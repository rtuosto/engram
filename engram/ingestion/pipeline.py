"""Per-Memory extraction pipeline.

Pipeline order (``docs/design/ingestion.md §6``):

1. Memory + Turn granule
2. Sentence segmentation
3. N-gram extraction (noun_chunk + SVO)
4. NER
5. Entity canonicalization
6. Claim extraction
7. Preference detection (fails closed per polarity gate)
8. Granule embedding (Turn + Sentence + N-gram → parallel vector index)
9. Temporal anchoring (TimeAnchor + ``temporal_at`` edges)

Per-instance state (:class:`InstanceState`) owns the GraphStore, the
parallel :class:`VectorIndex`, and the working caches (EntityRegistry,
monotonic ``memory_index``, speaker Entity index). The pipeline itself is
stateless — its construction carries the model dependencies and per-
polarity enablement gate.

**R5.** No LLM calls on this path. Everything here is spaCy + deterministic
embeddings + rapidfuzz + counts.

**R16.** Memories are never deduped. Content primitives (Entity / Claim /
Preference / Turn) are content-addressed; repeat content produces new
observation *edges* onto existing primitives, not duplicate primitives.
Aliases, co-occurrence, reinforcement counts, and current-truth are
derived (PR-D), not emitted here.

**No ``finalize`` pass in this PR.** Cross-Memory temporal edges and
co-occurrence were finalize-time concerns in the pre-pivot pipeline;
they're all derived in the new model. The derived-rebuild orchestrator
lands in PR-D.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from engram.config import MemoryConfig
from engram.ingestion.derived import DerivedIndex
from engram.ingestion.extractors.canonicalization import (
    EntityRegistry,
    canonicalize,
    normalize,
)
from engram.ingestion.extractors.claim import (
    ResolvedMention,
    extract_claims_from_sentence,
)
from engram.ingestion.extractors.ner import EntityMention, extract_mentions
from engram.ingestion.extractors.ngram import (
    extract_noun_chunk_ngrams,
    extract_svo_ngrams,
)
from engram.ingestion.extractors.preference import (
    PreferenceVerdict,
    build_preference_payload,
    classify_batch,
)
from engram.ingestion.extractors.segmentation import segment_turn
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_ABOUT,
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_MENTIONS,
    EDGE_PART_OF,
    EDGE_TEMPORAL_AT,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_NGRAM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LAYER_ENTITY,
    LAYER_RELATIONSHIP,
    LAYER_TEMPORAL,
    ClaimPayload,
    EdgeAttrs,
    EntityPayload,
    MemoryPayload,
    NgramPayload,
    TimeAnchorPayload,
    TurnPayload,
    entity_identity,
    memory_identity,
    node_id,
    round_iso_timestamp,
    segment_identity,
    time_anchor_identity,
    turn_identity,
)
from engram.ingestion.vector_index import (
    GRANULARITY_NGRAM,
    GRANULARITY_SENTENCE,
    GRANULARITY_TURN,
    VectorIndex,
)
from engram.models import Memory

SPEAKER_ENTITY_TYPE = "SPEAKER"
ANONYMOUS_SPEAKER = "__anonymous__"


@dataclass
class InstanceState:
    """Per-engram-instance ingest state.

    One engram instance holds one memory — no conversation-id partitioning.
    ``memory_index`` is a monotonic counter so Memory nodes from repeat
    ``ingest`` calls never collide (R16: Memories are events, never
    deduplicated).

    ``vector_index`` is the parallel granule-embedding store (``P10``).
    Lazy-initialized on the first ingest so ``create_state`` doesn't need
    to know the embedding dim up front (the pipeline discovers it from
    the first batch of granule embeddings).
    """

    store: GraphStore
    entity_registry: EntityRegistry = field(default_factory=EntityRegistry)
    speaker_to_entity_id: dict[str, str] = field(default_factory=dict)
    memory_index: int = 0
    vector_index: VectorIndex | None = None
    # Cached derived-rebuild snapshot. ``None`` until the first rebuild.
    # Staleness is detected via fingerprint; mutation is never in-place.
    derived: DerivedIndex | None = None


@dataclass(frozen=True, slots=True)
class _SegmentInfo:
    """Per-Sentence context passed to downstream stages.

    ``sent`` is the spaCy sentence span (or :class:`tests._fake_nlp.FakeSent`
    in tests); we keep a reference instead of re-iterating ``doc.sents`` to
    avoid double-filtering empty sentences across stages.
    """

    segment_id: str
    char_span: tuple[int, int]
    sent: object


@dataclass(frozen=True, slots=True)
class _MemoryWork:
    """Intermediate per-Memory bundle. Internal; not persisted."""

    memory_id: str
    turn_id: str
    mentions: tuple[EntityMention, ...]
    entity_id_by_span: dict[tuple[int, int], str]
    speaker_entity_id: str
    segments: tuple[_SegmentInfo, ...]


@dataclass(slots=True)
class _BatchBucket:
    """Mutable per-Memory bucket used by :meth:`IngestionPipeline.ingest_many`.

    Collects the structural output of stages 1–6 for one Memory while the
    batched model calls (stage 2 spaCy, stage 7 preference classification,
    stage 8 granule embedding) run once across the whole batch. ``work`` is
    ``None`` when spaCy produced no parsed doc for that Memory — the Turn
    granule still exists but no downstream claim / preference / n-gram /
    segment work happens for it (same fails-closed path as the sequential
    ``ingest`` with empty ``nlp_process`` output).
    """

    memory: Memory
    memory_id: str
    turn_id: str
    work: _MemoryWork | None
    anchor_sources: list[str]


class IngestionPipeline:
    """Orchestrates the per-Memory extraction pipeline.

    Model dependencies (spaCy pipe, preference centroids, embedding callable)
    are injected so tests can mock them. Production wires the real encoders
    via :func:`build_default_pipeline`.
    """

    def __init__(
        self,
        *,
        config: MemoryConfig,
        nlp_process: Callable[[list[str]], list[object]],
        preference_centroids: dict[str, np.ndarray],
        preference_embed: Callable[[list[str]], np.ndarray],
        granule_embed: Callable[[list[str]], np.ndarray],
        enabled_polarities: frozenset[str],
    ) -> None:
        self._config = config
        self._nlp_process = nlp_process
        self._centroids = preference_centroids
        self._preference_embed = preference_embed
        self._granule_embed = granule_embed
        self._enabled_polarities = enabled_polarities

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def create_state(self) -> InstanceState:
        return InstanceState(store=GraphStore(conversation_id="__instance__"))

    # ------------------------------------------------------------------
    # Memory ingest
    # ------------------------------------------------------------------

    def ingest(self, state: InstanceState, memory: Memory) -> None:
        if state.store.frozen:
            from engram.ingestion.graph import GraphFrozenError

            raise GraphFrozenError("cannot ingest into frozen engram instance")

        # Granule accumulator — populated by emit-granule stages in R2-stable
        # insertion order (Turn → Sentence(s) → N-gram(s)). Flushed to the
        # vector index in a single batched embed call at stage [8], after the
        # graph writes are done.
        granule_batch: list[tuple[str, str, str]] = []

        # Anchoring accumulator — granule + relationship node_ids created on
        # this ingest that should receive ``temporal_at`` edges at stage [9].
        # Order is insertion order; the emit call sorts at the edge site so
        # the final edge stream is R2-stable.
        anchor_sources: list[str] = []

        # [1] Memory + Turn granule.
        memory_id, memory_payload = self._emit_memory(state, memory)
        turn_id, _ = self._emit_turn_granule(
            state, memory_id, memory, granule_batch=granule_batch
        )
        anchor_sources.append(turn_id)

        # [2] spaCy process the Memory's content.
        docs = self._nlp_process([memory.content])
        if not docs:
            # Turn granule exists but has no sentences / n-grams — still embed
            # the whole-Memory representation so recall can find it.
            self._emit_granule_embeddings(state, granule_batch)
            self._emit_time_anchor(
                state, memory, anchor_sources=anchor_sources, memory_id=memory_id
            )
            return
        doc = docs[0]

        # [3] Segmentation.
        segments = self._emit_segments(
            state, turn_id, doc, memory_id=memory_id, granule_batch=granule_batch
        )
        anchor_sources.extend(seg.segment_id for seg in segments)

        # [4] N-gram extraction (noun_chunk + SVO).
        ngram_ids = self._emit_ngrams(
            state, doc, segments, memory_id=memory_id, granule_batch=granule_batch
        )
        anchor_sources.extend(ngram_ids)

        # [5] NER.
        mentions = extract_mentions(doc, turn_id)

        # [6] Entity canonicalization + mentions edges.
        entity_id_by_span: dict[tuple[int, int], str] = {}
        for mention in mentions:
            entity_id = self._canonicalize_and_link_mention(
                state, mention, turn_id, memory_id=memory_id, asserted_at=memory.timestamp
            )
            entity_id_by_span[mention.char_span] = entity_id

        speaker_label = memory.speaker or ANONYMOUS_SPEAKER
        speaker_entity_id = self._ensure_speaker_entity(state, speaker_label)

        work = _MemoryWork(
            memory_id=memory_id,
            turn_id=turn_id,
            mentions=tuple(mentions),
            entity_id_by_span=entity_id_by_span,
            speaker_entity_id=speaker_entity_id,
            segments=tuple(segments),
        )

        # [7] — claim extraction + preference detection per sentence.
        claim_pref_ids = self._extract_claims_and_preferences(
            state, work, asserted_at=memory.timestamp
        )
        anchor_sources.extend(claim_pref_ids)

        # [8] — granule embedding. One batched call per ingest; rows appended
        # to state.vector_index in insertion order.
        self._emit_granule_embeddings(state, granule_batch)

        # [9] — temporal anchoring. TimeAnchor node + ``temporal_at`` edges
        # from every granule + relationship created on this ingest. Skipped
        # silently when ``memory.timestamp`` is None (fails closed — there is
        # no anchor to attach to).
        self._emit_time_anchor(
            state, memory, anchor_sources=anchor_sources, memory_id=memory_id
        )

    def ingest_many(
        self, state: InstanceState, memories: Sequence[Memory]
    ) -> None:
        """Batched ingest across a sequence of Memories.

        Pipeline semantics match :meth:`ingest` per Memory — same nodes,
        same edges, same canonicalization state flow, same Memory-index
        sequence. The speedup comes from collapsing the three model calls
        across the batch dimension:

        * stage [2] — one ``nlp_process([all_memory_contents])`` call
          (spaCy's native pipe batching, instead of N single-doc calls).
        * stage [8] — one ``granule_embed(all_granule_texts)`` call over
          every Turn / Sentence / N-gram emitted across all Memories.
        * stage [7] — one ``preference_embed(all_sentence_texts)`` call
          over every Claim-bearing sentence across all Memories.

        Structural graph output (nodes / edge tuples / payloads) is
        byte-identical to ``[ingest(m) for m in memories]``. Edge
        ``weight`` on ``holds_preference`` and the float32 rows in
        :class:`VectorIndex` may drift at ~5e-8 because batch composition
        changes the numerics of batched transformer inference — this is
        covered by the structural-fingerprint guard at ``scripts/
        check_fingerprint_equivalence.py`` (R3/R4).

        **Append-only ordering (R16).** Memories are processed in the
        given order; ``state.memory_index`` advances monotonically; graph
        writes happen per-Memory in the same order the sequential path
        would make them. The batched model calls only pool the inputs —
        they never re-order observations.
        """
        if state.store.frozen:
            from engram.ingestion.graph import GraphFrozenError

            raise GraphFrozenError("cannot ingest into frozen engram instance")
        if not memories:
            return

        # Stage [2]: one spaCy call for the whole batch. Fall back to an
        # empty doc list if ``nlp_process`` mis-sizes its return (defensive;
        # real factory-wired callables always return one doc per input).
        texts = [m.content for m in memories]
        docs = self._nlp_process(texts)

        # Cross-Memory accumulators. Granules are appended in per-Memory
        # emission order (same order the sequential ingest would use) so
        # the vector index row ordering stays R2-stable.
        global_granule_batch: list[tuple[str, str, str]] = []
        global_pref_texts: list[str] = []
        # (bucket_index, claim_payload, asserted_at) per pending preference
        # entry, positionally aligned with ``global_pref_texts``.
        global_pref_meta: list[tuple[int, ClaimPayload, str | None]] = []
        buckets: list[_BatchBucket] = []

        for mem_idx, memory in enumerate(memories):
            # [1] Memory node + Turn granule.
            memory_id, _ = self._emit_memory(state, memory)
            turn_id, _ = self._emit_turn_granule(
                state, memory_id, memory, granule_batch=global_granule_batch
            )
            anchor_sources: list[str] = [turn_id]
            bucket = _BatchBucket(
                memory=memory,
                memory_id=memory_id,
                turn_id=turn_id,
                work=None,
                anchor_sources=anchor_sources,
            )
            buckets.append(bucket)

            doc = docs[mem_idx] if mem_idx < len(docs) else None
            if doc is None:
                # Same fails-closed shape as ``ingest`` with empty
                # ``nlp_process`` output: Turn granule stays, no downstream
                # structural work, TimeAnchor still fires at stage [9].
                continue

            # [3] Segmentation.
            segments = self._emit_segments(
                state,
                turn_id,
                doc,
                memory_id=memory_id,
                granule_batch=global_granule_batch,
            )
            anchor_sources.extend(seg.segment_id for seg in segments)

            # [4] N-gram extraction.
            ngram_ids = self._emit_ngrams(
                state,
                doc,
                segments,
                memory_id=memory_id,
                granule_batch=global_granule_batch,
            )
            anchor_sources.extend(ngram_ids)

            # [5] NER.
            mentions = extract_mentions(doc, turn_id)

            # [6] Entity canonicalization + mention edges.
            entity_id_by_span: dict[tuple[int, int], str] = {}
            for mention in mentions:
                entity_id = self._canonicalize_and_link_mention(
                    state,
                    mention,
                    turn_id,
                    memory_id=memory_id,
                    asserted_at=memory.timestamp,
                )
                entity_id_by_span[mention.char_span] = entity_id

            speaker_label = memory.speaker or ANONYMOUS_SPEAKER
            speaker_entity_id = self._ensure_speaker_entity(state, speaker_label)

            work = _MemoryWork(
                memory_id=memory_id,
                turn_id=turn_id,
                mentions=tuple(mentions),
                entity_id_by_span=entity_id_by_span,
                speaker_entity_id=speaker_entity_id,
                segments=tuple(segments),
            )
            bucket.work = work

            # [7a] Claim emission (Pass 1 of preference). Pending prefs
            # feed the global preference batch below.
            claim_ids, pending_prefs = self._emit_claims_collect_prefs(
                state, work, asserted_at=memory.timestamp
            )
            anchor_sources.extend(claim_ids)
            for claim_payload, sent_text in pending_prefs:
                global_pref_texts.append(sent_text)
                global_pref_meta.append((mem_idx, claim_payload, memory.timestamp))

        # [8] Stage — one granule embedding call for the whole batch.
        self._emit_granule_embeddings(state, global_granule_batch)

        # [7b] Stage — one preference classification call for the whole
        # batch. Verdicts align positionally with ``global_pref_meta``.
        if global_pref_texts:
            verdicts = classify_batch(
                global_pref_texts,
                self._centroids,
                self._preference_embed,
                margin_threshold=self._config.preference_discrimination_margin,
                enabled_polarities=self._enabled_polarities,
            )
            for (bucket_idx, claim_payload, asserted_at), verdict in zip(
                global_pref_meta, verdicts, strict=True
            ):
                if verdict is None:
                    continue
                bucket = buckets[bucket_idx]
                if bucket.work is None:
                    continue
                pref_id = self._emit_preference_from_verdict(
                    state,
                    bucket.work,
                    verdict=verdict,
                    claim_payload=claim_payload,
                    asserted_at=asserted_at,
                )
                bucket.anchor_sources.append(pref_id)

        # [9] Stage — per-Memory TimeAnchor + ``temporal_at`` edges. Same
        # fails-closed rule as ``ingest``: skipped when ``timestamp`` is None.
        for bucket in buckets:
            self._emit_time_anchor(
                state,
                bucket.memory,
                anchor_sources=bucket.anchor_sources,
                memory_id=bucket.memory_id,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_memory(
        self, state: InstanceState, memory: Memory
    ) -> tuple[str, MemoryPayload]:
        state.memory_index += 1
        mid = node_id(memory_identity(state.memory_index))
        metadata = tuple(sorted(memory.metadata, key=lambda kv: kv[0]))
        payload = MemoryPayload(
            memory_index=state.memory_index,
            content=memory.content,
            timestamp=memory.timestamp,
            speaker=memory.speaker,
            source=memory.source,
            metadata=metadata,
        )
        state.store.add_node(
            mid,
            labels=frozenset({LABEL_MEMORY}),
            payloads={LABEL_MEMORY: payload},
        )
        return mid, payload

    def _emit_turn_granule(
        self,
        state: InstanceState,
        memory_id: str,
        memory: Memory,
        *,
        granule_batch: list[tuple[str, str, str]],
    ) -> tuple[str, TurnPayload]:
        tid = node_id(turn_identity(memory_id))
        payload = TurnPayload(
            memory_id=memory_id,
            text=memory.content,
            speaker=memory.speaker,
            timestamp=memory.timestamp,
        )
        state.store.add_node(
            tid,
            labels=frozenset({LABEL_TURN}),
            payloads={LABEL_TURN: payload},
        )
        state.store.add_edge(
            tid,
            memory_id,
            EdgeAttrs(
                type=EDGE_PART_OF,
                weight=1.0,
                source_memory_id=memory_id,
                source_turn_id=tid,
                asserted_at=memory.timestamp,
            ),
        )
        granule_batch.append((tid, GRANULARITY_TURN, memory.content))
        return tid, payload

    def _emit_segments(
        self,
        state: InstanceState,
        turn_id: str,
        doc: object,
        *,
        memory_id: str,
        granule_batch: list[tuple[str, str, str]],
    ) -> list[_SegmentInfo]:
        """Emit UtteranceSegment nodes; return per-sentence context for
        downstream stages (n-gram extraction, claim / preference).

        Iterates ``doc.sents`` once, skipping empty/whitespace-only sentences
        in the same order :func:`segment_turn` does, and keeps a reference to
        each live sentence span so subsequent stages don't need to re-filter.
        """
        segments: list[_SegmentInfo] = []
        seg_index = 0
        for sent in getattr(doc, "sents", ()):
            text = str(getattr(sent, "text", "")).strip()
            if not text:
                continue
            start = int(getattr(sent, "start_char", 0))
            end = int(getattr(sent, "end_char", start + len(text)))
            seg_id = node_id(segment_identity(turn_id, seg_index))
            segments.append(
                _SegmentInfo(
                    segment_id=seg_id,
                    char_span=(start, end),
                    sent=sent,
                )
            )
            seg_index += 1

        # segment_turn() is re-used for its R2-sorted payload construction
        # (char spans and segment_index match the sentences above because
        # both iterators apply the same "strip empty" filter).
        for seg_id, payload in segment_turn(doc, turn_id):
            state.store.add_node(
                seg_id,
                labels=frozenset({LABEL_UTTERANCE_SEGMENT}),
                payloads={LABEL_UTTERANCE_SEGMENT: payload},
            )
            state.store.add_edge(
                seg_id,
                turn_id,
                EdgeAttrs(
                    type=EDGE_PART_OF,
                    weight=1.0,
                    source_memory_id=memory_id,
                    source_turn_id=turn_id,
                ),
            )
            granule_batch.append((seg_id, GRANULARITY_SENTENCE, payload.text))
        return segments

    def _emit_ngrams(
        self,
        state: InstanceState,
        doc: object,
        segments: list[_SegmentInfo],
        *,
        memory_id: str,
        granule_batch: list[tuple[str, str, str]],
    ) -> list[str]:
        """Emit N-gram nodes (noun chunks + SVO triples) with ``part_of``
        edges to their containing Sentence granule.

        Identity is ``(segment_id, ngram_kind, normalized_text)`` — an
        observation of the same phrase from the same Sentence converges to
        one node regardless of how many times the extractor visits it. N-gram
        granules carry no layer label in PR-B (``semantic`` is implicit in
        the parallel embedding index landing in PR-C).
        """
        if not segments:
            return []
        min_tokens = self._config.ngram_min_tokens
        segment_spans: list[tuple[tuple[int, int], str]] = [
            (seg.char_span, seg.segment_id) for seg in segments
        ]
        noun_chunk_ngrams = extract_noun_chunk_ngrams(
            doc, segment_spans, min_tokens=min_tokens
        )
        svo_ngrams: list[tuple[str, NgramPayload]] = []
        for seg in segments:
            svo_ngrams.extend(
                extract_svo_ngrams(seg.sent, seg.segment_id, min_tokens=min_tokens)
            )

        all_ngrams = sorted(
            noun_chunk_ngrams + svo_ngrams,
            key=lambda pair: (pair[1].char_span, pair[1].ngram_kind, pair[1].normalized_text),
        )
        # N-gram identity collapses repeated visits to the same
        # (segment, kind, normalized_text); dedup on node_id before
        # appending to the embedding batch so the vector index rejects
        # duplicates that never make it into the graph.
        seen_ngram_ids: set[str] = set()
        emitted: list[str] = []
        for ngram_id, payload in all_ngrams:
            if ngram_id in seen_ngram_ids:
                continue
            seen_ngram_ids.add(ngram_id)
            state.store.add_node(
                ngram_id,
                labels=frozenset({LABEL_NGRAM}),
                payloads={LABEL_NGRAM: payload},
            )
            state.store.add_edge(
                ngram_id,
                payload.segment_id,
                EdgeAttrs(
                    type=EDGE_PART_OF,
                    weight=1.0,
                    source_memory_id=memory_id,
                    source_turn_id=payload.segment_id,
                ),
            )
            granule_batch.append((ngram_id, GRANULARITY_NGRAM, payload.surface_form))
            emitted.append(ngram_id)
        return emitted

    def _emit_granule_embeddings(
        self,
        state: InstanceState,
        granule_batch: list[tuple[str, str, str]],
    ) -> None:
        """Stage [8]: batched MiniLM embedding for every granule emitted on
        this ingest call.

        ``granule_batch`` is ordered by pipeline emission: Turn first, then
        Sentences in segment order, then N-grams in ``(char_span, kind,
        text)`` order (see :meth:`_emit_ngrams`). That's the order the
        vector index persists; it must be R2-stable across repeat ingests.

        The index's dim is discovered from the first embed call and frozen
        for the life of the state. A subsequent ingest whose embed function
        returns a different dim is a configuration error.
        """
        if not granule_batch:
            return

        texts = [text for _nid, _gran, text in granule_batch]
        vectors = self._granule_embed(texts)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(granule_batch):
            raise RuntimeError(
                f"granule_embed returned shape {arr.shape}; "
                f"expected ({len(granule_batch)}, dim)"
            )
        dim = int(arr.shape[1])

        if state.vector_index is None:
            state.vector_index = VectorIndex(dim=dim)
        elif state.vector_index.dim != dim:
            raise RuntimeError(
                f"granule_embed produced dim {dim}; state.vector_index "
                f"was initialized with dim {state.vector_index.dim}"
            )

        for (node_id_, granularity, _text), row in zip(granule_batch, arr, strict=True):
            state.vector_index.add(node_id_, granularity, row)

    def _canonicalize_and_link_mention(
        self,
        state: InstanceState,
        mention: EntityMention,
        turn_id: str,
        *,
        memory_id: str,
        asserted_at: str | None,
    ) -> str:
        entity_id, payload, _is_new = canonicalize(
            mention,
            state.entity_registry,
            match_threshold=self._config.canonicalization_match_threshold,
        )
        state.store.add_node(
            entity_id,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: payload},
            layers=frozenset({LAYER_ENTITY}),
        )
        state.store.add_edge(
            turn_id,
            entity_id,
            EdgeAttrs(
                type=EDGE_MENTIONS,
                weight=1.0,
                source_memory_id=memory_id,
                source_turn_id=turn_id,
                asserted_at=asserted_at,
                surface_form=mention.surface_form,
            ),
        )
        return entity_id

    def _ensure_speaker_entity(self, state: InstanceState, speaker: str) -> str:
        if speaker in state.speaker_to_entity_id:
            return state.speaker_to_entity_id[speaker]
        normalized = normalize(speaker)
        entity_id = node_id(entity_identity(normalized, SPEAKER_ENTITY_TYPE))
        payload = EntityPayload(
            canonical_form=normalized,
            entity_type=SPEAKER_ENTITY_TYPE,
        )
        state.store.add_node(
            entity_id,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: payload},
            layers=frozenset({LAYER_ENTITY}),
        )
        # Register so future NER mentions of the same speaker label resolve
        # to the same node deterministically.
        state.entity_registry.by_type_and_form[(SPEAKER_ENTITY_TYPE, normalized)] = entity_id
        state.speaker_to_entity_id[speaker] = entity_id
        return entity_id

    def _extract_claims_and_preferences(
        self,
        state: InstanceState,
        work: _MemoryWork,
        *,
        asserted_at: str | None,
    ) -> list[str]:
        """Returns the list of Claim + Preference node IDs emitted on this
        ingest (de-duplicated in insertion order). Used by stage [9] to
        attach ``temporal_at`` edges to freshly observed relationships.

        Preference classification is batched per-Memory: all claim sentences
        for this Memory are embedded in one ``preference_embed`` call.
        ``ingest_many`` further widens the batch across Memories; the split
        between pass-1 (claim emission) and pass-2 (preference classification)
        is factored into :meth:`_emit_claims_collect_prefs` and
        :meth:`_emit_prefs_from_verdicts` so both paths share the per-claim
        graph-write code.

        Persisted output is byte-identical to the per-claim path because
        ``dump_state`` sorts nodes by ``node_id`` and edges by ``(src, dst,
        type)`` — insertion order of Claim and Preference nodes does not
        affect the msgpack bytes (see persist.py:151,158).
        """
        claim_ids, pending_prefs = self._emit_claims_collect_prefs(
            state, work, asserted_at=asserted_at
        )

        if not pending_prefs:
            return list(claim_ids)

        verdicts = classify_batch(
            [text for _, text in pending_prefs],
            self._centroids,
            self._preference_embed,
            margin_threshold=self._config.preference_discrimination_margin,
            enabled_polarities=self._enabled_polarities,
        )
        pref_ids = self._emit_prefs_from_verdicts(
            state, work, pending_prefs, verdicts, asserted_at=asserted_at
        )

        seen: set[str] = set()
        out: list[str] = []
        for nid in list(claim_ids) + list(pref_ids):
            if nid in seen:
                continue
            seen.add(nid)
            out.append(nid)
        return out

    def _emit_claims_collect_prefs(
        self,
        state: InstanceState,
        work: _MemoryWork,
        *,
        asserted_at: str | None,
    ) -> tuple[list[str], list[tuple[ClaimPayload, str]]]:
        """Pass 1: emit Claim nodes/edges for a single Memory and collect
        ``(claim_payload, sentence_text)`` pairs that feed preference
        classification. Shared by :meth:`ingest` and :meth:`ingest_many`.

        The returned ``pending_prefs`` list preserves segment/claim order so
        downstream preference emission stays R2-stable.
        """
        resolved = [
            ResolvedMention(
                entity_id=work.entity_id_by_span[(m.char_span[0], m.char_span[1])],
                char_span=m.char_span,
            )
            for m in work.mentions
            if (m.char_span[0], m.char_span[1]) in work.entity_id_by_span
        ]

        claim_ids: list[str] = []
        pending_prefs: list[tuple[ClaimPayload, str]] = []
        for seg in work.segments:
            sent = seg.sent
            sent_start, sent_end = seg.char_span
            sent_mentions = [
                rm
                for rm in resolved
                if rm.char_span[0] >= sent_start and rm.char_span[1] <= sent_end
            ]
            sent_text = str(getattr(sent, "text", "")).strip()

            claims = extract_claims_from_sentence(
                sent,
                speaker_entity_id=work.speaker_entity_id,
                mentions=sent_mentions,
                subject_required=self._config.claim_subject_required,
            )

            for claim_id, claim_payload in claims:
                self._emit_claim(
                    state, work, claim_id, claim_payload, asserted_at=asserted_at
                )
                claim_ids.append(claim_id)
                pending_prefs.append((claim_payload, sent_text))

        return claim_ids, pending_prefs

    def _emit_prefs_from_verdicts(
        self,
        state: InstanceState,
        work: _MemoryWork,
        pending_prefs: Sequence[tuple[ClaimPayload, str]],
        verdicts: Sequence[PreferenceVerdict | None],
        *,
        asserted_at: str | None,
    ) -> list[str]:
        """Pass 2: given a positionally-aligned ``(pending_prefs, verdicts)``
        pair, emit a Preference node for each non-``None`` verdict. Used by
        both :meth:`_extract_claims_and_preferences` (per-Memory batch) and
        :meth:`ingest_many` (cross-Memory batch).
        """
        pref_ids: list[str] = []
        for (claim_payload, _sent_text), verdict in zip(
            pending_prefs, verdicts, strict=True
        ):
            if verdict is None:
                continue
            pref_id = self._emit_preference_from_verdict(
                state,
                work,
                verdict=verdict,
                claim_payload=claim_payload,
                asserted_at=asserted_at,
            )
            pref_ids.append(pref_id)
        return pref_ids

    def _emit_claim(
        self,
        state: InstanceState,
        work: _MemoryWork,
        claim_id: str,
        claim_payload: ClaimPayload,
        *,
        asserted_at: str | None,
    ) -> None:
        state.store.add_node(
            claim_id,
            labels=frozenset({LABEL_CLAIM}),
            payloads={LABEL_CLAIM: claim_payload},
            layers=frozenset({LAYER_RELATIONSHIP}),
        )
        state.store.add_edge(
            work.turn_id,
            claim_id,
            EdgeAttrs(
                type=EDGE_ASSERTS,
                weight=1.0,
                source_memory_id=work.memory_id,
                source_turn_id=work.turn_id,
                asserted_at=asserted_at,
            ),
        )
        candidates: set[str] = set()
        if claim_payload.subject_id:
            candidates.add(claim_payload.subject_id)
        if claim_payload.object_id:
            candidates.add(claim_payload.object_id)
        for entity_id in sorted(candidates):
            if not state.store.has_node(entity_id):
                continue
            state.store.add_edge(
                claim_id,
                entity_id,
                EdgeAttrs(
                    type=EDGE_ABOUT,
                    weight=1.0,
                    source_memory_id=work.memory_id,
                    source_turn_id=work.turn_id,
                    asserted_at=asserted_at,
                ),
            )

    def _emit_preference_from_verdict(
        self,
        state: InstanceState,
        work: _MemoryWork,
        *,
        verdict: PreferenceVerdict,
        claim_payload: ClaimPayload,
        asserted_at: str | None,
    ) -> str:
        pref_id, pref_payload = build_preference_payload(
            verdict,
            claim_payload=claim_payload,
            speaker_entity_id=work.speaker_entity_id,
        )
        state.store.add_node(
            pref_id,
            labels=frozenset({LABEL_PREFERENCE}),
            payloads={LABEL_PREFERENCE: pref_payload},
            layers=frozenset({LAYER_RELATIONSHIP}),
        )
        # Observation: holder Entity → Preference node.
        state.store.add_edge(
            work.speaker_entity_id,
            pref_id,
            EdgeAttrs(
                type=EDGE_HOLDS_PREFERENCE,
                weight=verdict.confidence,
                source_memory_id=work.memory_id,
                source_turn_id=work.turn_id,
                asserted_at=asserted_at,
            ),
        )
        if pref_payload.target_id and state.store.has_node(pref_payload.target_id):
            state.store.add_edge(
                pref_id,
                pref_payload.target_id,
                EdgeAttrs(
                    type=EDGE_ABOUT,
                    weight=1.0,
                    source_memory_id=work.memory_id,
                    source_turn_id=work.turn_id,
                    asserted_at=asserted_at,
                ),
            )
        return pref_id


    def _emit_time_anchor(
        self,
        state: InstanceState,
        memory: Memory,
        *,
        anchor_sources: list[str],
        memory_id: str,
    ) -> None:
        """Stage [9]: emit (or reuse) a TimeAnchor node for the Memory's
        timestamp and attach ``temporal_at`` edges from every granule and
        relationship created on this ingest.

        Fails closed when ``memory.timestamp`` is ``None`` — no anchor is
        created and no edges are emitted. The temporal layer is opt-in per
        observation, not a runtime precondition.

        Anchors are content-addressed by the rounded ISO timestamp (see
        :func:`engram.ingestion.schema.round_iso_timestamp`), so repeat
        ingests at the same rounded moment converge onto one node. ``R16``:
        the anchor's payload is never mutated — two observations at the
        same instant share the node via incoming edges, which is the
        R16-legal form of "reinforcement."
        """
        if memory.timestamp is None or not anchor_sources:
            return

        rounded = round_iso_timestamp(
            memory.timestamp, self._config.time_anchor_resolution
        )
        anchor_id = node_id(time_anchor_identity(rounded))
        state.store.add_node(
            anchor_id,
            labels=frozenset({LABEL_TIME_ANCHOR}),
            payloads={LABEL_TIME_ANCHOR: TimeAnchorPayload(iso_timestamp=rounded)},
            layers=frozenset({LAYER_TEMPORAL}),
        )
        # Dedup + sort for R2-stable edge iteration: even though
        # ``anchor_sources`` is already insertion-ordered, subsequent walks
        # over ``MultiDiGraph`` sort by (src, dst, key), so only the dedup
        # matters. Sorting here additionally guards against insertion-order
        # drift between the pipeline and downstream test fixtures.
        for src_id in sorted(set(anchor_sources)):
            if not state.store.has_node(src_id):
                continue
            state.store.add_edge(
                src_id,
                anchor_id,
                EdgeAttrs(
                    type=EDGE_TEMPORAL_AT,
                    weight=1.0,
                    source_memory_id=memory_id,
                    source_turn_id=src_id,
                    asserted_at=rounded,
                ),
            )


__all__ = [
    "ANONYMOUS_SPEAKER",
    "IngestionPipeline",
    "InstanceState",
    "SPEAKER_ENTITY_TYPE",
]
