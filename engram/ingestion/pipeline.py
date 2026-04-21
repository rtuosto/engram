"""Per-Memory extraction pipeline.

Pipeline order (``docs/design/ingestion.md §6``):

1. Memory + Turn granule
2. Sentence segmentation
3. NER
4. Entity canonicalization
5. Claim extraction
6. Preference detection (fails closed per polarity gate)

Per-instance state (:class:`InstanceState`) owns the GraphStore and the
working caches (EntityRegistry, monotonic ``memory_index``, speaker Entity
index). The pipeline itself is stateless — its construction carries the
model dependencies and per-polarity enablement gate.

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

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from engram.config import MemoryConfig
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
from engram.ingestion.extractors.preference import (
    build_preference_payload,
    classify,
)
from engram.ingestion.extractors.segmentation import segment_turn
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_ABOUT,
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_PREFERENCE,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    ClaimPayload,
    EdgeAttrs,
    EntityPayload,
    MemoryPayload,
    TurnPayload,
    entity_identity,
    memory_identity,
    node_id,
    turn_identity,
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
    """

    store: GraphStore
    entity_registry: EntityRegistry = field(default_factory=EntityRegistry)
    speaker_to_entity_id: dict[str, str] = field(default_factory=dict)
    memory_index: int = 0


@dataclass(frozen=True, slots=True)
class _MemoryWork:
    """Intermediate per-Memory bundle. Internal; not persisted."""

    memory_id: str
    turn_id: str
    mentions: tuple[EntityMention, ...]
    entity_id_by_span: dict[tuple[int, int], str]
    speaker_entity_id: str


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
        enabled_polarities: frozenset[str],
    ) -> None:
        self._config = config
        self._nlp_process = nlp_process
        self._centroids = preference_centroids
        self._preference_embed = preference_embed
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

        # [1] Memory + Turn granule.
        memory_id, memory_payload = self._emit_memory(state, memory)
        turn_id, _ = self._emit_turn_granule(state, memory_id, memory)

        # [2] spaCy process the Memory's content.
        docs = self._nlp_process([memory.content])
        if not docs:
            return
        doc = docs[0]

        # [3] Segmentation.
        self._emit_segments(state, turn_id, doc, memory_id=memory_id)

        # [4] NER.
        mentions = extract_mentions(doc, turn_id)

        # [5] Entity canonicalization + mentions edges.
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
        )

        # [6] + [7] — claim extraction + preference detection per sentence.
        self._extract_claims_and_preferences(
            state, work, doc, asserted_at=memory.timestamp
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
        self, state: InstanceState, memory_id: str, memory: Memory
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
        return tid, payload

    def _emit_segments(
        self, state: InstanceState, turn_id: str, doc: object, *, memory_id: str
    ) -> None:
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
        doc: object,
        *,
        asserted_at: str | None,
    ) -> None:
        resolved = [
            ResolvedMention(
                entity_id=work.entity_id_by_span[(m.char_span[0], m.char_span[1])],
                char_span=m.char_span,
            )
            for m in work.mentions
            if (m.char_span[0], m.char_span[1]) in work.entity_id_by_span
        ]

        for sent in getattr(doc, "sents", ()):
            sent_start = int(getattr(sent, "start_char", 0))
            sent_end = int(getattr(sent, "end_char", sent_start))
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
                self._emit_claim(state, work, claim_id, claim_payload, asserted_at=asserted_at)
                self._maybe_emit_preference(
                    state,
                    work,
                    claim_payload=claim_payload,
                    sentence_text=sent_text,
                    asserted_at=asserted_at,
                )

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

    def _maybe_emit_preference(
        self,
        state: InstanceState,
        work: _MemoryWork,
        *,
        claim_payload: ClaimPayload,
        sentence_text: str,
        asserted_at: str | None,
    ) -> None:
        verdict = classify(
            sentence_text,
            self._centroids,
            self._preference_embed,
            margin_threshold=self._config.preference_discrimination_margin,
            enabled_polarities=self._enabled_polarities,
        )
        if verdict is None:
            return
        pref_id, pref_payload = build_preference_payload(
            verdict,
            claim_payload=claim_payload,
            speaker_entity_id=work.speaker_entity_id,
        )
        state.store.add_node(
            pref_id,
            labels=frozenset({LABEL_PREFERENCE}),
            payloads={LABEL_PREFERENCE: pref_payload},
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


__all__ = [
    "ANONYMOUS_SPEAKER",
    "IngestionPipeline",
    "InstanceState",
    "SPEAKER_ENTITY_TYPE",
]
