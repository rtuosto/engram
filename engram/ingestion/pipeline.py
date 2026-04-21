"""Stage orchestrator — wires the six Tier-1 extractors into a pipeline.

Pipeline order (``docs/design/ingestion.md §3``):

1. Segmentation
2. NER
3. Canonicalization
4. Claim extraction
5. Preference detection (fails closed per polarity gate)
6. Co-occurrence (accumulated; emitted at :meth:`finalize_conversation`)

Per-conversation state (:class:`ConversationState`) owns the GraphStore and
the working caches (EntityRegistry, CoOccurrenceCounter, speaker entity
index). The pipeline itself is stateless — its construction carries the
model dependencies and per-polarity enablement gate.

**R5.** No LLM calls on this path. Everything here is spaCy + deterministic
embeddings + rapidfuzz + counts.
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
from engram.ingestion.extractors.co_occurrence import CoOccurrenceCounter
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
    EDGE_TEMPORAL_AFTER,
    EDGE_TEMPORAL_BEFORE,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_PREFERENCE,
    LABEL_SESSION,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    EdgeAttrs,
    EntityPayload,
    SessionPayload,
    TurnPayload,
    entity_identity,
    node_id,
    session_identity,
    turn_identity,
)
from engram.models import Session, Turn

SPEAKER_ENTITY_TYPE = "SPEAKER"


@dataclass
class ConversationState:
    """Per-conversation ingest state — graph + working caches."""

    conversation_id: str
    store: GraphStore
    entity_registry: EntityRegistry = field(default_factory=EntityRegistry)
    co_occurrence: CoOccurrenceCounter = field(default_factory=CoOccurrenceCounter)
    speaker_to_entity_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _TurnWork:
    """Intermediate per-Turn bundle. Internal; not persisted."""

    turn_id: str
    turn_payload: TurnPayload
    mentions: tuple[EntityMention, ...]
    entity_id_by_span: dict[tuple[int, int], str]
    mentioned_entity_ids: tuple[str, ...]
    speaker_entity_id: str


class IngestionPipeline:
    """Orchestrates the Tier-1 extraction pipeline.

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

    def create_state(self, conversation_id: str) -> ConversationState:
        return ConversationState(
            conversation_id=conversation_id,
            store=GraphStore(conversation_id=conversation_id),
        )

    # ------------------------------------------------------------------
    # Session ingest
    # ------------------------------------------------------------------

    def ingest_session(self, state: ConversationState, session: Session) -> None:
        if state.store.frozen:
            from engram.ingestion.graph import GraphFrozenError

            raise GraphFrozenError(
                f"cannot ingest into frozen conversation {state.conversation_id!r}"
            )

        session_nid = self._ensure_session_node(state, session)
        docs = self._nlp_process([t.text for t in session.turns])

        # Stage 1: segmentation + Turn nodes + part_of edges (seg→turn, turn→session).
        turn_work: list[_TurnWork] = []
        for turn, doc in zip(session.turns, docs, strict=True):
            turn_id, turn_payload = self._ensure_turn_node(state, turn, session_nid)
            self._emit_segments(state, turn_id, doc)

            # Stage 2: NER.
            mentions = extract_mentions(doc, turn_id)

            # Stage 3: canonicalization + mentions edges.
            entity_id_by_span: dict[tuple[int, int], str] = {}
            mentioned_ids: list[str] = []
            for mention in mentions:
                entity_id = self._canonicalize_and_link_mention(
                    state, mention, turn_id, turn.timestamp
                )
                entity_id_by_span[mention.char_span] = entity_id
                mentioned_ids.append(entity_id)

            # Speaker entity (always attached; not an NER mention).
            speaker_entity_id = self._ensure_speaker_entity(state, turn.speaker)
            mentioned_ids.append(speaker_entity_id)

            # Accumulate co-occurrence on distinct canonical entities observed.
            state.co_occurrence.observe_turn(mentioned_ids)

            turn_work.append(
                _TurnWork(
                    turn_id=turn_id,
                    turn_payload=turn_payload,
                    mentions=tuple(mentions),
                    entity_id_by_span=entity_id_by_span,
                    mentioned_entity_ids=tuple(sorted(set(mentioned_ids))),
                    speaker_entity_id=speaker_entity_id,
                )
            )

        # Stage 4 + 5: claim extraction + preference detection, per Turn.
        # We run this *after* canonicalization for the whole session so that
        # first-person pronouns in later turns resolve against speaker entities
        # created earlier in the session.
        for work, doc in zip(turn_work, docs, strict=True):
            self._extract_claims_and_preferences(state, work, doc)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize_conversation(self, state: ConversationState) -> None:
        if state.store.frozen:
            return  # idempotent

        # Temporal edges at the Turn level — deterministic from indexes.
        turn_ids_sorted = self._turn_ids_in_order(state)
        for earlier, later in zip(turn_ids_sorted, turn_ids_sorted[1:], strict=False):
            state.store.add_edge(
                earlier,
                later,
                EdgeAttrs(type=EDGE_TEMPORAL_BEFORE, weight=1.0),
            )
            state.store.add_edge(
                later,
                earlier,
                EdgeAttrs(type=EDGE_TEMPORAL_AFTER, weight=1.0),
            )

        # Co-occurrence emission.
        for src, dst, attrs in state.co_occurrence.edges():
            # Defensive: skip self-edges (possible if observe_turn was ever
            # fed duplicate entity IDs, which it dedupes, but be explicit).
            if src == dst:
                continue
            if not state.store.has_node(src) or not state.store.has_node(dst):
                continue
            state.store.add_edge(src, dst, attrs)

        state.store.freeze()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_session_node(self, state: ConversationState, session: Session) -> str:
        sid = node_id(session_identity(state.conversation_id, session.session_index))
        if not state.store.has_node(sid):
            state.store.add_node(
                sid,
                labels=frozenset({LABEL_SESSION}),
                payloads={
                    LABEL_SESSION: SessionPayload(
                        conversation_id=state.conversation_id,
                        session_index=session.session_index,
                        timestamp=session.timestamp,
                    )
                },
            )
        return sid

    def _ensure_turn_node(
        self, state: ConversationState, turn: Turn, session_nid: str
    ) -> tuple[str, TurnPayload]:
        tid = node_id(
            turn_identity(state.conversation_id, turn.session_index, turn.turn_index)
        )
        payload = TurnPayload(
            speaker=turn.speaker,
            text=turn.text,
            conversation_id=state.conversation_id,
            session_index=turn.session_index,
            turn_index=turn.turn_index,
            timestamp=turn.timestamp,
        )
        state.store.add_node(
            tid,
            labels=frozenset({LABEL_TURN}),
            payloads={LABEL_TURN: payload},
        )
        state.store.add_edge(tid, session_nid, EdgeAttrs(type=EDGE_PART_OF, weight=1.0))
        return tid, payload

    def _emit_segments(self, state: ConversationState, turn_id: str, doc: object) -> None:
        for seg_id, payload in segment_turn(doc, turn_id):
            state.store.add_node(
                seg_id,
                labels=frozenset({LABEL_UTTERANCE_SEGMENT}),
                payloads={LABEL_UTTERANCE_SEGMENT: payload},
            )
            state.store.add_edge(
                seg_id,
                turn_id,
                EdgeAttrs(type=EDGE_PART_OF, weight=1.0, source_turn_id=turn_id),
            )

    def _canonicalize_and_link_mention(
        self,
        state: ConversationState,
        mention: EntityMention,
        turn_id: str,
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
            EdgeAttrs(type=EDGE_MENTIONS, weight=1.0, source_turn_id=turn_id, asserted_at=asserted_at),
        )
        return entity_id

    def _ensure_speaker_entity(self, state: ConversationState, speaker: str) -> str:
        if speaker in state.speaker_to_entity_id:
            return state.speaker_to_entity_id[speaker]
        normalized = normalize(speaker)
        entity_id = node_id(entity_identity(normalized, SPEAKER_ENTITY_TYPE))
        payload = EntityPayload(
            canonical_form=normalized,
            entity_type=SPEAKER_ENTITY_TYPE,
            aliases=(speaker,),
        )
        state.store.add_node(
            entity_id,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: payload},
        )
        # Register in canonicalization registry too so future NER mentions
        # of the speaker's label don't create a duplicate (unlikely, but
        # deterministic).
        state.entity_registry.by_type_and_form[(SPEAKER_ENTITY_TYPE, normalized)] = (
            entity_id,
            (speaker,),
        )
        state.speaker_to_entity_id[speaker] = entity_id
        return entity_id

    def _extract_claims_and_preferences(
        self, state: ConversationState, work: _TurnWork, doc: object
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
                turn_id=work.turn_id,
                speaker_entity_id=work.speaker_entity_id,
                mentions=sent_mentions,
                asserted_at=work.turn_payload.timestamp,
                subject_required=self._config.claim_subject_required,
            )

            for claim_id, claim_payload in claims:
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
                        source_turn_id=work.turn_id,
                        asserted_at=claim_payload.asserted_at,
                    ),
                )
                about_ids = sorted(
                    {claim_payload.subject_id, claim_payload.object_id} - {None, ""}
                )
                for entity_id in about_ids:
                    if not state.store.has_node(entity_id):
                        continue
                    state.store.add_edge(
                        claim_id,
                        entity_id,
                        EdgeAttrs(
                            type=EDGE_ABOUT,
                            weight=1.0,
                            source_turn_id=work.turn_id,
                            asserted_at=claim_payload.asserted_at,
                        ),
                    )

                verdict = classify(
                    sent_text or work.turn_payload.text,
                    self._centroids,
                    self._preference_embed,
                    margin_threshold=self._config.preference_discrimination_margin,
                    enabled_polarities=self._enabled_polarities,
                )
                if verdict is None:
                    continue
                pref_payload = build_preference_payload(
                    verdict,
                    claim_id=claim_id,
                    claim_payload=claim_payload,
                    speaker_entity_id=work.speaker_entity_id,
                )
                state.store.add_node(
                    claim_id,
                    labels=frozenset({LABEL_CLAIM, LABEL_PREFERENCE}),
                    payloads={LABEL_PREFERENCE: pref_payload},
                )
                state.store.add_edge(
                    work.speaker_entity_id,
                    claim_id,
                    EdgeAttrs(
                        type=EDGE_HOLDS_PREFERENCE,
                        weight=verdict.confidence,
                        source_turn_id=work.turn_id,
                        asserted_at=claim_payload.asserted_at,
                    ),
                )
                if pref_payload.target_id and state.store.has_node(pref_payload.target_id):
                    state.store.add_edge(
                        claim_id,
                        pref_payload.target_id,
                        EdgeAttrs(
                            type=EDGE_ABOUT,
                            weight=1.0,
                            source_turn_id=work.turn_id,
                            asserted_at=claim_payload.asserted_at,
                        ),
                    )

    def _turn_ids_in_order(self, state: ConversationState) -> list[str]:
        ids: list[tuple[int, int, str]] = []
        for node_id_, attrs in state.store.iter_nodes():
            if LABEL_TURN not in attrs.get("labels", frozenset()):
                continue
            payload: TurnPayload = attrs[LABEL_TURN]
            ids.append((payload.session_index, payload.turn_index, node_id_))
        ids.sort()
        return [nid for _, _, nid in ids]


__all__ = [
    "ConversationState",
    "IngestionPipeline",
    "SPEAKER_ENTITY_TYPE",
]
