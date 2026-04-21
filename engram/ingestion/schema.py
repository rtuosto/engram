"""Node and edge payload dataclasses plus content-addressed identity.

The graph's interior types. See ``docs/design/ingestion.md §2`` for the full
rationale and identity-field table.

**All payloads are ``@dataclass(frozen=True, slots=True)``** — frozen for
``R2`` determinism (no hidden mutation), slotted for memory. Mutability lives
on the ``MultiDiGraph`` wrapper, not on payloads.

**Node IDs are content-addressed.** ``sha256(canonical_form).hexdigest()[:16]``
where ``canonical_form`` is a sorted-key JSON string over the node type's
*identity fields*. Same identity fields in any process → same ID, insertion
order irrelevant. This is the R2 story for node identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Node type labels — canonical strings. Match manifesto §3 node taxonomy.
# ---------------------------------------------------------------------------

LABEL_TURN: Final[str] = "turn"
LABEL_UTTERANCE_SEGMENT: Final[str] = "utterance_segment"
LABEL_ENTITY: Final[str] = "entity"
LABEL_CLAIM: Final[str] = "claim"
LABEL_PREFERENCE: Final[str] = "preference"
LABEL_EVENT: Final[str] = "event"
LABEL_EPISODE: Final[str] = "episode"
LABEL_SESSION: Final[str] = "session"

ALL_NODE_LABELS: Final[frozenset[str]] = frozenset({
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LABEL_ENTITY,
    LABEL_CLAIM,
    LABEL_PREFERENCE,
    LABEL_EVENT,
    LABEL_EPISODE,
    LABEL_SESSION,
})

# ---------------------------------------------------------------------------
# Edge type labels — Tier 1 only. Tier 2 / Tier 3 ship in later iterations
# (docs/design/ingestion.md §2).
# ---------------------------------------------------------------------------

EDGE_PART_OF: Final[str] = "part_of"
EDGE_MENTIONS: Final[str] = "mentions"
EDGE_ASSERTS: Final[str] = "asserts"
EDGE_HOLDS_PREFERENCE: Final[str] = "holds_preference"
EDGE_ABOUT: Final[str] = "about"
EDGE_CO_OCCURS_WITH: Final[str] = "co_occurs_with"
EDGE_TEMPORAL_BEFORE: Final[str] = "temporal_before"
EDGE_TEMPORAL_AFTER: Final[str] = "temporal_after"

TIER_1_EDGE_TYPES: Final[frozenset[str]] = frozenset({
    EDGE_PART_OF,
    EDGE_MENTIONS,
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_ABOUT,
    EDGE_CO_OCCURS_WITH,
    EDGE_TEMPORAL_BEFORE,
    EDGE_TEMPORAL_AFTER,
})

# ---------------------------------------------------------------------------
# Preference polarities.
# ---------------------------------------------------------------------------

POLARITY_LIKES: Final[str] = "likes"
POLARITY_DISLIKES: Final[str] = "dislikes"
POLARITY_WANTS: Final[str] = "wants"
POLARITY_AVOIDS: Final[str] = "avoids"
POLARITY_COMMITS_TO: Final[str] = "commits_to"
POLARITY_REJECTS: Final[str] = "rejects"

PREFERENCE_POLARITIES: Final[tuple[str, ...]] = (
    POLARITY_LIKES,
    POLARITY_DISLIKES,
    POLARITY_WANTS,
    POLARITY_AVOIDS,
    POLARITY_COMMITS_TO,
    POLARITY_REJECTS,
)

# ---------------------------------------------------------------------------
# Content-addressed node ID.
# ---------------------------------------------------------------------------

NODE_ID_BYTES: Final[int] = 8  # 16 hex chars = 64-bit; collision-safe at our scale


def node_id(identity: dict[str, object]) -> str:
    """Deterministic content-addressed ID from a node's identity fields.

    ``identity`` is a plain dict; JSON-serialized with ``sort_keys=True`` for
    R2 determinism. Values must be JSON-safe — use plain str / int / None /
    nested dicts / tuples (tuples serialize as lists; that's fine because
    identity fields are all positional).
    """
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[: NODE_ID_BYTES * 2]


# ---------------------------------------------------------------------------
# Node payload dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnPayload:
    """One speaker utterance; the raw text unit from the dataset."""

    speaker: str
    text: str
    conversation_id: str
    session_index: int
    turn_index: int
    timestamp: str | None  # ISO-8601 if provided; never wall-clock


@dataclass(frozen=True, slots=True)
class UtteranceSegmentPayload:
    """A sentence-level span inside a Turn. Produced by spaCy ``Doc.sents``."""

    text: str
    turn_id: str
    segment_index: int
    char_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class EntityPayload:
    """A canonicalized real-world entity mention.

    ``aliases`` is the deterministic-sorted tuple of surface forms observed
    for this entity (stored as a tuple rather than a set for R2).
    """

    canonical_form: str
    entity_type: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimPayload:
    """A speaker's assertion at a point in time — not a world truth.

    ``subject_id`` / ``object_id`` reference Entity node IDs when the parser
    resolves them to a canonical entity; ``object_literal`` carries values
    that don't resolve to an Entity (numbers, dates, quoted strings).
    """

    subject_id: str
    predicate: str
    object_id: str | None
    object_literal: str | None
    asserted_by_turn_id: str
    asserted_at: str | None
    modality: str  # asserted | negated | hypothetical | interrogative
    tense: str     # past | present | future | habitual


@dataclass(frozen=True, slots=True)
class PreferencePayload:
    """A Preference overlay on a Claim (multi-label ``{claim, preference}``)."""

    holder_id: str
    polarity: str
    target_id: str | None
    target_literal: str | None
    source_claim_id: str
    confidence: float  # centroid-discrimination score, [0, 1]


@dataclass(frozen=True, slots=True)
class EventPayload:
    """Tier 2 — not extracted in the first PR. Shape declared for schema completeness."""

    canonical_description: str
    interval_start: str | None
    interval_end: str | None
    participant_ids: tuple[str, ...]  # sorted


@dataclass(frozen=True, slots=True)
class EpisodePayload:
    """Tier 2 — not extracted in the first PR."""

    conversation_id: str
    cluster_id: int
    member_turn_ids: tuple[str, ...]  # sorted
    summary: str | None


@dataclass(frozen=True, slots=True)
class SessionPayload:
    """A conversation slice; the unit of ``MemorySystem.ingest_session``."""

    conversation_id: str
    session_index: int
    timestamp: str | None


# ---------------------------------------------------------------------------
# Edge attributes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgeAttrs:
    """Attributes on a typed edge. ``weight`` is evidence strength in ``[0, 1]``.

    The graph stores normalized evidence; recall composes it with per-intent
    weight profiles at recall-planning time. Do not stuff recall weights here.
    """

    type: str
    weight: float = 1.0
    source_turn_id: str | None = None
    asserted_at: str | None = None


# ---------------------------------------------------------------------------
# Identity constructors — thin helpers that build the identity dict, then
# node_id(). Keeping them here (not on payload classes) means the payloads
# stay pure data.
# ---------------------------------------------------------------------------


def turn_identity(conversation_id: str, session_index: int, turn_index: int) -> dict[str, object]:
    return {
        "type": LABEL_TURN,
        "conversation_id": conversation_id,
        "session_index": session_index,
        "turn_index": turn_index,
    }


def segment_identity(turn_id: str, segment_index: int) -> dict[str, object]:
    return {
        "type": LABEL_UTTERANCE_SEGMENT,
        "turn_id": turn_id,
        "segment_index": segment_index,
    }


def entity_identity(canonical_form: str, entity_type: str) -> dict[str, object]:
    return {
        "type": LABEL_ENTITY,
        "canonical_form": canonical_form,
        "entity_type": entity_type,
    }


def claim_identity(
    subject_id: str,
    predicate: str,
    object_id: str | None,
    object_literal: str | None,
    asserted_by_turn_id: str,
) -> dict[str, object]:
    return {
        "type": LABEL_CLAIM,
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "object_literal": object_literal,
        "asserted_by_turn_id": asserted_by_turn_id,
    }


def session_identity(conversation_id: str, session_index: int) -> dict[str, object]:
    return {
        "type": LABEL_SESSION,
        "conversation_id": conversation_id,
        "session_index": session_index,
    }
