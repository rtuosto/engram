"""Node and edge payload dataclasses plus content-addressed identity.

The graph's interior types. See ``docs/design/ingestion.md §5`` for the full
rationale and identity-field table.

**All payloads are ``@dataclass(frozen=True, slots=True)``** — frozen for
``R2`` determinism (no hidden mutation), slotted for memory. Mutability lives
on the ``MultiDiGraph`` wrapper, not on payloads.

**Node IDs are content-addressed.** ``sha256(canonical_form).hexdigest()[:16]``
where ``canonical_form`` is a sorted-key JSON string over the node type's
*identity fields*. Same identity fields in any process → same ID, insertion
order irrelevant. This is the R2 story for node identity.

**R16 discipline.** Primary primitives (Entity / Claim / Preference / N-gram)
are content-addressed and never mutated after creation. Anything that would
look like mutation (alias accumulation, reinforcement counts, preference
polarity flips) is a derived index rebuilt from primary — see
``engram/ingestion/derived.py`` (landed in PR-D).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Node type labels — canonical strings. Match manifesto §3 node taxonomy.
# ---------------------------------------------------------------------------

LABEL_MEMORY: Final[str] = "memory"
LABEL_TURN: Final[str] = "turn"
LABEL_UTTERANCE_SEGMENT: Final[str] = "utterance_segment"
LABEL_NGRAM: Final[str] = "ngram"
LABEL_ENTITY: Final[str] = "entity"
LABEL_CLAIM: Final[str] = "claim"
LABEL_PREFERENCE: Final[str] = "preference"
LABEL_TIME_ANCHOR: Final[str] = "time_anchor"
LABEL_EVENT: Final[str] = "event"
LABEL_EPISODE: Final[str] = "episode"

ALL_NODE_LABELS: Final[frozenset[str]] = frozenset({
    LABEL_MEMORY,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LABEL_NGRAM,
    LABEL_ENTITY,
    LABEL_CLAIM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_EVENT,
    LABEL_EPISODE,
})

# ---------------------------------------------------------------------------
# Layer labels — content-classification labels on nodes (§3).
#
# ``semantic`` is deliberately *not* a node label: it lives in the parallel
# granule embedding store (PR-C), not on the graph node. Granules (Memory /
# Turn / UtteranceSegment / N-gram) therefore carry an empty ``layers`` set
# in PR-B; Entity / Claim / Preference carry their classification label.
# ---------------------------------------------------------------------------

LAYER_ENTITY: Final[str] = "entity"
LAYER_RELATIONSHIP: Final[str] = "relationship"
LAYER_TEMPORAL: Final[str] = "temporal"
LAYER_EPISODIC: Final[str] = "episodic"

ALL_LAYER_LABELS: Final[frozenset[str]] = frozenset({
    LAYER_ENTITY,
    LAYER_RELATIONSHIP,
    LAYER_TEMPORAL,
    LAYER_EPISODIC,
})

# ---------------------------------------------------------------------------
# N-gram kinds — one string per extractor that emits n-grams. Feeds the
# N-gram node's identity so two extractors emitting the same phrase from
# the same Sentence produce distinct nodes (they are distinct observations
# about that phrase, coming from different parses).
# ---------------------------------------------------------------------------

NGRAM_KIND_NOUN_CHUNK: Final[str] = "noun_chunk"
NGRAM_KIND_SVO: Final[str] = "svo"

ALL_NGRAM_KINDS: Final[frozenset[str]] = frozenset({
    NGRAM_KIND_NOUN_CHUNK,
    NGRAM_KIND_SVO,
})

# ---------------------------------------------------------------------------
# Edge type labels — Tier 1 (primary only). Derived edge types land in PR-D.
# ---------------------------------------------------------------------------

EDGE_PART_OF: Final[str] = "part_of"
EDGE_MENTIONS: Final[str] = "mentions"
EDGE_ASSERTS: Final[str] = "asserts"
EDGE_HOLDS_PREFERENCE: Final[str] = "holds_preference"
EDGE_ABOUT: Final[str] = "about"
EDGE_TEMPORAL_AT: Final[str] = "temporal_at"

# Derived edge types (emitted by :mod:`engram.ingestion.derived`, not the
# ingest pipeline). ``temporal_before`` / ``temporal_after`` chain the
# TimeAnchors in sorted-ISO order; ``co_occurs_with`` links Entity pairs
# mentioned together under a shared Memory. In PR-D these live in sidecar
# indexes rather than the graph itself (see ``derived.py``).
EDGE_TEMPORAL_BEFORE: Final[str] = "temporal_before"
EDGE_TEMPORAL_AFTER: Final[str] = "temporal_after"
EDGE_CO_OCCURS_WITH: Final[str] = "co_occurs_with"

TIER_1_EDGE_TYPES: Final[frozenset[str]] = frozenset({
    EDGE_PART_OF,
    EDGE_MENTIONS,
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_ABOUT,
    EDGE_TEMPORAL_AT,
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
class MemoryPayload:
    """One ingest event's raw payload — the agent's original submission.

    ``memory_index`` is a monotonic per-instance counter assigned at ingest.
    It is the Memory's identity — same content in two ``ingest`` calls
    produces two distinct Memory nodes (``R16``: Memories are never
    deduplicated; each observation is an event in its own right).
    """

    memory_index: int
    content: str
    timestamp: str | None
    speaker: str | None
    source: str | None
    metadata: tuple[tuple[str, str], ...]  # sorted for R2


@dataclass(frozen=True, slots=True)
class TurnPayload:
    """Whole-Memory granule. Participates in the granularity hierarchy and
    (in PR-C) carries the whole-Memory embedding.

    In the engram-as-tool model, one ``ingest(memory)`` call creates one
    Turn granule shadowing the Memory; the ``memory_id`` links the two.
    """

    memory_id: str
    text: str
    speaker: str | None
    timestamp: str | None


@dataclass(frozen=True, slots=True)
class UtteranceSegmentPayload:
    """A sentence-level span inside a Turn. Produced by spaCy ``Doc.sents``."""

    text: str
    turn_id: str
    segment_index: int
    char_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class NgramPayload:
    """A key-phrase granule inside an UtteranceSegment (§3, §5).

    Two extractors emit n-grams: noun chunks (``doc.noun_chunks``) and SVO
    dependency subtrees (subject + root verb + object). ``ngram_kind``
    distinguishes them; ``normalized_text`` is the R2-stable identity key
    (lowercase NFKC); ``surface_form`` preserves the original casing for
    recall-time display.
    """

    normalized_text: str
    surface_form: str
    segment_id: str
    ngram_kind: str  # "noun_chunk" | "svo"
    char_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class EntityPayload:
    """A canonicalized real-world entity mention.

    **No ``aliases`` field.** R16: primary primitives do not carry
    observation-derived data. Aliases are a derived index, rebuilt from the
    inbound ``mentions`` edges by :mod:`engram.ingestion.derived` (PR-D).
    """

    canonical_form: str
    entity_type: str


@dataclass(frozen=True, slots=True)
class ClaimPayload:
    """A speaker's assertion at a point in time — not a world truth.

    ``subject_id`` / ``object_id`` reference Entity node IDs when the parser
    resolves them to a canonical entity; ``object_literal`` carries values
    that don't resolve to an Entity (numbers, dates, quoted strings).

    Claims are content-addressed by ``(subject_id, predicate,
    object_id_or_literal)`` — the same assertion from two different
    Memories produces two ``asserts`` edges into the same Claim node (R16
    reinforcement via edge enumeration).
    """

    subject_id: str
    predicate: str
    object_id: str | None
    object_literal: str | None
    modality: str  # asserted | negated | hypothetical | interrogative
    tense: str     # past | present | future | habitual


@dataclass(frozen=True, slots=True)
class PreferencePayload:
    """A separate content-addressed Preference node (not a Claim overlay).

    Content-addressed by ``(holder_id, polarity, target_id_or_literal)``.
    Observations are ``holds_preference`` edges from the speaker Entity;
    reinforcement counts and "what does X currently think about Y" are
    derived (PR-D) — not stored on the payload.
    """

    holder_id: str
    polarity: str              # likes | dislikes | wants | avoids | commits_to | rejects
    target_id: str | None
    target_literal: str | None


@dataclass(frozen=True, slots=True)
class TimeAnchorPayload:
    """A shared temporal reference for granules + relationships (``§3``).

    Content-addressed by the ISO-8601 timestamp *rounded* to the configured
    resolution (see :data:`engram.config.MemoryConfig.time_anchor_resolution`).
    Multiple observations that share a rounded timestamp converge onto one
    anchor; recall walks these anchors to answer "everything observed in
    March" or "relationships established before this memory."
    """

    iso_timestamp: str  # rounded to configured resolution


@dataclass(frozen=True, slots=True)
class EventPayload:
    """Tier 2 — not extracted in this PR. Shape declared for schema completeness."""

    canonical_description: str
    interval_start: str | None
    interval_end: str | None
    participant_ids: tuple[str, ...]  # sorted


@dataclass(frozen=True, slots=True)
class EpisodePayload:
    """Tier 2 — not extracted in this PR. Derived by the episodic clusterer (PR-D)."""

    entity_id: str
    topic_signature: str
    time_window_start: str
    time_window_end: str
    member_granule_count: int


# ---------------------------------------------------------------------------
# Edge attributes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgeAttrs:
    """Attributes on a typed edge.

    ``weight`` is evidence strength in ``[0, 1]``. The graph stores
    normalized evidence; recall composes it with per-intent weight profiles
    at recall-planning time. Do not stuff recall weights here.

    ``source_memory_id`` is the provenance anchor — every primary edge
    traces to exactly one Memory. ``source_turn_id`` is retained as the
    most-specific granule ID that generated the edge (Turn, Sentence, or
    N-gram in future PRs).

    ``surface_form`` is the raw pre-canonicalization text span that generated
    this observation — populated on ``mentions`` edges so the derived
    alias-set rebuild (PR-D) can walk entity-inbound mentions and collect
    distinct surface forms without replaying the NER stage.
    """

    type: str
    weight: float = 1.0
    source_memory_id: str | None = None
    source_turn_id: str | None = None
    asserted_at: str | None = None
    surface_form: str | None = None


# ---------------------------------------------------------------------------
# Identity constructors — thin helpers that build the identity dict, then
# node_id(). Keeping them here (not on payload classes) means the payloads
# stay pure data.
# ---------------------------------------------------------------------------


def memory_identity(memory_index: int) -> dict[str, object]:
    return {
        "type": LABEL_MEMORY,
        "memory_index": memory_index,
    }


def turn_identity(memory_id: str) -> dict[str, object]:
    return {
        "type": LABEL_TURN,
        "memory_id": memory_id,
    }


def segment_identity(turn_id: str, segment_index: int) -> dict[str, object]:
    return {
        "type": LABEL_UTTERANCE_SEGMENT,
        "turn_id": turn_id,
        "segment_index": segment_index,
    }


def ngram_identity(
    segment_id: str, ngram_kind: str, normalized_text: str
) -> dict[str, object]:
    """N-gram identity — same normalized phrase + same kind + same Sentence
    converges to one node. N-grams are scoped to their containing Sentence
    so that otherwise-identical phrases in two Sentences are two distinct
    granules (each with its own semantic embedding and ``part_of`` edge).
    """
    return {
        "type": LABEL_NGRAM,
        "segment_id": segment_id,
        "ngram_kind": ngram_kind,
        "normalized_text": normalized_text,
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
) -> dict[str, object]:
    """Claim identity. R16: same assertion content → same Claim node.

    Note: ``asserted_by_turn_id`` is NOT part of identity. Two Memories
    asserting the same thing must converge to one Claim node; the source
    granule is recorded on the ``asserts`` edge, not the Claim itself.
    """
    return {
        "type": LABEL_CLAIM,
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "object_literal": object_literal,
    }


def preference_identity(
    holder_id: str,
    polarity: str,
    target_id: str | None,
    target_literal: str | None,
) -> dict[str, object]:
    """Preference identity — content-addressed separately from Claim (R16)."""
    return {
        "type": LABEL_PREFERENCE,
        "holder_id": holder_id,
        "polarity": polarity,
        "target_id": target_id,
        "target_literal": target_literal,
    }


def time_anchor_identity(iso_timestamp: str) -> dict[str, object]:
    """TimeAnchor identity — one node per distinct rounded ISO timestamp.

    The caller is responsible for rounding to the configured resolution
    (:func:`round_iso_timestamp`). Two Memories that share the same rounded
    timestamp converge onto one anchor; recall walks anchors to group "all
    observations at this moment in time."
    """
    return {
        "type": LABEL_TIME_ANCHOR,
        "iso_timestamp": iso_timestamp,
    }


# ---------------------------------------------------------------------------
# TimeAnchor resolution.
# ---------------------------------------------------------------------------

TIME_ANCHOR_RESOLUTION_SECOND: Final[str] = "second"
TIME_ANCHOR_RESOLUTION_MINUTE: Final[str] = "minute"
TIME_ANCHOR_RESOLUTION_HOUR: Final[str] = "hour"
TIME_ANCHOR_RESOLUTION_DAY: Final[str] = "day"

ALL_TIME_ANCHOR_RESOLUTIONS: Final[frozenset[str]] = frozenset({
    TIME_ANCHOR_RESOLUTION_SECOND,
    TIME_ANCHOR_RESOLUTION_MINUTE,
    TIME_ANCHOR_RESOLUTION_HOUR,
    TIME_ANCHOR_RESOLUTION_DAY,
})


def round_iso_timestamp(iso_timestamp: str, resolution: str) -> str:
    """Round an ISO-8601 timestamp to the configured resolution.

    Accepts the subset of ISO-8601 the ingest API accepts (
    ``YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]``). The output preserves the
    original timezone suffix where possible.

    Unrecognized formats raise :class:`ValueError` — ingest upstream should
    fail closed on malformed timestamps before they reach this function.
    """
    if resolution not in ALL_TIME_ANCHOR_RESOLUTIONS:
        raise ValueError(
            f"unknown time_anchor_resolution={resolution!r}; "
            f"expected one of {sorted(ALL_TIME_ANCHOR_RESOLUTIONS)}"
        )

    from datetime import datetime

    # Normalize a trailing "Z" to "+00:00" for fromisoformat; preserve the
    # exact suffix the caller supplied so anchors with offsets aren't
    # silently renormalized to UTC.
    suffix = ""
    body = iso_timestamp
    if body.endswith("Z"):
        body = body[:-1] + "+00:00"
        suffix = "Z"
    try:
        dt = datetime.fromisoformat(body)
    except ValueError as exc:
        raise ValueError(
            f"cannot round non-ISO-8601 timestamp {iso_timestamp!r}: {exc}"
        ) from exc

    if resolution == TIME_ANCHOR_RESOLUTION_DAY:
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif resolution == TIME_ANCHOR_RESOLUTION_HOUR:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    elif resolution == TIME_ANCHOR_RESOLUTION_MINUTE:
        dt = dt.replace(second=0, microsecond=0)
    else:  # second
        dt = dt.replace(microsecond=0)

    rendered = dt.isoformat()
    if suffix == "Z" and rendered.endswith("+00:00"):
        rendered = rendered[: -len("+00:00")] + "Z"
    return rendered
