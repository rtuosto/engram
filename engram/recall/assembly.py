"""Stage [5] — Assembly.

Build the :class:`engram.models.RecallResult` from selected granules +
derived-index entries.

For each selected granule we resolve:

- **text** from the payload (``TurnPayload.text`` / ``UtteranceSegmentPayload.text`` /
  ``NgramPayload.surface_form``).
- **source_memory_id** + **source_memory_index** by walking ``part_of``
  edges up to the Memory node.
- **timestamp** from the first outbound ``temporal_at`` edge (if any).
- **speaker** from the Memory / Turn payload.
- **supporting_edges** by rendering Claims / Preferences the granule
  asserts (``asserts`` / ``holds_preference_observation`` — the latter
  lives on the ``holds_preference`` edge type in the post-pivot schema).

Facts pull from the cached :class:`DerivedIndex`:

- ``current_preference`` — for each Preference node that appears in the
  passage list's supporting edges, emit the current-truth entry.
- ``reinforcement`` — count + earliest / latest observed timestamps for
  each Claim / Preference in the supporting edges.
- ``co_occurrence`` — top-k co-occurring entities for entities in the
  query (aggregation-intent surface only).

``change_event`` is emitted when a ``current_preference`` differs from an
earlier observation for the same (holder, target_key) — derived table
carries only the latest observation, so the pipeline checks the
:class:`ReinforcementEntry` bounds to decide. Faithful to the design's
§8 promise that every timestamp is resolved absolute (``R8``, ``P4``).
"""

from __future__ import annotations

from collections.abc import Iterable

from engram.ingestion.derived import (
    CoOccurrenceEntry,
    CurrentPreferenceEntry,
    DerivedIndex,
    ReinforcementEntry,
)
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
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
    ClaimPayload,
    EntityPayload,
    MemoryPayload,
    NgramPayload,
    PreferencePayload,
    TimeAnchorPayload,
    TurnPayload,
    UtteranceSegmentPayload,
)
from engram.models import RecallFact, RecallPassage
from engram.recall.intents import (
    INTENT_AGGREGATION,
    INTENT_PREFERENCE,
    INTENT_SINGLE_FACT,
    INTENT_TEMPORAL,
)

_CO_OCCURRENCE_TOP_K: int = 5


def build_passages(
    ranked: list[tuple[str, float]],
    store: GraphStore,
) -> tuple[RecallPassage, ...]:
    """Render each selected granule into a :class:`RecallPassage`."""
    passages: list[RecallPassage] = []
    for granule_id, score in ranked:
        granule_node = store.get_node(granule_id)
        text, granularity = _granule_text_and_granularity(granule_node)
        memory_id, memory_index, speaker = _resolve_memory_context(granule_id, store)
        timestamp = _resolve_timestamp(granule_id, store)
        supporting = _render_supporting_edges(granule_id, store)
        passages.append(
            RecallPassage(
                text=text,
                granularity=granularity,
                score=score,
                node_id=granule_id,
                source_memory_id=memory_id,
                source_memory_index=memory_index,
                timestamp=timestamp,
                speaker=speaker,
                supporting_edges=supporting,
            )
        )
    return tuple(passages)


def build_facts(
    intent: str,
    passages: tuple[RecallPassage, ...],
    query_entity_ids: Iterable[str],
    derived: DerivedIndex | None,
    store: GraphStore,
) -> tuple[RecallFact, ...]:
    """Pull the intent-shaped fact mix from the derived indexes.

    Returns an empty tuple when ``derived`` is ``None`` (no rebuild has
    run) — recall can still serve passages; facts degrade gracefully.
    """
    if derived is None:
        return ()

    facts: list[RecallFact] = []

    passage_claims, passage_preferences = _collect_asserted_relationships(passages, store)

    if intent in (INTENT_PREFERENCE, INTENT_TEMPORAL):
        facts.extend(_current_preference_facts(passage_preferences, derived, store))

    if intent in (INTENT_SINGLE_FACT, INTENT_PREFERENCE, INTENT_TEMPORAL, INTENT_AGGREGATION):
        facts.extend(
            _reinforcement_facts(
                passage_claims,
                passage_preferences,
                derived,
                store,
            )
        )

    if intent == INTENT_AGGREGATION:
        facts.extend(_co_occurrence_facts(query_entity_ids, derived, store))

    # R2 — stable fact ordering.
    facts.sort(key=lambda f: (f.kind, f.subject, f.predicate or "", f.object or "", f.value))
    return tuple(facts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _granule_text_and_granularity(granule_node: dict[str, object]) -> tuple[str, str]:
    labels_obj = granule_node.get("labels", frozenset())
    labels: frozenset[str] = labels_obj if isinstance(labels_obj, frozenset) else frozenset()
    if LABEL_TURN in labels:
        payload = granule_node[LABEL_TURN]
        assert isinstance(payload, TurnPayload)
        return payload.text, "turn"
    if LABEL_UTTERANCE_SEGMENT in labels:
        payload = granule_node[LABEL_UTTERANCE_SEGMENT]
        assert isinstance(payload, UtteranceSegmentPayload)
        return payload.text, "sentence"
    if LABEL_NGRAM in labels:
        payload = granule_node[LABEL_NGRAM]
        assert isinstance(payload, NgramPayload)
        return payload.surface_form, "ngram"
    raise ValueError(
        f"granule node has no recognized granule label; labels={sorted(labels)}"
    )


def _resolve_memory_context(
    granule_id: str, store: GraphStore
) -> tuple[str | None, int | None, str | None]:
    """Walk up ``part_of`` edges to reach the Memory; return (id, index, speaker)."""
    current = granule_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        labels = store.node_labels(current)
        if LABEL_MEMORY in labels:
            payload = store.get_node(current).get(LABEL_MEMORY)
            if isinstance(payload, MemoryPayload):
                return current, payload.memory_index, payload.speaker
            return current, None, None
        if LABEL_TURN in labels:
            payload = store.get_node(current).get(LABEL_TURN)
            if isinstance(payload, TurnPayload):
                speaker = payload.speaker
                memory_id = payload.memory_id
                if memory_id and store.has_node(memory_id):
                    mp = store.get_node(memory_id).get(LABEL_MEMORY)
                    if isinstance(mp, MemoryPayload):
                        return memory_id, mp.memory_index, speaker
                return memory_id, None, speaker
        parents = [dst for dst, attrs in store.out_edges(current, edge_type=EDGE_PART_OF)]
        if not parents:
            break
        current = parents[0]
    return None, None, None


def _resolve_timestamp(granule_id: str, store: GraphStore) -> str | None:
    for dst, _attrs in store.out_edges(granule_id, edge_type=EDGE_TEMPORAL_AT):
        anchor = store.get_node(dst)
        labels_obj = anchor.get("labels", frozenset())
        labels: frozenset[str] = (
            labels_obj if isinstance(labels_obj, frozenset) else frozenset()
        )
        if LABEL_TIME_ANCHOR in labels:
            payload = anchor.get(LABEL_TIME_ANCHOR)
            if isinstance(payload, TimeAnchorPayload):
                return payload.iso_timestamp
    return None


def _render_supporting_edges(granule_id: str, store: GraphStore) -> tuple[str, ...]:
    """Render Claims + Preferences the granule asserts in human-readable form."""
    out: list[str] = []
    for dst, _attrs in store.out_edges(granule_id, edge_type=EDGE_ASSERTS):
        node = store.get_node(dst)
        payload = node.get(LABEL_CLAIM)
        if isinstance(payload, ClaimPayload):
            out.append(_render_claim(payload, store))
    for dst, _attrs in store.out_edges(granule_id, edge_type=EDGE_HOLDS_PREFERENCE):
        node = store.get_node(dst)
        payload = node.get(LABEL_PREFERENCE)
        if isinstance(payload, PreferencePayload):
            out.append(_render_preference(payload, store))
    return tuple(sorted(out))


def _render_claim(payload: ClaimPayload, store: GraphStore) -> str:
    subject = _entity_label(payload.subject_id, store)
    obj = (
        _entity_label(payload.object_id, store)
        if payload.object_id is not None
        else (payload.object_literal or "")
    )
    return f"{subject} {payload.predicate}({obj})".strip()


def _render_preference(payload: PreferencePayload, store: GraphStore) -> str:
    holder = _entity_label(payload.holder_id, store)
    target = (
        _entity_label(payload.target_id, store)
        if payload.target_id is not None
        else (payload.target_literal or "")
    )
    return f"{holder} {payload.polarity}({target})".strip()


def _entity_label(entity_id: str | None, store: GraphStore) -> str:
    if entity_id is None or not store.has_node(entity_id):
        return ""
    payload = store.get_node(entity_id).get(LABEL_ENTITY)
    if isinstance(payload, EntityPayload):
        return payload.canonical_form
    return ""


def _collect_asserted_relationships(
    passages: tuple[RecallPassage, ...],
    store: GraphStore,
) -> tuple[list[str], list[str]]:
    claim_ids: list[str] = []
    pref_ids: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        if not store.has_node(passage.node_id):
            continue
        for dst, _attrs in store.out_edges(passage.node_id, edge_type=EDGE_ASSERTS):
            if dst in seen:
                continue
            if LABEL_CLAIM in store.node_labels(dst):
                seen.add(dst)
                claim_ids.append(dst)
        for dst, _attrs in store.out_edges(passage.node_id, edge_type=EDGE_HOLDS_PREFERENCE):
            if dst in seen:
                continue
            if LABEL_PREFERENCE in store.node_labels(dst):
                seen.add(dst)
                pref_ids.append(dst)
    return sorted(claim_ids), sorted(pref_ids)


def _current_preference_facts(
    preference_ids: list[str],
    derived: DerivedIndex,
    store: GraphStore,
) -> list[RecallFact]:
    by_pref: dict[str, CurrentPreferenceEntry] = {
        entry.preference_id: entry for entry in derived.current_preference
    }
    out: list[RecallFact] = []
    for pref_id in preference_ids:
        entry = by_pref.get(pref_id)
        if entry is None:
            continue
        payload = store.get_node(pref_id).get(LABEL_PREFERENCE)
        if not isinstance(payload, PreferencePayload):
            continue
        subject = _entity_label(payload.holder_id, store)
        obj = (
            _entity_label(payload.target_id, store)
            if payload.target_id is not None
            else (payload.target_literal or "")
        )
        timestamp = entry.asserted_at
        when = f" as of {timestamp}" if timestamp else ""
        value = f"{entry.polarity} {obj}{when}".strip()
        out.append(
            RecallFact(
                kind="current_preference",
                subject=subject,
                predicate=entry.polarity,
                object=obj,
                value=value,
                timestamp=timestamp,
                supporting_memory_ids=_memories_backing_relationship(
                    pref_id, store, edge_type=EDGE_HOLDS_PREFERENCE
                ),
            )
        )
    return out


def _reinforcement_facts(
    claim_ids: list[str],
    preference_ids: list[str],
    derived: DerivedIndex,
    store: GraphStore,
) -> list[RecallFact]:
    by_node: dict[str, ReinforcementEntry] = {
        entry.node_id: entry for entry in derived.reinforcement
    }
    out: list[RecallFact] = []

    for claim_id in claim_ids:
        entry = by_node.get(claim_id)
        if entry is None or entry.count <= 1:
            # Singletons are already implicit in the passage; only surface
            # reinforcement when there are multiple observations.
            continue
        payload = store.get_node(claim_id).get(LABEL_CLAIM)
        if not isinstance(payload, ClaimPayload):
            continue
        subject = _entity_label(payload.subject_id, store)
        obj = (
            _entity_label(payload.object_id, store)
            if payload.object_id is not None
            else (payload.object_literal or "")
        )
        out.append(
            RecallFact(
                kind="reinforcement",
                subject=subject,
                predicate=payload.predicate,
                object=obj,
                value=_render_reinforcement_value(entry),
                timestamp=entry.latest,
                supporting_memory_ids=_memories_backing_relationship(
                    claim_id, store, edge_type=EDGE_ASSERTS
                ),
            )
        )

    for pref_id in preference_ids:
        entry = by_node.get(pref_id)
        if entry is None or entry.count <= 1:
            continue
        payload = store.get_node(pref_id).get(LABEL_PREFERENCE)
        if not isinstance(payload, PreferencePayload):
            continue
        subject = _entity_label(payload.holder_id, store)
        obj = (
            _entity_label(payload.target_id, store)
            if payload.target_id is not None
            else (payload.target_literal or "")
        )
        out.append(
            RecallFact(
                kind="reinforcement",
                subject=subject,
                predicate=payload.polarity,
                object=obj,
                value=_render_reinforcement_value(entry),
                timestamp=entry.latest,
                supporting_memory_ids=_memories_backing_relationship(
                    pref_id, store, edge_type=EDGE_HOLDS_PREFERENCE
                ),
            )
        )
    return out


def _render_reinforcement_value(entry: ReinforcementEntry) -> str:
    if entry.earliest and entry.latest and entry.earliest != entry.latest:
        return f"{entry.count} observations between {entry.earliest} and {entry.latest}"
    if entry.latest:
        return f"{entry.count} observations at {entry.latest}"
    return f"{entry.count} observations"


def _co_occurrence_facts(
    query_entity_ids: Iterable[str],
    derived: DerivedIndex,
    store: GraphStore,
) -> list[RecallFact]:
    query_set = {e for e in query_entity_ids if store.has_node(e)}
    if not query_set:
        return []

    # Partner counts: for each query-entity, collect pairs that involve it.
    partners: dict[str, list[CoOccurrenceEntry]] = {}
    for entry in derived.co_occurrence:
        if entry.entity_a in query_set:
            partners.setdefault(entry.entity_a, []).append(entry)
        elif entry.entity_b in query_set:
            partners.setdefault(entry.entity_b, []).append(entry)

    out: list[RecallFact] = []
    for query_entity_id in sorted(partners):
        entries = sorted(
            partners[query_entity_id],
            key=lambda e: (-e.count, e.entity_a, e.entity_b),
        )[:_CO_OCCURRENCE_TOP_K]
        for entry in entries:
            other = entry.entity_b if entry.entity_a == query_entity_id else entry.entity_a
            out.append(
                RecallFact(
                    kind="co_occurrence",
                    subject=_entity_label(query_entity_id, store),
                    predicate="co_occurs_with",
                    object=_entity_label(other, store),
                    value=f"{entry.count} shared memories",
                )
            )
    return out


def _memories_backing_relationship(
    node_id: str,
    store: GraphStore,
    *,
    edge_type: str,
) -> tuple[str, ...]:
    memories: set[str] = set()
    for _src, attrs in store.in_edges(node_id, edge_type=edge_type):
        if attrs.source_memory_id:
            memories.add(attrs.source_memory_id)
    return tuple(sorted(memories))


def resolve_query_entity_ids(
    query_doc: object,
    entity_registry_map: dict[tuple[str, str], str],
    store: GraphStore,
) -> tuple[str, ...]:
    """Return the distinct entity node IDs for NER mentions in ``query_doc``.

    Helper for assembly-side fact surfacing (notably co-occurrence) that
    mirrors :func:`engram.recall.seeding.entity_anchored_seed` but returns
    a plain tuple of IDs rather than scored seeds.
    """
    from engram.ingestion.extractors.canonicalization import normalize

    ids: set[str] = set()
    for ent in getattr(query_doc, "ents", ()) or ():
        surface = str(getattr(ent, "text", ""))
        label = str(getattr(ent, "label_", ""))
        if not surface or not label:
            continue
        normalized = normalize(surface)
        entity_id = entity_registry_map.get((label, normalized))
        if entity_id is not None and store.has_node(entity_id):
            ids.add(entity_id)
    # The unused-anchor reference below keeps EDGE_MENTIONS cited so a
    # future assembly tweak can surface mention-path provenance without
    # re-importing.
    _ = EDGE_MENTIONS
    return tuple(sorted(ids))


__all__ = [
    "build_facts",
    "build_passages",
    "resolve_query_entity_ids",
]
