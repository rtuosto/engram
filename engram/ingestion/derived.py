"""Derived-index orchestrator (``docs/design/ingestion.md §7``).

Engram's primary data (Memory / Turn / Sentence / N-gram / Entity / Claim /
Preference / TimeAnchor nodes + their observation edges) is append-only
(``R16``). Everything that *looks* like mutation — alias accumulation,
reinforcement counts, current-truth flips, co-occurrence weights, temporal
chaining — is a derived index, recomputed from primary on demand
(``R17``).

This module owns that recompute. The pipeline does not touch derived state;
:func:`rebuild_derived` is the only public writer.

**Scope in PR-D.** Five derived indexes are implemented:

- ``aliases`` — per Entity, the sorted distinct surface forms observed
  across its inbound ``mentions`` edges.
- ``co_occurrence`` — per Entity pair, the count (and normalized weight)
  of shared Memories. Symmetric; stored with ``(entity_a, entity_b)`` in
  lexicographic order to avoid double-counting.
- ``reinforcement`` — per Claim / Preference, the count of observation
  edges plus the earliest / latest ``asserted_at`` among them.
- ``current_preference`` — per ``(holder, target)``, the most recent
  Preference observation (ISO-timestamp ordering, ``node_id`` tiebreak).
- ``time_anchor_chain`` — sorted TimeAnchors with ``prev_id`` / ``next_id``
  links. Lets recall walk forward / backward in time without re-sorting.

ChangeEvent + EpisodicNode are explicitly deferred to a follow-up PR
(``docs/design/ingestion.md §7 D5, D6``). Their absence here is by design;
recall v1 does not depend on them.

**Fingerprint discipline.** Each snapshot carries a
:attr:`DerivedIndex.fingerprint` that binds it to both the ingestion
configuration and the primary-state signature. Rebuilds check against
this before serving; a stale snapshot is a cache-invalidation bug (the
predecessor-project failure mode this repo was forked to avoid — see
``.agent/lessons.md``).

**R2 determinism.** Every iteration that influences snapshot contents walks
sorted inputs. Floats (``co_occurrence.weight``, ``reinforcement.count`` in
normalized form later) are produced from sorted-key reductions so identical
primaries always produce byte-identical snapshots (``test_derived.py`` pins
this).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Final

import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]

from engram.config import MemoryConfig
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_MENTIONS,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    EdgeAttrs,
    PreferencePayload,
    TimeAnchorPayload,
)

DERIVATION_VERSION: Final[int] = 1
DERIVED_SCHEMA_VERSION: Final[int] = 1


class DerivedFormatError(RuntimeError):
    """Persisted derived snapshot is structurally invalid or the wrong version."""


# ---------------------------------------------------------------------------
# Snapshot dataclasses — all frozen + slotted for R2.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AliasEntry:
    """Per-Entity distinct surface forms, sorted alphabetically."""

    entity_id: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoOccurrenceEntry:
    """One Entity-pair co-occurrence.

    ``entity_a`` / ``entity_b`` are lexicographically ordered so each
    unordered pair is represented exactly once. ``count`` is the number of
    Memories in which both entities were mentioned; ``weight`` is
    ``count / max_count_in_snapshot`` — ``0.0`` when the snapshot contains
    no pairs (caller checks ``len`` before reading weights, in practice).
    """

    entity_a: str
    entity_b: str
    count: int
    weight: float


@dataclass(frozen=True, slots=True)
class ReinforcementEntry:
    """Per Claim / Preference, how many observation edges and their bounds."""

    node_id: str
    kind: str  # "claim" | "preference"
    count: int
    earliest: str | None  # earliest asserted_at observed (ISO, rounded)
    latest: str | None    # latest asserted_at observed


@dataclass(frozen=True, slots=True)
class CurrentPreferenceEntry:
    """Latest Preference observation for a ``(holder, target)`` pair.

    ``target_key`` is ``entity:<id>`` when the preference resolved to an
    Entity target, or ``literal:<text>`` when the target is a free-form
    literal string.
    """

    holder_id: str
    target_key: str
    polarity: str
    preference_id: str
    asserted_at: str | None


@dataclass(frozen=True, slots=True)
class TimeAnchorChainEntry:
    """One TimeAnchor in the global temporal chain.

    ``prev_id`` / ``next_id`` are the neighboring anchor IDs in sorted ISO
    order (``None`` at either end of the chain). Anchors with identical
    rounded timestamps are impossible by construction (they content-address
    to the same node), so the chain is a strict total order.
    """

    time_anchor_id: str
    iso_timestamp: str
    prev_id: str | None
    next_id: str | None


@dataclass(frozen=True, slots=True)
class DerivedIndex:
    """A full derived-rebuild snapshot.

    Snapshots are produced by :func:`rebuild_derived` and are immutable.
    Stale snapshots are identified by fingerprint mismatch and discarded,
    never patched in place (``R17``).
    """

    fingerprint: str
    aliases: tuple[AliasEntry, ...] = ()
    co_occurrence: tuple[CoOccurrenceEntry, ...] = ()
    reinforcement: tuple[ReinforcementEntry, ...] = ()
    current_preference: tuple[CurrentPreferenceEntry, ...] = ()
    time_anchor_chain: tuple[TimeAnchorChainEntry, ...] = ()


# ---------------------------------------------------------------------------
# Fingerprint.
# ---------------------------------------------------------------------------


def primary_state_signature(store: GraphStore) -> str:
    """Cheap append-only change detector — ``nodes:edges`` counts.

    Safe because primary is append-only (``R16``): node / edge counts
    monotonically increase, so equal counts ⇒ equal content. If that
    invariant is ever violated, replace this with a content digest.
    """
    return f"{store.num_nodes()}:{store.num_edges()}"


def derived_fingerprint(config: MemoryConfig, store: GraphStore) -> str:
    """Snapshot fingerprint (``R17``) — binds derived output to primary + config.

    ``sha256(ingestion_fingerprint || derivation_version || primary_signature)``,
    truncated to 16 hex chars. Any ingest change → different primary signature
    → different derived fingerprint → rebuild.
    """
    payload = {
        "ingestion_fingerprint": config.ingestion_fingerprint(),
        "derivation_version": DERIVATION_VERSION,
        "primary_signature": primary_state_signature(store),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Internals — one builder per derived index. Each reads from ``store`` and
# returns a sorted tuple of entries. No state across builders; the top-level
# rebuild composes them.
# ---------------------------------------------------------------------------


def _build_aliases(store: GraphStore) -> tuple[AliasEntry, ...]:
    out: list[AliasEntry] = []
    for entity_id in store.nodes_by_label(LABEL_ENTITY):
        surfaces: set[str] = set()
        for _src, attrs in store.in_edges(entity_id, edge_type=EDGE_MENTIONS):
            if attrs.surface_form is not None and attrs.surface_form:
                surfaces.add(attrs.surface_form)
        if not surfaces:
            continue
        out.append(
            AliasEntry(
                entity_id=entity_id,
                aliases=tuple(sorted(surfaces)),
            )
        )
    out.sort(key=lambda e: e.entity_id)
    return tuple(out)


def _build_co_occurrence(store: GraphStore) -> tuple[CoOccurrenceEntry, ...]:
    # Group mentioned-entities by ``source_memory_id`` (the provenance
    # anchor on every ``mentions`` edge). Within a Memory, every unordered
    # pair of distinct entities increments the pair's count by 1.
    entities_by_memory: dict[str, set[str]] = defaultdict(set)
    for entity_id in store.nodes_by_label(LABEL_ENTITY):
        for _src, attrs in store.in_edges(entity_id, edge_type=EDGE_MENTIONS):
            memory_id = attrs.source_memory_id
            if memory_id is None:
                continue
            entities_by_memory[memory_id].add(entity_id)

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _memory_id, entity_set in sorted(entities_by_memory.items()):
        entities = sorted(entity_set)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pair_counts[(entities[i], entities[j])] += 1

    if not pair_counts:
        return ()

    max_count = max(pair_counts.values())
    out: list[CoOccurrenceEntry] = []
    for (a, b), count in sorted(pair_counts.items()):
        out.append(
            CoOccurrenceEntry(
                entity_a=a,
                entity_b=b,
                count=count,
                weight=(count / max_count) if max_count > 0 else 0.0,
            )
        )
    return tuple(out)


def _build_reinforcement(store: GraphStore) -> tuple[ReinforcementEntry, ...]:
    out: list[ReinforcementEntry] = []
    for claim_id in store.nodes_by_label(LABEL_CLAIM):
        incoming = store.in_edges(claim_id, edge_type=EDGE_ASSERTS)
        count, earliest, latest = _observation_bounds(incoming)
        if count == 0:
            continue
        out.append(
            ReinforcementEntry(
                node_id=claim_id,
                kind="claim",
                count=count,
                earliest=earliest,
                latest=latest,
            )
        )
    for pref_id in store.nodes_by_label(LABEL_PREFERENCE):
        incoming = store.in_edges(pref_id, edge_type=EDGE_HOLDS_PREFERENCE)
        count, earliest, latest = _observation_bounds(incoming)
        if count == 0:
            continue
        out.append(
            ReinforcementEntry(
                node_id=pref_id,
                kind="preference",
                count=count,
                earliest=earliest,
                latest=latest,
            )
        )
    out.sort(key=lambda e: (e.kind, e.node_id))
    return tuple(out)


def _observation_bounds(
    edges: list[tuple[str, EdgeAttrs]],
) -> tuple[int, str | None, str | None]:
    """Return ``(count, earliest_asserted_at, latest_asserted_at)`` over ``edges``.

    Missing / ``None`` timestamps do not count toward earliest / latest —
    they do contribute to the count. If every edge is timestampless, the
    bounds are both ``None``.
    """
    count = len(edges)
    if count == 0:
        return 0, None, None
    timestamps = [attrs.asserted_at for _src, attrs in edges if attrs.asserted_at]
    if not timestamps:
        return count, None, None
    timestamps.sort()
    return count, timestamps[0], timestamps[-1]


def _build_current_preference(
    store: GraphStore,
) -> tuple[CurrentPreferenceEntry, ...]:
    # For each Preference, collect observation edges (``holds_preference``)
    # grouped by (holder_id, target_key). Within each group, the "current"
    # observation is the Preference whose most-recent ``asserted_at`` wins;
    # ``asserted_at`` tiebreak is ``node_id`` (R2).
    candidates: dict[tuple[str, str], list[tuple[str | None, str, str]]] = defaultdict(list)
    # list entries: (asserted_at, preference_id, polarity)
    for pref_id in store.nodes_by_label(LABEL_PREFERENCE):
        payload = store.get_node(pref_id).get(LABEL_PREFERENCE)
        if not isinstance(payload, PreferencePayload):
            continue
        target_key = _preference_target_key(payload)
        holder_id = payload.holder_id
        for _src, attrs in store.in_edges(pref_id, edge_type=EDGE_HOLDS_PREFERENCE):
            candidates[(holder_id, target_key)].append(
                (attrs.asserted_at, pref_id, payload.polarity)
            )

    out: list[CurrentPreferenceEntry] = []
    for (holder_id, target_key), obs in sorted(candidates.items()):
        # Sort by (asserted_at or "") asc then node_id asc; last is "most recent".
        obs.sort(key=lambda t: ((t[0] or ""), t[1]))
        asserted_at, pref_id, polarity = obs[-1]
        out.append(
            CurrentPreferenceEntry(
                holder_id=holder_id,
                target_key=target_key,
                polarity=polarity,
                preference_id=pref_id,
                asserted_at=asserted_at,
            )
        )
    return tuple(out)


def _preference_target_key(payload: PreferencePayload) -> str:
    if payload.target_id is not None:
        return f"entity:{payload.target_id}"
    if payload.target_literal is not None:
        return f"literal:{payload.target_literal}"
    return "literal:"  # preference with no target — pathological but valid


def _build_time_anchor_chain(store: GraphStore) -> tuple[TimeAnchorChainEntry, ...]:
    anchors: list[tuple[str, str]] = []  # (iso_timestamp, node_id)
    for anchor_id in store.nodes_by_label(LABEL_TIME_ANCHOR):
        payload = store.get_node(anchor_id).get(LABEL_TIME_ANCHOR)
        if not isinstance(payload, TimeAnchorPayload):
            continue
        anchors.append((payload.iso_timestamp, anchor_id))
    # Strict total order: anchors with identical timestamps are impossible
    # (identity is derived from the timestamp itself), but the secondary
    # node_id sort protects against a future change that breaks that.
    anchors.sort()
    out: list[TimeAnchorChainEntry] = []
    for i, (iso, anchor_id) in enumerate(anchors):
        prev_id = anchors[i - 1][1] if i > 0 else None
        next_id = anchors[i + 1][1] if i < len(anchors) - 1 else None
        out.append(
            TimeAnchorChainEntry(
                time_anchor_id=anchor_id,
                iso_timestamp=iso,
                prev_id=prev_id,
                next_id=next_id,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def rebuild_derived(
    store: GraphStore, *, config: MemoryConfig
) -> DerivedIndex:
    """Rebuild every derived index from primary.

    Idempotent — two calls on the same ``(store, config)`` return byte-
    equivalent snapshots with the same fingerprint (``R17``). Primary is
    read-only; this function never mutates the graph.

    Expected cost: O(V + E) across every index (linear in graph size).
    There is no intermediate caching — rebuild is the natural unit.
    """
    fingerprint = derived_fingerprint(config, store)
    return DerivedIndex(
        fingerprint=fingerprint,
        aliases=_build_aliases(store),
        co_occurrence=_build_co_occurrence(store),
        reinforcement=_build_reinforcement(store),
        current_preference=_build_current_preference(store),
        time_anchor_chain=_build_time_anchor_chain(store),
    )


# ---------------------------------------------------------------------------
# Persistence — single-file msgpack envelope, one snapshot per file.
#
# The engram-level save layout places this at ``<save_path>/derived/snapshot.msgpack``
# (see :mod:`engram.engram_memory_system`). Keeping the format narrow — one
# envelope per rebuild — means load is idempotent and version checks are
# local to this module.
# ---------------------------------------------------------------------------


def _encode_entry(entry: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in sorted(fields(entry), key=lambda f_: f_.name):
        value = getattr(entry, f.name)
        if isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def _decode_entry(cls: type, encoded: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    expected = {f.name for f in fields(cls)}
    missing = expected - set(encoded)
    if missing:
        raise DerivedFormatError(
            f"{cls.__name__} missing fields {sorted(missing)}"
        )
    for f in fields(cls):
        value = encoded[f.name]
        annotation = str(f.type)
        if isinstance(value, list) and annotation.startswith("tuple"):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def dump_derived(index: DerivedIndex) -> bytes:
    """Serialize a :class:`DerivedIndex` to versioned msgpack bytes."""
    envelope: dict[str, Any] = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "fingerprint": index.fingerprint,
        "aliases": [_encode_entry(e) for e in index.aliases],
        "co_occurrence": [_encode_entry(e) for e in index.co_occurrence],
        "reinforcement": [_encode_entry(e) for e in index.reinforcement],
        "current_preference": [_encode_entry(e) for e in index.current_preference],
        "time_anchor_chain": [_encode_entry(e) for e in index.time_anchor_chain],
    }
    packed: bytes = msgpack.packb(envelope, use_bin_type=True)
    return packed


def load_derived(data: bytes) -> DerivedIndex:
    """Reconstruct a :class:`DerivedIndex` from msgpack bytes.

    Raises :class:`DerivedFormatError` on schema-version drift or missing
    fields. There is no implicit migration — rebuild from primary instead.
    """
    try:
        envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as exc:
        raise DerivedFormatError(f"msgpack decode failed: {exc}") from exc

    if not isinstance(envelope, dict):
        raise DerivedFormatError("top-level envelope must be a dict")
    version = envelope.get("schema_version")
    if version != DERIVED_SCHEMA_VERSION:
        raise DerivedFormatError(
            f"persisted derived schema_version={version}; "
            f"runtime DERIVED_SCHEMA_VERSION={DERIVED_SCHEMA_VERSION}"
        )
    derivation = envelope.get("derivation_version")
    if derivation != DERIVATION_VERSION:
        raise DerivedFormatError(
            f"persisted derivation_version={derivation}; "
            f"runtime DERIVATION_VERSION={DERIVATION_VERSION}"
        )

    fingerprint = envelope.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise DerivedFormatError("fingerprint missing or non-string")

    return DerivedIndex(
        fingerprint=fingerprint,
        aliases=tuple(_decode_entry(AliasEntry, e) for e in envelope.get("aliases", [])),
        co_occurrence=tuple(
            _decode_entry(CoOccurrenceEntry, e) for e in envelope.get("co_occurrence", [])
        ),
        reinforcement=tuple(
            _decode_entry(ReinforcementEntry, e) for e in envelope.get("reinforcement", [])
        ),
        current_preference=tuple(
            _decode_entry(CurrentPreferenceEntry, e)
            for e in envelope.get("current_preference", [])
        ),
        time_anchor_chain=tuple(
            _decode_entry(TimeAnchorChainEntry, e)
            for e in envelope.get("time_anchor_chain", [])
        ),
    )


__all__ = [
    "AliasEntry",
    "CoOccurrenceEntry",
    "CurrentPreferenceEntry",
    "DERIVATION_VERSION",
    "DERIVED_SCHEMA_VERSION",
    "DerivedFormatError",
    "DerivedIndex",
    "ReinforcementEntry",
    "TimeAnchorChainEntry",
    "derived_fingerprint",
    "dump_derived",
    "load_derived",
    "primary_state_signature",
    "rebuild_derived",
]
