"""Extraction coverage — what's in the graph, grouped for at-a-glance review.

Produces a :class:`CoverageReport` with per-label node counts, per-layer
node counts, per-edge-type edge counts, per-ngram-kind granule counts,
and a few top-line totals (n_memories, n_granules, n_entities, etc.).

**Uses only the read-only graph surface** (``iter_nodes`` / ``iter_edges``)
— diagnostics never touches the runtime path.

**R2.** All count maps are returned as tuples of ``(key, count)`` sorted
by key so two stores with identical content produce byte-identical
reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_NGRAM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Snapshot of a :class:`GraphStore`'s contents.

    ``nodes_by_label`` — number of nodes carrying each label. A node
    carrying two labels (``memory`` + ``turn`` is impossible, but a
    granule-with-layer is possible via entity_relationship lifting)
    counts under every label it declares.

    ``nodes_by_layer`` — number of nodes in each content-classification
    layer (``entity`` / ``relationship`` / ``temporal`` / ``episodic``).
    Granules with no layer count under ``(unlayered)``.

    ``ngram_kinds`` — distribution of n-gram granules by ``ngram_kind``
    (``noun_chunk`` / ``svo``); empty for stores with no n-grams.

    ``edges_by_type`` — edge-type distribution.

    ``totals`` — a small convenience map with ``n_nodes``, ``n_edges``,
    ``n_memories``, ``n_entities``, ``n_claims``, ``n_preferences``,
    ``n_time_anchors``, ``n_granules`` (the granularity hierarchy:
    memory + turn + utterance_segment + ngram).
    """

    nodes_by_label: tuple[tuple[str, int], ...]
    nodes_by_layer: tuple[tuple[str, int], ...]
    ngram_kinds: tuple[tuple[str, int], ...]
    edges_by_type: tuple[tuple[str, int], ...]
    totals: tuple[tuple[str, int], ...]


_GRANULARITY_LABELS = frozenset({
    LABEL_MEMORY,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LABEL_NGRAM,
})


def extraction_coverage(store: GraphStore) -> CoverageReport:
    """Summarize ``store``'s contents without mutating it.

    Reads once via ``iter_nodes`` / ``iter_edges``; O(V+E).
    """
    label_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    ngram_kind_counts: Counter[str] = Counter()
    n_memories = n_entities = n_claims = n_preferences = 0
    n_time_anchors = n_granules = n_nodes = 0

    for _node_id, attrs in store.iter_nodes():
        n_nodes += 1
        labels = attrs.get("labels", frozenset())
        if not isinstance(labels, frozenset):
            # Defensive: iter_nodes should return frozensets, but a
            # user of the GraphStore API might have attached a list.
            labels = frozenset(labels) if isinstance(labels, (list, tuple, set)) else frozenset()
        for label in labels:
            label_counts[label] += 1
        if labels & _GRANULARITY_LABELS:
            n_granules += 1

        layers = attrs.get("layers", frozenset())
        if not isinstance(layers, frozenset):
            layers = frozenset(layers) if isinstance(layers, (list, tuple, set)) else frozenset()
        if not layers:
            layer_counts["(unlayered)"] += 1
        else:
            for layer in layers:
                layer_counts[layer] += 1

        if LABEL_NGRAM in labels:
            ngram_payload = attrs.get(LABEL_NGRAM)
            kind = _ngram_kind(ngram_payload)
            if kind:
                ngram_kind_counts[kind] += 1

        if LABEL_MEMORY in labels:
            n_memories += 1
        if LABEL_ENTITY in labels:
            n_entities += 1
        if LABEL_CLAIM in labels:
            n_claims += 1
        if LABEL_PREFERENCE in labels:
            n_preferences += 1
        if LABEL_TIME_ANCHOR in labels:
            n_time_anchors += 1

    edge_counts: Counter[str] = Counter()
    n_edges = 0
    for _src, _dst, edge_type, _edge_attrs in store.iter_edges():
        edge_counts[edge_type] += 1
        n_edges += 1

    totals = {
        "n_claims": n_claims,
        "n_edges": n_edges,
        "n_entities": n_entities,
        "n_granules": n_granules,
        "n_memories": n_memories,
        "n_nodes": n_nodes,
        "n_preferences": n_preferences,
        "n_time_anchors": n_time_anchors,
    }

    return CoverageReport(
        nodes_by_label=tuple(sorted(label_counts.items())),
        nodes_by_layer=tuple(sorted(layer_counts.items())),
        ngram_kinds=tuple(sorted(ngram_kind_counts.items())),
        edges_by_type=tuple(sorted(edge_counts.items())),
        totals=tuple(sorted(totals.items())),
    )


def _ngram_kind(payload: Any) -> str | None:
    kind = getattr(payload, "ngram_kind", None)
    return kind if isinstance(kind, str) else None


__all__ = [
    "CoverageReport",
    "extraction_coverage",
]
