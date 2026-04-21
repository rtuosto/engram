"""Stage [2] — Seeding.

Two parallel sources of seed granules per ``docs/design/recall.md §5``:

1. **Semantic seeding** — cosine-KNN over the parallel granule embedding
   index (:class:`engram.ingestion.vector_index.VectorIndex`), per-granularity
   with an intent-specific weight profile.

2. **Entity-anchored seeding** — spaCy NER over the query. For each entity
   mention that resolves (by normalized surface form) to a known entity in
   the corpus, seed the entity node and its directly-connected granules via
   the inbound ``mentions`` edges.

Seed scores from both sources are merged by taking the max per node_id
(R2-stable). The combined seed set is capped at
:attr:`engram.config.MemoryConfig.recall_seed_count_total`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from engram.ingestion.extractors.canonicalization import (
    EntityRegistry,
    normalize,
)
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import EDGE_MENTIONS
from engram.ingestion.vector_index import (
    GRANULARITY_NGRAM,
    GRANULARITY_SENTENCE,
    GRANULARITY_TURN,
    VectorIndex,
)

# Score applied to entity-anchored granules (inbound mentions edges).
ENTITY_GRANULE_SEED_SCORE: float = 0.8
# Score applied to the entity node itself.
ENTITY_SEED_SCORE: float = 1.0


class _DocLike(Protocol):
    """Duck-typed spaCy Doc subset the query-NER pass reads.

    Only ``ents`` is required; each entry must expose ``text`` and
    ``label_`` so :func:`entity_anchored_seed` can normalize + look up in
    the registry.
    """

    @property
    def ents(self) -> tuple[object, ...]: ...  # pragma: no cover - protocol


def semantic_seed(
    query: str,
    vector_index: VectorIndex,
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    granularity_weights: dict[str, float],
    top_n_per_granularity: int,
) -> list[tuple[str, float]]:
    """Vector-index nearest-neighbor seeding per granularity.

    ``granularity_weights[granularity] -> float`` scales both the fetched
    candidate count (``ceil(top_n_per_granularity * weight)``) and the
    similarity score written into the returned seed list. Granularities
    absent from ``granularity_weights`` contribute zero seeds.
    """
    if len(vector_index) == 0:
        return []

    vectors = embed_fn([query])
    if vectors.ndim != 2 or vectors.shape[0] != 1:
        raise ValueError(
            f"embed_fn must return (1, d); got {vectors.shape}"
        )
    q = vectors[0]

    hits: dict[str, float] = {}
    for granularity in (GRANULARITY_TURN, GRANULARITY_SENTENCE, GRANULARITY_NGRAM):
        weight = float(granularity_weights.get(granularity, 0.0))
        if weight <= 0.0:
            continue
        k = max(1, int(round(top_n_per_granularity * weight)))
        for node_id, similarity in vector_index.knn(q, k, granularity_filter=granularity):
            scaled = similarity * weight
            if scaled <= 0.0:
                continue
            existing = hits.get(node_id)
            if existing is None or scaled > existing:
                hits[node_id] = scaled

    out = sorted(hits.items(), key=lambda pair: (-pair[1], pair[0]))
    return out


def entity_anchored_seed(
    query_doc: _DocLike,
    entity_registry: EntityRegistry,
    store: GraphStore,
) -> list[tuple[str, float]]:
    """Seed entity nodes + their directly-mentioned granules.

    For each NER mention in ``query_doc.ents``: normalize the surface
    (NFKC casefold), look up ``(label_, normalized)`` in
    :attr:`EntityRegistry.by_type_and_form`, and if it resolves, emit seeds
    for the entity and every granule with an inbound ``mentions`` edge.

    Unmatched mentions fall through silently — the semantic seed pass is
    the fallback signal.
    """
    seeds: dict[str, float] = {}
    for ent in getattr(query_doc, "ents", ()) or ():
        surface = str(getattr(ent, "text", ""))
        label = str(getattr(ent, "label_", ""))
        if not surface or not label:
            continue
        normalized = normalize(surface)
        entity_id = entity_registry.by_type_and_form.get((label, normalized))
        if entity_id is None or not store.has_node(entity_id):
            continue
        if seeds.get(entity_id, 0.0) < ENTITY_SEED_SCORE:
            seeds[entity_id] = ENTITY_SEED_SCORE
        for granule_id, _attrs in store.in_edges(entity_id, edge_type=EDGE_MENTIONS):
            if seeds.get(granule_id, 0.0) < ENTITY_GRANULE_SEED_SCORE:
                seeds[granule_id] = ENTITY_GRANULE_SEED_SCORE
    return sorted(seeds.items(), key=lambda pair: (-pair[1], pair[0]))


def merge_seeds(
    *seed_lists: list[tuple[str, float]],
    total_cap: int,
) -> list[tuple[str, float]]:
    """Union seeds by node_id keeping the max score; cap at ``total_cap``.

    Output is sorted by ``(-score, node_id)`` for R2 stability. When the
    union exceeds ``total_cap``, lower-scoring tail is dropped.
    """
    combined: dict[str, float] = {}
    for seeds in seed_lists:
        for node_id, score in seeds:
            if combined.get(node_id, 0.0) < score:
                combined[node_id] = score
    ranked = sorted(combined.items(), key=lambda pair: (-pair[1], pair[0]))
    if total_cap > 0:
        ranked = ranked[:total_cap]
    return ranked


__all__ = [
    "ENTITY_GRANULE_SEED_SCORE",
    "ENTITY_SEED_SCORE",
    "entity_anchored_seed",
    "merge_seeds",
    "semantic_seed",
]
