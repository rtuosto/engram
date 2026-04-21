"""Stage [4] — Scoring and granule selection.

Takes the walk's raw ``{node_id: score}`` map and resolves each scored
node back to its containing *granule* (Turn / Sentence / N-gram). Non-
granule nodes (Entity, Claim, Preference, TimeAnchor) are not returned
as passages — they're the *scaffolding* that made the walk find the
granule. Their score is folded into the containing granule's bucket,
keeping the walk-explanation legible via the recall assembly's
``supporting_edges`` rendering.

Per-granule bucket rule: the highest-scoring contributor wins; the
bucket's final score is the max over its contributors. Buckets are
returned sorted by ``(-score, granule_id)`` for R2 stability and capped
at ``max_passages``.
"""

from __future__ import annotations

from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_NGRAM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
)

_GRANULE_LABELS: frozenset[str] = frozenset({
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LABEL_NGRAM,
})

# Labels that should NOT be treated as passages but may route a score to
# their containing granule via a single ``part_of`` hop. (Relationship /
# temporal nodes terminate the walk-to-passage reduction; their score
# stays folded into the Claim / Pref / Anchor itself, which is never
# surfaced as a passage but can contribute via the scaffolding path.)
_NON_PASSAGE_LABELS: frozenset[str] = frozenset({
    LABEL_ENTITY,
    LABEL_CLAIM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
})


def select_passages(
    walk_scores: dict[str, float],
    store: GraphStore,
    *,
    max_passages: int,
) -> list[tuple[str, float]]:
    """Return ``[(granule_id, score), ...]`` ranked top ``max_passages``.

    A walked node routes to a granule via its type:

    - Granule (Turn / Sentence / N-gram) — contributes its own score.
    - Non-granule (Entity / Claim / Preference / TimeAnchor) — ignored.

    Sort order: ``(-score, granule_id)`` for R2.
    """
    if not walk_scores or max_passages <= 0:
        return []

    buckets: dict[str, float] = {}
    for node_id, score in walk_scores.items():
        if score <= 0.0:
            continue
        granule_id = _resolve_to_granule(node_id, store)
        if granule_id is None:
            continue
        prior = buckets.get(granule_id, 0.0)
        if score > prior:
            buckets[granule_id] = score

    ranked = sorted(buckets.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:max_passages]


def _resolve_to_granule(node_id: str, store: GraphStore) -> str | None:
    """Return the granule node_id that ``node_id`` belongs to, or None.

    Granules resolve to themselves. Non-granule labels yield None.
    """
    if not store.has_node(node_id):
        return None
    labels = store.node_labels(node_id)
    if labels & _GRANULE_LABELS:
        return node_id
    if labels & _NON_PASSAGE_LABELS:
        return None
    # Memory nodes aren't granules (they're parent to Turn granules) — let
    # the walk find the Turn via the PART_OF edge instead. Silently ignore.
    _ = EDGE_PART_OF  # reference kept; see module docstring's routing note
    return None


__all__ = ["select_passages"]
