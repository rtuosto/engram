"""Stage [3] — Bounded typed-edge BFS expansion.

Thin wrapper around :meth:`engram.ingestion.graph.GraphStore.bfs`. Keeping
the wrapper makes the recall pipeline's contract a single call-site rather
than exposing the ``GraphStore`` method directly to the orchestrator.

**Why no reranker.** Walk scores encode seed similarity × edge-weight
product × depth decay — legible and tunable (``docs/design/recall.md §6``).
A learned reranker enters only if diagnostics shows a specific ranking-
quality bottleneck the walk can't fix.
"""

from __future__ import annotations

from collections.abc import Sequence

from engram.ingestion.graph import GraphStore


def expand(
    store: GraphStore,
    seeds: Sequence[tuple[str, float]],
    edge_weights: dict[str, float],
    *,
    max_depth: int,
    max_frontier: int,
) -> dict[str, float]:
    """Return ``{node_id: score}`` from a bounded typed-edge BFS.

    Delegates to :meth:`GraphStore.bfs`. Returns the union of seed scores
    and walk-propagated scores (max per node). Callers pass only edge
    types they want walked — types absent from ``edge_weights`` are
    skipped by the underlying traversal.
    """
    return store.bfs(
        list(seeds),
        edge_weights,
        max_depth=max_depth,
        max_frontier=max_frontier,
    )


__all__ = ["expand"]
