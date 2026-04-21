"""``GraphStore`` — the narrow storage interface Recall depends on.

A thin wrapper around :class:`networkx.MultiDiGraph` with R2-compliant
iteration order. See ``docs/design/ingestion.md §1`` for the rationale;
this module is the swap point — nothing above it imports ``networkx``.

**R2 discipline.** Every method whose output order is observable returns
results in a deterministic order (sorted by node_id, then edge type). The
underlying ``MultiDiGraph``'s natural iteration order is hash-seed-dependent;
this wrapper sorts on the way out.

**Freeze semantics.** Callers may call :meth:`freeze` after ingestion
completes (e.g., before a derived-rebuild pass in PR-D). After freeze,
writes raise :class:`GraphFrozenError`. Reads remain open.

**Multi-labeling.** A node carries ``labels: frozenset[str]`` plus one payload
attribute per label (key = label name). Adding a second label merges; payloads
do not overwrite each other.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final, cast

import networkx as nx

from engram.ingestion.schema import EdgeAttrs

DEFAULT_MAX_FRONTIER: Final[int] = 256


class GraphFrozenError(RuntimeError):
    """Raised when a write is attempted on a frozen GraphStore."""


class NodeNotFoundError(KeyError):
    """Raised when an operation references a node that does not exist."""


@dataclass
class GraphStore:
    """One engram instance's graph.

    ``conversation_id`` is retained as an opaque tag (default
    ``"__instance__"``) — it has no semantic meaning in the
    post-pivot architecture (``R1``: one instance = one memory, no
    conversation partitioning). Exposed to Recall read-only (callers
    should not mutate after :meth:`freeze`).
    """

    conversation_id: str
    _graph: nx.MultiDiGraph[str] = field(default_factory=nx.MultiDiGraph)
    _frozen: bool = False
    _labels_index: dict[str, set[str]] = field(default_factory=dict)
    _layers_index: dict[str, set[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def _require_writable(self) -> None:
        if self._frozen:
            raise GraphFrozenError(
                f"GraphStore({self.conversation_id!r}) is frozen; mutation not permitted"
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        *,
        labels: frozenset[str],
        payloads: dict[str, object] | None = None,
        layers: frozenset[str] = frozenset(),
    ) -> None:
        """Add or extend a node.

        If the node already exists, ``labels`` and ``layers`` are each
        unioned with the existing sets and each payload in ``payloads`` is
        attached under its label-name key (overwriting any prior value for
        the same label).

        ``payloads`` is a ``{label: payload_dataclass}`` dict. Keys must all
        appear in ``labels`` (an unrelated payload key is a programming error).

        ``layers`` is the content-classification layer set (see
        ``docs/design/ingestion.md §3``: ``entity`` | ``relationship`` |
        ``temporal`` | ``episodic``). ``semantic`` is *not* a layer label —
        granules carry their semantic signal in the parallel embedding
        index. Default ``frozenset()`` keeps non-classified granule nodes
        valid.
        """
        self._require_writable()
        payloads = payloads or {}
        unknown = set(payloads) - set(labels)
        if unknown:
            raise ValueError(
                f"payload keys {sorted(unknown)} not present in labels {sorted(labels)}"
            )

        if self._graph.has_node(node_id):
            existing_labels: frozenset[str] = self._graph.nodes[node_id]["labels"]
            new_labels = existing_labels | labels
            self._graph.nodes[node_id]["labels"] = new_labels
            for label, payload in payloads.items():
                self._graph.nodes[node_id][label] = payload
            for label in new_labels - existing_labels:
                self._labels_index.setdefault(label, set()).add(node_id)

            existing_layers: frozenset[str] = self._graph.nodes[node_id].get(
                "layers", frozenset()
            )
            new_layers = existing_layers | layers
            if new_layers != existing_layers:
                self._graph.nodes[node_id]["layers"] = new_layers
                for layer in new_layers - existing_layers:
                    self._layers_index.setdefault(layer, set()).add(node_id)
        else:
            attrs: dict[str, object] = {"labels": labels, "layers": layers}
            attrs.update(payloads)
            self._graph.add_node(node_id, **attrs)
            for label in labels:
                self._labels_index.setdefault(label, set()).add(node_id)
            for layer in layers:
                self._layers_index.setdefault(layer, set()).add(node_id)

    def add_edge(self, src: str, dst: str, attrs: EdgeAttrs) -> None:
        """Add a typed edge. Parallel edges of the same type are not allowed;
        calling :meth:`add_edge` with a duplicate ``(src, dst, type)`` overwrites.

        Endpoints must exist — violating this raises :class:`NodeNotFoundError`.
        """
        self._require_writable()
        if not self._graph.has_node(src):
            raise NodeNotFoundError(f"src node {src!r} does not exist")
        if not self._graph.has_node(dst):
            raise NodeNotFoundError(f"dst node {dst!r} does not exist")
        self._graph.add_edge(src, dst, key=attrs.type, attrs=attrs)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> dict[str, object]:
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(node_id)
        return dict(self._graph.nodes[node_id])

    def node_labels(self, node_id: str) -> frozenset[str]:
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(node_id)
        return cast("frozenset[str]", self._graph.nodes[node_id]["labels"])

    def nodes_by_label(self, label: str) -> list[str]:
        """Sorted list of node IDs carrying ``label``."""
        return sorted(self._labels_index.get(label, set()))

    def node_layers(self, node_id: str) -> frozenset[str]:
        """Layer labels attached to ``node_id`` (``entity`` / ``relationship``
        / ``temporal`` / ``episodic``). Empty frozenset for granule nodes and
        Memories (``docs/design/ingestion.md §3``: ``semantic`` lives in the
        parallel embedding index, not on the node)."""
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(node_id)
        return cast("frozenset[str]", self._graph.nodes[node_id].get("layers", frozenset()))

    def nodes_by_layer(self, layer: str) -> list[str]:
        """Sorted list of node IDs carrying ``layer``.

        Primary use site is recall (PR-E): intent-driven seeding and
        expansion work against layer partitions, not raw node-type labels.
        """
        return sorted(self._layers_index.get(layer, set()))

    def has_edge(self, src: str, dst: str, edge_type: str) -> bool:
        return self._graph.has_edge(src, dst, key=edge_type)

    def get_edge_attrs(self, src: str, dst: str, edge_type: str) -> EdgeAttrs:
        if not self._graph.has_edge(src, dst, key=edge_type):
            raise KeyError((src, dst, edge_type))
        return cast("EdgeAttrs", self._graph.edges[src, dst, edge_type]["attrs"])

    def out_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[tuple[str, EdgeAttrs]]:
        """Deterministic-sorted outgoing edges from ``node_id``.

        Sort key is ``(dst_node_id, edge_type)`` — R2-stable.
        """
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(node_id)
        out: list[tuple[str, EdgeAttrs]] = []
        for _, dst, key, data in self._graph.out_edges(node_id, keys=True, data=True):
            if edge_type is not None and key != edge_type:
                continue
            out.append((dst, data["attrs"]))
        out.sort(key=lambda pair: (pair[0], pair[1].type))
        return out

    def in_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[tuple[str, EdgeAttrs]]:
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(node_id)
        out: list[tuple[str, EdgeAttrs]] = []
        for src, _, key, data in self._graph.in_edges(node_id, keys=True, data=True):
            if edge_type is not None and key != edge_type:
                continue
            out.append((src, data["attrs"]))
        out.sort(key=lambda pair: (pair[0], pair[1].type))
        return out

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, object]]]:
        """Sorted iteration over ``(node_id, attrs)`` — R2 order."""
        for node_id in sorted(self._graph.nodes):
            yield node_id, dict(self._graph.nodes[node_id])

    def iter_edges(self) -> Iterator[tuple[str, str, str, EdgeAttrs]]:
        """Sorted iteration over ``(src, dst, edge_type, attrs)`` — R2 order."""
        edges = [
            (src, dst, key, data["attrs"])
            for src, dst, key, data in self._graph.edges(keys=True, data=True)
        ]
        edges.sort(key=lambda t: (t[0], t[1], t[2]))
        yield from edges

    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Traversal (generic typed-edge BFS; Recall is the primary caller)
    # ------------------------------------------------------------------

    def bfs(
        self,
        seeds: Sequence[tuple[str, float]],
        edge_weights: dict[str, float],
        *,
        max_depth: int,
        max_frontier: int = DEFAULT_MAX_FRONTIER,
    ) -> dict[str, float]:
        """Typed-edge BFS; returns ``{node_id: score}``.

        Starts from ``seeds`` (each with an initial score). Each step expands
        along out-edges whose ``edge_type`` appears in ``edge_weights``, and
        propagates ``score_src * edge_attrs.weight * edge_weights[edge_type]``
        to the destination. Frontier is capped per layer at ``max_frontier``
        (keep top-scored; ties broken lexicographic by ``node_id``).

        Undirected edges (e.g. ``co_occurs_with``) should be emitted in both
        directions at ingest; this traversal follows out-edges only.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        scores: dict[str, float] = {}
        for node_id, s in seeds:
            if s <= 0.0:
                continue
            scores[node_id] = max(scores.get(node_id, 0.0), s)

        frontier = dict(scores)
        for _depth in range(max_depth):
            next_frontier: dict[str, float] = {}
            for src_id, src_score in sorted(frontier.items()):
                for dst_id, attrs in self.out_edges(src_id):
                    w = edge_weights.get(attrs.type)
                    if w is None or w <= 0.0:
                        continue
                    contrib = src_score * attrs.weight * w
                    if contrib <= 0.0:
                        continue
                    next_frontier[dst_id] = max(next_frontier.get(dst_id, 0.0), contrib)

            if not next_frontier:
                break

            if len(next_frontier) > max_frontier:
                ranked = sorted(
                    next_frontier.items(),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:max_frontier]
                next_frontier = dict(ranked)

            for node_id, s in next_frontier.items():
                scores[node_id] = max(scores.get(node_id, 0.0), s)

            frontier = next_frontier

        return scores


__all__ = [
    "DEFAULT_MAX_FRONTIER",
    "GraphFrozenError",
    "GraphStore",
    "NodeNotFoundError",
]
