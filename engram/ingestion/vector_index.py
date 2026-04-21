"""Parallel granule-embedding vector index — PR-C patch 4.

The graph encodes knowledge; the vector index ranks candidates (``P10``).
Granules (Turn / Sentence / N-gram) carry MiniLM embeddings in a parallel
``numpy.ndarray`` keyed by node ID, not on the graph node. Recall's semantic
seeding walks this index, not the graph.

**Shape.** ``(N, dim)`` float32 matrix + two parallel Python lists:
``_node_ids`` (row-index → node_id) and ``_granularities`` (row-index →
``turn`` | ``sentence`` | ``ngram``). Adding a granule appends one row;
rebuilding never moves existing rows (``R2``: insertion order is observable
after save, so it must be stable).

**Vectors are stored L2-normalized.** Cosine similarity reduces to a dot
product at query time. ``add`` normalizes on the way in; ``knn`` does not
re-normalize (saves a pass per query). The query vector must also arrive
normalized — ``knn`` asserts this isn't worth the cost.

**Persistence sidecar.** :meth:`save` emits two files — ``embeddings.npy``
(the matrix) and ``node_ids.json`` (the row mapping + granularities + dim).
Schema-versioned at the JSON level; same rebuild-or-die contract as the
primary msgpack store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

VECTOR_INDEX_SCHEMA_VERSION: Final[int] = 1

GRANULARITY_TURN: Final[str] = "turn"
GRANULARITY_SENTENCE: Final[str] = "sentence"
GRANULARITY_NGRAM: Final[str] = "ngram"

ALL_GRANULARITIES: Final[frozenset[str]] = frozenset({
    GRANULARITY_TURN,
    GRANULARITY_SENTENCE,
    GRANULARITY_NGRAM,
})


class VectorIndexFormatError(RuntimeError):
    """On-disk sidecar is structurally invalid or schema-version drifted."""


@dataclass
class VectorIndex:
    """Append-only parallel embedding store for granule nodes.

    The index owns a ``(N, dim)`` float32 matrix of L2-normalized row
    vectors plus two parallel lists mapping row → node_id and row →
    granularity. Rows are never moved after insertion (R2: two ingests of
    the same log must produce byte-identical sidecars).
    """

    dim: int
    _vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _node_ids: list[str] = field(default_factory=list)
    _granularities: list[str] = field(default_factory=list)
    _row_by_id: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive; got {self.dim}")
        if self._vectors.size == 0:
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        if self._vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector matrix width {self._vectors.shape[1]} != dim {self.dim}"
            )
        if not (len(self._node_ids) == len(self._granularities) == self._vectors.shape[0]):
            raise ValueError(
                "node_ids / granularities / vectors arrays are out of sync: "
                f"{len(self._node_ids)} / {len(self._granularities)} / "
                f"{self._vectors.shape[0]}"
            )
        if not self._row_by_id and self._node_ids:
            self._row_by_id = {nid: i for i, nid in enumerate(self._node_ids)}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(self, node_id: str, granularity: str, vector: np.ndarray) -> None:
        """Append a granule embedding. Duplicate ``node_id`` is an error.

        The input vector is converted to float32, L2-normalized, and stored
        as a new row. If the vector has zero norm it's stored as-is (zero
        vector) — recall's knn will score it 0 against any query, fail-closed.
        """
        if granularity not in ALL_GRANULARITIES:
            raise ValueError(
                f"unknown granularity {granularity!r}; expected one of {sorted(ALL_GRANULARITIES)}"
            )
        if node_id in self._row_by_id:
            raise ValueError(f"node_id {node_id!r} already present in vector index")

        row = np.asarray(vector, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.dim:
            raise ValueError(
                f"vector dim {row.shape[0]} != index dim {self.dim}"
            )
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row = row / norm

        # numpy's vstack on an empty (0, dim) array preserves dim — the
        # __post_init__ check made sure the initial matrix has the right
        # second axis so this is safe without a special-case.
        self._vectors = np.vstack([self._vectors, row[np.newaxis, :]]).astype(
            np.float32, copy=False
        )
        new_row_index = len(self._node_ids)
        self._node_ids.append(node_id)
        self._granularities.append(granularity)
        self._row_by_id[node_id] = new_row_index

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._node_ids)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._row_by_id

    def vector_for(self, node_id: str) -> np.ndarray:
        """Return the stored (normalized) vector for ``node_id``. Copy."""
        if node_id not in self._row_by_id:
            raise KeyError(node_id)
        row: np.ndarray = self._vectors[self._row_by_id[node_id]].copy()
        return row

    def granularity_for(self, node_id: str) -> str:
        if node_id not in self._row_by_id:
            raise KeyError(node_id)
        return self._granularities[self._row_by_id[node_id]]

    def node_ids(self) -> tuple[str, ...]:
        """Row-index-ordered tuple of all node IDs in the index."""
        return tuple(self._node_ids)

    def knn(
        self,
        query: np.ndarray,
        k: int,
        *,
        granularity_filter: str | frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return the top-``k`` nearest neighbors by cosine similarity.

        ``query`` is normalized defensively; the stored vectors are already
        normalized at :meth:`add` time so the dot product is the cosine sim
        in ``[-1, 1]``. Ties are broken lexicographically by ``node_id`` for
        R2 stability.

        ``granularity_filter`` may be a single granularity string or a
        ``frozenset`` thereof; ``None`` means no filter. Filtering happens
        before the top-k selection — recall's semantic-seeding pass calls
        ``knn(query, k, granularity_filter="sentence")`` to pull only
        sentence-level candidates.
        """
        if k <= 0:
            return []
        if len(self._node_ids) == 0:
            return []

        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self.dim}"
            )
        qnorm = float(np.linalg.norm(q))
        if qnorm == 0.0:
            return []
        q = q / qnorm

        scores = self._vectors @ q  # (N,) cosine similarities, since rows are normalized

        if granularity_filter is None:
            allowed_mask = None
        else:
            if isinstance(granularity_filter, str):
                allowed = frozenset({granularity_filter})
            else:
                allowed = frozenset(granularity_filter)
            unknown = allowed - ALL_GRANULARITIES
            if unknown:
                raise ValueError(
                    f"unknown granularity in filter: {sorted(unknown)}"
                )
            allowed_mask = np.fromiter(
                (g in allowed for g in self._granularities),
                dtype=bool,
                count=len(self._granularities),
            )

        if allowed_mask is not None:
            candidate_idx = np.nonzero(allowed_mask)[0]
            if candidate_idx.size == 0:
                return []
        else:
            candidate_idx = np.arange(len(self._node_ids))

        # Sort candidates by (-score, node_id). numpy can't break float ties
        # lexicographically in one pass, so we do a Python sort on the
        # candidate rows — bounded by N at this scale, no faiss needed.
        candidates: list[tuple[str, float]] = [
            (self._node_ids[i], float(scores[i])) for i in candidate_idx
        ]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        return candidates[:k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, embeddings_path: Path, node_ids_path: Path) -> None:
        """Write the two sidecar files — ``R2`` byte-stable.

        ``embeddings.npy`` is a raw numpy save of the ``(N, dim)`` matrix.
        ``node_ids.json`` is an ASCII JSON envelope carrying the schema
        version, ``dim``, and parallel ``node_ids`` / ``granularities``
        lists. Row order matches ``_vectors``.
        """
        embeddings_path = Path(embeddings_path)
        node_ids_path = Path(node_ids_path)

        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        node_ids_path.parent.mkdir(parents=True, exist_ok=True)

        # allow_pickle=False is the default on modern numpy; be explicit.
        np.save(embeddings_path, self._vectors, allow_pickle=False)

        envelope = {
            "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
            "dim": self.dim,
            "node_ids": list(self._node_ids),
            "granularities": list(self._granularities),
        }
        node_ids_path.write_text(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, embeddings_path: Path, node_ids_path: Path) -> VectorIndex:
        """Reconstruct the index from its sidecars.

        Raises :class:`VectorIndexFormatError` on schema-version drift or
        structural issues. No implicit migration — bump the version, ship a
        migration, or reingest.
        """
        embeddings_path = Path(embeddings_path)
        node_ids_path = Path(node_ids_path)
        if not embeddings_path.exists():
            raise VectorIndexFormatError(f"embeddings file missing: {embeddings_path}")
        if not node_ids_path.exists():
            raise VectorIndexFormatError(f"node_ids file missing: {node_ids_path}")

        try:
            envelope = json.loads(node_ids_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorIndexFormatError(f"node_ids.json decode failed: {exc}") from exc
        if not isinstance(envelope, dict):
            raise VectorIndexFormatError("node_ids.json must be a JSON object")

        version = envelope.get("schema_version")
        if version != VECTOR_INDEX_SCHEMA_VERSION:
            raise VectorIndexFormatError(
                f"persisted vector-index schema_version={version}; "
                f"runtime VECTOR_INDEX_SCHEMA_VERSION={VECTOR_INDEX_SCHEMA_VERSION}"
            )

        dim = envelope.get("dim")
        if not isinstance(dim, int) or dim <= 0:
            raise VectorIndexFormatError(f"invalid dim in envelope: {dim!r}")

        node_ids = envelope.get("node_ids")
        granularities = envelope.get("granularities")
        if not isinstance(node_ids, list) or not all(isinstance(x, str) for x in node_ids):
            raise VectorIndexFormatError("node_ids must be a list of strings")
        if not isinstance(granularities, list) or not all(
            isinstance(x, str) for x in granularities
        ):
            raise VectorIndexFormatError("granularities must be a list of strings")
        if len(node_ids) != len(granularities):
            raise VectorIndexFormatError(
                f"node_ids ({len(node_ids)}) / granularities ({len(granularities)}) length mismatch"
            )

        vectors = np.load(embeddings_path, allow_pickle=False)
        if vectors.ndim != 2:
            raise VectorIndexFormatError(
                f"embeddings.npy must be 2-D; got shape {vectors.shape}"
            )
        if vectors.shape[0] != len(node_ids):
            raise VectorIndexFormatError(
                f"row count {vectors.shape[0]} != node_ids length {len(node_ids)}"
            )
        if vectors.shape[1] != dim:
            raise VectorIndexFormatError(
                f"vector dim {vectors.shape[1]} != envelope dim {dim}"
            )

        return cls(
            dim=dim,
            _vectors=np.asarray(vectors, dtype=np.float32).copy(),
            _node_ids=list(node_ids),
            _granularities=list(granularities),
        )


__all__ = [
    "ALL_GRANULARITIES",
    "GRANULARITY_NGRAM",
    "GRANULARITY_SENTENCE",
    "GRANULARITY_TURN",
    "VECTOR_INDEX_SCHEMA_VERSION",
    "VectorIndex",
    "VectorIndexFormatError",
]
