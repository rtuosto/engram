"""Intent-classification seed fixtures and centroid construction.

Mirrors :mod:`engram.ingestion.preferences` — hand-authored synthetic seeds
per intent, disjoint held-out set for the per-intent discrimination gate,
and content-addressed hashes so seed edits invalidate the recall fingerprint
(``R3``, ``R4``, ``docs/design/recall.md §4``).

**Benchmark hygiene (``P8``, ``M3``, ``M4``).** Seeds and held-out queries
are synthetic; none are drawn from LongMemEval or LOCOMO. Contamination risk
is zero by construction.

**Lazy centroid construction.** Centroids are computed on first use from the
seeds file via a caller-supplied embedding function. No pre-serialized
centroid artifact — that would be a second source of truth prone to drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import numpy as np

INTENT_SINGLE_FACT: Final[str] = "single_fact"
INTENT_AGGREGATION: Final[str] = "aggregation"
INTENT_PREFERENCE: Final[str] = "preference"
INTENT_TEMPORAL: Final[str] = "temporal"
INTENT_ENTITY_RESOLUTION: Final[str] = "entity_resolution"

INTENTS: Final[tuple[str, ...]] = (
    INTENT_SINGLE_FACT,
    INTENT_AGGREGATION,
    INTENT_PREFERENCE,
    INTENT_TEMPORAL,
    INTENT_ENTITY_RESOLUTION,
)

_PKG_DIR: Final[Path] = Path(__file__).parent
SEEDS_FILE: Final[Path] = _PKG_DIR / "seeds.json"
HELDOUT_FILE: Final[Path] = _PKG_DIR / "heldout.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


INTENT_SEED_HASH: Final[str] = _file_hash(SEEDS_FILE)
INTENT_HELDOUT_HASH: Final[str] = _file_hash(HELDOUT_FILE)


def load_seeds() -> dict[str, tuple[str, ...]]:
    """Return ``{intent: tuple_of_seed_queries}`` in canonical intent order."""
    raw = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
    _validate_shape(raw, SEEDS_FILE)
    return {intent: tuple(raw[intent]) for intent in INTENTS}


def load_heldout() -> dict[str, tuple[str, ...]]:
    """Held-out queries for the per-intent discrimination gate."""
    raw = json.loads(HELDOUT_FILE.read_text(encoding="utf-8"))
    _validate_shape(raw, HELDOUT_FILE)
    return {intent: tuple(raw[intent]) for intent in INTENTS}


def _validate_shape(raw: object, source: Path) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must deserialize to a dict of {{intent: [queries]}}")
    missing = set(INTENTS) - set(raw)
    if missing:
        raise ValueError(f"{source} missing intents: {sorted(missing)}")
    for intent, queries in raw.items():
        if intent not in INTENTS:
            raise ValueError(f"{source} has unknown intent: {intent!r}")
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            raise ValueError(f"{source}[{intent!r}] must be a list of strings")
        if len(queries) == 0:
            raise ValueError(f"{source}[{intent!r}] is empty — fail-closed would drop every query")


def compute_intent_centroids(
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, np.ndarray]:
    """Build per-intent centroids as the mean of seed embeddings.

    ``embed_fn(list_of_strings)`` must return a ``(n, d)`` ndarray. Returned
    centroids are L2-normalized so cosine similarity reduces to a dot
    product at classification time.
    """
    seeds = load_seeds()
    centroids: dict[str, np.ndarray] = {}
    for intent in INTENTS:
        vectors = embed_fn(list(seeds[intent]))
        if vectors.ndim != 2 or vectors.shape[0] != len(seeds[intent]):
            raise ValueError(
                f"embed_fn must return (n, d); got {vectors.shape} for intent {intent!r}"
            )
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            centroid = centroid / norm
        centroids[intent] = centroid.astype(np.float32, copy=False)
    return centroids


def median_intent_margin(
    centroids: Mapping[str, np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, float]:
    """Per-intent median discrimination margin on the held-out set.

    ``margin = cos(query, centroid_true) - max_{q!=true} cos(query, centroid_q)``

    Returns ``{intent: median_margin}``. Callers decide whether each intent
    clears the per-corpus gate; this function is pure measurement.
    """
    heldout = load_heldout()
    results: dict[str, float] = {}
    for intent in INTENTS:
        vectors = embed_fn(list(heldout[intent]))
        margins: list[float] = []
        for row in vectors:
            row_norm = float(np.linalg.norm(row))
            unit = row / row_norm if row_norm > 0.0 else row
            scores = {i: float(unit @ centroids[i]) for i in INTENTS}
            top = scores[intent]
            second = max(v for i, v in scores.items() if i != intent)
            margins.append(top - second)
        margins.sort()
        mid = len(margins) // 2
        if len(margins) % 2 == 0:
            results[intent] = (margins[mid - 1] + margins[mid]) / 2.0
        else:
            results[intent] = margins[mid]
    return results


__all__ = [
    "HELDOUT_FILE",
    "INTENT_AGGREGATION",
    "INTENT_ENTITY_RESOLUTION",
    "INTENT_HELDOUT_HASH",
    "INTENT_PREFERENCE",
    "INTENT_SEED_HASH",
    "INTENT_SINGLE_FACT",
    "INTENT_TEMPORAL",
    "INTENTS",
    "SEEDS_FILE",
    "compute_intent_centroids",
    "load_heldout",
    "load_seeds",
    "median_intent_margin",
]
