"""Preference-detection fixtures and centroid construction.

See ``docs/design/ingestion.md §5`` for the validation protocol. This package
owns the hand-authored synthetic seeds and held-out sentences used to build
and validate the prototype-embedding centroids.

**Benchmark hygiene (P8, M3, M4).** Seeds and held-out sentences are
speaker-agnostic and never drawn from LongMemEval or LOCOMO. Contamination
risk is zero by construction.

**Fingerprint coupling (R3).** The seed file content hash is exposed as
:data:`SEED_HASH`. :mod:`engram.config` imports this constant and enters it
into ``MemoryConfig._INGESTION_FIELDS`` so editing the seeds invalidates the
ingestion fingerprint — cached graphs built under old seeds are rejected.

**Lazy centroid construction.** Centroids are computed on first use from the
seeds file, via a caller-supplied embedding function. No pre-serialized
centroid artifact — that would be a second source of truth prone to drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import numpy as np

from engram.ingestion.schema import PREFERENCE_POLARITIES

_PKG_DIR: Final[Path] = Path(__file__).parent
SEEDS_FILE: Final[Path] = _PKG_DIR / "seeds.json"
HELDOUT_FILE: Final[Path] = _PKG_DIR / "heldout.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


SEED_HASH: Final[str] = _file_hash(SEEDS_FILE)
HELDOUT_HASH: Final[str] = _file_hash(HELDOUT_FILE)


def load_seeds() -> dict[str, tuple[str, ...]]:
    """Return ``{polarity: tuple_of_seed_sentences}``.

    Tuples (not lists) and ``sorted`` polarity iteration via
    :data:`PREFERENCE_POLARITIES` keep downstream output R2-deterministic.
    """
    raw = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
    _validate_shape(raw, SEEDS_FILE)
    return {p: tuple(raw[p]) for p in PREFERENCE_POLARITIES}


def load_heldout() -> dict[str, tuple[str, ...]]:
    """Held-out sentences for the per-polarity discrimination gate (§5)."""
    raw = json.loads(HELDOUT_FILE.read_text(encoding="utf-8"))
    _validate_shape(raw, HELDOUT_FILE)
    return {p: tuple(raw[p]) for p in PREFERENCE_POLARITIES}


def _validate_shape(raw: object, source: Path) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must deserialize to a dict of {{polarity: [sentences]}}")
    missing = set(PREFERENCE_POLARITIES) - set(raw)
    if missing:
        raise ValueError(f"{source} missing polarities: {sorted(missing)}")
    for p, sentences in raw.items():
        if p not in PREFERENCE_POLARITIES:
            raise ValueError(f"{source} has unknown polarity: {p!r}")
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise ValueError(f"{source}[{p!r}] must be a list of strings")
        if len(sentences) == 0:
            raise ValueError(f"{source}[{p!r}] is empty — fail-closed would drop every Preference")


def compute_centroids(
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, np.ndarray]:
    """Build per-polarity centroids as the mean of seed embeddings.

    ``embed_fn(list_of_strings)`` must return a ``(n, d)`` ndarray. Caller
    owns the embedding model — this function is model-agnostic so we can mock
    it in unit tests and feed the real encoder in integration.
    """
    seeds = load_seeds()
    centroids: dict[str, np.ndarray] = {}
    for polarity in PREFERENCE_POLARITIES:
        vectors = embed_fn(list(seeds[polarity]))
        if vectors.ndim != 2 or vectors.shape[0] != len(seeds[polarity]):
            raise ValueError(
                f"embed_fn must return (n, d); got {vectors.shape} for polarity {polarity!r}"
            )
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            centroid = centroid / norm
        centroids[polarity] = centroid.astype(np.float32, copy=False)
    return centroids


def median_discrimination_margin(
    centroids: Mapping[str, np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, float]:
    """Per-polarity median ``margin`` on the held-out set (§5).

    ``margin = cos(sentence, centroid_true) - max_{q!=true} cos(sentence, centroid_q)``

    Returns ``{polarity: median_margin}``. Callers decide whether each
    polarity clears the per-corpus gate — this function is pure measurement.
    """
    heldout = load_heldout()
    results: dict[str, float] = {}
    for polarity in PREFERENCE_POLARITIES:
        vectors = embed_fn(list(heldout[polarity]))
        margins: list[float] = []
        for row in vectors:
            row_norm = float(np.linalg.norm(row))
            unit = row / row_norm if row_norm > 0.0 else row
            scores = {p: float(unit @ centroids[p]) for p in PREFERENCE_POLARITIES}
            top = scores[polarity]
            second = max(v for p, v in scores.items() if p != polarity)
            margins.append(top - second)
        margins.sort()
        mid = len(margins) // 2
        if len(margins) % 2 == 0:
            results[polarity] = (margins[mid - 1] + margins[mid]) / 2.0
        else:
            results[polarity] = margins[mid]
    return results


__all__ = [
    "HELDOUT_FILE",
    "HELDOUT_HASH",
    "SEEDS_FILE",
    "SEED_HASH",
    "compute_centroids",
    "load_heldout",
    "load_seeds",
    "median_discrimination_margin",
]
