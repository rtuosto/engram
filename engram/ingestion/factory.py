"""Builders that wire the real spaCy / sentence-transformers models.

Kept separate from :mod:`engram.ingestion.pipeline` so tests can import the
pipeline without pulling in heavy model loads. ``build_default_pipeline``
is the production entry point.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from engram.config import MemoryConfig
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import (
    compute_centroids,
    median_discrimination_margin,
)
from engram.ingestion.schema import PREFERENCE_POLARITIES


def _seed_rngs(seed: int) -> None:
    """Seed every RNG that could bleed into ingestion output.

    Called once per pipeline construction. PYTHONHASHSEED can only be set at
    process launch, but we set it defensively in case a caller re-spawns.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import random as _random

        _random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass


def _load_spacy(model_name: str) -> Callable[[list[str]], list[object]]:
    """Return a batching wrapper around ``spacy.load(model_name).pipe``."""
    import spacy  # local import: heavy dependency, model weights on first call

    nlp = spacy.load(model_name)

    def process(texts: list[str]) -> list[object]:
        # spaCy's pipe is R2-stable on CPU for fixed model versions. We
        # materialize the list to a stable order before handing to the
        # pipeline.
        return list(nlp.pipe(texts))

    return process


def _load_embed_fn(model_name: str) -> Callable[[list[str]], np.ndarray]:
    """Return a normalized-embedding wrapper around sentence-transformers."""
    from sentence_transformers import SentenceTransformer  # local import

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    return embed


def build_default_pipeline(config: MemoryConfig) -> IngestionPipeline:
    """Wire the real spaCy model, preference encoder, centroids, and gate.

    Called lazily by :class:`EngramGraphMemorySystem` on the first ingest so
    a bare construct-and-reset cycle doesn't pay the model-load cost.
    """
    _seed_rngs(config.random_seed)

    nlp_process = _load_spacy(config.spacy_model)
    preference_embed = _load_embed_fn(config.preference_embedding_model)

    centroids = compute_centroids(preference_embed)
    margins = median_discrimination_margin(centroids, preference_embed)
    enabled = frozenset(
        p for p in PREFERENCE_POLARITIES if margins[p] >= config.preference_discrimination_margin
    )

    return IngestionPipeline(
        config=config,
        nlp_process=nlp_process,
        preference_centroids=centroids,
        preference_embed=preference_embed,
        enabled_polarities=enabled,
    )


__all__ = ["build_default_pipeline"]
