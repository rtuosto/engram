"""Builder that wires real spaCy + sentence-transformers for the recall pipeline.

Kept separate from :mod:`engram.recall.pipeline` so tests can import the
pipeline without pulling in heavy model loads. Mirrors
:mod:`engram.ingestion.factory` — same NLP/embed model sourcing, identical
RNG-seeding discipline so ingest + recall stay deterministic across a shared
``MemoryConfig``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from engram.config import MemoryConfig
from engram.recall.intents import compute_intent_centroids
from engram.recall.pipeline import RecallPipeline


def _seed_rngs(seed: int) -> None:
    """Match :func:`engram.ingestion.factory._seed_rngs` for recall paths."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import random as _random

        _random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass


def _load_spacy(model_name: str) -> Callable[[list[str]], list[object]]:
    import spacy  # local import: heavy dependency

    nlp = spacy.load(model_name)

    def process(texts: list[str]) -> list[object]:
        return list(nlp.pipe(texts))

    return process


def _load_embed_fn(model_name: str) -> Callable[[list[str]], np.ndarray]:
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


def build_default_recall_pipeline(config: MemoryConfig) -> RecallPipeline:
    """Construct a production-wired :class:`RecallPipeline`.

    Uses :attr:`MemoryConfig.embedding_model` for both the query embedder
    (so query + granules share an embedding space) and for computing intent
    centroids from the hand-authored seed queries.
    """
    _seed_rngs(config.random_seed)

    query_embed = _load_embed_fn(config.embedding_model)
    nlp_process = _load_spacy(config.spacy_model)
    intent_centroids = compute_intent_centroids(query_embed)

    return RecallPipeline(
        config=config,
        intent_centroids=intent_centroids,
        query_embed=query_embed,
        nlp_process=nlp_process,
    )


__all__ = ["build_default_recall_pipeline"]
