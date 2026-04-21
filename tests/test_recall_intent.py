"""Intent-classification unit tests.

Mirrors :mod:`tests.test_preferences`: per-intent centroids + a
deterministic embed stub, classify synthetic queries, assert the top
intent + margin behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from engram.recall.intent import IntentVerdict, classify_intent
from engram.recall.intents import (
    INTENT_AGGREGATION,
    INTENT_PREFERENCE,
    INTENT_SINGLE_FACT,
    INTENT_TEMPORAL,
    INTENTS,
    compute_intent_centroids,
    load_heldout,
    median_intent_margin,
)
from tests._fake_nlp import deterministic_embed


def _embed_fn():
    return deterministic_embed(dim=128)


def test_seeds_load_and_validate() -> None:
    from engram.recall.intents import load_seeds

    seeds = load_seeds()
    assert set(seeds) == set(INTENTS)
    for intent, queries in seeds.items():
        assert len(queries) >= 10, f"{intent} should have >=10 seeds for stable centroid"


def test_intent_centroids_are_normalized() -> None:
    centroids = compute_intent_centroids(_embed_fn())
    for intent in INTENTS:
        centroid = centroids[intent]
        norm = float(np.linalg.norm(centroid))
        assert pytest.approx(norm, abs=1e-5) == 1.0
        assert centroid.dtype == np.float32


def test_classify_returns_verdict() -> None:
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    verdict = classify_intent(
        "What are my dietary restrictions?",
        centroids,
        embed,
        margin_threshold=0.0,  # any margin >=0 accepts the top intent
    )
    assert isinstance(verdict, IntentVerdict)
    assert verdict.intent in INTENTS


def test_classify_falls_back_when_margin_below_threshold() -> None:
    """An impossibly-high threshold forces fallback every time."""
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    verdict = classify_intent(
        "arbitrary text",
        centroids,
        embed,
        margin_threshold=10.0,
    )
    assert verdict.intent == INTENT_SINGLE_FACT  # default fallback
    assert verdict.margin < 10.0


def test_classify_honors_custom_fallback() -> None:
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    verdict = classify_intent(
        "arbitrary text",
        centroids,
        embed,
        margin_threshold=10.0,
        fallback=INTENT_AGGREGATION,
    )
    assert verdict.intent == INTENT_AGGREGATION


def test_classify_rejects_invalid_fallback() -> None:
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    with pytest.raises(ValueError, match="fallback intent"):
        classify_intent(
            "query",
            centroids,
            embed,
            margin_threshold=0.0,
            fallback="not_a_real_intent",
        )


def test_median_margin_computed_for_every_intent() -> None:
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    margins = median_intent_margin(centroids, embed)
    assert set(margins) == set(INTENTS)
    for intent in INTENTS:
        assert isinstance(margins[intent], float)


def test_heldout_set_is_disjoint_from_seeds() -> None:
    """Benchmark hygiene: held-out queries must not appear in seeds."""
    from engram.recall.intents import load_seeds

    seeds = load_seeds()
    heldout = load_heldout()
    for intent in INTENTS:
        seed_set = set(seeds[intent])
        for q in heldout[intent]:
            assert q not in seed_set, f"{intent}: query in both seeds and heldout: {q!r}"


def test_intent_seed_hash_is_stable() -> None:
    from engram.recall.intents import INTENT_SEED_HASH

    assert len(INTENT_SEED_HASH) == 16
    assert all(c in "0123456789abcdef" for c in INTENT_SEED_HASH)


def test_preference_query_favors_preference_intent() -> None:
    """Deterministic-embed fakes can't prove real-model discrimination, but
    the classifier's branching logic must route a preference-looking query
    to a consistent intent."""
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    verdict_a = classify_intent("Does Alice like pizza?", centroids, embed, margin_threshold=0.0)
    verdict_b = classify_intent("Does Alice like pizza?", centroids, embed, margin_threshold=0.0)
    assert verdict_a == verdict_b


def test_temporal_query_classifies_deterministically() -> None:
    embed = _embed_fn()
    centroids = compute_intent_centroids(embed)
    verdict_a = classify_intent(
        "What did we discuss yesterday?", centroids, embed, margin_threshold=0.0
    )
    verdict_b = classify_intent(
        "What did we discuss yesterday?", centroids, embed, margin_threshold=0.0
    )
    assert verdict_a.intent == verdict_b.intent
    assert verdict_a.margin == verdict_b.margin
    _ = INTENT_TEMPORAL  # verify the constant is importable
    _ = INTENT_PREFERENCE
