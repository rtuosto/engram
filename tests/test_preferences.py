"""Preference fixture + classifier behavior tests.

Validates the shipped seed / held-out files load cleanly, the centroid
construction is deterministic under a fake embedder, and the classifier
fails closed as documented.

The integration against the real ``all-mpnet-base-v2`` model lives in a
separate ``slow``-marked test (not in this file) — see §5 ingress criterion.
"""

from __future__ import annotations

from engram.ingestion.extractors.preference import PreferenceVerdict, classify
from engram.ingestion.preferences import (
    HELDOUT_FILE,
    HELDOUT_HASH,
    SEED_HASH,
    SEEDS_FILE,
    compute_centroids,
    load_heldout,
    load_seeds,
    median_discrimination_margin,
)
from engram.ingestion.schema import PREFERENCE_POLARITIES
from tests._fake_nlp import deterministic_embed


def test_seed_and_heldout_files_exist() -> None:
    assert SEEDS_FILE.exists()
    assert HELDOUT_FILE.exists()


def test_seed_and_heldout_hashes_are_16_hex() -> None:
    assert len(SEED_HASH) == 16
    assert len(HELDOUT_HASH) == 16
    int(SEED_HASH, 16)  # raises if not hex
    int(HELDOUT_HASH, 16)


def test_load_seeds_covers_every_polarity_and_nonempty() -> None:
    seeds = load_seeds()
    assert set(seeds.keys()) == set(PREFERENCE_POLARITIES)
    for polarity, sentences in seeds.items():
        assert len(sentences) >= 5, f"polarity {polarity} has {len(sentences)} seeds (§5: 10-20)"


def test_load_heldout_disjoint_from_seeds() -> None:
    seeds = load_seeds()
    heldout = load_heldout()
    assert set(heldout.keys()) == set(PREFERENCE_POLARITIES)
    for polarity in PREFERENCE_POLARITIES:
        # Held-out sentences must not be direct duplicates of training seeds.
        overlap = set(heldout[polarity]) & set(seeds[polarity])
        assert not overlap, f"polarity {polarity} leaks {overlap} across train/held-out"


def test_compute_centroids_returns_normalized_vectors() -> None:
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    assert set(centroids.keys()) == set(PREFERENCE_POLARITIES)
    import numpy as np

    for centroid in centroids.values():
        assert centroid.shape == (16,)
        # mean of L2-normalized vectors → may have norm < 1, but not NaN.
        assert not np.isnan(centroid).any()


def test_median_discrimination_margin_per_polarity_is_finite() -> None:
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    margins = median_discrimination_margin(centroids, embed)
    assert set(margins.keys()) == set(PREFERENCE_POLARITIES)
    import math
    for polarity, margin in margins.items():
        assert math.isfinite(margin), f"polarity {polarity} margin is {margin}"


def test_classify_fails_closed_below_margin() -> None:
    """A classifier with an unrealistically high margin threshold must emit None."""
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    verdict = classify(
        "I love spicy food",
        centroids,
        embed,
        margin_threshold=2.0,  # impossibly high
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )
    assert verdict is None


def test_classify_fails_closed_when_polarity_disabled() -> None:
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    # Enable only one polarity. If the top polarity happens to be the
    # enabled one, accept that — but assert that disabling every polarity
    # always yields None.
    nothing_enabled = classify(
        "I love spicy food",
        centroids,
        embed,
        margin_threshold=-1.0,
        enabled_polarities=frozenset(),
    )
    assert nothing_enabled is None


def test_classify_returns_verdict_when_enabled_and_above_margin() -> None:
    """With a permissive margin and all polarities enabled, some input must classify."""
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    verdict = classify(
        "I love spicy food",
        centroids,
        embed,
        margin_threshold=-1.0,  # impossibly low — always passes
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )
    assert isinstance(verdict, PreferenceVerdict)
    assert verdict.polarity in PREFERENCE_POLARITIES
