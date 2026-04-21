"""Stage [7] — Preference detection.

Prototype-centroid classification per ``docs/design/ingestion.md §9``.
Model-free at the caller-visible surface: the detector takes precomputed
centroids and a text-embedding callable; production wires the real encoder,
tests can pass a mock.

**Fails closed.** A Sentence whose top-polarity margin is below
``MemoryConfig.preference_discrimination_margin`` emits no Preference node
and no ``holds_preference`` edge. R6-compliant: low-confidence structure
would taint multi-Memory aggregation.

**R16.** Preference is its own content-addressed node (not a Claim
overlay). Identity: ``(holder_id, polarity, target_id_or_literal)``. Two
sentences expressing the same preference converge to one Preference node;
each expression is a separate ``holds_preference`` edge from the speaker
Entity (evidence strength on the edge weight). Reinforcement is derived.

**Per-polarity fails closed (held-out gate).** A polarity whose median
margin on the disjoint held-out set sits below the runtime margin threshold
is *blanket-disabled* for the whole corpus; no Preferences of that polarity
are ever emitted. Computed by callers via
:func:`engram.ingestion.preferences.median_discrimination_margin`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from engram.ingestion.schema import (
    PREFERENCE_POLARITIES,
    ClaimPayload,
    PreferencePayload,
    node_id,
    preference_identity,
)


@dataclass(frozen=True, slots=True)
class PreferenceVerdict:
    """Output of the per-Claim classifier — may be ``None`` when fails closed."""

    polarity: str
    confidence: float


def classify(
    claim_text: str,
    centroids: Mapping[str, np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    margin_threshold: float,
    enabled_polarities: frozenset[str],
) -> PreferenceVerdict | None:
    """Return the top polarity and confidence for ``claim_text``, or ``None``.

    - Embeds ``claim_text`` via ``embed_fn``.
    - Computes cosine similarity to each polarity's centroid.
    - Top polarity wins only if ``top - second >= margin_threshold`` **and**
      the top polarity is in ``enabled_polarities``.
    - Returns ``None`` when the gate fails — caller keeps the Claim as-is.
    """
    if not enabled_polarities:
        return None
    vectors = embed_fn([claim_text])
    if vectors.shape[0] != 1:
        raise ValueError(f"embed_fn must return a (1, d) vector for a single input; got {vectors.shape}")
    row = vectors[0]
    norm = float(np.linalg.norm(row))
    unit = row / norm if norm > 0.0 else row

    scores: dict[str, float] = {p: float(unit @ centroids[p]) for p in PREFERENCE_POLARITIES}

    # Pick top allowed polarity + next-best across *all* polarities (disallowed
    # included) so a disabled polarity still suppresses its neighbors.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_polarity, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else float("-inf")

    if top_polarity not in enabled_polarities:
        return None
    if (top_score - second_score) < margin_threshold:
        return None
    confidence = max(0.0, min(1.0, top_score))
    return PreferenceVerdict(polarity=top_polarity, confidence=confidence)


def build_preference_payload(
    verdict: PreferenceVerdict,
    claim_payload: ClaimPayload,
    speaker_entity_id: str,
) -> tuple[str, PreferencePayload]:
    """Materialize a content-addressed Preference node (R16).

    Returns ``(preference_node_id, payload)``. The holder is the speaker.
    Target comes from the Claim's object slot — entity-resolved target
    preferred; falls back to literal. ``verdict.confidence`` is not on the
    payload — it belongs on the ``holds_preference`` edge weight
    (per-observation evidence strength).
    """
    payload = PreferencePayload(
        holder_id=speaker_entity_id,
        polarity=verdict.polarity,
        target_id=claim_payload.object_id,
        target_literal=claim_payload.object_literal,
    )
    pid = node_id(
        preference_identity(
            holder_id=payload.holder_id,
            polarity=payload.polarity,
            target_id=payload.target_id,
            target_literal=payload.target_literal,
        )
    )
    return pid, payload


__all__ = [
    "PreferenceVerdict",
    "build_preference_payload",
    "classify",
]
