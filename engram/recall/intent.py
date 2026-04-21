"""Stage [1] — Intent classification.

Prototype-centroid classifier per ``docs/design/recall.md §4``. Same pattern
as :mod:`engram.ingestion.extractors.preference`: centroids are hand-authored
seed-query means; a query is embedded once and scored against all five
intent centroids; the top-over-second margin must clear
:attr:`engram.config.MemoryConfig.intent_discrimination_margin`, otherwise
the classifier falls back to ``single_fact`` with low ``intent_confidence``.

**R6 compliance.** No English-specific regex. No rule-based "does it start
with 'how many'" heuristics. The classifier is pure cosine geometry over
frozen seed centroids.

**Fails closed.** Below-margin → ``single_fact`` + margin returned verbatim.
The recall pipeline surfaces the raw margin in :class:`RecallResult` so the
agent can choose to re-query.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from engram.recall.intents import INTENT_SINGLE_FACT, INTENTS


@dataclass(frozen=True, slots=True)
class IntentVerdict:
    """Classifier output — always populated (fallback is a valid verdict)."""

    intent: str
    margin: float


def classify_intent(
    query: str,
    centroids: Mapping[str, np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
    *,
    margin_threshold: float,
    fallback: str = INTENT_SINGLE_FACT,
) -> IntentVerdict:
    """Classify ``query`` into one of :data:`INTENTS` (fails closed).

    Returns :class:`IntentVerdict` with the resolved ``intent`` and the
    raw ``margin = top - second``. When ``margin < margin_threshold``,
    ``intent`` is ``fallback`` but ``margin`` still carries the actual
    value — callers use it as :attr:`RecallResult.intent_confidence`.
    """
    if fallback not in INTENTS:
        raise ValueError(f"fallback intent {fallback!r} not in {INTENTS}")

    vectors = embed_fn([query])
    if vectors.ndim != 2 or vectors.shape[0] != 1:
        raise ValueError(
            f"embed_fn must return a (1, d) vector for a single query; got {vectors.shape}"
        )
    row = vectors[0]
    norm = float(np.linalg.norm(row))
    unit = row / norm if norm > 0.0 else row

    scores: dict[str, float] = {i: float(unit @ centroids[i]) for i in INTENTS}
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_intent, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else float("-inf")
    margin = top_score - second_score

    if margin < margin_threshold:
        return IntentVerdict(intent=fallback, margin=margin)
    return IntentVerdict(intent=top_intent, margin=margin)


__all__ = [
    "IntentVerdict",
    "classify_intent",
]
