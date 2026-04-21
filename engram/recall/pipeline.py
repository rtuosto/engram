"""Five-stage recall pipeline orchestrator (``docs/design/recall.md §3``).

Ties together intent classification, seeding (semantic + entity-anchored),
bounded typed-edge BFS, passage selection, and :class:`RecallResult`
assembly. Engram makes zero LLM calls on this path (``R5``, ``R13``); the
pipeline is pure NLP + vector search + graph traversal + structured
composition.

**Lazy derived rebuild.** :meth:`RecallPipeline.recall` checks the derived
snapshot's fingerprint; stale → calls :func:`rebuild_derived` before fact
assembly. Most invocations are a no-op (primary unchanged since last
rebuild).

**Recall fingerprint.** Returned on every :class:`RecallResult`. Composed
from the config's ``recall_fingerprint``, the query string, and the
supplied :class:`RecallContext`. Benchmark cache keys use this directly.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from engram.config import MemoryConfig
from engram.ingestion.derived import (
    DerivedIndex,
    derived_fingerprint,
    rebuild_derived,
)
from engram.ingestion.pipeline import InstanceState
from engram.models import RecallResult
from engram.recall.assembly import (
    build_facts,
    build_passages,
    resolve_query_entity_ids,
)
from engram.recall.context import RecallContext
from engram.recall.expansion import expand
from engram.recall.intent import IntentVerdict, classify_intent
from engram.recall.intents import INTENTS
from engram.recall.scoring import select_passages
from engram.recall.seeding import (
    entity_anchored_seed,
    merge_seeds,
    semantic_seed,
)
from engram.recall.weights import load_weights


@dataclass(frozen=True, slots=True)
class _Stage:
    name: str
    start: float


class RecallPipeline:
    """Orchestrates the five recall stages.

    Construction takes model / data dependencies (intent centroids, query
    embedder, query NER callable). The pipeline is stateless — all
    per-call state is threaded through :meth:`recall`'s arguments.
    """

    def __init__(
        self,
        *,
        config: MemoryConfig,
        intent_centroids: dict[str, np.ndarray],
        query_embed: Callable[[list[str]], np.ndarray],
        nlp_process: Callable[[list[str]], list[object]],
    ) -> None:
        self._config = config
        self._intent_centroids = intent_centroids
        self._query_embed = query_embed
        self._nlp_process = nlp_process
        gran_weights, edge_weights = load_weights()
        self._granularity_weights = gran_weights
        self._edge_weights = edge_weights

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recall(
        self,
        state: InstanceState,
        query: str,
        *,
        context: RecallContext,
    ) -> RecallResult:
        """Run the five-stage pipeline and return a :class:`RecallResult`."""
        timings: list[tuple[str, float]] = []
        total_start = time.perf_counter()

        # [1] Intent classification (or honor the agent's hint).
        stage = _Stage("classify", time.perf_counter())
        verdict = self._resolve_intent(query, context.intent_hint)
        timings.append((stage.name, _elapsed_ms(stage.start)))

        intent = verdict.intent
        granularity_weights = self._granularity_weights[intent]
        edge_weights = self._edge_weights[intent]

        # [2] Seeding.
        stage = _Stage("seed", time.perf_counter())
        seeds, query_entity_ids = self._seed(state, query, granularity_weights)
        timings.append((stage.name, _elapsed_ms(stage.start)))

        if not seeds:
            timings.append(("expand", 0.0))
            timings.append(("score", 0.0))
            stage = _Stage("assemble", time.perf_counter())
            empty = self._empty_result(verdict, query, context, timings, total_start)
            return empty

        # [3] Expansion.
        stage = _Stage("expand", time.perf_counter())
        walk = expand(
            state.store,
            seeds,
            edge_weights,
            max_depth=self._config.recall_max_depth,
            max_frontier=self._config.recall_max_frontier,
        )
        timings.append((stage.name, _elapsed_ms(stage.start)))

        # [4] Scoring / selection.
        stage = _Stage("score", time.perf_counter())
        max_passages = (
            context.max_passages
            if context.max_passages is not None
            else self._config.recall_max_passages
        )
        ranked = select_passages(walk, state.store, max_passages=max_passages)
        timings.append((stage.name, _elapsed_ms(stage.start)))

        # [5] Assembly.
        stage = _Stage("assemble", time.perf_counter())
        derived = self._ensure_derived(state)
        passages = build_passages(ranked, state.store)
        facts = build_facts(intent, passages, query_entity_ids, derived, state.store)
        timings.append((stage.name, _elapsed_ms(stage.start)))

        total_ms = _elapsed_ms(total_start)
        timings.append(("total", total_ms))

        return RecallResult(
            passages=passages,
            facts=facts,
            intent=intent,
            intent_confidence=verdict.margin,
            timing_ms=tuple(timings),
            recall_fingerprint=_recall_fingerprint(self._config, query, context),
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _resolve_intent(self, query: str, intent_hint: str | None) -> IntentVerdict:
        if intent_hint is not None:
            if intent_hint not in INTENTS:
                raise ValueError(
                    f"intent_hint {intent_hint!r} not in {INTENTS}"
                )
            # The agent's hint skips classification; confidence is 1.0 by
            # construction because the agent asserted it.
            return IntentVerdict(intent=intent_hint, margin=1.0)
        return classify_intent(
            query,
            self._intent_centroids,
            self._query_embed,
            margin_threshold=self._config.intent_discrimination_margin,
        )

    def _seed(
        self,
        state: InstanceState,
        query: str,
        granularity_weights: dict[str, float],
    ) -> tuple[list[tuple[str, float]], tuple[str, ...]]:
        semantic: list[tuple[str, float]] = []
        if state.vector_index is not None and len(state.vector_index) > 0:
            semantic = semantic_seed(
                query,
                state.vector_index,
                self._query_embed,
                granularity_weights=granularity_weights,
                top_n_per_granularity=self._config.recall_top_n_per_granularity,
            )

        docs = self._nlp_process([query])
        query_doc = docs[0] if docs else None
        registry_map = state.entity_registry.by_type_and_form

        entity_seeds: list[tuple[str, float]] = []
        query_entity_ids: tuple[str, ...] = ()
        if query_doc is not None:
            entity_seeds = entity_anchored_seed(
                query_doc,  # type: ignore[arg-type]
                state.entity_registry,
                state.store,
            )
            query_entity_ids = resolve_query_entity_ids(
                query_doc, registry_map, state.store
            )

        merged = merge_seeds(
            semantic,
            entity_seeds,
            total_cap=self._config.recall_seed_count_total,
        )
        return merged, query_entity_ids

    def _ensure_derived(self, state: InstanceState) -> DerivedIndex | None:
        expected = derived_fingerprint(self._config, state.store)
        if state.derived is not None and state.derived.fingerprint == expected:
            return state.derived
        snapshot = rebuild_derived(state.store, config=self._config)
        state.derived = snapshot
        return snapshot

    def _empty_result(
        self,
        verdict: IntentVerdict,
        query: str,
        context: RecallContext,
        timings: list[tuple[str, float]],
        total_start: float,
    ) -> RecallResult:
        timings.append(("total", _elapsed_ms(total_start)))
        return RecallResult(
            passages=(),
            facts=(),
            intent=verdict.intent,
            intent_confidence=verdict.margin,
            timing_ms=tuple(timings),
            recall_fingerprint=_recall_fingerprint(self._config, query, context),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 6)


def _recall_fingerprint(
    config: MemoryConfig, query: str, context: RecallContext
) -> str:
    payload: dict[str, object] = {
        "config": config.recall_fingerprint(),
        "query": query,
        "now": context.now,
        "timezone": context.timezone,
        "max_passages": context.max_passages,
        "intent_hint": context.intent_hint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


__all__ = [
    "RecallPipeline",
]
