"""Recall play-by-play diagnostic.

Re-runs the five recall stages against a persisted :class:`InstanceState`
while capturing every intermediate dataset — classified intent cosines,
per-granularity semantic seeds, entity-anchored seeds, BFS frontiers at
each depth, per-edge-type traversal counts, granule bucket resolution,
and final passage selection.

**Production-path safe.** This module does not hook into
:class:`engram.recall.pipeline.RecallPipeline`. It replicates the pipeline
stages here (calling the same leaf functions — ``classify_intent``,
``semantic_seed``, ``entity_anchored_seed``, ``select_passages``,
``build_passages``, ``build_facts``) and inlines the BFS so it can capture
per-depth frontiers. The production recall path is bit-for-bit unchanged.
That matters for R2/R3/R4 — tracing is diagnostic, not part of the bar.

**R5/R13 compliance.** Zero LLM calls. The same embedder + spaCy NER
the recall pipeline uses are reused (passed in via ``RecallPipeline``);
tracing does not add any model dependencies.

Typical use (from :mod:`engram.engram_memory_system.EngramGraphMemorySystem`
via :meth:`recall_trace` or :mod:`scripts.trace_recall`)::

    result, trace = traced_recall(pipeline, state, query, context)
    print(trace.pretty())
    # or
    import json; json.dump(trace.to_dict(), ..., indent=2)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engram.config import MemoryConfig
from engram.ingestion.derived import (
    derived_fingerprint,
    rebuild_derived,
)
from engram.ingestion.graph import GraphStore
from engram.ingestion.pipeline import InstanceState
from engram.ingestion.schema import (
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_NGRAM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
)
from engram.ingestion.vector_index import (
    GRANULARITY_NGRAM,
    GRANULARITY_SENTENCE,
    GRANULARITY_TURN,
)
from engram.models import RecallResult
from engram.recall.assembly import (
    build_facts,
    build_passages,
    resolve_query_entity_ids,
)
from engram.recall.context import RecallContext
from engram.recall.intent import IntentVerdict
from engram.recall.intents import INTENT_SINGLE_FACT, INTENTS
from engram.recall.pipeline import RecallPipeline, _recall_fingerprint  # noqa: PLC2701
from engram.recall.scoring import select_passages
from engram.recall.seeding import (
    ENTITY_GRANULE_SEED_SCORE,
    ENTITY_SEED_SCORE,
    entity_anchored_seed,
    merge_seeds,
    semantic_seed,
)

_PREVIEW_CHARS: int = 160


# ---------------------------------------------------------------------------
# Stage-level dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntentStageTrace:
    """Stage [1] — intent classification."""

    used_hint: bool
    intent_hint: str | None
    scores_by_intent: dict[str, float]
    chosen: str
    margin: float
    margin_threshold: float
    fell_back: bool
    fallback_intent: str


@dataclass(frozen=True, slots=True)
class SeedEntry:
    """One entry in the merged seed list after cap."""

    node_id: str
    score: float
    source: str  # "semantic" | "entity" | "entity_granule"
    granularity: str | None  # "turn" | "sentence" | "ngram" | None for entities
    text_preview: str | None


@dataclass(frozen=True, slots=True)
class SeedStageTrace:
    """Stage [2] — seeding (semantic + entity-anchored, merged + capped)."""

    query_entity_ids: tuple[str, ...]
    query_entity_surfaces: tuple[str, ...]
    granularity_weights: dict[str, float]
    top_n_per_granularity: int
    total_cap: int
    # Raw seeds, pre-merge:
    semantic_seed_count: int
    entity_seed_count: int
    # Merged + capped:
    merged_seed_count: int
    was_capped: bool
    # Expose top-K of each source + full merged list (capped to a readable
    # size in pretty-print but full in the JSON dump).
    semantic_top: tuple[SeedEntry, ...]
    entity_top: tuple[SeedEntry, ...]
    merged: tuple[SeedEntry, ...]


@dataclass(frozen=True, slots=True)
class ExpandStep:
    """One BFS depth-step within stage [3]."""

    depth: int
    frontier_in_size: int  # size at start of step
    edges_considered: int
    edges_traversed: int  # passed edge_weights / positive-contrib filter
    edges_by_type: dict[str, int]
    frontier_out_size_before_cap: int
    frontier_out_size: int  # after max_frontier cap
    was_capped: bool
    newly_reached: int  # nodes added to global scores this step


@dataclass(frozen=True, slots=True)
class ExpandStageTrace:
    """Stage [3] — bounded typed-edge BFS."""

    edge_weights: dict[str, float]
    max_depth: int
    max_frontier: int
    seed_count: int
    final_node_count: int
    total_edges_traversed: int
    steps: tuple[ExpandStep, ...]


@dataclass(frozen=True, slots=True)
class ScoredGranule:
    """One granule bucket result with routing + preview."""

    granule_id: str
    granularity: str  # "turn" | "sentence" | "ngram"
    score: float
    text_preview: str


@dataclass(frozen=True, slots=True)
class DroppedNode:
    """A walk-visited node that did not route to a granule passage."""

    node_id: str
    label: str  # single label name (the first granule-ish / non-granule-ish label)
    score: float


@dataclass(frozen=True, slots=True)
class ScoreStageTrace:
    """Stage [4] — granule bucket resolution + ranking."""

    walk_node_count: int
    max_passages: int
    granules_considered: int
    selected: tuple[ScoredGranule, ...]
    # Non-granule scores that were ignored (Entity / Claim / Preference /
    # TimeAnchor / Memory) — bucket for understanding why a walk-reached
    # node didn't appear in passages.
    dropped_non_granules: tuple[DroppedNode, ...]


@dataclass(frozen=True, slots=True)
class AssembleStageTrace:
    """Stage [5] — derived-rebuild gate + assembly counts."""

    derived_was_stale: bool
    derived_rebuilt: bool
    passages_assembled: int
    facts_assembled: int
    facts_by_type: dict[str, int]  # {"current_preference": n, ...}


@dataclass(frozen=True, slots=True)
class RecallTrace:
    """Full play-by-play of a single recall call."""

    query: str
    now: str | None
    timezone: str | None
    max_passages: int | None
    intent_hint: str | None
    recall_fingerprint: str

    intent: IntentStageTrace
    seed: SeedStageTrace
    expand: ExpandStageTrace
    score: ScoreStageTrace
    assemble: AssembleStageTrace

    timing_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (ndarray/tuple normalization)."""
        return _to_jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def pretty(self) -> str:
        """Human-readable play-by-play."""
        return _pretty(self)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def traced_recall(
    pipeline: RecallPipeline,
    state: InstanceState,
    query: str,
    context: RecallContext,
) -> tuple[RecallResult, RecallTrace]:
    """Execute all five recall stages and return ``(result, trace)``.

    Replicates the orchestration in :meth:`RecallPipeline.recall` but
    captures every intermediate dataset. The returned :class:`RecallResult`
    is equivalent to what the production path would produce for the same
    inputs (the leaf functions are identical).
    """
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    config: MemoryConfig = pipeline._config  # noqa: SLF001
    intent_centroids = pipeline._intent_centroids  # noqa: SLF001
    query_embed = pipeline._query_embed  # noqa: SLF001
    nlp_process = pipeline._nlp_process  # noqa: SLF001
    gran_weights_by_intent = pipeline._granularity_weights  # noqa: SLF001
    edge_weights_by_intent = pipeline._edge_weights  # noqa: SLF001

    # ----- [1] Intent classification ----------------------------------
    t0 = time.perf_counter()
    intent_trace, verdict = _trace_intent(
        query=query,
        intent_hint=context.intent_hint,
        centroids=intent_centroids,
        embed_fn=query_embed,
        margin_threshold=config.intent_discrimination_margin,
    )
    timings["intent_ms"] = _elapsed_ms(t0)
    intent = verdict.intent
    granularity_weights = gran_weights_by_intent[intent]
    edge_weights = edge_weights_by_intent[intent]

    # ----- [2] Seeding ------------------------------------------------
    t0 = time.perf_counter()
    seed_trace, merged_seeds, query_entity_ids = _trace_seed(
        state=state,
        query=query,
        embed_fn=query_embed,
        nlp_process=nlp_process,
        granularity_weights=granularity_weights,
        top_n_per_granularity=config.recall_top_n_per_granularity,
        total_cap=config.recall_seed_count_total,
    )
    timings["seed_ms"] = _elapsed_ms(t0)

    if not merged_seeds:
        # Short-circuit: no seeds → empty result (same contract as
        # RecallPipeline._empty_result, but we still return a trace for
        # every stage so the diagnostic is complete.)
        expand_trace = ExpandStageTrace(
            edge_weights=dict(edge_weights),
            max_depth=config.recall_max_depth,
            max_frontier=config.recall_max_frontier,
            seed_count=0,
            final_node_count=0,
            total_edges_traversed=0,
            steps=(),
        )
        score_trace = ScoreStageTrace(
            walk_node_count=0,
            max_passages=_effective_max_passages(context, config),
            granules_considered=0,
            selected=(),
            dropped_non_granules=(),
        )
        assemble_trace = AssembleStageTrace(
            derived_was_stale=False,
            derived_rebuilt=False,
            passages_assembled=0,
            facts_assembled=0,
            facts_by_type={},
        )
        timings["expand_ms"] = 0.0
        timings["score_ms"] = 0.0
        timings["assemble_ms"] = 0.0
        timings["total_ms"] = _elapsed_ms(total_start)
        empty = RecallResult(
            passages=(),
            facts=(),
            intent=verdict.intent,
            intent_confidence=verdict.margin,
            timing_ms=tuple(timings.items()),
            recall_fingerprint=_recall_fingerprint(config, query, context),
        )
        trace = RecallTrace(
            query=query,
            now=context.now,
            timezone=context.timezone,
            max_passages=context.max_passages,
            intent_hint=context.intent_hint,
            recall_fingerprint=empty.recall_fingerprint or "",
            intent=intent_trace,
            seed=seed_trace,
            expand=expand_trace,
            score=score_trace,
            assemble=assemble_trace,
            timing_ms=timings,
        )
        return empty, trace

    # ----- [3] Expansion (inlined BFS with per-step capture) ----------
    t0 = time.perf_counter()
    walk_scores, expand_trace = _trace_expand(
        store=state.store,
        seeds=merged_seeds,
        edge_weights=edge_weights,
        max_depth=config.recall_max_depth,
        max_frontier=config.recall_max_frontier,
    )
    timings["expand_ms"] = _elapsed_ms(t0)

    # ----- [4] Scoring / selection ------------------------------------
    t0 = time.perf_counter()
    max_passages = _effective_max_passages(context, config)
    ranked = select_passages(walk_scores, state.store, max_passages=max_passages)
    score_trace = _trace_score(
        walk_scores=walk_scores,
        ranked=ranked,
        store=state.store,
        max_passages=max_passages,
    )
    timings["score_ms"] = _elapsed_ms(t0)

    # ----- [5] Assembly + derived rebuild gate ------------------------
    t0 = time.perf_counter()
    expected_fp = derived_fingerprint(config, state.store)
    was_stale = state.derived is None or state.derived.fingerprint != expected_fp
    rebuilt = False
    if was_stale:
        state.derived = rebuild_derived(state.store, config=config)
        rebuilt = True
    derived = state.derived

    passages = build_passages(ranked, state.store)
    facts = build_facts(intent, passages, query_entity_ids, derived, state.store)
    facts_by_type: dict[str, int] = {}
    for f in facts:
        facts_by_type[f.kind] = facts_by_type.get(f.kind, 0) + 1

    assemble_trace = AssembleStageTrace(
        derived_was_stale=was_stale,
        derived_rebuilt=rebuilt,
        passages_assembled=len(passages),
        facts_assembled=len(facts),
        facts_by_type=facts_by_type,
    )
    timings["assemble_ms"] = _elapsed_ms(t0)
    timings["total_ms"] = _elapsed_ms(total_start)

    result = RecallResult(
        passages=passages,
        facts=facts,
        intent=intent,
        intent_confidence=verdict.margin,
        timing_ms=tuple(timings.items()),
        recall_fingerprint=_recall_fingerprint(config, query, context),
    )
    trace = RecallTrace(
        query=query,
        now=context.now,
        timezone=context.timezone,
        max_passages=context.max_passages,
        intent_hint=context.intent_hint,
        recall_fingerprint=result.recall_fingerprint or "",
        intent=intent_trace,
        seed=seed_trace,
        expand=expand_trace,
        score=score_trace,
        assemble=assemble_trace,
        timing_ms=timings,
    )
    return result, trace


# ---------------------------------------------------------------------------
# Per-stage helpers
# ---------------------------------------------------------------------------


def _trace_intent(
    *,
    query: str,
    intent_hint: str | None,
    centroids: dict[str, np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
    margin_threshold: float,
) -> tuple[IntentStageTrace, IntentVerdict]:
    if intent_hint is not None:
        if intent_hint not in INTENTS:
            raise ValueError(f"intent_hint {intent_hint!r} not in {INTENTS}")
        trace = IntentStageTrace(
            used_hint=True,
            intent_hint=intent_hint,
            scores_by_intent={},
            chosen=intent_hint,
            margin=1.0,
            margin_threshold=margin_threshold,
            fell_back=False,
            fallback_intent=INTENT_SINGLE_FACT,
        )
        return trace, IntentVerdict(intent=intent_hint, margin=1.0)

    vectors = embed_fn([query])
    row = vectors[0]
    norm = float(np.linalg.norm(row))
    unit = row / norm if norm > 0.0 else row
    scores = {i: float(unit @ centroids[i]) for i in INTENTS}
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top, second = ordered[0], ordered[1] if len(ordered) > 1 else (None, float("-inf"))
    top_intent, top_score = top
    second_score = second[1] if isinstance(second, tuple) else float("-inf")
    margin = top_score - second_score
    fell_back = margin < margin_threshold
    chosen = INTENT_SINGLE_FACT if fell_back else top_intent
    trace = IntentStageTrace(
        used_hint=False,
        intent_hint=None,
        scores_by_intent=scores,
        chosen=chosen,
        margin=margin,
        margin_threshold=margin_threshold,
        fell_back=fell_back,
        fallback_intent=INTENT_SINGLE_FACT,
    )
    return trace, IntentVerdict(intent=chosen, margin=margin)


def _trace_seed(
    *,
    state: InstanceState,
    query: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    nlp_process: Callable[[list[str]], list[object]],
    granularity_weights: dict[str, float],
    top_n_per_granularity: int,
    total_cap: int,
) -> tuple[SeedStageTrace, list[tuple[str, float]], tuple[str, ...]]:
    semantic: list[tuple[str, float]] = []
    if state.vector_index is not None and len(state.vector_index) > 0:
        semantic = semantic_seed(
            query,
            state.vector_index,
            embed_fn,
            granularity_weights=granularity_weights,
            top_n_per_granularity=top_n_per_granularity,
        )

    docs = nlp_process([query])
    query_doc = docs[0] if docs else None
    entity_seeds: list[tuple[str, float]] = []
    query_entity_ids: tuple[str, ...] = ()
    query_entity_surfaces: list[str] = []
    if query_doc is not None:
        entity_seeds = entity_anchored_seed(
            query_doc,  # type: ignore[arg-type]
            state.entity_registry,
            state.store,
        )
        query_entity_ids = resolve_query_entity_ids(
            query_doc, state.entity_registry.by_type_and_form, state.store
        )
        for ent in getattr(query_doc, "ents", ()) or ():
            surface = str(getattr(ent, "text", "")).strip()
            if surface:
                query_entity_surfaces.append(surface)

    merged = merge_seeds(semantic, entity_seeds, total_cap=total_cap)
    was_capped = (len(semantic) + len(entity_seeds)) > len(merged)

    semantic_top = tuple(_seed_entry(n, s, "semantic", state) for n, s in semantic[:20])
    entity_top = tuple(
        _seed_entry(
            n,
            s,
            _entity_seed_kind(s),
            state,
        )
        for n, s in entity_seeds[:20]
    )
    merged_top = tuple(
        _seed_entry(
            n,
            s,
            _merged_seed_source(n, semantic, entity_seeds),
            state,
        )
        for n, s in merged
    )

    trace = SeedStageTrace(
        query_entity_ids=query_entity_ids,
        query_entity_surfaces=tuple(query_entity_surfaces),
        granularity_weights=dict(granularity_weights),
        top_n_per_granularity=top_n_per_granularity,
        total_cap=total_cap,
        semantic_seed_count=len(semantic),
        entity_seed_count=len(entity_seeds),
        merged_seed_count=len(merged),
        was_capped=was_capped,
        semantic_top=semantic_top,
        entity_top=entity_top,
        merged=merged_top,
    )
    return trace, list(merged), query_entity_ids


def _entity_seed_kind(score: float) -> str:
    # ENTITY_SEED_SCORE (1.0) → entity node; ENTITY_GRANULE_SEED_SCORE (0.8) → anchored granule.
    if score >= ENTITY_SEED_SCORE - 1e-9:
        return "entity"
    if abs(score - ENTITY_GRANULE_SEED_SCORE) < 1e-9:
        return "entity_granule"
    return "entity"


def _merged_seed_source(
    node_id: str,
    semantic: Sequence[tuple[str, float]],
    entity_seeds: Sequence[tuple[str, float]],
) -> str:
    sem = next((s for (n, s) in semantic if n == node_id), None)
    ent = next((s for (n, s) in entity_seeds if n == node_id), None)
    if sem is not None and ent is not None:
        return "both"
    if sem is not None:
        return "semantic"
    return "entity"


def _seed_entry(
    node_id: str, score: float, source: str, state: InstanceState
) -> SeedEntry:
    granularity = None
    text_preview = None
    if state.vector_index is not None and node_id in state.vector_index:
        granularity = state.vector_index.granularity_for(node_id)
    text_preview = _text_preview_for_node(node_id, state.store)
    return SeedEntry(
        node_id=node_id,
        score=float(score),
        source=source,
        granularity=granularity,
        text_preview=text_preview,
    )


def _trace_expand(
    *,
    store: GraphStore,
    seeds: list[tuple[str, float]],
    edge_weights: dict[str, float],
    max_depth: int,
    max_frontier: int,
) -> tuple[dict[str, float], ExpandStageTrace]:
    """Mirror of :meth:`GraphStore.bfs` with per-step capture."""
    scores: dict[str, float] = {}
    for nid, s in seeds:
        if s <= 0.0:
            continue
        scores[nid] = max(scores.get(nid, 0.0), s)
    frontier = dict(scores)
    steps: list[ExpandStep] = []
    total_edges_traversed = 0

    for depth in range(max_depth):
        if not frontier:
            break
        frontier_in_size = len(frontier)
        next_frontier: dict[str, float] = {}
        edges_considered = 0
        edges_traversed = 0
        edges_by_type: dict[str, int] = {}
        for src_id, src_score in sorted(frontier.items()):
            for dst_id, attrs in store.out_edges(src_id):
                edges_considered += 1
                w = edge_weights.get(attrs.type)
                if w is None or w <= 0.0:
                    continue
                contrib = src_score * attrs.weight * w
                if contrib <= 0.0:
                    continue
                edges_traversed += 1
                edges_by_type[attrs.type] = edges_by_type.get(attrs.type, 0) + 1
                next_frontier[dst_id] = max(next_frontier.get(dst_id, 0.0), contrib)

        if not next_frontier:
            steps.append(ExpandStep(
                depth=depth + 1,
                frontier_in_size=frontier_in_size,
                edges_considered=edges_considered,
                edges_traversed=edges_traversed,
                edges_by_type=edges_by_type,
                frontier_out_size_before_cap=0,
                frontier_out_size=0,
                was_capped=False,
                newly_reached=0,
            ))
            total_edges_traversed += edges_traversed
            break

        frontier_out_before = len(next_frontier)
        was_capped = frontier_out_before > max_frontier
        if was_capped:
            ranked = sorted(next_frontier.items(), key=lambda kv: (-kv[1], kv[0]))[:max_frontier]
            next_frontier = dict(ranked)

        newly_reached = 0
        for nid, s in next_frontier.items():
            prior = scores.get(nid, 0.0)
            if s > prior:
                if prior == 0.0:
                    newly_reached += 1
                scores[nid] = s

        steps.append(ExpandStep(
            depth=depth + 1,
            frontier_in_size=frontier_in_size,
            edges_considered=edges_considered,
            edges_traversed=edges_traversed,
            edges_by_type=edges_by_type,
            frontier_out_size_before_cap=frontier_out_before,
            frontier_out_size=len(next_frontier),
            was_capped=was_capped,
            newly_reached=newly_reached,
        ))
        total_edges_traversed += edges_traversed
        frontier = next_frontier

    trace = ExpandStageTrace(
        edge_weights=dict(edge_weights),
        max_depth=max_depth,
        max_frontier=max_frontier,
        seed_count=len(seeds),
        final_node_count=len(scores),
        total_edges_traversed=total_edges_traversed,
        steps=tuple(steps),
    )
    return scores, trace


_GRANULE_LABELS = frozenset({LABEL_TURN, LABEL_UTTERANCE_SEGMENT, LABEL_NGRAM})
_NON_GRANULE_LABELS = frozenset({
    LABEL_ENTITY,
    LABEL_CLAIM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_MEMORY,
})
_LABEL_TO_GRANULARITY = {
    LABEL_TURN: GRANULARITY_TURN,
    LABEL_UTTERANCE_SEGMENT: GRANULARITY_SENTENCE,
    LABEL_NGRAM: GRANULARITY_NGRAM,
}


def _trace_score(
    *,
    walk_scores: dict[str, float],
    ranked: list[tuple[str, float]],
    store: GraphStore,
    max_passages: int,
) -> ScoreStageTrace:
    granules_considered = 0
    selected: list[ScoredGranule] = []
    dropped: list[DroppedNode] = []

    ranked_set = {nid for (nid, _) in ranked}
    for nid, score in sorted(walk_scores.items(), key=lambda kv: (-kv[1], kv[0])):
        if score <= 0.0:
            continue
        if not store.has_node(nid):
            continue
        labels = store.node_labels(nid)
        if labels & _GRANULE_LABELS:
            granules_considered += 1
            continue
        # walk reached a non-granule (entity / claim / preference / time anchor / memory)
        primary = next(iter(sorted(labels))) if labels else "unknown"
        dropped.append(DroppedNode(node_id=nid, label=primary, score=float(score)))

    for nid, score in ranked:
        labels = store.node_labels(nid)
        granularity = next(
            (g for lbl, g in _LABEL_TO_GRANULARITY.items() if lbl in labels),
            "unknown",
        )
        selected.append(
            ScoredGranule(
                granule_id=nid,
                granularity=granularity,
                score=float(score),
                text_preview=_text_preview_for_node(nid, store) or "",
            )
        )
    # Keep a bounded number of dropped nodes to avoid flooding the trace.
    dropped_sorted = sorted(dropped, key=lambda d: (-d.score, d.node_id))[:25]

    return ScoreStageTrace(
        walk_node_count=len(walk_scores),
        max_passages=max_passages,
        granules_considered=granules_considered,
        selected=tuple(selected),
        dropped_non_granules=tuple(dropped_sorted),
        # ranked_set is used to mark whether a granule was selected vs
        # considered-but-not-picked in a future revision. Not exposed now.
    )


def _text_preview_for_node(node_id: str, store: GraphStore) -> str | None:
    if not store.has_node(node_id):
        return None
    labels = store.node_labels(node_id)
    attrs = store.get_node(node_id)
    text: str | None = None
    if LABEL_TURN in labels and LABEL_TURN in attrs:
        text = getattr(attrs[LABEL_TURN], "text", None)
    elif LABEL_UTTERANCE_SEGMENT in labels and LABEL_UTTERANCE_SEGMENT in attrs:
        text = getattr(attrs[LABEL_UTTERANCE_SEGMENT], "text", None)
    elif LABEL_NGRAM in labels and LABEL_NGRAM in attrs:
        text = getattr(attrs[LABEL_NGRAM], "surface_form", None)
    elif LABEL_ENTITY in labels and LABEL_ENTITY in attrs:
        text = getattr(attrs[LABEL_ENTITY], "canonical_form", None)
    if text is None:
        return None
    if len(text) > _PREVIEW_CHARS:
        return text[:_PREVIEW_CHARS] + "…"
    return text


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _effective_max_passages(context: RecallContext, config: MemoryConfig) -> int:
    if context.max_passages is not None:
        return context.max_passages
    return config.recall_max_passages


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {
            f: _to_jsonable(getattr(obj, f))
            for f in obj.__dataclass_fields__
        }
    if isinstance(obj, (frozenset, set)):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return repr(obj)


# ---------------------------------------------------------------------------
# Pretty-printer (human-readable play-by-play)
# ---------------------------------------------------------------------------


def _pretty(trace: RecallTrace) -> str:
    lines: list[str] = []
    w = lines.append

    def _tl(seconds_key: str) -> str:
        ms = trace.timing_ms.get(seconds_key, 0.0)
        return f"{ms:.1f}ms"

    w("=" * 78)
    w(f"RecallTrace   query={trace.query!r}")
    w(f"  intent_hint={trace.intent_hint}   max_passages={trace.max_passages}")
    w(f"  recall_fingerprint={trace.recall_fingerprint}")
    w(f"  total={_tl('total_ms')}")
    w("")

    # [1] Intent ---------------------------------------------------------
    t = trace.intent
    w(f"[1] INTENT  ({_tl('intent_ms')})")
    if t.used_hint:
        w(f"   hint-supplied: chose {t.chosen!r} (confidence=1.0)")
    else:
        ordered = sorted(t.scores_by_intent.items(), key=lambda kv: -kv[1])
        for i, (intent, score) in enumerate(ordered):
            marker = " <-- chosen" if intent == t.chosen and not t.fell_back else ""
            w(f"   {intent:22s}  cos={score:+.4f}{marker}")
        w(
            f"   margin={t.margin:.4f}  threshold={t.margin_threshold:.4f}  "
            f"{'BELOW → fallback='+t.fallback_intent if t.fell_back else 'clear'}"
        )
    w("")

    # [2] Seed -----------------------------------------------------------
    s = trace.seed
    w(f"[2] SEED  ({_tl('seed_ms')})")
    w(
        f"   semantic={s.semantic_seed_count}   entity={s.entity_seed_count}   "
        f"merged={s.merged_seed_count}   capped={s.was_capped}   total_cap={s.total_cap}"
    )
    gw = s.granularity_weights
    w(
        "   granularity_weights: "
        + ", ".join(f"{k}={gw[k]:.2f}" for k in (GRANULARITY_TURN, GRANULARITY_SENTENCE, GRANULARITY_NGRAM) if k in gw)
    )
    if s.query_entity_surfaces:
        matched = set(s.query_entity_ids)
        w(
            f"   query-NER: {list(s.query_entity_surfaces)}  "
            f"resolved_ids={len(matched)}"
        )
    else:
        w("   query-NER: no entity mentions detected")
    w("   merged seeds (top 15):")
    for e in s.merged[:15]:
        gr = e.granularity or "-"
        preview = (e.text_preview or "").replace("\n", " ")[:80]
        w(f"     {e.score:+.3f}  [{e.source:14s}][{gr:8s}]  {preview!r}")
    w("")

    # [3] Expand ---------------------------------------------------------
    e = trace.expand
    w(f"[3] EXPAND  ({_tl('expand_ms')})")
    w(
        f"   seeds_in={e.seed_count}  final_nodes={e.final_node_count}  "
        f"total_edges_traversed={e.total_edges_traversed}  max_depth={e.max_depth}  "
        f"max_frontier={e.max_frontier}"
    )
    w("   edge_weights:")
    for etype in sorted(e.edge_weights):
        w(f"     {etype:22s}  w={e.edge_weights[etype]:.2f}")
    for step in e.steps:
        edge_summary = ", ".join(
            f"{t}={n}" for t, n in sorted(step.edges_by_type.items(), key=lambda kv: -kv[1])
        ) or "(none)"
        cap_note = f" → capped to {step.frontier_out_size}" if step.was_capped else ""
        w(
            f"   depth {step.depth}: frontier_in={step.frontier_in_size}  "
            f"edges {step.edges_traversed}/{step.edges_considered} traversed  "
            f"out_frontier={step.frontier_out_size_before_cap}{cap_note}  "
            f"new_nodes={step.newly_reached}"
        )
        w(f"     by_type: {edge_summary}")
    w("")

    # [4] Score ----------------------------------------------------------
    sc = trace.score
    w(f"[4] SCORE  ({_tl('score_ms')})")
    w(
        f"   walk_nodes={sc.walk_node_count}  granules_considered={sc.granules_considered}  "
        f"max_passages={sc.max_passages}  selected={len(sc.selected)}"
    )
    for i, g in enumerate(sc.selected[:12]):
        preview = (g.text_preview or "").replace("\n", " ")[:100]
        w(f"   #{i+1:2d} {g.score:+.3f}  [{g.granularity:8s}]  {preview!r}")
    if sc.dropped_non_granules:
        w(f"   top non-granule walk hits (not surfaced as passages):")
        for d in sc.dropped_non_granules[:6]:
            w(f"     {d.score:+.3f}  [{d.label:18s}]  {d.node_id}")
    w("")

    # [5] Assemble -------------------------------------------------------
    a = trace.assemble
    w(f"[5] ASSEMBLE  ({_tl('assemble_ms')})")
    w(
        f"   derived_stale={a.derived_was_stale}  rebuilt={a.derived_rebuilt}  "
        f"passages={a.passages_assembled}  facts={a.facts_assembled}"
    )
    if a.facts_by_type:
        w(
            "   fact types: "
            + ", ".join(f"{k}={v}" for k, v in sorted(a.facts_by_type.items()))
        )
    w("=" * 78)
    return "\n".join(lines)


__all__ = [
    "AssembleStageTrace",
    "DroppedNode",
    "ExpandStageTrace",
    "ExpandStep",
    "IntentStageTrace",
    "RecallTrace",
    "ScoreStageTrace",
    "ScoredGranule",
    "SeedEntry",
    "SeedStageTrace",
    "traced_recall",
]
