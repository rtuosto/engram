"""Tests for the :mod:`engram.diagnostics.recall_trace` play-by-play.

The trace must fire at every one of the five stages and the captured
events must faithfully mirror the computation the production
:meth:`RecallPipeline.recall` would have done. We reuse the same
synthetic fixture the recall-pipeline tests build.
"""

from __future__ import annotations

import asyncio

from engram import EngramGraphMemorySystem
from engram.diagnostics import RecallTrace, traced_recall
from engram.recall.context import RecallContext
from engram.recall.intents import INTENTS
from tests.test_recall_pipeline import _ingest_fixture, _make_system


def _ingested_system() -> tuple[EngramGraphMemorySystem, object]:
    system, pipeline = _make_system()
    asyncio.run(_ingest_fixture(system))
    return system, pipeline


def test_trace_fires_at_every_stage() -> None:
    """Every one of the five stages records its dataclass."""
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None

    result, trace = traced_recall(
        pipeline,
        state,
        "What does Alice like?",
        RecallContext(),
    )

    assert isinstance(trace, RecallTrace)
    # Stages are all present (non-null).
    assert trace.intent is not None
    assert trace.seed is not None
    assert trace.expand is not None
    assert trace.score is not None
    assert trace.assemble is not None

    # Timings are populated.
    for key in ("intent_ms", "seed_ms", "expand_ms", "score_ms", "assemble_ms", "total_ms"):
        assert key in trace.timing_ms


def test_trace_intent_stage_reports_all_intent_cosines() -> None:
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    i = trace.intent
    if not i.used_hint:
        # All intents appear in scores_by_intent; chosen is the top.
        assert set(i.scores_by_intent) == set(INTENTS)
        ordered = sorted(i.scores_by_intent.items(), key=lambda kv: -kv[1])
        # If we didn't fall back, chosen equals the argmax.
        if not i.fell_back:
            assert i.chosen == ordered[0][0]


def test_trace_intent_hint_records_bypass() -> None:
    """Agent hint → used_hint=True, scores_by_intent empty, confidence 1.0."""
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline,
        state,
        "What does Alice like?",
        RecallContext(intent_hint="preference"),
    )
    assert trace.intent.used_hint is True
    assert trace.intent.intent_hint == "preference"
    assert trace.intent.chosen == "preference"
    assert trace.intent.scores_by_intent == {}
    assert trace.intent.margin == 1.0


def test_trace_seed_stage_reports_semantic_and_entity() -> None:
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    s = trace.seed
    # Query contains 'Alice' — the fixture's fake NER lights up on it.
    assert "Alice" in s.query_entity_surfaces
    assert s.semantic_seed_count > 0
    assert s.merged_seed_count > 0
    assert s.merged_seed_count <= s.total_cap
    # Merged entries have source labels we understand.
    for entry in s.merged:
        assert entry.source in ("semantic", "entity", "entity_granule", "both")


def test_trace_expand_stage_captures_per_step_detail() -> None:
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    e = trace.expand
    # Walk fires at least once on a non-empty graph.
    assert e.seed_count > 0
    assert e.final_node_count >= e.seed_count  # seeds themselves remain scored
    assert len(e.steps) >= 1
    # Edge accounting is self-consistent.
    for step in e.steps:
        assert step.edges_traversed <= step.edges_considered
        assert sum(step.edges_by_type.values()) == step.edges_traversed
        assert step.frontier_out_size <= e.max_frontier
        if step.was_capped:
            assert step.frontier_out_size == e.max_frontier


def test_trace_score_stage_matches_recall_result() -> None:
    """Selected granules in the score trace line up with the RecallResult."""
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    result, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    assert len(trace.score.selected) == len(result.passages)
    # Granularities line up (same ordering).
    for scored, passage in zip(trace.score.selected, result.passages, strict=True):
        assert scored.granule_id == passage.node_id
        assert scored.score == passage.score


def test_trace_assemble_stage_counts_facts_by_type() -> None:
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    result, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    a = trace.assemble
    assert a.passages_assembled == len(result.passages)
    assert a.facts_assembled == len(result.facts)
    if result.facts:
        assert sum(a.facts_by_type.values()) == len(result.facts)


def test_trace_empty_state_short_circuits_cleanly() -> None:
    """No ingests → empty result + trace with zero-sized stages."""
    system, pipeline = _make_system()
    # Force create_state so we have an empty but real state.
    from engram.ingestion.pipeline import IngestionPipeline

    ingestion = system._get_pipeline()  # noqa: SLF001
    assert isinstance(ingestion, IngestionPipeline)
    system._state = ingestion.create_state()  # noqa: SLF001
    state = system.get_state()
    assert state is not None

    result, trace = traced_recall(
        pipeline, state, "any query", RecallContext()
    )
    assert result.passages == ()
    assert trace.seed.merged_seed_count == 0
    assert trace.expand.final_node_count == 0
    assert trace.score.selected == ()
    assert trace.assemble.passages_assembled == 0


def test_trace_to_json_is_serializable() -> None:
    """Dataclass → dict → json round-trip must not raise."""
    import json

    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    blob = trace.to_json()
    restored = json.loads(blob)
    assert "intent" in restored
    assert "seed" in restored
    assert "expand" in restored


def test_trace_pretty_output_runs() -> None:
    """pretty() must not raise and must include stage markers."""
    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )
    text = trace.pretty()
    for marker in ("[1] INTENT", "[2] SEED", "[3] EXPAND", "[4] SCORE", "[5] ASSEMBLE"):
        assert marker in text


def test_trace_render_html_self_contained() -> None:
    """HTML dashboard renders as a complete, self-contained document.

    The artifact is supposed to be shareable — no external fetches, JSON
    embedded inline, and all five stage containers present so the
    rendered dashboard has everything it needs to paint without a server.
    """
    from engram.diagnostics import render_trace_html

    system, pipeline = _ingested_system()
    state = system.get_state()
    assert state is not None
    _, trace = traced_recall(
        pipeline, state, "What does Alice like?", RecallContext()
    )

    doc = render_trace_html(trace)

    # Skeleton sanity
    assert doc.startswith("<!DOCTYPE html>")
    assert "</html>" in doc

    # No external dependencies — everything inline
    lowered = doc.lower()
    assert "<script src=" not in lowered
    assert "<link rel=\"stylesheet\"" not in lowered

    # Trace JSON embedded
    assert 'id="trace-data"' in doc
    assert trace.intent.chosen in doc or trace.intent.fallback_intent in doc

    # All five stage containers rendered (tab buttons present)
    for stage in ("intent", "seed", "expand", "score", "assemble"):
        assert f'data-stage="{stage}"' in doc
        assert f'id="stage-{stage}"' in doc

    # BFS step controls present
    assert 'id="bfs-range"' in doc
    assert 'id="bfs-prev"' in doc
    assert 'id="bfs-next"' in doc

    # Query is included in the header (HTML-escaped)
    assert "What does Alice like?" in doc
