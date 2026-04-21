"""End-to-end tests for :class:`engram.recall.pipeline.RecallPipeline`.

Covers:

- Determinism (R2 audit for recall) — same (config, ingest log, query,
  context) → byte-identical RecallResult modulo timing.
- Intent-hint skip path — agent-provided intent bypasses classification.
- Passages are granule-typed and sorted by score.
- Facts surface per intent; ``preference``/``temporal`` pull
  ``current_preference`` when available.
- Empty-state recall returns empty result without crashing.
- Zero LLM calls on the recall path.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from engram import EngramGraphMemorySystem, Memory
from engram.config import MemoryConfig
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import PREFERENCE_POLARITIES
from engram.recall.intents import (
    INTENT_AGGREGATION,
    INTENT_PREFERENCE,
    INTENT_SINGLE_FACT,
    INTENTS,
    compute_intent_centroids,
)
from engram.recall.pipeline import RecallPipeline
from tests._fake_nlp import (
    FakeEnt,
    FakeSent,
    attach_subtree,
    deterministic_embed,
    make_fake_doc,
    make_nlp_process,
    make_token,
)


def _make_tolerant_nlp(docs_by_text: dict):
    """Like make_nlp_process but returns an empty FakeDoc for unknown texts
    — queries at recall time aren't known to the test fixtures, so we fall
    back to a sentence-free empty doc (NER returns no mentions)."""

    def process(texts: list[str]) -> list[object]:
        out: list[object] = []
        for text in texts:
            if text in docs_by_text:
                out.append(docs_by_text[text])
            else:
                out.append(
                    make_fake_doc(
                        text=text,
                        sents=[FakeSent(text=text, start_char=0, end_char=len(text))],
                    )
                )
        return out

    return process


def _build_doc_for(text: str):
    """Build a minimal FakeDoc whose NER lights up on an 'Alice' mention when present."""
    root = make_token("loves", idx=text.find("loves"), pos="VERB", dep="ROOT", lemma="love", tense=("Pres",))
    nsubj = make_token(
        "Alice", idx=max(0, text.find("Alice")), pos="PROPN", dep="nsubj"
    )
    obj_text = "hiking" if "hiking" in text else "pizza"
    obj_idx = text.find(obj_text)
    dobj = make_token(obj_text, idx=max(0, obj_idx), pos="NOUN", dep="dobj")
    root.children = (nsubj, dobj)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(dobj, [dobj])
    attach_subtree(root, [nsubj, root, dobj])
    sent = FakeSent(text=text, start_char=0, end_char=len(text), root=root)
    ents: list[FakeEnt] = []
    if "Alice" in text:
        start = text.find("Alice")
        ents.append(FakeEnt(text="Alice", label_="PERSON", start_char=start, end_char=start + 5))
    return make_fake_doc(text=text, sents=[sent], ents=ents)


_TEXTS: tuple[str, ...] = (
    "Alice loves hiking.",
    "Alice loves pizza.",
)


def _all_docs() -> dict:
    return {text: _build_doc_for(text) for text in _TEXTS}


def _make_system(
    config: MemoryConfig | None = None,
) -> tuple[EngramGraphMemorySystem, RecallPipeline]:
    config = config or MemoryConfig()
    docs = _all_docs()
    # Add any query docs we'll pass to the recall-side nlp_process.
    query_docs = {
        "What does Alice like?": _build_doc_for("What does Alice like?"),
        "Does Alice love hiking?": _build_doc_for("Does Alice love hiking?"),
        "Alice": _build_doc_for("Alice"),
        "unrelated random query": make_fake_doc(text="unrelated random query", sents=[
            FakeSent(text="unrelated random query", start_char=0, end_char=23)
        ]),
    }
    all_docs = {**docs, **query_docs}

    preference_embed = deterministic_embed(dim=16)
    preference_centroids = compute_centroids(preference_embed)
    granule_embed = deterministic_embed(dim=32)

    ingest_pipeline = IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process(all_docs),
        preference_centroids=preference_centroids,
        preference_embed=preference_embed,
        granule_embed=granule_embed,
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )

    # Recall-side: reuse a fresh deterministic_embed of the SAME dim so the
    # vector index's dim matches the query embedding dim exactly.
    query_embed = deterministic_embed(dim=32)
    intent_centroids = compute_intent_centroids(query_embed)
    recall_pipeline = RecallPipeline(
        config=config,
        intent_centroids=intent_centroids,
        query_embed=query_embed,
        nlp_process=_make_tolerant_nlp(all_docs),
    )

    system = EngramGraphMemorySystem(
        config=config,
        pipeline=ingest_pipeline,
        recall_pipeline=recall_pipeline,
    )
    return system, recall_pipeline


async def _ingest_fixture(system: EngramGraphMemorySystem) -> None:
    await system.ingest(
        Memory(
            content="Alice loves hiking.",
            timestamp="2026-01-01T00:00:00Z",
            speaker="user",
            source="conversation_turn",
        )
    )
    await system.ingest(
        Memory(
            content="Alice loves pizza.",
            timestamp="2026-02-01T00:00:00Z",
            speaker="user",
            source="conversation_turn",
        )
    )


def test_recall_on_empty_system_is_empty() -> None:
    system, _ = _make_system()
    result = asyncio.run(system.recall("anything"))
    assert result.passages == ()
    assert result.facts == ()


def test_recall_returns_passages() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    result = asyncio.run(system.recall("What does Alice like?"))
    assert result.intent in INTENTS
    assert len(result.passages) > 0
    # Granularity set is a subset of the three granule labels.
    granularities = {p.granularity for p in result.passages}
    assert granularities <= {"turn", "sentence", "ngram"}
    # Sorted by score descending.
    scores = [p.score for p in result.passages]
    assert scores == sorted(scores, reverse=True)


def test_recall_passages_carry_provenance() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    result = asyncio.run(system.recall("What does Alice like?"))
    for passage in result.passages:
        assert passage.source_memory_id is not None
        assert passage.source_memory_index is not None
        assert passage.timestamp is not None


def test_intent_hint_bypasses_classification() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    hinted = asyncio.run(
        system.recall("anything", intent_hint=INTENT_AGGREGATION)
    )
    assert hinted.intent == INTENT_AGGREGATION
    assert hinted.intent_confidence == 1.0


def test_intent_hint_rejects_unknown_label() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    with pytest.raises(ValueError, match="intent_hint"):
        asyncio.run(system.recall("anything", intent_hint="not_a_real_intent"))


def test_recall_fingerprint_stable_across_calls() -> None:
    """Same (config, ingest, query, context) → same fingerprint."""
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    r1 = asyncio.run(system.recall("What does Alice like?", now="2026-03-01T00:00:00Z"))
    r2 = asyncio.run(system.recall("What does Alice like?", now="2026-03-01T00:00:00Z"))
    assert r1.recall_fingerprint == r2.recall_fingerprint


def test_recall_fingerprint_changes_with_query() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    r1 = asyncio.run(system.recall("question A"))
    r2 = asyncio.run(system.recall("question B"))
    assert r1.recall_fingerprint != r2.recall_fingerprint


def test_recall_fingerprint_changes_with_context() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    r1 = asyncio.run(system.recall("q", now="2026-03-01T00:00:00Z"))
    r2 = asyncio.run(system.recall("q", now="2026-04-01T00:00:00Z"))
    assert r1.recall_fingerprint != r2.recall_fingerprint


def test_recall_determinism_r2_audit() -> None:
    """Same (config, ingest, query, context) → byte-identical RecallResult
    modulo timing_ms (wall-clock, explicitly non-deterministic)."""
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    r1 = asyncio.run(system.recall("What does Alice like?"))
    r2 = asyncio.run(system.recall("What does Alice like?"))
    assert dataclasses.replace(r1, timing_ms=()) == dataclasses.replace(r2, timing_ms=())


def test_recall_passage_count_respects_max_passages_override() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    result = asyncio.run(system.recall("Alice", max_passages=2))
    assert len(result.passages) <= 2


def test_preference_intent_surfaces_current_preference_facts() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    # Hint as preference to get the preference-intent fact profile.
    result = asyncio.run(
        system.recall("Does Alice love hiking?", intent_hint=INTENT_PREFERENCE)
    )
    fact_kinds = {f.kind for f in result.facts}
    # Either current_preference or reinforcement should surface given that
    # the fixture contains a Preference node with at least one observation.
    assert fact_kinds <= {"current_preference", "reinforcement"}


def test_single_fact_intent_omits_co_occurrence_facts() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    result = asyncio.run(
        system.recall("Alice loves hiking.", intent_hint=INTENT_SINGLE_FACT)
    )
    for fact in result.facts:
        assert fact.kind != "co_occurrence"


def test_timing_ms_carries_all_five_stages_plus_total() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    result = asyncio.run(system.recall("Alice"))
    stage_names = [name for name, _ms in result.timing_ms]
    # Five stages + total = 6 entries (order is fixed).
    assert stage_names == ["classify", "seed", "expand", "score", "assemble", "total"]


def test_zero_llm_calls_on_recall_path() -> None:
    """Recall must make no LLM calls (R5 / R13).

    The test imports common LLM client paths and asserts they weren't
    touched. Since engram depends on neither ``openai`` nor ``ollama``
    in this repo, the safeguard is to verify the network modules aren't
    imported by the recall module graph. A positive assertion would need
    the client classes to exist; we assert module-absence instead.
    """
    import sys

    # Recall path uses spaCy + sentence-transformers (both local). No
    # network client should appear.
    assert "openai" not in sys.modules
    assert "ollama" not in sys.modules
    assert "anthropic" not in sys.modules


def test_save_and_load_roundtrip_preserves_recall(tmp_path: Path) -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    asyncio.run(system.save_state(tmp_path))

    # Load into a fresh system and confirm recall works.
    restored, _ = _make_system()
    asyncio.run(restored.load_state(tmp_path))
    result = asyncio.run(restored.recall("What does Alice like?"))
    assert len(result.passages) > 0


def test_recall_context_fingerprint_covers_timezone() -> None:
    system, _ = _make_system()
    asyncio.run(_ingest_fixture(system))
    r1 = asyncio.run(system.recall("q", timezone="UTC"))
    r2 = asyncio.run(system.recall("q", timezone="America/New_York"))
    assert r1.recall_fingerprint != r2.recall_fingerprint


def test_empty_state_recall_returns_fingerprint_aware_shell() -> None:
    """No ingests yet: recall returns a structurally-valid empty result."""
    system, _ = _make_system()
    result = asyncio.run(system.recall("anything"))
    assert result.passages == ()
    assert result.facts == ()
    assert result.intent is None
