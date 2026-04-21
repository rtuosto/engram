"""End-to-end test of :class:`EngramGraphMemorySystem` with mocked models.

Exercises the full verb surface: ingest → finalize → save → load → inspect.
Uses fakes from :mod:`tests._fake_nlp` so the test is fast and hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from engram import EngramGraphMemorySystem, MemorySystem
from engram.config import MemoryConfig
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import (
    EDGE_ASSERTS,
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_SESSION,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    PREFERENCE_POLARITIES,
)
from engram.models import Session, Turn
from tests._fake_nlp import (
    FakeEnt,
    FakeSent,
    attach_subtree,
    deterministic_embed,
    make_fake_doc,
    make_nlp_process,
    make_token,
)


def _build_session() -> tuple[Session, dict]:
    text = "Alice loves hiking."
    root = make_token("loves", idx=6, pos="VERB", dep="ROOT", lemma="love", tense=("Pres",))
    nsubj = make_token("Alice", idx=0, pos="PROPN", dep="nsubj")
    dobj = make_token("hiking", idx=12, pos="NOUN", dep="dobj")
    root.children = (nsubj, dobj)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(dobj, [dobj])
    attach_subtree(root, [nsubj, root, dobj])
    sent = FakeSent(text=text, start_char=0, end_char=len(text), root=root)
    doc = make_fake_doc(
        text=text,
        sents=[sent],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=0, end_char=5)],
    )
    session = Session(
        session_index=1,
        turns=(
            Turn(
                speaker="user",
                text=text,
                session_index=1,
                turn_index=1,
                timestamp="2026-01-01T00:00:00Z",
            ),
        ),
        timestamp="2026-01-01T00:00:00Z",
    )
    return session, {text: doc}


def _make_system(config: MemoryConfig | None = None) -> EngramGraphMemorySystem:
    config = config or MemoryConfig()
    _session, docs = _build_session()
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    pipeline = IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process(docs),
        preference_centroids=centroids,
        preference_embed=embed,
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )
    return EngramGraphMemorySystem(config=config, pipeline=pipeline)


def test_implements_memory_system_protocol() -> None:
    system = _make_system()
    assert isinstance(system, MemorySystem)


def test_ingest_and_finalize_populates_expected_labels() -> None:
    system = _make_system()
    session, _ = _build_session()

    async def run() -> None:
        await system.ingest_session(session, "c1")
        await system.finalize_conversation("c1")

    asyncio.run(run())
    state = system.get_state("c1")
    assert state is not None
    store = state.store
    assert store.frozen is True
    assert len(store.nodes_by_label(LABEL_SESSION)) == 1
    assert len(store.nodes_by_label(LABEL_TURN)) == 1
    assert len(store.nodes_by_label(LABEL_UTTERANCE_SEGMENT)) == 1
    assert len(store.nodes_by_label(LABEL_ENTITY)) >= 1
    assert len(store.nodes_by_label(LABEL_CLAIM)) == 1


def test_save_state_and_load_state_roundtrip(tmp_path: Path) -> None:
    system = _make_system()
    session, _ = _build_session()

    async def ingest() -> None:
        await system.ingest_session(session, "c1")
        await system.finalize_conversation("c1")
        await system.save_state(tmp_path)

    asyncio.run(ingest())

    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    assert (tmp_path / "c1.msgpack").exists()

    restored = _make_system()
    asyncio.run(restored.load_state(tmp_path))
    state_a = system.get_state("c1")
    state_b = restored.get_state("c1")
    assert state_a is not None and state_b is not None
    assert state_a.store.num_nodes() == state_b.store.num_nodes()
    assert state_a.store.num_edges() == state_b.store.num_edges()


def test_answer_question_raises_until_recall_ships() -> None:
    system = _make_system()
    with pytest.raises(NotImplementedError):
        asyncio.run(system.answer_question("any question", "c1"))


def test_ingest_rejects_frozen_conversation() -> None:
    system = _make_system()
    session, _ = _build_session()

    async def run() -> None:
        await system.ingest_session(session, "c1")
        await system.finalize_conversation("c1")
        # Second ingest on frozen conversation must fail.
        await system.ingest_session(session, "c1")

    from engram.ingestion.graph import GraphFrozenError

    with pytest.raises(GraphFrozenError):
        asyncio.run(run())


def test_reset_clears_conversations() -> None:
    system = _make_system()
    session, _ = _build_session()

    async def run() -> None:
        await system.ingest_session(session, "c1")
        await system.reset()

    asyncio.run(run())
    assert system.get_state("c1") is None


def test_basic_edge_types_present() -> None:
    system = _make_system()
    session, _ = _build_session()

    async def run() -> None:
        await system.ingest_session(session, "c1")
        await system.finalize_conversation("c1")

    asyncio.run(run())
    state = system.get_state("c1")
    assert state is not None
    edges_by_type: dict[str, int] = {}
    for _src, _dst, edge_type, _attrs in state.store.iter_edges():
        edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1
    for required in (EDGE_PART_OF, EDGE_MENTIONS, EDGE_ASSERTS):
        assert required in edges_by_type, f"missing edge type {required}: {edges_by_type}"
