"""End-to-end test of :class:`EngramGraphMemorySystem` with mocked models.

Exercises the post-pivot verb surface: ingest(Memory) → save → load →
inspect. Uses fakes from :mod:`tests._fake_nlp` so the test is fast and
hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from engram import EngramGraphMemorySystem, Memory, MemorySystem
from engram.config import MemoryConfig
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import (
    EDGE_ASSERTS,
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    PREFERENCE_POLARITIES,
)
from tests._fake_nlp import (
    FakeEnt,
    FakeSent,
    attach_subtree,
    deterministic_embed,
    make_fake_doc,
    make_nlp_process,
    make_token,
)


def _build_memory() -> tuple[Memory, dict]:
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
    memory = Memory(
        content=text,
        timestamp="2026-01-01T00:00:00Z",
        speaker="user",
        source="conversation_turn",
    )
    return memory, {text: doc}


def _make_system(config: MemoryConfig | None = None) -> EngramGraphMemorySystem:
    config = config or MemoryConfig()
    _memory, docs = _build_memory()
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    pipeline = IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process(docs),
        preference_centroids=centroids,
        preference_embed=embed,
        granule_embed=deterministic_embed(dim=16),
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )
    return EngramGraphMemorySystem(config=config, pipeline=pipeline)


def test_implements_memory_system_protocol() -> None:
    system = _make_system()
    assert isinstance(system, MemorySystem)


def test_ingest_populates_expected_labels() -> None:
    system = _make_system()
    memory, _ = _build_memory()

    async def run() -> None:
        await system.ingest(memory)

    asyncio.run(run())
    state = system.get_state()
    assert state is not None
    store = state.store
    assert len(store.nodes_by_label(LABEL_MEMORY)) == 1
    assert len(store.nodes_by_label(LABEL_TURN)) == 1
    assert len(store.nodes_by_label(LABEL_UTTERANCE_SEGMENT)) == 1
    assert len(store.nodes_by_label(LABEL_ENTITY)) >= 1
    assert len(store.nodes_by_label(LABEL_CLAIM)) == 1


def test_save_state_and_load_state_roundtrip(tmp_path: Path) -> None:
    system = _make_system()
    memory, _ = _build_memory()

    async def ingest() -> None:
        await system.ingest(memory)
        await system.save_state(tmp_path)

    asyncio.run(ingest())

    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    assert (tmp_path / "primary.msgpack").exists()

    restored = _make_system()
    asyncio.run(restored.load_state(tmp_path))
    state_a = system.get_state()
    state_b = restored.get_state()
    assert state_a is not None and state_b is not None
    assert state_a.store.num_nodes() == state_b.store.num_nodes()
    assert state_a.store.num_edges() == state_b.store.num_edges()


def test_recall_raises_until_pr_e_ships() -> None:
    system = _make_system()
    with pytest.raises(NotImplementedError):
        asyncio.run(system.recall("any question"))


def test_repeat_ingest_creates_new_memory_node() -> None:
    """R16: each ingest is an observation event — no dedup, even on identical content."""
    system = _make_system()
    memory, _ = _build_memory()

    async def run() -> None:
        await system.ingest(memory)
        await system.ingest(memory)

    asyncio.run(run())
    state = system.get_state()
    assert state is not None
    # Two ingest calls → two Memory nodes even with identical content.
    assert len(state.store.nodes_by_label(LABEL_MEMORY)) == 2
    # Turn granule is Memory-scoped (identity = memory_id), so two too.
    assert len(state.store.nodes_by_label(LABEL_TURN)) == 2
    # But the Entity ("Alice") is content-addressed — one node, two inbound edges.
    alice_ids = state.store.nodes_by_label(LABEL_ENTITY)
    assert len(alice_ids) >= 1


def test_reset_clears_state() -> None:
    system = _make_system()
    memory, _ = _build_memory()

    async def run() -> None:
        await system.ingest(memory)
        await system.reset()

    asyncio.run(run())
    assert system.get_state() is None


def test_basic_edge_types_present() -> None:
    system = _make_system()
    memory, _ = _build_memory()

    async def run() -> None:
        await system.ingest(memory)

    asyncio.run(run())
    state = system.get_state()
    assert state is not None
    edges_by_type: dict[str, int] = {}
    for _src, _dst, edge_type, _attrs in state.store.iter_edges():
        edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1
    for required in (EDGE_PART_OF, EDGE_MENTIONS, EDGE_ASSERTS):
        assert required in edges_by_type, f"missing edge type {required}: {edges_by_type}"
