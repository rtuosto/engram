"""Unit tests for :mod:`engram.ingestion.vector_index`.

PR-C (patch 4) adds a parallel granule-embedding store. These tests cover
the VectorIndex class directly — add / knn / granularity filter / save-load
roundtrip / schema-version discipline. Pipeline integration (the vector
index being populated during ingest) is covered in
``test_memory_system_integration.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engram.ingestion.vector_index import (
    GRANULARITY_NGRAM,
    GRANULARITY_SENTENCE,
    GRANULARITY_TURN,
    VECTOR_INDEX_SCHEMA_VERSION,
    VectorIndex,
    VectorIndexFormatError,
)


def _unit(seed: int, dim: int = 8) -> np.ndarray:
    """Deterministic unit vector from an integer seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# add() and knn()
# ---------------------------------------------------------------------------


def test_add_appends_rows_in_insertion_order() -> None:
    idx = VectorIndex(dim=4)
    idx.add("alpha", GRANULARITY_TURN, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("beta", GRANULARITY_SENTENCE, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    idx.add("gamma", GRANULARITY_NGRAM, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    assert len(idx) == 3
    assert idx.node_ids() == ("alpha", "beta", "gamma")
    assert idx.granularity_for("beta") == GRANULARITY_SENTENCE


def test_add_normalizes_input_vector() -> None:
    idx = VectorIndex(dim=2)
    idx.add("v", GRANULARITY_TURN, np.array([3.0, 4.0], dtype=np.float32))
    stored = idx.vector_for("v")
    np.testing.assert_allclose(np.linalg.norm(stored), 1.0, atol=1e-6)
    # direction preserved
    np.testing.assert_allclose(stored, np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)


def test_add_duplicate_node_id_rejected() -> None:
    idx = VectorIndex(dim=2)
    idx.add("v", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="already present"):
        idx.add("v", GRANULARITY_SENTENCE, np.array([0.0, 1.0], dtype=np.float32))


def test_add_rejects_wrong_dim() -> None:
    idx = VectorIndex(dim=4)
    with pytest.raises(ValueError, match="dim"):
        idx.add("v", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))


def test_add_rejects_unknown_granularity() -> None:
    idx = VectorIndex(dim=2)
    with pytest.raises(ValueError, match="unknown granularity"):
        idx.add("v", "session", np.array([1.0, 0.0], dtype=np.float32))


def test_add_zero_vector_is_stored_as_is() -> None:
    """Zero vectors never match anything in knn — fail-closed behavior."""
    idx = VectorIndex(dim=2)
    idx.add("zero", GRANULARITY_TURN, np.zeros(2, dtype=np.float32))
    stored = idx.vector_for("zero")
    assert np.linalg.norm(stored) == 0.0
    # knn treats a zero query as empty, but a zero stored vector just scores 0
    # for all queries — never top-k unless every candidate scores 0.
    query = np.array([1.0, 0.0], dtype=np.float32)
    idx.add("real", GRANULARITY_TURN, query)
    hits = idx.knn(query, k=1)
    assert hits == [("real", pytest.approx(1.0, abs=1e-6))]


def test_knn_returns_top_k_sorted_by_score() -> None:
    idx = VectorIndex(dim=4)
    # Craft candidates with known cosine similarity to query [1,0,0,0]:
    # "best" = 1.0, "mid" = 0.8, "low" = 0.1
    idx.add("best", GRANULARITY_TURN, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("mid", GRANULARITY_TURN, np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32))
    idx.add("low", GRANULARITY_TURN, np.array([0.1, 0.0, 0.9, 0.0], dtype=np.float32))

    hits = idx.knn(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert [node_id for node_id, _score in hits] == ["best", "mid"]


def test_knn_ties_break_by_node_id() -> None:
    """Identical scores resolve lexicographically — R2 stability."""
    idx = VectorIndex(dim=2)
    # Two candidates that score identically against the query.
    idx.add("zeta", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    idx.add("alpha", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    idx.add("mu", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))

    hits = idx.knn(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [n for n, _ in hits] == ["alpha", "mu", "zeta"]


def test_knn_granularity_filter_single_string() -> None:
    idx = VectorIndex(dim=2)
    idx.add("turn_a", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    idx.add("sent_a", GRANULARITY_SENTENCE, np.array([1.0, 0.0], dtype=np.float32))
    idx.add("ngram_a", GRANULARITY_NGRAM, np.array([1.0, 0.0], dtype=np.float32))

    query = np.array([1.0, 0.0], dtype=np.float32)
    sent_only = idx.knn(query, k=5, granularity_filter=GRANULARITY_SENTENCE)
    assert [n for n, _ in sent_only] == ["sent_a"]


def test_knn_granularity_filter_frozenset() -> None:
    idx = VectorIndex(dim=2)
    idx.add("turn_a", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    idx.add("sent_a", GRANULARITY_SENTENCE, np.array([1.0, 0.0], dtype=np.float32))
    idx.add("ngram_a", GRANULARITY_NGRAM, np.array([1.0, 0.0], dtype=np.float32))

    query = np.array([1.0, 0.0], dtype=np.float32)
    hits = idx.knn(
        query, k=5, granularity_filter=frozenset({GRANULARITY_SENTENCE, GRANULARITY_NGRAM})
    )
    assert sorted(n for n, _ in hits) == ["ngram_a", "sent_a"]


def test_knn_empty_index_returns_empty_list() -> None:
    idx = VectorIndex(dim=8)
    hits = idx.knn(_unit(0), k=5)
    assert hits == []


def test_knn_k_zero_returns_empty() -> None:
    idx = VectorIndex(dim=2)
    idx.add("v", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    assert idx.knn(np.array([1.0, 0.0], dtype=np.float32), k=0) == []


def test_knn_zero_query_returns_empty() -> None:
    idx = VectorIndex(dim=2)
    idx.add("v", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    assert idx.knn(np.zeros(2, dtype=np.float32), k=5) == []


def test_knn_granularity_filter_unknown_raises() -> None:
    idx = VectorIndex(dim=2)
    idx.add("v", GRANULARITY_TURN, np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="unknown granularity"):
        idx.knn(
            np.array([1.0, 0.0], dtype=np.float32),
            k=1,
            granularity_filter="session",
        )


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


def _build_index() -> VectorIndex:
    idx = VectorIndex(dim=8)
    idx.add("turn_1", GRANULARITY_TURN, _unit(1))
    idx.add("sent_1", GRANULARITY_SENTENCE, _unit(2))
    idx.add("sent_2", GRANULARITY_SENTENCE, _unit(3))
    idx.add("ngram_1", GRANULARITY_NGRAM, _unit(4))
    return idx


def test_save_load_roundtrip(tmp_path: Path) -> None:
    original = _build_index()
    emb_path = tmp_path / "embeddings.npy"
    nid_path = tmp_path / "node_ids.json"
    original.save(emb_path, nid_path)

    restored = VectorIndex.load(emb_path, nid_path)
    assert restored.dim == original.dim
    assert restored.node_ids() == original.node_ids()
    for node_id in original.node_ids():
        np.testing.assert_array_equal(
            restored.vector_for(node_id), original.vector_for(node_id)
        )
        assert restored.granularity_for(node_id) == original.granularity_for(node_id)


def test_save_is_byte_stable(tmp_path: Path) -> None:
    """R2: same index saved twice produces byte-identical sidecars."""
    a = _build_index()
    b = _build_index()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    a.save(dir_a / "embeddings.npy", dir_a / "node_ids.json")
    b.save(dir_b / "embeddings.npy", dir_b / "node_ids.json")

    assert (dir_a / "embeddings.npy").read_bytes() == (dir_b / "embeddings.npy").read_bytes()
    assert (dir_a / "node_ids.json").read_text(encoding="utf-8") == (
        dir_b / "node_ids.json"
    ).read_text(encoding="utf-8")


def test_neighbor_lookups_stable_across_roundtrip(tmp_path: Path) -> None:
    original = _build_index()
    query = _unit(2)  # matches sent_1 exactly
    before = original.knn(query, k=4)

    emb_path = tmp_path / "embeddings.npy"
    nid_path = tmp_path / "node_ids.json"
    original.save(emb_path, nid_path)
    restored = VectorIndex.load(emb_path, nid_path)

    after = restored.knn(query, k=4)
    # Node order must match; scores match to numerical tolerance.
    assert [n for n, _ in before] == [n for n, _ in after]
    for (_, s_before), (_, s_after) in zip(before, after, strict=True):
        assert s_before == pytest.approx(s_after, abs=1e-6)


def test_load_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    original = _build_index()
    emb_path = tmp_path / "embeddings.npy"
    nid_path = tmp_path / "node_ids.json"
    original.save(emb_path, nid_path)

    envelope = json.loads(nid_path.read_text(encoding="utf-8"))
    envelope["schema_version"] = VECTOR_INDEX_SCHEMA_VERSION + 1
    nid_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(VectorIndexFormatError, match="schema_version"):
        VectorIndex.load(emb_path, nid_path)


def test_load_rejects_missing_sidecar(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexFormatError, match="missing"):
        VectorIndex.load(tmp_path / "nope.npy", tmp_path / "nope.json")


def test_load_rejects_length_mismatch(tmp_path: Path) -> None:
    original = _build_index()
    emb_path = tmp_path / "embeddings.npy"
    nid_path = tmp_path / "node_ids.json"
    original.save(emb_path, nid_path)

    envelope = json.loads(nid_path.read_text(encoding="utf-8"))
    envelope["node_ids"] = envelope["node_ids"][:-1]  # drop one
    nid_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(VectorIndexFormatError):
        VectorIndex.load(emb_path, nid_path)


# ---------------------------------------------------------------------------
# Pipeline integration smoke — populated during ingest, survives save/load
# ---------------------------------------------------------------------------


def test_pipeline_populates_vector_index() -> None:
    """After one ingest, the InstanceState carries a VectorIndex whose node_ids
    cover every emitted granule (Turn + Sentences + N-grams)."""
    import asyncio

    from engram import EngramGraphMemorySystem, Memory
    from engram.config import MemoryConfig
    from engram.ingestion.pipeline import IngestionPipeline
    from engram.ingestion.preferences import compute_centroids
    from engram.ingestion.schema import (
        LABEL_NGRAM,
        LABEL_TURN,
        LABEL_UTTERANCE_SEGMENT,
        PREFERENCE_POLARITIES,
    )
    from tests._fake_nlp import (
        FakeEnt,
        FakeNounChunk,
        FakeSent,
        attach_subtree,
        deterministic_embed,
        make_fake_doc,
        make_nlp_process,
        make_token,
    )

    text = "Alice loves hiking."
    root = make_token("loves", idx=6, pos="VERB", dep="ROOT", lemma="love", tense=("Pres",))
    nsubj = make_token("Alice", idx=0, pos="PROPN", dep="nsubj")
    dobj = make_token("hiking", idx=12, pos="NOUN", dep="dobj")
    root.children = (nsubj, dobj)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(dobj, [dobj])
    attach_subtree(root, [nsubj, root, dobj])
    sent = FakeSent(text=text, start_char=0, end_char=len(text), root=root)
    chunk = FakeNounChunk(
        text="hiking trip",
        start_char=12,
        end_char=19,
        tokens=(
            make_token("hiking", idx=12, pos="NOUN"),
            make_token("trip", idx=15, pos="NOUN"),
        ),
    )
    doc = make_fake_doc(
        text=text,
        sents=[sent],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=0, end_char=5)],
        noun_chunks=[chunk],
    )
    memory = Memory(content=text, timestamp="2026-01-01T00:00:00Z", speaker="user")

    config = MemoryConfig()
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    pipeline = IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process({text: doc}),
        preference_centroids=centroids,
        preference_embed=embed,
        granule_embed=deterministic_embed(dim=16),
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )
    system = EngramGraphMemorySystem(config=config, pipeline=pipeline)
    asyncio.run(system.ingest(memory))

    state = system.get_state()
    assert state is not None
    assert state.vector_index is not None

    turn_ids = set(state.store.nodes_by_label(LABEL_TURN))
    segment_ids = set(state.store.nodes_by_label(LABEL_UTTERANCE_SEGMENT))
    ngram_ids = set(state.store.nodes_by_label(LABEL_NGRAM))

    indexed = set(state.vector_index.node_ids())
    assert turn_ids.issubset(indexed), f"missing turns: {turn_ids - indexed}"
    assert segment_ids.issubset(indexed), f"missing segments: {segment_ids - indexed}"
    assert ngram_ids.issubset(indexed), f"missing n-grams: {ngram_ids - indexed}"
    # No non-granule nodes are in the index.
    assert indexed == turn_ids | segment_ids | ngram_ids


def test_memory_system_save_state_writes_embeddings_sidecar(tmp_path: Path) -> None:
    """PR-C save layout: embeddings.npy + node_ids.json alongside primary.msgpack."""
    import asyncio

    from engram import EngramGraphMemorySystem, Memory
    from engram.config import MemoryConfig
    from engram.engram_memory_system import (
        EMBEDDINGS_FILENAME,
        MANIFEST_FILENAME,
        NODE_IDS_FILENAME,
        PRIMARY_FILENAME,
    )
    from engram.ingestion.pipeline import IngestionPipeline
    from engram.ingestion.preferences import compute_centroids
    from engram.ingestion.schema import PREFERENCE_POLARITIES
    from tests._fake_nlp import (
        FakeEnt,
        FakeSent,
        attach_subtree,
        deterministic_embed,
        make_fake_doc,
        make_nlp_process,
        make_token,
    )

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
    memory = Memory(content=text, timestamp="2026-01-01T00:00:00Z", speaker="user")

    config = MemoryConfig()
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    pipeline = IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process({text: doc}),
        preference_centroids=centroids,
        preference_embed=embed,
        granule_embed=deterministic_embed(dim=16),
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )
    system = EngramGraphMemorySystem(config=config, pipeline=pipeline)
    asyncio.run(system.ingest(memory))
    asyncio.run(system.save_state(tmp_path))

    for name in (MANIFEST_FILENAME, PRIMARY_FILENAME, EMBEDDINGS_FILENAME, NODE_IDS_FILENAME):
        assert (tmp_path / name).exists(), f"missing {name}"

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["has_embeddings"] is True
    from engram.ingestion.persist import SCHEMA_VERSION as _SCHEMA_VERSION

    assert manifest["schema_version"] == _SCHEMA_VERSION

    # Roundtrip the whole system and confirm the vector index restored.
    restored = EngramGraphMemorySystem(config=config, pipeline=pipeline)
    asyncio.run(restored.load_state(tmp_path))
    state_a = system.get_state()
    state_b = restored.get_state()
    assert state_a is not None and state_b is not None
    assert state_b.vector_index is not None
    assert state_b.vector_index.node_ids() == state_a.vector_index.node_ids()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# R2 audit extension — sidecar bytes match across two independent ingests
# ---------------------------------------------------------------------------


def test_ingest_produces_byte_identical_sidecars(tmp_path: Path) -> None:
    """Two independent pipelines ingesting the same Memories produce
    byte-identical embeddings.npy + node_ids.json."""
    import asyncio

    from engram import EngramGraphMemorySystem, Memory
    from engram.config import MemoryConfig
    from engram.engram_memory_system import EMBEDDINGS_FILENAME, NODE_IDS_FILENAME
    from engram.ingestion.pipeline import IngestionPipeline
    from engram.ingestion.preferences import compute_centroids
    from engram.ingestion.schema import PREFERENCE_POLARITIES
    from tests._fake_nlp import (
        FakeEnt,
        FakeSent,
        attach_subtree,
        deterministic_embed,
        make_fake_doc,
        make_nlp_process,
        make_token,
    )

    def build_doc_pair():
        text = "Alice loves hiking."
        root = make_token(
            "loves", idx=6, pos="VERB", dep="ROOT", lemma="love", tense=("Pres",)
        )
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
        return text, doc

    def run_and_save(save_dir: Path) -> None:
        config = MemoryConfig()
        text, doc = build_doc_pair()
        embed = deterministic_embed(dim=16)
        centroids = compute_centroids(embed)
        pipeline = IngestionPipeline(
            config=config,
            nlp_process=make_nlp_process({text: doc}),
            preference_centroids=centroids,
            preference_embed=embed,
            granule_embed=deterministic_embed(dim=16),
            enabled_polarities=frozenset(PREFERENCE_POLARITIES),
        )
        system = EngramGraphMemorySystem(config=config, pipeline=pipeline)
        asyncio.run(system.ingest(
            Memory(content=text, timestamp="2026-01-01T00:00:00Z", speaker="user")
        ))
        asyncio.run(system.save_state(save_dir))

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    run_and_save(dir_a)
    run_and_save(dir_b)

    assert (dir_a / EMBEDDINGS_FILENAME).read_bytes() == (
        dir_b / EMBEDDINGS_FILENAME
    ).read_bytes()
    assert (dir_a / NODE_IDS_FILENAME).read_text(encoding="utf-8") == (
        dir_b / NODE_IDS_FILENAME
    ).read_text(encoding="utf-8")
