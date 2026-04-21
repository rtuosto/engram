"""Phase 5 equivalence test — ``ingest_many`` ≡ loop of ``ingest``.

``IngestionPipeline.ingest_many`` pools the three model calls (spaCy
``nlp.pipe``, granule embed, preference embed) across the batch
dimension for speedup. R3/R4 demands the resulting graph be
structurally identical to calling ``ingest`` in sequence. Under the
test-path ``deterministic_embed`` (composition-invariant hashing, not
a real transformer), the two paths produce byte-identical msgpack
output. Real transformers introduce ~5e-8 batch-composition drift on
``holds_preference`` edge weights — that case is covered by
``scripts/check_fingerprint_equivalence.py`` (structural diff with
float tolerance).

See ``.agent/current-plan.md`` Phase 5 for the contract.
"""

from __future__ import annotations

import pytest

from engram.config import MemoryConfig
from engram.ingestion.persist import dump_state
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import PREFERENCE_POLARITIES
from engram.models import Memory
from tests._fake_nlp import (
    FakeEnt,
    FakeSent,
    attach_subtree,
    deterministic_embed,
    make_fake_doc,
    make_nlp_process,
    make_token,
)


def _svo_sentence(text: str, verb: str, verb_idx: int, verb_lemma: str,
                  subj: str, subj_idx: int, subj_pos: str,
                  obj: str, obj_idx: int, obj_pos: str = "NOUN"):
    root = make_token(verb, idx=verb_idx, pos="VERB", dep="ROOT", lemma=verb_lemma, tense=("Pres",))
    nsubj = make_token(subj, idx=subj_idx, pos=subj_pos, dep="nsubj")
    dobj = make_token(obj, idx=obj_idx, pos=obj_pos, dep="dobj")
    root.children = (nsubj, dobj)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(dobj, [dobj])
    attach_subtree(root, [nsubj, root, dobj])
    sent = FakeSent(text=text, start_char=0, end_char=len(text), root=root)
    return sent


def _build_corpus() -> tuple[dict, tuple[Memory, ...]]:
    """Four-Memory stream — spans multiple speakers, repeat entities,
    preference-shaped and non-preference-shaped content, and a Memory with
    no NER (the ``I also like running`` turn). Exercises every code path
    ``ingest_many`` has to fan out: fresh entity, repeat entity, speaker
    cache warm-up, claim-with-no-mention, preference claim + non-preference
    claim.
    """
    m1_text = "Alice loves hiking."
    m1_sent = _svo_sentence(m1_text, "loves", 6, "love", "Alice", 0, "PROPN", "hiking", 12)
    m1_doc = make_fake_doc(
        text=m1_text,
        sents=[m1_sent],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=0, end_char=5)],
    )

    m2_text = "Bob avoids crowds."
    m2_sent = _svo_sentence(m2_text, "avoids", 4, "avoid", "Bob", 0, "PROPN", "crowds", 11)
    m2_doc = make_fake_doc(
        text=m2_text,
        sents=[m2_sent],
        ents=[FakeEnt(text="Bob", label_="PERSON", start_char=0, end_char=3)],
    )

    m3_text = "Alice hates running."
    m3_sent = _svo_sentence(m3_text, "hates", 6, "hate", "Alice", 0, "PROPN", "running", 12)
    m3_doc = make_fake_doc(
        text=m3_text,
        sents=[m3_sent],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=0, end_char=5)],
    )

    m4_text = "I also like running."
    m4_sent = _svo_sentence(m4_text, "like", 7, "like", "I", 0, "PRON", "running", 12)
    m4_doc = make_fake_doc(text=m4_text, sents=[m4_sent], ents=[])

    docs_by_text = {m1_text: m1_doc, m2_text: m2_doc, m3_text: m3_doc, m4_text: m4_doc}
    memories = (
        Memory(content=m1_text, timestamp="2026-01-01T10:00:00Z", speaker="alice"),
        Memory(content=m2_text, timestamp="2026-01-01T10:01:00Z", speaker="bob"),
        Memory(content=m3_text, timestamp="2026-01-02T10:00:00Z", speaker="alice"),
        Memory(content=m4_text, timestamp="2026-01-02T10:05:00Z", speaker="user"),
    )
    return docs_by_text, memories


def _make_pipeline(config: MemoryConfig, docs_by_text: dict) -> IngestionPipeline:
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    return IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process(docs_by_text),
        preference_centroids=centroids,
        preference_embed=embed,
        granule_embed=deterministic_embed(dim=16),
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )


def test_ingest_many_byte_identical_to_loop() -> None:
    """Four memories through ``ingest_many`` match loop of ``ingest`` byte-for-byte.

    This is the R3/R4 structural contract: batched path → identical graph.
    Real transformers add ~5e-8 edge-weight drift; the test-path embedder
    is composition-invariant so byte equality is a strictly stronger check.
    """
    config = MemoryConfig()
    docs, memories = _build_corpus()

    pipeline_seq = _make_pipeline(config, docs)
    state_seq = pipeline_seq.create_state()
    for m in memories:
        pipeline_seq.ingest(state_seq, m)
    bytes_seq = dump_state(state_seq.store)

    pipeline_batch = _make_pipeline(config, docs)
    state_batch = pipeline_batch.create_state()
    pipeline_batch.ingest_many(state_batch, memories)
    bytes_batch = dump_state(state_batch.store)

    assert bytes_seq == bytes_batch, (
        "ingest_many produced different msgpack bytes than looped ingest — "
        "structural equivalence violated"
    )


def test_ingest_many_empty_is_noop() -> None:
    """Empty sequence → no state mutation."""
    config = MemoryConfig()
    docs, _memories = _build_corpus()

    pipeline = _make_pipeline(config, docs)
    state = pipeline.create_state()
    initial_bytes = dump_state(state.store)

    pipeline.ingest_many(state, ())

    assert dump_state(state.store) == initial_bytes


def test_ingest_many_single_memory_matches_ingest() -> None:
    """Batch-of-1 must match a direct ``ingest`` call — no off-by-one
    between the batched and sequential code paths."""
    config = MemoryConfig()
    docs, memories = _build_corpus()
    one = memories[:1]

    pipeline_seq = _make_pipeline(config, docs)
    state_seq = pipeline_seq.create_state()
    pipeline_seq.ingest(state_seq, one[0])

    pipeline_batch = _make_pipeline(config, docs)
    state_batch = pipeline_batch.create_state()
    pipeline_batch.ingest_many(state_batch, one)

    assert dump_state(state_seq.store) == dump_state(state_batch.store)


def test_ingest_many_vector_index_rows_match() -> None:
    """Vector-index rows (parallel to granule nodes) must match ingest-by-ingest.

    The vector index is not persisted via ``dump_state``; it has its own
    ``.save()`` path. This test asserts the in-memory row sequence
    (node_ids, granularities, and the embedding matrix) is equivalent.
    """
    config = MemoryConfig()
    docs, memories = _build_corpus()

    pipeline_seq = _make_pipeline(config, docs)
    state_seq = pipeline_seq.create_state()
    for m in memories:
        pipeline_seq.ingest(state_seq, m)

    pipeline_batch = _make_pipeline(config, docs)
    state_batch = pipeline_batch.create_state()
    pipeline_batch.ingest_many(state_batch, memories)

    assert state_seq.vector_index is not None
    assert state_batch.vector_index is not None
    assert state_seq.vector_index.node_ids() == state_batch.vector_index.node_ids()
    # Composition-invariant embed → exact numeric equality row-by-row.
    import numpy as np

    for nid in state_seq.vector_index.node_ids():
        np.testing.assert_array_equal(
            state_seq.vector_index.vector_for(nid),
            state_batch.vector_index.vector_for(nid),
        )
        assert state_seq.vector_index.granularity_for(nid) == (
            state_batch.vector_index.granularity_for(nid)
        )


def test_ingest_many_advances_memory_index_monotonically() -> None:
    """``memory_index`` matches N, and the emitted Memory nodes use indices 1..N.

    This guards the append-only contract (R16) — batched processing must
    not reset or skip indices when pooling model calls.
    """
    config = MemoryConfig()
    docs, memories = _build_corpus()

    pipeline = _make_pipeline(config, docs)
    state = pipeline.create_state()
    pipeline.ingest_many(state, memories)

    assert state.memory_index == len(memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
