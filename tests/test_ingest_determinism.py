"""R2 audit — the keystone determinism test.

Every ingestion guarantee collapses if this one fails. See
``docs/design/ingestion.md §2`` and §13: "write this first."

Two ingests of the same synthetic Memory sequence, in the same process,
with the same config must produce byte-identical msgpack state. Violations
indicate unstable iteration (set/dict order leaking through), wall-clock
values, unseeded randomness, or aggregation-order-dependent floats.
"""

from __future__ import annotations

import pytest

from engram.config import MemoryConfig
from engram.ingestion.persist import dump_conversation
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


def _build_corpus():
    """Three-Memory synthetic stream with NER, subjects, and candidate preferences."""
    t1_text = "Alice loves hiking."
    t1_root = make_token(
        "loves", idx=6, pos="VERB", dep="ROOT", lemma="love", tense=("Pres",)
    )
    t1_nsubj = make_token("Alice", idx=0, pos="PROPN", dep="nsubj")
    t1_dobj = make_token("hiking", idx=12, pos="NOUN", dep="dobj")
    t1_root.children = (t1_nsubj, t1_dobj)
    attach_subtree(t1_nsubj, [t1_nsubj])
    attach_subtree(t1_dobj, [t1_dobj])
    attach_subtree(t1_root, [t1_nsubj, t1_root, t1_dobj])
    t1_sent = FakeSent(text=t1_text, start_char=0, end_char=len(t1_text), root=t1_root)
    t1_doc = make_fake_doc(
        text=t1_text,
        sents=[t1_sent],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=0, end_char=5)],
    )

    t2_text = "Bob avoids crowds."
    t2_root = make_token(
        "avoids", idx=4, pos="VERB", dep="ROOT", lemma="avoid", tense=("Pres",)
    )
    t2_nsubj = make_token("Bob", idx=0, pos="PROPN", dep="nsubj")
    t2_dobj = make_token("crowds", idx=11, pos="NOUN", dep="dobj")
    t2_root.children = (t2_nsubj, t2_dobj)
    attach_subtree(t2_nsubj, [t2_nsubj])
    attach_subtree(t2_dobj, [t2_dobj])
    attach_subtree(t2_root, [t2_nsubj, t2_root, t2_dobj])
    t2_sent = FakeSent(text=t2_text, start_char=0, end_char=len(t2_text), root=t2_root)
    t2_doc = make_fake_doc(
        text=t2_text,
        sents=[t2_sent],
        ents=[FakeEnt(text="Bob", label_="PERSON", start_char=0, end_char=3)],
    )

    t3_text = "I also like running."
    t3_root = make_token(
        "like", idx=7, pos="VERB", dep="ROOT", lemma="like", tense=("Pres",)
    )
    t3_nsubj = make_token("I", idx=0, pos="PRON", dep="nsubj")
    t3_dobj = make_token("running", idx=12, pos="NOUN", dep="dobj")
    t3_root.children = (t3_nsubj, t3_dobj)
    attach_subtree(t3_nsubj, [t3_nsubj])
    attach_subtree(t3_dobj, [t3_dobj])
    attach_subtree(t3_root, [t3_nsubj, t3_root, t3_dobj])
    t3_sent = FakeSent(text=t3_text, start_char=0, end_char=len(t3_text), root=t3_root)
    t3_doc = make_fake_doc(text=t3_text, sents=[t3_sent], ents=[])

    docs_by_text = {t1_text: t1_doc, t2_text: t2_doc, t3_text: t3_doc}

    memories = (
        Memory(content=t1_text, timestamp="2026-01-01T10:00:00Z", speaker="user"),
        Memory(content=t2_text, timestamp="2026-01-01T10:01:00Z", speaker="user"),
        Memory(content=t3_text, timestamp="2026-01-02T10:00:00Z", speaker="user"),
    )

    return docs_by_text, memories


def _make_pipeline(config: MemoryConfig, docs_by_text: dict):
    embed = deterministic_embed(dim=16)
    centroids = compute_centroids(embed)
    return IngestionPipeline(
        config=config,
        nlp_process=make_nlp_process(docs_by_text),
        preference_centroids=centroids,
        preference_embed=embed,
        enabled_polarities=frozenset(PREFERENCE_POLARITIES),
    )


def _run_ingest(config: MemoryConfig) -> bytes:
    docs_by_text, memories = _build_corpus()
    pipeline = _make_pipeline(config, docs_by_text)
    state = pipeline.create_state()
    for memory in memories:
        pipeline.ingest(state, memory)
    return dump_conversation(state.store)


def test_ingest_same_process_byte_identical() -> None:
    """Same process, same config, same Memory sequence → identical msgpack."""
    config = MemoryConfig()
    bytes_a = _run_ingest(config)
    bytes_b = _run_ingest(config)
    assert bytes_a == bytes_b, "R2 violation: second ingest produced different bytes"


def test_ingest_independent_states_byte_identical() -> None:
    """Two independent pipeline instances also agree byte-for-byte."""
    config = MemoryConfig()
    docs, memories = _build_corpus()

    pipeline_1 = _make_pipeline(config, docs)
    state_1 = pipeline_1.create_state()
    for m in memories:
        pipeline_1.ingest(state_1, m)

    pipeline_2 = _make_pipeline(config, docs)
    state_2 = pipeline_2.create_state()
    for m in memories:
        pipeline_2.ingest(state_2, m)

    assert dump_conversation(state_1.store) == dump_conversation(state_2.store)


def test_ingest_differs_when_config_differs() -> None:
    """Sanity: a config change MUST change the serialized bytes.

    Without this the determinism test above would pass trivially for a
    pipeline that ignored config.
    """
    assert MemoryConfig(canonicalization_match_threshold=0.99).ingestion_fingerprint() != (
        MemoryConfig(canonicalization_match_threshold=0.01).ingestion_fingerprint()
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
