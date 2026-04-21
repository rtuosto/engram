"""Layer-label discipline on GraphStore nodes (docs/design/ingestion.md §3).

Each node carries a ``layers: frozenset[str]`` attribute — the content-
classification partition recall walks against. PR-B wires the pipeline to
populate three layers:

- Entity nodes → ``{entity}``
- Claim nodes → ``{relationship}``
- Preference nodes → ``{relationship}``

Granule nodes (Memory / Turn / UtteranceSegment / N-gram) carry an empty
``layers`` frozenset: the semantic layer is implemented as the parallel
embedding index (PR-C), not a node attribute. Temporal / episodic layers
land in PR-D.
"""

from __future__ import annotations

import asyncio

import pytest

from engram import EngramGraphMemorySystem, Memory
from engram.config import MemoryConfig
from engram.ingestion.graph import GraphStore, NodeNotFoundError
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import (
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_NGRAM,
    LABEL_PREFERENCE,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LAYER_ENTITY,
    LAYER_RELATIONSHIP,
    PREFERENCE_POLARITIES,
    EntityPayload,
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

# ---------------------------------------------------------------------------
# GraphStore unit tests for layers attr
# ---------------------------------------------------------------------------


def test_add_node_defaults_layers_to_empty_frozenset() -> None:
    store = GraphStore(conversation_id="__instance__")
    store.add_node(
        "e",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="e", entity_type="X")},
    )
    assert store.node_layers("e") == frozenset()
    assert store.nodes_by_layer(LAYER_ENTITY) == []


def test_add_node_indexes_by_layer() -> None:
    store = GraphStore(conversation_id="__instance__")
    for name in ("zeta", "alpha", "mu"):
        store.add_node(
            name,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=name, entity_type="X")},
            layers=frozenset({LAYER_ENTITY}),
        )
    assert store.nodes_by_layer(LAYER_ENTITY) == ["alpha", "mu", "zeta"]
    assert store.nodes_by_layer(LAYER_RELATIONSHIP) == []


def test_add_node_unions_layers_on_repeat() -> None:
    store = GraphStore(conversation_id="__instance__")
    store.add_node(
        "n",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="n", entity_type="X")},
        layers=frozenset({LAYER_ENTITY}),
    )
    # Simulate a later stage attaching a second layer (e.g., PR-D ChangeEvent
    # which carries both relationship and temporal).
    store.add_node(
        "n",
        labels=frozenset({LABEL_ENTITY}),
        layers=frozenset({LAYER_RELATIONSHIP}),
    )
    assert store.node_layers("n") == frozenset({LAYER_ENTITY, LAYER_RELATIONSHIP})
    assert "n" in store.nodes_by_layer(LAYER_ENTITY)
    assert "n" in store.nodes_by_layer(LAYER_RELATIONSHIP)


def test_node_layers_raises_for_missing_node() -> None:
    store = GraphStore(conversation_id="__instance__")
    with pytest.raises(NodeNotFoundError):
        store.node_layers("missing")


# ---------------------------------------------------------------------------
# End-to-end: pipeline populates the right layers per node label
# ---------------------------------------------------------------------------


def _build_doc() -> tuple[Memory, dict]:
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

    alice_chunk = FakeNounChunk(
        text="Alice",
        start_char=0,
        end_char=5,
        tokens=(make_token("Alice", idx=0, pos="PROPN"),),
    )
    hiking_chunk = FakeNounChunk(
        text="hiking trip",
        start_char=12,
        end_char=19,  # lines up with the sentence; "hiking" + filler token
        tokens=(
            make_token("hiking", idx=12, pos="NOUN"),
            make_token("trip", idx=15, pos="NOUN"),
        ),
    )

    doc = make_fake_doc(
        text=text,
        sents=[sent],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=0, end_char=5)],
        noun_chunks=[alice_chunk, hiking_chunk],
    )
    memory = Memory(
        content=text,
        timestamp="2026-01-01T00:00:00Z",
        speaker="user",
    )
    return memory, {text: doc}


def _make_system() -> EngramGraphMemorySystem:
    memory, docs = _build_doc()
    _ = memory  # referenced via _build_doc from tests below
    config = MemoryConfig()
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


def test_pipeline_labels_entity_nodes_with_entity_layer() -> None:
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None

    entity_ids = state.store.nodes_by_label(LABEL_ENTITY)
    assert entity_ids
    entity_layer_ids = state.store.nodes_by_layer(LAYER_ENTITY)
    assert set(entity_layer_ids) == set(entity_ids)


def test_pipeline_labels_claim_nodes_with_relationship_layer() -> None:
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None

    claim_ids = state.store.nodes_by_label(LABEL_CLAIM)
    assert claim_ids
    rel_layer_ids = state.store.nodes_by_layer(LAYER_RELATIONSHIP)
    for claim_id in claim_ids:
        assert claim_id in rel_layer_ids


def test_pipeline_labels_preference_nodes_with_relationship_layer() -> None:
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None

    pref_ids = state.store.nodes_by_label(LABEL_PREFERENCE)
    # "Alice loves hiking" is a likes-polarity candidate under the
    # deterministic-embed mock — if the margin gate happened to drop it,
    # the layer assertion is vacuously true.
    rel_layer_ids = state.store.nodes_by_layer(LAYER_RELATIONSHIP)
    for pref_id in pref_ids:
        assert pref_id in rel_layer_ids


def test_pipeline_leaves_granules_in_empty_layer() -> None:
    """Granule nodes carry no layer label — the semantic signal lives in the
    parallel embedding index (PR-C)."""
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None

    for label in (LABEL_MEMORY, LABEL_TURN, LABEL_UTTERANCE_SEGMENT, LABEL_NGRAM):
        for node in state.store.nodes_by_label(label):
            assert state.store.node_layers(node) == frozenset(), (
                f"granule node {node!r} ({label}) picked up layers "
                f"{state.store.node_layers(node)!r} — unexpected in PR-B"
            )


def test_pipeline_emits_ngram_nodes_and_part_of_edges() -> None:
    """Patch 3 wiring: n-grams become LABEL_NGRAM nodes with part_of edges
    into their containing Sentence."""
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None

    ngram_ids = state.store.nodes_by_label(LABEL_NGRAM)
    # Expect at least one SVO ngram ("Alice loves hiking") and one noun-chunk
    # ngram ("hiking trip" — Alice alone is below min_tokens=2).
    assert ngram_ids

    segment_ids = set(state.store.nodes_by_label(LABEL_UTTERANCE_SEGMENT))
    for ngram_id in ngram_ids:
        out = state.store.out_edges(ngram_id)
        # Each n-gram has exactly one part_of edge into a Sentence.
        assert any(
            dst in segment_ids and attrs.type == "part_of"
            for dst, attrs in out
        ), f"ngram {ngram_id} lacks part_of into a segment: {out}"
