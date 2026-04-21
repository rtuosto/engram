"""extraction_coverage — node/edge/layer counts over a hand-built store."""

from __future__ import annotations

from engram.diagnostics import extraction_coverage
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_NGRAM,
    LABEL_TURN,
    LAYER_ENTITY,
    NGRAM_KIND_NOUN_CHUNK,
    NGRAM_KIND_SVO,
    EdgeAttrs,
    EntityPayload,
    MemoryPayload,
    NgramPayload,
    TurnPayload,
)


def _store_with_mixed_content() -> GraphStore:
    store = GraphStore(conversation_id="__instance__")
    store.add_node(
        "m1",
        labels=frozenset({LABEL_MEMORY}),
        payloads={LABEL_MEMORY: MemoryPayload(
            memory_index=0,
            content="hello world",
            timestamp=None,
            speaker=None,
            source=None,
            metadata=(),
        )},
    )
    store.add_node(
        "t1",
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(memory_id="m1", text="hi", speaker="u", timestamp=None)},
    )
    store.add_node(
        "n1",
        labels=frozenset({LABEL_NGRAM}),
        payloads={LABEL_NGRAM: NgramPayload(
            normalized_text="eiffel tower",
            surface_form="Eiffel Tower",
            segment_id="s1",
            ngram_kind=NGRAM_KIND_NOUN_CHUNK,
            char_span=(0, 12),
        )},
    )
    store.add_node(
        "n2",
        labels=frozenset({LABEL_NGRAM}),
        payloads={LABEL_NGRAM: NgramPayload(
            normalized_text="alice visits paris",
            surface_form="Alice visits Paris",
            segment_id="s1",
            ngram_kind=NGRAM_KIND_SVO,
            char_span=(0, 18),
        )},
    )
    store.add_node(
        "e1",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="alice", entity_type="PERSON")},
        layers=frozenset({LAYER_ENTITY}),
    )
    store.add_edge("n1", "t1", EdgeAttrs(type=EDGE_PART_OF, weight=1.0))
    store.add_edge("n2", "t1", EdgeAttrs(type=EDGE_PART_OF, weight=1.0))
    store.add_edge("t1", "e1", EdgeAttrs(type=EDGE_MENTIONS, weight=1.0, surface_form="Alice"))
    return store


def test_coverage_label_counts() -> None:
    report = extraction_coverage(_store_with_mixed_content())
    as_dict = dict(report.nodes_by_label)
    assert as_dict[LABEL_MEMORY] == 1
    assert as_dict[LABEL_TURN] == 1
    assert as_dict[LABEL_NGRAM] == 2
    assert as_dict[LABEL_ENTITY] == 1


def test_coverage_layer_counts_split_entity_vs_unlayered() -> None:
    report = extraction_coverage(_store_with_mixed_content())
    as_dict = dict(report.nodes_by_layer)
    assert as_dict[LAYER_ENTITY] == 1
    # Memory + Turn + 2 N-grams = 4 granule/memory nodes without a layer.
    assert as_dict["(unlayered)"] == 4


def test_coverage_edge_counts() -> None:
    report = extraction_coverage(_store_with_mixed_content())
    as_dict = dict(report.edges_by_type)
    assert as_dict[EDGE_PART_OF] == 2
    assert as_dict[EDGE_MENTIONS] == 1


def test_coverage_ngram_kinds() -> None:
    report = extraction_coverage(_store_with_mixed_content())
    as_dict = dict(report.ngram_kinds)
    assert as_dict[NGRAM_KIND_NOUN_CHUNK] == 1
    assert as_dict[NGRAM_KIND_SVO] == 1


def test_coverage_totals_granule_count_includes_memory_turn_ngrams() -> None:
    report = extraction_coverage(_store_with_mixed_content())
    as_dict = dict(report.totals)
    assert as_dict["n_memories"] == 1
    assert as_dict["n_entities"] == 1
    assert as_dict["n_granules"] == 4  # Memory + Turn + 2 N-grams
    assert as_dict["n_claims"] == 0
    assert as_dict["n_preferences"] == 0
    assert as_dict["n_nodes"] == 5
    assert as_dict["n_edges"] == 3


def test_coverage_is_deterministic() -> None:
    a = extraction_coverage(_store_with_mixed_content())
    b = extraction_coverage(_store_with_mixed_content())
    assert a == b


def test_coverage_empty_store_all_zeros() -> None:
    store = GraphStore(conversation_id="__instance__")
    report = extraction_coverage(store)
    as_dict = dict(report.totals)
    assert as_dict["n_nodes"] == 0
    assert as_dict["n_edges"] == 0
    assert report.nodes_by_label == ()
    assert report.edges_by_type == ()
