"""Unit tests for recall stage modules (seeding / expansion / scoring / assembly).

These tests exercise each stage in isolation against a hand-built graph so
a regression in one stage doesn't hide behind a pipeline-level aggregate.
"""

from __future__ import annotations

import numpy as np

from engram.ingestion.extractors.canonicalization import (
    EntityRegistry,
    normalize,
)
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_ENTITY,
    LABEL_NGRAM,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    EdgeAttrs,
    EntityPayload,
    NgramPayload,
    TurnPayload,
    UtteranceSegmentPayload,
    entity_identity,
    ngram_identity,
    node_id,
    segment_identity,
    turn_identity,
)
from engram.ingestion.vector_index import (
    GRANULARITY_NGRAM,
    GRANULARITY_SENTENCE,
    GRANULARITY_TURN,
    VectorIndex,
)
from engram.recall.expansion import expand
from engram.recall.scoring import select_passages
from engram.recall.seeding import (
    entity_anchored_seed,
    merge_seeds,
    semantic_seed,
)
from tests._fake_nlp import (
    FakeEnt,
    FakeSent,
    deterministic_embed,
    make_fake_doc,
)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_semantic_seed_pulls_from_each_granularity() -> None:
    idx = VectorIndex(dim=4)
    idx.add("turn_a", GRANULARITY_TURN, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("sent_a", GRANULARITY_SENTENCE, np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32))
    idx.add("ngram_a", GRANULARITY_NGRAM, np.array([0.8, 0.0, 0.6, 0.0], dtype=np.float32))

    def embed(texts):
        # constant direction query aligned with turn_a
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype=np.float32)

    weights = {"turn": 1.0, "sentence": 0.5, "ngram": 0.5}
    seeds = semantic_seed(
        "Q", idx, embed, granularity_weights=weights, top_n_per_granularity=4
    )
    ids = {node_id for node_id, _ in seeds}
    assert {"turn_a", "sent_a", "ngram_a"} <= ids
    # The weighting scales similarity; turn_a should score highest.
    seeds_map = dict(seeds)
    assert seeds_map["turn_a"] >= seeds_map["sent_a"]


def test_semantic_seed_skips_zero_weight_granularities() -> None:
    idx = VectorIndex(dim=4)
    idx.add("t1", GRANULARITY_TURN, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("s1", GRANULARITY_SENTENCE, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def embed(texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype=np.float32)

    seeds = semantic_seed(
        "Q",
        idx,
        embed,
        granularity_weights={"turn": 1.0, "sentence": 0.0, "ngram": 0.0},
        top_n_per_granularity=8,
    )
    ids = {node_id for node_id, _ in seeds}
    assert "t1" in ids
    assert "s1" not in ids


def test_semantic_seed_empty_index() -> None:
    idx = VectorIndex(dim=4)

    def embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    seeds = semantic_seed(
        "Q", idx, embed, granularity_weights={"turn": 1.0}, top_n_per_granularity=2
    )
    assert seeds == []


def test_entity_anchored_seed_resolves_mentions() -> None:
    store = GraphStore(conversation_id="__instance__")
    # Build an Entity and a Turn granule with a mentions edge.
    normalized = normalize("Alice")
    entity_id = node_id(entity_identity(normalized, "PERSON"))
    store.add_node(
        entity_id,
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form=normalized, entity_type="PERSON")},
    )
    # A Turn granule that mentions Alice.
    memory_id = "mem_x"
    turn_id = node_id(turn_identity(memory_id))
    store.add_node(
        turn_id,
        labels=frozenset({LABEL_TURN}),
        payloads={
            LABEL_TURN: TurnPayload(
                memory_id=memory_id, text="Alice loves hiking.", speaker="user", timestamp=None
            )
        },
    )
    store.add_edge(
        turn_id,
        entity_id,
        EdgeAttrs(type=EDGE_MENTIONS, weight=1.0, source_memory_id=memory_id, surface_form="Alice"),
    )

    registry = EntityRegistry()
    registry.by_type_and_form[("PERSON", normalized)] = entity_id

    doc = make_fake_doc(
        text="What did Alice say?",
        sents=[FakeSent(text="What did Alice say?", start_char=0, end_char=19)],
        ents=[FakeEnt(text="Alice", label_="PERSON", start_char=9, end_char=14)],
    )
    seeds = entity_anchored_seed(doc, registry, store)
    ids = {nid for nid, _ in seeds}
    assert entity_id in ids
    assert turn_id in ids


def test_entity_anchored_seed_ignores_unresolved_mentions() -> None:
    store = GraphStore(conversation_id="__instance__")
    registry = EntityRegistry()
    doc = make_fake_doc(
        text="unknown entity query",
        sents=[FakeSent(text="unknown entity query", start_char=0, end_char=20)],
        ents=[FakeEnt(text="Unknown", label_="PERSON", start_char=0, end_char=7)],
    )
    assert entity_anchored_seed(doc, registry, store) == []


def test_merge_seeds_keeps_max_and_caps() -> None:
    merged = merge_seeds(
        [("a", 0.5), ("b", 0.8)],
        [("a", 0.9), ("c", 0.3)],
        total_cap=2,
    )
    assert merged == [("a", 0.9), ("b", 0.8)]


def test_merge_seeds_cap_zero_returns_all() -> None:
    """total_cap=0 is treated as "no cap" so tests and callers can turn the
    cap off without reaching into the internals."""
    merged = merge_seeds(
        [("a", 0.5), ("b", 0.3)],
        total_cap=0,
    )
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def test_expand_respects_max_depth() -> None:
    store = GraphStore(conversation_id="__instance__")
    memory_id = "mem_x"
    turn_id = node_id(turn_identity(memory_id))
    seg_id = node_id(segment_identity(turn_id, 0))
    ngram_id = node_id(ngram_identity(seg_id, "noun_chunk", "alice"))

    store.add_node(
        turn_id,
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(memory_id=memory_id, text="Alice loves hiking.", speaker="user", timestamp=None)},
    )
    store.add_node(
        seg_id,
        labels=frozenset({LABEL_UTTERANCE_SEGMENT}),
        payloads={
            LABEL_UTTERANCE_SEGMENT: UtteranceSegmentPayload(
                text="Alice loves hiking.", turn_id=turn_id, segment_index=0, char_span=(0, 19)
            )
        },
    )
    store.add_node(
        ngram_id,
        labels=frozenset({LABEL_NGRAM}),
        payloads={
            LABEL_NGRAM: NgramPayload(
                normalized_text="alice",
                surface_form="Alice",
                segment_id=seg_id,
                ngram_kind="noun_chunk",
                char_span=(0, 5),
            )
        },
    )
    store.add_edge(ngram_id, seg_id, EdgeAttrs(type=EDGE_PART_OF, weight=1.0))
    store.add_edge(seg_id, turn_id, EdgeAttrs(type=EDGE_PART_OF, weight=1.0))

    # max_depth=1 → from ngram we can reach seg but not turn.
    scores = expand(
        store,
        [(ngram_id, 1.0)],
        edge_weights={"part_of": 1.0},
        max_depth=1,
        max_frontier=32,
    )
    assert ngram_id in scores
    assert seg_id in scores
    assert turn_id not in scores

    # max_depth=2 → reaches turn.
    scores = expand(
        store,
        [(ngram_id, 1.0)],
        edge_weights={"part_of": 1.0},
        max_depth=2,
        max_frontier=32,
    )
    assert turn_id in scores


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_select_passages_buckets_by_granule() -> None:
    store = GraphStore(conversation_id="__instance__")
    memory_id = "mem_x"
    turn_id = node_id(turn_identity(memory_id))
    seg_id = node_id(segment_identity(turn_id, 0))
    store.add_node(
        turn_id,
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(memory_id=memory_id, text="...", speaker=None, timestamp=None)},
    )
    store.add_node(
        seg_id,
        labels=frozenset({LABEL_UTTERANCE_SEGMENT}),
        payloads={
            LABEL_UTTERANCE_SEGMENT: UtteranceSegmentPayload(
                text="hi", turn_id=turn_id, segment_index=0, char_span=(0, 2)
            )
        },
    )
    store.add_node(
        "entity_x",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="x", entity_type="MISC")},
    )

    walk_scores = {turn_id: 0.9, seg_id: 1.0, "entity_x": 2.0}
    picked = select_passages(walk_scores, store, max_passages=5)
    picked_map = dict(picked)
    # Entity node is dropped; granules survive with their own scores.
    assert "entity_x" not in picked_map
    assert turn_id in picked_map
    assert seg_id in picked_map
    # Order is by score desc.
    assert picked[0][0] == seg_id


def test_select_passages_respects_max_passages() -> None:
    store = GraphStore(conversation_id="__instance__")
    ids: list[str] = []
    for i in range(4):
        tid = node_id({"type": LABEL_TURN, "memory_id": f"m_{i}"})
        ids.append(tid)
        store.add_node(
            tid,
            labels=frozenset({LABEL_TURN}),
            payloads={LABEL_TURN: TurnPayload(memory_id=f"m_{i}", text="t", speaker=None, timestamp=None)},
        )

    walk = {tid: float(i + 1) for i, tid in enumerate(ids)}
    picked = select_passages(walk, store, max_passages=2)
    assert len(picked) == 2
    # Highest scores first
    assert picked[0][1] == 4.0
    assert picked[1][1] == 3.0


def test_select_passages_empty_input() -> None:
    store = GraphStore(conversation_id="__instance__")
    assert select_passages({}, store, max_passages=4) == []


def _unused_placeholder() -> None:
    """Deterministic-embed is unused here but imported; reference once so
    linting doesn't complain about test helpers that may migrate in."""
    _ = deterministic_embed(dim=4)
