"""GraphStore behavior — R2 iteration order, freeze, BFS.

The tests focus on guarantees the interface promises to Recall: sorted
neighbor iteration, freeze semantics, multi-label unification.
"""

from __future__ import annotations

import pytest

from engram.ingestion.graph import GraphFrozenError, GraphStore, NodeNotFoundError
from engram.ingestion.schema import (
    EDGE_ABOUT,
    EDGE_MENTIONS,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_PREFERENCE,
    LABEL_TURN,
    ClaimPayload,
    EdgeAttrs,
    EntityPayload,
    PreferencePayload,
    TurnPayload,
)


def _make_store() -> GraphStore:
    return GraphStore(conversation_id="c1")


def test_add_node_then_merge_labels() -> None:
    store = _make_store()
    claim = ClaimPayload(
        subject_id="alice_id",
        predicate="like",
        object_id=None,
        object_literal="food",
        asserted_by_turn_id="turn_id",
        asserted_at=None,
        modality="asserted",
        tense="present",
    )
    pref = PreferencePayload(
        holder_id="alice_id",
        polarity="likes",
        target_id=None,
        target_literal="food",
        source_claim_id="claim1",
        confidence=0.5,
    )
    store.add_node("n1", labels=frozenset({LABEL_CLAIM}), payloads={LABEL_CLAIM: claim})
    # Second call unions labels and attaches preference payload.
    store.add_node("n1", labels=frozenset({LABEL_CLAIM, LABEL_PREFERENCE}), payloads={LABEL_PREFERENCE: pref})

    assert store.node_labels("n1") == frozenset({LABEL_CLAIM, LABEL_PREFERENCE})
    attrs = store.get_node("n1")
    assert attrs[LABEL_CLAIM] == claim
    assert attrs[LABEL_PREFERENCE] == pref


def test_nodes_by_label_is_sorted() -> None:
    store = _make_store()
    for name in ["zeta", "alpha", "mu", "kappa"]:
        store.add_node(
            name,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=name, entity_type="X", aliases=(name,))},
        )
    assert store.nodes_by_label(LABEL_ENTITY) == ["alpha", "kappa", "mu", "zeta"]
    assert store.nodes_by_label(LABEL_TURN) == []


def test_add_edge_rejects_missing_endpoint() -> None:
    store = _make_store()
    store.add_node(
        "a",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="a", entity_type="X", aliases=("a",))},
    )
    with pytest.raises(NodeNotFoundError):
        store.add_edge("a", "missing", EdgeAttrs(type=EDGE_ABOUT))


def test_parallel_edge_types_allowed() -> None:
    store = _make_store()
    store.add_node(
        "a",
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(speaker="u", text="t", conversation_id="c1", session_index=1, turn_index=1, timestamp=None)},
    )
    store.add_node(
        "b",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="b", entity_type="X", aliases=("b",))},
    )
    store.add_edge("a", "b", EdgeAttrs(type=EDGE_MENTIONS))
    store.add_edge("a", "b", EdgeAttrs(type=EDGE_ABOUT))
    assert store.has_edge("a", "b", EDGE_MENTIONS)
    assert store.has_edge("a", "b", EDGE_ABOUT)


def test_freeze_blocks_writes() -> None:
    store = _make_store()
    store.add_node(
        "a",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="a", entity_type="X", aliases=("a",))},
    )
    store.freeze()
    assert store.frozen is True
    with pytest.raises(GraphFrozenError):
        store.add_node("b", labels=frozenset({LABEL_ENTITY}), payloads={LABEL_ENTITY: EntityPayload(canonical_form="b", entity_type="X", aliases=("b",))})


def test_out_edges_are_sorted() -> None:
    store = _make_store()
    store.add_node(
        "src",
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(speaker="u", text="t", conversation_id="c1", session_index=1, turn_index=1, timestamp=None)},
    )
    for name in ["zulu", "alpha", "mike"]:
        store.add_node(
            name,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=name, entity_type="X", aliases=(name,))},
        )
        store.add_edge("src", name, EdgeAttrs(type=EDGE_MENTIONS))

    observed = [dst for dst, _ in store.out_edges("src")]
    assert observed == ["alpha", "mike", "zulu"]


def test_bfs_propagates_scores() -> None:
    """Typed-edge BFS composes ``score_src * edge_weight * per_type_weight``."""
    store = _make_store()
    for name in ("seed", "hop1", "hop2"):
        store.add_node(
            name,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=name, entity_type="X", aliases=(name,))},
        )
    store.add_edge("seed", "hop1", EdgeAttrs(type=EDGE_MENTIONS, weight=0.5))
    store.add_edge("hop1", "hop2", EdgeAttrs(type=EDGE_MENTIONS, weight=0.5))

    scores = store.bfs(
        seeds=[("seed", 1.0)],
        edge_weights={EDGE_MENTIONS: 1.0},
        max_depth=2,
    )
    assert scores["seed"] == 1.0
    assert scores["hop1"] == pytest.approx(0.5)
    assert scores["hop2"] == pytest.approx(0.25)


def test_bfs_respects_edge_type_whitelist() -> None:
    store = _make_store()
    for name in ("seed", "via_mentions", "via_about"):
        store.add_node(
            name,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=name, entity_type="X", aliases=(name,))},
        )
    store.add_edge("seed", "via_mentions", EdgeAttrs(type=EDGE_MENTIONS, weight=1.0))
    store.add_edge("seed", "via_about", EdgeAttrs(type=EDGE_ABOUT, weight=1.0))

    scores = store.bfs(
        seeds=[("seed", 1.0)],
        edge_weights={EDGE_MENTIONS: 1.0},
        max_depth=1,
    )
    assert "via_mentions" in scores
    assert "via_about" not in scores


def test_bfs_frontier_cap_is_deterministic() -> None:
    store = _make_store()
    store.add_node(
        "seed",
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="seed", entity_type="X", aliases=("seed",))},
    )
    # Create 5 equal-weight neighbors; cap at 2.
    for name in ("a", "b", "c", "d", "e"):
        store.add_node(
            name,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=name, entity_type="X", aliases=(name,))},
        )
        store.add_edge("seed", name, EdgeAttrs(type=EDGE_MENTIONS, weight=1.0))

    scores = store.bfs(
        seeds=[("seed", 1.0)],
        edge_weights={EDGE_MENTIONS: 1.0},
        max_depth=1,
        max_frontier=2,
    )
    frontier_nodes = sorted(k for k in scores if k != "seed")
    # Ties broken lexicographic — "a" and "b" win.
    assert frontier_nodes == ["a", "b"]
