"""Persistence roundtrip + schema-version error tests (R12).

A ``dump_conversation`` followed by ``load_conversation`` must reconstruct
the GraphStore faithfully — same nodes, same labels, same payloads, same
edges, same edge attrs, same frozen state. Schema-version drift must raise
:class:`SchemaVersionMismatch` rather than silently migrating.
"""

from __future__ import annotations

import msgpack
import pytest

from engram.ingestion.graph import GraphStore
from engram.ingestion.persist import (
    MEMORY_SYSTEM_ID,
    SCHEMA_VERSION,
    PersistFormatError,
    SchemaVersionMismatch,
    dump_conversation,
    load_conversation,
)
from engram.ingestion.schema import (
    EDGE_ABOUT,
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_PREFERENCE,
    LABEL_TURN,
    ClaimPayload,
    EdgeAttrs,
    EntityPayload,
    MemoryPayload,
    PreferencePayload,
    TurnPayload,
    claim_identity,
    entity_identity,
    memory_identity,
    node_id,
    preference_identity,
    turn_identity,
)


def _build_store() -> GraphStore:
    store = GraphStore(conversation_id="__instance__")

    memory_id = node_id(memory_identity(1))
    turn_id = node_id(turn_identity(memory_id))
    alice_id = node_id(entity_identity("alice", "PERSON"))
    user_id = node_id(entity_identity("user", "SPEAKER"))
    claim_id = node_id(claim_identity(alice_id, "like", None, "spicy food"))
    pref_id = node_id(preference_identity(user_id, "likes", None, "spicy food"))

    store.add_node(
        memory_id,
        labels=frozenset({LABEL_MEMORY}),
        payloads={
            LABEL_MEMORY: MemoryPayload(
                memory_index=1,
                content="Alice likes spicy food.",
                timestamp="2026-01-01T00:00:00Z",
                speaker="user",
                source="conversation_turn",
                metadata=(),
            )
        },
    )
    store.add_node(
        turn_id,
        labels=frozenset({LABEL_TURN}),
        payloads={
            LABEL_TURN: TurnPayload(
                memory_id=memory_id,
                text="Alice likes spicy food.",
                speaker="user",
                timestamp="2026-01-01T00:00:00Z",
            )
        },
    )
    store.add_node(
        alice_id,
        labels=frozenset({LABEL_ENTITY}),
        payloads={
            LABEL_ENTITY: EntityPayload(canonical_form="alice", entity_type="PERSON")
        },
    )
    store.add_node(
        user_id,
        labels=frozenset({LABEL_ENTITY}),
        payloads={
            LABEL_ENTITY: EntityPayload(canonical_form="user", entity_type="SPEAKER")
        },
    )
    store.add_node(
        claim_id,
        labels=frozenset({LABEL_CLAIM}),
        payloads={
            LABEL_CLAIM: ClaimPayload(
                subject_id=alice_id,
                predicate="like",
                object_id=None,
                object_literal="spicy food",
                modality="asserted",
                tense="present",
            )
        },
    )
    store.add_node(
        pref_id,
        labels=frozenset({LABEL_PREFERENCE}),
        payloads={
            LABEL_PREFERENCE: PreferencePayload(
                holder_id=user_id,
                polarity="likes",
                target_id=None,
                target_literal="spicy food",
            )
        },
    )

    store.add_edge(
        turn_id, memory_id,
        EdgeAttrs(type=EDGE_PART_OF, weight=1.0, source_memory_id=memory_id, source_turn_id=turn_id),
    )
    store.add_edge(
        turn_id, claim_id,
        EdgeAttrs(
            type=EDGE_ASSERTS, weight=1.0, source_memory_id=memory_id, source_turn_id=turn_id,
            asserted_at="2026-01-01T00:00:00Z",
        ),
    )
    store.add_edge(
        claim_id, alice_id,
        EdgeAttrs(type=EDGE_ABOUT, weight=1.0, source_memory_id=memory_id, source_turn_id=turn_id),
    )
    store.add_edge(
        user_id, pref_id,
        EdgeAttrs(
            type=EDGE_HOLDS_PREFERENCE, weight=0.42,
            source_memory_id=memory_id, source_turn_id=turn_id,
            asserted_at="2026-01-01T00:00:00Z",
        ),
    )
    return store


def test_roundtrip_preserves_nodes_and_edges() -> None:
    original = _build_store()
    original.freeze()
    data = dump_conversation(original)
    restored = load_conversation(data)

    assert restored.conversation_id == original.conversation_id
    assert restored.frozen is True
    assert restored.num_nodes() == original.num_nodes()
    assert restored.num_edges() == original.num_edges()

    for (nid_o, attrs_o), (nid_r, attrs_r) in zip(
        original.iter_nodes(), restored.iter_nodes(), strict=True
    ):
        assert nid_o == nid_r
        assert attrs_o["labels"] == attrs_r["labels"]
        for label in attrs_o["labels"]:
            assert attrs_o[label] == attrs_r[label]

    for (src_o, dst_o, type_o, attrs_o), (src_r, dst_r, type_r, attrs_r) in zip(
        original.iter_edges(), restored.iter_edges(), strict=True
    ):
        assert (src_o, dst_o, type_o) == (src_r, dst_r, type_r)
        assert attrs_o == attrs_r


def test_roundtrip_is_byte_stable() -> None:
    store = _build_store()
    store.freeze()
    assert dump_conversation(store) == dump_conversation(store)


def test_schema_version_mismatch_rejects_older() -> None:
    store = _build_store()
    data = dump_conversation(store)
    envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    envelope["schema_version"] = SCHEMA_VERSION - 1
    tampered = msgpack.packb(envelope, use_bin_type=True)
    with pytest.raises(SchemaVersionMismatch):
        load_conversation(tampered)


def test_schema_version_mismatch_rejects_newer() -> None:
    store = _build_store()
    data = dump_conversation(store)
    envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    envelope["schema_version"] = SCHEMA_VERSION + 1
    tampered = msgpack.packb(envelope, use_bin_type=True)
    with pytest.raises(SchemaVersionMismatch):
        load_conversation(tampered)


def test_foreign_memory_system_id_rejected() -> None:
    store = _build_store()
    data = dump_conversation(store)
    envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    envelope["memory_system_id"] = "not_engram"
    tampered = msgpack.packb(envelope, use_bin_type=True)
    with pytest.raises(PersistFormatError):
        load_conversation(tampered)


def test_missing_schema_version_rejected() -> None:
    store = _build_store()
    data = dump_conversation(store)
    envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    del envelope["schema_version"]
    tampered = msgpack.packb(envelope, use_bin_type=True)
    with pytest.raises(PersistFormatError):
        load_conversation(tampered)


def test_memory_system_id_constant_matches() -> None:
    """MEMORY_SYSTEM_ID should align with EngramGraphMemorySystem's id."""
    from engram import EngramGraphMemorySystem

    assert EngramGraphMemorySystem.memory_system_id == MEMORY_SYSTEM_ID
