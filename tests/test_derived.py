"""Derived-rebuild tests (PR-D patch 7, ``docs/design/ingestion.md §7``).

Covers:

- Fingerprint tracks ingestion_fingerprint × primary-state signature.
- Rebuild is idempotent (``R17``): two rebuilds on unchanged primary produce
  byte-equivalent snapshots with identical fingerprints.
- Alias collection walks ``mentions.surface_form`` — distinct surfaces per
  Entity, sorted.
- Co-occurrence counts per-Memory pairs; symmetric ordering; weight = 1.0
  for the most-common pair.
- Reinforcement counts inbound observation edges; earliest / latest
  ``asserted_at`` across those edges.
- Current-preference tracks the latest observation for ``(holder, target)``.
- TimeAnchor chain is sorted by ISO timestamp with ``prev_id`` / ``next_id``
  links at the expected positions.
- ``dump_derived`` / ``load_derived`` roundtrip preserves every entry.
- ``EngramGraphMemorySystem.rebuild_derived`` caches the snapshot on
  :class:`InstanceState` and round-trips through save/load.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from engram import EngramGraphMemorySystem, Memory
from engram.config import MemoryConfig
from engram.engram_memory_system import (
    DERIVED_DIRNAME,
    DERIVED_SNAPSHOT_FILENAME,
)
from engram.ingestion.derived import (
    DerivedFormatError,
    DerivedIndex,
    derived_fingerprint,
    dump_derived,
    load_derived,
    rebuild_derived,
)
from engram.ingestion.graph import GraphStore
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import (
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_TURN,
    PREFERENCE_POLARITIES,
    ClaimPayload,
    EdgeAttrs,
    EntityPayload,
    MemoryPayload,
    PreferencePayload,
    TimeAnchorPayload,
    TurnPayload,
    claim_identity,
    entity_identity,
    memory_identity,
    node_id,
    preference_identity,
    time_anchor_identity,
    turn_identity,
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

# ---------------------------------------------------------------------------
# Synthetic GraphStore fixtures — bypass the pipeline so derived tests are
# insensitive to extractor drift. One Memory, two speakers, overlapping
# mentions, two observations of a Preference that change polarity.
# ---------------------------------------------------------------------------


def _mem(memory_index: int, content: str, timestamp: str) -> tuple[str, MemoryPayload]:
    mid = node_id(memory_identity(memory_index))
    return mid, MemoryPayload(
        memory_index=memory_index,
        content=content,
        timestamp=timestamp,
        speaker="user",
        source=None,
        metadata=(),
    )


def _seed_store() -> GraphStore:
    store = GraphStore(conversation_id="__instance__")

    # Three Memories spanning two rounded TimeAnchors.
    m1_id, m1 = _mem(1, "Alice and Bob met in Paris.", "2026-01-01T10:00:00Z")
    m2_id, m2 = _mem(2, "Alice likes pizza.", "2026-01-01T11:00:00Z")
    m3_id, m3 = _mem(3, "Actually Alice dislikes pizza now.", "2026-02-01T10:00:00Z")

    for mid, m in ((m1_id, m1), (m2_id, m2), (m3_id, m3)):
        store.add_node(mid, labels=frozenset({LABEL_MEMORY}), payloads={LABEL_MEMORY: m})

    t1_id = node_id(turn_identity(m1_id))
    t2_id = node_id(turn_identity(m2_id))
    t3_id = node_id(turn_identity(m3_id))
    for tid, memid, text, ts in (
        (t1_id, m1_id, "Alice and Bob met in Paris.", "2026-01-01T10:00:00Z"),
        (t2_id, m2_id, "Alice likes pizza.", "2026-01-01T11:00:00Z"),
        (t3_id, m3_id, "Actually Alice dislikes pizza now.", "2026-02-01T10:00:00Z"),
    ):
        store.add_node(
            tid,
            labels=frozenset({LABEL_TURN}),
            payloads={LABEL_TURN: TurnPayload(memory_id=memid, text=text, speaker="user", timestamp=ts)},
        )
        store.add_edge(
            tid, memid, EdgeAttrs(type=EDGE_PART_OF, source_memory_id=memid, source_turn_id=tid)
        )

    # Entities — canonical forms distinct by content.
    alice_id = node_id(entity_identity("alice", "PERSON"))
    bob_id = node_id(entity_identity("bob", "PERSON"))
    paris_id = node_id(entity_identity("paris", "GPE"))
    user_id = node_id(entity_identity("user", "SPEAKER"))
    for eid, form, et in (
        (alice_id, "alice", "PERSON"),
        (bob_id, "bob", "PERSON"),
        (paris_id, "paris", "GPE"),
        (user_id, "user", "SPEAKER"),
    ):
        store.add_node(
            eid,
            labels=frozenset({LABEL_ENTITY}),
            payloads={LABEL_ENTITY: EntityPayload(canonical_form=form, entity_type=et)},
        )

    # m1: Alice + Bob + Paris co-occur.
    store.add_edge(
        t1_id, alice_id,
        EdgeAttrs(type=EDGE_MENTIONS, source_memory_id=m1_id, source_turn_id=t1_id,
                  asserted_at="2026-01-01T10:00:00Z", surface_form="Alice"),
    )
    store.add_edge(
        t1_id, bob_id,
        EdgeAttrs(type=EDGE_MENTIONS, source_memory_id=m1_id, source_turn_id=t1_id,
                  asserted_at="2026-01-01T10:00:00Z", surface_form="Bob"),
    )
    store.add_edge(
        t1_id, paris_id,
        EdgeAttrs(type=EDGE_MENTIONS, source_memory_id=m1_id, source_turn_id=t1_id,
                  asserted_at="2026-01-01T10:00:00Z", surface_form="Paris"),
    )
    # m2: Alice only (+ a secondary surface form to test alias dedup).
    store.add_edge(
        t2_id, alice_id,
        EdgeAttrs(type=EDGE_MENTIONS, source_memory_id=m2_id, source_turn_id=t2_id,
                  asserted_at="2026-01-01T11:00:00Z", surface_form="alice"),
    )
    # m3: Alice again, new surface.
    store.add_edge(
        t3_id, alice_id,
        EdgeAttrs(type=EDGE_MENTIONS, source_memory_id=m3_id, source_turn_id=t3_id,
                  asserted_at="2026-02-01T10:00:00Z", surface_form="Alice"),
    )

    # Claim: Alice-met-Bob, asserted in m1 only.
    claim_id = node_id(claim_identity(alice_id, "meet", bob_id, None))
    store.add_node(
        claim_id,
        labels=frozenset({LABEL_CLAIM}),
        payloads={
            LABEL_CLAIM: ClaimPayload(
                subject_id=alice_id, predicate="meet",
                object_id=bob_id, object_literal=None,
                modality="asserted", tense="past",
            )
        },
    )
    store.add_edge(
        t1_id, claim_id,
        EdgeAttrs(type=EDGE_ASSERTS, source_memory_id=m1_id, source_turn_id=t1_id,
                  asserted_at="2026-01-01T10:00:00Z"),
    )

    # Preferences: likes-pizza in m2, dislikes-pizza in m3 (same holder+target,
    # different polarity). Current-truth index should pick the latest.
    pref_like_id = node_id(preference_identity(alice_id, "likes", None, "pizza"))
    pref_dislike_id = node_id(preference_identity(alice_id, "dislikes", None, "pizza"))
    for pid, pol in ((pref_like_id, "likes"), (pref_dislike_id, "dislikes")):
        store.add_node(
            pid,
            labels=frozenset({LABEL_PREFERENCE}),
            payloads={
                LABEL_PREFERENCE: PreferencePayload(
                    holder_id=alice_id, polarity=pol,
                    target_id=None, target_literal="pizza",
                )
            },
        )
    store.add_edge(
        alice_id, pref_like_id,
        EdgeAttrs(type=EDGE_HOLDS_PREFERENCE, source_memory_id=m2_id, source_turn_id=t2_id,
                  asserted_at="2026-01-01T11:00:00Z", weight=0.8),
    )
    store.add_edge(
        alice_id, pref_dislike_id,
        EdgeAttrs(type=EDGE_HOLDS_PREFERENCE, source_memory_id=m3_id, source_turn_id=t3_id,
                  asserted_at="2026-02-01T10:00:00Z", weight=0.9),
    )

    # TimeAnchors — two distinct rounded moments.
    for iso in ("2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z", "2026-02-01T10:00:00Z"):
        aid = node_id(time_anchor_identity(iso))
        store.add_node(
            aid,
            labels=frozenset({LABEL_TIME_ANCHOR}),
            payloads={LABEL_TIME_ANCHOR: TimeAnchorPayload(iso_timestamp=iso)},
        )

    return store


# ---------------------------------------------------------------------------
# rebuild_derived — unit tests against the seeded store
# ---------------------------------------------------------------------------


def test_rebuild_is_idempotent() -> None:
    config = MemoryConfig()
    store = _seed_store()
    a = rebuild_derived(store, config=config)
    b = rebuild_derived(store, config=config)
    assert a == b
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_ingestion_config_changes() -> None:
    store = _seed_store()
    fp_default = derived_fingerprint(MemoryConfig(), store)
    fp_tweaked = derived_fingerprint(
        MemoryConfig(canonicalization_match_threshold=0.99), store
    )
    assert fp_default != fp_tweaked


def test_fingerprint_changes_when_primary_grows() -> None:
    config = MemoryConfig()
    store = _seed_store()
    fp_before = derived_fingerprint(config, store)
    # Add a spurious node; fingerprint should move.
    store.add_node(
        node_id(memory_identity(99)),
        labels=frozenset({LABEL_MEMORY}),
        payloads={
            LABEL_MEMORY: MemoryPayload(
                memory_index=99, content="later", timestamp=None,
                speaker=None, source=None, metadata=(),
            )
        },
    )
    assert derived_fingerprint(config, store) != fp_before


def test_aliases_collect_distinct_surface_forms() -> None:
    config = MemoryConfig()
    index = rebuild_derived(_seed_store(), config=config)
    alice_id = node_id(entity_identity("alice", "PERSON"))
    by_entity = {e.entity_id: e for e in index.aliases}
    assert alice_id in by_entity
    # Two distinct surfaces observed for Alice ("Alice" + "alice"); sorted.
    assert by_entity[alice_id].aliases == ("Alice", "alice")


def test_aliases_omit_entities_with_no_mentions() -> None:
    config = MemoryConfig()
    store = _seed_store()
    # user Entity has no inbound mentions edges — it's a speaker, not a
    # mention target. Should be absent from the alias index.
    index = rebuild_derived(store, config=config)
    user_id = node_id(entity_identity("user", "SPEAKER"))
    assert all(e.entity_id != user_id for e in index.aliases)


def test_co_occurrence_counts_per_memory_pairs() -> None:
    config = MemoryConfig()
    index = rebuild_derived(_seed_store(), config=config)
    alice_id = node_id(entity_identity("alice", "PERSON"))
    bob_id = node_id(entity_identity("bob", "PERSON"))
    paris_id = node_id(entity_identity("paris", "GPE"))
    # Only m1 has Alice+Bob+Paris together — three pairs, each count=1.
    pairs = {
        (e.entity_a, e.entity_b): (e.count, e.weight) for e in index.co_occurrence
    }
    # Pairs are lexicographic.
    ab = tuple(sorted((alice_id, bob_id)))
    ap = tuple(sorted((alice_id, paris_id)))
    bp = tuple(sorted((bob_id, paris_id)))
    assert pairs[ab] == (1, 1.0)
    assert pairs[ap] == (1, 1.0)
    assert pairs[bp] == (1, 1.0)


def test_co_occurrence_is_empty_without_pairs() -> None:
    """Single-entity Memories produce no pairs (nothing to co-occur with)."""
    store = GraphStore(conversation_id="__instance__")
    m_id, m = _mem(1, "solo", "2026-01-01T00:00:00Z")
    t_id = node_id(turn_identity(m_id))
    alice_id = node_id(entity_identity("alice", "PERSON"))
    store.add_node(m_id, labels=frozenset({LABEL_MEMORY}), payloads={LABEL_MEMORY: m})
    store.add_node(
        t_id,
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(memory_id=m_id, text="solo", speaker="user", timestamp=None)},
    )
    store.add_node(
        alice_id,
        labels=frozenset({LABEL_ENTITY}),
        payloads={LABEL_ENTITY: EntityPayload(canonical_form="alice", entity_type="PERSON")},
    )
    store.add_edge(
        t_id, alice_id,
        EdgeAttrs(type=EDGE_MENTIONS, source_memory_id=m_id, source_turn_id=t_id,
                  surface_form="Alice"),
    )
    assert rebuild_derived(store, config=MemoryConfig()).co_occurrence == ()


def test_reinforcement_counts_and_bounds() -> None:
    index = rebuild_derived(_seed_store(), config=MemoryConfig())
    alice_id = node_id(entity_identity("alice", "PERSON"))
    bob_id = node_id(entity_identity("bob", "PERSON"))
    claim_id = node_id(claim_identity(alice_id, "meet", bob_id, None))
    pref_like_id = node_id(preference_identity(alice_id, "likes", None, "pizza"))
    by_id = {e.node_id: e for e in index.reinforcement}

    assert by_id[claim_id].count == 1
    assert by_id[claim_id].kind == "claim"
    assert by_id[claim_id].earliest == "2026-01-01T10:00:00Z"
    assert by_id[claim_id].latest == "2026-01-01T10:00:00Z"

    assert by_id[pref_like_id].kind == "preference"
    assert by_id[pref_like_id].count == 1


def test_current_preference_picks_latest_asserted() -> None:
    index = rebuild_derived(_seed_store(), config=MemoryConfig())
    alice_id = node_id(entity_identity("alice", "PERSON"))
    current = [
        e for e in index.current_preference
        if e.holder_id == alice_id and e.target_key == "literal:pizza"
    ]
    assert len(current) == 1
    # m3 is later than m2 — dislikes wins.
    assert current[0].polarity == "dislikes"
    assert current[0].asserted_at == "2026-02-01T10:00:00Z"


def test_time_anchor_chain_sorted_with_prev_next_links() -> None:
    index = rebuild_derived(_seed_store(), config=MemoryConfig())
    chain = index.time_anchor_chain
    assert len(chain) == 3
    isos = [e.iso_timestamp for e in chain]
    assert isos == sorted(isos)
    assert chain[0].prev_id is None
    assert chain[0].next_id == chain[1].time_anchor_id
    assert chain[1].prev_id == chain[0].time_anchor_id
    assert chain[1].next_id == chain[2].time_anchor_id
    assert chain[-1].next_id is None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_dump_and_load_roundtrip() -> None:
    original = rebuild_derived(_seed_store(), config=MemoryConfig())
    restored = load_derived(dump_derived(original))
    assert restored == original


def test_dump_is_byte_stable() -> None:
    index = rebuild_derived(_seed_store(), config=MemoryConfig())
    assert dump_derived(index) == dump_derived(index)


def test_load_rejects_wrong_schema_version() -> None:
    import msgpack  # type: ignore[import-not-found]

    empty = DerivedIndex(fingerprint="abc")
    envelope = msgpack.unpackb(dump_derived(empty), raw=False, strict_map_key=False)
    envelope["schema_version"] = envelope["schema_version"] + 1
    tampered = msgpack.packb(envelope, use_bin_type=True)
    with pytest.raises(DerivedFormatError):
        load_derived(tampered)


# ---------------------------------------------------------------------------
# End-to-end via EngramGraphMemorySystem — rebuild_derived + save/load
# ---------------------------------------------------------------------------


def _build_doc() -> tuple[Memory, dict]:
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
    return memory, {text: doc}


def _make_system() -> EngramGraphMemorySystem:
    config = MemoryConfig()
    _memory, docs = _build_doc()
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


def test_rebuild_derived_caches_on_state() -> None:
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    assert system.get_state().derived is None  # type: ignore[union-attr]
    snapshot = system.rebuild_derived()
    assert snapshot is not None
    assert system.get_state().derived is snapshot  # type: ignore[union-attr]


def test_rebuild_derived_returns_none_without_state() -> None:
    system = _make_system()
    assert system.rebuild_derived() is None


def test_derived_snapshot_persists_through_save_load(tmp_path: Path) -> None:
    system = _make_system()
    memory, _ = _build_doc()
    asyncio.run(system.ingest(memory))
    system.rebuild_derived()

    asyncio.run(system.save_state(tmp_path))
    assert (tmp_path / DERIVED_DIRNAME / DERIVED_SNAPSHOT_FILENAME).exists()

    restored = _make_system()
    asyncio.run(restored.load_state(tmp_path))
    restored_state = restored.get_state()
    assert restored_state is not None
    assert restored_state.derived is not None
    assert restored_state.derived == system.get_state().derived  # type: ignore[union-attr]
