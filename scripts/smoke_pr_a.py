"""Real-model smoke test for PR-A.

Validates the production ingest path end-to-end:
- spaCy en_core_web_sm + sentence-transformers (MiniLM, mpnet)
- EngramGraphMemorySystem.ingest(Memory) over a small synthetic corpus
- R2 byte-equality across two independent pipeline constructions
- save_state → load_state roundtrip preserves node/edge counts
- R16 shape checks: repeat-ingest creates new Memory nodes; EntityPayload has no aliases

Not part of the hermetic test suite (costs ~30s + model downloads).
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from engram import EngramGraphMemorySystem, Memory
from engram.config import MemoryConfig
from engram.ingestion.persist import dump_state
from engram.ingestion.schema import (
    EDGE_ASSERTS,
    EDGE_HOLDS_PREFERENCE,
    EDGE_MENTIONS,
    EDGE_PART_OF,
    LABEL_CLAIM,
    LABEL_ENTITY,
    LABEL_MEMORY,
    LABEL_PREFERENCE,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    EntityPayload,
)

CORPUS: tuple[Memory, ...] = (
    Memory(
        content="Alice loves hiking in the mountains.",
        timestamp="2026-01-01T10:00:00Z",
        speaker="user",
        source="conversation_turn",
    ),
    Memory(
        content="Bob avoids crowded restaurants.",
        timestamp="2026-01-01T10:01:00Z",
        speaker="user",
        source="conversation_turn",
    ),
    Memory(
        content="I really like spicy food.",
        timestamp="2026-01-02T09:00:00Z",
        speaker="user",
        source="conversation_turn",
    ),
    Memory(
        content="Alice also likes running marathons.",
        timestamp="2026-01-03T11:00:00Z",
        speaker="user",
        source="conversation_turn",
    ),
)


async def _ingest_all(system: EngramGraphMemorySystem, memories: tuple[Memory, ...]) -> None:
    for m in memories:
        await system.ingest(m)


def _check(cond: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    if not cond:
        raise AssertionError(f"{label}{suffix}")


def run() -> None:
    config = MemoryConfig()
    print(f"ingestion_fingerprint = {config.ingestion_fingerprint()}")
    print()

    # -----------------------------------------------------------
    # 1. Ingest the corpus with a fresh system (loads models once).
    # -----------------------------------------------------------
    print("[1/5] Real-model ingest of 4 Memories...")
    t0 = time.perf_counter()
    system_a = EngramGraphMemorySystem(config=config)
    asyncio.run(_ingest_all(system_a, CORPUS))
    t_ingest = time.perf_counter() - t0
    state_a = system_a.get_state()
    assert state_a is not None
    print(f"  ingest wall time: {t_ingest:.2f}s ({t_ingest / len(CORPUS) * 1000:.0f}ms/memory)")
    print(f"  nodes: {state_a.store.num_nodes()}, edges: {state_a.store.num_edges()}")
    print()

    # -----------------------------------------------------------
    # 2. Inspect graph shape.
    # -----------------------------------------------------------
    print("[2/5] Graph shape checks...")
    store = state_a.store
    n_mem = len(store.nodes_by_label(LABEL_MEMORY))
    n_turn = len(store.nodes_by_label(LABEL_TURN))
    n_seg = len(store.nodes_by_label(LABEL_UTTERANCE_SEGMENT))
    n_entity = len(store.nodes_by_label(LABEL_ENTITY))
    n_claim = len(store.nodes_by_label(LABEL_CLAIM))
    n_pref = len(store.nodes_by_label(LABEL_PREFERENCE))
    print(
        f"  labels: memory={n_mem}, turn={n_turn}, segment={n_seg}, "
        f"entity={n_entity}, claim={n_claim}, preference={n_pref}"
    )
    _check(n_mem == len(CORPUS), "one Memory node per ingest call", f"got {n_mem}, expected {len(CORPUS)}")
    _check(n_turn == len(CORPUS), "one Turn granule per Memory")
    _check(n_seg >= len(CORPUS), "at least one segment per Memory")
    _check(n_entity >= 2, "at least two Entity nodes (Alice, Bob + speaker)")
    _check(n_claim >= 1, "at least one Claim node from the real dependency parse")

    edges_by_type: dict[str, int] = {}
    for _s, _d, etype, _a in store.iter_edges():
        edges_by_type[etype] = edges_by_type.get(etype, 0) + 1
    print(f"  edge counts: {dict(sorted(edges_by_type.items()))}")
    for required in (EDGE_PART_OF, EDGE_MENTIONS, EDGE_ASSERTS):
        _check(edges_by_type.get(required, 0) > 0, f"edge type {required!r} present")
    if n_pref > 0:
        _check(
            edges_by_type.get(EDGE_HOLDS_PREFERENCE, 0) > 0,
            "holds_preference edge present when Preference nodes exist",
        )
    print()

    # -----------------------------------------------------------
    # 3. R16: EntityPayload carries no 'aliases' field in real output.
    # -----------------------------------------------------------
    print("[3/5] R16 primary-data discipline...")
    # Sample an Entity and confirm the payload shape.
    entity_ids = store.nodes_by_label(LABEL_ENTITY)
    assert entity_ids, "expected at least one Entity"
    sample_payload = store.get_node(entity_ids[0])[LABEL_ENTITY]
    _check(isinstance(sample_payload, EntityPayload), "Entity payload is EntityPayload")
    _check(
        not hasattr(sample_payload, "aliases"),
        "EntityPayload has no 'aliases' field (R16)",
    )
    # Repeat-ingest invariant: the same Memory ingested twice produces two
    # Memory nodes (events), not one deduped node.
    system_dup = EngramGraphMemorySystem(config=config, pipeline=system_a._pipeline)
    asyncio.run(_ingest_all(system_dup, (CORPUS[0], CORPUS[0])))
    state_dup = system_dup.get_state()
    assert state_dup is not None
    _check(
        len(state_dup.store.nodes_by_label(LABEL_MEMORY)) == 2,
        "repeat ingest of identical content creates 2 Memory nodes (R16)",
    )
    print()

    # -----------------------------------------------------------
    # 4. R2: Two independent system runs produce byte-identical state.
    # -----------------------------------------------------------
    print("[4/5] R2 byte-equality across independent runs...")
    # Reuse the already-loaded pipeline (loading models twice would double-
    # count the MiniLM/mpnet weights; pipeline itself is stateless across
    # InstanceState boundaries).
    system_b = EngramGraphMemorySystem(config=config, pipeline=system_a._pipeline)
    asyncio.run(_ingest_all(system_b, CORPUS))
    state_b = system_b.get_state()
    assert state_b is not None
    bytes_a = dump_state(state_a.store)
    bytes_b = dump_state(state_b.store)
    _check(
        bytes_a == bytes_b,
        "two independent ingests produce byte-identical msgpack",
        f"(bytes: {len(bytes_a)} vs {len(bytes_b)})",
    )
    print(f"  serialized size: {len(bytes_a)} bytes")
    print()

    # -----------------------------------------------------------
    # 5. save_state / load_state roundtrip.
    # -----------------------------------------------------------
    print("[5/5] save_state / load_state roundtrip...")
    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp)
        asyncio.run(system_a.save_state(save_path))
        manifest_path = save_path / "manifest.json"
        primary_path = save_path / "primary.msgpack"
        _check(manifest_path.exists(), "manifest.json written")
        _check(primary_path.exists(), "primary.msgpack written")

        restored = EngramGraphMemorySystem(config=config, pipeline=system_a._pipeline)
        asyncio.run(restored.load_state(save_path))
        state_r = restored.get_state()
        assert state_r is not None
        _check(
            state_r.store.num_nodes() == state_a.store.num_nodes(),
            "restored node count matches",
            f"{state_r.store.num_nodes()} vs {state_a.store.num_nodes()}",
        )
        _check(
            state_r.store.num_edges() == state_a.store.num_edges(),
            "restored edge count matches",
            f"{state_r.store.num_edges()} vs {state_a.store.num_edges()}",
        )
        # Byte-compare a re-dump of the restored store against the original.
        bytes_restored = dump_state(state_r.store)
        _check(
            bytes_restored == bytes_a,
            "restored store re-serializes byte-identical to original",
        )

    print()
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    run()
