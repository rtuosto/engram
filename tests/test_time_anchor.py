"""TimeAnchor emission + rounding tests (PR-D patch 6).

Validates stage [9] of the pipeline (``docs/design/ingestion.md §6``):

- ``round_iso_timestamp`` handles second / minute / hour / day resolutions
  and preserves the "Z" suffix when the input carries it.
- TimeAnchor nodes are content-addressed by the rounded ISO timestamp;
  two Memories at the same rounded moment share one anchor.
- Every granule (Turn + Sentence + N-gram) and relationship (Claim +
  Preference) created during the ingest gains a ``temporal_at`` edge into
  the anchor.
- Memories without a timestamp emit no anchor (fails closed).
- TimeAnchor nodes carry the ``temporal`` layer and empty payloads for
  granules are preserved under ``node_layers``.
"""

from __future__ import annotations

import asyncio

import pytest

from engram import EngramGraphMemorySystem, Memory
from engram.config import MemoryConfig
from engram.ingestion.pipeline import IngestionPipeline
from engram.ingestion.preferences import compute_centroids
from engram.ingestion.schema import (
    EDGE_TEMPORAL_AT,
    LABEL_CLAIM,
    LABEL_NGRAM,
    LABEL_PREFERENCE,
    LABEL_TIME_ANCHOR,
    LABEL_TURN,
    LABEL_UTTERANCE_SEGMENT,
    LAYER_TEMPORAL,
    PREFERENCE_POLARITIES,
    TimeAnchorPayload,
    node_id,
    round_iso_timestamp,
    time_anchor_identity,
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
# round_iso_timestamp unit tests
# ---------------------------------------------------------------------------


def test_round_to_second_strips_microseconds() -> None:
    assert round_iso_timestamp("2026-01-01T10:30:45.123456Z", "second") == (
        "2026-01-01T10:30:45Z"
    )


def test_round_to_minute_zeroes_seconds() -> None:
    assert round_iso_timestamp("2026-01-01T10:30:45Z", "minute") == (
        "2026-01-01T10:30:00Z"
    )


def test_round_to_hour_zeroes_seconds_and_minutes() -> None:
    assert round_iso_timestamp("2026-01-01T10:30:45Z", "hour") == (
        "2026-01-01T10:00:00Z"
    )


def test_round_to_day_zeroes_time_components() -> None:
    assert round_iso_timestamp("2026-01-01T10:30:45Z", "day") == (
        "2026-01-01T00:00:00Z"
    )


def test_round_preserves_tz_offset_suffix() -> None:
    assert round_iso_timestamp(
        "2026-01-01T10:30:45.999+05:30", "minute"
    ) == "2026-01-01T10:30:00+05:30"


def test_round_rejects_unknown_resolution() -> None:
    with pytest.raises(ValueError, match="unknown time_anchor_resolution"):
        round_iso_timestamp("2026-01-01T00:00:00Z", "millisecond")


def test_round_rejects_malformed_timestamp() -> None:
    with pytest.raises(ValueError, match="cannot round"):
        round_iso_timestamp("yesterday-ish", "second")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _build_memory(timestamp: str | None = "2026-01-01T10:30:45Z") -> tuple[Memory, dict]:
    text = "Alice loves hiking."
    root = make_token("loves", idx=6, pos="VERB", dep="ROOT", lemma="love", tense=("Pres",))
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
        noun_chunks=[alice_chunk, hiking_chunk],
    )
    memory = Memory(content=text, timestamp=timestamp, speaker="user")
    return memory, {text: doc}


def _make_system(config: MemoryConfig | None = None) -> EngramGraphMemorySystem:
    config = config or MemoryConfig()
    _memory, docs = _build_memory()
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


def test_emits_time_anchor_with_temporal_layer() -> None:
    system = _make_system()
    memory, _ = _build_memory()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None
    anchors = state.store.nodes_by_label(LABEL_TIME_ANCHOR)
    assert len(anchors) == 1
    anchor_id = anchors[0]
    assert state.store.node_layers(anchor_id) == frozenset({LAYER_TEMPORAL})
    payload = state.store.get_node(anchor_id)[LABEL_TIME_ANCHOR]
    assert isinstance(payload, TimeAnchorPayload)
    assert payload.iso_timestamp == "2026-01-01T10:30:45Z"


def test_anchor_is_content_addressed() -> None:
    system = _make_system()
    memory, _ = _build_memory()
    expected = node_id(time_anchor_identity("2026-01-01T10:30:45Z"))
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None
    assert state.store.has_node(expected)


def test_repeat_ingest_at_same_moment_shares_one_anchor() -> None:
    """R16: TimeAnchors are content-addressed by rounded ISO timestamp.
    Two Memories at the same moment produce one anchor with multiple
    inbound ``temporal_at`` edges (reinforcement via edges, not mutation).
    """
    system = _make_system()
    memory, _ = _build_memory()
    asyncio.run(system.ingest(memory))
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None
    assert len(state.store.nodes_by_label(LABEL_TIME_ANCHOR)) == 1
    anchor_id = state.store.nodes_by_label(LABEL_TIME_ANCHOR)[0]
    incoming = state.store.in_edges(anchor_id, edge_type=EDGE_TEMPORAL_AT)
    assert len(incoming) >= 2  # at least one edge per ingest's turn


def test_different_rounded_moments_produce_distinct_anchors() -> None:
    system = _make_system()
    memory_a, _ = _build_memory(timestamp="2026-01-01T10:30:45Z")
    memory_b, _ = _build_memory(timestamp="2026-01-01T10:30:46Z")
    asyncio.run(system.ingest(memory_a))
    asyncio.run(system.ingest(memory_b))
    state = system.get_state()
    assert state is not None
    assert len(state.store.nodes_by_label(LABEL_TIME_ANCHOR)) == 2


def test_granules_and_relationships_get_temporal_at_edges() -> None:
    system = _make_system()
    memory, _ = _build_memory()
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None

    anchors = state.store.nodes_by_label(LABEL_TIME_ANCHOR)
    assert len(anchors) == 1
    anchor_id = anchors[0]

    # Every granule + relationship should have outgoing temporal_at into the
    # anchor. (Memory nodes and Entity nodes do *not* — the anchor hangs off
    # observations, not identity primitives.)
    for label in (
        LABEL_TURN,
        LABEL_UTTERANCE_SEGMENT,
        LABEL_NGRAM,
        LABEL_CLAIM,
        LABEL_PREFERENCE,
    ):
        for nid in state.store.nodes_by_label(label):
            out = state.store.out_edges(nid, edge_type=EDGE_TEMPORAL_AT)
            assert any(dst == anchor_id for dst, _ in out), (
                f"{label} node {nid!r} missing temporal_at edge to anchor"
            )


def test_memory_without_timestamp_emits_no_anchor() -> None:
    system = _make_system()
    memory, _ = _build_memory(timestamp=None)
    asyncio.run(system.ingest(memory))
    state = system.get_state()
    assert state is not None
    assert state.store.nodes_by_label(LABEL_TIME_ANCHOR) == []


def test_day_resolution_collapses_intraday_observations() -> None:
    """Config.time_anchor_resolution=day sends same-day Memories to one anchor."""
    config = MemoryConfig(time_anchor_resolution="day")
    system = _make_system(config=config)
    memory_a, _ = _build_memory(timestamp="2026-01-01T10:30:45Z")
    memory_b, _ = _build_memory(timestamp="2026-01-01T23:59:59Z")
    asyncio.run(system.ingest(memory_a))
    asyncio.run(system.ingest(memory_b))
    state = system.get_state()
    assert state is not None
    anchors = state.store.nodes_by_label(LABEL_TIME_ANCHOR)
    assert len(anchors) == 1
    payload = state.store.get_node(anchors[0])[LABEL_TIME_ANCHOR]
    assert isinstance(payload, TimeAnchorPayload)
    assert payload.iso_timestamp == "2026-01-01T00:00:00Z"
