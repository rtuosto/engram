"""Shared data contracts across module boundaries.

This module holds the handful of types every other module depends on:

- :class:`Turn`, :class:`Session` — input shape for :meth:`MemorySystem.ingest_session`.
- :class:`RetrievedNode` — graph-agnostic projection of what Recall returns.
- :class:`AnswerResult` — what :meth:`MemorySystem.answer_question` returns.

All types are frozen dataclasses (``R2`` — ingestion must be deterministic given
the same inputs; mutable shared state is a determinism hazard).

These types are deliberately narrow. The interior graph representation inside
:mod:`engram.ingestion` and :mod:`engram.recall` may be richer, but only the
fields defined here cross module boundaries — see ``docs/DESIGN-MANIFESTO.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Turn:
    """One utterance by one speaker inside a Session.

    ``speaker`` is conventionally ``"user"`` or ``"assistant"`` but the
    ingestion layer must tolerate other values (dataset-specific).
    ``session_index`` and ``turn_index`` are 1-based positional anchors.
    """

    speaker: str
    text: str
    session_index: int
    turn_index: int
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """An ordered sequence of Turns sharing a timestamp and conversation slot.

    Sessions are the unit of ingestion (``MemorySystem.ingest_session``).
    ``timestamp`` is an absolute ISO-8601 string when known; ``None`` when the
    dataset only provides a relative ordering.
    """

    session_index: int
    turns: tuple[Turn, ...]
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedNode:
    """A graph node exposed to Benchmarking / Diagnostics via Recall.

    The interior graph may have richer types (Turn, Utterance Segment, Claim,
    Preference, Event, Episode, …). This class is the *projection* that crosses
    module boundaries — it carries just enough metadata for diagnostic metrics
    (``session_density``, ``needle_recall@k``, ``completeness``) to work without
    peeking at the graph.

    ``node_type`` should be a stable string drawn from the node-type taxonomy
    in ``docs/DESIGN-MANIFESTO.md §3`` (``turn``, ``utterance_segment``,
    ``entity``, ``claim``, ``preference``, ``event``, ``episode``, ``session``).
    """

    node_id: str
    node_type: str
    text: str
    conversation_id: str
    session_index: int | None = None
    turn_index: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """The result of :meth:`MemorySystem.answer_question`.

    ``context`` is the final assembled string handed to the answerer — the
    single source of truth for "what did the LLM see". Required for diagnostic
    classification (``R15``) and prompt-miss detection (``M5``).

    ``retrieved_nodes`` is the rank-ordered subgraph projection. Benchmarking
    uses it for retrieval-side metrics; Diagnostics uses it for extraction- and
    graph-gap classification.

    ``retrieval_time_ms`` and ``answer_time_ms`` sum to ``total_time_ms`` plus
    any overhead; all three are reported (``K4``).
    """

    answer: str
    context: str
    retrieved_nodes: tuple[RetrievedNode, ...]
    retrieval_time_ms: float
    answer_time_ms: float
    total_time_ms: float
