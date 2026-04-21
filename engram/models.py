"""Shared data contracts across module boundaries.

Engram is a memory tool exposed to an outside agent (``R13``, ``P13``). The
agent calls :meth:`MemorySystem.ingest` to write and :meth:`MemorySystem.recall`
to read. These types are the only shapes that cross module boundaries — see
``docs/DESIGN-MANIFESTO.md §6``.

- :class:`Memory` — the ingest input. Whatever the agent decides to record
  (a conversation turn, a file excerpt, a note). One ``ingest(memory)`` call =
  one permanent observation event, never deduplicated (``R16``).
- :class:`RecallResult` / :class:`RecallPassage` / :class:`RecallFact` — the
  structured output of :meth:`MemorySystem.recall`. Engram returns context
  for the agent to reason over, not an answer (``R9``, ``R13``).
- :class:`Session` / :class:`Turn` — retained as **optional** helper types.
  External callers (the benchmark) may use them to build a stream of
  Memories; the protocol itself no longer requires them.

All types are frozen dataclasses (``R2`` — ingestion must be deterministic
given the same inputs; mutable shared state is a determinism hazard).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Memory:
    """One atomic observation event handed to :meth:`MemorySystem.ingest`.

    ``content`` is the text payload (required). Everything else is
    agent-supplied metadata; engram is agnostic to what the agent decides
    constitutes a single memory (``docs/design/ingestion.md §1``).

    ``timestamp`` is an ISO-8601 absolute string when the agent knows it;
    ``None`` when only relative ordering is available. ``speaker`` is a
    freeform label (conventionally ``"user"`` / ``"assistant"`` but anything
    is accepted). ``source`` is a short string describing provenance
    (``"conversation_turn"``, ``"file:notes.md"``, …).

    ``metadata`` is a sorted tuple of ``(key, value)`` pairs rather than a
    dict so the payload is hashable and R2-deterministic. Callers may pass a
    ``dict`` to :meth:`MemorySystem.ingest` — the implementation normalizes
    it to sorted tuples before the Memory node is content-addressed.
    """

    content: str
    timestamp: str | None = None
    speaker: str | None = None
    source: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Turn:
    """One utterance by one speaker inside a Session (helper type).

    Retained for external callers that build a stream of Memories from a
    conversational dataset. The protocol no longer requires it.
    """

    speaker: str
    text: str
    session_index: int
    turn_index: int
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """An ordered sequence of Turns (helper type).

    Retained for external callers. The protocol no longer requires it —
    each Turn typically maps to one :class:`Memory`.
    """

    session_index: int
    turns: tuple[Turn, ...]
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedNode:
    """A graph node projection exposed to external callers via Recall.

    The interior graph may have richer types (Turn, Sentence, N-gram,
    Entity, Claim, Preference, …). This class is the *projection* that
    crosses module boundaries — it carries just enough metadata for
    diagnostic metrics to work without peeking at the graph. See
    ``docs/DESIGN-MANIFESTO.md §3`` for the node taxonomy.
    """

    node_id: str
    node_type: str
    text: str
    memory_id: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecallPassage:
    """A ranked snippet of remembered content returned by :meth:`recall`.

    ``text`` is the granule content; ``granularity`` is ``turn`` |
    ``sentence`` | ``ngram``. ``score`` is the walk-score from expansion.
    ``node_id`` is the granule's graph node id. ``source_memory_id`` anchors
    provenance. ``source_memory_index`` is the monotonic ingest order of
    that Memory. ``supporting_edges`` is a tuple of human-readable strings
    describing Claims / Preferences this granule asserts (see
    ``docs/design/recall.md §8``).
    """

    text: str
    granularity: str
    score: float
    node_id: str
    source_memory_id: str | None = None
    source_memory_index: int | None = None
    timestamp: str | None = None
    speaker: str | None = None
    supporting_edges: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallFact:
    """A pre-computed answer the agent can cite directly.

    Engram resolves arithmetic, aggregations, and current-truth lookups
    (``P4``, ``R8``) so the agent never has to compute them.

    ``kind`` is one of ``current_preference`` | ``reinforcement`` |
    ``change_event`` | ``co_occurrence``. ``value`` is the rendered literal
    (``"dislikes (as of 2026-03-05; previously liked, 47 reinforcements)"``).
    ``supporting_memory_ids`` is provenance — the Memories backing the fact.
    """

    kind: str
    subject: str
    value: str
    predicate: str | None = None
    object: str | None = None
    timestamp: str | None = None
    supporting_memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Structured output of :meth:`MemorySystem.recall`.

    ``passages`` is the ranked list of retrieved snippets. ``facts`` are
    pre-computed entries from derived indexes. ``intent`` is the detected
    query intent; ``intent_confidence`` is the discrimination margin (~0
    means weakly-classified and the pipeline fell back to ``single_fact``).
    ``timing_ms`` is a stage-level breakdown (``classify`` / ``seed`` /
    ``expand`` / ``score`` / ``assemble`` / ``total``).
    ``recall_fingerprint`` is the audit key the benchmark caches against.

    Recall does not produce an answer (``R9``, ``R13``). The agent sees this
    structure and composes the final response.
    """

    passages: tuple[RecallPassage, ...]
    facts: tuple[RecallFact, ...] = ()
    intent: str | None = None
    intent_confidence: float = 0.0
    timing_ms: tuple[tuple[str, float], ...] = ()
    recall_fingerprint: str | None = None


__all__ = [
    "Memory",
    "RecallFact",
    "RecallPassage",
    "RecallResult",
    "RetrievedNode",
    "Session",
    "Turn",
]
