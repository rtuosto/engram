"""The ``MemorySystem`` protocol — the only surface external callers touch.

``R1``: Single ``MemorySystem`` protocol per experiment family. Only
``ingest_session``, ``finalize_conversation``, ``answer_question``, ``reset``,
``save_state``, ``load_state`` are public. No sideband hooks that let one
experiment peek into another's internals.

``P8``: The external benchmark reads only this surface; it never inspects the
interior graph. Rewrites of the memory system must keep this contract
byte-compatible or bump a major version — the benchmark is the measurement
instrument and must remain stable.

``R4``: ``answer_fingerprint`` transitively includes ``ingestion_fingerprint``.
Fingerprints live on :class:`engram.config.MemoryConfig` (not on the
``MemorySystem`` itself) — the protocol below exposes identity/version metadata
but leaves cache key composition to the external benchmark harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from engram.models import AnswerResult, Session


@runtime_checkable
class MemorySystem(Protocol):
    """Agent-agnostic memory system contract.

    Implementations may be graph-backed, embedding-backed, or anything else —
    only the public verbs below are visible to external callers (the benchmark
    harness; this repo's :mod:`engram.diagnostics`).

    **Identity.** ``memory_system_id`` is a stable string identifying the
    implementation family (``"engram_graph"``, ``"null"``, …). Cache keys
    include it so unrelated systems don't collide in shared caches.

    **Versioning (``R12``).** ``memory_version`` is a monotonically increasing
    string that MUST be bumped when on-disk state format or retrieval/answer
    semantics change. Load paths handle missing / old / future versions with
    a clear error, not a silent crash. (This is distinct from
    ``engram.__version__``, which is the Python package semver.)
    """

    memory_system_id: str
    memory_version: str

    async def ingest_session(self, session: Session, conversation_id: str) -> None:
        """Append a single session to the per-conversation graph.

        Order-sensitive: sessions for a given ``conversation_id`` must be
        ingested in ascending ``session_index``. Calling this out of order is
        undefined behavior.

        Must be deterministic (``R2``): same ``(config, session, conversation_id)``
        → identical internal state mutation.

        Must not call an LLM on the default path (``R5``). LLM-based extraction
        enhancements are opt-in and flag-gated.
        """
        ...

    async def finalize_conversation(self, conversation_id: str) -> None:
        """Run end-of-conversation passes for a conversation.

        Called once after the last ``ingest_session`` for a given conversation.
        Episode detection, corpus-signal derivation, cross-session edge
        construction, and any other end-of-stream work happens here.

        Must be deterministic (``R2``) — same ingested history → same finalized
        graph. After this returns, ``answer_question(question, conversation_id)``
        must be usable against the finalized graph.
        """
        ...

    async def answer_question(self, question: str, conversation_id: str) -> AnswerResult:
        """Answer ``question`` using the graph for ``conversation_id``.

        Executes the full Recall pipeline (``docs/DESIGN-MANIFESTO.md §3``):
        intent classification → seed → expand → rank → assemble → answer.

        Exactly **one** LLM call happens here — the answerer (``K5``). Any
        additional LLM call at recall time requires an approved exception.

        The returned ``AnswerResult`` carries the final answer, the assembled
        context, the retrieved subgraph projection, and per-stage timings.
        """
        ...

    async def reset(self) -> None:
        """Clear all in-memory state. Equivalent to reconstructing the object.

        After ``reset`` a fresh ingestion must produce identical state to a
        process that started clean (``R2`` — determinism). Does not touch
        persisted state files — ``load_state`` / ``save_state`` handle that.
        """
        ...

    async def save_state(self, path: Path) -> None:
        """Persist the full memory system state to ``path``.

        On-disk format MUST be versioned (``R12``). Implementations should
        write a version marker alongside the state payload so ``load_state``
        can detect format drift.

        Must be deterministic (``R2``): same internal state → byte-identical
        output file modulo timestamps, which MUST NOT appear in the payload.
        """
        ...

    async def load_state(self, path: Path) -> None:
        """Restore state previously written by ``save_state``.

        Load paths MUST handle missing / old / future versions with a clear
        error (``R12``). No silent migration; no silent crash on format drift.
        """
        ...
