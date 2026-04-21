"""The ``MemorySystem`` protocol — the only surface external callers touch.

``R1``: Single ``MemorySystem`` protocol per experiment family. The public
verbs are :meth:`ingest`, :meth:`recall`, :meth:`reset`, :meth:`save_state`,
:meth:`load_state`. **No conversation / session IDs** — engram instances
hold one memory. Isolation is the caller's responsibility (instantiate
separately, or call :meth:`reset`).

``P13``: Engram is a memory tool; the outside agent calls :meth:`recall`,
reasons over the returned :class:`RecallResult`, and composes an answer.
Engram never produces an answer and never calls an LLM (``R5``, ``R13``).

``P8``: The external benchmark reads only this surface; it never inspects
the interior graph. Rewrites must keep this contract byte-compatible or
bump :attr:`memory_version` — the benchmark is the measurement instrument
and must remain stable.

``R4``: ``recall_fingerprint`` transitively includes ``ingestion_fingerprint``.
Fingerprints live on :class:`engram.config.MemoryConfig` (not on the
``MemorySystem`` itself) — the protocol below exposes identity / version
metadata but leaves cache key composition to the external benchmark harness.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from engram.models import Memory, RecallResult


@runtime_checkable
class MemorySystem(Protocol):
    """Agent-agnostic memory system contract.

    Implementations may be graph-backed, embedding-backed, or anything else —
    only the public verbs below are visible to external callers (the
    benchmark harness; this repo's :mod:`engram.diagnostics`).

    **Identity.** ``memory_system_id`` is a stable string identifying the
    implementation family (``"engram_graph"``, ``"null"``, …). Cache keys
    include it so unrelated systems don't collide in shared caches.

    **Versioning (``R12``).** ``memory_version`` is a monotonically
    increasing string that MUST be bumped when on-disk state format or
    retrieval semantics change. Load paths handle missing / old / future
    versions with a clear error, not a silent crash. (Distinct from
    ``engram.__version__``, which is the Python package semver.)
    """

    memory_system_id: str
    memory_version: str

    async def ingest(self, memory: Memory) -> None:
        """Append a single :class:`Memory` observation to the graph.

        One call = one permanent observation event. Memory nodes are never
        deduplicated (``R16``). Content primitives (Entity / N-gram / Claim
        / Preference) *are* content-addressed — repeat content produces new
        *observation edges* on the same primitive, not duplicate primitives.
        This is how counting works: edge enumeration.

        Must be deterministic (``R2``): same ``(config, Memory sequence)`` →
        identical internal state. Must not call an LLM (``R5``).
        """
        ...

    async def ingest_many(self, memories: Iterable[Memory]) -> None:
        """Append a sequence of :class:`Memory` observations in order.

        Semantically equivalent to ``for m in memories: await ingest(m)`` —
        same ``R2`` determinism, same ``R16`` append-only ordering. The
        default implementation does exactly that; implementations MAY
        override with a batched variant that pools model calls across the
        batch dimension. Any override MUST produce a graph that is
        structurally identical to the sequential path (same node IDs,
        same edge tuples, same payloads) — numeric drift in edge weights
        from batched transformer inference is tolerated within the
        structural-fingerprint guard (``docs/ARCHITECTURE.md`` Key Decisions
        2026-04-21; ``scripts/check_fingerprint_equivalence.py``).

        Must not call an LLM (``R5``, ``R13``).
        """
        for memory in memories:
            await self.ingest(memory)

    async def recall(
        self,
        query: str,
        *,
        now: str | None = None,
        timezone: str | None = None,
        max_passages: int | None = None,
        intent_hint: str | None = None,
    ) -> RecallResult:
        """Return a :class:`RecallResult` for ``query`` (``docs/design/recall.md``).

        Engram does **not** produce an answer (``R9``, ``R13``) — it returns
        structured context (ranked passages + pre-computed facts) that the
        outside agent reasons over.

        Engram makes **zero** LLM calls at recall time (``R5``, ``K5``). The
        pipeline is intent classification → seeding → typed-edge BFS
        expansion → scoring → assembly.

        ``now`` / ``timezone`` give the agent-supplied recall clock used for
        temporal arithmetic (``R8``); engram resolves relative time
        references at recall planning, never in the output. ``max_passages``
        caps the returned passage count. ``intent_hint`` lets the agent
        bypass intent classification for high-confidence calls.
        """
        ...

    async def reset(self) -> None:
        """Clear all in-memory state.

        After ``reset`` a fresh ingest sequence must produce identical state
        to a process that started clean (``R2``). Does not touch persisted
        state files — :meth:`load_state` / :meth:`save_state` handle that.
        """
        ...

    async def save_state(self, path: Path) -> None:
        """Persist full memory-system state to ``path``.

        On-disk format MUST be versioned (``R12``). Implementations write a
        version marker alongside the state payload so :meth:`load_state`
        can detect format drift.

        Must be deterministic (``R2``): same internal state → byte-identical
        output modulo timestamps, which MUST NOT appear in the payload.
        """
        ...

    async def load_state(self, path: Path) -> None:
        """Restore state previously written by :meth:`save_state`.

        Load paths MUST handle missing / old / future versions with a clear
        error (``R12``). No silent migration; no silent crash on format drift.
        """
        ...
