"""Recall-time deterministic context — :class:`RecallContext`.

The agent supplies optional system info (current wall clock, timezone) via
:meth:`MemorySystem.recall`'s keyword args; the pipeline wraps them in a
frozen context object so downstream stages have a single typed handle to
pass around.

**R8 / P4.** ``now`` is used to resolve relative time expressions at recall
planning (never in the output). If ``now`` is ``None`` the pipeline does
not invent one — relative expressions in the query remain unresolved and
temporal-fact rendering falls back to absolute stamps only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecallContext:
    """Deterministic system inputs the agent optionally supplies.

    Kept frozen for the same ``R2`` reasons as every other public dataclass
    — two identical contexts must be byte-equal for the recall fingerprint
    to match.
    """

    now: str | None = None
    timezone: str | None = None
    max_passages: int | None = None
    intent_hint: str | None = None


__all__ = ["RecallContext"]
