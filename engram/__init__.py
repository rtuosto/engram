"""engram — graph-based memory system for LLM agents.

Three modules with strict boundaries (see ``docs/DESIGN-MANIFESTO.md §6``):

- :mod:`engram.ingestion` — Memories → graph + fingerprint.
- :mod:`engram.recall` — query → :class:`RecallResult` (no LLM calls).
- :mod:`engram.diagnostics` — failure classification + coverage reports.

Benchmarking + the answerer agent live in a separate repository
(``agent-memory-benchmark``) and consume engram through the
:class:`engram.MemorySystem` protocol — the only public surface external
callers touch.
"""

from engram.engram_memory_system import EngramGraphMemorySystem
from engram.memory_system import MemorySystem
from engram.models import (
    Memory,
    RecallFact,
    RecallPassage,
    RecallResult,
    RetrievedNode,
    Session,
    Turn,
)

__all__ = [
    "EngramGraphMemorySystem",
    "Memory",
    "MemorySystem",
    "RecallFact",
    "RecallPassage",
    "RecallResult",
    "RetrievedNode",
    "Session",
    "Turn",
]

__version__ = "0.0.1"
