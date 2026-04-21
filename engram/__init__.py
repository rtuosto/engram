"""engram — graph-based memory system for LLM agents.

Three modules with strict boundaries (see ``docs/DESIGN-MANIFESTO.md §6``):

- :mod:`engram.ingestion` — sessions → graph + fingerprint.
- :mod:`engram.recall` — question → subgraph + context + one answerer call.
- :mod:`engram.diagnostics` — failure classification + coverage reports.

Benchmarking lives in a separate repository (``agent-memory-benchmark``) and
consumes engram through the :class:`engram.MemorySystem` protocol — the only
public surface external callers touch.
"""

from engram.memory_system import MemorySystem
from engram.models import AnswerResult, RetrievedNode, Session, Turn

__all__ = [
    "AnswerResult",
    "MemorySystem",
    "RetrievedNode",
    "Session",
    "Turn",
]

__version__ = "0.0.1"
