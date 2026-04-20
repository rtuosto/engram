"""engram — graph-based memory system for LLM agents.

Four modules with strict boundaries (see ``docs/DESIGN-MANIFESTO.md §6``):

- :mod:`engram.ingestion` — sessions → graph + fingerprint.
- :mod:`engram.recall` — question → subgraph + context + one answerer call.
- :mod:`engram.benchmarking` — dataset orchestration + judging + scoring.
- :mod:`engram.diagnostics` — failure classification + coverage reports.

Only :class:`engram.MemorySystem` is the public surface benchmarks touch.
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
