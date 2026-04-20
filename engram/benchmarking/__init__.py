"""engram.benchmarking — dataset orchestration + judging + scoring + replicates.

**Responsibility.** Run a memory system against LongMemEval-s (primary) and
LOCOMO (validation); score; persist; compare.

**Public verbs.** ``run``, ``resume``, ``rejudge``, ``summarize``, ``compare``,
``baseline``, ``ablation``.

**Owns.** Dataset loaders, judge abstraction + prompts, run directory layout,
scorecard rendering, cache layout, replicate orchestration.

**Does not touch.** The memory system's internals. This module reads only the
``MemorySystem`` protocol surface; it never inspects the graph.

**Stability guarantee.** Must be bit-stable across memory-system rewrites
(``P8``). Judge prompts, dataset loaders, and scoring rules are sacrosanct —
changes require an explicit, documented re-baseline.

Cache-tainted ablations are invalid (``M3``): the runner refuses to publish
results when the answer cache predates the relevant fingerprint. Retrieval-only
claims do not ship without a full-benchmark run (``M4``).
"""
