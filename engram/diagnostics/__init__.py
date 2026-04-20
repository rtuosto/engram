"""engram.diagnostics — per-run failure classification + coverage + audits.

**Responsibility.** Given a run's results, classify each failure and surface
actionable patterns.

**Public verbs.** ``classify_failures``, ``bucket_breakdown``, ``needle_overlap``,
``extraction_coverage``, ``fingerprint_audit``.

**Owns.** Failure classification (``R15`` enum:
``extraction_miss | graph_gap | retrieval_miss | partial_retrieval | prompt_miss | answerer_miss``),
gold-term overlap, extraction-coverage reports, fingerprint-discipline audits,
commit-over-commit regression reports.

**Does not touch.** The memory system's runtime path. Read-only. Never writes to
caches. Never mutates run artifacts.

**Stability guarantee.** Classifications are versioned. Comparing classifications
across versions requires a migration note.

Diagnostics runs first on every failure (``M8``); classification is logged with
the commit. Layer-decision-tree discipline (``M5``) comes from here: fix the
deepest broken layer first (``P9``), never patch an upper layer to compensate
for a lower one.
"""
