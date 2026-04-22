"""engram.diagnostics — per-run failure classification + coverage + audits.

**Responsibility.** Given a run's results, classify each failure and surface
actionable patterns.

**Public verbs.** :func:`classify_failures`, :func:`bucket_breakdown`,
:func:`needle_overlap`, :func:`extraction_coverage`, :func:`fingerprint_audit`.

**Owns.** Failure classification (R15 enum:
``extraction_miss | graph_gap | retrieval_miss | partial_retrieval |
output_miss | agent_miss``; plus a neutral ``correct`` bucket so
aggregation keeps complete denominators), gold-term overlap,
extraction-coverage reports, fingerprint-discipline audits,
commit-over-commit regression reports.

**Does not touch.** The memory system's runtime path. Read-only. Never
writes to caches. Never mutates run artifacts.

**Stability guarantee.** Classifications are versioned. Comparing
classifications across versions requires a migration note.

Diagnostics runs first on every failure (M8); classification is logged
with the commit. Layer-decision-tree discipline (M5) comes from here:
fix the deepest broken layer first (P9), never patch an upper layer to
compensate for a lower one.
"""

from engram.diagnostics.audit import (
    FingerprintAuditResult,
    fingerprint_audit,
)
from engram.diagnostics.coverage import (
    CoverageReport,
    extraction_coverage,
)
from engram.diagnostics.failures import (
    DEFAULT_PARTIAL_THRESHOLD,
    BucketReport,
    FailureCase,
    FailureInput,
    FailureKind,
    bucket_breakdown,
    classify_failures,
)
from engram.diagnostics.overlap import (
    Overlap,
    extract_key_terms,
    needle_overlap,
)
from engram.diagnostics.recall_trace import (
    AssembleStageTrace,
    DroppedNode,
    ExpandStageTrace,
    ExpandStep,
    IntentStageTrace,
    RecallTrace,
    ScoredGranule,
    ScoreStageTrace,
    SeedEntry,
    SeedStageTrace,
    traced_recall,
)
from engram.diagnostics.recall_trace_html import render_html as render_trace_html

__all__ = [
    "DEFAULT_PARTIAL_THRESHOLD",
    "AssembleStageTrace",
    "BucketReport",
    "CoverageReport",
    "DroppedNode",
    "ExpandStageTrace",
    "ExpandStep",
    "FailureCase",
    "FailureInput",
    "FailureKind",
    "FingerprintAuditResult",
    "IntentStageTrace",
    "Overlap",
    "RecallTrace",
    "ScoreStageTrace",
    "ScoredGranule",
    "SeedEntry",
    "SeedStageTrace",
    "bucket_breakdown",
    "classify_failures",
    "extract_key_terms",
    "extraction_coverage",
    "fingerprint_audit",
    "needle_overlap",
    "render_trace_html",
    "traced_recall",
]
