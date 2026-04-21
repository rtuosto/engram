"""Fingerprint audit — detect config drift across two ``MemoryConfig`` snapshots.

Used by CI and by ad-hoc comparison tooling to identify *which* field
changed when two fingerprints diverge. Pure function over a pair of
``MemoryConfig`` instances (or their pre-rendered dicts) — never reads
the graph, never touches caches.

**Two use cases.**

- **Same-commit audit.** Two configs built at different call sites
  should produce identical fingerprints. Different field values with
  the same fingerprint is an R3 violation (fingerprint doesn't cover a
  field that affects graph output); different fingerprints with
  identical fields is a seed-hash drift.
- **Cross-commit audit.** A recall-cache miss after a config change
  tells the user *which* field broke the cache. The audit lists
  diverging fields with both values so the user can decide whether to
  roll back, re-ingest, or bless the change.

**R2.** Diverging field list is sorted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from engram.config import MemoryConfig


@dataclass(frozen=True, slots=True)
class FingerprintAuditResult:
    """Result of :func:`fingerprint_audit`.

    ``ingestion_match`` — whether both configs produce the same
    ``ingestion_fingerprint``. ``recall_match`` — same for
    ``recall_fingerprint``. ``diverging_fields`` is a sorted tuple of
    ``(field_name, value_a, value_b)`` triples for every field where
    the two configs disagree (including fields that aren't in the
    fingerprint partition — those are informational, not errors).
    """

    ingestion_match: bool
    recall_match: bool
    ingestion_fingerprint_a: str
    ingestion_fingerprint_b: str
    recall_fingerprint_a: str
    recall_fingerprint_b: str
    diverging_fields: tuple[tuple[str, Any, Any], ...]


def fingerprint_audit(a: MemoryConfig, b: MemoryConfig) -> FingerprintAuditResult:
    """Compare two configs field-by-field and fingerprint-by-fingerprint.

    Every declared field is compared; the ``diverging_fields`` tuple
    lists each mismatch. The two fingerprint matches are computed
    independently so a recall-only change (e.g. a weights-hash bump)
    surfaces as ``ingestion_match=True, recall_match=False``.
    """
    dict_a = asdict(a)
    dict_b = asdict(b)

    all_fields = sorted(set(dict_a) | set(dict_b))
    diverging = tuple(
        (name, dict_a.get(name), dict_b.get(name))
        for name in all_fields
        if dict_a.get(name) != dict_b.get(name)
    )

    ingestion_fp_a = a.ingestion_fingerprint()
    ingestion_fp_b = b.ingestion_fingerprint()
    recall_fp_a = a.recall_fingerprint()
    recall_fp_b = b.recall_fingerprint()

    return FingerprintAuditResult(
        ingestion_match=ingestion_fp_a == ingestion_fp_b,
        recall_match=recall_fp_a == recall_fp_b,
        ingestion_fingerprint_a=ingestion_fp_a,
        ingestion_fingerprint_b=ingestion_fp_b,
        recall_fingerprint_a=recall_fp_a,
        recall_fingerprint_b=recall_fp_b,
        diverging_fields=diverging,
    )


__all__ = [
    "FingerprintAuditResult",
    "fingerprint_audit",
]
