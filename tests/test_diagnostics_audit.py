"""fingerprint_audit — config-drift detector."""

from __future__ import annotations

from dataclasses import replace

from engram.config import MemoryConfig
from engram.diagnostics import fingerprint_audit


def test_audit_identical_configs_match() -> None:
    a = MemoryConfig()
    b = MemoryConfig()
    result = fingerprint_audit(a, b)
    assert result.ingestion_match is True
    assert result.recall_match is True
    assert result.diverging_fields == ()
    assert result.ingestion_fingerprint_a == result.ingestion_fingerprint_b
    assert result.recall_fingerprint_a == result.recall_fingerprint_b


def test_audit_ingestion_field_change_breaks_both_fingerprints() -> None:
    a = MemoryConfig()
    b = replace(a, ngram_min_tokens=3)
    result = fingerprint_audit(a, b)
    assert result.ingestion_match is False
    # R4 transitivity: recall fingerprint must diverge too.
    assert result.recall_match is False
    diverging_names = {name for name, _, _ in result.diverging_fields}
    assert "ngram_min_tokens" in diverging_names


def test_audit_recall_only_field_leaves_ingestion_match_intact() -> None:
    a = MemoryConfig()
    b = replace(a, recall_max_depth=4)
    result = fingerprint_audit(a, b)
    # Recall-only field: ingestion fingerprint stays stable.
    assert result.ingestion_match is True
    assert result.recall_match is False
    diverging_names = {name for name, _, _ in result.diverging_fields}
    assert diverging_names == {"recall_max_depth"}


def test_audit_diverging_fields_sorted_by_name() -> None:
    a = MemoryConfig()
    b = replace(a, ngram_min_tokens=3, recall_max_depth=4, random_seed=1)
    result = fingerprint_audit(a, b)
    names = [name for name, _, _ in result.diverging_fields]
    assert names == sorted(names)


def test_audit_reports_both_values_per_divergence() -> None:
    a = MemoryConfig(ngram_min_tokens=2)
    b = replace(a, ngram_min_tokens=3)
    result = fingerprint_audit(a, b)
    triples = {name: (va, vb) for name, va, vb in result.diverging_fields}
    assert triples["ngram_min_tokens"] == (2, 3)
