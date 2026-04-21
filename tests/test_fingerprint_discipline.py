"""Verification step 4: ``MemoryConfig`` fingerprint-discipline CI gate.

These tests are the load-bearing guard against the predecessor's most expensive
failure mode — cache invalidation traps, where a config change silently reused
a cached answer from a different config because fingerprints didn't budge.

Cited rules:

- ``P6`` — every artifact has a fingerprint; no fingerprint, no cache.
- ``R3`` — ``ingestion_fingerprint`` includes every config field that affects
  node / edge production.
- ``R4`` — ``recall_fingerprint`` transitively includes
  ``ingestion_fingerprint``.

Fails CI when:

- A field is added to the dataclass but not assigned to a partition
  (``test_every_field_is_categorized``).
- Partitions overlap (``test_partitions_are_disjoint``).
- Two identical configs disagree on a fingerprint
  (``test_identical_configs_match``).
- A config diff produces a fingerprint match (``test_ingestion_*``,
  ``test_recall_field_change_does_not_alter_ingestion_fingerprint``).
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from engram.config import MemoryConfig


def test_every_field_is_categorized() -> None:
    """Every dataclass field must appear in exactly one partition set.

    This is the core R3 discipline: add a field, update a partition.
    """
    declared = {f.name for f in fields(MemoryConfig)}
    categorized = MemoryConfig._INGESTION_FIELDS | MemoryConfig._RECALL_FIELDS
    missing = declared - categorized
    extra = categorized - declared
    assert not missing, (
        f"Uncategorized MemoryConfig field(s) — add to _INGESTION_FIELDS or "
        f"_RECALL_FIELDS: {sorted(missing)}"
    )
    assert not extra, (
        f"Partition references non-existent field(s): {sorted(extra)}"
    )


def test_partitions_are_disjoint() -> None:
    overlap = MemoryConfig._INGESTION_FIELDS & MemoryConfig._RECALL_FIELDS
    assert not overlap, f"Partition not disjoint: {sorted(overlap)}"


def test_identical_configs_match() -> None:
    a = MemoryConfig()
    b = MemoryConfig()
    assert a.ingestion_fingerprint() == b.ingestion_fingerprint()
    assert a.recall_fingerprint() == b.recall_fingerprint()


def test_fingerprints_are_stable_across_runs() -> None:
    """Same config → same fingerprint twice. No wall-clock or PID leakage (R2)."""
    c = MemoryConfig()
    assert c.ingestion_fingerprint() == c.ingestion_fingerprint()
    assert c.recall_fingerprint() == c.recall_fingerprint()


def test_fingerprints_are_hex_digests() -> None:
    c = MemoryConfig()
    assert len(c.ingestion_fingerprint()) == 16
    assert len(c.recall_fingerprint()) == 16
    assert all(ch in "0123456789abcdef" for ch in c.ingestion_fingerprint())
    assert all(ch in "0123456789abcdef" for ch in c.recall_fingerprint())


@pytest.mark.parametrize("field_name", sorted(MemoryConfig._INGESTION_FIELDS))
def test_ingestion_field_change_alters_ingestion_fingerprint(field_name: str) -> None:
    """Perturbing any ingestion field must change ingestion_fingerprint (R3)."""
    base = MemoryConfig()
    perturbed = replace(base, **{field_name: _perturb(getattr(base, field_name))})
    assert base.ingestion_fingerprint() != perturbed.ingestion_fingerprint(), (
        f"Changing {field_name} left ingestion_fingerprint unchanged — R3 violation"
    )


@pytest.mark.parametrize("field_name", sorted(MemoryConfig._INGESTION_FIELDS))
def test_ingestion_field_change_propagates_to_recall_fingerprint(field_name: str) -> None:
    """R4 transitivity: any ingestion change invalidates the recall fingerprint."""
    base = MemoryConfig()
    perturbed = replace(base, **{field_name: _perturb(getattr(base, field_name))})
    assert base.recall_fingerprint() != perturbed.recall_fingerprint(), (
        f"Changing {field_name} left recall_fingerprint unchanged — R4 violation"
    )


@pytest.mark.parametrize("field_name", sorted(MemoryConfig._RECALL_FIELDS))
def test_recall_field_change_alters_recall_fingerprint(field_name: str) -> None:
    base = MemoryConfig()
    perturbed = replace(base, **{field_name: _perturb(getattr(base, field_name))})
    assert base.recall_fingerprint() != perturbed.recall_fingerprint(), (
        f"Changing {field_name} left recall_fingerprint unchanged"
    )


@pytest.mark.parametrize("field_name", sorted(MemoryConfig._RECALL_FIELDS))
def test_recall_field_change_does_not_alter_ingestion_fingerprint(field_name: str) -> None:
    """Recall-only fields MUST NOT shift the ingestion fingerprint — otherwise
    a pure recall-tweak would over-invalidate cached ingestion state."""
    base = MemoryConfig()
    perturbed = replace(base, **{field_name: _perturb(getattr(base, field_name))})
    assert base.ingestion_fingerprint() == perturbed.ingestion_fingerprint(), (
        f"Changing {field_name} shifted ingestion_fingerprint — partition bug"
    )


def test_invariant_check_fires_on_partition_bug() -> None:
    """Subclass the config with a broken partition → __post_init__ raises."""
    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass(frozen=True, slots=True)
    class BrokenConfig(MemoryConfig):
        # A new field the subclass adds but forgets to categorize.
        new_ingestion_knob: str = "x"

        # Inherit partitions unchanged — new_ingestion_knob is uncategorized.
        _INGESTION_FIELDS: ClassVar[frozenset[str]] = MemoryConfig._INGESTION_FIELDS
        _RECALL_FIELDS: ClassVar[frozenset[str]] = MemoryConfig._RECALL_FIELDS

    with pytest.raises(RuntimeError, match="not categorized"):
        BrokenConfig()


def _perturb(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + "-perturbed"
    raise TypeError(f"no perturbation for type {type(value).__name__}")
