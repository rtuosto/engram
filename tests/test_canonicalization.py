"""Entity canonicalization determinism + merge semantics (§6).

R2-critical: tie-breaking must be deterministic under any insertion order.
"""

from __future__ import annotations

from engram.ingestion.extractors.canonicalization import (
    EntityRegistry,
    canonicalize,
    normalize,
)
from engram.ingestion.extractors.ner import EntityMention


def _mention(surface: str, entity_type: str = "PERSON", span: tuple[int, int] = (0, 0)) -> EntityMention:
    return EntityMention(
        surface_form=surface,
        entity_type=entity_type,
        char_span=span,
        turn_id="t",
    )


def test_normalize_nfkc_casefold() -> None:
    # NFKC decomposes the ligature then casefolds.
    assert normalize("ﬁle") == "file"
    assert normalize("ALICE") == "alice"
    assert normalize("  Bob  ") == "bob"


def test_exact_match_reuses_existing_node() -> None:
    registry = EntityRegistry()
    id_a, payload_a, new_a = canonicalize(_mention("Alice"), registry, match_threshold=0.85)
    id_b, payload_b, new_b = canonicalize(_mention("alice"), registry, match_threshold=0.85)
    assert new_a is True
    assert new_b is False
    assert id_a == id_b
    assert set(payload_b.aliases) == {"Alice", "alice"}


def test_fuzzy_match_above_threshold_merges() -> None:
    registry = EntityRegistry()
    canonicalize(_mention("Dr. Alice Smith"), registry, match_threshold=0.7)
    id_b, payload_b, new_b = canonicalize(_mention("Alice Smith"), registry, match_threshold=0.7)
    assert new_b is False
    assert "Alice Smith" in payload_b.aliases


def test_type_gate_prevents_cross_type_merge() -> None:
    registry = EntityRegistry()
    id_company, _, _ = canonicalize(_mention("Apple", entity_type="ORG"), registry, match_threshold=0.5)
    id_fruit, _, is_new = canonicalize(_mention("apple", entity_type="FOOD"), registry, match_threshold=0.5)
    assert id_company != id_fruit
    assert is_new is True


def test_threshold_below_creates_new_entity() -> None:
    registry = EntityRegistry()
    canonicalize(_mention("Alice"), registry, match_threshold=0.85)
    id_b, _, is_new = canonicalize(_mention("Bob"), registry, match_threshold=0.85)
    assert is_new is True


def test_determinism_per_sequence() -> None:
    """Same sequence, different runs → byte-identical output (R2)."""
    mentions = [
        _mention("Alice", span=(0, 5)),
        _mention("alice", span=(10, 15)),
        _mention("Dr. Alice Smith", span=(20, 35)),
        _mention("Bob", span=(40, 43)),
    ]

    def run() -> list[tuple[str, tuple[str, ...]]]:
        reg = EntityRegistry()
        out: list[tuple[str, tuple[str, ...]]] = []
        for m in mentions:
            eid, payload, _ = canonicalize(m, reg, match_threshold=0.85)
            out.append((eid, payload.aliases))
        return out

    assert run() == run()


def test_insertion_order_invariant_when_no_merges() -> None:
    """Disjoint entities produce the same IDs regardless of arrival order.

    Content addressing guarantees this: the ID is a hash of the canonical
    form and type, not of what existed at arrival time.
    """
    mentions = [
        _mention("Alice", span=(0, 5)),
        _mention("Bob", span=(10, 13)),
        _mention("Charlie", span=(20, 27)),
    ]

    def run(order: list[int]) -> set[str]:
        reg = EntityRegistry()
        seen: set[str] = set()
        for idx in order:
            eid, _, _ = canonicalize(mentions[idx], reg, match_threshold=0.99)
            seen.add(eid)
        return seen

    base = run([0, 1, 2])
    for order in ([2, 1, 0], [1, 0, 2], [0, 2, 1]):
        assert run(order) == base


def test_aliases_sorted_deterministically() -> None:
    """Aliases tuple is always sorted; insertion-order-invariant."""
    reg_forward = EntityRegistry()
    reg_reverse = EntityRegistry()
    for surface in ["Zed", "alice", "Alice"]:
        canonicalize(_mention(surface), reg_forward, match_threshold=0.0)
    for surface in ["alice", "Zed", "Alice"]:
        canonicalize(_mention(surface), reg_reverse, match_threshold=0.0)

    # With threshold=0 everything merges into the first entity of the type.
    fw_alias = next(iter(reg_forward.by_type_and_form.values()))[1]
    rv_alias = next(iter(reg_reverse.by_type_and_form.values()))[1]
    assert fw_alias == rv_alias
