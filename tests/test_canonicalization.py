"""Entity canonicalization determinism + merge semantics (§10).

R2-critical: tie-breaking must be deterministic under any insertion order.

R16: :class:`EntityPayload` has no ``aliases`` field. The canonicalizer
resolves mention surfaces to Entity node IDs; alias collection is a derived
index (PR-D) rebuilt from inbound ``mentions`` edges.
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
    id_a, _payload_a, new_a = canonicalize(_mention("Alice"), registry, match_threshold=0.85)
    id_b, _payload_b, new_b = canonicalize(_mention("alice"), registry, match_threshold=0.85)
    assert new_a is True
    assert new_b is False
    assert id_a == id_b


def test_fuzzy_match_above_threshold_merges() -> None:
    registry = EntityRegistry()
    id_a, _, _ = canonicalize(_mention("Dr. Alice Smith"), registry, match_threshold=0.7)
    id_b, _, new_b = canonicalize(_mention("Alice Smith"), registry, match_threshold=0.7)
    assert new_b is False
    assert id_a == id_b


def test_type_gate_prevents_cross_type_merge() -> None:
    registry = EntityRegistry()
    id_company, _, _ = canonicalize(_mention("Apple", entity_type="ORG"), registry, match_threshold=0.5)
    id_fruit, _, is_new = canonicalize(_mention("apple", entity_type="FOOD"), registry, match_threshold=0.5)
    assert id_company != id_fruit
    assert is_new is True


def test_threshold_below_creates_new_entity() -> None:
    registry = EntityRegistry()
    canonicalize(_mention("Alice"), registry, match_threshold=0.85)
    _id_b, _, is_new = canonicalize(_mention("Bob"), registry, match_threshold=0.85)
    assert is_new is True


def test_determinism_per_sequence() -> None:
    """Same sequence, different runs → byte-identical output (R2)."""
    mentions = [
        _mention("Alice", span=(0, 5)),
        _mention("alice", span=(10, 15)),
        _mention("Dr. Alice Smith", span=(20, 35)),
        _mention("Bob", span=(40, 43)),
    ]

    def run() -> list[str]:
        reg = EntityRegistry()
        out: list[str] = []
        for m in mentions:
            eid, _payload, _ = canonicalize(m, reg, match_threshold=0.85)
            out.append(eid)
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


def test_payload_has_no_aliases_field() -> None:
    """R16: aliases live in the derived index, not on the Entity payload."""
    from dataclasses import fields

    from engram.ingestion.schema import EntityPayload

    names = {f.name for f in fields(EntityPayload)}
    assert "aliases" not in names, (
        "EntityPayload.aliases reintroduces primary-data mutation (R16 violation) — "
        "aliases must stay derived"
    )
