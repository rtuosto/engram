"""Stage [5] — Entity canonicalization.

Minimum-viable algorithm per ``docs/design/ingestion.md §10``: Unicode NFKC
lowercasing + ``rapidfuzz.fuzz.token_set_ratio`` against existing entities
of the same ``entity_type``. Feature infrastructure is wired for future
embedding and co-occurrence features; their weights are 0 in this PR.

**Determinism.** Tie-breaking is a strict total order:
``(higher similarity, alphabetical canonical_form, lexicographic node_id)``.
R2-compliant under any mention insertion order.

**Entity-type gate.** Canonicalization only merges within identical
``entity_type`` (``PERSON`` ↔ ``PERSON``; ``ORG`` ↔ ``ORG``). Prevents
"Apple" the company colliding with "apple" the food.

**R16.** :class:`EntityPayload` does not carry an ``aliases`` field.
Aliases are a derived index rebuilt from the inbound ``mentions`` edges
(PR-D). The registry tracks observed surface forms purely so the pipeline
can attach them to ``mentions`` edges as provenance; they are not part of
the Entity node's primary payload.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from rapidfuzz.fuzz import token_set_ratio  # type: ignore[import-untyped]

from engram.ingestion.extractors.ner import EntityMention
from engram.ingestion.schema import (
    EntityPayload,
    entity_identity,
    node_id,
)

# Similarity-feature weights. Only string is active in this PR; embedding
# and co-occurrence slots are wired so a future PR can enable them without
# reshaping the function signature. Changing any of these affects graph
# output and would need to enter MemoryConfig._INGESTION_FIELDS.
SIMILARITY_WEIGHTS: dict[str, float] = {
    "string": 1.0,
    "embedding": 0.0,
    "co_occurrence": 0.0,
}


@dataclass
class EntityRegistry:
    """Per-(entity_type, normalized_form) index of canonical entities.

    Value shape is ``(node_id, first_seen_surface)`` — the registry just
    resolves a surface form to a canonical node ID. Full alias sets are
    rebuilt from the graph's inbound ``mentions`` edges in the derived
    layer (R16).
    """

    # (entity_type, normalized_form) -> node_id
    by_type_and_form: dict[tuple[str, str], str]

    def __init__(self) -> None:
        self.by_type_and_form = {}

    def types(self) -> list[str]:
        return sorted({t for t, _ in self.by_type_and_form})

    def forms_of_type(self, entity_type: str) -> list[tuple[str, str]]:
        """Sorted list of ``(normalized_form, node_id)`` for one type."""
        items = [
            (form, nid)
            for (t, form), nid in self.by_type_and_form.items()
            if t == entity_type
        ]
        items.sort(key=lambda it: it[0])
        return items


def normalize(surface: str) -> str:
    """Unicode NFKC casefolded form used for exact-match and similarity."""
    return unicodedata.normalize("NFKC", surface).casefold().strip()


def canonicalize(
    mention: EntityMention,
    registry: EntityRegistry,
    *,
    match_threshold: float,
) -> tuple[str, EntityPayload, bool]:
    """Resolve ``mention`` to a canonical entity node.

    Returns ``(entity_node_id, payload, is_new)``. Mutates ``registry``
    only when a genuinely new canonical form is observed. Aliases are
    *not* tracked on the payload (R16) — the pipeline records the surface
    form on the ``mentions`` edge instead.

    **Fails closed if ``entity_type`` is empty** — callers should pre-filter
    (NER already drops empty labels). ``match_threshold`` is the
    ``token_set_ratio`` cut-off in ``[0, 100]``; MemoryConfig stores it as a
    0–1 float, so the pipeline multiplies by 100 before calling.
    """
    if not mention.entity_type:
        raise ValueError("canonicalize: entity_type must be non-empty")

    normalized = normalize(mention.surface_form)
    key = (mention.entity_type, normalized)

    # Rule 1: exact normalized match.
    existing = registry.by_type_and_form.get(key)
    if existing is not None:
        return existing, EntityPayload(
            canonical_form=normalized,
            entity_type=mention.entity_type,
        ), False

    # Rule 2: above-threshold fuzzy match against same-type entities.
    best_score = -1.0
    best_entry: tuple[str, str] | None = None
    for existing_form, existing_id in registry.forms_of_type(mention.entity_type):
        score = float(token_set_ratio(normalized, existing_form))
        # Strict total order: (higher similarity, alphabetical canonical_form,
        # lexicographic node_id). We iterate forms in alphabetical order
        # already; only update when strictly better score.
        if score > best_score:
            best_score = score
            best_entry = (existing_form, existing_id)

    if best_entry is not None and best_score >= match_threshold * 100.0:
        existing_form, existing_id = best_entry
        return existing_id, EntityPayload(
            canonical_form=existing_form,
            entity_type=mention.entity_type,
        ), False

    # Rule 3: new canonical entity.
    new_id = node_id(entity_identity(normalized, mention.entity_type))
    registry.by_type_and_form[key] = new_id
    return new_id, EntityPayload(
        canonical_form=normalized,
        entity_type=mention.entity_type,
    ), True


__all__ = [
    "SIMILARITY_WEIGHTS",
    "EntityRegistry",
    "canonicalize",
    "normalize",
]
