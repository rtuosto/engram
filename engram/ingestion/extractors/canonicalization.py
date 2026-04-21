"""Stage [3] — Entity canonicalization.

Minimum-viable algorithm per ``docs/design/ingestion.md §6``: Unicode NFKC
lowercasing + ``rapidfuzz.fuzz.token_set_ratio`` against existing entities
of the same ``entity_type``. Feature infrastructure is wired for future
embedding and co-occurrence features; their weights are 0 in Tier 1.

**Determinism.** Tie-breaking is a strict total order:
``(higher similarity, alphabetical canonical_form, lexicographic node_id)``.
R2-compliant under any mention insertion order.

**Entity-type gate.** Canonicalization only merges within identical
``entity_type`` (``PERSON`` ↔ ``PERSON``; ``ORG`` ↔ ``ORG``). Prevents
"Apple" the company colliding with "apple" the food.
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

# Similarity-feature weights. Only string is active in Tier 1; embedding
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

    Tracks surface aliases observed per canonical form so the payload's
    ``aliases`` tuple stays deterministic under any mention-arrival order.
    """

    # (entity_type, normalized_form) -> (node_id, sorted tuple of aliases)
    by_type_and_form: dict[tuple[str, str], tuple[str, tuple[str, ...]]]

    def __init__(self) -> None:
        self.by_type_and_form = {}

    def types(self) -> list[str]:
        return sorted({t for t, _ in self.by_type_and_form})

    def forms_of_type(self, entity_type: str) -> list[tuple[str, str, tuple[str, ...]]]:
        """Sorted list of ``(normalized_form, node_id, aliases)`` for one type."""
        items = [
            (form, node_id, aliases)
            for (t, form), (node_id, aliases) in self.by_type_and_form.items()
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

    Returns ``(entity_node_id, payload, is_new)``. Mutates ``registry`` —
    either merges the mention's surface as an alias onto an existing entity,
    or records a brand-new canonical form.

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
        entity_id, aliases = existing
        if mention.surface_form not in aliases:
            aliases = tuple(sorted({*aliases, mention.surface_form}))
            registry.by_type_and_form[key] = (entity_id, aliases)
        return entity_id, EntityPayload(
            canonical_form=normalized,
            entity_type=mention.entity_type,
            aliases=aliases,
        ), False

    # Rule 2: above-threshold fuzzy match against same-type entities.
    best_score = -1.0
    best_entry: tuple[str, str, tuple[str, ...]] | None = None
    for existing_form, existing_id, existing_aliases in registry.forms_of_type(
        mention.entity_type
    ):
        score = float(token_set_ratio(normalized, existing_form))
        # Strict total order: (higher similarity, alphabetical canonical_form,
        # lexicographic node_id). We iterate forms in alphabetical order
        # already; only update when strictly better score.
        if score > best_score:
            best_score = score
            best_entry = (existing_form, existing_id, existing_aliases)

    if best_entry is not None and best_score >= match_threshold * 100.0:
        existing_form, existing_id, existing_aliases = best_entry
        merged_aliases = tuple(sorted({*existing_aliases, mention.surface_form}))
        registry.by_type_and_form[(mention.entity_type, existing_form)] = (
            existing_id,
            merged_aliases,
        )
        return existing_id, EntityPayload(
            canonical_form=existing_form,
            entity_type=mention.entity_type,
            aliases=merged_aliases,
        ), False

    # Rule 3: new canonical entity.
    new_id = node_id(entity_identity(normalized, mention.entity_type))
    aliases: tuple[str, ...] = (mention.surface_form,)
    registry.by_type_and_form[key] = (new_id, aliases)
    return new_id, EntityPayload(
        canonical_form=normalized,
        entity_type=mention.entity_type,
        aliases=aliases,
    ), True


__all__ = [
    "SIMILARITY_WEIGHTS",
    "EntityRegistry",
    "canonicalize",
    "normalize",
]
