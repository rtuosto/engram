"""Stage [2] — NER.

Extract entity mentions from a spaCy Doc. Mentions are *text spans with a
type* — not yet canonical Entity nodes; that's Stage [3].

**Determinism.** spaCy's CPU NER is bit-stable on a fixed model. GPU mode
falls under the R14 float-determinism budget (±2pp provisional); see
``docs/design/ingestion.md §7``.

**Output is ordered by ``(char_start, char_end)``** so downstream stages
iterate mentions in reading order regardless of spaCy's internal ordering.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EntityMention:
    """One NER-emitted span inside a Turn.

    ``surface_form`` is the raw text (before canonicalization's Unicode
    normalization). ``entity_type`` is spaCy's label — ``PERSON``, ``ORG``,
    ``GPE``, ``DATE``, etc.
    """

    surface_form: str
    entity_type: str
    char_span: tuple[int, int]
    turn_id: str


def extract_mentions(doc: object, turn_id: str) -> list[EntityMention]:
    """Return mentions from ``doc``, ordered by ``(start_char, end_char)``.

    ``doc`` is a spaCy Doc (or any object exposing ``.ents`` whose items have
    ``.text``, ``.label_``, ``.start_char``, ``.end_char``). Empty / whitespace
    surfaces are dropped — fails closed on degenerate spans.
    """
    mentions: list[EntityMention] = []
    for ent in getattr(doc, "ents", ()):
        surface = str(getattr(ent, "text", "")).strip()
        if not surface:
            continue
        label = str(getattr(ent, "label_", "")).strip()
        if not label:
            continue
        start = int(getattr(ent, "start_char", 0))
        end = int(getattr(ent, "end_char", start + len(surface)))
        mentions.append(
            EntityMention(
                surface_form=surface,
                entity_type=label,
                char_span=(start, end),
                turn_id=turn_id,
            )
        )
    mentions.sort(key=lambda m: (m.char_span[0], m.char_span[1], m.surface_form))
    return mentions


__all__ = ["EntityMention", "extract_mentions"]
