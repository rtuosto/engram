"""Stage [4] — Claim extraction.

Extract ``(subject, predicate, object)`` triples from the dependency parse
of each UtteranceSegment's tokens. See ``docs/design/ingestion.md §3 [4]``.

**Fails closed.** No identifiable subject **or** no predicate → no Claim
emitted. A claim whose subject slot is unresolved but whose predicate is
clear is still dropped; partial claims taint multi-session aggregation.

**First-person pronouns resolve to the Turn's speaker.** "I", "me", "my"
resolve to ``speaker_entity_id``. Second- and third-person pronouns are
not resolved in Tier 1 (coreference is a deferred upgrade per
``docs/design/ingestion.md §6`` "Known Tier-1 limitations").
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.ingestion.extractors.ner import EntityMention
from engram.ingestion.schema import (
    ClaimPayload,
    claim_identity,
    node_id,
)

_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})

# Heuristics for modality / tense. Deliberately small: morph-feature-driven
# extraction; overfitting to one dataset's phrasings is R6-forbidden.
_MODAL_HYPOTHETICAL = frozenset({"would", "could", "might", "may", "should"})


@dataclass(frozen=True, slots=True)
class ResolvedMention:
    """A canonicalized mention — an entity_node_id plus the source span."""

    entity_id: str
    char_span: tuple[int, int]


def _char_span_of(token: object) -> tuple[int, int]:
    start = int(getattr(token, "idx", 0))
    text_len = len(str(getattr(token, "text", "")))
    return (start, start + text_len)


def _subtree_span(token: object) -> tuple[int, int]:
    """Character span covering the full subtree under ``token``."""
    subtree = list(getattr(token, "subtree", (token,)))
    if not subtree:
        return _char_span_of(token)
    starts = [int(getattr(t, "idx", 0)) for t in subtree]
    ends = [int(getattr(t, "idx", 0)) + len(str(getattr(t, "text", ""))) for t in subtree]
    return (min(starts), max(ends))


def _resolve_span(
    span: tuple[int, int],
    mentions_by_span: list[ResolvedMention],
) -> str | None:
    """Return the entity_id whose mention span is enclosed in ``span``, if any.

    If multiple enclosed mentions exist, the first one (by start offset, then
    end offset) wins — deterministic under any order of ``mentions_by_span``.
    """
    start, end = span
    enclosed = [
        m for m in mentions_by_span if m.char_span[0] >= start and m.char_span[1] <= end
    ]
    enclosed.sort(key=lambda m: (m.char_span[0], m.char_span[1]))
    return enclosed[0].entity_id if enclosed else None


def _find_nsubj(verb: object) -> object | None:
    for child in getattr(verb, "children", ()):
        if getattr(child, "dep_", "") in {"nsubj", "nsubjpass"}:
            return child
    return None


def _find_object(verb: object) -> object | None:
    for child in getattr(verb, "children", ()):
        if getattr(child, "dep_", "") in {"dobj", "attr"}:
            return child
    # Fall back to prepositional object via any prep child.
    for child in getattr(verb, "children", ()):
        if getattr(child, "dep_", "") == "prep":
            for grand in getattr(child, "children", ()):
                if getattr(grand, "dep_", "") == "pobj":
                    return grand
    return None


def _has_negation(verb: object) -> bool:
    return any(
        getattr(child, "dep_", "") == "neg" for child in getattr(verb, "children", ())
    )


def _modal_aux(verb: object) -> str | None:
    for child in getattr(verb, "children", ()):
        if getattr(child, "dep_", "") in {"aux", "auxpass"}:
            token_text = str(getattr(child, "text", "")).casefold()
            if token_text in _MODAL_HYPOTHETICAL:
                return token_text
            if token_text == "will":
                return "will"
    return None


def _tense_from_morph(verb: object) -> str:
    morph = getattr(verb, "morph", None)
    if morph is None:
        return "present"
    try:
        tense_values = morph.get("Tense") if hasattr(morph, "get") else []
    except Exception:
        tense_values = []
    if "Past" in tense_values:
        return "past"
    if "Pres" in tense_values:
        return "present"
    return "present"


def _modality(verb: object, sentence_text: str) -> str:
    if sentence_text.rstrip().endswith("?"):
        return "interrogative"
    if _has_negation(verb):
        return "negated"
    modal = _modal_aux(verb)
    if modal in _MODAL_HYPOTHETICAL:
        return "hypothetical"
    return "asserted"


def _tense(verb: object) -> str:
    modal = _modal_aux(verb)
    if modal == "will":
        return "future"
    return _tense_from_morph(verb)


def extract_claims_from_sentence(
    sent: object,
    turn_id: str,
    speaker_entity_id: str | None,
    mentions: list[ResolvedMention],
    asserted_at: str | None,
    *,
    subject_required: bool,
) -> list[tuple[str, ClaimPayload]]:
    """Extract claims from one spaCy sentence span.

    Returns sorted ``(claim_node_id, payload)`` pairs — deterministic under
    any caller iteration order.
    """
    root = getattr(sent, "root", None)
    if root is None:
        return []
    if getattr(root, "pos_", "") not in {"VERB", "AUX"}:
        return []

    predicate = str(getattr(root, "lemma_", "")).casefold().strip()
    if not predicate:
        return []

    nsubj_tok = _find_nsubj(root)
    if nsubj_tok is None:
        if subject_required:
            return []
        subject_id = None
    else:
        nsubj_text = str(getattr(nsubj_tok, "text", "")).casefold()
        if nsubj_text in _FIRST_PERSON and speaker_entity_id is not None:
            subject_id = speaker_entity_id
        else:
            subject_id = _resolve_span(_subtree_span(nsubj_tok), mentions)
            if subject_id is None and subject_required:
                return []

    obj_tok = _find_object(root)
    object_id: str | None = None
    object_literal: str | None = None
    if obj_tok is not None:
        obj_span = _subtree_span(obj_tok)
        object_id = _resolve_span(obj_span, mentions)
        if object_id is None:
            literal_text = "".join(
                str(getattr(tok, "text_with_ws", getattr(tok, "text", "")))
                for tok in getattr(obj_tok, "subtree", (obj_tok,))
            ).strip()
            object_literal = literal_text or None

    sentence_text = str(getattr(sent, "text", ""))
    modality = _modality(root, sentence_text)
    tense = _tense(root)

    # Subject-not-required path: anchor identity on the speaker slot so two
    # anonymous claims with the same predicate on the same turn don't
    # collide. Callers should usually keep subject_required=True.
    identity_subject = subject_id if subject_id is not None else (speaker_entity_id or "")

    cid = node_id(
        claim_identity(
            subject_id=identity_subject,
            predicate=predicate,
            object_id=object_id,
            object_literal=object_literal,
            asserted_by_turn_id=turn_id,
        )
    )
    payload = ClaimPayload(
        subject_id=subject_id or "",
        predicate=predicate,
        object_id=object_id,
        object_literal=object_literal,
        asserted_by_turn_id=turn_id,
        asserted_at=asserted_at,
        modality=modality,
        tense=tense,
    )
    return [(cid, payload)]


def extract_claims_from_doc(
    doc: object,
    turn_id: str,
    speaker_entity_id: str | None,
    mentions: list[EntityMention],
    entity_id_by_span: dict[tuple[int, int], str],
    asserted_at: str | None,
    *,
    subject_required: bool,
) -> list[tuple[str, ClaimPayload]]:
    """Iterate over ``doc.sents`` and extract claims from each sentence."""
    resolved_mentions = [
        ResolvedMention(
            entity_id=entity_id_by_span[(m.char_span[0], m.char_span[1])],
            char_span=m.char_span,
        )
        for m in mentions
        if (m.char_span[0], m.char_span[1]) in entity_id_by_span
    ]

    claims: list[tuple[str, ClaimPayload]] = []
    for sent in getattr(doc, "sents", ()):
        claims.extend(
            extract_claims_from_sentence(
                sent,
                turn_id=turn_id,
                speaker_entity_id=speaker_entity_id,
                mentions=resolved_mentions,
                asserted_at=asserted_at,
                subject_required=subject_required,
            )
        )
    claims.sort(key=lambda pair: pair[0])
    return claims


__all__ = [
    "ResolvedMention",
    "extract_claims_from_doc",
    "extract_claims_from_sentence",
]
