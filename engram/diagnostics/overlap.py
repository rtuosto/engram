"""Gold-term overlap — the cheapest signal about retrieval quality.

``needle_overlap(gold, text)`` extracts content terms from a gold answer
string and measures how many appear verbatim in a candidate text. It is
the kernel the R15 failure classifier uses to decide whether the right
content reached the agent (high overlap) vs whether retrieval lost it
(zero overlap).

**Fails closed.** An empty ``gold`` yields zero content terms and an
``Overlap(recall=0.0, ...)`` with an empty ``found`` tuple. Callers
should gate on ``overlap.terms`` being non-empty before making
classifier decisions.

**No LLM, no regex on speech acts.** Content-term extraction is strict
Unicode NFKC casefold + 3-char minimum + stop-word filter. The stop-word
list is intentionally small and English-only; longer stop-lists invite
R6-style English-specific behavior and are deferred until a hypothesis
demands them.

**R2 discipline.** ``Overlap`` fields are sorted tuples — two calls with
the same ``(gold, text)`` produce byte-identical output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

# Small English stop-word set. Deliberately short — every entry has a
# reason. If adding, cite the miss it fixes.
_STOP_WORDS: Final[frozenset[str]] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "has", "his", "how", "its", "may", "new",
    "now", "old", "see", "way", "who", "did", "get", "got", "had", "him",
    "let", "say", "she", "too", "use", "also", "been", "some", "than",
    "them", "then", "they", "this", "that", "what", "when", "will", "with",
    "from", "have", "more", "would", "about", "could", "other", "their",
    "there", "these", "which", "should", "because", "through",
    "days", "weeks", "months", "years", "hours", "including", "last",
    "first", "acceptable", "correct", "answer", "response",
})

_MIN_TERM_LEN: Final[int] = 3
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-zA-Z0-9]+")


@dataclass(frozen=True, slots=True)
class Overlap:
    """One ``needle_overlap`` call's result.

    - ``terms`` — sorted unique content terms extracted from gold.
    - ``found`` — subset of ``terms`` present in the candidate text.
    - ``missed`` — ``terms - found``, sorted.
    - ``recall`` — ``len(found) / len(terms)`` when non-empty, else ``0.0``.
    """

    terms: tuple[str, ...]
    found: tuple[str, ...]
    missed: tuple[str, ...]
    recall: float


def extract_key_terms(gold: str) -> tuple[str, ...]:
    """Pull content terms from a gold answer string.

    NFKC normalize + casefold, tokenize on ASCII alphanumerics, drop
    tokens shorter than 3 chars or in the stop-word list, return sorted
    unique tuple.
    """
    if not gold:
        return ()
    normalized = unicodedata.normalize("NFKC", gold).casefold()
    tokens = _TOKEN_PATTERN.findall(normalized)
    return tuple(sorted({
        t for t in tokens if len(t) >= _MIN_TERM_LEN and t not in _STOP_WORDS
    }))


def needle_overlap(gold: str, text: str) -> Overlap:
    """Measure content-term overlap between ``gold`` and ``text``.

    Both strings are NFKC-casefolded before comparison (the candidate in
    full — no tokenization needed since ``in`` substring match is
    sufficient). Determinism: output tuples are sorted.
    """
    terms = extract_key_terms(gold)
    if not terms:
        return Overlap(terms=(), found=(), missed=(), recall=0.0)

    haystack = unicodedata.normalize("NFKC", text).casefold()
    found = tuple(t for t in terms if t in haystack)
    missed = tuple(t for t in terms if t not in haystack)
    return Overlap(
        terms=terms,
        found=found,
        missed=missed,
        recall=len(found) / len(terms),
    )


__all__ = [
    "Overlap",
    "extract_key_terms",
    "needle_overlap",
]
