"""Stage [3] — N-gram extraction.

Two deterministic extractors emit key-phrase granules per
``docs/design/ingestion.md §3, §6``:

- :func:`extract_noun_chunk_ngrams` walks spaCy ``doc.noun_chunks`` and
  emits one n-gram per noun chunk whose containing Sentence is known.
- :func:`extract_svo_ngrams` walks the dependency parse of one Sentence and
  emits one n-gram per ``(nsubj / nsubjpass) + root verb + (dobj / attr /
  prep→pobj)`` triple.

**Fails closed.** N-grams below :class:`MemoryConfig.ngram_min_tokens`, all
stop-word, or with empty normalized text are dropped (R6-style — low-signal
structure would taint multi-Memory aggregation). Stop-word detection uses
spaCy's ``token.is_stop`` attribute; callers that mock the Doc must either
populate ``is_stop`` (``False`` default is fine for tests using real words)
or accept that their mock tokens count as non-stop-words.

**R2 discipline.** Output lists are sorted by
``(char_span, ngram_kind, normalized_text)`` before return so downstream
pipeline iteration is deterministic regardless of spaCy's internal ordering.

**Identity.** N-gram nodes are content-addressed by ``(segment_id,
ngram_kind, normalized_text)`` — two sentences containing "the cat"
produce two distinct nodes (each with its own part-of edge and its own
embedding under PR-C). One sentence's two extractors ("the cat" both
as noun_chunk and as SVO subject) produce two nodes too — they represent
two different parses of the same surface phrase.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from engram.ingestion.schema import (
    NGRAM_KIND_NOUN_CHUNK,
    NGRAM_KIND_SVO,
    NgramPayload,
    ngram_identity,
    node_id,
)


@dataclass(frozen=True, slots=True)
class _ResolvedNgram:
    """One pre-insert n-gram (before node_id + payload construction).

    Internal — the module surface is ``extract_*`` functions returning the
    final ``(node_id, payload)`` pairs.
    """

    normalized_text: str
    surface_form: str
    segment_id: str
    ngram_kind: str
    char_span: tuple[int, int]


def _normalize(text: str) -> str:
    """NFKC + casefold + whitespace collapse — the R2 identity string."""
    nfkc = unicodedata.normalize("NFKC", text).casefold().strip()
    # Collapse internal whitespace runs to a single space so "hello  world"
    # and "hello world" converge to the same n-gram.
    return " ".join(nfkc.split())


def _token_is_stop(tok: object) -> bool:
    """Return True if the token is a stop word. Defaults False on mocks
    that don't carry ``is_stop``."""
    return bool(getattr(tok, "is_stop", False))


def _count_content_tokens(tokens: Iterable[object]) -> int:
    """Count non-whitespace, non-stop-word tokens inside the span."""
    count = 0
    for tok in tokens:
        text = str(getattr(tok, "text", "")).strip()
        if not text:
            continue
        if _token_is_stop(tok):
            continue
        count += 1
    return count


def _chunk_tokens(chunk: object) -> list[object]:
    """Return the token list of a noun chunk / span.

    spaCy Spans are iterable over tokens; mocks that expose a ``tokens``
    attribute (tuple or list) are also accepted for tests that do not want
    to implement the iterator protocol.
    """
    toks = getattr(chunk, "tokens", None)
    if toks is not None:
        return list(toks)
    try:
        return list(iter(chunk))  # type: ignore[call-overload]
    except TypeError:
        return []


def _chunk_span(chunk: object, *, fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    start = int(getattr(chunk, "start_char", fallback[0]))
    end = int(getattr(chunk, "end_char", fallback[1]))
    return (start, end)


def extract_noun_chunk_ngrams(
    doc: object,
    segment_index_by_span: list[tuple[tuple[int, int], str]],
    *,
    min_tokens: int,
) -> list[tuple[str, NgramPayload]]:
    """Extract noun-chunk n-grams from ``doc.noun_chunks``.

    ``segment_index_by_span`` is a list of ``((start, end), segment_id)``
    pairs describing the character spans of every Sentence granule. N-grams
    whose span does not fall inside any Sentence are dropped (a chunk
    straddling sentence boundaries is silently ignored — the SVO extractor
    handles cross-sentence semantics, not this one).
    """
    chunks = list(getattr(doc, "noun_chunks", ()))
    resolved: list[_ResolvedNgram] = []
    for chunk in chunks:
        surface = str(getattr(chunk, "text", "")).strip()
        if not surface:
            continue
        normalized = _normalize(surface)
        if not normalized:
            continue
        tokens = _chunk_tokens(chunk)
        if _count_content_tokens(tokens) < min_tokens:
            continue
        span = _chunk_span(chunk)
        segment_id = _enclosing_segment(span, segment_index_by_span)
        if segment_id is None:
            continue
        resolved.append(
            _ResolvedNgram(
                normalized_text=normalized,
                surface_form=surface,
                segment_id=segment_id,
                ngram_kind=NGRAM_KIND_NOUN_CHUNK,
                char_span=span,
            )
        )
    return _finalize(resolved)


def extract_svo_ngrams(
    sent: object,
    segment_id: str,
    *,
    min_tokens: int,
) -> list[tuple[str, NgramPayload]]:
    """Extract (subject, verb, object) phrase n-grams from one Sentence.

    Subject = the first ``nsubj``/``nsubjpass`` child of the root; Object =
    the first ``dobj``/``attr`` child, falling back to a prep→pobj chain.
    Both slots expand to their ``subtree``. The phrase is rendered in
    source-order (subject, verb, object) and emitted as a single n-gram
    whose char span covers the subject's earliest offset through the
    object's latest offset.

    Sentences whose root is not a verb, whose subject is missing, or whose
    combined subject+object+verb has fewer than ``min_tokens`` content
    tokens emit nothing.
    """
    root = getattr(sent, "root", None)
    if root is None:
        return []
    if getattr(root, "pos_", "") not in {"VERB", "AUX"}:
        return []

    subj_tok = _first_child(root, {"nsubj", "nsubjpass"})
    if subj_tok is None:
        return []

    obj_tok = _first_child(root, {"dobj", "attr"})
    if obj_tok is None:
        prep = _first_child(root, {"prep"})
        if prep is not None:
            obj_tok = _first_child(prep, {"pobj"})
    if obj_tok is None:
        return []

    subj_tokens = list(getattr(subj_tok, "subtree", (subj_tok,)))
    obj_tokens = list(getattr(obj_tok, "subtree", (obj_tok,)))
    verb_tokens = [root]

    all_tokens = subj_tokens + verb_tokens + obj_tokens
    if _count_content_tokens(all_tokens) < min_tokens:
        return []

    subj_surface = _render_tokens(subj_tokens)
    obj_surface = _render_tokens(obj_tokens)
    verb_surface = str(getattr(root, "text", "")).strip()
    surface = " ".join(part for part in (subj_surface, verb_surface, obj_surface) if part)
    if not surface:
        return []

    normalized = _normalize(surface)
    if not normalized:
        return []

    span_start = min(
        int(getattr(t, "idx", 0))
        for t in (subj_tokens + verb_tokens + obj_tokens)
    )
    span_end = max(
        int(getattr(t, "idx", 0)) + len(str(getattr(t, "text", "")))
        for t in (subj_tokens + verb_tokens + obj_tokens)
    )

    resolved = _ResolvedNgram(
        normalized_text=normalized,
        surface_form=surface,
        segment_id=segment_id,
        ngram_kind=NGRAM_KIND_SVO,
        char_span=(span_start, span_end),
    )
    return _finalize([resolved])


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _first_child(token: object, deps: set[str]) -> object | None:
    for child in getattr(token, "children", ()):
        if getattr(child, "dep_", "") in deps:
            return cast("object", child)
    return None


def _render_tokens(tokens: Iterable[object]) -> str:
    parts: list[str] = []
    for tok in tokens:
        text = str(getattr(tok, "text", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _enclosing_segment(
    span: tuple[int, int],
    segment_spans: list[tuple[tuple[int, int], str]],
) -> str | None:
    start, end = span
    for (s_start, s_end), segment_id in segment_spans:
        if start >= s_start and end <= s_end:
            return segment_id
    return None


def _finalize(resolved: list[_ResolvedNgram]) -> list[tuple[str, NgramPayload]]:
    """Deduplicate by identity, sort, and materialize payloads.

    Two extractions with the same ``(segment_id, ngram_kind, normalized_text)``
    collapse to one node — n-grams are content-addressed. We keep the first
    one we saw (stable under input sort order) for ``surface_form`` and
    ``char_span``; callers relying on per-observation provenance should use
    the inbound ``part_of`` edge, not re-observe the node.
    """
    seen: dict[tuple[str, str, str], _ResolvedNgram] = {}
    for r in resolved:
        key = (r.segment_id, r.ngram_kind, r.normalized_text)
        if key not in seen:
            seen[key] = r

    out: list[tuple[str, NgramPayload]] = []
    for r in seen.values():
        payload = NgramPayload(
            normalized_text=r.normalized_text,
            surface_form=r.surface_form,
            segment_id=r.segment_id,
            ngram_kind=r.ngram_kind,
            char_span=r.char_span,
        )
        nid = node_id(
            ngram_identity(
                segment_id=r.segment_id,
                ngram_kind=r.ngram_kind,
                normalized_text=r.normalized_text,
            )
        )
        out.append((nid, payload))
    out.sort(key=lambda pair: (pair[1].char_span, pair[1].ngram_kind, pair[1].normalized_text))
    return out


__all__ = [
    "extract_noun_chunk_ngrams",
    "extract_svo_ngrams",
]
