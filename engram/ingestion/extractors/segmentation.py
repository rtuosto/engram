"""Stage [1] — Segmentation.

Split each Turn into UtteranceSegment nodes via spaCy's dependency-parse-aware
sentence boundary detection. Surface-regex boundaries are forbidden under R6:
regex overfits to the phrasings that happen to appear in one dataset.

**Fails closed.** A Turn producing zero ``Doc.sents`` (empty / whitespace /
punctuation-only) emits no UtteranceSegment nodes; the Turn itself still exists.
"""

from __future__ import annotations

from collections.abc import Iterable

from engram.ingestion.schema import (
    UtteranceSegmentPayload,
    node_id,
    segment_identity,
)


def segment_turn(doc: object, turn_id: str) -> list[tuple[str, UtteranceSegmentPayload]]:
    """Return sorted ``(segment_node_id, payload)`` pairs for the given Doc.

    ``doc`` is a spaCy :class:`~spacy.tokens.Doc`. We accept ``object`` here
    to keep this module import-free of spaCy for test mocking; it uses only
    the public ``.sents`` iterator and ``.text`` / ``.start_char`` / ``.end_char``.

    Segmentation is left-to-right; ``segment_index`` increments monotonically
    and segments with zero-length text are dropped.
    """
    segments: list[tuple[str, UtteranceSegmentPayload]] = []
    sents: Iterable[object] = getattr(doc, "sents", ())
    seg_index = 0
    for sent in sents:
        text = str(getattr(sent, "text", "")).strip()
        if not text:
            continue
        start = int(getattr(sent, "start_char", 0))
        end = int(getattr(sent, "end_char", start + len(text)))
        payload = UtteranceSegmentPayload(
            text=text,
            turn_id=turn_id,
            segment_index=seg_index,
            char_span=(start, end),
        )
        seg_id = node_id(segment_identity(turn_id, seg_index))
        segments.append((seg_id, payload))
        seg_index += 1
    return segments


__all__ = ["segment_turn"]
