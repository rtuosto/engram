"""needle_overlap — key-term extraction + recall fraction."""

from __future__ import annotations

from engram.diagnostics.overlap import extract_key_terms, needle_overlap


def test_extract_key_terms_drops_short_tokens_and_stopwords() -> None:
    terms = extract_key_terms("The Eiffel Tower is in Paris.")
    # "the", "is", "in" are stop-words; everything else >=3 chars survives
    # and is sorted, case-folded, deduped.
    assert terms == ("eiffel", "paris", "tower")


def test_extract_key_terms_casefolds_ascii() -> None:
    terms = extract_key_terms("Paris PARIS paris")
    # Casefold + dedup: all three collapse to one token.
    assert terms == ("paris",)


def test_extract_key_terms_empty_gold() -> None:
    assert extract_key_terms("") == ()


def test_needle_overlap_full_recall() -> None:
    ov = needle_overlap(
        "Paris is the capital of France.",
        "We visited Paris, the capital, and France together.",
    )
    assert ov.recall == 1.0
    assert ov.found == ("capital", "france", "paris")
    assert ov.missed == ()


def test_needle_overlap_partial_recall() -> None:
    ov = needle_overlap("Paris France Germany", "Only Paris mentioned.")
    assert ov.recall == 1 / 3
    assert "paris" in ov.found
    assert "germany" in ov.missed


def test_needle_overlap_zero_recall_on_empty_text() -> None:
    ov = needle_overlap("something valuable", "")
    assert ov.recall == 0.0
    assert ov.found == ()
    assert set(ov.missed) == set(ov.terms)


def test_needle_overlap_empty_gold_returns_zero_recall_no_terms() -> None:
    ov = needle_overlap("", "any text here")
    assert ov.terms == ()
    assert ov.recall == 0.0


def test_needle_overlap_is_deterministic() -> None:
    # R2: two calls return byte-identical tuples.
    a = needle_overlap("Paris France Germany", "Only Paris mentioned.")
    b = needle_overlap("Paris France Germany", "Only Paris mentioned.")
    assert a == b
