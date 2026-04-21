"""Per-extractor unit tests for :mod:`engram.ingestion.extractors.ngram`.

Two extractors, tested independently with hermetic FakeDocs / FakeSents:

- :func:`extract_noun_chunk_ngrams` — spaCy ``doc.noun_chunks`` → n-gram
  nodes, scoped to their containing Sentence.
- :func:`extract_svo_ngrams` — per-sentence dependency-parse SVO triples →
  one n-gram per sentence where subject + verb + object resolve.

Coverage priorities: R16 content-addressed identity, R6 fails-closed (below
``min_tokens`` / all-stop-word surfaces dropped), and R2 sort order.
"""

from __future__ import annotations

from engram.ingestion.extractors.ngram import (
    extract_noun_chunk_ngrams,
    extract_svo_ngrams,
)
from engram.ingestion.schema import (
    NGRAM_KIND_NOUN_CHUNK,
    NGRAM_KIND_SVO,
    ngram_identity,
    node_id,
)
from tests._fake_nlp import (
    FakeNounChunk,
    FakeSent,
    attach_subtree,
    make_fake_doc,
    make_token,
)


def _build_noun_chunk(text: str, start: int, tokens: list) -> FakeNounChunk:
    return FakeNounChunk(
        text=text,
        start_char=start,
        end_char=start + len(text),
        tokens=tuple(tokens),
    )


# ---------------------------------------------------------------------------
# extract_noun_chunk_ngrams
# ---------------------------------------------------------------------------


def test_noun_chunk_emits_node_scoped_to_its_segment() -> None:
    cat = make_token("cat", idx=4, pos="NOUN")
    the = make_token("the", idx=0, pos="DET", is_stop=True)
    chunk = _build_noun_chunk("the cat", start=0, tokens=[the, cat])
    doc = make_fake_doc(text="the cat sat.", noun_chunks=[chunk])

    out = extract_noun_chunk_ngrams(
        doc,
        [((0, 12), "seg_id_1")],
        min_tokens=1,
    )
    assert len(out) == 1
    ngram_id, payload = out[0]
    assert payload.normalized_text == "the cat"
    assert payload.surface_form == "the cat"
    assert payload.segment_id == "seg_id_1"
    assert payload.ngram_kind == NGRAM_KIND_NOUN_CHUNK
    assert payload.char_span == (0, 7)
    # Identity is content-addressed.
    expected = node_id(ngram_identity("seg_id_1", NGRAM_KIND_NOUN_CHUNK, "the cat"))
    assert ngram_id == expected


def test_noun_chunk_below_min_tokens_is_dropped() -> None:
    # "the" is a stop word; non-stop token count = 0 < min_tokens=2.
    the = make_token("the", idx=0, pos="DET", is_stop=True)
    chunk = _build_noun_chunk("the", start=0, tokens=[the])
    doc = make_fake_doc(text="the cat.", noun_chunks=[chunk])

    out = extract_noun_chunk_ngrams(doc, [((0, 8), "seg_id")], min_tokens=2)
    assert out == []


def test_noun_chunk_all_stop_words_is_dropped() -> None:
    the = make_token("the", idx=0, pos="DET", is_stop=True)
    a = make_token("a", idx=4, pos="DET", is_stop=True)
    chunk = _build_noun_chunk("the a", start=0, tokens=[the, a])
    doc = make_fake_doc(text="the a thing.", noun_chunks=[chunk])

    out = extract_noun_chunk_ngrams(doc, [((0, 12), "seg_id")], min_tokens=1)
    assert out == []


def test_noun_chunk_outside_segment_is_dropped() -> None:
    """A chunk whose span doesn't fall inside any Sentence is dropped."""
    cat = make_token("cat", idx=100, pos="NOUN")
    dog = make_token("dog", idx=104, pos="NOUN")
    chunk = _build_noun_chunk("cat dog", start=100, tokens=[cat, dog])
    doc = make_fake_doc(text="x" * 50, noun_chunks=[chunk])

    # Sentence only covers 0–50.
    out = extract_noun_chunk_ngrams(doc, [((0, 50), "seg_id")], min_tokens=1)
    assert out == []


def test_noun_chunk_output_is_sorted_by_char_span() -> None:
    later = _build_noun_chunk(
        "bigger concept",
        start=20,
        tokens=[
            make_token("bigger", idx=20, pos="ADJ"),
            make_token("concept", idx=27, pos="NOUN"),
        ],
    )
    earlier = _build_noun_chunk(
        "green apple",
        start=0,
        tokens=[
            make_token("green", idx=0, pos="ADJ"),
            make_token("apple", idx=6, pos="NOUN"),
        ],
    )
    # Feed in reverse order; expect earlier span first on output.
    doc = make_fake_doc(text="x" * 60, noun_chunks=[later, earlier])
    out = extract_noun_chunk_ngrams(doc, [((0, 60), "seg_id")], min_tokens=1)
    assert [p.normalized_text for _, p in out] == ["green apple", "bigger concept"]


def test_noun_chunk_casing_and_unicode_normalized() -> None:
    """Identity is NFKC + casefold — "Café" and "café" converge."""
    a_tok = make_token("Café", idx=0, pos="NOUN")
    chunk = _build_noun_chunk("Café", start=0, tokens=[a_tok])
    doc = make_fake_doc(text="Café.", noun_chunks=[chunk])
    out_a = extract_noun_chunk_ngrams(doc, [((0, 5), "seg_id")], min_tokens=1)

    b_tok = make_token("café", idx=0, pos="NOUN")
    chunk_b = _build_noun_chunk("café", start=0, tokens=[b_tok])
    doc_b = make_fake_doc(text="café.", noun_chunks=[chunk_b])
    out_b = extract_noun_chunk_ngrams(doc_b, [((0, 5), "seg_id")], min_tokens=1)

    assert out_a[0][0] == out_b[0][0]  # same node_id


# ---------------------------------------------------------------------------
# extract_svo_ngrams
# ---------------------------------------------------------------------------


def _build_svo_sentence(text: str, *, subj: str, verb: str, obj: str) -> FakeSent:
    subj_idx = 0
    verb_idx = len(subj) + 1
    obj_idx = verb_idx + len(verb) + 1

    root = make_token(verb, idx=verb_idx, pos="VERB", lemma=verb.lower())
    nsubj = make_token(subj, idx=subj_idx, pos="PROPN", dep="nsubj")
    dobj = make_token(obj, idx=obj_idx, pos="NOUN", dep="dobj")
    root.children = (nsubj, dobj)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(dobj, [dobj])
    attach_subtree(root, [nsubj, root, dobj])
    return FakeSent(text=text, start_char=0, end_char=len(text), root=root)


def test_svo_emits_one_ngram_per_sentence() -> None:
    sent = _build_svo_sentence("Alice loves hiking.", subj="Alice", verb="loves", obj="hiking")
    out = extract_svo_ngrams(sent, "seg_id_0", min_tokens=1)
    assert len(out) == 1
    _, payload = out[0]
    assert payload.normalized_text == "alice loves hiking"
    assert payload.surface_form == "Alice loves hiking"
    assert payload.ngram_kind == NGRAM_KIND_SVO
    assert payload.segment_id == "seg_id_0"


def test_svo_without_subject_emits_nothing() -> None:
    root = make_token("run", idx=0, pos="VERB")
    sent = FakeSent(text="run.", start_char=0, end_char=4, root=root)
    out = extract_svo_ngrams(sent, "seg_id", min_tokens=1)
    assert out == []


def test_svo_without_object_emits_nothing() -> None:
    root = make_token("runs", idx=6, pos="VERB")
    nsubj = make_token("Alice", idx=0, pos="PROPN", dep="nsubj")
    root.children = (nsubj,)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(root, [nsubj, root])
    sent = FakeSent(text="Alice runs.", start_char=0, end_char=11, root=root)
    out = extract_svo_ngrams(sent, "seg_id", min_tokens=1)
    assert out == []


def test_svo_prepositional_object_fallback() -> None:
    """No direct object → walk prep → pobj."""
    park = make_token("park", idx=21, pos="NOUN", dep="pobj")
    in_prep = make_token("in", idx=18, pos="ADP", dep="prep", children=(park,))
    attach_subtree(park, [park])
    nsubj = make_token("Alice", idx=0, pos="PROPN", dep="nsubj")
    root = make_token("walks", idx=6, pos="VERB", lemma="walk")
    root.children = (nsubj, in_prep)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(root, [nsubj, root, in_prep, park])
    sent = FakeSent(
        text="Alice walks in the park.", start_char=0, end_char=24, root=root
    )
    out = extract_svo_ngrams(sent, "seg_id", min_tokens=1)
    assert len(out) == 1
    assert "alice" in out[0][1].normalized_text
    assert "walks" in out[0][1].normalized_text
    assert "park" in out[0][1].normalized_text


def test_svo_non_verb_root_emits_nothing() -> None:
    root = make_token("cat", idx=0, pos="NOUN")
    sent = FakeSent(text="cat.", start_char=0, end_char=4, root=root)
    assert extract_svo_ngrams(sent, "seg_id", min_tokens=1) == []


def test_svo_min_tokens_gate_drops_low_signal_triples() -> None:
    """All-stop-word subject + object + stopword verb below threshold."""
    root = make_token("is", idx=4, pos="VERB", lemma="be", is_stop=True)
    nsubj = make_token("It", idx=0, pos="PRON", dep="nsubj", is_stop=True)
    dobj = make_token("a", idx=7, pos="DET", dep="dobj", is_stop=True)
    root.children = (nsubj, dobj)
    attach_subtree(nsubj, [nsubj])
    attach_subtree(dobj, [dobj])
    attach_subtree(root, [nsubj, root, dobj])
    sent = FakeSent(text="It is a.", start_char=0, end_char=8, root=root)
    out = extract_svo_ngrams(sent, "seg_id", min_tokens=2)
    assert out == []


def test_svo_identity_is_deterministic_across_calls() -> None:
    """Two extractions from the same sentence produce the same node_id."""
    s1 = _build_svo_sentence("Alice loves hiking.", subj="Alice", verb="loves", obj="hiking")
    s2 = _build_svo_sentence("Alice loves hiking.", subj="Alice", verb="loves", obj="hiking")
    out1 = extract_svo_ngrams(s1, "seg_id_0", min_tokens=1)
    out2 = extract_svo_ngrams(s2, "seg_id_0", min_tokens=1)
    assert out1[0][0] == out2[0][0]


def test_svo_different_sentences_yield_different_node_ids() -> None:
    """Same phrase in different Sentences produces distinct N-gram nodes
    (R16: N-gram identity includes ``segment_id`` by design)."""
    s1 = _build_svo_sentence("Alice loves hiking.", subj="Alice", verb="loves", obj="hiking")
    s2 = _build_svo_sentence("Alice loves hiking.", subj="Alice", verb="loves", obj="hiking")
    out1 = extract_svo_ngrams(s1, "seg_id_0", min_tokens=1)
    out2 = extract_svo_ngrams(s2, "seg_id_1", min_tokens=1)
    assert out1[0][0] != out2[0][0]
