"""classify_failures — R15 decision tree + bucket_breakdown."""

from __future__ import annotations

from engram.diagnostics import (
    FailureInput,
    FailureKind,
    bucket_breakdown,
    classify_failures,
)
from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    EDGE_PART_OF,
    LABEL_NGRAM,
    LABEL_TURN,
    NGRAM_KIND_NOUN_CHUNK,
    EdgeAttrs,
    NgramPayload,
    TurnPayload,
)
from engram.models import RecallFact, RecallPassage, RecallResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passage(text: str, score: float = 1.0) -> RecallPassage:
    return RecallPassage(text=text, granularity="sentence", score=score, node_id="n")


def _result(passages: tuple[RecallPassage, ...], facts: tuple[RecallFact, ...] = ()) -> RecallResult:
    return RecallResult(passages=passages, facts=facts)


def _input(
    *,
    gold: str,
    passages: tuple[RecallPassage, ...],
    facts: tuple[RecallFact, ...] = (),
    judged_correct: bool = False,
    qid: str = "q1",
    bucket: str | None = None,
) -> FailureInput:
    return FailureInput(
        question_id=qid,
        gold=gold,
        recall_result=_result(passages, facts),
        judged_correct=judged_correct,
        bucket=bucket,
    )


def _store_with_gold_terms(terms: tuple[str, ...], *, connect: bool = True) -> GraphStore:
    """Build a store where each gold term appears in a distinct N-gram payload.

    When ``connect`` is False, the N-grams are added without edges linking
    them — that is the graph_gap signal.
    """
    store = GraphStore(conversation_id="__instance__")
    store.add_node(
        "t1",
        labels=frozenset({LABEL_TURN}),
        payloads={LABEL_TURN: TurnPayload(memory_id="m1", text=" ".join(terms), speaker="u", timestamp=None)},
    )
    for i, term in enumerate(terms):
        nid = f"ng{i}"
        store.add_node(
            nid,
            labels=frozenset({LABEL_NGRAM}),
            payloads={LABEL_NGRAM: NgramPayload(
                normalized_text=term,
                surface_form=term,
                segment_id="s1",
                ngram_kind=NGRAM_KIND_NOUN_CHUNK,
                char_span=(0, len(term)),
            )},
        )
        if connect:
            store.add_edge(nid, "t1", EdgeAttrs(type=EDGE_PART_OF, weight=1.0))
    return store


# ---------------------------------------------------------------------------
# One fixture per enum arm
# ---------------------------------------------------------------------------


def test_correct_case_is_not_an_r15_miss() -> None:
    case = _input(
        gold="Paris",
        passages=(_passage("Something unrelated"),),
        judged_correct=True,
    )
    (out,) = classify_failures([case])
    assert out.kind is FailureKind.CORRECT


def test_extraction_miss_when_gold_absent_from_graph_and_passages() -> None:
    store = _store_with_gold_terms(("alice", "paris"))
    case = _input(
        gold="Eiffel Tower",
        passages=(_passage("Alice visits Paris"),),
    )
    (out,) = classify_failures([case], store=store)
    # "eiffel" / "tower" are not in any node payload → extraction_miss.
    assert out.kind is FailureKind.EXTRACTION_MISS


def test_retrieval_miss_when_gold_in_graph_but_not_in_passages() -> None:
    store = _store_with_gold_terms(("eiffel", "tower", "paris"))
    case = _input(
        gold="Eiffel Tower",
        passages=(_passage("Something completely unrelated"),),
    )
    (out,) = classify_failures([case], store=store)
    assert out.kind is FailureKind.RETRIEVAL_MISS


def test_partial_retrieval_when_some_but_not_enough_terms_land_in_passages() -> None:
    case = _input(
        gold="Paris France Germany",
        # Only 1 of 3 content terms surfaces → recall ≈ 0.33 < 0.5 default.
        passages=(_passage("visited Paris"),),
    )
    (out,) = classify_failures([case])
    assert out.kind is FailureKind.PARTIAL_RETRIEVAL


def test_output_miss_when_passages_cover_gold_but_facts_do_not() -> None:
    case = _input(
        gold="Alice Paris",
        passages=(_passage("Alice mentioned Paris last week"),),
        facts=(RecallFact(kind="co_occurrence", subject="bob", value="something else"),),
    )
    (out,) = classify_failures([case])
    assert out.kind is FailureKind.OUTPUT_MISS


def test_agent_miss_when_both_passages_and_facts_cover_gold() -> None:
    case = _input(
        gold="Alice Paris",
        passages=(_passage("Alice mentioned Paris last week"),),
        facts=(RecallFact(kind="co_occurrence", subject="alice", value="co-occurs with paris"),),
    )
    (out,) = classify_failures([case])
    assert out.kind is FailureKind.AGENT_MISS


def test_graph_gap_when_nodes_hit_but_no_edges_connect_them() -> None:
    # Two distinct nodes carry the two gold terms, but no edges wire them up.
    store = _store_with_gold_terms(("eiffel", "tower"), connect=False)
    case = _input(
        gold="Eiffel Tower",
        passages=(_passage("Something completely unrelated"),),
    )
    (out,) = classify_failures([case], store=store)
    assert out.kind is FailureKind.GRAPH_GAP


# ---------------------------------------------------------------------------
# Fallbacks / misc
# ---------------------------------------------------------------------------


def test_retrieval_miss_fallback_when_store_not_supplied() -> None:
    # Without a store, the classifier can't distinguish extraction vs
    # retrieval and falls through to retrieval_miss.
    case = _input(
        gold="Eiffel Tower",
        passages=(_passage("Something unrelated"),),
    )
    (out,) = classify_failures([case])
    assert out.kind is FailureKind.RETRIEVAL_MISS


def test_classify_output_sorted_by_qid_then_gold() -> None:
    a = _input(gold="B", passages=(), qid="q2")
    b = _input(gold="A", passages=(), qid="q1")
    out = classify_failures([a, b])
    assert [c.question_id for c in out] == ["q1", "q2"]


def test_partial_threshold_is_configurable() -> None:
    case = _input(
        gold="Paris France Germany",
        passages=(_passage("visited Paris and Germany"),),  # recall = 2/3
    )
    # Default threshold 0.5 → enough retrieved → output/agent miss path.
    (default_out,) = classify_failures([case])
    assert default_out.kind in (FailureKind.OUTPUT_MISS, FailureKind.AGENT_MISS)

    # Raise threshold to 0.9 → 2/3 now counts as partial_retrieval.
    (strict_out,) = classify_failures([case], partial_threshold=0.9)
    assert strict_out.kind is FailureKind.PARTIAL_RETRIEVAL


def test_bucket_breakdown_aggregates_counts_per_bucket() -> None:
    cases = [
        _input(gold="x", passages=(), qid="q1", bucket="temporal"),
        _input(gold="x", passages=(), qid="q2", bucket="temporal", judged_correct=True),
        _input(gold="x", passages=(), qid="q3", bucket="preference"),
    ]
    classified = classify_failures(cases)
    report = bucket_breakdown(classified)
    buckets = dict(report.buckets)

    temporal_counts = dict(buckets["temporal"])
    assert temporal_counts[FailureKind.CORRECT.value] == 1
    assert temporal_counts[FailureKind.RETRIEVAL_MISS.value] == 1
    preference_counts = dict(buckets["preference"])
    assert preference_counts[FailureKind.RETRIEVAL_MISS.value] == 1

    totals = dict(report.totals)
    assert totals[FailureKind.RETRIEVAL_MISS.value] == 2
    assert totals[FailureKind.CORRECT.value] == 1

    assert report.total_cases == 3
    assert report.total_correct == 1


def test_bucket_breakdown_handles_none_bucket() -> None:
    cases = [_input(gold="x", passages=(), qid="q1")]
    classified = classify_failures(cases)
    report = bucket_breakdown(classified)
    assert dict(report.buckets).get("(none)") is not None


def test_classify_failures_is_deterministic() -> None:
    store = _store_with_gold_terms(("paris", "france"))
    case = _input(gold="Paris France", passages=(_passage("unrelated"),))
    a = classify_failures([case], store=store)
    b = classify_failures([case], store=store)
    assert a == b
