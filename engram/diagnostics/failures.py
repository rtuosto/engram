"""R15 failure classification + per-bucket aggregation.

Consumes ``FailureInput`` records (one per benchmark question) and emits
one :class:`FailureCase` per input, tagged with a :class:`FailureKind`
enum value.

**The R15 decision tree** (``docs/DESIGN-MANIFESTO.md §4``):

1. ``overlap(gold, passages) == 0``

   a. ``overlap(gold, any node payload) == 0`` → ``extraction_miss``.
      Engram never learned it from the input stream.

   b. ``overlap(gold, any node payload) > 0`` → ``retrieval_miss``.
      It's in the graph; recall didn't seed / walk to it.

2. ``0 < overlap(gold, passages) < partial_threshold`` →
   ``partial_retrieval``. Some content reached the agent, but not
   enough for a confident answer.

3. ``overlap(gold, passages) >= partial_threshold`` AND
   ``judged_correct == False``:

   a. ``overlap(gold, facts) == 0`` → ``output_miss``. Right content
      retrieved, but the structured ``facts`` tuple didn't surface the
      pre-computed entry that would have answered the question.

   b. ``overlap(gold, facts) > 0`` → ``agent_miss``. The agent had
      everything it needed and still answered wrong.

4. ``graph_gap`` — emitted when gold terms appear across multiple node
   payloads but no connecting edge (``mentions`` / ``asserts`` /
   ``holds_preference`` / ``part_of``) joins them. This is an
   ingestion-layer bug: the primitives landed but the structural links
   didn't. Signals that a derived-index rebuild can't recover from
   missing primary edges.

**Correctly-answered cases.** When ``judged_correct`` is ``True`` the
classifier emits :attr:`FailureKind.CORRECT` — included in the output so
bucket aggregation can report counts per bucket without losing the
denominators.

**Thresholds.** ``partial_threshold`` defaults to ``0.5`` (half the gold
terms present). Callers tuning the classifier for a tighter / looser
buckets can override per-call.

**R2.** Output tuples are sorted by ``question_id`` (when provided) and
then by ``gold`` for ties — byte-stable across caller iteration order.

**Read-only.** This module never mutates its inputs. ``store`` is
inspected via the read-only ``iter_nodes`` / ``iter_edges`` surface.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Any, Final

from engram.diagnostics.overlap import Overlap, extract_key_terms, needle_overlap
from engram.ingestion.graph import GraphStore
from engram.models import RecallFact, RecallResult


class FailureKind(StrEnum):
    """R15 enum. Values are stable identifiers used in reports.

    ``CORRECT`` is not part of R15 — it's added as a neutral bucket for
    judged-correct questions so aggregation produces complete per-bucket
    counts without a second pass over the input.
    """

    EXTRACTION_MISS = "extraction_miss"
    GRAPH_GAP = "graph_gap"
    RETRIEVAL_MISS = "retrieval_miss"
    PARTIAL_RETRIEVAL = "partial_retrieval"
    OUTPUT_MISS = "output_miss"
    AGENT_MISS = "agent_miss"
    CORRECT = "correct"


DEFAULT_PARTIAL_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class FailureInput:
    """One benchmark question + engram's output + the judge verdict.

    ``question_id`` is a stable identifier the caller controls; empty
    string is allowed but makes sort order gold-only. ``gold`` is the
    canonical answer string. ``bucket`` is a caller-defined key (for
    LongMemEval, this is ``question_type``). ``judged_correct`` is the
    judge's binary verdict.

    ``recall_result`` is engram's structured output for the question.
    ``generated_answer`` is the agent's final answer — retained for
    report rendering but not inspected by the classifier itself (the
    judge verdict is the source of truth).
    """

    question_id: str
    gold: str
    recall_result: RecallResult
    judged_correct: bool
    generated_answer: str = ""
    bucket: str | None = None


@dataclass(frozen=True, slots=True)
class FailureCase:
    """One classified case. ``kind`` is the R15 bucket; ``explanation`` is
    a short human-readable string suitable for a report row.

    ``passage_overlap`` / ``node_overlap`` / ``fact_overlap`` are kept so
    downstream consumers (e.g. PR-G's calibrator) can use the raw
    signals rather than re-deriving them.
    """

    question_id: str
    gold: str
    kind: FailureKind
    explanation: str
    passage_overlap: Overlap
    node_overlap: Overlap
    fact_overlap: Overlap
    bucket: str | None = None


@dataclass(frozen=True, slots=True)
class BucketReport:
    """Per-bucket aggregation of classified cases.

    ``buckets`` is a sorted tuple of ``(bucket_key, counts)`` where
    ``counts`` is a sorted tuple of ``(kind_value, count)``. ``totals``
    is the same shape but summed across buckets.
    """

    buckets: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    totals: tuple[tuple[str, int], ...]
    total_cases: int = 0
    total_correct: int = 0


def classify_failures(
    cases: Iterable[FailureInput],
    store: GraphStore | None = None,
    *,
    partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
) -> tuple[FailureCase, ...]:
    """Classify each input into an R15 bucket.

    ``store`` is optional — when omitted, the ``extraction_miss`` /
    ``retrieval_miss`` split collapses to ``retrieval_miss`` (we can't
    tell whether the content is in the graph without looking). Callers
    that can supply the store should, since that's the most actionable
    split.

    See module docstring for the decision tree.
    """
    if not 0.0 <= partial_threshold <= 1.0:
        raise ValueError(
            f"partial_threshold must be in [0, 1]; got {partial_threshold}"
        )

    # Pre-compute the store's node payload text once per call; the
    # classifier consults it per-question but the corpus itself is
    # stable.
    node_haystack = _collect_node_haystack(store) if store is not None else ""

    out: list[FailureCase] = []
    for case in cases:
        passage_text = "\n".join(p.text for p in case.recall_result.passages)
        fact_text = "\n".join(_render_fact(f) for f in case.recall_result.facts)

        p_overlap = needle_overlap(case.gold, passage_text)
        f_overlap = needle_overlap(case.gold, fact_text)
        n_overlap = needle_overlap(case.gold, node_haystack) if store is not None else p_overlap

        kind, explanation = _classify_one(
            case=case,
            passage_overlap=p_overlap,
            node_overlap=n_overlap,
            fact_overlap=f_overlap,
            store=store,
            partial_threshold=partial_threshold,
        )
        out.append(
            FailureCase(
                question_id=case.question_id,
                gold=case.gold,
                kind=kind,
                explanation=explanation,
                passage_overlap=p_overlap,
                node_overlap=n_overlap,
                fact_overlap=f_overlap,
                bucket=case.bucket,
            )
        )

    out.sort(key=lambda c: (c.question_id, c.gold))
    return tuple(out)


def bucket_breakdown(
    cases: Iterable[FailureCase],
    *,
    bucket_key: str = "bucket",
) -> BucketReport:
    """Aggregate classified cases per bucket.

    ``bucket_key`` selects which attribute of :class:`FailureCase` is
    the grouping key; ``"bucket"`` is the default (matches the field on
    :class:`FailureInput`). Cases whose bucket is ``None`` are grouped
    under the literal string ``"(none)"`` so the report has no silent
    None keys.
    """
    materialized = list(cases)
    per_bucket: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    total_cases = 0
    total_correct = 0

    for c in materialized:
        bucket = getattr(c, bucket_key, None)
        label = bucket if isinstance(bucket, str) and bucket else "(none)"
        per_bucket[label][c.kind.value] += 1
        totals[c.kind.value] += 1
        total_cases += 1
        if c.kind is FailureKind.CORRECT:
            total_correct += 1

    buckets_sorted = tuple(
        (bucket, tuple(sorted(counts.items())))
        for bucket, counts in sorted(per_bucket.items())
    )
    return BucketReport(
        buckets=buckets_sorted,
        totals=tuple(sorted(totals.items())),
        total_cases=total_cases,
        total_correct=total_correct,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _classify_one(
    *,
    case: FailureInput,
    passage_overlap: Overlap,
    node_overlap: Overlap,
    fact_overlap: Overlap,
    store: GraphStore | None,
    partial_threshold: float,
) -> tuple[FailureKind, str]:
    if case.judged_correct:
        return FailureKind.CORRECT, f"recall={passage_overlap.recall:.0%}"

    # Arm 1 — no gold terms in recall output.
    if passage_overlap.recall == 0.0:
        if store is None:
            return (
                FailureKind.RETRIEVAL_MISS,
                "no overlap with passages; store not supplied to split extraction vs retrieval",
            )
        if node_overlap.recall == 0.0:
            return (
                FailureKind.EXTRACTION_MISS,
                f"none of {len(passage_overlap.terms)} gold terms present in any node payload",
            )
        # Content IS in the graph. Check whether any edge connects nodes
        # carrying gold terms — if there are hits but they are
        # disconnected, that's a graph_gap; otherwise retrieval_miss.
        if _has_graph_gap(case.gold, store):
            return (
                FailureKind.GRAPH_GAP,
                "gold terms present in nodes but no connecting edge found",
            )
        return (
            FailureKind.RETRIEVAL_MISS,
            f"gold terms in graph ({len(node_overlap.found)}/{len(node_overlap.terms)}) "
            "but not retrieved",
        )

    # Arm 2 — some but not enough.
    if passage_overlap.recall < partial_threshold:
        return (
            FailureKind.PARTIAL_RETRIEVAL,
            f"passage recall {passage_overlap.recall:.0%} below {partial_threshold:.0%}",
        )

    # Arm 3 — enough retrieved but judged wrong.
    if fact_overlap.recall == 0.0:
        return (
            FailureKind.OUTPUT_MISS,
            f"passage recall {passage_overlap.recall:.0%} but no gold terms in facts",
        )
    return (
        FailureKind.AGENT_MISS,
        f"passage recall {passage_overlap.recall:.0%} and fact overlap "
        f"{fact_overlap.recall:.0%} — agent had what it needed",
    )


def _collect_node_haystack(store: GraphStore) -> str:
    """Concatenate every node payload's string-valued fields."""
    parts: list[str] = []
    for _node_id, attrs in store.iter_nodes():
        for value in attrs.values():
            parts.extend(_render_value(value))
    return "\n".join(parts)


def _render_value(value: Any) -> list[str]:
    """Flatten a node-attrs value into string fragments.

    Dataclass payloads (EntityPayload, NgramPayload, ClaimPayload, …)
    contribute each of their string-typed fields. ``frozenset`` /
    ``list`` / ``tuple`` values are flattened recursively.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)) or value is None:
        return []
    if isinstance(value, (frozenset, set, list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_render_value(item))
        return out
    if is_dataclass(value) and not isinstance(value, type):
        out = []
        for f in fields(value):
            out.extend(_render_value(getattr(value, f.name)))
        return out
    return []


def _render_fact(fact: RecallFact) -> str:
    """Stringify a RecallFact for overlap scanning."""
    parts = [fact.kind, fact.subject, fact.value]
    if fact.predicate:
        parts.append(fact.predicate)
    if fact.object:
        parts.append(fact.object)
    if fact.timestamp:
        parts.append(fact.timestamp)
    return " ".join(parts)


def _has_graph_gap(gold: str, store: GraphStore) -> bool:
    """Heuristic: multiple nodes carry gold terms but no edge connects any pair.

    A "graph gap" is the specific failure mode where ingestion emitted
    the right primitives but the edge extractor didn't link them. We
    detect it conservatively: if two or more distinct nodes carry
    overlapping gold terms yet the subgraph induced by those nodes has
    zero edges, we call it a gap. Single-node-hit cases fall through to
    ``retrieval_miss``.
    """
    terms = extract_key_terms(gold)
    if not terms:
        return False

    hit_nodes: list[str] = []
    for node_id, attrs in store.iter_nodes():
        payload_parts: list[str] = []
        for value in attrs.values():
            payload_parts.extend(_render_value(value))
        if _has_any_term(" ".join(payload_parts), terms):
            hit_nodes.append(node_id)

    if len(hit_nodes) < 2:
        return False

    hit_set = set(hit_nodes)
    for src, dst, _edge_type, _attrs in store.iter_edges():
        if src in hit_set and dst in hit_set:
            return False
    return True


def _has_any_term(text: str, terms: Iterable[str]) -> bool:
    hay = unicodedata.normalize("NFKC", text).casefold()
    return any(term in hay for term in terms)


__all__ = [
    "BucketReport",
    "DEFAULT_PARTIAL_THRESHOLD",
    "FailureCase",
    "FailureInput",
    "FailureKind",
    "bucket_breakdown",
    "classify_failures",
]
