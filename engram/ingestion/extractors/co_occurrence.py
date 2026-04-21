"""Stage [6] — Co-occurrence.

Per-conversation entity-pair counter. Each Turn contributes one count per
unordered pair of distinct canonical Entity nodes mentioned in that Turn.
At :meth:`finalize_conversation` the pipeline emits bidirectional
``co_occurs_with`` edges with weight normalized to ``[0, 1]`` against the
max pair-count in the conversation — per ``docs/design/ingestion.md §3 [6]``.

**Pruning.** No hard count floor in this stage; the recall-time weight
optimizer drives thin edges toward zero in the seeding/expansion weight
vector.
"""

from __future__ import annotations

from collections import Counter

from engram.ingestion.schema import EDGE_CO_OCCURS_WITH, EdgeAttrs

# Unordered pair key is a sorted tuple of two entity_ids. ``Counter`` keeps
# the add / mutate ergonomics but we normalize keys explicitly to guarantee
# R2 — iteration over a Counter preserves insertion order, but we're going
# to sort before emit anyway.
PairKey = tuple[str, str]


def pair_key(a: str, b: str) -> PairKey:
    return (a, b) if a < b else (b, a)


class CoOccurrenceCounter:
    """Accumulates per-conversation pair counts."""

    def __init__(self) -> None:
        self._counter: Counter[PairKey] = Counter()

    def observe_turn(self, entity_ids_in_turn: list[str]) -> None:
        """Increment the pair counter for every unordered pair in the Turn.

        Deduplicates within-turn so a Turn mentioning "Alice" twice doesn't
        inflate her pair counts against other mentioned entities.
        """
        unique = sorted(set(entity_ids_in_turn))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                self._counter[pair_key(unique[i], unique[j])] += 1

    def max_count(self) -> int:
        return max(self._counter.values(), default=0)

    def edges(
        self, source_turn_id: str | None = None
    ) -> list[tuple[str, str, EdgeAttrs]]:
        """Emit bidirectional ``co_occurs_with`` edges, sorted for R2.

        Returns a sorted list of ``(src, dst, attrs)`` with *both* directions
        present (src<dst and src>dst). Weight is ``count / max_count``.

        ``source_turn_id`` is not generally meaningful for an aggregate edge
        — it's left ``None`` by default, per the §2 "drop the field on
        aggregate edges" open question.
        """
        if not self._counter:
            return []
        peak = self.max_count()
        if peak <= 0:
            return []

        emitted: list[tuple[str, str, EdgeAttrs]] = []
        for (a, b), count in self._counter.items():
            weight = count / peak
            attrs = EdgeAttrs(
                type=EDGE_CO_OCCURS_WITH,
                weight=weight,
                source_turn_id=source_turn_id,
                asserted_at=None,
            )
            emitted.append((a, b, attrs))
            emitted.append((b, a, attrs))
        emitted.sort(key=lambda t: (t[0], t[1]))
        return emitted


__all__ = ["CoOccurrenceCounter", "pair_key"]
