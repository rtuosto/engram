"""``MemoryConfig`` — the config object behind every fingerprint.

``P6`` — every artifact has a fingerprint; no fingerprint, no cache.
``R3`` — ``ingestion_fingerprint`` includes every config field that affects
node / edge production; a config diff without a fingerprint diff is a bug.
``R4`` — ``recall_fingerprint`` transitively includes
``ingestion_fingerprint``; any ingest change invalidates every downstream
recall result.

**Discipline.** Every declared field must belong to exactly one of
:attr:`MemoryConfig._INGESTION_FIELDS` or :attr:`MemoryConfig._RECALL_FIELDS`.
A field without a category is a bug caught by the class invariant check in
``__post_init__`` — and redundantly by ``test_fingerprint_discipline.py``, which
fails CI.

The partition:

- **Ingestion fields** affect graph production. Changing one invalidates the
  ingested graph and (by ``R4``) every cached recall result.
- **Recall fields** affect recall output but not the graph. Changing one
  invalidates cached recall results; ingested graphs remain valid. The
  answerer (model / temperature) no longer lives here — it belongs to the
  external benchmark harness (``docs/design/recall.md §11``).

Cache layout / log verbosity / other non-semantic knobs go in a separate
``RuntimeOptions`` object (not here). If you find yourself reaching for a
"metadata" category inside ``MemoryConfig``, you're almost certainly adding a
runtime option, not a config field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import ClassVar

from engram.ingestion.preferences import SEED_HASH as _PREFERENCE_SEED_HASH
from engram.recall.intents import INTENT_SEED_HASH as _INTENT_SEED_HASH
from engram.recall.weights import WEIGHTS_HASH as _RECALL_WEIGHTS_HASH


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Frozen configuration object. Fingerprints are derived from field partitions."""

    # --- Ingestion fields -------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    preference_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    preference_seed_hash: str = _PREFERENCE_SEED_HASH
    canonicalization_match_threshold: float = 0.85
    preference_discrimination_margin: float = 0.05
    claim_subject_required: bool = True
    ngram_min_tokens: int = 2
    random_seed: int = 0
    # Rounding granularity for TimeAnchor node identity (PR-D).
    # One of "second" | "minute" | "hour" | "day". Coarser resolutions
    # collapse more observations onto the same anchor, trading precision for
    # a smaller temporal layer.
    time_anchor_resolution: str = "second"

    # --- Recall fields ----------------------------------------------------
    # Fingerprint-tracked content hashes over hand-authored fixture files
    # (seeds.json, weights.json). Editing a fixture invalidates
    # recall_fingerprint transitively.
    intent_seed_hash: str = _INTENT_SEED_HASH
    recall_weights_hash: str = _RECALL_WEIGHTS_HASH
    intent_discrimination_margin: float = 0.05
    recall_max_depth: int = 3
    recall_max_frontier: int = 256
    recall_max_passages: int = 16
    recall_seed_count_total: int = 64
    # Per-granularity seed budget before the intent weight profile is applied.
    # Single int rather than a table — §5 granularity weights scale this.
    recall_top_n_per_granularity: int = 12

    # --- Partition (must cover every field declared above) ---------------
    _INGESTION_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "embedding_model",
        "spacy_model",
        "preference_embedding_model",
        "preference_seed_hash",
        "canonicalization_match_threshold",
        "preference_discrimination_margin",
        "claim_subject_required",
        "ngram_min_tokens",
        "random_seed",
        "time_anchor_resolution",
    })
    _RECALL_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "intent_seed_hash",
        "recall_weights_hash",
        "intent_discrimination_margin",
        "recall_max_depth",
        "recall_max_frontier",
        "recall_max_passages",
        "recall_seed_count_total",
        "recall_top_n_per_granularity",
    })

    def __post_init__(self) -> None:
        """Class invariant: every declared field is categorized exactly once."""
        declared = {f.name for f in fields(self)}
        categorized = self._INGESTION_FIELDS | self._RECALL_FIELDS
        overlap = self._INGESTION_FIELDS & self._RECALL_FIELDS
        missing = declared - categorized
        extra = categorized - declared

        if overlap:
            raise RuntimeError(
                f"MemoryConfig partition is not disjoint — fields in both "
                f"_INGESTION_FIELDS and _RECALL_FIELDS: {sorted(overlap)}"
            )
        if missing:
            raise RuntimeError(
                f"MemoryConfig field(s) not categorized — add to "
                f"_INGESTION_FIELDS or _RECALL_FIELDS: {sorted(missing)} (R3)"
            )
        if extra:
            raise RuntimeError(
                f"MemoryConfig partition references non-existent field(s): "
                f"{sorted(extra)}"
            )

    def ingestion_fingerprint(self) -> str:
        """SHA-256 of the ingestion-affecting fields, truncated to 16 hex chars.

        Determinism (``R2``): keys are sorted; values are JSON-serializable.
        If you add a field that resists JSON serialization, convert it to a
        stable string representation here — do not skip it (``R3``).
        """
        return _hash(self._extract(self._INGESTION_FIELDS))

    def recall_fingerprint(self) -> str:
        """SHA-256 of the recall-affecting fields + ingestion fingerprint (``R4``).

        Transitivity is realized by including ``_ingestion_fingerprint`` as a
        key in the recall payload. Any ingestion change → different ingestion
        fingerprint → different recall fingerprint, invalidating every cached
        recall result.
        """
        payload = self._extract(self._RECALL_FIELDS)
        payload["_ingestion_fingerprint"] = self.ingestion_fingerprint()
        return _hash(payload)

    def _extract(self, field_names: frozenset[str]) -> dict[str, object]:
        full = asdict(self)
        return {name: full[name] for name in sorted(field_names)}


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
