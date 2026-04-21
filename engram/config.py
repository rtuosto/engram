"""``MemoryConfig`` — the config object behind every fingerprint.

``P6`` — every artifact has a fingerprint; no fingerprint, no cache.
``R3`` — ``ingestion_fingerprint`` includes every config field that affects
node / edge production; a config diff without a fingerprint diff is a bug.
``R4`` — ``answer_fingerprint`` transitively includes ``ingestion_fingerprint``;
any ingest change invalidates every downstream answer.

**Discipline.** Every declared field must belong to exactly one of
:attr:`MemoryConfig._INGESTION_FIELDS` or :attr:`MemoryConfig._ANSWER_FIELDS`.
A field without a category is a bug caught by the class invariant check in
``__post_init__`` — and redundantly by ``test_fingerprint_discipline.py``, which
fails CI.

The partition:

- **Ingestion fields** affect graph production. Changing one invalidates the
  ingested graph and (by ``R4``) every cached answer.
- **Answer fields** affect the answer but not the graph. Changing one
  invalidates cached answers; ingested graphs remain valid.

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


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Frozen configuration object. Fingerprints are derived from field partitions."""

    # --- Ingestion fields -------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    preference_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    preference_seed_hash: str = _PREFERENCE_SEED_HASH
    canonicalization_match_threshold: float = 0.85
    preference_discrimination_margin: float = 0.05
    claim_subject_required: bool = True
    ngram_min_tokens: int = 2
    random_seed: int = 0

    # --- Answer fields ----------------------------------------------------
    answerer_model: str = "ollama:llama3.1:8b"
    answerer_temperature: float = 0.0
    recall_top_k: int = 10
    context_char_budget: int = 16000

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
    })
    _ANSWER_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "answerer_model",
        "answerer_temperature",
        "recall_top_k",
        "context_char_budget",
    })

    def __post_init__(self) -> None:
        """Class invariant: every declared field is categorized exactly once."""
        declared = {f.name for f in fields(self)}
        categorized = self._INGESTION_FIELDS | self._ANSWER_FIELDS
        overlap = self._INGESTION_FIELDS & self._ANSWER_FIELDS
        missing = declared - categorized
        extra = categorized - declared

        if overlap:
            raise RuntimeError(
                f"MemoryConfig partition is not disjoint — fields in both "
                f"_INGESTION_FIELDS and _ANSWER_FIELDS: {sorted(overlap)}"
            )
        if missing:
            raise RuntimeError(
                f"MemoryConfig field(s) not categorized — add to "
                f"_INGESTION_FIELDS or _ANSWER_FIELDS: {sorted(missing)} (R3)"
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

    def answer_fingerprint(self) -> str:
        """SHA-256 of the answer-affecting fields + ingestion fingerprint (``R4``).

        Transitivity is realized by including ``_ingestion_fingerprint`` as a
        key in the answer payload. Any ingestion change → different ingestion
        fingerprint → different answer fingerprint, invalidating every cached
        answer.
        """
        payload = self._extract(self._ANSWER_FIELDS)
        payload["_ingestion_fingerprint"] = self.ingestion_fingerprint()
        return _hash(payload)

    def _extract(self, field_names: frozenset[str]) -> dict[str, object]:
        full = asdict(self)
        return {name: full[name] for name in sorted(field_names)}


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
