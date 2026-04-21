"""engram.ingestion — conversation stream → populated graph with deterministic fingerprint.

**Responsibility.** Convert a conversation stream into a populated graph with a
deterministic fingerprint.

**Public surface.** Reached via :class:`engram.EngramGraphMemorySystem`, which
implements the ``MemorySystem`` protocol verbs (``ingest_session``,
``finalize_conversation``, ``save_state``, ``load_state``, ``reset``). Interior
modules — :mod:`.graph`, :mod:`.persist`, :mod:`.pipeline`, :mod:`.schema`,
:mod:`.extractors`, :mod:`.preferences` — are not part of the external contract.

**Owns (Tier 1).** Segmentation, NER, entity canonicalization, claim + preference
extraction, temporal + co-occurrence edges, ingestion fingerprinting. Event /
Episode extraction and Tier-3 semantic edges (``supports`` / ``contradicts`` /
``refers_back_to``) are deferred to later design-doc iterations — see
``docs/design/ingestion.md §2``.

**Does not touch.** Query text, answer generation, benchmark orchestration (lives
in the external ``agent-memory-benchmark`` repo), judge prompts, cache file layout.

**Stability guarantee.** The ingestion fingerprint fully covers output state:
identical fingerprint ⇒ identical state. Violating this guarantee is a bug
(``docs/DESIGN-MANIFESTO.md §R2, §R3``).

No LLM calls in the default path (``R5``). Any LLM-based enhancement is opt-in,
flag-gated, batched per conversation, and budget-capped — and the non-LLM path
must always be fully functional on its own.
"""
