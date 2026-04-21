"""engram.ingestion — conversation stream → populated graph with deterministic fingerprint.

**Responsibility.** Convert a conversation stream into a populated graph with a
deterministic fingerprint.

**Public verbs.** ``ingest_session``, ``finalize_conversation``, ``export_state``,
``import_state``, ``fingerprint``.

**Owns.** Segmentation, NER, entity canonicalization, claim / preference / event
extraction, temporal resolution, edge construction, episode detection,
corpus-signal derivation, ingestion fingerprinting.

**Does not touch.** Query text, answer generation, benchmark orchestration (lives
in the external ``agent-memory-benchmark`` repo), judge prompts, cache file layout.

**Stability guarantee.** The ingestion fingerprint fully covers output state:
identical fingerprint ⇒ identical state. Violating this guarantee is a bug
(``docs/DESIGN-MANIFESTO.md §R2, §R3``).

No LLM calls in the default path (``R5``). Any LLM-based enhancement is opt-in,
flag-gated, batched per conversation, and budget-capped — and the non-LLM path
must always be fully functional on its own.
"""
