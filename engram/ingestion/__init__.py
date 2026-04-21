"""engram.ingestion — Memory stream → populated graph with deterministic fingerprint.

**Responsibility.** Convert a stream of :class:`engram.Memory` observations
into a populated graph + parallel vector index (later PRs) with a
deterministic fingerprint.

**Public surface.** Reached via :class:`engram.EngramGraphMemorySystem`,
which implements the ``MemorySystem`` protocol verbs (``ingest``,
``recall``, ``save_state``, ``load_state``, ``reset``). Interior modules —
:mod:`.graph`, :mod:`.persist`, :mod:`.pipeline`, :mod:`.schema`,
:mod:`.extractors`, :mod:`.preferences` — are not part of the external
contract.

**Owns (primary).** Memory + Turn granule emission, segmentation, NER,
entity canonicalization, claim + preference extraction, ingestion
fingerprinting. N-gram extraction (PR-B), granule embeddings (PR-C),
TimeAnchor nodes (PR-D), and derived-index rebuilds — co-occurrence, alias
sets, reinforcement counts, current-truth, episodic clusters — land in
later patches per ``docs/design/ingestion.md §12``.

**Does not touch.** Query text, answer generation, benchmark orchestration
(lives in the external ``agent-memory-benchmark`` repo), judge prompts,
cache file layout.

**Stability guarantee.** The ingestion fingerprint fully covers output
state: identical fingerprint ⇒ identical state. Violating this guarantee
is a bug (``docs/DESIGN-MANIFESTO.md §R2, §R3``).

**R5.** No LLM calls on the ingest path. Any LLM-based enhancement would
be opt-in, flag-gated, batched, and budget-capped — and the non-LLM path
must always be fully functional on its own.
"""
