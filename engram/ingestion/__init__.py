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

**Owns (primary).** Memory + Turn granule emission, segmentation, N-gram
extraction, NER, entity canonicalization, claim + preference extraction,
granule embeddings, TimeAnchor nodes + ``temporal_at`` edges, and the
derived-rebuild orchestrator (alias sets, co-occurrence, reinforcement
counts, current-truth, TimeAnchor chain). Episodic clusters + ChangeEvents
remain deferred per ``docs/design/ingestion.md §7 D5–D6``.

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
