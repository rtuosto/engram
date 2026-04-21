"""engram.recall — query → structured ``RecallResult``.

**Responsibility.** Given a query string and a corpus, produce a ranked
:class:`engram.models.RecallResult` of passages plus pre-computed facts
from the derived indexes. Engram makes zero LLM calls on this path
(``R5``, ``R13``).

**Pipeline (``docs/design/recall.md §3``).** Five stages:

1. :mod:`engram.recall.intent` — prototype-centroid intent classification.
2. :mod:`engram.recall.seeding` — semantic (vector index) + entity-anchored.
3. :mod:`engram.recall.expansion` — bounded typed-edge BFS.
4. :mod:`engram.recall.scoring` — per-granule selection.
5. :mod:`engram.recall.assembly` — passage + fact composition.

:class:`engram.recall.pipeline.RecallPipeline` orchestrates the stages.
Construct in tests with hand-built centroids and embed callables; call
:func:`engram.recall.factory.build_default_recall_pipeline` for the
production wiring (lazy spaCy + sentence-transformers load).

**Public surface.** Reached via :class:`engram.EngramGraphMemorySystem`'s
``recall`` verb. Interior modules — :mod:`.intent`, :mod:`.seeding`,
:mod:`.expansion`, :mod:`.scoring`, :mod:`.assembly`, :mod:`.pipeline`,
:mod:`.context`, :mod:`.intents`, :mod:`.weights` — are not part of the
external contract.

**Owns.** Intent classification (``R6``: prototype-embedding centroids,
never English regex), seeding, subgraph expansion, scoring, result
assembly, intent seed + weight fixtures (``engram/recall/intents/*.json``,
``engram/recall/weights.json``).

**Does not touch.** Node / edge creation, benchmark scoring, judge
prompts, the ingestion fingerprint. The answerer agent and its LLM live
in the external ``agent-memory-benchmark`` repo (``R13``).

**Stability guarantee.** ``recall_fingerprint`` transitively includes
``ingestion_fingerprint`` plus every recall-side config field + the query
+ :class:`RecallContext` (``R4``).

Retrieval returns a structured result — passages plus pre-computed facts —
not a flat list (``R9``). Temporal / arithmetic computation happens here
or at ingest, never in the answer prompt (``P4``, ``R8``).
"""
