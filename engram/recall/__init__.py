"""engram.recall — question + graph → ranked subgraph + context + one answerer call.

**Responsibility.** Given a question and a graph, produce a ranked subgraph, an
assembled context string, and one answer call.

**Public verbs.** ``classify_intent``, ``seed``, ``expand``, ``rank``,
``assemble_context``, ``answer``.

**Owns.** Intent classification (``R6``: prototype-embedding centroids, never
English regex), seeding, subgraph expansion, reranking, context assembly, answer
prompt template (``R13``: one file, template changes reviewed like API changes).

**Does not touch.** Node / edge creation, benchmark scoring (lives in the
external ``agent-memory-benchmark`` repo), judge prompts, the ingestion
fingerprint.

**Stability guarantee.** ``answer_fingerprint`` transitively includes
``ingestion_fingerprint`` plus every recall-side config field (``R4``).

Retrieval returns a subgraph — nodes plus justifying edges — not a flat list
(``R9``). Context assembly preserves locality by default (``R11``); pruning
requires oracle-tested justification on the affected bucket (``P3``).

Temporal / arithmetic computation happens here or at ingest, never in the answer
prompt (``P4``, ``R8``): the answerer relays literals, it does not execute
procedures.
"""
