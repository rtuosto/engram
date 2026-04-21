"""Deterministic extractors — one stage per module.

Each extractor is a pure function (or a stateless class) that consumes its
stage's inputs and emits the nodes / edges for its stage. Stage order is
fixed by ``docs/design/ingestion.md §3``:

1. :mod:`.segmentation` — Turn.text → UtteranceSegmentPayload
2. :mod:`.ner` — spaCy Doc → entity mentions (text spans, not yet nodes)
3. :mod:`.canonicalization` — mentions → Entity node_ids (merge or create)
4. :mod:`.claim` — dependency parse → ClaimPayload
5. :mod:`.preference` — Claim + embedding centroids → PreferencePayload (fails closed)
6. :mod:`.co_occurrence` — pairwise entity counts → co_occurs_with edges (at finalize)

All extractors are R2-deterministic for a fixed spaCy / embedding model.
Model choice lives in :class:`engram.config.MemoryConfig`; the
``ingestion_fingerprint`` covers it transitively via ``R3``.
"""
