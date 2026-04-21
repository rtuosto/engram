"""Deterministic extractors — one stage per module.

Each extractor is a pure function (or a stateless class) that consumes its
stage's inputs and emits the nodes / edges for its stage. Stage order is
fixed by ``docs/design/ingestion.md §6``:

1. :mod:`.segmentation` — Turn.text → UtteranceSegmentPayload
2. :mod:`.ngram` — Doc + segment spans → NgramPayload (noun_chunk + SVO)
3. :mod:`.ner` — spaCy Doc → entity mentions (text spans, not yet nodes)
4. :mod:`.canonicalization` — mentions → Entity node_ids (merge or create)
5. :mod:`.claim` — dependency parse → ClaimPayload
6. :mod:`.preference` — Sentence + embedding centroids → Preference node (fails closed)

Co-occurrence, alias sets, and reinforcement counts are **derived** (R17),
rebuilt from primary by :mod:`engram.ingestion.derived` (PR-D), not emitted
at ingest time.

All extractors are R2-deterministic for a fixed spaCy / embedding model.
Model choice lives in :class:`engram.config.MemoryConfig`; the
``ingestion_fingerprint`` covers it transitively via ``R3``.
"""
