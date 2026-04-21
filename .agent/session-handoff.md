# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

---

## Session: 2026-04-20 — engram foundations + Tier-1 ingestion

### What Was Done

- Cloned `github.com/rtuosto/engram` into `~/code/engram`.
- Ran `~/code/agent-bootstrap/setup.sh`: CLAUDE.md, .cursorrules, .agent/, docs/ARCHITECTURE.md, .gitignore installed.
- Created feature branch `feat/engram-foundations`.
- Wrote `docs/DESIGN-MANIFESTO.md` — the binding architectural contract. Approved by user 2026-04-20.
- Wrote `docs/ARCHITECTURE.md` — technical map.
- Committed the foundations (`b842b22`), module scaffolds (`1d7a263`), `MemorySystem` protocol (`d2b06c6`), `MemoryConfig` + fingerprint-discipline CI gate (`5cb494c`).
- Seeded `.agent/lessons.md` with 10 hard-won lessons ported from the predecessor.
- **2026-04-20 (later):** Removed `engram/benchmarking/` — benchmarking lives in a separate `agent-memory-benchmark` repo and consumes engram through the `MemorySystem` protocol. Three modules: `ingestion`, `recall`, `diagnostics`.
- **2026-04-20 (later still):** Wrote `docs/design/ingestion.md` — the complete Tier-1 ingestion design (graph storage, schema, pipeline, model choices, preference validation, canonicalization, operational policy, deps/tests). Merged to `main`.
- **2026-04-20 (Tier-1 implementation):** Shipped `feat/ingestion-tier1`:
  - `engram/ingestion/schema.py` — content-addressed node IDs + frozen dataclass payloads (Turn, UtteranceSegment, Entity, Claim, Preference, Event, Episode, Session) + EdgeAttrs + Tier-1 edge constants.
  - `engram/ingestion/graph.py` — `GraphStore` over `networkx.MultiDiGraph` with R2-sorted iteration, multi-labeling, freeze semantics, typed-edge BFS.
  - `engram/ingestion/persist.py` — schema-versioned msgpack serializer; `SCHEMA_VERSION=1`; `SchemaVersionMismatch` / `PersistFormatError` on drift.
  - `engram/ingestion/preferences/` — seed + held-out fixtures (72 seed + 42 held-out sentences across 6 polarities), `compute_centroids`, `median_discrimination_margin`, seed-hash fingerprint coupling.
  - `engram/ingestion/extractors/` — `segmentation`, `ner`, `canonicalization` (string-similarity via rapidfuzz, feature infra wired with 0 weights for embedding/co_occurrence), `claim` (SVO + first-person → speaker), `preference` (fails-closed classifier), `co_occurrence` (normalized pair counts).
  - `engram/ingestion/pipeline.py` — `IngestionPipeline` orchestrator + `ConversationState`.
  - `engram/ingestion/factory.py` — real-model wiring (spaCy `en_core_web_sm` + sentence-transformers `all-mpnet-base-v2` for preferences + `all-MiniLM-L6-v2` for topical embeddings).
  - `engram/engram_memory_system.py` — `EngramGraphMemorySystem` implementing the `MemorySystem` protocol (Tier-1 ingestion; `answer_question` raises `NotImplementedError` until recall ships).
  - Tests: R2 keystone, persist roundtrip + version drift, GraphStore + BFS, canonicalization determinism, MemorySystem integration (save/load), preference fixtures + classifier gates. 99 tests pass; ruff clean.

### Current State

- Branch: `feat/ingestion-tier1` (open PR against `main`)
- `main` at `33e1424` — foundations + three-module consolidation + ingestion design doc
- Tests: 99 passing (local), ruff clean
- Build: `pip install -e .` succeeds; heavy deps (spaCy/sentence-transformers) pulled on install
- External artifacts downloaded lazily at first real ingest: `python -m spacy download en_core_web_sm` + HF model cache on first encode.

### What's Next

**Immediate:** review + merge `feat/ingestion-tier1` once CI is green. After merge, the next natural pushes are:

1. **Calibrate provisional values.** After a real-model run on a small synthetic corpus:
   - `canonicalization_match_threshold` (ships at 0.85)
   - `preference_discrimination_margin` (ships at 0.05)
   - Float-determinism budget (ships at ±2pp provisional)
   Each re-baseline is an M1 hypothesis with a per-bucket expected delta; PRs cite the hypothesis.

2. **Recall design doc** (`docs/design/recall.md`). The questions the design needs to resolve:
   - Intent taxonomy prototypes: single-fact, aggregation, preference, temporal, entity-resolution. Labeled seed queries per intent (hygiene: speaker-agnostic, NOT drawn from LME-s / LOCOMO).
   - Seeding: per-intent embedding-sim weights into UtteranceSegments vs. Entities.
   - Expansion: per-intent edge-weight vectors for the bounded walk (the ~35 scalars the optimizer tunes).
   - Ranking: cross-encoder choice + canonical subgraph→text rendering.
   - Context assembly rules (R11 — locality preserved, contiguous turn ordering).
   - Answerer prompt template (R13 — single file-owned).

3. **Diagnostics design doc** (`docs/design/diagnostics.md`). Reads `(AnswerResult, gold)` and the graph interior:
   - R15 classifier enum.
   - Oracle subgraph from gold annotations.
   - `needle_recall@k` / `session_density` / `completeness`.
   - Extraction-coverage report.
   - Fingerprint-audit pass over the external benchmark's cache layout.

4. **Tier-2 edges (Event / Episode)** and **Tier-3 edges (`supports` / `contradicts` / `refers_back_to`)** each gated on a separate design-doc iteration + M1 hypothesis + deterministic extractor.

### Open Questions

- Whether to ship a precomputed `centroids.json` fixture or keep lazy centroid construction at startup. Currently lazy — drift-free, loads the embedding model eagerly on first ingest. Swap to precomputed if startup latency becomes a bottleneck (unlikely at LME-s scale).
- `EdgeAttrs.source_turn_id` on aggregate edges like `co_occurs_with` is currently `None`. Design §2 flagged this as open. Decision deferred until recall needs it.

### Gotchas

- Windows + Git-bash: forward-slash paths only (`/c/Users/...`), no `--no-verify` on commits.
- The predecessor repo (`agent-memory`) is SIBLING, not parent. Do not accidentally modify it.
- The answerer is `ollama:llama3.1:8b`. Requires local Ollama. **Not yet wired** — `answer_question` raises `NotImplementedError` pending the recall design doc.
- Model downloads are lazy. First real ingest pulls spaCy `en_core_web_sm` (~12 MB) and sentence-transformer weights (~80 MB for MiniLM + ~420 MB for mpnet). Without them, `build_default_pipeline` fails loudly at construction.
- Numpy pin is `>=1.26,<3` (loosened from the design doc's `<2` because Python 3.13 requires numpy 2.x).
- Tests use `tests/_fake_nlp.py` fakes — no model downloads in the hermetic test suite. A `slow`-marked integration test against the real model is a future add.
