# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

---

## Session: 2026-04-21 (2) — ingestion-perf Phase 5

### What Was Done

Landed `perf/ingest-many` — `MemorySystem.ingest_many(memories)` batched
variant. Structural R3/R4 equivalence verified with both the fake
embedder (byte-identical msgpack) and real transformers (structural
fingerprint `OK nodes=271 edges=539` on the 20-memory synthetic subset).

- **Protocol extension** ([engram/memory_system.py](../engram/memory_system.py)):
  added `async def ingest_many(self, memories: Iterable[Memory])` with a
  concrete default body that loops `await self.ingest(m)`. Any
  implementation may override with a batched variant that pools model
  calls across the batch — the contract is "same graph as sequential,
  within the structural fingerprint guard's 1e-5 weight tolerance."
- **EngramGraphMemorySystem override** ([engram/engram_memory_system.py](../engram/engram_memory_system.py))
  → delegates to `IngestionPipeline.ingest_many(state, memory_seq)`.
- **Pipeline batched path** ([engram/ingestion/pipeline.py](../engram/ingestion/pipeline.py)):
  split `_extract_claims_and_preferences` into `_emit_claims_collect_prefs`
  (pass 1) + `_emit_prefs_from_verdicts` (pass 2). New
  `IngestionPipeline.ingest_many` method:
  - one `nlp_process([all_memory_contents])` call (spaCy pipe batching)
  - one `granule_embed(all_granule_texts)` call (MiniLM batch)
  - one `classify_batch(all_sentence_texts, ...)` call (mpnet batch)
  - per-memory graph writes, canonicalization, speaker caching, and
    TimeAnchor emission stay sequential so R16 append-only ordering is
    preserved exactly.
- **Protocol-shape test** ([tests/test_memory_system_protocol.py](../tests/test_memory_system_protocol.py))
  — added `ingest_many` to `_EXPECTED_VERBS` and to the `_FakeMemory`
  witness (default loop impl). All six verbs now tested for async +
  docstring citation.
- **New equivalence test** ([tests/test_ingest_many_equivalence.py](../tests/test_ingest_many_equivalence.py))
  — four assertions on the batched path: byte-identical msgpack vs
  sequential under the composition-invariant fake embedder,
  empty-is-noop, single-memory batch matches single ingest, vector-index
  row equality.
- **Profile harness extension** ([scripts/profile_ingestion.py](../scripts/profile_ingestion.py))
  — new `--batch-size` flag. `--batch-size 0` (default) uses sequential
  ingest (Phase 1/2 shape); `> 0` drives `ingest_many` in chunks and
  tags the artifact with `corpus=synthetic+batchN`.

### Measured deltas (50-memory synthetic corpus, single run)

| Driver                 | total_s | mean_ms | nlp | pref | granule |
| ---------------------- | ------- | ------- | --- | ---- | ------- |
| sequential (Phase 2)   | 1.003   | 20.07   | 50  | 37   | 50      |
| `ingest_many`, size=10 | 0.453   | 9.06    | 5   | 5    | 5       |
| `ingest_many`, size=50 | 0.388   | 7.77    | 1   | 1    | 1       |

**2.58× end-to-end speedup** at batch-size 50. Graph totals identical
across all three runs (50 memories, 39 entities, 9 claims, 5
preferences, 489 granules, 1296 edges). Preference-embed time dropped
322ms → 17.7ms (−94%). Artifacts: `profiling/phase5-seq.json`,
`profiling/phase5-batch10.json`, `profiling/phase5-batch50.json`.

### Current State

- Branch: `perf/ingest-many`
- Tests: 265 passing (added 5 new in `test_ingest_many_equivalence.py`
  and the `ingest_many`-parametrized protocol-shape tests on top of the
  prior 258).
- Build: clean.

### Gotchas

- The Protocol default for `ingest_many` is a concrete loop — if you add
  a new `MemorySystem` implementation, you only need to implement
  `ingest` to be structurally compatible, but for
  `isinstance(x, MemorySystem)` (`runtime_checkable`) to pass you still
  need the attribute. The `_FakeMemory` witness shows the minimum shape.
- R3/R4 guard remains structural (same graph topology + 1e-5 weight
  tolerance), NOT bytes — batched mpnet drifts at the 7th–8th decimal
  exactly like Phase 2. `scripts/check_fingerprint_equivalence.py`
  already handles this; don't revert to byte-hash comparison.
- Model-load cost (`~10s` for spaCy + MiniLM + mpnet + centroid probes)
  is still amortized-once-per-process. Phase 6 (module-scope cache
  keyed on the full ingestion fingerprint) remains unimplemented.

---

## Session: 2026-04-21 — ingestion-perf Phase 1 + Phase 2

### What Was Done

Landed `perf/ingestion-speedup` — merged to main. Three commits:

- **Phase 1** (`0335650`): per-stage ingestion profile harness
  [`scripts/profile_ingestion.py`](../scripts/profile_ingestion.py). Wraps
  the three injected model callables (spaCy, mpnet preference,
  MiniLM granule) with `time.perf_counter()` timers before handing
  them to `IngestionPipeline`. Emits a JSON artifact per commit
  under `profiling/`. Supports `--cprofile` for drill-downs and
  `--corpus longmemeval` as a best-effort loader.
  Baseline `profiling/ingestion-c40d5bc.json` (main, 40-mem synthetic):
  total 836ms, preference_embed 313.7ms (37.5%, 45 unbatched calls),
  nlp 259ms, granule_embed 236.5ms.

- **Phase 2** (`b325f53`): hoisted mpnet preference-classification
  out of the per-sentence loop. New API
  `engram.ingestion.extractors.preference.classify_batch(texts, ...)`;
  `IngestionPipeline._extract_claims_and_preferences` now collects
  `(claim_payload, sentence_text)` tuples while emitting Claim nodes
  inline, then fires a single batched `preference_embed` call per
  Memory. Per-row L2 normalization + centroid dots stay in a Python
  loop so the scoring arithmetic is bit-identical to single-text
  `classify()`.
  Measured: mpnet **calls 45 → 30 (−33%)**, preference_embed
  **−7.5%**. Total-time delta is inside run-to-run noise (±5%) on the
  synthetic corpus because templates average only 1.5 claim events
  per call-site. Real corpora with multi-claim memories will batch
  more aggressively.

- **Phase 2 fingerprint guard**
  [`scripts/check_fingerprint_equivalence.py`](../scripts/check_fingerprint_equivalence.py).
  First pass used a raw msgpack SHA-256 — flagged drift because
  `holds_preference` edge weights moved ~5e-8 (batched vs singleton
  encoder numerics from attention-mask padding). Graph topology is
  byte-identical: same 271 nodes, same 539 edges, same 5 Preference
  node IDs, same polarities, same claim identities. Upgraded the
  guard to compare structure (node IDs, edge tuples, payloads exact;
  edge weights within 1e-5). That is the correct R3/R4 proxy — "does
  the ingested graph mean the same thing," not "do the bytes match
  to the last ULP." Any new Preference, polarity flip, or payload
  change still fails immediately.

### cProfile-driven decisions (commit `8748597`)

- **Phase 3 (VectorIndex amortized-O(1) append): DROPPED.**
  `vector_index.add` = 6ms across 266 calls (1.7% of stage 8).
  Plan's <5% threshold. The `np.vstack` pattern is still a textbook
  anti-pattern; only revisit at benchmark scale (tens of thousands
  of granules).
- **Phase 4 (canonicalization index): DEFERRED.**
  `canonicalize` = 1ms across 156 calls (0.4% of total ingest) on
  the synthetic corpus. Revisit when a real LongMemEval-s run
  produces hundreds of entities per instance.
- **Phase 5 (batched `ingest_many` API): not started.** Largest
  scope remaining. Would collapse stages 2+7+8 across multiple
  memories into one forward pass each. Needs protocol extension.
- **Phase 6 (model-load cache): not started.** 10s of model load
  per pipeline construction. Low effort; useful for benchmark runs
  that spawn multiple engram instances.

### Current State

- Branch: `perf/ingestion-speedup` merged into `main`.
- Tests: all 258 passing.
- Build: clean.

### Gotchas

- `sentence-transformers` batched forward passes produce float32
  outputs that differ from singletons at the 7th–8th decimal. Any
  future "batch more inputs" optimization must use the structural
  fingerprint guard (not a raw byte hash) or it will appear to
  regress R3/R4.
- `profiling/` is tracked in-tree. Each phase's artifact is pinned
  by commit SHA so regressions are a `diff` away.
- `scripts/check_fingerprint_equivalence.py --emit` requires the
  synthetic corpus from `scripts/profile_ingestion.py`; running it
  against an older commit that lacks the profile harness needs a
  self-contained re-implementation of the corpus (see the pattern
  in `_tmp_baseline_hash.py` used during this session).

---

## Session: 2026-04-20 — PR-F (diagnostics module skeleton)

### What Was Done

Landed PR-F on `feat/pr-f-diagnostics`. Greenfield `engram/diagnostics/`
per manifesto §6 — the third and final module in the `ingestion /
recall / diagnostics` triad. Read-only (never writes caches, never
mutates run artifacts); zero LLM calls (R5, R13).

**Public verbs (five, per manifesto).**
- `classify_failures(cases, store=None, *, partial_threshold=0.5)` —
  R15 decision-tree classifier. Emits one `FailureCase` per
  `FailureInput`; `FailureKind(StrEnum)` covers
  `extraction_miss | graph_gap | retrieval_miss | partial_retrieval |
  output_miss | agent_miss` plus a neutral `correct` bucket so
  aggregation preserves denominators.
- `bucket_breakdown(cases, *, bucket_key="bucket")` — per-bucket counts
  per kind + totals + `total_cases` / `total_correct`. None-buckets
  group under the literal `"(none)"` key.
- `needle_overlap(gold, text)` — content-term overlap kernel
  (NFKC casefold + ASCII tokenize + 3-char min + small stop-word list).
  Returns sorted tuples for R2 stability.
- `extraction_coverage(store)` — nodes-by-label, nodes-by-layer
  (unlayered granules grouped under `(unlayered)`), ngram_kinds,
  edges-by-type, and a totals map
  (`n_nodes / n_edges / n_memories / n_entities / n_claims /
  n_preferences / n_time_anchors / n_granules`).
- `fingerprint_audit(a, b)` — full-field diff between two
  `MemoryConfig` instances. Separate `ingestion_match` / `recall_match`
  booleans + sorted `diverging_fields` tuple of `(name, value_a,
  value_b)`.

**R15 decision tree** (see `failures.py` docstring):
1. Overlap(gold, passages) == 0
   → extraction_miss if store also has zero node overlap
   → graph_gap if ≥2 nodes hit but no connecting edge
   → retrieval_miss otherwise
2. 0 < overlap < threshold → partial_retrieval
3. overlap ≥ threshold AND wrong
   → output_miss if facts don't surface the gold
   → agent_miss if they do

**Files created.**
- `engram/diagnostics/overlap.py` (~100 lines)
- `engram/diagnostics/failures.py` (~270 lines)
- `engram/diagnostics/coverage.py` (~140 lines)
- `engram/diagnostics/audit.py` (~80 lines)
- Rewrote `engram/diagnostics/__init__.py` with correct enum names
  (the stub had drifted to `prompt_miss / answerer_miss`) and
  re-exports for all public verbs.

**Tests (33 new; 258 total, was 225 on PR-E).**
- `tests/test_diagnostics_overlap.py` (8) — key-term extraction,
  stop-word filtering, casefold, NFKC on empty input, full / partial /
  zero recall, determinism.
- `tests/test_diagnostics_coverage.py` (7) — label counts, layer
  splits (entity vs unlayered), edge counts, ngram_kinds split, totals
  incl. n_granules, determinism, empty-store zeros.
- `tests/test_diagnostics_audit.py` (5) — identical match, ingestion
  field bumps both fingerprints (R4 transitivity), recall-only field
  leaves ingestion match intact, sorted divergence, (value_a, value_b)
  preservation.
- `tests/test_diagnostics_failures.py` (13) — one fixture per enum
  arm (CORRECT, EXTRACTION_MISS, RETRIEVAL_MISS, PARTIAL_RETRIEVAL,
  OUTPUT_MISS, AGENT_MISS, GRAPH_GAP), no-store fallback, sort order,
  configurable threshold, bucket_breakdown aggregation + None-bucket
  handling, classifier determinism.

**Out of scope (per plan).**
- PR-G (calibrator) — needs these classifications as its objective
  function but is its own PR.
- Commit-over-commit regression report — needs a run-history store.
- Benchmark CLI integration — per user direction, benchmark plumbing
  lives in `agent-memory-benchmark` (PR-J).

Test suite: 258 passing (was 225); ruff clean; mypy clean in
diagnostics scope (0 errors in 4 source + 4 test files). Pre-existing
56-error baseline on the rest of the repo is unchanged by this PR.

### Current State

- Branch: `feat/pr-f-diagnostics` (open; not yet PR'd against main)
- `main` at `bcd01c9` — PR-E + mypy cleanup merged
- Tests: 258 passing, ruff clean, mypy 56 pre-existing errors
  (0 new in diagnostics scope)
- Build: `pip install -e .` unchanged
- `SCHEMA_VERSION = 3`, `memory_version = 0.4.0` (unchanged —
  diagnostics is read-only; no on-disk format changes).

### What's Next

**Immediate:** open PR-F → main → merge.

The greenfield module triad (ingestion / recall / diagnostics) is now
complete. Remaining roadmap (documented in the prior session):

- **PR-G** — Per-intent weight calibrator. Uses this PR's
  `classify_failures` as the objective function over
  `recall/intents/heldout.json`. Offline tuner; pipeline untouched.
- **PR-H** — Co-occurrence windowing `(1h, 1d, all-time)`
  (`derivation_version: 1 → 2`).
- **PR-I** — ChangeEvent + EpisodicNode emission (manifesto §7 D5/D6).
  `SCHEMA_VERSION: 3 → 4`.
- **PR-J** — Benchmark adapter for new engram (lives in
  `agent-memory-benchmark`, not in this repo — per user direction).
  Blocks the first real benchmark number.

### Open Questions

- `_has_graph_gap` is a conservative heuristic (≥2 hit nodes +
  zero edges connecting any pair). It will miss cases where the gap
  sits between a seed granule and a distant gold node — the actual
  structural-gap detector would need BFS from the seed. Deferred
  until we have benchmark cases that show the heuristic misses.
- The `CORRECT` enum arm is not in the manifesto's R15 list — added
  here as a neutral bucket so `bucket_breakdown` has complete
  denominators without a second pass over inputs. If this becomes
  confusing in consumer code, split into two return types.
- Key-term extraction is ASCII-only (`[a-zA-Z0-9]+` token regex).
  Matches the predecessor's kernel. Non-ASCII gold answers
  (LongMemEval is English; LOCOMO has some Unicode) would silently
  extract zero terms. If benchmarks surface this, tighten to `\w+`
  with a Unicode category filter.

### Gotchas

- `FailureInput.bucket` is optional and defaults to `None`;
  `bucket_breakdown` maps that to the literal string `"(none)"`. If
  you filter buckets in a report, watch for that key rather than
  assuming `None`.
- `classify_failures` returns sorted output by `(question_id, gold)`.
  If you build a `FailureInput` list from an iterable that's already
  ordered and your report assumes input-order preservation, wrap
  inputs with monotonic `question_id`s instead.
- `fingerprint_audit.diverging_fields` includes non-fingerprint-carrying
  fields too (if any ever exist). Today the partition covers every
  field, so the list is a strict subset of fingerprint inputs.

---

## Session: 2026-04-20 — PR-E (recall implementation)

### What Was Done

Landed PR-E on `feat/pr-e-recall`. Greenfield `engram/recall/` per
[`docs/design/recall.md`](../docs/design/recall.md). Engram makes zero LLM
calls on the recall path (R5, R13).

**Config (R3/R4 rename).**
- `MemoryConfig._ANSWER_FIELDS → _RECALL_FIELDS`. `answer_fingerprint →
  recall_fingerprint`. Removed `answerer_model` / `answerer_temperature` /
  `context_char_budget` / `recall_top_k` (moved to the benchmark per
  design §11).
- New recall fields:
  - `intent_seed_hash` (content hash of `engram/recall/intents/seeds.json`)
  - `recall_weights_hash` (content hash of `engram/recall/weights.json`)
  - `intent_discrimination_margin: float = 0.05` (provisional)
  - `recall_max_depth: int = 3`
  - `recall_max_frontier: int = 256`
  - `recall_max_passages: int = 16`
  - `recall_seed_count_total: int = 64`
  - `recall_top_n_per_granularity: int = 12`
- Fingerprint discipline test renamed along with the partition — 42 tests
  covering the rename still pass.

**Models.**
- `RecallPassage` gained `source_memory_id`, `source_memory_index`,
  `speaker`, `supporting_edges`. `RecallFact` reshaped to carry `kind` +
  `value` + `supporting_memory_ids` per design §2. `RecallResult` gained
  `intent_confidence`, `timing_ms: tuple[(stage, ms)]`, `recall_fingerprint`.

**Recall module (new).**
- `engram/recall/intents/` — hand-authored `seeds.json` + `heldout.json`
  over five intents (`single_fact`, `aggregation`, `preference`,
  `temporal`, `entity_resolution`). 12 seeds per intent; 6 disjoint
  held-out queries per intent. Content hash (`INTENT_SEED_HASH`) exposed
  to config.
- `engram/recall/weights.json` — per-intent granularity + edge-type
  weight table. Values provisional per design §15. Content hash
  (`WEIGHTS_HASH`) exposed to config.
- `engram/recall/intent.py` — `classify_intent(query, centroids, embed_fn,
  margin_threshold, fallback)`. R6-compliant prototype-centroid classifier;
  below-margin → fallback + raw margin returned verbatim (fails closed).
- `engram/recall/seeding.py` — `semantic_seed` (KNN per granularity,
  weight-scaled counts + scores), `entity_anchored_seed` (spaCy NER over
  the query → entity + directly-mentioned granules), `merge_seeds` (max
  per node_id + total_cap).
- `engram/recall/expansion.py` — thin `expand()` wrapper over
  `GraphStore.bfs(seeds, edge_weights, max_depth, max_frontier)`.
- `engram/recall/scoring.py` — `select_passages(walk_scores, store,
  max_passages)`; routes granule nodes to themselves, drops non-granule
  nodes (Entity / Claim / Preference / TimeAnchor).
- `engram/recall/assembly.py` — `build_passages` (text + granularity +
  provenance from `part_of`-walked Memory + `temporal_at`-resolved
  timestamp + supporting `asserts` / `holds_preference` edges),
  `build_facts` (intent-shaped fact mix: `current_preference` for
  preference/temporal; `reinforcement` when count>1; `co_occurrence` for
  aggregation), `resolve_query_entity_ids`.
- `engram/recall/context.py` — `RecallContext` frozen dataclass
  (`now`, `timezone`, `max_passages`, `intent_hint`).
- `engram/recall/pipeline.py` — `RecallPipeline.recall(state, query,
  context)` orchestrates [classify → seed → expand → score → assemble].
  Timings recorded per stage (`timing_ms`) plus a `total` entry.
  `recall_fingerprint = sha256(config.recall_fingerprint || query ||
  context)[:16]`. Lazy derived rebuild via fingerprint check — stale
  snapshot triggers a fresh `rebuild_derived` before fact assembly.
- `engram/recall/factory.py` — `build_default_recall_pipeline(config)`
  mirrors `build_default_pipeline`; lazy-loads spaCy + sentence-transformers.

**System integration.**
- `EngramGraphMemorySystem.__init__` gains `recall_pipeline: RecallPipeline
  | None = None` (test injection; production builds lazily on first call).
- `EngramGraphMemorySystem.recall(...)` — no-op-then-structured-empty when
  `state is None` (benchmark caches still get a fingerprint-shaped
  envelope); otherwise dispatches to `RecallPipeline`.
- `EngramGraphMemorySystem._get_recall_pipeline()` — lazy builder.

**Public re-exports.** `engram.RecallContext` added to the package surface.

**Tests.** Three new files (40 tests).
- `tests/test_recall_intent.py` (11) — seed/heldout schema, centroid
  normalization, verdict shape, custom fallback, invalid fallback
  rejection, median margin per intent, seed/heldout disjointness,
  content-hash stability, classification determinism.
- `tests/test_recall_stages.py` (11) — `semantic_seed` per-granularity
  weighting, zero-weight skip, empty index; `entity_anchored_seed`
  resolves mentions and drops unresolved; `merge_seeds` keeps max +
  caps; `expand` respects `max_depth`; `select_passages` buckets per
  granule, drops non-granule nodes, respects `max_passages`, empty input.
- `tests/test_recall_pipeline.py` (17) — empty-state recall, passage
  retrieval + provenance fields, intent-hint bypass, invalid hint
  rejection, recall fingerprint stability/variation under query/context,
  determinism R2 audit (byte-identical modulo timing), `max_passages`
  override, preference-intent fact surfacing, `single_fact` omits
  co-occurrence, timing_ms stage names, zero-LLM-calls sentinel,
  save/load roundtrip preserves recall, timezone invalidation.

Test suite: 225 passing (was 178 on PR-D); ruff clean; mypy 16 pre-existing
errors (unchanged).

### Current State

- Branch: `feat/pr-e-recall` (open; not yet PR'd against main)
- `main` at `bf3460a` — PR-D merged
- Tests: 225 passing, ruff clean, mypy 16 pre-existing errors
- Build: `pip install -e .` unchanged
- `SCHEMA_VERSION = 3`, `memory_version = 0.4.0` (unchanged — recall is
  pure runtime; no on-disk format changes).

### What's Next

**Immediate:** open PR-E → main → merge.

After PR-E lands, the rewrite roadmap is complete:

1. ~~**PR-B** — n-gram granularity + layer labels.~~
2. ~~**PR-C** — granule embeddings + parallel vector index.~~
3. ~~**PR-D** — TimeAnchor + derived-rebuild orchestrator.~~
4. ~~**PR-E** — recall implementation (greenfield `engram/recall/`).~~ (this session)

Natural follow-ups (each its own PR, each grounded in a measured hypothesis):

- Per-intent weight calibration via Diagnostics-owned tuner (design §15).
- Co-occurrence windowing `(1 hour, 1 day, all-time)` (PR-D follow-up).
- ChangeEvent + EpisodicNode emission (manifesto §7 D5, D6).
- First full benchmark run against `agent-memory-benchmark` to baseline
  the new architecture vs the predecessor's 76%.

### Open Questions

- Intent classifier runs on the deterministic-embed fake in tests — real
  centroid discrimination on MiniLM is validated by
  `median_intent_margin`, which the factory currently does not gate on.
  A per-intent fails-closed policy (mirroring preference polarities)
  is a natural follow-up once benchmark data exists.
- N-gram passages use `NgramPayload.surface_form` as their passage text
  (short — can be <20 chars). The design §14 flags including the parent
  Sentence's text with the n-gram highlighted; deferred until post-
  benchmark measurement tells us whether agents benefit.
- `locality preservation` (R11 context around a Sentence passage) is
  **not** implemented — design §8 describes it but recall v1 ships
  without it; added as an item once a concrete gain-hypothesis exists.
- `max_frontier` eviction in `GraphStore.bfs` truncates the frontier but
  not the accumulated score map. If a pathological graph pushes scores
  that later get surfaced as passages despite being frontier-evicted
  mid-walk, recall surfaces them anyway. Not an issue at benchmark
  scale; flagging for the scaling triggers doc.

### Gotchas

- The recall pipeline's `timing_ms` is wall-clock — byte-identity of
  `RecallResult` across two runs requires zeroing or stripping the
  `timing_ms` field (the R2 audit test does this via
  `dataclasses.replace(r, timing_ms=())`). Any future test that compares
  two RecallResults end-to-end must follow the same convention.
- `EngramGraphMemorySystem.recall()` on an un-ingested system returns an
  empty `RecallResult(passages=(), intent=None)` rather than raising.
  This preserves benchmark-cache key construction but **does not** set
  `recall_fingerprint` — callers expecting a fingerprint must gate on
  `state is not None` before relying on it.
- `engram/recall/__init__.py` is kept empty of re-exports (just the
  module docstring) to avoid circular imports via `engram.config →
  engram.recall.intents`. Import the runtime classes from their explicit
  submodules (`engram.recall.pipeline`, `engram.recall.context`).
- Seeds / weights files are fingerprint-tracked: editing them shifts
  `intent_seed_hash` / `recall_weights_hash` → `recall_fingerprint`
  changes → every cached recall result invalidates. This is intentional
  (R4). Be deliberate about edits.

---

## Session: 2026-04-20 — PR-D (TimeAnchor + derived-rebuild orchestrator)

### What Was Done

Landed PR-D on `feat/pr-d-temporal-derived`. Patches 6 + 7 from `docs/design/ingestion.md §12`:

**Patch 6 — TimeAnchor + `temporal_at` edges.**
- New `LABEL_TIME_ANCHOR` + `TimeAnchorPayload` + `time_anchor_identity(iso_timestamp)`.
- `EDGE_TEMPORAL_AT` promoted into `TIER_1_EDGE_TYPES`; `EDGE_TEMPORAL_BEFORE`/`EDGE_TEMPORAL_AFTER` + `EDGE_CO_OCCURS_WITH` moved to a "derived edges" block with the comment that they live in sidecars, not the graph, in PR-D.
- `round_iso_timestamp(iso, resolution)` helper — supports `second | minute | hour | day`; preserves trailing "Z" suffix; preserves `+HH:MM` offsets; raises `ValueError` on unknown resolution or malformed timestamp.
- `MemoryConfig.time_anchor_resolution: str = "second"`, categorized under `_INGESTION_FIELDS`.
- `EdgeAttrs` gains `surface_form: str | None = None`. Populated on `mentions` edges so the derived alias-rebuild can walk entity-inbound mentions without replaying NER.
- Pipeline stage [9]: after granule embedding, emit (or reuse) a TimeAnchor for `memory.timestamp` (rounded) and attach `temporal_at` edges from every granule and relationship created on this ingest. Anchors carry `{temporal}` layer. Fails closed when `memory.timestamp is None`.
- Anchor sources tracked via `anchor_sources: list[str]` accumulator; stage [9] sorts + dedupes before edge emission (R2-stable).

**Patch 7 — derived-rebuild orchestrator.**
- New `engram/ingestion/derived.py` (487 lines). Five sidecar indexes, all frozen + slotted:
  - `AliasEntry` — per Entity, sorted distinct surface forms from inbound `mentions.surface_form`.
  - `CoOccurrenceEntry` — per Entity-pair (lexicographically ordered), per-Memory co-occurrence count + normalized weight.
  - `ReinforcementEntry` — per Claim / Preference, count of inbound observation edges + earliest / latest `asserted_at`.
  - `CurrentPreferenceEntry` — per `(holder_id, target_key)`, the most recent Preference observation (ISO-timestamp order, `node_id` tiebreak). `target_key` is `entity:<id>` or `literal:<text>`.
  - `TimeAnchorChainEntry` — TimeAnchors sorted by ISO, each carrying `prev_id` / `next_id` links.
- `rebuild_derived(store, config)` — idempotent composer; builds all five indexes in one pass.
- `derived_fingerprint(config, store) = sha256(ingestion_fp || derivation_version || "nodes:edges")[:16]`. Append-only primary means node + edge counts are a collision-safe change detector.
- `dump_derived(index) / load_derived(bytes)` — msgpack envelope with `schema_version=1`, `derivation_version=1`, `DerivedFormatError` on mismatch.
- ChangeEvent + EpisodicNode explicitly deferred (manifesto §7 D5, D6) — noted in module docstring.

**System integration.**
- `InstanceState.derived: DerivedIndex | None` — cached snapshot, populated by an explicit rebuild.
- `EngramGraphMemorySystem.rebuild_derived() -> DerivedIndex | None` — public sync method (returns `None` when no state yet). Recall (PR-E) will call this lazily on entry.
- `save_state` now writes `derived/snapshot.msgpack` when a snapshot is cached; manifest gains `has_derived: bool` + `derived_fingerprint: str | None`.
- `load_state` reads the sidecar when declared, recomputes the expected fingerprint against the freshly-loaded primary, and drops the snapshot if stale (so a rebuild regenerates on next use).

**Persistence rename + schema bump.**
- `SCHEMA_VERSION: 2 → 3`. Needed because `EdgeAttrs` gained a field (`surface_form`); v2 saves are rejected at envelope level, no migration shim.
- `persist._KIND_TO_CLS` gains `"time_anchor": TimeAnchorPayload`.
- `memory_version: 0.3.0 → 0.4.0`.
- Filename constants: `DERIVED_DIRNAME = "derived"`, `DERIVED_SNAPSHOT_FILENAME = "snapshot.msgpack"`.

**Tests.**
- `tests/test_time_anchor.py` — 14 tests. Rounding units across all four resolutions, malformed-input rejection, content-addressing, repeat-ingest convergence, distinct rounded moments, per-granule + per-relationship `temporal_at` edges, timestamp-less ingest fails closed, day-resolution collapses intraday.
- `tests/test_derived.py` — 16 tests. Rebuild idempotency (R17), fingerprint responds to config changes + primary growth, alias collection dedupes surfaces and omits un-mentioned entities, co-occurrence pair counts + sort order + empty case, reinforcement bounds, current-preference latest-wins across polarity flip, temporal-chain prev/next linkage, dump/load roundtrip + byte-stability + schema-mismatch rejection, end-to-end `rebuild_derived()` caching + save/load persistence.

Updated `tests/test_vector_index.py` to read `SCHEMA_VERSION` from `engram.ingestion.persist` instead of hard-coding `2` (so future bumps don't require test edits).

Test suite: 178 passing (was 148 on PR-C); ruff clean; mypy 16 pre-existing errors (unchanged).

### Current State

- Branch: `feat/pr-d-temporal-derived` (open; not yet PR'd against main)
- `main` at `d2ff48e` — PR-C merged
- Tests: 178 passing, ruff clean, mypy 16 pre-existing errors
- Build: `pip install -e .` unchanged
- `SCHEMA_VERSION = 3`. v1 / v2 saves are rejected on load.

### What's Next

**Immediate:** open PR-D → main → merge.

Then the remaining roadmap:

1. ~~**PR-B** — n-gram granularity + layer labels.~~
2. ~~**PR-C** — granule embeddings + parallel vector index.~~
3. ~~**PR-D** — TimeAnchor + derived-rebuild orchestrator.~~ (this session)
4. **PR-E** — recall implementation (greenfield `engram/recall/`).

### Open Questions

- Co-occurrence window currently is "all-time at Memory granularity." Design §14 lists `(1 hour, 1 day, all-time)` as the target — left as a follow-up calibration PR since recall doesn't depend on per-window weights yet.
- ChangeEvent / EpisodicNode node types remain declared (`LABEL_EVENT`, `LABEL_EPISODE`) but unemitted. Adding them is a standalone PR once recall lands and has an appetite for their shape.
- Derived sidecar persistence is one msgpack file (`snapshot.msgpack`) covering all five indexes. The design's §11 layout lists separate files per index — no functional difference; the one-file layout is simpler and rebuild is the atomic unit. Can be split later without changing the public contract.
- `primary_state_signature` is `nodes:edges`. R16 append-only makes this collision-safe today; if we ever introduce derived-edge emission into the graph (we don't in PR-D), this needs to swap to a content digest.

### Gotchas

- `surface_form` is on `EdgeAttrs` now — any test that builds `EdgeAttrs` directly for `mentions` edges and expects the alias index to pick it up must pass `surface_form=...`. The derived tests do this; existing tests that don't need aliases are untouched.
- `load_state` silently drops a stale derived snapshot (primary-signature mismatch) rather than raising. Recall-side callers should expect `state.derived` to potentially be `None` after load and call `rebuild_derived()` to regenerate.
- `time_anchor_identity` is scoped to the *rounded* ISO timestamp string. Changing `time_anchor_resolution` mid-corpus would duplicate anchors for the same wall-clock moment — `ingestion_fingerprint` covers this so fingerprint-aware caches will invalidate.
- Preference `target_key` in the current-preference index uses the literal string prefix `entity:` or `literal:`. Recall consumers must compose this same way when looking up "does Alice currently like pizza." Helper could be exposed; left inline for now.

---

## Session: 2026-04-20 — PR-C (granule embeddings + parallel vector index)

### What Was Done

Landed PR-C on `feat/pr-c-embeddings`. Patch 4 from `docs/design/ingestion.md §12`:

**Patch 4 — granule embeddings + parallel vector index.**
- New `engram/ingestion/vector_index.py` with `VectorIndex` dataclass:
  - `add(node_id, granularity, vector)` — appends a row; rejects duplicate IDs; L2-normalizes defensively (zero vectors stored as-is, fail-closed on knn).
  - `knn(query, k, *, granularity_filter=None)` — brute-force cosine top-k; normalizes the query; ties broken by `node_id` (R2). `granularity_filter` accepts a single string or frozenset over `{turn, sentence, ngram}`.
  - `save(embeddings_path, node_ids_path)` / `VectorIndex.load(...)` — two-file sidecar: `embeddings.npy` raw numpy matrix + `node_ids.json` schema-versioned envelope `{schema_version, dim, node_ids, granularities}`.
  - `VECTOR_INDEX_SCHEMA_VERSION = 1` with its own `VectorIndexFormatError`.
- `InstanceState` gains `vector_index: VectorIndex | None`, lazy-initialized on the first ingest (dim discovered from the first embed batch, frozen thereafter).
- `IngestionPipeline.__init__` now takes `granule_embed: Callable[[list[str]], np.ndarray]` — the MiniLM batch encoder.
- Pipeline stage [8] (`_emit_granule_embeddings`): one batched `granule_embed(texts)` call per ingest; rows appended to the vector index in pipeline emission order (Turn → Sentences in segment_index order → N-grams in `(char_span, kind, normalized_text)` order). N-gram identity collisions within a single ingest are deduped before the index call.
- `factory.build_default_pipeline` wires a second `_load_embed_fn(config.embedding_model)` for granules (previously only the preference-embed model was loaded).

**Persistence rename + schema bump.**
- `persist.dump_conversation` / `load_conversation` → `dump_state` / `load_state`.
- `SCHEMA_VERSION: 1 → 2`. Primary msgpack payload shape is unchanged from v1; the bump prevents silent loading of v1 saves that don't have the vector-index sidecar.
- `EngramGraphMemorySystem.save_state` writes `embeddings.npy` + `node_ids.json` alongside `primary.msgpack`; manifest gains `has_embeddings: bool`. `load_state` restores the vector index when the manifest declares it.
- `memory_version` bumped `0.2.0` → `0.3.0`.
- New filename constants: `EMBEDDINGS_FILENAME`, `NODE_IDS_FILENAME`.

**Tests.** New `tests/test_vector_index.py` — 23 tests across:
- Unit: add/normalize/dup-reject, knn top-k + tiebreak, granularity filter (single + frozenset), zero-vector + zero-query fail-closed paths.
- Persistence: roundtrip preserves vectors + granularities; save is byte-stable; neighbor lookups stable across save/load; schema-version mismatch rejected; length-mismatch sidecars rejected.
- Pipeline integration: every granule (Turn + Sentence + N-gram) is indexed after ingest; no non-granules leak in.
- `save_state` round-trip: sidecars present, manifest declares them, restored state carries an equivalent `VectorIndex`.
- R2 sidecar audit: two independent pipeline constructions produce byte-identical `embeddings.npy` + `node_ids.json`.

Touched existing tests (`test_persist_roundtrip`, `test_ingest_determinism`, `test_memory_system_integration`, `test_layers`) to use the renamed persist functions and pass `granule_embed` to `IngestionPipeline`.

Test suite: 146 passing (was 123 on PR-B); ruff clean; mypy 16 pre-existing errors (unchanged).

### Current State

- Branch: `feat/pr-c-embeddings` (open; not yet PR'd against main)
- `main` at `fb08c76` — PR-B merged
- Tests: 146 passing, ruff clean, mypy 16 pre-existing errors
- Build: `pip install -e .` unchanged
- `SCHEMA_VERSION = 2`. Old v1 files will be rejected on load (no migration shim; reingest).

### What's Next

**Immediate:** open PR-C → main → merge.

Then the remaining roadmap:

1. ~~**PR-B** — n-gram granularity + layer labels.~~
2. ~~**PR-C** — granule embeddings + parallel vector index + `dump_state`/`load_state` rename + `SCHEMA_VERSION` bump to 2.~~ (this session)
3. **PR-D** — TimeAnchor + derived-rebuild orchestrator.
4. **PR-E** — recall implementation (greenfield `engram/recall/`).

### Open Questions

- `granule_embed` is a separate callable from `preference_embed`, both loaded in `factory.py`. They currently use different models (MiniLM vs mpnet), so the duplication is correct. If a future config unifies them, collapse to one encoder.
- Vector-index brute-force cosine is fine at the design's scale triggers; swap to faiss only after one of the §2 thresholds fires.
- Real MiniLM determinism depends on torch determinism flags already set by `factory._seed_rngs`. The R2 audit tests use the fake `deterministic_embed`; a real-model R2 sidecar byte-check lives in the smoke script (to be refreshed in a follow-up).

### Gotchas

- Zero-norm input vectors are stored as zeros; they score 0 against every query and will never appear in knn top-k unless every candidate scores 0 (fail-closed). If this ever bites someone, add a strict mode that raises.
- `VectorIndex.add` is O(N) because `np.vstack` reallocates. At the design's granule counts per engram instance this is dominated by the embed call; revisit if ingest p50 exceeds the §2 threshold.
- Sidecar filenames are `embeddings.npy` + `node_ids.json`. Don't rename without bumping both `SCHEMA_VERSION` and `VECTOR_INDEX_SCHEMA_VERSION`.
- `_emit_ngrams` now dedupes repeat `ngram_id` within a single ingest before adding to the embedding batch. The underlying `add_node` / `add_edge` already tolerated duplicates (label-union + edge-overwrite); the new dedup is for `VectorIndex.add`, which rejects duplicate node_ids.

---

## Session: 2026-04-20 — PR-B (n-gram granularity + layer labels)

### What Was Done

Landed PR-B on `feat/pr-b-ngram-layers`. Patches 3 + 5 from `docs/design/ingestion.md §12`:

**Patch 3 — n-gram granularity.**
- New `engram/ingestion/extractors/ngram.py` with two deterministic extractors:
  - `extract_noun_chunk_ngrams(doc, segment_spans, *, min_tokens)` — walks spaCy `doc.noun_chunks`, maps each chunk to its enclosing Sentence, drops all-stop-word / below-min-tokens surfaces (fails closed, R6-style).
  - `extract_svo_ngrams(sent, segment_id, *, min_tokens)` — per-sentence dep-parse SVO triple (nsubj/nsubjpass + root verb + dobj/attr, with prep→pobj fallback).
- New `NgramPayload` dataclass and `ngram_identity(segment_id, ngram_kind, normalized_text)` helper. Content-addressed by that triple — two sentences sharing a phrase produce two nodes (each granule carries its own `part_of` edge and eventually its own embedding in PR-C), while two visits to the same phrase within one Sentence converge.
- `LABEL_NGRAM` added; registered in `persist._KIND_TO_CLS` (schema-version stays at 1 — the payload dict is additive and doesn't require a bump).
- Pipeline stage [3] runs both extractors, emits N-gram nodes + `part_of` (N-gram → Sentence) edges. Output is sorted by `(char_span, ngram_kind, normalized_text)` so pipeline iteration order is R2-stable.
- New `MemoryConfig.ngram_min_tokens = 2` (provisional, calibration deferred) — categorized under `_INGESTION_FIELDS` so the fingerprint-discipline test covers it.

**Patch 5 — layer labels.**
- `GraphStore.add_node(..., layers: frozenset[str] = frozenset())` accepts a content-classification layer set; second call unions layers.
- `GraphStore.node_layers(node_id)` + `GraphStore.nodes_by_layer(layer)` read helpers (R2-sorted).
- Pipeline populates layers:
  - Entity nodes → `{entity}` (includes speaker entities).
  - Claim nodes → `{relationship}`.
  - Preference nodes → `{relationship}`.
- Granule nodes (Memory / Turn / UtteranceSegment / N-gram) carry empty `frozenset()` — per design, `semantic` is implicit in the parallel embedding index (landing PR-C), not a node label. Temporal / episodic layers land in PR-D.
- `persist.py` loader now handles the new `layers` node attr (missing = empty frozenset, forward-compatible with pre-PR-B dumps).

**Shared-iteration refactor.** The pipeline used to re-iterate `doc.sents` in segmentation, n-gram, and claim/preference stages — three places that had to agree on the "skip empty" filter. Consolidated to one iteration in `_emit_segments`, which now returns a `list[_SegmentInfo]` carrying `(segment_id, char_span, sent)` for downstream stages. Prevents the drift bug where two stages disagree about sentence boundaries.

**Tests.** Two new test files:
- `tests/test_ngram_extractor.py` — 14 per-extractor units (identity, sorting, content-address, min-tokens gate, span/segment resolution, fail-closed paths).
- `tests/test_layers.py` — 9 tests covering GraphStore layer discipline (default empty, index, union-on-repeat, missing-node error) and pipeline integration (entity/claim/preference layered correctly, granules leave layers empty, n-gram nodes + part_of edges present).

Test suite: 123 passing (was 98 on PR-A); ruff clean; mypy down from 19 → 16 errors (my changes introduced zero new errors).

### Current State

- Branch: `feat/pr-b-ngram-layers` (open; not yet PR'd against main)
- `main` at `c4e2ddb` — PR-A merged
- Tests: 123 passing, ruff clean, mypy 16 pre-existing errors (unchanged touch)
- Build: `pip install -e .` unchanged
- `SCHEMA_VERSION` still at `1` — on-disk dataclass registry gained `NgramPayload`, but existing files decode because no required fields changed. PR-C will bump to `2` when it adds the parallel vector index.

### What's Next

**Immediate:** open PR-B → main → merge.

Then proceed through the roadmap in `.agent/current-plan.md`:

1. ~~**PR-B** — n-gram granularity + layer labels.~~ (this session)
2. **PR-C** — granule embeddings + parallel vector index + `dump_state`/`load_state` rename + `SCHEMA_VERSION` bump to 2.
3. **PR-D** — TimeAnchor + derived-rebuild orchestrator.
4. **PR-E** — recall implementation (greenfield `engram/recall/`).

### Open Questions

- N-gram identity includes `segment_id`, so a phrase repeated across sentences creates distinct nodes. Recall's cross-sentence connection will have to happen through Entity nodes (already shared) or the co-occurrence derived index (PR-D). No blocker; flagging for PR-E's seeding design.
- N-gram embeddings aren't computed yet — PR-C adds them. Until then, granules have no vector representation and recall cannot use the semantic-layer index.

### Gotchas

- The pipeline ingest order now runs n-gram extraction right after segmentation and before NER (so the n-gram stage sees the full Doc and the live segment spans). The stage-numbered code comments in `pipeline.py` reflect the new order; the module docstring lists stages as a bullet list without absolute indexes.
- `FakeDoc.noun_chunks` now exists as an optional field; existing tests don't pass it and default to empty tuple → noun-chunk extractor returns nothing. Tests that care must construct `FakeNounChunk` instances (see `tests/test_ngram_extractor.py` for patterns).
- `FakeToken.is_stop = False` default. Tests exercising the min-tokens gate must set `is_stop=True` on tokens meant to be filtered out.

---

## Session: 2026-04-20 — PR-A (protocol pivot + R16 primary-data discipline)

### What Was Done

Landed PR-A on `feat/pr-a-protocol-pivot`, the most disruptive of the five PRs in the post-pivot roadmap. Patches 1 + 2 from `docs/design/ingestion.md §12`:

**Protocol surface (patch 1).**
- `MemorySystem` protocol reshaped: verbs are now `ingest(memory)`, `recall(query, *, now, timezone, max_passages, intent_hint)`, `reset`, `save_state`, `load_state`. Dropped `ingest_session`, `finalize_conversation`, `answer_question`. No more `conversation_id` anywhere.
- New `Memory` dataclass (`engram/models.py`): `content`, `timestamp`, `speaker`, `source`, `metadata` (sorted tuple).
- New result types: `RecallResult`, `RecallPassage`, `RecallFact`. `AnswerResult` removed.
- `Session` / `Turn` retained as optional helper types (external callers may still build Memories from conversational datasets).
- `recall` stub raises `NotImplementedError` — real implementation lands in PR-E.

**R16 primary-data discipline (patch 2).**
- `EntityPayload.aliases` removed from the payload — aliases are derived from `mentions` edges (PR-D wires the rebuild).
- Preference is now its own content-addressed node (`preference_identity(holder_id, polarity, target_id_or_literal)`). No more multi-label overlay on Claim.
- `holds_preference` edge goes from speaker Entity → Preference node (with confidence on the edge weight, not the payload).
- `ClaimPayload` shrunk: identity-relevant fields only. `asserted_by_turn_id` / `asserted_at` removed (they live on the `asserts` edge now).
- `CoOccurrenceCounter` + `finalize_conversation` removed. Temporal-before/after and co-occurrence both move to derived rebuild (PR-D).
- `memory_index` is a monotonic per-instance counter — two `ingest` calls with identical content produce two Memory nodes (each is an observation event; R16: never dedup Memories).
- New node labels + payloads: `LABEL_MEMORY`, `MemoryPayload`; `TurnPayload` now points to its parent `memory_id` instead of carrying conversation/session/turn indexes.
- `EngramGraphMemorySystem` holds one `InstanceState` (GraphStore + registry + speaker map + memory_index). `reset` clears it. Persistence layout simplified to `manifest.json` + `primary.msgpack`.
- `memory_version` bumped `0.1.0` → `0.2.0`.

**Tests.** Rewrote six test files for the new shape; added R16 coverage (repeat-ingest creates new Memory nodes; EntityPayload has no aliases field). 98 tests pass, ruff clean, mypy unchanged at pre-PR baseline (net −7 errors).

### Current State

- Branch: `feat/pr-a-protocol-pivot` (open; not yet PR'd against main)
- `main` at `d9f95d3` — docs pivot landed, but code still has the pre-pivot shape
- Tests: 98 passing, ruff clean
- Build: `pip install -e .` unchanged
- `SCHEMA_VERSION` still at `1` — primary on-disk format changed (new `MemoryPayload`, dropped `SessionPayload`), but we're bumping `memory_version` not `SCHEMA_VERSION` since the file layout itself (envelope shape) is the same. PR-C will bump `SCHEMA_VERSION` to `2` when it adds the parallel vector index.

### What's Next

**Immediate:** open PR-A → main → merge.

After PR-A lands, proceed through the roadmap in `.agent/current-plan.md`:

1. **PR-B** — n-gram granularity (`extractors/ngram.py`) + layer labels on nodes. Additive, small.
2. **PR-C** — granule embeddings + parallel vector index + `dump_state`/`load_state` rename + `SCHEMA_VERSION` bump to 2.
3. **PR-D** — TimeAnchor + derived-rebuild orchestrator (co-occurrence, alias sets, reinforcement counts, current-truth; ChangeEvent + EpisodicNode deferred).
4. **PR-E** — recall implementation (greenfield `engram/recall/`).

Each PR updates this file with what shipped.

### Open Questions

- `load_state` restores `memory_index=0` on restore; if future flows mix load + fresh ingest, we need to derive max `memory_index` from the restored graph's Memory nodes. Deferred until someone needs it.
- `GraphStore.conversation_id` field lingers as a vestigial opaque tag (`"__instance__"`). Could delete in a follow-up, but each removal touches msgpack envelope shape → `SCHEMA_VERSION` bump. Deferred to PR-C (where we bump anyway).

### Gotchas

- Windows + Git-bash: forward-slash paths only, no `--no-verify` on commits.
- `memory_version` is `"0.2.0"` now; any persisted state from pre-PR-A `EngramGraphMemorySystem` won't load (manifest schema moved from per-conversation files to `primary.msgpack`). No migration shim; reingest.
- `Preference` is a separate node, not a Claim label. Tests / diagnostics that counted "claim ∧ preference" multi-label nodes must now count Preference nodes directly.
- First real ingest still downloads spaCy + MiniLM + mpnet weights on first call — unchanged.
