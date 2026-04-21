# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

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
