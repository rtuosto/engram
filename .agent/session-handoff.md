# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

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
