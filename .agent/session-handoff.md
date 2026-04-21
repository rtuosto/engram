# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

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
