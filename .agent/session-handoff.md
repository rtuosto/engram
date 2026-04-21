# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

---

## Session: 2026-04-20 — engram foundations bootstrap

### What Was Done

- Cloned `github.com/rtuosto/engram` into `~/code/engram`.
- Ran `~/code/agent-bootstrap/setup.sh`: CLAUDE.md, .cursorrules, .agent/, docs/ARCHITECTURE.md, .gitignore installed.
- Created feature branch `feat/engram-foundations`.
- Wrote `docs/DESIGN-MANIFESTO.md` — the binding architectural contract. Approved by user 2026-04-20.
- Wrote `docs/ARCHITECTURE.md` — technical map.
- Committed the foundations (`b842b22`), module scaffolds (`1d7a263`), `MemorySystem` protocol (`d2b06c6`), `MemoryConfig` + fingerprint-discipline CI gate (`5cb494c`).
- Seeded `.agent/lessons.md` with 10 hard-won lessons ported from the predecessor.
- **2026-04-20 (later):** Removed `engram/benchmarking/` — benchmarking lives in a separate `agent-memory-benchmark` repo and consumes engram through the `MemorySystem` protocol. Updated manifesto §6/§7/§8/Verification, ARCHITECTURE.md (diagram + components + data flow + deps + status), README.md, CLAUDE.md project context, module `__init__.py` docstrings, the two test files, and this file to match. Engram is now three modules: `ingestion`, `recall`, `diagnostics`.

### Current State

- Branch: `feat/engram-foundations`
- Tests: none yet (no code)
- Build: no code to build
- Commits: pending first commit of the foundations

### What's Next

**Verification skeleton status.** Steps 1–4 done. Step 5 (external benchmark integration smoke) happens in `agent-memory-benchmark`, not here — this repo's obligation is to stay installable and protocol-stable. There is no step 6 in this repo's checklist anymore.

1. ✅ Manifesto approved.
2. ✅ Scaffold module dirs — commit `1d7a263`.
3. ✅ `MemorySystem` protocol with rule-citing docstrings — commit `d2b06c6`.
4. ✅ Fingerprint-discipline CI test — commit `5cb494c`.
5. ⏭ External benchmark integration smoke (in `agent-memory-benchmark`).

**Real work starts here — memory system design.** Next design pushes, in order:

- **Ingestion design** (`engram/ingestion/`). Concrete decisions needed before any extractor ships:
  - Graph storage: in-memory `networkx` vs. `kuzu` vs. DuckDB+PGQ vs. custom. Constraints: R2 (determinism) and R12 (versioned persistence).
  - Finalize node/edge schema from manifesto §3 sketch → concrete dataclasses.
  - Extraction pipeline order: segmentation → NER → entity canonicalization → claim → preference → temporal resolution → event grouping → episode detection → corpus signals.
  - spaCy model choice; embedding model choice (`all-MiniLM-L6-v2` default, but P5 says it can't discriminate speech acts — likely need a secondary index).
  - Preference-detection prototype centroids + held-out discrimination protocol (R6, fails-closed per §3.5).
  - Entity canonicalization algorithm (string + embedding + co-occurrence, deterministic tie-breaking).

- **Recall design** (`engram/recall/`). Concrete decisions:
  - Intent taxonomy prototypes: single-fact, aggregation, preference, temporal, entity-resolution. Labeled seed queries per intent.
  - Seeding: embedding-sim weights per intent, into Utterance Segments vs. Entity nodes.
  - Expansion: per-intent edge-weight schema for the bounded walk.
  - Ranking: cross-encoder choice + canonical subgraph→text rendering.
  - Context assembly rules (R11 — locality preserved, contiguous turn ordering).
  - Answerer prompt template (R13 — single file-owned template, no scattered f-strings).

- **Diagnostics design** (`engram/diagnostics/`). The input contract is `(AnswerResult, gold annotations)` handed in by the external benchmark; engram's diagnostics reads the graph interior when classification requires it.
  - R15 classifier: `extraction_miss | graph_gap | retrieval_miss | partial_retrieval | prompt_miss | answerer_miss`.
  - Oracle subgraph computation from gold annotations.
  - `needle_recall@k`, `session_density`, `completeness` implementations.
  - Extraction-coverage report (for a conversation, what % of gold entities/claims did extraction capture?).
  - Fingerprint-audit pass (cache hit rates, invalidation correctness, per K7) — the caches live in the benchmark repo, so this pass reads their layout and cross-checks against `MemoryConfig.*_fingerprint()`.

Every PR in the design phase cites the rule(s) it implements or the M1 hypothesis it tests.

### Open Questions

- Which Python project tooling? (`uv` / `poetry` / plain `pip` + `pyproject.toml`). Defer to first implementation commit; default is `uv` + `pyproject.toml` unless the user picks otherwise.
- Which graph storage layer? Candidates (implementation-time decision): in-memory `networkx`, `kuzu`, DuckDB + PGQ, or a custom NumPy-backed adjacency representation. Must satisfy R2 (determinism) and R12 (versioned persistence).
- Sentence-transformers model to start with? Default `all-MiniLM-L6-v2` per predecessor, but we already know it fails for speech acts (lesson 2026-04-20). May want an NLI-tuned model as secondary index for Preferences.

### Gotchas

- Windows + Git-bash: forward-slash paths only (`/c/Users/...`), no `--no-verify` on commits.
- Never commit to `main` — even the first commit goes on a feature branch (per bootstrap CLAUDE.md).
- The predecessor repo (`agent-memory`) is SIBLING, not parent. Do not accidentally modify it. `cd` deliberately if you ever need to consult it.
- The answerer is `ollama:llama3.1:8b`. Requires local Ollama running. Not yet wired.
- Cache discipline matters FROM DAY ONE. Even the skeleton should have `ingestion_fingerprint` / `answer_fingerprint` plumbing in place — the fingerprint-discipline test (verification step 4) is a load-bearing gate.
