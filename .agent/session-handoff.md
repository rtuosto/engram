# Session Handoff

> This file is the bridge between agent sessions. Update it at the END of every session.
> Read it at the START of every session. Don't skip this — it's how you maintain continuity.

---

## Session: 2026-04-20 — engram foundations bootstrap

### What Was Done

- Cloned `github.com/rtuosto/engram` into `~/code/engram`.
- Ran `~/code/agent-bootstrap/setup.sh`: CLAUDE.md, .cursorrules, .agent/, docs/ARCHITECTURE.md, .gitignore installed.
- Created feature branch `feat/engram-foundations`.
- Wrote `docs/DESIGN-MANIFESTO.md` — the binding architectural contract (11 principles, 15 design rules, graph sketch, 7 KPIs, 9 methodology rules, 4 module boundaries, anti-patterns, verification checklist). Approved by user 2026-04-20.
- Wrote `docs/ARCHITECTURE.md` — technical map with system diagram, component table, data flow, decision log, deps.
- Customized `CLAUDE.md` Project-Specific Context with engram stack, four-module boundaries, non-negotiables excerpts, cost discipline, predecessor repo note.
- Seeded `.agent/lessons.md` with 10 hard-won lessons ported from the predecessor `agent-memory` project (cache traps, retrieval-only gains, FP non-determinism, speech-act vs. topic embeddings, prompt drift, etc.).
- Updated this file.

### Current State

- Branch: `feat/engram-foundations`
- Tests: none yet (no code)
- Build: no code to build
- Commits: pending first commit of the foundations

### What's Next

Verification checklist from `docs/DESIGN-MANIFESTO.md` — §Verification. Six steps before any extraction/retrieval implementation:

1. ✅ Manifesto approved.
2. ⏭ Scaffold four module dirs (`ingestion/`, `recall/`, `benchmarking/`, `diagnostics/`) with boundary docstrings + stub tests.
3. ⏭ Define `MemorySystem` protocol with rule-citing docstrings (the five public verbs).
4. ⏭ Write fingerprint discipline CI test (two configs differing on one field → fingerprints differ; identical configs → identical fingerprints).
5. ⏭ Wire benchmark harness (LongMemEval-s loader + judge + runner) against a `NullMemorySystem` that answers "I don't know"; produce a scorecard.
6. ⏭ Wire diagnostics failure-classification taxonomy (R15 enum) against the `NullMemorySystem` run.

After 1–6 green, every subsequent PR must cite the rule(s) it implements or the hypothesis (M1) it tests.

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
