# engram

A graph-based memory system for LLM agents. Measured against LongMemEval and LOCOMO by the external [`agent-memory-benchmark`](https://github.com/rtuosto/agent-memory-benchmark) repo, which consumes this package through the `MemorySystem` protocol.

**Status:** pre-implementation. Design contract is in [`docs/DESIGN-MANIFESTO.md`](docs/DESIGN-MANIFESTO.md); technical map is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Code lands only after the Verification checklist in the manifesto is green.

## North star

100% on LongMemEval-s 100q with `ollama:llama3.1:8b` as the answerer. No paid APIs. The memory system alone closes the gap.

## Why a rewrite

The predecessor [`agent-memory`](https://github.com/rtuosto/agent-memory) plateaued at 76.0% ± 1.73pp after six months of tuning a flat multi-layer RRF architecture. Two buckets dominate the remaining headroom: multi-session-aggregation (53.8%) and single-session-preference (16.7%). Both are graph-shaped failures — preference requires speech-act structure, aggregation requires cross-session entity/event connections. Flat embedding layers can't represent either.

engram is graph-first because the failures are.

## Modules

- [`engram/ingestion/`](engram/ingestion/) — sessions → graph; deterministic fingerprint.
- [`engram/recall/`](engram/recall/) — question → subgraph + context + one answerer call.
- [`engram/diagnostics/`](engram/diagnostics/) — per-run failure classification and coverage reports.

Benchmarking (dataset loading, judging, scoring, cache layout, replicate orchestration) lives in the external [`agent-memory-benchmark`](https://github.com/rtuosto/agent-memory-benchmark) repo. It calls into engram through the `MemorySystem` protocol exposed by `engram/__init__.py`.

(All three modules are pre-implementation; only the protocol surface and config exist today.)

## Design contract

Read [`docs/DESIGN-MANIFESTO.md`](docs/DESIGN-MANIFESTO.md) first. It contains 11 principles, 15 design rules, a graph architecture sketch, 7 KPIs, 9 optimization-methodology rules, module boundaries, and a verification checklist. Every PR cites the rule(s) it implements or the hypothesis it tests.

## Agent workflow

This repo uses the agent-bootstrap contract (see [`CLAUDE.md`](CLAUDE.md)):

- Never commit to `main`. All work on `feat/...`, `fix/...`, `refactor/...`, `docs/...`, `chore/...`, `test/...` branches.
- Read `.agent/lessons.md` and `.agent/session-handoff.md` at the start of every session.
- Update `.agent/session-handoff.md` at the end of every session.
- Log mistakes in `.agent/lessons.md`. Never repeat them.

## Local development

To be wired once ingestion and recall implementations land. Will require:

- Python 3.11+
- Ollama running locally with `llama3.1:8b` pulled (answerer lives inside recall)

Dataset downloads and judge invocation are the external benchmark's concern.

## License

TBD.
