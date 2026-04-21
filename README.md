# engram

A graph-based memory system exposed as a tool to LLM agents. An outside agent (an LLM with tool-calling) decides what to save (calling `ingest(memory)`) and when to query (calling `recall(query)`). Engram itself never calls an LLM — it's all deterministic NLP + graph traversal + embedding similarity.

Measured against LongMemEval and LOCOMO by the external [`agent-memory-benchmark`](https://github.com/rtuosto/agent-memory-benchmark) repo, which both consumes engram through the `MemorySystem` protocol and implements the answerer agent that uses engram as a tool.

**Status:** ingestion Tier-1 implementation has landed; architecture pivot to engram-as-tool is in flight (see [`docs/design/ingestion.md`](docs/design/ingestion.md) §12 for the patch path). Recall design is in [`docs/design/recall.md`](docs/design/recall.md). Code lands only after each section of the manifesto is green.

## North star

100% on LongMemEval-s 100q achieved by an outside agent powered by `ollama:llama3.1:8b`, with engram as the agent's memory tool. No paid APIs. **Engram itself makes zero LLM calls** — at ingest, at recall, ever. The benchmark implements the agent. The memory system alone closes the gap.

## Why a rewrite

The predecessor [`agent-memory`](https://github.com/rtuosto/agent-memory) plateaued at 76.0% ± 1.73pp after six months of tuning a flat multi-layer RRF architecture. Two buckets dominate the remaining headroom: multi-session-aggregation (53.8%) and single-session-preference (16.7%). Both are graph-shaped failures — preference requires speech-act structure, aggregation requires cross-memory entity / event connections. Flat embedding layers can't represent either.

engram is graph-first because the failures are.

## Architecture in one paragraph

Engram indexes ingested Memories at four granularities (Session, Turn, Sentence, N-gram) across five content layers (Episodic, Entity, Relationship, Temporal, Semantic). Primary data — Memories, granules, entities, claims, preferences, time anchors — is append-only, content-addressed where appropriate, and never mutated. Derived indexes — co-occurrence counts, alias sets, current-truth indexes, change events, episodic clusters — are rebuildable from primary. Recall classifies query intent, seeds via vector-index nearest-neighbor on granule embeddings, expands via typed-edge BFS with intent-specific weights, and returns a structured `RecallResult` (passages + pre-computed facts + provenance) for the outside agent to consume.

## Modules

- [`engram/ingestion/`](engram/ingestion/) — Memories → graph + parallel vector index. Deterministic fingerprint.
- [`engram/recall/`](engram/recall/) — query → structured `RecallResult`. No LLM calls.
- [`engram/diagnostics/`](engram/diagnostics/) — per-run failure classification (R15 enum), coverage reports, fingerprint audits.

Benchmarking — dataset loading, judging, scoring, cache layout, replicate orchestration, **and the answerer agent itself** — lives in the external [`agent-memory-benchmark`](https://github.com/rtuosto/agent-memory-benchmark) repo. It calls into engram through the `MemorySystem` protocol exposed by `engram/__init__.py`.

## The two core verbs

```python
async def ingest(memory: Memory) -> None
async def recall(query: str, *, now: str | None = None, timezone: str | None = None,
                 max_passages: int | None = None) -> RecallResult
```

Plus `reset`, `save_state`, `load_state`. No conversation IDs, no session IDs — one engram instance holds one memory; isolation is the caller's responsibility via `reset` or separate instances.

## Design contract

Read [`docs/DESIGN-MANIFESTO.md`](docs/DESIGN-MANIFESTO.md) first. It contains 13 principles, 17 design rules, a graph architecture sketch (5 layers × 4 granularities, append-only primary + rebuildable derived), 7 KPIs, 9 optimization-methodology rules, module boundaries, and a verification checklist. Every PR cites the rule(s) it implements or the hypothesis it tests.

Per-module design docs in [`docs/design/`](docs/design/):
- [`ingestion.md`](docs/design/ingestion.md) — Memory shape, layers, granularities, append-only primary, derived rebuilds, extraction pipeline, patch path.
- [`recall.md`](docs/design/recall.md) — five-stage pipeline, `RecallResult` shape, intent classification, seeding, expansion, scoring, assembly.

## Agent workflow

This repo uses the agent-bootstrap contract (see [`CLAUDE.md`](CLAUDE.md)):

- Never commit to `main`. All work on `feat/...`, `fix/...`, `refactor/...`, `docs/...`, `chore/...`, `test/...` branches.
- Read `.agent/lessons.md` and `.agent/session-handoff.md` at the start of every session.
- Update `.agent/session-handoff.md` at the end of every session.
- Log mistakes in `.agent/lessons.md`. Never repeat them.

## Local development

To be wired once the recall implementation lands. Will require:

- Python 3.11+
- spaCy `en_core_web_sm` model: `python -m spacy download en_core_web_sm`
- sentence-transformers (downloads `all-MiniLM-L6-v2` and `all-mpnet-base-v2` on first use; cached to `~/.cache/huggingface/`)

The benchmark agent's answerer (Ollama with `llama3.1:8b`) lives in the benchmark repo, not here.

## License

TBD.
