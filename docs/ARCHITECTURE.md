# Engram — Architecture

> The binding design contract lives in [`DESIGN-MANIFESTO.md`](DESIGN-MANIFESTO.md). This file is the evolving technical map — it grows as modules are built. If the two ever disagree, the manifesto wins; update this file instead.

## Overview

Engram is a graph-based memory system exposed as a tool to LLM agents. An outside agent (an LLM with tool-calling) decides what to save (calling `ingest(memory)`) and when to query (calling `recall(query)`). Engram never calls an LLM itself.

The graph is heterogeneous and multi-labeled. It indexes ingested Memories at four granularities (Session, Turn, Sentence, N-gram) across five content layers (Episodic, Entity, Relationship, Temporal, Semantic). Primary data is append-only; derived indexes are rebuildable.

Measurement against **LongMemEval-s** (primary) and **LOCOMO** (validation) lives in the external [`agent-memory-benchmark`](https://github.com/rtuosto/agent-memory-benchmark) repo, which implements an LLM agent that uses engram as a recall tool — simulating production deployment.

**North star:** an outside agent powered by `ollama:llama3.1:8b`, with engram as its memory tool, hits 100% on LongMemEval-s 100q. No paid APIs. Engram itself makes zero LLM calls. See [`DESIGN-MANIFESTO.md`](DESIGN-MANIFESTO.md) for the full principle set, design rules, KPIs, methodology, and verification gates.

## System Diagram

```
   agent-memory-benchmark (external repo)
   ───────────────────────────────────────────────────────────────────
                  ┌────────────────────────────┐
   Dataset    ──▶ │ benchmark/                 │
   (LME-s,        │  loader · runner · cache   │
    LOCOMO)       │  · replicates              │
                  └─────────────┬──────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │ agent (LLM loop)           │ ◀── 1 answerer call per question
                  │  uses engram as tool       │     (ollama:llama3.1:8b)
                  └─────────────┬──────────────┘
                                │ ingest(memory) / recall(query)
                                │
   engram (this repo)           │
   ───────────────────────────────────────────────────────────────────
                  ┌─────────────▼──────────────┐
                  │ MemorySystem               │
                  │  ingest · recall · reset   │
                  │  · save_state · load_state │
                  └──┬──────────────────────┬──┘
                     │                      │
   ┌─────────────────▼─────────┐  ┌─────────▼──────────────┐
   │ ingestion/                │  │ recall/                │
   │  segment · n-gram · NER   │  │  intent · seed · expand│──▶ RecallResult
   │  · canonicalize · claim   │  │  · score · assemble    │    (structured output
   │  · preference · embed     │  └────────────────────────┘     for the agent)
   │  · derived rebuilds       │
   │  · graph + vector index   │
   └──────────┬────────────────┘
              │ writes append-only primary;
              │ rebuilds derived indexes
              ▼
        ┌─────────────────────────────────┐
        │ Graph (5 layers × 4 granularities) +
        │  parallel embedding vector index │
        └─────────────────────────────────┘
                                       ▲
                  ┌────────────────────┴────────────────────┐
                  │ diagnostics/                            │
                  │  classify · coverage · fingerprint audit│ ◀── (AnswerResult, gold)
                  └─────────────────────────────────────────┘     from the benchmark
```

## Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `ingestion/` | Convert each Memory into graph nodes and edges (append-only primary, rebuildable derived). Maintain ingestion fingerprint. | `engram/ingestion/` |
| `recall/` | Convert a query into a structured `RecallResult` for the outside agent. No LLM calls. | `engram/recall/` |
| `diagnostics/` | Failure classification (R15), fingerprint audits, coverage reports. Read-only. | `engram/diagnostics/` |
| `engram.MemorySystem` | Public protocol — the only surface external callers touch. Five verbs: `ingest`, `recall`, `reset`, `save_state`, `load_state`. | `engram/__init__.py` |
| benchmarking + answerer agent | Orchestration, dataset loading, judging, scoring, replicates, **and the LLM agent that uses engram as a tool** — **external** | `agent-memory-benchmark` repo |

## Data Flow

1. **Ingest.** The agent calls `MemorySystem.ingest(memory)`. Engram segments the Memory's content into granules (Turn → Sentences → N-grams), runs NER, canonicalizes entities, extracts claims and preferences, embeds each granule, and anchors timestamps. All primary data is append-only.
2. **Derive (lazy).** Before the next recall, engram rebuilds derived indexes from primary if they're stale: co-occurrence counts, alias sets, current-truth indexes per (holder, target), change-event nodes when current truth flips, episodic clusters. Rebuilds are idempotent.
3. **Recall.** The agent calls `MemorySystem.recall(query, *, now, timezone, max_results)`. Engram classifies intent, seeds via vector-index nearest-neighbor on granule embeddings, expands via typed-edge BFS with intent-specific weights, scores the resulting subgraph, and returns a structured `RecallResult` (ranked passages + supporting edges + provenance).
4. **Agent answers.** The benchmark's agent serializes `RecallResult` into its tool-call response, may issue more recall calls, and eventually produces an answer with one `llama3.1:8b` call. The judge scores the answer.
5. **Diagnose.** `diagnostics.classify_failures(...)` consumes `(AnswerResult, gold_annotations)` from the benchmark plus engram's graph internals and produces an enum classification (`extraction_miss | graph_gap | retrieval_miss | partial_retrieval | output_miss | agent_miss`).

## Key Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| Engram is a memory tool, not an answerer | Mirrors production: agents use memory tools. Engram never calls an LLM. | 2026-04-20 |
| Two core verbs: `ingest(memory)` and `recall(query)` | No conversation IDs in the API. The agent decides what's a memory; one engram instance = one memory. | 2026-04-20 |
| Five layers × four granularities in the graph | Rich indexing narrows the search space even if the agent only sees a subset of returned content. | 2026-04-20 |
| Append-only primary, rebuildable derived | Time-travel queries, attribution, and reinforcement counting fall out for free. | 2026-04-20 |
| Content-addressed primitives, observations as edges | Counting "how many times the user said X" works via edge enumeration; entities are deduplicated automatically. | 2026-04-20 |
| Graph-first architecture (over flat multi-layer RRF) | Predecessor plateaued at 76% on flat layers; red buckets are graph-shaped. | 2026-04-20 |
| No LLM at ingest or recall on default path | Cost + determinism + R5/R13 discipline; LLM is opt-in enhancement layer. | 2026-04-20 |
| `llama3.1:8b` as the benchmark agent's answerer | North-star thesis: memory quality closes the gap, not model capability. | 2026-04-20 |
| LongMemEval-s 100q as primary benchmark | Diverse bucket structure; fits in iteration budget. | 2026-04-20 |
| Greenfield repo (not in-place rewrite of `agent-memory`) | Clean slate; predecessor stays runnable as a baseline. | 2026-04-20 |
| `recall` returns no answer; the agent answers | Production-realistic; engram never owns the answer prompt. | 2026-04-20 |
| No reranker in recall v1 | Walk scores are legible and tunable; add a reranker only when diagnostics shows ranking quality is the bottleneck. | 2026-04-20 |

## External Dependencies

| Service | Purpose | Notes |
|---------|---------|-------|
| Ollama | The benchmark agent's answerer. **Not invoked from engram.** | `llama3.1:8b` is the canonical answerer; lives outside engram. |
| Sentence-Transformers | Embeddings for granule semantic indexing and seeding | `all-MiniLM-L6-v2` for granule embeddings; `all-mpnet-base-v2` for preference centroids. |
| spaCy | Sentence splitting, n-gram extraction, NER, dependency parses | Enables no-LLM-at-ingest (R5) extraction |
| rapidfuzz | String similarity for entity canonicalization | Fast, deterministic. |
| networkx | In-memory `MultiDiGraph` storage | One graph per engram instance. |
| msgpack | Versioned persistence | Schema version on every save (R12). |
| numpy | Embedding vector index storage | Parallel to the graph; keyed by node ID. |

Dataset access (LongMemEval-s, LOCOMO) is owned by the external `agent-memory-benchmark` repo and is not a direct dependency of engram.

## Status & Next Steps

**Current state.** Ingestion is complete through PR-D (patches 1–7 of `docs/design/ingestion.md §12`): post-pivot `MemorySystem` protocol (`ingest` / `recall` / `reset` / `save_state` / `load_state`), R16 primary-data discipline, n-gram granularity + layer labels, granule embeddings + parallel vector index, TimeAnchor nodes + `temporal_at` edges, and the derived-rebuild orchestrator (aliases, co-occurrence, reinforcement, current-preference, TimeAnchor chain). `recall()` raises `NotImplementedError` — the implementation lands in PR-E.

**Where design attention goes next.**

1. **Recall implementation** — `docs/design/recall.md` is locked; the greenfield `engram/recall/` module (intent classifier, seeding, expansion, scoring, assembly) is next.
2. **Diagnostics design** — R15 classifier over `(RecallResult, gold_annotations)`, oracle subgraph computation, `needle_recall@k` / `granule_density` / `completeness` metrics, extraction-coverage reports, K7 fingerprint audits.
3. **Follow-ups** — ChangeEvent + EpisodicNode derived indexes; co-occurrence windows beyond per-Memory; calibration of provisional thresholds (`ngram_min_tokens`, `canonicalization_match_threshold`, `preference_discrimination_margin`).

Every design-phase PR cites the rule(s) it implements or the M1 hypothesis (target bucket, expected pp gain, mechanism, validation threshold, falsification condition) it tests.

## Local Development

Not yet wired — see the top-level `README.md` for setup once recall implementation lands.
