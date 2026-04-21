# Engram — Architecture

> The binding design contract lives in [`DESIGN-MANIFESTO.md`](DESIGN-MANIFESTO.md). This file is the evolving technical map — it grows as modules are built. If the two ever disagree, the manifesto wins; update this file instead.

## Overview

Engram is a graph-based memory system for LLM agents. It converts a stream of conversation sessions into a heterogeneous, multi-labeled graph (nodes: Turn, Utterance Segment, Entity, Claim, Preference, Event, Episode, Session; edges: typed relations) and, given a question, retrieves a ranked subgraph that feeds a single-pass answerer (`ollama:llama3.1:8b`).

Measurement against **LongMemEval-s** (primary) and **LOCOMO** (validation) lives in the external [`agent-memory-benchmark`](https://github.com/rtuosto/agent-memory-benchmark) repo, which consumes this package through the `MemorySystem` protocol.

**North star:** 100% on LongMemEval-s 100q with `llama3.1:8b`. No paid APIs. See [`DESIGN-MANIFESTO.md`](DESIGN-MANIFESTO.md) for the full principle set, design rules, KPIs, methodology, and verification gates.

## System Diagram

```
  engram (this repo)
  ───────────────────────────────────────────────────────────────────
                  ┌────────────────────────────┐
  Conversation ──▶│ ingestion/                 │──▶ Graph (nodes + edges + fingerprint)
  (sessions)      │  segment · NER · canon.    │        │
                  │  · claims · preferences    │        │
                  │  · events · episodes       │        │
                  │  · corpus signals          │        │
                  └────────────────────────────┘        │
                                                        ▼
                  ┌────────────────────────────┐
       Query ────▶│ recall/                    │──▶ Subgraph + context + 1 answerer call
                  │  intent · seed · expand    │              │
                  │  · rank · assemble · answer│              │
                  └────────────────────────────┘              │
                                                              │
                  ┌────────────────────────────┐              │
                  │ diagnostics/               │◀─────────────┘
                  │  classify · coverage       │──▶ Per-run failure classification
                  │  · fingerprint audit       │      (R15 enum)
                  └────────────────────────────┘
                             ▲
                             │  calls `MemorySystem.*` verbs
                             │
  agent-memory-benchmark (external repo) ──────────────────────────────
                  ┌────────────────────────────┐
                  │ benchmarking/              │──▶ Scorecards (LME-s, LOCOMO)
                  │  datasets · judge · runner │
                  │  · cache · replicates      │
                  └────────────────────────────┘
```

## Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `ingestion/` | Convert sessions → graph; deterministic fingerprint | `engram/ingestion/` |
| `recall/` | Subgraph retrieval + context assembly + 1 answerer call | `engram/recall/` |
| `diagnostics/` | Failure classification (R15), fingerprint audits, coverage reports | `engram/diagnostics/` |
| `engram.MemorySystem` | Public protocol — the only surface external callers touch | `engram/__init__.py` |
| benchmarking | LME-s and LOCOMO orchestration, judging, scoring, caching — **external** | `agent-memory-benchmark` repo |

## Data Flow

1. **Ingest.** `MemorySystem.ingest_session(session, conversation_id)` appends a session to the per-conversation graph; `finalize_conversation(conversation_id)` runs end-of-conversation passes (episode clustering, corpus-signal derivation). Output: deterministic graph + `ingestion_fingerprint`.
2. **Answer.** `MemorySystem.answer_question(question, conversation_id)` runs Recall: intent classification → seeding → bounded subgraph walk → cross-encoder rerank → context assembly → single answerer call. Output: `AnswerResult(answer, retrieved_units, timing)`.
3. **Benchmark (external).** The `agent-memory-benchmark` repo orchestrates ingest + answer + judge across a dataset by calling `MemorySystem.*` verbs. It owns its own cache keyed by `(memory_system_id, ingestion_fingerprint, answer_fingerprint, dataset_hash, question_id)` — engram exposes the fingerprints; composing the cache key is the benchmark's job.
4. **Diagnose.** `diagnostics.classify_failures(...)` produces a per-question enum classification (`extraction_miss | graph_gap | retrieval_miss | partial_retrieval | prompt_miss | answerer_miss`) from an `AnswerResult` plus gold annotations handed in by the benchmark, using engram-internal knowledge of the graph where needed. Bucket-level breakdowns and extraction-coverage / fingerprint-audit reports are also emitted here.

## Key Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| Graph-first architecture (over flat multi-layer RRF) | Predecessor plateaued at 76% on flat layers; red buckets are graph-shaped | 2026-04-20 |
| No LLM at ingest on default path | Cost + determinism + R5 discipline; LLM is opt-in enhancement layer | 2026-04-20 |
| `llama3.1:8b` as permanent answerer | North-star thesis: memory quality closes the gap, not model capability | 2026-04-20 |
| LongMemEval-s 100q as primary benchmark | Diverse bucket structure; fits in iteration budget | 2026-04-20 |
| Greenfield repo (not in-place rewrite of `agent-memory`) | Clean slate; predecessor stays runnable as a baseline | 2026-04-20 |

## External Dependencies

| Service | Purpose | Notes |
|---------|---------|-------|
| Ollama | Local inference for answerer | `llama3.1:8b` is the canonical answerer; no paid API in default path. The judge lives in the external benchmark repo. |
| Sentence-Transformers | Embeddings for seeding and ranking | To be chosen during Ingestion implementation; `all-MiniLM-L6-v2` is the predecessor's default |
| spaCy | Sentence splitting, NER, dependency parses | Enables no-LLM-at-ingest (R5) extraction |

Dataset access (LongMemEval-s, LOCOMO) is owned by the external `agent-memory-benchmark` repo and is not a direct dependency of engram.

## Status & Next Steps

**Current state.** Verification skeleton steps 1–4 complete (manifesto, module scaffolds, `MemorySystem` protocol, fingerprint-discipline CI). Step 5 (external benchmark integration smoke) happens in the `agent-memory-benchmark` repo, not here — engram's job is to stay installable and protocol-stable.

**Where design attention goes next — the memory system itself.** The three modules have boundary docstrings but no interiors. Design planning and implementation follow this order:

1. **Ingestion** — graph storage choice (must satisfy R2, R12), concrete node/edge schema from the manifesto §3 sketch, deterministic extraction pipeline (segmentation → NER → canonicalization → claim → preference → temporal → event → episode → corpus signals), preference-detection discrimination protocol (R6, fails-closed).
2. **Recall** — intent taxonomy prototypes and seed queries (R6, no English-specific regex), per-intent seeding and expansion edge-weight schemas, ranker, R11-compliant context assembly, R13 single-file answerer prompt template.
3. **Diagnostics** — R15 classifier over `(AnswerResult, gold_annotations)`, oracle subgraph computation, `needle_recall@k` / `session_density` / `completeness` metrics, extraction-coverage reports, K7 fingerprint audits. Input shape is defined by the external benchmark's run-result contract.

Every design-phase PR cites the rule(s) it implements or the M1 hypothesis (target bucket, expected pp gain, mechanism, validation threshold, falsification condition) it tests.

## Local Development

Not yet wired — see the top-level `README.md` for setup once implementation begins.
