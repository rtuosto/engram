# Engram — Architecture

> The binding design contract lives in [`DESIGN-MANIFESTO.md`](DESIGN-MANIFESTO.md). This file is the evolving technical map — it grows as modules are built. If the two ever disagree, the manifesto wins; update this file instead.

## Overview

Engram is a graph-based memory system for LLM agents, benchmarked against **LongMemEval-s** (primary) and **LOCOMO** (validation). It converts a stream of conversation sessions into a heterogeneous, multi-labeled graph (nodes: Turn, Utterance Segment, Entity, Claim, Preference, Event, Episode, Session; edges: typed relations) and, given a question, retrieves a ranked subgraph that feeds a single-pass answerer (`ollama:llama3.1:8b`).

**North star:** 100% on LongMemEval-s 100q with `llama3.1:8b`. No paid APIs. See [`DESIGN-MANIFESTO.md`](DESIGN-MANIFESTO.md) for the full principle set, design rules, KPIs, methodology, and verification gates.

## System Diagram

```
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
                  │  intent · seed · expand    │
                  │  · rank · assemble · answer│
                  └────────────────────────────┘
                             │
                             ▼
                  ┌────────────────────────────┐
                  │ benchmarking/              │──▶ Scorecards (LME-s, LOCOMO)
                  │  datasets · judge · runner │
                  │  · cache · replicates      │
                  └────────────────────────────┘
                             │
                             ▼
                  ┌────────────────────────────┐
                  │ diagnostics/               │──▶ Per-run failure classification
                  │  classify · coverage       │      (R15 enum)
                  │  · fingerprint audit       │
                  └────────────────────────────┘
```

## Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `ingestion/` | Convert sessions → graph; deterministic fingerprint | `engram/ingestion/` |
| `recall/` | Subgraph retrieval + context assembly + 1 answerer call | `engram/recall/` |
| `benchmarking/` | LME-s and LOCOMO orchestration, judging, scoring, caching | `engram/benchmarking/` |
| `diagnostics/` | Failure classification (R15), fingerprint audits, coverage reports | `engram/diagnostics/` |
| `engram.MemorySystem` | Public protocol — the only surface benchmarks touch | `engram/__init__.py` (TBD) |

## Data Flow

1. **Ingest.** `MemorySystem.ingest_session(session, conversation_id)` appends a session to the per-conversation graph; `finalize_conversation(conversation_id)` runs end-of-conversation passes (episode clustering, corpus-signal derivation). Output: deterministic graph + `ingestion_fingerprint`.
2. **Answer.** `MemorySystem.answer_question(question, conversation_id)` runs Recall: intent classification → seeding → bounded subgraph walk → cross-encoder rerank → context assembly → single answerer call. Output: `AnswerResult(answer, retrieved_units, timing)`.
3. **Benchmark.** `benchmarking.run(dataset, memory_system, answer_model, judge_model)` orchestrates ingest + answer + judge across a dataset. Caches are keyed by `(memory_system_id, ingestion_fingerprint, answer_fingerprint, dataset_hash, question_id)`.
4. **Diagnose.** `diagnostics.classify_failures(run)` produces a per-question enum classification (`extraction_miss | graph_gap | retrieval_miss | partial_retrieval | prompt_miss | answerer_miss`) and bucket-level breakdowns.

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
| Ollama | Local inference for answerer + (optional) judge | `llama3.1:8b` is the canonical answerer; no paid API in default path |
| Sentence-Transformers | Embeddings for seeding and ranking | To be chosen during Ingestion implementation; `all-MiniLM-L6-v2` is the predecessor's default |
| spaCy | Sentence splitting, NER, dependency parses | Enables no-LLM-at-ingest (R5) extraction |
| LongMemEval dataset | Primary benchmark (500q; we use 100q s-split) | Loader pattern ported from `agent-memory/benchmark/datasets/longmemeval.py` |
| LOCOMO dataset | Validation benchmark | Loader pattern ported from `agent-memory/benchmark/datasets/locomo.py` |

## Local Development

Not yet wired — see the top-level `README.md` for setup once implementation begins. The current repo contains design docs and the verification-skeleton plan only.
