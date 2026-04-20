# Engram — Design Manifesto

**Status:** Pre-implementation contract. Binding on all subsequent design and implementation decisions.
**Scope:** This repository's architecture, methodology, and KPIs. Supersedes any conflicting guidance elsewhere.
**Reference codebase:** `agent-memory` (sibling project). Concepts may be ported; no files are copied.

---

## Context — why engram exists

The predecessor system (`agent-memory`) plateaued at **76.0% ± 1.73pp** on LongMemEval-s 100q with `llama3.1:8b`, after six months of tuning a flat multi-layer RRF architecture (RAW + SENTENCE + ENTITY + EPISODIC + PREFERENCE FAISS indexes, cross-encoder reranker, context-window expansion). Two buckets dominate the remaining headroom:

- Multi-session-aggregation: **53.8%** (26 qids, 10 stable losses)
- Single-session-preference: **16.7%** (6 qids, 5 stable losses)

Two weeks of knob-tuning moved the needle in tenths of a point. Every experiment that changed *structure* — episodic layer, adaptive top-N, context-window — moved it in whole points. The evidence is that the next 10–15 points live in what we store and how we connect it, not in how we weight what we already store.

**North star:** 100% on LongMemEval-s 100q using `ollama:llama3.1:8b` as the answerer. No paid APIs, no model swaps. The memory system alone closes the gap.

**Minimum acceptable outcome of the rewrite:** match or beat 76% with an architecture that has a credible path to 100%.

**Primary thesis:** embedding + NER + rule-based extraction, graph structure over flat layers. LLM at ingest is a possible future enhancement, NOT the default path.

---

## 1. Principles

Each principle is a normative statement plus the lesson it encodes. Every design decision traces back to one of these.

**P1 — Representation before ranking.** The system is a representation problem, not a retrieval-tuning problem. Six months of knob-tuning plateaued at 76%. If a change only moves embedding scores, it is unlikely to be load-bearing.

**P2 — Corpus-aware, not query-brainstormed.** LLM query expansion regressed because 8b invents vocabulary the user never wrote. Any expansion signal comes from the indexed corpus — observed aliases, co-occurrences, session vocabulary.

**P3 — Context is answer-critical; pruning is hostile by default.** Aggressive rerank pruning and LLM-judge filters dropped accuracy by removing load-bearing non-needle scaffolding (speaker, tone, timing, clarifications). Default response to uncertainty: keep it.

**P4 — The 8b answerer relays, it does not execute.** Structured date-math prompts regressed −8pp. Any computation — dates, counts, sums — is done at ingest or recall planning and handed to the answerer as a literal string.

**P5 — Speech acts are first-class, distinct from topics.** `all-MiniLM-L6-v2` embeds by topic. "I love X" and "X is a problem" sit near each other in topic space but are opposite in preference space. Preferences, opinions, negations, commitments get their own representation — not a cosine away.

**P6 — Every artifact has a fingerprint; no fingerprint, no cache.** Cache-invalidation traps cost days on the predecessor. Every derived artifact is addressable by a hash of every input that can change it.

**P7 — Evidence lives above noise.** Ground-truth noise floor is ±1.73pp; FP non-determinism adds another ±4pp commit-to-commit. A claimed improvement smaller than the noise budget is not an improvement.

**P8 — Benchmarking is a measurement instrument, not a subject of optimization.** If benchmark code, prompts, or judges are tweaked to help the memory system, measurements become non-comparable. Benchmarking is hostile to self-serving changes.

**P9 — Diagnose deepest, fix deepest.** On failure, classify the miss first: extraction → retrieval → prompt → answerer. Fix the deepest broken layer. Do not patch an upper layer to compensate for a lower one.

**P10 — Graph encodes knowledge; embeddings rank candidates.** Embeddings answer "near this topic"; graphs answer "connected through this relation." Our two red buckets are graph-shaped. The rewrite is graph-first because the failures are.

**P11 — LLM at ingest is a future layer, not a foundation.** Primary extraction is deterministic: NER, dependency parses, rule-based claim detection, prototype-embedding classification. An LLM enhancement layer may be added later, gated behind a budget flag, and must never become a precondition for correctness.

---

## 2. Design Rules

Engineering constraints that flow from the principles. Reviewed at PR time; violations block merges.

**R1.** Single `MemorySystem` protocol per experiment family. Only `ingest_session`, `finalize_conversation`, `answer_question`, `reset`, `save_state`, `load_state` are public. No sideband hooks.

**R2.** Ingestion is deterministic given (config, corpus). Same inputs → byte-identical serialized state. No wall-clock values, no PID-seeded randomness, no unsorted set iteration in output.

**R3.** `ingestion_fingerprint` includes every config field that affects node/edge production. A config diff without a fingerprint diff is a bug.

**R4.** `answer_fingerprint` transitively includes `ingestion_fingerprint`. Any ingest change invalidates every downstream answer.

**R5.** No LLM calls in the default ingestion path. Any LLM-based enhancement is opt-in, behind a config flag, batched per-conversation, and budget-capped. The non-LLM path must always be fully functional on its own.

**R6.** No English-specific regex in intent or speech-act classification. Use prototype-embedding centroids with validated discrimination, or labeled seed queries. Hardcoded patterns overfit to benchmark phrasings.

**R7.** No LLM query expansion over raw queries. Expansion reads only corpus-derived signals (aliases, co-occurrences, session vocabulary).

**R8.** Temporal arithmetic happens at ingest or recall planning, never in the answer prompt. The answerer receives resolved absolute dates.

**R9.** Retrieval returns a subgraph (nodes + justifying edges), not a flat list. Context assembly decides ordering and format.

**R10.** Speech acts and topics live on separate indices or separate node properties. A preference is not a weighted sentence.

**R11.** Context assembly preserves locality by default. A selected node's temporal neighbors are eligible for inclusion. Any pruning requires oracle-tested justification on the affected bucket.

**R12.** No unversioned persisted state. Explicit schema version on every save; load paths handle missing / old / future versions with clear errors.

**R13.** The answerer prompt is a single file-owned template. Changes are reviewed like API changes. No f-strings scattered across modules.

**R14.** Float-path determinism is declared, not assumed. Fixed seeds, sorted inputs, stable reduction order where feasible; measured and budgeted where not.

**R15.** Diagnostic classifier outputs are an enum, not a string: `extraction_miss | graph_gap | retrieval_miss | partial_retrieval | prompt_miss | answerer_miss`. Explanations are a separate field.

---

## 3. Graph Architecture Sketch

High-level. Libraries, storage layout, and on-disk format are implementation choices made module-by-module.

### Node types (labeled)

- **Turn** — one utterance by one speaker at a timestamp. Verbatim text. Source of truth.
- **Utterance Segment** — sentence-grained unit under a Turn; finest text anchor for retrieval.
- **Entity** — person, place, organization, artifact, concept. Canonical form + aliases.
- **Claim** — a proposition extracted from Turns (subject, predicate, object, asserted_by, asserted_at). Named "Claim" not "Fact" to signal it is a *speaker's assertion* in time, not a world truth.
- **Preference** — Claim subtype with `polarity` (`likes | dislikes | wants | avoids | commits_to | rejects`) and `holder`. First-class speech-act node.
- **Event** — situation anchored by absolute timestamp or interval. Connects the Turns that describe it.
- **Episode** — narrative thread spanning one or more sessions. First-class node (promoted from "unit with raw_unit_ids extra" in the predecessor).
- **Session** — container for Turns sharing a timestamp and conversation position.

Heterogeneous, multi-labeled graph. A node may carry multiple labels (e.g., a Claim that is also a Preference).

### Edge types (labeled)

- `part_of` (Turn → Session, Utterance Segment → Turn, Turn → Episode)
- `mentions` (Turn → Entity, Claim → Entity)
- `asserts` (Turn → Claim)
- `holds_preference` (Entity[speaker] → Preference)
- `about` (Claim → Entity, Preference → Entity)
- `during` (Event → Session, Claim → Event)
- `temporal_before` / `temporal_after` (Event → Event, Turn → Turn)
- `supports` / `contradicts` (Claim ↔ Claim, Claim ↔ Preference) — conflict detection at recall
- `refers_back_to` (Turn → Turn) — anaphora / callback
- `co_occurs_with` (Entity ↔ Entity) — corpus-derived, weights P2 expansion

### Extraction (no-LLM path)

Primary extractors are deterministic:

1. **Segmentation** — dependency-parse-aware sentence splitting per Turn.
2. **NER** — batched per conversation, existing pipeline pattern.
3. **Entity canonicalization** — alias clustering via string similarity + embedding similarity + corpus co-occurrence; deterministic tie-breaking.
4. **Claim extraction** — subject-verb-object triples from dependency parses; speaker attribution from Turn metadata; tense/modality from parse features.
5. **Preference detection** — prototype-embedding centroids *with validated discrimination on a held-out set*, combined with dependency-parse signals (modal verbs, preference predicates). Fails closed: if the classifier can't discriminate above noise, no Preference node is emitted and the Claim stands on its own.
6. **Temporal resolution** — deterministic library for explicit dates; relative expressions resolved against `asserted_at`.
7. **Event grouping** — temporal proximity + entity overlap + topic coherence.
8. **Episode detection** — clustered over the enriched graph.
9. **Corpus signals** — `co_occurs_with` counts, alias tables, per-speaker vocabulary — built after extraction, feed back into expansion.

If a step's output quality is not validated above noise on a held-out set, that step does not ship. P11 applies: the non-LLM path stands on its own.

### Recall (subgraph retrieval)

1. **Intent classification** — corpus-prototype embeddings (R6). Intents: single-fact, aggregation, preference, temporal, entity-resolution.
2. **Seeding** — embedding similarity into Utterance Segments and Entity nodes; intent-weighted.
3. **Expansion** — bounded walk along typed edges. Edge weights are intent-shaped (aggregation expands `co_occurs_with` + `part_of_episode`; preference expands `holds_preference` + `about`).
4. **Ranking** — cross-encoder on a canonicalized textual rendering of each subgraph neighborhood.
5. **Assembly** — contiguous context preserving turn order (R11), with speaker and timestamp metadata, and edge-justifications where they help the answerer (e.g., "preference held by Alice, expressed 2024-05-12").
6. **Answer** — fixed prompt template (R13), answerer sees only resolved literals (R8).

### LLM enhancement layer (future)

Optional, flag-gated. Conceivable enhancements:

- Claim extraction from turns where dependency parses are ambiguous.
- Preference-polarity confirmation where embedding confidence is low.
- Episode summarization for recall-time rendering.

Each is an opt-in experiment with its own hypothesis and evidence bar (M2). None is a precondition for the system being correct.

---

## 4. KPIs

Priority order with targets. Non-negotiable in regression analysis.

**K1 — LongMemEval-s 100q accuracy (primary).** Target 100%. Minimum acceptable post-rewrite 80%. Reported as mean ± std over 3 replicates. Claim threshold: `current_score + 2 × 1.73pp`.

**K2 — Per-bucket accuracy.** Multi-session-aggregation ≥ 90% (was 53.8%). Single-session-preference ≥ 90% (was 16.7%). No other bucket regresses by more than 1 question.

**K3 — Retrieval diagnostics (per-bucket).**
- `needle_recall@k` — all gold turns present in retrieved subgraph.
- `session_density` — fraction of retrieved content from gold sessions.
- `completeness` — gold entity mentions present.
- `retrieval_miss` rate ≤ 5% of wrong answers.
- `extraction_miss` rate — tracked separately from retrieval.

**K4 — Latency.** p50 ingest per session and p50 retrieve per question reported; p95 answer end-to-end reported. Not optimized until accuracy targets are hit.

**K5 — Cost.**
- Ingest LLM calls per question amortized — target 0 on default path (R5).
- Recall LLM calls per question — target exactly 1 (the answerer).
- Any additional LLM call requires an approved exception.

**K6 — Stability.** Ground-truth noise floor ±1.73pp. FP non-determinism budget ±4pp commit-to-commit. Claims within these budgets require 3 replicates; within 2× requires 5.

**K7 — Fingerprint discipline.**
- Ingestion cache hit rate on unchanged-config rerun: 100%.
- Answer cache invalidation on ingestion-fingerprint change: 100%.
- Verified by a fingerprint-audit test that fails CI if violated.

### Minimum per-commit scoreboard

Every commit touching memory code emits:
- LME-s 100q accuracy (1 replicate quick signal; 3 for claims).
- Per-bucket delta vs main.
- `{extraction_miss, graph_gap, retrieval_miss, partial_retrieval, prompt_miss, answerer_miss}` counts.
- Ingest p50 and retrieve p50.
- Ingestion cache hit rate on warm rerun.
- Commit SHA, ingestion_fingerprint, answer_fingerprint.

---

## 5. Optimization Methodology

These rules prevent the specific failure modes that burned the predecessor.

**M1 — Every change starts as a written hypothesis.** Format: `(target bucket | expected gain in pp | mechanism in 1–2 sentences | validation threshold | falsification condition)`. No hypothesis, no branch.

**M2 — Minimum evidence bar.** One of:
- ≥ 1.73pp full-benchmark gain across 3 replicates, lower CI bound above zero.
- Oracle-subgraph test showing mechanism improves a diagnosed bucket by ≥ 2 questions, with no regression elsewhere on full benchmark.
- Diagnostic-only change that improves signal quality, with no scored delta required.

**M3 — Cache-tainted ablations are invalid.** Every ablation run starts from a clean answer cache (ingestion cache may persist iff the change is post-ingest). The runner refuses to publish a result if the answer cache predates the relevant fingerprint.

**M4 — Retrieval-only improvements do not ship without full-benchmark runs.** `needle_recall@k` is necessary but not sufficient. Pruning has a history of improving retrieval metrics while regressing accuracy.

**M5 — Layer decision tree before coding.** Classify the failure first:
- Gold fact present in retrieved subgraph, answerer misses → prompt or answerer.
- Gold fact absent from retrieved subgraph, present in graph → retrieval (seeding / expansion / ranking).
- Gold fact absent from graph → extraction or ingestion.
Fix the deepest broken layer (P9). Never patch above to compensate for below.

**M6 — Noise-budget-aware replicates.** 1 for sanity; 3 within ±4pp; 5 within 2×.

**M7 — Kill criteria.** When the hypothesis's falsification condition is met, the branch is abandoned. Revival requires a new hypothesis citing why the original failed.

**M8 — Diagnostics runs first on every failure.** Classification logged with the commit.

**M9 — Competitor features enter as hypotheses, not as code.** Hindsight's four-network design, CARA, etc. are candidate experiments each with their own hypothesis and threshold.

---

## 6. Module Boundaries

Four top-level modules. Each owns a strict slice; cross-module calls go through the public verbs only.

### Ingestion

- **Responsibility:** Convert a conversation stream into a populated graph with a deterministic fingerprint.
- **Public verbs:** `ingest_session`, `finalize_conversation`, `export_state`, `import_state`, `fingerprint`.
- **Owns:** segmentation, NER, entity canonicalization, claim/preference/event extraction, temporal resolution, edge construction, episode detection, corpus-signal derivation, ingestion fingerprinting.
- **Does not touch:** query text, answer generation, benchmark orchestration, judge prompts, cache file layout.
- **Stability:** fingerprint fully covers output state; identical fingerprint ⇒ identical state.

### Recall

- **Responsibility:** Given a question and a graph, produce a ranked subgraph, an assembled context string, and one answer call.
- **Public verbs:** `classify_intent`, `seed`, `expand`, `rank`, `assemble_context`, `answer`.
- **Owns:** intent classification, seeding, subgraph expansion, reranking, context assembly, answer-prompt template.
- **Does not touch:** node/edge creation, benchmark scoring, judge prompts, the ingestion fingerprint.
- **Stability:** answer fingerprint transitively covers ingestion fingerprint plus all recall-side config.

### Benchmarking

- **Responsibility:** Run a memory system against LongMemEval-s (primary) and LOCOMO (validation), score, persist, compare.
- **Public verbs:** `run`, `resume`, `rejudge`, `summarize`, `compare`, `baseline`, `ablation`.
- **Owns:** dataset loaders, judge abstraction + prompts, run directory layout, scorecard rendering, cache layout, replicate orchestration.
- **Does not touch:** the memory system's internals. Reads only the `MemorySystem` protocol surface; never inspects the graph.
- **Stability:** must be bit-stable across memory-system rewrites. Judge or dataset changes require an explicit re-baseline.

### Diagnostics

- **Responsibility:** Given a run's results, classify each failure and surface actionable patterns.
- **Public verbs:** `classify_failures`, `bucket_breakdown`, `needle_overlap`, `extraction_coverage`, `fingerprint_audit`.
- **Owns:** failure classification (R15), gold-term overlap, extraction-coverage reports, fingerprint-discipline audits, regression reports.
- **Does not touch:** runtime path. Read-only. Never writes to caches. Never mutates run artifacts.
- **Stability:** may evolve freely, but classifications are versioned.

---

## 7. Reference material in `agent-memory` (read, don't copy)

These files in the sibling `agent-memory` repo encode concepts that survive the rewrite. Read them for shape; do not port code.

- `benchmark/memory_interface.py` — `MemorySystem` ABC, `AnswerResult`. Shape of R1.
- `memory/config.py` — dual-fingerprint discipline. Shape of R3, R4, P6.
- `benchmark/runner.py` — ingestion/answer/judge cache layering, replicate orchestration, resume semantics. Shape of M3, R12.
- `benchmark/cache.py` — cache key construction, dataset hashing, prompt fingerprint. Shape of P6.
- `benchmark/datasets/longmemeval.py`, `benchmark/datasets/locomo.py` — dataset interfaces. Directly informative; reimplement to the same shape.
- `memory/ingestion/ner.py` — batched GPU NER pattern. Shape of R5 (batching invariant applies even without LLM).
- `memory/ingestion/episodes.py` — episode clustering concept. Will be promoted to a first-class node.
- `memory/retrieval/reranker.py` — cross-encoder integration pattern with session diversity + min_unique_sessions.
- `scripts/diagnose_lme.py` — failure-classification kernel (`retrieval_miss | partial_retrieval | model_miss`). Seed of the Diagnostics taxonomy (R15).
- `.agent/lessons.md` — hard-won rules from the predecessor. Ported key entries into this repo's `.agent/lessons.md` as seed lessons.
- `.agent/architecture-brief.md` (DECISIONS block) — frozen decisions on the predecessor (north star, phase roadmap, cost/model constraints).

---

## 8. Anti-Patterns — explicit DO NOT

- **Do not add LLM query expansion over raw queries.** P2, R7.
- **Do not ask the 8b answerer to perform arithmetic.** P4, R8.
- **Do not use English-specific regex for intent or speech-act classification.** R6.
- **Do not use topic-embedding cosine as a preference matcher without validated discrimination.** P5, R10.
- **Do not add an LLM-judge filter between retrieval and the answerer.** P3.
- **Do not ship aggressive reranker pruning without oracle validation on the affected bucket.** P3, R11.
- **Do not add a config field without adding it to the correct fingerprint.** P6, R3, R4.
- **Do not let Benchmarking "help" the memory system.** P8.
- **Do not fix the prompt to paper over a retrieval miss, or fix retrieval to paper over an extraction miss.** P9, M5.
- **Do not claim an improvement smaller than ±1.73pp on a single replicate.** P7, M2, M6.
- **Do not persist ingestion state via unversioned pickle.** R12.
- **Do not scatter answerer-prompt f-strings across modules.** R13.
- **Do not mint a new Node or Edge type without adding it to the ingestion fingerprint and wiring it into extraction.** R3, R5.
- **Do not import competitor ideas as code.** M9.
- **Do not run an ablation against a warm answer cache whose fingerprint predates the change.** M3.
- **Do not make the default path depend on an LLM at ingest.** P11, R5.

---

## Verification — how we know the rewrite works

Before any extraction/retrieval implementation lands:

1. **Manifesto approved** (this document).
2. **Skeleton modules created** — four module directories (`ingestion/`, `recall/`, `benchmarking/`, `diagnostics/`) with the module-boundary docstrings from §6 as the top-of-module comment, and one stub test per module that imports it.
3. **`MemorySystem` protocol defined** — the five public verbs from §6, typed, with docstrings citing the relevant rules.
4. **Fingerprint discipline test** — a CI test that creates two configs differing only on one field, verifies fingerprints differ; and two configs identical in all fields, verifies fingerprints match. Fails CI otherwise.
5. **Benchmark harness wired** — LongMemEval-s loader + judge + runner ported, with an empty `NullMemorySystem` that answers "I don't know" to everything. It must produce a scorecard (≈15–25% expected from abstention-allowed questions).
6. **Diagnostics wired** — the failure-classification taxonomy of R15 implemented against the `NullMemorySystem` run; produces a breakdown.

Only after 1–6 are green does implementation of actual extraction/retrieval begin. Every subsequent PR must cite the rule(s) it implements or the hypothesis (M1) it tests.

After implementation begins, the rewrite is considered a success when:

- LME-s 100q accuracy ≥ 80% (minimum acceptable, §K1), with a credible measured path to 100% supported by the per-bucket scoreboard (§K2).
- Multi-session-aggregation ≥ 75% and SSP ≥ 75% — the two currently-red buckets moved decisively.
- `retrieval_miss` and `extraction_miss` rates below the targets in §K3.
- Fingerprint-discipline test and replicate-eval scoreboard stable across 3 consecutive commits on `main`.
