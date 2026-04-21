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

**North star:** 100% on LongMemEval-s 100q achieved by an outside agent powered by `ollama:llama3.1:8b`, with engram as the memory tool the agent calls. No paid APIs, no model swaps. The memory system alone closes the gap. **Engram itself never calls an LLM** — not at ingest, not at recall. The benchmark implements the agent (a single answerer call per question), simulating how engram is used in production: as a tool an LLM agent invokes to save and recall context.

**Minimum acceptable outcome of the rewrite:** match or beat 76% with an architecture that has a credible path to 100%.

**Primary thesis:** embedding + NER + rule-based extraction, graph structure over flat layers. LLM enhancements (at ingest or recall) are possible future layers, never preconditions for correctness.

---

## 1. Principles

Each principle is a normative statement plus the lesson it encodes. Every design decision traces back to one of these.

**P1 — Representation before ranking.** The system is a representation problem, not a retrieval-tuning problem. Six months of knob-tuning plateaued at 76%. If a change only moves embedding scores, it is unlikely to be load-bearing.

**P2 — Corpus-aware, not query-brainstormed.** LLM query expansion regressed because 8b invents vocabulary the user never wrote. Any expansion signal comes from the indexed corpus — observed aliases, co-occurrences, granule vocabulary.

**P3 — Context is answer-critical; pruning is hostile by default.** Aggressive rerank pruning and LLM-judge filters dropped accuracy by removing load-bearing non-needle scaffolding (speaker, tone, timing, clarifications). Default response to uncertainty: keep it. The outside agent decides what to use; engram's job is to surface what could matter.

**P4 — The agent relays, it does not execute.** Structured date-math prompts regressed −8pp on the predecessor. Any computation — dates, counts, sums, aggregations — is resolved inside engram (at ingest or recall planning) and handed to the outside agent as literal strings.

**P5 — Speech acts are first-class, distinct from topics.** `all-MiniLM-L6-v2` embeds by topic. "I love X" and "X is a problem" sit near each other in topic space but are opposite in preference space. Preferences, opinions, negations, commitments get their own representation — not a cosine away.

**P6 — Every artifact has a fingerprint; no fingerprint, no cache.** Cache-invalidation traps cost days on the predecessor. Every derived artifact is addressable by a hash of every input that can change it.

**P7 — Evidence lives above noise.** Ground-truth noise floor is ±1.73pp; FP non-determinism adds another ±4pp commit-to-commit. A claimed improvement smaller than the noise budget is not an improvement.

**P8 — The benchmark is a measurement instrument, not a subject of optimization.** Benchmarking lives in a separate repo (`agent-memory-benchmark`) and calls this package through the `MemorySystem` protocol. The benchmark also implements the answerer agent that uses engram as a tool. If benchmark code, prompts, or judges are tweaked to help the memory system, measurements become non-comparable. The benchmark is hostile to self-serving changes — engram never reaches into it, never reimplements it, never special-cases for it.

**P9 — Diagnose deepest, fix deepest.** On failure, classify the miss first: extraction → retrieval → recall-output → agent. Fix the deepest broken layer. Do not patch an upper layer to compensate for a lower one.

**P10 — Graph encodes knowledge; embeddings rank candidates.** Embeddings answer "near this topic"; graphs answer "connected through this relation." Our two red buckets are graph-shaped. The rewrite is graph-first because the failures are.

**P11 — No LLM in engram's runtime path.** Primary extraction is deterministic: NER, dependency parses, rule-based claim detection, prototype-embedding classification. Recall is deterministic: similarity search + typed-edge traversal + structured assembly. An LLM enhancement layer may be added later (e.g., for ambiguous claim extraction), gated behind a budget flag, and must never become a precondition for correctness. The outside agent may be an LLM; that's the agent's concern, not engram's.

**P12 — Append-only primary; derived indexes are rebuildable.** Every Memory the agent ingests is a permanent primary record. Engram never mutates or deletes past observations. Aggregates (alias sets, co-occurrence counts, current-truth indexes, episodic clusters) are derived from primary and rebuilt, not mutated in place. Time-travel queries, attribution, and reinforcement counting fall out for free. Counting "how many times did the user say X" works because primary is preserved.

**P13 — Engram is a tool; the agent is outside.** Engram exposes two core verbs: `ingest(memory)` and `recall(query)`. The agent that uses them lives outside engram — typically an LLM with tool-calling. Engram never produces answers, constructs answer prompts, or calls an LLM. The benchmark implements the agent, exactly as production deployments will.

---

## 2. Design Rules

Engineering constraints that flow from the principles. Reviewed at PR time; violations block merges.

**R1.** Single `MemorySystem` protocol per experiment family. Only `ingest`, `recall`, `reset`, `save_state`, `load_state` are public. No sideband hooks. No conversation-id or session-id arguments on the public surface — engram instances hold one memory; isolation between agents/sessions is the caller's job via `reset` or separate instances.

**R2.** Ingestion and recall are deterministic given (config, ingested log, query). Same inputs → byte-identical serialized state, identical recall output. No wall-clock values, no PID-seeded randomness, no unsorted set iteration in output.

**R3.** `ingestion_fingerprint` includes every config field that affects node/edge production or derived-index computation. A config diff without a fingerprint diff is a bug.

**R4.** `recall_fingerprint` transitively includes `ingestion_fingerprint`. Any ingest-side change invalidates every downstream recall result.

**R5.** No LLM calls in engram's runtime paths (ingest or recall). Any LLM-based enhancement is opt-in, behind a config flag, and measured as a separate layer. The non-LLM path must always be fully functional on its own.

**R6.** No English-specific regex in intent or speech-act classification. Use prototype-embedding centroids with validated discrimination, or labeled seed queries. Hardcoded patterns overfit to benchmark phrasings.

**R7.** No LLM query expansion over raw queries. Expansion reads only corpus-derived signals (aliases, co-occurrences, granule vocabulary).

**R8.** Temporal arithmetic happens at ingest or recall planning, never in the recall output. The outside agent receives resolved absolute dates and pre-computed aggregates as literal strings.

**R9.** Recall returns a structured result (ranked granule nodes + their supporting edges + layer metadata + provenance), not a plain text answer. Shape is designed for serialization into an LLM tool-call response.

**R10.** Speech acts and topics live on separate indices or separate node properties. A preference is not a weighted sentence.

**R11.** Recall output preserves locality by default. A selected granule's temporal neighbors and source Memory are eligible for inclusion. Any pruning requires oracle-tested justification on the affected bucket.

**R12.** No unversioned persisted state. Explicit schema version on every save; load paths handle missing / old / future versions with clear errors.

**R13.** Recall is a memory tool, not an answerer. Engram's `recall` verb returns structured content for consumption by an outside LLM agent. Engram never constructs answer prompts, never calls LLMs, never produces answers. The benchmark, simulating an agent, owns the answerer prompt and the LLM call.

**R14.** Float-path determinism is declared, not assumed. Fixed seeds, sorted inputs, stable reduction order where feasible; measured and budgeted where not.

**R15.** Diagnostic classifier outputs are an enum, not a string: `extraction_miss | graph_gap | retrieval_miss | partial_retrieval | output_miss | agent_miss`. Explanations are a separate field. (`output_miss` covers cases where the right content was retrieved but recall's structured output didn't surface it usefully; `agent_miss` covers cases where the outside agent had what it needed and still answered wrong.)

**R16.** Memories are append-only; content primitives are content-addressed. Every `ingest(memory)` call produces a new Memory node. Content primitives (entities, n-grams, claim content, sentence text) are deduplicated by content-addressed ID — same content → same node. Observations of a primitive are edges from the containing granule to the primitive. Counting, time-travel, and reinforcement queries work via edge enumeration.

**R17.** Derived indexes are rebuildable, not mutable. Co-occurrence counts, alias sets, current-truth indexes, episodic clusters live in a derived layer that is recomputed from primary, not mutated in place. The derived layer carries its own fingerprint = `(ingestion_fingerprint, derivation_config_fingerprint)`.

---

## 3. Graph Architecture Sketch

High-level. Libraries, storage layout, and on-disk format are implementation choices made module-by-module.

### Five layers (kinds of memory content)

- **Episodic** — clustered memories about an entity + topic over a time span. Derived from primary observations. Example: "discussions with Alice about the project from 2026-03-01 to 2026-03-15."
- **Entity** — the nouns. People, places, things, concepts. Canonicalized; one node per (canonical_form, type), shared across all memories that reference them.
- **Relationship** — typed connections: likes / dislikes / wants / avoids / commits_to / rejects, co-occurrence, assertions. Each relationship carries an `asserted_at` attribute and an outbound `temporal_at` edge to a TimeAnchor (see Temporal layer).
- **Temporal** — time-based ordering of granules and relationships. TimeAnchor nodes and `temporal_at` / `temporal_before` / `temporal_after` edges live here. Lets us group "everything Alice said about pizza in March 2026" or "all relationships established before this Memory."
- **Semantic** — the embedding vector attached to each granule (session, turn, sentence, n-gram). Meaning, not facts. Implemented as a parallel vector index over granule node IDs; nearest-neighbor search is the recall-time entry point for semantic similarity.

### Four granularities (how finely we index source text)

Granularity is a label on each granule node, orthogonal to its layer.

- **Session** — a group of related Memories from the same source over a time window (typically derived from temporal proximity + source label, not required as input).
- **Turn** — one ingested Memory's primary content. The Memory boundary IS a Turn boundary.
- **Sentence** — a single sentence inside a Turn, segmented via spaCy.
- **N-gram** — a key phrase inside a Sentence: noun chunks (`doc.noun_chunks`) and dependency subtrees (subject + verb + object). Captures the parts of a sentence that carry information when the rest is noise.

Every ingested Memory produces nodes at every granularity that applies to its content. A Memory consisting of one sentence produces a Turn node, a Sentence node, and N-gram nodes — but no Session node until grouping kicks in. A Memory consisting of a multi-sentence document produces a Turn, multiple Sentences, and many N-grams.

### Append-only primary, rebuildable derived

**Primary nodes** are immutable once created. Memory nodes capture each ingest event with its timestamp. Granule nodes (Turn, Sentence, N-gram) capture the segmented structure. Entity nodes are created the first time their canonical form is seen and never updated thereafter — the node itself carries only `(canonical_form, type)`.

**Observations are edges.** When a Sentence "Alice likes pizza" is ingested, edges link the Sentence to the Entity "Alice", the Entity "pizza", and a Claim node "Alice-likes-pizza" (content-addressed by subject+predicate+object). The next time someone says "Alice likes pizza", the Sentence is new, the Entity nodes are reused, the Claim is reused, and a new edge from the new Sentence to the existing Claim is added. Counting reinforcements = counting edges into the Claim node.

**Derived indexes** are rebuilt from primary. Co-occurrence counts, current-preference indexes, change-event nodes, episodic clusters, alias sets — all of these are computed from primary and overwritten on rebuild. Recall reads derived for fast common queries; falls through to primary for time-travel or full-history queries.

### Edge types

Primary edges (set on ingest, never mutated):

- `part_of` — Sentence → Turn, N-gram → Sentence, Turn → Memory.
- `mentions` — Granule → Entity.
- `asserts` — Granule → Claim.
- `holds_preference` — Entity[holder] → Preference.
- `about` — Claim → Entity, Preference → Entity.
- `temporal_at` — Granule → TimeAnchor, Relationship → TimeAnchor.

Derived edges (rebuilt by `finalize_derived`):

- `co_occurs_with` — Entity ↔ Entity, weighted by per-window count normalized to [0, 1].
- `superseded_by` — Relationship → Relationship, when a later relationship contradicts an earlier one with the same (holder, target).
- `temporal_before` / `temporal_after` — Granule → Granule, derived from TimeAnchor ordering.
- `cluster_of` — EpisodicNode → Granule, derived from entity+topic+time clustering.

### Extraction pipeline (no-LLM path)

Per-Memory:

1. **Segmentation** — spaCy `doc.sents`. Produces Sentence nodes.
2. **N-gram extraction** — spaCy noun chunks + dependency subtrees. Produces N-gram nodes.
3. **NER** — spaCy NER over the Doc. Produces entity mentions.
4. **Entity canonicalization** — string similarity (NFKC + token-set ratio) within entity type. Same canonical form → same Entity node ID. Future: embedding similarity + co-occurrence as additional features.
5. **Claim extraction** — SVO from dependency parses. Produces Claim nodes (content-addressed by `(subject, predicate, object)`); each observation is an `asserts` edge from the Sentence.
6. **Preference detection** — prototype-embedding centroids with validated discrimination on a held-out set. Fails closed: below-margin observations stay as plain Claims. Each detection produces a Preference node (content-addressed) and a `holds_preference` edge from the speaker.
7. **Granule embedding** — every granule (Turn, Sentence, N-gram) gets a MiniLM embedding stored as a node attribute and indexed in the parallel vector store.
8. **Temporal anchoring** — TimeAnchor nodes for the Memory's timestamp; `temporal_at` edges from the Memory's granules and relationships.

Derived rebuild (run lazily before recall, or explicitly):

9. **Co-occurrence counts** over Entity pairs within a time window.
10. **Alias sets** by walking Entity ← `mentions` ← Granule.
11. **Current-truth indexes** for relationships per `(holder, target)`, taking the latest observation by TimeAnchor ordering.
12. **Change-event nodes** when current-truth flips.
13. **Episodic clusters** by entity+topic+time-window grouping.

If a step's output quality is not validated above noise on a held-out set, that step does not ship. P11 applies: the non-LLM path stands on its own.

### Recall (subgraph retrieval, no LLM)

1. **Intent classification** — corpus-prototype embeddings (R6) over the agent's query. Intents: single-fact, aggregation, preference, temporal, entity-resolution.
2. **Seeding** — semantic-layer nearest-neighbor search over granule embeddings; intent-weighted across granularities (a temporal query may seed Session/Turn-level nodes; a preference query may seed N-gram-level nodes).
3. **Expansion** — bounded typed-edge walk from seeds. Per-intent edge weights — aggregation expands `co_occurs_with` + `cluster_of`; preference expands `holds_preference` + `about` + `superseded_by`; temporal expands `temporal_before` + `temporal_after`.
4. **Scoring** — walk-derived score per node = `seed_similarity × edge_weight_product × depth_decay`. No cross-encoder reranker in v1: walk scores are legible and tunable; we add a reranker only when diagnostics shows ranking quality is the bottleneck.
5. **Assembly** — structured `RecallResult` with ranked granule passages, supporting edges, layer metadata, and provenance (source Memory IDs and timestamps). The agent gets text plus structured signals.

The `RecallResult` shape is iterated based on agent behavior — first cut is a list of passages with metadata; richer structures (timelines, change events, derived facts) added when diagnostics shows the agent benefits.

### LLM enhancement layer (future)

Optional, flag-gated. Conceivable enhancements:

- Claim extraction from sentences where dependency parses are ambiguous.
- Preference-polarity confirmation where embedding confidence is low.
- Episodic-cluster summarization for richer recall output.

Each is an opt-in experiment with its own hypothesis and evidence bar (M2). None is a precondition for the system being correct.

---

## 4. KPIs

Priority order with targets. Non-negotiable in regression analysis.

**K1 — LongMemEval-s 100q accuracy (primary).** Target 100%. Minimum acceptable post-rewrite 80%. Reported as mean ± std over 3 replicates. Claim threshold: `current_score + 2 × 1.73pp`. Measured against the benchmark's reference agent (single answerer call per question with `llama3.1:8b` and engram as the recall tool).

**K2 — Per-bucket accuracy.** Multi-session-aggregation ≥ 90% (was 53.8%). Single-session-preference ≥ 90% (was 16.7%). No other bucket regresses by more than 1 question.

**K3 — Retrieval diagnostics (per-bucket).**
- `needle_recall@k` — all gold turns present in retrieved subgraph.
- `granule_density` — fraction of recall output drawn from gold granules.
- `completeness` — gold entity mentions present.
- `retrieval_miss` rate ≤ 5% of wrong answers.
- `extraction_miss` rate — tracked separately from retrieval.

**K4 — Latency.** p50 ingest per Memory, p50 recall per query reported; p95 agent end-to-end (benchmark-side) reported. Engram-side targets: ingest p50 < 200ms per Memory, recall p50 < 100ms per query (excluding agent's LLM call). Not optimized until accuracy targets are hit.

**K5 — LLM call discipline.**
- Engram LLM calls per ingest — exactly 0. (R5)
- Engram LLM calls per recall — exactly 0. (R5, R13)
- Benchmark agent LLM calls per question — exactly 1 (the answerer). The benchmark may evolve its agent design (more tool calls, multi-step reasoning) in later experiments; each variant is a separate run with its own scorecard.

**K6 — Stability.** Ground-truth noise floor ±1.73pp. FP non-determinism budget ±4pp commit-to-commit. Claims within these budgets require 3 replicates; within 2× requires 5.

**K7 — Fingerprint discipline.**
- Ingestion cache hit rate on unchanged-config rerun: 100%.
- Recall cache invalidation on ingestion-fingerprint change: 100%.
- Verified by a fingerprint-audit test that fails CI if violated.

### Minimum per-commit scoreboard

Every commit touching memory code emits:
- LME-s 100q accuracy (1 replicate quick signal; 3 for claims).
- Per-bucket delta vs main.
- `{extraction_miss, graph_gap, retrieval_miss, partial_retrieval, output_miss, agent_miss}` counts.
- Ingest p50 and recall p50.
- Ingestion cache hit rate on warm rerun.
- Commit SHA, ingestion_fingerprint, recall_fingerprint.

---

## 5. Optimization Methodology

These rules prevent the specific failure modes that burned the predecessor.

**M1 — Every change starts as a written hypothesis.** Format: `(target bucket | expected gain in pp | mechanism in 1–2 sentences | validation threshold | falsification condition)`. No hypothesis, no branch.

**M2 — Minimum evidence bar.** One of:
- ≥ 1.73pp full-benchmark gain across 3 replicates, lower CI bound above zero.
- Oracle-subgraph test showing mechanism improves a diagnosed bucket by ≥ 2 questions, with no regression elsewhere on full benchmark.
- Diagnostic-only change that improves signal quality, with no scored delta required.

**M3 — Cache-tainted ablations are invalid.** Every ablation run starts from a clean recall cache (ingestion cache may persist iff the change is post-ingest). The runner refuses to publish a result if the recall cache predates the relevant fingerprint.

**M4 — Retrieval-only improvements do not ship without full-benchmark runs.** `needle_recall@k` is necessary but not sufficient. Pruning has a history of improving retrieval metrics while regressing accuracy.

**M5 — Layer decision tree before coding.** Classify the failure first:
- Gold fact present in recall output, agent misses → output structure or agent.
- Gold fact retrieved but not surfaced in `RecallResult` → output structure (R9).
- Gold fact absent from recall output, present in graph → retrieval (seeding / expansion / scoring).
- Gold fact absent from graph → extraction or ingestion.
Fix the deepest broken layer (P9). Never patch above to compensate for below.

**M6 — Noise-budget-aware replicates.** 1 for sanity; 3 within ±4pp; 5 within 2×.

**M7 — Kill criteria.** When the hypothesis's falsification condition is met, the branch is abandoned. Revival requires a new hypothesis citing why the original failed.

**M8 — Diagnostics runs first on every failure.** Classification logged with the commit.

**M9 — Competitor features enter as hypotheses, not as code.** Hindsight's four-network design, CARA, etc. are candidate experiments each with their own hypothesis and threshold.

---

## 6. Module Boundaries

Three top-level modules inside this repo. Each owns a strict slice; cross-module calls go through the public verbs only. Benchmarking lives in a separate repo (`agent-memory-benchmark`) which also implements the answerer agent that uses engram as a tool.

### Ingestion

- **Responsibility:** Convert each ingested Memory into a populated graph with a deterministic fingerprint. Maintain primary append-only data and rebuild derived indexes.
- **Public surface (via `MemorySystem`):** `ingest(memory)`. Internal verbs: per-stage extractors, derived-rebuild orchestrator, persistence.
- **Owns:** segmentation, n-gram extraction, NER, entity canonicalization, claim/preference extraction, granule embedding + vector index, temporal anchoring, derived-index rebuilds (co-occurrence, current-truth, change events, episodic clusters), ingestion fingerprinting.
- **Does not touch:** query text, recall output formatting, benchmark orchestration, judge prompts, the agent's prompt template.
- **Stability:** fingerprint fully covers output state; identical fingerprint ⇒ identical graph.

### Recall

- **Responsibility:** Given a query and the current graph, produce a structured `RecallResult` that an outside agent can serialize into a tool-call response.
- **Public surface (via `MemorySystem`):** `recall(query, *, now=None, timezone=None, max_results=None)`. Internal verbs: `classify_intent`, `seed`, `expand`, `score`, `assemble`.
- **Owns:** intent classification, seeding (semantic + entity-anchored), bounded typed-edge expansion, walk scoring, structured output assembly.
- **Does not touch:** node/edge creation, derived-index rebuilds, benchmark scoring, judge prompts, the agent's LLM call.
- **Stability:** recall fingerprint transitively covers ingestion fingerprint plus all recall-side config.

### Diagnostics

- **Responsibility:** Given a benchmark run's results plus optional gold annotations, classify each failure and surface actionable patterns.
- **Public verbs:** `classify_failures`, `bucket_breakdown`, `needle_overlap`, `extraction_coverage`, `fingerprint_audit`.
- **Owns:** failure classification (R15), gold-term overlap, extraction-coverage reports, fingerprint-discipline audits, regression reports.
- **Does not touch:** runtime path. Read-only. Never writes to caches. Never mutates run artifacts.
- **Stability:** may evolve freely, but classifications are versioned.

### Benchmarking + answerer agent (external — `agent-memory-benchmark` repo)

Out of scope for this repository. The external benchmark owns dataset loaders, judge abstraction + prompts, run directory layout, scorecard rendering, cache layout, replicate orchestration, **and the answerer agent itself** (the LLM loop that calls engram's `recall` tool and produces the answer). It reads only the `MemorySystem` protocol surface and never inspects engram's graph internals. It must be bit-stable across memory-system rewrites; judge or agent-prompt changes there require an explicit re-baseline.

Engram's responsibility is to hold the protocol stable (R1), hold the fingerprints honest (R3/R4), and never "help" the benchmark or the agent.

---

## 7. Reference material in `agent-memory` (read, don't copy)

These files in the sibling `agent-memory` repo encode concepts that survive the rewrite. Read them for shape; do not port code.

**For engram (this repo):**

- `benchmark/memory_interface.py` — `MemorySystem` ABC, `AnswerResult`. Shape of R1 — but note the rewrite drops `answer_question` (engram no longer answers).
- `memory/config.py` — dual-fingerprint discipline. Shape of R3, R4, P6.
- `memory/ingestion/ner.py` — batched GPU NER pattern. Shape of R5 (batching invariant applies even without LLM).
- `memory/ingestion/episodes.py` — episode clustering concept. Promoted to derived layer in the rewrite.
- `memory/retrieval/reranker.py` — cross-encoder integration pattern. Reference only; v1 recall ships without a reranker (walk scores are the ranking signal).
- `scripts/diagnose_lme.py` — failure-classification kernel. Seed of engram's Diagnostics taxonomy (R15).
- `.agent/lessons.md` — hard-won rules from the predecessor. Ported key entries into this repo's `.agent/lessons.md` as seed lessons.
- `.agent/architecture-brief.md` (DECISIONS block) — frozen decisions on the predecessor.

**For the external `agent-memory-benchmark` repo (read from there, not engram):**

- `benchmark/runner.py` — ingestion/recall/judge cache layering, replicate orchestration, resume semantics. Shape of M3, R12.
- `benchmark/cache.py` — cache key construction, dataset hashing, prompt fingerprint. Shape of P6.
- `benchmark/datasets/longmemeval.py`, `benchmark/datasets/locomo.py` — dataset interfaces.

---

## 8. Anti-Patterns — explicit DO NOT

- **Do not call an LLM from inside engram.** P11, P13, R5, R13. Not at ingest, not at recall. The outside agent may use an LLM; that's not engram's runtime path.
- **Do not produce answers from engram.** Engram returns structured retrieval output; the agent answers. R9, R13.
- **Do not add LLM query expansion over raw queries.** P2, R7.
- **Do not bake date arithmetic, counts, or aggregations into the agent's response path.** P4, R8. Pre-compute inside engram and surface as literal strings.
- **Do not use English-specific regex for intent or speech-act classification.** R6.
- **Do not use topic-embedding cosine as a preference matcher without validated discrimination.** P5, R10.
- **Do not add an LLM-judge filter between recall and the agent.** P3.
- **Do not ship aggressive output pruning without oracle validation on the affected bucket.** P3, R11.
- **Do not add a config field without adding it to the correct fingerprint.** P6, R3, R4.
- **Do not let the external benchmark "help" the memory system, and do not reach into it from here.** P8. No conditionals on question-type, dataset, or gold annotations in engram code.
- **Do not patch one layer to paper over a miss in a deeper layer.** P9, M5.
- **Do not claim an improvement smaller than ±1.73pp on a single replicate.** P7, M2, M6.
- **Do not persist ingestion state via unversioned pickle.** R12.
- **Do not mutate primary nodes after they're created.** P12, R16. Updates to "current truth" live in the derived layer.
- **Do not deduplicate Memory ingest events.** P12, R16. Each `ingest` call is a permanent observation; reinforcement is counted via edge enumeration.
- **Do not mint a new Node or Edge type without adding it to the ingestion fingerprint and wiring it into extraction.** R3, R5.
- **Do not import competitor ideas as code.** M9.
- **Do not run an ablation against a warm cache whose fingerprint predates the change.** M3.
- **Do not make the default path depend on an LLM at ingest or at recall.** P11, R5.

---

## Verification — how we know the rewrite works

Before any extraction/retrieval implementation lands:

1. **Manifesto approved** (this document).
2. **Skeleton modules created** — three module directories (`ingestion/`, `recall/`, `diagnostics/`) with the module-boundary docstrings from §6 as the top-of-module comment, and one stub test per module that imports it.
3. **`MemorySystem` protocol defined** — the five public verbs from §6, typed, with docstrings citing the relevant rules.
4. **Fingerprint discipline test** — a CI test that creates two configs differing only on one field, verifies fingerprints differ; and two configs identical in all fields, verifies fingerprints match. Fails CI otherwise.
5. **External benchmark integration smoke** — the `agent-memory-benchmark` repo imports engram (editable install / path import), wires its agent to call engram's `recall` tool, runs against LongMemEval-s, and produces a scorecard. This lives in the benchmark repo; engram's role is to stay installable and protocol-stable.

Only after 1–5 are green does implementation of actual extraction/retrieval begin. Every subsequent PR must cite the rule(s) it implements or the hypothesis (M1) it tests.

After implementation begins, the rewrite is considered a success when:

- LME-s 100q accuracy ≥ 80% (minimum acceptable, §K1), with a credible measured path to 100% supported by the per-bucket scoreboard (§K2).
- Multi-session-aggregation ≥ 75% and SSP ≥ 75% — the two currently-red buckets moved decisively.
- `retrieval_miss` and `extraction_miss` rates below the targets in §K3.
- Fingerprint-discipline test and replicate-eval scoreboard stable across 3 consecutive commits on `main`.
