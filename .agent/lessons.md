# Engram — Agent Lessons Learned

> Read this file at the START of every session. These are mistakes that have already been made — many inherited from the predecessor `agent-memory` project. Don't repeat them. Add new entries when things go wrong.

---

## 2026-04-20 — Knob-tuning plateau triggered the rewrite

**What happened:** The predecessor `agent-memory` system spent two weeks iterating on knobs (reranker top_n, preference layer, LLM query expansion, temporal math prompts) and moved from 76% to 76% on LongMemEval-s 100q.
**Root cause:** Flat multi-layer RRF architecture had exhausted its tunable range. The remaining 10–15pp lived in what we store and how we connect it (structure), not in weights or widths.
**What I should have done:** Pivoted to structural/representational changes once knob-tuning showed sub-noise-floor gains across 3+ consecutive attempts.
**Rule:** When three successive experiments all land inside the noise floor (±1.73pp), stop tuning and audit the representation.

## 2026-04-20 — `--no-cache-ingestion` is a trap

**What happened:** Tested a retrieval change with ingestion cache busted but answer cache intact. Scores were identical to baseline, implying "no change" — but the answer cache was serving baseline answers despite new retrieval.
**Root cause:** Answer cache key did not transitively include ingestion fingerprint. Ingestion changed; answer cache didn't invalidate.
**What I should have done:** Either fully busted the answer cache or ensured `answer_fingerprint` transitively included `ingestion_fingerprint`.
**Rule:** After ANY ingestion or retrieval code change, use `--no-cache` (full bust) OR verify fingerprint-invalidation behavior first. A config diff without a fingerprint diff is a bug. `answer_fingerprint` MUST transitively include `ingestion_fingerprint` (manifesto R4).

## 2026-04-20 — Cache-tainted ablations invalidate conclusions

**What happened:** A 2026-03-22 ablation claimed all variants scored equivalently (~75%); a refactor built on that conclusion regressed −33pp when re-run with cache-bust.
**Root cause:** Cache keys were identical across configs, so all variants served the same cached answers.
**What I should have done:** Verified each config produces different answers before trusting "all equivalent" ablation results.
**Rule:** Before trusting an ablation, audit the cache: verify each config produces different `answer_fingerprint` values, OR start every ablation run from a clean answer cache (manifesto M3). The runner should refuse to publish results if the answer cache predates the relevant fingerprint.

## 2026-04-20 — Retrieval-only recall lifts don't translate to 100q

**What happened:** Session-diverse keyword expansion gained +1 needle on 6 oracle-annotated qids but regressed −4pp on the full 100q benchmark.
**Root cause:** Retrieval metrics (needle_recall@k, session_density) ignore non-needle context. Pruning improved retrieval metrics while removing load-bearing reasoning scaffolding the 8b answerer used.
**What I should have done:** Required a full-benchmark run before shipping any retrieval change.
**Rule:** Retrieval-only improvements never ship without a full-benchmark A/B (manifesto M4). Needle_recall is necessary, not sufficient.

## 2026-04-20 — Structured procedure prompts regress on 8b

**What happened:** Introduced explicit CASE-A (same-month) / CASE-B (cross-month) date-math branches into the prompt; regressed −8pp on 20 temporal-math qids.
**Root cause:** `llama3.1:8b` mechanically applies wrong branches when procedures are spelled out. It relays pre-computed values reliably but does not execute structured multi-step procedures.
**What I should have done:** Computed the date-diff programmatically at ingest or recall planning, handed the answerer a literal.
**Rule:** Temporal/arithmetic computation happens at ingest or recall planning, never in the answer prompt (manifesto P4, R8). The answerer relays literals; it does not execute procedures.

## 2026-04-20 — LLM query expansion regresses on personal corpora

**What happened:** Tested LLM-driven query expansion at retrieval time on 11 oracle qids (34 needles); yielded +1 needle total, failed evidence gate 3×.
**Root cause:** The expansion LLM brainstormed brand names and synonyms that had zero overlap with the user's idiosyncratic corpus vocabulary.
**What I should have done:** Derived expansion signals from the indexed corpus (observed aliases, co-occurrences, session vocabulary) — not from the query LLM's training distribution.
**Rule:** No LLM query expansion over raw queries (manifesto R7). Expansion reads only corpus-derived signals. P2 — corpus-aware, not query-brainstormed.

## 2026-04-20 — LLM-judge filter between retrieval and answerer drops accuracy

**What happened:** Added a post-retrieval LLM filter to prune non-needle units. 30q benchmark: −16.7pp micro, −20pp macro. Needle density went up 13% → 22% but accuracy regressed.
**Root cause:** Non-needle units carry load-bearing reasoning scaffolding (speaker, tone, timing, clarifications) the 8b answerer uses. Pruning them is hostile.
**What I should have done:** Accepted that context pruning is almost always net-negative, even when retrieval metrics improve.
**Rule:** Do not add an LLM-judge filter between retrieval and the answerer (manifesto anti-pattern). Context is answer-critical; pruning is hostile by default (P3).

## 2026-04-20 — Sentence-transformers embed by topic, not speech act

**What happened:** Tried sentence-level preference detection via latent-concept centroid matching on MiniLM embeddings. Held-out p90 (0.26) was indistinguishable from random user sentences (p90 = 0.28).
**Root cause:** `all-MiniLM-L6-v2` embeds by topic. "I love hiking" and "hiking is a problem" sit close in topic space but are opposites in preference space. The embedding geometry doesn't encode speech-act polarity.
**What I should have done:** Either used an NLI-tuned or sentiment-aware embedder for speech-act classification, or made speech acts a first-class node property rather than a cosine threshold.
**Rule:** Speech acts and topics live on separate indices or node properties (manifesto P5, R10). A preference is not a weighted sentence. Do not use topic-embedding cosine as a preference matcher without validated discrimination on a held-out set.

## 2026-04-20 — Commit-to-commit floating-point non-determinism is ±4pp

**What happened:** Two commits with bit-identical config produced 75% and 78% on the same 100q benchmark. Retrieval-unit counts drifted ±5–10 per question.
**Root cause:** FAISS + sentence-transformers + llama.cpp all have small FP non-determinism across builds/runtime environments. Cumulative across 100q questions and 10 retrieval + 1 answer call each.
**What I should have done:** Always paired a candidate run with a fresh baseline on the same commit, and budgeted ±4pp into all comparisons.
**Rule:** Ground-truth noise floor is ±1.73pp; commit-to-commit FP non-determinism adds ±4pp (manifesto P7, K6). Claims within ±4pp require 3 replicates; within 2× require 5.

## 2026-04-20 — Oracle-test prompt silently drifted from production prompt

**What happened:** `oracle_test.py` and `phase0_sweep.py` hand-built their own prompts that diverged from the production pipeline when config gates (`enumerate_on_aggregation`) changed. One qid flipped "38"→"44" when fixed.
**Root cause:** Prompt construction was duplicated across diagnostic tools instead of going through a shared builder.
**What I should have done:** Had diagnostic tools import the same `build_answer_prompt` the production pipeline uses.
**Rule:** The answerer prompt is a single file-owned template (manifesto R13). No f-strings scattered across modules. Diagnostic tools MUST use the same builder as production.

## 2026-04-20 — Sweep meta.json can silently drift from the real baseline

**What happened:** Sweep `BASELINE_OVERRIDES` silently diverged from the baseline's CLI flags (e.g., `keyword_expand_top_k` defaulted back to None). Comparisons were invalid.
**Root cause:** Sweep meta didn't dump full argv; reviewers compared only the fields they thought they'd changed.
**What I should have done:** Always dumped full argv + full memory_config to a companion file; diffed ALL fields.
**Rule:** For any tagged sweep or benchmark run, dump the full argv and full config to a companion file. Diff ALL memory_config fields, not just the ones you think you changed.
