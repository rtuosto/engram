# Recall — Design

> **Scope.** The recall side of engram under the post-pivot architecture. Engram is a memory tool for an outside agent; `recall(query)` is one of the two core verbs. This doc covers: the recall signature, the five-stage pipeline (intent → seed → expand → score → assemble), the `RecallResult` shape, intent classification, seeding strategy, bounded typed-edge expansion, scoring, output assembly with pre-computed facts, dependencies on ingestion patches, tests, and provisional values.
> **Status.** Draft for review. The binding rules are in [`../DESIGN-MANIFESTO.md`](../DESIGN-MANIFESTO.md); this doc is implementation-level and evolves as the module is built.
> **Architecture context.** The graph and ingestion design that recall reads from is in [`ingestion.md`](ingestion.md).

---

## 1. The recall verb

`recall` is one of the two public verbs on `MemorySystem`. It takes a query string and optional deterministic system context, and returns a structured result the outside agent can consume.

```python
async def recall(
    self,
    query: str,
    *,
    now: str | None = None,                # ISO-8601, used to resolve "yesterday" etc.
    timezone: str | None = None,           # IANA name (e.g., "America/New_York")
    max_passages: int | None = None,       # cap on returned passages; None = config default
    intent_hint: str | None = None,        # optional override; engram still classifies
) -> RecallResult: ...
```

**No conversation_id, no session scoping.** One engram instance holds one memory; the agent calls `reset()` between isolated workloads, or instantiates separate engram instances if it needs to keep memories partitioned.

**Engram makes zero LLM calls.** Recall is deterministic NLP + vector search + typed-edge graph traversal + structured assembly.

---

## 2. The `RecallResult` shape

Recall returns a structured object the agent serializes into its tool-call response. The shape is intentionally first-cut and iterates as we measure agent behavior. Two top-level fields:

- `passages` — ranked text snippets (granule content) with provenance and supporting edges. The "search results" view.
- `facts` — pre-computed answers from derived indexes (current truth, reinforcement counts, change events). The "we already worked it out for you" view.

```python
@dataclass(frozen=True, slots=True)
class RecallPassage:
    text: str                              # what the agent puts in its context
    granularity: str                       # "session" | "turn" | "sentence" | "ngram"
    score: float                           # walk-derived score; higher = more relevant
    source_memory_id: str                  # provenance — the Memory this came from
    source_memory_index: int               # monotonic ingest order
    timestamp: str | None                  # ISO-8601 from the source TimeAnchor
    speaker: str | None
    supporting_edges: tuple[str, ...]      # human-readable: "Alice asserts likes(pizza)"


@dataclass(frozen=True, slots=True)
class RecallFact:
    """A pre-computed answer the agent can cite directly.

    Engram resolves arithmetic, aggregations, and current-truth lookups
    (P4, R8) so the agent never has to compute them.
    """
    kind: str                              # "current_preference" | "reinforcement" | "change_event" | "co_occurrence"
    subject: str                           # canonical entity form, e.g., "alice"
    predicate: str | None                  # e.g., "likes"
    object: str | None                     # e.g., "pizza"
    value: str                             # rendered literal, e.g., "47 times since 2026-01-15"
    timestamp: str | None                  # latest observation when applicable
    supporting_memory_ids: tuple[str, ...] # provenance


@dataclass(frozen=True, slots=True)
class RecallResult:
    passages: tuple[RecallPassage, ...]
    facts: tuple[RecallFact, ...]
    intent: str                            # classified intent label
    intent_confidence: float               # discrimination margin; ~0 means weakly-classified
    timing_ms: dict[str, float]            # stage-level: classify, seed, expand, score, assemble, total
    recall_fingerprint: str                # for cache keys and audit
```

**Why facts as a separate field.** The outside agent shouldn't have to count, sort, or do date math (P4, R8). If the agent asks "what does Alice think about pizza?", we should hand back `RecallFact(kind="current_preference", subject="alice", object="pizza", value="dislikes (as of 2026-03-05; previously liked, 47 reinforcements over 6 months)")` plus the supporting passages — so the agent can answer with the literal string OR cite the underlying observations as needed.

**Iteration is expected.** First cut ships `passages` + `facts` only. Diagnostics will tell us whether agents benefit from richer structure (timelines, comparison summaries, source-clustered views). Each addition is an M1-hypothesis PR.

---

## 3. Five-stage pipeline

```
query (str) + RecallContext
     │
     ▼
[1] Intent classification          → intent label + confidence
     │
     ▼
[2] Seeding                         → set of (granule_node_id, seed_score)
     │
     ▼
[3] Expansion (typed-edge BFS)      → scored subgraph of granules + relationships
     │
     ▼
[4] Scoring                         → final per-node score; top-k selection
     │
     ▼
[5] Assembly                        → RecallResult (passages + facts + provenance)
```

No reranker stage in v1. Walk scores are the ranking signal — legible, tunable, owned. A reranker enters the design only when diagnostics shows a specific ranking-quality bottleneck that walk scores can't fix.

---

## 4. [1] Intent classification

### Intents

Five intents (matching the predecessor's red/yellow bucket structure):

- `single_fact` — "What did Alice say about the project?"
- `aggregation` — "How many times have I mentioned my sister?", "What's everyone's favorite food?"
- `preference` — "Does Alice like pizza?", "What are my dietary restrictions?"
- `temporal` — "When did Alice change her mind?", "What did we discuss yesterday?"
- `entity_resolution` — "Who is the project manager?", "Where is Bob from?"

### Method (R6)

Prototype-embedding centroids per intent. Same pattern as preference detection in ingestion (§5 of [`ingestion.md`](ingestion.md)).

- For each intent, hand-author 10–20 synthetic seed queries. Speaker-agnostic. Never drawn from LME-s / LOCOMO (P8, M3, M4).
- Stored at `engram/recall/intents/seeds.json`. Content hash → `MemoryConfig.intent_seed_hash`. Editing seeds invalidates the recall fingerprint (R3, R4).
- Centroid = mean of seed embeddings per intent (using the same `embedding_model` as ingestion's granule embeddings — `all-MiniLM-L6-v2`).
- Held-out set at `engram/recall/intents/heldout.json` with 30–60 disjoint queries. Per-intent median discrimination margin gates that intent's downstream weight profile.

### Runtime

```python
def classify_intent(query, centroids, embed_fn, *, margin_threshold, fallback="single_fact"):
    e = embed_fn([query])[0]
    e = e / norm(e) if norm(e) > 0 else e
    scores = {intent: float(e @ centroids[intent]) for intent in INTENTS}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_intent, top_score = ranked[0]
    second_score = ranked[1][1]
    margin = top_score - second_score
    if margin < margin_threshold:
        return fallback, margin            # below margin: default intent, signal weak classification
    return top_intent, margin
```

### Fails closed

Below-margin queries default to `single_fact` and report low `intent_confidence` in the `RecallResult`. The agent sees this and can choose to re-query with a clarified phrasing.

### Config

- `MemoryConfig.intent_seed_hash: str` — derived from the seeds file content.
- `MemoryConfig.intent_discrimination_margin: float = 0.05` (provisional).

Both in `_RECALL_FIELDS` (new partition; recall fields are answer-side under the existing R3/R4 model).

---

## 5. [2] Seeding

Find the granule nodes that match the query semantically (and where applicable, structurally).

### Two seeding sources

1. **Semantic seeding** — vector-index nearest-neighbor search on granule embeddings.
2. **Entity-anchored seeding** — if the query mentions a known entity, seed the entity node and its directly-connected granules.

### Semantic seeding

```python
def semantic_seed(query, vector_index, *, top_n_per_granularity, intent_weights):
    e = embed_fn([query])[0]
    e = e / norm(e)
    # vector_index supports per-granularity slicing
    hits = []
    for granularity in ("session", "turn", "sentence", "ngram"):
        n = top_n_per_granularity[granularity] * intent_weights[granularity]
        for node_id, similarity in vector_index.knn(e, granularity=granularity, k=n):
            hits.append((node_id, similarity * intent_weights[granularity]))
    return dedup_by_id_keeping_max_score(hits)
```

`intent_weights[granularity]` lets each intent prefer different granularities:
- `single_fact` — favors sentence + n-gram (specific phrasing matters).
- `aggregation` — favors turn + session (broader context).
- `preference` — favors n-gram (preference statements are often short phrases).
- `temporal` — favors session + turn (time spans).
- `entity_resolution` — favors n-gram (entity names live in noun chunks).

### Entity-anchored seeding

```python
def entity_seed(query, doc_for_query, entity_index):
    seeds = []
    # spaCy NER over the query
    for ent in doc_for_query.ents:
        normalized = normalize(ent.text)
        if normalized in entity_index.by_normalized_form[ent.label_]:
            entity_node = entity_index.by_normalized_form[ent.label_][normalized]
            # Seed the entity itself + its directly-mentioned granules
            seeds.append((entity_node.id, 1.0))
            for granule_id, _ in graph.in_edges(entity_node.id, edge_type="mentions"):
                seeds.append((granule_id, 0.8))
    return seeds
```

Combined seed set = semantic seeds + entity-anchored seeds, deduplicated by node ID keeping the maximum score.

### Config

- `MemoryConfig.recall_top_n_per_granularity_<intent>_<granularity>: int` — flattened (5 intents × 4 granularities = 20 fields) OR a single hash field over a JSON weights file. Decision: **single hash field** (`recall_seeding_weights_hash`) over `engram/recall/weights.json`, mirroring the preference-seed pattern. Weights file is human-editable, fingerprint-tracked.
- `MemoryConfig.recall_seed_count_total: int = 64` — cap on total seeds before expansion.

---

## 6. [3] Expansion — bounded typed-edge BFS

Take the seed set, walk along typed edges with per-intent edge weights, accumulate scores.

### Edge weights per intent

For each intent, a vector of weights over edge types — `dict[str, float]`. Examples:

- `single_fact` — high weight on `asserts`, `mentions`, `about`; lower on `co_occurs_with`.
- `aggregation` — high weight on `co_occurs_with`, `cluster_of` (episodic), `mentions`; lower on `asserts`.
- `preference` — high weight on `holds_preference`, `about`, `superseded_by`; medium on `mentions`.
- `temporal` — high weight on `temporal_at`, `temporal_before`, `temporal_after`; medium on `asserts`.
- `entity_resolution` — high weight on `mentions`, `co_occurs_with`; lower on relationship edges.

Total search space across 5 intents × ~12 edge types ≈ 60 scalars. Stored in `engram/recall/weights.json` alongside seeding weights. Optimized later by a Diagnostics-owned tuner against `needle_recall@k`.

### Walk

Reuses `GraphStore.bfs(seeds, edge_weights, *, max_depth, max_frontier)` from the Tier-1 implementation:

```python
def expand(seeds, edge_weights, *, max_depth, max_frontier):
    """Returns {node_id: score} where score = seed_score × edge_weight_product × depth_decay."""
    return graph.bfs(seeds, edge_weights, max_depth=max_depth, max_frontier=max_frontier)
```

### Config

- `MemoryConfig.recall_max_depth: int = 3`
- `MemoryConfig.recall_max_frontier: int = 256`
- `MemoryConfig.recall_edge_weights_hash: str` — content hash of the weights file.

### Why no reranker (v1)

Walk scores already encode:
- **Seed similarity** — how close the granule is to the query semantically.
- **Edge-weight product** — how much the path supports the intent.
- **Depth decay** — closer to seed = more relevant.

These are legible (you can read off why a node scored what it did). A learned reranker is a black box trained on (query, passage) data we don't own. Until diagnostics shows walk-score-based ranking putting the wrong thing at rank 1 in a specific bucket, we don't add one.

---

## 7. [4] Scoring and selection

Score per candidate node = walk's accumulated score. Sort descending, take top-k, deduplicate by source granule (don't return the same Sentence twice as both a Sentence-level and N-gram-level hit; keep the most specific granule with the highest score).

```python
def score_and_select(walk_scores, *, max_passages):
    by_granule = defaultdict(lambda: (0.0, None))   # granule_id -> (score, contributor_node_id)
    for node_id, score in walk_scores.items():
        granule_id = resolve_to_containing_granule(node_id)
        if score > by_granule[granule_id][0]:
            by_granule[granule_id] = (score, node_id)
    ranked = sorted(by_granule.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(granule_id, score) for granule_id, (score, _) in ranked[:max_passages]]
```

`max_passages` defaults from `MemoryConfig.recall_max_passages: int = 16` and may be overridden per-call.

---

## 8. [5] Assembly — RecallResult

Build the `RecallResult` from selected granules and pre-computed derived facts.

### Passages

For each selected granule:

```python
def build_passage(granule_id, score, graph):
    granule = graph.nodes[granule_id]
    # Extract supporting edges — render Claims and Preferences this granule asserts
    supporting = []
    for _, claim_id, attrs in graph.out_edges(granule_id, edge_type="asserts"):
        supporting.append(render_claim(claim_id, graph))   # "Alice asserts likes(pizza)"
    for _, pref_id, attrs in graph.out_edges(granule_id, edge_type="holds_preference_observation"):
        supporting.append(render_preference(pref_id, graph))
    timestamp = first_outbound_timeanchor(granule_id, graph)
    return RecallPassage(
        text=granule.text,
        granularity=granule.granularity,
        score=score,
        source_memory_id=granule.parent_memory_id,
        source_memory_index=resolve_memory_index(granule_id, graph),
        timestamp=timestamp,
        speaker=resolve_speaker(granule_id, graph),
        supporting_edges=tuple(supporting),
    )
```

### Facts (the pre-computed answers)

After selecting passages, look up derived-index entries that are likely relevant:

- **Current-preference facts.** For each (holder, target) pair appearing in the passages or query, look up the current-truth index. If a current preference exists, emit a `RecallFact(kind="current_preference", ..., value="dislikes (as of 2026-03-05; previously liked, 47 reinforcements)")`.
- **Reinforcement counts.** For each Claim or Preference appearing in the passages, look up its reinforcement count and emit a fact: `RecallFact(kind="reinforcement", value="47 observations between 2026-01-15 and 2026-03-05")`.
- **Change events.** If a current-truth flip exists for a (holder, target) in the query's scope, emit a `RecallFact(kind="change_event", value="changed from likes to dislikes on 2026-03-02")`.
- **Co-occurrence facts.** For aggregation intents, surface top-k co-occurring entities for any entity in the query.

Facts are intent-shaped — different intents return different fact mixes:

| Intent | Default facts surfaced |
|---|---|
| `single_fact` | reinforcement (only if asked-about claim exists) |
| `aggregation` | co_occurrence, reinforcement |
| `preference` | current_preference, change_event, reinforcement |
| `temporal` | change_event, reinforcement (with timestamps) |
| `entity_resolution` | (none — passages carry the answer) |

### Locality preservation (R11)

When a Sentence-level passage is selected, the assembly step considers including its neighboring sentences (within the same Turn) if they're not already in the passage list and if doing so would help the agent. Default: include the previous and next sentence at half-score, marked `granularity="sentence"` with a "context_for=<original granule_id>" attribute on `supporting_edges`. Pruning this context requires oracle-tested justification.

### Resolved literals only (R8, P4)

Every timestamp in the output is resolved to ISO-8601 absolute. "Yesterday" in the query is resolved against `RecallContext.now` before any facts are emitted. The agent never needs to do date math.

### Recall fingerprint

`recall_fingerprint = sha256(ingestion_fingerprint || recall_config_fingerprint || query || RecallContext)`. The benchmark uses this in its cache key.

---

## 9. Recall context

The agent supplies optional deterministic system info via the `recall` call's keyword args.

- `now: str | None` — ISO-8601 timestamp used to resolve relative time expressions ("yesterday", "last week", "this morning") in the query AND in derived-fact rendering. If absent, relative expressions in the query stay unresolved (engram does not invent a "now").
- `timezone: str | None` — IANA name. If absent, all timestamps are treated as UTC.
- `max_passages` and `intent_hint` are per-call overrides on config defaults.

`RecallContext` is its own frozen dataclass for forward extension (future fields might include `agent_id`, `priority_entities`, etc.).

---

## 10. Operational policy

**Async (R1).** `recall` is `async`; implementation wraps synchronous embedding + traversal via `asyncio.to_thread`.

**Lazy derived rebuild.** `recall` checks the derived-index fingerprint; if stale relative to primary, calls `rebuild_derived()` before serving. Most of the time this is a no-op (primary unchanged since last rebuild).

**No LLM calls (R5, R13).** Verified by a CI test that mocks any LLM client and asserts it's never called during a recall path.

**Float-determinism budget (R14, K6).** Same ±2pp budget as ingestion. Vector search uses sorted tie-breaking by node ID.

**Recall fingerprint covers the full input.** Test: same `(config, ingest log, query, RecallContext)` → byte-identical `RecallResult`. R2 audit for recall.

**Identity / version.** No new identity surface — engram's `memory_system_id` and `memory_version` cover both ingest and recall.

---

## 11. MemoryConfig changes

New fields under `_RECALL_FIELDS` (rename existing `_ANSWER_FIELDS`; the answerer is now external):

```python
intent_seed_hash: str               # content hash of intents/seeds.json
intent_discrimination_margin: float = 0.05
recall_seeding_weights_hash: str    # content hash of recall/weights.json (granularity weights)
recall_edge_weights_hash: str       # content hash of recall/weights.json (edge weights)
recall_max_depth: int = 3
recall_max_frontier: int = 256
recall_max_passages: int = 16
recall_seed_count_total: int = 64
```

Removed (move to benchmark): `answerer_model`, `answerer_temperature`. The benchmark's agent owns these.

`context_char_budget` → either dropped or repurposed as `recall_text_char_budget` (cap on combined `RecallPassage.text` length). My pick: **drop** in v1; add later if agents complain.

`recall_fingerprint` is computed analogously to the existing `answer_fingerprint` but only over `_RECALL_FIELDS` + `_INGESTION_FIELDS`.

---

## 12. Dependencies on ingestion patches

This recall design assumes the ingestion patches in [`ingestion.md`](ingestion.md) §12 have landed:

- **Patch 1 — protocol surface.** Required: the `recall` verb exists on `MemorySystem` and `Memory` is the ingest input.
- **Patch 4 — granule embeddings + parallel vector index.** Required: semantic seeding has nothing to seed against without per-granule embeddings.
- **Patch 5 — layer labels on nodes.** Required: per-granularity seeding queries the `granularity` label.
- **Patch 6 — TimeAnchor + temporal edges.** Required for temporal intent's expansion and for RecallFact timestamp resolution.
- **Patch 7 — derived-rebuild orchestrator.** Required for facts (current-truth, reinforcement counts, change events).

Patch 3 (n-gram extraction) is not strictly required for v1 recall but greatly helps preference and entity-resolution intents — n-gram-level seeds are the most precise.

---

## 13. Tests

| Test | What it proves | Tier |
|---|---|---|
| **Recall determinism (R2 audit).** Same `(config, ingest log, query, context)` twice → byte-identical `RecallResult`. | R2 for recall path. **Write first.** | required |
| Intent classification — synthetic queries → expected intent labels (with `intent_hint=None`). | R6 centroid discrimination works. | required |
| Intent fails-closed — below-margin queries return fallback intent and report low confidence. | R6 + R13 fail-closed. | required |
| Seeding — semantic + entity-anchored produce expected granule hits on a fixture graph. | Seeding correctness. | required |
| Expansion — typed-edge BFS terminates at `max_depth` and respects `max_frontier`. | Walk bounds. | required |
| Scoring — passage selection deduplicates by source granule and sorts by score. | Scoring correctness. | required |
| Assembly — `RecallResult` carries the right facts per intent. | Pre-computed facts surfacing. | required |
| Locality preservation — sentence-level passages include neighboring sentences when configured. | R11. | required |
| Recall makes zero LLM calls — mock any LLM client; assert never called during recall. | R5, R13. | required |
| Fingerprint coverage — every new `MemoryConfig` field categorized. | R3. | required (auto via existing test) |
| Recall fingerprint changes when config / weights / query / context change. | R4. | required |
| Vector-index roundtrip — save-load doesn't change neighbor lookups. | Semantic-layer R12. | required |
| Per-intent end-to-end — for each intent, a fixture query returns expected passage shape and fact mix. | Pipeline wiring. | required |
| Replicate stability — N runs with the same config / ingest / query / context produce identical timing distributions within budget. | K6. | nice-to-have, slow-marked |

---

## 14. Open questions

- **Should `RecallResult.facts` be deduplicated against `RecallResult.passages`?** If a passage already says "Alice likes pizza" and we also surface a `current_preference` fact, the agent sees both. My initial pick: yes, surface both — facts are pre-computed and the agent benefits from the literal value. Revisit if agent prompts get bloated.
- **N-gram-level passages — text or context?** A passage with `granularity="ngram"` has very short `text` (e.g., "spicy Thai food"). Useful as a fact anchor but too short as standalone context. Default: include the parent Sentence's text in the passage with the n-gram highlighted in `supporting_edges`. Revisit after measurement.
- **Vector-index cardinality vs. recall latency.** Brute-force cosine over all granules is fine at LME-s scale (~10k–100k granules). At LOCOMO scale or a real-world deployment, this could become the recall bottleneck. Plan: monitor `recall` p50; switch to FAISS or HNSW when it crosses 100ms (per scaling triggers in `ingestion.md`).
- **Tool-call schema for recall.** What JSON shape does the LLM agent see when it calls the `recall` tool? Default: a Pydantic-style schema generated from `RecallResult`'s dataclass shape, exposed via `engram.tool_schema()` (a helper for benchmark consumption). Lives at the engram-package boundary; not in this design doc but flagged for the implementation PR.

---

## 15. Provisional values awaiting calibration

These ship as defaults; explicitly provisional, re-baselined after the first benchmark run.

- `intent_discrimination_margin = 0.05`
- `recall_max_depth = 3`
- `recall_max_frontier = 256`
- `recall_max_passages = 16`
- `recall_seed_count_total = 64`
- Per-intent edge weights — initial values picked by hand per the rationale in §6, encoded in `engram/recall/weights.json`.
- Per-intent granularity seeding weights — initial values picked by hand per §5, encoded in same weights file.

The Diagnostics-owned `needle_recall@k` optimizer (a later PR) replaces the hand-picked weights with measured ones.
