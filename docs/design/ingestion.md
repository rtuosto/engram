# Ingestion — Design

> **Scope.** The ingestion side of engram under the post-pivot architecture. Engram is a memory tool for an outside agent; `ingest(memory)` is one of the two core verbs. This doc covers: the `Memory` shape, graph storage, the 5-layer × 4-granularity model, append-only primary data, derived-index rebuilds, the deterministic extraction pipeline, model choices, validation protocols, dependencies, tests, and the patch path from the existing Tier-1 implementation.
> **Status.** Draft for review; revises the prior version of this doc that predated the engram-as-tool pivot. The binding rules are in [`../DESIGN-MANIFESTO.md`](../DESIGN-MANIFESTO.md); this doc is implementation-level and evolves as the module is built.

---

## 1. The `Memory` shape

`ingest` accepts a single `Memory` per call. Each call is a permanent observation event — never deduplicated.

```python
@dataclass(frozen=True, slots=True)
class Memory:
    content: str                             # the text payload (required)
    timestamp: str | None = None             # ISO-8601, agent-supplied
    speaker: str | None = None               # "user", "assistant", or any freeform label
    source: str | None = None                # where this came from: "conversation_turn", "file:notes.md", ...
    metadata: dict[str, str] = field(default_factory=dict)  # freeform agent hints
    # (later: attachments / multimodal fields when needed)
```

Engram is agnostic to what the agent decides constitutes a memory. The benchmark may ingest one Memory per conversation turn; a real-world deployment may ingest a long document as a single Memory and let segmentation handle it. Multi-modal fields are deferred — text-only ingest in the first cut.

The tool-call schema engram exposes to the agent mirrors this dataclass's fields.

---

## 2. Graph storage

### Constraints

- **R2 — Determinism.** Same `(config, ingested log)` → byte-identical serialized state. No wall-clock values, no PID-seeded randomness, no unsorted set iteration in output.
- **R12 — Versioned persistence.** Every on-disk artifact carries an explicit schema version; load paths handle missing / old / future versions with a clear error.
- **R16 — Memories append-only; primitives content-addressed.** Memory nodes are timestamped events, never deduplicated. Entity / N-gram / Claim / Preference content is content-addressed (same content → same node ID); observations are edges.
- **R17 — Derived indexes are rebuildable.** Co-occurrence counts, alias sets, current-truth indexes, episodic clusters live in a derived layer that is recomputed from primary, not mutated in place.
- **P10 — Graph encodes knowledge; embeddings rank candidates.** The graph is a data structure we traverse; vector search runs over a parallel index.
- **Scale.** LongMemEval-s: hundreds of conversations, tens of thousands of granule nodes per conversation. LOCOMO comparable. Not RAM-bound at this scale.

### Decision

**One in-memory `networkx.MultiDiGraph` per engram instance, plus a parallel embedding vector index, persisted through schema-versioned msgpack.**

- One engram instance holds one memory. No conversation-id partitioning at the storage layer.
- `MultiDiGraph` (not `DiGraph`) — parallel edges of different types between the same pair of nodes are load-bearing. A Sentence both `mentions` an Entity and `asserts` a Claim about it via different edges.
- Embeddings live as a parallel `numpy` matrix keyed by node ID (granule-only — Session, Turn, Sentence, N-gram). Nearest-neighbor search is a brute-force cosine-sim pass at this scale; swap to `faiss` if the scale triggers below fire.
- Custom serializer (msgpack with a `schema_version` envelope, never `pickle` — version-brittle, opaque). msgpack preserves binary embedding bytes without base64.

### Scaling triggers (when we revisit storage)

We swap the backing store with data, a benchmark, and a new design-doc section if any of:

| Signal | Threshold | Why |
|---|---|---|
| Working-set memory per instance | > 2 GB | Python object overhead becomes painful |
| Granule count in one instance | > 5 M | Brute-force vector search starts dominating recall latency |
| `ingest` p50 (per Memory) | > 200 ms | Tool latency hurts agent throughput |
| `recall` p50 (excl. agent's LLM) | > 100 ms | Recall budget eaten by storage, not by traversal |
| msgpack snapshot size | > 1 GB | Save/load becomes a noticeable run share |

### R2 serialization discipline (non-negotiable)

1. **Sort before iterate.** Any iteration whose output order is observable walks `sorted(items, key=str)`.
2. **No `datetime.now()`.** Timestamps in the graph come from the ingested Memory.
3. **No PID-seeded randomness.** All RNGs explicitly seeded from config (R14).
4. **Floats: stable reduction order.** Aggregations sort contributors by ID before summing.
5. **Set → `frozenset`, sorted tuple on serialize.** Python set iteration is hash-seed-dependent.

A CI test (`tests/test_ingest_determinism.py`) ingests the same synthetic log twice in one process and asserts byte-equality. This is an R2 audit, not a nice-to-have.

---

## 3. Layers and granularities

### Five layers

Layers are content-classification labels. A node may carry multiple layer labels (a Claim is both `relationship` and, when time-anchored, indirectly `temporal` via `temporal_at`).

- **Episodic** — clustered memories about an entity + topic over a time span. Lives in the derived layer (computed from primary observations grouped by entity + topic + time-window). Example: an EpisodicNode "Alice + project + 2026-03" linking the granules where Alice and the project were discussed in March.
- **Entity** — canonicalized nouns. One Entity node per (canonical_form, entity_type), shared across all Memories that reference them.
- **Relationship** — typed connections: preferences (likes/dislikes/wants/avoids/commits_to/rejects), assertions (Claims), co-occurrence. Relationships carry `asserted_at` attributes and outbound `temporal_at` edges to TimeAnchors.
- **Temporal** — TimeAnchor nodes (one per distinct timestamp ingested) plus `temporal_at` / `temporal_before` / `temporal_after` edges. Lets us group "everything observed in March 2026" or "all relationships established before this Memory."
- **Semantic** — the embedding vector attached to each granule node, indexed in a parallel vector store. Nearest-neighbor search is the recall-side entry point for "find granules with similar meaning to this query."

### Four granularities

Granularity is a label on each granule node. Granules carry semantic embeddings.

- **Session** — a derived grouping of related Memories from the same source within a time window. Not an input; computed during derived rebuild.
- **Turn** — one ingested Memory's primary content. Memory boundary IS Turn boundary.
- **Sentence** — a single sentence inside a Turn, segmented via spaCy.
- **N-gram** — a key phrase inside a Sentence. spaCy noun chunks (`doc.noun_chunks`) and dependency subtrees (subject + verb + object) — both are NLP-fast and parseable, no LLM.

### Layers as labels, semantic as parallel index

Each node carries:
- `granularity: str` (one of `session | turn | sentence | ngram`) for granule nodes; absent for non-granules (Entity, Claim, Preference, EpisodicNode, TimeAnchor).
- `layers: frozenset[str]` covering `episodic | entity | relationship | temporal | semantic`.

The semantic layer is implemented as a parallel `numpy.ndarray` of shape `(num_granules, embedding_dim)` plus a `node_ids: list[str]` array. Adding a granule appends a row; rebuilding doesn't move existing rows (R2).

---

## 4. Append-only primary, rebuildable derived

### Primary data — created on `ingest`, never mutated

**Memory nodes** — one per ingest call; carry `content`, `timestamp`, `speaker`, `source`, `metadata`, and a `memory_index` (monotonic per-instance). Never deduplicated.

**Granule nodes** — Turn (one per Memory), Sentence (per spaCy `Doc.sents`), N-gram (per noun chunk + dep subtree). Each granule has its own content + char span + parent reference + embedding.

**Entity nodes** — content-addressed by `(canonical_form, entity_type)`. Created the first time the canonical form is seen; never updated thereafter. Aliases / observation counts live in derived indexes.

**Claim nodes** — content-addressed by `(subject_id, predicate, object_id_or_literal)`. Created the first time this assertion appears; never updated. Each observation is an `asserts` edge from the source granule.

**Preference nodes** — content-addressed by `(holder_id, polarity, target_id_or_literal)`. Same lifecycle as Claims. Each observation is a `holds_preference` edge from the speaker entity (with the source granule referenced via the edge attrs).

**TimeAnchor nodes** — content-addressed by the ISO-8601 timestamp (rounded to a configurable resolution, default 1 second). One per distinct ingested timestamp.

**Primary edges** (set on `ingest`, never mutated):
- `part_of` — Sentence → Turn, N-gram → Sentence, Turn → Memory.
- `mentions` — Granule → Entity. Multiple Granules may mention the same Entity.
- `asserts` — Granule → Claim. Multiple Granules may assert the same Claim (reinforcement).
- `holds_preference` — Entity[holder] → Preference. Multiple observations of the same Preference are multiple edges (reinforcement).
- `about` — Claim → Entity, Preference → Entity.
- `temporal_at` — Granule → TimeAnchor, Relationship → TimeAnchor.

### Derived indexes — rebuilt from primary, never mutated in place

Triggered lazily before recall (or explicitly via an internal `rebuild_derived()` pass).

- **Alias sets per Entity.** For each Entity node, the sorted tuple of distinct surface forms observed across all its `mentions` edges. Stored as a sidecar dict, not on the Entity node payload.
- **Co-occurrence edges** — `co_occurs_with` between Entity pairs, weighted by per-window count normalized to `[0, 1]`. Rebuilt by walking `mentions` edges within configurable time windows.
- **Reinforcement counts per Claim / Preference.** For each Claim or Preference, the count of inbound observation edges (and earliest / latest timestamps). Stored as a sidecar index.
- **Current-truth index for relationships.** For each `(holder, target)` pair, the latest preference observation (by TimeAnchor ordering). Used by recall to answer "what does X currently think about Y."
- **Change-event nodes.** When current-truth flips for a `(holder, target)` (e.g., Alice was `likes` pizza, now `dislikes`), emit a synthetic ChangeEvent node with `temporal_at` edges to both the old and new TimeAnchors. Lets recall answer "when did X change their mind about Y."
- **Episodic clusters.** Group granules by `(entity, topic, time-window)`; emit an EpisodicNode per cluster with `cluster_of` edges to its member granules.
- **`temporal_before` / `temporal_after` between TimeAnchors.** Derived from the sorted set of TimeAnchors.

Derived data carries its own fingerprint: `(ingestion_fingerprint, derivation_config_fingerprint)`. Derived rebuild is idempotent. If primary hasn't changed since the last rebuild, recall reuses the existing derived snapshot.

---

## 5. Node and edge schema (concrete)

### Identity functions

```python
def node_id(identity: dict[str, object]) -> str:
    """sha256(sorted-key JSON of identity).hex[:16]"""
```

| Node type | Identity fields | Layer label(s) | Granularity label |
|---|---|---|---|
| Memory | `(memory_index)` (monotonic per instance) | — | — |
| Turn (granule) | `(memory_id)` — same as the Memory's `memory_index`-derived ID | `semantic` | `turn` |
| Sentence | `(turn_id, sentence_index)` | `semantic` | `sentence` |
| N-gram | `(sentence_id, ngram_kind, normalized_text)` (`ngram_kind` in `noun_chunk | svo`) | `semantic` | `ngram` |
| Entity | `(canonical_form, entity_type)` | `entity` | — |
| Claim | `(subject_id, predicate, object_id_or_literal)` | `relationship` | — |
| Preference | `(holder_id, polarity, target_id_or_literal)` | `relationship` | — |
| TimeAnchor | `(iso_timestamp_rounded)` | `temporal` | — |
| EpisodicNode (derived) | `(entity_id, topic_signature, time_window_start, time_window_end)` | `episodic` | — |
| ChangeEvent (derived) | `(holder_id, target_id, time_anchor_id, old_polarity, new_polarity)` | `relationship`, `temporal` | — |

### Frozen payloads

All payload dataclasses are `@dataclass(frozen=True, slots=True)`. Same convention as the Tier-1 codebase.

```python
@dataclass(frozen=True, slots=True)
class MemoryPayload:
    memory_index: int
    content: str
    timestamp: str | None
    speaker: str | None
    source: str | None
    metadata: tuple[tuple[str, str], ...]   # sorted for R2

@dataclass(frozen=True, slots=True)
class TurnPayload:
    memory_id: str

@dataclass(frozen=True, slots=True)
class SentencePayload:
    text: str
    turn_id: str
    sentence_index: int
    char_span: tuple[int, int]

@dataclass(frozen=True, slots=True)
class NgramPayload:
    normalized_text: str
    surface_form: str
    sentence_id: str
    ngram_kind: str            # "noun_chunk" | "svo"
    char_span: tuple[int, int]

@dataclass(frozen=True, slots=True)
class EntityPayload:
    canonical_form: str
    entity_type: str
    # aliases NOT stored here — derived from `mentions` edges

@dataclass(frozen=True, slots=True)
class ClaimPayload:
    subject_id: str
    predicate: str
    object_id: str | None
    object_literal: str | None

@dataclass(frozen=True, slots=True)
class PreferencePayload:
    holder_id: str
    polarity: str              # likes | dislikes | wants | avoids | commits_to | rejects
    target_id: str | None
    target_literal: str | None

@dataclass(frozen=True, slots=True)
class TimeAnchorPayload:
    iso_timestamp: str         # rounded to configured resolution

@dataclass(frozen=True, slots=True)
class EpisodicNodePayload:
    entity_id: str
    topic_signature: str       # short stable digest of the topic centroid
    time_window_start: str
    time_window_end: str
    member_granule_count: int

@dataclass(frozen=True, slots=True)
class ChangeEventPayload:
    holder_id: str
    target_id: str
    time_anchor_id: str
    old_polarity: str
    new_polarity: str
```

Note that `EntityPayload` no longer carries `aliases`. Aliases are a derived index. Same for `Claim` / `Preference` reinforcement counts — derived, not on the node.

### Edge attrs

```python
@dataclass(frozen=True, slots=True)
class EdgeAttrs:
    type: str                              # primary or derived edge type
    weight: float = 1.0                    # evidence strength in [0, 1]
    source_memory_id: str | None = None    # which Memory this observation came from
    source_granule_id: str | None = None   # the most-specific granule (Sentence or N-gram) that generated this edge
    asserted_at: str | None = None         # ISO-8601 (TimeAnchor's timestamp)
```

`source_memory_id` is the provenance anchor — every primary edge traces to exactly one Memory, supporting the time-travel and attribution properties from P12.

---

## 6. Extraction pipeline (deterministic, no LLM)

Per-Memory, in order. Each stage consumes its predecessor's output and emits nodes / edges.

```
Memory (ingest input)
  │
  ▼
[1] Memory node + Turn node + part_of (Turn → Memory)
  │
  ▼
[2] Sentence segmentation       → Sentence nodes + part_of (Sentence → Turn)
  │
  ▼
[3] N-gram extraction           → N-gram nodes + part_of (N-gram → Sentence)
  │
  ▼
[4] NER                         → entity mentions (text spans)
  │
  ▼
[5] Entity canonicalization     → Entity nodes + mentions edges
  │
  ▼
[6] Claim extraction            → Claim nodes + asserts edges + about edges
  │
  ▼
[7] Preference detection        → Preference nodes (fails closed) + holds_preference + about
  │
  ▼
[8] Granule embedding           → vector index update (Turn, Sentence, N-gram embeddings)
  │
  ▼
[9] Temporal anchoring          → TimeAnchor node + temporal_at edges
                                  (from Turn, Sentence, N-gram, Claim, Preference)

(rebuild_derived runs lazily before recall, or explicitly:)
[D1] Co-occurrence edges        (rewrite)
[D2] Alias sets                 (rewrite)
[D3] Reinforcement counts       (rewrite)
[D4] Current-truth index        (rewrite)
[D5] Change-event nodes         (rewrite)
[D6] Episodic clusters          (rewrite)
[D7] Temporal_before / after between TimeAnchors  (rewrite)
```

### Per-stage notes

**[1] Memory + Turn.** Memory node carries the ingest payload; Turn node is a granule shadow that participates in the granularity hierarchy and gets its own embedding (the whole-Memory representation).

**[2] Segmentation.** spaCy `Doc.sents` with dependency-parse-aware sentence boundaries. Empty / whitespace-only sentences dropped (fails closed).

**[3] N-gram extraction.** Two extractors run side-by-side:
- `noun_chunk` extractor: `doc.noun_chunks`, normalized to lowercase NFKC, dropped if all stop words or below 2 tokens.
- `svo` extractor: walk the dependency parse, emit a phrase per (subject, root verb, object) triple — stable text rendering that's stable under R2.

Each n-gram becomes a node; `part_of` from N-gram to its containing Sentence.

**[4] NER.** spaCy NER over the Doc. Mention spans extracted with surface form + entity type + char span.

**[5] Entity canonicalization.** Tier-1 algorithm: NFKC + casefold + `rapidfuzz.token_set_ratio` against existing entities of the same type. Threshold default 0.85. Same content → same Entity node ID (content-addressed). Tie-breaking: `(higher similarity, alphabetical canonical_form, lexicographic node_id)`. Feature infrastructure wired for embedding similarity + co-occurrence with weights = 0; enabled later under M1 hypotheses.

**[6] Claim extraction.** SVO triples from dependency parses. Subject = `nsubj` subtree (resolved to Entity or first-person → speaker Entity). Predicate = `lemma_` of root verb. Object = `dobj` / `attr` / `pobj` subtree (resolved to Entity or held as `object_literal`). Modality + tense from spaCy morph features. Fails closed if subject can't be resolved.

**[7] Preference detection.** Prototype-centroid classifier per polarity. Fails closed below per-polarity discrimination margin. Synthetic seeds + held-out validation as in the Tier-1 implementation. Detail in §8.

**[8] Granule embedding.** MiniLM (`all-MiniLM-L6-v2`) for all granules — Turn, Sentence, N-gram. Each embedding stored in the parallel vector index keyed by node ID. Embedding is also persisted on the node attrs for recovery if the index is lost.

**[9] Temporal anchoring.** TimeAnchor node for the Memory's timestamp (rounded to configured resolution). `temporal_at` edges from Turn / Sentence / N-gram / Claim / Preference observations to the TimeAnchor.

### Cross-cutting fails-closed policy

Every extractor obeys: **emit nothing below the declared confidence threshold for its output node type.** Below-threshold output that slips through taints multi-Memory aggregation. Thresholds live in `MemoryConfig._INGESTION_FIELDS` and feed the ingestion fingerprint:

- `canonicalization_match_threshold: float`
- `claim_subject_required: bool = True`
- `preference_discrimination_margin: float`
- `ngram_min_tokens: int = 2`

---

## 7. Derived rebuilds

`rebuild_derived()` is idempotent. Triggered lazily before recall if the primary fingerprint has advanced since the last rebuild, or explicitly by callers that want to batch.

### D1 — Co-occurrence

For each pair of Entity nodes, count co-occurrences within configurable time windows (default: per-Memory, plus per-1-hour and per-1-day windows). Emit `co_occurs_with` edges with `weight = count / max_pair_count_in_window`. One edge per (pair, window).

### D2 — Alias sets

For each Entity, walk inbound `mentions` edges. Collect distinct surface forms from the source granules. Store as `aliases: tuple[str, ...]` in a sidecar `derived/alias_index.msgpack`.

### D3 — Reinforcement counts

For each Claim and Preference node, count inbound observation edges and record `(count, earliest_observation, latest_observation)`. Stored in `derived/reinforcement_index.msgpack`.

### D4 — Current-truth index

For each `(holder_id, target)` pair (where target = `target_id` or `target_literal`), find the latest Preference observation by TimeAnchor ordering. Stored in `derived/current_preference_index.msgpack`. Recall consults this for "what does X currently think about Y" queries in O(1).

### D5 — Change-event nodes

When the current-truth index records a polarity flip for `(holder, target)` (e.g., `likes` → `dislikes`), emit a synthetic `ChangeEvent` node and `temporal_at` edges to both old and new TimeAnchors.

### D6 — Episodic clusters

Group granules by `(entity, topic_centroid, time_window)`:
- For each Entity, gather granules that mention it.
- Within those, cluster by topic (k-means on granule embeddings, k chosen by silhouette or elbow on a held-out fixture).
- Within each topic cluster, group by time window (default: 1 day).
- Emit one `EpisodicNode` per (entity, topic, window) with `cluster_of` edges to member granules.

Episodic clustering is the most expensive derived step and may be tier-2 in implementation order.

### D7 — Temporal-before / after between TimeAnchors

Sort all TimeAnchor nodes by their ISO timestamp. Emit `temporal_before` / `temporal_after` edges between consecutive anchors. Lets recall walk forward/backward in time without scanning timestamps.

### Rebuild fingerprint

Derived snapshot fingerprint = `sha256(ingestion_fingerprint || derivation_config_fingerprint)`. Stored alongside the snapshot. Recall checks this on read; if mismatched, rebuild before serving.

---

## 8. Model choices

Three models feed ingestion; each is a `MemoryConfig._INGESTION_FIELDS` entry.

| Role | Model | `MemoryConfig` field | Rationale |
|---|---|---|---|
| NLP pipeline — segmentation, n-gram extraction, NER, dep parse, morph | `en_core_web_sm` | `spacy_model` | CPU-deterministic, ~12 MB, fast. Transformer variant deferred until NER quality is a diagnosed blocker. |
| Granule embedding — semantic-layer indexing, claim / entity proximity | `all-MiniLM-L6-v2` | `embedding_model` | Predecessor default; 384-dim; well-characterized. Topical only — known to flatten speech acts (P5). |
| Preference discrimination | `all-mpnet-base-v2` | `preference_embedding_model` | NLI-aware; discriminates assertion / negation / preference structure. 768-dim. |

Model swaps require an M1 hypothesis (target bucket, expected pp delta) and pass the K6 replicate gate. The fingerprint-discipline test catches forgotten categorizations automatically.

---

## 9. Preference-detection validation protocol

Unchanged in spirit from the Tier-1 doc; restated here for completeness.

### Prototype centroid construction

For each polarity in `{likes, dislikes, wants, avoids, commits_to, rejects}`:
- 10–20 hand-authored synthetic seed sentences per polarity (~60–120 total). Speaker-agnostic, varied in surface form / tense / intensity.
- Centroid = `mean(preference_embed(seed_i))` per polarity. L2-normalized.
- Stored at `engram/ingestion/preferences/seeds.json`. The file's content hash enters `MemoryConfig.preference_seed_hash` (R3 — editing seeds invalidates the ingestion fingerprint).

### Held-out discrimination measurement

- 30–60 hand-authored sentences across the same polarities, **disjoint from the training seeds and never drawn from LME-s / LOCOMO** (P8, M3, M4).
- Stored at `engram/ingestion/preferences/heldout.json`.
- Per-sentence: `margin = cos(sentence, centroid_p) - max_{q != p} cos(sentence, centroid_q)`.
- Per-polarity: median margin must clear `MemoryConfig.preference_discrimination_margin` for that polarity to emit Preferences. Polarities below the gate fail closed (no Preferences of that polarity emitted anywhere).

### Runtime gate

`preference_discrimination_margin: float = 0.05` (provisional; calibrated from held-out before the first benchmark run).

---

## 10. Entity canonicalization

Tier-1 algorithm; extends naturally under the new architecture.

```python
def canonicalize(mention, entity_index):
    normalized = unicodedata.normalize("NFKC", mention.surface_form).casefold().strip()
    same_type = entity_index.by_type[mention.entity_type]

    # 1. Exact normalized match.
    if normalized in same_type.by_normalized_form:
        return same_type.by_normalized_form[normalized]

    # 2. Above-threshold fuzzy match (token_set_ratio).
    best = max(
        same_type.values(),
        key=lambda e: token_set_ratio(normalized, e.normalized_form),
        default=None,
    )
    if best and token_set_ratio(normalized, best.normalized_form) >= threshold:
        return best  # link mention as new edge; no payload mutation

    # 3. New Entity node (content-addressed by (normalized, entity_type)).
    return Entity.new(canonical_form=normalized, entity_type=mention.entity_type)
```

Threshold default: `MemoryConfig.canonicalization_match_threshold = 0.85`. Tie-breaking: `(higher similarity, alphabetical canonical_form, lexicographic node_id)`. Type gate prevents "Apple" the company from colliding with "apple" the food.

Feature infrastructure wired for embedding similarity + co-occurrence as additional features at weight 0; future PR with M1 hypothesis enables them.

**Pronoun coreference** ("my sister" → "Alice") not handled in v1. Diagnosed as `extraction_miss` in R15; addressed in a later PR.

---

## 11. Operational policy

**Async execution model (R1).** Protocol verbs are `async`. Implementation wraps synchronous model calls via `asyncio.to_thread`. No intra-instance concurrency.

**Float-determinism budget (R14, K6).** Provisional ±2pp commit-to-commit. Enforced via:
- `torch.use_deterministic_algorithms(True)` on model init.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` at process start.
- `PYTHONHASHSEED=0` for dict-order determinism.
- All RNGs seeded from `MemoryConfig.random_seed: int = 0`.

**Identity and version.**
- `memory_system_id = "engram_graph"` — stable across patch versions; part of cache isolation.
- `memory_version = "0.1.0"` (semver). `MINOR` bumps on `SCHEMA_VERSION` bumps; `PATCH` on implementation fixes.

**Embedding storage.** Vector matrix lives alongside the graph msgpack (`embeddings.npy` + `node_ids.json` sidecar files). Changing `embedding_model` bumps the ingestion fingerprint, invalidating the snapshot.

**Save-file layout.**

```
<save_path>/
  manifest.json                       # memory_system_id, memory_version, schema_version,
                                      # ingestion_fingerprint, derived_fingerprint
  primary.msgpack                     # nodes + edges (append-only primary)
  embeddings.npy                      # parallel vector index
  node_ids.json                       # row-index → node_id mapping for embeddings.npy
  derived/
    alias_index.msgpack
    reinforcement_index.msgpack
    current_preference_index.msgpack
    co_occurrence.msgpack
    episodic.msgpack
    timeanchor_chain.msgpack
```

`load_state` verifies `schema_version`, `memory_system_id`, and `ingestion_fingerprint`; rebuilds derived if `derived_fingerprint` is missing or mismatched.

---

## 12. Patch path from current Tier-1

The Tier-1 implementation already shipped (`engram/ingestion/{schema,graph,persist,extractors,pipeline,factory}.py` + `engram_memory_system.py`). It uses `Session`/`Turn`/`conversation_id` as inputs and has `answer_question` as a stub. The architecture pivot doesn't require a rewrite — it requires a sequence of patches.

**Status: patches 1–7 are shipped (PR-A / PR-B / PR-C / PR-D). `recall()` still raises `NotImplementedError`; PR-E implements it.**

**Patch 1 — protocol surface (smallest, highest priority).** *(shipped — PR-A)*
- `MemorySystem` protocol: drop `answer_question`, drop `finalize_conversation`, drop `conversation_id` from all verbs. Add `recall(query, *, now, timezone, max_results)`. Rename `ingest_session(session, conversation_id)` → `ingest(memory)`.
- `Memory` dataclass added to `engram/models.py`. `Session` and `Turn` retained as optional helper types (the benchmark may use them to build Memories), but no longer required by the protocol.
- `EngramGraphMemorySystem`: adapt internal state to one-instance-one-memory; `reset` clears everything.
- Fingerprint-discipline test updated.

**Patch 2 — drop primary-data mutations (R16 enforcement).** *(shipped — PR-A)*
- `EntityPayload.aliases` removed. Alias collection moves to derived.
- Claim → Preference label-merging removed. Preferences become separate content-addressed nodes linked by `holds_preference` from the speaker entity.
- Co-occurrence accumulator promoted to derived.

**Patch 3 — add n-gram granularity.** *(shipped — PR-B)*
- New `extractors/ngram.py` running `doc.noun_chunks` + a small dep-subtree SVO extractor.
- New `NgramPayload` dataclass.
- N-gram nodes + `part_of` edges from N-gram → Sentence in the pipeline.

**Patch 4 — add granule embedding storage + parallel vector index.** *(shipped — PR-C)*
- New `engram/ingestion/vector_index.py` wrapping the `numpy` matrix + node-id list.
- Pipeline computes MiniLM embeddings for every granule (Turn, Sentence, N-gram) and inserts into the index.
- `dump_conversation` / `load_conversation` rename → `dump_state` / `load_state`; vector index serializes alongside.
- `SCHEMA_VERSION` bumps to 2.

**Patch 5 — add layer labels.** *(shipped — PR-B)*
- Each node gets a `layers: frozenset[str]` attribute populated by the extractor.
- `GraphStore.nodes_by_layer(label)` helper for recall-side use.

**Patch 6 — TimeAnchor + temporal layer.** *(shipped — PR-D)*
- New TimeAnchor node + `temporal_at` edges from observations.
- `temporal_before` / `temporal_after` between TimeAnchors moves to derived rebuild (lives on the derived `TimeAnchorChainEntry`, not as graph edges in PR-D).

**Patch 7 — derived-rebuild orchestrator.** *(shipped — PR-D)*
- New `engram/ingestion/derived.py` with `rebuild_derived(store, *, config)`.
- Co-occurrence, alias sets, reinforcement counts, current-truth index, and TimeAnchor chain implemented.
- ChangeEvent + EpisodicNode remain deferred to a follow-up PR (not blocking recall v1).
- Lazy trigger: recall (PR-E) will check `state.derived.fingerprint` against `derived_fingerprint(config, store)` and call `rebuild_derived` if stale. `EngramGraphMemorySystem.rebuild_derived()` is exposed now for tests and diagnostics.

Each patch is its own PR with R3 fingerprint coverage, an R2 audit run, and at least one test exercising the new behavior.

---

## 13. Dependencies and tests

### Runtime dependencies (pinned in pyproject.toml — already shipped)

```toml
networkx              = ">=3.2,<4"
msgpack               = ">=1.0,<2"
spacy                 = ">=3.7,<4"
sentence-transformers = ">=2.7,<6"
numpy                 = ">=1.26,<3"
httpx                 = ">=0.27,<1"
rapidfuzz             = ">=3.9,<4"
```

`en_core_web_sm` and sentence-transformer weights are downloaded on first use and not pinned.

### Test strategy

| Test | What it proves | Status |
|---|---|---|
| **Ingest determinism (R2 audit).** Ingest a synthetic log of Memories twice in one process; byte-compare serialized state. | End-to-end determinism. **Write first.** | shipped (Tier-1) — extends to cover Memories |
| Fingerprint coverage. Every `MemoryConfig` field is categorized. | R3 discipline. | shipped (Tier-1) |
| Per-extractor units — segmentation, n-gram, NER (mock), canonicalization, claim, preference. | Each extractor deterministic on fixture input. | shipped for current set; n-gram pending |
| Integration — synthetic log → assert graph shape per layer × granularity. | Pipeline wiring works end-to-end. | shipped; reshape under patch 1 |
| Append-only invariants. Re-ingesting the same Memory content produces a new Memory node and only edges into existing primitives. | R16 enforcement. | new |
| Derived idempotency. Rebuilding derived twice produces byte-identical output. | R17. | new |
| Vector-index roundtrip. Save → load → identical embeddings + identical neighbor lookups. | Vector-index R12. | new |
| Preference-detector held-out validation. Median margin ≥ threshold on shipped held-out set. | Fails-closed gate calibration. | shipped (slow-marked) |
| Cross-process determinism. Serialize from two separate Python processes; byte-compare. | Catches process-global nondeterminism. | new (slow-marked) |

---

## 14. Open questions

- **TimeAnchor resolution.** Round to second / minute / hour? Default 1 second; revisit if anchor count explodes on long corpora.
- **Episodic-cluster topic signature.** k-means on granule embeddings is the obvious choice; the topic_signature stored on the EpisodicNode needs to be a stable, R2-deterministic digest. Open: hash of cluster-centroid bytes vs. hash of member-granule IDs.
- **Co-occurrence window definition.** Per-Memory + per-N-day? What N values? Default to {1 hour, 1 day, all-time}; revisit after first benchmark run.
- **Reinforcement counts vs change events at recall.** A query "does Alice like pizza" — return current-truth node directly, or always include reinforcement count + change history? Defer to recall design.

---

## 15. Provisional values awaiting calibration

These defaults ship with the patches; explicitly provisional, re-baselined after the first benchmark replicate run.

- `canonicalization_match_threshold = 0.85`
- `preference_discrimination_margin = 0.05`
- `ngram_min_tokens = 2`
- TimeAnchor resolution = 1 second
- Co-occurrence windows = `(1 hour, 1 day, all-time)`
- Float-determinism budget = ±2pp
