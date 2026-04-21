# Ingestion — Design

> **Scope.** The complete ingestion design for the first implementation PR (Tier-1 edges per §2). Graph storage (§1), node/edge schema (§2), extraction pipeline (§3), model choices (§4), preference-detection validation (§5), entity canonicalization (§6), operational policy (§7), dependencies and tests (§8).
> **Deferred to later design-doc iterations.** Event / Episode node extraction (Tier 2), coreference-augmented canonicalization, Tier 3 edges (`supports` / `contradicts` / `refers_back_to`). Each a separate design-doc iteration with its own M1 hypothesis before implementation.
> **Status.** Draft for review. The binding rules are in [`../DESIGN-MANIFESTO.md`](../DESIGN-MANIFESTO.md); this doc is implementation-level and evolves as the module is built.

---

## 1. Graph Storage

### Constraints

- **R2 — Determinism.** Same `(config, corpus)` → byte-identical serialized state. No wall-clock values, no PID-seeded randomness, no unsorted set iteration in output. This is the dominant constraint.
- **R12 — Versioned persistence.** Every on-disk artifact carries an explicit schema version; load paths handle missing / old / future versions with a clear error, not a silent crash.
- **P10 — Graph encodes knowledge; embeddings rank candidates.** The graph is a data structure we traverse; it is *not* a query engine whose optimizer we trust. Recall-side traversals are typed-edge BFS with small fan-out — we write them by hand against a simple API.
- **Scale.** LongMemEval-s: 100 conversations, up to ~50 sessions each, on the order of tens of thousands of nodes per conversation at the finest granularity. LOCOMO is comparable. We are not RAM-bound. Anything that scales beyond this is premature.
- **Iteration speed.** The node/edge schema will change many times during the design phase. The cost of re-serializing a corpus after a schema bump must be seconds, not hours.

### Options considered

| Option | R2 story | R12 story | Schema iteration | Footprint | Notes |
|---|---|---|---|---|---|
| `networkx` in-memory + custom serializer | Easy — `sorted(G.nodes(data=True))` everywhere | We own it — explicit version marker, small audit surface | Pure-Python dataclasses, rename-at-will | Zero external infra | Introspectable; no query language (we write traversals ourselves — fine per P10) |
| `kuzu` embedded graph DB | Query plans may reorder; need to sort results everywhere | Format versioning is upstream's; drift on upgrades is a risk | Cypher-schema migrations needed on every bump | External native lib | Cypher is nice but we don't need it for typed-edge BFS |
| DuckDB + SQL/PGQ | Attainable with `ORDER BY` everywhere; PGQ planner is young | DuckDB has a versioned on-disk format | SQL migrations | Mature, columnar | PGQ still evolving in DuckDB; overkill for tens of thousands of nodes |
| Custom numpy-adjacency | Excellent — fixed types, sorted buckets, reproducible floats | We own it | Largest — every schema change touches array layouts | Minimal at runtime | High implementation burden for a problem that is not yet performance-bound |

### Decision

**In-memory `networkx.MultiDiGraph` per conversation, persisted through an explicit `engram.ingestion.persist` module that writes schema-versioned msgpack.**

- **Per-conversation graph, not a single global graph.** `MemorySystem.ingest_session(session, conversation_id)` and `answer_question(question, conversation_id)` are conversation-scoped. Keeping one `MultiDiGraph` per `conversation_id` makes the R2 story trivial (no cross-conversation ordering to worry about) and matches how the benchmark iterates.
- **`MultiDiGraph` (not `DiGraph`)** — parallel edges of different types between the same pair of nodes are load-bearing. A Turn can both `mentions` an Entity and `about` it via a different edge path.
- **Custom serializer, not `pickle` / `nx.node_link_data` / `nx.gml`.** Pickle is R12-hostile (version-brittle, opaque). `node_link_data` loses type information. We write a small module that emits `{"schema_version": N, "memory_system_id": ..., "conversations": {...}}` and refuses to load foreign versions.
- **msgpack over JSON.** Both are deterministic with sorted keys. msgpack is smaller and preserves binary embedding vectors without base64. We keep a `--format=json` debug mode for human inspection; the canonical format is msgpack.

### Consequences

- **Pros.**
  - Schema iteration is dataclass refactors — minutes, not hours.
  - R2 is shallow work: one sorted-iteration helper and one canonical serializer.
  - R12 audit surface is small — one module, one schema version enum.
  - No external infra; tests are pure Python.
  - Transparent: every decision is a Python attribute we can print.
- **Cons (with explicit mitigations, not hand-waves).**
  - *Traversals are hand-written code.* Mitigated by writing **one** generic typed-edge BFS parameterized by `(edge_type_weights: dict[str, float], max_depth: int, max_frontier: int)`. Adding an edge type to the manifesto is a dict entry, not a new function. Code size stays constant as the edge taxonomy grows.
  - *In-memory only; Python object overhead dominates at scale.* Not a current bottleneck; current scale (LME-s, LOCOMO) sits well under the thresholds below. Mitigated by a `GraphStore` interface with explicit swap triggers (see "Scaling triggers" below) so the call is made against data, not vibes.
  - *Weight semantics could diverge across edge types* (count-based `co_occurs_with` vs. evidence-strength `supports` vs. similarity-based `refers_back_to`). Mitigated by: **every edge-type weight normalized to `[0, 1]`** with its meaning documented in the edge-type table, and **intent-specific weight profiles live at the recall-planning layer**, not baked into the graph. The graph stores normalized evidence; recall composes it.
  - *Serialization throughput is Python-bound.* Mitigated by per-conversation files and batch-friendly msgpack.
- **What this forecloses.**
  - Cypher / PGQ query ergonomics. If Recall ever grows a query language, we swap storage — the `MemorySystem` protocol hides this. Not expected for 1.0.
- **Escape hatch.** The ingestion module exposes a narrow `GraphStore` interface around the `MultiDiGraph`: add-node, add-edge, neighbors-by-edge-type (returns sorted neighbors for R2), snapshot-for-persist, restore-from-persist, and the generic typed-edge traversal above. Swapping the backing store later is a localized change — nothing above this interface sees networkx.

### Scaling triggers (when we revisit the storage decision)

The decision is networkx-now-with-a-swap-path, not networkx-forever. We revisit — with data, a benchmark, and a new design-doc section — if *any* of the following holds on a representative corpus:

| Signal | Threshold | Why it matters |
|---|---|---|
| Working-set memory per conversation | > 500 MB | Python object overhead starts hurting at ~1M nodes; beyond 500 MB we're paging |
| Peak corpus RAM (all conversations in parallel eval) | > 16 GB | Above comfortable laptop / dev box headroom |
| `ingest_session` p50 | > 2 s | Iteration speed dies if a 50-session corpus takes over a minute to load |
| `answer_question` p50 traversal time (in-graph only, excl. LLM) | > 100 ms | Recall budget is eaten by storage, not by embedding or reranking |
| Single conversation `co_occurs_with` edge count | > 500k | Signals a combinatorial explosion we should probably cap at ingest, not store |
| Single-conversation msgpack size | > 200 MB | Load/save becomes a noticeable fraction of a run |

None of these are current concerns on LME-s or LOCOMO. They exist so that the swap is triggered by measured pain, not by speculation, and so that no one reopens the debate in a one-off PR without data.

### R2 Serialization Discipline (non-negotiable)

Every function that emits a byte string does so under these rules:

1. **Sort before iterate.** Any iteration whose output order is observable (persistence, fingerprint, log) walks `sorted(items, key=str)`.
2. **No `datetime.now()`.** Timestamps in the graph are copied from inputs. Wall-clock values never appear in persisted state.
3. **No PID-seeded randomness.** All RNGs are explicitly seeded from config. See R14.
4. **Floats: stable reduction order.** When we aggregate (e.g. `co_occurs_with` weights), we sort contributors by ID before summing. Budgeted non-determinism is declared in config, not assumed.
5. **Set → `frozenset`, converted to sorted tuple on serialize.** Python's `set` iteration order is hash-seed-dependent; sorted tuples are not.

A CI test (future: `tests/test_ingest_determinism.py`) ingests the same synthetic corpus twice in one process and asserts byte-equality on the serialized output. This is an R2 audit, not a nice-to-have.

---

## 2. Node and Edge Schema

### Node identity

Every node has a content-addressed ID:

```
node_id = sha256(canonical_form).hexdigest()[:16]
```

`canonical_form` is a stable, sorted-key JSON string built from the node's *identity fields* (below). IDs are:

- **Deterministic under R2** — same identity fields → same ID.
- **Insertion-order-invariant** — a Turn's ID is the same whether it arrives first or last.
- **Collision-safe at our scale** — 64-bit truncation gives 2⁶⁴ space for ≪10⁶ nodes per corpus.
- **Globally unique across conversations where appropriate** — a canonicalized Entity "Alice" shared across two conversations has the same ID, enabling (future) cross-conversation aggregation. A Turn is conversation-scoped and includes `conversation_id` in its identity fields.

Identity fields per node type:

| Node type | Identity fields |
|---|---|
| Turn | `(conversation_id, session_index, turn_index)` |
| Utterance Segment | `(turn_id, segment_index)` |
| Entity | `(canonical_form, entity_type)` — shared across conversations after canonicalization |
| Claim | `(subject_id, predicate, object_id, asserted_by_turn_id)` — a claim is a *speaker's assertion at a point in time*, not a world truth |
| Preference | same as Claim (a Preference is a Claim subtype) |
| Event | `(canonical_description_hash, interval_start, interval_end)` — see §2.timestamps |
| Episode | `(conversation_id, episode_cluster_id)` where cluster ID is deterministic from its member Turn IDs |
| Session | `(conversation_id, session_index)` |

### Multi-labeling

A Claim that is also a Preference is **one node**, not two. Multi-labeling is expressed via the `labels` attribute:

```python
node.labels = frozenset({"claim", "preference"})
```

Per-label payloads are merged onto the same node's attribute dict, namespaced by label:

```python
G.add_node(
    node_id,
    labels=frozenset({"claim", "preference"}),
    claim=ClaimPayload(subject_id=..., predicate="likes", object_id=...),
    preference=PreferencePayload(holder_id=..., polarity="likes"),
)
```

This matches the manifesto's "heterogeneous, multi-labeled graph" (§3) and keeps `(labels, payloads)` co-located without proliferating `is_same_as` edges.

### Node payloads (frozen dataclasses)

All payload dataclasses are `@dataclass(frozen=True, slots=True)`. The graph wrapper holds mutability; the payloads do not. This matches `engram.models` conventions (see [engram/models.py](../../engram/models.py)).

```python
@dataclass(frozen=True, slots=True)
class TurnPayload:
    speaker: str
    text: str
    conversation_id: str
    session_index: int
    turn_index: int
    timestamp: str | None  # ISO-8601 if present in source; never wall-clock

@dataclass(frozen=True, slots=True)
class UtteranceSegmentPayload:
    text: str
    turn_id: str
    segment_index: int
    char_span: tuple[int, int]  # offsets into Turn.text

@dataclass(frozen=True, slots=True)
class EntityPayload:
    canonical_form: str
    entity_type: str            # PERSON | ORG | GPE | ARTIFACT | CONCEPT | ...
    aliases: tuple[str, ...]    # deterministic: sorted

@dataclass(frozen=True, slots=True)
class ClaimPayload:
    subject_id: str             # Entity node id
    predicate: str              # normalized verb/relation
    object_id: str | None       # Entity node id; None for intransitive
    object_literal: str | None  # for "I am 42" where object is a value, not an entity
    asserted_by_turn_id: str
    asserted_at: str | None     # ISO-8601, resolved at ingest (R8)
    modality: str               # asserted | negated | hypothetical | interrogative
    tense: str                  # past | present | future | habitual

@dataclass(frozen=True, slots=True)
class PreferencePayload:
    holder_id: str              # Entity node id, usually the speaker
    polarity: str               # likes | dislikes | wants | avoids | commits_to | rejects
    target_id: str | None
    target_literal: str | None
    source_claim_id: str        # the Claim this Preference overlays
    confidence: float           # centroid-discrimination score; gated by R6 threshold

@dataclass(frozen=True, slots=True)
class EventPayload:
    canonical_description: str
    interval_start: str | None  # ISO-8601
    interval_end: str | None    # ISO-8601
    participant_ids: tuple[str, ...]  # Entity node ids, sorted

@dataclass(frozen=True, slots=True)
class EpisodePayload:
    conversation_id: str
    cluster_id: int             # deterministic from member turn ids
    member_turn_ids: tuple[str, ...]  # sorted
    summary: str | None         # optional; set only if the LLM-enhancement layer is enabled

@dataclass(frozen=True, slots=True)
class SessionPayload:
    conversation_id: str
    session_index: int
    timestamp: str | None
```

### Edges

A `MultiDiGraph` with typed edges. Edge identity is the triple `(src, dst, key)` where `key` is the edge type. Attributes:

```python
@dataclass(frozen=True, slots=True)
class EdgeAttrs:
    type: str                          # edge-type identifier from the reference table below
    weight: float = 1.0                # evidence strength, normalized to [0, 1]
    source_turn_id: str | None = None  # which Turn introduced this edge (provenance)
    asserted_at: str | None = None     # ISO-8601 when relevant (supports/contradicts/etc.)
```

`weight` here is the edge's *evidence strength* — how strongly the corpus supports the relation — normalized to `[0, 1]` with per-edge-type semantics documented below. It is **not** the recall-time weight used during subgraph expansion; that lives in a per-intent weight vector at recall-planning (see "Recall-side weight optimization" below).

Parallel edges of different `type` between the same pair are valid (Claim → Entity can be both `mentions` and `about`). Parallel edges of the *same* type are not — add-edge upserts, accumulating evidence into `weight` where appropriate (e.g. `co_occurs_with`).

#### Extraction-cost-driven tiering

Manifesto §3 enumerates an aspirational edge inventory (twelve types). Not all ship on day one. The filter for inclusion is *extraction cost*, not *recall utility* — weight optimization handles recall-side relevance automatically. If extraction is cheap and deterministic, ship the edge; the optimizer can drive its recall weight toward zero for intents that don't benefit. If extraction requires semantic reasoning (entailment, anaphora, paraphrase), R5 defers it.

Rationale: the marginal *recall-side* cost of a speculative edge type collapses under automated weight optimization (one extra scalar in a ~50-dimensional search space). The marginal *ingest-side* cost does not — each edge type demands a deterministic extractor, R2 audit, R3 fingerprint coupling, and serialization schema. So the right filter is "cheap to extract", not "proven to help recall."

**Tier 1 — cheap deterministic extraction; shipped with the first ingestion PR.**

| Edge | From → To | Extraction | Weight semantics |
|---|---|---|---|
| `part_of` | Turn → Session, UtteranceSegment → Turn | structural, direct from inputs | 1.0 (structural) |
| `mentions` | Turn → Entity, Claim → Entity | NER + canonicalization | 1.0 (presence) |
| `asserts` | Turn → Claim | dependency-parse SVO extraction | 1.0 (presence) |
| `holds_preference` | Entity[speaker] → Preference | Preference detector output + Turn.speaker | Preference detector confidence, rescaled to [0, 1] |
| `about` | Claim → Entity, Preference → Entity | Claim.object_id / Preference.target_id | 1.0 (presence) |
| `co_occurs_with` | Entity ↔ Entity (bidirectional) | entity-pair counting per conversation | count normalized by max-pair-count in conversation |
| `temporal_before` / `temporal_after` | Turn → Turn (Turn level only) | from `(session_index, turn_index)` ordering; free | 1.0 (ordinal) |

Seven edge types. All deterministic, all extractable from NER + dependency parses + counts + ordering. No LLM, no semantic reasoning at ingest.

**Tier 2 — unlocked by first-class Event / Episode node detection.**

| Edge | From → To | Unlocked by |
|---|---|---|
| `during` | Claim → Event, Event → Session | Event extraction |
| `part_of` | Turn → Episode | Episode clustering |
| `temporal_before` / `temporal_after` | Event → Event | Event extraction |

Promotion gate: Event and Episode node extraction landing as a separate design-doc iteration, with their own determinism and discrimination validation. These edges come "for free" once the nodes exist.

**Tier 3 — requires semantic reasoning at ingest (R5 pressure).**

| Edge | From → To | Why deferred |
|---|---|---|
| `supports` | Claim ↔ Claim, Claim ↔ Preference | entailment detection — NLI model or LLM |
| `contradicts` | Claim ↔ Claim, Claim ↔ Preference | same |
| `refers_back_to` | Turn → Turn | anaphora / coreference resolution |

Promotion gate: an M1 hypothesis tying a diagnosed bucket gap to the structure these edges would provide, **plus** a deterministic (non-LLM) extractor validated above noise on a held-out set. Until then, conflict detection at recall (manifesto §3.Recall) operates over claim target overlap without explicit edges.

#### Recall-side weight optimization

Per-intent edge weights are parameters to optimize, not hand-tune. Shipping an edge is a commitment to extract it; shipping a *weight* for it is a commitment to a hypothesis.

- **Objective.** `needle_recall@k` on oracle-annotated questions (LME-s `answer_session_ids`, LOCOMO equivalent), per intent.
- **Search space.** Per-intent edge-type weight vector (~5 intents × ~7 Tier-1 edges ≈ 35 scalars), plus walk depth, frontier size, and seed-count-per-intent. ~50 scalars at Tier-1 scope.
- **Strategy.** Black-box optimization (Optuna / CMA-ES) against the oracle. LLM-free — minutes on LME-s.
- **Gate (M4 discipline).** `needle_recall@k` improvements can regress full-benchmark accuracy (see [lessons 2026-04-20](../../.agent/lessons.md)). Optimized weights ship through the same replicate + pp-threshold gate as any other change. Optimize on dev; validate on the full benchmark.
- **Infrastructure owner.** Diagnostics (`needle_overlap`; manifesto §6 Diagnostics verbs). The optimizer is a thin wrapper.

When a Tier 2 or Tier 3 edge type is promoted, its weight enters the search space on the next optimization run. No manual weight-setting beyond an initial sensible guess.

### Timestamps

- ISO-8601 strings throughout, always timezone-aware when source provides one, always explicit about when it does not (field is typed `str | None`).
- No naive `datetime` objects in payloads. No `datetime.now()`.
- Temporal arithmetic (R8) happens at ingest or recall-planning and stores resolved absolute strings on Event / Claim payloads. The answerer never sees relative expressions.
- Dataset-provided timestamps (e.g. LongMemEval's `question_date`, `haystack_dates`) are the source of truth; missing values stay `None`.

### Frozen vs. mutable

- **Payloads are frozen.** No hidden mutation.
- **The `MultiDiGraph` is mutable during ingest.** `finalize_conversation(conversation_id)` sets a `frozen: bool` flag on the conversation's `GraphStore` wrapper; subsequent writes raise `GraphFrozenError`. This enforces the ingest→finalize→read lifecycle that recall depends on.
- **Saved state is immutable.** `save_state` writes a content-hashed filename (`{answer_fingerprint}.msgpack`) alongside a manifest; `load_state` verifies the hash on read.

### Schema versioning (R12)

A single `SCHEMA_VERSION: Final[int] = 1` constant in `engram.ingestion.persist`. The version increments whenever:

- Any payload dataclass gains or loses a field.
- Any edge type is added or removed.
- Identity-field lists change (would rewrite every node ID).

Every persisted file carries `{"schema_version": N, ...}` as its first key. Load paths:

- Exact match → load.
- Older version → raise `SchemaVersionMismatch(have=N, found=M)` with a message pointing to a migration tool. Migrations are separate PRs with their own tests; they never run implicitly.
- Newer version → same error. No silent downgrade.

This is the R12 contract, stated in ≤20 lines of Python.

### Ingestion fingerprint coupling (R3)

Every config field that influences node/edge *production* — model identifiers, segmentation rules, canonicalization thresholds — lives in `MemoryConfig._INGESTION_FIELDS` (see [engram/config.py](../../engram/config.py)). The `SCHEMA_VERSION` is also part of the ingestion fingerprint: a schema bump invalidates every persisted graph, even if config is otherwise unchanged.

The existing fingerprint-discipline test (`tests/test_fingerprint_discipline.py`) already catches forgotten fields. A new ingest-determinism test will catch "field categorized but iteration is unstable" bugs where the fingerprint matches but bytes differ.

---

## 3. Extraction Pipeline

A linear sequence. Each stage consumes its predecessor's output and emits nodes / edges onto the `MultiDiGraph`. Stage order mirrors manifesto §3.

```
Session + Turns (input)
    │
    ▼
[1] Segmentation            → UtteranceSegment nodes + part_of edges
    │
    ▼
[2] NER                     → entity mentions (text spans; not yet nodes)
    │
    ▼
[3] Canonicalization        → Entity nodes (deduplicated) + mentions edges
    │
    ▼
[4] Claim extraction        → Claim nodes + asserts edges + about edges
    │
    ▼
[5] Preference detection    → Preference labels on Claim nodes (fails-closed)
    │                          + holds_preference + about edges
    ▼
[6] Co-occurrence           → co_occurs_with edges (at finalize)

finalize_conversation:
    - co-occurrence edge emission across the full conversation
    - temporal_before / temporal_after at Turn level (from indexes)
    - GraphStore.freeze()
```

### [1] Segmentation

- **Algorithm.** spaCy `Doc.sents` with dependency-parse-aware boundary detection.
- **Why not regex.** R6 — surface-pattern boundaries overfit to dataset phrasings.
- **Input.** `Turn.text`.
- **Output.** `UtteranceSegmentPayload(text, turn_id, segment_index, char_span)`, left-to-right `segment_index`.
- **Determinism.** spaCy CPU tokenizer / parser is bit-stable.
- **Fails closed.** A turn producing zero segments (empty / punctuation-only) emits no UtteranceSegment nodes; the Turn still exists.

### [2] NER

- **Algorithm.** spaCy pipeline NER over the Doc.
- **Batching.** Accumulate all Turns in a session; process via `Language.pipe` at the end of `ingest_session`. Batch sizes 20–50.
- **Output.** Entity mentions — `(surface_form, entity_type, char_span, turn_id)` tuples. Not yet Entity nodes.
- **Determinism.** CPU-deterministic. GPU mode covered by R14 budget (§7).

### [3] Entity canonicalization

Full algorithm in §6. Pipeline summary:

- **Input.** Entity mentions from [2].
- **Output.** Entity nodes (one per canonical form) + `mentions` edges from Turns.
- **Fails closed.** Mentions that don't meet the clustering threshold are dropped; no "pending" state.

### [4] Claim extraction

- **Algorithm.** Over each UtteranceSegment's dependency parse, extract `(subject, predicate, object)` triples:
  - `subject` = subtree under the `nsubj` token, resolved to a canonical Entity if possible.
  - `predicate` = lemma of the `ROOT` verb.
  - `object` = subtree under `dobj` / `attr` / `pobj`, resolved to a canonical Entity or held as `object_literal`.
- **Tense / modality.** From spaCy morph features (`VerbForm`, `Tense`, `Mood`).
- **Speaker attribution.** Inherited from `Turn.speaker`.
- **Output.** `ClaimPayload` + `asserts` edge (Turn → Claim) + `about` edge (Claim → Entity) for each entity in subject / object.
- **Fails closed.** No identifiable subject **or** no predicate → no Claim. Literals (numbers, dates, quoted strings) go in `object_literal` when no Entity resolves.

### [5] Preference detection

Full protocol in §5. Pipeline summary:

- **Input.** Each Claim from [4].
- **Classifier.** Prototype-embedding centroids per polarity, computed from hand-authored synthetic seeds.
- **Output.** `PreferencePayload` co-located on the same node as the parent Claim (multi-label `{claim, preference}`), plus `holds_preference` edge from the speaker Entity and `about` edge to the target.
- **Fails closed.** Below-margin Claims remain pure Claims. SSP accuracy depends entirely on prototype quality — see §5.

### [6] Co-occurrence (corpus signal)

- **Algorithm.** Per-conversation entity-pair counter. Within each Turn, every unordered pair of canonical Entity nodes increments a counter.
- **Edge emission.** During `finalize_conversation`: emit bidirectional `co_occurs_with` edges, `weight = count / max_pair_count_in_conversation` — normalized to `[0, 1]` per §2 edge-weight rule.
- **Pruning.** No hard count floor; the recall-time weight optimizer is responsible for ignoring thin edges.

### Finalize-conversation passes

1. Co-occurrence edge emission (above).
2. `temporal_before` / `temporal_after` at Turn level, derived from `(session_index, turn_index)` — O(n) sorted pass over Turn nodes.
3. `GraphStore.freeze()` — sets `frozen = True`; subsequent writes raise `GraphFrozenError`.

### Fails-closed policy (cross-cutting)

Every extractor obeys one rule: **emit nothing below the declared confidence threshold for its output node type.** No "partial" nodes, no "low-confidence" flags. Thresholds live in `MemoryConfig._INGESTION_FIELDS` and feed the ingestion fingerprint:

- `canonicalization_match_threshold: float`
- `claim_subject_required: bool = True`  (dependency parse must produce a resolved subject)
- `preference_discrimination_margin: float`

Violating fails-closed is a correctness bug, not a quality knob. A below-threshold output that slips through taints multi-session-aggregation (P1, P3, R6).

---

## 4. Model Choices

Three models feed ingestion. Each is a `MemoryConfig._INGESTION_FIELDS` entry — changing any bumps the ingestion fingerprint (R3) and every downstream answer (R4).

| Role | Model | `MemoryConfig` field | Rationale |
|---|---|---|---|
| NLP pipeline — segmentation, NER, dep parse, morph | `en_core_web_sm` | `spacy_model` | CPU-deterministic, ~12 MB, fast. Transformer variant (`_trf`, 400+ MB) deferred until NER quality is a diagnosed blocker — avoid speculative capability (P1) |
| Topical embedding — seeding, co-occurrence weighting, canonicalization feature #2 | `all-MiniLM-L6-v2` | `embedding_model` | Predecessor default; 384-dim; well-characterized. Known limitation: collapses speech acts onto topic proximity (P5) — hence the separate preference-discrimination model below |
| Preference discrimination | `all-mpnet-base-v2` | `preference_embedding_model` *(new)* | NLI-aware; discriminates assertion / negation / preference structure that MiniLM flattens. 768-dim. Ships day one per P5 + R10 |

**New `MemoryConfig` field** (to add alongside the existing ones; the partition invariant catches forgetting this):

```python
preference_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
# _INGESTION_FIELDS |= {"preference_embedding_model"}
```

**Upgrade criteria.** A model swap requires an M1 hypothesis (target bucket, expected pp delta) and passes the K6 replicate gate. The fingerprint-discipline test catches forgotten categorizations automatically.

**Not added in Tier 1.** Cross-encoder reranker (lives in Recall). Coreference resolver (upgrades §6; non-trivial without an LLM).

---

## 5. Preference-Detection Validation Protocol

SSP is the lower of the two red buckets (16.7% on the predecessor). Preference detection is the single Tier-1 mechanism pointed at it. This section specifies how to get it right without contaminating the benchmark (M3, M4, P8).

### Prototype centroid construction

For each polarity in `{likes, dislikes, wants, avoids, commits_to, rejects}`:

- **Seeds.** 10–20 hand-authored synthetic sentences per polarity (≈ 60–120 total). Sentences are speaker-agnostic ("I love spicy food" / "I avoid early meetings") and vary in surface form, tense, and intensity.
- **Centroid.** `mean(preference_embed(seed_i))` — mean-pool of seed embeddings per polarity.
- **Storage.** Precomputed, source-controlled at `engram/ingestion/preferences/centroids.json` (seeds + computed centroids). The seed-file content hash enters `MemoryConfig` as `preference_seed_hash: str` — in `_INGESTION_FIELDS`, so editing seeds invalidates the ingestion fingerprint (R3).

### Held-out discrimination measurement

- **Held-out set.** 30–60 hand-authored sentences across the same six polarities, constructed *separately* from the training seeds. Lives at `engram/ingestion/preferences/heldout.json`. **Never drawn from LongMemEval or LOCOMO questions.**
- **Metric.** For each held-out sentence with true polarity p:

  ```
  margin(sentence) = cos(sentence, centroid_p) - max(cos(sentence, centroid_q) for q ≠ p)
  ```

- **Per-polarity gate.** Median `margin` must clear the discrimination threshold for that polarity to emit Preference nodes. Polarities that fail the gate fail closed for the whole corpus — no Preference-of-polarity-p nodes emitted anywhere.

### Runtime threshold

`MemoryConfig.preference_discrimination_margin: float = 0.05` — provisional; calibrated from held-out data before the first benchmark run.

At ingest:

```python
for claim in claims:
    e = preference_embed(claim.text)
    scores = {p: cos(e, centroids[p]) for p in POLARITIES}
    top_p = max(scores, key=scores.get)
    second = max(scores[q] for q in POLARITIES if q != top_p)
    margin = scores[top_p] - second
    if margin >= preference_discrimination_margin and scores[top_p] >= min_centroid_score:
        emit_preference(claim, polarity=top_p, confidence=scores[top_p])
    # else: fail closed — the Claim stands on its own
```

### Iteration discipline (M3 / M4 / P8)

- Held-out set is **never** populated from LongMemEval or LOCOMO questions. Contamination risk is zero.
- When the detector misses on real SSP cases (diagnosed via `extraction_miss` in R15), iterate on **seeds**, not on thresholds, and **only using the detector's own outputs** on the held-out set — never LME-s gold labels.
- Each iteration is an M1 hypothesis: "adding seeds X, Y to polarity p will raise held-out median margin by ≥ ε without regressing other polarities." The hypothesis carries to the benchmark PR as its pp-delta target.

### Known Tier-1 limitation

If the synthetic distribution doesn't match real LME-s SSP phrasings, the detector fails closed and SSP stays near 16.7% baseline. That is correct behavior — R6 forbids emitting low-confidence structure. The fix is better seeds, not a looser threshold.

---

## 6. Entity Canonicalization

Multi-session-aggregation depends on entity identity across sessions. "Alice" in session 1 and "Alice" in session 5 must resolve to the same Entity node. Tier 1 ships a minimum-viable canonicalizer and the architectural infrastructure for feature expansion.

### Minimum-viable algorithm (first PR)

String similarity only, with Unicode-normalized lowercasing.

```python
def canonicalize(mention, existing_entities_by_type):
    normalized = unicodedata.normalize("NFKC", mention.surface_form).casefold().strip()
    same_type = existing_entities_by_type.get(mention.entity_type, {})

    # 1. Exact normalized match
    if normalized in same_type.by_normalized_form:
        return same_type.by_normalized_form[normalized]

    # 2. Above-threshold fuzzy match (token_set_ratio)
    best = max(
        same_type.values(),
        key=lambda e: token_set_ratio(normalized, e.normalized_form),
        default=None,
    )
    if best and token_set_ratio(normalized, best.normalized_form) >= canonicalization_match_threshold:
        return best  # merge; add mention.surface_form as alias

    # 3. New Entity node (fails-closed if type is missing)
    return Entity.new(canonical_form=normalized, entity_type=mention.entity_type, ...)
```

- **Similarity.** `rapidfuzz.fuzz.token_set_ratio` — token-set Jaccard; handles word-order shuffles and partial matches.
- **Threshold.** `MemoryConfig.canonicalization_match_threshold: float = 0.85` — provisional; calibrated on a synthetic fixture.
- **Entity-type gate.** Canonicalization only merges within identical `entity_type` (PERSON ↔ PERSON; ORG ↔ ORG). Prevents "Apple" the company colliding with "apple" the food.
- **Determinism.** Tie-breaking is a strict total order: `(higher similarity, alphabetical canonical_form, lexicographic node_id_hash)`. R2-compliant under any insertion order.

### Feature infrastructure — wired, disabled

The canonicalization function takes a weighted feature vector even in Tier 1:

```python
SIMILARITY_WEIGHTS = {
    "string":        1.0,   # Tier 1: the only feature used
    "embedding":     0.0,   # wired; enabled in a future PR with M1 hypothesis
    "co_occurrence": 0.0,   # wired; enabled in a future PR with M1 hypothesis
}
```

Ship the infrastructure, keep non-string weights at zero. A future PR proposes non-zero weights with measured bucket-impact evidence. Commits the architectural shape without committing the extraction complexity.

### Known Tier-1 limitations (explicit)

- **Pronoun coreference** ("my sister" → "Alice") not handled. Requires embedding-sim + dialogue-context resolution. Diagnosed as `extraction_miss` in R15; addressed in a later PR with its own hypothesis.
- **Nicknames** ("Bob" vs. "Robert") not handled unless string-similar enough to cross the threshold. Same diagnostic path.
- **Cross-conversation entity sharing.** Tier-1 canonicalization is per-conversation. Two conversations that each mention "Alice" produce two distinct Entity nodes. Cross-conversation sharing is out of LME-s scope and deferred.

---

## 7. Operational Policy

Cross-cutting policy pinned by rule.

**Async execution model (R1).** Protocol verbs are `async` (contract). Implementation is synchronous model calls wrapped via `asyncio.to_thread`. Parallelism is at the `MemorySystem`-instance level — the external benchmark runs conversations in parallel with separate instances. No intra-instance concurrency.

**Float-determinism budget (R14, K6).** Commit-to-commit FP drift budget: **±2pp** on the full LME-s benchmark. Provisional — validated after the first three commits produce replicate statistics; loosened to predecessor ±4pp only if empirics demand. Enforce:

- `torch.use_deterministic_algorithms(True)` on model init.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` at process start.
- `PYTHONHASHSEED=0` for dict-order determinism.
- All RNGs seeded from `MemoryConfig.random_seed: int = 0` (add to `_INGESTION_FIELDS`).
- Accept residual CUDNN drift within budget.

**Identity and version.**
- `memory_system_id = "engram_graph"` — stable across patch versions; part of cache isolation.
- `version = "0.1.0"` (semver). `MINOR` bumps on `SCHEMA_VERSION` bumps (§2); `PATCH` on implementation fixes.

**Embedding storage.** Embedding vectors live as node attributes inside the graph msgpack. A sidecar file solves a non-existent problem — changing `embedding_model` bumps the ingestion fingerprint (R3/R4), which invalidates the graph anyway.

**Batching boundary.** Per-`ingest_session`. Sessions are naturally ~5–50 turns — GPU-friendly. Batching at `finalize_conversation` would make K4 `ingest_session` latency meaningless.

**Save-file layout.** One msgpack per conversation at `save/<memory_system_id>/<ingestion_fingerprint>/<conversation_id>.msgpack`, plus a top-level manifest listing conversation IDs and their content hashes. Per-conversation files keep R2 audits local and scale with incremental work.

---

## 8. Dependencies & Test Strategy

### Runtime dependencies

```toml
# pyproject.toml [project.dependencies]
networkx              = ">=3.2,<4"
msgpack               = ">=1.0,<2"
spacy                 = ">=3.7,<4"
sentence-transformers = ">=2.7,<3"
numpy                 = ">=1.26,<2"
httpx                 = ">=0.27,<1"
rapidfuzz             = ">=3.9,<4"
```

**Not pinned in pyproject.** spaCy language model `en_core_web_sm` and sentence-transformer weights — downloaded on first use (`python -m spacy download en_core_web_sm`; sentence-transformers auto-caches to `~/.cache/huggingface/`). Setup documented in `README.md`.

### Test strategy — the R2 audit is the keystone

| Test | What it proves | When |
|---|---|---|
| **Ingest determinism (R2 audit).** Ingest a synthetic 2-session corpus twice in one process, byte-compare serialized state | End-to-end determinism discipline | Every CI run. Non-negotiable. **Write this first.** |
| Fingerprint coverage (existing `test_fingerprint_discipline.py`) | Every `MemoryConfig` field is categorized | Every CI run |
| Per-extractor unit tests — segmentation, NER (mock), canonicalization, claim, preference | Each extractor deterministic on fixture input | Every CI run |
| Integration — 2-session synthetic corpus → assert graph shape (node counts by label, edge counts by type, identity-invariant ordering) | Pipeline wiring works end-to-end | Every CI run |
| Preference-detector held-out validation — median margin ≥ threshold on the shipped held-out set | Fails-closed gate calibration is correct | Every CI run, `slow`-marked |
| Cross-process determinism — serialize from two separate Python processes with same seed; byte-compare | Catches process-global nondeterminism beyond R14 budget | Nightly / release; `slow`-marked |

Predecessor lessons 2026-04-20 document a full day lost to FP non-determinism that passed unit tests but failed session-to-session. The R2-audit test exists so we don't repeat that.

**Out of this module's test scope.** Full LongMemEval-s runs (external benchmark). Cross-encoder reranker (Recall). Answerer prompt (Recall).

---

## Open questions surfaced by this iteration

- **Do we need a `labels` index on the graph for fast "all Preferences in this conversation" lookups?** Probably yes — it's how recall seeds typed subgraph walks. Could be a separate `dict[str, set[str]]` on the `GraphStore` wrapper, populated on add-node. Defer until recall-side traversal patterns firm up.
- **Should `EdgeAttrs.source_turn_id` be multi-valued?** A `co_occurs_with` edge accumulates evidence from many turns. Options: single turn ID (loses evidence) vs. tuple of turn IDs (grows with weight) vs. drop the field on aggregate edges. Defer.

## Provisional values awaiting calibration

These defaults ship with the first PR but are explicitly provisional — re-baselined after the first benchmark replicate run.

- `canonicalization_match_threshold = 0.85`
- `preference_discrimination_margin = 0.05`
- Float-determinism budget = ±2pp
- Preference seed file content (≈60–120 sentences across 6 polarities) — drafted as part of first-PR work.
- Preference held-out file content (≈30–60 sentences) — drafted as part of first-PR work.
