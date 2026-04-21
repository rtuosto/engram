"""``EngramGraphMemorySystem`` — the concrete :class:`MemorySystem` implementation.

Implements ingest per ``docs/design/ingestion.md``. Recall
(:meth:`recall`) is a separate design-doc iteration and raises
:class:`NotImplementedError` here; external callers that only need
ingest + save-state semantics work today.

**One engram instance holds one memory (``R1``).** No conversation-id
partitioning. Isolation is the caller's responsibility (instantiate
separately, or call :meth:`reset`).

**Persistence layout** (R12, manifesto §K7):

```
<save_path>/
    manifest.json     # memory_system_id, memory_version, schema_version,
                      #   ingestion_fingerprint, has_primary, has_embeddings
    primary.msgpack   # the GraphStore's nodes + edges
    embeddings.npy    # parallel granule vector index (PR-C)
    node_ids.json     # row-index → node_id mapping for embeddings.npy
```

``load_state`` verifies ``schema_version`` matches
:data:`persist.SCHEMA_VERSION`, ``memory_system_id`` matches this instance's
id, and every declared sidecar decodes cleanly. No implicit migration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from engram.config import MemoryConfig
from engram.ingestion.derived import (
    DerivedFormatError,
    DerivedIndex,
    derived_fingerprint,
    dump_derived,
    load_derived,
    rebuild_derived,
)
from engram.ingestion.persist import (
    MEMORY_SYSTEM_ID,
    SCHEMA_VERSION,
    PersistFormatError,
    SchemaVersionMismatch,
    dump_state,
    load_state,
)
from engram.ingestion.pipeline import IngestionPipeline, InstanceState
from engram.ingestion.vector_index import VectorIndex
from engram.models import Memory, RecallResult
from engram.recall.context import RecallContext
from engram.recall.pipeline import RecallPipeline

MANIFEST_FILENAME: Final[str] = "manifest.json"
PRIMARY_FILENAME: Final[str] = "primary.msgpack"
EMBEDDINGS_FILENAME: Final[str] = "embeddings.npy"
NODE_IDS_FILENAME: Final[str] = "node_ids.json"
DERIVED_DIRNAME: Final[str] = "derived"
DERIVED_SNAPSHOT_FILENAME: Final[str] = "snapshot.msgpack"


class EngramGraphMemorySystem:
    """Graph-backed MemorySystem implementation.

    Construct with an optional pre-built :class:`IngestionPipeline` (tests
    inject a mocked pipeline; production lazily loads spaCy + sentence-
    transformers on first ingest).
    """

    memory_system_id: str = MEMORY_SYSTEM_ID
    memory_version: str = "0.4.0"

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        pipeline: IngestionPipeline | None = None,
        recall_pipeline: RecallPipeline | None = None,
    ) -> None:
        self._config: MemoryConfig = config or MemoryConfig()
        self._pipeline: IngestionPipeline | None = pipeline
        self._recall_pipeline: RecallPipeline | None = recall_pipeline
        self._state: InstanceState | None = None

    # ------------------------------------------------------------------
    # Public MemorySystem surface (R1).
    # ------------------------------------------------------------------

    async def ingest(self, memory: Memory) -> None:
        pipeline = self._get_pipeline()
        if self._state is None:
            self._state = pipeline.create_state()
        pipeline.ingest(self._state, memory)

    async def ingest_many(self, memories: Iterable[Memory]) -> None:
        """Batched variant of :meth:`ingest`.

        Collapses the three model calls (spaCy ``nlp.pipe``, granule
        embedding, preference embedding) across the batch dimension while
        keeping per-Memory graph writes in order. Structural output is
        identical to looping :meth:`ingest`; edge-weight / vector-row
        numerics may drift at ~5e-8 due to batch-composition changes in
        transformer inference (guard:
        ``scripts/check_fingerprint_equivalence.py``, R3/R4).
        """
        memory_seq: tuple[Memory, ...] = tuple(memories)
        if not memory_seq:
            return
        pipeline = self._get_pipeline()
        if self._state is None:
            self._state = pipeline.create_state()
        pipeline.ingest_many(self._state, memory_seq)

    async def recall(
        self,
        query: str,
        *,
        now: str | None = None,
        timezone: str | None = None,
        max_passages: int | None = None,
        intent_hint: str | None = None,
    ) -> RecallResult:
        if self._state is None:
            # No ingests yet — return an empty shell with the recall fingerprint
            # so the benchmark's cache key remains well-defined.
            from engram.models import RecallResult as _RR

            return _RR(passages=(), intent=None, intent_confidence=0.0)
        pipeline = self._get_recall_pipeline()
        context = RecallContext(
            now=now,
            timezone=timezone,
            max_passages=max_passages,
            intent_hint=intent_hint,
        )
        return pipeline.recall(self._state, query, context=context)

    async def reset(self) -> None:
        self._state = None

    def rebuild_derived(self) -> DerivedIndex | None:
        """Rebuild the derived-index snapshot and cache it on ``InstanceState``.

        Idempotent: re-running on unchanged primary returns a snapshot with
        the same fingerprint as the previous rebuild (``R17``).

        Returns ``None`` when no state exists yet (no ingests). Recall (PR-E)
        will call this lazily on its entry path; for now it's exposed as a
        sync method so tests and diagnostic callers can exercise the path.
        """
        if self._state is None:
            return None
        snapshot = rebuild_derived(self._state.store, config=self._config)
        self._state.derived = snapshot
        return snapshot

    async def save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        ingestion_fingerprint = self._config.ingestion_fingerprint()

        has_primary = self._state is not None
        has_embeddings = (
            self._state is not None
            and self._state.vector_index is not None
            and len(self._state.vector_index) > 0
        )
        has_derived = self._state is not None and self._state.derived is not None
        derived_fp: str | None = None

        if self._state is not None:
            (path / PRIMARY_FILENAME).write_bytes(dump_state(self._state.store))
        if has_embeddings:
            assert self._state is not None and self._state.vector_index is not None
            self._state.vector_index.save(
                path / EMBEDDINGS_FILENAME,
                path / NODE_IDS_FILENAME,
            )
        if has_derived:
            assert self._state is not None and self._state.derived is not None
            derived_dir = path / DERIVED_DIRNAME
            derived_dir.mkdir(parents=True, exist_ok=True)
            (derived_dir / DERIVED_SNAPSHOT_FILENAME).write_bytes(
                dump_derived(self._state.derived)
            )
            derived_fp = self._state.derived.fingerprint

        manifest = {
            "memory_system_id": self.memory_system_id,
            "memory_version": self.memory_version,
            "schema_version": SCHEMA_VERSION,
            "ingestion_fingerprint": ingestion_fingerprint,
            "has_primary": has_primary,
            "has_embeddings": has_embeddings,
            "has_derived": has_derived,
            "derived_fingerprint": derived_fp,
        }
        (path / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    async def load_state(self, path: Path) -> None:
        manifest_path = path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise PersistFormatError(f"manifest not found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("memory_system_id") != self.memory_system_id:
            raise PersistFormatError(
                f"memory_system_id={manifest.get('memory_system_id')!r}; "
                f"expected {self.memory_system_id!r}"
            )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"persisted schema_version={manifest.get('schema_version')}; "
                f"runtime SCHEMA_VERSION={SCHEMA_VERSION}"
            )

        if not manifest.get("has_primary", False):
            self._state = None
            return

        primary_path = path / PRIMARY_FILENAME
        if not primary_path.exists():
            raise PersistFormatError(
                f"manifest declared has_primary=True but {primary_path} missing"
            )

        pipeline = self._get_pipeline()
        store = load_state(primary_path.read_bytes())
        state = pipeline.create_state()
        state.store = store

        if manifest.get("has_embeddings", False):
            embeddings_path = path / EMBEDDINGS_FILENAME
            node_ids_path = path / NODE_IDS_FILENAME
            if not embeddings_path.exists() or not node_ids_path.exists():
                raise PersistFormatError(
                    "manifest declared has_embeddings=True but vector-index "
                    f"sidecar files are missing under {path}"
                )
            state.vector_index = VectorIndex.load(embeddings_path, node_ids_path)

        if manifest.get("has_derived", False):
            derived_path = path / DERIVED_DIRNAME / DERIVED_SNAPSHOT_FILENAME
            if not derived_path.exists():
                raise PersistFormatError(
                    "manifest declared has_derived=True but derived snapshot "
                    f"file is missing under {path}"
                )
            try:
                loaded = load_derived(derived_path.read_bytes())
            except DerivedFormatError as exc:
                raise PersistFormatError(f"derived snapshot: {exc}") from exc
            # Fingerprint audit: if the persisted derived fingerprint doesn't
            # match the one we'd compute against the freshly-loaded primary,
            # the snapshot is stale — drop it so the next recall rebuilds.
            expected = derived_fingerprint(self._config, store)
            if loaded.fingerprint == expected:
                state.derived = loaded
            # else: silently discard; a rebuild on next use regenerates.

        # memory_index is not restored from persisted primary — the graph
        # already reflects every ingested Memory. A follow-up ingest
        # continues from `state.memory_index + max-observed-index`; for
        # now we leave it at 0 since the protocol doesn't allow post-load
        # ingestion in a well-defined way (callers either load OR ingest,
        # not both). If mixed flows become real, we'll derive the max
        # memory_index from the loaded graph here.
        self._state = state

    # ------------------------------------------------------------------
    # Helpers (not part of the protocol surface; tests may touch them).
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            from engram.ingestion.factory import build_default_pipeline

            self._pipeline = build_default_pipeline(self._config)
        return self._pipeline

    def _get_recall_pipeline(self) -> RecallPipeline:
        if self._recall_pipeline is None:
            from engram.recall.factory import build_default_recall_pipeline

            self._recall_pipeline = build_default_recall_pipeline(self._config)
        return self._recall_pipeline

    def get_state(self) -> InstanceState | None:
        """Escape hatch for diagnostics / tests — do not use from Recall."""
        return self._state


__all__ = [
    "DERIVED_DIRNAME",
    "DERIVED_SNAPSHOT_FILENAME",
    "EMBEDDINGS_FILENAME",
    "EngramGraphMemorySystem",
    "MANIFEST_FILENAME",
    "NODE_IDS_FILENAME",
    "PRIMARY_FILENAME",
]
