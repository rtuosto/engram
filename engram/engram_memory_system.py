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
from pathlib import Path
from typing import Final

from engram.config import MemoryConfig
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

MANIFEST_FILENAME: Final[str] = "manifest.json"
PRIMARY_FILENAME: Final[str] = "primary.msgpack"
EMBEDDINGS_FILENAME: Final[str] = "embeddings.npy"
NODE_IDS_FILENAME: Final[str] = "node_ids.json"


class EngramGraphMemorySystem:
    """Graph-backed MemorySystem implementation.

    Construct with an optional pre-built :class:`IngestionPipeline` (tests
    inject a mocked pipeline; production lazily loads spaCy + sentence-
    transformers on first ingest).
    """

    memory_system_id: str = MEMORY_SYSTEM_ID
    memory_version: str = "0.3.0"

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        pipeline: IngestionPipeline | None = None,
    ) -> None:
        self._config: MemoryConfig = config or MemoryConfig()
        self._pipeline: IngestionPipeline | None = pipeline
        self._state: InstanceState | None = None

    # ------------------------------------------------------------------
    # Public MemorySystem surface (R1).
    # ------------------------------------------------------------------

    async def ingest(self, memory: Memory) -> None:
        pipeline = self._get_pipeline()
        if self._state is None:
            self._state = pipeline.create_state()
        pipeline.ingest(self._state, memory)

    async def recall(
        self,
        query: str,
        *,
        now: str | None = None,
        timezone: str | None = None,
        max_passages: int | None = None,
        intent_hint: str | None = None,
    ) -> RecallResult:
        raise NotImplementedError(
            "Recall is landing in PR-E — see docs/design/recall.md. "
            "This PR ships the protocol surface only."
        )

    async def reset(self) -> None:
        self._state = None

    async def save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        ingestion_fingerprint = self._config.ingestion_fingerprint()

        has_primary = self._state is not None
        has_embeddings = (
            self._state is not None
            and self._state.vector_index is not None
            and len(self._state.vector_index) > 0
        )

        if self._state is not None:
            (path / PRIMARY_FILENAME).write_bytes(dump_state(self._state.store))
        if has_embeddings:
            assert self._state is not None and self._state.vector_index is not None
            self._state.vector_index.save(
                path / EMBEDDINGS_FILENAME,
                path / NODE_IDS_FILENAME,
            )

        manifest = {
            "memory_system_id": self.memory_system_id,
            "memory_version": self.memory_version,
            "schema_version": SCHEMA_VERSION,
            "ingestion_fingerprint": ingestion_fingerprint,
            "has_primary": has_primary,
            "has_embeddings": has_embeddings,
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

    def get_state(self) -> InstanceState | None:
        """Escape hatch for diagnostics / tests — do not use from Recall."""
        return self._state


__all__ = [
    "EMBEDDINGS_FILENAME",
    "EngramGraphMemorySystem",
    "MANIFEST_FILENAME",
    "NODE_IDS_FILENAME",
    "PRIMARY_FILENAME",
]
