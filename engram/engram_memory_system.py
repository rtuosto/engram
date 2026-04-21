"""``EngramGraphMemorySystem`` — the concrete :class:`MemorySystem` implementation.

Implements ingestion in Tier-1 scope per ``docs/design/ingestion.md``.
Recall (:meth:`answer_question`) is a separate design-doc iteration and
raises :class:`NotImplementedError` here; external callers that only need
ingest + save-state semantics work today.

**Persistence layout** (R12, manifesto §K7):

```
<save_path>/
    manifest.json                       # memory_system_id, version,
                                        # schema_version, ingestion_fingerprint,
                                        # list of conversation_ids
    <conversation_id>.msgpack           # one per conversation
```

``load_state`` verifies ``schema_version`` matches :data:`persist.SCHEMA_VERSION`,
``memory_system_id`` matches this instance's id, and per-conversation files
decode cleanly. No implicit migration.
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
    dump_conversation,
    load_conversation,
)
from engram.ingestion.pipeline import ConversationState, IngestionPipeline
from engram.models import AnswerResult, Session

MANIFEST_FILENAME: Final[str] = "manifest.json"
CONVERSATION_SUFFIX: Final[str] = ".msgpack"


class EngramGraphMemorySystem:
    """Graph-backed MemorySystem implementation.

    Construct with an optional pre-built :class:`IngestionPipeline` (tests
    inject a mocked pipeline; production lazily loads spaCy + sentence-
    transformers on first ingest).
    """

    memory_system_id: str = MEMORY_SYSTEM_ID
    version: str = "0.1.0"

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        pipeline: IngestionPipeline | None = None,
    ) -> None:
        self._config: MemoryConfig = config or MemoryConfig()
        self._pipeline: IngestionPipeline | None = pipeline
        self._conversations: dict[str, ConversationState] = {}

    # ------------------------------------------------------------------
    # Public MemorySystem surface (R1).
    # ------------------------------------------------------------------

    async def ingest_session(self, session: Session, conversation_id: str) -> None:
        pipeline = self._get_pipeline()
        state = self._conversations.get(conversation_id)
        if state is None:
            state = pipeline.create_state(conversation_id)
            self._conversations[conversation_id] = state
        pipeline.ingest_session(state, session)

    async def finalize_conversation(self, conversation_id: str) -> None:
        state = self._conversations.get(conversation_id)
        if state is None:
            return
        pipeline = self._get_pipeline()
        pipeline.finalize_conversation(state)

    async def answer_question(self, question: str, conversation_id: str) -> AnswerResult:
        raise NotImplementedError(
            "Recall is a separate design-doc iteration — see docs/design/ "
            "(pending 'recall.md'). Tier-1 ingestion ships without answer_question."
        )

    async def reset(self) -> None:
        self._conversations.clear()

    async def save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        ingestion_fingerprint = self._config.ingestion_fingerprint()
        conversations_sorted = sorted(self._conversations.items())

        for conversation_id, state in conversations_sorted:
            target = path / f"{conversation_id}{CONVERSATION_SUFFIX}"
            target.write_bytes(dump_conversation(state.store))

        manifest = {
            "memory_system_id": self.memory_system_id,
            "version": self.version,
            "schema_version": SCHEMA_VERSION,
            "ingestion_fingerprint": ingestion_fingerprint,
            "conversation_ids": [cid for cid, _ in conversations_sorted],
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

        pipeline = self._get_pipeline()
        restored: dict[str, ConversationState] = {}
        for conversation_id in manifest.get("conversation_ids", []):
            target = path / f"{conversation_id}{CONVERSATION_SUFFIX}"
            if not target.exists():
                raise PersistFormatError(
                    f"manifest lists {conversation_id!r} but file {target} missing"
                )
            store = load_conversation(target.read_bytes())
            state = pipeline.create_state(conversation_id)
            state.store = store
            restored[conversation_id] = state
        self._conversations = restored

    # ------------------------------------------------------------------
    # Helpers (not part of the protocol surface; tests may touch them).
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            from engram.ingestion.factory import build_default_pipeline

            self._pipeline = build_default_pipeline(self._config)
        return self._pipeline

    def get_state(self, conversation_id: str) -> ConversationState | None:
        """Escape hatch for diagnostics / tests — do not use from Recall."""
        return self._conversations.get(conversation_id)


__all__ = [
    "CONVERSATION_SUFFIX",
    "EngramGraphMemorySystem",
    "MANIFEST_FILENAME",
]
