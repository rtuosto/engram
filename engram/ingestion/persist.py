"""Schema-versioned msgpack persistence for a ``GraphStore``.

See ``docs/design/ingestion.md §11`` and the R12 contract: every persisted
file carries an explicit ``schema_version``; load paths handle missing /
old / future versions with a clear error, not a silent crash.

**R2 discipline.** Every iteration that influences output order is sorted
before packing. Python 3.7+ dict iteration is insertion-ordered, so building
dicts in sorted order yields byte-stable msgpack output.

**Scope.** One in-memory graph per file (one engram instance). The multi-
file save layout (primary + embeddings + derived) is the
:class:`engram.engram_memory_system`-level concern; this module owns the
single-file primary contract.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Final

import msgpack  # type: ignore[import-not-found]
import numpy as np

from engram.ingestion.graph import GraphStore
from engram.ingestion.schema import (
    ClaimPayload,
    EdgeAttrs,
    EntityPayload,
    EpisodePayload,
    EventPayload,
    MemoryPayload,
    NgramPayload,
    PreferencePayload,
    TurnPayload,
    UtteranceSegmentPayload,
)

SCHEMA_VERSION: Final[int] = 1
MEMORY_SYSTEM_ID: Final[str] = "engram_graph"

# Dataclass kind registry — each payload declares a stable string so the
# loader can dispatch. Adding a new payload type requires an entry here and
# a SCHEMA_VERSION bump.
_KIND_TO_CLS: Final[dict[str, type]] = {
    "memory": MemoryPayload,
    "turn": TurnPayload,
    "utterance_segment": UtteranceSegmentPayload,
    "ngram": NgramPayload,
    "entity": EntityPayload,
    "claim": ClaimPayload,
    "preference": PreferencePayload,
    "event": EventPayload,
    "episode": EpisodePayload,
    "edge_attrs": EdgeAttrs,
}
_CLS_TO_KIND: Final[dict[type, str]] = {cls: kind for kind, cls in _KIND_TO_CLS.items()}

# Tag used to distinguish tagged dicts (dataclasses, ndarrays) from plain
# dicts on the wire. Chosen to be unambiguous and never appear as a payload
# field name.
_KIND_TAG: Final[str] = "__engram_kind__"
_NDARRAY_KIND: Final[str] = "ndarray"


class SchemaVersionMismatch(RuntimeError):
    """Persisted file's schema_version does not match the current SCHEMA_VERSION."""


class PersistFormatError(RuntimeError):
    """Persisted file is structurally invalid — missing required keys, wrong types, etc."""


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def _encode_value(value: Any) -> Any:
    """Recursively convert a value into msgpack-friendly primitives.

    - Dataclass payloads → tagged dict with ``__engram_kind__`` + sorted field dict
    - ``frozenset`` / ``set`` → sorted list (strings / primitives only)
    - ``tuple`` → list (msgpack treats them identically)
    - ``np.ndarray`` → tagged dict with shape, dtype, and raw bytes
    - ``dict`` → dict with keys sorted
    - primitives → themselves
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    if isinstance(value, np.ndarray):
        return {
            _KIND_TAG: _NDARRAY_KIND,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tobytes(order="C"),
        }

    if is_dataclass(value) and not isinstance(value, type):
        kind = _CLS_TO_KIND.get(type(value))
        if kind is None:
            raise PersistFormatError(
                f"unknown dataclass type {type(value).__name__}; "
                f"add to persist._KIND_TO_CLS and bump SCHEMA_VERSION"
            )
        encoded: dict[str, Any] = {_KIND_TAG: kind}
        for f in sorted(fields(value), key=lambda f_: f_.name):
            encoded[f.name] = _encode_value(getattr(value, f.name))
        return encoded

    if isinstance(value, (frozenset, set)):
        return [_encode_value(v) for v in sorted(value, key=str)]

    if isinstance(value, tuple):
        return [_encode_value(v) for v in value]

    if isinstance(value, list):
        return [_encode_value(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _encode_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}

    raise PersistFormatError(f"cannot encode value of type {type(value).__name__}")


def _encode_node_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Encode a node attribute dict; ``labels`` always serializes as a sorted list."""
    encoded: dict[str, Any] = {}
    for key in sorted(attrs.keys()):
        encoded[key] = _encode_value(attrs[key])
    return encoded


def dump_conversation(store: GraphStore) -> bytes:
    """Serialize ``store`` to versioned msgpack bytes — R2-deterministic."""
    nodes_payload: list[dict[str, Any]] = []
    for node_id, attrs in store.iter_nodes():
        nodes_payload.append({
            "id": node_id,
            "attrs": _encode_node_attrs(attrs),
        })

    edges_payload: list[dict[str, Any]] = []
    for src, dst, edge_type, edge_attrs in store.iter_edges():
        edges_payload.append({
            "src": src,
            "dst": dst,
            "type": edge_type,
            "attrs": _encode_value(edge_attrs),
        })

    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "memory_system_id": MEMORY_SYSTEM_ID,
        "conversation_id": store.conversation_id,
        "frozen": store.frozen,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }
    return msgpack.packb(envelope, use_bin_type=True)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        kind = value.get(_KIND_TAG)
        if kind == _NDARRAY_KIND:
            dtype = np.dtype(value["dtype"])
            shape = tuple(value["shape"])
            return np.frombuffer(value["data"], dtype=dtype).reshape(shape).copy()
        if kind is not None:
            cls = _KIND_TO_CLS.get(kind)
            if cls is None:
                raise SchemaVersionMismatch(
                    f"persisted file references unknown dataclass kind {kind!r}"
                )
            kwargs: dict[str, Any] = {}
            expected = {f.name for f in fields(cls)}
            for k, v in value.items():
                if k == _KIND_TAG:
                    continue
                if k not in expected:
                    raise SchemaVersionMismatch(
                        f"{cls.__name__} has no field {k!r} at SCHEMA_VERSION={SCHEMA_VERSION}"
                    )
                kwargs[k] = _decode_value(v)
            missing = expected - set(kwargs)
            if missing:
                raise SchemaVersionMismatch(
                    f"{cls.__name__} missing fields {sorted(missing)} in persisted file"
                )
            # Restore tuple fields that were dumped as lists.
            for f in fields(cls):
                if f.name in kwargs and isinstance(kwargs[f.name], list):
                    annotation = str(f.type)
                    if annotation.startswith("tuple"):
                        kwargs[f.name] = tuple(kwargs[f.name])
            return cls(**kwargs)
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


def _decode_node_attrs(encoded: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in encoded.items():
        decoded = _decode_value(value)
        if key in ("labels", "layers"):
            if not isinstance(decoded, list):
                raise PersistFormatError(
                    f"node {key} must serialize as a list; got {type(decoded)}"
                )
            attrs[key] = frozenset(decoded)
        else:
            attrs[key] = decoded
    return attrs


def load_conversation(data: bytes) -> GraphStore:
    """Reconstruct a ``GraphStore`` from its msgpack bytes.

    Raises :class:`SchemaVersionMismatch` on version drift and
    :class:`PersistFormatError` on structural issues.
    """
    try:
        envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as exc:
        raise PersistFormatError(f"msgpack decode failed: {exc}") from exc

    if not isinstance(envelope, dict):
        raise PersistFormatError("top-level envelope must be a dict")

    version = envelope.get("schema_version")
    if version is None:
        raise PersistFormatError("missing schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"persisted schema_version={version}; runtime SCHEMA_VERSION={SCHEMA_VERSION}. "
            f"No implicit migration; run a migration tool or reingest."
        )

    system_id = envelope.get("memory_system_id")
    if system_id != MEMORY_SYSTEM_ID:
        raise PersistFormatError(
            f"memory_system_id={system_id!r}; expected {MEMORY_SYSTEM_ID!r}"
        )

    conversation_id = envelope.get("conversation_id")
    if not isinstance(conversation_id, str):
        raise PersistFormatError("conversation_id missing or non-string")

    store = GraphStore(conversation_id=conversation_id)

    for entry in envelope.get("nodes", []):
        node_id = entry["id"]
        raw_attrs = entry["attrs"]
        attrs = _decode_node_attrs(raw_attrs)
        labels = attrs.pop("labels", frozenset())
        layers = attrs.pop("layers", frozenset())
        payloads = {k: v for k, v in attrs.items() if k in labels}
        # Any attrs keyed by a label name are payloads. Older files without
        # a ``layers`` key default to an empty frozenset — forward-compatible
        # with pre-PR-B dumps until we bump SCHEMA_VERSION in PR-C.
        store.add_node(node_id, labels=labels, payloads=payloads, layers=layers)

    for entry in envelope.get("edges", []):
        attrs = _decode_value(entry["attrs"])
        if not isinstance(attrs, EdgeAttrs):
            raise PersistFormatError(
                f"edge attrs decoded to {type(attrs).__name__}; expected EdgeAttrs"
            )
        store.add_edge(entry["src"], entry["dst"], attrs)

    if envelope.get("frozen"):
        store.freeze()

    return store


# ---------------------------------------------------------------------------
# Convenience helpers for asdict-based shallow comparison (tests).
# ---------------------------------------------------------------------------


def payload_to_dict(payload: Any) -> dict[str, Any]:
    """``asdict`` wrapper that works on our frozen-slotted payloads."""
    return asdict(payload)


__all__ = [
    "MEMORY_SYSTEM_ID",
    "PersistFormatError",
    "SCHEMA_VERSION",
    "SchemaVersionMismatch",
    "dump_conversation",
    "load_conversation",
    "payload_to_dict",
]
