"""Per-intent weight profiles (``docs/design/recall.md §5, §6``).

Weights live in ``engram/recall/weights.json`` alongside this module. One
file, two sections: ``granularity`` (per-intent seeding weights) and
``edges`` (per-intent edge-type walk weights). The file's content hash is
carried in :data:`WEIGHTS_HASH` and exposed to :mod:`engram.config` via
``MemoryConfig.recall_weights_hash`` so edits invalidate
``recall_fingerprint`` (``R3``, ``R4``).

Values are provisional defaults (``docs/design/recall.md §15``). A
Diagnostics-owned tuner will replace them with measured ones once the
benchmark exists; until then the hand-authored values encode the
intent-rationale bullets in §5 and §6.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from engram.recall.intents import INTENTS

WEIGHTS_FILE: Final[Path] = Path(__file__).parent / "weights.json"


class WeightsFormatError(RuntimeError):
    """The weights file is structurally invalid or missing an expected intent."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


WEIGHTS_HASH: Final[str] = _file_hash(WEIGHTS_FILE)


def load_weights() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return ``(granularity_weights, edge_weights)`` keyed by intent.

    ``granularity_weights[intent][granularity] -> float``
    ``edge_weights[intent][edge_type] -> float``

    Raises :class:`WeightsFormatError` if the file is malformed, if any
    intent in :data:`engram.recall.intents.INTENTS` is missing, or if a
    per-intent sub-dict is non-numeric.
    """
    raw = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise WeightsFormatError(f"{WEIGHTS_FILE} must deserialize to a dict")
    granularity = raw.get("granularity")
    edges = raw.get("edges")
    if not isinstance(granularity, dict):
        raise WeightsFormatError(f"{WEIGHTS_FILE}: missing 'granularity' dict")
    if not isinstance(edges, dict):
        raise WeightsFormatError(f"{WEIGHTS_FILE}: missing 'edges' dict")

    gran_out: dict[str, dict[str, float]] = {}
    edge_out: dict[str, dict[str, float]] = {}
    for intent in INTENTS:
        if intent not in granularity:
            raise WeightsFormatError(
                f"{WEIGHTS_FILE}: 'granularity' missing intent {intent!r}"
            )
        if intent not in edges:
            raise WeightsFormatError(
                f"{WEIGHTS_FILE}: 'edges' missing intent {intent!r}"
            )
        gran_out[intent] = {k: float(v) for k, v in granularity[intent].items()}
        edge_out[intent] = {k: float(v) for k, v in edges[intent].items()}
    return gran_out, edge_out


__all__ = [
    "WEIGHTS_FILE",
    "WEIGHTS_HASH",
    "WeightsFormatError",
    "load_weights",
]
