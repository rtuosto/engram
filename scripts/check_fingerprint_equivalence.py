"""R3/R4 structural-equivalence guard for ingestion-perf phases.

Re-ingests a fixed synthetic corpus on this commit and compares its
graph output against a pinned baseline. Flags any semantic change —
different nodes, different edges, different payload fields — while
tolerating sub-ULP float drift in numeric edge attributes.

**Why not a raw msgpack byte hash?** Batched sentence-transformer
inference produces float32 outputs that differ at the 7th–8th decimal
from singleton inference (attention-mask padding, batched BLAS kernels).
The graph that results is structurally identical — same node IDs, same
edge endpoints, same claim/preference identities — but the
``holds_preference`` edge weights (cosine confidences) drift by ~5e-8.
A raw byte check flags that as "drift" even though nothing semantically
changed. R3/R4 is about "does the ingested graph mean the same thing,"
not "do the bytes match to the last ULP."

What this script actually checks:

- node_id set matches exactly
- edge ``(src, dst, type)`` multiset matches exactly
- every node payload matches exactly (strings, ints, tuples, etc.)
- every edge attribute matches exactly EXCEPT ``weight``, which must
  be within ``FLOAT_TOLERANCE`` (default 1e-5)

Any other difference raises SystemExit(1). A new Preference popping
into existence, or a polarity flipping, is caught immediately.

Usage::

    # Capture a baseline on main:
    python scripts/check_fingerprint_equivalence.py --emit > /tmp/baseline.json

    # On a candidate commit, verify structural equivalence:
    python scripts/check_fingerprint_equivalence.py --against /tmp/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from engram import EngramGraphMemorySystem
from engram.config import MemoryConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from profile_ingestion import _synthetic_corpus  # noqa: E402

FLOAT_TOLERANCE: float = 1e-5
WEIGHT_ATTR: str = "weight"


def _payload_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _payload_to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (frozenset, set)):
        return sorted(_payload_to_jsonable(v) for v in value)
    if isinstance(value, tuple):
        return [_payload_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _payload_to_jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return repr(value)


def _node_fingerprint(nid: str, attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": nid,
        "labels": sorted(attrs.get("labels", frozenset())),
        "layers": sorted(attrs.get("layers", frozenset())),
        "payloads": {
            k: _payload_to_jsonable(v) for k, v in sorted(attrs.get("payloads", {}).items())
        },
    }


def _edge_fingerprint(src: str, dst: str, etype: str, attrs: Any) -> dict[str, Any]:
    blob = _payload_to_jsonable(attrs)
    weight = blob.pop(WEIGHT_ATTR, None) if isinstance(blob, dict) else None
    return {
        "src": src,
        "dst": dst,
        "type": etype,
        "attrs": blob,
        "weight": weight,
    }


def capture(n_memories: int) -> dict[str, Any]:
    async def _run() -> None:
        for m in _synthetic_corpus(n_memories):
            await system.ingest(m)

    system = EngramGraphMemorySystem(MemoryConfig())
    asyncio.run(_run())
    state = system.get_state()
    assert state is not None

    nodes = [
        _node_fingerprint(nid, attrs) for nid, attrs in state.store.iter_nodes()
    ]
    edges = [
        _edge_fingerprint(src, dst, etype, eattrs)
        for src, dst, etype, eattrs in state.store.iter_edges()
    ]
    nodes.sort(key=lambda n: n["id"])
    edges.sort(key=lambda e: (e["src"], e["dst"], e["type"]))
    return {"n_memories": n_memories, "nodes": nodes, "edges": edges}


def _drift(a: Any, b: Any) -> str | None:
    if isinstance(a, float) and isinstance(b, float):
        if abs(a - b) <= FLOAT_TOLERANCE:
            return None
        return f"float drift {a} vs {b} (|delta|={abs(a - b):.3e})"
    if a != b:
        return f"value mismatch {a!r} vs {b!r}"
    return None


def _report_diff(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    base_ids = {n["id"] for n in baseline["nodes"]}
    curr_ids = {n["id"] for n in current["nodes"]}
    if base_ids != curr_ids:
        missing = base_ids - curr_ids
        extra = curr_ids - base_ids
        if missing:
            issues.append(f"nodes missing in current: {sorted(missing)[:10]}...")
        if extra:
            issues.append(f"nodes extra in current: {sorted(extra)[:10]}...")

    base_by_id = {n["id"]: n for n in baseline["nodes"]}
    curr_by_id = {n["id"]: n for n in current["nodes"]}
    for nid in sorted(base_ids & curr_ids):
        bn, cn = base_by_id[nid], curr_by_id[nid]
        for field in ("labels", "layers", "payloads"):
            if bn[field] != cn[field]:
                issues.append(f"node {nid} {field} mismatch: {bn[field]!r} vs {cn[field]!r}")

    # Edge tuples — compared as sorted lists.
    be = baseline["edges"]
    ce = current["edges"]
    if len(be) != len(ce):
        issues.append(f"edge count mismatch: baseline={len(be)} current={len(ce)}")
        return issues
    for i, (b, c) in enumerate(zip(be, ce, strict=True)):
        for field in ("src", "dst", "type", "attrs"):
            if b[field] != c[field]:
                issues.append(
                    f"edge #{i} {field} mismatch: {b[field]!r} vs {c[field]!r}"
                )
        drift = _drift(b["weight"], c["weight"])
        if drift:
            issues.append(f"edge #{i} ({b['src']}->{b['dst']}/{b['type']}) {drift}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="R3/R4 structural-equivalence guard")
    parser.add_argument("--n-memories", type=int, default=20)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true", help="Print baseline JSON to stdout.")
    group.add_argument("--against", type=Path, help="Baseline JSON path; exit 1 on drift.")
    args = parser.parse_args()

    current = capture(args.n_memories)

    if args.emit:
        json.dump(current, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    baseline = json.loads(args.against.read_text(encoding="utf-8"))
    issues = _report_diff(baseline, current)
    if not issues:
        print(
            f"OK  nodes={len(current['nodes'])}  edges={len(current['edges'])}  "
            f"(float tolerance {FLOAT_TOLERANCE:.0e})"
        )
        return
    print(
        f"DRIFT  nodes_base={len(baseline['nodes'])}  nodes_curr={len(current['nodes'])}  "
        f"issues={len(issues)}",
        file=sys.stderr,
    )
    for issue in issues[:20]:
        print(f"  - {issue}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
