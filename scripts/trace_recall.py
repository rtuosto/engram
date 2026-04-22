"""Recall play-by-play CLI — trace one query through the five-stage pipeline.

Loads a persisted engram state (or spins up a fresh in-memory instance and
ingests a scripted corpus), executes :func:`engram.diagnostics.traced_recall`
on the supplied query, and prints the human-readable play-by-play. Pass
``--json`` to emit a machine-readable JSON dump instead.

Example — trace a query against a saved benchmark state::

    python scripts/trace_recall.py \\
        --state ~/code/agent-memory-benchmark/cache/ingestion/<hash> \\
        --query "What was my personal best time in the charity 5K run?"

Example — trace against the synthetic profile corpus (no state required)::

    python scripts/trace_recall.py \\
        --synthetic 40 \\
        --query "What does Alice love?"

Emits to stdout. Exit code 0 unless the query raises.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from engram import EngramGraphMemorySystem
from engram.config import MemoryConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


async def _ingest_synthetic(system: EngramGraphMemorySystem, n: int) -> None:
    from profile_ingestion import _synthetic_corpus  # noqa: PLC0415

    for m in _synthetic_corpus(n):
        await system.ingest(m)


async def _run(
    query: str,
    *,
    state_path: Path | None,
    synthetic_n: int | None,
    now: str | None,
    timezone: str | None,
    max_passages: int | None,
    intent_hint: str | None,
    as_json: bool,
    html_path: Path | None,
) -> None:
    system = EngramGraphMemorySystem(MemoryConfig())

    if state_path is not None:
        if not state_path.exists():
            raise SystemExit(f"state path not found: {state_path}")
        await system.load_state(state_path)
    elif synthetic_n is not None:
        await _ingest_synthetic(system, synthetic_n)
    else:
        raise SystemExit("must supply --state <dir> or --synthetic <n>")

    result, trace = await system.recall_trace(
        query,
        now=now,
        timezone=timezone,
        max_passages=max_passages,
        intent_hint=intent_hint,
    )

    if html_path is not None:
        from engram.diagnostics import render_trace_html

        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_trace_html(trace), encoding="utf-8")
        print(f"dashboard: {html_path}")

    if as_json:
        payload = {
            "query": query,
            "result": {
                "intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "passage_count": len(result.passages),
                "fact_count": len(result.facts),
                "recall_fingerprint": result.recall_fingerprint,
            },
            "trace": trace.to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif html_path is None:
        print(trace.pretty())
        # Also a result-line summary so stdout is useful by itself.
        print()
        print(
            f"Result: {len(result.passages)} passages, {len(result.facts)} facts, "
            f"intent={result.intent!r}, conf={result.intent_confidence:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--state",
        type=Path,
        help="Path to a persisted engram state directory (load via load_state).",
    )
    source.add_argument(
        "--synthetic",
        type=int,
        metavar="N",
        help="Skip load_state; ingest N synthetic-corpus memories and trace against that.",
    )
    parser.add_argument("--query", required=True, help="The query string to trace.")
    parser.add_argument("--now", default=None)
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument(
        "--intent-hint",
        default=None,
        help="Skip intent classification with an agent-style hint (one of the INTENTS).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit trace as JSON (for programmatic consumption).",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write a self-contained HTML dashboard to PATH. Single file, "
            "inline CSS+JS, no server required — open directly in a browser."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        _run(
            args.query,
            state_path=args.state,
            synthetic_n=args.synthetic,
            now=args.now,
            timezone=args.timezone,
            max_passages=args.max_passages,
            intent_hint=args.intent_hint,
            as_json=args.json,
            html_path=args.html,
        )
    )


if __name__ == "__main__":
    main()
