"""Phase 1 ingestion profile harness.

Measures engram's ingestion pipeline with real models (spaCy + MiniLM + mpnet).
Emits a JSON artifact under ``profiling/`` that all later ingestion-perf
phases diff against.

**Why.** The performance plan at ``.agent/current-plan.md`` gates every
optimization on a measured delta against a reproducible baseline. This is
that baseline.

**Measurement model.** The three model-dependent callables wired by
:func:`engram.ingestion.factory.build_default_pipeline` are the hottest
stages by code inspection. We wrap each with a counter + perf-counter
timer before handing them to :class:`IngestionPipeline`:

- ``nlp_process`` (spaCy sentence + dependency parse, stage [2])
- ``preference_embed`` (mpnet, called per-sentence at stage [7])
- ``granule_embed`` (MiniLM, batched once per ingest at stage [8])

Per-ingest wall-clock is also recorded. What's *not* measured directly
per-stage: canonicalization (rapidfuzz), graph writes, temporal anchoring.
Those fall into ``other_ms`` (total_ingest - sum-of-measured-stages). An
optional ``--cprofile`` flag dumps a ``.prof`` sample so we can drill into
``other`` if it turns out to be large.

**Corpus.** Synthetic by default — the shape (sentence count per memory,
entity density, claim density) matters more than content for timing. If
``agent-memory-benchmark`` is importable, ``--corpus longmemeval`` will
pull the first N turns from LongMemEval-s instead.

Usage::

    python scripts/profile_ingestion.py --n-memories 50
    python scripts/profile_ingestion.py --n-memories 20 --cprofile
    python scripts/profile_ingestion.py --corpus longmemeval --n-memories 100
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import pstats
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engram import EngramGraphMemorySystem, Memory
from engram.config import MemoryConfig
from engram.diagnostics import extraction_coverage
from engram.ingestion.factory import build_default_pipeline
from engram.ingestion.pipeline import IngestionPipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILING_DIR = REPO_ROOT / "profiling"


@dataclass
class StageMeter:
    """Wall-clock accumulator for one injected callable."""

    name: str
    calls: int = 0
    items: int = 0
    total_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "items": self.items,
            "total_ms": round(self.total_s * 1000.0, 3),
            "mean_ms_per_call": round((self.total_s / self.calls) * 1000.0, 3)
            if self.calls
            else 0.0,
            "mean_ms_per_item": round((self.total_s / self.items) * 1000.0, 3)
            if self.items
            else 0.0,
        }


@dataclass
class ProfileRun:
    commit: str
    memories: int
    corpus: str
    model_load_s: float
    total_ingest_s: float
    per_ingest_ms: list[float] = field(default_factory=list)
    stage_meters: dict[str, StageMeter] = field(default_factory=dict)
    graph_totals: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        measured_stage_s = sum(m.total_s for m in self.stage_meters.values())
        other_s = max(0.0, self.total_ingest_s - measured_stage_s)
        per_ingest = sorted(self.per_ingest_ms)
        n = len(per_ingest) or 1
        p50 = per_ingest[n // 2]
        p95 = per_ingest[min(n - 1, int(n * 0.95))]
        return {
            "commit": self.commit,
            "memories": self.memories,
            "corpus": self.corpus,
            "model_load_s": round(self.model_load_s, 3),
            "total_ingest_s": round(self.total_ingest_s, 3),
            "mean_ingest_ms": round((self.total_ingest_s / max(1, self.memories)) * 1000.0, 3),
            "p50_ingest_ms": round(p50, 3),
            "p95_ingest_ms": round(p95, 3),
            "max_ingest_ms": round(max(per_ingest), 3) if per_ingest else 0.0,
            "stages": {name: m.as_dict() for name, m in self.stage_meters.items()},
            "other_ms": round(other_s * 1000.0, 3),
            "graph_totals": self.graph_totals,
        }


def _wrap_callable(
    inner: Callable[[list[str]], Any],
    meter: StageMeter,
) -> Callable[[list[str]], Any]:
    """Wrap an injected callable so each call charges its wall-clock to ``meter``."""

    def wrapped(texts: list[str]) -> Any:
        t0 = time.perf_counter()
        try:
            return inner(texts)
        finally:
            meter.calls += 1
            meter.items += len(texts)
            meter.total_s += time.perf_counter() - t0

    return wrapped


def _build_instrumented_pipeline(
    config: MemoryConfig, stage_meters: dict[str, StageMeter]
) -> tuple[IngestionPipeline, float]:
    """Build a real pipeline, wrapping the three model callables with timers.

    Returns ``(pipeline, model_load_s)``. Model load includes spaCy +
    sentence-transformers + centroid computation + held-out-margin probes.
    """
    t0 = time.perf_counter()
    base = build_default_pipeline(config)
    model_load_s = time.perf_counter() - t0

    for name in ("nlp_process", "preference_embed", "granule_embed"):
        stage_meters[name] = StageMeter(name=name)

    instrumented = IngestionPipeline(
        config=config,
        nlp_process=_wrap_callable(base._nlp_process, stage_meters["nlp_process"]),
        preference_centroids=base._centroids,
        preference_embed=_wrap_callable(
            base._preference_embed, stage_meters["preference_embed"]
        ),
        granule_embed=_wrap_callable(base._granule_embed, stage_meters["granule_embed"]),
        enabled_polarities=base._enabled_polarities,
    )
    return instrumented, model_load_s


# ----------------------------------------------------------------------
# Corpora
# ----------------------------------------------------------------------

SYNTHETIC_TEMPLATES: tuple[str, ...] = (
    "Alice loves hiking in the mountains with her dog Rex. "
    "She prefers weekends because the trails near Seattle get too crowded on weekdays. "
    "Last weekend she drove to Mount Rainier and hiked eight miles.",
    "Bob avoids crowded restaurants because the noise bothers him. "
    "He recommends the quiet cafe on Pine Street that opens at seven in the morning. "
    "The owner Maria makes excellent espresso and knows every regular by name.",
    "Carol visited Paris last March and toured the Louvre for three hours. "
    "She particularly liked the Delacroix paintings on the second floor. "
    "Afterwards she walked along the Seine and ate dinner at a small bistro.",
    "David works as a backend engineer at Stripe in San Francisco. "
    "He is learning Rust on weekends and has been contributing to tokio. "
    "His manager Priya scheduled a one-on-one every Wednesday morning.",
    "Emily dislikes spicy food but loves Thai curries made with coconut milk. "
    "Her favorite restaurant is Nong's on Burnside. "
    "The owner recognizes her and always checks that the curry is mild.",
    "Frank started running marathons after turning forty and finished Boston last year. "
    "He trains with a group that meets at six in the morning near Fenway Park. "
    "His coach Marcus coordinates the weekly long runs.",
    "Grace is allergic to peanuts and has to check every menu carefully. "
    "At the new Thai place downtown she ordered the pad see ew without peanuts. "
    "The chef came out personally to confirm there was no cross-contamination.",
    "Henry collects vintage watches and owns a 1965 Omega Speedmaster. "
    "He bought it at an auction in Geneva for twelve thousand dollars. "
    "His wife Jane thinks the hobby is extravagant but tolerates it.",
)


def _synthetic_corpus(n: int) -> list[Memory]:
    base_day = 1
    memories: list[Memory] = []
    for i in range(n):
        tpl = SYNTHETIC_TEMPLATES[i % len(SYNTHETIC_TEMPLATES)]
        timestamp = f"2026-01-{base_day + (i % 28):02d}T{(i % 24):02d}:00:00Z"
        memories.append(
            Memory(
                content=tpl,
                timestamp=timestamp,
                speaker="user",
                source="synthetic_profile",
            )
        )
    return memories


def _longmemeval_corpus(n: int) -> list[Memory] | None:
    """Best-effort load; returns ``None`` if the benchmark repo isn't on path."""
    try:
        from agent_memory_benchmark.datasets.longmemeval import LongMemEvalDataset  # type: ignore
    except Exception:
        return None
    try:
        ds = LongMemEvalDataset.load_s(limit=max(1, n // 8))  # type: ignore[attr-defined]
    except Exception:
        return None

    memories: list[Memory] = []
    for case in ds.cases():
        for session in case.sessions:
            for turn in session.turns:
                memories.append(
                    Memory(
                        content=turn.text,
                        timestamp=turn.timestamp,
                        speaker=turn.speaker,
                        source="longmemeval",
                    )
                )
                if len(memories) >= n:
                    return memories
    return memories or None


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


async def _ingest_all(
    system: EngramGraphMemorySystem,
    memories: list[Memory],
    per_ingest_ms: list[float],
) -> None:
    for m in memories:
        t0 = time.perf_counter()
        await system.ingest(m)
        per_ingest_ms.append((time.perf_counter() - t0) * 1000.0)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _graph_totals(system: EngramGraphMemorySystem) -> dict[str, int]:
    state = system.get_state()
    if state is None:
        return {}
    report = extraction_coverage(state.store)
    return {k: int(v) for k, v in report.totals}


def run_profile(
    *,
    n_memories: int,
    corpus: str,
    enable_cprofile: bool,
    out_path: Path | None,
) -> ProfileRun:
    if corpus == "longmemeval":
        mems = _longmemeval_corpus(n_memories)
        if mems is None:
            print(
                "[profile] agent-memory-benchmark not importable; "
                "falling back to synthetic corpus",
                file=sys.stderr,
            )
            corpus = "synthetic"
            mems = _synthetic_corpus(n_memories)
    else:
        mems = _synthetic_corpus(n_memories)

    config = MemoryConfig()
    stage_meters: dict[str, StageMeter] = {}
    pipeline, model_load_s = _build_instrumented_pipeline(config, stage_meters)
    system = EngramGraphMemorySystem(config, pipeline=pipeline)

    per_ingest_ms: list[float] = []
    profiler: cProfile.Profile | None = None
    if enable_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    t0 = time.perf_counter()
    asyncio.run(_ingest_all(system, mems, per_ingest_ms))
    total_s = time.perf_counter() - t0

    if profiler is not None:
        profiler.disable()

    run = ProfileRun(
        commit=_git_commit(),
        memories=len(mems),
        corpus=corpus,
        model_load_s=model_load_s,
        total_ingest_s=total_s,
        per_ingest_ms=per_ingest_ms,
        stage_meters=stage_meters,
        graph_totals=_graph_totals(system),
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(run.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if profiler is not None:
            prof_path = out_path.with_suffix(".prof")
            profiler.dump_stats(str(prof_path))
            # Also emit a readable top-50 to stdout for quick review.
            stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
            print(f"\n[profile] cProfile top-30 by cumulative time:")
            stats.print_stats(30)
            print(f"[profile] full .prof dumped to {prof_path}")

    return run


def _print_summary(run: ProfileRun) -> None:
    d = run.as_dict()
    print("\n=== ingestion profile ===")
    print(f"commit:          {d['commit']}")
    print(f"corpus:          {d['corpus']}")
    print(f"memories:        {d['memories']}")
    print(f"model_load_s:    {d['model_load_s']}")
    print(f"total_ingest_s:  {d['total_ingest_s']}")
    print(
        f"mean_ingest_ms:  {d['mean_ingest_ms']}   "
        f"(p50 {d['p50_ingest_ms']}  p95 {d['p95_ingest_ms']}  max {d['max_ingest_ms']})"
    )
    print(f"other_ms:        {d['other_ms']}")
    print(f"\n--- per-stage (measured callables) ---")
    for name, m in d["stages"].items():
        print(
            f"  {name:18s}  calls={m['calls']:4d}  items={m['items']:5d}  "
            f"total_ms={m['total_ms']:10.3f}  "
            f"mean/call={m['mean_ms_per_call']:.3f}ms  "
            f"mean/item={m['mean_ms_per_item']:.3f}ms"
        )
    if run.graph_totals:
        print(f"\n--- graph totals ---")
        for k, v in run.graph_totals.items():
            print(f"  {k:18s}  {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-memories", type=int, default=20)
    parser.add_argument(
        "--corpus", choices=("synthetic", "longmemeval"), default="synthetic"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON artifact path (default: profiling/ingestion-<commit>.json)",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Also dump a .prof sample next to the JSON artifact.",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        commit = _git_commit()
        out_path = PROFILING_DIR / f"ingestion-{commit}.json"

    run = run_profile(
        n_memories=args.n_memories,
        corpus=args.corpus,
        enable_cprofile=args.cprofile,
        out_path=out_path,
    )
    _print_summary(run)
    print(f"\n[profile] artifact: {out_path}")


if __name__ == "__main__":
    main()
