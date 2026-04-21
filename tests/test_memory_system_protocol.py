"""Verification step 3: ``MemorySystem`` protocol has the R1-mandated shape.

The protocol is the single surface external callers touch — the
``agent-memory-benchmark`` repo and this repo's :mod:`engram.diagnostics`
(``docs/DESIGN-MANIFESTO.md §R1``, ``§6``). This test pins the shape so
that adding or removing a public verb is an intentional, reviewable change
— not a silent refactor that breaks the measurement instrument.

A minimal ``FakeMemory`` implementation is included both as a runtime-check
witness (``isinstance(fake, MemorySystem)``) and as executable documentation
of what a valid implementation looks like.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from engram import Memory, MemorySystem, RecallResult

_EXPECTED_VERBS = frozenset({
    "ingest",
    "recall",
    "reset",
    "save_state",
    "load_state",
})


_EXPECTED_ATTRIBUTES = frozenset({"memory_system_id", "memory_version"})


class _FakeMemory:
    """Minimal MemorySystem implementation — no-op stubs for shape verification."""

    memory_system_id: str = "fake"
    memory_version: str = "0.0.1-test"

    async def ingest(self, memory: Memory) -> None:
        return None

    async def recall(
        self,
        query: str,
        *,
        now: str | None = None,
        timezone: str | None = None,
        max_passages: int | None = None,
        intent_hint: str | None = None,
    ) -> RecallResult:
        return RecallResult(passages=())

    async def reset(self) -> None:
        return None

    async def save_state(self, path: Path) -> None:
        return None

    async def load_state(self, path: Path) -> None:
        return None


def test_protocol_exposes_expected_verbs() -> None:
    """All R1 public verbs must exist on the Protocol as callables."""
    for verb in _EXPECTED_VERBS:
        assert hasattr(MemorySystem, verb), f"MemorySystem missing public verb: {verb}"


def test_protocol_has_no_deprecated_verbs() -> None:
    """Pre-pivot verbs must be gone — the protocol no longer carries them."""
    for removed in ("ingest_session", "finalize_conversation", "answer_question"):
        assert not hasattr(MemorySystem, removed), (
            f"deprecated verb {removed} still on MemorySystem — "
            f"post-pivot surface is ingest/recall/reset/save_state/load_state"
        )


def test_protocol_exposes_identity_attributes() -> None:
    for name in _EXPECTED_ATTRIBUTES:
        assert name in MemorySystem.__annotations__, (
            f"MemorySystem missing identity attribute: {name}"
        )


def test_fake_memory_satisfies_protocol() -> None:
    """A minimal implementation passes the runtime isinstance check.

    If this fails, the Protocol has drifted (a new required verb was added
    without updating the fake, or vice-versa).
    """
    fake = _FakeMemory()
    assert isinstance(fake, MemorySystem)


@pytest.mark.parametrize("verb", sorted(_EXPECTED_VERBS))
def test_public_verbs_are_async(verb: str) -> None:
    """Every public verb is async — Recall and Ingestion both hit IO / LLMs."""
    fake = _FakeMemory()
    method = getattr(fake, verb)
    assert inspect.iscoroutinefunction(method), f"{verb} must be async"


@pytest.mark.parametrize("verb", sorted(_EXPECTED_VERBS))
def test_public_verbs_have_docstrings(verb: str) -> None:
    """Every public verb docstring must cite at least one manifesto rule or principle."""
    method = getattr(MemorySystem, verb)
    doc = method.__doc__ or ""
    assert doc.strip(), f"{verb} missing docstring"
    has_rule_citation = any(
        marker in doc
        for marker in (
            "R1",
            "R2",
            "R4",
            "R5",
            "R8",
            "R9",
            "R12",
            "R13",
            "R16",
            "P8",
            "K5",
            "M5",
            "docs/design",
            "docs/DESIGN-MANIFESTO",
        )
    )
    assert has_rule_citation, (
        f"{verb} docstring must cite at least one manifesto rule/principle — "
        f"got: {doc[:120]!r}"
    )


async def test_fake_memory_roundtrip() -> None:
    """End-to-end smoke: instantiate → ingest → recall → reset."""
    fake = _FakeMemory()
    memory = Memory(
        content="hello world",
        timestamp="2026-04-20T10:00:00Z",
        speaker="user",
    )
    await fake.ingest(memory)
    result = await fake.recall("what did I say?")
    assert isinstance(result, RecallResult)
    assert result.passages == ()
    await fake.reset()


def test_recall_kwargs_shape() -> None:
    """``recall`` signature must accept the context kwargs (now, timezone, max_passages, intent_hint)."""
    sig = inspect.signature(MemorySystem.recall)
    expected_params = {"self", "query", "now", "timezone", "max_passages", "intent_hint"}
    assert set(sig.parameters) == expected_params, (
        f"recall signature mismatch: {sorted(sig.parameters)} vs {sorted(expected_params)}"
    )
