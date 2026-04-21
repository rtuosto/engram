"""Verification step 3: ``MemorySystem`` protocol has the R1-mandated shape.

The protocol is the single surface external callers touch — the
``agent-memory-benchmark`` repo and this repo's :mod:`engram.diagnostics`
(``docs/DESIGN-MANIFESTO.md §R1``, ``§6``). This test pins the shape so that
adding or removing a public verb is an intentional, reviewable change — not a
silent refactor that breaks the measurement instrument.

A minimal ``FakeMemory`` implementation is included both as a runtime-check
witness (``isinstance(fake, MemorySystem)``) and as executable documentation
of what a valid implementation looks like.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from engram import AnswerResult, MemorySystem, RetrievedNode, Session, Turn

_EXPECTED_VERBS = frozenset({
    "ingest_session",
    "finalize_conversation",
    "answer_question",
    "reset",
    "save_state",
    "load_state",
})


_EXPECTED_ATTRIBUTES = frozenset({"memory_system_id", "version"})


class _FakeMemory:
    """Minimal MemorySystem implementation — no-op stubs for shape verification."""

    memory_system_id: str = "fake"
    version: str = "0.0.1-test"

    async def ingest_session(self, session: Session, conversation_id: str) -> None:
        return None

    async def finalize_conversation(self, conversation_id: str) -> None:
        return None

    async def answer_question(
        self, question: str, conversation_id: str
    ) -> AnswerResult:
        return AnswerResult(
            answer="",
            context="",
            retrieved_nodes=(),
            retrieval_time_ms=0.0,
            answer_time_ms=0.0,
            total_time_ms=0.0,
        )

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
        for marker in ("R1", "R2", "R4", "R5", "R12", "P8", "K5", "M5", "docs/DESIGN-MANIFESTO")
    )
    assert has_rule_citation, (
        f"{verb} docstring must cite at least one manifesto rule/principle — "
        f"got: {doc[:100]!r}"
    )


async def test_fake_memory_roundtrip() -> None:
    """End-to-end smoke: instantiate → ingest → finalize → answer → reset."""
    fake = _FakeMemory()
    session = Session(
        session_index=1,
        timestamp="2026-04-20T10:00:00",
        turns=(
            Turn(speaker="user", text="hello", session_index=1, turn_index=1),
            Turn(speaker="assistant", text="hi", session_index=1, turn_index=2),
        ),
    )
    await fake.ingest_session(session, "conv-1")
    await fake.finalize_conversation("conv-1")
    result = await fake.answer_question("what did I say?", "conv-1")
    assert isinstance(result, AnswerResult)
    assert result.answer == ""
    assert isinstance(result.retrieved_nodes, tuple)
    await fake.reset()


def test_retrieved_node_is_frozen() -> None:
    node = RetrievedNode(
        node_id="n1",
        node_type="turn",
        text="hello",
        conversation_id="c1",
    )
    with pytest.raises(AttributeError):
        node.node_id = "n2"  # type: ignore[misc]
