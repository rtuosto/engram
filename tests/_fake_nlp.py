"""Test-only fakes for spaCy Docs and embedding callables.

These keep ingestion tests hermetic — no model downloads, no GPU, no float
non-determinism from heavy models. Each fake implements the subset of the
real API the extractors touch. If an extractor grows a new spaCy attribute
dependency, add it here rather than loosening the test to tolerate it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FakeMorph:
    tense: tuple[str, ...] = ()

    def get(self, feat: str) -> list[str]:
        if feat == "Tense":
            return list(self.tense)
        return []


@dataclass
class FakeToken:
    text: str
    idx: int
    lemma_: str = ""
    pos_: str = ""
    dep_: str = ""
    label_: str = ""
    morph: FakeMorph = field(default_factory=FakeMorph)
    children: tuple[FakeToken, ...] = ()
    # subtree defaults to [self]; tests that care supply it explicitly.
    _subtree: tuple[FakeToken, ...] | None = None

    @property
    def subtree(self) -> tuple[FakeToken, ...]:
        return self._subtree if self._subtree is not None else (self,)

    @property
    def text_with_ws(self) -> str:
        return self.text + " "


@dataclass
class FakeEnt:
    text: str
    label_: str
    start_char: int
    end_char: int


@dataclass
class FakeSent:
    text: str
    start_char: int
    end_char: int
    root: FakeToken | None = None


@dataclass
class FakeDoc:
    """Minimal stand-in for :class:`spacy.tokens.Doc`."""

    text: str = ""
    sents: tuple[FakeSent, ...] = ()
    ents: tuple[FakeEnt, ...] = ()


def make_token(
    text: str,
    idx: int,
    *,
    pos: str = "",
    dep: str = "",
    lemma: str | None = None,
    tense: tuple[str, ...] = (),
    children: Iterable[FakeToken] = (),
) -> FakeToken:
    children_tup = tuple(children)
    token = FakeToken(
        text=text,
        idx=idx,
        lemma_=(lemma if lemma is not None else text).casefold(),
        pos_=pos,
        dep_=dep,
        morph=FakeMorph(tense=tense),
        children=children_tup,
    )
    return token


def attach_subtree(token: FakeToken, subtree: Iterable[FakeToken]) -> FakeToken:
    """Helper: set a token's subtree (for dep-parse simulations)."""
    token._subtree = tuple(subtree)
    return token


def make_fake_doc(
    text: str,
    sents: Iterable[FakeSent] = (),
    ents: Iterable[FakeEnt] = (),
) -> FakeDoc:
    return FakeDoc(text=text, sents=tuple(sents), ents=tuple(ents))


def deterministic_embed(dim: int = 16):
    """Return an ``embed_fn(list[str]) -> (n, dim) ndarray`` seeded by string hash.

    Each input text is hashed to a stable byte sequence, unpacked to floats,
    L2-normalized, and returned. Caller code that assumes normalized vectors
    works unchanged; tests can feed any strings without contamination from
    prior calls.
    """

    def embed(texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            # Use the hash as float32 bytes — 256 bits = 8 float32s. Tile if dim > 8.
            needed = dim * 4
            if len(seed_bytes) < needed:
                reps = (needed + len(seed_bytes) - 1) // len(seed_bytes)
                seed_bytes = (seed_bytes * reps)[:needed]
            else:
                seed_bytes = seed_bytes[:needed]
            row = np.frombuffer(seed_bytes, dtype=np.uint32).astype(np.float32)
            # Map to [-1, 1] before normalizing — avoids degenerate all-positive vectors.
            row = (row / np.uint32(0xFFFFFFFF)) * 2.0 - 1.0
            norm = np.linalg.norm(row)
            if norm > 0.0:
                row = row / norm
            rows.append(row.astype(np.float32))
        return np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float32)

    return embed


def make_nlp_process(docs_by_text: dict[str, FakeDoc]):
    """Return a fake ``nlp_process`` that looks up pre-built Docs by text."""

    def process(texts: list[str]) -> list[object]:
        out: list[object] = []
        for text in texts:
            if text not in docs_by_text:
                raise KeyError(
                    f"make_nlp_process: no FakeDoc registered for text {text!r}. "
                    f"Add it to the docs_by_text dict in the test."
                )
            out.append(docs_by_text[text])
        return out

    return process


__all__ = [
    "FakeDoc",
    "FakeEnt",
    "FakeMorph",
    "FakeSent",
    "FakeToken",
    "attach_subtree",
    "deterministic_embed",
    "make_fake_doc",
    "make_nlp_process",
    "make_token",
]
