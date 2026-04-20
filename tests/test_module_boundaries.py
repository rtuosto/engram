"""Verification step 2: four module dirs importable with boundary docstrings.

Each top-level module (``ingestion``, ``recall``, ``benchmarking``,
``diagnostics``) must be importable and carry a non-empty module docstring.
The docstring is the top-of-module boundary contract — deleting it is a
boundary violation that this test catches.
"""

from __future__ import annotations

import importlib

import pytest

_MODULES = (
    "engram",
    "engram.ingestion",
    "engram.recall",
    "engram.benchmarking",
    "engram.diagnostics",
)


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_importable(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_has_boundary_docstring(module_name: str) -> None:
    module = importlib.import_module(module_name)
    doc = module.__doc__ or ""
    assert doc.strip(), f"{module_name} must carry a boundary docstring"
    assert len(doc) > 100, (
        f"{module_name} docstring must describe Responsibility / Owns / "
        f"Does not touch / Stability — got {len(doc)} chars"
    )


def test_engram_exposes_version() -> None:
    import engram

    assert isinstance(engram.__version__, str)
    assert engram.__version__
