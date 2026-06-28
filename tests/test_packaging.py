"""Phase 0 packaging sanity checks.

Verifies the package imports, its version, that every subpackage is importable,
that the expected top-level directories exist, and that ``pyproject.toml``
parses and declares the expected project metadata.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import slow_wave

SUBPACKAGES = ["stream", "memory", "agent", "dream", "eval", "repro"]
TOP_LEVEL_DIRS = ["configs", "tests", "paper", "docs"]


def test_package_imports_and_version() -> None:
    assert slow_wave.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_importable(name: str) -> None:
    module = importlib.import_module(f"slow_wave.{name}")
    assert module is not None
    assert module.__name__ == f"slow_wave.{name}"


@pytest.mark.parametrize("dirname", TOP_LEVEL_DIRS)
def test_top_level_dirs_exist(repo_root: Path, dirname: str) -> None:
    assert (repo_root / dirname).is_dir(), f"expected top-level dir: {dirname}/"


def test_pyproject_parses_and_declares_metadata(repo_root: Path) -> None:
    pyproject = repo_root / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml is missing"

    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)

    project = data["project"]
    assert project["name"] == "slow-wave"
    assert "requires-python" in project
