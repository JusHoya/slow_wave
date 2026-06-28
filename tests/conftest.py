"""Shared pytest fixtures for the Slow Wave test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def force_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the deterministic mock LLM path by removing any API key.

    Tests that assert reproducible, LLM-independent behavior depend on this so
    they never accidentally make a real (non-deterministic) Claude API call.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """A fresh temporary output directory for a single run."""
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def smoke_config_path(repo_root: Path) -> Path:
    """Path to the canonical smoke config (``configs/smoke.yaml``)."""
    return repo_root / "configs" / "smoke.yaml"
