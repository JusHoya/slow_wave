"""Tests for the no-sleep wake runner + manifest (Phase 2, WS-AGENT, EC1).

The runner must write a manifest that carries the populated accuracy matrix
``R[i,j]`` (``results.accuracy_matrix``) alongside full cost telemetry
(``cost.api_calls``, ``cost.tokens.total``, ``results.telemetry``), round-trip
cleanly via :func:`slow_wave.repro.manifest.read_manifest`, and refuse to run
without a stream.
"""

from __future__ import annotations

import pytest

from slow_wave.agent.runner import run_agent
from slow_wave.config import Config, load_config
from slow_wave.repro.manifest import read_manifest


@pytest.fixture
def smoke_cfg(repo_root):
    """The unbounded no-sleep baseline config (``configs/agent_smoke.yaml``)."""
    return load_config(repo_root / "configs" / "agent_smoke.yaml")


def test_run_agent_writes_populated_manifest(smoke_cfg, tmp_path, force_mock_llm):
    """The manifest carries a populated R[i,j] + cost telemetry and round-trips."""
    out = tmp_path / "out"
    path = run_agent(smoke_cfg, out_dir=out)

    assert path.exists()
    assert path == out / "agent" / "manifest.json"

    manifest = read_manifest(path)

    # results.accuracy_matrix populated and the right shape.
    R = manifest.results["accuracy_matrix"]["R"]
    assert len(R) == smoke_cfg.stream.n_tasks
    assert all(len(row) == smoke_cfg.stream.n_tasks for row in R)
    assert R  # non-empty

    # Cost telemetry recorded (EC1).
    assert manifest.cost.api_calls >= 1
    assert manifest.cost.tokens.total > 0

    # Continual metrics + telemetry travelled in results.
    assert "continual_metrics" in manifest.results
    assert "memory_footprint" in manifest.results
    tel = manifest.results["telemetry"]
    assert tel["n_items_ingested"] == (
        smoke_cfg.stream.n_tasks * smoke_cfg.stream.items_per_task
    )
    assert tel["api_calls"] == manifest.cost.api_calls


def test_manifest_round_trips_identically(smoke_cfg, tmp_path, force_mock_llm):
    """Reading the written manifest twice yields identical results payloads."""
    path = run_agent(smoke_cfg, out_dir=tmp_path / "out")
    first = read_manifest(path)
    second = read_manifest(path)
    assert first.results == second.results
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_run_agent_requires_stream(tmp_path):
    """A config with no stream raises ValueError before doing any work."""
    cfg = Config(experiment="no-stream-agent")
    assert cfg.stream is None
    with pytest.raises(ValueError):
        run_agent(cfg, out_dir=tmp_path)
