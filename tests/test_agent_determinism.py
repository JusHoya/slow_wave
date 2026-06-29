"""Phase 2 integration: agent-run determinism (DX1) + confound re-check (FR1.6).

Two cross-module guards a red-team relies on:

* **DX1 reproducibility.** Two ``run_agent`` invocations with the same config +
  seed produce byte-identical non-LLM outputs — the accuracy matrix ``R[i,j]``,
  the per-tier memory footprint, and the eviction/retrieval counts — recorded in
  the manifest's ``deterministic_probe`` and ``results``. (LLM token counts /
  wall-clock are the only flagged-nondeterministic fields and are not compared.)
* **EC1 shape.** The written manifest carries a populated ``R[i,j]`` *and* cost
  telemetry, and round-trips through ``read_manifest``.
* **FR1.6 confound.** No object reachable from the agent's live memory (episodic,
  semantic, archival) — nor the stream's online view — can reach a ground-truth
  relevance label.

All runs use the deterministic hash embedder + mock LLM (``force_mock_llm``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slow_wave.agent.runner import run_agent
from slow_wave.agent.wake import WakeAgent
from slow_wave.config import load_config
from slow_wave.embeddings import get_embedder
from slow_wave.repro.manifest import read_manifest
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.guard import assert_no_label_leak, online_view
from slow_wave.stream.probes import build_probe_set


@pytest.fixture
def smoke_cfg(repo_root: Path):
    return load_config(repo_root / "configs" / "agent_smoke.yaml")


def test_run_agent_is_deterministic(smoke_cfg, force_mock_llm, tmp_path) -> None:
    """Two runs of the same config+seed produce identical non-LLM outputs."""
    p1 = run_agent(smoke_cfg, out_dir=tmp_path / "run1")
    p2 = run_agent(smoke_cfg, out_dir=tmp_path / "run2")
    m1 = read_manifest(p1)
    m2 = read_manifest(p2)

    # Deterministic probe = R + footprint + eviction/retrieval/item counts.
    assert m1.deterministic_probe == m2.deterministic_probe
    # Results payload's deterministic members match exactly.
    assert m1.results["accuracy_matrix"] == m2.results["accuracy_matrix"]
    assert m1.results["memory_footprint"] == m2.results["memory_footprint"]
    assert m1.results["continual_metrics"] == m2.results["continual_metrics"]


def test_manifest_carries_R_and_cost(smoke_cfg, force_mock_llm, tmp_path) -> None:
    """EC1: the manifest carries a populated R[i,j] plus cost telemetry."""
    path = run_agent(smoke_cfg, out_dir=tmp_path)
    m = read_manifest(path)

    R = m.results["accuracy_matrix"]["R"]
    n_tasks = m.results["accuracy_matrix"]["n_tasks"]
    assert n_tasks == smoke_cfg.stream.n_tasks
    assert len(R) == n_tasks and all(len(row) == n_tasks for row in R)
    assert all(0.0 <= v <= 1.0 for row in R for v in row)

    # Cost telemetry is present and consistent.
    assert m.cost.api_calls >= 1  # agent_smoke uses reasoning_calls: per_task
    assert m.cost.tokens.total == m.cost.tokens.input + m.cost.tokens.output
    assert m.cost.tokens.total > 0
    assert "telemetry" in m.results and "memory_footprint" in m.results


def test_no_label_leak_from_live_memory(smoke_cfg, force_mock_llm) -> None:
    """FR1.6: nothing reachable from live memory or the online view leaks a label."""
    set_global_seeds(smoke_cfg.seed)
    stream = generate_stream(smoke_cfg.stream, derive_seed(smoke_cfg.seed, "stream"))
    probe_set = build_probe_set(stream)
    embedder = get_embedder(smoke_cfg)
    result = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)

    # The online view the agent consumed is label-free.
    for item in online_view(stream):
        assert_no_label_leak(item)

    # Every live memory entry across all three tiers is label-free.
    substrate = result.substrate
    entries = (
        substrate.episodic.all_entries()
        + substrate.semantic.all_entries()
        + substrate.archival.all_entries()
    )
    assert entries  # non-vacuous: there is memory to check
    for entry in entries:
        assert_no_label_leak(entry)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
