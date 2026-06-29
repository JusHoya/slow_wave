"""Phase 2 exit criterion #6 — the no-sleep baseline demonstrably forgets.

This is the cross-module integration check a red-team inspects for EC6: on a
*noisy* stream with a capacity-bounded episodic store and **no dream cycle**
(no consolidation, no semantic transfer, no signal protection), the wake agent
must measurably forget earlier tasks. We assert a strictly negative backward
transfer (BWT < 0) and positive average forgetting computed from the agent's own
``R[i,j]``, and tie the forgetting to the mechanism: evicted episodic entries are
demoted to (and recoverable from) the auditable archival tier (EC4), not deleted.

The whole run uses the deterministic hash embedder + mock LLM, so it is offline,
fast, and reproducible (``force_mock_llm`` guarantees no real API call).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slow_wave.agent.wake import WakeAgent
from slow_wave.config import load_config
from slow_wave.embeddings import get_embedder
from slow_wave.memory.schema import MemoryTier
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set


@pytest.fixture
def forgetting_cfg(repo_root: Path):
    """The bounded, noisy Phase 2 forgetting config."""
    return load_config(repo_root / "configs" / "agent_forgetting.yaml")


def _run(cfg):
    """Run the no-sleep wake agent end-to-end on ``cfg`` and return WakeResult."""
    set_global_seeds(cfg.seed)
    stream = generate_stream(cfg.stream, derive_seed(cfg.seed, "stream"))
    probe_set = build_probe_set(stream)
    embedder = get_embedder(cfg)
    return WakeAgent(cfg, embedder).run(stream, probe_set), stream


def test_baseline_forgets_negative_bwt(forgetting_cfg, force_mock_llm) -> None:
    """EC6: BWT is measurably negative and average forgetting is positive."""
    result, _ = _run(forgetting_cfg)
    metrics = result.metrics

    # The phenomenon under study exists: the baseline forgot earlier tasks.
    assert metrics.bwt < -0.05, f"expected clearly negative BWT, got {metrics.bwt}"
    assert metrics.average_forgetting > 0.0, (
        f"expected positive average forgetting, got {metrics.average_forgetting}"
    )
    # Per-task forgetting is non-empty and at least one task was forgotten.
    assert metrics.per_task_forgetting
    assert any(f > 0.0 for f in metrics.per_task_forgetting)


def test_forgetting_visible_in_accuracy_matrix(forgetting_cfg, force_mock_llm) -> None:
    """EC6: the current task is learned (R[t][t]=1) but the first task decays."""
    result, _ = _run(forgetting_cfg)
    R = result.accuracy_matrix.R
    n = result.accuracy_matrix.n_tasks
    assert n >= 2

    # Each task is mastered when it is the current task (capacity > items_per_task).
    for t in range(n):
        assert R[t][t] == pytest.approx(1.0), f"R[{t}][{t}]={R[t][t]} (current task not learned)"

    # By the final cutoff, the first task has been forgotten (evicted).
    assert R[n - 1][0] < R[0][0], "first task accuracy did not decay by the last task"
    assert R[n - 1][0] == pytest.approx(0.0), (
        f"expected the first task fully forgotten, got R[{n-1}][0]={R[n-1][0]}"
    )


def test_forgetting_is_demotion_not_deletion(forgetting_cfg, force_mock_llm) -> None:
    """EC4×EC6: evictions demote to a recoverable archival tier (no hard delete)."""
    result, _ = _run(forgetting_cfg)
    substrate = result.substrate

    # Forgetting happened via eviction, and evictions were demoted, not dropped.
    assert result.telemetry.n_evicted > 0
    archived = substrate.archival.all_entries()
    assert archived, "expected demoted entries in the archival tier"

    # A demoted fact-bearing entry is fully recoverable with its content intact.
    fact_archived = [e for e in archived if e.fact is not None]
    assert fact_archived, "expected at least one demoted fact-bearing entry"
    sample = fact_archived[0]
    recovered = substrate.archival.recover(sample.entry_id)
    assert recovered is not None
    assert recovered.tier is MemoryTier.ARCHIVAL
    assert recovered.content == sample.content
    assert recovered.fact == sample.fact
    assert recovered.provenance == sample.provenance


def test_gating_no_semantic_writes_in_baseline(forgetting_cfg, force_mock_llm) -> None:
    """FR3.1 gating: a no-sleep run writes nothing to the SEMANTIC store."""
    result, _ = _run(forgetting_cfg)
    assert len(result.substrate.semantic.all_entries()) == 0
    assert result.footprint.semantic.n_entries == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
