"""Tests for the Phase 5 sweep runner (WS1, ``slow_wave.eval.grid``).

Exercises the three across-cell sweeps + the assembled committed artifact on the
tiny, fully-deterministic ``configs/phase5_smoke.yaml`` grid (mock LLM + hash
embeddings), asserting the WS1 contract in ``docs/PHASE5_CONTRACT.md``:

* :func:`run_regime_sweep` returns a :class:`GridResult` with the three regimes,
  every :class:`RegimeCell` keyed by all nine arms with one entry per seed, the
  full preregistered primary endpoint embedded + mirrored, every per-cell
  manifest on disk + round-tripping as a :class:`Manifest`, and a DX2 coverage
  note stating cells run and 0 dropped;
* :func:`run_length_sweep` returns a :class:`LengthSweep` ascending in L with the
  unbounded ``long_context`` arm's memory non-decreasing in L while ``full_dream``
  stays bounded (grows far slower than the keep-everything ceiling);
* :func:`run_sim_vs_real` returns a :class:`SimRealResult` with one arm per key
  arm, right-length retention curves, finite agreement scalars in ``[-1, 1]`` and
  a non-empty note;
* :func:`run_phase5` writes a ``phase5_result.json`` that round-trips as a
  :class:`Phase5Result` (``analysis is None``, ``model_mocked`` True, non-empty
  ``manifest_paths`` that all exist, ``git_commit`` set), and two calls on the
  smoke config produce a **byte-identical** file (DX1 determinism).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from slow_wave.config import load_config
from slow_wave.eval.grid import (
    run_length_sweep,
    run_phase5,
    run_regime_sweep,
    run_sim_vs_real,
)
from slow_wave.eval.phase5_schema import (
    GridResult,
    LengthSweep,
    Phase5Result,
    SimRealResult,
)
from slow_wave.repro.manifest import Manifest, read_manifest

_SMOKE_CONFIG = "configs/phase5_smoke.yaml"
_NINE_ARMS = {
    "no_sleep",
    "replay_only",
    "downscale_only",
    "random_pruning",
    "full_dream",
    "reflection",
    "oracle",
    "long_context",
    "aa",
}


# --------------------------------------------------------------------------- #
# Shared fixtures: run the full smoke pipeline ONCE and share its artifacts.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def smoke_phase5(tmp_path_factory):
    """Run ``run_phase5`` once on the smoke config; share the result + artifacts."""
    cfg = load_config(_SMOKE_CONFIG)
    out = tmp_path_factory.mktemp("phase5_run")
    path = run_phase5(cfg, out)
    result = Phase5Result.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return {"cfg": cfg, "out": out, "path": path, "result": result}


# --------------------------------------------------------------------------- #
# run_regime_sweep — the arm x regime x seed grid (EC1)
# --------------------------------------------------------------------------- #
def test_regime_sweep_shape_and_arms(smoke_phase5) -> None:
    """Three regimes, every cell keyed by all nine arms with one entry per seed."""
    grid: GridResult = smoke_phase5["result"].grid
    assert isinstance(grid, GridResult)
    assert {c.regime.name for c in grid.regimes} == {
        "signal_rich",
        "balanced",
        "distractor_heavy",
    }
    assert grid.primary_regime == "distractor_heavy"
    assert set(grid.arms) == _NINE_ARMS

    n_seeds = len(grid.seeds)
    assert n_seeds == 3
    for cell in grid.regimes:
        for mapping in (
            cell.acc_by_arm,
            cell.prune_precision_by_arm,
            cell.prune_recall_by_arm,
            cell.prune_f1_by_arm,
            cell.signal_retention_by_arm,
            cell.total_tokens_by_arm,
            cell.memory_vectors_by_arm,
        ):
            assert set(mapping) == _NINE_ARMS
            assert all(len(v) == n_seeds for v in mapping.values())


def test_regime_sweep_embeds_full_primary_endpoint(smoke_phase5) -> None:
    """Each cell embeds the FULL primary-endpoint object + mirrored scalars."""
    grid: GridResult = smoke_phase5["result"].grid
    for cell in grid.regimes:
        pe = cell.primary_endpoint
        assert pe is not None
        assert pe.name == "acc_diff_full_dream_vs_no_sleep"
        assert pe.treatment_arm == "full_dream" and pe.baseline_arm == "no_sleep"
        # The mirrored scalar fields equal the embedded object's fields.
        assert cell.primary_value == pytest.approx(pe.value)
        assert cell.primary_verdict == pe.verdict
        assert cell.primary_ci_lo == pytest.approx(pe.difference_ci.lo)
        assert cell.primary_ci_hi == pytest.approx(pe.difference_ci.hi)
        # The A/A floor + budget verdict are surfaced.
        assert cell.aa_abs_difference >= 0.0
        assert isinstance(cell.aa_significant, bool)
        assert isinstance(cell.budget_matched, bool)


def test_regime_sweep_manifests_exist_and_round_trip(smoke_phase5) -> None:
    """Every per-cell manifest exists on disk and round-trips as a Manifest."""
    grid: GridResult = smoke_phase5["result"].grid
    for cell in grid.regimes:
        from pathlib import Path

        assert Path(cell.manifest_path).exists()
        manifest = read_manifest(cell.manifest_path)
        assert isinstance(manifest, Manifest)
        # EC1: every run's manifest carries cost + git provenance.
        assert manifest.cost.api_calls >= 0
        assert "commit" in manifest.git.model_dump()
        exp = manifest.results["experiment"]
        assert set(exp["arms_run"]) == _NINE_ARMS


def test_regime_sweep_coverage_note_no_drop(smoke_phase5) -> None:
    """DX2: coverage notes state the cell count and that 0 were dropped."""
    grid: GridResult = smoke_phase5["result"].grid
    joined = " ".join(grid.coverage_notes)
    assert "0 dropped" in joined
    # 3 regimes x 9 arms x 3 seeds = 81 cells, stated explicitly.
    assert "81 cells" in joined


def test_run_regime_sweep_standalone(tmp_path) -> None:
    """The standalone runner returns a GridResult and writes a real manifest."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.seeds = [0, 1]  # trim for speed; shape is seed-count independent
    grid = run_regime_sweep(cfg, tmp_path)
    assert isinstance(grid, GridResult)
    assert len(grid.regimes) == 3
    for cell in grid.regimes:
        assert all(len(v) == 2 for v in cell.acc_by_arm.values())
        assert isinstance(read_manifest(cell.manifest_path), Manifest)


# --------------------------------------------------------------------------- #
# run_length_sweep — the long-context crossover data (EC6)
# --------------------------------------------------------------------------- #
def test_length_sweep_ascending_and_per_seed(smoke_phase5) -> None:
    """Points ascend in L; each key arm has one ACC/token entry per seed."""
    sweep: LengthSweep = smoke_phase5["result"].length_sweep
    assert isinstance(sweep, LengthSweep)
    lengths = [p.n_tasks for p in sweep.points]
    assert lengths == sorted(lengths)
    assert lengths == [2, 3, 4]
    for point in sweep.points:
        for arm in ("full_dream", "long_context", "no_sleep"):
            assert arm in point.acc_by_arm
            assert len(point.acc_by_arm[arm]) == len(point.seeds)
            assert len(point.total_tokens_by_arm[arm]) == len(point.seeds)
            assert len(point.memory_vectors_by_arm[arm]) == len(point.seeds)


def test_length_sweep_long_context_monotone_full_dream_bounded(smoke_phase5) -> None:
    """``long_context`` memory is non-decreasing in L; ``full_dream`` stays bounded.

    ``long_context`` (episodic capacity 0) never evicts, so its memory grows with
    every additional task — monotone non-decreasing in L for each seed.
    ``full_dream`` consolidates under a bounded episodic capacity, so its memory
    grows far slower than the keep-everything ceiling (bounded relative to it).
    """
    sweep: LengthSweep = smoke_phase5["result"].length_sweep
    n_seeds = len(sweep.points[0].seeds)

    # long_context per-seed memory is non-decreasing in L.
    for s in range(n_seeds):
        lc = [p.memory_vectors_by_arm["long_context"][s] for p in sweep.points]
        assert all(b >= a for a, b in zip(lc, lc[1:])), lc

    lc_mean = [
        float(np.mean(p.memory_vectors_by_arm["long_context"])) for p in sweep.points
    ]
    fd_mean = [
        float(np.mean(p.memory_vectors_by_arm["full_dream"])) for p in sweep.points
    ]
    # full_dream grows strictly slower than the unbounded ceiling (bounded).
    assert (fd_mean[-1] - fd_mean[0]) < (lc_mean[-1] - lc_mean[0])
    # ... and at the largest L it is no larger than the keep-everything arm.
    assert fd_mean[-1] <= lc_mean[-1] + 1e-9


def test_run_length_sweep_default_key_arms(tmp_path) -> None:
    """Default key arms are the crossover treatment/ceiling/baseline trio."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.seeds = [0, 1]
    sweep = run_length_sweep(cfg, tmp_path, lengths=[2, 3])
    assert sweep.treatment_arm == "full_dream"
    assert sweep.baseline_arm == "long_context"
    assert [p.n_tasks for p in sweep.points] == [2, 3]
    for point in sweep.points:
        assert set(point.arms) == {"full_dream", "long_context", "no_sleep"}


# --------------------------------------------------------------------------- #
# run_sim_vs_real — accelerated-sim vs real-long-horizon agreement (EC3)
# --------------------------------------------------------------------------- #
def test_sim_vs_real_arms_curves_and_agreement(smoke_phase5) -> None:
    """One arm per key arm; right-length curves; finite agreement in [-1, 1]."""
    sr: SimRealResult = smoke_phase5["result"].sim_real
    assert isinstance(sr, SimRealResult)
    assert {a.arm_name for a in sr.arms} == {"full_dream", "no_sleep", "long_context"}
    assert sr.sim_n_tasks == 3 and sr.real_n_tasks == 6
    n_seeds = len(sr.seeds)
    for arm in sr.arms:
        assert len(arm.acc_sim_per_seed) == n_seeds
        assert len(arm.acc_real_per_seed) == n_seeds
        # Retention curves: one final-row per seed, of the stream's length.
        assert len(arm.retention_sim.final_row_per_seed) == n_seeds
        assert len(arm.retention_real.final_row_per_seed) == n_seeds
        assert all(len(row) == sr.sim_n_tasks for row in arm.retention_sim.final_row_per_seed)
        assert all(len(row) == sr.real_n_tasks for row in arm.retention_real.final_row_per_seed)

    for scalar in (sr.pearson_agreement, sr.spearman_agreement):
        assert np.isfinite(scalar)
        assert -1.0 <= scalar <= 1.0
    assert isinstance(sr.ranking_preserved, bool)
    assert sr.max_abs_acc_divergence >= 0.0
    assert sr.note  # non-empty honest interpretation


def test_run_sim_vs_real_single_arm_zero_variance_note(tmp_path) -> None:
    """A single key arm yields agreement 1.0 with an honest note (never raises)."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.seeds = [0, 1]
    sr = run_sim_vs_real(
        cfg,
        tmp_path,
        key_arms=["full_dream"],
        sim_n_tasks=2,
        sim_compression=60.0,
        real_n_tasks=3,
        real_compression=1.0,
    )
    assert len(sr.arms) == 1
    assert sr.pearson_agreement == 1.0
    assert sr.spearman_agreement == 1.0
    assert "1.0" in sr.note and sr.note


# --------------------------------------------------------------------------- #
# run_phase5 — the assembled committed artifact + determinism (DX1)
# --------------------------------------------------------------------------- #
def test_phase5_artifact_round_trips(smoke_phase5) -> None:
    """The written phase5_result.json round-trips and is well-formed (WS1)."""
    path = smoke_phase5["path"]
    result: Phase5Result = smoke_phase5["result"]
    assert path.name == "phase5_result.json" and path.exists()
    assert result.analysis is None
    assert result.model_mocked is True
    assert result.git_commit  # provenance set (this is a git checkout)
    assert result.experiment == "phase5-smoke"

    # Every per-cell manifest path is non-empty and on disk.
    assert result.manifest_paths
    from pathlib import Path

    assert all(Path(p).exists() for p in result.manifest_paths)

    # Retention seed-bands for the primary regime: one curve per arm.
    assert result.grid.primary_regime in result.retention
    curves = result.retention[result.grid.primary_regime]
    assert {c.arm_name for c in curves} == _NINE_ARMS
    assert all(c.n_tasks == 3 for c in curves)
    assert all(len(c.final_row_per_seed) == len(result.grid.seeds) for c in curves)


def test_phase5_byte_identical_determinism(tmp_path) -> None:
    """Two run_phase5 calls on the smoke config produce a byte-identical file."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.seeds = [0, 1]  # trim for speed; determinism is seed-count independent
    out = tmp_path / "det"

    p1 = run_phase5(cfg, out)
    bytes_1 = p1.read_bytes()
    p2 = run_phase5(cfg, out)  # same out_dir => same manifest_paths strings
    bytes_2 = p2.read_bytes()

    assert p1 == p2
    assert bytes_1 == bytes_2
    # Sanity: it really is a valid Phase5Result with analysis still None.
    restored = Phase5Result.model_validate(json.loads(bytes_2.decode("utf-8")))
    assert restored.analysis is None
