"""End-to-end Phase 5 integration test (orchestrator-owned).

Runs the whole Phase 5 pipeline on the smoke config — the grid + length sweep +
sim-vs-real (WS1), the analysis verdicts (WS2), and the figures (WS3) — in one
temp directory and asserts that **every Phase 5 exit criterion** leaves a
well-formed artifact:

* **EC1** arm × distractor-regime × seed grid completes; every cell has a
  manifest with cost + git hash; coverage logged.
* **EC2** ≥ the configured seeds per arm; a realized-power report is produced.
* **EC3** a real long-horizon vs accelerated sim run per key arm + a sim-vs-real
  agreement figure/number.
* **EC4** every figure regenerates from the committed result with a caption
  stating *n* and the CI method.
* **EC5** the preregistered primary endpoint is computed with an unambiguous
  verdict.
* **EC6** the long-context crossover is reported (found or absence stated).
* **EC7** a negative-result-form mapping is present (and the secondary contrast
  is always populated).

This is the smoke-scale check (fast, deterministic, mock LLM); the science-scale
8-seed grid is ``configs/phase5_full.yaml`` and its committed artifacts live
under ``phase5/`` + ``paper/``. The figure assertions are guarded by
``importorskip("matplotlib")`` so the suite stays green in CI (which has no
matplotlib).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from slow_wave.config import load_config
from slow_wave.eval.analysis import analyze, write_analysis
from slow_wave.eval.grid import run_phase5
from slow_wave.eval.phase5_schema import Phase5Result
from slow_wave.repro.manifest import read_manifest

_SMOKE_CONFIG = "configs/phase5_smoke.yaml"
_REGIMES = {"signal_rich", "balanced", "distractor_heavy"}
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


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    """Run grid → analysis once into a temp dir; return the artifacts.

    Module-scoped so the (relatively expensive) smoke pipeline runs a single time
    for every assertion below.
    """
    out = tmp_path_factory.mktemp("phase5_e2e")
    cfg = load_config(_SMOKE_CONFIG)
    result_path = run_phase5(cfg, out_dir=out)
    result = Phase5Result.model_validate_json(Path(result_path).read_text("utf-8"))
    report = analyze(result, stats_seed=0)
    analysis_json, results_md = write_analysis(result, report, out)
    return {
        "out": out,
        "result_path": Path(result_path),
        "result": result,
        "report": report,
        "analysis_json": analysis_json,
        "results_md": results_md,
    }


def test_ec1_regime_grid_with_manifests(pipeline):
    """EC1: arm × regime × seed grid ran; every cell has a manifest w/ cost+git."""
    result: Phase5Result = pipeline["result"]
    grid = result.grid
    assert {c.regime.name for c in grid.regimes} == _REGIMES
    assert grid.primary_regime == "distractor_heavy"
    seeds = result.grid.seeds
    assert len(seeds) >= 3  # smoke floor
    for cell in grid.regimes:
        assert set(cell.acc_by_arm) == _NINE_ARMS
        for arm, accs in cell.acc_by_arm.items():
            assert len(accs) == len(seeds), arm
    # Every per-cell manifest exists and carries cost + git provenance.
    assert result.manifest_paths
    for rel in result.manifest_paths:
        # run_phase5 records the path as written: absolute under a tmp out-dir.
        p = Path(rel)
        assert p.exists(), f"missing manifest {rel}"
        man = read_manifest(p)
        assert man.cost.tokens.total >= 0
        assert man.git is not None  # commit may be None on a shallow checkout
    # Coverage logged (DX2: no silent caps).
    assert any("dropped" in n.lower() or "ran" in n.lower() for n in grid.coverage_notes)


def test_ec2_power_report(pipeline):
    """EC2: realized-power report exists; n_seeds matches the grid; floor known."""
    power = pipeline["report"].power
    assert power.n_seeds == len(pipeline["result"].grid.seeds)
    assert power.floor == 5
    assert power.required_n_for_observed >= 1
    # The boolean flags must be CONSISTENT with the realized numbers, not constants
    # (a hardcoded True/False would diverge from these relations).
    assert power.floor_met == (power.n_seeds >= power.floor)
    assert power.powered_for_observed == (power.n_seeds >= power.required_n_for_observed)
    assert isinstance(power.note, str) and power.note


def test_ec3_sim_vs_real(pipeline):
    """EC3: a real long-horizon vs accelerated sim run per key arm + agreement."""
    sr = pipeline["result"].sim_real
    assert sr.real_n_tasks > sr.sim_n_tasks  # long-horizon is longer than sim
    assert sr.arms, "no sim-vs-real arms"
    for arm in sr.arms:
        assert len(arm.acc_sim_per_seed) == len(sr.seeds)
        assert len(arm.acc_real_per_seed) == len(sr.seeds)
        assert arm.retention_real.n_tasks == sr.real_n_tasks
        assert arm.retention_sim.n_tasks == sr.sim_n_tasks
    assert -1.0 <= sr.spearman_agreement <= 1.0
    assert -1.0 <= sr.pearson_agreement <= 1.0
    assert sr.note
    assert pipeline["report"].sim_real_agreement_note


def test_ec5_primary_endpoint_verdict(pipeline):
    """EC5: the preregistered primary endpoint is computed with a clear verdict."""
    rep = pipeline["report"]
    assert rep.primary_endpoint_name == "acc_diff_full_dream_vs_no_sleep"
    assert rep.primary_verdict in {
        "confirmed",
        "refuted",
        "inconclusive",
        "not-computed",
    }
    # On the distractor-heavy smoke stream the mechanism confirms; the CI + test
    # + effect are all present regardless of sign.
    if rep.primary_verdict != "not-computed":
        assert rep.primary_value is not None
        assert rep.primary_ci_lo is not None and rep.primary_ci_hi is not None
        assert rep.primary_ci_method  # DX4: the caption needs a method string
        assert rep.primary_effect_name == "cohens_d"
    # The verdict must be CONSISTENT with the numbers (catches a hardcoded/flipped
    # verdict constant on real grid output): 'confirmed' requires a positive effect
    # clearing the noise floor with a CI excluding zero (the prereg gate).
    if rep.primary_verdict == "confirmed":
        assert rep.primary_value is not None and rep.primary_value > 0.0
        assert rep.primary_ci_lo is not None and rep.primary_ci_lo > 0.0
        assert rep.exceeds_noise_floor is True


def test_ec6_crossover_reported(pipeline):
    """EC6: the long-context crossover is reported (found or absence stated)."""
    cx = pipeline["report"].crossover
    assert cx.lengths, "no swept lengths"
    assert len(cx.acc_per_token_treatment) == len(cx.lengths)
    assert len(cx.acc_per_token_baseline) == len(cx.lengths)
    # Either a crossover length is found, or the note explicitly states absence.
    if cx.crossover_found:
        assert cx.crossover_length in cx.lengths
    else:
        assert cx.crossover_length is None
        assert "no" in cx.note.lower() or "absence" in cx.note.lower()
    assert cx.note


def test_ec7_negative_form_mapping(pipeline):
    """EC7: a negative-result-form mapping + the always-on secondary contrast."""
    neg = pipeline["report"].negative
    assert isinstance(neg.applicable, bool)
    # The registered secondary contrast (dreaming vs replay-only) is ALWAYS
    # computed, regardless of the primary verdict.
    assert "full_dream vs replay_only" in neg.secondary_contrasts
    assert neg.secondary_contrasts["full_dream vs replay_only"]
    assert neg.note
    if neg.applicable:
        assert neg.matched_forms
        assert neg.regime_tie
    elif neg.matched_forms:
        # A registered negative PATTERN is observed but does not overturn H1; the
        # note must surface it honestly rather than claim nothing matched (DX2).
        assert "no negative-result form applies" not in neg.note


def test_tmr_targeting_effect(pipeline):
    """FR5.3: the TMR-style replay-targeting effect is computed vs the benchmark."""
    tmr = pipeline["report"].tmr
    assert tmr.benchmark_g == pytest.approx(0.29)
    assert tmr.replay_arms and tmr.no_replay_arms
    assert tmr.signal_retention_replay and tmr.signal_retention_no_replay
    assert isinstance(tmr.exceeds_benchmark, bool)
    assert "analogue" in tmr.note.lower() or "proxy" in tmr.note.lower()


def test_results_md_carries_mock_caveat(pipeline):
    """DX5: the written RESULTS.md opens with the mock-LLM caveat + every section."""
    md = pipeline["results_md"].read_text("utf-8").lower()
    assert "mock" in md and "not a scientific claim" in md
    for token in ("primary endpoint", "crossover", "tmr", "power"):
        assert token in md, token
    # analysis.json round-trips.
    data = json.loads(pipeline["analysis_json"].read_text("utf-8"))
    assert data["primary_endpoint_name"] == "acc_diff_full_dream_vs_no_sleep"


def test_ec4_figures_regenerate_with_captions(pipeline):
    """EC4: every figure regenerates from the committed result; captions w/ n+CI."""
    pytest.importorskip("matplotlib")
    from slow_wave.paper.figures import FIGURES, _seed_count, generate_all_figures

    fig_dir = pipeline["out"] / "figures"
    written = generate_all_figures(pipeline["result_path"], fig_dir)
    assert written
    manifest = json.loads((fig_dir / "figures_manifest.json").read_text("utf-8"))
    assert set(manifest) == set(FIGURES)
    assert len(FIGURES) == 7
    for key, entry in manifest.items():
        pdf = fig_dir / entry["pdf"]
        assert pdf.exists() and pdf.stat().st_size > 0, key
        cap = entry["caption"].lower()
        # The 'n' is not just present — it equals the real seed count behind the
        # figure (catches a mislabeled 'n=' or an 'n=0' empty-data figure; DX4).
        m = re.search(r"n=(\d+) seeds", cap)
        assert m, f"{key}: caption missing 'n=<int> seeds' ({entry['caption']!r})"
        assert int(m.group(1)) == _seed_count(pipeline["result"], key), key
        assert "ci" in cap, key  # states the CI method (DX4)
        assert "mock" in cap, key  # DX5 caveat on every figure
