"""Tests for slow_wave.paper.figures (Phase 5, WS3, EC4, DX4/DX5).

These pin the figures layer's contract:

* the module imports cleanly **without matplotlib** and exposes exactly the seven
  deliverable figure keys (the matplotlib/CI rule — these tests run in CI where
  matplotlib is absent);
* every matplotlib-touching test is guarded by ``pytest.importorskip``;
* :func:`generate_all_figures` renders seven non-empty vector PDFs plus a
  ``figures_manifest.json`` whose every caption is non-empty and states n (seeds),
  the CI method, and the mock-LLM caveat;
* each individual ``figure_*`` returns an existing non-empty path;
* a :class:`Phase5Result` with ``analysis=None`` raises a clear ``ValueError``.

The synthetic :class:`Phase5Result` fixture is built **by hand** via the
``phase5_schema`` constructors (no dependence on the parallel WS1/WS2 modules),
including a fully populated :class:`AnalysisReport` and realistic per-seed arrays.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

# The module MUST import without matplotlib (lazy import rule); import it at the
# top level so the "imports cleanly + 7 keys" tests run even where matplotlib is
# absent (CI). Rendering tests below guard with pytest.importorskip.
from slow_wave.paper import figures
from slow_wave.eval.phase5_schema import (
    AnalysisReport,
    CrossoverResult,
    GridResult,
    LengthPoint,
    LengthSweep,
    NegativeFormMapping,
    Phase5Result,
    PowerReport,
    RegimeCell,
    RegimeMix,
    RetentionCurve,
    SimRealArm,
    SimRealResult,
    TMRResult,
)

SEEDS = [0, 1, 2]
ARMS = ["no_sleep", "replay_only", "downscale_only", "full_dream", "reflection", "long_context"]
PRIMARY_REGIME = "distractor_heavy"
MOCK_CAVEAT = (
    "MOCK-LLM CAVEAT: every number is a mechanism demonstration in the synthetic "
    "+ deterministic mock-LLM regime, NOT a scientific claim about a real Claude model."
)


# --------------------------------------------------------------------------- #
# Synthetic fixture builders (by hand)
# --------------------------------------------------------------------------- #
def _s3(a: float, b: float, c: float) -> list[float]:
    """A three-seed per-seed list."""
    return [a, b, c]


def _retention_curve(arm: str, base: float) -> RetentionCurve:
    """A 3-task, 3-seed retention curve descending from ``base`` (forgetting)."""
    rows = [
        [round(base - 0.10 + 0.01 * s, 4), round(base - 0.02 + 0.01 * s, 4),
         round(base + 0.05 + 0.01 * s, 4)]
        for s in range(3)
    ]
    return RetentionCurve(arm_name=arm, n_tasks=3, final_row_per_seed=rows, seeds=SEEDS)


def _regime_cell(name: str, mix: RegimeMix, manifest: str, *, primary: bool) -> RegimeCell:
    """One regime's nine-arm summary cell with realistic per-arm per-seed arrays."""
    acc = {
        "no_sleep": _s3(0.50, 0.52, 0.48),
        "replay_only": _s3(0.60, 0.62, 0.58),
        "downscale_only": _s3(0.58, 0.55, 0.57),
        "full_dream": _s3(0.72, 0.70, 0.74),
        "reflection": _s3(0.66, 0.64, 0.68),
        "long_context": _s3(0.80, 0.82, 0.79),
    }
    precision = {
        "no_sleep": _s3(0.0, 0.0, 0.0),
        "replay_only": _s3(0.70, 0.72, 0.68),
        "downscale_only": _s3(0.74, 0.71, 0.76),
        "full_dream": _s3(0.85, 0.83, 0.87),
        "reflection": _s3(0.78, 0.80, 0.76),
        "long_context": _s3(0.0, 0.0, 0.0),
    }
    recall = {
        "no_sleep": _s3(0.0, 0.0, 0.0),
        "replay_only": _s3(0.60, 0.58, 0.62),
        "downscale_only": _s3(0.66, 0.64, 0.68),
        "full_dream": _s3(0.80, 0.78, 0.82),
        "reflection": _s3(0.70, 0.72, 0.68),
        "long_context": _s3(0.0, 0.0, 0.0),
    }
    f1 = {
        arm: [
            round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
            for p, r in zip(precision[arm], recall[arm])
        ]
        for arm in ARMS
    }
    signal_retention = {
        "no_sleep": _s3(0.55, 0.57, 0.53),
        "replay_only": _s3(0.82, 0.80, 0.84),
        "downscale_only": _s3(0.60, 0.62, 0.58),
        "full_dream": _s3(0.90, 0.88, 0.92),
        "reflection": _s3(0.85, 0.83, 0.87),
        "long_context": _s3(0.98, 0.99, 0.97),
    }
    tokens = {
        "no_sleep": _s3(1000.0, 1010.0, 990.0),
        "replay_only": _s3(1400.0, 1420.0, 1380.0),
        "downscale_only": _s3(1300.0, 1290.0, 1310.0),
        "full_dream": _s3(1800.0, 1820.0, 1780.0),
        "reflection": _s3(1600.0, 1590.0, 1610.0),
        "long_context": _s3(5200.0, 5300.0, 5100.0),
    }
    vectors = {
        "no_sleep": _s3(40.0, 41.0, 39.0),
        "replay_only": _s3(55.0, 54.0, 56.0),
        "downscale_only": _s3(45.0, 46.0, 44.0),
        "full_dream": _s3(60.0, 61.0, 59.0),
        "reflection": _s3(58.0, 57.0, 59.0),
        "long_context": _s3(300.0, 305.0, 295.0),
    }
    return RegimeCell(
        regime=mix,
        manifest_path=manifest,
        arms=ARMS,
        seeds=SEEDS,
        acc_by_arm=acc,
        prune_precision_by_arm=precision,
        prune_recall_by_arm=recall,
        prune_f1_by_arm=f1,
        signal_retention_by_arm=signal_retention,
        total_tokens_by_arm=tokens,
        memory_vectors_by_arm=vectors,
        primary_value=0.21 if primary else 0.12,
        primary_verdict="confirmed" if primary else "inconclusive",
        primary_ci_lo=0.10 if primary else -0.01,
        primary_ci_hi=0.32 if primary else 0.25,
        aa_abs_difference=0.02,
        aa_significant=False,
        budget_matched=True,
    )


def _length_sweep() -> LengthSweep:
    """A 3-length sweep over full_dream / long_context / no_sleep, ascending in L."""
    key_arms = ["full_dream", "long_context", "no_sleep"]
    points = []
    for L in (2, 3, 5):
        acc = {
            "full_dream": _s3(0.62 + 0.03 * L, 0.60 + 0.03 * L, 0.64 + 0.03 * L),
            "long_context": _s3(0.80, 0.82, 0.79),
            "no_sleep": _s3(0.50 - 0.02 * L, 0.52 - 0.02 * L, 0.48 - 0.02 * L),
        }
        tokens = {
            "full_dream": _s3(900.0 * L, 905.0 * L, 895.0 * L),
            "long_context": _s3(2000.0 * L * L, 2010.0 * L * L, 1990.0 * L * L),
            "no_sleep": _s3(700.0 * L, 705.0 * L, 695.0 * L),
        }
        vectors = {
            "full_dream": _s3(60.0, 61.0, 59.0),
            "long_context": _s3(100.0 * L, 101.0 * L, 99.0 * L),
            "no_sleep": _s3(40.0, 41.0, 39.0),
        }
        points.append(
            LengthPoint(
                n_tasks=L, items_per_task=4, arms=key_arms, seeds=SEEDS,
                acc_by_arm=acc, total_tokens_by_arm=tokens, memory_vectors_by_arm=vectors,
            )
        )
    return LengthSweep(
        treatment_arm="full_dream", baseline_arm="long_context", points=points,
        coverage_notes=["ran 3 lengths x 3 arms x 3 seeds = 27 cells; 0 dropped"],
    )


def _sim_real() -> SimRealResult:
    """Per-arm sim vs real comparison with high agreement."""
    arms = [
        SimRealArm(
            arm_name="full_dream",
            acc_sim_per_seed=_s3(0.72, 0.70, 0.74),
            acc_real_per_seed=_s3(0.70, 0.69, 0.72),
            retention_sim=_retention_curve("full_dream", 0.72),
            retention_real=_retention_curve("full_dream", 0.70),
        ),
        SimRealArm(
            arm_name="no_sleep",
            acc_sim_per_seed=_s3(0.50, 0.52, 0.48),
            acc_real_per_seed=_s3(0.48, 0.50, 0.47),
            retention_sim=_retention_curve("no_sleep", 0.50),
            retention_real=_retention_curve("no_sleep", 0.48),
        ),
        SimRealArm(
            arm_name="long_context",
            acc_sim_per_seed=_s3(0.80, 0.82, 0.79),
            acc_real_per_seed=_s3(0.81, 0.80, 0.82),
            retention_sim=_retention_curve("long_context", 0.80),
            retention_real=_retention_curve("long_context", 0.81),
        ),
    ]
    return SimRealResult(
        arms=arms, sim_n_tasks=3, sim_compression=60.0, real_n_tasks=10,
        real_compression=1.0, seeds=SEEDS, pearson_agreement=0.99,
        spearman_agreement=1.0, ranking_preserved=True, inversions=[],
        max_abs_acc_divergence=0.02,
        note="sim predicts real magnitude and ranking; no inversion.",
    )


def _analysis() -> AnalysisReport:
    """A fully populated AnalysisReport (crossover/TMR/power/negative)."""
    crossover = CrossoverResult(
        metric="acc_per_token", treatment_arm="full_dream", baseline_arm="long_context",
        lengths=[2, 3, 5],
        acc_per_token_treatment=[0.0020, 0.0040, 0.0060],
        acc_per_token_baseline=[0.0050, 0.0045, 0.0040],
        acc_gap_baseline_minus_treatment=[0.10, 0.08, 0.05],
        crossover_length=5, crossover_found=True, raw_accuracy_crossover_length=None,
        note="cost-adjusted crossover found at L=5; long_context still leads on raw ACC.",
    )
    tmr = TMRResult(
        replay_arms=["full_dream", "replay_only", "reflection"],
        no_replay_arms=["no_sleep", "downscale_only"],
        signal_retention_replay=[0.90, 0.88, 0.92, 0.82, 0.80, 0.84, 0.85, 0.83, 0.87],
        signal_retention_no_replay=[0.55, 0.57, 0.53, 0.60, 0.62, 0.58],
        mean_lift=0.281, hedges_g=0.45, g_ci_lo=0.12, g_ci_hi=0.81, benchmark_g=0.29,
        exceeds_benchmark=True,
        note="replay-targeting analogue (not a literal cued-TMR protocol); exceeds 0.29.",
    )
    power = PowerReport(
        n_seeds=3, floor=5, floor_met=False, observed_effect_d=1.2,
        required_n_for_observed=8, powered_for_observed=False,
        note="smoke fixture uses 3 seeds; the full grid uses 8 (above the floor of 5).",
    )
    negative = NegativeFormMapping(
        applicable=False, matched_forms=[],
        regime_tie="distractor_heavy (signal=0.34, distractor=0.40, noise=0.26)",
        secondary_contrasts={"full_dream vs replay_only": "full_dream > replay_only"},
        note="primary confirmed; no negative form applies.",
    )
    return AnalysisReport(
        primary_endpoint_name="acc_diff_full_dream_vs_no_sleep",
        primary_verdict="confirmed", primary_value=0.21, primary_ci_lo=0.10,
        primary_ci_hi=0.32, primary_ci_method="percentile bootstrap",
        primary_test_name="wilcoxon_signed_rank", primary_test_p=0.03,
        primary_effect_name="cohens_d", primary_effect_value=1.2,
        primary_effect_magnitude="large", noise_floor=0.02, exceeds_noise_floor=True,
        n_seeds=3, crossover=crossover, tmr=tmr, power=power, negative=negative,
        per_regime_verdicts={"signal_rich": "inconclusive", "distractor_heavy": "confirmed"},
        sim_real_agreement_note="sim and real agree; no inversion (EC3).",
        headline="full_dream beats no_sleep on retention in the distractor-heavy regime.",
        mock_llm_caveat=MOCK_CAVEAT,
    )


def _build_result(*, with_analysis: bool = True) -> Phase5Result:
    """Assemble a complete synthetic Phase5Result."""
    mix_signal = RegimeMix(name="signal_rich", signal=0.70, distractor=0.20, noise=0.10)
    mix_distractor = RegimeMix(name=PRIMARY_REGIME, signal=0.34, distractor=0.40, noise=0.26)
    grid = GridResult(
        regimes=[
            _regime_cell("signal_rich", mix_signal,
                         "phase5/regime_signal_rich/manifest.json", primary=False),
            _regime_cell(PRIMARY_REGIME, mix_distractor,
                         "phase5/regime_distractor_heavy/manifest.json", primary=True),
        ],
        arms=ARMS, seeds=SEEDS, primary_regime=PRIMARY_REGIME,
        coverage_notes=["ran 2 regimes x 6 arms x 3 seeds = 36 cells; 0 dropped"],
    )
    retention = {PRIMARY_REGIME: [_retention_curve(arm, base)
                                  for arm, base in zip(ARMS, [0.50, 0.60, 0.58, 0.72, 0.66, 0.80])]}
    return Phase5Result(
        experiment="phase5_smoke", scenario="domain_incremental",
        git_commit="deadbeef", model_id="claude-opus-4-8", model_mocked=True,
        grid=grid, length_sweep=_length_sweep(), sim_real=_sim_real(),
        retention=retention,
        analysis=_analysis() if with_analysis else None,
        coverage_notes=["phase5 smoke: all sweeps complete; 0 dropped"],
        manifest_paths=[
            "phase5/regime_signal_rich/manifest.json",
            "phase5/regime_distractor_heavy/manifest.json",
        ],
    )


def _write(result: Phase5Result, tmp_path: Path) -> Path:
    """Serialize a Phase5Result to a JSON file and return its path."""
    path = tmp_path / "phase5_result.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Tests that run WITHOUT matplotlib (the CI-safe guarantees)
# --------------------------------------------------------------------------- #
def test_figures_dict_has_seven_keys() -> None:
    """FIGURES exposes exactly the seven deliverable keys (no matplotlib needed)."""
    assert len(figures.FIGURES) == 7
    assert set(figures.FIGURES) == {
        "retention_curves", "ablation_table", "mechanism_pr", "cost_pareto",
        "long_context_crossover", "tmr_targeting", "sim_vs_real",
    }
    # Every value is a distinct .pdf filename.
    assert all(v.endswith(".pdf") for v in figures.FIGURES.values())
    assert len(set(figures.FIGURES.values())) == 7


def test_module_has_no_toplevel_matplotlib_import() -> None:
    """The module must not import matplotlib at module scope (CI has none)."""
    source = Path(figures.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "matplotlib", (
                    "matplotlib must be imported lazily inside functions, not at top level"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root != "matplotlib", (
                "matplotlib must be imported lazily inside functions, not at top level"
            )


def test_analysis_none_raises_value_error(tmp_path: Path) -> None:
    """A Phase5Result with analysis=None raises a clear ValueError (no rendering)."""
    path = _write(_build_result(with_analysis=False), tmp_path)
    with pytest.raises(ValueError, match="analysis"):
        figures.generate_all_figures(path, tmp_path / "figs")


# --------------------------------------------------------------------------- #
# Rendering tests (guarded — matplotlib absent in CI)
# --------------------------------------------------------------------------- #
def test_generate_all_figures_writes_pdfs_and_manifest(tmp_path: Path) -> None:
    """generate_all_figures writes 7 non-empty PDFs + a well-formed manifest."""
    pytest.importorskip("matplotlib")
    result_path = _write(_build_result(), tmp_path)
    out_dir = tmp_path / "figs"
    written = figures.generate_all_figures(result_path, out_dir)

    # Seven PDFs, each non-empty.
    pdfs = [p for p in written if p.suffix == ".pdf"]
    assert len(pdfs) == 7
    for pdf in pdfs:
        assert pdf.exists() and pdf.stat().st_size > 0

    # Manifest present and well-formed.
    manifest_path = out_dir / "figures_manifest.json"
    assert manifest_path in written
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == set(figures.FIGURES)
    for key, entry in manifest.items():
        assert entry["pdf"].endswith(".pdf")
        assert entry["png"].endswith(".png")
        caption = entry["caption"]
        assert caption, f"empty caption for {key}"
        assert "n=" in caption, f"caption for {key} lacks an n= token"
        assert "CI" in caption, f"caption for {key} lacks a CI-method token"
        assert "MOCK-LLM CAVEAT" in caption, f"caption for {key} lacks the mock caveat"
        # The referenced files actually exist and are non-empty.
        assert (out_dir / entry["pdf"]).stat().st_size > 0
        assert (out_dir / entry["png"]).stat().st_size > 0


def test_each_individual_figure_returns_nonempty_path(tmp_path: Path) -> None:
    """Each figure_* returns an existing, non-empty PDF path."""
    pytest.importorskip("matplotlib")
    result = _build_result()
    out_dir = tmp_path / "figs"
    fns = [
        figures.figure_retention_curves,
        figures.figure_ablation_table,
        figures.figure_mechanism_pr,
        figures.figure_cost_pareto,
        figures.figure_long_context_crossover,
        figures.figure_tmr_targeting,
        figures.figure_sim_vs_real,
    ]
    assert len(fns) == 7
    for fn in fns:
        path = fn(result, out_dir)
        assert isinstance(path, Path)
        assert path.suffix == ".pdf"
        assert path.exists() and path.stat().st_size > 0
        # The PNG sibling is also written.
        assert path.with_suffix(".png").stat().st_size > 0


def test_no_crossover_branch_renders(tmp_path: Path) -> None:
    """The crossover figure also renders when no crossover was found (EC6 absence)."""
    pytest.importorskip("matplotlib")
    result = _build_result()
    result.analysis.crossover.crossover_found = False
    result.analysis.crossover.crossover_length = None
    out_dir = tmp_path / "figs"
    path = figures.figure_long_context_crossover(result, out_dir)
    assert path.exists() and path.stat().st_size > 0
    caption = figures._build_caption(result, "long_context_crossover")
    assert "no cost-adjusted crossover" in caption
