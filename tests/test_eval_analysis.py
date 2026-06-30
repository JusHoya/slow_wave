"""Tests for slow_wave.eval.analysis (Phase 5, WS2, EC5-EC7, FR5.3, DX3/DX5).

These pin the analysis layer's verdict logic on **hand-built** ``Phase5Result``
fixtures (independent of WS1's grid runner): the preregistered primary-endpoint
verdict is surfaced exactly as the harness computed it and its name is asserted
against the committed preregistration (DX3); the long-context crossover (EC6),
the TMR-style targeting effect (FR5.3), the realized power (EC2), and the
negative-result-form mapping (EC7) are computed correctly; and ``write_analysis``
emits a deterministic ``analysis.json`` + a ``paper/RESULTS.md`` that opens with
the mandatory mock-LLM caveat (DX5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slow_wave.eval.analysis import analyze, run_analysis, write_analysis
from slow_wave.eval.metrics import ContinualMetrics
from slow_wave.eval.phase5_schema import (
    AnalysisReport,
    GridResult,
    LengthPoint,
    LengthSweep,
    Phase5Result,
    RegimeCell,
    RegimeMix,
    SimRealResult,
)
from slow_wave.eval.prereg import (
    NonPreregisteredEndpointError,
    compute_primary_endpoint,
    load_preregistration,
)
from slow_wave.eval.schema import (
    AAResult,
    ArmCost,
    ArmResult,
    BootstrapCI,
    EffectSize,
    Preregistration,
    PrimaryEndpoint,
)
from slow_wave.eval.schema import TestResult as _TestResult  # aliased: avoids pytest collecting it
from slow_wave.memory.schema import MemoryFootprint, MemoryTier, StoreFootprint
from slow_wave.stream.schema import AccuracyMatrix, CLScenario

PRIMARY = "acc_diff_full_dream_vs_no_sleep"
SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Fixtures: the committed preregistration + its path
# --------------------------------------------------------------------------- #
@pytest.fixture
def prereg_path(repo_root: Path) -> str:
    """Path to the committed preregistration artifact (DX3 binding)."""
    return str(repo_root / "prereg" / "preregistration.yaml")


@pytest.fixture
def prereg(prereg_path: str) -> Preregistration:
    """The parsed committed preregistration."""
    return load_preregistration(prereg_path)


# --------------------------------------------------------------------------- #
# Synthetic ArmResult / PrimaryEndpoint builders
# --------------------------------------------------------------------------- #
def _empty_footprint(dim: int = 384) -> MemoryFootprint:
    """A zero footprint with one well-formed StoreFootprint per tier."""
    return MemoryFootprint(
        episodic=StoreFootprint(
            tier=MemoryTier.EPISODIC, n_entries=0, n_vectors=0, dim=dim, bytes=0
        ),
        semantic=StoreFootprint(
            tier=MemoryTier.SEMANTIC, n_entries=0, n_vectors=0, dim=dim, bytes=0
        ),
        archival=StoreFootprint(
            tier=MemoryTier.ARCHIVAL, n_entries=0, n_vectors=0, dim=dim, bytes=0
        ),
        total_bytes=0,
    )


def _zero_cost() -> ArmCost:
    """A zero ArmCost (the endpoint reads only continual_metrics.acc)."""
    return ArmCost(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        api_calls=0,
        retrieval_calls=0,
        memory_vectors=0,
        memory_bytes=0,
    )


def _arm_result(arm_name: str, seed: int, acc: float) -> ArmResult:
    """A minimal valid ArmResult whose paired final ACC is ``acc``."""
    clamped = float(min(max(acc, 0.0), 1.0))
    matrix = AccuracyMatrix(
        n_tasks=1, scenario=CLScenario.TASK_INCREMENTAL, R=[[clamped]]
    )
    metrics = ContinualMetrics(
        acc=acc, bwt=0.0, fwt=0.0, average_forgetting=0.0, per_task_forgetting=[]
    )
    return ArmResult(
        arm_name=arm_name,
        seed=seed,
        scenario=CLScenario.TASK_INCREMENTAL.value,
        accuracy_matrix=matrix,
        continual_metrics=metrics,
        footprint=_empty_footprint(),
        cost=_zero_cost(),
    )


def _compute_endpoint(
    prereg: Preregistration,
    full_dream: list[float],
    no_sleep: list[float],
    *,
    seeds: list[int] = SEEDS,
    aa_floor: float = 0.01,
) -> PrimaryEndpoint:
    """Compute a real PrimaryEndpoint from per-seed accs (harness-equivalent)."""
    grid: list[ArmResult] = []
    for seed, acc in zip(seeds, full_dream):
        grid.append(_arm_result("full_dream", seed, acc))
    for seed, acc in zip(seeds, no_sleep):
        grid.append(_arm_result("no_sleep", seed, acc))
    aa = AAResult(
        reference_arm="no_sleep",
        seed_a=seeds[0],
        seed_b=seeds[1],
        value_a=0.60,
        value_b=0.60 + aa_floor,
        abs_difference=aa_floor,
        significant=False,
        test=None,
    )
    return compute_primary_endpoint(
        prereg, grid, requested_name=PRIMARY, aa=aa, rng=np.random.default_rng(0)
    )


def _manual_endpoint(
    *,
    name: str = PRIMARY,
    value: float = 0.2,
    d: float = 1.5,
    ci: tuple[float, float] = (0.10, 0.30),
    noise_floor: float = 0.01,
    p_value: float = 0.03,
    magnitude: str = "large",
    verdict: str = "confirmed",
) -> PrimaryEndpoint:
    """A directly-constructed PrimaryEndpoint (for the power / DX3 fixtures)."""
    return PrimaryEndpoint(
        name=name,
        description="paired ACC diff full_dream - no_sleep",
        treatment_arm="full_dream",
        baseline_arm="no_sleep",
        value=value,
        difference_ci=BootstrapCI(
            point=value,
            lo=ci[0],
            hi=ci[1],
            level=0.95,
            method="percentile bootstrap",
            n_resamples=10000,
            statistic="paired_mean_diff",
        ),
        effect=EffectSize(
            name="cohens_d", value=d, lo=d - 0.4, hi=d + 0.4, level=0.95, magnitude=magnitude
        ),
        test=_TestResult(
            test="wilcoxon_signed_rank", statistic=0.0, p_value=p_value, n=len(SEEDS)
        ),
        noise_floor=noise_floor,
        exceeds_noise_floor=abs(value) > noise_floor,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# Regime cell / grid / sweep / Phase5Result builders
# --------------------------------------------------------------------------- #
def _regime_cell(
    *,
    name: str = "distractor_heavy",
    signal: float = 0.34,
    distractor: float = 0.40,
    noise: float = 0.26,
    seeds: list[int] = SEEDS,
    acc_by_arm: dict[str, list[float]],
    signal_retention_by_arm: dict[str, list[float]],
    total_tokens_by_arm: dict[str, list[float]] | None = None,
    primary_endpoint: PrimaryEndpoint | None = None,
    manifest_path: str = "runs/phase5/regime_distractor_heavy/manifest.json",
    aa_abs_difference: float = 0.01,
) -> RegimeCell:
    """Build a RegimeCell with consistent per-arm per-seed arrays."""
    arms = list(acc_by_arm.keys())
    n = len(seeds)
    zeros = {a: [0.0] * n for a in arms}
    return RegimeCell(
        regime=RegimeMix(name=name, signal=signal, distractor=distractor, noise=noise),
        manifest_path=manifest_path,
        arms=arms,
        seeds=seeds,
        acc_by_arm=acc_by_arm,
        prune_precision_by_arm=zeros,
        prune_recall_by_arm=zeros,
        prune_f1_by_arm=zeros,
        signal_retention_by_arm=signal_retention_by_arm,
        total_tokens_by_arm=total_tokens_by_arm or {a: [1000.0] * n for a in arms},
        memory_vectors_by_arm={a: [10.0] * n for a in arms},
        primary_endpoint=primary_endpoint,
        primary_value=primary_endpoint.value if primary_endpoint else None,
        primary_verdict=primary_endpoint.verdict if primary_endpoint else None,
        primary_ci_lo=primary_endpoint.difference_ci.lo if primary_endpoint else None,
        primary_ci_hi=primary_endpoint.difference_ci.hi if primary_endpoint else None,
        aa_abs_difference=aa_abs_difference,
        aa_significant=False,
        budget_matched=True,
    )


def _length_sweep(
    points: list[LengthPoint],
    *,
    treatment_arm: str = "full_dream",
    baseline_arm: str = "long_context",
) -> LengthSweep:
    """Build a LengthSweep over the given points."""
    return LengthSweep(
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        points=points,
        coverage_notes=["synthetic sweep"],
    )


def _length_point(
    n_tasks: int,
    *,
    acc: dict[str, float],
    tokens: dict[str, float],
    seeds: list[int] = (0, 1, 2),
) -> LengthPoint:
    """A LengthPoint with constant per-seed ACC / token values per arm."""
    seeds = list(seeds)
    arms = list(acc.keys())
    return LengthPoint(
        n_tasks=n_tasks,
        items_per_task=4,
        arms=arms,
        seeds=seeds,
        acc_by_arm={a: [acc[a]] * len(seeds) for a in arms},
        total_tokens_by_arm={a: [tokens[a]] * len(seeds) for a in arms},
        memory_vectors_by_arm={a: [float(n_tasks)] * len(seeds) for a in arms},
    )


def _sim_real(note: str = "sim and real retention curves agree; arm ranking preserved.") -> SimRealResult:
    """A minimal SimRealResult (analyze reads only its note)."""
    return SimRealResult(
        arms=[],
        sim_n_tasks=3,
        sim_compression=60.0,
        real_n_tasks=10,
        real_compression=1.0,
        seeds=[0, 1, 2],
        pearson_agreement=0.95,
        spearman_agreement=1.0,
        ranking_preserved=True,
        inversions=[],
        max_abs_acc_divergence=0.05,
        note=note,
    )


def _phase5_result(
    cells: list[RegimeCell],
    *,
    primary_regime: str = "distractor_heavy",
    length_sweep: LengthSweep | None = None,
    sim_real: SimRealResult | None = None,
) -> Phase5Result:
    """Assemble a Phase5Result from regime cells + sweeps (analysis=None)."""
    arms = cells[0].arms
    grid = GridResult(
        regimes=cells,
        arms=arms,
        seeds=cells[0].seeds,
        primary_regime=primary_regime,
        coverage_notes=["synthetic grid; 0 dropped"],
    )
    if length_sweep is None:
        length_sweep = _crossover_sweep()
    if sim_real is None:
        sim_real = _sim_real()
    return Phase5Result(
        experiment="phase5_test",
        scenario=CLScenario.TASK_INCREMENTAL.value,
        git_commit="deadbeef",
        grid=grid,
        length_sweep=length_sweep,
        sim_real=sim_real,
        manifest_paths=["runs/phase5/regime_distractor_heavy/manifest.json"],
    )


# --- canned sweeps -------------------------------------------------------- #
def _crossover_sweep() -> LengthSweep:
    """A sweep where full_dream acc/token overtakes long_context at L=3."""
    return _length_sweep(
        [
            _length_point(2, acc={"full_dream": 0.60, "long_context": 0.80}, tokens={"full_dream": 100.0, "long_context": 100.0}),
            _length_point(3, acc={"full_dream": 0.60, "long_context": 0.80}, tokens={"full_dream": 100.0, "long_context": 200.0}),
            _length_point(5, acc={"full_dream": 0.60, "long_context": 0.80}, tokens={"full_dream": 100.0, "long_context": 400.0}),
        ]
    )


def _no_crossover_sweep() -> LengthSweep:
    """A sweep where long_context acc/token always dominates full_dream."""
    return _length_sweep(
        [
            _length_point(2, acc={"full_dream": 0.50, "long_context": 0.90}, tokens={"full_dream": 200.0, "long_context": 100.0}),
            _length_point(3, acc={"full_dream": 0.50, "long_context": 0.90}, tokens={"full_dream": 200.0, "long_context": 100.0}),
        ]
    )


def _full_cell(
    prereg: Preregistration,
    *,
    full_dream: list[float],
    no_sleep: list[float],
    replay_only: list[float] | None = None,
    aa_floor: float = 0.01,
) -> RegimeCell:
    """A primary cell with a computed endpoint + TMR-able signal_retention."""
    n = len(SEEDS)
    replay_only = replay_only or [a - 0.03 for a in full_dream]
    acc_by_arm = {
        "full_dream": full_dream,
        "no_sleep": no_sleep,
        "replay_only": replay_only,
        "reflection": [a - 0.05 for a in full_dream],
        "downscale_only": no_sleep,
    }
    signal_retention_by_arm = {
        "full_dream": [0.90, 0.92, 0.88, 0.91, 0.89],
        "replay_only": [0.86, 0.88, 0.85, 0.87, 0.84],
        "reflection": [0.87, 0.86, 0.88, 0.85, 0.86],
        "no_sleep": [0.55, 0.52, 0.57, 0.54, 0.56],
        "downscale_only": [0.58, 0.60, 0.57, 0.59, 0.56],
    }
    endpoint = _compute_endpoint(prereg, full_dream, no_sleep, aa_floor=aa_floor)
    return _regime_cell(
        acc_by_arm=acc_by_arm,
        signal_retention_by_arm=signal_retention_by_arm,
        primary_endpoint=endpoint,
        aa_abs_difference=aa_floor,
    )


# --------------------------------------------------------------------------- #
# PRIMARY endpoint (EC5) + DX3
# --------------------------------------------------------------------------- #
def test_analyze_confirmed_surfaces_cell_verdict(prereg: Preregistration, prereg_path: str) -> None:
    """full_dream strictly > no_sleep -> confirmed verdict, positive effect (EC5)."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)

    assert report.primary_endpoint_name == PRIMARY
    assert report.primary_verdict == "confirmed"
    assert report.primary_verdict == cell.primary_verdict  # matches recorded verdict
    assert report.primary_value is not None and report.primary_value > 0.0
    assert report.primary_effect_value is not None and report.primary_effect_value > 0.0
    assert report.primary_ci_lo is not None and report.primary_ci_lo > 0.0
    assert report.primary_ci_method == cell.primary_endpoint.difference_ci.method
    assert report.exceeds_noise_floor is True
    assert report.n_seeds == len(SEEDS)
    assert report.mock_llm_caveat  # DX5 caveat always present


def test_analyze_equal_arms_non_confirmed_ci_spans_zero(
    prereg: Preregistration, prereg_path: str
) -> None:
    """Equal arms -> a non-confirmed verdict with a CI spanning zero (EC5)."""
    accs = [0.60, 0.62, 0.58, 0.61, 0.59]
    cell = _full_cell(prereg, full_dream=list(accs), no_sleep=list(accs), replay_only=list(accs))
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)

    assert report.primary_verdict != "confirmed"
    assert report.primary_ci_lo is not None and report.primary_ci_hi is not None
    assert report.primary_ci_lo <= 0.0 <= report.primary_ci_hi


def test_analyze_uses_prereg_primary_endpoint_name(
    prereg: Preregistration, prereg_path: str
) -> None:
    """The surfaced endpoint name is exactly the prereg's primary endpoint (DX3)."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)
    assert report.primary_endpoint_name == prereg.primary_endpoint


def test_analyze_refuses_non_preregistered_endpoint(prereg_path: str) -> None:
    """A cell endpoint with a non-preregistered name is refused (DX3)."""
    bad_endpoint = _manual_endpoint(name="acc_diff_reflection_vs_no_sleep", verdict="confirmed")
    cell = _regime_cell(
        acc_by_arm={
            "full_dream": [0.8] * 5,
            "no_sleep": [0.6] * 5,
            "replay_only": [0.7] * 5,
        },
        signal_retention_by_arm={
            "full_dream": [0.9] * 5,
            "no_sleep": [0.5] * 5,
        },
        primary_endpoint=bad_endpoint,
    )
    with pytest.raises(NonPreregisteredEndpointError):
        analyze(_phase5_result([cell]), prereg_path=prereg_path)


def test_analyze_not_computed_when_endpoint_missing(prereg_path: str) -> None:
    """No cell endpoint and an unreadable manifest -> verdict not-computed (DX2)."""
    cell = _regime_cell(
        acc_by_arm={
            "full_dream": [0.7] * 5,
            "no_sleep": [0.6] * 5,
            "replay_only": [0.65] * 5,
        },
        signal_retention_by_arm={
            "full_dream": [0.9] * 5,
            "no_sleep": [0.5] * 5,
        },
        primary_endpoint=None,
        manifest_path="runs/phase5/does_not_exist/manifest.json",
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)
    assert report.primary_verdict == "not-computed"
    assert report.primary_value is None
    assert report.primary_ci_method == "n/a"
    assert report.primary_endpoint_name == PRIMARY
    # Power falls back to the sentinel + note when the effect is unavailable.
    assert report.power.required_n_for_observed == 9999
    assert report.power.note


# --------------------------------------------------------------------------- #
# CROSSOVER (EC6)
# --------------------------------------------------------------------------- #
def test_crossover_found_at_known_length(prereg: Preregistration, prereg_path: str) -> None:
    """acc/token overtakes the baseline at the known L=3 (EC6)."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    result = _phase5_result([cell], length_sweep=_crossover_sweep())
    report = analyze(result, prereg_path=prereg_path)

    cr = report.crossover
    assert cr.crossover_found is True
    assert cr.crossover_length == 3
    assert cr.lengths == [2, 3, 5]
    # long_context keeps the raw-accuracy lead (no raw crossover).
    assert cr.raw_accuracy_crossover_length is None
    assert "L=3" in cr.note


def test_crossover_absent_states_absence(prereg: Preregistration, prereg_path: str) -> None:
    """When acc/token never overtakes, crossover_found is False + absence stated (EC6)."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    result = _phase5_result([cell], length_sweep=_no_crossover_sweep())
    report = analyze(result, prereg_path=prereg_path)

    cr = report.crossover
    assert cr.crossover_found is False
    assert cr.crossover_length is None
    assert "no cost-adjusted crossover" in cr.note.lower()


# --------------------------------------------------------------------------- #
# TMR (FR5.3)
# --------------------------------------------------------------------------- #
def test_tmr_positive_lift_exceeds_benchmark(prereg: Preregistration, prereg_path: str) -> None:
    """Replay arms with higher signal_retention -> g>0, finite CI, exceeds 0.29."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)
    tmr = report.tmr

    assert tmr.mean_lift > 0.0
    assert tmr.hedges_g > 0.0
    assert np.isfinite(tmr.g_ci_lo) and np.isfinite(tmr.g_ci_hi)
    assert tmr.g_ci_lo <= tmr.g_ci_hi
    assert tmr.benchmark_g == 0.29
    assert tmr.exceeds_benchmark == (tmr.hedges_g > 0.29)
    assert tmr.exceeds_benchmark is True  # the canned lift is large
    assert "full_dream" in tmr.replay_arms and "no_sleep" in tmr.no_replay_arms
    assert "proxy" in tmr.note.lower()


# --------------------------------------------------------------------------- #
# POWER (EC2)
# --------------------------------------------------------------------------- #
def test_power_large_effect_is_powered(prereg_path: str) -> None:
    """A large observed d yields a small required n and powered_for_observed True."""
    endpoint = _manual_endpoint(value=0.20, d=2.0, ci=(0.12, 0.28), verdict="confirmed")
    cell = _regime_cell(
        acc_by_arm={"full_dream": [0.8] * 5, "no_sleep": [0.6] * 5, "replay_only": [0.78] * 5},
        signal_retention_by_arm={"full_dream": [0.9] * 5, "no_sleep": [0.5] * 5},
        primary_endpoint=endpoint,
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)

    assert report.power.n_seeds == 5
    assert report.power.floor_met is True
    assert report.power.observed_effect_d == pytest.approx(2.0)
    assert report.power.required_n_for_observed <= 5
    assert report.power.powered_for_observed is True


def test_power_tiny_effect_needs_many_seeds(prereg_path: str) -> None:
    """A tiny observed d yields a large required n, not powered, with a note."""
    endpoint = _manual_endpoint(value=0.01, d=0.05, ci=(-0.02, 0.04), magnitude="negligible", verdict="refuted")
    cell = _regime_cell(
        acc_by_arm={"full_dream": [0.61] * 5, "no_sleep": [0.60] * 5, "replay_only": [0.60] * 5},
        signal_retention_by_arm={"full_dream": [0.9] * 5, "no_sleep": [0.5] * 5},
        primary_endpoint=endpoint,
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)

    assert report.power.required_n_for_observed > 1000
    assert report.power.powered_for_observed is False
    assert "Colas" in report.power.note


# --------------------------------------------------------------------------- #
# NEGATIVE mapping (EC7)
# --------------------------------------------------------------------------- #
def test_negative_mapping_for_refuted_primary(prereg: Preregistration, prereg_path: str) -> None:
    """A refuted primary -> applicable True, >=1 matched form, regime tie, secondary."""
    # Near-equal arms -> CI spans zero -> refuted; full_dream not > replay_only.
    cell = _full_cell(
        prereg,
        full_dream=[0.60, 0.61, 0.59, 0.62, 0.58],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
        replay_only=[0.61, 0.62, 0.60, 0.63, 0.59],
    )
    assert cell.primary_verdict == "refuted"
    result = _phase5_result([cell], length_sweep=_no_crossover_sweep())
    report = analyze(result, prereg_path=prereg_path)

    neg = report.negative
    assert neg.applicable is True
    assert len(neg.matched_forms) >= 1
    assert all(form in prereg.negative_result_forms for form in neg.matched_forms)
    assert neg.regime_tie.startswith("distractor_heavy")
    # The registered secondary contrast is always populated.
    assert "full_dream vs replay_only" in neg.secondary_contrasts


def test_negative_secondary_always_populated_when_confirmed(
    prereg: Preregistration, prereg_path: str
) -> None:
    """Even a confirmed primary records the full_dream-vs-replay_only contrast."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
        replay_only=[0.62, 0.63, 0.61, 0.64, 0.60],
    )
    report = analyze(_phase5_result([cell]), prereg_path=prereg_path)
    assert "full_dream vs replay_only" in report.negative.secondary_contrasts


# --------------------------------------------------------------------------- #
# per-regime verdicts + sim-real note + headline
# --------------------------------------------------------------------------- #
def test_per_regime_verdicts_and_sim_real_note(
    prereg: Preregistration, prereg_path: str
) -> None:
    """per_regime_verdicts spans every cell; sim-real note comes from result."""
    primary = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    other = _regime_cell(
        name="balanced",
        signal=0.50,
        distractor=0.30,
        noise=0.20,
        acc_by_arm={"full_dream": [0.7] * 5, "no_sleep": [0.68] * 5, "replay_only": [0.69] * 5},
        signal_retention_by_arm={"full_dream": [0.8] * 5, "no_sleep": [0.6] * 5},
        primary_endpoint=_manual_endpoint(value=0.02, d=0.3, ci=(-0.01, 0.05), verdict="refuted"),
        manifest_path="runs/phase5/regime_balanced/manifest.json",
    )
    result = _phase5_result(
        [primary, other], sim_real=_sim_real(note="documented inversion at scale (finding).")
    )
    report = analyze(result, prereg_path=prereg_path)

    assert set(report.per_regime_verdicts) == {"distractor_heavy", "balanced"}
    assert report.per_regime_verdicts["distractor_heavy"] == "confirmed"
    assert report.sim_real_agreement_note == "documented inversion at scale (finding)."
    assert report.headline  # non-empty plain-language headline


# --------------------------------------------------------------------------- #
# Determinism (DX1)
# --------------------------------------------------------------------------- #
def test_analyze_is_deterministic(prereg: Preregistration, prereg_path: str) -> None:
    """Two analyses with the same stats_seed are byte-identical (DX1)."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    result = _phase5_result([cell])
    a = analyze(result, prereg_path=prereg_path, stats_seed=0)
    b = analyze(result, prereg_path=prereg_path, stats_seed=0)
    assert a.model_dump() == b.model_dump()


# --------------------------------------------------------------------------- #
# write_analysis + run_analysis
# --------------------------------------------------------------------------- #
def test_write_analysis_emits_both_files(
    prereg: Preregistration, prereg_path: str, tmp_path: Path
) -> None:
    """write_analysis writes analysis.json + RESULTS.md and rewrites the artifact."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    result = _phase5_result([cell])
    report = analyze(result, prereg_path=prereg_path)

    analysis_path, md_path = write_analysis(result, report, tmp_path)

    assert analysis_path.exists() and md_path.exists()
    # analysis.json round-trips as an AnalysisReport.
    loaded = AnalysisReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    assert loaded.model_dump() == report.model_dump()

    # RESULTS.md opens with the mock-LLM caveat and names every section keyword.
    md = md_path.read_text(encoding="utf-8")
    assert "MECHANISM DEMONSTRATION" in md  # the DX5 caveat
    assert "primary endpoint" in md
    assert "crossover" in md
    assert "TMR" in md
    assert "power" in md

    # The artifact is rewritten with analysis filled in (input object untouched).
    rewritten = json.loads((tmp_path / "phase5" / "phase5_result.json").read_text(encoding="utf-8"))
    assert rewritten["analysis"] is not None
    assert result.analysis is None
    assert Phase5Result.model_validate(rewritten).analysis is not None


def test_write_analysis_is_byte_identical_on_rerun(
    prereg: Preregistration, prereg_path: str, tmp_path: Path
) -> None:
    """Re-running write_analysis produces byte-identical JSON (DX1)."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    result = _phase5_result([cell])
    report = analyze(result, prereg_path=prereg_path)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    write_analysis(result, report, out_a)
    write_analysis(result, report, out_b)
    assert (out_a / "phase5" / "analysis.json").read_bytes() == (
        out_b / "phase5" / "analysis.json"
    ).read_bytes()


def test_run_analysis_loads_writes_and_returns_path(
    prereg: Preregistration, prereg_path: str, tmp_path: Path
) -> None:
    """run_analysis loads a result file, writes artifacts, returns analysis.json."""
    cell = _full_cell(
        prereg,
        full_dream=[0.80, 0.82, 0.78, 0.81, 0.79],
        no_sleep=[0.60, 0.62, 0.58, 0.61, 0.59],
    )
    result = _phase5_result([cell])
    phase5_dir = tmp_path / "phase5"
    phase5_dir.mkdir(parents=True)
    result_path = phase5_dir / "phase5_result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )

    analysis_path = run_analysis(result_path, prereg_path=prereg_path)

    assert analysis_path == phase5_dir / "analysis.json"
    assert analysis_path.exists()
    loaded = AnalysisReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    assert loaded.primary_endpoint_name == PRIMARY
    # The original artifact was rewritten in place with the analysis filled in.
    rewritten = Phase5Result.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert rewritten.analysis is not None
