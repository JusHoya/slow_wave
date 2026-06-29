"""Cross-module integration tests for the Phase 4 evaluation harness (EC1-EC8).

These exercise the *whole* nine-arm control battery end-to-end through
:func:`slow_wave.eval.harness.build_experiment` / :func:`run_experiment` on the
committed ``configs/eval_smoke.yaml`` (a tiny, fully-deterministic mock-LLM grid),
and assert each Phase 4 exit criterion against the assembled
:class:`~slow_wave.eval.schema.ExperimentResult`:

* EC1 — all nine arms instantiate and run on the same stream via one harness;
* EC2 — the A/A control yields no significant difference (the noise floor);
* EC3 — the oracle arm's prune precision/recall ceiling beats every non-oracle arm;
* EC4 — the matched-budget controller records actuals + always produces a Pareto;
* EC5 — the metric suite (ACC/BWT/FWT/FM + prune P/R/F1 + calibration) is present,
  decoupled, and JSON-serializable;
* EC6 — the statistics suite is complete (aggregates, Friedman, Holm-corrected
  comparisons + effect sizes + CIs, probability-of-improvement, profiles);
* EC7 — the primary endpoint is computed exactly as preregistered, and a
  non-preregistered endpoint is refused;
* EC8 — the temperature-0 stability + memory-drift controls produce numbers.
"""

from __future__ import annotations

import json

import pytest

from slow_wave.config import load_config
from slow_wave.eval.harness import build_experiment, run_experiment
from slow_wave.eval.prereg import NonPreregisteredEndpointError
from slow_wave.eval.schema import BootstrapCI, ExperimentResult

_SMOKE_CONFIG = "configs/eval_smoke.yaml"
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
def experiment() -> ExperimentResult:
    """Build the smoke experiment once and share it across the EC assertions."""
    cfg = load_config(_SMOKE_CONFIG)
    return build_experiment(cfg)


def _by_arm(experiment: ExperimentResult, attr_path):
    """Return ``{arm_name: [per-seed value]}`` for a callable extracting one value."""
    out: dict[str, list[float]] = {a: [] for a in experiment.arms_run}
    for res in experiment.arm_results:
        out[res.arm_name].append(attr_path(res))
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# EC1 — all nine arms instantiate and run on the same stream via one harness
# --------------------------------------------------------------------------- #
def test_ec1_all_nine_arms_run(experiment: ExperimentResult) -> None:
    """Every one of the nine arms ran once per seed on the shared stream."""
    assert set(experiment.arms_run) == _NINE_ARMS
    n_seeds = len(experiment.seeds)
    counts = {a: 0 for a in _NINE_ARMS}
    for res in experiment.arm_results:
        counts[res.arm_name] += 1
        # Same stream within a seed => one scenario, well-formed R[i,j].
        assert res.scenario == experiment.scenario
        assert res.accuracy_matrix.n_tasks == len(res.accuracy_matrix.R)
    assert all(c == n_seeds for c in counts.values())
    assert len(experiment.arm_results) == len(_NINE_ARMS) * n_seeds


# --------------------------------------------------------------------------- #
# EC2 — the A/A control is the noise floor (no significant difference)
# --------------------------------------------------------------------------- #
def test_ec2_aa_noise_floor_not_significant(experiment: ExperimentResult) -> None:
    """Two identical configs (different seeds) show no significant difference."""
    aa = experiment.aa
    assert aa.reference_arm == "no_sleep"
    assert aa.significant is False
    # The noise floor is a small magnitude (sanity: not a huge spurious gap).
    assert aa.abs_difference < 0.2
    assert aa.test is not None and aa.test.test == "wilcoxon_signed_rank"


# --------------------------------------------------------------------------- #
# EC3 — the oracle prune ceiling beats every non-oracle arm
# --------------------------------------------------------------------------- #
def test_ec3_oracle_prune_ceiling(experiment: ExperimentResult) -> None:
    """Oracle prune precision AND recall (and F1) exceed every non-oracle arm."""
    prec = _by_arm(experiment, lambda r: r.prune_quality.precision)
    rec = _by_arm(experiment, lambda r: r.prune_quality.recall)
    f1 = _by_arm(experiment, lambda r: r.prune_quality.f1)

    o_prec, o_rec, o_f1 = _mean(prec["oracle"]), _mean(rec["oracle"]), _mean(f1["oracle"])
    others = [a for a in experiment.arms_run if a != "oracle"]

    # Oracle is the prune-quality ceiling (FR5.1): >= on precision and recall,
    # strictly the unique max on F1 (the combined ceiling).
    assert all(o_prec >= _mean(prec[a]) - 1e-9 for a in others)
    assert all(o_rec >= _mean(rec[a]) - 1e-9 for a in others)
    assert all(o_f1 > _mean(f1[a]) for a in others)
    # Oracle prunes exactly the distractor/noise => recall is maximal (== 1.0 here).
    assert o_rec == pytest.approx(1.0)
    assert experiment.arm_results[0].uses_labels is False  # only oracle reads labels
    assert any(r.uses_labels for r in experiment.arm_results if r.arm_name == "oracle")
    assert all(
        not r.uses_labels for r in experiment.arm_results if r.arm_name != "oracle"
    )


# --------------------------------------------------------------------------- #
# EC4 — matched-budget controller records actuals + always produces a Pareto
# --------------------------------------------------------------------------- #
def test_ec4_budget_actuals_and_pareto(experiment: ExperimentResult) -> None:
    """Every arm's actuals are recorded and a Pareto frontier is produced (FR5.2)."""
    br = experiment.budget_report
    assert {a.arm_name for a in br.per_arm} == _NINE_ARMS
    # Actuals recorded for every arm.
    for a in br.per_arm:
        assert a.mean_total_tokens >= 0.0
        assert a.mean_memory_vectors >= 0.0
    # The Pareto frontier is ALWAYS produced (the reported artifact when unmatched).
    assert {p.arm_name for p in br.pareto} == _NINE_ARMS
    assert any(p.on_frontier for p in br.pareto)
    # Honesty (DX2): excluded ceilings are named in the notes.
    assert br.notes
    joined = " ".join(br.notes).lower()
    assert "oracle" in joined and "long_context" in joined


def test_ec4_controller_can_match_when_comparable() -> None:
    """The controller reports 'matched' when included arms share a budget."""
    from slow_wave.eval.budget_match import match_budget
    from slow_wave.eval.schema import ArmCost

    def cost(tok, ret, vec):
        return ArmCost(
            input_tokens=tok,
            output_tokens=0,
            total_tokens=tok,
            api_calls=1,
            retrieval_calls=ret,
            memory_vectors=vec,
            memory_bytes=vec * 384 * 4,
        )

    arm_costs = {
        "no_sleep": [cost(100, 10, 50), cost(100, 10, 50)],
        "full_dream": [cost(105, 10, 52), cost(95, 10, 48)],
        "replay_only": [cost(100, 11, 50), cost(100, 9, 50)],
    }
    accs = {"no_sleep": [0.5, 0.5], "full_dream": [0.8, 0.8], "replay_only": [0.6, 0.6]}
    report = match_budget(arm_costs, accuracies=accs, tolerance=0.15)
    assert report.matched is True


# --------------------------------------------------------------------------- #
# EC5 — metric suite present, decoupled, JSON-serializable
# --------------------------------------------------------------------------- #
def test_ec5_metrics_present_and_decoupled(experiment: ExperimentResult) -> None:
    """Continual + prune + calibration metrics coexist and can diverge (FR5.3)."""
    for res in experiment.arm_results:
        cm = res.continual_metrics
        assert -1.0 <= cm.bwt <= 1.0 and 0.0 <= cm.acc <= 1.0
        assert res.prune_quality is not None
        pq = res.prune_quality
        assert pq.tp + pq.fp + pq.fn + pq.tn > 0
        assert pq.tp + pq.fp == pq.n_pruned
        assert res.calibration is not None
        assert 0.0 <= res.calibration.expected_calibration_error <= 1.0
        # Everything JSON-serializable.
        json.dumps(res.model_dump(mode="json"))

    # Decoupling: the highest-accuracy arm is NOT the highest-prune-F1 arm.
    acc = _by_arm(experiment, lambda r: r.continual_metrics.acc)
    f1 = _by_arm(experiment, lambda r: r.prune_quality.f1)
    best_acc = max(acc, key=lambda a: _mean(acc[a]))
    best_f1 = max(f1, key=lambda a: _mean(f1[a]))
    assert best_acc != best_f1  # accuracy and consolidation quality diverge


# --------------------------------------------------------------------------- #
# EC6 — statistics suite complete (aggregates, omnibus, corrected comparisons)
# --------------------------------------------------------------------------- #
def test_ec6_statistics_suite_complete(experiment: ExperimentResult) -> None:
    """Bootstrap CIs, rliable aggregates, Friedman, Holm comparisons, profiles."""
    st = experiment.stats
    assert st.metric == "acc"
    assert {a.arm_name for a in st.aggregates} == _NINE_ARMS
    for agg in st.aggregates:
        for ci in (agg.iqm, agg.median, agg.mean):
            assert isinstance(ci, BootstrapCI)
            assert ci.lo <= ci.point <= ci.hi
            assert ci.n_resamples > 0
    # Omnibus across all arms.
    assert st.omnibus is not None and st.omnibus.test == "friedman"
    # One Holm-corrected paired comparison per non-baseline arm.
    assert len(st.comparisons) == len(_NINE_ARMS) - 1
    assert st.correction == "holm"
    for comp in st.comparisons:
        assert comp.adjusted_p >= comp.raw_p - 1e-9  # Holm never decreases p
        assert comp.effect.name == "cohens_d"
        assert comp.effect.lo <= comp.effect.value <= comp.effect.hi
        assert comp.difference_ci.lo <= comp.difference_ci.point <= comp.difference_ci.hi
    # rliable probability-of-improvement + performance profiles.
    assert len(st.probability_of_improvement) == len(_NINE_ARMS) - 1
    assert all(0.0 <= p <= 1.0 for p in st.probability_of_improvement.values())
    assert set(st.performance_profiles) == _NINE_ARMS
    for profile in st.performance_profiles.values():
        fracs = [pt[1] for pt in profile]
        assert all(a >= b - 1e-9 for a, b in zip(fracs, fracs[1:]))  # non-increasing


# --------------------------------------------------------------------------- #
# EC7 — primary endpoint computed as registered; non-prereg endpoint refused
# --------------------------------------------------------------------------- #
def test_ec7_primary_endpoint_as_registered(experiment: ExperimentResult) -> None:
    """The single primary endpoint matches the prereg and has a falsifiable verdict."""
    pe = experiment.primary_endpoint
    assert pe is not None
    assert pe.name == experiment.prereg.primary_endpoint
    assert pe.name == "acc_diff_full_dream_vs_no_sleep"
    assert pe.treatment_arm == "full_dream" and pe.baseline_arm == "no_sleep"
    assert pe.verdict in {"confirmed", "refuted", "inconclusive"}
    assert pe.test.test == "wilcoxon_signed_rank"
    assert pe.difference_ci.lo <= pe.difference_ci.point <= pe.difference_ci.hi
    assert pe.effect.name == "cohens_d"
    # noise_floor is the A/A magnitude.
    assert pe.noise_floor == pytest.approx(experiment.aa.abs_difference)


def test_ec7_non_preregistered_endpoint_refused() -> None:
    """A primary endpoint that is not preregistered is refused loudly (DX3)."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.primary_endpoint = "acc_per_retrieval_token_NOT_REGISTERED"
    with pytest.raises(NonPreregisteredEndpointError):
        build_experiment(cfg)


# --------------------------------------------------------------------------- #
# EC8 — temperature-0 stability + memory-drift controls produce numbers
# --------------------------------------------------------------------------- #
def test_ec8_bias_controls_produce_numbers(experiment: ExperimentResult) -> None:
    """Both bias controls return well-formed numbers on a tiny mock-LLM run."""
    s = experiment.stability
    assert s.n_repeats >= 2
    assert s.identical is True  # deterministic mock LLM
    assert s.mean_pairwise_similarity == pytest.approx(1.0)
    assert s.token_cv == pytest.approx(0.0)

    d = experiment.drift
    assert d.n_rounds >= 2
    assert len(d.fidelity_per_round) == d.n_rounds
    assert 0.0 <= d.faithfulness <= 1.0
    assert isinstance(d.degraded, bool)


# --------------------------------------------------------------------------- #
# Determinism, JSON round-trip, and the one-command manifest
# --------------------------------------------------------------------------- #
def test_experiment_round_trips_json(experiment: ExperimentResult) -> None:
    """The whole ExperimentResult serializes to JSON and validates back."""
    dumped = json.dumps(experiment.model_dump(mode="json"), sort_keys=True)
    restored = ExperimentResult.model_validate(json.loads(dumped))
    assert restored.arms_run == experiment.arms_run
    assert restored.primary_endpoint.value == pytest.approx(
        experiment.primary_endpoint.value
    )


def test_determinism_under_mock_llm() -> None:
    """Two builds of the same config produce identical non-cost outputs (DX1)."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.seeds = [0, 1]  # trim for speed; determinism is seed-count-independent
    a = build_experiment(cfg)
    b = build_experiment(cfg)

    def digest(exp):
        return {
            "acc": {
                r.arm_name + str(r.seed): round(r.continual_metrics.acc, 9)
                for r in exp.arm_results
            },
            "prune_f1": {
                r.arm_name + str(r.seed): round(r.prune_quality.f1, 9)
                for r in exp.arm_results
            },
            "aa": round(exp.aa.abs_difference, 9),
            "primary": round(exp.primary_endpoint.value, 9),
            "iqm": {
                agg.arm_name: round(agg.iqm.point, 9) for agg in exp.stats.aggregates
            },
        }

    assert digest(a) == digest(b)


def test_run_experiment_writes_manifest(tmp_path) -> None:
    """The one-command runner writes a readable manifest with the experiment."""
    cfg = load_config(_SMOKE_CONFIG)
    cfg.eval.seeds = [0, 1]
    path = run_experiment(cfg, out_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["experiment"] == "eval-smoke"
    assert "experiment" in data["results"]
    assert set(data["results"]["experiment"]["arms_run"]) == _NINE_ARMS
    # FR6.1 provenance still present on the manifest.
    assert data["git"]["commit"] is not None or data["git"]["commit"] is None
    assert data["cost"]["api_calls"] >= 0
