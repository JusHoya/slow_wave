"""The one evaluation harness: nine arms, matched budget, stats, prereg (Phase 4).

This is the orchestrator glue that ties the five Phase 4 workstreams into a single
runnable experiment (PRD §8 Phase 4; see ``docs/PHASE4_CONTRACT.md``):

* :func:`run_arm` materializes one control arm (WS-ARMS), runs the
  :class:`~slow_wave.agent.wake.WakeAgent` over a shared stream, and rolls the
  result up into an :class:`~slow_wave.eval.schema.ArmResult` — continual metrics
  (Phase 2), mechanism-level prune quality + calibration (WS-METRICS), and the
  realized cost (wake + dream telemetry).
* :func:`run_experiment` runs the **nine-arm × seed grid on one shared stream per
  seed** (same stream within a seed; both stream and sampling vary across seeds,
  FR5.4), then layers the matched-budget controller + Pareto (WS-BUDGET), the
  statistics suite (WS-STATS), the A/A noise-floor control, the preregistration's
  primary endpoint (WS-PREREG), and the temperature-0 stability + memory-drift
  bias controls (WS-PREREG), and writes a single :class:`ExperimentResult` into a
  run manifest.

All non-LLM outputs are reproducible bit-for-bit given a fixed config + seeds
under the mock LLM (DX1); the bootstrap statistics are deterministic given
``cfg.eval.stats_seed``. The LLM token/wall-clock fields are flagged
nondeterministic in the manifest.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from slow_wave.agent.wake import WakeAgent
from slow_wave.config import Config
from slow_wave.embeddings import get_embedder
from slow_wave.eval.arms import ARM_REGISTRY, build_arm
from slow_wave.eval.budget_match import match_budget
from slow_wave.eval.controls import memory_drift, temperature_zero_stability
from slow_wave.eval.prereg import (
    NonPreregisteredEndpointError,
    compute_primary_endpoint,
    load_preregistration,
)
from slow_wave.eval.prune_metrics import calibration_curve, prune_quality
from slow_wave.eval.schema import (
    AAResult,
    ArmCost,
    ArmResult,
    Comparison,
    ExperimentResult,
    PrimaryEndpoint,
    StatsReport,
)
from slow_wave.eval.stats import (
    effect_size_cohens_d,
    friedman,
    holm_correction,
    paired_diff_ci,
    performance_profile,
    probability_of_improvement,
    robust_aggregate,
    wilcoxon_signed_rank,
)
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set
from slow_wave.stream.schema import Stream

logger = logging.getLogger(__name__)

#: Prime seed offset for the A/A control's second seed family ("two identical
#: configs, DIFFERENT seeds", FR5.1) so it never collides with a configured seed.
_AA_SEED_OFFSET = 7919

#: Performance-profile thresholds (rliable): the accuracy survival curve sample
#: points, ascending so the profile is non-increasing.
_PERF_PROFILE_TAUS: list[float] = [round(0.1 * i, 1) for i in range(11)]


def run_arm(
    name: str,
    base_cfg: Config,
    stream: Stream,
    probe_set,
    embedder,
    seed: int,
) -> ArmResult:
    """Materialize and run one control arm, returning its :class:`ArmResult`.

    Builds the arm via :func:`slow_wave.eval.arms.build_arm` (deep-merging the
    arm's overrides onto ``base_cfg`` and stamping ``seed``), runs the
    :class:`~slow_wave.agent.wake.WakeAgent` over ``stream`` with the arm's
    sleep hook, then rolls up the continual metrics (Phase 2), the mechanism-level
    prune quality + calibration curve (computed offline against the stream's
    ground-truth labels — the sanctioned offline label read), and the realized
    cost (wake telemetry + the arm's dream/prune telemetry).

    Args:
        name: The arm's registry name (must be in
            :data:`slow_wave.eval.arms.ARM_REGISTRY`).
        base_cfg: The base experiment config to merge the arm overrides onto.
        stream: The shared stream the arm runs on (one per seed).
        probe_set: The held-out probe set scored to fill ``R[i,j]``.
        embedder: A shared embedder duck-type (``.dim``, ``.encode``).
        seed: The run seed (stamped onto the effective config).

    Returns:
        A fully populated :class:`ArmResult` (``primary_endpoint_value`` is left
        ``None`` — the harness fills it after the prereg endpoint is computed).
    """
    arm = build_arm(name, base_cfg, stream, seed)

    start = time.perf_counter()
    agent = WakeAgent(arm.cfg, embedder)
    wake = agent.run(stream, probe_set, sleep_hook=arm.sleep_hook)
    wall = time.perf_counter() - start

    substrate = wake.substrate
    tel = wake.telemetry
    d_in, d_out, d_api = arm.dream_tokens()

    footprint = wake.footprint
    active_vectors = footprint.episodic.n_vectors + footprint.semantic.n_vectors
    cost = ArmCost(
        input_tokens=tel.input_tokens + d_in,
        output_tokens=tel.output_tokens + d_out,
        total_tokens=tel.input_tokens + tel.output_tokens + d_in + d_out,
        api_calls=tel.api_calls + d_api,
        retrieval_calls=tel.retrieval_calls,
        memory_vectors=active_vectors,
        memory_bytes=footprint.total_bytes,
        wall_clock_s=wall,
        p95_latency_s=tel.p95_latency_s,
    )

    return ArmResult(
        arm_name=name,
        seed=seed,
        scenario=stream.scenario.value,
        accuracy_matrix=wake.accuracy_matrix,
        continual_metrics=wake.metrics,
        footprint=footprint,
        cost=cost,
        n_failure_events=len(substrate.failure_events),
        n_dream_cycles=arm.n_cycles(),
        generator_fidelity=arm.generator_fidelity(),
        uses_labels=arm.spec.uses_labels,
        prune_quality=prune_quality(stream, substrate),
        calibration=calibration_curve(stream, substrate),
        primary_endpoint_value=None,
    )


def _build_stats_report(
    arm_names: list[str],
    acc_by_arm: dict[str, list[float]],
    baseline_arm: str,
    *,
    level: float,
    n_resamples: int,
    family_alpha: float,
    rng: np.random.Generator,
) -> StatsReport:
    """Assemble the full statistics suite over the arm × seed accuracy grid (FR5.4).

    Computes per-arm rliable robust aggregates (IQM/median/mean + bootstrap CIs),
    the Friedman omnibus across all arms, Holm-corrected paired arm-vs-baseline
    comparisons (Wilcoxon + Cohen's d + paired-difference CI), rliable
    probability-of-improvement over the baseline, and per-arm performance
    profiles.

    Args:
        arm_names: Arms in report order.
        acc_by_arm: Per-arm list of per-seed final ACC (aligned by seed order).
        baseline_arm: The arm every comparison is made against.
        level: Confidence level for CIs.
        n_resamples: Bootstrap resamples.
        family_alpha: Family-wise alpha for the multiple-comparison correction.
        rng: Explicit bootstrap RNG (DX1).

    Returns:
        The populated :class:`StatsReport`.
    """
    aggregates = [
        robust_aggregate(
            arm, acc_by_arm[arm], level=level, n_resamples=n_resamples, rng=rng
        )
        for arm in arm_names
    ]

    # Friedman omnibus across all arms (needs >= 3 aligned arms).
    omnibus = None
    if len(arm_names) >= 3:
        lengths = {len(acc_by_arm[a]) for a in arm_names}
        if len(lengths) == 1 and lengths.pop() >= 1:
            try:
                omnibus = friedman([acc_by_arm[a] for a in arm_names])
            except ValueError as exc:  # k<3 / uneven — recorded, not hidden
                logger.info("friedman omnibus skipped: %s", exc)

    # Paired arm-vs-baseline comparisons with Holm correction across the family.
    baseline_acc = acc_by_arm[baseline_arm]
    contrast_arms = [a for a in arm_names if a != baseline_arm]
    tests = [wilcoxon_signed_rank(acc_by_arm[a], baseline_acc) for a in contrast_arms]
    raw_ps = [t.p_value for t in tests]
    corrected = holm_correction(raw_ps, alpha=family_alpha)

    comparisons: list[Comparison] = []
    for arm, test, (adj_p, reject) in zip(contrast_arms, tests, corrected):
        effect = effect_size_cohens_d(
            acc_by_arm[arm],
            baseline_acc,
            paired=True,
            level=level,
            n_resamples=n_resamples,
            rng=rng,
        )
        diff_ci = paired_diff_ci(
            acc_by_arm[arm],
            baseline_acc,
            level=level,
            n_resamples=n_resamples,
            rng=rng,
        )
        comparisons.append(
            Comparison(
                label=f"{arm} vs {baseline_arm}",
                test=test,
                raw_p=test.p_value,
                adjusted_p=adj_p,
                reject=reject,
                effect=effect,
                difference_ci=diff_ci,
            )
        )

    poi = {
        f"{arm} > {baseline_arm}": probability_of_improvement(
            acc_by_arm[arm], baseline_acc
        )
        for arm in contrast_arms
    }
    profiles = {
        arm: performance_profile(acc_by_arm[arm], _PERF_PROFILE_TAUS)
        for arm in arm_names
    }

    return StatsReport(
        metric="acc",
        arms=list(arm_names),
        aggregates=aggregates,
        omnibus=omnibus,
        comparisons=comparisons,
        probability_of_improvement=poi,
        performance_profiles=profiles,
        correction="holm",
        family_alpha=family_alpha,
    )


def _run_aa_control(
    base_cfg: Config,
    embedder,
    seeds: list[int],
    aa_family_a_acc: list[float],
    *,
    family_alpha: float,
) -> AAResult:
    """Run the A/A noise-floor control: identical config, two seed families (EC2).

    Family A is the already-computed ``aa`` arm accuracy at the configured seeds;
    family B re-runs the *same* reference config at a disjoint, prime-offset seed
    family (FR5.1: "two identical configs, different seeds"). A paired Wilcoxon
    (by index) over the two families establishes that the difference is not
    significant — the noise floor any claimed effect must exceed.

    Args:
        base_cfg: The base config (its ``eval.aa_reference_arm`` is the config
            run twice).
        embedder: The shared embedder.
        seeds: The configured seed list (family A seeds).
        aa_family_a_acc: The ``aa`` arm's per-seed ACC from the main grid (family A).
        family_alpha: Significance threshold for the verdict.

    Returns:
        The :class:`AAResult` noise floor.
    """
    ref = base_cfg.eval.aa_reference_arm
    family_b_acc: list[float] = []
    for s in seeds:
        b_seed = s + _AA_SEED_OFFSET
        stream_b = generate_stream(base_cfg.stream, _stream_seed(b_seed))
        probe_b = build_probe_set(stream_b)
        res = run_arm(ref, base_cfg, stream_b, probe_b, embedder, b_seed)
        family_b_acc.append(res.continual_metrics.acc)

    value_a = float(np.mean(aa_family_a_acc)) if aa_family_a_acc else 0.0
    value_b = float(np.mean(family_b_acc)) if family_b_acc else 0.0
    test = None
    significant = False
    if len(aa_family_a_acc) == len(family_b_acc) and len(family_b_acc) >= 1:
        test = wilcoxon_signed_rank(aa_family_a_acc, family_b_acc)
        significant = test.p_value < family_alpha

    return AAResult(
        reference_arm=ref,
        seed_a=seeds[0] if seeds else 0,
        seed_b=(seeds[0] + _AA_SEED_OFFSET) if seeds else _AA_SEED_OFFSET,
        value_a=value_a,
        value_b=value_b,
        abs_difference=abs(value_a - value_b),
        significant=significant,
        test=test,
    )


def _stream_seed(seed: int) -> int:
    """Return the stream-generation seed derived from a master ``seed`` (FR5.4)."""
    from slow_wave.repro.seeding import derive_seed

    return derive_seed(seed, "stream")


def build_experiment(cfg: Config) -> ExperimentResult:
    """Run the full nine-arm control battery and return its result (Phase 4).

    For each seed in ``cfg.eval.seeds`` a single stream is generated (so every arm
    shares one stream within a seed; both stream and sampling vary across seeds,
    FR5.4) and every arm in ``cfg.eval.arms`` is run on it. The harness then
    matches budgets + builds the Pareto frontier (FR5.2), computes the statistics
    suite (FR5.4), the A/A noise-floor control (FR5.1), the preregistration's
    primary endpoint (FR5.5), and the temperature-0 stability + memory-drift bias
    controls (FR5.6), assembling everything into one
    :class:`~slow_wave.eval.schema.ExperimentResult`. This is the pure
    (no-I/O) core; :func:`run_experiment` wraps it to write a manifest.

    Args:
        cfg: The experiment config; ``cfg.stream`` must be set and every name in
            ``cfg.eval.arms`` must resolve in the arm registry.

    Returns:
        The assembled :class:`~slow_wave.eval.schema.ExperimentResult`.

    Raises:
        ValueError: If ``cfg.stream`` is ``None`` or an arm name is unknown.
        NonPreregisteredEndpointError: If ``cfg.eval.primary_endpoint`` does not
            match the committed preregistration (DX3).
    """
    from slow_wave.repro.seeding import set_global_seeds

    if cfg.stream is None:
        raise ValueError("run_experiment requires cfg.stream to be set.")
    unknown = [a for a in cfg.eval.arms if a not in ARM_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown arm(s) {unknown}; known arms: {sorted(ARM_REGISTRY)}"
        )

    set_global_seeds(cfg.seed)
    arm_names = list(cfg.eval.arms)
    seeds = list(cfg.eval.seeds)
    embedder = get_embedder(cfg)
    rng = np.random.default_rng(cfg.eval.stats_seed)

    coverage: list[str] = []
    arm_results: list[ArmResult] = []
    scenario = cfg.stream.scenario.value
    stream_ids: list[str] = []

    # The arm x seed grid: one shared stream per seed (DX2: log coverage).
    for seed in seeds:
        stream = generate_stream(cfg.stream, _stream_seed(seed))
        probe_set = build_probe_set(stream)
        stream_ids.append(stream.stream_id)
        for name in arm_names:
            arm_results.append(
                run_arm(name, cfg, stream, probe_set, embedder, seed)
            )
    coverage.append(
        f"ran {len(arm_names)} arms x {len(seeds)} seeds = {len(arm_results)} "
        f"runs; no cells dropped"
    )

    # Group per-arm per-seed actuals (aligned by seed order for pairing).
    acc_by_arm: dict[str, list[float]] = {a: [] for a in arm_names}
    cost_by_arm: dict[str, list[ArmCost]] = {a: [] for a in arm_names}
    for res in arm_results:
        acc_by_arm[res.arm_name].append(res.continual_metrics.acc)
        cost_by_arm[res.arm_name].append(res.cost)

    # Matched-budget controller + Pareto (FR5.2).
    budget_report = match_budget(
        cost_by_arm,
        accuracies=acc_by_arm,
        tolerance=cfg.eval.budget_tolerance,
        target_tokens=cfg.eval.token_budget,
    )

    # Statistics suite (FR5.4).
    stats = _build_stats_report(
        arm_names,
        acc_by_arm,
        cfg.eval.baseline_arm,
        level=cfg.eval.ci_level,
        n_resamples=cfg.eval.bootstrap_resamples,
        family_alpha=0.05,
        rng=rng,
    )

    # A/A noise-floor control (FR5.1 / EC2).
    aa_acc = acc_by_arm.get("aa", acc_by_arm.get(cfg.eval.aa_reference_arm, []))
    aa = _run_aa_control(cfg, embedder, seeds, aa_acc, family_alpha=0.05)

    # Preregistration primary endpoint (FR5.5 / EC7); refuse a non-prereg endpoint.
    prereg = load_preregistration(cfg.eval.prereg_path)
    primary: PrimaryEndpoint | None = None
    try:
        primary = compute_primary_endpoint(
            prereg,
            arm_results,
            requested_name=cfg.eval.primary_endpoint,
            aa=aa,
            ci_level=cfg.eval.ci_level,
            n_resamples=cfg.eval.bootstrap_resamples,
            rng=rng,
        )
    except NonPreregisteredEndpointError as exc:
        coverage.append(f"primary endpoint refused (non-preregistered): {exc}")
        raise
    except ValueError as exc:  # e.g. treatment/baseline arm absent from grid
        coverage.append(f"primary endpoint not computed: {exc}")

    if primary is not None:
        for res in arm_results:
            if res.arm_name == prereg.treatment_arm:
                res.primary_endpoint_value = res.continual_metrics.acc

    # Bias controls (FR5.6 / EC8): temperature-0 stability + memory drift.
    stability = temperature_zero_stability(
        cfg, n_repeats=cfg.eval.stability_repeats
    )
    drift = memory_drift(cfg, n_rounds=cfg.eval.drift_rounds)

    return ExperimentResult(
        experiment=cfg.experiment,
        scenario=scenario,
        stream_id=stream_ids[0] if stream_ids else "",
        arms_run=arm_names,
        seeds=seeds,
        arm_results=arm_results,
        budget_report=budget_report,
        stats=stats,
        aa=aa,
        primary_endpoint=primary,
        prereg=prereg,
        stability=stability,
        drift=drift,
        coverage_notes=coverage,
    )


def run_experiment(cfg: Config, out_dir: str | Path | None = None) -> Path:
    """Run the control battery (:func:`build_experiment`) and write its manifest.

    Wraps :func:`build_experiment` with timing + manifest I/O: writes the
    assembled :class:`~slow_wave.eval.schema.ExperimentResult` to
    ``<out>/eval/manifest.json`` (one command:
    ``python -m slow_wave.eval.runner --config configs/eval_smoke.yaml``) and
    prints a human summary.

    Args:
        cfg: The experiment config (see :func:`build_experiment`).
        out_dir: Output root; defaults to ``cfg.output_dir``.

    Returns:
        The path to the written manifest.
    """
    start = time.perf_counter()
    experiment = build_experiment(cfg)
    wall = time.perf_counter() - start
    path = _write_manifest(cfg, get_embedder(cfg), experiment, out_dir, wall)
    _print_summary(experiment)
    return path


def _aggregate_llm(cfg: Config, experiment: ExperimentResult):
    """Build the summed-cost LLM stand-in for the manifest's ``llm``/``cost`` block.

    Args:
        cfg: The experiment config (for the model id).
        experiment: The assembled experiment result (its arm costs are summed).

    Returns:
        A small object exposing the ``.text``/``.model_id``/``.input_tokens``/
        ``.output_tokens``/``.mocked``/``.stop_reason`` duck-type
        :func:`slow_wave.repro.manifest.new_manifest` expects, plus the total
        ``api_calls``.
    """
    from dataclasses import dataclass

    @dataclass
    class _AggregateLLM:
        text: str
        model_id: str
        input_tokens: int
        output_tokens: int
        mocked: bool
        stop_reason: str = "aggregate"

    total_in = sum(r.cost.input_tokens for r in experiment.arm_results)
    total_out = sum(r.cost.output_tokens for r in experiment.arm_results)
    verdict = (
        experiment.primary_endpoint.verdict
        if experiment.primary_endpoint is not None
        else "not-computed"
    )
    summary = (
        f"eval harness: {len(experiment.arms_run)} arms x {len(experiment.seeds)} "
        f"seeds; primary={verdict}; matched={experiment.budget_report.matched}"
    )
    llm = _AggregateLLM(
        text=summary,
        model_id=cfg.model.id,
        input_tokens=total_in,
        output_tokens=total_out,
        mocked=True,  # mock LLM unless ANTHROPIC_API_KEY is set during arm runs
    )
    return llm, total_in + total_out


def _write_manifest(
    cfg: Config, embedder, experiment: ExperimentResult, out_dir, wall: float
) -> Path:
    """Write the experiment into a run manifest under ``<out>/eval/manifest.json``."""
    from slow_wave.repro.gitinfo import git_info
    from slow_wave.repro.manifest import new_manifest, write_manifest
    from slow_wave.repro.seeding import derive_seed

    llm, _ = _aggregate_llm(cfg, experiment)
    total_api = sum(r.cost.api_calls for r in experiment.arm_results)

    pe = experiment.primary_endpoint
    deterministic_probe = {
        "scenario": experiment.scenario,
        "arms_run": experiment.arms_run,
        "seeds": experiment.seeds,
        "acc_by_arm": {
            a: [
                round(r.continual_metrics.acc, 6)
                for r in experiment.arm_results
                if r.arm_name == a
            ]
            for a in experiment.arms_run
        },
        "prune_f1_by_arm": {
            a: [
                round(r.prune_quality.f1, 6)
                for r in experiment.arm_results
                if r.arm_name == a and r.prune_quality is not None
            ]
            for a in experiment.arms_run
        },
        "budget_matched": experiment.budget_report.matched,
        "aa_abs_difference": round(experiment.aa.abs_difference, 6),
        "aa_significant": experiment.aa.significant,
        "primary_endpoint": None
        if pe is None
        else {"name": pe.name, "value": round(pe.value, 6), "verdict": pe.verdict},
        "stability_identical": experiment.stability.identical,
        "drift_degraded": experiment.drift.degraded,
    }

    manifest = new_manifest(
        cfg=cfg,
        embedder=embedder,
        llm=llm,
        seeds={"master": cfg.seed, **{f"seed_{s}": s for s in experiment.seeds}},
        deterministic_probe=deterministic_probe,
        wall_clock_s=wall,
        git=git_info(),
        api_calls=total_api,
        results={"experiment": experiment.model_dump(mode="json")},
    )

    out_path = Path(out_dir or cfg.output_dir) / "eval" / "manifest.json"
    written = write_manifest(manifest, out_path)
    fallback_reason = getattr(embedder, "fallback_reason", None)
    if fallback_reason:
        logger.warning("Embedder fell back: %s", fallback_reason)
    print(f"[eval] manifest written to {written}")
    return written


def _print_summary(experiment: ExperimentResult) -> None:
    """Print a one-line-per-section human summary of the experiment."""
    pe = experiment.primary_endpoint
    print(
        f"[eval] experiment={experiment.experiment} "
        f"arms={len(experiment.arms_run)} seeds={len(experiment.seeds)}"
    )
    print(
        f"[eval] budget matched={experiment.budget_report.matched} "
        f"target_tokens={experiment.budget_report.target_tokens:.1f}"
    )
    print(
        f"[eval] A/A noise floor abs_diff={experiment.aa.abs_difference:.4f} "
        f"significant={experiment.aa.significant}"
    )
    if pe is not None:
        print(
            f"[eval] primary endpoint {pe.name}={pe.value:.4f} "
            f"CI=[{pe.difference_ci.lo:.4f},{pe.difference_ci.hi:.4f}] "
            f"verdict={pe.verdict}"
        )
    print(
        f"[eval] stability identical={experiment.stability.identical} "
        f"drift degraded={experiment.drift.degraded}"
    )
