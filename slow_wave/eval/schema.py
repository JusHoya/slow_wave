"""Shared result models for the Phase 4 evaluation harness (authoritative).

This module is the **cross-module contract** for Phase 4 (see
``docs/PHASE4_CONTRACT.md``). Every workstream returns one of the pydantic
models defined here; the harness rolls them into an :class:`ExperimentResult`
and the runner serializes that into the run manifest. Independent workstreams
import the models from here and **must not redefine them**.

Design principles
-----------------
* **JSON-first & deterministic (DX1).** Every model is JSON-serializable with
  stable key order (``model_dump(mode="json")`` + ``sort_keys=True`` at write
  time). All non-LLM fields are exact (counts, fractions, byte sizes) so two runs
  from the same config + seed are byte-identical under the mock LLM.
* **Decoupled metrics (FR5.3).** Mechanism-level consolidation quality
  (:class:`PruneQuality`, :class:`CalibrationCurve`) is reported **separately**
  from downstream accuracy (:class:`~slow_wave.eval.metrics.ContinualMetrics`):
  the two can — and the bench expects them to — diverge.
* **Honesty by construction (DX2).** Anything bounded (a Pareto-dropped arm, a
  mismatched budget, a coverage cap) is surfaced as an explicit field, never
  dropped silently.
* **``extra="forbid"`` everywhere** so a typo'd field fails loudly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from slow_wave.eval.metrics import ContinualMetrics
from slow_wave.memory.schema import MemoryFootprint
from slow_wave.stream.schema import AccuracyMatrix


# --------------------------------------------------------------------------- #
# Arms (WS-ARMS)
# --------------------------------------------------------------------------- #
class ArmSpec(BaseModel):
    """Declarative specification of one control arm (FR5.1, DX6).

    An arm is a *named* configuration of the existing wake + dream machinery plus
    an optional sleep-window operator (random-prune / oracle-prune). The harness
    materializes an arm by deep-merging :attr:`config_overrides` onto the base
    config and attaching the arm's sleep hook. Adding a new arm is implementing a
    builder and registering it (DX6) — no harness edits.

    Attributes:
        name: Canonical registry name (e.g. ``"full_dream"``).
        description: One-line human-readable description.
        config_overrides: Nested dict deep-merged onto the base config (e.g.
            ``{"dream": {"enabled": True, "augment_enabled": False}}``).
        uses_labels: ``True`` ONLY for the ``oracle`` arm, which is permitted to
            read ground-truth labels offline as the upper-bound ceiling (FR5.1).
            Every other arm MUST be ``False`` — the confound guard (FR1.6) still
            holds for them. The harness asserts this invariant.
        family: Coarse grouping for reporting (``"baseline"``, ``"ablation"``,
            ``"control"``, ``"ceiling"``, ``"treatment"``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    uses_labels: bool = False
    family: Literal[
        "baseline", "ablation", "control", "ceiling", "treatment"
    ] = "ablation"


class ArmCost(BaseModel):
    """Realized cost of running one arm under one seed (FR5.2, FR5.3 cost).

    Attributes:
        input_tokens: Summed prompt tokens (wake + dream).
        output_tokens: Summed completion tokens (wake + dream).
        total_tokens: ``input_tokens + output_tokens``.
        api_calls: Total LLM calls (wake + dream).
        retrieval_calls: Wake-time context retrieval calls.
        memory_vectors: Final live vector count across active tiers
            (episodic + semantic) — the matched "final memory vector count"
            (FR5.2).
        memory_bytes: Final total footprint in bytes (all tiers).
        wall_clock_s: Wall-clock duration in seconds (nondeterministic).
        p95_latency_s: 95th-percentile per-item latency (nondeterministic).
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    total_tokens: int
    api_calls: int
    retrieval_calls: int
    memory_vectors: int
    memory_bytes: int
    wall_clock_s: float = 0.0
    p95_latency_s: float = 0.0


class LabelCount(BaseModel):
    """Counts of stream items by ground-truth label (offline scoring only)."""

    model_config = ConfigDict(extra="forbid")

    signal: int = 0
    distractor: int = 0
    noise: int = 0


class PruneQuality(BaseModel):
    """Mechanism-level consolidation quality: pruning vs. ground truth (FR5.3).

    Computed **offline** (the only sanctioned label read, via
    :func:`slow_wave.stream.schema.offline_labels`) by comparing what the run
    **retained** in active memory against what it **pruned** (demoted to archival
    or never consolidated), scored against ground-truth relevance. "Positive" =
    *pruned* and the target of a good prune is a ``distractor``/``noise`` item;
    ``signal`` items should be retained. This is the bench's superpower (FR5.3):
    consolidation quality measured against known relevance, **decoupled** from
    downstream accuracy.

    Attributes:
        precision: Of items pruned, fraction that were truly distractor/noise
            (TP / (TP + FP)). ``0.0`` when nothing was pruned.
        recall: Of all distractor/noise items, fraction that were pruned
            (TP / (TP + FN)). ``0.0`` when there were none.
        f1: Harmonic mean of precision and recall (``0.0`` if either is 0).
        tp: Pruned items that were distractor/noise (correct prunes).
        fp: Pruned items that were signal (wrongly pruned).
        fn: Retained items that were distractor/noise (wrongly kept).
        tn: Retained items that were signal (correctly kept).
        n_retained: Items judged retained.
        n_pruned: Items judged pruned.
        retained_by_label: Retained counts split by ground-truth label.
        pruned_by_label: Pruned counts split by ground-truth label.
        signal_retention: Fraction of signal items retained (the protective
            complement of recall; a good arm keeps signal AND prunes noise).
    """

    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_retained: int
    n_pruned: int
    retained_by_label: LabelCount
    pruned_by_label: LabelCount
    signal_retention: float


class CalibrationBin(BaseModel):
    """One bin of the salience/decay-vs-true-relevance calibration curve.

    Attributes:
        lo: Inclusive lower edge of the normalized-salience bin.
        hi: Exclusive upper edge of the bin (inclusive for the last bin).
        n: Number of items falling in the bin.
        mean_salience: Mean normalized salience of the items in the bin.
        frac_signal: Fraction of the bin's items whose ground-truth label is
            ``signal`` (the "true relevance" the salience is calibrated against).
    """

    model_config = ConfigDict(extra="forbid")

    lo: float
    hi: float
    n: int
    mean_salience: float
    frac_signal: float


class CalibrationCurve(BaseModel):
    """Decay/salience-vs-true-relevance calibration curve (FR5.3).

    A reliability-diagram-style curve: items are binned by their final normalized
    salience and each bin reports the fraction that are truly ``signal``. A
    well-calibrated consolidation has ``frac_signal`` increasing with salience.
    Reported **decoupled** from accuracy.

    Attributes:
        bins: The ordered calibration bins.
        n_items: Total items binned.
        expected_calibration_error: Mean absolute gap between each bin's
            mean_salience and its frac_signal, weighted by bin count (ECE).
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[CalibrationBin]
    n_items: int
    expected_calibration_error: float


class ArmResult(BaseModel):
    """The complete result of running one arm under one seed.

    WS-ARMS populates everything except :attr:`prune_quality` and
    :attr:`calibration` (filled by the harness via WS-METRICS, keeping ARMS
    decoupled from METRICS). The harness also fills :attr:`primary_endpoint_value`
    after the prereg-named endpoint is computed.

    Attributes:
        arm_name: The arm's registry name.
        seed: The master seed this run used.
        scenario: The stream's CL scenario (no cross-scenario aggregation).
        accuracy_matrix: The filled ``R[i][j]``.
        continual_metrics: ACC/BWT/FWT/forgetting over ``R`` (FR5.3).
        footprint: Per-tier memory footprint after the run.
        cost: Realized cost (FR5.2/FR5.3).
        n_failure_events: Logged write-protection / failure events.
        n_dream_cycles: Dream cycles run (0 for no-sleep/long-context).
        generator_fidelity: Mean generative-augment fidelity over the run, or
            ``None`` if augment never ran (FR5.3 generator fidelity).
        uses_labels: Echo of the arm's label-permission flag (oracle only).
        prune_quality: Mechanism-level prune quality (filled by harness).
        calibration: Salience-vs-relevance calibration (filled by harness).
        primary_endpoint_value: This arm/seed's value of the prereg primary
            endpoint's per-arm term (filled by harness; ``None`` until computed).
    """

    model_config = ConfigDict(extra="forbid")

    arm_name: str
    seed: int
    scenario: str
    accuracy_matrix: AccuracyMatrix
    continual_metrics: ContinualMetrics
    footprint: MemoryFootprint
    cost: ArmCost
    n_failure_events: int = 0
    n_dream_cycles: int = 0
    generator_fidelity: float | None = None
    uses_labels: bool = False
    prune_quality: PruneQuality | None = None
    calibration: CalibrationCurve | None = None
    primary_endpoint_value: float | None = None


# --------------------------------------------------------------------------- #
# Matched budget + Pareto (WS-BUDGET)
# --------------------------------------------------------------------------- #
class ArmBudgetActuals(BaseModel):
    """Per-arm realized budget actuals + match verdict (FR5.2).

    Attributes:
        arm_name: The arm's registry name.
        mean_total_tokens: Mean total tokens over the arm's seeds.
        mean_retrieval_calls: Mean retrieval calls over the arm's seeds.
        mean_memory_vectors: Mean final active vector count over the arm's seeds.
        tokens_within_tolerance: Whether mean tokens are within tolerance of the
            target.
        retrieval_within_tolerance: Whether mean retrieval calls are within
            tolerance of the target.
        memory_within_tolerance: Whether mean memory vectors are within tolerance
            of the target.
        matched: Conjunction of the three within-tolerance flags.
    """

    model_config = ConfigDict(extra="forbid")

    arm_name: str
    mean_total_tokens: float
    mean_retrieval_calls: float
    mean_memory_vectors: float
    tokens_within_tolerance: bool
    retrieval_within_tolerance: bool
    memory_within_tolerance: bool
    matched: bool


class ParetoPoint(BaseModel):
    """One (accuracy, compute) point on the cost/accuracy trade-off (FR5.2).

    Attributes:
        arm_name: The arm's registry name.
        accuracy: The arm's mean primary accuracy (ACC).
        compute_tokens: The arm's mean total tokens (the compute axis).
        memory_vectors: The arm's mean final active vector count.
        on_frontier: Whether this point is Pareto-non-dominated (max accuracy,
            min compute).
    """

    model_config = ConfigDict(extra="forbid")

    arm_name: str
    accuracy: float
    compute_tokens: float
    memory_vectors: float
    on_frontier: bool


class BudgetReport(BaseModel):
    """Matched-budget verdict + Pareto frontier (FR5.2).

    Attributes:
        matched: Whether every (non-ceiling) arm matched all three budget axes
            within tolerance.
        tolerance: The fractional tolerance used.
        target_tokens: The token target arms were matched against.
        target_retrieval: The retrieval-call target.
        target_memory_vectors: The active-vector-count target.
        per_arm: Per-arm actuals + verdicts.
        pareto: The accuracy-vs-compute Pareto frontier, ALWAYS produced (it is
            the reported artifact when matching is infeasible, FR5.2).
        notes: DX2 honesty notes (e.g. which arms are intentionally excluded from
            matching, like the long-context ceiling).
    """

    model_config = ConfigDict(extra="forbid")

    matched: bool
    tolerance: float
    target_tokens: float
    target_retrieval: float
    target_memory_vectors: float
    per_arm: list[ArmBudgetActuals]
    pareto: list[ParetoPoint]
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Statistics (WS-STATS)
# --------------------------------------------------------------------------- #
class BootstrapCI(BaseModel):
    """A bootstrap confidence interval for a scalar statistic (FR5.4).

    Attributes:
        point: The point estimate on the observed sample.
        lo: Lower CI bound.
        hi: Upper CI bound.
        level: Confidence level (e.g. 0.95).
        method: CI method, e.g. ``"percentile"``.
        n_resamples: Number of bootstrap resamples.
        statistic: Name of the statistic (e.g. ``"mean"``, ``"iqm"``).
    """

    model_config = ConfigDict(extra="forbid")

    point: float
    lo: float
    hi: float
    level: float
    method: str
    n_resamples: int
    statistic: str


class EffectSize(BaseModel):
    """A standardized effect size with a CI (FR5.4 — not just p-values).

    Attributes:
        name: ``"cohens_d"`` (paired/unpaired) or ``"cliffs_delta"``.
        value: The point estimate.
        lo: Lower CI bound.
        hi: Upper CI bound.
        level: Confidence level of the CI.
        magnitude: Qualitative label (``"negligible"``/``"small"``/``"medium"``/
            ``"large"``) per conventional thresholds.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    lo: float
    hi: float
    level: float
    magnitude: str


class TestResult(BaseModel):
    """The result of a significance test (FR5.4).

    Attributes:
        test: Test name (``"wilcoxon_signed_rank"``, ``"friedman"``,
            ``"nemenyi"``...).
        statistic: The test statistic.
        p_value: The (uncorrected) p-value.
        n: Sample size (pairs / blocks).
        detail: Optional extra info (e.g. df, tie corrections, group labels).
    """

    model_config = ConfigDict(extra="forbid")

    test: str
    statistic: float
    p_value: float
    n: int
    detail: dict[str, Any] = Field(default_factory=dict)


class Comparison(BaseModel):
    """One arm-vs-control comparison with corrected significance (FR5.4).

    Attributes:
        label: ``"<arm> vs <control>"``.
        test: The paired test applied (typically Wilcoxon).
        raw_p: Uncorrected p-value.
        adjusted_p: Multiple-comparison-adjusted p-value (e.g. Holm).
        reject: Whether H0 is rejected at the family-wise alpha after correction.
        effect: The standardized effect size with CI for this comparison.
        difference_ci: Bootstrap CI of the paired mean difference.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    test: TestResult
    raw_p: float
    adjusted_p: float
    reject: bool
    effect: EffectSize
    difference_ci: BootstrapCI


class RobustAggregate(BaseModel):
    """rliable-style robust aggregate for one arm's score samples (FR5.4).

    Implements the Agarwal et al. (2021) aggregates in pure numpy: the
    interquartile mean (IQM), median, and mean, each with a bootstrap CI.

    Attributes:
        arm_name: The arm the scores belong to.
        n: Number of score samples (seeds).
        iqm: IQM with bootstrap CI.
        median: Median with bootstrap CI.
        mean: Mean with bootstrap CI.
    """

    model_config = ConfigDict(extra="forbid")

    arm_name: str
    n: int
    iqm: BootstrapCI
    median: BootstrapCI
    mean: BootstrapCI


class StatsReport(BaseModel):
    """The full statistics suite over the arm x seed grid (FR5.4).

    Attributes:
        metric: Name of the per-run score the statistics were computed over
            (e.g. ``"acc"``).
        arms: The arms included, in report order.
        aggregates: Per-arm robust aggregates (IQM/median/mean + CIs).
        omnibus: The Friedman omnibus test across all arms (or ``None`` if < 3
            arms / insufficient data).
        comparisons: Arm-vs-control paired comparisons with Holm-corrected
            significance and effect sizes.
        probability_of_improvement: ``{"<armA> > <armB>": P}`` rliable
            probability-of-improvement entries for the registered contrasts.
        performance_profiles: ``{arm_name: [(tau, frac_runs >= tau), ...]}``
            performance-profile sample points (rliable).
        correction: The multiple-comparison method used (e.g. ``"holm"``).
        family_alpha: The family-wise error rate controlled for.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    arms: list[str]
    aggregates: list[RobustAggregate]
    omnibus: TestResult | None
    comparisons: list[Comparison]
    probability_of_improvement: dict[str, float] = Field(default_factory=dict)
    performance_profiles: dict[str, list[list[float]]] = Field(default_factory=dict)
    correction: str = "holm"
    family_alpha: float = 0.05


# --------------------------------------------------------------------------- #
# Preregistration + bias controls (WS-PREREG)
# --------------------------------------------------------------------------- #
class Preregistration(BaseModel):
    """The committed registered-report artifact, parsed (FR5.5, DX3).

    Mirrors ``prereg/preregistration.yaml``. The analysis reads the **single**
    :attr:`primary_endpoint` from here and refuses to compute any other (DX3:
    falsifiability visible in code).

    Attributes:
        title: Short title of the preregistration.
        hypothesis_h1: The alternative hypothesis (the effect claimed).
        hypothesis_h0: The null hypothesis (the noise floor / no effect).
        primary_endpoint: The single pre-specified primary endpoint name.
        primary_endpoint_description: Prose definition of how it is computed.
        treatment_arm: The treatment arm name in the primary contrast.
        baseline_arm: The baseline arm name in the primary contrast.
        seed_plan: Description of the seed plan (>= 5; both seed types).
        power_analysis: Description/figures of the power analysis (Colas 2018).
        tests: The pre-specified statistical tests.
        rejection_criteria: Explicit, falsifiable rejection criteria.
        negative_result_forms: The pre-specified negative-result forms (PRD §9.1).
        committed_git_hash: Git commit the prereg was committed at (filled when
            known; ``None`` in the artifact itself).
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    hypothesis_h1: str
    hypothesis_h0: str
    primary_endpoint: str
    primary_endpoint_description: str
    treatment_arm: str
    baseline_arm: str
    seed_plan: str
    power_analysis: str
    tests: list[str]
    rejection_criteria: list[str]
    negative_result_forms: list[str] = Field(default_factory=list)
    committed_git_hash: str | None = None


class StabilityResult(BaseModel):
    """Temperature-0 stability control for the dream summarizer (FR5.6).

    Quantifies run-to-run variance of the summarization output under fixed
    inputs. Under the deterministic mock LLM the repeats are byte-identical
    (:attr:`identical` True, :attr:`distinct_outputs` == 1); with a real
    temperature-0 model they may still drift, which this number captures.

    Attributes:
        n_repeats: Number of repeated summarization calls on the same input.
        distinct_outputs: Number of distinct output texts seen.
        identical: Whether all repeats produced byte-identical output.
        mean_pairwise_similarity: Mean pairwise cosine similarity of the repeat
            embeddings in ``[0, 1]`` (1.0 == perfectly stable).
        token_cv: Coefficient of variation of output token counts.
    """

    model_config = ConfigDict(extra="forbid")

    n_repeats: int
    distinct_outputs: int
    identical: bool
    mean_pairwise_similarity: float
    token_cv: float


class DriftResult(BaseModel):
    """Memory-drift detector over repeated summarization (FR5.6).

    Re-summarizes a memory's own summary repeatedly and tracks fidelity to the
    *original* source. A monotonic decline flags silent corruption — repeated
    summarization degrading rather than distilling memory.

    Attributes:
        n_rounds: Number of re-summarization rounds.
        fidelity_per_round: Cosine fidelity of each round's output to the
            original source, in round order.
        faithfulness: Final-round fidelity to the original source.
        degraded: Whether fidelity declined past the drift threshold (silent
            corruption flagged).
        monotonic_decline: Whether fidelity was non-increasing across rounds.
        drift_threshold: The fidelity-drop threshold used to set ``degraded``.
    """

    model_config = ConfigDict(extra="forbid")

    n_rounds: int
    fidelity_per_round: list[float]
    faithfulness: float
    degraded: bool
    monotonic_decline: bool
    drift_threshold: float


# --------------------------------------------------------------------------- #
# A/A control + whole experiment (orchestrator harness)
# --------------------------------------------------------------------------- #
class AAResult(BaseModel):
    """The A/A noise-floor control (FR5.1, exit criterion #2).

    Runs the reference arm under two different seeds (identical config) and tests
    for *no* significant difference on the primary endpoint — the floor any real
    effect must clear.

    Attributes:
        reference_arm: The arm run twice.
        seed_a: First seed.
        seed_b: Second seed.
        value_a: Primary-endpoint per-arm value under seed_a.
        value_b: Primary-endpoint per-arm value under seed_b.
        abs_difference: ``|value_a - value_b|`` — the noise-floor magnitude.
        significant: Whether the difference is statistically significant
            (expected ``False``: it is a noise floor).
        test: The test used for the verdict.
    """

    model_config = ConfigDict(extra="forbid")

    reference_arm: str
    seed_a: int
    seed_b: int
    value_a: float
    value_b: float
    abs_difference: float
    significant: bool
    test: TestResult | None = None


class PrimaryEndpoint(BaseModel):
    """The computed primary endpoint with its verdict (FR5.5, DX3).

    Attributes:
        name: The endpoint name (must equal the prereg's).
        description: Echo of the prereg description.
        treatment_arm: Treatment arm.
        baseline_arm: Baseline arm.
        value: The point estimate of the primary endpoint (e.g. paired mean ACC
            difference, treatment - baseline).
        difference_ci: Bootstrap CI of the endpoint.
        effect: Standardized effect size with CI.
        test: The pre-specified paired test result.
        noise_floor: The A/A absolute difference the effect must exceed.
        exceeds_noise_floor: Whether ``|value|`` exceeds ``noise_floor``.
        verdict: One of ``"confirmed"``, ``"refuted"``, ``"inconclusive"``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    treatment_arm: str
    baseline_arm: str
    value: float
    difference_ci: BootstrapCI
    effect: EffectSize
    test: TestResult
    noise_floor: float
    exceeds_noise_floor: bool
    verdict: Literal["confirmed", "refuted", "inconclusive"]


class ExperimentResult(BaseModel):
    """The complete Phase 4 experiment output (orchestrator harness).

    Attributes:
        experiment: The experiment name.
        scenario: The shared CL scenario (no cross-scenario aggregation).
        stream_id: The shared stream id all arms ran on.
        arms_run: The arm names that ran.
        seeds: The seeds used.
        arm_results: Every (arm, seed) :class:`ArmResult`.
        budget_report: The matched-budget verdict + Pareto frontier.
        stats: The statistics suite over the grid.
        aa: The A/A noise-floor control.
        primary_endpoint: The prereg primary endpoint with its verdict, or
            ``None`` if it was not computed.
        prereg: The parsed preregistration the analysis was bound to.
        stability: The temperature-0 stability control.
        drift: The memory-drift detector result.
        coverage_notes: DX2 honesty log of grid coverage and any dropped cells.
    """

    model_config = ConfigDict(extra="forbid")

    experiment: str
    scenario: str
    stream_id: str
    arms_run: list[str]
    seeds: list[int]
    arm_results: list[ArmResult]
    budget_report: BudgetReport
    stats: StatsReport
    aa: AAResult
    primary_endpoint: PrimaryEndpoint | None
    prereg: Preregistration
    stability: StabilityResult
    drift: DriftResult
    coverage_notes: list[str] = Field(default_factory=list)
