"""Shared result models for the Phase 5 experiment grid, analysis & plots.

This module is the **cross-module contract** for Phase 5 (PRD §8, Phase 5; see
``docs/PHASE5_CONTRACT.md``). Phase 5 layers three things on top of the Phase 4
nine-arm harness (:func:`slow_wave.eval.harness.build_experiment`):

* a **sweep runner** (WS1, ``slow_wave/eval/grid.py``) that runs the arm ×
  **distractor-regime** × seed grid (EC1), a **stream-length sweep** for the
  long-context crossover (EC6), and **sim-vs-real** long-horizon vs accelerated
  runs (EC3), serializing everything into one :class:`Phase5Result`;
* an **analysis** layer (WS2, ``slow_wave/eval/analysis.py``) that reads a
  :class:`Phase5Result` and produces the :class:`AnalysisReport` — the primary
  endpoint verdict surfaced exactly as preregistered (EC5), the long-context
  crossover verdict (EC6), the TMR-style targeting effect (FR5.3), the realized
  power vs the committed analysis (EC2), and the negative-result-form mapping
  (EC7);
* a **figures** layer (WS3, ``slow_wave/paper/figures.py``) that regenerates the
  deliverable figures from a committed :class:`Phase5Result` (EC4).

Design principles (inherited from Phase 0–4 — non-negotiable)
-------------------------------------------------------------
* **JSON-first & deterministic (DX1).** Every model is JSON-serializable with
  stable key order. All non-LLM fields are exact (counts, fractions) so two runs
  from the same config + seed are byte-identical under the mock LLM.
* **Honesty by construction (DX2).** Anything bounded (a dropped cell, a sweep
  cap, an unmatched budget) is surfaced as an explicit field + ``coverage_notes``
  line, never dropped silently.
* **No overclaiming (DX5).** Every report carries a ``mock_llm_caveat`` so the
  H1/H0 verdict is read as a **mechanism demonstration in the synthetic + mock
  regime**, never as a scientific claim about a real Claude model.
* **``extra="forbid"`` everywhere** so a typo'd field fails loudly.

The Phase 4 :class:`~slow_wave.eval.schema.ExperimentResult` is reused verbatim
as the per-cell unit (one regime = one full nine-arm experiment); Phase 5 only
adds the *across-cell* models below. Import these models from here and **do not
redefine them**.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from slow_wave.eval.schema import PrimaryEndpoint

# --------------------------------------------------------------------------- #
# WS1-owned: raw sweep data (the grid runner populates these)
# --------------------------------------------------------------------------- #


class RegimeMix(BaseModel):
    """A distractor-regime label mix (one point in the EC1 regime sweep).

    Attributes:
        name: Human-readable regime name (e.g. ``"distractor_heavy"``).
        signal: Target fraction of ``signal`` items.
        distractor: Target fraction of ``distractor`` items.
        noise: Target fraction of ``noise`` items.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    signal: float
    distractor: float
    noise: float


class RegimeCell(BaseModel):
    """Summary of one regime's full nine-arm experiment (one EC1 grid column).

    Carries the *summary* arrays the figures and analysis need (per-arm per-seed
    ACC and prune-F1, the regime's primary-endpoint verdict, the A/A floor) plus
    a relative pointer to the full per-cell :class:`ExperimentResult` manifest on
    disk. The full manifest is not embedded (it is large); the canonical
    committed artifact figures read is the enclosing :class:`Phase5Result`.

    Attributes:
        regime: The regime's label mix.
        manifest_path: Repo-relative path to the per-cell experiment manifest.
        arms: Arms run in this cell (report order).
        seeds: Seeds used (aligned with the per-seed lists below).
        acc_by_arm: ``{arm -> [per-seed final ACC]}`` (aligned to ``seeds``).
        prune_precision_by_arm: ``{arm -> [per-seed prune precision]}``.
        prune_recall_by_arm: ``{arm -> [per-seed prune recall]}``.
        prune_f1_by_arm: ``{arm -> [per-seed prune F1]}``.
        signal_retention_by_arm: ``{arm -> [per-seed signal_retention]}`` (for
            the TMR-style targeting effect, FR5.3).
        total_tokens_by_arm: ``{arm -> [per-seed total tokens]}`` (cost axis).
        memory_vectors_by_arm: ``{arm -> [per-seed final active vector count]}``.
        primary_endpoint: The regime's full preregistered primary endpoint
            (value + CI + test + effect + verdict, computed by the harness exactly
            as registered), or ``None`` if not computed for this regime. This is
            the authoritative EC5 object; the scalar fields below mirror it for
            convenience.
        primary_value: The regime's primary-endpoint point estimate (mean paired
            ACC diff, treatment − baseline), or ``None`` if not computed.
        primary_verdict: ``"confirmed"``/``"refuted"``/``"inconclusive"`` for this
            regime, or ``None``.
        primary_ci_lo: Lower bound of the regime's primary-endpoint bootstrap CI.
        primary_ci_hi: Upper bound of the regime's primary-endpoint bootstrap CI.
        aa_abs_difference: The A/A noise-floor magnitude for this regime.
        aa_significant: Whether the A/A difference was significant (expected
            ``False`` — it is a noise floor).
        budget_matched: Whether the matched-budget controller matched this regime.
    """

    model_config = ConfigDict(extra="forbid")

    regime: RegimeMix
    manifest_path: str
    arms: list[str]
    seeds: list[int]
    acc_by_arm: dict[str, list[float]]
    prune_precision_by_arm: dict[str, list[float]]
    prune_recall_by_arm: dict[str, list[float]]
    prune_f1_by_arm: dict[str, list[float]]
    signal_retention_by_arm: dict[str, list[float]]
    total_tokens_by_arm: dict[str, list[float]]
    memory_vectors_by_arm: dict[str, list[float]]
    primary_endpoint: PrimaryEndpoint | None = None
    primary_value: float | None = None
    primary_verdict: str | None = None
    primary_ci_lo: float | None = None
    primary_ci_hi: float | None = None
    aa_abs_difference: float = 0.0
    aa_significant: bool = False
    budget_matched: bool = False


class GridResult(BaseModel):
    """The arm × distractor-regime × seed grid (EC1).

    Attributes:
        regimes: One :class:`RegimeCell` per swept distractor regime (>= 3).
        arms: The arm names run in every cell.
        seeds: The seed list run in every cell (>= 5; both seed types vary).
        primary_regime: Name of the regime whose primary-endpoint verdict is the
            headline (the registered distractor-heavy regime).
        coverage_notes: DX2 honesty log of cells run and any dropped (none).
    """

    model_config = ConfigDict(extra="forbid")

    regimes: list[RegimeCell]
    arms: list[str]
    seeds: list[int]
    primary_regime: str
    coverage_notes: list[str] = Field(default_factory=list)


class LengthPoint(BaseModel):
    """One stream-length point in the long-context crossover sweep (EC6 data).

    Attributes:
        n_tasks: The stream length L (number of task segments) at this point.
        items_per_task: Items per task at this point (held fixed across L).
        arms: Key arms measured at this length (e.g. full_dream, long_context).
        seeds: Seeds aligned with the per-seed lists.
        acc_by_arm: ``{arm -> [per-seed ACC]}`` at this length.
        total_tokens_by_arm: ``{arm -> [per-seed total tokens]}`` at this length.
        memory_vectors_by_arm: ``{arm -> [per-seed final vector count]}``.
    """

    model_config = ConfigDict(extra="forbid")

    n_tasks: int
    items_per_task: int
    arms: list[str]
    seeds: list[int]
    acc_by_arm: dict[str, list[float]]
    total_tokens_by_arm: dict[str, list[float]]
    memory_vectors_by_arm: dict[str, list[float]]


class LengthSweep(BaseModel):
    """The stream-length sweep feeding the long-context crossover (EC6).

    Attributes:
        treatment_arm: The consolidation arm (default ``"full_dream"``).
        baseline_arm: The stuff-it-in-context arm (default ``"long_context"``).
        points: One :class:`LengthPoint` per swept length, ascending in L.
        coverage_notes: DX2 honesty log (lengths run, any dropped).
    """

    model_config = ConfigDict(extra="forbid")

    treatment_arm: str = "full_dream"
    baseline_arm: str = "long_context"
    points: list[LengthPoint]
    coverage_notes: list[str] = Field(default_factory=list)


class RetentionCurve(BaseModel):
    """Per-seed retention curve for one arm (accuracy on each task at stream end).

    Each inner list is the **final row** of that seed's accuracy matrix
    ``R[T-1][:]`` — accuracy on task *j* after the whole stream — so the figures
    layer can draw a seed-band retention curve. Stored raw (per seed); means and
    CI bands are computed downstream so the figure caption can state *n* + method.

    Attributes:
        arm_name: The arm the curve belongs to.
        n_tasks: Number of tasks (curve length).
        final_row_per_seed: ``[[R[T-1][0..T-1] for each seed], ...]``.
        seeds: Seeds aligned with ``final_row_per_seed``.
    """

    model_config = ConfigDict(extra="forbid")

    arm_name: str
    n_tasks: int
    final_row_per_seed: list[list[float]]
    seeds: list[int]


class SimRealArm(BaseModel):
    """One arm's accelerated-sim vs real-long-horizon comparison (EC3).

    Attributes:
        arm_name: The arm.
        acc_sim_per_seed: Per-seed ACC on the accelerated (short, high-
            compression) sim-time stream.
        acc_real_per_seed: Per-seed ACC on the real (long, low-compression)
            long-horizon stream.
        retention_sim: The sim retention curve (final-row R per seed).
        retention_real: The real retention curve (final-row R per seed).
    """

    model_config = ConfigDict(extra="forbid")

    arm_name: str
    acc_sim_per_seed: list[float]
    acc_real_per_seed: list[float]
    retention_sim: RetentionCurve
    retention_real: RetentionCurve


class SimRealResult(BaseModel):
    """Sim-vs-real agreement: does time-compression distort retention? (EC3).

    The bench iterates on **accelerated sim-time** (short streams, high
    compression factor); a few **real long-horizon** runs (long streams, low
    compression) validate that the compression does not distort the retention
    curves or the arm ranking. A documented inversion at scale is a *finding*,
    not a failure (EC3).

    Attributes:
        arms: Per-arm sim/real comparisons.
        sim_n_tasks: Stream length used for the accelerated sim runs.
        sim_compression: ``sim_time.compression_factor`` of the sim runs.
        real_n_tasks: Stream length used for the real long-horizon runs.
        real_compression: ``sim_time.compression_factor`` of the real runs.
        seeds: Seeds used (aligned across arms).
        pearson_agreement: Pearson r between the per-arm (mean) sim ACC vector and
            the per-arm real ACC vector — how well sim predicts real magnitude.
        spearman_agreement: Spearman rank correlation of the same vectors — how
            well sim predicts the real *ranking* of arms.
        ranking_preserved: Whether the arm ordering by ACC is identical sim→real.
        inversions: DX2 list of arm pairs whose ranking flipped sim→real.
        max_abs_acc_divergence: Max over arms of ``|mean ACC_sim − mean
            ACC_real|`` — the worst-case magnitude distortion.
        note: Honest one-line interpretation (agreement vs documented inversion).
    """

    model_config = ConfigDict(extra="forbid")

    arms: list[SimRealArm]
    sim_n_tasks: int
    sim_compression: float
    real_n_tasks: int
    real_compression: float
    seeds: list[int]
    pearson_agreement: float
    spearman_agreement: float
    ranking_preserved: bool
    inversions: list[str] = Field(default_factory=list)
    max_abs_acc_divergence: float
    note: str = ""


# --------------------------------------------------------------------------- #
# WS2-owned: analysis verdicts (the analysis layer populates these)
# --------------------------------------------------------------------------- #


class CrossoverResult(BaseModel):
    """The long-context crossover verdict (EC6), computed from a LengthSweep.

    The ``long_context`` arm keeps the whole stream (unbounded memory → never
    forgets) at maximum cost; ``full_dream`` consolidates under bounded capacity.
    The crossover is the stream length L beyond which consolidation is *worth its
    cost* — i.e. its accuracy-per-token overtakes the long-context arm's. Both a
    raw-accuracy crossover and the cost-adjusted crossover are reported; EC6 is
    satisfied either by a found crossover **or** an explicit statement of its
    absence in the swept range.

    Attributes:
        metric: The cost-adjusted axis used for the headline crossover
            (``"acc_per_token"``).
        treatment_arm: The consolidation arm (``"full_dream"``).
        baseline_arm: The stuff-it-in-context arm (``"long_context"``).
        lengths: The swept lengths L (ascending).
        acc_per_token_treatment: Treatment accuracy-per-token at each L.
        acc_per_token_baseline: Baseline accuracy-per-token at each L.
        acc_gap_baseline_minus_treatment: ``ACC(long_context) − ACC(full_dream)``
            at each L (the raw-accuracy gap the consolidation pays to close).
        crossover_length: Smallest L where the cost-adjusted metric crosses
            (treatment overtakes baseline), or ``None`` if no crossover in range.
        crossover_found: Whether a cost-adjusted crossover exists in the range.
        raw_accuracy_crossover_length: Smallest L where raw ACC(treatment) >=
            ACC(baseline), or ``None`` (long-context usually dominates on raw
            accuracy, so this is often ``None`` — that is the expected finding).
        note: Honest statement of the crossover (found at L=…) or its absence.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = "acc_per_token"
    treatment_arm: str = "full_dream"
    baseline_arm: str = "long_context"
    lengths: list[int]
    acc_per_token_treatment: list[float]
    acc_per_token_baseline: list[float]
    acc_gap_baseline_minus_treatment: list[float]
    crossover_length: int | None = None
    crossover_found: bool = False
    raw_accuracy_crossover_length: int | None = None
    note: str = ""


class TMRResult(BaseModel):
    """TMR-style targeting effect: retention lift for the replayed subset (FR5.3).

    Targeted Memory Reactivation (TMR) in sleep neuroscience boosts retention of a
    *cued* subset. The bench analogue: prioritized REPLAY preferentially
    re-potentiates the mission-relevant (``signal``) subset, so the standardized
    lift in **signal retention** for replay-enabled arms over replay-disabled arms
    is the bench's TMR effect, benchmarked against the Hu et al. (2020) TMR
    meta-analytic Hedges' g ≈ 0.29.

    Attributes:
        replay_arms: Arms whose dream cycle replays (e.g. full_dream, replay_only,
            reflection).
        no_replay_arms: Arms with no replay (e.g. no_sleep, downscale_only).
        signal_retention_replay: Pooled per-(arm,seed) signal_retention for the
            replay arms.
        signal_retention_no_replay: Pooled per-(arm,seed) signal_retention for the
            no-replay arms.
        mean_lift: ``mean(replay) − mean(no_replay)`` signal-retention lift.
        hedges_g: Standardized lift (bias-corrected Cohen's d, unpaired).
        g_ci_lo: Lower bootstrap CI bound on ``hedges_g``.
        g_ci_hi: Upper bootstrap CI bound on ``hedges_g``.
        benchmark_g: The Hu et al. (2020) TMR meta-analytic g (0.29).
        exceeds_benchmark: Whether ``hedges_g`` exceeds ``benchmark_g``.
        note: Honest one-line interpretation + the proxy caveat (this is a
            replay-targeting analogue, not a literal cued-TMR protocol).
    """

    model_config = ConfigDict(extra="forbid")

    replay_arms: list[str]
    no_replay_arms: list[str]
    signal_retention_replay: list[float]
    signal_retention_no_replay: list[float]
    mean_lift: float
    hedges_g: float
    g_ci_lo: float
    g_ci_hi: float
    benchmark_g: float = 0.29
    exceeds_benchmark: bool = False
    note: str = ""


class PowerReport(BaseModel):
    """Realized power vs the committed power analysis (EC2; Colas et al. 2018).

    Attributes:
        n_seeds: Number of paired seeds used.
        floor: The preregistered seed floor (5).
        floor_met: Whether ``n_seeds >= floor``.
        observed_effect_d: The observed standardized paired effect (Cohen's d) of
            the primary contrast.
        required_n_for_observed: Paired n needed to detect the observed d at
            alpha=0.05 (two-sided), power=0.8, per Colas et al. (2018).
        powered_for_observed: Whether ``n_seeds >= required_n_for_observed``.
        note: Honest statement of whether N is consistent with the committed
            analysis, and (if the observed effect is small) what N would be
            needed.
    """

    model_config = ConfigDict(extra="forbid")

    n_seeds: int
    floor: int = 5
    floor_met: bool = True
    observed_effect_d: float
    required_n_for_observed: int
    powered_for_observed: bool
    note: str = ""


class NegativeFormMapping(BaseModel):
    """Mapping of a negative/refuted result to a preregistered form (EC7).

    Only meaningful when the primary verdict is not ``"confirmed"`` (or when a
    secondary contrast is negative, e.g. dreaming ≯ replay-only). Ties the form to
    the datasheet regime that produced it.

    Attributes:
        applicable: Whether a negative-result form applies (primary refuted, or a
            registered secondary negative was observed).
        matched_forms: The preregistered negative-result form strings that match
            (from the prereg ``negative_result_forms``).
        regime_tie: The datasheet regime (name + label mix) the negative result is
            tied to.
        secondary_contrasts: ``{contrast_label -> short verdict}`` for the
            registered secondary negative checks (e.g. ``"full_dream vs
            replay_only"``) regardless of the primary verdict.
        note: Honest interpretation.
    """

    model_config = ConfigDict(extra="forbid")

    applicable: bool
    matched_forms: list[str] = Field(default_factory=list)
    regime_tie: str = ""
    secondary_contrasts: dict[str, str] = Field(default_factory=dict)
    note: str = ""


class AnalysisReport(BaseModel):
    """The Phase 5 written analysis, machine-readable (EC5–EC7).

    Surfaces the preregistered primary-endpoint verdict exactly as computed by the
    harness (EC5), the long-context crossover (EC6), the TMR targeting effect
    (FR5.3), the realized power (EC2), and the negative-result-form mapping (EC7).
    The companion ``paper/RESULTS.md`` is generated from this object.

    Attributes:
        primary_endpoint_name: The single preregistered primary endpoint name.
        primary_verdict: ``"confirmed"``/``"refuted"``/``"inconclusive"`` — the
            unambiguous H1/H0 verdict (EC5).
        primary_value: The primary-endpoint point estimate (mean paired ACC diff).
        primary_ci_lo: Lower bootstrap CI bound on the primary endpoint.
        primary_ci_hi: Upper bootstrap CI bound on the primary endpoint.
        primary_ci_method: The CI method string (e.g. ``"percentile bootstrap"``)
            for figure captions / DX4.
        primary_test_name: The preregistered test (e.g. ``"wilcoxon_signed_rank"``).
        primary_test_p: The test p-value.
        primary_effect_name: The standardized effect-size name (Cohen's d).
        primary_effect_value: The effect-size point estimate.
        primary_effect_magnitude: Qualitative magnitude label.
        noise_floor: The A/A noise floor the effect must exceed.
        exceeds_noise_floor: Whether the effect exceeds the floor.
        n_seeds: Paired seeds behind the verdict.
        crossover: The long-context crossover verdict (EC6).
        tmr: The TMR-style targeting effect (FR5.3).
        power: The realized-power report (EC2).
        negative: The negative-result-form mapping (EC7).
        per_regime_verdicts: ``{regime_name -> verdict}`` across the regime sweep.
        sim_real_agreement_note: One-line sim-vs-real summary (EC3).
        headline: A one-paragraph plain-language headline of the result.
        mock_llm_caveat: The mandatory honesty caveat (DX5): all numbers are a
            mechanism demonstration in the synthetic + deterministic-mock-LLM
            regime, NOT a scientific claim about a real Claude model.
    """

    model_config = ConfigDict(extra="forbid")

    primary_endpoint_name: str
    primary_verdict: Literal["confirmed", "refuted", "inconclusive", "not-computed"]
    primary_value: float | None
    primary_ci_lo: float | None
    primary_ci_hi: float | None
    primary_ci_method: str
    primary_test_name: str | None
    primary_test_p: float | None
    primary_effect_name: str | None
    primary_effect_value: float | None
    primary_effect_magnitude: str | None
    noise_floor: float
    exceeds_noise_floor: bool
    n_seeds: int
    crossover: CrossoverResult
    tmr: TMRResult
    power: PowerReport
    negative: NegativeFormMapping
    per_regime_verdicts: dict[str, str] = Field(default_factory=dict)
    sim_real_agreement_note: str = ""
    headline: str = ""
    mock_llm_caveat: str = ""


# --------------------------------------------------------------------------- #
# Top-level Phase 5 artifact (WS1 writes; WS2 fills `analysis`; WS3 reads)
# --------------------------------------------------------------------------- #


class Phase5Result(BaseModel):
    """The complete Phase 5 experiment + analysis artifact (the committed file).

    WS1 (the grid runner) writes everything except :attr:`analysis`; WS2 (the
    analysis layer) reads the file, computes :attr:`analysis`, and rewrites it;
    WS3 (figures) reads the final file to regenerate every figure (EC4). The file
    is written to ``runs/phase5/phase5_result.json`` and is the canonical
    committed artifact (one-command repro reads it).

    Attributes:
        experiment: The experiment name.
        scenario: The shared CL scenario (no cross-scenario aggregation, FR5.6).
        git_commit: The git commit the grid ran at (provenance, EC1).
        model_id: The configured model id (recorded; mock under no API key).
        model_mocked: Whether the LLM was mocked (True in this regime; DX5).
        grid: The arm × regime × seed grid (EC1).
        length_sweep: The stream-length sweep feeding the crossover (EC6).
        sim_real: The sim-vs-real agreement result (EC3).
        retention: ``{regime_name -> [RetentionCurve per arm]}`` for the retention
            figure (seed bands), at minimum for the primary regime.
        analysis: The written analysis verdicts (EC5–EC7); ``None`` until WS2
            fills it.
        coverage_notes: DX2 honesty log spanning every sweep (cells run, dropped).
        manifest_paths: Repo-relative paths to every per-cell experiment manifest
            (EC1: every run has a manifest with cost + git hash).
    """

    model_config = ConfigDict(extra="forbid")

    experiment: str
    scenario: str
    git_commit: str | None = None
    model_id: str = "claude-opus-4-8"
    model_mocked: bool = True
    grid: GridResult
    length_sweep: LengthSweep
    sim_real: SimRealResult
    retention: dict[str, list[RetentionCurve]] = Field(default_factory=dict)
    analysis: AnalysisReport | None = None
    coverage_notes: list[str] = Field(default_factory=list)
    manifest_paths: list[str] = Field(default_factory=list)
