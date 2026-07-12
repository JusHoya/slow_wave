"""Phase 5 analysis & verdict layer (WS2; EC5–EC7, FR5.3, DX3/DX5).

This module reads a :class:`~slow_wave.eval.phase5_schema.Phase5Result` produced
by the grid runner (WS1) and distils it into the machine-readable
:class:`~slow_wave.eval.phase5_schema.AnalysisReport` plus a human-readable
``paper/RESULTS.md`` writeup. It performs **no** plotting (that is WS3) and
introduces **no** new science — it surfaces the preregistered primary-endpoint
verdict exactly as the harness computed it and layers the secondary readouts the
exit criteria demand:

* **EC5 (primary endpoint).** The single preregistered endpoint
  (``acc_diff_full_dream_vs_no_sleep``) is surfaced from the primary regime
  cell's embedded :class:`~slow_wave.eval.schema.PrimaryEndpoint` — value, CI (+
  method), the preregistered test (+ p), the standardized effect (+ magnitude),
  and the A/A noise floor it must clear. The endpoint *name* is asserted against
  the committed preregistration (DX3): a non-preregistered endpoint is refused
  via :func:`~slow_wave.eval.prereg.assert_primary_endpoint` before any verdict
  is reported.
* **EC6 (long-context crossover).** From the stream-length sweep, the
  cost-adjusted (accuracy-per-token) and raw-accuracy crossovers between the
  consolidation arm and the stuff-it-in-context arm are computed; EC6 is
  satisfied by a found crossover **or** an explicit statement of its absence.
* **FR5.3 (TMR-style targeting).** The standardized signal-retention lift of the
  replay-enabled arms over the replay-disabled arms is benchmarked against the
  Hu et al. (2020) cued-TMR meta-analytic Hedges' *g* ≈ 0.29 (a replay-targeting
  *analogue*, not a literal cued-TMR protocol — DX5).
* **EC2 (realized power).** The realized seed count is compared against the
  paired-design *n* required to detect the observed effect (Colas et al. 2018).
* **EC7 (negative-result mapping).** A non-confirmed primary (or a negative
  registered secondary contrast) is mapped to one of the six preregistered
  negative-result forms (PRD §9.1) and tied to the datasheet regime that
  produced it; the ``full_dream`` vs ``replay_only`` secondary contrast is
  **always** computed.

Design principles (inherited from Phase 0–4 — non-negotiable)
-------------------------------------------------------------
* **numpy + stdlib + pydantic + slow_wave only.** All inference is delegated to
  the pure-numpy :mod:`slow_wave.eval.stats`; ``scipy``/``statsmodels``/
  ``rliable``/``matplotlib`` are never imported.
* **Determinism (DX1).** Every bootstrap draws from a generator seeded
  deterministically from ``stats_seed`` via
  :func:`~slow_wave.repro.seeding.derive_seed`, so the report is byte-identical
  on a re-run.
* **No overclaiming (DX5).** Every report carries the mandatory
  ``mock_llm_caveat``: all numbers are a **mechanism demonstration** in the
  synthetic + deterministic-mock-LLM regime, never a scientific claim about a
  real Claude model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

from slow_wave.eval import stats as swstats
from slow_wave.eval.phase5_schema import (
    AnalysisReport,
    CrossoverResult,
    LengthSweep,
    NegativeFormMapping,
    Phase5Result,
    PowerReport,
    RegimeCell,
    SimRealResult,
    TMRResult,
)
from slow_wave.eval.prereg import (
    assert_primary_endpoint,
    load_preregistration,
)
from slow_wave.eval.schema import PrimaryEndpoint, Preregistration
from slow_wave.repro.manifest import read_manifest
from slow_wave.repro.seeding import derive_seed

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants pinned by the contract / preregistration
# --------------------------------------------------------------------------- #
_DEFAULT_PREREG = "prereg/preregistration.yaml"

#: Standard normal quantiles for the Colas et al. (2018) paired-n formula.
_Z_975 = 1.959964  # z_{0.975} (two-sided alpha = 0.05)
_Z_80 = 0.841621  # z_{0.8}   (power = 0.8)

#: Replay-enabled vs replay-disabled arms for the TMR-style targeting effect.
_TMR_REPLAY_ARMS = ("full_dream", "replay_only", "reflection")
_TMR_NO_REPLAY_ARMS = ("no_sleep", "downscale_only")

#: Hu et al. (2020) cued-TMR meta-analytic Hedges' g.
_BENCHMARK_G = 0.29

#: Preregistered seed floor (FR5.4); a sentinel required-n for a vanishing effect.
_SEED_FLOOR = 5
_REQUIRED_N_SENTINEL = 9999

#: The registered secondary contrast that is always computed (PRD §9.1 #1).
_SECONDARY_LABEL = "full_dream vs replay_only"

#: The mandatory DX5 honesty caveat surfaced everywhere a number is reported.
MOCK_LLM_CAVEAT = (
    "MECHANISM DEMONSTRATION ONLY. Every number in this analysis was produced in "
    "the synthetic fact-world stream under the deterministic mock LLM (no "
    "ANTHROPIC_API_KEY on this box), so it is reproducible bit-for-bit. The H1/H0 "
    "verdict is therefore a statement about the bench mechanism in this synthetic "
    "+ mock-LLM regime, NOT a scientific claim about a real Claude model. "
    "Neuroscience (sleep, replay, TMR) is motivation and analogy only, never a "
    "proof of biological fidelity or real-model efficacy (DX5)."
)


# --------------------------------------------------------------------------- #
# Primary regime cell + endpoint resolution (EC5, DX3)
# --------------------------------------------------------------------------- #
def _primary_cell(result: Phase5Result) -> RegimeCell:
    """Return the registered primary-regime :class:`RegimeCell`.

    Falls back to the first cell (with a DX2 warning) if the named primary
    regime is missing, and raises only when the grid has no cells at all.

    Args:
        result: The Phase 5 artifact to analyze.

    Returns:
        The primary regime's :class:`RegimeCell`.

    Raises:
        ValueError: If the grid carries no regime cells.
    """
    name = result.grid.primary_regime
    for cell in result.grid.regimes:
        if cell.regime.name == name:
            return cell
    if result.grid.regimes:
        logger.warning(
            "primary regime %r not found in grid; falling back to first cell %r (DX2).",
            name,
            result.grid.regimes[0].regime.name,
        )
        return result.grid.regimes[0]
    raise ValueError("Phase5Result grid carries no regime cells; cannot analyze.")


def _endpoint_from_manifest(manifest_path: str) -> PrimaryEndpoint | None:
    """Best-effort load of a cell's primary endpoint from its manifest on disk.

    The harness serializes the whole :class:`ExperimentResult` under
    ``manifest.results['experiment']``; the primary endpoint (if computed) lives
    at ``...['primary_endpoint']``. Any I/O or validation failure is swallowed
    and reported as ``None`` so the analysis degrades to ``not-computed`` rather
    than crashing (DX2).

    Args:
        manifest_path: Repo-relative path to the per-cell experiment manifest.

    Returns:
        The parsed :class:`PrimaryEndpoint`, or ``None`` if it is absent /
        unreadable.
    """
    try:
        manifest = read_manifest(manifest_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning(
            "could not read per-cell manifest %r for the primary endpoint fallback: %s",
            manifest_path,
            exc,
        )
        return None
    raw = manifest.results.get("experiment", {}).get("primary_endpoint")
    if raw is None:
        return None
    try:
        return PrimaryEndpoint.model_validate(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        logger.warning("manifest %r primary_endpoint did not validate: %s", manifest_path, exc)
        return None


def _resolve_primary_endpoint(
    cell: RegimeCell, prereg: Preregistration
) -> PrimaryEndpoint | None:
    """Surface the cell's full :class:`PrimaryEndpoint`, asserting its name (DX3).

    Reads the endpoint the harness embedded on the cell; if it is absent, falls
    back to the per-cell manifest. Whenever an endpoint is found, its *name* is
    validated against the committed preregistration via
    :func:`~slow_wave.eval.prereg.assert_primary_endpoint` — refusing a
    non-preregistered endpoint before any verdict is surfaced.

    Args:
        cell: The primary regime cell.
        prereg: The parsed preregistration binding the analysis.

    Returns:
        The resolved :class:`PrimaryEndpoint`, or ``None`` if neither the cell
        nor its manifest carries one.

    Raises:
        slow_wave.eval.prereg.NonPreregisteredEndpointError: If a resolved
            endpoint's name is not the prereg's single primary endpoint (DX3).
    """
    endpoint = cell.primary_endpoint
    if endpoint is None:
        endpoint = _endpoint_from_manifest(cell.manifest_path)
    if endpoint is not None:
        # DX3: refuse a non-preregistered endpoint before reporting any verdict.
        assert_primary_endpoint(prereg, endpoint.name)
    return endpoint


# --------------------------------------------------------------------------- #
# EC6 — long-context crossover
# --------------------------------------------------------------------------- #
def _compute_crossover(sweep: LengthSweep) -> CrossoverResult:
    """Compute the cost-adjusted + raw long-context crossover verdict (EC6).

    For each swept length L (ascending) the treatment and baseline
    accuracy-per-token are ``mean(ACC) / max(mean(total_tokens), 1)`` (the
    divide-by-zero guard). The headline crossover is the smallest L at which the
    treatment's accuracy-per-token overtakes the baseline's; the raw-accuracy
    crossover is the smallest L at which raw treatment ACC reaches the baseline's.
    EC6 is satisfied by a found crossover **or** an explicit absence statement.

    Args:
        sweep: The stream-length sweep (EC6 data) from the Phase 5 artifact.

    Returns:
        The populated :class:`CrossoverResult` with an honest ``note``.
    """
    treatment = sweep.treatment_arm
    baseline = sweep.baseline_arm
    points = sorted(sweep.points, key=lambda p: p.n_tasks)

    lengths: list[int] = []
    apt_treatment: list[float] = []
    apt_baseline: list[float] = []
    acc_gap: list[float] = []
    crossover_length: int | None = None
    raw_crossover_length: int | None = None

    for point in points:
        if treatment not in point.acc_by_arm or baseline not in point.acc_by_arm:
            continue
        acc_t = float(np.mean(point.acc_by_arm[treatment]))
        acc_b = float(np.mean(point.acc_by_arm[baseline]))
        tok_t = float(np.mean(point.total_tokens_by_arm.get(treatment, [0.0]) or [0.0]))
        tok_b = float(np.mean(point.total_tokens_by_arm.get(baseline, [0.0]) or [0.0]))
        apt_t = acc_t / max(tok_t, 1.0)
        apt_b = acc_b / max(tok_b, 1.0)

        lengths.append(point.n_tasks)
        apt_treatment.append(apt_t)
        apt_baseline.append(apt_b)
        acc_gap.append(acc_b - acc_t)

        if crossover_length is None and apt_t > apt_b:
            crossover_length = point.n_tasks
        if raw_crossover_length is None and acc_t >= acc_b:
            raw_crossover_length = point.n_tasks

    found = crossover_length is not None
    if not lengths:
        note = (
            f"no length-sweep point carries both {treatment!r} and {baseline!r}; "
            f"the long-context crossover is not evaluable (DX2)."
        )
    elif found:
        raw_txt = (
            f" Raw-accuracy crossover at L={raw_crossover_length}."
            if raw_crossover_length is not None
            else f" No raw-accuracy crossover in {lengths} ({baseline} keeps the raw-ACC lead)."
        )
        note = (
            f"cost-adjusted crossover found at L={crossover_length}: {treatment} "
            f"accuracy-per-token overtakes {baseline} within the swept range "
            f"{lengths}." + raw_txt
        )
    else:
        note = (
            f"no cost-adjusted crossover in the swept range {lengths}; {baseline} "
            f"dominates on raw accuracy at maximum cost (EC6: absence stated)."
        )

    logger.info(
        "crossover: found=%s at L=%s (raw L=%s) over lengths %s",
        found,
        crossover_length,
        raw_crossover_length,
        lengths,
    )
    return CrossoverResult(
        metric="acc_per_token",
        treatment_arm=treatment,
        baseline_arm=baseline,
        lengths=lengths,
        acc_per_token_treatment=apt_treatment,
        acc_per_token_baseline=apt_baseline,
        acc_gap_baseline_minus_treatment=acc_gap,
        crossover_length=crossover_length,
        crossover_found=found,
        raw_accuracy_crossover_length=raw_crossover_length,
        note=note,
    )


# --------------------------------------------------------------------------- #
# FR5.3 — TMR-style targeting effect
# --------------------------------------------------------------------------- #
def _hedges_correction(n1: int, n2: int) -> float:
    """Return the small-sample Hedges' bias correction ``1 - 3/(4(n1+n2)-9)``.

    Guards the degenerate ``4(n1+n2)-9 <= 0`` case (only for tiny pooled
    samples) by returning ``1.0`` (no correction).

    Args:
        n1: Size of the replay pool.
        n2: Size of the no-replay pool.

    Returns:
        The multiplicative bias-correction factor in ``(0, 1]``.
    """
    denom = 4 * (n1 + n2) - 9
    return 1.0 - 3.0 / denom if denom > 0 else 1.0


def _compute_tmr(cell: RegimeCell, stats_seed: int) -> TMRResult:
    """Compute the replay-vs-no-replay signal-retention targeting effect (FR5.3).

    Pools per-(arm, seed) ``signal_retention`` for the replay-enabled arms
    (``full_dream``/``replay_only``/``reflection``) against the replay-disabled
    arms (``no_sleep``/``downscale_only``), all from the primary regime cell. The
    standardized lift is a bias-corrected **unpaired** Cohen's d (Hedges' g): the
    point estimate is :func:`slow_wave.eval.stats.cohens_d` (``paired=False``)
    times the Hedges factor, and the CI reuses the bootstrap CI of unpaired
    Cohen's d from :func:`~slow_wave.eval.stats.effect_size_cohens_d` scaled by
    the same factor (a documented approximation). Benchmarked against
    ``g = 0.29`` (Hu et al. 2020).

    Args:
        cell: The primary regime cell carrying ``signal_retention_by_arm``.
        stats_seed: Master seed for the bootstrap CI (DX1).

    Returns:
        The populated :class:`TMRResult` (proxy caveat in ``note``).
    """
    replay_arms = [a for a in _TMR_REPLAY_ARMS if a in cell.signal_retention_by_arm]
    no_replay_arms = [a for a in _TMR_NO_REPLAY_ARMS if a in cell.signal_retention_by_arm]
    replay = [v for a in replay_arms for v in cell.signal_retention_by_arm[a]]
    no_replay = [v for a in no_replay_arms for v in cell.signal_retention_by_arm[a]]
    n1, n2 = len(replay), len(no_replay)

    proxy = (
        "Proxy caveat: this is a prioritized-replay targeting ANALOGUE, not a "
        "literal cued-TMR protocol."
    )
    if n1 == 0 or n2 == 0:
        note = (
            "insufficient replay/no-replay signal_retention samples to estimate the "
            f"TMR-style effect; reported as zero (DX2). {proxy}"
        )
        logger.warning("TMR effect: n1=%d n2=%d -> not estimable", n1, n2)
        return TMRResult(
            replay_arms=replay_arms,
            no_replay_arms=no_replay_arms,
            signal_retention_replay=replay,
            signal_retention_no_replay=no_replay,
            mean_lift=0.0,
            hedges_g=0.0,
            g_ci_lo=0.0,
            g_ci_hi=0.0,
            benchmark_g=_BENCHMARK_G,
            exceeds_benchmark=False,
            note=note,
        )

    a = np.asarray(replay, dtype=float)
    b = np.asarray(no_replay, dtype=float)
    mean_lift = float(np.mean(a) - np.mean(b))
    correction = _hedges_correction(n1, n2)
    hedges_g = float(swstats.cohens_d(a, b, paired=False) * correction)

    rng = np.random.default_rng(derive_seed(stats_seed, "tmr"))
    effect = swstats.effect_size_cohens_d(a, b, paired=False, rng=rng)
    g_ci_lo = float(effect.lo * correction)
    g_ci_hi = float(effect.hi * correction)
    exceeds = hedges_g > _BENCHMARK_G

    note = (
        f"replay arms {replay_arms} vs no-replay {no_replay_arms}: mean signal-"
        f"retention lift={mean_lift:+.3f}, bias-corrected unpaired Hedges' "
        f"g={hedges_g:.3f} (95% CI [{g_ci_lo:.3f}, {g_ci_hi:.3f}], bootstrap on "
        f"unpaired Cohen's d x Hedges factor). Benchmark Hu et al. (2020) "
        f"g={_BENCHMARK_G}: {'exceeds' if exceeds else 'does not exceed'} it. {proxy}"
    )
    logger.info("TMR effect: g=%.4f ci=[%.4f, %.4f] exceeds=%s", hedges_g, g_ci_lo, g_ci_hi, exceeds)
    return TMRResult(
        replay_arms=replay_arms,
        no_replay_arms=no_replay_arms,
        signal_retention_replay=replay,
        signal_retention_no_replay=no_replay,
        mean_lift=mean_lift,
        hedges_g=hedges_g,
        g_ci_lo=g_ci_lo,
        g_ci_hi=g_ci_hi,
        benchmark_g=_BENCHMARK_G,
        exceeds_benchmark=exceeds,
        note=note,
    )


# --------------------------------------------------------------------------- #
# EC2 — realized power
# --------------------------------------------------------------------------- #
def _compute_power(n_seeds: int, observed_d: float, *, computed: bool) -> PowerReport:
    """Compare realized N against the n required for the observed effect (EC2).

    Uses the Colas et al. (2018) paired-design formula
    ``n = ceil((z_{0.975} + z_{0.8})^2 / d^2) + 1``. A vanishing effect (``d ->
    0``) yields a large sentinel required-n with an explanatory note.

    Args:
        n_seeds: The realized number of paired seeds.
        observed_d: ``|primary effect Cohen's d|``.
        computed: Whether the primary endpoint was actually computed (drives the
            sentinel note wording).

    Returns:
        The populated :class:`PowerReport`.
    """
    floor_met = n_seeds >= _SEED_FLOOR
    d = abs(float(observed_d))
    if d < 1e-9:
        required_n = _REQUIRED_N_SENTINEL
        if computed:
            note = (
                f"observed paired effect d~0; required n set to the sentinel "
                f"{required_n} (no feasible N detects a vanishing effect)."
            )
        else:
            note = (
                f"primary endpoint not computed, so the observed effect is "
                f"unavailable; required n set to the sentinel {required_n}."
            )
    else:
        required_n = math.ceil(((_Z_975 + _Z_80) ** 2) / (d**2)) + 1
        note = (
            f"to detect the observed paired d={d:.3f} at alpha=0.05 (two-sided), "
            f"power=0.8, Colas et al. (2018) requires n>={required_n} paired "
            f"seeds; the run used n={n_seeds}."
        )
    powered = n_seeds >= required_n
    if not floor_met:
        note += f" WARNING: n_seeds={n_seeds} is below the preregistered floor of {_SEED_FLOOR}."
    logger.info(
        "power: n=%d required=%d powered=%s floor_met=%s (d=%.4f)",
        n_seeds,
        required_n,
        powered,
        floor_met,
        d,
    )
    return PowerReport(
        n_seeds=n_seeds,
        floor=_SEED_FLOOR,
        floor_met=floor_met,
        observed_effect_d=d,
        required_n_for_observed=required_n,
        powered_for_observed=powered,
        note=note,
    )


# --------------------------------------------------------------------------- #
# EC7 — negative-result mapping
# --------------------------------------------------------------------------- #
def _match_form(forms: list[str], needle: str) -> str | None:
    """Return the first preregistered negative form containing ``needle``."""
    low = needle.lower()
    for form in forms:
        if low in form.lower():
            return form
    return None


def _compute_negative(
    cell: RegimeCell,
    prereg: Preregistration,
    primary_verdict: str,
    crossover: CrossoverResult,
    stats_seed: int,
) -> NegativeFormMapping:
    """Map a non-confirmed/secondary-negative result to a prereg form (EC7).

    Always computes the registered ``full_dream`` vs ``replay_only`` secondary
    contrast (PRD §9.1 #1) from the primary cell's per-seed ACC via
    :func:`~slow_wave.eval.stats.paired_diff_ci` +
    :func:`~slow_wave.eval.stats.wilcoxon_signed_rank`. The mapping is
    ``applicable`` when the primary verdict is not ``"confirmed"`` OR the
    secondary contrast is negative; matched forms are drawn from
    ``prereg.negative_result_forms`` and the result is tied to the datasheet
    regime.

    Args:
        cell: The primary regime cell (per-seed ACC + regime mix).
        prereg: The parsed preregistration (negative-result forms).
        primary_verdict: The surfaced primary verdict.
        crossover: The computed crossover verdict (drives the long-context form).
        stats_seed: Master seed for the secondary-contrast bootstrap (DX1).

    Returns:
        The populated :class:`NegativeFormMapping`.
    """
    forms = list(prereg.negative_result_forms)
    secondary_contrasts: dict[str, str] = {}
    secondary_negative = False
    sec_note = ""

    if "full_dream" in cell.acc_by_arm and "replay_only" in cell.acc_by_arm:
        a = np.asarray(cell.acc_by_arm["full_dream"], dtype=float)
        b = np.asarray(cell.acc_by_arm["replay_only"], dtype=float)
        m = min(a.size, b.size)
        a, b = a[:m], b[:m]
        value = float(np.mean(a - b)) if m else 0.0
        rng = np.random.default_rng(derive_seed(stats_seed, "neg_full_dream_vs_replay_only"))
        ci = swstats.paired_diff_ci(a, b, rng=rng)
        test = swstats.wilcoxon_signed_rank(a, b)
        excludes_zero = ci.lo > 0.0 or ci.hi < 0.0
        if value > 0.0 and excludes_zero:
            verdict = (
                f"full_dream > replay_only (delta={value:+.3f}, 95% CI "
                f"[{ci.lo:.3f}, {ci.hi:.3f}], wilcoxon p={test.p_value:.3f})"
            )
        else:
            secondary_negative = True
            verdict = (
                f"full_dream NOT > replay_only (delta={value:+.3f}, 95% CI "
                f"[{ci.lo:.3f}, {ci.hi:.3f}] includes/<=0, wilcoxon p={test.p_value:.3f})"
            )
        secondary_contrasts[_SECONDARY_LABEL] = verdict
    else:
        secondary_contrasts[_SECONDARY_LABEL] = (
            "not-computed (replay_only and/or full_dream arm absent from the primary cell)"
        )
        sec_note = "replay_only arm absent; the secondary contrast is not evaluable. "

    applicable = (primary_verdict != "confirmed") or secondary_negative

    matched: list[str] = []
    if secondary_negative:
        form = _match_form(forms, "replay-only")
        if form:
            matched.append(form)
    if not crossover.crossover_found:
        form = _match_form(forms, "Long-context wins")
        if form:
            matched.append(form)
    if applicable and not matched:
        # Guarantee at least one tied form when a negative result applies.
        form = _match_form(forms, "replay-only") or (forms[0] if forms else None)
        if form:
            matched.append(form)
    matched = list(dict.fromkeys(matched))  # dedup, preserve order

    regime = cell.regime
    regime_tie = (
        f"{regime.name} (signal={regime.signal}, distractor={regime.distractor}, "
        f"noise={regime.noise})"
    )

    if applicable:
        note = (
            sec_note
            + f"negative-result form(s) apply and are tied to regime "
            f"{regime.name!r}. Primary verdict={primary_verdict!r}; the registered "
            f"secondary full_dream-vs-replay_only contrast is "
            f"{'NEGATIVE' if secondary_negative else 'positive'}."
        )
    elif matched:
        # Primary confirmed + positive secondary, but a registered negative
        # PATTERN is still observed (e.g. no long-context crossover ⇒ the
        # "Long-context wins" form). Surface it honestly instead of claiming
        # nothing matched — it is a secondary cost/length trade-off, not a
        # refutation of the primary-endpoint contrast (DX2/DX5).
        forms_str = "; ".join(repr(f) for f in matched)
        note = (
            sec_note
            + "no negative-result form OVERTURNS H1: the primary endpoint is confirmed "
            "and the registered secondary contrast is positive. However, the following "
            f"preregistered negative pattern(s) ARE observed and reported (see the "
            f"crossover / EC6 section): {forms_str}. These concern secondary cost/length "
            "trade-offs, not the primary-endpoint contrast (reported on the "
            "accuracy-vs-compute Pareto frontier, since budgets were not matched "
            "within tolerance)."
        )
    else:
        note = (
            sec_note
            + "no negative-result form applies: the primary endpoint is confirmed and "
            "the registered secondary contrast is positive."
        )
    logger.info(
        "negative mapping: applicable=%s matched=%d secondary_negative=%s",
        applicable,
        len(matched),
        secondary_negative,
    )
    return NegativeFormMapping(
        applicable=applicable,
        matched_forms=matched,
        regime_tie=regime_tie,
        secondary_contrasts=secondary_contrasts,
        note=note,
    )


# --------------------------------------------------------------------------- #
# Cross-cell + sim/real summaries
# --------------------------------------------------------------------------- #
def _cell_verdict(cell: RegimeCell) -> str:
    """Return a regime cell's primary verdict (mirror, endpoint, or not-computed)."""
    if cell.primary_verdict:
        return cell.primary_verdict
    if cell.primary_endpoint is not None:
        return cell.primary_endpoint.verdict
    return "not-computed"


def _sim_real_note(sim_real: SimRealResult) -> str:
    """Return the sim-vs-real one-liner (EC3), generating one if WS1 left it blank."""
    if sim_real.note.strip():
        return sim_real.note
    return (
        f"sim-vs-real agreement: pearson={sim_real.pearson_agreement:.3f}, "
        f"spearman={sim_real.spearman_agreement:.3f}, "
        f"ranking_preserved={sim_real.ranking_preserved}, "
        f"max|delta ACC|={sim_real.max_abs_acc_divergence:.3f}."
    )


def _headline(
    regime_name: str,
    primary_verdict: str,
    primary_value: float | None,
    exceeds_noise_floor: bool,
    crossover: CrossoverResult,
    tmr: TMRResult,
    power: PowerReport,
) -> str:
    """Compose a plain-language, non-overclaiming one-paragraph headline."""
    verdict_map = {
        "confirmed": f"CONFIRMED H1 in the {regime_name!r} regime",
        "refuted": f"REFUTED H1 (failed to reject H0) in the {regime_name!r} regime",
        "inconclusive": f"INCONCLUSIVE in the {regime_name!r} regime (budgets unmatched)",
        "not-computed": f"primary endpoint NOT COMPUTED for the {regime_name!r} regime",
    }
    verdict_txt = verdict_map.get(primary_verdict, primary_verdict)
    val_txt = (
        f" (mean paired ACC diff full_dream - no_sleep = {primary_value:+.3f}, "
        f"{'clears' if exceeds_noise_floor else 'within'} the A/A noise floor)"
        if primary_value is not None
        else ""
    )
    cross_txt = (
        f"a long-context crossover appears at L={crossover.crossover_length}"
        if crossover.crossover_found
        else "no long-context crossover appears in the swept range (long_context "
        "keeps the raw-accuracy lead at maximum cost)"
    )
    tmr_txt = (
        f"the TMR-style replay-targeting effect is g={tmr.hedges_g:.3f} versus the "
        f"0.29 benchmark ({'exceeds' if tmr.exceeds_benchmark else 'does not exceed'})"
    )
    power_txt = (
        f"realized N={power.n_seeds} (floor {power.floor} "
        f"{'met' if power.floor_met else 'NOT met'}); the observed effect needs "
        f"N>={power.required_n_for_observed}"
    )
    return (
        f"Mechanism demonstration under the deterministic mock LLM: {verdict_txt}"
        f"{val_txt}. On the long-horizon sweep {cross_txt}; {tmr_txt}; {power_txt}. "
        f"These are synthetic + mock-LLM mechanism results, not claims about a real "
        f"Claude model."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def analyze(
    result: Phase5Result,
    *,
    prereg_path: str = _DEFAULT_PREREG,
    stats_seed: int = 0,
) -> AnalysisReport:
    """Compute the Phase 5 analysis verdicts (EC5–EC7) from a Phase5Result.

    Surfaces the preregistered primary-endpoint verdict exactly as the harness
    computed it (EC5; the endpoint *name* is asserted against the committed
    preregistration, DX3), then layers the long-context crossover (EC6), the
    TMR-style targeting effect (FR5.3), the realized power (EC2), and the
    negative-result-form mapping (EC7). Every randomized statistic draws from a
    generator seeded deterministically from ``stats_seed`` (DX1), so the report
    is byte-identical on a re-run. The mandatory ``mock_llm_caveat`` (DX5) is
    always set.

    Args:
        result: The Phase 5 artifact (grid + length sweep + sim-vs-real) to
            analyze. ``result.analysis`` is ignored / overwritten downstream.
        prereg_path: Path to the committed ``preregistration.yaml`` that binds
            the analysis (DX3).
        stats_seed: Master seed for the bootstrap CIs (TMR + secondary contrast).

    Returns:
        The populated :class:`~slow_wave.eval.phase5_schema.AnalysisReport`.

    Raises:
        slow_wave.eval.prereg.NonPreregisteredEndpointError: If the primary
            cell's endpoint name is not the prereg's single primary endpoint
            (DX3).
        ValueError: If the grid carries no regime cells.
    """
    prereg = load_preregistration(prereg_path)
    cell = _primary_cell(result)
    n_seeds = len(cell.seeds)

    # --- PRIMARY (EC5) -------------------------------------------------------
    endpoint = _resolve_primary_endpoint(cell, prereg)  # asserts the name (DX3)
    if endpoint is not None:
        primary_endpoint_name = endpoint.name
        primary_verdict = endpoint.verdict
        primary_value: float | None = endpoint.value
        primary_ci_lo: float | None = endpoint.difference_ci.lo
        primary_ci_hi: float | None = endpoint.difference_ci.hi
        primary_ci_method = endpoint.difference_ci.method
        primary_test_name: str | None = endpoint.test.test
        primary_test_p: float | None = endpoint.test.p_value
        primary_effect_name: str | None = endpoint.effect.name
        primary_effect_value: float | None = endpoint.effect.value
        primary_effect_magnitude: str | None = endpoint.effect.magnitude
        noise_floor = endpoint.noise_floor
        exceeds_noise_floor = endpoint.exceeds_noise_floor
        observed_d = abs(endpoint.effect.value)
    else:
        logger.warning(
            "primary endpoint absent from the primary cell and its manifest; "
            "verdict = not-computed (DX2)."
        )
        primary_endpoint_name = prereg.primary_endpoint
        primary_verdict = "not-computed"
        primary_value = None
        primary_ci_lo = None
        primary_ci_hi = None
        primary_ci_method = "n/a"
        primary_test_name = None
        primary_test_p = None
        primary_effect_name = None
        primary_effect_value = None
        primary_effect_magnitude = None
        noise_floor = float(cell.aa_abs_difference)
        exceeds_noise_floor = False
        observed_d = 0.0

    # --- secondary readouts --------------------------------------------------
    crossover = _compute_crossover(result.length_sweep)
    tmr = _compute_tmr(cell, stats_seed)
    power = _compute_power(n_seeds, observed_d, computed=endpoint is not None)
    negative = _compute_negative(cell, prereg, primary_verdict, crossover, stats_seed)

    per_regime_verdicts = {c.regime.name: _cell_verdict(c) for c in result.grid.regimes}
    sim_real_agreement_note = _sim_real_note(result.sim_real)
    headline = _headline(
        cell.regime.name,
        primary_verdict,
        primary_value,
        exceeds_noise_floor,
        crossover,
        tmr,
        power,
    )

    return AnalysisReport(
        primary_endpoint_name=primary_endpoint_name,
        primary_verdict=primary_verdict,
        primary_value=primary_value,
        primary_ci_lo=primary_ci_lo,
        primary_ci_hi=primary_ci_hi,
        primary_ci_method=primary_ci_method,
        primary_test_name=primary_test_name,
        primary_test_p=primary_test_p,
        primary_effect_name=primary_effect_name,
        primary_effect_value=primary_effect_value,
        primary_effect_magnitude=primary_effect_magnitude,
        noise_floor=noise_floor,
        exceeds_noise_floor=exceeds_noise_floor,
        n_seeds=n_seeds,
        crossover=crossover,
        tmr=tmr,
        power=power,
        negative=negative,
        per_regime_verdicts=per_regime_verdicts,
        sim_real_agreement_note=sim_real_agreement_note,
        headline=headline,
        mock_llm_caveat=MOCK_LLM_CAVEAT,
    )


def _fmt(value: float | None, spec: str = ".4f") -> str:
    """Format an optional float for the markdown writeup (``n/a`` for ``None``)."""
    return "n/a" if value is None else format(value, spec)


def _render_results_md(result: Phase5Result, report: AnalysisReport) -> str:
    """Render ``paper/RESULTS.md`` from the analysis report (DX4/DX5 writeup).

    Opens with the mandatory mock-LLM caveat, then the primary verdict, the
    per-regime table, the long-context crossover (EC6), the TMR effect vs 0.29,
    the realized power (EC2), the sim-vs-real summary (EC3), and the
    negative-result mapping (EC7).

    Args:
        result: The Phase 5 artifact (provenance for the header).
        report: The computed analysis report (the writeup's only data source).

    Returns:
        The complete markdown document as a string.
    """
    cr = report.crossover
    tmr = report.tmr
    pw = report.power
    neg = report.negative

    lines: list[str] = []
    lines.append("# Slow Wave — Phase 5 Results")
    lines.append("")
    lines.append("> **MOCK-LLM CAVEAT (DX5).** " + report.mock_llm_caveat)
    lines.append("")
    lines.append(
        f"Experiment `{result.experiment}` · scenario `{result.scenario}` · "
        f"git `{result.git_commit or 'unknown'}` · model `{result.model_id}` "
        f"(mocked={result.model_mocked})."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(report.headline)
    lines.append("")

    # --- Primary endpoint (EC5) ---------------------------------------------
    lines.append("## Primary endpoint (EC5)")
    lines.append("")
    lines.append(
        f"The single preregistered **primary endpoint** is "
        f"`{report.primary_endpoint_name}` (full_dream - no_sleep, paired by "
        f"seed). Verdict: **{report.primary_verdict.upper()}** "
        f"(n={report.n_seeds} paired seeds)."
    )
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Point estimate (paired mean ACC diff) | {_fmt(report.primary_value)} |")
    lines.append(
        f"| 95% CI | [{_fmt(report.primary_ci_lo)}, {_fmt(report.primary_ci_hi)}] "
        f"({report.primary_ci_method}) |"
    )
    lines.append(
        f"| Test | {report.primary_test_name or 'n/a'} "
        f"(p={_fmt(report.primary_test_p)}) |"
    )
    lines.append(
        f"| Effect | {report.primary_effect_name or 'n/a'} = "
        f"{_fmt(report.primary_effect_value)} ({report.primary_effect_magnitude or 'n/a'}) |"
    )
    lines.append(
        f"| A/A noise floor | {_fmt(report.noise_floor)} "
        f"(exceeded={report.exceeds_noise_floor}) |"
    )
    lines.append("")

    # --- Per-regime verdicts -------------------------------------------------
    lines.append("## Per-regime verdicts")
    lines.append("")
    lines.append("| Regime | Verdict |")
    lines.append("| --- | --- |")
    for name, verdict in report.per_regime_verdicts.items():
        marker = " (primary)" if name == result.grid.primary_regime else ""
        lines.append(f"| {name}{marker} | {verdict} |")
    lines.append("")

    # --- Crossover (EC6) -----------------------------------------------------
    lines.append("## Long-context crossover (EC6)")
    lines.append("")
    lines.append(cr.note)
    lines.append("")
    if cr.lengths:
        lines.append(
            f"Cost-adjusted metric: `{cr.metric}` for `{cr.treatment_arm}` vs "
            f"`{cr.baseline_arm}`."
        )
        lines.append("")
        lines.append("| L | acc/token (treatment) | acc/token (baseline) | ACC gap (base - treat) |")
        lines.append("| --- | --- | --- | --- |")
        for i, length in enumerate(cr.lengths):
            lines.append(
                f"| {length} | {cr.acc_per_token_treatment[i]:.6g} | "
                f"{cr.acc_per_token_baseline[i]:.6g} | "
                f"{cr.acc_gap_baseline_minus_treatment[i]:+.4f} |"
            )
        lines.append("")
        lines.append(
            f"Cost-adjusted crossover length: "
            f"{cr.crossover_length if cr.crossover_found else 'none in range'}; "
            f"raw-accuracy crossover length: "
            f"{cr.raw_accuracy_crossover_length if cr.raw_accuracy_crossover_length is not None else 'none in range'}."
        )
        lines.append("")

    # --- TMR (FR5.3) ---------------------------------------------------------
    lines.append("## TMR-style targeting effect (FR5.3)")
    lines.append("")
    lines.append(
        f"Replay arms {tmr.replay_arms} vs no-replay arms {tmr.no_replay_arms}: "
        f"mean signal-retention lift = {tmr.mean_lift:+.4f}; bias-corrected "
        f"Hedges' **g = {tmr.hedges_g:.4f}** (95% CI [{tmr.g_ci_lo:.4f}, "
        f"{tmr.g_ci_hi:.4f}]). Benchmark (Hu et al. 2020) **g = "
        f"{tmr.benchmark_g}** — {'exceeds' if tmr.exceeds_benchmark else 'does not exceed'}."
    )
    lines.append("")
    lines.append(tmr.note)
    lines.append("")

    # --- Power (EC2) ---------------------------------------------------------
    lines.append("## Realized power (EC2)")
    lines.append("")
    lines.append(
        f"Realized **power**: n_seeds = {pw.n_seeds} (floor {pw.floor}, met="
        f"{pw.floor_met}); observed |d| = {pw.observed_effect_d:.4f}; required n "
        f"for the observed effect = {pw.required_n_for_observed} "
        f"(powered={pw.powered_for_observed})."
    )
    lines.append("")
    lines.append(pw.note)
    lines.append("")

    # --- Sim-vs-real (EC3) ---------------------------------------------------
    lines.append("## Sim-vs-real agreement (EC3)")
    lines.append("")
    lines.append(report.sim_real_agreement_note)
    lines.append("")

    # --- Negative-result mapping (EC7) --------------------------------------
    lines.append("## Negative-result mapping (EC7)")
    lines.append("")
    lines.append(f"Applicable: **{neg.applicable}**. Regime tie: {neg.regime_tie}.")
    lines.append("")
    if neg.matched_forms:
        lines.append("Matched preregistered negative-result forms:")
        lines.append("")
        for form in neg.matched_forms:
            lines.append(f"- {form}")
        lines.append("")
    lines.append("Registered secondary contrasts:")
    lines.append("")
    for label, verdict in neg.secondary_contrasts.items():
        lines.append(f"- **{label}**: {verdict}")
    lines.append("")
    lines.append(neg.note)
    lines.append("")
    return "\n".join(lines)


def write_analysis(
    result: Phase5Result, report: AnalysisReport, out_dir: Path
) -> tuple[Path, Path]:
    """Write ``analysis.json`` + ``paper/RESULTS.md`` and rewrite the artifact.

    Writes ``<out>/phase5/analysis.json`` (the report as deterministic,
    key-sorted JSON), ``<out>/paper/RESULTS.md`` (the human-readable writeup
    generated from ``report``), and rewrites ``<out>/phase5/phase5_result.json``
    with ``result.analysis = report`` so the committed artifact carries the
    verdicts (EC4/EC5). All JSON is ``sort_keys``-stable so a re-run is
    byte-identical (DX1).

    Args:
        result: The Phase 5 artifact (its ``analysis`` is replaced by ``report``
            in the rewritten file; the passed object is not mutated).
        report: The computed analysis report.
        out_dir: Base output directory; ``phase5/`` and ``paper/`` subtrees are
            created under it.

    Returns:
        ``(analysis_json_path, results_md_path)``.
    """
    out_dir = Path(out_dir)
    phase5_dir = out_dir / "phase5"
    paper_dir = out_dir / "paper"
    phase5_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = phase5_dir / "analysis.json"
    analysis_payload = json.dumps(
        report.model_dump(mode="json"), indent=2, sort_keys=True
    )
    analysis_path.write_text(analysis_payload + "\n", encoding="utf-8")

    results_md_path = paper_dir / "RESULTS.md"
    results_md_path.write_text(_render_results_md(result, report), encoding="utf-8")

    # Rewrite the canonical artifact with the analysis filled in (no mutation of
    # the passed object — model_copy keeps `analyze` re-runnable on `result`).
    result_with_analysis = result.model_copy(update={"analysis": report})
    result_payload = json.dumps(
        result_with_analysis.model_dump(mode="json"), indent=2, sort_keys=True
    )
    (phase5_dir / "phase5_result.json").write_text(result_payload + "\n", encoding="utf-8")

    logger.info(
        "wrote analysis -> %s, results -> %s, rewrote %s",
        analysis_path,
        results_md_path,
        phase5_dir / "phase5_result.json",
    )
    return analysis_path, results_md_path


def run_analysis(
    result_path: str | Path,
    *,
    prereg_path: str = _DEFAULT_PREREG,
    out_dir: str | Path | None = None,
) -> Path:
    """Load a Phase5Result, analyze it, write the artifacts, return analysis.json.

    Args:
        result_path: Path to ``phase5_result.json`` (WS1's output, ``analysis``
            may be ``None``).
        prereg_path: Path to the committed preregistration (DX3).
        out_dir: Base output directory; defaults to the result's grandparent so
            ``runs/phase5/phase5_result.json`` writes ``runs/phase5/analysis.json``
            and rewrites the artifact in place.

    Returns:
        The path to the written ``analysis.json``.
    """
    result_path = Path(result_path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    result = Phase5Result.model_validate(data)
    report = analyze(result, prereg_path=prereg_path)
    base = Path(out_dir) if out_dir is not None else result_path.parent.parent
    analysis_path, _ = write_analysis(result, report, base)
    return analysis_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``python -m slow_wave.eval.analysis`` CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m slow_wave.eval.analysis",
        description=(
            "Compute the Phase 5 analysis (EC5-EC7) from a phase5_result.json and "
            "write analysis.json + paper/RESULTS.md (mock-LLM mechanism demo; DX5)."
        ),
    )
    parser.add_argument(
        "--result",
        required=True,
        help="Path to runs/phase5/phase5_result.json (WS1 output).",
    )
    parser.add_argument(
        "--prereg",
        default=_DEFAULT_PREREG,
        help="Path to the committed preregistration.yaml (DX3).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Base output dir (default: the result file's grandparent).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: analyze a Phase5Result and write its artifacts.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)
    analysis_path = run_analysis(args.result, prereg_path=args.prereg, out_dir=args.out)
    print(analysis_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
