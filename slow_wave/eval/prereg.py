"""Preregistration guard + primary-endpoint analysis (Phase 4, FR5.5, DX3).

This module makes the bench's falsifiability **visible in code** (DX3): the
analysis reads the *single* primary endpoint named in the committed
``prereg/preregistration.yaml`` artifact and **refuses** to compute any other.
A request for a non-preregistered endpoint raises
:class:`NonPreregisteredEndpointError` *before* any statistic is touched, so an
analyst cannot quietly p-hack a different contrast into the primary slot.

The primary endpoint itself (``acc_diff_full_dream_vs_no_sleep``) is the paired,
per-seed difference in final Average Accuracy between the treatment arm
(``full_dream``) and the baseline arm (``no_sleep``), aggregated as the mean
paired difference over seeds and reported with:

* a percentile bootstrap 95% CI of the mean paired difference,
* a standardized paired effect size (Cohen's d) with its own bootstrap CI,
* a paired Wilcoxon signed-rank test,

then judged against the A/A noise floor. The verdict is ``"confirmed"`` only
when the effect is positive, its CI excludes zero, **and** it clears the A/A
noise floor — exactly the rejection criteria pre-committed in the artifact.

All randomness flows through an explicitly-injected ``rng`` (DX1): two runs with
the same config + seed are byte-identical under the mock LLM. The statistics are
delegated to :mod:`slow_wave.eval.stats` (numpy-only, no scipy).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from slow_wave.eval.schema import AAResult, ArmResult, PrimaryEndpoint, Preregistration
from slow_wave.eval.stats import (
    effect_size_cohens_d,
    paired_diff_ci,
    wilcoxon_signed_rank,
)

logger = logging.getLogger(__name__)


class NonPreregisteredEndpointError(ValueError):
    """Raised when analysis is asked for an endpoint the prereg did not name.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    still catch it, while callers that care about the falsifiability guard
    (DX3) can catch this specific type.
    """


def load_preregistration(path: str | Path) -> Preregistration:
    """Parse the committed registered-report YAML into a :class:`Preregistration`.

    Args:
        path: Path to the committed ``preregistration.yaml`` artifact.

    Returns:
        The validated :class:`Preregistration` (``extra="forbid"``: every key in
        the artifact is load-bearing, and an unexpected key fails loudly).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML root is not a mapping.
        pydantic.ValidationError: If the data does not satisfy the schema.
    """
    prereg_path = Path(path)
    if not prereg_path.is_file():
        raise FileNotFoundError(f"Preregistration file not found: {prereg_path}")

    with prereg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Preregistration file {prereg_path} must contain a YAML mapping at "
            f"the top level, got {type(data).__name__}."
        )

    return Preregistration.model_validate(data)


def assert_primary_endpoint(prereg: Preregistration, requested_name: str) -> None:
    """Refuse any endpoint other than the prereg's single primary one (DX3).

    This is the falsifiability guard rendered in code: analysis may only ever
    compute the endpoint that was pre-committed, so a different contrast cannot
    be slipped into the primary slot after seeing the data.

    Args:
        prereg: The parsed preregistration.
        requested_name: The endpoint name the caller is asking to compute.

    Raises:
        NonPreregisteredEndpointError: If ``requested_name`` is not exactly
            ``prereg.primary_endpoint``.
    """
    if requested_name != prereg.primary_endpoint:
        raise NonPreregisteredEndpointError(
            f"Refusing to compute non-preregistered primary endpoint "
            f"{requested_name!r}; the committed preregistration names "
            f"{prereg.primary_endpoint!r} as the single primary endpoint (DX3)."
        )


def compute_primary_endpoint(
    prereg: Preregistration,
    arm_results: list[ArmResult],
    *,
    requested_name: str,
    aa: AAResult,
    ci_level: float = 0.95,
    n_resamples: int = 10000,
    rng: np.random.Generator,
) -> PrimaryEndpoint:
    """Compute the prereg primary endpoint with its falsifiable verdict (FR5.5).

    The endpoint is the **paired, per-seed** difference in final Average Accuracy
    (``continual_metrics.acc``) between the treatment arm and the baseline arm
    named in the preregistration, paired by seed. The point estimate is the mean
    paired difference (treatment - baseline); it is reported with a percentile
    bootstrap CI (:func:`slow_wave.eval.stats.paired_diff_ci`), a standardized
    paired Cohen's d with CI (:func:`~slow_wave.eval.stats.effect_size_cohens_d`),
    and a paired Wilcoxon signed-rank test
    (:func:`~slow_wave.eval.stats.wilcoxon_signed_rank`).

    The endpoint name is validated **first** via :func:`assert_primary_endpoint`,
    so a non-preregistered endpoint is refused before any statistic is computed
    (DX3). The verdict follows the pre-committed rejection criteria:

    * ``"confirmed"`` iff ``value > 0`` **and** the CI excludes zero **and** the
      effect exceeds the A/A noise floor;
    * else ``"refuted"`` if the CI includes zero;
    * else ``"inconclusive"``.

    Args:
        prereg: The parsed preregistration binding the analysis.
        arm_results: Every ``(arm, seed)`` :class:`ArmResult`; the treatment and
            baseline arms are selected from these by ``arm_name`` and paired by
            ``seed``.
        requested_name: The endpoint name the caller asks to compute (must equal
            ``prereg.primary_endpoint``).
        aa: The A/A noise-floor control; ``aa.abs_difference`` is the floor the
            effect must clear to count as confirmed.
        ci_level: Confidence level for both bootstrap CIs.
        n_resamples: Number of bootstrap resamples (passed to the stats routines).
        rng: The explicitly-seeded generator for the bootstrap (DX1). Keyword-only.

    Returns:
        The populated :class:`PrimaryEndpoint` with its verdict.

    Raises:
        NonPreregisteredEndpointError: If ``requested_name`` is not the prereg's
            primary endpoint (raised before any computation).
        ValueError: If the treatment and baseline arms share no common seed (no
            paired observations to compute the endpoint from).
    """
    # DX3: refuse a non-preregistered endpoint FIRST, before any statistics.
    assert_primary_endpoint(prereg, requested_name)

    treatment = prereg.treatment_arm
    baseline = prereg.baseline_arm

    treat_by_seed = {r.seed: r for r in arm_results if r.arm_name == treatment}
    base_by_seed = {r.seed: r for r in arm_results if r.arm_name == baseline}
    common_seeds = sorted(set(treat_by_seed) & set(base_by_seed))

    if not common_seeds:
        raise ValueError(
            f"No paired seeds between treatment arm {treatment!r} and baseline "
            f"arm {baseline!r}; cannot compute the primary endpoint."
        )

    # Pair BY SEED on final ACC (treatment - baseline). a/b are aligned per seed.
    a = np.array(
        [treat_by_seed[s].continual_metrics.acc for s in common_seeds], dtype=float
    )
    b = np.array(
        [base_by_seed[s].continual_metrics.acc for s in common_seeds], dtype=float
    )

    value = float(np.mean(a - b))
    difference_ci = paired_diff_ci(
        a, b, level=ci_level, n_resamples=n_resamples, rng=rng
    )
    effect = effect_size_cohens_d(
        a, b, paired=True, level=ci_level, n_resamples=n_resamples, rng=rng
    )
    test = wilcoxon_signed_rank(a, b)

    noise_floor = float(aa.abs_difference)
    exceeds_noise_floor = abs(value) > noise_floor

    ci_excludes_zero = difference_ci.lo > 0.0 or difference_ci.hi < 0.0
    ci_includes_zero = not ci_excludes_zero

    if value > 0.0 and ci_excludes_zero and exceeds_noise_floor:
        verdict = "confirmed"
    elif ci_includes_zero:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    logger.info(
        "primary endpoint %s: value=%.6f ci=[%.6f, %.6f] noise_floor=%.6f "
        "exceeds_floor=%s -> %s (n=%d paired seeds)",
        requested_name,
        value,
        difference_ci.lo,
        difference_ci.hi,
        noise_floor,
        exceeds_noise_floor,
        verdict,
        len(common_seeds),
    )

    return PrimaryEndpoint(
        name=requested_name,
        description=prereg.primary_endpoint_description,
        treatment_arm=treatment,
        baseline_arm=baseline,
        value=value,
        difference_ci=difference_ci,
        effect=effect,
        test=test,
        noise_floor=noise_floor,
        exceeds_noise_floor=exceeds_noise_floor,
        verdict=verdict,
    )
