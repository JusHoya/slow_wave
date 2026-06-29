"""Tests for slow_wave.eval.stats (Phase 4, WS-STATS).

These tests pin the numpy-only statistics suite from the Phase 4 contract
(``docs/PHASE4_CONTRACT.md``, WS-STATS / EC6). Every estimator is checked
against a hand-computed or otherwise *known* answer (IQM of ``[1..8]`` is 4.5,
Cohen's *d* against its closed form, a worked Wilcoxon/Friedman case, the Holm
step-down on a known p-vector, probability-of-improvement at its dominance/tie
extremes) via :func:`pytest.approx`. Determinism (DX1) is asserted by running
each randomized routine twice with the same seeded generator and demanding
byte-identical results.

One clearly-marked test (:func:`test_crosscheck_against_scipy`) is guarded by
``pytest.importorskip("scipy")`` — scipy is in the dev venv (so it runs locally)
but absent from CI (so it skips). The rest of the suite passes with no scipy /
statsmodels / rliable / matplotlib present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slow_wave.eval.schema import BootstrapCI, EffectSize, RobustAggregate
from slow_wave.eval.schema import TestResult as _TestResult  # aliased: avoids pytest collecting it
from slow_wave.eval.stats import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    effect_size_cliffs_delta,
    effect_size_cohens_d,
    friedman,
    holm_correction,
    iqm,
    paired_diff_ci,
    performance_profile,
    probability_of_improvement,
    robust_aggregate,
    wilcoxon_signed_rank,
)

STATS_SEED = 20260629  # cfg.stats_seed analogue for the determinism checks.


def _rng(seed: int = STATS_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# IQM (hand value)
# --------------------------------------------------------------------------- #
def test_iqm_hand_value() -> None:
    """IQM of [1..8] is the mean of the middle four {3,4,5,6} = 4.5."""
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)


def test_iqm_trims_outliers() -> None:
    """IQM ignores the extreme tails a plain mean would chase."""
    # n=12: trim floor(0.25*12)=3 from each end -> mean of the middle six.
    data = [-100, -50, 1, 2, 3, 4, 5, 6, 7, 8, 50, 100]
    middle = [2, 3, 4, 5, 6, 7]
    assert iqm(data) == pytest.approx(sum(middle) / len(middle))


def test_iqm_empty_is_zero() -> None:
    """An empty sample yields 0.0 (degenerate, never raises)."""
    assert iqm([]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Bootstrap CIs
# --------------------------------------------------------------------------- #
def test_bootstrap_mean_ci_brackets_true_mean() -> None:
    """A 95% mean CI on a known normal sample brackets the true mean."""
    sample = np.random.default_rng(20260629).normal(5.0, 1.0, size=300)
    ci = bootstrap_ci(
        sample, statistic="mean", level=0.95, n_resamples=3000, rng=_rng(7)
    )
    assert isinstance(ci, BootstrapCI)
    assert ci.method == "percentile"
    assert ci.statistic == "mean"
    assert ci.lo <= ci.point <= ci.hi
    assert ci.lo <= 5.0 <= ci.hi


def test_bootstrap_ci_deterministic() -> None:
    """Two calls with the same seeded rng are byte-identical (DX1)."""
    sample = [0.1, 0.5, 0.5, 0.7, 0.9, 0.3, 0.8]
    a = bootstrap_ci(sample, n_resamples=2000, rng=_rng())
    b = bootstrap_ci(sample, n_resamples=2000, rng=_rng())
    assert a.model_dump() == b.model_dump()


def test_bootstrap_ci_degenerate_single_point() -> None:
    """n < 2 collapses to a degenerate interval lo == hi == point."""
    ci = bootstrap_ci([3.5], statistic="mean", rng=_rng())
    assert ci.point == pytest.approx(3.5)
    assert ci.lo == pytest.approx(3.5)
    assert ci.hi == pytest.approx(3.5)


def test_bootstrap_ci_name_override_and_iqm_statistic() -> None:
    """The statistic field is overridable and IQM is a valid statistic."""
    ci = bootstrap_ci(
        [1, 2, 3, 4, 5, 6, 7, 8],
        statistic="iqm",
        n_resamples=500,
        rng=_rng(),
        name="custom_iqm",
    )
    assert ci.statistic == "custom_iqm"
    assert ci.point == pytest.approx(4.5)


def test_bootstrap_ci_unknown_statistic_raises() -> None:
    """An unsupported statistic name raises ValueError."""
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2, 3], statistic="variance", rng=_rng())


def test_paired_diff_ci_constant_difference() -> None:
    """A constant paired difference gives that constant as the point estimate."""
    a = [2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = paired_diff_ci(a, b, n_resamples=2000, rng=_rng())
    assert ci.statistic == "paired_mean_diff"
    assert ci.point == pytest.approx(1.0)
    assert ci.lo <= 1.0 <= ci.hi
    # Identical seeded rng -> identical result (DX1).
    again = paired_diff_ci(a, b, n_resamples=2000, rng=_rng())
    assert ci.model_dump() == again.model_dump()


def test_paired_diff_ci_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_diff_ci([1.0, 2.0], [1.0], rng=_rng())


def test_robust_aggregate_bundles_three_cis() -> None:
    """robust_aggregate returns IQM/median/mean CIs and is deterministic."""
    sample = [0.2, 0.4, 0.6, 0.8, 1.0]
    agg = robust_aggregate("full_dream", sample, n_resamples=1500, rng=_rng())
    assert isinstance(agg, RobustAggregate)
    assert agg.arm_name == "full_dream"
    assert agg.n == 5
    assert agg.iqm.statistic == "iqm"
    assert agg.median.statistic == "median"
    assert agg.mean.statistic == "mean"
    assert agg.mean.point == pytest.approx(0.6)
    assert agg.median.point == pytest.approx(0.6)
    again = robust_aggregate("full_dream", sample, n_resamples=1500, rng=_rng())
    assert agg.model_dump() == again.model_dump()


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def test_cohens_d_paired_hand_value() -> None:
    """Paired d_z = mean(a-b)/std(a-b, ddof=1) = 2.5 / sqrt(5/3)."""
    a = [2.0, 4.0, 6.0, 8.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert cohens_d(a, b, paired=True) == pytest.approx(2.5 / math.sqrt(5 / 3))


def test_cohens_d_unpaired_hand_value() -> None:
    """Unpaired d = (mean_a - mean_b)/s_pooled = 2.5 / sqrt(25/6)."""
    a = [2.0, 4.0, 6.0, 8.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert cohens_d(a, b, paired=False) == pytest.approx(2.5 / math.sqrt(25 / 6))


def test_cohens_d_zero_variance_is_zero() -> None:
    """A zero standard deviation yields 0.0 rather than dividing by zero."""
    assert cohens_d([3.0, 3.0, 3.0], [3.0, 3.0, 3.0], paired=True) == 0.0


def test_effect_size_cohens_d_magnitude_and_ci() -> None:
    """The effect-size wrapper echoes cohens_d, labels magnitude, and is exact."""
    a = [2.0, 4.0, 6.0, 8.0, 10.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    es = effect_size_cohens_d(a, b, paired=True, n_resamples=1500, rng=_rng())
    assert isinstance(es, EffectSize)
    assert es.name == "cohens_d"
    assert es.value == pytest.approx(cohens_d(a, b, paired=True))
    assert es.magnitude == "large"
    assert es.lo <= es.value <= es.hi
    again = effect_size_cohens_d(a, b, paired=True, n_resamples=1500, rng=_rng())
    assert es.model_dump() == again.model_dump()


def test_cliffs_delta_extremes() -> None:
    """Cliff's delta is +1 under full dominance and 0 for identical samples."""
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_effect_size_cliffs_delta_magnitude() -> None:
    """Full dominance is a 'large' Cliff's delta and is deterministic."""
    es = effect_size_cliffs_delta([5, 6, 7, 8], [1, 2, 3, 4], n_resamples=1000, rng=_rng())
    assert es.name == "cliffs_delta"
    assert es.value == pytest.approx(1.0)
    assert es.magnitude == "large"
    again = effect_size_cliffs_delta(
        [5, 6, 7, 8], [1, 2, 3, 4], n_resamples=1000, rng=_rng()
    )
    assert es.model_dump() == again.model_dump()


# --------------------------------------------------------------------------- #
# Significance tests (known cases)
# --------------------------------------------------------------------------- #
def test_wilcoxon_signed_rank_hand_case() -> None:
    """a=[1..8] vs zeros: all positive ranks -> T=0, n=8, known two-sided p.

    W+ = 1+...+8 = 36, W- = 0, T = 0, mean = n(n+1)/4 = 18, se = sqrt(51),
    z = (0 - 18 + 0.5)/sqrt(51); p = erfc(|z|/sqrt(2)) ~= 0.0142662.
    """
    res = wilcoxon_signed_rank([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0])
    assert isinstance(res, _TestResult)
    assert res.test == "wilcoxon_signed_rank"
    assert res.statistic == pytest.approx(0.0)
    assert res.n == 8
    assert res.detail["r_plus"] == pytest.approx(36.0)
    assert res.detail["r_minus"] == pytest.approx(0.0)
    assert res.p_value == pytest.approx(0.014266186701446928, abs=1e-9)
    assert res.p_value < 0.05


def test_wilcoxon_all_ties_is_null() -> None:
    """All-zero differences drop out: statistic 0, p 1.0, n 0."""
    res = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res.statistic == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)
    assert res.n == 0
    assert res.detail["zeros_dropped"] == 3


def test_wilcoxon_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


def test_friedman_hand_case() -> None:
    """Perfectly separated arms (A>B>C in every block) give chi2=8, df=2.

    Rank sums R=(12,8,4); chi2 = 12/(n k (k+1)) * sum(R^2) - 3 n (k+1)
    = 12/48 * 224 - 48 = 8; for df=2 the chi-square sf is exp(-x/2) = exp(-4).
    """
    res = friedman([[10, 20, 30, 40], [5, 15, 25, 35], [1, 11, 21, 31]])
    assert res.test == "friedman"
    assert res.statistic == pytest.approx(8.0)
    assert res.n == 4
    assert res.detail["df"] == 2
    assert res.detail["k"] == 3
    assert res.detail["rank_sums"] == pytest.approx([12.0, 8.0, 4.0])
    assert res.p_value == pytest.approx(math.exp(-4.0))


def test_friedman_requires_three_arms() -> None:
    with pytest.raises(ValueError):
        friedman([[1, 2, 3], [4, 5, 6]])


def test_friedman_uneven_lengths_raise() -> None:
    with pytest.raises(ValueError):
        friedman([[1, 2, 3], [4, 5], [7, 8, 9]])


# --------------------------------------------------------------------------- #
# Holm correction (known p-vector)
# --------------------------------------------------------------------------- #
def test_holm_correction_known_vector() -> None:
    """Step-down on [0.01, 0.04, 0.03, 0.005] at alpha=0.05.

    Sorted: 0.005, 0.01, 0.03, 0.04 with multipliers 4,3,2,1 -> raw
    0.02, 0.03, 0.06, 0.04; monotone -> 0.02, 0.03, 0.06, 0.06. Mapped back to
    input order [idx0, idx1, idx2, idx3].
    """
    result = holm_correction([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    expected = [(0.03, True), (0.06, False), (0.06, False), (0.02, True)]
    assert [r[1] for r in result] == [e[1] for e in expected]
    for (adj, _), (exp_adj, _) in zip(result, expected):
        assert adj == pytest.approx(exp_adj)


def test_holm_correction_empty() -> None:
    assert holm_correction([]) == []


def test_holm_correction_all_significant() -> None:
    """Tiny p-values all survive the step-down."""
    result = holm_correction([0.001, 0.002, 0.003], alpha=0.05)
    assert all(reject for _, reject in result)


# --------------------------------------------------------------------------- #
# rliable comparison primitives
# --------------------------------------------------------------------------- #
def test_probability_of_improvement_extremes() -> None:
    """1.0 when a strictly dominates b; 0.5 on all ties."""
    assert probability_of_improvement([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert probability_of_improvement([2, 2, 2], [2, 2, 2]) == pytest.approx(0.5)


def test_probability_of_improvement_partial() -> None:
    """A single tie across one pair contributes 0.5 to the count."""
    # a=[2], b=[1,2,3]: 2>1, 2==2 (0.5), 2<3 -> (1 + 0.5)/3 = 0.5
    assert probability_of_improvement([2], [1, 2, 3]) == pytest.approx(0.5)


def test_performance_profile_values_and_monotonic() -> None:
    """Profile reports fraction >= tau and is non-increasing in tau."""
    profile = performance_profile([0.1, 0.4, 0.6, 0.9], [0.0, 0.5, 1.0])
    assert profile == [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    fractions = [frac for _, frac in profile]
    assert all(x >= y for x, y in zip(fractions, fractions[1:]))


def test_performance_profile_finer_grid_non_increasing() -> None:
    rng = _rng()
    sample = rng.random(40)
    taus = list(np.linspace(0.0, 1.0, 21))
    fractions = [frac for _, frac in performance_profile(sample, taus)]
    assert all(x >= y for x, y in zip(fractions, fractions[1:]))


# --------------------------------------------------------------------------- #
# Determinism sweep (DX1): every randomized fn byte-identical at one seed
# --------------------------------------------------------------------------- #
def _run_all_randomized(seed: int) -> list[dict]:
    """Run every randomized routine off a single seeded rng, dumping results."""
    rng = _rng(seed)
    a = [0.3, 0.5, 0.6, 0.8, 0.9]
    b = [0.2, 0.4, 0.4, 0.5, 0.7]
    out: list[dict] = []
    out.append(bootstrap_ci(a, n_resamples=500, rng=rng).model_dump())
    out.append(paired_diff_ci(a, b, n_resamples=500, rng=rng).model_dump())
    out.append(robust_aggregate("x", a, n_resamples=500, rng=rng).model_dump())
    out.append(
        effect_size_cohens_d(a, b, paired=True, n_resamples=500, rng=rng).model_dump()
    )
    out.append(
        effect_size_cohens_d(a, b, paired=False, n_resamples=500, rng=rng).model_dump()
    )
    out.append(
        effect_size_cliffs_delta(a, b, n_resamples=500, rng=rng).model_dump()
    )
    return out


def test_all_randomized_functions_byte_identical() -> None:
    """Two passes off the same stats_seed produce byte-identical structures."""
    first = _run_all_randomized(STATS_SEED)
    second = _run_all_randomized(STATS_SEED)
    assert first == second


def test_results_are_json_dumpable() -> None:
    """Every returned model round-trips through JSON mode (manifest-ready)."""
    rng = _rng()
    models = [
        bootstrap_ci([0.1, 0.2, 0.3], n_resamples=200, rng=rng),
        robust_aggregate("arm", [0.1, 0.2, 0.3, 0.4], n_resamples=200, rng=rng),
        effect_size_cohens_d([1, 2, 3, 4], [0, 1, 2, 3], n_resamples=200, rng=rng),
        wilcoxon_signed_rank([1, 2, 3, 4], [0, 1, 2, 3]),
        friedman([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),
    ]
    for model in models:
        dumped = model.model_dump(mode="json")
        assert isinstance(dumped, dict)


# --------------------------------------------------------------------------- #
# Optional cross-check against scipy (skipped in CI; runs in the dev venv)
# --------------------------------------------------------------------------- #
def test_crosscheck_against_scipy() -> None:
    """Validate Wilcoxon + Friedman against scipy where it is installed.

    Guarded by ``pytest.importorskip`` so it runs in the dev venv (scipy present)
    and skips cleanly in CI (scipy absent) — the rest of the suite never imports
    scipy.
    """
    scipy_stats = pytest.importorskip("scipy.stats")

    a = [5, 3, 8, 2, 9, 1, 7]
    b = [4, 3, 6, 5, 9, 2, 4]
    mine = wilcoxon_signed_rank(a, b)
    ref = scipy_stats.wilcoxon(a, b, correction=True, mode="approx", alternative="two-sided")
    assert mine.statistic == pytest.approx(float(ref.statistic))
    assert mine.p_value == pytest.approx(float(ref.pvalue), rel=1e-9, abs=1e-12)

    groups = [[1, 2, 3, 4], [2, 2, 3, 4], [3, 3, 3, 4]]
    mine_f = friedman(groups)
    ref_f = scipy_stats.friedmanchisquare(*groups)
    assert mine_f.statistic == pytest.approx(float(ref_f.statistic))
    assert mine_f.p_value == pytest.approx(float(ref_f.pvalue), rel=1e-9, abs=1e-12)
