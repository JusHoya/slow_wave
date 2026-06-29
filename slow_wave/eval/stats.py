"""Robust statistics suite for the Phase 4 evaluation harness (FR5.4, EC6).

This module is the bench's inference layer: it turns the per-seed score samples
produced by the nine control arms into bootstrap confidence intervals, rliable
robust aggregates, standardized effect sizes, non-parametric significance tests,
and a multiple-comparison correction. Every routine is implemented **from
scratch in numpy** so the suite runs in the lean CI image, which ships only
``pydantic``/``pyyaml``/``numpy``/``anthropic``/``pytest`` — there is no
``scipy``/``statsmodels``/``rliable``/``matplotlib`` to lean on (see
``docs/PHASE4_CONTRACT.md``, "LEAN CORE, NUMPY-ONLY STATS").

Design principles
-----------------
* **Pure & deterministic (DX1).** Every randomized routine takes an explicit
  ``rng: numpy.random.Generator`` and draws all of its randomness from it; two
  calls with the same seeded generator return byte-identical results. Nothing
  here touches the ``numpy.random`` global or Python ``hash()``.
* **numpy + stdlib + pydantic only.** The chi-square survival function (for
  Friedman) is evaluated through a NumPy-Recipes regularized incomplete-gamma
  implementation; the normal tail (for Wilcoxon) through :func:`math.erfc`. No
  third-party statistics package is imported.
* **JSON-first results.** Every public estimator returns one of the pydantic
  models in :mod:`slow_wave.eval.schema` (``extra="forbid"``); the harness rolls
  these into the run manifest.

Literature
----------
* Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife* —
  percentile bootstrap CIs.
* Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the
  Statistical Precipice* (NeurIPS) — IQM, performance profiles, probability of
  improvement.
* Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods* — the signed-
  rank test (normal approximation with tie + continuity correction).
* Friedman, M. (1937). *The Use of Ranks to Avoid the Assumption of Normality*;
  see also Demšar, J. (2006). *Statistical Comparisons of Classifiers over
  Multiple Data Sets* (JMLR) — the repeated-measures omnibus.
* Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure* —
  step-down family-wise error control.
* Cohen, J. (1988). *Statistical Power Analysis* — the standardized mean
  difference *d* and its magnitude thresholds.
* Cliff, N. (1993) / Romano, J. et al. (2006) — the dominance statistic
  (Cliff's delta) and its magnitude thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from slow_wave.eval.schema import (
    BootstrapCI,
    EffectSize,
    RobustAggregate,
    TestResult,
)

ArrayLike = Sequence[float] | np.ndarray

_SUPPORTED_STATISTICS = ("mean", "median", "iqm")
_IQM_TRIM = 0.25  # discard this fraction from each tail (Agarwal et al. 2021).


# --------------------------------------------------------------------------- #
# Small numeric primitives (pure)
# --------------------------------------------------------------------------- #
def _as1d(x: ArrayLike) -> np.ndarray:
    """Return ``x`` as a flat float64 array (a copy-safe view for indexing)."""
    return np.asarray(x, dtype=float).ravel()


def iqm(samples: ArrayLike) -> float:
    """Interquartile mean: the mean of the middle 50% of the samples.

    The IQM (Agarwal et al. 2021) sorts the sample, discards the bottom and top
    ``25%`` (``floor(0.25 * n)`` elements from each tail, matching a 25%-trimmed
    mean), and averages the remainder. It is far more robust to the heavy-tailed,
    outlier-prone score distributions of small-seed experiments than the mean,
    while staying more efficient than the median.

    Args:
        samples: The score sample (any 1-D array-like).

    Returns:
        The interquartile mean, e.g. ``iqm([1..8]) == mean(3, 4, 5, 6) == 4.5``.
        Returns ``0.0`` for an empty sample (degenerate, never raises).
    """
    x = np.sort(_as1d(samples))
    n = x.size
    if n == 0:
        return 0.0
    lo = int(_IQM_TRIM * n)
    hi = n - lo
    if hi <= lo:  # too few points to trim both tails — fall back to the mean.
        return float(np.mean(x))
    return float(np.mean(x[lo:hi]))


def _iqm_rows(matrix: np.ndarray) -> np.ndarray:
    """Vectorized IQM along axis 1 of a ``(n_resamples, n)`` resample matrix."""
    n = matrix.shape[1]
    s = np.sort(matrix, axis=1)
    lo = int(_IQM_TRIM * n)
    hi = n - lo
    if hi <= lo:
        return s.mean(axis=1)
    return s[:, lo:hi].mean(axis=1)


def _statistic_scalar(x: np.ndarray, statistic: str) -> float:
    """Evaluate ``statistic`` on the observed 1-D sample ``x`` (empty -> 0.0)."""
    if x.size == 0:
        return 0.0
    if statistic == "mean":
        return float(np.mean(x))
    if statistic == "median":
        return float(np.median(x))
    if statistic == "iqm":
        return iqm(x)
    raise ValueError(
        f"unknown statistic {statistic!r}; expected one of {_SUPPORTED_STATISTICS}"
    )


def _statistic_rows(matrix: np.ndarray, statistic: str) -> np.ndarray:
    """Evaluate ``statistic`` along axis 1 of a resample matrix (vectorized)."""
    if statistic == "mean":
        return matrix.mean(axis=1)
    if statistic == "median":
        return np.median(matrix, axis=1)
    if statistic == "iqm":
        return _iqm_rows(matrix)
    raise ValueError(
        f"unknown statistic {statistic!r}; expected one of {_SUPPORTED_STATISTICS}"
    )


def _percentile_bounds(values: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided percentile CI bounds at confidence ``level`` over ``values``."""
    alpha = 1.0 - level
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Return 1-based ranks of ``x`` with ties resolved by the average rank.

    A small, deterministic ``scipy.stats.rankdata`` replacement (stable sort) for
    the Wilcoxon and Friedman implementations.
    """
    n = x.size
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:  # a tie block spanning sorted positions i..j (0-based).
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0  # mean of ranks i+1..j+1
        i = j + 1
    return ranks


def _tie_term(values: np.ndarray) -> float:
    """Return ``sum_g (t_g**3 - t_g)`` over tie groups of sizes ``t_g``."""
    if values.size == 0:
        return 0.0
    counts = np.unique(values, return_counts=True)[1].astype(float)
    return float(np.sum(counts**3 - counts))


# --------------------------------------------------------------------------- #
# Chi-square tail via regularized incomplete gamma (Numerical Recipes)
# --------------------------------------------------------------------------- #
def _gser(a: float, x: float, *, itmax: int = 300, eps: float = 1e-14) -> float:
    """Lower regularized incomplete gamma ``P(a, x)`` by series (``x < a + 1``)."""
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float, *, itmax: int = 300, eps: float = 1e-14) -> float:
    """Upper regularized incomplete gamma ``Q(a, x)`` by the Lentz continued
    fraction (``x >= a + 1``)."""
    fpmin = 1e-300
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _gammaq(a: float, x: float) -> float:
    """Upper regularized incomplete gamma ``Q(a, x) = 1 - P(a, x)``."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("_gammaq requires a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function ``P(X > x)`` for ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0.0:
        return 1.0
    return float(min(1.0, max(0.0, _gammaq(df / 2.0, x / 2.0))))


# --------------------------------------------------------------------------- #
# Bootstrap CIs (Efron percentile)
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    samples: ArrayLike,
    *,
    statistic: str = "mean",
    level: float = 0.95,
    n_resamples: int = 10000,
    rng: np.random.Generator,
    name: str | None = None,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval for a statistic (Efron 1979).

    Resamples ``samples`` with replacement ``n_resamples`` times, recomputes
    ``statistic`` on each resample, and reports the central ``level`` percentile
    interval of the resample distribution alongside the point estimate on the
    observed sample.

    Args:
        samples: The observed sample (1-D array-like).
        statistic: One of ``"mean"``, ``"median"``, ``"iqm"``.
        level: Two-sided confidence level (e.g. ``0.95``).
        n_resamples: Number of bootstrap resamples.
        rng: Explicit NumPy generator; identical seeded generators give
            byte-identical intervals (DX1).
        name: Optional override for the reported ``statistic`` field (e.g.
            ``"paired_mean_diff"``); defaults to ``statistic``.

    Returns:
        A :class:`~slow_wave.eval.schema.BootstrapCI`. For ``n < 2`` the interval
        is degenerate (``lo == hi == point``) — there is nothing to resample.
    """
    if statistic not in _SUPPORTED_STATISTICS:
        raise ValueError(
            f"unknown statistic {statistic!r}; expected one of {_SUPPORTED_STATISTICS}"
        )
    x = _as1d(samples)
    n = x.size
    point = _statistic_scalar(x, statistic)
    stat_name = name if name is not None else statistic
    if n < 2:
        lo = hi = point
    else:
        idx = rng.integers(0, n, size=(n_resamples, n))
        resamples = x[idx]
        boot = _statistic_rows(resamples, statistic)
        lo, hi = _percentile_bounds(boot, level)
    return BootstrapCI(
        point=point,
        lo=lo,
        hi=hi,
        level=level,
        method="percentile",
        n_resamples=n_resamples,
        statistic=stat_name,
    )


def paired_diff_ci(
    a: ArrayLike,
    b: ArrayLike,
    *,
    level: float = 0.95,
    n_resamples: int = 10000,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Percentile bootstrap CI of the paired mean difference ``mean(a - b)``.

    The pairs are resampled jointly — resampling the difference vector
    ``d = a - b`` is exactly resampling pair indices together — so the CI honours
    the within-pair correlation. The reported ``statistic`` is
    ``"paired_mean_diff"``.

    Args:
        a: First member of each pair (1-D array-like).
        b: Second member of each pair, same length as ``a``.
        level: Two-sided confidence level.
        n_resamples: Number of bootstrap resamples.
        rng: Explicit NumPy generator (DX1).

    Returns:
        A :class:`~slow_wave.eval.schema.BootstrapCI` of ``mean(a - b)``.

    Raises:
        ValueError: If ``a`` and ``b`` differ in length.
    """
    av = _as1d(a)
    bv = _as1d(b)
    if av.size != bv.size:
        raise ValueError(
            f"paired_diff_ci requires equal-length samples; got {av.size} vs {bv.size}"
        )
    return bootstrap_ci(
        av - bv,
        statistic="mean",
        level=level,
        n_resamples=n_resamples,
        rng=rng,
        name="paired_mean_diff",
    )


def robust_aggregate(
    arm_name: str,
    samples: ArrayLike,
    *,
    level: float = 0.95,
    n_resamples: int = 10000,
    rng: np.random.Generator,
) -> RobustAggregate:
    """rliable-style robust aggregate for one arm's score samples (Agarwal 2021).

    Bundles the interquartile mean, median, and mean of ``samples``, each with
    its own percentile bootstrap CI. The three CIs are drawn in order from the
    single supplied ``rng`` so the whole aggregate is deterministic.

    Args:
        arm_name: The arm the scores belong to.
        samples: The arm's per-seed scores (1-D array-like).
        level: Two-sided confidence level for each CI.
        n_resamples: Number of bootstrap resamples per aggregate.
        rng: Explicit NumPy generator (DX1).

    Returns:
        A populated :class:`~slow_wave.eval.schema.RobustAggregate`.
    """
    x = _as1d(samples)
    return RobustAggregate(
        arm_name=arm_name,
        n=int(x.size),
        iqm=bootstrap_ci(
            x, statistic="iqm", level=level, n_resamples=n_resamples, rng=rng
        ),
        median=bootstrap_ci(
            x, statistic="median", level=level, n_resamples=n_resamples, rng=rng
        ),
        mean=bootstrap_ci(
            x, statistic="mean", level=level, n_resamples=n_resamples, rng=rng
        ),
    )


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def _cohens_magnitude(d: float) -> str:
    """Qualitative magnitude of Cohen's *d* by Cohen's (1988) thresholds."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _cliffs_magnitude(d: float) -> str:
    """Qualitative magnitude of Cliff's delta by Romano et al. (2006)."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def cohens_d(a: ArrayLike, b: ArrayLike, *, paired: bool = True) -> float:
    """Standardized mean difference between ``a`` and ``b`` (Cohen 1988).

    For the **paired** case this is the within-subject ``d_z``:
    ``mean(a - b) / std(a - b)`` (sample std, ``ddof=1``). For the **unpaired**
    case it is the pooled-SD standardized difference
    ``(mean(a) - mean(b)) / s_pooled`` with
    ``s_pooled = sqrt(((n_a - 1) var_a + (n_b - 1) var_b) / (n_a + n_b - 2))``.

    Args:
        a: First sample (1-D array-like).
        b: Second sample.
        paired: Whether to use the paired ``d_z`` (default) or the unpaired
            pooled-SD form.

    Returns:
        The standardized difference, or ``0.0`` if the relevant standard
        deviation is zero / the sample is too small to estimate it.

    Raises:
        ValueError: If ``paired`` and the samples differ in length.
    """
    av = _as1d(a)
    bv = _as1d(b)
    if paired:
        if av.size != bv.size:
            raise ValueError("paired cohens_d requires equal-length samples")
        d = av - bv
        if d.size < 2:
            return 0.0
        sd = float(np.std(d, ddof=1))
        return 0.0 if sd == 0.0 else float(np.mean(d) / sd)
    na, nb = av.size, bv.size
    if na < 2 or nb < 2:
        return 0.0
    var_a = float(np.var(av, ddof=1))
    var_b = float(np.var(bv, ddof=1))
    sp = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return 0.0 if sp == 0.0 else float((np.mean(av) - np.mean(bv)) / sp)


def effect_size_cohens_d(
    a: ArrayLike,
    b: ArrayLike,
    *,
    paired: bool = True,
    level: float = 0.95,
    n_resamples: int = 10000,
    rng: np.random.Generator,
) -> EffectSize:
    """Cohen's *d* with a percentile bootstrap CI and magnitude label.

    The point estimate is :func:`cohens_d`; the CI is the bootstrap distribution
    of *d* (paired -> resample pairs jointly; unpaired -> resample each sample
    independently). Magnitude follows Cohen (1988): ``|d| < 0.2`` negligible,
    ``< 0.5`` small, ``< 0.8`` medium, else large.

    Args:
        a: First sample.
        b: Second sample.
        paired: Paired ``d_z`` (default) vs unpaired pooled-SD *d*.
        level: Two-sided confidence level.
        n_resamples: Number of bootstrap resamples.
        rng: Explicit NumPy generator (DX1).

    Returns:
        An :class:`~slow_wave.eval.schema.EffectSize` with ``name="cohens_d"``.
    """
    av = _as1d(a)
    bv = _as1d(b)
    point = cohens_d(av, bv, paired=paired)
    if paired:
        n = av.size
        if n < 2:
            lo = hi = point
        else:
            idx = rng.integers(0, n, size=(n_resamples, n))
            d = av[idx] - bv[idx]
            mean = d.mean(axis=1)
            sd = d.std(axis=1, ddof=1)
            vals = np.divide(mean, sd, out=np.zeros_like(mean), where=sd != 0.0)
            lo, hi = _percentile_bounds(vals, level)
    else:
        na, nb = av.size, bv.size
        if na < 2 or nb < 2:
            lo = hi = point
        else:
            ra = av[rng.integers(0, na, size=(n_resamples, na))]
            rb = bv[rng.integers(0, nb, size=(n_resamples, nb))]
            va = ra.var(axis=1, ddof=1)
            vb = rb.var(axis=1, ddof=1)
            sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
            num = ra.mean(axis=1) - rb.mean(axis=1)
            vals = np.divide(num, sp, out=np.zeros_like(sp), where=sp != 0.0)
            lo, hi = _percentile_bounds(vals, level)
    return EffectSize(
        name="cohens_d",
        value=point,
        lo=lo,
        hi=hi,
        level=level,
        magnitude=_cohens_magnitude(point),
    )


def cliffs_delta(a: ArrayLike, b: ArrayLike) -> float:
    """Cliff's delta: the dominance probability ``P(a>b) - P(a<b)`` (Cliff 1993).

    Over all ``n_a * n_b`` cross pairs, ``delta = (#(a_i > b_j) - #(a_i < b_j)) /
    (n_a * n_b)``. It ranges in ``[-1, 1]``: ``+1`` iff every ``a`` exceeds every
    ``b``, ``0`` for complete overlap. Non-parametric — invariant to monotone
    rescaling and robust to outliers.

    Args:
        a: First sample (1-D array-like).
        b: Second sample.

    Returns:
        Cliff's delta, or ``0.0`` if either sample is empty.
    """
    av = _as1d(a)
    bv = _as1d(b)
    na, nb = av.size, bv.size
    if na == 0 or nb == 0:
        return 0.0
    diff = av[:, None] - bv[None, :]
    gt = int(np.count_nonzero(diff > 0))
    lt = int(np.count_nonzero(diff < 0))
    return (gt - lt) / (na * nb)


def effect_size_cliffs_delta(
    a: ArrayLike,
    b: ArrayLike,
    *,
    level: float = 0.95,
    n_resamples: int = 10000,
    rng: np.random.Generator,
) -> EffectSize:
    """Cliff's delta with a percentile bootstrap CI and magnitude label.

    The point estimate is :func:`cliffs_delta`; the CI bootstraps both samples
    independently. Magnitude follows Romano et al. (2006): ``|d| < 0.147``
    negligible, ``< 0.33`` small, ``< 0.474`` medium, else large.

    Args:
        a: First sample.
        b: Second sample.
        level: Two-sided confidence level.
        n_resamples: Number of bootstrap resamples.
        rng: Explicit NumPy generator (DX1).

    Returns:
        An :class:`~slow_wave.eval.schema.EffectSize` with
        ``name="cliffs_delta"``.
    """
    av = _as1d(a)
    bv = _as1d(b)
    point = cliffs_delta(av, bv)
    na, nb = av.size, bv.size
    if na < 1 or nb < 1:
        lo = hi = point
    else:
        ra = av[rng.integers(0, na, size=(n_resamples, na))]
        rb = bv[rng.integers(0, nb, size=(n_resamples, nb))]
        diff = ra[:, :, None] - rb[:, None, :]
        gt = np.count_nonzero(diff > 0, axis=(1, 2))
        lt = np.count_nonzero(diff < 0, axis=(1, 2))
        vals = (gt - lt) / (na * nb)
        lo, hi = _percentile_bounds(vals, level)
    return EffectSize(
        name="cliffs_delta",
        value=point,
        lo=lo,
        hi=hi,
        level=level,
        magnitude=_cliffs_magnitude(point),
    )


# --------------------------------------------------------------------------- #
# Significance tests
# --------------------------------------------------------------------------- #
def wilcoxon_signed_rank(a: ArrayLike, b: ArrayLike) -> TestResult:
    """Paired Wilcoxon signed-rank test, normal approximation (Wilcoxon 1945).

    Computes the within-pair differences ``d = a - b``, drops exact zeros, ranks
    ``|d|`` with average ties, and sums the ranks of positive (``r_plus``) and
    negative (``r_minus``) differences. The reported statistic is
    ``T = min(r_plus, r_minus)``. The two-sided p-value uses the normal
    approximation with the standard tie correction to the variance and a
    continuity correction of ``0.5``:

    ``z = (T - n(n+1)/4 - 0.5*sign(T - mean)) / sqrt(var)`` where
    ``var = [n(n+1)(2n+1) - 0.5*sum(t^3 - t)] / 24`` over tie groups of size
    ``t``, and ``p = erfc(|z| / sqrt(2))``.

    Args:
        a: First member of each pair (1-D array-like).
        b: Second member, same length as ``a``.

    Returns:
        A :class:`~slow_wave.eval.schema.TestResult` with
        ``test="wilcoxon_signed_rank"``; ``n`` is the number of non-zero pairs.
        ``detail`` carries ``r_plus``/``r_minus``/``z``/``zeros_dropped``/
        ``tie_correction``. An all-ties input yields ``statistic=0``,
        ``p_value=1.0``, ``n=0``.

    Raises:
        ValueError: If ``a`` and ``b`` differ in length.
    """
    av = _as1d(a)
    bv = _as1d(b)
    if av.size != bv.size:
        raise ValueError("wilcoxon_signed_rank requires equal-length samples")
    d = av - bv
    nonzero = d[d != 0.0]
    zeros_dropped = int(d.size - nonzero.size)
    count = nonzero.size
    if count == 0:
        return TestResult(
            test="wilcoxon_signed_rank",
            statistic=0.0,
            p_value=1.0,
            n=0,
            detail={
                "r_plus": 0.0,
                "r_minus": 0.0,
                "z": 0.0,
                "zeros_dropped": zeros_dropped,
                "tie_correction": 0.0,
                "continuity_correction": True,
            },
        )
    abs_d = np.abs(nonzero)
    ranks = _rankdata(abs_d)
    r_plus = float(np.sum(ranks[nonzero > 0]))
    r_minus = float(np.sum(ranks[nonzero < 0]))
    stat = min(r_plus, r_minus)
    mean = count * (count + 1) * 0.25
    tie_term = _tie_term(abs_d)
    var = (count * (count + 1) * (2 * count + 1) - 0.5 * tie_term) / 24.0
    if var <= 0.0:
        z = 0.0
        p = 1.0
    else:
        cc = 0.5 * float(np.sign(stat - mean))
        z = (stat - mean - cc) / math.sqrt(var)
        p = min(1.0, max(0.0, math.erfc(abs(z) / math.sqrt(2.0))))
    return TestResult(
        test="wilcoxon_signed_rank",
        statistic=float(stat),
        p_value=float(p),
        n=int(count),
        detail={
            "r_plus": r_plus,
            "r_minus": r_minus,
            "z": float(z),
            "zeros_dropped": zeros_dropped,
            "tie_correction": tie_term,
            "continuity_correction": True,
        },
    )


def friedman(groups: Sequence[ArrayLike]) -> TestResult:
    """Friedman repeated-measures omnibus over ``k >= 3`` arms (Friedman 1937).

    ``groups`` is a sequence of ``k`` equal-length samples — ``groups[j][i]`` is
    arm *j*'s score in block (seed) *i*. Within each block the ``k`` arm scores
    are ranked (average ties); with ``R_j`` the rank sum of arm *j* the statistic
    is

    ``chi2 = (12 / (n k (k+1)) * sum_j R_j^2 - 3 n (k+1)) / C``

    where the tie correction ``C = 1 - sum(t^3 - t) / (n (k^3 - k))`` over all tie
    groups. Under H0 it is approximately chi-square with ``df = k - 1``.

    Args:
        groups: ``k`` aligned per-arm samples (each of length ``n`` blocks).

    Returns:
        A :class:`~slow_wave.eval.schema.TestResult` with ``test="friedman"``;
        ``statistic`` is the (tie-corrected) chi-square, ``n`` is the number of
        blocks, and ``detail`` carries ``df``, ``k``, ``rank_sums``, and
        ``tie_correction``.

    Raises:
        ValueError: If ``k < 3`` or the groups are not all the same length.
    """
    cols = [_as1d(g) for g in groups]
    k = len(cols)
    if k < 3:
        raise ValueError(f"friedman requires k >= 3 arms; got {k}")
    n = cols[0].size
    if n == 0:
        raise ValueError("friedman requires at least one block")
    if any(c.size != n for c in cols):
        raise ValueError("friedman requires all groups to have the same length")
    data = np.column_stack(cols)  # shape (n_blocks, k)
    ranks = np.empty_like(data, dtype=float)
    tie_sum = 0.0
    for i in range(n):
        ranks[i] = _rankdata(data[i])
        tie_sum += _tie_term(data[i])
    rank_sums = ranks.sum(axis=0)
    ssbn = float(np.sum(rank_sums**2))
    chi2 = 12.0 / (n * k * (k + 1)) * ssbn - 3.0 * n * (k + 1)
    correction = 1.0 - tie_sum / (n * (k**3 - k))
    if correction != 0.0:
        chi2 /= correction
    df = k - 1
    p = _chi2_sf(chi2, df)
    return TestResult(
        test="friedman",
        statistic=float(chi2),
        p_value=float(p),
        n=int(n),
        detail={
            "df": df,
            "k": k,
            "rank_sums": [float(v) for v in rank_sums],
            "tie_correction": float(tie_sum),
        },
    )


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #
def holm_correction(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down correction (Holm 1979), input order preserved.

    Sorts the ``m`` p-values ascending, scales the ``i``-th smallest by its
    remaining-test multiplier ``(m - i)``, enforces monotone non-decreasing
    adjusted p-values, and clips to ``1.0``. Because the adjusted sequence is
    monotone, ``reject := adjusted_p <= alpha`` reproduces the step-down decision
    exactly (the first failure stops all subsequent rejections).

    Args:
        p_values: The raw per-comparison p-values.
        alpha: Family-wise error rate to control.

    Returns:
        A list of ``(adjusted_p, reject)`` pairs in the **input order**.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")  # ascending; stable for determinism
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * float(p[idx])
        running = max(running, scaled)
        adjusted_sorted[rank] = min(1.0, running)
    out: list[tuple[float, bool]] = [(0.0, False)] * m
    for rank, idx in enumerate(order):
        adj = float(adjusted_sorted[rank])
        out[idx] = (adj, adj <= alpha)
    return out


# --------------------------------------------------------------------------- #
# rliable comparison primitives
# --------------------------------------------------------------------------- #
def probability_of_improvement(a: ArrayLike, b: ArrayLike) -> float:
    """rliable probability of improvement ``P(a > b)`` over all cross pairs.

    ``P(a > b) = (#(a_i > b_j) + 0.5 * #(a_i == b_j)) / (n_a * n_b)`` (Agarwal et
    al. 2021): ties contribute ``0.5``. It is ``1.0`` when every ``a`` strictly
    dominates every ``b`` and ``0.5`` when the samples are identical / all ties.

    Args:
        a: The candidate sample (1-D array-like).
        b: The reference sample.

    Returns:
        The probability of improvement in ``[0, 1]``; ``0.5`` if either sample is
        empty (no information).
    """
    av = _as1d(a)
    bv = _as1d(b)
    na, nb = av.size, bv.size
    if na == 0 or nb == 0:
        return 0.5
    diff = av[:, None] - bv[None, :]
    wins = int(np.count_nonzero(diff > 0))
    ties = int(np.count_nonzero(diff == 0))
    return (wins + 0.5 * ties) / (na * nb)


def performance_profile(
    samples: ArrayLike, taus: Sequence[float]
) -> list[list[float]]:
    """Performance profile: the run-score survival curve over thresholds ``taus``.

    For each threshold ``tau`` reports the fraction of runs whose score is
    ``>= tau`` (Agarwal et al. 2021). The curve is non-increasing in ``tau`` for
    ascending ``taus``.

    Args:
        samples: The per-run scores (1-D array-like).
        taus: The thresholds to evaluate the profile at.

    Returns:
        ``[[tau, fraction >= tau], ...]`` in the order of ``taus`` (a list of
        2-element lists so the result is JSON-dumpable as in
        :class:`~slow_wave.eval.schema.StatsReport`).
    """
    x = _as1d(samples)
    n = x.size
    profile: list[list[float]] = []
    for tau in taus:
        frac = 0.0 if n == 0 else float(np.count_nonzero(x >= tau)) / n
        profile.append([float(tau), frac])
    return profile
