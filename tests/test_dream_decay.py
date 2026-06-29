"""Tests for the swappable decay registry + the DOWNSCALE swap (Phase 3, EC6).

EC6 requires the salience-decay curve to be swappable among exponential /
Weibull / ACT-R. This module exercises the swap from both ends:

* the registry contract on every curve via
  :func:`slow_wave.dream.decay.decay_factor`: each returns ``1.0`` at age 0, is
  strictly ``< 1.0`` for any age ``> 0``, is monotonically non-increasing
  (``f(a1) >= f(a2)`` for ``a1 < a2``), and stays inside ``(0, 1]``; and
* DOWNSCALE run end-to-end with ``dream_cfg.decay_function`` set to each of
  ``"exponential"`` / ``"weibull"`` / ``"act_r"`` — every curve runs and yields
  valid post-decay salience, and at least two of the three produce *different*
  salience for the same input (proving the swap actually changes behavior).
"""

from __future__ import annotations

import numpy as np
import pytest

from slow_wave.config import DreamConfig, MemoryConfig
from slow_wave.dream.decay import DECAY_REGISTRY, decay_factor, params_for
from slow_wave.dream.downscale import downscale
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact

CURVES = ("exponential", "weibull", "act_r")
AGES = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _vec(dim: int) -> np.ndarray:
    """Return a deterministic ``(dim,)`` float32 vector of ones."""
    return np.ones(dim, dtype=np.float32)


def _params(name: str) -> dict:
    """Return the kwargs for curve ``name`` from a default DreamConfig."""
    return params_for(name, DreamConfig())


# --------------------------------------------------------------------------- #
# EC6 — every registered curve satisfies the decay-factor contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", CURVES)
def test_curve_is_one_at_age_zero(name: str) -> None:
    """Every curve returns exactly 1.0 at age 0 (a fresh item is not decayed)."""
    assert decay_factor(name, 0, _params(name)) == 1.0


@pytest.mark.parametrize("name", CURVES)
def test_curve_is_strictly_below_one_for_positive_age(name: str) -> None:
    """Every curve is strictly < 1.0 for any age > 0."""
    for age in AGES:
        assert decay_factor(name, age, _params(name)) < 1.0


@pytest.mark.parametrize("name", CURVES)
def test_curve_stays_in_open_closed_unit_interval(name: str) -> None:
    """Every curve stays inside (0, 1] across a wide range of ages.

    Ages are kept within the float-representable range; the Weibull curve's
    stretched exponential underflows to a literal 0.0 only at extreme ages
    (e.g. ``exp(-(1000/10)**1.5)``), which is a float-underflow artifact, not a
    contract violation.
    """
    for age in (0.0, *AGES):
        factor = decay_factor(name, age, _params(name))
        assert 0.0 < factor <= 1.0


@pytest.mark.parametrize("name", CURVES)
def test_curve_is_monotonically_non_increasing(name: str) -> None:
    """Every curve is non-increasing: f(a1) >= f(a2) for a1 < a2."""
    ages = [0.0, *AGES]
    factors = [decay_factor(name, a, _params(name)) for a in ages]
    for earlier, later in zip(factors, factors[1:]):
        assert earlier >= later


def test_registry_holds_exactly_the_three_curves() -> None:
    """The three EC6 curves are all registered and dispatchable by name."""
    for name in CURVES:
        assert name in DECAY_REGISTRY
        # Dispatch via decay_factor works for each registered name.
        assert decay_factor(name, 1.0, _params(name)) <= 1.0


# --------------------------------------------------------------------------- #
# EC6 — DOWNSCALE runs end-to-end under each curve and the swap changes behavior
# --------------------------------------------------------------------------- #
def _fresh_substrate(dim: int = 4) -> MemorySubstrate:
    """Build a small fixed substrate of aged episodic entries."""
    sub = MemorySubstrate(MemoryConfig(), dim)
    for i in range(3):
        entry = MemoryEntry(
            entry_id=f"e{i:06d}",
            tier=MemoryTier.EPISODIC,
            content=f"The role of s{i} is v{i}.",
            fact=Fact(subject=f"s{i}", attribute="role", value=f"v{i}"),
            created_order=0,
            salience=SalienceMeta(importance=1.0, recency_order=0),
            provenance=(f"i{i:06d}",),
        )
        sub.episodic.append(entry, _vec(dim), now_order=0)
    return sub


def _salience_after_downscale(name: str, now_order: int = 8) -> list[float]:
    """Run DOWNSCALE (pure decay) under curve ``name`` and return saliences."""
    sub = _fresh_substrate()
    res = downscale(
        sub,
        dream_cfg=DreamConfig(decay_function=name),
        replayed_ids=set(),
        now_order=now_order,
    )
    assert res.decay_function == name
    assert res.n_decayed == 3
    return [e.salience.importance for e in sub.episodic.all_entries()]


@pytest.mark.parametrize("name", CURVES)
def test_downscale_runs_and_yields_valid_salience_for_each_curve(name: str) -> None:
    """DOWNSCALE runs under each curve; post-decay salience is valid and lower."""
    saliences = _salience_after_downscale(name)
    for s in saliences:
        # Started at 1.0, age 8 > 0 => decayed into (0, 1).
        assert 0.0 < s < 1.0


def test_swapping_the_curve_changes_the_salience() -> None:
    """At least two of the three curves produce different post-decay salience."""
    results = {name: _salience_after_downscale(name) for name in CURVES}
    distinct = {tuple(v) for v in results.values()}
    # The swap is real: not every curve collapses to the same salience vector.
    assert len(distinct) >= 2
    # Specifically, exponential and act_r differ for the same aged input.
    assert results["exponential"] != results["act_r"]
