"""Tests for slow_wave.eval.budget_match (Phase 4, WS-BUDGET).

These tests pin the matched-budget controller and the accuracy/compute Pareto
frontier from the Phase 4 contract (EC4). They use **hand-built** ``ArmCost``
lists with means worked out by hand so every verdict is exact:

* (a) all included arms within tolerance => ``matched is True`` and
  ``target_tokens`` equals the median-of-means;
* (b) perturb one included arm beyond tolerance => ``matched is False`` and that
  arm's ``tokens_within_tolerance is False`` while its actuals are still recorded;
* (c) ``pareto_frontier`` on a known set marks dominating points on-frontier and
  dominated ones off it;
* (d) excluded ceilings appear in ``per_arm``/``pareto`` but never flip
  ``matched``.

Everything is asserted to be deterministic and JSON-dumpable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from slow_wave.eval.budget_match import match_budget, pareto_frontier
from slow_wave.eval.schema import (
    ArmBudgetActuals,
    ArmCost,
    BudgetReport,
    ParetoPoint,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cost(total_tokens: int, retrieval: int, vectors: int) -> ArmCost:
    """Build an ArmCost pinned on the three matched axes (others are filler)."""
    return ArmCost(
        input_tokens=total_tokens,
        output_tokens=0,
        total_tokens=total_tokens,
        api_calls=0,
        retrieval_calls=retrieval,
        memory_vectors=vectors,
        memory_bytes=0,
    )


def _two_seed_arm(
    mean_tokens: int, mean_retrieval: int, mean_vectors: int
) -> list[ArmCost]:
    """Two seeds straddling the requested means (so means exercise averaging).

    Each axis is set to ``mean - d`` and ``mean + d`` so the per-arm mean is
    exactly the requested value. ``d`` is small and axis-specific.
    """
    return [
        _cost(mean_tokens - 50, mean_retrieval - 2, mean_vectors - 5),
        _cost(mean_tokens + 50, mean_retrieval + 2, mean_vectors + 5),
    ]


# Three INCLUDED arms whose per-axis means are 900/1000/1100 (tokens),
# 18/20/22 (retrieval), 90/100/110 (vectors). Medians: 1000 / 20 / 100.
#   tokens band  = 1000 +/- 0.15*1000 = [850, 1150]   -> 900,1000,1100 all in
#   retrieval    =   20 +/- 0.15*20   = [17, 23]       -> 18,20,22 all in
#   vectors      =  100 +/- 0.15*100  = [85, 115]      -> 90,100,110 all in
def _included_arms() -> dict[str, list[ArmCost]]:
    return {
        "no_sleep": _two_seed_arm(1000, 20, 100),
        "replay_only": _two_seed_arm(1100, 22, 110),
        "full_dream": _two_seed_arm(900, 18, 90),
    }


# Excluded ceilings/control: oracle & long_context overspend wildly,
# aa happens to sit on-budget. None of them affect the verdict or the targets.
def _excluded_arms() -> dict[str, list[ArmCost]]:
    return {
        "oracle": _two_seed_arm(5000, 100, 500),
        "long_context": _two_seed_arm(50000, 200, 5000),
        "aa": _two_seed_arm(1000, 20, 100),
    }


def _accuracies() -> dict[str, list[float]]:
    return {
        "no_sleep": [0.50, 0.50],
        "replay_only": [0.60, 0.60],
        "full_dream": [0.70, 0.70],
        "oracle": [0.95, 0.95],
        "long_context": [0.97, 0.97],
        "aa": [0.50, 0.50],
    }


# --------------------------------------------------------------------------- #
# (a) all included arms within tolerance -> matched True, target = median
# --------------------------------------------------------------------------- #
def test_all_included_within_tolerance_matched_true() -> None:
    """Every included arm in-band => matched True; targets are median-of-means."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    report = match_budget(arm_costs, accuracies=_accuracies())

    assert isinstance(report, BudgetReport)
    assert report.matched is True

    # Targets are the median over INCLUDED arms only.
    assert report.target_tokens == pytest.approx(np.median([1000.0, 1100.0, 900.0]))
    assert report.target_tokens == pytest.approx(1000.0)
    assert report.target_retrieval == pytest.approx(20.0)
    assert report.target_memory_vectors == pytest.approx(100.0)

    # Every included arm is matched on all three axes.
    by_name = {a.arm_name: a for a in report.per_arm}
    for name in ("no_sleep", "replay_only", "full_dream"):
        a = by_name[name]
        assert a.tokens_within_tolerance is True
        assert a.retrieval_within_tolerance is True
        assert a.memory_within_tolerance is True
        assert a.matched is True


def test_per_arm_covers_every_arm_including_excluded() -> None:
    """per_arm holds an ArmBudgetActuals for EVERY arm, excluded ones included."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    report = match_budget(arm_costs, accuracies=_accuracies())

    names = {a.arm_name for a in report.per_arm}
    assert names == set(arm_costs)
    assert all(isinstance(a, ArmBudgetActuals) for a in report.per_arm)

    # Recorded actuals are the hand-built means.
    by_name = {a.arm_name: a for a in report.per_arm}
    assert by_name["oracle"].mean_total_tokens == pytest.approx(5000.0)
    assert by_name["long_context"].mean_memory_vectors == pytest.approx(5000.0)


# --------------------------------------------------------------------------- #
# (b) perturb one included arm beyond tolerance -> matched False, actuals kept
# --------------------------------------------------------------------------- #
def test_perturbed_included_arm_flips_matched_false() -> None:
    """An out-of-band included arm flips matched and its tokens flag, keeps actuals."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    # Blow full_dream's tokens far outside any plausible band (means -> 2000).
    arm_costs["full_dream"] = _two_seed_arm(2000, 18, 90)

    report = match_budget(arm_costs, accuracies=_accuracies())

    assert report.matched is False

    by_name = {a.arm_name: a for a in report.per_arm}
    fd = by_name["full_dream"]
    # The violating axis is flagged off...
    assert fd.tokens_within_tolerance is False
    assert fd.matched is False
    # ...while the other axes stay in tolerance and the actuals ARE recorded.
    assert fd.retrieval_within_tolerance is True
    assert fd.memory_within_tolerance is True
    assert fd.mean_total_tokens == pytest.approx(2000.0)
    assert fd.mean_retrieval_calls == pytest.approx(18.0)
    assert fd.mean_memory_vectors == pytest.approx(90.0)

    # The other included arms are still individually within tolerance.
    assert by_name["no_sleep"].matched is True
    assert by_name["replay_only"].matched is True


# --------------------------------------------------------------------------- #
# (c) pareto_frontier marks dominating points on, dominated points off
# --------------------------------------------------------------------------- #
def test_pareto_frontier_known_set() -> None:
    """Dominating (high-acc/low-cost) points are on-frontier; dominated are off."""
    points = [
        ("A", 0.90, 100.0, 10.0),  # frontier: best cost, high acc
        ("B", 0.50, 500.0, 50.0),  # dominated by A (lower acc AND higher cost)
        ("C", 0.95, 300.0, 30.0),  # frontier: highest acc (costs more than A)
        ("D", 0.70, 200.0, 20.0),  # dominated by A (0.90>=0.70 and 100<=200)
    ]
    result = pareto_frontier(points)

    # Order preserved.
    assert [p.arm_name for p in result] == ["A", "B", "C", "D"]
    on = {p.arm_name: p.on_frontier for p in result}
    assert on == {"A": True, "B": False, "C": True, "D": False}
    assert all(isinstance(p, ParetoPoint) for p in result)


def test_pareto_frontier_ties_both_on_frontier() -> None:
    """Identical points dominate neither (no strict inequality) -> both kept."""
    points = [
        ("X", 0.80, 100.0, 10.0),
        ("Y", 0.80, 100.0, 10.0),
    ]
    result = pareto_frontier(points)
    assert all(p.on_frontier for p in result)


def test_pareto_frontier_single_point_on_frontier() -> None:
    """A lone point is trivially non-dominated."""
    result = pareto_frontier([("solo", 0.42, 123.0, 7.0)])
    assert len(result) == 1
    assert result[0].on_frontier is True
    assert result[0].accuracy == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# (d) excluded ceilings appear in per_arm/pareto but never flip matched
# --------------------------------------------------------------------------- #
def test_excluded_ceilings_do_not_gate_matched() -> None:
    """Wildly-overspending ceilings appear everywhere but don't flip matched."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    report = match_budget(arm_costs, accuracies=_accuracies())

    assert report.matched is True  # despite oracle/long_context overspend

    by_name = {a.arm_name: a for a in report.per_arm}
    # The ceilings are recorded and (correctly) flagged out-of-tolerance...
    assert by_name["oracle"].tokens_within_tolerance is False
    assert by_name["oracle"].matched is False
    assert by_name["long_context"].matched is False
    # ...yet the overall verdict stayed True (they don't gate it).

    # They also appear on the Pareto frontier as points.
    pareto_names = {p.arm_name for p in report.pareto}
    assert {"oracle", "long_context", "aa"} <= pareto_names

    # The ceilings sit on the frontier (high accuracy), the cheap baselines off.
    on = {p.arm_name: p.on_frontier for p in report.pareto}
    assert on["full_dream"] is True
    assert on["oracle"] is True
    assert on["long_context"] is True
    assert on["no_sleep"] is False
    assert on["replay_only"] is False
    assert on["aa"] is False


def test_notes_name_excluded_ceilings() -> None:
    """DX2 honesty: notes name each excluded ceiling present in the grid."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    report = match_budget(arm_costs, accuracies=_accuracies())

    joined = " ".join(report.notes)
    for arm in ("oracle", "long_context", "aa"):
        assert arm in joined


def test_target_tokens_override_skips_included_median() -> None:
    """An explicit target_tokens is used verbatim; retrieval/memory stay medians."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    report = match_budget(
        arm_costs, accuracies=_accuracies(), target_tokens=900.0
    )
    assert report.target_tokens == pytest.approx(900.0)
    # The other axes are still derived as median-of-means over included arms.
    assert report.target_retrieval == pytest.approx(20.0)
    assert report.target_memory_vectors == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Determinism + JSON-dumpability
# --------------------------------------------------------------------------- #
def test_deterministic_across_two_runs() -> None:
    """Same inputs => byte-identical JSON (DX1)."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    r1 = match_budget(arm_costs, accuracies=_accuracies())
    r2 = match_budget(arm_costs, accuracies=_accuracies())
    dump1 = json.dumps(r1.model_dump(mode="json"), sort_keys=True)
    dump2 = json.dumps(r2.model_dump(mode="json"), sort_keys=True)
    assert dump1 == dump2


def test_report_is_json_dumpable() -> None:
    """The whole report round-trips through JSON without error."""
    arm_costs = {**_included_arms(), **_excluded_arms()}
    report = match_budget(arm_costs, accuracies=_accuracies())
    blob = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    restored = BudgetReport.model_validate(json.loads(blob))
    assert restored.matched == report.matched
    assert len(restored.per_arm) == len(report.per_arm)
    assert len(restored.pareto) == len(report.pareto)
