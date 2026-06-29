"""Matched-budget controller + accuracy/compute Pareto frontier (Phase 4, WS-BUDGET).

Comparing consolidation strategies is only fair when they spend the **same**
budget: a dream arm that quietly burns 10x the tokens of the no-sleep baseline
would "win" on compute it was never charged for. This module equalizes the three
budget axes the bench cares about — total tokens, retrieval calls, and final
memory-vector count — across arms within a fractional tolerance and **records the
realized actuals** (FR5.2). When budgets cannot be matched (the ceilings
``oracle`` / ``long_context`` are *meant* to overspend), the honest artifact is
the accuracy-vs-compute **Pareto frontier**, which this module always produces.

Design principles
-----------------
* **Honesty by construction (DX2).** Every excluded ceiling and every
  out-of-tolerance arm is surfaced — in :class:`~slow_wave.eval.schema.ArmBudgetActuals`
  per-axis flags, in :attr:`~slow_wave.eval.schema.BudgetReport.notes`, and via
  ``logger`` — never silently dropped. The Pareto frontier is *always* emitted,
  so an unmatched grid still yields its reported artifact (FR5.2).
* **Robust common target.** The matching target is the **median** over included
  arms of each arm's per-axis mean. The median resists a single runaway arm
  dragging the whole target, which a mean would not.
* **Pure & deterministic (DX1).** No randomness, no I/O. Means/medians are exact
  reductions over the input; results are JSON-dumpable with stable order
  following the input dict's insertion order.
* **numpy + stdlib + pydantic + ``slow_wave`` only** (no scipy/pandas/matplotlib).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from slow_wave.eval.schema import (
    ArmBudgetActuals,
    ArmCost,
    BudgetReport,
    ParetoPoint,
)

logger = logging.getLogger(__name__)


# A 4-tuple describing one point on the cost/accuracy plane.
ParetoInput = tuple[str, float, float, float]


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of ``values`` as a float, or ``0.0`` if empty.

    Args:
        values: The numbers to average.

    Returns:
        ``float(np.mean(values))`` for a non-empty sequence, else ``0.0`` (an
        empty sequence never raises — a missing arm contributes a zero mean).
    """
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _median(values: Sequence[float]) -> float:
    """Return the median of ``values`` as a float, or ``0.0`` if empty.

    Args:
        values: The numbers to take the median of.

    Returns:
        ``float(np.median(values))`` for a non-empty sequence, else ``0.0`` (an
        empty included set yields a zero target rather than ``nan``).
    """
    if len(values) == 0:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def _within_tolerance(mean: float, target: float, tolerance: float) -> bool:
    """Whether ``mean`` is within ``tolerance`` (fractional) of ``target``.

    Implements the contract rule ``|mean - target| <= tolerance * max(target, 1)``.
    The ``max(target, 1)`` floor keeps the band sane when the target is small or
    zero (so a zero target still admits a small absolute slack, not an exact
    match only).

    Args:
        mean: The arm's realized per-axis mean.
        target: The common per-axis target.
        tolerance: The fractional tolerance (e.g. ``0.15`` for +/-15%).

    Returns:
        ``True`` iff ``mean`` falls inside the tolerance band around ``target``.
    """
    return abs(mean - target) <= tolerance * max(target, 1.0)


def pareto_frontier(points: Sequence[ParetoInput]) -> list[ParetoPoint]:
    """Mark the accuracy-vs-compute Pareto-non-dominated points (FR5.2).

    The objective is to **maximize accuracy** and **minimize compute tokens**. A
    point ``p`` is *dominated* (hence off the frontier) iff some other point ``q``
    has ``q.accuracy >= p.accuracy`` AND ``q.compute_tokens <= p.compute_tokens``
    with at least one of those inequalities strict; otherwise ``p`` is on the
    frontier. Ties (two points with identical accuracy and tokens) dominate
    neither — both stay on the frontier. ``memory_vectors`` is carried through for
    reporting but is **not** part of the domination test.

    Args:
        points: Sequence of ``(arm_name, accuracy, compute_tokens, memory_vectors)``
            tuples, typically one per arm using per-arm means.

    Returns:
        One :class:`~slow_wave.eval.schema.ParetoPoint` per input, in input order
        (deterministic, stable), with :attr:`~slow_wave.eval.schema.ParetoPoint.on_frontier`
        set per the rule above.
    """
    result: list[ParetoPoint] = []
    for i, (name_i, acc_i, tok_i, vec_i) in enumerate(points):
        dominated = False
        for j, (_, acc_j, tok_j, _) in enumerate(points):
            if i == j:
                continue
            if (
                acc_j >= acc_i
                and tok_j <= tok_i
                and (acc_j > acc_i or tok_j < tok_i)
            ):
                dominated = True
                break
        result.append(
            ParetoPoint(
                arm_name=name_i,
                accuracy=float(acc_i),
                compute_tokens=float(tok_i),
                memory_vectors=float(vec_i),
                on_frontier=not dominated,
            )
        )
    return result


def match_budget(
    arm_costs: dict[str, list[ArmCost]],
    *,
    accuracies: dict[str, list[float]],
    tolerance: float = 0.15,
    target_tokens: float | None = None,
    exclude: Sequence[str] = ("oracle", "long_context", "aa"),
) -> BudgetReport:
    """Equalize budgets across arms within tolerance and report the verdict (FR5.2).

    For each arm the per-axis mean over its seeds is computed for the three budget
    axes (total tokens, retrieval calls, final memory-vector count). The common
    matching target on each axis is the **median over the included (non-excluded)
    arms** of that axis's per-arm means — a robust common budget. Each arm is then
    flagged per axis via :func:`_within_tolerance`. The overall grid is
    ``matched`` iff **every included arm** is within tolerance on **all three**
    axes; excluded ceilings (``oracle`` / ``long_context`` / ``aa``) have their
    actuals recorded and flagged but never gate the verdict (DX2).

    The accuracy-vs-compute :func:`pareto_frontier` is **always** produced over
    *all* arms — it is the reported artifact when budget matching is infeasible
    (FR5.2).

    Args:
        arm_costs: Map of arm name to its per-seed :class:`~slow_wave.eval.schema.ArmCost`
            list (one entry per seed). Insertion order fixes the report order.
        accuracies: Map of arm name to its per-seed ACC list, supplying the Pareto
            accuracy axis. A missing arm contributes a zero mean accuracy.
        tolerance: Fractional tolerance for the within-tolerance band (default
            ``0.15`` == +/-15%).
        target_tokens: Explicit token target; when ``None`` (default) the token
            target is the median-of-means over included arms. The retrieval and
            memory-vector targets are *always* derived as the median-of-means
            (there is no override for them).
        exclude: Arm names excluded from the matching verdict and target
            derivation — the cost ceilings/controls (default ``oracle``,
            ``long_context``, ``aa``). They still appear in ``per_arm`` and
            ``pareto``.

    Returns:
        A :class:`~slow_wave.eval.schema.BudgetReport` with the per-axis targets,
        per-arm actuals + verdicts for every arm, the overall ``matched`` flag,
        the always-present Pareto frontier, and DX2 honesty notes naming the
        excluded ceilings.
    """
    exclude_set = set(exclude)

    # Per-arm per-axis means (one mean per axis per arm, over its seeds).
    means: dict[str, tuple[float, float, float]] = {}
    for arm, costs in arm_costs.items():
        mean_tokens = _mean([c.total_tokens for c in costs])
        mean_retrieval = _mean([c.retrieval_calls for c in costs])
        mean_vectors = _mean([c.memory_vectors for c in costs])
        means[arm] = (mean_tokens, mean_retrieval, mean_vectors)

    # Targets: median over INCLUDED arms of each axis's per-arm means.
    included = [arm for arm in arm_costs if arm not in exclude_set]
    if target_tokens is None:
        resolved_target_tokens = _median([means[arm][0] for arm in included])
    else:
        resolved_target_tokens = float(target_tokens)
    target_retrieval = _median([means[arm][1] for arm in included])
    target_memory_vectors = _median([means[arm][2] for arm in included])

    # Per-arm actuals + verdicts for EVERY arm (excluded ones are flagged too,
    # but they do not gate the overall verdict).
    per_arm: list[ArmBudgetActuals] = []
    overall_matched = True
    for arm in arm_costs:
        mean_tokens, mean_retrieval, mean_vectors = means[arm]
        tokens_ok = _within_tolerance(mean_tokens, resolved_target_tokens, tolerance)
        retrieval_ok = _within_tolerance(mean_retrieval, target_retrieval, tolerance)
        memory_ok = _within_tolerance(mean_vectors, target_memory_vectors, tolerance)
        arm_matched = tokens_ok and retrieval_ok and memory_ok
        per_arm.append(
            ArmBudgetActuals(
                arm_name=arm,
                mean_total_tokens=mean_tokens,
                mean_retrieval_calls=mean_retrieval,
                mean_memory_vectors=mean_vectors,
                tokens_within_tolerance=tokens_ok,
                retrieval_within_tolerance=retrieval_ok,
                memory_within_tolerance=memory_ok,
                matched=arm_matched,
            )
        )
        if arm not in exclude_set and not arm_matched:
            overall_matched = False
            logger.info(
                "Budget mismatch on included arm %r: tokens_ok=%s retrieval_ok=%s "
                "memory_ok=%s (means tok=%.1f ret=%.1f vec=%.1f; targets tok=%.1f "
                "ret=%.1f vec=%.1f)",
                arm,
                tokens_ok,
                retrieval_ok,
                memory_ok,
                mean_tokens,
                mean_retrieval,
                mean_vectors,
                resolved_target_tokens,
                target_retrieval,
                target_memory_vectors,
            )

    # Pareto frontier over ALL arms (always produced — FR5.2 reported artifact).
    pareto_points: list[ParetoInput] = []
    for arm in arm_costs:
        mean_tokens, _, mean_vectors = means[arm]
        mean_acc = _mean(accuracies.get(arm, []))
        pareto_points.append((arm, mean_acc, mean_tokens, mean_vectors))
    pareto = pareto_frontier(pareto_points)

    # DX2 honesty notes: name every excluded ceiling present in the grid.
    notes: list[str] = []
    for arm in exclude:
        if arm in arm_costs:
            note = (
                f"Arm '{arm}' excluded from budget matching (cost ceiling/control): "
                "its actuals are recorded in per_arm but do not gate `matched`."
            )
            notes.append(note)
            logger.info(note)
    if not overall_matched:
        msg = (
            "Budgets not matched within tolerance %.3f; Pareto frontier is the "
            "reported artifact (FR5.2)." % tolerance
        )
        notes.append(msg)
        logger.warning(msg)

    return BudgetReport(
        matched=overall_matched,
        tolerance=tolerance,
        target_tokens=resolved_target_tokens,
        target_retrieval=target_retrieval,
        target_memory_vectors=target_memory_vectors,
        per_arm=per_arm,
        pareto=pareto,
        notes=notes,
    )
