"""Tests for slow_wave.eval.metrics (Phase 2, WS-EVAL).

These tests pin the exact continual-learning point-metric definitions from the
Phase 2 contract. They use small, hand-built matrices with arithmetic worked out
by hand so every scalar is asserted against a known number (via
:func:`pytest.approx`), plus the canonical no-forgetting reference: the
lower-triangular ones matrix the Phase 1 oracle produces (``R[i][j] == 1`` for
``j <= i``, else ``0``).
"""

from __future__ import annotations

import pytest

from slow_wave.eval.metrics import (
    ContinualMetrics,
    average_accuracy,
    average_forgetting,
    backward_transfer,
    compute_continual_metrics,
    forward_transfer,
    per_task_forgetting,
)
from slow_wave.stream.schema import AccuracyMatrix, CLScenario


# --------------------------------------------------------------------------- #
# Reference fixtures
# --------------------------------------------------------------------------- #
def _lower_triangular_ones(n: int) -> list[list[float]]:
    """Return the no-forgetting oracle matrix: 1.0 for j <= i, else 0.0."""
    return [[1.0 if j <= i else 0.0 for j in range(n)] for i in range(n)]


# A hand-crafted forgetting matrix (T == 3) with values worked out by hand:
#   ACC                 = mean(R[2]) = (0.5 + 0.7 + 1.0) / 3
#   BWT                 = ((0.5-1.0) + (0.7-1.0)) / 2          = -0.4
#   per_task_forgetting = [max(1.0,0.8)-0.5, max(0.3,1.0)-0.7] = [0.5, 0.3]
#   average_forgetting  = (0.5 + 0.3) / 2                      = 0.4
#   FWT (zero baseline) = (R[0][1] + R[1][2]) / 2 = (0.3 + 0.4) / 2 = 0.35
_FORGETTING_R = [
    [1.0, 0.3, 0.2],  # after task 0 (some forward transfer to tasks 1, 2)
    [0.8, 1.0, 0.4],  # after task 1: task 0 dropped 1.0 -> 0.8
    [0.5, 0.7, 1.0],  # after task 2: task 0 -> 0.5, task 1 -> 0.7
]


# --------------------------------------------------------------------------- #
# No-forgetting reference (lower-triangular ones)
# --------------------------------------------------------------------------- #
def test_lower_triangular_ones_no_forgetting() -> None:
    """The oracle skeleton: perfect ACC, zero BWT, zero forgetting."""
    R = _lower_triangular_ones(4)

    assert average_accuracy(R) == pytest.approx(1.0)
    assert backward_transfer(R) == pytest.approx(0.0)
    assert average_forgetting(R) == pytest.approx(0.0)

    ptf = per_task_forgetting(R)
    assert len(ptf) == 3  # n_tasks - 1
    assert all(v == pytest.approx(0.0) for v in ptf)


def test_lower_triangular_ones_forward_transfer_zero() -> None:
    """No above-diagonal mass => forward transfer is zero."""
    R = _lower_triangular_ones(4)
    assert forward_transfer(R) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Hand-crafted forgetting matrix (known values)
# --------------------------------------------------------------------------- #
def test_forgetting_matrix_exact_values() -> None:
    """Every metric equals the hand-computed number; BWT < 0, forgetting > 0."""
    R = _FORGETTING_R

    assert average_accuracy(R) == pytest.approx((0.5 + 0.7 + 1.0) / 3)
    assert backward_transfer(R) == pytest.approx(-0.4)
    assert per_task_forgetting(R) == pytest.approx([0.5, 0.3])
    assert average_forgetting(R) == pytest.approx(0.4)
    assert forward_transfer(R) == pytest.approx(0.35)

    # Direction sanity (the EC6 spirit): the baseline forgot.
    assert backward_transfer(R) < 0
    assert average_forgetting(R) > 0


def test_forward_transfer_explicit_baseline() -> None:
    """A non-zero baseline is subtracted from the above-diagonal entries."""
    R = _FORGETTING_R
    baseline = [0.0, 0.1, 0.1]
    # ((0.3 - 0.1) + (0.4 - 0.1)) / 2 = (0.2 + 0.3) / 2 = 0.25
    assert forward_transfer(R, baseline=baseline) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Edge cases: T == 1 and T == 0 (no raising, all zeros)
# --------------------------------------------------------------------------- #
def test_single_task_returns_zeros() -> None:
    """T == 1: ACC is the lone entry; transfer/forgetting are zero, no raise."""
    R = [[0.9]]

    assert average_accuracy(R) == pytest.approx(0.9)
    assert backward_transfer(R) == pytest.approx(0.0)
    assert forward_transfer(R) == pytest.approx(0.0)
    assert per_task_forgetting(R) == []
    assert average_forgetting(R) == pytest.approx(0.0)


def test_empty_matrix_returns_zeros() -> None:
    """T == 0: bare-list helpers return zeros / empty without raising."""
    R: list[list[float]] = []

    assert average_accuracy(R) == pytest.approx(0.0)
    assert backward_transfer(R) == pytest.approx(0.0)
    assert forward_transfer(R) == pytest.approx(0.0)
    assert per_task_forgetting(R) == []
    assert average_forgetting(R) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# compute_continual_metrics: round-trips through an AccuracyMatrix
# --------------------------------------------------------------------------- #
def test_compute_continual_metrics_roundtrip() -> None:
    """Bundles the matrix into a valid, dumpable ContinualMetrics."""
    matrix = AccuracyMatrix(
        n_tasks=3, scenario=CLScenario.TASK_INCREMENTAL, R=_FORGETTING_R
    )
    metrics = compute_continual_metrics(matrix)

    assert isinstance(metrics, ContinualMetrics)
    assert metrics.acc == pytest.approx((0.5 + 0.7 + 1.0) / 3)
    assert metrics.bwt == pytest.approx(-0.4)
    assert metrics.fwt == pytest.approx(0.35)
    assert metrics.average_forgetting == pytest.approx(0.4)
    assert metrics.per_task_forgetting == pytest.approx([0.5, 0.3])

    dumped = metrics.model_dump()
    assert set(dumped) == {
        "acc",
        "bwt",
        "fwt",
        "average_forgetting",
        "per_task_forgetting",
    }
    assert dumped["per_task_forgetting"] == pytest.approx([0.5, 0.3])


def test_compute_continual_metrics_no_forgetting_oracle() -> None:
    """The oracle matrix bundles to perfect ACC and zero forgetting/BWT."""
    matrix = AccuracyMatrix(
        n_tasks=3,
        scenario=CLScenario.TASK_INCREMENTAL,
        R=_lower_triangular_ones(3),
    )
    metrics = compute_continual_metrics(matrix)

    assert metrics.acc == pytest.approx(1.0)
    assert metrics.bwt == pytest.approx(0.0)
    assert metrics.average_forgetting == pytest.approx(0.0)
    assert metrics.per_task_forgetting == pytest.approx([0.0, 0.0])
