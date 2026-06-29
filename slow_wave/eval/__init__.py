"""Evaluation metrics for the Slow Wave bench (Phase 2+).

Phase 2 ships the continual-learning point metrics derived from the accuracy
matrix ``R[i,j]`` (ACC, BWT, FWT, per-task forgetting); see
:mod:`slow_wave.eval.metrics` and ``docs/PHASE2_CONTRACT.md``. The Phase 4
statistics suite (bootstrap CIs, rliable robust aggregates, significance tests)
will build on top of these.
"""

from __future__ import annotations

from slow_wave.eval.metrics import (
    ContinualMetrics,
    average_accuracy,
    average_forgetting,
    backward_transfer,
    compute_continual_metrics,
    forward_transfer,
    per_task_forgetting,
)

__all__ = [
    "ContinualMetrics",
    "average_accuracy",
    "average_forgetting",
    "backward_transfer",
    "compute_continual_metrics",
    "forward_transfer",
    "per_task_forgetting",
]
