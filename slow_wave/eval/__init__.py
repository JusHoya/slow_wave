"""Evaluation metrics, control battery & statistics for the Slow Wave bench.

Phase 2 ships the continual-learning point metrics derived from the accuracy
matrix ``R[i,j]`` (ACC, BWT, FWT, per-task forgetting); see
:mod:`slow_wave.eval.metrics`. Phase 4 adds the full evaluation apparatus (PRD §8
Phase 4; ``docs/PHASE4_CONTRACT.md``):

* :mod:`slow_wave.eval.arms` — the nine control arms (registry + builder).
* :mod:`slow_wave.eval.prune_metrics` — mechanism-level prune precision/recall/F1
  and the decay-vs-relevance calibration curve (decoupled from accuracy).
* :mod:`slow_wave.eval.stats` — bootstrap CIs, rliable robust aggregates,
  significance tests, effect sizes, and multiple-comparison correction (numpy).
* :mod:`slow_wave.eval.budget_match` — the matched-budget controller + Pareto.
* :mod:`slow_wave.eval.prereg` — the preregistration guard + primary endpoint.
* :mod:`slow_wave.eval.controls` — the temperature-0 stability + memory-drift
  bias controls.
* :mod:`slow_wave.eval.harness` — the one harness wiring it all together, and
  :mod:`slow_wave.eval.runner` — the ``python -m slow_wave.eval.runner`` CLI.
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
