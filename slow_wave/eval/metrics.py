"""Continual-learning point metrics over an accuracy matrix (Phase 2, WS-EVAL).

This module turns a continual-learning **accuracy matrix** ``R[i][j]`` (built by
the stream oracle or the wake agent; see
:class:`~slow_wave.stream.schema.AccuracyMatrix`) into the standard scalar
metrics used to characterise transfer and catastrophic forgetting. It is the
point-metric layer the rest of the bench reports against; Phase 4 layers a
statistics suite (CIs, significance) **on top** of these primitives.

Design principles
-----------------
* **Pure & deterministic.** Every function is a closed-form reduction over the
  matrix — no randomness, no I/O, no heavy dependencies. The only imports are
  :mod:`pydantic` (for the result model) and
  :class:`~slow_wave.stream.schema.AccuracyMatrix`. Plain-Python arithmetic is
  used throughout (these are a handful of sums over an ``n_tasks x n_tasks``
  grid; numpy would be overkill).
* **Exact, literature-grounded definitions.** ``R`` is ``list[list[float]]`` of
  shape ``n_tasks x n_tasks`` with ``R[i][j]`` the accuracy on task *j* after
  training through task *i* (0-based, every entry in ``[0, 1]``). ``T`` denotes
  ``n_tasks``. Backward transfer follows Lopez-Paz & Ranzato (2017); per-task
  forgetting follows Chaudhry et al. (2018).
* **Degenerate streams never raise.** A single-task stream (``T == 1``) has no
  earlier tasks to transfer to/from, and an empty matrix (``T == 0``) has no
  tasks at all; both yield zeros / empty lists rather than dividing by zero.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from slow_wave.stream.schema import AccuracyMatrix


class ContinualMetrics(BaseModel):
    """Bundle of scalar continual-learning metrics for one accuracy matrix.

    Attributes:
        acc: Average accuracy over every task after the final task (ACC).
        bwt: Backward transfer; negative values indicate the model forgot
            earlier tasks (Lopez-Paz & Ranzato 2017).
        fwt: Forward transfer; positive values indicate above-baseline accuracy
            on a task *before* it was trained.
        average_forgetting: Mean per-task forgetting; ``>= 0`` typically, larger
            means more forgetting.
        per_task_forgetting: Per-task forgetting for tasks ``0 .. T-2`` (the last
            task cannot be forgotten yet), so its length is ``n_tasks - 1``.
    """

    model_config = ConfigDict(extra="forbid")

    acc: float
    bwt: float
    fwt: float
    average_forgetting: float
    per_task_forgetting: list[float]


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of ``values``, or ``0.0`` when empty.

    Args:
        values: The numbers to average.

    Returns:
        ``sum(values) / len(values)`` for a non-empty list, else ``0.0``.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def average_accuracy(R: list[list[float]]) -> float:
    """Compute average accuracy (ACC) over the final row of ``R``.

    ACC is the mean over *j* of ``R[T-1][j]`` — the accuracy on every task once
    the whole stream has been processed (after the last task).

    Args:
        R: The ``n_tasks x n_tasks`` accuracy matrix.

    Returns:
        The mean of the last row, or ``0.0`` when ``T == 0`` (empty matrix).
    """
    T = len(R)
    if T == 0:
        return 0.0
    return _mean(R[T - 1])


def backward_transfer(R: list[list[float]]) -> float:
    """Compute backward transfer (BWT) over ``R`` (Lopez-Paz & Ranzato 2017).

    ``BWT = (1 / (T - 1)) * sum_{i=0}^{T-2} (R[T-1][i] - R[i][i])`` — the average,
    over earlier tasks, of the change between accuracy right after a task was
    learned (its diagonal entry) and accuracy at the end of the stream (the last
    row). Negative values mean the model forgot earlier tasks.

    Args:
        R: The ``n_tasks x n_tasks`` accuracy matrix.

    Returns:
        The backward-transfer scalar, or ``0.0`` when ``T <= 1`` (no earlier
        task to transfer back to).
    """
    T = len(R)
    if T <= 1:
        return 0.0
    deltas = [R[T - 1][i] - R[i][i] for i in range(T - 1)]
    return _mean(deltas)


def forward_transfer(
    R: list[list[float]], baseline: list[float] | None = None
) -> float:
    """Compute forward transfer (FWT) over ``R``.

    ``FWT = (1 / (T - 1)) * sum_{i=1}^{T-1} (R[i-1][i] - baseline[i])`` — the
    average, over later tasks, of accuracy on a task *just before* it was trained
    (the entry one step above the diagonal) relative to a ``baseline``. The
    baseline models random / pre-training accuracy and defaults to zeros, in
    which case FWT is simply the mean of the just-above-diagonal entries.

    Args:
        R: The ``n_tasks x n_tasks`` accuracy matrix.
        baseline: Per-task baseline accuracies indexed by task. Only entries
            ``baseline[1] .. baseline[T-1]`` are read. Defaults to all-zeros
            (length ``T``).

    Returns:
        The forward-transfer scalar, or ``0.0`` when ``T <= 1`` (no later task
        to transfer forward to).
    """
    T = len(R)
    if T <= 1:
        return 0.0
    if baseline is None:
        baseline = [0.0] * T
    deltas = [R[i - 1][i] - baseline[i] for i in range(1, T)]
    return _mean(deltas)


def per_task_forgetting(R: list[list[float]]) -> list[float]:
    """Compute per-task forgetting over ``R`` (Chaudhry et al. 2018).

    For each task *j* in ``0 .. T-2`` the forgetting is
    ``max_{l in 0..T-2} R[l][j] - R[T-1][j]`` — the drop from the best accuracy
    ever recorded for task *j* (over all training stages before the last) to its
    final accuracy. The last task is excluded: it has only just been learned and
    cannot have been forgotten yet.

    Args:
        R: The ``n_tasks x n_tasks`` accuracy matrix.

    Returns:
        A list of length ``T - 1`` (one entry per task ``0 .. T-2``), or ``[]``
        when ``T <= 1``.
    """
    T = len(R)
    if T <= 1:
        return []
    return [max(R[l][j] for l in range(T - 1)) - R[T - 1][j] for j in range(T - 1)]


def average_forgetting(R: list[list[float]]) -> float:
    """Compute the mean of :func:`per_task_forgetting` over ``R``.

    Args:
        R: The ``n_tasks x n_tasks`` accuracy matrix.

    Returns:
        The mean per-task forgetting, or ``0.0`` when there are no earlier tasks
        (``T <= 1``, so the per-task list is empty).
    """
    return _mean(per_task_forgetting(R))


def compute_continual_metrics(matrix: AccuracyMatrix) -> ContinualMetrics:
    """Bundle every point metric for ``matrix`` into a :class:`ContinualMetrics`.

    Reads ``matrix.R`` and computes :func:`average_accuracy`,
    :func:`backward_transfer`, :func:`forward_transfer` (zero baseline),
    :func:`per_task_forgetting`, and :func:`average_forgetting`.

    Args:
        matrix: A validated accuracy matrix to summarise.

    Returns:
        The populated :class:`ContinualMetrics` for the matrix.
    """
    R = matrix.R
    return ContinualMetrics(
        acc=average_accuracy(R),
        bwt=backward_transfer(R),
        fwt=forward_transfer(R),
        average_forgetting=average_forgetting(R),
        per_task_forgetting=per_task_forgetting(R),
    )
