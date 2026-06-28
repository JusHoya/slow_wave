"""Probe-set construction and the accuracy-matrix oracle (Phase 1, WS3).

This module turns a generated :class:`~slow_wave.stream.schema.Stream` into the
held-out **probe set** that the bench queries, and computes a well-formed
continual-learning **accuracy matrix** ``R[i][j]`` against a trivial
perfect-memory **oracle** (see ``docs/PHASE1_CONTRACT.md`` — section "WS3 —
Probe set / R[i,j] / oracle", and Phase 1 exit criterion #6).

Design principles
-----------------
* **Offline construction is allowed to read labels.** Building the probe set is
  *offline* scoring code, so it may read relevance labels via the sanctioned
  :func:`~slow_wave.stream.schema.offline_labels` accessor to decide which items
  are ``signal``. The probes themselves carry only the surface ``query`` and a
  ground-truth ``answer`` used for offline scoring — never a relevance label.
* **One probe per probed key.** A probed key is a ``(subject, attribute)`` pair
  asserted by at least one ``signal`` item. Each probed key yields exactly one
  :class:`~slow_wave.stream.schema.Probe` whose canonical ``answer`` is the
  **latest** signal value for that key over the whole stream (by ``order``), so
  contradictions resolve to the final asserted value.
* **Determinism (FR1.4).** No randomness: keys are selected and ordered by a
  stable sort, the per-task cap takes the first ``probes_per_task`` keys by
  sorted key, and ``probe_id`` is assigned over a stable global order. The same
  stream therefore always yields a byte-identical probe set.
* **Trivial oracle.** :class:`OracleAgent` memorizes every fact it *sees* (it is
  not given labels) and answers a probe with the latest value recorded for the
  probe's key. With ``contradiction_rate=0.0`` it produces a lower-triangular
  ones accuracy matrix — the "well-formed skeleton with known answers" check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from slow_wave.stream.schema import (
    AccuracyMatrix,
    CLScenario,
    Label,
    Probe,
    ProbeSet,
    Stream,
    StreamItem,
    offline_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class _KeyInfo:
    """Accumulated facts about one probed ``(subject, attribute)`` key.

    Attributes:
        task_index: Task in which the key's first signal assertion appears.
        first_order: ``order`` of the first signal assertion of the key.
        answer: The latest signal value seen for the key (by ``order``).
        latest_order: ``order`` of the latest signal assertion (book-keeping so
            the latest value wins regardless of item iteration order).
    """

    task_index: int
    first_order: int
    answer: str
    latest_order: int


def build_probe_set(stream: Stream) -> ProbeSet:
    """Build the held-out probe set for ``stream`` (FR1.5, exit #6).

    Derives one probe per probed ``(subject, attribute)`` key — a key asserted by
    at least one ``signal`` item, where signal membership is read offline via
    :func:`~slow_wave.stream.schema.offline_labels`. For each key the canonical
    ``answer`` is the **latest** signal value over the whole stream (by
    ``order``), ``available_after_order`` is the ``order`` of the **first** signal
    assertion of the key, and ``task_index`` is the task in which the key first
    appears.

    Construction is fully deterministic: keys are grouped by their first task,
    each task is capped at ``stream.config.probes_per_task`` keys (the first N by
    sorted key), and ``probe_id`` (``"p000000"`` ...) is assigned over the stable
    ``(task_index, subject, attribute)`` order of the selected keys.
    ``task_id_visible`` is ``True`` only for the task-incremental scenario.

    Args:
        stream: The fully-built stream to derive probes from.

    Returns:
        A :class:`~slow_wave.stream.schema.ProbeSet` carrying the stream's
        scenario, ``n_tasks``, and the ordered list of probes.
    """
    labels = offline_labels(stream)

    # Accumulate per-key info, iterating items in stream order so the last
    # signal assertion seen is the latest value (largest order wins).
    info: dict[tuple[str, str], _KeyInfo] = {}
    for item in sorted(stream.items, key=lambda it: it.order):
        if item.fact is None:
            continue
        if labels.get(item.item_id) is not Label.SIGNAL:
            continue
        key = item.fact.key()
        existing = info.get(key)
        if existing is None:
            info[key] = _KeyInfo(
                task_index=item.task_index,
                first_order=item.order,
                answer=item.fact.value,
                latest_order=item.order,
            )
        else:
            if item.order < existing.first_order:
                existing.first_order = item.order
                existing.task_index = item.task_index
            if item.order >= existing.latest_order:
                existing.latest_order = item.order
                existing.answer = item.fact.value

    # Group keys by their first task, then deterministically cap each task.
    by_task: dict[int, list[tuple[str, str]]] = {}
    for key, meta in info.items():
        by_task.setdefault(meta.task_index, []).append(key)

    cap = stream.config.probes_per_task
    selected: list[tuple[str, str]] = []
    for task_index in sorted(by_task):
        keys_sorted = sorted(by_task[task_index])
        # DX2 (honesty by construction): never drop coverage silently. When a
        # task has more probed keys than the per-task cap, log exactly how many
        # keys were excluded so the bounded coverage is auditable.
        n_dropped = max(0, len(keys_sorted) - cap)
        if n_dropped:
            logger.info(
                "probe cap: task %d has %d probed keys; keeping %d, dropping %d "
                "(stream_id=%s, probes_per_task=%d)",
                task_index,
                len(keys_sorted),
                cap,
                n_dropped,
                stream.stream_id,
                cap,
            )
        selected.extend(keys_sorted[:cap])

    # Stable global order for probe_id assignment.
    selected.sort(key=lambda k: (info[k].task_index, k[0], k[1]))

    task_id_visible = stream.scenario == CLScenario.TASK_INCREMENTAL
    probes: list[Probe] = []
    for idx, key in enumerate(selected):
        subject, attribute = key
        meta = info[key]
        probes.append(
            Probe(
                probe_id=f"p{idx:06d}",
                task_index=meta.task_index,
                subject=subject,
                attribute=attribute,
                answer=meta.answer,
                available_after_order=meta.first_order,
                query=f"What is the {attribute} of {subject}?",
                task_id_visible=task_id_visible,
            )
        )

    return ProbeSet(scenario=stream.scenario, n_tasks=stream.n_tasks, probes=probes)


class OracleAgent:
    """Trivial perfect-memory oracle for the probe-set skeleton check (exit #6).

    The oracle memorizes the value of **every fact-bearing item it sees** — it is
    never handed relevance labels, it simply records the most recent value for
    each ``(subject, attribute)`` key it observes. Because distractor facts live
    in a disjoint namespace from probed keys, memorizing them is harmless. This
    is exactly the "knows the answers if it has seen them" baseline used to prove
    the probe set / accuracy matrix is well formed.
    """

    def __init__(self) -> None:
        """Create an oracle with empty memory."""
        self._memory: dict[tuple[str, str], str] = {}

    def reset(self) -> None:
        """Forget all observed facts (used before each accuracy-matrix replay)."""
        self._memory = {}

    def observe(self, item: StreamItem) -> None:
        """Memorize the fact carried by ``item``, if any.

        Records ``(subject, attribute) -> value`` for fact-bearing items;
        pure-noise items (``fact is None``) are ignored. Later observations
        overwrite earlier ones, so replaying items in stream order leaves the
        latest value for each key.

        Args:
            item: The stream item being observed.
        """
        if item.fact is not None:
            self._memory[item.fact.key()] = item.fact.value

    def answer(self, probe: Probe) -> str:
        """Answer ``probe`` with the latest value recorded for its key.

        Args:
            probe: The probe to answer.

        Returns:
            The latest value observed for the probe's ``(subject, attribute)``
            key, or ``""`` if the key has never been observed.
        """
        return self._memory.get((probe.subject, probe.attribute), "")


def compute_accuracy_matrix(
    stream: Stream,
    probe_set: ProbeSet,
    agent: "OracleAgent | None" = None,
) -> AccuracyMatrix:
    """Compute the continual-learning accuracy matrix ``R[i][j]`` (exit #6).

    For each cutoff task ``i`` in ``0 .. n_tasks-1``: reset ``agent``, replay
    every item with ``task_index <= i`` in ``order`` (calling
    :meth:`OracleAgent.observe`), then for each task ``j`` set ``R[i][j]`` to the
    fraction of task ``j``'s probes the agent answers correctly
    (``agent.answer(probe) == probe.answer``). A task with no probes contributes
    ``0.0``.

    With a ``contradiction_rate=0.0`` stream the oracle yields a lower-triangular
    ones matrix: ``R[i][j] == 1.0`` for ``j <= i`` (the key's signal was seen by
    cutoff ``i``) and ``0.0`` for ``j > i`` (not yet introduced).

    Args:
        stream: The stream whose items are replayed.
        probe_set: The probe set scored against the stream.
        agent: The oracle to use. A fresh :class:`OracleAgent` is created when
            ``None``.

    Returns:
        A validated :class:`~slow_wave.stream.schema.AccuracyMatrix` of shape
        ``n_tasks x n_tasks`` with every entry in ``[0, 1]``.
    """
    if agent is None:
        agent = OracleAgent()

    n_tasks = stream.n_tasks
    items_in_order = sorted(stream.items, key=lambda it: it.order)

    probes_by_task: dict[int, list[Probe]] = {j: [] for j in range(n_tasks)}
    for probe in probe_set.probes:
        if 0 <= probe.task_index < n_tasks:
            probes_by_task[probe.task_index].append(probe)

    matrix: list[list[float]] = []
    for i in range(n_tasks):
        agent.reset()
        for item in items_in_order:
            if item.task_index <= i:
                agent.observe(item)

        row: list[float] = []
        for j in range(n_tasks):
            probes_j = probes_by_task[j]
            if not probes_j:
                row.append(0.0)
                continue
            correct = sum(1 for p in probes_j if agent.answer(p) == p.answer)
            row.append(correct / len(probes_j))
        matrix.append(row)

    return AccuracyMatrix(n_tasks=n_tasks, scenario=stream.scenario, R=matrix)
