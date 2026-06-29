"""The no-sleep wake agent and its run loop (Phase 2, WS-AGENT, FR3.1-FR3.3).

This is the **catastrophic-forgetting reference**: a frozen-weights agent that
runs a Phase 1 stream end-to-end in a single online pass with *no dream cycle*.
It writes every observation into the EPISODIC tier, optionally makes a
budget-gated per-task Claude reasoning call (whose output is telemetry only —
the reasoning never writes the SEMANTIC store; consolidation is gated to sleep,
FR3.1), and after each task evaluates every held-out probe against current
memory to fill one row of the continual-learning accuracy matrix ``R[i][j]``.

Design principles
-----------------
* **Label-free by construction (FR1.6).** The loop consumes only
  :func:`slow_wave.stream.guard.online_view` — it never reads ``offline_labels``,
  ``stream.ground_truth``, or ``probe.answer`` (the latter is read only by the
  *offline* scoring comparison, exactly as the Phase 1 oracle does).
* **Crisp, deterministic answers.** :meth:`WakeAgent.answer` resolves a probe by
  exact ``(subject, attribute)`` key lookup over the *active* tiers (episodic +
  semantic, never archival), returning the latest-``created_order`` value. So an
  answer is correct iff the fact is still live in active memory — evicted /
  forgotten facts are simply wrong, which keeps ``R[i][j]`` immune to
  embedding-similarity fuzz and byte-identical across runs (DX1).
* **Gating invariant (FR3.1).** The wake loop performs *no* semantic writes; the
  SEMANTIC store is empty after a baseline run (only Phase 3's dream engine ever
  writes it).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from slow_wave.agent.budget import TokenBudgetController
from slow_wave.eval.metrics import ContinualMetrics, compute_continual_metrics
from slow_wave.llm import complete as _default_complete
from slow_wave.memory.retrieval import retrieve
from slow_wave.memory.salience import novelty_score
from slow_wave.memory.schema import (
    MemoryEntry,
    MemoryFootprint,
    MemoryTier,
    SalienceMeta,
)
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.guard import online_view
from slow_wave.stream.schema import AccuracyMatrix, Probe, ProbeSet, Stream

logger = logging.getLogger(__name__)


@dataclass
class WakeTelemetry:
    """Cost / activity telemetry accumulated over a wake run (EC1).

    All counters are deterministic given ``(cfg, stream, probe_set)`` under the
    mock LLM *except* the wall-clock-derived :attr:`step_latencies_s` (and the
    token counts once a real Claude call is made), which are flagged as
    nondeterministic in the manifest.

    Attributes:
        api_calls: Number of LLM calls actually made.
        input_tokens: Cumulative prompt tokens across all LLM calls.
        output_tokens: Cumulative completion tokens across all LLM calls.
        reasoning_calls_made: Per-task reasoning calls that were executed.
        reasoning_calls_skipped: Per-task reasoning calls skipped by the budget.
        retrieval_calls: Wake-time context :func:`retrieve` calls made (one per
            executed per-task reasoning step, grounding that call's prompt).
        n_items_ingested: Stream items written into the episodic tier.
        n_evicted: Episodic entries evicted (demoted to archival) during the run.
        step_latencies_s: Per-item ingestion latencies, in seconds.
    """

    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls_made: int = 0
    reasoning_calls_skipped: int = 0
    retrieval_calls: int = 0
    n_items_ingested: int = 0
    n_evicted: int = 0
    step_latencies_s: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Return the total tokens used (``input_tokens + output_tokens``)."""
        return self.input_tokens + self.output_tokens

    @property
    def p95_latency_s(self) -> float:
        """Return the 95th-percentile per-item latency (nearest-rank), or 0.0.

        Returns:
            The 95th-percentile of :attr:`step_latencies_s` using the nearest-rank
            method, or ``0.0`` when no steps were recorded.
        """
        if not self.step_latencies_s:
            return 0.0
        ordered = sorted(self.step_latencies_s)
        rank = max(1, math.ceil(0.95 * len(ordered)))
        return float(ordered[rank - 1])

    def as_dict(self) -> dict:
        """Return a JSON-safe, key-sorted dict of the telemetry for the manifest.

        Returns:
            A plain ``dict`` (sorted by key) carrying every counter plus the
            derived :attr:`total_tokens` and :attr:`p95_latency_s`.
        """
        payload = {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_calls_made": self.reasoning_calls_made,
            "reasoning_calls_skipped": self.reasoning_calls_skipped,
            "retrieval_calls": self.retrieval_calls,
            "n_items_ingested": self.n_items_ingested,
            "n_evicted": self.n_evicted,
            "p95_latency_s": self.p95_latency_s,
            "step_latencies_s": [float(x) for x in self.step_latencies_s],
        }
        return dict(sorted(payload.items()))


@dataclass
class WakeResult:
    """The complete result of a single :meth:`WakeAgent.run`.

    Attributes:
        accuracy_matrix: The filled continual-learning accuracy matrix ``R[i][j]``.
        metrics: The bundled continual-learning point metrics over the matrix.
        footprint: The per-tier memory footprint after the run.
        telemetry: The accumulated cost/activity telemetry.
        substrate: The memory substrate the run wrote to (for separation /
            gating / provenance assertions in tests).
    """

    accuracy_matrix: AccuracyMatrix
    metrics: ContinualMetrics
    footprint: MemoryFootprint
    telemetry: WakeTelemetry
    substrate: MemorySubstrate


class WakeAgent:
    """The no-sleep wake agent (the catastrophic-forgetting reference, FR3.1-3.3).

    Owns a :class:`~slow_wave.memory.stores.MemorySubstrate` and a
    :class:`~slow_wave.agent.budget.TokenBudgetController`, and runs a Phase 1
    stream in a single online pass: ingest every item into the episodic tier,
    optionally make a budget-gated per-task reasoning call (telemetry only — no
    semantic writes), and after each task evaluate every probe against current
    active memory to fill one row of ``R[i][j]``.
    """

    def __init__(self, cfg, embedder, *, llm_complete=None) -> None:
        """Build the wake agent's memory substrate and budget controller.

        Args:
            cfg: The experiment :class:`~slow_wave.config.Config` (supplies the
                memory, agent, and model sub-configs).
            embedder: An embedder duck-type exposing ``.dim`` and
                ``.encode(list[str]) -> np.ndarray``.
            llm_complete: Optional injectable completion callable with the
                signature of :func:`slow_wave.llm.complete`; defaults to it.
        """
        self.cfg = cfg
        self.embedder = embedder
        self.llm_complete = llm_complete if llm_complete is not None else _default_complete
        self.substrate = MemorySubstrate(cfg.memory, dim=embedder.dim)
        self.budget = TokenBudgetController(cfg.agent.token_budget)
        #: Number of genuine (non-mocked) LLM calls made during the run, so the
        #: runner can record ``mocked`` honestly on its aggregate cost object.
        self.n_real_calls: int = 0

    def run(self, stream: Stream, probe_set: ProbeSet) -> WakeResult:
        """Run the no-sleep wake pass and return its result (FR3.1-FR3.3).

        Single online pass, evaluating after each task:

        1. Take the label-free online view of ``stream``.
        2. Batch-embed item contents and probe queries once each.
        3. For each task ``t`` (ascending): ingest its items into the episodic
           tier (with a per-item wake-time context retrieve), make the optional
           budget-gated reasoning call, then evaluate every probe to fill row
           ``R[t][j]`` (fraction of task ``j``'s probes answered correctly).
        4. Bundle the accuracy matrix, continual metrics, and footprint.

        Args:
            stream: The stream to process (consumed via the label-free online
                view only — labels are never read).
            probe_set: The held-out probe set scored after each task.

        Returns:
            A :class:`WakeResult`. Deterministic given ``(cfg, stream,
            probe_set)`` under the mock LLM.
        """
        cfg = self.cfg
        telemetry = WakeTelemetry()
        n_tasks = stream.n_tasks

        # 1) Label-free online view (FR1.6): never touches labels/ground_truth.
        items = online_view(stream)

        # 2) Batch-embed item contents and probe queries exactly once.
        item_embeddings = self.embedder.encode([item.content for item in items])
        probes = list(probe_set.probes)
        probe_embeddings = self.embedder.encode([p.query for p in probes])

        emb_by_order: dict[int, np.ndarray] = {}
        items_by_task: dict[int, list] = {t: [] for t in range(n_tasks)}
        for idx, item in enumerate(items):
            emb_by_order[item.order] = item_embeddings[idx]
            items_by_task.setdefault(item.task_index, []).append(item)

        probe_idx_by_id = {p.probe_id: i for i, p in enumerate(probes)}
        probes_by_task: dict[int, list[Probe]] = {j: [] for j in range(n_tasks)}
        for probe in probes:
            if 0 <= probe.task_index < n_tasks:
                probes_by_task[probe.task_index].append(probe)

        top_k = cfg.memory.retrieval_top_k
        now_order = 0

        # 3) Online pass, one row of R per task.
        rows: list[list[float]] = []
        for t in range(n_tasks):
            last_embedding: np.ndarray | None = None
            for item in sorted(items_by_task.get(t, []), key=lambda it: it.order):
                step_start = time.perf_counter()
                now_order = item.order
                embedding = emb_by_order[item.order]
                last_embedding = embedding

                if cfg.memory.novelty_enabled:
                    _, semantic_matrix = self.substrate.semantic.snapshot()
                    novelty = novelty_score(embedding, semantic_matrix)
                else:
                    novelty = 0.0

                entry = MemoryEntry(
                    entry_id=f"e{item.order:06d}",
                    tier=MemoryTier.EPISODIC,
                    content=item.content,
                    fact=item.fact,
                    created_order=item.order,
                    salience=SalienceMeta(
                        importance=cfg.memory.base_salience,
                        recency_order=item.order,
                        novelty=novelty,
                    ),
                    provenance=(item.item_id,),
                )

                evicted = self.substrate.observe(entry, embedding, now_order=item.order)
                telemetry.n_items_ingested += 1
                telemetry.n_evicted += len(evicted)

                telemetry.step_latencies_s.append(time.perf_counter() - step_start)

            # Optional, budget-gated per-task reasoning step (FR3.1: retrieve from
            # memory, then call Claude for reasoning). No semantic writes occur —
            # the output is telemetry/realism only; probe answers are read
            # deterministically from memory, not from the LLM.
            if cfg.agent.reasoning_calls == "per_task" and last_embedding is not None:
                est = len(cfg.agent.reasoning_prompt) // 4 + cfg.model.max_tokens
                if not self.budget.can_afford(est):
                    self.budget.skip()
                    telemetry.reasoning_calls_skipped += 1
                else:
                    # Retrieve recent context from active memory to ground the
                    # reasoning call. Read-only (touch=False): retrieving a memory
                    # for context must not, by itself, re-potentiate it — selective
                    # re-potentiation of accessed/replayed memories is a *dream*
                    # operator (FR4.3 DOWNSCALE "decay all, protect signal"), not
                    # baseline wake behavior. Keeping it read-only makes the
                    # baseline a clean catastrophic-forgetting reference whose
                    # eviction reflects observation recency alone.
                    context = retrieve(
                        self.substrate,
                        last_embedding,
                        top_k,
                        now_order=now_order,
                        mem_cfg=cfg.memory,
                        touch=False,
                    )
                    telemetry.retrieval_calls += 1
                    if context:
                        ctx = " | ".join(entry.content for entry, _ in context)
                        prompt = f"{cfg.agent.reasoning_prompt}\n\nRecent memory:\n{ctx}"
                    else:
                        prompt = cfg.agent.reasoning_prompt
                    result = self.llm_complete(cfg, prompt)
                    self.budget.record(result.input_tokens, result.output_tokens)
                    telemetry.api_calls += 1
                    telemetry.input_tokens += result.input_tokens
                    telemetry.output_tokens += result.output_tokens
                    telemetry.reasoning_calls_made += 1
                    if not result.mocked:
                        self.n_real_calls += 1

            # Evaluate every probe against current active memory -> row R[t].
            row: list[float] = []
            for j in range(n_tasks):
                probes_j = probes_by_task[j]
                if not probes_j:
                    row.append(0.0)
                    continue
                correct = 0
                for probe in probes_j:
                    query_vec = probe_embeddings[probe_idx_by_id[probe.probe_id]]
                    if self.answer(probe, query_vec, now_order) == probe.answer:
                        correct += 1
                row.append(correct / len(probes_j))
            rows.append(row)

        # 4) Bundle results.
        matrix = AccuracyMatrix(n_tasks=n_tasks, scenario=stream.scenario, R=rows)
        metrics = compute_continual_metrics(matrix)
        footprint = self.substrate.footprint()
        return WakeResult(
            accuracy_matrix=matrix,
            metrics=metrics,
            footprint=footprint,
            telemetry=telemetry,
            substrate=self.substrate,
        )

    def answer(self, probe: Probe, query_vec: np.ndarray, now_order: int) -> str:
        """Answer ``probe`` from active memory by exact-key lookup (read-only).

        Gathers entries whose ``fact.key()`` equals ``(probe.subject,
        probe.attribute)`` from the *active* tiers (episodic + semantic, never
        archival) and returns the ``fact.value`` of the one with the largest
        ``created_order`` (latest-wins, so contradictions resolve to the final
        value); returns ``""`` if no active entry asserts the key. The lookup is
        read-only (no touch) and never reads ``probe.answer``.

        Args:
            probe: The probe to answer (only ``subject``/``attribute`` are read).
            query_vec: The probe-query embedding; accepted for signature symmetry
                and future similarity-gated answering (unused by the exact-key
                lookup).
            now_order: The current stream order (accepted for symmetry; the
                read-only lookup does not mutate recency).

        Returns:
            The latest active value for the probe's key, or ``""`` if absent.
        """
        key = (probe.subject, probe.attribute)
        candidates = self.substrate.episodic.find_by_key(key)
        candidates += self.substrate.semantic.find_by_key(key)
        if not candidates:
            return ""
        best = max(candidates, key=lambda e: (e.created_order, e.entry_id))
        return best.fact.value if best.fact is not None else ""
