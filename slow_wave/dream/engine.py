"""Two-phase dream-cycle orchestrator for the dream engine (Phase 3, WS-ENGINE).

:class:`DreamEngine` is the conductor that composes the five already-landed dream
operators (REPLAY / TRANSFER / DOWNSCALE / CONFLICT / GENERATIVE-AUGMENT) into a
single offline NREM->REM cycle, gates that cycle to scheduled sleep windows, and
records every cycle into a run-level :class:`~slow_wave.dream.schema.DreamTelemetry`.

The cycle is the canonical structure from ``docs/PHASE3_CONTRACT.md``:

* **NREM** — ``REPLAY -> TRANSFER -> DOWNSCALE -> (CONFLICT, optional)``.
* **REM** — ``GENERATIVE-AUGMENT``.

Each operator is an **independent toggle**, so the engine instantiates and runs
for any of the 2^4 on/off combinations of (replay, transfer, downscale, augment)
— including the all-off empty cycle that records ``operators_run == []`` (EC1) —
with ``conflict`` as the optional fifth NREM step (FR4.7).

Two invariants carry the engine:

* **Determinism (DX1).** The only randomness is a single per-cycle
  :class:`numpy.random.Generator` derived via
  ``derive_seed(cfg.seed, f"dream_cycle_{cycle_index}")`` and threaded into
  REPLAY, TRANSFER, and AUGMENT. There is no use of the ``numpy.random`` globals,
  so a fixed ``(cfg, seed, substrate)`` under the mock LLM yields a byte-identical
  :class:`~slow_wave.dream.schema.DreamCycleResult`.
* **Gating (FR4.5/EC3).** Semantic writes only ever happen inside a cycle, and a
  cycle only runs when :meth:`DreamEngine.sleep_hook` says sleep — on the fixed
  ``sleep_every_n_tasks`` schedule, or additionally on accumulated wake "churn"
  in ``adaptive`` sleep-pressure mode (FR4.6).
"""

from __future__ import annotations

import logging

import numpy as np

import slow_wave.llm
from slow_wave.config import Config
from slow_wave.dream.augment import augment
from slow_wave.dream.conflict import resolve_conflicts
from slow_wave.dream.downscale import downscale
from slow_wave.dream.replay import replay
from slow_wave.dream.schema import DreamCycleResult, DreamTelemetry
from slow_wave.dream.transfer import transfer
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.repro.seeding import derive_seed

logger = logging.getLogger(__name__)


class DreamEngine:
    """The two-phase (NREM->REM) dream-cycle orchestrator (FR4, EC1).

    Holds the experiment :class:`~slow_wave.config.Config`, an injectable
    ``llm_complete`` callable, and a run-level
    :class:`~slow_wave.dream.schema.DreamTelemetry` into which every cycle is
    folded. An internal ``_last_cycle_order`` cursor (initialized to ``-1`` so the
    first cycle's candidate pool is the whole episodic store) tracks the stream
    order of the previous cycle, which the engine uses both to select the recent
    candidate pool and to measure wake churn for adaptive sleep pressure.

    Attributes:
        cfg: The experiment configuration (supplies ``dream`` toggles/scheduling,
            ``memory.recency_half_life``, and ``seed``).
        llm_complete: The default completion callable used when the wake agent
            does not pass one through (defaults to :func:`slow_wave.llm.complete`).
        telemetry: The run-level roll-up of every cycle this engine has run.
    """

    def __init__(self, cfg: Config, *, llm_complete=None) -> None:
        """Initialize the engine for a single run.

        Args:
            cfg: The experiment :class:`~slow_wave.config.Config`.
            llm_complete: Optional injectable completion callable with the
                ``complete(cfg, prompt, system=None) -> LLMResult`` signature;
                defaults to :func:`slow_wave.llm.complete`.
        """
        self.cfg = cfg
        self.llm_complete = llm_complete or slow_wave.llm.complete
        self.telemetry = DreamTelemetry()
        #: Stream order of the previous cycle; ``-1`` before the first cycle, so
        #: the first candidate pool is the whole episodic store (and adaptive
        #: churn is measured from the run's start).
        self._last_cycle_order: int = -1

    def sleep_hook(
        self,
        substrate: MemorySubstrate,
        *,
        embedder,
        llm_complete,
        now_order: int,
        task_index: int,
    ) -> DreamCycleResult | None:
        """Run a dream cycle iff gating says sleep; the wake-agent callback (FR4.5).

        Matches the keyword signature the wake agent invokes
        (``sleep_hook(substrate, *, embedder, llm_complete, now_order,
        task_index)``) and is the *only* place semantic writes are allowed
        (gating, EC3). Gating (FR4.5/FR4.6):

        * If ``cfg.dream.enabled`` is ``False`` => never sleep (return ``None``);
          the run stays byte-identical to the Phase 2 no-sleep baseline.
        * ``"fixed"`` mode: sleep iff
          ``(task_index + 1) % cfg.dream.sleep_every_n_tasks == 0``.
        * ``"adaptive"`` mode: the fixed condition **or** — when
          ``cfg.dream.sleep_pressure_churn_threshold > 0`` — accumulated wake
          churn ``now_order - last_cycle_order >= threshold`` (the
          SWA-homeostasis analogue).

        On a sleep it calls :meth:`run_cycle` (preferring the passed
        ``llm_complete``, falling back to the engine's) and returns its result;
        otherwise it returns ``None``.

        Args:
            substrate: The memory substrate consolidated this cycle.
            embedder: The embedder duck-type (``.dim`` / ``.encode``).
            llm_complete: The completion callable the wake agent threads through;
                ``None`` falls back to ``self.llm_complete``.
            now_order: The current stream order (the sleep window).
            task_index: The 0-based task segment this hook fires after.

        Returns:
            The :class:`~slow_wave.dream.schema.DreamCycleResult` if a cycle ran,
            else ``None``.
        """
        dream_cfg = self.cfg.dream
        if not dream_cfg.enabled:
            return None

        fixed_due = (task_index + 1) % dream_cfg.sleep_every_n_tasks == 0
        should_sleep = fixed_due
        if dream_cfg.sleep_pressure_mode == "adaptive":
            churn = now_order - self._last_cycle_order
            if (
                dream_cfg.sleep_pressure_churn_threshold > 0
                and churn >= dream_cfg.sleep_pressure_churn_threshold
            ):
                should_sleep = True

        if not should_sleep:
            return None

        return self.run_cycle(
            substrate,
            embedder=embedder,
            llm_complete=(llm_complete or self.llm_complete),
            now_order=now_order,
            task_index=task_index,
        )

    def run_cycle(
        self,
        substrate: MemorySubstrate,
        *,
        embedder,
        llm_complete,
        now_order: int,
        task_index: int,
    ) -> DreamCycleResult:
        """Run one two-phase dream cycle and record it into the telemetry (FR4).

        The cycle (each step gated by its operator toggle, EC1):

        * **Candidate pool** — episodic entries with
          ``created_order > last_cycle_order`` (the whole episodic store on the
          first cycle).
        * **NREM**
            * **REPLAY** samples the candidate pool; the *replayed set* is the
              sampled entries resolved via ``episodic.get`` (``None`` misses
              dropped). Empty when REPLAY is off.
            * **TRANSFER** consolidates the *transfer source* = the replayed set
              if REPLAY is on, else the whole candidate pool.
            * **DOWNSCALE** decays all live salience and re-potentiates the
              replayed ids (pure decay when REPLAY is off).
            * **CONFLICT** (optional) demotes same-key contradictions.
        * **REM**
            * **GENERATIVE-AUGMENT** synthesizes pseudo-episodes from the transfer
              source (or the candidate pool when the transfer source is empty).

        The single per-cycle ``rng`` (``derive_seed(cfg.seed,
        f"dream_cycle_{cycle_index}")``) is threaded into REPLAY/TRANSFER/AUGMENT
        — the only randomness in the cycle (DX1). ``operators_run`` lists exactly
        the operators that ran, in execution order (``[]`` when all are off), and
        ``api_calls``/``input_tokens``/``output_tokens`` are summed across TRANSFER
        and AUGMENT (the only operators that call the LLM). Finally the
        ``_last_cycle_order`` cursor advances to ``now_order`` and the result is
        folded into :attr:`telemetry`.

        Args:
            substrate: The memory substrate to consolidate (mutated in place).
            embedder: The embedder duck-type (``.dim`` / ``.encode``).
            llm_complete: The completion callable used by TRANSFER and AUGMENT.
            now_order: The current stream order (the sleep window).
            task_index: The 0-based task segment this cycle ran after.

        Returns:
            The assembled :class:`~slow_wave.dream.schema.DreamCycleResult`.
        """
        cfg = self.cfg
        dream_cfg = cfg.dream

        cycle_index = self.telemetry.n_cycles
        rng = np.random.default_rng(derive_seed(cfg.seed, f"dream_cycle_{cycle_index}"))

        candidates = [
            entry
            for entry in substrate.episodic.all_entries()
            if entry.created_order > self._last_cycle_order
        ]

        operators_run: list[str] = []
        replay_res = None
        transfer_res = None
        downscale_res = None
        conflict_res = None
        augment_res = None

        # ---- NREM -------------------------------------------------------- #
        if dream_cfg.replay_enabled:
            replay_res = replay(
                candidates,
                dream_cfg=dream_cfg,
                rng=rng,
                now_order=now_order,
                recency_half_life=cfg.memory.recency_half_life,
            )
            operators_run.append("replay")
            replayed = [
                substrate.episodic.get(entry_id)
                for entry_id in replay_res.sampled_ids()
            ]
            replayed = [entry for entry in replayed if entry is not None]
        else:
            replayed = []

        # The transfer source is the replayed set when REPLAY is on, otherwise the
        # whole recent candidate pool (canonical cycle structure).
        transfer_source = replayed if dream_cfg.replay_enabled else candidates

        if dream_cfg.transfer_enabled:
            transfer_res = transfer(
                substrate,
                transfer_source,
                cfg=cfg,
                dream_cfg=dream_cfg,
                embedder=embedder,
                llm_complete=llm_complete,
                rng=rng,
                now_order=now_order,
            )
            operators_run.append("transfer")

        if dream_cfg.downscale_enabled:
            downscale_res = downscale(
                substrate,
                dream_cfg=dream_cfg,
                replayed_ids={entry.entry_id for entry in replayed},
                now_order=now_order,
            )
            operators_run.append("downscale")

        if dream_cfg.conflict_enabled:
            conflict_res = resolve_conflicts(
                substrate,
                dream_cfg=dream_cfg,
                now_order=now_order,
            )
            operators_run.append("conflict")

        # ---- REM --------------------------------------------------------- #
        if dream_cfg.augment_enabled:
            augment_source = transfer_source if transfer_source else candidates
            augment_res = augment(
                substrate,
                augment_source,
                cfg=cfg,
                dream_cfg=dream_cfg,
                embedder=embedder,
                llm_complete=llm_complete,
                rng=rng,
                now_order=now_order,
            )
            operators_run.append("augment")

        # The LLM is only called by TRANSFER and AUGMENT; sum their cost.
        api_calls = 0
        input_tokens = 0
        output_tokens = 0
        for op_res in (transfer_res, augment_res):
            if op_res is not None:
                api_calls += op_res.api_calls
                input_tokens += op_res.input_tokens
                output_tokens += op_res.output_tokens

        result = DreamCycleResult(
            cycle_index=cycle_index,
            at_order=now_order,
            task_index=task_index,
            operators_run=operators_run,
            replay=replay_res,
            transfer=transfer_res,
            downscale=downscale_res,
            augment=augment_res,
            conflict=conflict_res,
            api_calls=api_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.info(
            "dream cycle %d at order %d (task %d): ran %s (%d call(s))",
            cycle_index,
            now_order,
            task_index,
            operators_run or "[]",
            api_calls,
        )

        self._last_cycle_order = now_order
        self.telemetry.record(result)
        return result
