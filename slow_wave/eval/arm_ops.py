"""Custom sleep-window operators for the Phase 4 control arms (WS-ARMS).

This module implements the two *non-dream* sleep-window operators the control
battery needs (see ``docs/PHASE4_CONTRACT.md`` — "WS-ARMS"): a
ground-truth-blind **random prune** negative control and a label-reading
**oracle prune** ceiling. Both are factory functions that return a callable with
the exact :meth:`slow_wave.agent.wake.WakeAgent` sleep-hook signature
``(substrate, *, embedder, llm_complete, now_order, task_index)`` so they drop
straight into the wake loop in place of
:meth:`slow_wave.dream.engine.DreamEngine.sleep_hook`.

Design principles
-----------------
* **Demote, never delete (EC4/EC7).** Both operators retire entries via
  :meth:`slow_wave.memory.stores.MemorySubstrate.demote_entry`, which moves the
  entry to the auditable archival tier (recoverable) rather than hard-deleting
  it. Nothing here ever ``pop``\\ s an entry out of memory permanently.
* **Determinism (DX1).** The random operator's only randomness is a per-window
  :class:`numpy.random.Generator` seeded via
  ``derive_seed(seed, f"random_prune_{cycle_index}")`` — never Python ``hash()``
  or the ``numpy.random`` global — so the demoted-id set is byte-identical across
  two runs with the same seed. The oracle operator iterates entries in insertion
  order and reads a fixed label map, so it is deterministic by construction.
* **Confound guard (FR1.6) is sacred.** :func:`make_random_prune_hook` never
  reads ground-truth labels (it is blind to salience *and* relevance). Only
  :func:`make_oracle_prune_hook` reads labels, and only via the sanctioned
  :func:`slow_wave.stream.schema.offline_labels` accessor — the single permitted
  label use in the whole arm registry (the ``oracle`` arm's
  ``ArmSpec.uses_labels`` is ``True``).
* **Honesty by construction (DX2).** Every scheduled window logs how many of how
  many active entries it demoted; nothing is bounded silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from slow_wave.config import Config
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.repro.seeding import derive_seed
from slow_wave.stream.schema import Label, Stream, offline_labels

logger = logging.getLogger(__name__)


@dataclass
class PruneTelemetry:
    """Accumulated activity of a custom prune operator over a wake run (DX2).

    Mutated in place by the hook returned from :func:`make_random_prune_hook` /
    :func:`make_oracle_prune_hook`. The harness reads these counters to fill the
    arm's ``n_dream_cycles`` (here, prune cycles) and to audit exactly which
    entries were demoted.

    Attributes:
        n_cycles: Number of *scheduled* sleep windows the operator ran in
            (counted whether or not it found anything to demote).
        n_demoted: Total entries demoted to archival across the run.
        demoted_ids: The demoted entry ids, in the order they were demoted
            (deterministic; no duplicates — a demoted entry leaves active memory).
    """

    n_cycles: int = 0
    n_demoted: int = 0
    demoted_ids: list[str] = field(default_factory=list)


def make_random_prune_hook(
    cfg: Config,
    *,
    seed: int,
    telemetry: PruneTelemetry,
    prune_fraction: float = 0.5,
):
    """Build the ground-truth-blind random-prune sleep hook (the negative control).

    Returns a :class:`slow_wave.agent.wake.WakeAgent` sleep hook (signature
    ``(substrate, *, embedder, llm_complete, now_order, task_index)``) that, at
    each *scheduled* window — ``(task_index + 1) % cfg.dream.sleep_every_n_tasks
    == 0`` — demotes ``round(prune_fraction * n_active)`` of the active entries
    (episodic + semantic) chosen **uniformly at random**, blind to salience
    **and** to ground-truth labels. It is the registry's ground-truth-blind
    negative control: pruning with no signal at all.

    Demotion goes through :meth:`MemorySubstrate.demote_entry` (demote, not
    delete), so every pruned entry stays recoverable in the archival tier.
    Randomness is a per-window ``numpy.random.default_rng(derive_seed(seed,
    f"random_prune_{cycle_index}"))`` where ``cycle_index`` is the operator's
    pre-increment :attr:`PruneTelemetry.n_cycles`, so the demoted-id set is
    byte-identical run-to-run for a fixed seed (DX1). The number demoted is
    logged each window (DX2). Labels are never read (FR1.6).

    Args:
        cfg: The arm's effective configuration (only
            ``cfg.dream.sleep_every_n_tasks`` is read, for the schedule gate).
        seed: The master seed the per-window RNG is derived from.
        telemetry: The :class:`PruneTelemetry` mutated in place each window.
        prune_fraction: Fraction of active entries to demote each window
            (default ``0.5``); clamped so it never exceeds the active count.

    Returns:
        A callable with the wake-agent sleep-hook signature; it returns ``None``
        (the wake loop ignores a custom hook's return value).
    """

    def hook(
        substrate: MemorySubstrate,
        *,
        embedder,
        llm_complete,
        now_order: int,
        task_index: int,
    ):
        """Demote a random fraction of active entries at a scheduled window."""
        if (task_index + 1) % cfg.dream.sleep_every_n_tasks != 0:
            return None

        cycle_index = telemetry.n_cycles
        rng = np.random.default_rng(
            derive_seed(seed, f"random_prune_{cycle_index}")
        )

        active_ids = [e.entry_id for e in substrate.episodic.all_entries()]
        active_ids += [e.entry_id for e in substrate.semantic.all_entries()]
        n_active = len(active_ids)
        n_demote = min(n_active, int(round(prune_fraction * n_active)))

        chosen: list[str] = []
        if n_demote > 0:
            picks = rng.choice(n_active, size=n_demote, replace=False)
            # Sort the chosen ids for a deterministic, readable demotion order;
            # the *set* is already fixed by the seeded RNG.
            chosen = sorted(active_ids[int(i)] for i in picks)

        n_done = 0
        for entry_id in chosen:
            if substrate.demote_entry(
                entry_id, reason="random_pruning", at_order=now_order
            ):
                telemetry.demoted_ids.append(entry_id)
                n_done += 1

        telemetry.n_demoted += n_done
        telemetry.n_cycles = cycle_index + 1
        logger.info(
            "random_pruning cycle %d at order %d (task %d): demoted %d of %d "
            "active entries (blind to salience and labels)",
            cycle_index,
            now_order,
            task_index,
            n_done,
            n_active,
        )
        return None

    return hook


def make_oracle_prune_hook(
    cfg: Config,
    stream: Stream,
    *,
    telemetry: PruneTelemetry,
):
    """Build the label-reading oracle-prune sleep hook (the prune-quality ceiling).

    Returns a :class:`slow_wave.agent.wake.WakeAgent` sleep hook that, at each
    *scheduled* window — ``(task_index + 1) % cfg.dream.sleep_every_n_tasks ==
    0`` — demotes **exactly** the active episodic entries whose source stream
    item is ground-truth ``distractor`` or ``noise``, retaining every ``signal``
    item. This is the sanctioned upper bound on prune quality: it is the only
    operator in the registry that reads ground-truth labels, and it reads them
    only through :func:`slow_wave.stream.schema.offline_labels` (the ``oracle``
    arm's ``ArmSpec.uses_labels`` is ``True``).

    For each active episodic entry ``e`` whose ``e.provenance[0]`` is a stream
    ``item_id`` labelled ``distractor``/``noise``, the entry is demoted via
    :meth:`MemorySubstrate.demote_entry` (demote, not delete; recoverable from
    archival). Entries whose ``provenance[0]`` is not a stream item id (e.g. a
    consolidated/pseudo entry) are left alone. Entries are iterated in insertion
    order, so the operator is deterministic (DX1) with no RNG at all. The label
    map is read **once** at hook construction — the single sanctioned label use.

    Args:
        cfg: The arm's effective configuration (only
            ``cfg.dream.sleep_every_n_tasks`` is read, for the schedule gate).
        stream: The stream being run; its ground-truth labels are read once via
            :func:`offline_labels` (the one permitted label read, FR1.6).
        telemetry: The :class:`PruneTelemetry` mutated in place each window.

    Returns:
        A callable with the wake-agent sleep-hook signature; it returns ``None``.
    """
    # The ONE sanctioned ground-truth read in the arm registry (FR1.6): the
    # oracle arm declares uses_labels=True. Read once, here, at construction.
    labels = offline_labels(stream)
    prunable = {Label.DISTRACTOR, Label.NOISE}

    def hook(
        substrate: MemorySubstrate,
        *,
        embedder,
        llm_complete,
        now_order: int,
        task_index: int,
    ):
        """Demote exactly the active distractor/noise episodics at a window."""
        if (task_index + 1) % cfg.dream.sleep_every_n_tasks != 0:
            return None

        cycle_index = telemetry.n_cycles
        # Snapshot active episodics in insertion order (deterministic).
        to_demote: list[str] = []
        n_active = 0
        for entry in substrate.episodic.all_entries():
            n_active += 1
            if not entry.provenance:
                continue
            item_id = entry.provenance[0]
            if labels.get(item_id) in prunable:
                to_demote.append(entry.entry_id)

        n_done = 0
        for entry_id in to_demote:
            if substrate.demote_entry(
                entry_id, reason="oracle_prune", at_order=now_order
            ):
                telemetry.demoted_ids.append(entry_id)
                n_done += 1

        telemetry.n_demoted += n_done
        telemetry.n_cycles = cycle_index + 1
        logger.info(
            "oracle_prune cycle %d at order %d (task %d): demoted %d of %d "
            "active episodics (distractor/noise sources only)",
            cycle_index,
            now_order,
            task_index,
            n_done,
            n_active,
        )
        return None

    return hook
