"""DOWNSCALE operator for the dream engine (Phase 3, FR4.3, EC2).

DOWNSCALE is the NREM "decay all, protect signal" step of a dream cycle: every
live memory item's salience is multiplied by a swappable decay factor (EC6,
:mod:`slow_wave.dream.decay`), and the small set of items REPLAY re-sampled this
cycle is then *re-potentiated* — boosted above the global decay and given a fresh
recency stamp so it ages from "now" rather than from when it was last seen. This
is the synaptic-homeostasis (SHY) analogue: downscale the field, protect the
replayed signal.

The operator only ever **mutates salience** of entries already present in the
EPISODIC and SEMANTIC stores (it never adds, removes, or demotes entries, and
never touches the archival tier). It uses no randomness, so it is a pure
deterministic function of ``(substrate state, dream_cfg, now_order)`` (DX1), and
it reads only confound-free :class:`~slow_wave.memory.schema.MemoryEntry`
salience — never a relevance label (FR1.6).
"""

from __future__ import annotations

import logging

from slow_wave.config import DreamConfig
from slow_wave.dream.decay import decay_factor, params_for
from slow_wave.dream.schema import DownscaleResult
from slow_wave.memory.stores import MemorySubstrate

logger = logging.getLogger(__name__)


def downscale(
    substrate: MemorySubstrate,
    *,
    dream_cfg: DreamConfig,
    replayed_ids: set[str],
    now_order: int,
) -> DownscaleResult:
    """Apply global salience decay, then re-potentiate replayed items (FR4.3).

    For every live entry in the EPISODIC and SEMANTIC stores the entry's age is
    ``max(0, now_order - e.salience.recency_order)`` and its importance is
    multiplied by the swappable decay factor for that age::

        factor = decay_factor(dream_cfg.decay_function, age,
                              params_for(dream_cfg.decay_function, dream_cfg))
        e.salience.importance *= factor

    An entry whose id is in ``replayed_ids`` is then re-potentiated: its
    importance is additionally multiplied by ``dream_cfg.repotentiate_boost``
    (``>= 1``, so a replayed item ends strictly above an otherwise-identical
    non-replayed one — EC2) and its ``recency_order`` is reset to ``now_order``
    so it ages from this cycle onward. When ``replayed_ids`` is empty this is a
    pure global decay.

    Only salience is mutated: no entry is added, removed, or demoted, and the
    archival tier is untouched. The function uses no randomness, so identical
    ``(substrate state, dream_cfg, now_order)`` inputs yield identical salience
    (DX1), and it reads only confound-free salience fields (FR1.6).

    Args:
        substrate: The memory substrate whose EPISODIC + SEMANTIC entries are
            decayed in place.
        dream_cfg: The dream configuration supplying ``decay_function``, the flat
            decay knobs, and ``repotentiate_boost``.
        replayed_ids: Ids of the entries REPLAY sampled this cycle (the protected
            "signal"); empty for pure decay.
        now_order: The current stream order (the sleep window), used to compute
            each entry's age and to re-stamp re-potentiated entries.

    Returns:
        A :class:`~slow_wave.dream.schema.DownscaleResult` recording the decay
        curve used, the number of entries decayed and re-potentiated, and the
        mean importance over the decayed entries before and after the pass
        (``0.0`` for both when no entry was decayed).
    """
    params = params_for(dream_cfg.decay_function, dream_cfg)

    n_decayed = 0
    n_repotentiated = 0
    sum_before = 0.0
    sum_after = 0.0

    for store in (substrate.episodic, substrate.semantic):
        for entry in store.all_entries():
            salience = entry.salience
            sum_before += salience.importance

            age = max(0, now_order - salience.recency_order)
            factor = decay_factor(dream_cfg.decay_function, age, params)
            salience.importance *= factor

            if entry.entry_id in replayed_ids:
                salience.importance *= dream_cfg.repotentiate_boost
                salience.recency_order = now_order
                n_repotentiated += 1

            sum_after += salience.importance
            n_decayed += 1

    mean_before = sum_before / n_decayed if n_decayed else 0.0
    mean_after = sum_after / n_decayed if n_decayed else 0.0

    logger.info(
        "downscale[%s]: decayed %d entries, re-potentiated %d "
        "(mean salience %.6f -> %.6f) at order %d",
        dream_cfg.decay_function,
        n_decayed,
        n_repotentiated,
        mean_before,
        mean_after,
        now_order,
    )

    return DownscaleResult(
        decay_function=dream_cfg.decay_function,
        n_decayed=n_decayed,
        n_repotentiated=n_repotentiated,
        mean_salience_before=mean_before,
        mean_salience_after=mean_after,
    )
