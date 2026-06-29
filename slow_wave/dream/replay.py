"""REPLAY operator for the dream engine (Phase 3, FR4.1, EC1).

REPLAY is the opening NREM step of a dream cycle: it re-samples a bounded
"ripple" batch of recent episodic memories for the rest of the cycle to
consolidate, re-potentiate, and augment. Two strategies are supported:

* ``"uniform"`` — the DQN experience-replay baseline (Mnih et al. 2015): every
  candidate is equally likely.
* ``"prioritized"`` — prioritized experience replay (Schaul et al. 2016): a
  candidate's sample mass scales with its salience (recency x importance x
  novelty x surprise), and an importance-sampling weight is recorded per sample
  so a downstream learner can correct the induced bias. The weights are
  *logged*, never silently dropped (DX2).

Sampling is always WITHOUT replacement and the surplus past
``dream_cfg.replay_sample_size`` is recorded as ``n_dropped`` and logged (DX2:
the bounded coverage is honest, never hidden). All randomness flows through the
caller-supplied :class:`numpy.random.Generator`, so a fixed rng state plus fixed
inputs yields byte-identical sampled ids, probabilities, and IS-weights (DX1).
The operator reads only confound-free :class:`~slow_wave.memory.schema.MemoryEntry`
salience — never a relevance label (FR1.6).
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.config import DreamConfig
from slow_wave.dream.schema import ReplayResult, ReplaySample
from slow_wave.memory.salience import recency_factor
from slow_wave.memory.schema import MemoryEntry

logger = logging.getLogger(__name__)


def _prioritized_priorities(
    candidates: list[MemoryEntry],
    *,
    now_order: int,
    recency_half_life: float,
    eps: float,
    alpha: float,
) -> np.ndarray:
    """Return the Schaul et al. (2016) unnormalized priorities for ``candidates``.

    Each entry's priority is::

        ( recency_factor(now_order, recency_order, recency_half_life)
          * max(importance, 0)
          * (novelty + eps)
          * (1.0 + max(surprise, 0)) + eps ) ** alpha

    so a recent, important, novel, surprising entry earns more sample mass while
    the ``+ eps`` floor keeps every candidate strictly above zero (no entry is
    ever starved). ``alpha == 0`` collapses every priority to ``1.0`` (uniform).

    Args:
        candidates: The candidate episodic entries.
        now_order: The current stream order (for the recency term).
        recency_half_life: Half-life of the recency decay, in stream-item units.
        eps: Small positive floor added to novelty and to the product.
        alpha: Prioritization exponent (``0`` => uniform).

    Returns:
        A ``(len(candidates),)`` float64 array of unnormalized priorities.
    """
    priorities = np.empty(len(candidates), dtype=np.float64)
    for i, entry in enumerate(candidates):
        salience = entry.salience
        recency = recency_factor(now_order, salience.recency_order, recency_half_life)
        importance = max(salience.importance, 0.0)
        novelty = salience.novelty + eps
        surprise = 1.0 + max(salience.surprise, 0.0)
        base = recency * importance * novelty * surprise + eps
        priorities[i] = base**alpha
    return priorities


def replay(
    candidates: list[MemoryEntry],
    *,
    dream_cfg: DreamConfig,
    rng: np.random.Generator,
    now_order: int,
    recency_half_life: float = 64.0,
) -> ReplayResult:
    """Sample up to ``dream_cfg.replay_sample_size`` entries from ``candidates``.

    The recent-episodic candidate pool is sampled WITHOUT replacement under the
    configured ``dream_cfg.replay_strategy``:

    * ``"uniform"``: every candidate has priority ``1.0`` and probability
      ``1 / N``; samples are drawn with ``rng.choice(..., replace=False)`` and
      every importance-sampling weight is ``1.0``.
    * ``"prioritized"`` (Schaul et al. 2016): each candidate's priority is
      computed by :func:`_prioritized_priorities`, probability is
      ``priority / sum(priority)`` over the whole pool, and samples are drawn
      with those probabilities. The importance-sampling weight for a sampled
      entry is ``(1 / (N * probability)) ** 1.0`` normalized so the largest is
      ``1.0`` (all in ``(0, 1]``).

    ``n_sampled = min(replay_sample_size, len(candidates))`` and
    ``n_dropped = len(candidates) - n_sampled`` — the candidates the
    ``replay_sample_size`` cap excluded, recorded on the result and ``logger``-ed
    (DX2). An empty pool or ``replay_sample_size == 0`` returns an empty
    :class:`~slow_wave.dream.schema.ReplayResult` (``n_candidates`` set,
    ``n_sampled == 0``). If every priority underflows to zero the operator falls
    back to a uniform distribution so sampling stays well-defined.

    All randomness flows through ``rng``, so identical ``(candidates, dream_cfg,
    rng state, now_order)`` yields byte-identical sampled ids, probabilities, and
    IS-weights (DX1).

    Args:
        candidates: The recent-episodic candidate pool to sample from.
        dream_cfg: The dream configuration supplying ``replay_strategy``,
            ``replay_sample_size``, ``replay_priority_alpha``, and
            ``replay_priority_eps``.
        rng: The :class:`numpy.random.Generator` that owns all randomness for
            this pass (determinism, DX1).
        now_order: The current stream order (the sleep window), used for the
            prioritized recency term.
        recency_half_life: Half-life of the recency decay, in stream-item units
            (defaults to ``64.0``; the engine passes ``memory.recency_half_life``).

    Returns:
        A :class:`~slow_wave.dream.schema.ReplayResult` recording the strategy,
        the pool size, the sampled/dropped counts, and one
        :class:`~slow_wave.dream.schema.ReplaySample` (priority, probability,
        IS-weight) per sampled entry.
    """
    strategy = dream_cfg.replay_strategy
    n_candidates = len(candidates)
    sample_size = dream_cfg.replay_sample_size
    n_sampled = min(sample_size, n_candidates)
    n_dropped = n_candidates - n_sampled

    # Degenerate cases: nothing to sample. Still report the pool + any drop.
    if n_sampled == 0:
        if n_dropped > 0:
            logger.info(
                "replay[%s]: dropped %d of %d candidates "
                "(sample_size=%d) at order %d",
                strategy,
                n_dropped,
                n_candidates,
                sample_size,
                now_order,
            )
        return ReplayResult(
            strategy=strategy,
            n_candidates=n_candidates,
            n_sampled=0,
            n_dropped=n_dropped,
        )

    if strategy == "uniform":
        priorities = np.ones(n_candidates, dtype=np.float64)
    else:  # "prioritized"
        priorities = _prioritized_priorities(
            candidates,
            now_order=now_order,
            recency_half_life=recency_half_life,
            eps=dream_cfg.replay_priority_eps,
            alpha=dream_cfg.replay_priority_alpha,
        )

    # Normalize to a valid distribution over the whole pool. Fall back to uniform
    # if every priority underflowed to zero (e.g. an extreme alpha).
    sum_priority = float(priorities.sum())
    if not np.isfinite(sum_priority) or sum_priority <= 0.0:
        probabilities = np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)
    else:
        probabilities = priorities / sum_priority

    # Sample WITHOUT replacement, all randomness via the passed rng (DX1).
    if strategy == "uniform":
        sampled_idx = rng.choice(n_candidates, size=n_sampled, replace=False)
    else:
        sampled_idx = rng.choice(
            n_candidates, size=n_sampled, replace=False, p=probabilities
        )

    # Importance-sampling weights for the sampled entries, normalized so max == 1.
    sampled_probs = probabilities[sampled_idx]
    raw_is = (1.0 / (n_candidates * sampled_probs)) ** 1.0
    max_is = float(raw_is.max())
    is_weights = raw_is / max_is if max_is > 0.0 else np.ones_like(raw_is)

    samples = [
        ReplaySample(
            entry_id=candidates[int(idx)].entry_id,
            priority=float(priorities[int(idx)]),
            probability=float(probabilities[int(idx)]),
            is_weight=float(is_weights[j]),
        )
        for j, idx in enumerate(sampled_idx)
    ]

    if n_dropped > 0:
        logger.info(
            "replay[%s]: sampled %d, dropped %d of %d candidates "
            "(sample_size=%d) at order %d",
            strategy,
            n_sampled,
            n_dropped,
            n_candidates,
            sample_size,
            now_order,
        )

    return ReplayResult(
        strategy=strategy,
        n_candidates=n_candidates,
        n_sampled=n_sampled,
        n_dropped=n_dropped,
        samples=samples,
        sum_is_weight=float(is_weights.sum()),
    )
