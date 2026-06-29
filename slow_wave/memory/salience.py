"""Salience scoring primitives for the memory substrate (Phase 2, WS-MEM).

Pure, deterministic functions that turn raw bookkeeping (stream order, embedding
similarity, importance weights) into the recency / novelty / retrieval /
eviction scalars used by the stores and the retrieval policy. They hold no
state, perform no I/O, and use no randomness, so two calls with identical inputs
return identical outputs (DX1).

The retrieval score follows the Park et al. (2023) "generative agents"
memory-stream policy: a weighted sum of recency, importance, and relevance. The
eviction score is its query-free counterpart (recency + importance only), used
to decide which episodic entry to forget when the buffer is full.
"""

from __future__ import annotations

import numpy as np


def recency_factor(now_order: int, last_order: int, half_life: float) -> float:
    """Return the exponential recency decay of an entry, in ``(0, 1]``.

    Computes ``0.5 ** (max(0, now_order - last_order) / half_life)``: an entry
    accessed at ``now_order`` scores ``1.0`` and decays by half every
    ``half_life`` stream-item units. ``last_order`` in the future (greater than
    ``now_order``) is clamped to a zero gap, so the factor never exceeds ``1.0``.

    Args:
        now_order: The current stream order.
        last_order: Stream order of the entry's last observation/access.
        half_life: Half-life of the decay, in stream-item units (``> 0``).

    Returns:
        The recency factor in ``(0, 1]``.
    """
    gap = max(0, now_order - last_order)
    return float(0.5 ** (gap / half_life))


def novelty_score(vector: np.ndarray, reference: np.ndarray | None) -> float:
    """Return how novel ``vector`` is versus a consolidated reference matrix.

    Defined as ``1 - max cosine similarity`` of ``vector`` to any row of
    ``reference`` (the consolidated/semantic matrix), clipped to ``[0, 1]``.
    Returns ``1.0`` when ``reference`` is ``None`` or empty — everything is novel
    versus an empty store — and also when ``vector`` has zero norm (no signal to
    compare). Rows of ``reference`` with zero norm contribute zero similarity.

    Args:
        vector: A ``(dim,)`` (or ``(1, dim)``) embedding to score.
        reference: A ``(n, dim)`` matrix of consolidated embeddings, or ``None``.

    Returns:
        The novelty score in ``[0, 1]``.
    """
    if reference is None:
        return 1.0
    ref = np.asarray(reference, dtype=np.float64)
    if ref.size == 0:
        return 1.0
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)

    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm == 0.0:
        return 1.0

    ref_norms = np.linalg.norm(ref, axis=1)
    denom = ref_norms * vec_norm
    dots = ref @ vec
    cos = np.where(denom > 0.0, dots / np.where(denom > 0.0, denom, 1.0), 0.0)
    max_cos = float(np.max(cos)) if cos.size else 0.0
    return float(np.clip(1.0 - max_cos, 0.0, 1.0))


def retrieval_score(
    *,
    recency: float,
    importance: float,
    relevance: float,
    w_recency: float,
    w_importance: float,
    w_relevance: float,
) -> float:
    """Return the weighted-sum memory-stream retrieval score (Park et al. 2023).

    Computes ``w_recency*recency + w_importance*importance +
    w_relevance*relevance`` (higher is better). The function is purely
    arithmetic, so passing NumPy arrays for ``recency``/``importance``/
    ``relevance`` returns the element-wise array of scores (the retrieval policy
    relies on this to vectorize over a whole store).

    Args:
        recency: Recency factor in ``(0, 1]``.
        importance: Salience importance, ``>= 0``.
        relevance: ``max(0, cosine)`` query relevance in ``[0, 1]``.
        w_recency: Weight on the recency term.
        w_importance: Weight on the importance term.
        w_relevance: Weight on the relevance term.

    Returns:
        The combined retrieval score.
    """
    return w_recency * recency + w_importance * importance + w_relevance * relevance


def eviction_score(
    *,
    recency: float,
    importance: float,
    w_recency: float,
    w_importance: float,
) -> float:
    """Return the query-free priority used to choose eviction victims.

    Computes ``w_recency*recency + w_importance*importance`` (no relevance term,
    since eviction happens without a query). The **lowest**-scoring live entry is
    evicted first, so old, low-importance entries are forgotten before recent or
    important ones.

    Args:
        recency: Recency factor in ``(0, 1]``.
        importance: Salience importance, ``>= 0``.
        w_recency: Weight on the recency term.
        w_importance: Weight on the importance term.

    Returns:
        The eviction priority (lower = evicted sooner).
    """
    return w_recency * recency + w_importance * importance
