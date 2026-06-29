"""Pluggable retrieval policies and the retrieve() entry point (WS-MEM, FR2.3).

Retrieval is a registry of named policies (DX6 reusability): a policy maps a
query vector plus a batch of ``(entries, vectors)`` to a per-entry score array,
and :func:`retrieve` gathers entries from the requested tiers, scores them, and
returns the top-k descending (with a stable ``entry_id`` tie-break).

The default policy, :class:`RecencyImportanceRelevancePolicy`, is the Park et al.
(2023) memory-stream weighted sum of recency × importance × relevance and is
auto-registered at import. Custom policies register via :func:`register_policy`
and are selected by name through :class:`MemoryConfig.retrieval_policy` or passed
explicitly to :func:`retrieve`.

The archival tier is **never** retrieved by the default ``tiers`` (it is the
forgotten tier); passing ``tiers=(MemoryTier.ARCHIVAL,)`` is permitted for
audit/tests only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from slow_wave.config import MemoryConfig
from slow_wave.memory.salience import recency_factor, retrieval_score
from slow_wave.memory.schema import MemoryEntry, MemoryTier
from slow_wave.memory.stores import MemorySubstrate


@runtime_checkable
class RetrievalPolicy(Protocol):
    """A named scoring policy over a batch of candidate entries.

    Implementations expose a ``name`` and a :meth:`score` that returns a
    ``(len(entries),)`` float array of scores (higher = better). They must be
    pure functions of their inputs so retrieval stays deterministic.
    """

    name: str

    def score(
        self,
        query_vec: np.ndarray,
        entries: list[MemoryEntry],
        vectors: np.ndarray,
        now_order: int,
        mem_cfg: MemoryConfig,
    ) -> np.ndarray:
        """Return a ``(len(entries),)`` float array of scores (higher better)."""
        ...


#: The global registry of retrieval policies, keyed by ``policy.name``.
REGISTRY: dict[str, RetrievalPolicy] = {}


def register_policy(policy: RetrievalPolicy) -> None:
    """Register ``policy`` under ``policy.name`` (overwrites any same-name entry).

    Args:
        policy: The policy to register; must expose ``name`` and ``score``.
    """
    REGISTRY[policy.name] = policy


def get_policy(name: str) -> RetrievalPolicy:
    """Return the registered policy named ``name``.

    Args:
        name: The policy name to look up.

    Returns:
        The registered policy.

    Raises:
        KeyError: If no policy is registered under ``name``.
    """
    return REGISTRY[name]


class RecencyImportanceRelevancePolicy:
    """Baseline recency × importance × relevance policy (Park et al. 2023).

    For each candidate entry the score is the weighted sum
    ``w_recency*recency + w_importance*importance + w_relevance*relevance`` where
    ``recency`` is the exponential decay of the entry's last-access order,
    ``importance`` is its salience weight, and ``relevance`` is
    ``max(0, cosine(query, entry_vector))``. Weights and half-life come from the
    :class:`MemoryConfig`. Scoring is vectorized over the whole batch.
    """

    name = "recency_importance_relevance"

    def score(
        self,
        query_vec: np.ndarray,
        entries: list[MemoryEntry],
        vectors: np.ndarray,
        now_order: int,
        mem_cfg: MemoryConfig,
    ) -> np.ndarray:
        """Score ``entries`` against ``query_vec`` (see class docstring).

        Args:
            query_vec: The ``(dim,)`` query embedding.
            entries: The candidate entries.
            vectors: An ``(n, dim)`` matrix aligned to ``entries``.
            now_order: The current stream order (for recency).
            mem_cfg: Supplies the half-life and the three term weights.

        Returns:
            A ``(len(entries),)`` float64 array of retrieval scores.
        """
        n = len(entries)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        recency = np.array(
            [
                recency_factor(
                    now_order, e.salience.recency_order, mem_cfg.recency_half_life
                )
                for e in entries
            ],
            dtype=np.float64,
        )
        importance = np.array(
            [e.salience.importance for e in entries], dtype=np.float64
        )

        query = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        mat = np.asarray(vectors, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0 or mat.shape[0] == 0:
            relevance = np.zeros(n, dtype=np.float64)
        else:
            row_norms = np.linalg.norm(mat, axis=1)
            denom = row_norms * query_norm
            dots = mat @ query
            cos = np.where(denom > 0.0, dots / np.where(denom > 0.0, denom, 1.0), 0.0)
            relevance = np.maximum(0.0, cos)

        return retrieval_score(
            recency=recency,
            importance=importance,
            relevance=relevance,
            w_recency=mem_cfg.weight_recency,
            w_importance=mem_cfg.weight_importance,
            w_relevance=mem_cfg.weight_relevance,
        )


# The baseline policy is the registry default (selected by MemoryConfig).
register_policy(RecencyImportanceRelevancePolicy())


def _store_for_tier(substrate: MemorySubstrate, tier: MemoryTier):
    """Return the store object backing ``tier`` on ``substrate`` (or ``None``)."""
    if tier is MemoryTier.EPISODIC:
        return substrate.episodic
    if tier is MemoryTier.SEMANTIC:
        return substrate.semantic
    if tier is MemoryTier.ARCHIVAL:
        return substrate.archival
    return None


def retrieve(
    substrate: MemorySubstrate,
    query_vec: np.ndarray,
    top_k: int,
    *,
    now_order: int,
    mem_cfg: MemoryConfig,
    tiers: tuple[MemoryTier, ...] = (MemoryTier.EPISODIC, MemoryTier.SEMANTIC),
    policy: RetrievalPolicy | None = None,
    touch: bool = False,
) -> list[tuple[MemoryEntry, float]]:
    """Retrieve the top-``top_k`` entries from the requested tiers.

    Gathers ``(entries, vectors)`` from each tier in ``tiers`` (pass a single
    tier for per-store retrieval), scores them with ``policy`` (default: the
    policy named ``mem_cfg.retrieval_policy`` via :func:`get_policy`), and sorts
    by score **descending with a stable tie-break on ``entry_id`` (ascending)**,
    so identical scores always order deterministically. If ``touch`` is ``True``,
    each returned entry's owning store is touched at ``now_order`` (wake-time
    retrieval bumps recency); eval-time retrieval passes ``touch=False`` and is
    read-only.

    The default ``tiers`` exclude ``ARCHIVAL`` (the forgotten tier);
    ``tiers=(MemoryTier.ARCHIVAL,)`` is allowed for audit/tests only.

    Args:
        substrate: The memory substrate to read from.
        query_vec: The ``(dim,)`` query embedding.
        top_k: Maximum number of ``(entry, score)`` pairs to return.
        now_order: The current stream order (recency + optional touch).
        mem_cfg: The memory config (policy name + scoring weights).
        tiers: Which tiers to gather candidates from.
        policy: An explicit policy; defaults to ``mem_cfg.retrieval_policy``.
        touch: Whether to bump recency on returned entries.

    Returns:
        Up to ``top_k`` ``(entry, score)`` pairs, highest score first.
    """
    if policy is None:
        policy = get_policy(mem_cfg.retrieval_policy)

    entries: list[MemoryEntry] = []
    mats: list[np.ndarray] = []
    owners: list = []
    for tier in tiers:
        store = _store_for_tier(substrate, tier)
        if store is None:
            continue
        tier_entries, tier_matrix = store.snapshot()
        if tier_entries:
            entries.extend(tier_entries)
            mats.append(tier_matrix)
            owners.extend([store] * len(tier_entries))

    if not entries:
        return []

    vectors = np.vstack(mats)
    scores = policy.score(query_vec, entries, vectors, now_order, mem_cfg)

    order = sorted(
        range(len(entries)),
        key=lambda i: (-float(scores[i]), entries[i].entry_id),
    )
    selected = order[:top_k]

    if touch:
        for i in selected:
            owners[i].touch(entries[i].entry_id, now_order)

    return [(entries[i], float(scores[i])) for i in selected]
