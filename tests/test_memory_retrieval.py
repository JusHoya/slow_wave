"""Tests for slow_wave.memory.retrieval (Phase 2, WS-MEM; FR2.3, DX6).

Covers the baseline recency × importance × relevance policy (an exact
subject+attribute match outranks a same-attribute/different-subject distractor
and noise), per-tier retrieval disjointness (EC2), the policy registry
round-trip (DX6 reusability), the deterministic ``entry_id`` tie-break, and the
``touch`` recency-bump contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from slow_wave.config import MemoryConfig
from slow_wave.memory.retrieval import (
    REGISTRY,
    RecencyImportanceRelevancePolicy,
    get_policy,
    register_policy,
    retrieve,
)
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _vec(dim: int, vals: list[float]) -> np.ndarray:
    """Return a ``(dim,)`` float32 vector with ``vals`` in its leading slots."""
    vec = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        vec[i] = x
    return vec


def _entry(
    entry_id: str,
    subject: str,
    attribute: str,
    value: str,
    order: int,
    *,
    tier: MemoryTier = MemoryTier.EPISODIC,
    importance: float = 1.0,
) -> MemoryEntry:
    """Build a small fact-bearing :class:`MemoryEntry`."""
    return MemoryEntry(
        entry_id=entry_id,
        tier=tier,
        content=f"The {attribute} of {subject} is {value}.",
        fact=Fact(subject=subject, attribute=attribute, value=value),
        created_order=order,
        salience=SalienceMeta(importance=importance, recency_order=order),
        provenance=(),
    )


# --------------------------------------------------------------------------- #
# Baseline policy ranking
# --------------------------------------------------------------------------- #
def test_baseline_policy_ranks_exact_match_above_distractor_and_noise() -> None:
    """Exact match > same-attribute distractor > noise under the baseline policy."""
    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)

    # All inserted at the same order with equal importance, so relevance decides.
    sub.observe(_entry("e_exact", "alice", "role", "engineer", 0), _vec(dim, [1, 0, 0, 0]), 0)
    # Distractor: partially aligned with the query (cosine 0.5).
    sub.observe(
        _entry("e_distractor", "bob", "role", "doctor", 0),
        _vec(dim, [0.5, 0.8660254, 0, 0]),
        0,
    )
    # Noise: orthogonal to the query (cosine 0).
    sub.observe(_entry("e_noise", "zzz", "misc", "blah", 0), _vec(dim, [0, 0, 0, 1]), 0)

    query = _vec(dim, [1, 0, 0, 0])
    results = retrieve(
        sub, query, top_k=3, now_order=0, mem_cfg=cfg, tiers=(MemoryTier.EPISODIC,)
    )
    ids = [e.entry_id for e, _ in results]
    assert ids[0] == "e_exact"
    assert ids.index("e_exact") < ids.index("e_distractor") < ids.index("e_noise")

    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)  # returned descending


def test_top_k_truncates() -> None:
    """retrieve returns at most top_k pairs, the highest-scoring ones."""
    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)
    sub.observe(_entry("e_exact", "alice", "role", "engineer", 0), _vec(dim, [1, 0, 0, 0]), 0)
    sub.observe(_entry("e_noise", "zzz", "misc", "blah", 0), _vec(dim, [0, 0, 0, 1]), 0)
    results = retrieve(
        sub, _vec(dim, [1, 0, 0, 0]), top_k=1, now_order=0, mem_cfg=cfg,
        tiers=(MemoryTier.EPISODIC,),
    )
    assert len(results) == 1
    assert results[0][0].entry_id == "e_exact"


# --------------------------------------------------------------------------- #
# EC2 — per-tier retrieval is disjoint
# --------------------------------------------------------------------------- #
def test_per_tier_retrieval_is_disjoint() -> None:
    """Retrieving from a single tier never returns another tier's entries (EC2)."""
    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)

    sub.observe(_entry("e0", "alice", "role", "engineer", 0), _vec(dim, [1, 0, 0, 0]), 0)
    sub.semantic.upsert(
        _entry("c0", "carol", "role", "artist", 0, tier=MemoryTier.SEMANTIC),
        _vec(dim, [1, 0, 0, 0]),
        0,
    )

    query = _vec(dim, [1, 0, 0, 0])
    epi = retrieve(sub, query, 10, now_order=0, mem_cfg=cfg, tiers=(MemoryTier.EPISODIC,))
    sem = retrieve(sub, query, 10, now_order=0, mem_cfg=cfg, tiers=(MemoryTier.SEMANTIC,))

    epi_ids = {e.entry_id for e, _ in epi}
    sem_ids = {e.entry_id for e, _ in sem}
    assert epi_ids == {"e0"}
    assert sem_ids == {"c0"}
    assert epi_ids.isdisjoint(sem_ids)

    # The default tiers union both active stores.
    both = retrieve(sub, query, 10, now_order=0, mem_cfg=cfg)
    assert {e.entry_id for e, _ in both} == {"e0", "c0"}


def test_archival_excluded_from_default_tiers() -> None:
    """Demoted entries are never returned by the default-tier retrieve."""
    dim = 4
    cfg = MemoryConfig(episodic_capacity=1)
    sub = MemorySubstrate(cfg, dim)
    sub.observe(_entry("e0", "alice", "role", "engineer", 0), _vec(dim, [1, 0, 0, 0]), 0)
    sub.observe(_entry("e1", "alice", "role", "manager", 1), _vec(dim, [1, 0, 0, 0]), 1)

    # e0 was evicted to archival.
    assert sub.archival.contains("e0")
    default = retrieve(sub, _vec(dim, [1, 0, 0, 0]), 10, now_order=1, mem_cfg=cfg)
    assert "e0" not in {e.entry_id for e, _ in default}
    # ... but an explicit archival audit retrieval can see it.
    audit = retrieve(
        sub, _vec(dim, [1, 0, 0, 0]), 10, now_order=1, mem_cfg=cfg,
        tiers=(MemoryTier.ARCHIVAL,),
    )
    assert "e0" in {e.entry_id for e, _ in audit}


# --------------------------------------------------------------------------- #
# DX6 — registry round-trip
# --------------------------------------------------------------------------- #
def test_default_policy_is_registered() -> None:
    """The baseline policy is auto-registered under its name at import."""
    assert "recency_importance_relevance" in REGISTRY
    assert isinstance(
        get_policy("recency_importance_relevance"), RecencyImportanceRelevancePolicy
    )


def test_register_and_get_custom_policy_round_trip() -> None:
    """A trivial custom policy registers, resolves, and drives retrieve (DX6)."""

    class IndexPolicy:
        """Scores entries by their position in the batch (last = best)."""

        name = "index_test_policy"

        def score(self, query_vec, entries, vectors, now_order, mem_cfg):
            return np.arange(len(entries), dtype=np.float64)

    policy = IndexPolicy()
    register_policy(policy)
    assert get_policy("index_test_policy") is policy

    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)
    sub.observe(_entry("e0", "a", "x", "1", 0), _vec(dim, [1, 0, 0, 0]), 0)
    sub.observe(_entry("e1", "b", "x", "2", 1), _vec(dim, [0, 1, 0, 0]), 1)

    results = retrieve(
        sub, _vec(dim, [1, 0, 0, 0]), 2, now_order=1, mem_cfg=cfg,
        tiers=(MemoryTier.EPISODIC,), policy=policy,
    )
    # e1 has the higher (later) index score, so it ranks first.
    assert results[0][0].entry_id == "e1"


def test_get_policy_unknown_raises_key_error() -> None:
    """get_policy raises KeyError for an unregistered name."""
    with pytest.raises(KeyError):
        get_policy("no_such_policy_xyz")


# --------------------------------------------------------------------------- #
# Determinism / tie-break
# --------------------------------------------------------------------------- #
def test_tie_break_is_stable_on_entry_id() -> None:
    """Entries with identical scores order by entry_id ascending."""
    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)
    # Identical vectors, salience and order => identical scores.
    sub.observe(_entry("e_b", "a", "x", "1", 0), _vec(dim, [1, 0, 0, 0]), 0)
    sub.observe(_entry("e_a", "b", "y", "2", 0), _vec(dim, [1, 0, 0, 0]), 0)

    results = retrieve(
        sub, _vec(dim, [1, 0, 0, 0]), 2, now_order=0, mem_cfg=cfg,
        tiers=(MemoryTier.EPISODIC,),
    )
    assert [e.entry_id for e, _ in results] == ["e_a", "e_b"]


def test_retrieve_is_deterministic() -> None:
    """Identical inputs yield identical retrieval order and scores."""
    dim = 4
    cfg = MemoryConfig()

    def run():
        sub = MemorySubstrate(cfg, dim)
        for i in range(6):
            sub.observe(_entry(f"e{i:06d}", f"s{i}", "role", f"v{i}", i), _vec(dim, [1, 0, 0, 0]), i)
        out = retrieve(
            sub, _vec(dim, [1, 0, 0, 0]), 4, now_order=6, mem_cfg=cfg,
            tiers=(MemoryTier.EPISODIC,),
        )
        return [(e.entry_id, round(s, 9)) for e, s in out]

    assert run() == run()


# --------------------------------------------------------------------------- #
# touch contract
# --------------------------------------------------------------------------- #
def test_touch_true_bumps_recency_and_access() -> None:
    """touch=True updates recency_order and access_count on returned entries."""
    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)
    sub.observe(_entry("e0", "a", "x", "1", 0), _vec(dim, [1, 0, 0, 0]), now_order=0)

    before = sub.episodic.get("e0").salience
    assert before.recency_order == 0
    access_before = before.access_count

    retrieve(
        sub, _vec(dim, [1, 0, 0, 0]), 1, now_order=50, mem_cfg=cfg,
        tiers=(MemoryTier.EPISODIC,), touch=True,
    )
    after = sub.episodic.get("e0").salience
    assert after.recency_order == 50
    assert after.access_count == access_before + 1


def test_touch_false_is_read_only() -> None:
    """touch=False leaves recency_order and access_count untouched."""
    dim = 4
    cfg = MemoryConfig()
    sub = MemorySubstrate(cfg, dim)
    sub.observe(_entry("e0", "a", "x", "1", 0), _vec(dim, [1, 0, 0, 0]), now_order=0)
    access_before = sub.episodic.get("e0").salience.access_count

    retrieve(
        sub, _vec(dim, [1, 0, 0, 0]), 1, now_order=50, mem_cfg=cfg,
        tiers=(MemoryTier.EPISODIC,), touch=False,
    )
    after = sub.episodic.get("e0").salience
    assert after.recency_order == 0
    assert after.access_count == access_before
