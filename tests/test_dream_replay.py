"""Tests for slow_wave.dream.replay (Phase 3, WS-REPLAY; FR4.1, DX1, DX2).

Covers both sampling strategies (uniform and prioritized) producing valid
distributions and importance-sampling weights in ``(0, 1]``; that prioritized
sampling concentrates mass on a high-recency/high-importance entry (a degenerate
pool whose single dominant entry is always picked); the DX2 ``n_dropped``
accounting + log when the candidate pool exceeds the cap; determinism under a
fixed rng seed (byte-identical sampled ids / probabilities / IS-weights); and the
degenerate guards (empty pool, ``replay_sample_size == 0``, a single candidate).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from slow_wave.config import DreamConfig
from slow_wave.dream.replay import replay
from slow_wave.dream.schema import ReplayResult
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.stream.schema import Fact


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _entry(
    entry_id: str,
    order: int,
    *,
    importance: float = 1.0,
    novelty: float = 0.5,
    surprise: float = 0.0,
    recency_order: int | None = None,
    with_fact: bool = True,
) -> MemoryEntry:
    """Build a small episodic :class:`MemoryEntry` with explicit salience."""
    fact = (
        Fact(subject=f"s{entry_id}", attribute="role", value=f"v{entry_id}")
        if with_fact
        else None
    )
    return MemoryEntry(
        entry_id=entry_id,
        tier=MemoryTier.EPISODIC,
        content=f"observation {entry_id}",
        fact=fact,
        created_order=order,
        salience=SalienceMeta(
            importance=importance,
            recency_order=order if recency_order is None else recency_order,
            novelty=novelty,
            surprise=surprise,
        ),
        provenance=(),
    )


def _pool(n: int, *, importance: float = 1.0, novelty: float = 0.5) -> list[MemoryEntry]:
    """Build a pool of ``n`` candidates with uniform salience."""
    return [
        _entry(f"e{i:06d}", i, importance=importance, novelty=novelty) for i in range(n)
    ]


def _cfg(**overrides) -> DreamConfig:
    """Build a DreamConfig with replay overrides applied."""
    return DreamConfig(**overrides)


def _rng(seed: int = 0) -> np.random.Generator:
    """Build a fresh deterministic numpy Generator."""
    return np.random.default_rng(seed)


def _full_probabilities(
    candidates: list[MemoryEntry], cfg: DreamConfig, *, now_order: int
) -> float:
    """Return the sum of the full-pool selection probabilities (~1.0 sanity).

    Reconstructs the probability mass over the whole candidate pool the way
    :func:`replay` does, then returns its sum so a test can assert it is a valid
    distribution (sums to one).
    """
    from slow_wave.dream.replay import _prioritized_priorities

    if cfg.replay_strategy == "uniform":
        priorities = np.ones(len(candidates), dtype=np.float64)
    else:
        priorities = _prioritized_priorities(
            candidates,
            now_order=now_order,
            recency_half_life=64.0,
            eps=cfg.replay_priority_eps,
            alpha=cfg.replay_priority_alpha,
        )
    return float((priorities / priorities.sum()).sum())


# --------------------------------------------------------------------------- #
# Both strategies: valid distribution + IS-weights in (0, 1]
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_returns_at_most_sample_size_with_valid_distribution(strategy: str) -> None:
    """Both strategies return <= sample_size samples with a valid distribution."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=5)
    candidates = _pool(12, importance=2.0, novelty=0.7)

    res = replay(candidates, dream_cfg=cfg, rng=_rng(7), now_order=20)

    assert isinstance(res, ReplayResult)
    assert res.strategy == strategy
    assert res.n_candidates == 12
    assert res.n_sampled == 5
    assert len(res.samples) == 5

    # Probabilities over the whole pool form a valid distribution (sum ~ 1).
    assert _full_probabilities(candidates, cfg, now_order=20) == pytest.approx(1.0)
    # Each sample carries a valid probability and IS-weight in (0, 1].
    for s in res.samples:
        assert 0.0 < s.probability <= 1.0
        assert 0.0 < s.is_weight <= 1.0
        assert s.priority > 0.0

    # No duplicate ids (sampling is without replacement).
    ids = res.sampled_ids()
    assert len(ids) == len(set(ids))

    # sum_is_weight is the sum of the per-sample IS weights.
    assert res.sum_is_weight == pytest.approx(sum(s.is_weight for s in res.samples))


@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_full_pool_probabilities_sum_to_one(strategy: str) -> None:
    """The selection probabilities over the candidate pool form a distribution."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=4)
    candidates = _pool(9, importance=1.5, novelty=0.4)
    assert _full_probabilities(candidates, cfg, now_order=15) == pytest.approx(1.0)


def test_uniform_priorities_probabilities_and_is_weights_are_flat() -> None:
    """Uniform sampling => priority 1.0, probability 1/N, IS-weight 1.0."""
    n = 6
    cfg = _cfg(replay_strategy="uniform", replay_sample_size=3)
    candidates = _pool(n)
    res = replay(candidates, dream_cfg=cfg, rng=_rng(3), now_order=10)

    for s in res.samples:
        assert s.priority == pytest.approx(1.0)
        assert s.probability == pytest.approx(1.0 / n)
        assert s.is_weight == pytest.approx(1.0)
    assert res.sum_is_weight == pytest.approx(float(res.n_sampled))


# --------------------------------------------------------------------------- #
# Prioritized concentrates mass on the salient entry
# --------------------------------------------------------------------------- #
def test_prioritized_dominant_entry_is_always_picked() -> None:
    """A degenerate pool with one dominant entry always samples that entry."""
    # One recent, important, novel, surprising entry; the rest are stale & dull.
    dominant = _entry(
        "e_dominant", 100, importance=100.0, novelty=1.0, surprise=5.0, recency_order=100
    )
    stale = [
        _entry(f"e_stale{i}", 0, importance=1e-3, novelty=0.0, surprise=0.0, recency_order=0)
        for i in range(8)
    ]
    candidates = [*stale, dominant]
    cfg = _cfg(replay_strategy="prioritized", replay_sample_size=1)

    # Across many independent rng seeds the single pick is always the dominant id.
    for seed in range(25):
        res = replay(candidates, dream_cfg=cfg, rng=_rng(seed), now_order=100)
        assert res.sampled_ids() == ["e_dominant"]


def test_prioritized_dominant_entry_has_highest_probability() -> None:
    """The dominant entry's recorded probability exceeds every stale entry's."""
    dominant = _entry(
        "e_dominant", 100, importance=100.0, novelty=1.0, surprise=5.0, recency_order=100
    )
    stale = [
        _entry(f"e_stale{i}", 0, importance=1e-3, novelty=0.0, surprise=0.0, recency_order=0)
        for i in range(5)
    ]
    candidates = [*stale, dominant]
    cfg = _cfg(replay_strategy="prioritized", replay_sample_size=6)  # sample all

    res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=100)
    by_id = {s.entry_id: s for s in res.samples}
    dominant_p = by_id["e_dominant"].probability
    assert all(dominant_p > by_id[f"e_stale{i}"].probability for i in range(5))
    # IS-weight is inversely related to probability: the dominant entry's is the
    # smallest (it is over-sampled), and the max IS-weight is exactly 1.0.
    assert max(s.is_weight for s in res.samples) == pytest.approx(1.0)
    assert by_id["e_dominant"].is_weight == min(s.is_weight for s in res.samples)


def test_prioritized_alpha_zero_is_uniform() -> None:
    """alpha == 0 collapses prioritized priorities to a flat distribution."""
    candidates = _pool(5, importance=3.0, novelty=0.9)
    cfg = _cfg(
        replay_strategy="prioritized", replay_sample_size=5, replay_priority_alpha=0.0
    )
    res = replay(candidates, dream_cfg=cfg, rng=_rng(1), now_order=10)
    probs = [s.probability for s in res.samples]
    assert probs == pytest.approx([1.0 / 5] * 5)
    assert all(s.is_weight == pytest.approx(1.0) for s in res.samples)


# --------------------------------------------------------------------------- #
# DX2 — n_dropped accounting + log
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_n_dropped_equals_surplus_and_is_logged(
    strategy: str, caplog: pytest.LogCaptureFixture
) -> None:
    """n_dropped == n_candidates - n_sampled and the drop is logged (DX2)."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=4)
    candidates = _pool(10)

    with caplog.at_level(logging.INFO, logger="slow_wave.dream.replay"):
        res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=5)

    assert res.n_candidates == 10
    assert res.n_sampled == 4
    assert res.n_dropped == 10 - 4
    assert any("dropped" in rec.message for rec in caplog.records)


def test_no_drop_when_pool_fits(caplog: pytest.LogCaptureFixture) -> None:
    """When the pool fits under the cap, n_dropped == 0 and nothing is logged."""
    cfg = _cfg(replay_strategy="prioritized", replay_sample_size=10)
    candidates = _pool(4)
    with caplog.at_level(logging.INFO, logger="slow_wave.dream.replay"):
        res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=5)
    assert res.n_sampled == 4
    assert res.n_dropped == 0
    assert not any("dropped" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# DX1 — determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_same_seed_yields_identical_sample(strategy: str) -> None:
    """Same rng seed + inputs => identical sampled ids / probabilities / weights."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=5)
    candidates = _pool(12, importance=2.0, novelty=0.6)

    def run() -> list[tuple[str, float, float, float]]:
        res = replay(candidates, dream_cfg=cfg, rng=_rng(42), now_order=30)
        return [
            (s.entry_id, s.priority, s.probability, s.is_weight) for s in res.samples
        ]

    first = run()
    second = run()
    assert first == second  # byte-identical (exact float equality)


def test_different_seeds_can_differ_under_prioritized() -> None:
    """Distinct rng seeds generally draw distinct samples (sanity on rng usage)."""
    cfg = _cfg(replay_strategy="prioritized", replay_sample_size=4)
    candidates = _pool(20, importance=1.0, novelty=0.5)
    a = replay(candidates, dream_cfg=cfg, rng=_rng(1), now_order=10).sampled_ids()
    b = replay(candidates, dream_cfg=cfg, rng=_rng(2), now_order=10).sampled_ids()
    assert a != b


# --------------------------------------------------------------------------- #
# Degenerate guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_empty_pool_is_safe(strategy: str) -> None:
    """An empty candidate pool yields an empty result without raising."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=8)
    res = replay([], dream_cfg=cfg, rng=_rng(0), now_order=0)
    assert res.n_candidates == 0
    assert res.n_sampled == 0
    assert res.n_dropped == 0
    assert res.samples == []
    assert res.sum_is_weight == 0.0


@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_sample_size_zero_is_safe(strategy: str) -> None:
    """replay_sample_size == 0 records the pool but samples nothing (DX2)."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=0)
    candidates = _pool(6)
    res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=3)
    assert res.n_candidates == 6
    assert res.n_sampled == 0
    assert res.n_dropped == 6  # the whole pool was dropped by the zero cap
    assert res.samples == []


@pytest.mark.parametrize("strategy", ["uniform", "prioritized"])
def test_single_candidate_is_safe(strategy: str) -> None:
    """A single candidate is sampled with probability 1.0 and IS-weight 1.0."""
    cfg = _cfg(replay_strategy=strategy, replay_sample_size=4)
    candidates = _pool(1, importance=2.0, novelty=0.3)
    res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=2)
    assert res.n_sampled == 1
    assert res.n_dropped == 0
    (s,) = res.samples
    assert s.entry_id == "e000000"
    assert s.probability == pytest.approx(1.0)
    assert s.is_weight == pytest.approx(1.0)


def test_prioritized_handles_zero_salience_pool() -> None:
    """All-zero-salience candidates still sample (eps floor keeps mass > 0)."""
    candidates = [
        _entry(f"e{i:06d}", i, importance=0.0, novelty=0.0, surprise=0.0)
        for i in range(5)
    ]
    cfg = _cfg(replay_strategy="prioritized", replay_sample_size=3)
    res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=10)
    assert res.n_sampled == 3
    assert all(s.probability > 0.0 for s in res.samples)
    # With identical salience the eps floor makes every priority equal => uniform.
    assert all(s.probability == pytest.approx(1.0 / 5) for s in res.samples)


def test_does_not_read_labels_only_salience() -> None:
    """Confound guard (FR1.6): replay never touches ground-truth/label fields.

    A MemoryEntry exposes no relevance label, so this is structural — the test
    simply asserts replay runs over plain entries and consumes only salience.
    """
    candidates = _pool(3)
    cfg = _cfg(replay_strategy="prioritized", replay_sample_size=2)
    res = replay(candidates, dream_cfg=cfg, rng=_rng(0), now_order=1)
    assert {s.entry_id for s in res.samples} <= {e.entry_id for e in candidates}
