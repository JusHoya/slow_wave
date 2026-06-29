"""Tests for slow_wave.dream.downscale (Phase 3, WS-DOWNSCALE, EC2).

DOWNSCALE (FR4.3) is the NREM "decay all, protect signal" step: it multiplies
every live EPISODIC + SEMANTIC entry's salience by a swappable decay factor and
then re-potentiates only the entries REPLAY sampled this cycle.

These tests pin the EC2 exit criterion:

* Two episodic entries with *identical* importance and recency_order, with one
  id placed in ``replayed_ids``, end the pass with the replayed entry's
  importance **strictly greater** than the non-replayed one (same decay applied
  to both, the re-potentiation boost wins) and the replayed entry's
  ``recency_order`` reset to ``now_order`` while the non-replayed one's is
  unchanged.
* Pure decay (empty ``replayed_ids``) strictly decreases every entry's salience
  for ``age > 0`` and leaves recency stamps untouched.
* The pass mutates only salience (no entry is added/removed/demoted) and is a
  deterministic function of ``(substrate state, dream_cfg, now_order)`` (DX1).

Entries are built directly from the schema types with deterministic float32
vectors (no generator / embedder dependency).
"""

from __future__ import annotations

import numpy as np

from slow_wave.config import DreamConfig, MemoryConfig
from slow_wave.dream.downscale import downscale
from slow_wave.dream.schema import DownscaleResult
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _vec(dim: int) -> np.ndarray:
    """Return a deterministic ``(dim,)`` float32 vector of ones."""
    return np.ones(dim, dtype=np.float32)


def _epi(
    entry_id: str,
    subject: str,
    value: str,
    *,
    importance: float,
    recency_order: int,
    created_order: int | None = None,
) -> MemoryEntry:
    """Build a fact-bearing EPISODIC :class:`MemoryEntry`."""
    order = created_order if created_order is not None else recency_order
    return MemoryEntry(
        entry_id=entry_id,
        tier=MemoryTier.EPISODIC,
        content=f"The role of {subject} is {value}.",
        fact=Fact(subject=subject, attribute="role", value=value),
        created_order=order,
        salience=SalienceMeta(importance=importance, recency_order=recency_order),
        provenance=(f"i_{entry_id}",),
    )


# --------------------------------------------------------------------------- #
# EC2 — re-potentiation: replayed salience > non-replayed under same decay
# --------------------------------------------------------------------------- #
def test_replayed_entry_ends_strictly_above_non_replayed() -> None:
    """Two identical entries diverge: the replayed one wins after decay (EC2)."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)

    # Identical importance AND identical recency_order => identical decay factor.
    rep = _epi("e000000", "alice", "engineer", importance=1.0, recency_order=0)
    non = _epi("e000001", "bob", "doctor", importance=1.0, recency_order=0)
    sub.episodic.append(rep, _vec(dim), now_order=0)
    sub.episodic.append(non, _vec(dim), now_order=0)

    dream_cfg = DreamConfig(decay_function="exponential", repotentiate_boost=1.5)
    res = downscale(
        sub, dream_cfg=dream_cfg, replayed_ids={"e000000"}, now_order=10
    )

    rep_after = sub.episodic.get("e000000").salience
    non_after = sub.episodic.get("e000001").salience

    # Same decay applied to both; the boost (>= 1) makes the replayed one win.
    assert rep_after.importance > non_after.importance
    # The replayed entry's recency is reset to now; the other is untouched.
    assert rep_after.recency_order == 10
    assert non_after.recency_order == 0

    # Result bookkeeping is honest.
    assert isinstance(res, DownscaleResult)
    assert res.decay_function == "exponential"
    assert res.n_decayed == 2
    assert res.n_repotentiated == 1


def test_repotentiation_equals_decay_times_boost() -> None:
    """The replayed entry's salience is exactly decay * boost of the original."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    rep = _epi("e000000", "alice", "engineer", importance=2.0, recency_order=0)
    non = _epi("e000001", "bob", "doctor", importance=2.0, recency_order=0)
    sub.episodic.append(rep, _vec(dim), now_order=0)
    sub.episodic.append(non, _vec(dim), now_order=0)

    rate = 0.1
    boost = 1.5
    now = 10
    dream_cfg = DreamConfig(
        decay_function="exponential",
        decay_exponential_rate=rate,
        repotentiate_boost=boost,
    )
    downscale(sub, dream_cfg=dream_cfg, replayed_ids={"e000000"}, now_order=now)

    factor = float(np.exp(-rate * now))
    rep_after = sub.episodic.get("e000000").salience.importance
    non_after = sub.episodic.get("e000001").salience.importance

    assert rep_after == 2.0 * factor * boost
    assert non_after == 2.0 * factor
    # The replayed entry strictly exceeds the non-replayed one by the boost.
    assert rep_after == non_after * boost


# --------------------------------------------------------------------------- #
# EC2 — pure decay (empty replayed_ids)
# --------------------------------------------------------------------------- #
def test_pure_decay_strictly_decreases_all_salience() -> None:
    """With no replayed ids, every aged entry's salience strictly decreases."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    befores = {}
    for i in range(3):
        e = _epi(
            f"e{i:06d}", f"s{i}", f"v{i}", importance=1.0 + i, recency_order=0
        )
        sub.episodic.append(e, _vec(dim), now_order=0)
        befores[e.entry_id] = e.salience.importance

    res = downscale(sub, dream_cfg=DreamConfig(), replayed_ids=set(), now_order=5)

    assert res.n_repotentiated == 0
    assert res.n_decayed == 3
    for entry in sub.episodic.all_entries():
        # age = 5 > 0 => factor < 1 => strict decrease.
        assert entry.salience.importance < befores[entry.entry_id]
        assert entry.salience.recency_order == 0  # untouched (not replayed)
    # Means reflect the strict drop.
    assert res.mean_salience_after < res.mean_salience_before


def test_age_zero_is_a_no_op_for_unreplayed_entries() -> None:
    """An entry with age 0 (recency == now) is not decayed (factor == 1.0)."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    e = _epi("e000000", "alice", "engineer", importance=1.0, recency_order=7)
    sub.episodic.append(e, _vec(dim), now_order=7)

    downscale(sub, dream_cfg=DreamConfig(), replayed_ids=set(), now_order=7)

    assert sub.episodic.get("e000000").salience.importance == 1.0


# --------------------------------------------------------------------------- #
# Semantic tier is decayed too; empty substrate is safe
# --------------------------------------------------------------------------- #
def test_semantic_entries_are_decayed_and_can_be_repotentiated() -> None:
    """DOWNSCALE walks the SEMANTIC tier as well as the EPISODIC tier."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    sem = MemoryEntry(
        entry_id="s000000",
        tier=MemoryTier.SEMANTIC,
        content="alice is an engineer",
        fact=Fact(subject="alice", attribute="role", value="engineer"),
        created_order=0,
        salience=SalienceMeta(importance=1.0, recency_order=0),
        provenance=("e000000",),
    )
    sub.semantic.upsert(sem, _vec(dim), now_order=0)

    res = downscale(
        sub,
        dream_cfg=DreamConfig(decay_function="exponential"),
        replayed_ids={"s000000"},
        now_order=4,
    )

    assert res.n_decayed == 1
    assert res.n_repotentiated == 1
    assert sub.semantic.get("s000000").salience.recency_order == 4


def test_empty_substrate_reports_zero_means() -> None:
    """No entries => zero counts and 0.0 means (no division by zero)."""
    sub = MemorySubstrate(MemoryConfig(), 4)
    res = downscale(sub, dream_cfg=DreamConfig(), replayed_ids=set(), now_order=3)
    assert res.n_decayed == 0
    assert res.n_repotentiated == 0
    assert res.mean_salience_before == 0.0
    assert res.mean_salience_after == 0.0


# --------------------------------------------------------------------------- #
# Salience-only mutation + determinism (DX1)
# --------------------------------------------------------------------------- #
def test_downscale_only_mutates_salience_no_membership_change() -> None:
    """DOWNSCALE never adds/removes/demotes entries — only salience changes."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    for i in range(3):
        sub.episodic.append(
            _epi(f"e{i:06d}", f"s{i}", f"v{i}", importance=1.0, recency_order=0),
            _vec(dim),
            now_order=0,
        )
    fp_before = sub.footprint().model_dump(mode="json")

    downscale(sub, dream_cfg=DreamConfig(), replayed_ids={"e000000"}, now_order=5)

    # Same membership in every tier (footprint counts/bytes unchanged).
    assert sub.footprint().model_dump(mode="json") == fp_before
    assert {e.entry_id for e in sub.episodic.all_entries()} == {
        "e000000",
        "e000001",
        "e000002",
    }
    assert sub.archival.footprint().n_entries == 0


def test_downscale_is_deterministic() -> None:
    """Identical substrate + cfg + now_order => identical salience (DX1)."""

    def run() -> list[tuple[str, float, int]]:
        dim = 4
        sub = MemorySubstrate(MemoryConfig(), dim)
        for i in range(4):
            sub.episodic.append(
                _epi(
                    f"e{i:06d}", f"s{i}", f"v{i}", importance=1.0 + 0.5 * i,
                    recency_order=i,
                ),
                _vec(dim),
                now_order=i,
            )
        downscale(
            sub,
            dream_cfg=DreamConfig(decay_function="weibull"),
            replayed_ids={"e000001", "e000003"},
            now_order=20,
        )
        return [
            (e.entry_id, e.salience.importance, e.salience.recency_order)
            for e in sub.episodic.all_entries()
        ]

    assert run() == run()
