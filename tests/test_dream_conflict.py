"""Tests for slow_wave.dream.conflict (Phase 3, WS-CONFLICT, FR4.7).

Covers the conflict / unlearning step's exit criteria:

* A same-key/different-value pair is one conflict; one entry is demoted and is
  **recoverable from the archival tier** (demote-not-delete, EC7) while the
  survivor stays active.
* ``conflict_demote_strategy="older"`` keeps the latest-``created_order`` entry
  (matching the wake agent's latest-wins answer); ``"lower_salience"`` keeps the
  highest-importance entry.
* A non-conflicting store (all distinct keys, or a same-key/same-value pair)
  yields all-zero counters and demotes nothing.
* Determinism — identical substrate + cfg yield identical demoted ids.

Entries are built directly from the schema types (no generator / embedder
dependency) and appended with deterministic float32 vectors.
"""

from __future__ import annotations

import numpy as np

from slow_wave.config import DreamConfig, MemoryConfig
from slow_wave.dream.conflict import resolve_conflicts
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact

DIM = 4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _vec(i: int) -> np.ndarray:
    """Return a deterministic ``(DIM,)`` float32 vector."""
    vec = np.ones(DIM, dtype=np.float32)
    vec[i % DIM] = 2.0
    return vec


def _episodic(
    entry_id: str,
    subject: str,
    attribute: str,
    value: str,
    order: int,
    *,
    importance: float = 1.0,
) -> MemoryEntry:
    """Build a fact-bearing EPISODIC :class:`MemoryEntry`."""
    return MemoryEntry(
        entry_id=entry_id,
        tier=MemoryTier.EPISODIC,
        content=f"The {attribute} of {subject} is {value}.",
        fact=Fact(subject=subject, attribute=attribute, value=value),
        created_order=order,
        salience=SalienceMeta(importance=importance, recency_order=order),
        provenance=(f"i{order:06d}",),
    )


def _seed(
    substrate: MemorySubstrate, entries: list[MemoryEntry]
) -> None:
    """Append ``entries`` to the substrate's episodic store with stable vectors."""
    for i, entry in enumerate(entries):
        substrate.observe(entry, _vec(i), now_order=entry.created_order)


def _substrate(entries: list[MemoryEntry], *, capacity: int = 0) -> MemorySubstrate:
    """Build a substrate seeded with ``entries`` in the episodic tier."""
    sub = MemorySubstrate(MemoryConfig(episodic_capacity=capacity), DIM)
    _seed(sub, entries)
    return sub


# --------------------------------------------------------------------------- #
# Core conflict detection + demote-not-delete (EC7)
# --------------------------------------------------------------------------- #
def test_same_key_different_value_demotes_one_recoverable() -> None:
    """Two same-key/different-value entries => 1 conflict, 1 demoted, recoverable."""
    sub = _substrate(
        [
            _episodic("e000000", "alice", "role", "intern", 0),
            _episodic("e000001", "alice", "role", "engineer", 1),
        ]
    )

    res = resolve_conflicts(sub, dream_cfg=DreamConfig(), now_order=99)

    assert res.n_conflicts_detected == 1
    assert res.n_demoted == 1
    assert len(res.demoted_entry_ids) == 1

    demoted_id = res.demoted_entry_ids[0]
    survivor_id = "e000001" if demoted_id == "e000000" else "e000000"

    # Demoted entry is GONE from the active episodic store ...
    assert sub.episodic.get(demoted_id) is None
    # ... but recoverable from archival (demote-not-delete, EC7).
    recovered = sub.archival.recover(demoted_id)
    assert recovered is not None
    assert sub.archival.contains(demoted_id)
    assert recovered.tier == MemoryTier.ARCHIVAL
    assert sub.archival.reason_for(demoted_id) == ("conflict_unlearning", 99)

    # Survivor stays active.
    assert sub.episodic.get(survivor_id) is not None


def test_older_strategy_keeps_latest_created_order() -> None:
    """With "older", the survivor is the latest-created_order entry (latest-wins)."""
    sub = _substrate(
        [
            _episodic("e000000", "alice", "role", "intern", 0),
            _episodic("e000001", "alice", "role", "engineer", 1),
            _episodic("e000002", "alice", "role", "manager", 2),
        ]
    )

    res = resolve_conflicts(
        sub, dream_cfg=DreamConfig(conflict_demote_strategy="older"), now_order=10
    )

    assert res.n_conflicts_detected == 1
    assert res.n_demoted == 2
    # The latest-created_order entry survives; the two older values are demoted.
    assert sub.episodic.get("e000002") is not None
    assert set(res.demoted_entry_ids) == {"e000000", "e000001"}
    for demoted_id in res.demoted_entry_ids:
        assert sub.episodic.get(demoted_id) is None
        assert sub.archival.recover(demoted_id) is not None


def test_lower_salience_strategy_keeps_highest_importance() -> None:
    """With "lower_salience", the highest-importance entry survives."""
    sub = _substrate(
        [
            # Latest order but LOW importance -> would survive under "older",
            # but must be demoted under "lower_salience".
            _episodic("e000000", "alice", "role", "intern", 0, importance=5.0),
            _episodic("e000001", "alice", "role", "engineer", 1, importance=0.5),
        ]
    )

    res = resolve_conflicts(
        sub,
        dream_cfg=DreamConfig(conflict_demote_strategy="lower_salience"),
        now_order=7,
    )

    assert res.n_conflicts_detected == 1
    assert res.n_demoted == 1
    # Highest-importance entry (the older one) survives.
    assert sub.episodic.get("e000000") is not None
    assert res.demoted_entry_ids == ["e000001"]
    assert sub.archival.recover("e000001") is not None


# --------------------------------------------------------------------------- #
# Non-conflicting stores are a no-op (zeros)
# --------------------------------------------------------------------------- #
def test_distinct_keys_no_conflict() -> None:
    """All-distinct keys => zero conflicts, nothing demoted."""
    sub = _substrate(
        [
            _episodic("e000000", "alice", "role", "intern", 0),
            _episodic("e000001", "bob", "role", "doctor", 1),
            _episodic("e000002", "carol", "city", "paris", 2),
        ]
    )

    res = resolve_conflicts(sub, dream_cfg=DreamConfig(), now_order=5)

    assert res.n_conflicts_detected == 0
    assert res.n_demoted == 0
    assert res.demoted_entry_ids == []
    assert sub.archival.footprint().n_entries == 0
    assert len(sub.episodic) == 3


def test_same_key_same_value_is_not_a_conflict() -> None:
    """A same-key pair asserting the SAME value is not a conflict (zeros)."""
    sub = _substrate(
        [
            _episodic("e000000", "alice", "role", "engineer", 0),
            _episodic("e000001", "alice", "role", "engineer", 1),
        ]
    )

    res = resolve_conflicts(sub, dream_cfg=DreamConfig(), now_order=5)

    assert res.n_conflicts_detected == 0
    assert res.n_demoted == 0
    assert res.demoted_entry_ids == []
    # Both stay active; nothing archived.
    assert sub.episodic.get("e000000") is not None
    assert sub.episodic.get("e000001") is not None
    assert sub.archival.footprint().n_entries == 0


def test_empty_substrate_is_noop() -> None:
    """An empty substrate yields all-zero counters without raising."""
    sub = MemorySubstrate(MemoryConfig(), DIM)
    res = resolve_conflicts(sub, dream_cfg=DreamConfig(), now_order=0)
    assert res.n_conflicts_detected == 0
    assert res.n_demoted == 0
    assert res.demoted_entry_ids == []


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_resolve_conflicts_is_deterministic() -> None:
    """Identical substrate + cfg yield identical demoted ids and counts."""

    def run() -> dict:
        sub = _substrate(
            [
                _episodic("e000000", "alice", "role", "intern", 0),
                _episodic("e000001", "alice", "role", "engineer", 1),
                _episodic("e000002", "bob", "city", "paris", 2),
                _episodic("e000003", "bob", "city", "london", 3),
                _episodic("e000004", "carol", "role", "artist", 4),
            ]
        )
        res = resolve_conflicts(sub, dream_cfg=DreamConfig(), now_order=20)
        return res.model_dump(mode="json")

    a = run()
    b = run()
    assert a == b
    # Two conflicting keys (alice/role, bob/city), one demote each.
    assert a["n_conflicts_detected"] == 2
    assert a["n_demoted"] == 2
    assert a["demoted_entry_ids"] == ["e000000", "e000002"]
