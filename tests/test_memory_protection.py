"""Tests for FR2.5 semantic write-protection (Phase 2, WS-MEM; EC5).

A protected consolidated fact must survive an attempt by a distractor write to
overwrite it with a different value: the conflicting upsert is refused, a
``FailureEvent(kind="protected_overwrite")`` lands in the failure sink (and is
logged), the protected value is preserved, and the upsert returns ``False``. The
positive control confirms an *unprotected* same-key upsert still overwrites
latest-wins and returns ``True``.
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.memory.schema import FailureEvent, MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import SemanticStore
from slow_wave.stream.schema import Fact


def _onehot(dim: int, i: int) -> np.ndarray:
    """Return a deterministic unit one-hot ``(dim,)`` float32 vector."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[i % dim] = 1.0
    return vec


def _sem_entry(
    entry_id: str,
    subject: str,
    attribute: str,
    value: str,
    order: int,
    *,
    protected: bool = False,
    provenance: tuple[str, ...] = (),
) -> MemoryEntry:
    """Build a SEMANTIC :class:`MemoryEntry` for a ``(subject, attribute)`` key."""
    return MemoryEntry(
        entry_id=entry_id,
        tier=MemoryTier.SEMANTIC,
        content=f"The {attribute} of {subject} is {value}.",
        fact=Fact(subject=subject, attribute=attribute, value=value),
        created_order=order,
        salience=SalienceMeta(importance=1.0, recency_order=order),
        provenance=provenance,
        protected=protected,
    )


def test_protected_overwrite_is_blocked_and_logged(caplog) -> None:
    """A distractor overwriting a protected fact is refused and logged (EC5)."""
    dim = 4
    store = SemanticStore(dim)
    sink: list[FailureEvent] = []

    # Establish a protected fact for (alice, role) = engineer.
    ok_first = store.upsert(
        _sem_entry(
            "c_protected", "alice", "role", "engineer", 0,
            protected=True, provenance=("i_signal",),
        ),
        _onehot(dim, 0),
        now_order=0,
        failure_sink=sink,
    )
    assert ok_first is True

    # A distractor tries to overwrite it with a different value.
    with caplog.at_level(logging.WARNING, logger="slow_wave.memory.stores"):
        ok_second = store.upsert(
            _sem_entry(
                "c_distractor", "alice", "role", "janitor", 5,
                provenance=("i_distractor",),
            ),
            _onehot(dim, 1),
            now_order=5,
            failure_sink=sink,
        )

    # The overwrite is refused.
    assert ok_second is False

    # The protected value is preserved (exactly one entry, the original).
    kept = store.find_by_key(("alice", "role"))
    assert len(kept) == 1
    assert kept[0].entry_id == "c_protected"
    assert kept[0].fact.value == "engineer"
    assert kept[0].protected is True

    # A FailureEvent landed in the sink and on the substrate-style sink.
    assert len(sink) == 1
    event = sink[0]
    assert isinstance(event, FailureEvent)
    assert event.kind == "protected_overwrite"
    assert event.key == ("alice", "role")
    assert event.old_value == "engineer"
    assert event.new_value == "janitor"
    assert event.source == "c_distractor"  # the offending write
    assert event.at_order == 5

    # The failure was also logged at WARNING.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert any("protected_overwrite" in r.getMessage() for r in warnings)


def test_unprotected_same_key_overwrites_latest_wins() -> None:
    """Positive control: an unprotected same-key upsert overwrites, returns True."""
    dim = 4
    store = SemanticStore(dim)
    sink: list[FailureEvent] = []

    assert store.upsert(
        _sem_entry("c0", "alice", "role", "engineer", 0, protected=False),
        _onehot(dim, 0),
        now_order=0,
        failure_sink=sink,
    )
    assert store.upsert(
        _sem_entry("c1", "alice", "role", "scientist", 3, protected=False),
        _onehot(dim, 1),
        now_order=3,
        failure_sink=sink,
    )

    kept = store.find_by_key(("alice", "role"))
    assert len(kept) == 1  # upsert by key, not append
    assert kept[0].entry_id == "c1"
    assert kept[0].fact.value == "scientist"  # latest wins
    assert sink == []  # no failure for an unprotected overwrite


def test_protected_same_value_reassertion_is_allowed() -> None:
    """Re-asserting a protected key with the SAME value is not a violation."""
    dim = 4
    store = SemanticStore(dim)
    sink: list[FailureEvent] = []

    assert store.upsert(
        _sem_entry("c0", "alice", "role", "engineer", 0, protected=True),
        _onehot(dim, 0),
        now_order=0,
        failure_sink=sink,
    )
    # Same value, even from a different entry id => allowed (no value change).
    assert store.upsert(
        _sem_entry("c1", "alice", "role", "engineer", 2, protected=True),
        _onehot(dim, 1),
        now_order=2,
        failure_sink=sink,
    )
    kept = store.find_by_key(("alice", "role"))
    assert len(kept) == 1
    assert kept[0].fact.value == "engineer"
    assert sink == []


def test_new_protected_key_is_a_normal_write() -> None:
    """A protected upsert for a brand-new key just writes (no prior to protect)."""
    dim = 4
    store = SemanticStore(dim)
    sink: list[FailureEvent] = []
    assert store.upsert(
        _sem_entry("c0", "dave", "role", "pilot", 0, protected=True),
        _onehot(dim, 0),
        now_order=0,
        failure_sink=sink,
    )
    assert store.find_by_key(("dave", "role"))[0].fact.value == "pilot"
    assert sink == []
