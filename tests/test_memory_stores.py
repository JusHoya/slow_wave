"""Tests for slow_wave.memory.stores (Phase 2, WS-MEM).

Covers the cross-module exit criteria the substrate is responsible for:

* **EC2** — episodic and semantic are physically separate objects; a write to one
  never appears in the other; ``substrate.footprint()`` reports each tier
  independently.
* **EC3** — provenance: an episodic entry points at its source stream ``item_id``
  and a consolidated entry points at its source episodic ``entry_id``\\ (s).
* **EC4** — eviction *demotes* (does not delete): an entry evicted from the
  episodic buffer is absent there but recoverable from the archival tier with
  content/fact intact.
* Determinism — identical inputs yield an identical eviction set and footprint.

The tests build entries directly from the schema types (no generator / embedder
dependency) and pass in deterministic one-hot vectors.
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.config import MemoryConfig
from slow_wave.memory.schema import (
    MemoryEntry,
    MemoryTier,
    SalienceMeta,
    StoreFootprint,
)
from slow_wave.memory.stores import (
    ArchivalStore,
    EpisodicStore,
    MemorySubstrate,
    SemanticStore,
)
from slow_wave.stream.schema import Fact


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _onehot(dim: int, i: int) -> np.ndarray:
    """Return a deterministic unit one-hot ``(dim,)`` float32 vector."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[i % dim] = 1.0
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
    provenance: tuple[str, ...] = (),
    protected: bool = False,
) -> MemoryEntry:
    """Build a small fact-bearing :class:`MemoryEntry`."""
    return MemoryEntry(
        entry_id=entry_id,
        tier=tier,
        content=f"The {attribute} of {subject} is {value}.",
        fact=Fact(subject=subject, attribute=attribute, value=value),
        created_order=order,
        salience=SalienceMeta(importance=importance, recency_order=order),
        provenance=provenance,
        protected=protected,
    )


# --------------------------------------------------------------------------- #
# EC2 — physical separation of episodic and semantic
# --------------------------------------------------------------------------- #
def test_episodic_and_semantic_are_distinct_objects() -> None:
    """A write to one store never appears in the other (EC2)."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)

    epi = _entry("e000000", "alice", "role", "engineer", 0, provenance=("i000000",))
    sub.observe(epi, _onehot(dim, 0), now_order=0)

    sem = _entry("c000000", "bob", "role", "doctor", 1, tier=MemoryTier.SEMANTIC)
    sub.semantic.upsert(sem, _onehot(dim, 1), now_order=1)

    assert sub.episodic is not sub.semantic
    assert sub.episodic.get("e000000") is not None
    assert sub.episodic.get("c000000") is None
    assert sub.semantic.get("c000000") is not None
    assert sub.semantic.get("e000000") is None
    # find_by_key is also per-store.
    assert sub.episodic.find_by_key(("bob", "role")) == []
    assert sub.semantic.find_by_key(("alice", "role")) == []


def test_footprint_reports_each_tier_independently() -> None:
    """substrate.footprint() reports per-tier counts/bytes and a summed total."""
    dim = 8
    sub = MemorySubstrate(MemoryConfig(), dim)
    sub.observe(_entry("e000000", "alice", "role", "engineer", 0), _onehot(dim, 0), 0)
    sub.observe(_entry("e000001", "carol", "role", "artist", 1), _onehot(dim, 1), 1)
    sub.semantic.upsert(
        _entry("c000000", "bob", "role", "doctor", 2, tier=MemoryTier.SEMANTIC),
        _onehot(dim, 2),
        2,
    )

    fp = sub.footprint()
    assert isinstance(fp.episodic, StoreFootprint)
    assert fp.episodic.tier == MemoryTier.EPISODIC
    assert fp.semantic.tier == MemoryTier.SEMANTIC
    assert fp.archival.tier == MemoryTier.ARCHIVAL

    assert fp.episodic.n_entries == 2
    assert fp.semantic.n_entries == 1
    assert fp.archival.n_entries == 0

    # bytes == n_vectors * dim * 4 (float32), reported independently per tier.
    assert fp.episodic.bytes == 2 * dim * 4
    assert fp.semantic.bytes == 1 * dim * 4
    assert fp.archival.bytes == 0
    assert fp.total_bytes == fp.episodic.bytes + fp.semantic.bytes + fp.archival.bytes


# --------------------------------------------------------------------------- #
# EC3 — provenance traces back to origin
# --------------------------------------------------------------------------- #
def test_episodic_provenance_points_at_source_item() -> None:
    """An episodic entry's provenance carries its source stream item_id (EC3)."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    epi = _entry("e000007", "alice", "role", "engineer", 7, provenance=("i000007",))
    sub.observe(epi, _onehot(dim, 0), now_order=7)

    got = sub.episodic.get("e000007")
    assert got is not None
    assert "i000007" in got.provenance  # trace the entry back to its origin item


def test_semantic_provenance_traces_to_episodic_sources() -> None:
    """A consolidated entry points at the episodic entry_ids it was built from."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(), dim)
    # Two episodic observations of the same key.
    sub.observe(
        _entry("e000000", "alice", "role", "intern", 0, provenance=("i000000",)),
        _onehot(dim, 0),
        0,
    )
    sub.observe(
        _entry("e000001", "alice", "role", "engineer", 1, provenance=("i000001",)),
        _onehot(dim, 1),
        1,
    )
    # A semantic entry consolidated from those episodic sources.
    sem = MemoryEntry(
        entry_id="c000000",
        tier=MemoryTier.SEMANTIC,
        content="alice is an engineer",
        fact=Fact(subject="alice", attribute="role", value="engineer"),
        created_order=1,
        salience=SalienceMeta(importance=1.0, recency_order=1),
        provenance=("e000000", "e000001"),
    )
    sub.semantic.upsert(sem, _onehot(dim, 1), now_order=1)

    consolidated = sub.semantic.find_by_key(("alice", "role"))[0]
    assert set(consolidated.provenance) == {"e000000", "e000001"}
    # Each source entry_id resolves to a real episodic entry with the same key.
    for src_id in consolidated.provenance:
        src = sub.episodic.get(src_id)
        assert src is not None
        assert src.key() == ("alice", "role")


# --------------------------------------------------------------------------- #
# EC4 — eviction demotes, it does not delete
# --------------------------------------------------------------------------- #
def test_eviction_demotes_to_recoverable_archival() -> None:
    """Over-capacity eviction removes from episodic but archives recoverably (EC4)."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(episodic_capacity=2), dim)

    evicted: list[MemoryEntry] = []
    for i in range(4):
        e = _entry(
            f"e{i:06d}", f"subj_{i}", "role", f"val_{i}", i, provenance=(f"i{i:06d}",)
        )
        evicted.extend(sub.observe(e, _onehot(dim, i), now_order=i))

    # capacity 2, four inserts => two evictions; the two oldest are forgotten.
    assert len(sub.episodic) == 2
    assert len(evicted) == 2
    evicted_ids = {e.entry_id for e in evicted}
    assert evicted_ids == {"e000000", "e000001"}

    for victim in evicted:
        # Absent from episodic ...
        assert sub.episodic.get(victim.entry_id) is None
        # ... but recoverable from archival, content/fact intact.
        recovered = sub.archival.recover(victim.entry_id)
        assert recovered is not None
        assert sub.archival.contains(victim.entry_id)
        assert recovered.tier == MemoryTier.ARCHIVAL
        assert recovered.content == victim.content
        assert recovered.fact == victim.fact
        assert recovered.provenance == victim.provenance


def test_unbounded_episodic_never_evicts() -> None:
    """capacity == 0 means unbounded: observe never evicts."""
    dim = 4
    sub = MemorySubstrate(MemoryConfig(episodic_capacity=0), dim)
    for i in range(10):
        evicted = sub.observe(
            _entry(f"e{i:06d}", "alice", "role", f"v{i}", i), _onehot(dim, i), i
        )
        assert evicted == []
    assert len(sub.episodic) == 10
    assert sub.archival.footprint().n_entries == 0


def test_archival_disabled_drops_with_log(caplog) -> None:
    """With archival disabled, evicted entries are dropped and logged (DX2)."""
    dim = 4
    sub = MemorySubstrate(
        MemoryConfig(episodic_capacity=1, archival_enabled=False), dim
    )
    sub.observe(_entry("e000000", "s0", "role", "v0", 0), _onehot(dim, 0), 0)
    with caplog.at_level(logging.INFO, logger="slow_wave.memory.stores"):
        evicted = sub.observe(_entry("e000001", "s1", "role", "v1", 1), _onehot(dim, 1), 1)

    assert len(evicted) == 1
    assert not sub.archival.contains("e000000")  # not recoverable when disabled
    assert any("e000000" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_eviction_and_footprint_are_deterministic() -> None:
    """Identical inputs yield an identical eviction set and footprint."""

    def run() -> tuple[list[str], dict]:
        sub = MemorySubstrate(MemoryConfig(episodic_capacity=3), 4)
        evicted_ids: list[str] = []
        for i in range(8):
            e = _entry(f"e{i:06d}", f"s_{i}", "role", f"v_{i}", i)
            evicted_ids.extend(
                v.entry_id for v in sub.observe(e, _onehot(4, i), now_order=i)
            )
        return evicted_ids, sub.footprint().model_dump(mode="json")

    ids_a, fp_a = run()
    ids_b, fp_b = run()
    assert ids_a == ids_b
    assert fp_a == fp_b
    assert len(ids_a) == 5  # 8 inserts - capacity 3


# --------------------------------------------------------------------------- #
# Store-level guards
# --------------------------------------------------------------------------- #
def test_episodic_append_rejects_wrong_tier() -> None:
    """EpisodicStore.append refuses a non-EPISODIC entry."""
    store = EpisodicStore(dim=4)
    bad = _entry("c0", "a", "x", "1", 0, tier=MemoryTier.SEMANTIC)
    try:
        store.append(bad, _onehot(4, 0), 0)
    except ValueError:
        pass
    else:  # pragma: no cover - explicit failure if no raise
        raise AssertionError("expected ValueError for wrong tier")


def test_semantic_and_archival_smoke() -> None:
    """SemanticStore upsert (new key) and ArchivalStore demote/recover basics."""
    dim = 4
    sem = SemanticStore(dim)
    assert sem.upsert(
        _entry("c0", "a", "x", "1", 0, tier=MemoryTier.SEMANTIC), _onehot(dim, 0), 0
    )
    assert sem.footprint().n_entries == 1

    arc = ArchivalStore(dim)
    entry = _entry("e0", "a", "x", "1", 0)
    arc.demote(entry, _onehot(dim, 0), reason="manual", at_order=5)
    assert arc.contains("e0")
    assert arc.recover("e0").tier == MemoryTier.ARCHIVAL
    assert arc.reason_for("e0") == ("manual", 5)
    # Re-demoting the same id does not raise.
    arc.demote(entry, None, reason="again", at_order=6)
    assert arc.reason_for("e0") == ("again", 6)
