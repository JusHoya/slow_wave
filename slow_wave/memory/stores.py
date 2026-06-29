"""The three physical memory stores and the substrate that ties them (WS-MEM).

This module implements the dual-store (plus archival) memory substrate the
Phase 2 wake agent writes to and reads from (FR2.1-FR2.5):

* :class:`EpisodicStore` — append-only observation buffer with capacity-bounded,
  query-free eviction (the source of the no-sleep baseline's forgetting).
* :class:`SemanticStore` — consolidated knowledge store with FR2.5
  write-protection (a protected fact refuses a conflicting overwrite).
* :class:`ArchivalStore` — the auditable "forgotten" tier evicted entries are
  *demoted* into rather than deleted (EC4).
* :class:`MemorySubstrate` — owns the three stores plus the run's failure-event
  sink and exposes :meth:`MemorySubstrate.observe` (append-then-demote) and a
  per-tier :meth:`MemorySubstrate.footprint`.

**Stores never embed.** Each store keeps entries *and* a parallel float32 vector
index (so a store's footprint is exactly ``n_vectors * dim * 4`` bytes), but the
vectors are computed by the caller (the agent) and passed in — the memory tier
has no dependency on the embedding backend.
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.config import MemoryConfig
from slow_wave.memory.salience import eviction_score, recency_factor
from slow_wave.memory.schema import (
    FailureEvent,
    MemoryEntry,
    MemoryFootprint,
    MemoryTier,
    StoreFootprint,
)

logger = logging.getLogger(__name__)


class _VectorStore:
    """Private base: entries plus a parallel float32 vector index.

    Holds the shared storage and query surface used by the episodic, semantic,
    and archival stores. Entries are kept in a dict keyed by ``entry_id`` (so
    insertion order is preserved and lookups are O(1)); vectors are kept in a
    parallel dict so a store's footprint is exactly ``n_vectors * dim * 4`` bytes
    of float32. Not part of the public contract.
    """

    def __init__(self, dim: int, tier: MemoryTier) -> None:
        """Initialize an empty store.

        Args:
            dim: Embedding dimensionality of the vector index.
            tier: The tier this store reports in its footprint.
        """
        self.dim: int = int(dim)
        self._tier: MemoryTier = tier
        self._entries: dict[str, MemoryEntry] = {}
        self._vectors: dict[str, np.ndarray] = {}

    # -- internal mutation ------------------------------------------------- #
    def _add(self, entry: MemoryEntry, embedding: np.ndarray | None) -> None:
        """Store ``entry`` and (optionally) its ``embedding``.

        Args:
            entry: The entry to store (keyed by ``entry.entry_id``).
            embedding: A ``(dim,)`` vector, or ``None`` to store no vector for
                this entry (archival entries may lack a preserved vector).

        Raises:
            ValueError: If ``embedding`` is provided with the wrong dimension.
        """
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self.dim:
                raise ValueError(
                    f"embedding dim {vec.shape[0]} != store dim {self.dim}"
                )
            self._vectors[entry.entry_id] = vec
        self._entries[entry.entry_id] = entry

    def _remove(self, entry_id: str) -> MemoryEntry | None:
        """Remove and return the entry with ``entry_id`` (and its vector)."""
        self._vectors.pop(entry_id, None)
        return self._entries.pop(entry_id, None)

    def pop(self, entry_id: str) -> tuple[MemoryEntry | None, np.ndarray | None]:
        """Remove ``entry_id`` and return its ``(entry, vector)`` (or ``(None, None)``).

        The public removal primitive the dream engine uses to move an active
        entry out of its store (e.g. conflict/unlearning demotion); the caller is
        responsible for demoting the returned entry to the archival tier (the
        substrate's :meth:`MemorySubstrate.demote_entry` does both).

        Args:
            entry_id: The entry to remove.

        Returns:
            A ``(entry, vector)`` tuple; ``(None, None)`` if the id is absent.
        """
        vec = self._vectors.get(entry_id)
        entry = self._remove(entry_id)
        return entry, vec

    # -- query surface ----------------------------------------------------- #
    def __len__(self) -> int:
        """Return the number of live entries in the store."""
        return len(self._entries)

    def touch(self, entry_id: str, now_order: int) -> None:
        """Record an access of ``entry_id`` at ``now_order``.

        Increments the entry's ``access_count`` and resets its ``recency_order``
        to ``now_order``. A no-op if the entry is not present.

        Args:
            entry_id: The entry to touch.
            now_order: The current stream order.
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        entry.salience.access_count += 1
        entry.salience.recency_order = now_order

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with ``entry_id``, or ``None`` if absent."""
        return self._entries.get(entry_id)

    def vector(self, entry_id: str) -> np.ndarray | None:
        """Return the stored vector for ``entry_id``, or ``None`` if absent."""
        return self._vectors.get(entry_id)

    def all_entries(self) -> list[MemoryEntry]:
        """Return the live entries in insertion order."""
        return list(self._entries.values())

    def find_by_key(self, key: tuple[str, str]) -> list[MemoryEntry]:
        """Return all live entries whose fact key equals ``key``."""
        return [e for e in self._entries.values() if e.key() == key]

    def snapshot(self) -> tuple[list[MemoryEntry], np.ndarray]:
        """Return ``(entries, matrix)`` with rows aligned to ``entries``.

        The matrix is an ``(n, dim)`` float32 array whose row ``i`` is the vector
        for ``entries[i]`` (a zero row if that entry has no stored vector).

        Returns:
            A tuple of the live entries (insertion order) and their aligned
            embedding matrix.
        """
        entries = list(self._entries.values())
        matrix = np.zeros((len(entries), self.dim), dtype=np.float32)
        for i, entry in enumerate(entries):
            vec = self._vectors.get(entry.entry_id)
            if vec is not None:
                matrix[i] = vec
        return entries, matrix

    def footprint(self) -> StoreFootprint:
        """Return this store's exact float32 vector-index footprint."""
        n_vectors = len(self._vectors)
        return StoreFootprint(
            tier=self._tier,
            n_entries=len(self._entries),
            n_vectors=n_vectors,
            dim=self.dim,
            bytes=n_vectors * self.dim * 4,
        )


class EpisodicStore(_VectorStore):
    """Append-only episodic buffer with capacity-bounded eviction (FR2.1).

    Every observation is appended (episodic memory does not deduplicate by key).
    When ``capacity > 0`` and the buffer overflows, the lowest-:func:`eviction_score`
    live entries are evicted and returned to the caller (the substrate demotes
    them to the archival tier). ``capacity == 0`` means unbounded — nothing is
    ever evicted, which is what makes a generously-sized baseline *not* forget.
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 0,
        half_life: float = 64.0,
        w_recency: float = 1.0,
        w_importance: float = 1.0,
    ) -> None:
        """Initialize the episodic store.

        Args:
            dim: Embedding dimensionality of the vector index.
            capacity: Maximum live entries; ``0`` means unbounded.
            half_life: Recency half-life used by the eviction score.
            w_recency: Eviction-score weight on the recency term.
            w_importance: Eviction-score weight on the importance term.
        """
        super().__init__(dim, MemoryTier.EPISODIC)
        self.capacity: int = int(capacity)
        self.half_life: float = float(half_life)
        self.w_recency: float = float(w_recency)
        self.w_importance: float = float(w_importance)
        # Buffer of the vectors of the most-recently-evicted entries, so the
        # substrate can preserve them when demoting to the archival tier.
        self._evicted_vectors: dict[str, np.ndarray | None] = {}

    def append(
        self, entry: MemoryEntry, embedding: np.ndarray, now_order: int
    ) -> list[MemoryEntry]:
        """Append ``entry`` (+ its vector) and evict if over capacity.

        If ``capacity > 0`` and the store now exceeds it, evicts the
        lowest-:func:`~slow_wave.memory.salience.eviction_score` live entries
        (scored at ``now_order``) until the store is back within capacity, and
        returns the evicted entries (callers demote them). Ties in eviction score
        are broken by ``(created_order, entry_id)`` ascending, so eviction is
        deterministic. ``capacity == 0`` is unbounded and returns ``[]``.

        Args:
            entry: The entry to append; ``entry.tier`` must be ``EPISODIC``.
            embedding: The entry's ``(dim,)`` embedding (computed by the caller).
            now_order: The current stream order (used to score recency).

        Returns:
            The list of evicted entries (empty if nothing was evicted).

        Raises:
            ValueError: If ``entry.tier`` is not ``EPISODIC``.
        """
        if entry.tier is not MemoryTier.EPISODIC:
            raise ValueError(
                f"EpisodicStore.append requires tier=EPISODIC, got {entry.tier}"
            )
        self._add(entry, embedding)
        self._evicted_vectors = {}

        evicted: list[MemoryEntry] = []
        if self.capacity > 0 and len(self._entries) > self.capacity:
            n_evict = len(self._entries) - self.capacity
            ranked = sorted(
                self._entries.values(),
                key=lambda e: (
                    eviction_score(
                        recency=recency_factor(
                            now_order, e.salience.recency_order, self.half_life
                        ),
                        importance=e.salience.importance,
                        w_recency=self.w_recency,
                        w_importance=self.w_importance,
                    ),
                    e.created_order,
                    e.entry_id,
                ),
            )
            for victim in ranked[:n_evict]:
                self._evicted_vectors[victim.entry_id] = self.vector(victim.entry_id)
                self._remove(victim.entry_id)
                evicted.append(victim)
        return evicted

    def evicted_vector(self, entry_id: str) -> np.ndarray | None:
        """Return the vector of an entry evicted by the last :meth:`append`.

        The substrate uses this to preserve an evicted entry's embedding when
        demoting it to the archival tier. Only valid for entries evicted by the
        most recent :meth:`append` call.

        Args:
            entry_id: The evicted entry's id.

        Returns:
            The evicted entry's vector, or ``None`` if unknown/unstored.
        """
        return self._evicted_vectors.get(entry_id)


class SemanticStore(_VectorStore):
    """Consolidated knowledge store with FR2.5 write-protection.

    Keyed by ``fact.key()``: an upsert with a new key appends, an upsert with an
    existing *unprotected* key overwrites latest-wins, and an upsert that would
    change the value of an existing *protected* key is **refused** — it logs a
    warning, records a :class:`~slow_wave.memory.schema.FailureEvent`, and leaves
    the protected entry intact (the EWC-spirit guard against a distractor
    clobbering a consolidated fact).

    In the no-sleep baseline this store stays empty — only the Phase 3 dream
    engine writes it (the wake loop's reasoning is gated, FR3.1).
    """

    def __init__(
        self,
        dim: int,
        half_life: float = 64.0,
        w_recency: float = 1.0,
        w_importance: float = 1.0,
    ) -> None:
        """Initialize the semantic store.

        Args:
            dim: Embedding dimensionality of the vector index.
            half_life: Recency half-life (kept for query-surface symmetry).
            w_recency: Recency weight (kept for symmetry; unused by upsert).
            w_importance: Importance weight (kept for symmetry; unused by upsert).
        """
        super().__init__(dim, MemoryTier.SEMANTIC)
        self.half_life: float = float(half_life)
        self.w_recency: float = float(w_recency)
        self.w_importance: float = float(w_importance)

    def upsert(
        self,
        entry: MemoryEntry,
        embedding: np.ndarray,
        now_order: int,
        failure_sink: list[FailureEvent] | None = None,
    ) -> bool:
        """Write a semantic ``entry``, honoring FR2.5 write-protection.

        If an existing entry with the same ``fact.key()`` is ``protected`` and
        the incoming ``fact.value`` differs, this appends a
        ``FailureEvent(kind="protected_overwrite", ...)`` to ``failure_sink``,
        logs a WARNING, does **not** overwrite (the protected entry is
        preserved), and returns ``False``. Otherwise it upserts by key
        (latest-wins for an unprotected same-key entry; a new key appends) and
        returns ``True``. An entry without a fact (no key) is always appended.

        Args:
            entry: The entry to write; ``entry.tier`` must be ``SEMANTIC``.
            embedding: The entry's ``(dim,)`` embedding (computed by the caller).
            now_order: The current stream order (recorded on any failure event).
            failure_sink: Optional list that protection failures are appended to.

        Returns:
            ``True`` if the write was applied, ``False`` if it was refused.

        Raises:
            ValueError: If ``entry.tier`` is not ``SEMANTIC``.
        """
        if entry.tier is not MemoryTier.SEMANTIC:
            raise ValueError(
                f"SemanticStore.upsert requires tier=SEMANTIC, got {entry.tier}"
            )

        key = entry.key()
        if key is not None:
            existing_matches = self.find_by_key(key)
            existing = existing_matches[0] if existing_matches else None
            if existing is not None:
                incoming_value = entry.fact.value if entry.fact is not None else None
                existing_value = (
                    existing.fact.value if existing.fact is not None else None
                )
                if existing.protected and incoming_value != existing_value:
                    detail = (
                        f"refused overwrite of protected key {key}: "
                        f"{existing_value!r} -> {incoming_value!r}"
                    )
                    event = FailureEvent(
                        kind="protected_overwrite",
                        at_order=now_order,
                        key=key,
                        detail=detail,
                        old_value=existing_value,
                        new_value=incoming_value,
                        source=entry.entry_id,
                    )
                    if failure_sink is not None:
                        failure_sink.append(event)
                    logger.warning(
                        "protected_overwrite blocked at order %d for key %s "
                        "(kept %r, rejected %r from %s)",
                        now_order,
                        key,
                        existing_value,
                        incoming_value,
                        entry.entry_id,
                    )
                    return False
                # Unprotected (or same-value) same-key write: latest wins.
                self._remove(existing.entry_id)

        self._add(entry, embedding)
        return True


class ArchivalStore(_VectorStore):
    """The auditable "forgotten" tier evicted entries are demoted into (EC4).

    Demotion preserves the entry's id/content/fact/provenance/salience (only its
    :class:`MemoryTier` flips to ``ARCHIVAL``) and records ``(reason, at_order)``
    per archived entry, so any forgetting is recoverable and auditable. The
    archival tier is never retrieved by the default policy.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the archival store.

        Args:
            dim: Embedding dimensionality of the vector index.
        """
        super().__init__(dim, MemoryTier.ARCHIVAL)
        self._meta: dict[str, tuple[str, int]] = {}

    def demote(
        self,
        entry: MemoryEntry,
        embedding: np.ndarray | None,
        reason: str,
        at_order: int,
    ) -> None:
        """Record ``entry`` in the archival tier (does not delete it).

        Sets ``entry.tier`` to ``ARCHIVAL`` while preserving its
        id/content/fact/provenance/salience, stores the (optional) embedding, and
        records ``(reason, at_order)`` for audit. Re-demoting an already-archived
        id replaces the record rather than raising.

        Args:
            entry: The entry being demoted (mutated in place: tier flips).
            embedding: The entry's vector to preserve, or ``None``.
            reason: Why the entry was demoted (e.g. ``"episodic_capacity"``).
            at_order: Stream order at which the demotion happened.
        """
        entry.tier = MemoryTier.ARCHIVAL
        if entry.entry_id in self._entries:
            self._remove(entry.entry_id)
        self._add(entry, embedding)
        self._meta[entry.entry_id] = (reason, at_order)

    def recover(self, entry_id: str) -> MemoryEntry | None:
        """Return the archived entry with ``entry_id`` (content/fact intact)."""
        return self._entries.get(entry_id)

    def contains(self, entry_id: str) -> bool:
        """Return whether ``entry_id`` is held in the archival tier."""
        return entry_id in self._entries

    def reason_for(self, entry_id: str) -> tuple[str, int] | None:
        """Return the ``(reason, at_order)`` recorded for an archived entry."""
        return self._meta.get(entry_id)


class MemorySubstrate:
    """The dual-store-plus-archival memory substrate (FR2.1-FR2.5).

    Owns the three physical stores, the run-wide failure-event sink, and the
    append-then-demote :meth:`observe` write path used by the wake agent. The
    stores are physically separate objects (EC2): a write to one never appears in
    another, and each tier's retrieval/footprint/forgetting is independently
    queryable.

    Attributes:
        episodic: The episodic observation buffer.
        semantic: The consolidated knowledge store.
        archival: The auditable forgotten tier.
        failure_events: Run-wide sink of recorded failure events.
        archival_enabled: Whether evicted entries are demoted (vs. dropped).
    """

    def __init__(self, mem_cfg: MemoryConfig, dim: int) -> None:
        """Build the substrate from a :class:`MemoryConfig`.

        Args:
            mem_cfg: The memory configuration (capacity, half-life, weights,
                archival toggle).
            dim: Embedding dimensionality shared by all three stores.
        """
        self.episodic = EpisodicStore(
            dim=dim,
            capacity=mem_cfg.episodic_capacity,
            half_life=mem_cfg.recency_half_life,
            w_recency=mem_cfg.weight_recency,
            w_importance=mem_cfg.weight_importance,
        )
        self.semantic = SemanticStore(
            dim=dim,
            half_life=mem_cfg.recency_half_life,
            w_recency=mem_cfg.weight_recency,
            w_importance=mem_cfg.weight_importance,
        )
        self.archival = ArchivalStore(dim=dim)
        self.failure_events: list[FailureEvent] = []
        self.archival_enabled: bool = mem_cfg.archival_enabled

    def observe(
        self, entry: MemoryEntry, embedding: np.ndarray, now_order: int
    ) -> list[MemoryEntry]:
        """Append ``entry`` to the episodic tier and demote any evictions.

        Appends ``entry`` (with its vector) to the episodic store. Any entries
        evicted by the append are demoted to the archival tier when
        :attr:`archival_enabled`, or dropped with a logged INFO line otherwise
        (DX2: never drop silently). Returns the evicted entries for telemetry.

        Args:
            entry: The (EPISODIC) entry to observe.
            embedding: The entry's ``(dim,)`` embedding (computed by the caller).
            now_order: The current stream order.

        Returns:
            The list of evicted entries (for the eviction count / telemetry).
        """
        evicted = self.episodic.append(entry, embedding, now_order)
        for victim in evicted:
            if self.archival_enabled:
                self.archival.demote(
                    victim,
                    self.episodic.evicted_vector(victim.entry_id),
                    reason="episodic_capacity",
                    at_order=now_order,
                )
            else:
                logger.info(
                    "dropping evicted episodic entry %s at order %d "
                    "(archival disabled; not recoverable)",
                    victim.entry_id,
                    now_order,
                )
        return evicted

    def demote_entry(
        self, entry_id: str, *, reason: str, at_order: int
    ) -> bool:
        """Demote one active entry (episodic or semantic) to archival (FR4.7).

        The dream-driven counterpart of capacity eviction: removes ``entry_id``
        from whichever active store holds it and demotes it to the auditable
        archival tier (demote-not-delete, EC7), or — when
        :attr:`archival_enabled` is ``False`` — drops it with a logged INFO line
        (DX2: never drop silently). Episodic is checked before semantic. Used by
        the conflict/unlearning step to retire a contradicting entry without
        destroying it.

        Args:
            entry_id: The active entry to demote.
            reason: Why it is being demoted (e.g. ``"conflict_unlearning"``).
            at_order: Stream order at which the demotion happened.

        Returns:
            ``True`` if an active entry was found and demoted (or dropped);
            ``False`` if no active store held ``entry_id``.
        """
        for store in (self.episodic, self.semantic):
            if store.get(entry_id) is not None:
                entry, vec = store.pop(entry_id)
                if entry is None:  # pragma: no cover - guarded by get() above
                    return False
                if self.archival_enabled:
                    self.archival.demote(entry, vec, reason=reason, at_order=at_order)
                else:
                    logger.info(
                        "dropping demoted entry %s at order %d "
                        "(archival disabled; not recoverable)",
                        entry_id,
                        at_order,
                    )
                return True
        return False

    def footprint(self) -> MemoryFootprint:
        """Return the per-tier footprint with ``total_bytes`` summed (EC2)."""
        episodic = self.episodic.footprint()
        semantic = self.semantic.footprint()
        archival = self.archival.footprint()
        return MemoryFootprint(
            episodic=episodic,
            semantic=semantic,
            archival=archival,
            total_bytes=episodic.bytes + semantic.bytes + archival.bytes,
        )
