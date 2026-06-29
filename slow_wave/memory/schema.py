"""Data model for the Slow Wave memory substrate (Phase 2, WS-MEM).

This module is the **authoritative cross-module contract** for the memory tier
(see ``docs/PHASE2_CONTRACT.md`` — section "WS-MEM — Memory substrate"). The
three physical stores, the salience primitives, the retrieval policy registry,
and the Phase 2 wake agent all build against the types defined here and must not
redefine them.

Design principles
-----------------
* **Confound separation by construction (FR1.6).** No memory object carries a
  relevance label or any name in
  :data:`slow_wave.stream.guard.BANNED_FIELD_NAMES`. The Phase 2 integration
  confound test walks live :class:`MemoryEntry` graphs with
  :func:`slow_wave.stream.guard.assert_no_label_leak`, so every field name below
  is deliberately confound-free.
* **Mutability where the biology demands it.** :class:`SalienceMeta` and
  :class:`MemoryEntry` are **mutable**: salience (recency, access count) updates
  on every access and an entry's :class:`MemoryTier` flips when it is demoted to
  the archival tier. The frozen value objects of Phase 1
  (:class:`~slow_wave.stream.schema.Fact`) are reused as-is, not redefined.
* **Determinism (DX1).** Every type is JSON-serializable with stable key order
  (pydantic ``model_dump(mode="json")``). Footprints are reported in exact bytes
  (``n_vectors * dim * 4`` for a float32 index) so two runs from the same config
  + seed are byte-identical.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from slow_wave.stream.schema import Fact


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class MemoryTier(str, Enum):
    """The physical tier an entry currently lives in (FR2.1, FR2.4).

    ``EPISODIC`` is the append-only, capacity-bounded observation buffer;
    ``SEMANTIC`` is the consolidated knowledge store (written only by the Phase 3
    dream engine); ``ARCHIVAL`` is the auditable "forgotten" tier that evicted
    entries are *demoted* into rather than deleted.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVAL = "archival"


# --------------------------------------------------------------------------- #
# Salience & entries
# --------------------------------------------------------------------------- #
class SalienceMeta(BaseModel):
    """Mutable salience bookkeeping for a single memory entry (FR2.2).

    Updated in place whenever the owning entry is observed or accessed, so it is
    intentionally **not** frozen. The fields feed the recency/importance/novelty
    terms of the baseline retrieval and eviction scores.

    Attributes:
        importance: Salience weight; starts at ``MemoryConfig.base_salience``.
        recency_order: Stream order of the last observation/access of the entry.
        access_count: Number of times the entry has been touched (retrieved).
        novelty: Embedding distance to the consolidated store at insert, in
            ``[0, 1]`` (``1.0`` == maximally novel / nothing similar yet).
        surprise: Reserved reward/surprise term; defaults to ``0.0``.
    """

    model_config = ConfigDict(extra="forbid")

    importance: float
    recency_order: int
    access_count: int = 0
    novelty: float = 0.0
    surprise: float = 0.0


class MemoryEntry(BaseModel):
    """A single memory entry held by one of the three physical stores.

    Mutable on purpose: :attr:`salience` changes on access and :attr:`tier`
    flips from ``EPISODIC`` to ``ARCHIVAL`` when the entry is demoted. The entry
    keeps a :attr:`provenance` pointer to its source(s) so any consolidated or
    episodic item can be traced back to its origin (FR2.x, EC3).

    Attributes:
        entry_id: Stable id; episodic entries use ``"e{order:06d}"``.
        tier: The tier the entry currently lives in.
        content: Natural-language surface form (what was embedded).
        fact: The asserted ``(subject, attribute, value)`` triple, or ``None``
            for an entry that asserts no structured fact.
        created_order: Stream order at creation (latest-wins tie-break for
            answers, so contradictions resolve to the final value).
        salience: Mutable salience bookkeeping.
        provenance: Source ids — stream ``item_id``\\ (s) for episodic entries,
            source ``entry_id``\\ (s) for consolidated/semantic entries (EC3).
        protected: EWC-spirit write-protection flag; a protected semantic entry
            refuses a conflicting overwrite (FR2.5).
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    tier: MemoryTier
    content: str
    fact: Fact | None = None
    created_order: int
    salience: SalienceMeta
    provenance: tuple[str, ...]
    protected: bool = False

    def key(self) -> tuple[str, str] | None:
        """Return the ``(subject, attribute)`` identity of the entry's fact.

        Returns:
            The fact's :meth:`~slow_wave.stream.schema.Fact.key`, or ``None`` if
            the entry carries no fact.
        """
        return self.fact.key() if self.fact is not None else None


# --------------------------------------------------------------------------- #
# Failure events & footprints
# --------------------------------------------------------------------------- #
class FailureEvent(BaseModel):
    """A logged memory-substrate failure (EC5; e.g. a protection violation).

    Attributes:
        kind: Failure category, e.g. ``"protected_overwrite"``.
        at_order: Stream order at which the failure occurred.
        key: The ``(subject, attribute)`` key involved, or ``None``.
        detail: Human-readable description of what happened.
        old_value: The value preserved by the substrate (e.g. the protected
            value), or ``None``.
        new_value: The rejected/offending value, or ``None``.
        source: Id of the offending write (e.g. the distractor entry id), or
            ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    at_order: int
    key: tuple[str, str] | None
    detail: str
    old_value: str | None = None
    new_value: str | None = None
    source: str | None = None


class StoreFootprint(BaseModel):
    """The memory footprint of a single physical store (EC2).

    Attributes:
        tier: Which tier this footprint describes.
        n_entries: Number of live entries held by the store.
        n_vectors: Number of vectors in the store's parallel float32 index.
        dim: Embedding dimensionality of the index.
        bytes: Index size in bytes, exactly ``n_vectors * dim * 4`` (float32).
    """

    model_config = ConfigDict(extra="forbid")

    tier: MemoryTier
    n_entries: int
    n_vectors: int
    dim: int
    bytes: int


class MemoryFootprint(BaseModel):
    """The combined footprint of the whole substrate, reported per tier (EC2).

    Attributes:
        episodic: Footprint of the episodic store.
        semantic: Footprint of the semantic store.
        archival: Footprint of the archival store.
        total_bytes: Sum of the three tiers' ``bytes``.
    """

    model_config = ConfigDict(extra="forbid")

    episodic: StoreFootprint
    semantic: StoreFootprint
    archival: StoreFootprint
    total_bytes: int
