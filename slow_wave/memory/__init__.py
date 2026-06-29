"""Dual-store memory substrate for the Slow Wave bench (Phase 2).

The public surface is re-exported here for convenience; see the per-module
docstrings and ``docs/PHASE2_CONTRACT.md`` for details:

* :mod:`slow_wave.memory.schema` — the memory data model (entries, tiers,
  salience, failure events, footprints).
* :mod:`slow_wave.memory.salience` — recency/novelty/retrieval/eviction scoring.
* :mod:`slow_wave.memory.stores` — the EPISODIC / SEMANTIC / ARCHIVAL physical
  stores and the :class:`~slow_wave.memory.stores.MemorySubstrate` that ties them.
* :mod:`slow_wave.memory.retrieval` — the pluggable retrieval policy + registry
  and the :func:`~slow_wave.memory.retrieval.retrieve` entry point (FR2.3).
"""

from __future__ import annotations

from slow_wave.memory.retrieval import (
    REGISTRY,
    RecencyImportanceRelevancePolicy,
    RetrievalPolicy,
    get_policy,
    register_policy,
    retrieve,
)
from slow_wave.memory.salience import (
    eviction_score,
    novelty_score,
    recency_factor,
    retrieval_score,
)
from slow_wave.memory.schema import (
    FailureEvent,
    MemoryEntry,
    MemoryFootprint,
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

__all__ = [
    # schema
    "FailureEvent",
    "MemoryEntry",
    "MemoryFootprint",
    "MemoryTier",
    "SalienceMeta",
    "StoreFootprint",
    # salience
    "eviction_score",
    "novelty_score",
    "recency_factor",
    "retrieval_score",
    # stores
    "ArchivalStore",
    "EpisodicStore",
    "MemorySubstrate",
    "SemanticStore",
    # retrieval
    "REGISTRY",
    "RecencyImportanceRelevancePolicy",
    "RetrievalPolicy",
    "get_policy",
    "register_policy",
    "retrieve",
]
