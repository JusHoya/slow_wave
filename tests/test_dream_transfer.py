"""Tests for the TRANSFER operator + CLS interleaving (Phase 3, WS-TRANSFER).

These cover the contract's "Tests must cover (EC4 + gating support)" bullets:

* a transfer over signal-bearing sources writes semantic entries that **preserve
  the source fact** and a provenance pointer back to the source episodic id;
* ``n_refused`` increments when a source would overwrite a **protected** same-key
  semantic entry, and the protected value is preserved (reusing the FR2.5 path);
* **EC4** — with a pre-populated semantic store, ``cls_interleave=True`` yields
  ``n_interleaved_items > 0`` and ``cls_interleave=False`` yields
  ``n_interleaved_items == 0`` on the same inputs (removable interleaving);
* noise (``fact=None``) sources are not consolidated;
* ``transfer_max_calls`` caps the per-cycle calls and records/logs
  ``n_skipped_calls``;
* determinism under the mock LLM (same ``rng`` + inputs => identical written ids
  and counts).

All LLM calls go through the deterministic mock (``force_mock_llm`` removes any
API key) so the structured results are byte-identical run-to-run (DX1).
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.config import Config, DreamConfig, MemoryConfig
from slow_wave.dream.transfer import transfer
from slow_wave.embeddings import HashEmbedder
from slow_wave.llm import complete
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact

DIM = 16


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _episodic(
    order: int,
    subject: str,
    attribute: str,
    value: str,
    *,
    importance: float = 1.0,
    novelty: float = 0.5,
) -> MemoryEntry:
    """Build a fact-bearing EPISODIC source entry with id ``e{order:06d}``."""
    return MemoryEntry(
        entry_id=f"e{order:06d}",
        tier=MemoryTier.EPISODIC,
        content=f"The {attribute} of {subject} is {value}.",
        fact=Fact(subject=subject, attribute=attribute, value=value),
        created_order=order,
        salience=SalienceMeta(
            importance=importance, recency_order=order, novelty=novelty
        ),
        provenance=(f"i{order:06d}",),
    )


def _noise(order: int) -> MemoryEntry:
    """Build a fact-free (noise) EPISODIC source entry (``fact=None``)."""
    return MemoryEntry(
        entry_id=f"e{order:06d}",
        tier=MemoryTier.EPISODIC,
        content=f"qux zlork wibble {order} salad noise",
        fact=None,
        created_order=order,
        salience=SalienceMeta(importance=0.5, recency_order=order),
        provenance=(f"i{order:06d}",),
    )


def _seed_semantic(
    substrate: MemorySubstrate,
    embedder: HashEmbedder,
    triples: list[tuple[str, str, str]],
    *,
    protected: bool = False,
) -> None:
    """Pre-populate the semantic store with prior consolidated entries."""
    for i, (subject, attribute, value) in enumerate(triples):
        content = f"Prior: the {attribute} of {subject} is {value}."
        entry = MemoryEntry(
            entry_id=f"sp{i:04d}",
            tier=MemoryTier.SEMANTIC,
            content=content,
            fact=Fact(subject=subject, attribute=attribute, value=value),
            created_order=i,
            salience=SalienceMeta(importance=1.0, recency_order=i),
            provenance=(),
            protected=protected,
        )
        assert substrate.semantic.upsert(entry, embedder.encode([content])[0], i)


def _substrate() -> MemorySubstrate:
    """A fresh substrate sized to the test embedder."""
    return MemorySubstrate(MemoryConfig(), dim=DIM)


def _cfg() -> Config:
    """A minimal top-level config (only ``model`` is used by the mock LLM)."""
    return Config(experiment="transfer-test")


# --------------------------------------------------------------------------- #
# Fact preservation + provenance
# --------------------------------------------------------------------------- #
def test_transfer_preserves_fact_and_provenance(force_mock_llm) -> None:
    """Consolidated semantic entries preserve the source fact + provenance."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()
    sources = [
        _episodic(0, "alice", "role", "engineer"),
        _episodic(1, "bob", "city", "paris"),
    ]

    res = transfer(
        substrate,
        sources,
        cfg=_cfg(),
        dream_cfg=DreamConfig(cls_interleave=False, transfer_batch_size=8),
        embedder=embedder,
        llm_complete=complete,
        rng=np.random.default_rng(0),
        now_order=100,
    )

    assert res.n_batches == 1
    assert res.n_consolidated == 2
    assert res.n_semantic_written == 2
    assert res.n_refused == 0
    assert res.api_calls == 1
    assert set(res.written_entry_ids) == {"s000000", "s000001"}

    # The semantic entry carries the source fact verbatim and traces to its
    # source episodic id; created_order is the source's (latest-wins).
    s0 = substrate.semantic.get("s000000")
    assert s0 is not None
    assert s0.tier is MemoryTier.SEMANTIC
    assert s0.fact == sources[0].fact
    assert s0.created_order == 0
    assert s0.provenance == ("e000000",)
    assert s0.salience.recency_order == 100
    # The natural-language content is the dream summary (mock), not the fact.
    assert s0.content
    assert s0.content != sources[0].content

    # Exact-key lookup still answers crisply from the consolidated fact.
    kept = substrate.semantic.find_by_key(("bob", "city"))
    assert len(kept) == 1
    assert kept[0].fact.value == "paris"


# --------------------------------------------------------------------------- #
# FR2.5 protected-overwrite refusal
# --------------------------------------------------------------------------- #
def test_transfer_refuses_protected_overwrite(force_mock_llm, caplog) -> None:
    """A source clobbering a protected same-key fact is refused (n_refused)."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()
    # Establish a *protected* consolidated fact (alice, role) = engineer.
    _seed_semantic(substrate, embedder, [("alice", "role", "engineer")], protected=True)

    # A source asserts a conflicting value for the same key.
    sources = [_episodic(5, "alice", "role", "janitor")]

    with caplog.at_level(logging.WARNING, logger="slow_wave.memory.stores"):
        res = transfer(
            substrate,
            sources,
            cfg=_cfg(),
            dream_cfg=DreamConfig(cls_interleave=False, transfer_batch_size=8),
            embedder=embedder,
            llm_complete=complete,
            rng=np.random.default_rng(0),
            now_order=100,
        )

    assert res.n_consolidated == 1
    assert res.n_semantic_written == 0
    assert res.n_refused == 1
    assert res.written_entry_ids == []

    # The protected value is preserved — exactly one entry, the original.
    kept = substrate.semantic.find_by_key(("alice", "role"))
    assert len(kept) == 1
    assert kept[0].entry_id == "sp0000"
    assert kept[0].fact.value == "engineer"
    assert kept[0].protected is True

    # The FR2.5 path recorded a protected_overwrite failure event and logged it.
    assert any(e.kind == "protected_overwrite" for e in substrate.failure_events)
    assert any("protected_overwrite" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# EC4 — CLS interleaving on/off, removable
# --------------------------------------------------------------------------- #
def test_cls_interleaving_on_pulls_prior_consolidated(force_mock_llm) -> None:
    """EC4 (on): with a populated semantic store, interleaving mixes prior memory."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()
    _seed_semantic(
        substrate, embedder, [("carol", "city", "rome"), ("dave", "city", "oslo")]
    )
    sources = [
        _episodic(10, "alice", "role", "engineer"),
        _episodic(11, "bob", "city", "paris"),
    ]

    res = transfer(
        substrate,
        sources,
        cfg=_cfg(),
        dream_cfg=DreamConfig(
            cls_interleave=True, cls_interleave_ratio=0.5, transfer_batch_size=8
        ),
        embedder=embedder,
        llm_complete=complete,
        rng=np.random.default_rng(0),
        now_order=100,
    )

    assert res.interleaved is True
    assert res.n_interleaved_items > 0  # prior consolidated memories were mixed in
    assert res.n_semantic_written == 2  # the new sources are still consolidated


def test_cls_interleaving_off_mixes_nothing(force_mock_llm) -> None:
    """EC4 (off): the same inputs with interleaving off mix no prior memory."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()
    _seed_semantic(
        substrate, embedder, [("carol", "city", "rome"), ("dave", "city", "oslo")]
    )
    sources = [
        _episodic(10, "alice", "role", "engineer"),
        _episodic(11, "bob", "city", "paris"),
    ]

    res = transfer(
        substrate,
        sources,
        cfg=_cfg(),
        dream_cfg=DreamConfig(cls_interleave=False, transfer_batch_size=8),
        embedder=embedder,
        llm_complete=complete,
        rng=np.random.default_rng(0),
        now_order=100,
    )

    assert res.interleaved is False
    assert res.n_interleaved_items == 0  # catastrophic-interference condition
    assert res.n_semantic_written == 2


def test_cls_interleaving_empty_semantic_is_zero_even_when_on(force_mock_llm) -> None:
    """On the first-ever cycle (empty semantic) interleaving pulls 0 even when on."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()  # semantic store is empty
    sources = [_episodic(0, "alice", "role", "engineer")]

    res = transfer(
        substrate,
        sources,
        cfg=_cfg(),
        dream_cfg=DreamConfig(cls_interleave=True, transfer_batch_size=8),
        embedder=embedder,
        llm_complete=complete,
        rng=np.random.default_rng(0),
        now_order=100,
    )

    assert res.interleaved is True
    assert res.n_interleaved_items == 0


# --------------------------------------------------------------------------- #
# Noise is never consolidated
# --------------------------------------------------------------------------- #
def test_noise_sources_not_consolidated(force_mock_llm) -> None:
    """Fact-free (noise) sources are filtered out before consolidation."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()
    sources = [
        _episodic(0, "alice", "role", "engineer"),
        _noise(1),
        _noise(2),
    ]

    res = transfer(
        substrate,
        sources,
        cfg=_cfg(),
        dream_cfg=DreamConfig(cls_interleave=False, transfer_batch_size=8),
        embedder=embedder,
        llm_complete=complete,
        rng=np.random.default_rng(0),
        now_order=100,
    )

    assert res.n_consolidated == 1
    assert res.n_semantic_written == 1
    assert res.written_entry_ids == ["s000000"]
    assert substrate.semantic.get("s000000") is not None
    # The noise sources produced no semantic entries.
    assert substrate.semantic.get("s000001") is None
    assert substrate.semantic.get("s000002") is None


# --------------------------------------------------------------------------- #
# transfer_max_calls cap (DX2)
# --------------------------------------------------------------------------- #
def test_transfer_max_calls_caps_and_records_skips(force_mock_llm, caplog) -> None:
    """The per-cycle call ceiling caps calls and records/logs n_skipped_calls."""
    embedder = HashEmbedder(DIM)
    substrate = _substrate()
    # 5 fact sources, batch_size=2 => 3 batches; cap at 1 call => 2 batches skipped.
    sources = [_episodic(i, f"subj{i}", "attr", f"val{i}") for i in range(5)]
    dream_cfg = DreamConfig(
        cls_interleave=False, transfer_batch_size=2, transfer_max_calls=1
    )

    with caplog.at_level(logging.WARNING, logger="slow_wave.dream.transfer"):
        res = transfer(
            substrate,
            sources,
            cfg=_cfg(),
            dream_cfg=dream_cfg,
            embedder=embedder,
            llm_complete=complete,
            rng=np.random.default_rng(0),
            now_order=100,
        )

    assert res.n_batches == 3
    assert res.api_calls == 1
    assert res.n_skipped_calls == 2
    assert res.api_calls + res.n_skipped_calls == res.n_batches
    # Only the first (un-skipped) batch's two sources were consolidated.
    assert res.n_consolidated == 2
    assert res.n_semantic_written == 2
    assert res.written_entry_ids == ["s000000", "s000001"]
    # Skipped batches left no semantic entries behind.
    assert substrate.semantic.get("s000002") is None
    assert substrate.semantic.get("s000004") is None
    # The skip is logged (DX2 — never dropped silently).
    assert any("transfer_max_calls" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Determinism (DX1)
# --------------------------------------------------------------------------- #
def test_transfer_is_deterministic_under_mock_llm(force_mock_llm) -> None:
    """Same rng + inputs => byte-identical written ids and counts (DX1)."""

    def run() -> dict:
        embedder = HashEmbedder(DIM)
        substrate = _substrate()
        _seed_semantic(
            substrate,
            embedder,
            [
                ("carol", "city", "rome"),
                ("dave", "city", "oslo"),
                ("eve", "city", "kyiv"),
            ],
        )
        sources = [_episodic(10 + i, f"s{i}", "attr", f"v{i}") for i in range(6)]
        res = transfer(
            substrate,
            sources,
            cfg=_cfg(),
            dream_cfg=DreamConfig(
                cls_interleave=True, cls_interleave_ratio=0.5, transfer_batch_size=4
            ),
            embedder=embedder,
            llm_complete=complete,
            rng=np.random.default_rng(42),
            now_order=200,
        )
        return res.model_dump()

    first = run()
    second = run()
    assert first == second
    # Sanity: the run actually did meaningful work (multiple batches + writes).
    assert first["n_batches"] == 2
    assert first["n_semantic_written"] == 6
    assert first["n_interleaved_items"] > 0
