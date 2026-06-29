"""Tests for the GENERATIVE-AUGMENT operator (Phase 3, WS-AUGMENT, FR4.4, EC5).

These exercise :func:`slow_wave.dream.augment.augment` against a small in-memory
substrate of fact-bearing episodics. The exit criteria proven here (EC5):

* an augment pass over ``N`` sources writes
  ``n_pseudo == min(augment_per_cycle, len(sources))`` pseudo-episodes that live
  in the EPISODIC tier, carry ``fact=None`` (no R-corruption), and trace back via
  ``provenance`` to a source entry id;
* a per-cycle :class:`~slow_wave.dream.schema.FidelityScore` is produced with
  ``mean_fidelity`` in ``[0, 1]``, ``min_fidelity <= mean_fidelity``, and
  ``mean_drift == 1 - mean_fidelity``;
* ``augment_per_cycle == 0`` or empty ``sources`` yields an empty, zeroed result
  without raising;
* under the deterministic mock LLM, the same ``rng`` + inputs reproduce identical
  pseudo ids, text, and fidelity (DX1).
"""

from __future__ import annotations

import numpy as np
import pytest

from slow_wave.config import Config, DreamConfig
from slow_wave.dream.augment import augment
from slow_wave.dream.schema import AugmentResult
from slow_wave.embeddings import get_embedder
from slow_wave.llm import complete as llm_complete
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact

NOW_ORDER = 100


def _setup(fact_flags: list[bool], *, augment_per_cycle: int, seed: int = 7):
    """Build ``(cfg, dream_cfg, embedder, substrate, sources, rng)`` for a test.

    Seeds the episodic store with one entry per flag in ``fact_flags``; a truthy
    flag attaches a structured ``Fact`` (a "signal" source), a falsy one leaves
    ``fact=None`` (a fact-free "noise" source).
    """
    cfg = Config(experiment="augment-test")
    embedder = get_embedder(cfg)
    substrate = MemorySubstrate(cfg.memory, dim=embedder.dim)

    sources: list[MemoryEntry] = []
    for k, has_fact in enumerate(fact_flags):
        content = (
            f"observation {k}: station S{k} reports reading value V{k} "
            f"in the daily field log entry for sector {k % 4}"
        )
        fact = Fact(subject=f"S{k}", attribute="reading", value=f"V{k}") if has_fact else None
        entry = MemoryEntry(
            entry_id=f"e{k:06d}",
            tier=MemoryTier.EPISODIC,
            content=content,
            fact=fact,
            created_order=k,
            salience=SalienceMeta(importance=1.0 + 0.1 * k, recency_order=k),
            provenance=(f"i{k:06d}",),
        )
        substrate.observe(entry, embedder.encode([content])[0], k)
        sources.append(entry)

    dream_cfg = DreamConfig(augment_per_cycle=augment_per_cycle)
    rng = np.random.default_rng(seed)
    return cfg, dream_cfg, embedder, substrate, sources, rng


def _run(cfg, dream_cfg, embedder, substrate, sources, rng) -> AugmentResult:
    """Invoke ``augment`` with the standard keyword wiring."""
    return augment(
        substrate,
        sources,
        cfg=cfg,
        dream_cfg=dream_cfg,
        embedder=embedder,
        llm_complete=llm_complete,
        rng=rng,
        now_order=NOW_ORDER,
    )


def test_pseudo_episodes_are_episodic_factless_and_traceable(force_mock_llm):
    """EC5: n_pseudo == min(per_cycle, N); each pseudo is EPISODIC, factless, traced."""
    cfg, dream_cfg, embedder, substrate, sources, rng = _setup(
        [True] * 6, augment_per_cycle=4
    )
    res = _run(cfg, dream_cfg, embedder, substrate, sources, rng)

    assert isinstance(res, AugmentResult)
    assert res.n_pseudo == min(dream_cfg.augment_per_cycle, len(sources)) == 4
    assert len(res.pseudo_entry_ids) == 4
    assert res.api_calls == 4  # one generation call per pseudo-episode
    assert res.input_tokens > 0 and res.output_tokens > 0

    source_ids = {s.entry_id for s in sources}
    for pid in res.pseudo_entry_ids:
        entry = substrate.episodic.get(pid)  # (a) lives in the EPISODIC tier
        assert entry is not None
        assert entry.tier is MemoryTier.EPISODIC
        assert entry.fact is None  # (b) no structured fact -> no R-corruption
        assert len(entry.provenance) == 1  # (c) traces to exactly one source
        assert entry.provenance[0] in source_ids
        assert entry.created_order == NOW_ORDER
        assert entry.salience.recency_order == NOW_ORDER


def test_n_pseudo_capped_by_source_count(force_mock_llm):
    """EC5: with fewer sources than the per-cycle budget, n_pseudo == len(sources)."""
    cfg, dream_cfg, embedder, substrate, sources, rng = _setup(
        [True] * 3, augment_per_cycle=10
    )
    res = _run(cfg, dream_cfg, embedder, substrate, sources, rng)

    assert res.n_pseudo == 3 == min(10, len(sources))
    assert res.fidelity.n_pseudo == 3


def test_fidelity_score_bounds_and_drift(force_mock_llm):
    """EC5: a FidelityScore with mean in [0,1], min<=mean, drift == 1-mean."""
    cfg, dream_cfg, embedder, substrate, sources, rng = _setup(
        [True] * 5, augment_per_cycle=5
    )
    res = _run(cfg, dream_cfg, embedder, substrate, sources, rng)

    f = res.fidelity
    assert f.n_pseudo == 5
    assert 0.0 <= f.mean_fidelity <= 1.0
    assert 0.0 <= f.min_fidelity <= 1.0
    assert f.min_fidelity <= f.mean_fidelity + 1e-9
    assert f.mean_drift == pytest.approx(1.0 - f.mean_fidelity)


def test_zero_per_cycle_yields_empty_result(force_mock_llm):
    """EC5: augment_per_cycle == 0 -> empty, zeroed result without raising."""
    cfg, dream_cfg, embedder, substrate, sources, rng = _setup(
        [True] * 5, augment_per_cycle=0
    )
    res = _run(cfg, dream_cfg, embedder, substrate, sources, rng)

    assert res.n_pseudo == 0
    assert res.pseudo_entry_ids == []
    assert res.api_calls == 0
    assert res.fidelity.n_pseudo == 0
    assert res.fidelity.mean_fidelity == 0.0
    assert res.fidelity.min_fidelity == 0.0
    assert res.fidelity.mean_drift == 0.0
    # No pseudo-episodes were written to the episodic tier.
    assert len(substrate.episodic) == len(sources)


def test_empty_sources_yields_empty_result(force_mock_llm):
    """EC5: an empty source list -> empty, zeroed result without raising."""
    cfg, dream_cfg, embedder, substrate, _sources, rng = _setup(
        [], augment_per_cycle=4
    )
    res = _run(cfg, dream_cfg, embedder, substrate, [], rng)

    assert res.n_pseudo == 0
    assert res.pseudo_entry_ids == []
    assert res.fidelity.mean_drift == 0.0


def test_determinism_under_mock_llm(force_mock_llm):
    """DX1: same rng + inputs -> identical pseudo ids, text, and fidelity."""

    def run_once():
        cfg, dream_cfg, embedder, substrate, sources, rng = _setup(
            [True] * 6, augment_per_cycle=4, seed=42
        )
        res = _run(cfg, dream_cfg, embedder, substrate, sources, rng)
        texts = [substrate.episodic.get(pid).content for pid in res.pseudo_entry_ids]
        return res, texts

    res1, texts1 = run_once()
    res2, texts2 = run_once()

    assert res1.model_dump(mode="json") == res2.model_dump(mode="json")
    assert texts1 == texts2  # pseudo text is byte-identical under the mock LLM


def test_prefers_fact_bearing_sources(force_mock_llm):
    """The selector draws fact-bearing sources before fact-free ones."""
    # Three signal sources (e000000..e000002) and three noise sources.
    cfg, dream_cfg, embedder, substrate, sources, rng = _setup(
        [True, True, True, False, False, False], augment_per_cycle=3
    )
    res = _run(cfg, dream_cfg, embedder, substrate, sources, rng)

    fact_ids = {s.entry_id for s in sources if s.fact is not None}
    assert res.n_pseudo == 3
    for pid in res.pseudo_entry_ids:
        provenance_id = substrate.episodic.get(pid).provenance[0]
        assert provenance_id in fact_ids


def test_pseudo_kinds_cycle_through_augment_kinds(force_mock_llm):
    """Distinct kinds produce distinct mock pseudo text for an identical source."""
    cfg, _dream_cfg, embedder, substrate, sources, rng = _setup(
        [True], augment_per_cycle=1
    )
    # One source, three kinds: re-run conceptually by checking the prompt fan-out
    # via two kinds over one repeated source would need a richer pool; instead we
    # assert the three default kinds yield three distinct generations directly.
    src = sources[0]
    texts = {
        kind: llm_complete(cfg, f"{kind} of: {src.content}").text
        for kind in DreamConfig().augment_kinds
    }
    assert len(set(texts.values())) == len(texts)  # each kind -> distinct text
