"""Cross-module Phase 3 integration checks: gating, provenance/archival, drift.

Exercises the dream engine wired into the real wake loop via ``sleep_hook`` and
the one-command :func:`slow_wave.dream.runner.run_dream`, covering the exit
criteria that span modules:

* **EC3 (gating)** — TRANSFER writes the SEMANTIC store *only* inside a scheduled
  sleep window; no semantic write happens during wake ingest.
* **EC7 (provenance + archival intact, no hard deletes)** — after a full dream
  run over a Phase-1 stream, every semantic entry traces to a still-existing
  source and every archived entry is recoverable (nothing was hard-deleted).
* **FR1.6 (confound guard)** — no live memory entry (episodic / semantic /
  archival) created by the dream engine carries a relevance label or banned field.
* **DX1 (determinism)** — two ``run_dream`` runs from the same config+seed produce
  identical reproducible outputs under the mock LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slow_wave.agent.wake import WakeAgent
from slow_wave.config import Config, DreamConfig, EmbeddingConfig, MemoryConfig
from slow_wave.dream.engine import DreamEngine
from slow_wave.dream.runner import provenance_archival_audit, run_dream
from slow_wave.embeddings import get_embedder
from slow_wave.repro.manifest import read_manifest
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream import guard
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set
from slow_wave.stream.schema import CLScenario, LabelMix, StreamGenConfig


def _dream_config(seed: int = 11, *, capacity: int = 20) -> Config:
    """A small, deterministic, all-operators-on dream config over a bounded store."""
    return Config(
        experiment="dream-integration",
        seed=seed,
        embedding=EmbeddingConfig(backend="hash", model="hash-bow-v1", dim=384),
        stream=StreamGenConfig(
            scenario=CLScenario.TASK_INCREMENTAL,
            n_tasks=3,
            items_per_task=12,
            label_mix=LabelMix(signal=0.5, distractor=0.3, noise=0.2),
            n_subjects_per_task=5,
            n_attributes=2,
            n_values=10,
            probes_per_task=3,
        ),
        memory=MemoryConfig(episodic_capacity=capacity, archival_enabled=True),
        dream=DreamConfig(
            enabled=True,
            replay_enabled=True,
            transfer_enabled=True,
            downscale_enabled=True,
            augment_enabled=True,
            conflict_enabled=True,
            sleep_every_n_tasks=1,
            replay_sample_size=10,
            transfer_batch_size=5,
            augment_per_cycle=2,
        ),
    )


def _build_run_inputs(cfg: Config):
    """Generate the stream + probe set + embedder for a config (mirrors run_dream)."""
    set_global_seeds(cfg.seed)
    stream = generate_stream(cfg.stream, derive_seed(cfg.seed, "stream"))
    probe_set = build_probe_set(stream)
    embedder = get_embedder(cfg)
    return stream, probe_set, embedder


def test_ec3_transfer_is_gated_to_sleep_windows(force_mock_llm) -> None:
    """EC3: no semantic write happens during wake ingest — only in sleep windows.

    A spy hook records the semantic-store size at the moment each sleep window
    opens. The first recorded value must be 0: the entire first task was ingested
    (and its reasoning call made) with the SEMANTIC store still empty, proving no
    semantic write occurred during wake. By run end the store is populated, so the
    writes happened *only* inside the dream cycles.
    """
    cfg = _dream_config()
    stream, probe_set, embedder = _build_run_inputs(cfg)
    engine = DreamEngine(cfg)
    agent = WakeAgent(cfg, embedder)

    semantic_size_at_sleep: list[int] = []
    real_hook = engine.sleep_hook

    def spy_hook(substrate, **kwargs):
        semantic_size_at_sleep.append(len(substrate.semantic))
        return real_hook(substrate, **kwargs)

    result = agent.run(stream, probe_set, sleep_hook=spy_hook)

    assert semantic_size_at_sleep, "no sleep window ever opened"
    # When the first sleep window opens, task 0 has been fully ingested but no
    # dream cycle has run yet => the semantic store must still be empty.
    assert semantic_size_at_sleep[0] == 0
    # After the run, consolidation has populated the semantic store.
    assert len(result.substrate.semantic) > 0


def test_ec3_no_sleep_hook_leaves_semantic_empty(force_mock_llm) -> None:
    """EC3 control: without the dream hook the SEMANTIC store stays empty.

    The Phase 2 gating invariant still holds — a wake run with no ``sleep_hook``
    performs zero semantic writes — so any semantic content in the dream run is
    attributable to the (gated) dream cycle, not to wake ingest.
    """
    cfg = _dream_config()
    stream, probe_set, embedder = _build_run_inputs(cfg)
    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set)  # no sleep_hook
    assert len(result.substrate.semantic) == 0


def test_ec7_provenance_and_archival_intact_no_hard_deletes(
    tmp_path: Path, force_mock_llm
) -> None:
    """EC7: a full dream run leaves provenance + archival audit intact (no deletes).

    The bounded episodic store forces eviction (→ archival demotion) and the
    cycle's transfer/augment/conflict steps write and demote, so the audit is a
    live check: every semantic entry traces to a still-existing source, and every
    archived entry is recoverable. Nothing is hard-deleted.
    """
    cfg = _dream_config(capacity=18)
    manifest = read_manifest(run_dream(cfg, out_dir=tmp_path))
    audit = manifest.results["provenance_archival_audit"]

    # Eviction actually happened (the bounded store demoted to archival)...
    assert audit["n_archival"] > 0
    # ...and every archived entry is recoverable (demote-not-delete, no deletes).
    assert audit["n_archival_recoverable"] == audit["n_archival"]

    # Consolidation actually happened, and every semantic entry's provenance
    # resolves to a source that still exists (live episodic or archived) — the
    # provenance audit is intact end-to-end.
    assert audit["n_semantic"] > 0
    assert audit["n_semantic_traceable"] == audit["n_semantic"]
    assert audit["example_semantic_provenance"] is not None


def test_ec7_live_substrate_provenance_and_recovery(force_mock_llm) -> None:
    """EC7 (object level): trace a real semantic entry back to its source episode.

    Drives the engine over a real stream and asserts, on the live substrate, that
    at least one semantic entry's provenance points to an entry recoverable from
    either the active episodic tier or the archival tier — the concrete
    "trace a consolidated item back to its origin" check, with no hard deletes.
    """
    cfg = _dream_config(capacity=16)
    stream, probe_set, embedder = _build_run_inputs(cfg)
    engine = DreamEngine(cfg)
    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set, sleep_hook=engine.sleep_hook)
    substrate = result.substrate

    semantic_entries = substrate.semantic.all_entries()
    assert semantic_entries, "dream run wrote no semantic entries"

    traced = 0
    for entry in semantic_entries:
        assert entry.provenance, f"semantic entry {entry.entry_id} has no provenance"
        for src_id in entry.provenance:
            if (
                substrate.episodic.get(src_id) is not None
                or substrate.archival.recover(src_id) is not None
            ):
                traced += 1
                break
    assert traced == len(semantic_entries)

    # No hard deletes: the provenance-archival audit agrees.
    audit = provenance_archival_audit(substrate)
    assert audit["n_archival_recoverable"] == audit["n_archival"]


def test_confound_guard_over_live_dream_entries(force_mock_llm) -> None:
    """FR1.6: no dream-created entry (any tier) carries a label or banned field.

    Walks every live entry across episodic, semantic, and archival tiers with the
    confound guard after a full dream run, so consolidated and pseudo-episode
    entries are proven label-free by construction.
    """
    cfg = _dream_config()
    stream, probe_set, embedder = _build_run_inputs(cfg)
    engine = DreamEngine(cfg)
    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set, sleep_hook=engine.sleep_hook)
    substrate = result.substrate

    all_entries = (
        substrate.episodic.all_entries()
        + substrate.semantic.all_entries()
        + substrate.archival.all_entries()
    )
    assert all_entries
    for entry in all_entries:
        guard.assert_no_label_leak(entry)


def test_run_dream_is_deterministic(tmp_path: Path, force_mock_llm) -> None:
    """DX1: two dream runs from the same config+seed reproduce identical outputs.

    Compares the full ``deterministic_probe`` block (R[i,j], footprint, dream
    cycle digest, semantic/pseudo/eviction counts, and the provenance audit) of
    two independent runs.
    """
    cfg = _dream_config()
    m1 = read_manifest(run_dream(cfg, out_dir=tmp_path / "a"))
    m2 = read_manifest(run_dream(cfg, out_dir=tmp_path / "b"))
    assert m1.deterministic_probe == m2.deterministic_probe
    # The dream-cycle structure digest specifically must match.
    assert (
        m1.deterministic_probe["dream_cycle_digest"]
        == m2.deterministic_probe["dream_cycle_digest"]
    )
