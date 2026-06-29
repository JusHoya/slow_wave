"""Tests for the no-sleep wake agent (Phase 2, WS-AGENT, FR3.1-FR3.3).

These exercise :class:`slow_wave.agent.wake.WakeAgent` end-to-end on a small
generated stream: it must return a well-formed square accuracy matrix with the
crisp present-implies-correct diagonal on the unbounded baseline, leave the
SEMANTIC store empty (the FR3.1 gating invariant), populate its telemetry, and
run purely from the label-free online view (proven by making ``offline_labels``
explode and confirming the run still succeeds).
"""

from __future__ import annotations

import pytest

import slow_wave.stream.schema as stream_schema
from slow_wave.agent.wake import WakeAgent, WakeResult
from slow_wave.config import load_config
from slow_wave.embeddings import get_embedder
from slow_wave.repro.seeding import derive_seed
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set


@pytest.fixture
def smoke_cfg(repo_root):
    """The unbounded no-sleep baseline config (``configs/agent_smoke.yaml``)."""
    return load_config(repo_root / "configs" / "agent_smoke.yaml")


def _build(cfg):
    """Return ``(stream, probe_set, embedder)`` for ``cfg`` (deterministic)."""
    stream = generate_stream(cfg.stream, derive_seed(cfg.seed, "stream"))
    probe_set = build_probe_set(stream)
    embedder = get_embedder(cfg)
    return stream, probe_set, embedder


def test_run_returns_well_formed_matrix(smoke_cfg, force_mock_llm):
    """The run yields a square ``n_tasks x n_tasks`` matrix with entries in [0, 1]."""
    stream, probe_set, embedder = _build(smoke_cfg)
    result = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)

    assert isinstance(result, WakeResult)
    R = result.accuracy_matrix
    assert R.n_tasks == stream.n_tasks
    assert len(R.R) == R.n_tasks
    for row in R.R:
        assert len(row) == R.n_tasks
        assert all(0.0 <= v <= 1.0 for v in row)


def test_unbounded_baseline_has_unit_diagonal(smoke_cfg, force_mock_llm):
    """Unbounded episodic store => present-implies-correct => ``R[t][t] == 1.0``."""
    stream, probe_set, embedder = _build(smoke_cfg)
    result = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)

    R = result.accuracy_matrix.R
    for t in range(result.accuracy_matrix.n_tasks):
        assert R[t][t] == 1.0
    # No-forgetting baseline: ACC is perfect and BWT is non-negative.
    assert result.metrics.acc == pytest.approx(1.0)
    assert result.metrics.bwt >= 0.0


def test_semantic_store_empty_after_run_gating(smoke_cfg, force_mock_llm):
    """FR3.1 gating: the wake loop performs no semantic writes (semantic empty)."""
    stream, probe_set, embedder = _build(smoke_cfg)
    result = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)

    assert len(result.substrate.semantic) == 0
    # Sanity: the episodic tier actually received every observation.
    assert len(result.substrate.episodic) == len(stream.items)


def test_telemetry_counts_populated(smoke_cfg, force_mock_llm):
    """Telemetry reflects ingestion, retrieval, and per-task reasoning calls."""
    stream, probe_set, embedder = _build(smoke_cfg)
    result = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)

    tel = result.telemetry
    assert tel.n_items_ingested == len(stream.items)
    # Each executed reasoning step does exactly one context retrieval to ground it.
    assert tel.retrieval_calls == tel.reasoning_calls_made
    # per_task reasoning with an unbounded budget => one call per task.
    assert tel.reasoning_calls_made == stream.n_tasks
    assert tel.reasoning_calls_skipped == 0
    assert tel.api_calls == stream.n_tasks
    assert tel.total_tokens > 0
    assert tel.p95_latency_s >= 0.0

    payload = tel.as_dict()
    assert payload["n_items_ingested"] == tel.n_items_ingested
    assert payload["total_tokens"] == tel.total_tokens
    # JSON-safe and key-sorted for the manifest.
    assert list(payload.keys()) == sorted(payload.keys())


def test_run_never_reads_labels(smoke_cfg, force_mock_llm, monkeypatch):
    """Confound (FR1.6): the run succeeds even if ``offline_labels`` explodes."""
    stream, probe_set, embedder = _build(smoke_cfg)

    def _boom(*args, **kwargs):
        raise AssertionError("the wake loop must never read offline labels (FR1.6)")

    # Patch every sanctioned label accessor the agent could conceivably reach.
    monkeypatch.setattr(stream_schema, "offline_labels", _boom)

    result = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)
    assert result.accuracy_matrix.n_tasks == stream.n_tasks


def test_run_is_deterministic(smoke_cfg, force_mock_llm):
    """Two runs from the same config+seed produce a byte-identical R + footprint."""
    stream, probe_set, embedder = _build(smoke_cfg)
    r1 = WakeAgent(smoke_cfg, embedder).run(stream, probe_set)

    stream2, probe_set2, embedder2 = _build(smoke_cfg)
    r2 = WakeAgent(smoke_cfg, embedder2).run(stream2, probe_set2)

    assert r1.accuracy_matrix.model_dump(mode="json") == r2.accuracy_matrix.model_dump(
        mode="json"
    )
    assert r1.footprint.model_dump(mode="json") == r2.footprint.model_dump(mode="json")
