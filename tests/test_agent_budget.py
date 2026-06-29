"""Tests for the wake agent's token-budget controller (Phase 2, WS-AGENT, FR3.3).

Two halves: pure unit tests of
:class:`slow_wave.agent.budget.TokenBudgetController`
(``can_afford``/``record``/``remaining``/``exhausted``/``skip``), and an
integration check that a tiny ``token_budget`` makes the wake loop *skip* its
per-task reasoning calls (staying within budget) while an unbounded budget makes
every per-task call.
"""

from __future__ import annotations

import pytest

from slow_wave.agent.budget import TokenBudgetController
from slow_wave.agent.wake import WakeAgent
from slow_wave.config import load_config
from slow_wave.embeddings import get_embedder
from slow_wave.repro.seeding import derive_seed
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set


# --------------------------------------------------------------------------- #
# Unit tests: TokenBudgetController
# --------------------------------------------------------------------------- #
def test_unbounded_budget_affords_everything():
    """A ``None`` budget never limits, never exhausts, and never reports remaining."""
    b = TokenBudgetController(None)
    assert b.remaining is None
    assert b.exhausted is False
    assert b.can_afford(10**12) is True
    b.record(1000, 500)
    assert b.total_spent == 1500
    assert b.spent_input == 1000
    assert b.spent_output == 500
    assert b.remaining is None
    assert b.exhausted is False
    assert b.can_afford(10**12) is True


def test_bounded_budget_affordability_and_exhaustion():
    """A finite budget tracks remaining, affordability, and exhaustion exactly."""
    b = TokenBudgetController(100)
    assert b.remaining == 100
    assert b.exhausted is False
    assert b.can_afford(100) is True
    assert b.can_afford(101) is False

    b.record(40, 30)  # total_spent = 70
    assert b.total_spent == 70
    assert b.remaining == 30
    assert b.exhausted is False
    assert b.can_afford(30) is True
    assert b.can_afford(31) is False

    b.record(30, 0)  # total_spent = 100
    assert b.total_spent == 100
    assert b.remaining == 0
    assert b.exhausted is True
    assert b.can_afford(1) is False


def test_skip_counts_and_is_safe(caplog):
    """``skip`` increments the counter and logs (DX2: never silent)."""
    b = TokenBudgetController(5)
    import logging

    with caplog.at_level(logging.INFO):
        b.skip()
        b.skip()
    assert b.n_skipped == 2
    assert any("skipping reasoning call" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Integration: budget gating in the wake loop
# --------------------------------------------------------------------------- #
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


def test_tiny_budget_skips_all_reasoning_calls(smoke_cfg, force_mock_llm):
    """A budget smaller than one call's estimate skips every per-task call."""
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.agent.token_budget = 5  # far below len(prompt)//4 + max_tokens
    stream, probe_set, embedder = _build(cfg)

    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set)
    tel = result.telemetry

    assert tel.reasoning_calls_skipped > 0
    assert tel.reasoning_calls_skipped == stream.n_tasks
    assert tel.reasoning_calls_made == 0
    assert tel.api_calls == 0
    assert agent.budget.n_skipped == stream.n_tasks
    assert agent.budget.total_spent == 0
    assert agent.budget.total_spent <= cfg.agent.token_budget
    # Accuracy is independent of reasoning (answers read from memory).
    assert result.accuracy_matrix.R[0][0] == 1.0


def test_unbounded_budget_makes_all_reasoning_calls(smoke_cfg, force_mock_llm):
    """A ``None`` budget executes every per-task reasoning call."""
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.agent.token_budget = None
    stream, probe_set, embedder = _build(cfg)

    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set)
    tel = result.telemetry

    assert tel.reasoning_calls_made == stream.n_tasks
    assert tel.reasoning_calls_skipped == 0
    assert tel.api_calls == stream.n_tasks
    assert tel.total_tokens > 0


def test_generous_budget_makes_all_calls_within_budget(smoke_cfg, force_mock_llm):
    """A large finite budget still affords every call and stays within ceiling."""
    cfg = smoke_cfg.model_copy(deep=True)
    cfg.agent.token_budget = 10**6
    stream, probe_set, embedder = _build(cfg)

    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set)
    tel = result.telemetry

    assert tel.reasoning_calls_made == stream.n_tasks
    assert tel.reasoning_calls_skipped == 0
    assert agent.budget.total_spent <= cfg.agent.token_budget
