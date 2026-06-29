"""Tests for slow_wave.eval.controls (Phase 4, WS-PREREG, FR5.6, EC8).

The two bias controls -- temperature-0 stability and the memory-drift detector
-- must produce well-formed numbers on a tiny run under the deterministic mock
LLM, hit the exact EC8 stability numbers (identical repeats), and be
byte-identical across two runs (DX1) and JSON-dumpable.
"""

from __future__ import annotations

import pytest

from slow_wave.config import Config
from slow_wave.eval.controls import memory_drift, temperature_zero_stability
from slow_wave.eval.schema import DriftResult, StabilityResult


@pytest.fixture
def cfg() -> Config:
    """A minimal config using the deterministic hash embedder + mock LLM."""
    return Config(experiment="controls-test")


# --------------------------------------------------------------------------- #
# Temperature-0 stability (EC8)
# --------------------------------------------------------------------------- #
def test_stability_mock_numbers(cfg: Config, force_mock_llm: None) -> None:
    """Under the mock LLM the summarizer is perfectly stable (the EC8 numbers)."""
    result = temperature_zero_stability(cfg, n_repeats=3)

    assert isinstance(result, StabilityResult)
    assert result.n_repeats == 3
    assert result.distinct_outputs == 1
    assert result.identical is True
    assert result.mean_pairwise_similarity == 1.0
    assert result.token_cv == 0.0


def test_stability_deterministic(cfg: Config, force_mock_llm: None) -> None:
    """Two stability runs are byte-identical (DX1) and JSON-dumpable."""
    a = temperature_zero_stability(cfg, n_repeats=3)
    b = temperature_zero_stability(cfg, n_repeats=3)
    assert a.model_dump() == b.model_dump()
    # JSON round-trips without error.
    assert StabilityResult.model_validate_json(a.model_dump_json()) == a


def test_stability_respects_n_repeats(cfg: Config, force_mock_llm: None) -> None:
    """The repeat count is echoed and the run still collapses to one output."""
    result = temperature_zero_stability(cfg, n_repeats=5)
    assert result.n_repeats == 5
    assert result.distinct_outputs == 1
    assert result.identical is True


def test_stability_custom_llm_with_variance(cfg: Config) -> None:
    """A noisy injected summarizer surfaces as >1 distinct output, not identical."""

    class _LLMResult:
        def __init__(self, text: str, output_tokens: int) -> None:
            self.text = text
            self.output_tokens = output_tokens

    calls = {"n": 0}

    def noisy_complete(_cfg, _prompt, system=None):  # noqa: ANN001
        calls["n"] += 1
        # Each call returns a different text and token count.
        return _LLMResult(text=f"summary variant {calls['n']}", output_tokens=10 + calls["n"])

    result = temperature_zero_stability(cfg, llm_complete=noisy_complete, n_repeats=3)
    assert result.n_repeats == 3
    assert result.distinct_outputs == 3
    assert result.identical is False
    assert result.token_cv > 0.0
    assert 0.0 <= result.mean_pairwise_similarity <= 1.0


# --------------------------------------------------------------------------- #
# Memory drift (EC8)
# --------------------------------------------------------------------------- #
def test_drift_well_formed(cfg: Config, force_mock_llm: None) -> None:
    """The drift detector returns a well-formed result on a tiny run."""
    result = memory_drift(cfg, n_rounds=3)

    assert isinstance(result, DriftResult)
    assert result.n_rounds == 3
    assert len(result.fidelity_per_round) == 3
    assert all(isinstance(f, float) for f in result.fidelity_per_round)
    # Cosine fidelities live in [-1, 1].
    assert all(-1.0 <= f <= 1.0 for f in result.fidelity_per_round)
    # faithfulness is the final round's fidelity.
    assert result.faithfulness == result.fidelity_per_round[-1]
    assert isinstance(result.degraded, bool)
    assert isinstance(result.monotonic_decline, bool)
    assert result.drift_threshold == pytest.approx(0.15)


def test_drift_deterministic(cfg: Config, force_mock_llm: None) -> None:
    """Two drift runs are byte-identical (DX1) and JSON-dumpable."""
    a = memory_drift(cfg, n_rounds=3)
    b = memory_drift(cfg, n_rounds=3)
    assert a.model_dump() == b.model_dump()
    assert DriftResult.model_validate_json(a.model_dump_json()) == a


def test_drift_custom_source_text(cfg: Config, force_mock_llm: None) -> None:
    """A caller-supplied source text is accepted and produces a clean result."""
    result = memory_drift(
        cfg, n_rounds=2, source_text="The thermostat failed at noon on Tuesday."
    )
    assert result.n_rounds == 2
    assert len(result.fidelity_per_round) == 2
