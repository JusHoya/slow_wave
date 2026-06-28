"""Tests for slow_wave.llm: deterministic mock path and real-path wiring.

The real-path test patches ``anthropic.Anthropic`` with a fake whose
``messages.create(**kwargs)`` records the request and returns a canned response,
so we verify the request is built correctly and the response parsed correctly
*without* a network call or a real API key.
"""

from __future__ import annotations

import pytest

from slow_wave.config import Config
from slow_wave.llm import LLMResult, complete


def _cfg(**model_overrides) -> Config:
    """Build a minimal Config, optionally overriding ModelConfig fields."""
    if model_overrides:
        return Config(experiment="llm-test", model=model_overrides)
    return Config(experiment="llm-test")


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str = "hi",
    input_tokens: int = 11,
    output_tokens: int = 3,
    stop_reason: str = "end_turn",
) -> dict:
    """Patch ``anthropic.Anthropic`` with a recording fake; return the recorder.

    Returns a dict that is populated with the kwargs passed to
    ``messages.create`` when ``complete`` runs.
    """
    import anthropic

    recorded: dict = {}

    class _Block:
        def __init__(self) -> None:
            self.type = "text"
            self.text = text

    class _Usage:
        def __init__(self) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class _Response:
        def __init__(self) -> None:
            self.content = [_Block()]
            self.usage = _Usage()
            self.stop_reason = stop_reason

    class _Messages:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return _Response()

    class _FakeAnthropic:
        def __init__(self, *args, **kwargs) -> None:
            self.messages = _Messages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    return recorded


# --- Mock path -------------------------------------------------------------


def test_mock_path_is_mocked_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no API key, complete returns a deterministic mock result."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _cfg()

    r1 = complete(cfg, "what is memory consolidation?")
    r2 = complete(cfg, "what is memory consolidation?")

    assert isinstance(r1, LLMResult)
    assert r1.mocked is True
    assert r1.stop_reason == "mock"
    assert r1.text == r2.text  # deterministic
    assert r1.text.startswith("[MOCK:")
    assert r1.input_tokens > 0
    assert r1.output_tokens > 0
    assert r1.model_id == cfg.model.id


def test_mock_path_is_prompt_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different prompts yield different deterministic mock text."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _cfg()

    assert complete(cfg, "prompt A").text != complete(cfg, "prompt B").text


# --- Real path (no network) ------------------------------------------------


def test_real_path_builds_request_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a key + fake SDK, complete builds the request and parses usage/text."""
    recorded = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = _cfg()

    result = complete(cfg, "hello", system="be terse")

    # Response parsed correctly.
    assert result.mocked is False
    assert result.text == "hi"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.stop_reason == "end_turn"
    assert result.model_id == cfg.model.id

    # Request built correctly.
    assert recorded["model"] == cfg.model.id
    assert recorded["max_tokens"] == cfg.model.max_tokens
    assert recorded["temperature"] == cfg.model.temperature
    assert recorded["messages"] == [{"role": "user", "content": "hello"}]
    assert recorded["system"] == "be terse"
    # top_p defaults to None and must be omitted entirely.
    assert "top_p" not in recorded


def test_real_path_includes_top_p_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """top_p is forwarded only when configured (not None)."""
    recorded = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = _cfg(top_p=0.9)

    complete(cfg, "hello")

    assert recorded["top_p"] == 0.9
    # system omitted when not provided.
    assert "system" not in recorded
