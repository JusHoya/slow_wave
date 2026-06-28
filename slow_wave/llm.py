"""Claude completion helper for the Slow Wave bench (Phase 0).

A single entry point, :func:`complete`, returns an :class:`LLMResult` regardless
of environment:

* **Real path** — when ``ANTHROPIC_API_KEY`` is set, the ``anthropic`` SDK is
  imported lazily and ``client.messages.create(...)`` is called with the
  configured sampling parameters. If the network call raises for any reason, we
  *recover* by falling back to the deterministic mock (with ``mocked=True`` and a
  note in the text) so a smoke run never hard-fails on a flaky API.
* **Mock path** — when no key is present, a deterministic, dependency-free
  response is synthesized from a SHA-256 of the inputs. Token counts are derived
  from text lengths. This keeps Phase 0 CI green offline and bit-for-bit
  reproducible.

:func:`complete` **never raises on a missing key** and ``anthropic`` is imported
only inside the real path, so importing this module never requires the SDK.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """The result of a single completion.

    Attributes:
        text: The generated (or mocked) text.
        model_id: The model id used (echoes ``cfg.model.id``).
        input_tokens: Prompt token count (real usage, or a mock estimate).
        output_tokens: Completion token count (real usage, or a mock estimate).
        mocked: ``True`` if produced by the deterministic mock path (no API call,
            or a recovered API failure); ``False`` for a genuine API response.
        stop_reason: The provider stop reason, ``"mock"`` for the mock path, or
            ``"mock-fallback"`` when a real call failed and was recovered.
    """

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    mocked: bool
    stop_reason: str | None = None


def _mock_complete(cfg, prompt: str, system: str | None = None) -> LLMResult:
    """Produce a deterministic mock completion from a hash of the inputs.

    The text is fully determined by ``(cfg.model.id, system, prompt)`` so two
    calls with identical inputs are byte-identical. Token counts are derived
    deterministically from text lengths.
    """
    digest = hashlib.sha256(
        (cfg.model.id + "|" + (system or "") + "|" + prompt).encode("utf-8")
    ).hexdigest()
    text = (
        f"[MOCK:{digest[:8]}] Memory consolidation is the offline reorganization "
        f"of recent experience into durable, generalizable memory."
    )
    return LLMResult(
        text=text,
        model_id=cfg.model.id,
        input_tokens=max(1, len(prompt) // 4),
        output_tokens=max(1, len(text) // 4),
        mocked=True,
        stop_reason="mock",
    )


def complete(cfg, prompt: str, system: str | None = None) -> LLMResult:
    """Complete ``prompt`` with Claude, or a deterministic mock if no key.

    Behavior:
        * If ``ANTHROPIC_API_KEY`` is set, call the Anthropic Messages API with
          ``cfg.model.{id,max_tokens,temperature}`` (and ``top_p`` only when set,
          and ``system`` only when provided). On **any** exception during the
          call we recover by returning the deterministic mock with
          ``mocked=True`` and a note in the text, rather than propagating the
          error — so a smoke run never hard-fails on a transient API problem.
        * Otherwise, return a deterministic mock (``mocked=True``).

    This function never raises on a missing API key.

    Args:
        cfg: A :class:`slow_wave.config.Config`.
        prompt: The user prompt.
        system: Optional system prompt.

    Returns:
        An :class:`LLMResult`.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # lazy: only needed on the real path

            client = anthropic.Anthropic()
            kwargs: dict = {
                "model": cfg.model.id,
                "max_tokens": cfg.model.max_tokens,
                "temperature": cfg.model.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if cfg.model.top_p is not None:
                kwargs["top_p"] = cfg.model.top_p
            if system is not None:
                kwargs["system"] = system

            resp = client.messages.create(**kwargs)
            text = "".join(
                block.text for block in resp.content if block.type == "text"
            )
            return LLMResult(
                text=text,
                model_id=cfg.model.id,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                mocked=False,
                stop_reason=resp.stop_reason,
            )
        except Exception as exc:  # recover rather than hard-fail the smoke
            logger.warning(
                "Anthropic API call failed (%s: %s); falling back to "
                "deterministic mock.",
                type(exc).__name__,
                exc,
            )
            base = _mock_complete(cfg, prompt, system)
            note = f"[LLM-FALLBACK {type(exc).__name__}] "
            text = note + base.text
            return LLMResult(
                text=text,
                model_id=base.model_id,
                input_tokens=base.input_tokens,
                output_tokens=max(1, len(text) // 4),
                mocked=True,
                stop_reason="mock-fallback",
            )

    return _mock_complete(cfg, prompt, system)
