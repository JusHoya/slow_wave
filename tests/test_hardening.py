"""Phase 0 hardening tests (added during red-team verification).

These close gaps the adversarial review identified:

* **Golden reproducibility values** — the determinism tests elsewhere only check
  "run twice -> identical". A silent algorithm change (``derive_seed`` digest
  size, ``HashEmbedder`` math, ``embedding_sha256`` rounding) would stay
  internally consistent yet break cross-version reproducibility — the very
  property Phase 0 exists to protect. We freeze concrete golden constants.
* **End-to-end FR6.1 completeness on the *written* artifact** — the schema test
  exercises ``new_manifest`` with fakes; here we run the real ``run_smoke`` and
  walk every FR6.1 path on the JSON file it produced.
* **Nested-config typo rejection** — a typo'd key under ``model:``/``embedding:``
  must fail loudly, not be silently dropped.
* **Real-path LLM fallback** — when a real API call raises, ``complete`` must
  recover to a flagged mock rather than propagate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slow_wave.config import Config, load_config
from slow_wave.llm import complete
from slow_wave.repro.seeding import derive_seed
from slow_wave.repro.smoke import run_smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"

# FR6.1 dotted paths the *written* smoke manifest must contain (PRD §5.6).
FR61_REQUIRED_PATHS = [
    "model.id",
    "model.sampling.temperature",
    "model.sampling.max_tokens",
    "model.sampling.top_p",
    "model.sampling.effort",
    "embedding.model",
    "embedding.version",
    "embedding.dim",
    "hyperparameters",
    "search_ranges",
    "seed_list",
    "seeds",
    "git.commit",
    "cost.wall_clock_s",
    "cost.tokens.input",
    "cost.tokens.output",
    "cost.tokens.total",
    "cost.api_calls",
    "sim_time.compression_factor",
]


def _get_path(data: dict, dotted: str):
    cur = data
    for part in dotted.split("."):
        assert isinstance(cur, dict) and part in cur, f"missing path: {dotted}"
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- #
# Golden reproducibility constants                                            #
# --------------------------------------------------------------------------- #
def test_derive_seed_golden_values():
    """Frozen golden seeds — guards against a silent derive_seed change."""
    assert derive_seed(7, "stream") == 994352881
    assert derive_seed(7, "agent") == 1134892007
    # Range invariant.
    assert 0 <= derive_seed(7, "stream") < 2**32


def test_smoke_probe_golden_values(tmp_path, monkeypatch):
    """Frozen golden probe for configs/smoke.yaml (seed 7, hash backend, mock LLM).

    Locks the deterministic_probe so a change to HashEmbedder, the sampling RNG,
    or embedding_sha256 that stays internally consistent still trips this test.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(SMOKE_CONFIG)
    manifest_path = run_smoke(cfg, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe = manifest["deterministic_probe"]

    assert probe["n_items"] == 8
    assert probe["sampling_order"] == [1, 3, 6, 4, 2, 0, 7, 5]
    assert (
        probe["embedding_sha256"]
        == "36d3e937257be586d52c68e0a45072ea323b0f3ada89f94a4f7b4786ec7f97f2"
    )


# --------------------------------------------------------------------------- #
# End-to-end FR6.1 completeness on the produced artifact                      #
# --------------------------------------------------------------------------- #
def test_written_smoke_manifest_has_every_fr61_field(tmp_path, monkeypatch):
    """Run the real smoke and verify FR6.1 completeness on the JSON it wrote."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config(SMOKE_CONFIG)
    manifest_path = run_smoke(cfg, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for dotted in FR61_REQUIRED_PATHS:
        _get_path(manifest, dotted)  # asserts presence

    # Spot-check a few values are sane, not just present.
    assert manifest["model"]["id"] == "claude-opus-4-8"
    assert manifest["model"]["mocked"] is True  # no key in this run
    assert isinstance(manifest["seed_list"], list) and manifest["seed_list"]
    tokens = manifest["cost"]["tokens"]
    assert tokens["total"] == tokens["input"] + tokens["output"]
    # nondeterministic fields are flagged, incl. at least one llm.* path.
    nd = manifest["nondeterministic_fields"]
    assert "created_at" in nd and "cost.wall_clock_s" in nd
    assert any(f == "llm" or f.startswith("llm.") for f in nd)


# --------------------------------------------------------------------------- #
# Nested-config typo rejection                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        {"experiment": "x", "model": {"temperatur": 0.5}},
        {"experiment": "x", "embedding": {"backend": "hash", "dimm": 8}},
        {"experiment": "x", "sim_time": {"compresion_factor": 2.0}},
        {"experiment": "x", "smoke": {"n_item": 4}},
    ],
)
def test_nested_config_rejects_unknown_keys(bad):
    """A typo'd key under any nested block must fail loudly (extra='forbid')."""
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


# --------------------------------------------------------------------------- #
# Real-path LLM fallback on API error                                         #
# --------------------------------------------------------------------------- #
def test_complete_recovers_when_real_call_raises(monkeypatch):
    """With a key set, a raising Anthropic client must recover to a flagged mock."""
    import anthropic

    class _BoomClient:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated API outage")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", _BoomClient)

    cfg = Config(experiment="fallback-test")
    result = complete(cfg, "hello")

    assert result.mocked is True
    assert result.stop_reason == "mock-fallback"
    assert "[LLM-FALLBACK" in result.text
    assert result.model_id == cfg.model.id
    assert result.input_tokens > 0 and result.output_tokens > 0
