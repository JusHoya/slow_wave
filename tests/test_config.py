"""Tests for slow_wave.config: model schema, YAML loading, and content_hash."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from slow_wave.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
SMOKE_YAML = CONFIGS_DIR / "smoke.yaml"
DEFAULT_YAML = CONFIGS_DIR / "default.yaml"


def test_load_smoke_config() -> None:
    """configs/smoke.yaml loads into a valid Config with the expected values."""
    cfg = load_config(SMOKE_YAML)

    assert isinstance(cfg, Config)
    assert cfg.experiment == "hello-bench-smoke"
    assert cfg.seed == 7
    assert cfg.embedding.backend == "hash"
    assert cfg.embedding.model == "hash-bow-v1"
    assert cfg.embedding.dim == 384
    assert cfg.model.id == "claude-opus-4-8"
    assert cfg.model.temperature == 0.0
    assert cfg.model.max_tokens == 64
    assert cfg.sim_time.compression_factor == 60.0
    assert cfg.smoke.n_items == 8


def test_load_default_config() -> None:
    """configs/default.yaml loads and uses the sentence-transformers backend."""
    cfg = load_config(DEFAULT_YAML)

    assert isinstance(cfg, Config)
    assert cfg.embedding.backend == "sentence-transformers"
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.dim == 384
    assert cfg.model.id == "claude-opus-4-8"
    assert cfg.model.max_tokens == 256


def test_content_hash_is_deterministic() -> None:
    """The same Config produces the same content_hash across instances."""
    cfg_a = load_config(SMOKE_YAML)
    cfg_b = load_config(SMOKE_YAML)

    digest = cfg_a.content_hash()
    assert isinstance(digest, str)
    assert len(digest) == 64  # sha256 hex
    assert digest == cfg_b.content_hash()


def test_content_hash_changes_when_field_changes() -> None:
    """Mutating any field changes the content_hash."""
    cfg = load_config(SMOKE_YAML)
    baseline = cfg.content_hash()

    changed = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert changed.content_hash() != baseline

    # A nested change is also reflected (model_dump is recursive).
    new_model = cfg.model.model_copy(update={"max_tokens": cfg.model.max_tokens + 1})
    changed_nested = cfg.model_copy(update={"model": new_model})
    assert changed_nested.content_hash() != baseline


def test_round_trip_model_dump() -> None:
    """Config.model_validate(cfg.model_dump()) reconstructs an equal Config."""
    cfg = load_config(SMOKE_YAML)
    round_tripped = Config.model_validate(cfg.model_dump())

    assert round_tripped == cfg
    assert round_tripped.content_hash() == cfg.content_hash()


def test_unknown_top_level_key_rejected() -> None:
    """extra='forbid' makes an unknown top-level key raise ValidationError."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"experiment": "x", "not_a_real_field": True}
        )


def test_missing_config_path_raises_clear_error() -> None:
    """A missing config path raises FileNotFoundError mentioning the path."""
    missing = CONFIGS_DIR / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)
