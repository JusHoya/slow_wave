"""Configuration models and loader for the Slow Wave bench (Phase 0).

This module pins the cross-module configuration contract (see
``docs/PHASE0_CONTRACT.md``). Other workstreams consume these field names
directly, so the schema here is authoritative and must not drift:

* WS3 (manifest) reads ``cfg.model.{id,temperature,max_tokens,top_p,effort}``,
  ``cfg.embedding.{backend,model,dim}``, ``cfg.sim_time.compression_factor``,
  ``cfg.hyperparameters``, ``cfg.search_ranges`` and ``cfg.content_hash()``.
* WS4 (embeddings/llm/smoke) reads ``cfg.smoke.{prompt,n_items,texts}``,
  ``cfg.seed`` and ``cfg.output_dir``.

The top-level :class:`Config` forbids unknown keys (``extra="forbid"``) so a
typo'd config field fails loudly rather than being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from slow_wave.stream.schema import StreamGenConfig


class ModelConfig(BaseModel):
    """LLM sampling configuration for agent reasoning and dream summarization."""

    model_config = ConfigDict(extra="forbid")

    id: str = "claude-opus-4-8"
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float | None = None
    effort: str | None = None  # adaptive-thinking effort knob, e.g. "low"/"medium"/"high"


class EmbeddingConfig(BaseModel):
    """Local embedding backend configuration for memory vectors.

    ``protected_namespaces=()`` is set because this model has a field literally
    named ``model``, which otherwise collides with pydantic's protected
    ``model_`` namespace and emits a spurious warning.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    backend: Literal["hash", "sentence-transformers"] = "hash"
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384


class SimTimeConfig(BaseModel):
    """Accelerated sim-time configuration for cheap wake/sleep iteration."""

    model_config = ConfigDict(extra="forbid")

    compression_factor: float = 1.0  # sim-time / wall-time


class SmokeConfig(BaseModel):
    """Parameters for the Phase 0 hello-bench smoke run."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = "In one sentence, state what memory consolidation is."
    n_items: int = 8
    texts: list[str] | None = None  # if None, smoke generates a deterministic corpus from n_items


class Config(BaseModel):
    """Top-level experiment configuration.

    Unknown top-level keys are rejected (``extra="forbid"``) so config typos are
    caught at load time. ``protected_namespaces=()`` silences the warning from
    the nested ``model`` field.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    experiment: str
    description: str = ""
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sim_time: SimTimeConfig = Field(default_factory=SimTimeConfig)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)
    stream: StreamGenConfig | None = None  # Phase 1: synthetic stream generation params
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    search_ranges: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "runs"

    def content_hash(self) -> str:
        """Return the sha256 hex digest of this config's canonical JSON.

        The digest is computed over ``self.model_dump(mode="json")`` serialized
        with sorted keys and no insignificant whitespace, so it is stable
        run-to-run and independent of field ordering in the source YAML.
        """

        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> Config:
    """Load and validate a :class:`Config` from a YAML file.

    Args:
        path: Path to a YAML config file (``str`` or :class:`~pathlib.Path`).

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML root is not a mapping (e.g. a list or scalar).
        pydantic.ValidationError: If the data does not satisfy the schema.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping at the top "
            f"level, got {type(data).__name__}."
        )

    return Config.model_validate(data)
