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


class MemoryConfig(BaseModel):
    """Phase 2 memory-substrate configuration (FR2.x).

    Controls the dual-store memory substrate the wake agent writes to and reads
    from: episodic-buffer capacity (and thus eviction/forgetting pressure), the
    baseline retrieval policy and its recency/importance/relevance weights, and
    the recency decay used by both salience and eviction. ``extra="forbid"`` so a
    typo'd knob fails loudly at load time.
    """

    model_config = ConfigDict(extra="forbid")

    episodic_capacity: int = Field(default=0, ge=0)
    """Max live entries in the EPISODIC store; ``0`` means unbounded (no
    eviction). A finite capacity below the stream's signal count is what makes
    the no-sleep baseline forget (Phase 2 exit criterion #6)."""

    archival_enabled: bool = True
    """When ``True`` (default), evicted entries are *demoted* to the auditable
    archival tier rather than discarded (FR2.4)."""

    retrieval_policy: str = "recency_importance_relevance"
    """Name of the registered retrieval policy (FR2.3; pluggable via the
    ``slow_wave.memory.retrieval`` registry)."""

    retrieval_top_k: int = Field(default=8, ge=1)
    """Number of entries the retrieval policy returns per query."""

    recency_half_life: float = Field(default=64.0, gt=0.0)
    """Half-life (in stream-item units) of the recency decay applied to salience
    and eviction priority."""

    weight_recency: float = Field(default=1.0, ge=0.0)
    weight_importance: float = Field(default=1.0, ge=0.0)
    weight_relevance: float = Field(default=1.0, ge=0.0)
    """Multiplicative weights for the recency × importance × relevance baseline
    retrieval score (Park et al. 2023 memory-stream policy)."""

    base_salience: float = Field(default=1.0, gt=0.0)
    """Initial importance assigned to a freshly ingested episodic entry."""

    novelty_enabled: bool = True
    """Whether to compute a novelty term (embedding distance to the consolidated
    semantic store) into salience (FR2.2)."""


class AgentConfig(BaseModel):
    """Phase 2 wake-agent configuration (FR3.x).

    Controls the no-sleep wake loop: the optional token-budget ceiling the agent
    self-moderates against (and the harness enforces), whether a Claude reasoning
    call is made per task segment, and the top-k used when answering held-out
    probes. ``extra="forbid"`` so a typo'd knob fails loudly at load time.
    """

    model_config = ConfigDict(extra="forbid")

    token_budget: int | None = None
    """Ceiling on total (input+output) LLM tokens for the whole run; ``None``
    means unbounded. Reasoning calls that would exceed it are skipped and logged
    (FR3.3, DX2)."""

    reasoning_calls: Literal["off", "per_task"] = "per_task"
    """Whether the wake loop makes a (mock-by-default) Claude reasoning call once
    per task segment (``"per_task"``) or never (``"off"``). Reasoning never
    writes the semantic store — writes are gated to sleep (FR3.1)."""

    reasoning_prompt: str = (
        "Review the recent observations and note any facts worth remembering."
    )
    """Prompt used for the per-task reasoning call (telemetry/realism only; it
    does not determine probe answers, which are read deterministically from
    memory by exact-key lookup over the active stores)."""


class DreamConfig(BaseModel):
    """Phase 3 dream-engine configuration (FR4.x).

    Controls the offline, two-phase (NREM-like -> REM-like) "dream" cycle that
    consolidates wake experience. Every one of the four operators is an
    independent on/off toggle (Phase 3 exit criterion #1, the 2^4 ablation
    matrix), so the engine instantiates and runs for any subset; ``conflict`` is
    the optional fifth operator (FR4.7). ``extra="forbid"`` so a typo'd knob
    fails loudly at load time.

    The dream cycle is **gated to scheduled sleep windows** (FR4.5): semantic
    writes occur *only* inside a cycle, never during wake ingest. With
    :attr:`enabled` ``False`` (the default) the engine never runs and the agent is
    byte-identical to the Phase 2 no-sleep baseline.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    """Master toggle. ``False`` (default) => no dream cycle ever runs (the Phase 2
    no-sleep baseline). ``True`` => the engine runs at scheduled sleep windows."""

    # -- Operator toggles (EC1: the 2^4 replay x transfer x downscale x augment
    #    matrix; conflict is the optional FR4.7 fifth operator). ------------- #
    replay_enabled: bool = True
    """REPLAY operator (FR4.1): re-sample recent episodics for the cycle."""
    transfer_enabled: bool = True
    """TRANSFER operator (FR4.2): consolidate sampled episodics into semantic."""
    downscale_enabled: bool = True
    """DOWNSCALE operator (FR4.3): global salience decay + replay re-potentiation."""
    augment_enabled: bool = True
    """GENERATIVE-AUGMENT operator (FR4.4, REM-like): synthesize pseudo-episodes."""
    conflict_enabled: bool = False
    """Optional conflict/unlearning step (FR4.7): demote contradicting entries."""

    # -- Scheduling & gating (FR4.5) / sleep-pressure controller (FR4.6) ----- #
    sleep_every_n_tasks: int = Field(default=1, ge=1)
    """Run a dream cycle at the end of every ``N`` task segments (fixed schedule;
    FR4.5). The sleep window is the only place semantic writes happen."""
    sleep_pressure_mode: Literal["fixed", "adaptive"] = "fixed"
    """``"fixed"`` => cycle every :attr:`sleep_every_n_tasks` tasks. ``"adaptive"``
    => additionally trigger a cycle once accumulated wake "memory churn" (items
    ingested since the last cycle) exceeds :attr:`sleep_pressure_churn_threshold`
    (SWA-homeostasis analogue, FR4.6)."""
    sleep_pressure_churn_threshold: int = Field(default=0, ge=0)
    """Churn (items since last cycle) that triggers an adaptive cycle; ``0`` =>
    disabled (adaptive mode then behaves like fixed). Ignored when
    :attr:`sleep_pressure_mode` is ``"fixed"``."""

    # -- REPLAY (FR4.1) ----------------------------------------------------- #
    replay_sample_size: int = Field(default=16, ge=0)
    """Number of recent episodics sampled per cycle (the "ripple" batch). When the
    candidate pool exceeds this, the surplus is logged as dropped (DX2)."""
    replay_strategy: Literal["uniform", "prioritized"] = "prioritized"
    """``"uniform"`` (DQN baseline; Mnih et al. 2015) or ``"prioritized"``
    (recency x relevance x novelty x surprise; Schaul et al. 2016, IS weights
    logged)."""
    replay_priority_alpha: float = Field(default=1.0, ge=0.0)
    """Prioritization exponent (0 => uniform even under ``"prioritized"``)."""
    replay_priority_eps: float = Field(default=1e-6, gt=0.0)
    """Small constant added to every priority so no entry has zero sample mass."""

    # -- TRANSFER (FR4.2) --------------------------------------------------- #
    transfer_batch_size: int = Field(default=8, ge=1)
    """Episodics distilled per Claude "dream summarization" batch."""
    cls_interleave: bool = True
    """Enforce CLS interleaving (FR4.2/EC4): each batch mixes new episodics with
    sampled prior consolidated memories. ``False`` is the on-purpose
    catastrophic-interference condition (no interleaving)."""
    cls_interleave_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    """Target fraction of each transfer batch drawn from prior consolidated
    (semantic) memory when :attr:`cls_interleave` is on."""
    transfer_max_calls: int | None = None
    """Optional ceiling on Claude calls per cycle for TRANSFER; ``None`` =>
    unbounded. Any batch skipped by the ceiling is logged (DX2)."""

    # -- DOWNSCALE (FR4.3) + swappable decay (EC6) -------------------------- #
    decay_function: Literal["exponential", "weibull", "act_r"] = "exponential"
    """Which swappable decay curve DOWNSCALE applies (EC6)."""
    repotentiate_boost: float = Field(default=1.5, ge=1.0)
    """Multiplicative boost applied to a replayed item's salience after global
    decay ("protect signal"); ``>= 1`` so a replayed item ends strictly above an
    identical non-replayed one (EC2)."""
    decay_exponential_rate: float = Field(default=0.1, gt=0.0)
    """``exponential`` decay rate: factor = ``exp(-rate * age)``."""
    decay_weibull_scale: float = Field(default=10.0, gt=0.0)
    decay_weibull_k: float = Field(default=1.5, gt=0.0)
    """``weibull`` decay: factor = ``exp(-(age/scale)**k)``."""
    decay_act_r: float = Field(default=0.5, gt=0.0)
    """``act_r`` base-level decay: factor = ``(1 + age) ** (-d)``."""

    # -- GENERATIVE-AUGMENT (FR4.4) ----------------------------------------- #
    augment_per_cycle: int = Field(default=4, ge=0)
    """Number of pseudo-episodes synthesized per cycle (REM-like)."""
    augment_kinds: list[str] = Field(
        default_factory=lambda: ["paraphrase", "abstraction", "counterfactual"]
    )
    """Kinds of pseudo-episode the generator cycles through."""

    # -- CONFLICT / unlearning (FR4.7) -------------------------------------- #
    conflict_demote_strategy: Literal["older", "lower_salience"] = "older"
    """When two consolidated entries assert the same key with different values,
    which one to demote (the ``"older"`` ``created_order``, or the
    ``"lower_salience"`` one); the survivor is never hard-deleted (FR4.7)."""


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
    memory: MemoryConfig = Field(default_factory=MemoryConfig)  # Phase 2: memory substrate
    agent: AgentConfig = Field(default_factory=AgentConfig)  # Phase 2: wake agent
    dream: DreamConfig = Field(default_factory=DreamConfig)  # Phase 3: dream engine
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
