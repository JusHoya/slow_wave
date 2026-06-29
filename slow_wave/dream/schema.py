"""Shared data model for the Slow Wave dream engine (Phase 3).

This module is the **authoritative cross-module contract** for the dream tier
(see ``docs/PHASE3_CONTRACT.md``). The four operators (replay / transfer /
downscale / generative-augment), the optional conflict step, and the two-phase
engine all build against the types defined here and must not redefine them.

Design principles
-----------------
* **JSON-serializable, deterministic results.** Every operator returns a pydantic
  result model so the engine can roll them up into a
  :class:`DreamCycleResult` and the runner can serialize the whole dream-cycle
  log into a run manifest with stable key order (``model_dump(mode="json")``).
  All numeric fields are plain floats/ints; given a fixed ``(cfg, stream, seed)``
  under the mock LLM the results are byte-identical run-to-run (DX1).
* **Honesty by construction (DX2).** Result models carry explicit
  ``n_dropped`` / ``n_skipped`` / ``n_refused`` counters so anything an operator
  bounds (top-N replay, a per-cycle call ceiling, a refused protected overwrite)
  is *recorded*, never silently swallowed.
* **Confound-free (FR1.6).** No field here is a relevance label or any name in
  :data:`slow_wave.stream.guard.BANNED_FIELD_NAMES`; the dream operators only ever
  read/write confound-free :class:`~slow_wave.memory.schema.MemoryEntry` graphs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DreamPhase(str, Enum):
    """The two sleep phases of a dream cycle (the sequential hypothesis, FR4).

    ``NREM`` runs REPLAY -> TRANSFER -> DOWNSCALE (+ optional CONFLICT); ``REM``
    runs GENERATIVE-AUGMENT. Phases are independently ablatable (an empty phase is
    legal) and the engine records which operators actually ran.
    """

    NREM = "nrem"
    REM = "rem"


# --------------------------------------------------------------------------- #
# REPLAY (FR4.1)
# --------------------------------------------------------------------------- #
class ReplaySample(BaseModel):
    """One episodic entry sampled for replay, with its priority bookkeeping.

    Attributes:
        entry_id: The sampled episodic entry's id.
        priority: The (unnormalized) priority assigned to the entry. ``1.0`` for
            every entry under uniform sampling.
        probability: The normalized selection probability the sample was drawn
            with (priority / sum of priorities over the candidate pool).
        is_weight: The importance-sampling correction weight
            ``(1 / (N * probability)) ** beta`` normalized to ``<= 1`` (Schaul et
            al. 2016); ``1.0`` under uniform sampling. Logged, never silently
            dropped.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    priority: float
    probability: float
    is_weight: float


class ReplayResult(BaseModel):
    """The outcome of one REPLAY pass (FR4.1).

    Attributes:
        strategy: ``"uniform"`` or ``"prioritized"``.
        n_candidates: Size of the recent-episodic candidate pool considered.
        n_sampled: Number of entries actually sampled (``<= replay_sample_size``).
        n_dropped: Candidates not sampled because of the ``replay_sample_size``
            cap (DX2: the bounded coverage is recorded, never hidden).
        samples: The sampled entries with priority/IS-weight bookkeeping.
        sum_is_weight: Sum of the sampled importance-sampling weights (a sanity
            handle for later bias correction).
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str
    n_candidates: int
    n_sampled: int
    n_dropped: int
    samples: list[ReplaySample] = Field(default_factory=list)
    sum_is_weight: float = 0.0

    def sampled_ids(self) -> list[str]:
        """Return the entry ids of the sampled episodics (sampling order)."""
        return [s.entry_id for s in self.samples]


# --------------------------------------------------------------------------- #
# TRANSFER (FR4.2)
# --------------------------------------------------------------------------- #
class TransferResult(BaseModel):
    """The outcome of one TRANSFER (episodic -> semantic) pass (FR4.2).

    Attributes:
        n_batches: Number of distillation batches processed.
        n_consolidated: Source episodics consolidated into semantic entries.
        n_semantic_written: Semantic upserts that were applied.
        n_refused: Semantic upserts refused by FR2.5 write-protection (a protected
            fact would have been clobbered) — recorded, not hidden.
        interleaved: Whether CLS interleaving was enabled for this pass (EC4).
        n_interleaved_items: Prior consolidated memories mixed into the batches
            (``0`` when interleaving is off — the catastrophic-interference
            condition).
        written_entry_ids: The semantic entry ids written (provenance audit).
        api_calls: Claude "dream summarization" calls made.
        input_tokens: Cumulative prompt tokens across those calls.
        output_tokens: Cumulative completion tokens across those calls.
        n_skipped_calls: Distillation batches skipped by ``transfer_max_calls``
            (DX2).
    """

    model_config = ConfigDict(extra="forbid")

    n_batches: int = 0
    n_consolidated: int = 0
    n_semantic_written: int = 0
    n_refused: int = 0
    interleaved: bool = False
    n_interleaved_items: int = 0
    written_entry_ids: list[str] = Field(default_factory=list)
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    n_skipped_calls: int = 0


# --------------------------------------------------------------------------- #
# DOWNSCALE (FR4.3)
# --------------------------------------------------------------------------- #
class DownscaleResult(BaseModel):
    """The outcome of one DOWNSCALE pass (FR4.3): decay all, protect signal.

    Attributes:
        decay_function: The swappable decay curve applied (``exponential`` /
            ``weibull`` / ``act_r``; EC6).
        n_decayed: Entries whose salience was multiplied by the decay factor.
        n_repotentiated: Replayed entries that were re-potentiated after decay
            (their salience boosted and recency reset).
        mean_salience_before: Mean importance over decayed entries before the pass.
        mean_salience_after: Mean importance over the same entries after the pass.
    """

    model_config = ConfigDict(extra="forbid")

    decay_function: str
    n_decayed: int = 0
    n_repotentiated: int = 0
    mean_salience_before: float = 0.0
    mean_salience_after: float = 0.0


# --------------------------------------------------------------------------- #
# GENERATIVE-AUGMENT (FR4.4)
# --------------------------------------------------------------------------- #
class FidelityScore(BaseModel):
    """Generator fidelity / drift of a cycle's pseudo-episodes (EC5, FR5.6).

    Fidelity is the cosine similarity between a pseudo-episode's embedding and its
    source episodic's embedding, in ``[0, 1]`` (higher = more faithful); drift is
    ``1 - fidelity``. Tracking this per cycle is what lets the bench flag when
    repeated summarization *degrades* rather than distills memory.

    Attributes:
        n_pseudo: Number of pseudo-episodes scored.
        mean_fidelity: Mean fidelity over the pseudo-episodes (``0.0`` if none).
        min_fidelity: Minimum fidelity (the worst drift; ``0.0`` if none).
        mean_drift: Mean drift ``1 - mean_fidelity`` (``0.0`` if none).
    """

    model_config = ConfigDict(extra="forbid")

    n_pseudo: int = 0
    mean_fidelity: float = 0.0
    min_fidelity: float = 0.0
    mean_drift: float = 0.0


class AugmentResult(BaseModel):
    """The outcome of one GENERATIVE-AUGMENT pass (FR4.4, REM-like).

    Attributes:
        n_pseudo: Pseudo-episodes synthesized and written to the episodic tier.
        pseudo_entry_ids: The ids of the written pseudo-episodes (provenance
            audit; each points back to its source episodic).
        fidelity: The per-cycle generator-fidelity/drift score (EC5).
        api_calls: Claude generation calls made.
        input_tokens: Cumulative prompt tokens across those calls.
        output_tokens: Cumulative completion tokens across those calls.
    """

    model_config = ConfigDict(extra="forbid")

    n_pseudo: int = 0
    pseudo_entry_ids: list[str] = Field(default_factory=list)
    fidelity: FidelityScore = Field(default_factory=FidelityScore)
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# --------------------------------------------------------------------------- #
# CONFLICT / unlearning (FR4.7)
# --------------------------------------------------------------------------- #
class ConflictResult(BaseModel):
    """The outcome of one optional CONFLICT/unlearning pass (FR4.7).

    Attributes:
        n_conflicts_detected: Same-key consolidated entries with differing values.
        n_demoted: Entries demoted to the archival tier to resolve conflicts
            (demote, never hard-delete — Crick-Mitchison "reverse learning").
        demoted_entry_ids: Ids of the demoted entries (archival audit).
    """

    model_config = ConfigDict(extra="forbid")

    n_conflicts_detected: int = 0
    n_demoted: int = 0
    demoted_entry_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Cycle + run roll-ups
# --------------------------------------------------------------------------- #
class DreamCycleResult(BaseModel):
    """The complete log of a single two-phase dream cycle (FR4).

    Attributes:
        cycle_index: 0-based index of this cycle within the run.
        at_order: Stream order at which the cycle ran (the sleep window).
        task_index: Task segment after which this cycle ran.
        operators_run: Names of the operators that actually executed, in order
            (e.g. ``["replay", "transfer", "downscale", "augment"]``). An
            ablated operator is absent — the honest record of what ran (EC1).
        replay/transfer/downscale/augment/conflict: Per-operator results, or
            ``None`` if that operator was disabled this cycle.
        api_calls: Total Claude calls made across the cycle.
        input_tokens: Total prompt tokens across the cycle.
        output_tokens: Total completion tokens across the cycle.
    """

    model_config = ConfigDict(extra="forbid")

    cycle_index: int
    at_order: int
    task_index: int
    operators_run: list[str] = Field(default_factory=list)
    replay: ReplayResult | None = None
    transfer: TransferResult | None = None
    downscale: DownscaleResult | None = None
    augment: AugmentResult | None = None
    conflict: ConflictResult | None = None
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class DreamTelemetry(BaseModel):
    """Run-level roll-up of every dream cycle, for the manifest (EC7).

    Attributes:
        n_cycles: Number of dream cycles that ran over the whole run.
        cycles: The per-cycle results, in execution order.
        api_calls: Total Claude calls made by the dream engine across the run.
        input_tokens: Total prompt tokens across the run's dream cycles.
        output_tokens: Total completion tokens across the run's dream cycles.
        n_semantic_written: Total semantic upserts applied across the run.
        n_pseudo: Total pseudo-episodes synthesized across the run.
        n_demoted_conflict: Total entries demoted by the conflict step.
        n_protected_refusals: Total protected-overwrite refusals during transfer.
    """

    model_config = ConfigDict(extra="forbid")

    n_cycles: int = 0
    cycles: list[DreamCycleResult] = Field(default_factory=list)
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    n_semantic_written: int = 0
    n_pseudo: int = 0
    n_demoted_conflict: int = 0
    n_protected_refusals: int = 0

    @property
    def total_tokens(self) -> int:
        """Return total dream tokens (``input_tokens + output_tokens``)."""
        return self.input_tokens + self.output_tokens

    def record(self, cycle: DreamCycleResult) -> None:
        """Append ``cycle`` and fold its counters into the run-level totals.

        Args:
            cycle: The just-completed dream cycle to record.
        """
        self.cycles.append(cycle)
        self.n_cycles += 1
        self.api_calls += cycle.api_calls
        self.input_tokens += cycle.input_tokens
        self.output_tokens += cycle.output_tokens
        if cycle.transfer is not None:
            self.n_semantic_written += cycle.transfer.n_semantic_written
            self.n_protected_refusals += cycle.transfer.n_refused
        if cycle.augment is not None:
            self.n_pseudo += cycle.augment.n_pseudo
        if cycle.conflict is not None:
            self.n_demoted_conflict += cycle.conflict.n_demoted
