"""The dream engine for the Slow Wave bench (Phase 3).

The four independently-ablatable operators plus the optional conflict step, the
swappable decay registry, the shared result models, and the two-phase
(NREM->REM) :class:`~slow_wave.dream.engine.DreamEngine` that composes them. See
the per-module docstrings and ``docs/PHASE3_CONTRACT.md`` for details:

* :mod:`slow_wave.dream.schema` — the dream result models (per-operator results,
  :class:`~slow_wave.dream.schema.DreamCycleResult`,
  :class:`~slow_wave.dream.schema.DreamTelemetry`).
* :mod:`slow_wave.dream.decay` — the swappable exponential / Weibull / ACT-R
  decay registry (EC6).
* :mod:`slow_wave.dream.replay` — REPLAY (FR4.1): uniform / prioritized sampling.
* :mod:`slow_wave.dream.transfer` — TRANSFER (FR4.2): episodic->semantic
  consolidation with CLS interleaving (EC4).
* :mod:`slow_wave.dream.downscale` — DOWNSCALE (FR4.3): global decay +
  replay re-potentiation (EC2).
* :mod:`slow_wave.dream.augment` — GENERATIVE-AUGMENT (FR4.4): pseudo-episodes +
  generator-fidelity tracking (EC5).
* :mod:`slow_wave.dream.conflict` — the optional conflict/unlearning step (FR4.7).
* :mod:`slow_wave.dream.engine` — the :class:`~slow_wave.dream.engine.DreamEngine`
  two-phase cycle + gating + sleep-pressure (EC1, FR4.5/4.6).

:mod:`slow_wave.dream.runner` (the one-command end-to-end dream run that writes
``R[i,j]`` + wake/dream telemetry to a manifest) is intentionally *not*
re-exported here so ``python -m slow_wave.dream.runner`` does not double-import
the module (mirrors :mod:`slow_wave.agent.runner`).
"""

from __future__ import annotations

from slow_wave.dream.augment import augment
from slow_wave.dream.conflict import resolve_conflicts
from slow_wave.dream.decay import (
    DECAY_REGISTRY,
    act_r_decay,
    decay_factor,
    exponential_decay,
    get_decay,
    params_for,
    register_decay,
    weibull_decay,
)
from slow_wave.dream.downscale import downscale
from slow_wave.dream.engine import DreamEngine
from slow_wave.dream.replay import replay
from slow_wave.dream.schema import (
    AugmentResult,
    ConflictResult,
    DownscaleResult,
    DreamCycleResult,
    DreamPhase,
    DreamTelemetry,
    FidelityScore,
    ReplayResult,
    ReplaySample,
    TransferResult,
)
from slow_wave.dream.transfer import transfer

__all__ = [
    # schema
    "AugmentResult",
    "ConflictResult",
    "DownscaleResult",
    "DreamCycleResult",
    "DreamPhase",
    "DreamTelemetry",
    "FidelityScore",
    "ReplayResult",
    "ReplaySample",
    "TransferResult",
    # decay
    "DECAY_REGISTRY",
    "act_r_decay",
    "decay_factor",
    "exponential_decay",
    "get_decay",
    "params_for",
    "register_decay",
    "weibull_decay",
    # operators
    "augment",
    "downscale",
    "replay",
    "resolve_conflicts",
    "transfer",
    # engine
    "DreamEngine",
]
