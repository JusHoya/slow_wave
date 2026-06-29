"""The nine control arms: registry + builder (Phase 4, WS-ARMS, FR5.1/DX6).

This module turns the authoritative toggle table in ``docs/PHASE4_CONTRACT.md``
("The nine arms") into runnable objects. An arm is a declarative
:class:`slow_wave.eval.schema.ArmSpec` (name + ``config_overrides`` + label
permission + family); :func:`build_arm` materializes one by deep-merging its
overrides onto a copy of the base :class:`slow_wave.config.Config`, stamping the
run seed, and attaching the right sleep-window machinery:

* **dream-driven arms** (``replay_only``, ``downscale_only``, ``full_dream``,
  ``reflection``) attach :meth:`slow_wave.dream.engine.DreamEngine.sleep_hook`;
* **no-sleep arms** (``no_sleep``, ``long_context``) attach no hook;
* **custom-hook controls** (``random_pruning``, ``oracle``) attach the operators
  from :mod:`slow_wave.eval.arm_ops`;
* **``aa``** builds the ``aa_reference_arm`` (default ``no_sleep``) verbatim.

Adding a new arm is implementing/registering an :class:`ArmSpec` — no harness
edits (DX6).

Design principles
-----------------
* **No base mutation.** :func:`build_arm` deep-merges via ``model_dump`` ->
  dict deep-merge -> ``Config.model_validate``, so the base config is never
  touched and the effective config is re-validated (``extra="forbid"``).
* **Confound guard (FR1.6).** Exactly one spec — ``oracle`` — has
  ``uses_labels=True``; :func:`build_arm` asserts that invariant for every arm,
  and only the oracle hook reads :func:`offline_labels`.
* **Determinism (DX1).** The effective config carries ``seed`` so the dream RNG
  and the random-prune RNG are reproducible; same seed -> identical sampled /
  demoted ids under the mock LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from slow_wave.config import Config
from slow_wave.dream.engine import DreamEngine
from slow_wave.eval.arm_ops import (
    PruneTelemetry,
    make_oracle_prune_hook,
    make_random_prune_hook,
)
from slow_wave.eval.schema import ArmSpec
from slow_wave.stream.schema import Stream

__all__ = ["ARM_REGISTRY", "RunnableArm", "build_arm"]


# --------------------------------------------------------------------------- #
# The nine arms — authoritative toggle mapping (see PHASE4_CONTRACT.md).
# --------------------------------------------------------------------------- #
#: The control battery, keyed by canonical arm name. Each ``config_overrides``
#: fully specifies the toggles the arm cares about, so the effective config is
#: correct regardless of the base config's dream defaults. Only ``oracle`` reads
#: labels.
ARM_REGISTRY: dict[str, ArmSpec] = {
    "no_sleep": ArmSpec(
        name="no_sleep",
        description="Wake-only baseline; no dream cycle (the forgetting reference).",
        config_overrides={"dream": {"enabled": False}},
        uses_labels=False,
        family="baseline",
    ),
    "replay_only": ArmSpec(
        name="replay_only",
        description=(
            "Mere re-exposure: replay + downscale re-potentiate the sampled "
            "episodics so they survive eviction; no transfer/augment."
        ),
        config_overrides={
            "dream": {
                "enabled": True,
                "replay_enabled": True,
                "transfer_enabled": False,
                "downscale_enabled": True,
                "augment_enabled": False,
                "conflict_enabled": False,
            }
        },
        uses_labels=False,
        family="ablation",
    ),
    "downscale_only": ArmSpec(
        name="downscale_only",
        description="Pure SHY homeostasis: global decay, re-potentiate nothing.",
        config_overrides={
            "dream": {
                "enabled": True,
                "replay_enabled": False,
                "transfer_enabled": False,
                "downscale_enabled": True,
                "augment_enabled": False,
                "conflict_enabled": False,
            }
        },
        uses_labels=False,
        family="ablation",
    ),
    "random_pruning": ArmSpec(
        name="random_pruning",
        description=(
            "Ground-truth-blind negative control: each sleep window demotes a "
            "random fraction of active entries (custom prune hook)."
        ),
        config_overrides={
            "dream": {
                "enabled": True,
                "replay_enabled": False,
                "transfer_enabled": False,
                "downscale_enabled": False,
                "augment_enabled": False,
                "conflict_enabled": False,
            }
        },
        uses_labels=False,
        family="control",
    ),
    "full_dream": ArmSpec(
        name="full_dream",
        description="The treatment: replay + transfer + downscale + augment all on.",
        config_overrides={
            "dream": {
                "enabled": True,
                "replay_enabled": True,
                "transfer_enabled": True,
                "downscale_enabled": True,
                "augment_enabled": True,
            }
        },
        uses_labels=False,
        family="treatment",
    ),
    "reflection": ArmSpec(
        name="reflection",
        description=(
            "Generative-Agents shallow synthesis: uniformly sample recent, "
            "summarize into semantic; no homeostasis/interleave/REM."
        ),
        config_overrides={
            "dream": {
                "enabled": True,
                "replay_enabled": True,
                "replay_strategy": "uniform",
                "transfer_enabled": True,
                "cls_interleave": False,
                "downscale_enabled": False,
                "augment_enabled": False,
                "conflict_enabled": False,
            }
        },
        uses_labels=False,
        family="control",
    ),
    "oracle": ArmSpec(
        name="oracle",
        description=(
            "Prune-quality ceiling: each sleep window demotes exactly the active "
            "distractor/noise episodics (reads ground-truth labels)."
        ),
        config_overrides={
            "dream": {
                "enabled": True,
                "replay_enabled": False,
                "transfer_enabled": False,
                "downscale_enabled": False,
                "augment_enabled": False,
                "conflict_enabled": False,
            }
        },
        uses_labels=True,
        family="ceiling",
    ),
    "long_context": ArmSpec(
        name="long_context",
        description=(
            "Unbounded-memory ceiling: dream disabled, episodic capacity 0 "
            "(never forgets; the whole stream stays in context)."
        ),
        config_overrides={
            "dream": {"enabled": False},
            "memory": {"episodic_capacity": 0},
        },
        uses_labels=False,
        family="ceiling",
    ),
    "aa": ArmSpec(
        name="aa",
        description=(
            "A/A noise-floor control: builds the aa_reference_arm (default "
            "no_sleep) verbatim; the harness runs it under two seeds."
        ),
        config_overrides={"dream": {"enabled": False}},
        uses_labels=False,
        family="control",
    ),
}

#: Arms whose sleep window is a custom (non-DreamEngine) prune operator.
_CUSTOM_HOOK_ARMS = frozenset({"random_pruning", "oracle"})


@dataclass
class RunnableArm:
    """A materialized arm ready to hand to :class:`slow_wave.agent.wake.WakeAgent`.

    Attributes:
        spec: The declarative :class:`ArmSpec` this arm was built from.
        cfg: The effective :class:`slow_wave.config.Config` (overrides applied,
            ``seed`` stamped; re-validated).
        sleep_hook: The wake-agent sleep-window callback, or ``None`` for the
            no-sleep / long-context arms.
        engine: The :class:`DreamEngine` backing a dream-driven arm (its
            telemetry is read after the run), or ``None`` for custom-hook arms.
        prune_telemetry: The :class:`PruneTelemetry` backing a custom-hook arm
            (``random_pruning`` / ``oracle``), or ``None`` otherwise.
    """

    spec: ArmSpec
    cfg: Config
    sleep_hook: Callable | None
    engine: DreamEngine | None = None
    prune_telemetry: PruneTelemetry | None = None

    def n_cycles(self) -> int:
        """Return the number of dream cycles (or prune cycles) the arm ran.

        Returns:
            ``engine.telemetry.n_cycles`` for a dream-driven arm,
            ``prune_telemetry.n_cycles`` for a custom-hook arm, else ``0``
            (no-sleep / long-context). Valid only after the wake run that uses
            :attr:`sleep_hook`.
        """
        if self.engine is not None:
            return self.engine.telemetry.n_cycles
        if self.prune_telemetry is not None:
            return self.prune_telemetry.n_cycles
        return 0

    def dream_tokens(self) -> tuple[int, int, int]:
        """Return ``(input_tokens, output_tokens, api_calls)`` spent by the engine.

        Returns:
            The dream engine's cumulative ``(input_tokens, output_tokens,
            api_calls)``, or ``(0, 0, 0)`` for custom-hook / no-dream arms (which
            never call the LLM in their sleep window).
        """
        if self.engine is not None:
            tel = self.engine.telemetry
            return (tel.input_tokens, tel.output_tokens, tel.api_calls)
        return (0, 0, 0)

    def generator_fidelity(self) -> float | None:
        """Return the mean generative-augment fidelity over the run, or ``None``.

        There is no single roll-up fidelity field, so this averages the per-cycle
        ``cycle.augment.fidelity.mean_fidelity`` over the cycles where augment
        actually produced pseudo-episodes (``augment is not None`` and
        ``n_pseudo > 0``). Returns ``None`` when no such cycle ran (augment off,
        or a non-dream arm).

        Returns:
            The mean augment fidelity in ``[0, 1]``, or ``None`` if augment never
            synthesized a pseudo-episode.
        """
        if self.engine is None:
            return None
        fidelities = [
            cycle.augment.fidelity.mean_fidelity
            for cycle in self.engine.telemetry.cycles
            if cycle.augment is not None and cycle.augment.n_pseudo > 0
        ]
        if not fidelities:
            return None
        return float(np.mean(fidelities))


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Recursively merge ``overrides`` into ``base`` in place (dicts merge).

    Nested ``dict`` values are merged key-by-key; any non-dict value (or a value
    whose base counterpart is not a dict) replaces the base value outright. Only
    ``base`` is mutated; ``overrides`` is read but never modified.

    Args:
        base: The mapping mutated in place (a fresh ``model_dump`` of the config).
        overrides: The override mapping deep-merged onto ``base``.
    """
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value


def _materialize_config(
    base_cfg: Config, overrides: dict[str, Any], seed: int
) -> Config:
    """Deep-merge ``overrides`` onto a copy of ``base_cfg`` and stamp ``seed``.

    Does not mutate ``base_cfg``: it works on a fresh ``base_cfg.model_dump()``
    dict, deep-merges the overrides into it, sets the top-level ``seed``, and
    re-validates through :meth:`Config.model_validate` so ``extra="forbid"``
    catches any typo'd override key.

    Args:
        base_cfg: The base configuration (never mutated).
        overrides: The arm's nested ``config_overrides``.
        seed: The run seed stamped onto the effective config.

    Returns:
        A freshly validated effective :class:`Config`.
    """
    merged = base_cfg.model_dump()
    _deep_merge(merged, overrides)
    merged["seed"] = seed
    return Config.model_validate(merged)


def build_arm(
    name: str, base_cfg: Config, stream: Stream, seed: int
) -> RunnableArm:
    """Materialize the named control arm into a :class:`RunnableArm` (FR5.1/DX6).

    Deep-merges ``ARM_REGISTRY[name].config_overrides`` onto a copy of
    ``base_cfg`` (``base_cfg`` is never mutated), stamps ``effective.seed =
    seed``, and attaches the arm's sleep machinery:

    * dream-driven arms: a :class:`DreamEngine` over the effective config; the
      sleep hook is :meth:`DreamEngine.sleep_hook` when ``dream.enabled`` else
      ``None``. ``n_cycles`` / ``dream_tokens`` / ``generator_fidelity`` read the
      engine's telemetry after the run.
    * ``random_pruning``: a :class:`PruneTelemetry` + :func:`make_random_prune_hook`;
      ``dream_tokens`` is ``(0, 0, 0)``.
    * ``oracle``: a :class:`PruneTelemetry` + :func:`make_oracle_prune_hook`
      (which reads labels — the one sanctioned use); asserts ``uses_labels``.
    * ``long_context`` / ``no_sleep``: dream disabled, so ``sleep_hook`` is
      ``None``.
    * ``aa``: builds ``base_cfg.eval.aa_reference_arm`` (default ``no_sleep``)
      verbatim and re-wraps it under the ``aa`` spec.

    Args:
        name: The arm's registry name.
        base_cfg: The base configuration to merge onto (never mutated).
        stream: The stream the arm will run on (passed to the oracle hook so it
            can read that stream's ground-truth labels).
        seed: The run seed stamped onto the effective config.

    Returns:
        A :class:`RunnableArm` whose ``cfg`` validates and whose ``sleep_hook``
        is ready for :meth:`slow_wave.agent.wake.WakeAgent.run`.

    Raises:
        KeyError: If ``name`` is not a registered arm.
        AssertionError: If the spec's ``uses_labels`` flag does not match the
            invariant (``True`` only for ``oracle``).
    """
    if name not in ARM_REGISTRY:
        raise KeyError(f"unknown arm {name!r}; known arms: {sorted(ARM_REGISTRY)}")
    spec = ARM_REGISTRY[name]

    # Confound guard (FR1.6): only the oracle arm may read ground-truth labels.
    assert spec.uses_labels == (name == "oracle"), (
        f"confound guard: arm {name!r} has uses_labels={spec.uses_labels}; "
        f"only 'oracle' may read labels"
    )

    # The A/A control runs the reference arm verbatim (re-wrapped under aa spec).
    if name == "aa":
        ref_name = base_cfg.eval.aa_reference_arm
        ref = build_arm(ref_name, base_cfg, stream, seed)
        return RunnableArm(
            spec=spec,
            cfg=ref.cfg,
            sleep_hook=ref.sleep_hook,
            engine=ref.engine,
            prune_telemetry=ref.prune_telemetry,
        )

    effective = _materialize_config(base_cfg, spec.config_overrides, seed)

    if name == "random_pruning":
        telemetry = PruneTelemetry()
        hook = make_random_prune_hook(effective, seed=seed, telemetry=telemetry)
        return RunnableArm(
            spec=spec, cfg=effective, sleep_hook=hook, prune_telemetry=telemetry
        )

    if name == "oracle":
        assert spec.uses_labels  # belt-and-braces: the one label-reading arm.
        telemetry = PruneTelemetry()
        hook = make_oracle_prune_hook(effective, stream, telemetry=telemetry)
        return RunnableArm(
            spec=spec, cfg=effective, sleep_hook=hook, prune_telemetry=telemetry
        )

    # Dream-family arms (including dream-disabled no_sleep / long_context): build
    # an engine and use its sleep hook only when the dream cycle is enabled.
    engine = DreamEngine(effective)
    hook = engine.sleep_hook if effective.dream.enabled else None
    return RunnableArm(spec=spec, cfg=effective, sleep_hook=hook, engine=engine)
