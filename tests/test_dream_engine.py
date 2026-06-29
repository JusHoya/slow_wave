"""Tests for the two-phase dream-cycle orchestrator (Phase 3, WS-ENGINE, EC1).

:class:`slow_wave.dream.engine.DreamEngine` composes the five landed operators
(REPLAY / TRANSFER / DOWNSCALE / CONFLICT / GENERATIVE-AUGMENT) into one
NREM->REM cycle, gates it to scheduled sleep windows, and rolls each cycle into a
run-level :class:`~slow_wave.dream.schema.DreamTelemetry`. The exit criteria
proven here:

* **EC1** — every one of the ``2^4 = 16`` on/off combinations of (replay,
  transfer, downscale, augment) instantiates and runs, and the returned
  :class:`~slow_wave.dream.schema.DreamCycleResult` reports ``operators_run``
  exactly equal to the enabled operator set (the all-off combo reports ``[]``).
  The 2x2 (replay x downscale) sub-matrix is exercised explicitly too.
* **Gating** — :meth:`DreamEngine.sleep_hook` returns ``None`` off-schedule and a
  cycle on-schedule; ``dream.enabled=False`` is always ``None``; ``adaptive``
  sleep pressure triggers a cycle on churn even when the fixed condition is unmet.
* **Determinism (DX1)** — two cycles on two freshly-built identical substrates
  (same cfg+seed, mock LLM) produce byte-identical
  :meth:`~pydantic.BaseModel.model_dump` payloads (sampled ids, written semantic
  ids, pseudo ids, salience-derived counts all identical).

All randomness flows through the deterministic hash embedder and the per-cycle
rng; the mock LLM is forced via the ``force_mock_llm`` fixture (no API key).
"""

from __future__ import annotations

import itertools

import pytest

from slow_wave.config import Config, DreamConfig
from slow_wave.dream.engine import DreamEngine
from slow_wave.dream.schema import DreamCycleResult
from slow_wave.embeddings import get_embedder
from slow_wave.llm import complete as llm_complete
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Fact

# A handful of fact-bearing episodics plus a couple of factless "noise" ones.
_FACTS = [
    ("alice", "role", "engineer"),
    ("bob", "city", "paris"),
    ("carol", "pet", "cat"),
    ("dave", "team", "blue"),
]
_NOISE = [
    "idle chatter about nothing in particular today",
    "the weather outside is mild and unremarkable",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_cfg(*, seed: int = 7, **dream_kwargs) -> Config:
    """Build a :class:`Config` with an ``enabled`` dream and the given toggles."""
    return Config(
        experiment="engine-test",
        seed=seed,
        dream=DreamConfig(enabled=True, **dream_kwargs),
    )


def _build_substrate(cfg: Config, embedder) -> MemorySubstrate:
    """Build a tiny substrate seeded with fact-bearing + noise episodics."""
    substrate = MemorySubstrate(cfg.memory, dim=embedder.dim)
    order = 0
    for subject, attribute, value in _FACTS:
        content = f"{subject}'s {attribute} is {value}."
        entry = MemoryEntry(
            entry_id=f"e{order:06d}",
            tier=MemoryTier.EPISODIC,
            content=content,
            fact=Fact(subject=subject, attribute=attribute, value=value),
            created_order=order,
            salience=SalienceMeta(importance=1.0, recency_order=order),
            provenance=(f"i{order:06d}",),
        )
        substrate.episodic.append(
            entry, embedder.encode([entry.content])[0], entry.created_order
        )
        order += 1
    for content in _NOISE:
        entry = MemoryEntry(
            entry_id=f"e{order:06d}",
            tier=MemoryTier.EPISODIC,
            content=content,
            fact=None,
            created_order=order,
            salience=SalienceMeta(importance=1.0, recency_order=order),
            provenance=(f"i{order:06d}",),
        )
        substrate.episodic.append(
            entry, embedder.encode([entry.content])[0], entry.created_order
        )
        order += 1
    return substrate


def _expected_ops(replay_en, transfer_en, downscale_en, augment_en) -> set[str]:
    """Return the operator-name set implied by the four toggles."""
    names = []
    if replay_en:
        names.append("replay")
    if transfer_en:
        names.append("transfer")
    if downscale_en:
        names.append("downscale")
    if augment_en:
        names.append("augment")
    return set(names)


# --------------------------------------------------------------------------- #
# EC1 — all 2^4 = 16 toggle combinations instantiate and run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "replay_en,transfer_en,downscale_en,augment_en",
    list(itertools.product([False, True], repeat=4)),
)
def test_ec1_all_sixteen_combinations_run(
    replay_en, transfer_en, downscale_en, augment_en, force_mock_llm
) -> None:
    """Each of the 16 combos runs and reports the exact enabled operator set."""
    cfg = _make_cfg(
        replay_enabled=replay_en,
        transfer_enabled=transfer_en,
        downscale_enabled=downscale_en,
        augment_enabled=augment_en,
    )
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    result = engine.run_cycle(
        substrate,
        embedder=embedder,
        llm_complete=llm_complete,
        now_order=10,
        task_index=0,
    )

    expected = _expected_ops(replay_en, transfer_en, downscale_en, augment_en)
    assert isinstance(result, DreamCycleResult)
    assert set(result.operators_run) == expected
    # operators_run preserves the canonical NREM->REM execution order.
    assert result.operators_run == [
        name
        for name in ("replay", "transfer", "downscale", "augment")
        if name in expected
    ]
    # Per-operator result objects are present iff their operator ran.
    assert (result.replay is not None) == replay_en
    assert (result.transfer is not None) == transfer_en
    assert (result.downscale is not None) == downscale_en
    assert (result.augment is not None) == augment_en
    # Conflict was not toggled on, so it never ran.
    assert result.conflict is None
    # The cycle was folded into telemetry exactly once.
    assert engine.telemetry.n_cycles == 1
    assert engine.telemetry.cycles[0] is result


def test_ec1_all_off_yields_empty_cycle(force_mock_llm) -> None:
    """The all-off combo is a legal empty cycle: operators_run == []."""
    cfg = _make_cfg(
        replay_enabled=False,
        transfer_enabled=False,
        downscale_enabled=False,
        augment_enabled=False,
    )
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    result = engine.run_cycle(
        substrate,
        embedder=embedder,
        llm_complete=llm_complete,
        now_order=10,
        task_index=0,
    )

    assert result.operators_run == []
    assert result.replay is result.transfer is result.downscale is None
    assert result.augment is result.conflict is None
    assert result.api_calls == 0
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.parametrize("replay_en", [False, True])
@pytest.mark.parametrize("downscale_en", [False, True])
def test_ec1_replay_x_downscale_sub_matrix(
    replay_en, downscale_en, force_mock_llm
) -> None:
    """Explicit 2x2 (replay x downscale) sub-matrix (transfer/augment off)."""
    cfg = _make_cfg(
        replay_enabled=replay_en,
        transfer_enabled=False,
        downscale_enabled=downscale_en,
        augment_enabled=False,
    )
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    result = engine.run_cycle(
        substrate,
        embedder=embedder,
        llm_complete=llm_complete,
        now_order=10,
        task_index=0,
    )

    assert set(result.operators_run) == _expected_ops(
        replay_en, False, downscale_en, False
    )
    assert (result.replay is not None) == replay_en
    assert (result.downscale is not None) == downscale_en


# --------------------------------------------------------------------------- #
# Gating + sleep-pressure (FR4.5 / FR4.6)
# --------------------------------------------------------------------------- #
def test_gating_fixed_schedule(force_mock_llm) -> None:
    """sleep_hook is None off the fixed schedule and a cycle when it divides."""
    cfg = _make_cfg(sleep_every_n_tasks=3)
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    def hook(now_order, task_index):
        return engine.sleep_hook(
            substrate,
            embedder=embedder,
            llm_complete=llm_complete,
            now_order=now_order,
            task_index=task_index,
        )

    # (task_index + 1) % 3 != 0 for task_index 0, 1.
    assert hook(now_order=5, task_index=0) is None
    assert hook(now_order=6, task_index=1) is None
    # (task_index + 1) % 3 == 0 for task_index 2 -> a cycle runs.
    result = hook(now_order=7, task_index=2)
    assert isinstance(result, DreamCycleResult)
    assert engine.telemetry.n_cycles == 1


def test_gating_disabled_is_always_none(force_mock_llm) -> None:
    """dream.enabled == False never sleeps, even when the schedule divides."""
    cfg = Config(
        experiment="engine-test",
        seed=7,
        dream=DreamConfig(enabled=False, sleep_every_n_tasks=1),
    )
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    # (0 + 1) % 1 == 0 would divide, but the master toggle is off.
    assert (
        engine.sleep_hook(
            substrate,
            embedder=embedder,
            llm_complete=llm_complete,
            now_order=5,
            task_index=0,
        )
        is None
    )
    assert engine.telemetry.n_cycles == 0


def test_gating_adaptive_triggers_on_churn(force_mock_llm) -> None:
    """Adaptive mode sleeps on churn >= threshold even off the fixed schedule."""
    # Fixed schedule would never fire within the run (every 100 tasks), so only
    # the churn threshold can trigger a cycle.
    cfg = _make_cfg(
        sleep_every_n_tasks=100,
        sleep_pressure_mode="adaptive",
        sleep_pressure_churn_threshold=3,
    )
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    def hook(now_order, task_index):
        return engine.sleep_hook(
            substrate,
            embedder=embedder,
            llm_complete=llm_complete,
            now_order=now_order,
            task_index=task_index,
        )

    # churn = now_order - last_cycle_order = 1 - (-1) = 2 < 3, fixed unmet -> None.
    assert hook(now_order=1, task_index=0) is None
    # churn = 5 - (-1) = 6 >= 3, fixed still unmet -> a cycle runs anyway.
    result = hook(now_order=5, task_index=1)
    assert isinstance(result, DreamCycleResult)
    assert engine.telemetry.n_cycles == 1


def test_gating_adaptive_threshold_zero_behaves_like_fixed(force_mock_llm) -> None:
    """Adaptive with threshold 0 only ever fires on the fixed schedule."""
    cfg = _make_cfg(
        sleep_every_n_tasks=2,
        sleep_pressure_mode="adaptive",
        sleep_pressure_churn_threshold=0,
    )
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    def hook(now_order, task_index):
        return engine.sleep_hook(
            substrate,
            embedder=embedder,
            llm_complete=llm_complete,
            now_order=now_order,
            task_index=task_index,
        )

    # Threshold 0 disables churn-triggering; only (task_index+1) % 2 == 0 fires.
    assert hook(now_order=10, task_index=0) is None
    assert isinstance(hook(now_order=20, task_index=1), DreamCycleResult)
    assert engine.telemetry.n_cycles == 1


# --------------------------------------------------------------------------- #
# Determinism (DX1)
# --------------------------------------------------------------------------- #
def test_run_cycle_is_deterministic(force_mock_llm) -> None:
    """Two cycles on identical fresh substrates produce identical results."""

    def run() -> dict:
        cfg = _make_cfg(conflict_enabled=True)  # all five operators active
        embedder = get_embedder(cfg)
        substrate = _build_substrate(cfg, embedder)
        engine = DreamEngine(cfg)
        result = engine.run_cycle(
            substrate,
            embedder=embedder,
            llm_complete=llm_complete,
            now_order=10,
            task_index=0,
        )
        return result.model_dump()

    first = run()
    second = run()
    assert first == second
    # Sanity: the deterministic payload actually carries the operator results.
    assert first["operators_run"] == [
        "replay",
        "transfer",
        "downscale",
        "conflict",
        "augment",
    ]
    assert first["transfer"]["written_entry_ids"]
    assert first["augment"]["pseudo_entry_ids"]


def test_telemetry_accumulates_across_cycles(force_mock_llm) -> None:
    """Successive run_cycle calls fold their cost into the run-level telemetry."""
    cfg = _make_cfg()
    embedder = get_embedder(cfg)
    substrate = _build_substrate(cfg, embedder)
    engine = DreamEngine(cfg)

    c0 = engine.run_cycle(
        substrate,
        embedder=embedder,
        llm_complete=llm_complete,
        now_order=10,
        task_index=0,
    )
    c1 = engine.run_cycle(
        substrate,
        embedder=embedder,
        llm_complete=llm_complete,
        now_order=20,
        task_index=1,
    )

    assert engine.telemetry.n_cycles == 2
    assert c0.cycle_index == 0
    assert c1.cycle_index == 1
    assert engine.telemetry.api_calls == c0.api_calls + c1.api_calls
    assert engine.telemetry.input_tokens == c0.input_tokens + c1.input_tokens
    assert engine.telemetry.output_tokens == c0.output_tokens + c1.output_tokens
