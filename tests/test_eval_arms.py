"""Tests for the nine control arms (Phase 4, WS-ARMS, EC1 + EC3 support).

:mod:`slow_wave.eval.arms` turns the authoritative toggle table in
``docs/PHASE4_CONTRACT.md`` into runnable arms, and
:mod:`slow_wave.eval.arm_ops` supplies the two custom sleep-window operators
(random prune / oracle prune). The contract checks proven here:

* **Registry** — ``ARM_REGISTRY`` has exactly the nine names, and only ``oracle``
  has ``uses_labels=True`` (the confound guard, FR1.6).
* **Builder** — :func:`build_arm` returns, for every arm, a ``RunnableArm`` whose
  effective ``cfg`` validates and whose toggles match the table; it never mutates
  the base config; it stamps ``effective.seed = seed``; unknown names raise
  ``KeyError``; and it asserts the ``uses_labels`` invariant.
* **EC1 (local)** — every arm runs end-to-end through
  :meth:`slow_wave.agent.wake.WakeAgent.run` on a real Phase-1 stream without
  error.
* **Mechanism** — ``random_pruning`` and ``oracle`` demote to the *archival*
  tier (recoverable; no hard deletes); ``oracle`` demotes only distractor/noise
  sources on a distractor-heavy stream; ``long_context`` never evicts.
* **Confound guard** — only the oracle path reads ``offline_labels``.
* **Determinism (DX1)** — same seed -> identical sampled / demoted ids.

The mock LLM is forced via the ``force_mock_llm`` fixture (no API key), and all
embeddings come from the deterministic hash backend.
"""

from __future__ import annotations

import pytest

from slow_wave.agent.wake import WakeAgent, WakeResult
from slow_wave.config import AgentConfig, Config, DreamConfig, MemoryConfig
from slow_wave.embeddings import get_embedder
from slow_wave.eval import arm_ops
from slow_wave.eval.arms import ARM_REGISTRY, RunnableArm, build_arm
from slow_wave.eval.schema import ArmSpec
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set
from slow_wave.stream.schema import (
    CLScenario,
    Label,
    LabelMix,
    StreamGenConfig,
    offline_labels,
)

# The canonical nine arms (order-independent).
_NINE_ARMS = {
    "no_sleep",
    "replay_only",
    "downscale_only",
    "random_pruning",
    "full_dream",
    "reflection",
    "oracle",
    "long_context",
    "aa",
}

# Per-arm expected effective toggles (only the keys an arm pins are checked).
_EXPECTED_TOGGLES: dict[str, dict] = {
    "no_sleep": {"enabled": False},
    "replay_only": {
        "enabled": True,
        "replay": True,
        "transfer": False,
        "downscale": True,
        "augment": False,
        "conflict": False,
    },
    "downscale_only": {
        "enabled": True,
        "replay": False,
        "transfer": False,
        "downscale": True,
        "augment": False,
        "conflict": False,
    },
    "random_pruning": {
        "enabled": True,
        "replay": False,
        "transfer": False,
        "downscale": False,
        "augment": False,
        "conflict": False,
    },
    "full_dream": {
        "enabled": True,
        "replay": True,
        "transfer": True,
        "downscale": True,
        "augment": True,
    },
    "reflection": {
        "enabled": True,
        "replay": True,
        "transfer": True,
        "downscale": False,
        "augment": False,
        "conflict": False,
        "replay_strategy": "uniform",
        "cls_interleave": False,
    },
    "oracle": {
        "enabled": True,
        "replay": False,
        "transfer": False,
        "downscale": False,
        "augment": False,
        "conflict": False,
    },
    "long_context": {"enabled": False, "episodic_capacity": 0},
    "aa": {"enabled": False},
}


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _make_base_cfg(seed: int = 0) -> Config:
    """Build a distractor-heavy, bounded-capacity base config for the arms.

    The bounded episodic capacity (below the signal count) makes the no-sleep
    baseline forget so the arms diverge, and the distractor-heavy label mix gives
    the oracle plenty to prune.
    """
    return Config(
        experiment="arms-test",
        seed=seed,
        stream=StreamGenConfig(
            scenario=CLScenario.TASK_INCREMENTAL,
            n_tasks=3,
            items_per_task=18,
            label_mix=LabelMix(signal=0.34, distractor=0.40, noise=0.26),
            n_subjects_per_task=6,
            n_attributes=4,
            n_values=12,
            probes_per_task=5,
            distractor_namespace_size=48,
            noise_vocab_size=96,
            noise_tokens=8,
        ),
        memory=MemoryConfig(
            episodic_capacity=30, archival_enabled=True, retrieval_top_k=12
        ),
        agent=AgentConfig(reasoning_calls="per_task"),
        dream=DreamConfig(enabled=True),
    )


@pytest.fixture
def base_cfg() -> Config:
    """A fresh distractor-heavy base :class:`Config`."""
    return _make_base_cfg()


@pytest.fixture
def stream():
    """A real Phase-1 distractor-heavy stream (deterministic)."""
    return generate_stream(_make_base_cfg().stream, seed=123)


@pytest.fixture
def probe_set(stream):
    """The held-out probe set for ``stream``."""
    return build_probe_set(stream)


@pytest.fixture
def embedder(base_cfg):
    """The deterministic hash embedder for the base config."""
    return get_embedder(base_cfg)


def _assert_toggles(cfg: Config, expected: dict) -> None:
    """Assert the effective ``cfg`` matches the pinned toggle subset."""
    d = cfg.dream
    checks = {
        "enabled": d.enabled,
        "replay": d.replay_enabled,
        "transfer": d.transfer_enabled,
        "downscale": d.downscale_enabled,
        "augment": d.augment_enabled,
        "conflict": d.conflict_enabled,
        "replay_strategy": d.replay_strategy,
        "cls_interleave": d.cls_interleave,
        "episodic_capacity": cfg.memory.episodic_capacity,
    }
    for key, want in expected.items():
        assert checks[key] == want, f"{key}: got {checks[key]!r}, want {want!r}"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_has_exactly_nine_arms() -> None:
    """ARM_REGISTRY holds exactly the nine canonical arm names."""
    assert set(ARM_REGISTRY) == _NINE_ARMS
    assert len(ARM_REGISTRY) == 9


def test_registry_spec_names_match_keys() -> None:
    """Each registry key equals its spec's ``name``."""
    for key, spec in ARM_REGISTRY.items():
        assert isinstance(spec, ArmSpec)
        assert spec.name == key


def test_only_oracle_uses_labels() -> None:
    """Only the ``oracle`` arm declares ``uses_labels=True`` (FR1.6)."""
    labelled = {name for name, spec in ARM_REGISTRY.items() if spec.uses_labels}
    assert labelled == {"oracle"}


# --------------------------------------------------------------------------- #
# Builder — config materialization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_NINE_ARMS))
def test_build_arm_returns_runnable_arm_with_valid_cfg(
    name, base_cfg, stream
) -> None:
    """Every arm materializes into a RunnableArm whose cfg re-validates."""
    arm = build_arm(name, base_cfg, stream, seed=0)
    assert isinstance(arm, RunnableArm)
    # Re-validating the effective config exercises extra="forbid".
    Config.model_validate(arm.cfg.model_dump())
    assert arm.spec is ARM_REGISTRY[name]


@pytest.mark.parametrize("name", sorted(_NINE_ARMS))
def test_build_arm_toggles_match_table(name, base_cfg, stream) -> None:
    """Each arm's effective toggles match the authoritative table."""
    arm = build_arm(name, base_cfg, stream, seed=0)
    _assert_toggles(arm.cfg, _EXPECTED_TOGGLES[name])


def test_build_arm_stamps_seed(base_cfg, stream) -> None:
    """build_arm stamps ``effective.seed = seed`` on every arm."""
    for name in _NINE_ARMS:
        arm = build_arm(name, base_cfg, stream, seed=37)
        assert arm.cfg.seed == 37


def test_build_arm_does_not_mutate_base_cfg(base_cfg, stream) -> None:
    """Building every arm leaves the base config byte-identical (deep-merge)."""
    before = base_cfg.model_dump()
    for name in _NINE_ARMS:
        build_arm(name, base_cfg, stream, seed=5)
    after = base_cfg.model_dump()
    assert before == after
    # Spot-check the load-bearing fields the overrides touch.
    assert base_cfg.seed == 0
    assert base_cfg.dream.enabled is True
    assert base_cfg.memory.episodic_capacity == 30


def test_build_arm_unknown_name_raises_keyerror(base_cfg, stream) -> None:
    """An unknown arm name raises ``KeyError`` (the caller guards)."""
    with pytest.raises(KeyError):
        build_arm("does_not_exist", base_cfg, stream, seed=0)


def test_build_arm_asserts_label_permission(monkeypatch, base_cfg, stream) -> None:
    """build_arm refuses a non-oracle spec that claims ``uses_labels=True``."""
    rogue = ArmSpec(
        name="rogue",
        description="illegally claims label access",
        config_overrides={"dream": {"enabled": False}},
        uses_labels=True,
        family="control",
    )
    monkeypatch.setitem(ARM_REGISTRY, "rogue", rogue)
    with pytest.raises(AssertionError):
        build_arm("rogue", base_cfg, stream, seed=0)


def test_aa_builds_reference_arm_verbatim(base_cfg, stream) -> None:
    """The ``aa`` arm reproduces the aa_reference_arm's effective config."""
    assert base_cfg.eval.aa_reference_arm == "no_sleep"
    aa = build_arm("aa", base_cfg, stream, seed=11)
    ref = build_arm("no_sleep", base_cfg, stream, seed=11)
    assert aa.cfg.model_dump() == ref.cfg.model_dump()
    assert aa.sleep_hook is None  # no_sleep has no sleep window
    assert aa.spec.name == "aa"


# --------------------------------------------------------------------------- #
# EC1 (local) — every arm runs end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_NINE_ARMS))
def test_each_arm_runs_end_to_end(
    name, base_cfg, stream, probe_set, embedder, force_mock_llm
) -> None:
    """Each arm runs through WakeAgent.run on a Phase-1 stream (WS-local EC1)."""
    arm = build_arm(name, base_cfg, stream, seed=0)
    result = WakeAgent(arm.cfg, embedder).run(
        stream, probe_set, sleep_hook=arm.sleep_hook
    )
    assert isinstance(result, WakeResult)
    assert result.accuracy_matrix.n_tasks == stream.n_tasks


# --------------------------------------------------------------------------- #
# RunnableArm telemetry accessors
# --------------------------------------------------------------------------- #
def test_runnable_arm_accessors_dream_vs_custom(
    base_cfg, stream, probe_set, embedder, force_mock_llm
) -> None:
    """n_cycles/dream_tokens/generator_fidelity reflect each arm's machinery."""
    # full_dream: the engine runs cycles, spends LLM tokens, and (augment on)
    # reports a generator fidelity.
    full = build_arm("full_dream", base_cfg, stream, seed=0)
    WakeAgent(full.cfg, embedder).run(stream, probe_set, sleep_hook=full.sleep_hook)
    assert full.n_cycles() == stream.n_tasks  # sleep_every_n_tasks defaults to 1
    in_tok, out_tok, api = full.dream_tokens()
    assert api > 0 and in_tok > 0 and out_tok > 0
    fid = full.generator_fidelity()
    assert fid is not None and 0.0 <= fid <= 1.0

    # random_pruning: a custom hook -> no dream tokens, fidelity is None.
    rand = build_arm("random_pruning", base_cfg, stream, seed=0)
    WakeAgent(rand.cfg, embedder).run(stream, probe_set, sleep_hook=rand.sleep_hook)
    assert rand.dream_tokens() == (0, 0, 0)
    assert rand.generator_fidelity() is None
    assert rand.n_cycles() == stream.n_tasks

    # no_sleep: no cycles, no tokens, no fidelity.
    none = build_arm("no_sleep", base_cfg, stream, seed=0)
    WakeAgent(none.cfg, embedder).run(stream, probe_set, sleep_hook=none.sleep_hook)
    assert none.n_cycles() == 0
    assert none.dream_tokens() == (0, 0, 0)
    assert none.generator_fidelity() is None

    # replay_only: dream runs but augment is off -> generator fidelity is None.
    replay = build_arm("replay_only", base_cfg, stream, seed=0)
    WakeAgent(replay.cfg, embedder).run(
        stream, probe_set, sleep_hook=replay.sleep_hook
    )
    assert replay.generator_fidelity() is None


# --------------------------------------------------------------------------- #
# Mechanism — demote-not-delete, oracle relevance, long-context
# --------------------------------------------------------------------------- #
def test_random_pruning_demotes_recoverably(
    base_cfg, stream, probe_set, embedder, force_mock_llm
) -> None:
    """random_pruning demotes to archival (no hard deletes; recoverable)."""
    arm = build_arm("random_pruning", base_cfg, stream, seed=0)
    result = WakeAgent(arm.cfg, embedder).run(
        stream, probe_set, sleep_hook=arm.sleep_hook
    )
    sub = result.substrate
    tel = arm.prune_telemetry
    assert tel is not None
    assert tel.n_demoted > 0
    assert tel.n_demoted == len(tel.demoted_ids)
    assert len(set(tel.demoted_ids)) == len(tel.demoted_ids)  # no duplicates
    for entry_id in tel.demoted_ids:
        # Recoverable from the archival tier...
        assert sub.archival.contains(entry_id)
        recovered = sub.archival.recover(entry_id)
        assert recovered is not None
        assert sub.archival.reason_for(entry_id)[0] == "random_pruning"
        # ...and gone from the active stores (no hard delete, but not live).
        assert sub.episodic.get(entry_id) is None
        assert sub.semantic.get(entry_id) is None


def test_oracle_demotes_only_distractor_noise(
    base_cfg, stream, probe_set, embedder, force_mock_llm
) -> None:
    """oracle demotes exactly distractor/noise sources; signals are retained."""
    arm = build_arm("oracle", base_cfg, stream, seed=0)
    result = WakeAgent(arm.cfg, embedder).run(
        stream, probe_set, sleep_hook=arm.sleep_hook
    )
    sub = result.substrate
    tel = arm.prune_telemetry
    labels = offline_labels(stream)
    assert tel is not None
    assert tel.n_demoted > 0
    for entry_id in tel.demoted_ids:
        recovered = sub.archival.recover(entry_id)
        assert recovered is not None  # demote-not-delete
        assert sub.archival.reason_for(entry_id)[0] == "oracle_prune"
        item_id = recovered.provenance[0]
        assert labels[item_id] in {Label.DISTRACTOR, Label.NOISE}
    # No signal item the oracle saw should have been demoted by it.
    demoted_items = {
        sub.archival.recover(eid).provenance[0] for eid in tel.demoted_ids
    }
    signal_items = {iid for iid, lab in labels.items() if lab is Label.SIGNAL}
    assert demoted_items.isdisjoint(signal_items)


def test_long_context_never_evicts(
    base_cfg, stream, probe_set, embedder, force_mock_llm
) -> None:
    """long_context (unbounded) holds every item; nothing is evicted/demoted."""
    arm = build_arm("long_context", base_cfg, stream, seed=0)
    assert arm.sleep_hook is None
    result = WakeAgent(arm.cfg, embedder).run(
        stream, probe_set, sleep_hook=arm.sleep_hook
    )
    sub = result.substrate
    assert len(sub.archival) == 0  # nothing demoted/evicted
    assert len(sub.episodic) == len(stream.items)  # every item retained live
    # Every signal item's representation is live in episodic.
    labels = offline_labels(stream)
    live_items = {
        e.provenance[0] for e in sub.episodic.all_entries() if e.provenance
    }
    signal_items = {iid for iid, lab in labels.items() if lab is Label.SIGNAL}
    assert signal_items <= live_items


# --------------------------------------------------------------------------- #
# Confound guard — only oracle reads labels
# --------------------------------------------------------------------------- #
def test_only_oracle_path_reads_labels(
    monkeypatch, base_cfg, stream, probe_set, embedder, force_mock_llm
) -> None:
    """Only the oracle path calls ``offline_labels`` (FR1.6)."""
    calls: list[str] = []
    real = arm_ops.offline_labels

    def spy(s):
        calls.append(s.stream_id)
        return real(s)

    monkeypatch.setattr(arm_ops, "offline_labels", spy)

    # Building + running every non-oracle arm reads no labels.
    for name in sorted(_NINE_ARMS - {"oracle"}):
        arm = build_arm(name, base_cfg, stream, seed=0)
        WakeAgent(arm.cfg, embedder).run(
            stream, probe_set, sleep_hook=arm.sleep_hook
        )
    assert calls == []

    # The oracle arm reads labels (once, at hook construction).
    build_arm("oracle", base_cfg, stream, seed=0)
    assert calls == [stream.stream_id]


# --------------------------------------------------------------------------- #
# Determinism (DX1)
# --------------------------------------------------------------------------- #
def _run_demoted_ids(name: str, *, seed: int) -> list[str]:
    """Build + run ``name`` from scratch and return its demoted-id list."""
    base = _make_base_cfg()
    strm = generate_stream(_make_base_cfg().stream, seed=123)
    pset = build_probe_set(strm)
    emb = get_embedder(base)
    arm = build_arm(name, base, strm, seed=seed)
    WakeAgent(arm.cfg, emb).run(strm, pset, sleep_hook=arm.sleep_hook)
    return list(arm.prune_telemetry.demoted_ids)


def test_random_pruning_is_deterministic(force_mock_llm) -> None:
    """Same seed -> byte-identical random-prune demotions (DX1)."""
    first = _run_demoted_ids("random_pruning", seed=0)
    second = _run_demoted_ids("random_pruning", seed=0)
    assert first == second
    assert first  # non-empty (it actually pruned)
    # A different seed changes the random selection.
    other = _run_demoted_ids("random_pruning", seed=1)
    assert other != first


def test_oracle_is_deterministic(force_mock_llm) -> None:
    """Same seed -> identical oracle demotions (DX1; no RNG, label-driven)."""
    first = _run_demoted_ids("oracle", seed=0)
    second = _run_demoted_ids("oracle", seed=0)
    assert first == second
    assert first


def test_full_dream_engine_is_deterministic(force_mock_llm) -> None:
    """full_dream's dream telemetry is byte-identical across two runs (DX1)."""

    def run() -> dict:
        base = _make_base_cfg()
        strm = generate_stream(_make_base_cfg().stream, seed=123)
        pset = build_probe_set(strm)
        emb = get_embedder(base)
        arm = build_arm("full_dream", base, strm, seed=0)
        WakeAgent(arm.cfg, emb).run(strm, pset, sleep_hook=arm.sleep_hook)
        return arm.engine.telemetry.model_dump()

    assert run() == run()
