"""Cross-module EC1 check: the 2^4 operator on/off matrix over a real stream.

Phase 3 exit criterion #1 requires that each of the four dream operators be
independently enabled/disabled by config and that *every* one of the 2^4 = 16
on/off combinations (plus the 2x2 replay x downscale sub-matrix) instantiates and
runs. ``tests/test_dream_engine.py`` proves this at the engine-unit level on
synthetic entries; this module proves it **end-to-end through the real runner**
(`run_dream`) on a genuine Phase-1 generated stream, and confirms the manifest's
recorded ``operators_run`` matches the enabled set for each combination.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from slow_wave.config import (
    Config,
    DreamConfig,
    EmbeddingConfig,
    MemoryConfig,
)
from slow_wave.dream.runner import run_dream
from slow_wave.repro.manifest import read_manifest
from slow_wave.stream.schema import CLScenario, LabelMix, StreamGenConfig

OPERATORS = ("replay_enabled", "transfer_enabled", "downscale_enabled", "augment_enabled")
OP_NAME = {
    "replay_enabled": "replay",
    "transfer_enabled": "transfer",
    "downscale_enabled": "downscale",
    "augment_enabled": "augment",
}


def _make_config(toggles: dict[str, bool], seed: int = 5) -> Config:
    """Build a tiny, dependency-free dream Config with the given operator toggles."""
    return Config(
        experiment="dream-matrix",
        seed=seed,
        embedding=EmbeddingConfig(backend="hash", model="hash-bow-v1", dim=384),
        stream=StreamGenConfig(
            scenario=CLScenario.TASK_INCREMENTAL,
            n_tasks=2,
            items_per_task=6,
            label_mix=LabelMix(signal=0.5, distractor=0.3, noise=0.2),
            n_subjects_per_task=4,
            n_attributes=2,
            n_values=8,
            probes_per_task=2,
        ),
        memory=MemoryConfig(episodic_capacity=8, archival_enabled=True),
        dream=DreamConfig(
            enabled=True,
            sleep_every_n_tasks=1,
            replay_sample_size=6,
            transfer_batch_size=4,
            augment_per_cycle=2,
            **toggles,
        ),
    )


@pytest.mark.parametrize(
    "combo", list(itertools.product([False, True], repeat=4)), ids=lambda c: "".join("1" if x else "0" for c_ in [c] for x in c_) if isinstance(c, tuple) else str(c)
)
def test_all_16_operator_combinations_run(
    combo: tuple[bool, ...], tmp_path: Path, force_mock_llm
) -> None:
    """EC1: every one of the 2^4 operator on/off combinations runs end-to-end.

    Runs the real ``run_dream`` over a generated Phase-1 stream for each
    combination and asserts the manifest records a ``operators_run`` (unioned over
    cycles) exactly equal to the enabled operator set — including the all-off
    combination, whose every cycle records ``operators_run == []``.
    """
    toggles = dict(zip(OPERATORS, combo))
    cfg = _make_config(toggles)

    manifest_path = run_dream(cfg, out_dir=tmp_path)
    manifest = read_manifest(manifest_path)

    dream_tel = manifest.results["dream_telemetry"]
    cycles = dream_tel["cycles"]
    assert dream_tel["n_cycles"] == 2  # sleep_every_n_tasks=1, n_tasks=2

    expected = {OP_NAME[k] for k, v in toggles.items() if v}
    for cycle in cycles:
        assert set(cycle["operators_run"]) == expected
        if not expected:
            assert cycle["operators_run"] == []


def test_2x2_replay_downscale_submatrix(tmp_path: Path, force_mock_llm) -> None:
    """EC1: the explicit 2x2 (replay-module x downscale-module) sub-matrix runs.

    With transfer + augment held off, all four (replay, downscale) on/off cells
    instantiate and run, each recording exactly its enabled subset.
    """
    for replay_on, downscale_on in itertools.product([False, True], repeat=2):
        cfg = _make_config(
            {
                "replay_enabled": replay_on,
                "transfer_enabled": False,
                "downscale_enabled": downscale_on,
                "augment_enabled": False,
            }
        )
        manifest = read_manifest(run_dream(cfg, out_dir=tmp_path))
        expected = set()
        if replay_on:
            expected.add("replay")
        if downscale_on:
            expected.add("downscale")
        for cycle in manifest.results["dream_telemetry"]["cycles"]:
            assert set(cycle["operators_run"]) == expected
