"""Golden-artifact regression guard for the COMMITTED Phase 5 deliverables.

The pipeline tests (``test_eval_phase5_integration``) re-run the *smoke* grid in a
temp dir; nothing otherwise pins the **shipped, science-scale 8-seed** numbers
the paper cites. This module loads the committed ``phase5/phase5_result.json``
(+ its embedded ``analysis``) and asserts its headline values, so a future code
change that silently moves the verdict / effect size / TMR lift / power is caught
in CI rather than at paper-submission time.

Every number here is the deterministic mock-LLM **mechanism-demonstration** value
(DX5 — not a claim about a real Claude model), reproduced bit-for-bit by
``make repro-phase5``. Tolerances are loose enough to survive a harmless
float-repr change but tight enough to catch a real regression.

The test self-skips if the committed artifact is absent (e.g. a sparse checkout),
so it never blocks CI on environments that do not ship the artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slow_wave.eval.phase5_schema import Phase5Result

_REPO = Path(__file__).resolve().parents[1]
_RESULT = _REPO / "phase5" / "phase5_result.json"

pytestmark = pytest.mark.skipif(
    not _RESULT.exists(),
    reason="committed phase5/phase5_result.json not present in this checkout",
)


@pytest.fixture(scope="module")
def committed() -> Phase5Result:
    """The committed science-scale Phase 5 artifact (with analysis filled)."""
    return Phase5Result.model_validate_json(_RESULT.read_text("utf-8"))


def test_committed_is_mock_mechanism_demo(committed: Phase5Result) -> None:
    """DX5: the shipped result is the mock-LLM mechanism demo, analysis present."""
    assert committed.model_mocked is True
    assert committed.analysis is not None


def test_committed_grid_shape(committed: Phase5Result) -> None:
    """EC1/EC2: 3 regimes x 9 arms x 8 seeds, distractor_heavy primary."""
    grid = committed.grid
    assert len(grid.seeds) == 8
    assert grid.primary_regime == "distractor_heavy"
    assert {c.regime.name for c in grid.regimes} == {
        "signal_rich",
        "balanced",
        "distractor_heavy",
    }
    for cell in grid.regimes:
        assert len(cell.acc_by_arm) == 9, cell.regime.name
        for arm, accs in cell.acc_by_arm.items():
            assert len(accs) == 8, f"{cell.regime.name}:{arm}"


def test_committed_primary_endpoint(committed: Phase5Result) -> None:
    """EC5: the preregistered endpoint is CONFIRMED with the shipped numbers."""
    a = committed.analysis
    assert a is not None
    assert a.primary_endpoint_name == "acc_diff_full_dream_vs_no_sleep"
    assert a.primary_verdict == "confirmed"
    assert a.primary_value == pytest.approx(0.3792, abs=3e-3)
    assert a.primary_effect_name == "cohens_d"
    assert a.primary_effect_value == pytest.approx(4.890, abs=2e-2)
    assert a.exceeds_noise_floor is True
    assert a.primary_ci_lo is not None and a.primary_ci_lo > 0.0
    assert a.primary_test_name == "wilcoxon_signed_rank"


def test_committed_tmr_targeting(committed: Phase5Result) -> None:
    """FR5.3: the shipped TMR-style lift exceeds the Hu 2020 g=0.29 benchmark."""
    t = committed.analysis.tmr
    assert t.benchmark_g == pytest.approx(0.29)
    assert t.hedges_g == pytest.approx(1.7017, abs=3e-3)
    assert t.exceeds_benchmark is True


def test_committed_crossover_absent(committed: Phase5Result) -> None:
    """EC6: the shipped result reports NO cost-adjusted long-context crossover."""
    assert committed.analysis.crossover.crossover_found is False


def test_committed_power(committed: Phase5Result) -> None:
    """EC2: realized power is consistent (n=8 >= floor 5, powered for observed d)."""
    p = committed.analysis.power
    assert p.n_seeds == 8
    assert p.floor == 5
    assert p.floor_met is True
    assert p.powered_for_observed is True
    assert p.powered_for_observed == (p.n_seeds >= p.required_n_for_observed)


def test_committed_manifests_exist_with_provenance(committed: Phase5Result) -> None:
    """EC1: every recorded per-cell manifest path resolves under the repo."""
    assert committed.manifest_paths
    for rel in committed.manifest_paths:
        assert (_REPO / rel).exists(), rel
