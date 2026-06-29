"""Tests for slow_wave.eval.prereg (Phase 4, WS-PREREG, FR5.5, EC7, DX3).

These pin the preregistration guard's falsifiability-in-code behavior: the
committed artifact parses, the analysis refuses any endpoint other than the
single preregistered one (DX3), and the primary-endpoint computation yields the
pre-committed verdicts on synthetic arm results (treatment strictly beating
baseline -> ``confirmed`` with ``value > 0``; equal arms -> a non-confirmed
verdict with a CI spanning zero).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slow_wave.eval.metrics import ContinualMetrics
from slow_wave.eval.prereg import (
    NonPreregisteredEndpointError,
    assert_primary_endpoint,
    compute_primary_endpoint,
    load_preregistration,
)
from slow_wave.eval.schema import (
    AAResult,
    ArmCost,
    ArmResult,
    Preregistration,
    PrimaryEndpoint,
)
from slow_wave.memory.schema import MemoryFootprint, MemoryTier, StoreFootprint
from slow_wave.stream.schema import AccuracyMatrix, CLScenario

PRIMARY = "acc_diff_full_dream_vs_no_sleep"


# --------------------------------------------------------------------------- #
# Synthetic fixture builders
# --------------------------------------------------------------------------- #
def _empty_footprint(dim: int = 384) -> MemoryFootprint:
    """A zero footprint with one well-formed StoreFootprint per tier."""
    return MemoryFootprint(
        episodic=StoreFootprint(
            tier=MemoryTier.EPISODIC, n_entries=0, n_vectors=0, dim=dim, bytes=0
        ),
        semantic=StoreFootprint(
            tier=MemoryTier.SEMANTIC, n_entries=0, n_vectors=0, dim=dim, bytes=0
        ),
        archival=StoreFootprint(
            tier=MemoryTier.ARCHIVAL, n_entries=0, n_vectors=0, dim=dim, bytes=0
        ),
        total_bytes=0,
    )


def _zero_cost() -> ArmCost:
    """A zero ArmCost (the endpoint reads only continual_metrics.acc)."""
    return ArmCost(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        api_calls=0,
        retrieval_calls=0,
        memory_vectors=0,
        memory_bytes=0,
    )


def _arm_result(arm_name: str, seed: int, acc: float) -> ArmResult:
    """A minimal valid ArmResult whose paired final ACC is ``acc``."""
    clamped = float(min(max(acc, 0.0), 1.0))
    matrix = AccuracyMatrix(
        n_tasks=1, scenario=CLScenario.TASK_INCREMENTAL, R=[[clamped]]
    )
    metrics = ContinualMetrics(
        acc=acc,
        bwt=0.0,
        fwt=0.0,
        average_forgetting=0.0,
        per_task_forgetting=[],
    )
    return ArmResult(
        arm_name=arm_name,
        seed=seed,
        scenario=CLScenario.TASK_INCREMENTAL.value,
        accuracy_matrix=matrix,
        continual_metrics=metrics,
        footprint=_empty_footprint(),
        cost=_zero_cost(),
    )


def _grid(treatment_accs: dict[int, float], baseline_accs: dict[int, float]) -> list[ArmResult]:
    """Build a paired arm-result grid for full_dream vs no_sleep."""
    results: list[ArmResult] = []
    for seed, acc in treatment_accs.items():
        results.append(_arm_result("full_dream", seed, acc))
    for seed, acc in baseline_accs.items():
        results.append(_arm_result("no_sleep", seed, acc))
    return results


def _aa(abs_difference: float = 0.01) -> AAResult:
    """An A/A control with a small noise floor."""
    return AAResult(
        reference_arm="no_sleep",
        seed_a=0,
        seed_b=1,
        value_a=0.60,
        value_b=0.60 + abs_difference,
        abs_difference=abs_difference,
        significant=False,
        test=None,
    )


@pytest.fixture
def prereg(repo_root: Path) -> Preregistration:
    """The committed preregistration artifact, parsed."""
    return load_preregistration(repo_root / "prereg" / "preregistration.yaml")


# --------------------------------------------------------------------------- #
# load_preregistration (EC7)
# --------------------------------------------------------------------------- #
def test_load_preregistration_parses_committed_artifact(
    prereg: Preregistration,
) -> None:
    """The committed artifact parses and names the expected primary endpoint."""
    assert prereg.primary_endpoint == PRIMARY
    assert prereg.treatment_arm == "full_dream"
    assert prereg.baseline_arm == "no_sleep"
    assert prereg.tests  # non-empty list of pre-specified tests
    assert prereg.rejection_criteria  # non-empty falsifiable criteria


# --------------------------------------------------------------------------- #
# assert_primary_endpoint (DX3)
# --------------------------------------------------------------------------- #
def test_assert_primary_endpoint_silent_for_correct(prereg: Preregistration) -> None:
    """The guard is silent for the preregistered endpoint name."""
    assert assert_primary_endpoint(prereg, PRIMARY) is None


def test_assert_primary_endpoint_raises_for_wrong(prereg: Preregistration) -> None:
    """The guard refuses any other endpoint name (DX3)."""
    with pytest.raises(NonPreregisteredEndpointError):
        assert_primary_endpoint(prereg, "acc_diff_reflection_vs_no_sleep")


# --------------------------------------------------------------------------- #
# compute_primary_endpoint (EC7 + DX3)
# --------------------------------------------------------------------------- #
def test_compute_confirmed_when_treatment_beats_baseline(
    prereg: Preregistration,
) -> None:
    """Treatment strictly beating baseline on every seed -> confirmed, value>0."""
    treatment = {0: 0.80, 1: 0.82, 2: 0.78, 3: 0.81, 4: 0.79}
    baseline = {0: 0.60, 1: 0.62, 2: 0.58, 3: 0.61, 4: 0.59}
    grid = _grid(treatment, baseline)

    result = compute_primary_endpoint(
        prereg,
        grid,
        requested_name=PRIMARY,
        aa=_aa(abs_difference=0.01),
        rng=np.random.default_rng(0),
    )

    assert isinstance(result, PrimaryEndpoint)
    assert result.name == PRIMARY
    assert result.treatment_arm == "full_dream"
    assert result.baseline_arm == "no_sleep"
    assert result.value > 0.0
    assert result.value == pytest.approx(0.20, abs=0.01)
    # CI excludes zero.
    assert result.difference_ci.lo > 0.0
    assert result.exceeds_noise_floor is True
    assert result.verdict == "confirmed"


def test_compute_refuted_for_equal_arms(prereg: Preregistration) -> None:
    """Identical arms -> non-confirmed verdict with a CI spanning zero."""
    accs = {0: 0.60, 1: 0.62, 2: 0.58, 3: 0.61, 4: 0.59}
    grid = _grid(accs, dict(accs))

    result = compute_primary_endpoint(
        prereg,
        grid,
        requested_name=PRIMARY,
        aa=_aa(abs_difference=0.01),
        rng=np.random.default_rng(0),
    )

    assert result.value == pytest.approx(0.0)
    # CI spans zero.
    assert result.difference_ci.lo <= 0.0 <= result.difference_ci.hi
    assert result.verdict != "confirmed"
    assert result.verdict == "refuted"


def test_compute_raises_for_non_preregistered_endpoint(
    prereg: Preregistration,
) -> None:
    """A wrong requested_name is refused before any statistic (DX3)."""
    grid = _grid({0: 0.8, 1: 0.8}, {0: 0.6, 1: 0.6})
    with pytest.raises(NonPreregisteredEndpointError):
        compute_primary_endpoint(
            prereg,
            grid,
            requested_name="something_else",
            aa=_aa(),
            rng=np.random.default_rng(0),
        )


def test_compute_is_deterministic_and_json_dumpable(
    prereg: Preregistration,
) -> None:
    """Two computations with the same rng seed are byte-identical (DX1)."""
    treatment = {0: 0.80, 1: 0.82, 2: 0.78, 3: 0.81, 4: 0.79}
    baseline = {0: 0.60, 1: 0.62, 2: 0.58, 3: 0.61, 4: 0.59}
    grid = _grid(treatment, baseline)

    a = compute_primary_endpoint(
        prereg, grid, requested_name=PRIMARY, aa=_aa(), rng=np.random.default_rng(7)
    )
    b = compute_primary_endpoint(
        prereg, grid, requested_name=PRIMARY, aa=_aa(), rng=np.random.default_rng(7)
    )
    assert a.model_dump() == b.model_dump()
    assert PrimaryEndpoint.model_validate_json(a.model_dump_json()) == a
