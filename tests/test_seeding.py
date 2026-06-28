"""Tests for slow_wave.repro.seeding (Phase 0)."""

from __future__ import annotations

import random

import numpy as np

from slow_wave.repro.seeding import derive_seed, set_global_seeds


def test_derive_seed_is_deterministic() -> None:
    """derive_seed is a pure function: same inputs -> same output."""
    assert derive_seed(0, "agent") == derive_seed(0, "agent")
    assert derive_seed(42, "stream") == derive_seed(42, "stream")


def test_derive_seed_in_uint32_range() -> None:
    """Derived seeds fall in [0, 2**32) for use as NumPy seeds."""
    for master in (0, 1, 7, 42, 2**31, 2**40):
        for name in ("agent", "stream", "master", ""):
            seed = derive_seed(master, name)
            assert isinstance(seed, int)
            assert 0 <= seed < 2**32


def test_derive_seed_distinct_names_give_distinct_seeds() -> None:
    """Different names (very likely) derive different seeds from one master."""
    names = ["agent", "stream", "master", "replay", "dream", "downscale"]
    seeds = {name: derive_seed(123, name) for name in names}
    # No collisions across this small, fixed set of stream names.
    assert len(set(seeds.values())) == len(names)


def test_derive_seed_distinct_masters_give_distinct_seeds() -> None:
    """The same name under different masters derives different seeds."""
    seeds = {m: derive_seed(m, "agent") for m in range(8)}
    assert len(set(seeds.values())) == 8


def test_set_global_seeds_makes_numpy_reproducible() -> None:
    """Two set_global_seeds(seed) calls reproduce numpy.random output."""
    set_global_seeds(2024)
    first = np.random.rand(5)
    set_global_seeds(2024)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_set_global_seeds_makes_random_reproducible() -> None:
    """Two set_global_seeds(seed) calls reproduce random.random output."""
    set_global_seeds(99)
    first = [random.random() for _ in range(5)]
    set_global_seeds(99)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_global_seeds_returns_none() -> None:
    """set_global_seeds returns None (it only has side effects)."""
    assert set_global_seeds(0) is None
