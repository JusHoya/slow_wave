"""Deterministic seeding utilities for the Slow Wave bench (Phase 0).

This module pins how randomness is seeded across the bench so that the
non-LLM portions of a run (embeddings, sampling orders, file layout) are
reproducible bit-for-bit given a fixed config + seed (Phase 0 exit
criterion #3 in ``docs/PHASE0_CONTRACT.md``).

Two primitives are provided:

* :func:`set_global_seeds` seeds the process-global RNGs (Python's
  :mod:`random` and NumPy's legacy global RNG). Newer code is expected to use
  :func:`numpy.random.default_rng` with an *explicit* seed (typically one
  derived via :func:`derive_seed`), so seeding the globals here is a
  belt-and-braces measure for any code path that still reaches for the
  module-level functions.
* :func:`derive_seed` deterministically derives a stable child seed from a
  master seed and a human-readable name (e.g. ``"agent"``, ``"stream"``), so
  independent random streams can be seeded reproducibly without colliding.

.. note::
   ``PYTHONHASHSEED`` controls Python's salted hashing of ``str``/``bytes``
   and **cannot** be changed once the interpreter has started. We never rely
   on ``hash()`` for any reproducible artifact (we use :mod:`hashlib` instead),
   so leaving ``PYTHONHASHSEED`` unset is fine. Pin it in the environment only
   if you additionally want set/dict iteration order over hashed keys to be
   stable across processes.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed the process-global RNGs for reproducibility.

    Seeds Python's :func:`random.seed` and NumPy's legacy global RNG
    (:func:`numpy.random.seed`). This makes subsequent calls to
    :func:`random.random`, :func:`numpy.random.rand`, etc. reproducible across
    two invocations with the same ``seed``.

    Code that needs an independent, explicitly-seeded stream should prefer
    ``numpy.random.default_rng(derive_seed(master, name))`` rather than relying
    on these globals.

    Args:
        seed: The master seed to apply to the global RNGs.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)


def derive_seed(master: int, name: str) -> int:
    """Derive a stable child seed from a master seed and a name.

    The derivation is a pure function of ``(master, name)`` and is stable
    across runs, processes, and platforms because it uses BLAKE2b
    (:mod:`hashlib`) rather than Python's salted built-in ``hash()``. The
    result is reduced to a 32-bit unsigned integer so it is a valid seed for
    NumPy's legacy RNG and :func:`numpy.random.default_rng`.

    Args:
        master: The master seed.
        name: A human-readable label for the derived stream
            (e.g. ``"agent"``, ``"stream"``).

    Returns:
        A deterministic integer seed in the range ``[0, 2**32)``.
    """

    digest = hashlib.blake2b(f"{master}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)
