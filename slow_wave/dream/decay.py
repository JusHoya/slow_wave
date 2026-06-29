"""Swappable salience-decay curves for the dream engine (Phase 3, FR4.3, EC6).

DOWNSCALE (FR4.3) multiplies every memory item's salience by a decay factor each
cycle ("decay all, protect signal"). The *shape* of that decay is ablatable
(Phase 3 exit criterion #6): exponential, Weibull, or ACT-R base-level
activation decay. This module provides the three curves plus a small registry so
a new curve can be added by implementing one function and registering it (DX6),
selected by name through :class:`~slow_wave.config.DreamConfig.decay_function`.

Every curve maps an **age** (units of accumulated decay since the entry was last
potentiated — the dream cycle passes ``now_order - recency_order``) to a
multiplicative factor in ``(0, 1]``. All three satisfy:

* ``f(0) == 1.0`` — a freshly potentiated item is not decayed (so re-potentiated
  "signal" items are protected); and
* monotonically non-increasing in age, bounded to ``(0, 1]``.

The functions are pure, deterministic, and depend only on NumPy-free standard
math, so two calls with identical inputs return identical outputs (DX1).
"""

from __future__ import annotations

import math
import sys
from typing import Callable

#: A decay curve: ``factor = f(age, **params)`` in ``(0, 1]``, ``f(0) == 1``.
DecayFn = Callable[..., float]

#: Smallest strictly-positive factor any curve may return. At extreme ages the
#: exact value of a curve (e.g. ``exp(-(1000/10)**1.5)``) underflows IEEE-754 to a
#: literal ``0.0``; clamping to this floor preserves the documented ``(0, 1]``
#: invariant (a decay factor is never *exactly* zero — memory fades, it does not
#: vanish in one step) without perceptibly changing any in-range result.
_MIN_FACTOR: float = sys.float_info.min


def exponential_decay(age: float, *, rate: float = 0.1) -> float:
    """Exponential decay ``exp(-rate * age)`` (the classic forgetting curve).

    Args:
        age: Non-negative age (accumulated decay units since last potentiation).
        rate: Decay rate ``> 0``; larger forgets faster.

    Returns:
        The decay factor ``exp(-rate * max(0, age))`` in ``(0, 1]`` (``1.0`` at
        ``age == 0``; floored at :data:`_MIN_FACTOR` so it never underflows to 0).
    """
    a = max(0.0, float(age))
    return max(_MIN_FACTOR, float(math.exp(-rate * a)))


def weibull_decay(age: float, *, scale: float = 10.0, k: float = 1.5) -> float:
    """Weibull (stretched-exponential) decay ``exp(-(age/scale)**k)``.

    A more flexible forgetting curve than the pure exponential: ``k < 1`` gives a
    heavy tail (slow late forgetting), ``k > 1`` a sharper drop-off.

    Args:
        age: Non-negative age (accumulated decay units since last potentiation).
        scale: Characteristic scale ``> 0`` (age at which the exponent is 1).
        k: Weibull shape ``> 0``.

    Returns:
        The decay factor in ``(0, 1]`` (``1.0`` at ``age == 0``; floored at
        :data:`_MIN_FACTOR` so it never underflows to 0 at extreme ages).
    """
    a = max(0.0, float(age))
    return max(_MIN_FACTOR, float(math.exp(-((a / scale) ** k))))


def act_r_decay(age: float, *, d: float = 0.5) -> float:
    """ACT-R base-level activation decay ``(1 + age) ** (-d)`` (Anderson 1991).

    The single-trace base-level learning equation, written as a multiplicative
    factor: power-law forgetting with a heavier tail than the exponential.

    Args:
        age: Non-negative age (accumulated decay units since last potentiation).
        d: Base-level decay exponent ``> 0`` (ACT-R's default is ``0.5``).

    Returns:
        The decay factor ``(1 + age) ** (-d)`` in ``(0, 1]`` (``1.0`` at
        ``age == 0``; floored at :data:`_MIN_FACTOR`).
    """
    a = max(0.0, float(age))
    return max(_MIN_FACTOR, float((1.0 + a) ** (-d)))


#: Registry of named decay curves (DX6). Keyed by the
#: :class:`~slow_wave.config.DreamConfig.decay_function` literal.
DECAY_REGISTRY: dict[str, DecayFn] = {
    "exponential": exponential_decay,
    "weibull": weibull_decay,
    "act_r": act_r_decay,
}


def register_decay(name: str, fn: DecayFn) -> None:
    """Register a decay curve under ``name`` (overwrites any same-name entry).

    Args:
        name: The registry key (matches ``DreamConfig.decay_function``).
        fn: The decay curve ``f(age, **params) -> float`` in ``(0, 1]``.
    """
    DECAY_REGISTRY[name] = fn


def get_decay(name: str) -> DecayFn:
    """Return the registered decay curve named ``name``.

    Args:
        name: The decay-curve name.

    Returns:
        The decay curve.

    Raises:
        KeyError: If no curve is registered under ``name``.
    """
    return DECAY_REGISTRY[name]


def decay_factor(name: str, age: float, params: dict | None = None) -> float:
    """Dispatch to the named decay curve with ``params`` and return the factor.

    Convenience wrapper so callers (DOWNSCALE) can pick the curve by name and
    pass its parameter dict in one call.

    Args:
        name: The decay-curve name (``"exponential"`` / ``"weibull"`` /
            ``"act_r"`` by default).
        age: Non-negative age passed to the curve.
        params: Keyword parameters for the curve (e.g. ``{"rate": 0.1}`` for
            ``exponential``); ``None`` => the curve's defaults.

    Returns:
        The decay factor in ``(0, 1]``.

    Raises:
        KeyError: If no curve is registered under ``name``.
    """
    fn = get_decay(name)
    return float(fn(age, **(params or {})))


def params_for(name: str, cfg) -> dict:
    """Return the keyword parameters for decay curve ``name`` from a config.

    Maps the flat :class:`~slow_wave.config.DreamConfig` decay knobs onto the
    keyword arguments each curve expects, so DOWNSCALE can call
    ``decay_factor(name, age, params_for(name, cfg))`` without knowing the
    per-curve parameter names.

    Args:
        name: The decay-curve name.
        cfg: A :class:`~slow_wave.config.DreamConfig` (or any object exposing the
            ``decay_exponential_rate`` / ``decay_weibull_scale`` /
            ``decay_weibull_k`` / ``decay_act_r`` attributes).

    Returns:
        The keyword-parameter dict for the named curve (empty for unknown names,
        so a custom registered curve falls back to its own defaults).
    """
    if name == "exponential":
        return {"rate": cfg.decay_exponential_rate}
    if name == "weibull":
        return {"scale": cfg.decay_weibull_scale, "k": cfg.decay_weibull_k}
    if name == "act_r":
        return {"d": cfg.decay_act_r}
    return {}
