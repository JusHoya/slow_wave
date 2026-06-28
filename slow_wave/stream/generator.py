"""Deterministic synthetic stream generator for the Slow Wave bench (Phase 1, WS1).

This module implements :func:`generate_stream`, the authoritative producer of a
:class:`~slow_wave.stream.schema.Stream` from a
:class:`~slow_wave.stream.schema.StreamGenConfig` and an integer seed. It is the
WS1 deliverable described in ``docs/PHASE1_CONTRACT.md`` ("WS1 — Generator") and
imports the shared data model from :mod:`slow_wave.stream.schema` without
redefining any of its types.

Determinism discipline (FR1.4)
------------------------------
Every random decision is drawn from an explicitly seeded
:class:`numpy.random.Generator` obtained via
``numpy.random.default_rng(derive_seed(seed, "<purpose>"))`` (purposes:
``"labels"``, ``"subjects"``, ``"values"``, ``"order"``, ``"noise"``). Python's
salted ``hash()`` is never used; the stream id derives from a
:mod:`hashlib` ``sha256`` digest over the canonical ``(seed, config)`` JSON. Two
calls with the same ``(config, seed)`` therefore produce an identical
:class:`Stream` and a byte-identical
``json.dumps(stream.model_dump(mode="json"), sort_keys=True)``.

The synthetic *fact world*
--------------------------
A stream is ``n_tasks`` contiguous task segments of ``items_per_task`` ingest
items each. Each item is one of three relevance kinds (the OFFLINE-ONLY label):

* ``signal`` — asserts a fact over the *probed* namespace (mission-relevant).
* ``distractor`` — asserts a plausibly-formed fact over a disjoint distractor
  namespace (``dsubj_*``) that is never probed.
* ``noise`` — a random token salad; ``fact=None``.

The probed namespace is shaped by the continual-learning scenario (FR1.2); the
distractor regime knobs ``contradiction_rate``/``drift``/``recency_bias``
(FR1.3) visibly reshape the signal facts.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from slow_wave.repro.seeding import derive_seed
from slow_wave.stream.schema import (
    CLScenario,
    Fact,
    GroundTruth,
    ItemKind,
    Label,
    LabelMix,
    Stream,
    StreamGenConfig,
    StreamItem,
)

__all__ = ["generate_stream"]


def _largest_remainder(mix: LabelMix, total: int) -> dict[Label, int]:
    """Allocate ``total`` items across the three labels by largest remainder.

    The largest-remainder (Hamilton) method floors each ``proportion * total``
    and then hands the leftover units, one at a time, to the labels with the
    largest fractional parts (ties broken by a fixed label order:
    signal < distractor < noise). The returned counts always sum **exactly** to
    ``total`` (FR1.1, Phase 1 exit criterion #2).

    Args:
        mix: The requested label proportions (sum to ~1.0, validated upstream).
        total: The number of items to allocate (``items_per_task``).

    Returns:
        A mapping ``Label -> count`` whose values sum to ``total``.
    """
    proportions: list[tuple[Label, float]] = [
        (Label.SIGNAL, mix.signal),
        (Label.DISTRACTOR, mix.distractor),
        (Label.NOISE, mix.noise),
    ]
    raw = [p * total for _, p in proportions]
    floors = [int(math.floor(x)) for x in raw]
    remainder = total - sum(floors)
    # Indices ordered by descending fractional part, ties by original index.
    ranked = sorted(
        range(len(raw)),
        key=lambda i: (-(raw[i] - math.floor(raw[i])), i),
    )
    counts = {lab: floors[i] for i, (lab, _) in enumerate(proportions)}
    for k in range(remainder):
        counts[proportions[ranked[k]][0]] += 1
    return counts


def _scenario_namespace(
    scenario: CLScenario, task_index: int, config: StreamGenConfig
) -> tuple[list[str], list[str]]:
    """Return the probed ``(subjects, attributes)`` vocab for a task (FR1.2).

    The continual-learning scenario shapes the probed namespace:

    * task-incremental: subjects ``subj_{t}_{k}`` (disjoint per task),
      attributes ``attr_{a}`` (shared).
    * domain-incremental: subjects ``dom{t}_subj_{k}`` (per-task "domain" input
      shift), attributes ``attr_{a}`` (shared answer space).
    * class-incremental: subjects ``subj_{t}_{k}``, attributes
      ``attr_t{t}_{a}`` (a growing set of attribute "classes").

    Args:
        scenario: The stream's single CL scenario.
        task_index: The 0-based task segment index.
        config: The generation config (supplies vocab sizes).

    Returns:
        ``(subjects, attributes)`` lists of natural-key strings for the task.
    """
    n_subj = config.n_subjects_per_task
    n_attr = config.n_attributes
    t = task_index
    if scenario is CLScenario.DOMAIN_INCREMENTAL:
        subjects = [f"dom{t}_subj_{k}" for k in range(n_subj)]
        attributes = [f"attr_{a}" for a in range(n_attr)]
    elif scenario is CLScenario.CLASS_INCREMENTAL:
        subjects = [f"subj_{t}_{k}" for k in range(n_subj)]
        attributes = [f"attr_t{t}_{a}" for a in range(n_attr)]
    else:  # TASK_INCREMENTAL (default)
        subjects = [f"subj_{t}_{k}" for k in range(n_subj)]
        attributes = [f"attr_{a}" for a in range(n_attr)]
    return subjects, attributes


def _sample_value_index(
    rng: np.random.Generator, n_values: int, drift: float, frac: float
) -> int:
    """Sample a value index in ``[0, n_values)`` with optional upward drift.

    ``drift`` biases the sampled index upward as a task progresses through the
    stream (FR1.3). Concretely the index is drawn uniformly from
    ``[floor(drift * frac * n_values), n_values)``, so the lower bound climbs
    with the task fraction ``frac`` (0 for the first task, 1 for the last). At
    ``drift == 0.0`` the lower bound is 0 and the distribution is uniform.

    Args:
        rng: The ``"values"`` random generator.
        n_values: Size of the value vocabulary (``>= 2``).
        drift: Drift strength in ``[0, 1]``.
        frac: The task's position fraction in ``[0, 1]``.

    Returns:
        A value index in ``[0, n_values)``.
    """
    lo = drift * frac
    u = float(rng.uniform(lo, 1.0))
    v = int(u * n_values)
    if v >= n_values:
        v = n_values - 1
    return v


def _n_contradictions(rate: float, n_signal: int) -> int:
    """Return how many of a task's signal items are contradictions (FR1.3).

    A contradiction re-asserts an already-introduced probed key with a new value,
    so at least one signal item must remain a fresh introduction. When
    ``rate > 0`` and the task has ``>= 2`` signal items the count is forced to at
    least one so the knob *visibly* takes effect; at ``rate == 0.0`` it is zero
    (every probed key is asserted exactly once).

    Args:
        rate: The configured ``contradiction_rate`` in ``[0, 1]``.
        n_signal: Number of signal items in the task.

    Returns:
        The number of contradiction items, in ``[0, max(0, n_signal - 1)]``.
    """
    if rate <= 0.0 or n_signal <= 1:
        return 0
    n = int(round(rate * n_signal))
    if n < 1:
        n = 1
    if n > n_signal - 1:
        n = n_signal - 1
    return n


def _choose_contradiction_flags(
    n_signal: int,
    n_contra: int,
    recency_bias: float,
    rng: np.random.Generator,
) -> list[bool]:
    """Flag which signal items (by within-task signal index) are contradictions.

    Signal index 0 is always a fresh introduction (it cannot contradict anything
    yet). The remaining indices are scored by
    ``(1 - recency_bias) * uniform + recency_bias * (index / (n_signal - 1))``
    and the top ``n_contra`` scores become contradictions. With
    ``recency_bias == 0.0`` selection is uniform; as ``recency_bias`` rises the
    positional term dominates and contradictions cluster toward the end of the
    task (FR1.3). The number of random draws is independent of ``recency_bias``,
    preserving determinism across knob settings for a fixed config.

    Args:
        n_signal: Number of signal items in the task.
        n_contra: Number of contradictions to place.
        recency_bias: Clustering strength in ``[0, 1]``.
        rng: The ``"order"`` random generator.

    Returns:
        A list of length ``n_signal`` of booleans (``True`` => contradiction).
    """
    flags = [False] * n_signal
    if n_contra <= 0 or n_signal <= 1:
        return flags
    candidates = list(range(1, n_signal))  # index 0 must introduce a key
    noise = rng.uniform(0.0, 1.0, size=len(candidates))
    denom = float(n_signal - 1)
    positional = np.array([i / denom for i in candidates], dtype=float)
    scores = (1.0 - recency_bias) * noise + recency_bias * positional
    ranked = np.argsort(-scores, kind="stable")
    for r in ranked[:n_contra]:
        flags[candidates[int(r)]] = True
    return flags


def generate_stream(config: StreamGenConfig, seed: int) -> Stream:
    """Generate a deterministic synthetic continual task stream (WS1).

    Produces ``config.n_tasks * config.items_per_task`` ingest items partitioned
    into contiguous task segments. Each item is assigned exactly one OFFLINE-ONLY
    relevance label (stored in :class:`~slow_wave.stream.schema.GroundTruth`, not
    on the item) and a surface form: signal/distractor items assert a
    ``(subject, attribute, value)`` fact, noise items carry a random token salad
    with ``fact=None``.

    The output is fully determined by ``(config, seed)``: all randomness flows
    from ``numpy.random.default_rng(derive_seed(seed, "<purpose>"))`` and the
    stream id is a stable hash, so two calls yield an identical :class:`Stream`
    and byte-identical canonical JSON (Phase 1 exit criterion #1).

    Args:
        config: The generation knobs (scenario, sizes, label mix, regime knobs).
        seed: The integer master seed for this stream.

    Returns:
        A validated :class:`~slow_wave.stream.schema.Stream` tagged with
        ``config.scenario`` and carrying one label per item.

    Raises:
        ValueError: If a task needs more distinct probed keys than the configured
            ``n_subjects_per_task * n_attributes`` vocabulary can supply (raise
            the vocab sizes, raise ``contradiction_rate``, or lower the signal
            proportion).
    """
    scenario = config.scenario
    n_tasks = config.n_tasks
    items_per_task = config.items_per_task
    n_values = config.n_values
    noise_vocab = config.noise_vocab_size
    noise_tokens = config.noise_tokens
    distractor_size = config.distractor_namespace_size

    rng_labels = np.random.default_rng(derive_seed(seed, "labels"))
    rng_subjects = np.random.default_rng(derive_seed(seed, "subjects"))
    rng_values = np.random.default_rng(derive_seed(seed, "values"))
    rng_order = np.random.default_rng(derive_seed(seed, "order"))
    rng_noise = np.random.default_rng(derive_seed(seed, "noise"))

    items: list[StreamItem] = []
    labels: dict[str, Label] = {}

    for task_index in range(n_tasks):
        frac = task_index / (n_tasks - 1) if n_tasks > 1 else 0.0
        subjects, attributes = _scenario_namespace(scenario, task_index, config)
        n_attr = len(attributes)

        # 1) Label counts (largest remainder) and deterministic placement.
        counts = _largest_remainder(config.label_mix, items_per_task)
        label_pool = (
            [Label.SIGNAL] * counts[Label.SIGNAL]
            + [Label.DISTRACTOR] * counts[Label.DISTRACTOR]
            + [Label.NOISE] * counts[Label.NOISE]
        )
        perm = rng_labels.permutation(items_per_task)
        label_seq = [label_pool[int(i)] for i in perm]

        # 2) Plan signal facts: which signals are fresh vs. contradictions.
        signal_positions = [
            p for p in range(items_per_task) if label_seq[p] == Label.SIGNAL
        ]
        n_signal = len(signal_positions)
        n_contra = _n_contradictions(config.contradiction_rate, n_signal)
        contra_flags = _choose_contradiction_flags(
            n_signal, n_contra, config.recency_bias, rng_order
        )
        n_fresh = n_signal - n_contra

        key_pool = [(s, a) for s in subjects for a in attributes]
        if n_fresh > len(key_pool):
            raise ValueError(
                f"task {task_index}: need {n_fresh} distinct probed keys but only "
                f"{len(key_pool)} are available "
                f"(n_subjects_per_task * n_attributes). Increase the vocab sizes, "
                f"raise contradiction_rate, or lower the signal proportion."
            )
        key_perm = rng_subjects.permutation(len(key_pool))
        fresh_keys = [key_pool[int(i)] for i in key_perm[:n_fresh]]
        fresh_cursor = 0

        signal_index = {p: si for si, p in enumerate(signal_positions)}
        introduced_value: dict[tuple[str, str], int] = {}
        introduced_order: list[tuple[str, str]] = []

        # 3) Emit items in position order (stable RNG draw order).
        for p in range(items_per_task):
            order = task_index * items_per_task + p
            item_id = f"i{order:06d}"
            label = label_seq[p]

            if label == Label.SIGNAL:
                if contra_flags[signal_index[p]]:
                    pick = int(rng_subjects.integers(0, len(introduced_order)))
                    key = introduced_order[pick]
                    v = _sample_value_index(
                        rng_values, n_values, config.drift, frac
                    )
                    if v == introduced_value[key]:
                        v = (v + 1) % n_values  # guarantee a real contradiction
                    introduced_value[key] = v
                    subject, attribute = key
                else:
                    subject, attribute = fresh_keys[fresh_cursor]
                    fresh_cursor += 1
                    v = _sample_value_index(
                        rng_values, n_values, config.drift, frac
                    )
                    introduced_value[(subject, attribute)] = v
                    introduced_order.append((subject, attribute))
                value = f"val_{v}"
                fact: Fact | None = Fact(
                    subject=subject, attribute=attribute, value=value
                )
                content = f"The {attribute} of {subject} is {value}."

            elif label == Label.DISTRACTOR:
                d = int(rng_subjects.integers(0, distractor_size))
                subject = f"dsubj_{d}"
                attribute = attributes[int(rng_subjects.integers(0, n_attr))]
                v = _sample_value_index(rng_values, n_values, config.drift, frac)
                value = f"val_{v}"
                fact = Fact(subject=subject, attribute=attribute, value=value)
                content = f"The {attribute} of {subject} is {value}."

            else:  # Label.NOISE
                toks = rng_noise.integers(0, noise_vocab, size=noise_tokens)
                content = " ".join(f"tok_{int(x)}" for x in toks)
                fact = None

            items.append(
                StreamItem(
                    item_id=item_id,
                    order=order,
                    kind=ItemKind.INGEST,
                    task_index=task_index,
                    content=content,
                    fact=fact,
                )
            )
            labels[item_id] = label

    digest_src = json.dumps(
        {"seed": seed, "config": config.canonical_dict()}, sort_keys=True
    )
    h = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:8]
    stream_id = f"{scenario.value}-s{seed}-{h}"

    return Stream(
        stream_id=stream_id,
        scenario=scenario,
        seed=seed,
        n_tasks=n_tasks,
        config=config,
        items=items,
        ground_truth=GroundTruth(labels=labels),
    )
