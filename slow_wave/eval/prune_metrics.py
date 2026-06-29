"""Mechanism-level prune metrics for the Phase 4 eval harness (WS-METRICS).

This module is the bench's *superpower* metric layer (FR5.3): it scores how well
a run **consolidated** memory — what it kept versus what it threw away — against
the synthetic stream's known ground-truth relevance, **decoupled** from any
downstream probe accuracy (the two can and should diverge). It answers "did the
arm prune the right things?" rather than "did the arm answer correctly?".

This is **offline scoring**, so — and *only* here, alongside the oracle arm — it
is sanctioned to read ground-truth labels, and only via
:func:`slow_wave.stream.schema.offline_labels` (the confound guard, FR1.6). No
online path, ``stream.ground_truth``, ``probe.answer``, or banned field is ever
touched directly.

Design principles
-----------------
* **Positive class = PRUNED (FR5.3).** A *correct* prune targets a
  ``distractor``/``noise`` item; a ``signal`` item should be *retained*. So a
  true positive is a distractor/noise item that was pruned, a false positive is a
  signal that was wrongly pruned, a false negative is a distractor/noise item
  wrongly kept, and a true negative is a signal correctly kept. ``precision`` and
  ``recall`` are the pruning quality; ``signal_retention`` is its protective
  complement (a good arm keeps signal *and* prunes noise).
* **Retention is structural, not similarity-based.** An item is *retained* iff it
  has a live representation in an **active** tier (episodic or semantic); archival
  ("forgotten") and never-consolidated items are *pruned*. The mapping back to a
  source stream item is exact (episodic provenance / semantic ``created_order``),
  so the metric is byte-identical across runs (DX1).
* **Degenerate inputs never raise.** An empty substrate, an empty stream, or a
  stream with no items of some label yields a well-formed zero result (precision/
  recall/F1 default to ``0.0``; the calibration curve is a zeroed reliability
  diagram) rather than dividing by zero.
* **numpy + stdlib + pydantic + slow_wave only.** No scipy/sklearn/matplotlib.
"""

from __future__ import annotations

import numpy as np

from slow_wave.eval.schema import (
    CalibrationBin,
    CalibrationCurve,
    LabelCount,
    PruneQuality,
)
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Label, Stream, offline_labels


def _active_importances_by_item(
    stream: Stream, substrate: MemorySubstrate
) -> dict[str, list[float]]:
    """Map each stream item to the salience of its LIVE active representations.

    The shared core of :func:`retained_item_ids` and :func:`calibration_curve`.
    Walks the **active** tiers (episodic + semantic, never archival) and resolves
    each live entry back to the source stream ``item_id`` it represents, using the
    same deterministic rule both public functions rely on:

    * an **episodic** entry maps to ``provenance[0]`` when that is a real stream
      ``item_id`` — which excludes generative-augment pseudo-episodes, whose
      ``provenance[0]`` is a source ``entry_id`` rather than an item id;
    * a **semantic** entry maps via ``order_to_item[created_order]`` because the
      TRANSFER operator preserves the source item's stream order.

    Args:
        stream: The scored stream (its items define the valid id/order space).
        substrate: The memory substrate after a run.

    Returns:
        A ``{item_id: [importance, ...]}`` dict holding, per retained item, the
        ``salience.importance`` of every active entry that represents it. Items
        with no active representation are absent (i.e. pruned).
    """
    valid = {it.item_id for it in stream.items}
    order_to_item = {it.order: it.item_id for it in stream.items}

    by_item: dict[str, list[float]] = {}
    for entry in substrate.episodic.all_entries():
        if entry.provenance and entry.provenance[0] in valid:
            by_item.setdefault(entry.provenance[0], []).append(
                float(entry.salience.importance)
            )
    for entry in substrate.semantic.all_entries():
        item_id = order_to_item.get(entry.created_order)
        if item_id is not None:
            by_item.setdefault(item_id, []).append(float(entry.salience.importance))
    return by_item


def retained_item_ids(stream: Stream, substrate: MemorySubstrate) -> set[str]:
    """Return the ids of stream items with a LIVE representation in active memory.

    An item is *retained* iff at least one entry in an **active** tier (episodic
    OR semantic — never archival) traces back to it. The mapping is deterministic:

    * **episodic**: an entry maps to ``provenance[0]`` when that is a real stream
      ``item_id`` (so augment pseudo-episodes, whose ``provenance[0]`` is an
      ``entry_id``, are *not* counted as retaining any item);
    * **semantic**: an entry maps to the item at ``order_to_item[created_order]``
      (TRANSFER preserves the source item's stream order).

    Args:
        stream: The scored stream.
        substrate: The memory substrate after a run.

    Returns:
        The set of retained stream ``item_id``\\ s (a subset of the stream's
        items). Everything else in the stream is considered pruned.
    """
    return set(_active_importances_by_item(stream, substrate))


def prune_quality(stream: Stream, substrate: MemorySubstrate) -> PruneQuality:
    """Score consolidation quality against ground truth (FR5.3, offline).

    Over **all** stream items (each carries exactly one ground-truth label),
    partitions them into retained (:func:`retained_item_ids`) and pruned (the
    rest), then scores the pruning with the positive class = PRUNED:

    * ``tp`` — pruned ``distractor``/``noise`` (correct prunes);
    * ``fp`` — pruned ``signal`` (wrongly pruned);
    * ``fn`` — retained ``distractor``/``noise`` (wrongly kept);
    * ``tn`` — retained ``signal`` (correctly kept);
    * ``precision = tp / (tp + fp)``, ``recall = tp / (tp + fn)``,
      ``f1 = 2PR / (P + R)``, ``signal_retention = tn / (tn + fp)`` — each
      ``0.0`` when its denominator is 0.

    By construction ``tp + fp + fn + tn == n_items``, ``tp + fp == n_pruned``,
    and ``tp + fn == #distractor + #noise``.

    Labels are read **only** via
    :func:`slow_wave.stream.schema.offline_labels` (FR1.6).

    Args:
        stream: The scored stream (supplies items + offline labels).
        substrate: The memory substrate after a run.

    Returns:
        A :class:`~slow_wave.eval.schema.PruneQuality` with all counts, the
        per-label retained/pruned breakdowns, and the four quality scores. An
        empty substrate yields all-pruned counts; an empty stream yields zeros.
    """
    labels = offline_labels(stream)
    retained = retained_item_ids(stream, substrate)
    negative = {Label.DISTRACTOR, Label.NOISE}

    tp = fp = fn = tn = 0
    retained_counts = {Label.SIGNAL: 0, Label.DISTRACTOR: 0, Label.NOISE: 0}
    pruned_counts = {Label.SIGNAL: 0, Label.DISTRACTOR: 0, Label.NOISE: 0}

    for item in stream.items:
        label = labels[item.item_id]
        if item.item_id in retained:
            retained_counts[label] += 1
            if label in negative:
                fn += 1  # distractor/noise wrongly kept
            else:
                tn += 1  # signal correctly kept
        else:
            pruned_counts[label] += 1
            if label in negative:
                tp += 1  # distractor/noise correctly pruned
            else:
                fp += 1  # signal wrongly pruned

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    signal_retention = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return PruneQuality(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_retained=tn + fn,
        n_pruned=tp + fp,
        retained_by_label=LabelCount(
            signal=retained_counts[Label.SIGNAL],
            distractor=retained_counts[Label.DISTRACTOR],
            noise=retained_counts[Label.NOISE],
        ),
        pruned_by_label=LabelCount(
            signal=pruned_counts[Label.SIGNAL],
            distractor=pruned_counts[Label.DISTRACTOR],
            noise=pruned_counts[Label.NOISE],
        ),
        signal_retention=signal_retention,
    )


def calibration_curve(
    stream: Stream, substrate: MemorySubstrate, *, n_bins: int = 10
) -> CalibrationCurve:
    """Build the salience-vs-true-relevance reliability curve (FR5.3, offline).

    A reliability diagram for consolidation: each stream item gets a *salience
    score* — the max ``salience.importance`` over its active representations, or
    ``0.0`` if it has none — min-max normalized to ``[0, 1]`` across items. Items
    are dropped into ``n_bins`` equal-width ``[0, 1]`` bins; each bin reports its
    count, mean normalized salience, and the fraction of its items that are truly
    ``signal``. A well-calibrated consolidation has ``frac_signal`` rising with
    salience. The expected calibration error is the count-weighted mean gap
    ``sum_b (n_b / N) * |mean_salience_b - frac_signal_b|`` (in ``[0, 1]``).

    Labels are read **only** via
    :func:`slow_wave.stream.schema.offline_labels` (FR1.6).

    Args:
        stream: The scored stream.
        substrate: The memory substrate after a run.
        n_bins: Number of equal-width salience bins (coerced to ``>= 1``).

    Returns:
        A :class:`~slow_wave.eval.schema.CalibrationCurve` whose bin counts sum to
        ``n_items``. An empty stream / empty substrate yields a well-formed zeroed
        curve (no raise).
    """
    n_bins = max(1, int(n_bins))
    labels = offline_labels(stream)
    by_item = _active_importances_by_item(stream, substrate)

    raw_scores: list[float] = []
    is_signal: list[bool] = []
    for item in stream.items:
        importances = by_item.get(item.item_id)
        raw_scores.append(max(importances) if importances else 0.0)
        is_signal.append(labels[item.item_id] == Label.SIGNAL)

    n_items = len(raw_scores)
    if n_items == 0:
        bins = [
            CalibrationBin(
                lo=b / n_bins,
                hi=(b + 1) / n_bins,
                n=0,
                mean_salience=0.0,
                frac_signal=0.0,
            )
            for b in range(n_bins)
        ]
        return CalibrationCurve(
            bins=bins, n_items=0, expected_calibration_error=0.0
        )

    scores = np.asarray(raw_scores, dtype=float)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi > lo:
        normalized = (scores - lo) / (hi - lo)
    else:
        normalized = np.zeros_like(scores)

    bin_norms: list[list[float]] = [[] for _ in range(n_bins)]
    bin_signal: list[list[bool]] = [[] for _ in range(n_bins)]
    for norm, signal in zip(normalized.tolist(), is_signal):
        idx = int(norm * n_bins)
        if idx >= n_bins:  # the last bin is inclusive of the upper edge (norm==1)
            idx = n_bins - 1
        elif idx < 0:
            idx = 0
        bin_norms[idx].append(float(norm))
        bin_signal[idx].append(signal)

    bins: list[CalibrationBin] = []
    ece = 0.0
    for b in range(n_bins):
        n_b = len(bin_norms[b])
        if n_b > 0:
            mean_salience = sum(bin_norms[b]) / n_b
            frac_signal = sum(1 for s in bin_signal[b] if s) / n_b
            ece += (n_b / n_items) * abs(mean_salience - frac_signal)
        else:
            mean_salience = 0.0
            frac_signal = 0.0
        bins.append(
            CalibrationBin(
                lo=b / n_bins,
                hi=(b + 1) / n_bins,
                n=n_b,
                mean_salience=mean_salience,
                frac_signal=frac_signal,
            )
        )

    return CalibrationCurve(
        bins=bins, n_items=n_items, expected_calibration_error=float(ece)
    )
