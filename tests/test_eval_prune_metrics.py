"""Tests for slow_wave.eval.prune_metrics (Phase 4, WS-METRICS).

These pin the exact prune-quality definitions from the Phase 4 contract
(``docs/PHASE4_CONTRACT.md`` — WS-METRICS, EC5). The positive class is PRUNED: a
correct prune targets a ``distractor``/``noise`` item and a ``signal`` item
should be retained.

Most assertions use **fully hand-built** ``Stream`` + ``MemorySubstrate``
fixtures with known labels and a known retained/pruned partition, so every metric
is checked against an exact number. One test additionally drives a realistic
substrate via :func:`slow_wave.stream.generator.generate_stream`,
:func:`slow_wave.stream.probes.build_probe_set`, and a tiny
:class:`slow_wave.agent.wake.WakeAgent` (hash embedder) to confirm the metrics are
well-formed, JSON-dumpable, and deterministic on real data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from slow_wave.agent.wake import WakeAgent
from slow_wave.config import (
    AgentConfig,
    Config,
    EmbeddingConfig,
    MemoryConfig,
)
from slow_wave.embeddings import get_embedder
from slow_wave.eval import prune_metrics
from slow_wave.eval.prune_metrics import (
    calibration_curve,
    prune_quality,
    retained_item_ids,
)
from slow_wave.eval.schema import CalibrationCurve, PruneQuality
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.repro.seeding import derive_seed
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set
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

DIM = 8


# --------------------------------------------------------------------------- #
# Hand-built fixtures
# --------------------------------------------------------------------------- #
def _vec(i: int) -> np.ndarray:
    """A deterministic unit one-hot ``(DIM,)`` float32 vector."""
    vec = np.zeros(DIM, dtype=np.float32)
    vec[i % DIM] = 1.0
    return vec


def _build_stream(labels: list[Label]) -> Stream:
    """Build a single-task stream with one item per requested label, in order."""
    items: list[StreamItem] = []
    ground_truth: dict[str, Label] = {}
    for order, label in enumerate(labels):
        item_id = f"i{order:06d}"
        items.append(
            StreamItem(
                item_id=item_id,
                order=order,
                kind=ItemKind.INGEST,
                task_index=0,
                content=f"fact {order}",
                fact=Fact(subject=f"s{order:03d}", attribute="att", value=f"v{order:03d}"),
            )
        )
        ground_truth[item_id] = label
    return Stream(
        stream_id="hand-built",
        scenario=CLScenario.TASK_INCREMENTAL,
        seed=0,
        n_tasks=1,
        config=StreamGenConfig(n_tasks=1, items_per_task=max(1, len(labels))),
        items=items,
        ground_truth=GroundTruth(labels=ground_truth),
    )


def _empty_substrate() -> MemorySubstrate:
    """An empty, unbounded substrate (nothing evicts on insert)."""
    return MemorySubstrate(
        MemoryConfig(episodic_capacity=0, archival_enabled=True), dim=DIM
    )


def _episodic_entry(item: StreamItem, importance: float = 1.0) -> MemoryEntry:
    """An episodic entry whose provenance points back at ``item``."""
    return MemoryEntry(
        entry_id=f"e{item.order:06d}",
        tier=MemoryTier.EPISODIC,
        content=item.content,
        fact=item.fact,
        created_order=item.order,
        salience=SalienceMeta(importance=importance, recency_order=item.order),
        provenance=(item.item_id,),
    )


def _retain_episodic(sub: MemorySubstrate, item: StreamItem, importance: float = 1.0) -> None:
    """Place a live episodic representation of ``item`` into active memory."""
    sub.episodic.append(_episodic_entry(item, importance), _vec(item.order), now_order=item.order)


def _prune_to_archival(sub: MemorySubstrate, item: StreamItem) -> None:
    """Demote ``item``'s representation to the archival tier (a pruned item)."""
    sub.archival.demote(
        _episodic_entry(item), _vec(item.order), reason="test_prune", at_order=item.order
    )


# --------------------------------------------------------------------------- #
# retained_item_ids — the exact retention rule
# --------------------------------------------------------------------------- #
def test_retention_rule_episodic_semantic_and_augment_exclusion() -> None:
    """Episodic provenance + semantic created_order retain; augment pseudo does not."""
    stream = _build_stream([Label.SIGNAL, Label.SIGNAL, Label.DISTRACTOR])
    sub = _empty_substrate()

    # Episodic entry for item 0 -> retained.
    _retain_episodic(sub, stream.items[0])

    # Augment pseudo-episode: provenance[0] is an entry_id, NOT a stream item_id.
    sub.episodic.append(
        MemoryEntry(
            entry_id="p000999",
            tier=MemoryTier.EPISODIC,
            content="pseudo",
            fact=None,
            created_order=999,
            salience=SalienceMeta(importance=1.0, recency_order=999),
            provenance=("e000005",),  # an entry_id, never a stream item
        ),
        _vec(5),
        now_order=999,
    )

    # Semantic entry whose created_order maps to item 1 (TRANSFER preserves order).
    sub.semantic.upsert(
        MemoryEntry(
            entry_id="sem-1",
            tier=MemoryTier.SEMANTIC,
            content="consolidated fact 1",
            fact=stream.items[1].fact,
            created_order=1,
            salience=SalienceMeta(importance=1.0, recency_order=1),
            provenance=("e000001",),
        ),
        _vec(1),
        now_order=10,
    )

    assert retained_item_ids(stream, sub) == {"i000000", "i000001"}


# --------------------------------------------------------------------------- #
# prune_quality — exact numbers
# --------------------------------------------------------------------------- #
def test_perfect_prune_precision_recall_f1_and_signal_retention() -> None:
    """Exactly the distractor/noise pruned + all signals retained -> all 1.0."""
    labels = [
        Label.SIGNAL,
        Label.SIGNAL,
        Label.SIGNAL,
        Label.DISTRACTOR,
        Label.DISTRACTOR,
        Label.NOISE,
    ]
    stream = _build_stream(labels)
    sub = _empty_substrate()
    for item, label in zip(stream.items, labels):
        if label is Label.SIGNAL:
            _retain_episodic(sub, item)
        else:
            _prune_to_archival(sub, item)

    pq = prune_quality(stream, sub)

    assert pq.precision == 1.0
    assert pq.recall == 1.0
    assert pq.f1 == 1.0
    assert pq.signal_retention == 1.0
    assert (pq.tp, pq.fp, pq.fn, pq.tn) == (3, 0, 0, 3)
    assert pq.n_retained == 3
    assert pq.n_pruned == 3
    assert pq.retained_by_label.signal == 3
    assert pq.retained_by_label.distractor == 0
    assert pq.retained_by_label.noise == 0
    assert pq.pruned_by_label.distractor == 2
    assert pq.pruned_by_label.noise == 1
    assert pq.pruned_by_label.signal == 0


def test_opposite_prune_yields_zero_recall() -> None:
    """Signals pruned, distractor/noise kept -> recall == 0 (and precision/f1 0)."""
    labels = [
        Label.SIGNAL,
        Label.SIGNAL,
        Label.DISTRACTOR,
        Label.NOISE,
    ]
    stream = _build_stream(labels)
    sub = _empty_substrate()
    for item, label in zip(stream.items, labels):
        if label is Label.SIGNAL:
            _prune_to_archival(sub, item)  # wrongly pruned signal
        else:
            _retain_episodic(sub, item)  # wrongly kept distractor/noise

    pq = prune_quality(stream, sub)

    assert pq.recall == 0.0
    assert pq.precision == 0.0
    assert pq.f1 == 0.0
    assert pq.signal_retention == 0.0
    assert (pq.tp, pq.fp, pq.fn, pq.tn) == (0, 2, 2, 0)


def test_count_invariants_hold_on_a_mixed_partition() -> None:
    """tp+fp+fn+tn == n_items; tp+fp == n_pruned; tp+fn == #distractor+#noise."""
    labels = [
        Label.SIGNAL,
        Label.SIGNAL,
        Label.SIGNAL,
        Label.DISTRACTOR,
        Label.DISTRACTOR,
        Label.NOISE,
        Label.NOISE,
    ]
    stream = _build_stream(labels)
    sub = _empty_substrate()
    # A deliberately imperfect partition: keep one distractor, prune one signal.
    retain_orders = {0, 1, 3}  # signal, signal, distractor
    for item in stream.items:
        if item.order in retain_orders:
            _retain_episodic(sub, item)
        else:
            _prune_to_archival(sub, item)

    pq = prune_quality(stream, sub)
    n_items = len(stream.items)
    n_neg = sum(1 for label in labels if label in (Label.DISTRACTOR, Label.NOISE))

    assert pq.tp + pq.fp + pq.fn + pq.tn == n_items
    assert pq.tp + pq.fp == pq.n_pruned
    assert pq.tn + pq.fn == pq.n_retained
    assert pq.tp + pq.fn == n_neg
    # retained = {signal, signal, distractor}; pruned = {signal, distractor, noise, noise}
    assert (pq.tp, pq.fp, pq.fn, pq.tn) == (3, 1, 1, 2)


def test_empty_substrate_is_safe_and_prunes_everything() -> None:
    """An empty substrate prunes all items without raising; recall over neg == 1."""
    labels = [Label.SIGNAL, Label.DISTRACTOR, Label.NOISE]
    stream = _build_stream(labels)
    sub = _empty_substrate()

    pq = prune_quality(stream, sub)

    assert pq.n_retained == 0
    assert pq.n_pruned == 3
    assert pq.tp == 2  # both distractor + noise pruned
    assert pq.fp == 1  # the signal was (wrongly) pruned too
    assert pq.recall == 1.0
    assert pq.signal_retention == 0.0


# --------------------------------------------------------------------------- #
# Labels are read ONLY via offline_labels
# --------------------------------------------------------------------------- #
def test_prune_quality_reads_labels_only_via_offline_labels(monkeypatch) -> None:
    """Probe the code path: the offline_labels accessor is the sole label source."""
    labels = [Label.SIGNAL, Label.DISTRACTOR, Label.NOISE]
    stream = _build_stream(labels)
    sub = _empty_substrate()
    _retain_episodic(sub, stream.items[0])

    calls: list[str] = []
    real = prune_metrics.offline_labels

    def _spy(s: Stream):
        calls.append(s.stream_id)
        return real(s)

    monkeypatch.setattr(prune_metrics, "offline_labels", _spy)

    pq = prune_quality(stream, sub)
    cc = calibration_curve(stream, sub)

    # Both offline scorers routed through offline_labels and nothing else.
    assert calls == ["hand-built", "hand-built"]
    assert pq.tn == 1  # the retained signal
    assert cc.n_items == 3


# --------------------------------------------------------------------------- #
# calibration_curve
# --------------------------------------------------------------------------- #
def test_calibration_bins_sum_and_ece_in_unit_interval_and_monotone() -> None:
    """Bins sum to n_items, ECE in [0,1], frac_signal rises with salience."""
    # High-salience signals, low-salience distractors -> well-calibrated.
    labels = [Label.SIGNAL] * 4 + [Label.DISTRACTOR] * 4
    stream = _build_stream(labels)
    sub = _empty_substrate()
    for item, label in zip(stream.items, labels):
        importance = 1.0 if label is Label.SIGNAL else 0.2
        _retain_episodic(sub, item, importance=importance)

    cc = calibration_curve(stream, sub, n_bins=10)

    assert sum(b.n for b in cc.bins) == cc.n_items == len(stream.items)
    assert len(cc.bins) == 10
    assert 0.0 <= cc.expected_calibration_error <= 1.0

    nonempty = [b for b in cc.bins if b.n > 0]
    assert len(nonempty) >= 2
    # The highest-salience populated bin is more signal-dense than the lowest.
    assert nonempty[-1].frac_signal > nonempty[0].frac_signal
    # frac_signal is non-decreasing across populated bins (well-calibrated).
    fracs = [b.frac_signal for b in nonempty]
    assert fracs == sorted(fracs)


def test_calibration_empty_stream_is_a_well_formed_zero_curve() -> None:
    """An empty stream yields a zeroed curve with the requested bin count."""
    stream = _build_stream([])  # no items
    sub = _empty_substrate()

    cc = calibration_curve(stream, sub, n_bins=5)

    assert isinstance(cc, CalibrationCurve)
    assert cc.n_items == 0
    assert cc.expected_calibration_error == 0.0
    assert len(cc.bins) == 5
    assert all(b.n == 0 for b in cc.bins)
    assert sum(b.n for b in cc.bins) == cc.n_items


def test_calibration_empty_substrate_is_safe() -> None:
    """No active representations -> all items score 0, no raise, bins still sum."""
    labels = [Label.SIGNAL, Label.DISTRACTOR, Label.NOISE]
    stream = _build_stream(labels)
    sub = _empty_substrate()

    cc = calibration_curve(stream, sub, n_bins=4)

    assert cc.n_items == 3
    assert sum(b.n for b in cc.bins) == 3
    assert 0.0 <= cc.expected_calibration_error <= 1.0


# --------------------------------------------------------------------------- #
# JSON-dumpability + determinism
# --------------------------------------------------------------------------- #
def test_results_are_json_dumpable_and_deterministic() -> None:
    """Hand-built results round-trip to JSON and are byte-identical across calls."""
    labels = [Label.SIGNAL, Label.SIGNAL, Label.DISTRACTOR, Label.NOISE]
    stream = _build_stream(labels)
    sub = _empty_substrate()
    _retain_episodic(sub, stream.items[0], importance=0.9)
    _retain_episodic(sub, stream.items[1], importance=0.4)
    _prune_to_archival(sub, stream.items[2])
    _prune_to_archival(sub, stream.items[3])

    pq1 = prune_quality(stream, sub)
    pq2 = prune_quality(stream, sub)
    cc1 = calibration_curve(stream, sub)
    cc2 = calibration_curve(stream, sub)

    pq_json = json.dumps(pq1.model_dump(mode="json"), sort_keys=True)
    cc_json = json.dumps(cc1.model_dump(mode="json"), sort_keys=True)
    assert pq_json == json.dumps(pq2.model_dump(mode="json"), sort_keys=True)
    assert cc_json == json.dumps(cc2.model_dump(mode="json"), sort_keys=True)
    # Sanity: the dumped objects parse back.
    assert json.loads(pq_json)["precision"] == pq1.precision
    assert json.loads(cc_json)["n_items"] == cc1.n_items


# --------------------------------------------------------------------------- #
# Realistic stream + substrate via the generator + WakeAgent
# --------------------------------------------------------------------------- #
def _realistic_substrate(seed: int) -> tuple[Stream, MemorySubstrate]:
    """Run a tiny no-sleep WakeAgent (hash embedder) over a distractor-heavy stream."""
    cfg = Config(
        experiment="prune-metrics-realistic",
        seed=seed,
        embedding=EmbeddingConfig(backend="hash", dim=64),
        stream=StreamGenConfig(
            scenario=CLScenario.TASK_INCREMENTAL,
            n_tasks=3,
            items_per_task=12,
            label_mix=LabelMix(signal=0.34, distractor=0.40, noise=0.26),
            n_subjects_per_task=5,
            n_attributes=2,
            n_values=10,
            probes_per_task=3,
        ),
        memory=MemoryConfig(episodic_capacity=10, archival_enabled=True),
        agent=AgentConfig(reasoning_calls="off"),  # no LLM calls at all
    )
    stream = generate_stream(cfg.stream, derive_seed(cfg.seed, "stream"))
    probe_set = build_probe_set(stream)
    embedder = get_embedder(cfg)
    result = WakeAgent(cfg, embedder).run(stream, probe_set)
    return stream, result.substrate


def test_realistic_run_metrics_are_well_formed_and_deterministic() -> None:
    """On a real bounded run: invariants hold, results are deterministic + JSON-able."""
    stream_a, sub_a = _realistic_substrate(seed=7)
    pq = prune_quality(stream_a, sub_a)
    cc = calibration_curve(stream_a, sub_a)

    n_items = len(stream_a.items)
    # The bounded episodic store must have forced some forgetting.
    assert pq.n_pruned > 0
    assert pq.tp + pq.fp + pq.fn + pq.tn == n_items
    assert pq.tp + pq.fp == pq.n_pruned
    assert 0.0 <= pq.precision <= 1.0
    assert 0.0 <= pq.recall <= 1.0
    assert 0.0 <= pq.signal_retention <= 1.0
    assert sum(b.n for b in cc.bins) == cc.n_items == n_items
    assert 0.0 <= cc.expected_calibration_error <= 1.0

    # Deterministic: a second identical run yields byte-identical JSON.
    stream_b, sub_b = _realistic_substrate(seed=7)
    pq_b = prune_quality(stream_b, sub_b)
    cc_b = calibration_curve(stream_b, sub_b)
    assert json.dumps(pq.model_dump(mode="json"), sort_keys=True) == json.dumps(
        pq_b.model_dump(mode="json"), sort_keys=True
    )
    assert json.dumps(cc.model_dump(mode="json"), sort_keys=True) == json.dumps(
        cc_b.model_dump(mode="json"), sort_keys=True
    )

    assert isinstance(pq, PruneQuality)
    assert isinstance(cc, CalibrationCurve)
