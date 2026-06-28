"""Tests for slow_wave.stream.generator (Phase 1, WS1).

Covers Phase 1 exit criteria #1 (determinism), #2 (label mix), and #3 (scenario
tagging / no cross-scenario mixing), plus the FR1.3 regime knobs
(contradictions, drift, recency bias).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

from slow_wave.stream.generator import generate_stream
from slow_wave.stream.schema import (
    CLScenario,
    ItemKind,
    Label,
    LabelMix,
    StreamGenConfig,
    assert_same_scenario,
    offline_labels,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cfg(**overrides) -> StreamGenConfig:
    """Build a StreamGenConfig with test-friendly defaults plus overrides."""
    base = dict(
        scenario=CLScenario.TASK_INCREMENTAL,
        n_tasks=3,
        items_per_task=40,
        label_mix=LabelMix(signal=0.5, distractor=0.3, noise=0.2),
        n_subjects_per_task=16,
        n_attributes=4,
        n_values=12,
        distractor_namespace_size=64,
        noise_vocab_size=128,
        noise_tokens=8,
    )
    base.update(overrides)
    return StreamGenConfig(**base)


def _canonical_json(stream) -> str:
    """The byte-identical canonical JSON the contract pins for determinism."""
    return json.dumps(stream.model_dump(mode="json"), sort_keys=True)


def _signal_facts_by_key(stream):
    """Group signal items' facts by ``(subject, attribute)`` (offline view)."""
    labels = offline_labels(stream)
    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for item in stream.items:
        if labels[item.item_id] == Label.SIGNAL and item.fact is not None:
            by_key[item.fact.key()].append(item)
    return by_key


def _contradiction_orders(stream) -> list[int]:
    """Stream orders of signal items that re-assert an already-seen key."""
    labels = offline_labels(stream)
    seen: set[tuple[str, str]] = set()
    orders: list[int] = []
    for item in sorted(stream.items, key=lambda it: it.order):
        if labels[item.item_id] != Label.SIGNAL or item.fact is None:
            continue
        key = item.fact.key()
        if key in seen:
            orders.append(item.order)
        else:
            seen.add(key)
    return orders


# --------------------------------------------------------------------------- #
# Exit criterion #1 — determinism
# --------------------------------------------------------------------------- #
def test_same_seed_yields_equal_stream_and_byte_identical_json() -> None:
    """Two calls with the same (config, seed) are identical (exit #1)."""
    cfg = _cfg(contradiction_rate=0.3, drift=0.5, recency_bias=0.5)
    a = generate_stream(cfg, seed=7)
    b = generate_stream(cfg, seed=7)
    assert a == b
    assert _canonical_json(a) == _canonical_json(b)
    assert a.stream_id == b.stream_id


def test_different_seed_changes_stream() -> None:
    """A different seed changes both the id and the serialized stream (exit #1)."""
    cfg = _cfg(contradiction_rate=0.3, drift=0.5, recency_bias=0.5)
    a = generate_stream(cfg, seed=7)
    c = generate_stream(cfg, seed=8)
    assert a.stream_id != c.stream_id
    assert _canonical_json(a) != _canonical_json(c)


def test_stream_id_format() -> None:
    """stream_id is f'{scenario}-s{seed}-{h}' with an 8-hex suffix (exit #1)."""
    cfg = _cfg()
    stream = generate_stream(cfg, seed=42)
    prefix = f"{cfg.scenario.value}-s42-"
    assert stream.stream_id.startswith(prefix)
    suffix = stream.stream_id[len(prefix) :]
    assert len(suffix) == 8
    assert all(ch in "0123456789abcdef" for ch in suffix)


def test_basic_structure() -> None:
    """order, task_index, item_id, kind and item count follow the contract."""
    cfg = _cfg(n_tasks=4, items_per_task=10)
    stream = generate_stream(cfg, seed=1)
    assert len(stream.items) == cfg.n_tasks * cfg.items_per_task
    assert stream.n_tasks == cfg.n_tasks
    for expected_order, item in enumerate(stream.items):
        assert item.order == expected_order
        assert item.item_id == f"i{expected_order:06d}"
        assert item.kind == ItemKind.INGEST
        assert item.task_index == expected_order // cfg.items_per_task


# --------------------------------------------------------------------------- #
# Exit criterion #2 — labels
# --------------------------------------------------------------------------- #
def test_every_item_has_exactly_one_label() -> None:
    """Each item maps to exactly one label, with no orphan labels (exit #2)."""
    cfg = _cfg()
    stream = generate_stream(cfg, seed=3)
    labels = offline_labels(stream)
    item_ids = {it.item_id for it in stream.items}
    assert set(labels) == item_ids
    assert len(labels) == len(stream.items)
    assert all(isinstance(v, Label) for v in labels.values())


@pytest.mark.parametrize("distractor", [0.1, 0.3, 0.5])
def test_label_proportions_match_requested(distractor: float) -> None:
    """Empirical label mix matches the request across >=3 ratios (exit #2)."""
    signal = 0.4
    noise = 1.0 - signal - distractor
    cfg = _cfg(
        items_per_task=100,
        n_tasks=3,
        n_subjects_per_task=64,
        label_mix=LabelMix(signal=signal, distractor=distractor, noise=noise),
    )
    stream = generate_stream(cfg, seed=11)
    labels = offline_labels(stream)
    total = len(labels)
    counts = Counter(v.value for v in labels.values())
    requested = {"signal": signal, "distractor": distractor, "noise": noise}
    for name, want in requested.items():
        got = counts.get(name, 0) / total
        assert abs(got - want) <= 0.02, f"{name}: got {got}, want {want}"


def test_largest_remainder_counts_sum_exactly() -> None:
    """Largest-remainder allocation sums to items_per_task even when fractional."""
    # 0.5/0.3/0.2 * 7 = 3.5/2.1/1.4 -> floors 3/2/1 (=6), leftover -> signal.
    cfg = _cfg(
        items_per_task=7,
        n_tasks=5,
        label_mix=LabelMix(signal=0.5, distractor=0.3, noise=0.2),
    )
    stream = generate_stream(cfg, seed=5)
    labels = offline_labels(stream)
    per_task: dict[int, Counter] = defaultdict(Counter)
    by_id = {it.item_id: it for it in stream.items}
    for item_id, label in labels.items():
        per_task[by_id[item_id].task_index][label.value] += 1
    assert len(per_task) == cfg.n_tasks
    for task_counts in per_task.values():
        assert sum(task_counts.values()) == cfg.items_per_task
        assert task_counts["signal"] == 4  # 3 + leftover unit (largest frac)
        assert task_counts["distractor"] == 2
        assert task_counts["noise"] == 1


# --------------------------------------------------------------------------- #
# Exit criterion #3 — scenario tagging and no cross-scenario mixing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scenario",
    [
        CLScenario.TASK_INCREMENTAL,
        CLScenario.DOMAIN_INCREMENTAL,
        CLScenario.CLASS_INCREMENTAL,
    ],
)
def test_scenario_tag_matches_request(scenario: CLScenario) -> None:
    """Stream.scenario equals the requested scenario for all three (exit #3)."""
    cfg = _cfg(scenario=scenario)
    stream = generate_stream(cfg, seed=2)
    assert stream.scenario == scenario


def test_assert_same_scenario_returns_and_raises() -> None:
    """assert_same_scenario returns on match and raises on mix (exit #3)."""
    a = generate_stream(_cfg(scenario=CLScenario.TASK_INCREMENTAL), seed=1)
    b = generate_stream(_cfg(scenario=CLScenario.TASK_INCREMENTAL), seed=2)
    assert assert_same_scenario([a, b]) == CLScenario.TASK_INCREMENTAL

    other = generate_stream(_cfg(scenario=CLScenario.DOMAIN_INCREMENTAL), seed=1)
    with pytest.raises(ValueError):
        assert_same_scenario([a, other])


@pytest.mark.parametrize(
    "scenario,subj_check,attr_check",
    [
        (
            CLScenario.TASK_INCREMENTAL,
            lambda s, t: s.startswith(f"subj_{t}_"),
            lambda a, t: a.startswith("attr_") and not a.startswith("attr_t"),
        ),
        (
            CLScenario.DOMAIN_INCREMENTAL,
            lambda s, t: s.startswith(f"dom{t}_subj_"),
            lambda a, t: a.startswith("attr_") and not a.startswith("attr_t"),
        ),
        (
            CLScenario.CLASS_INCREMENTAL,
            lambda s, t: s.startswith(f"subj_{t}_"),
            lambda a, t: a.startswith(f"attr_t{t}_"),
        ),
    ],
)
def test_probed_namespace_policy_per_scenario(scenario, subj_check, attr_check) -> None:
    """Signal facts use the scenario's probed subject/attribute naming (exit #3)."""
    cfg = _cfg(scenario=scenario, contradiction_rate=0.0)
    stream = generate_stream(cfg, seed=9)
    labels = offline_labels(stream)
    for item in stream.items:
        if labels[item.item_id] != Label.SIGNAL:
            continue
        assert item.fact is not None
        t = item.task_index
        assert subj_check(item.fact.subject, t), item.fact.subject
        assert attr_check(item.fact.attribute, t), item.fact.attribute


def test_distractor_namespace_disjoint_and_noise_factless() -> None:
    """Distractor subjects are dsubj_* (never probed); noise items have fact=None."""
    cfg = _cfg()
    stream = generate_stream(cfg, seed=4)
    labels = offline_labels(stream)
    signal_subjects = {
        it.fact.subject
        for it in stream.items
        if labels[it.item_id] == Label.SIGNAL and it.fact is not None
    }
    for item in stream.items:
        label = labels[item.item_id]
        if label == Label.DISTRACTOR:
            assert item.fact is not None
            assert item.fact.subject.startswith("dsubj_")
            assert item.fact.subject not in signal_subjects
        elif label == Label.NOISE:
            assert item.fact is None
            assert all(tok.startswith("tok_") for tok in item.content.split())


# --------------------------------------------------------------------------- #
# FR1.3 regime knobs visibly take effect
# --------------------------------------------------------------------------- #
def test_no_contradictions_when_rate_zero() -> None:
    """With contradiction_rate=0 every probed key's signal value is unique."""
    cfg = _cfg(items_per_task=80, n_tasks=2, n_subjects_per_task=64,
               contradiction_rate=0.0)
    stream = generate_stream(cfg, seed=6)
    by_key = _signal_facts_by_key(stream)
    assert by_key  # there are signal items
    for key, items in by_key.items():
        assert len(items) == 1, f"key {key} asserted {len(items)} times at rate 0"


def test_contradictions_present_when_rate_positive() -> None:
    """With contradiction_rate>0 some probed key is reasserted with new values."""
    cfg = _cfg(items_per_task=80, n_tasks=2, n_subjects_per_task=64,
               contradiction_rate=0.3)
    stream = generate_stream(cfg, seed=6)
    by_key = _signal_facts_by_key(stream)
    contradicted = {
        key: items
        for key, items in by_key.items()
        if len(items) >= 2 and len({it.fact.value for it in items}) >= 2
    }
    assert contradicted, "expected at least one contradicted key at rate 0.3"


def test_drift_biases_values_upward_across_tasks() -> None:
    """With drift>0 later tasks' signal values skew to higher indices."""
    cfg = _cfg(
        n_tasks=4,
        items_per_task=200,
        n_subjects_per_task=64,
        n_values=12,
        label_mix=LabelMix(signal=0.6, distractor=0.2, noise=0.2),
        contradiction_rate=0.0,
        drift=0.95,
    )
    stream = generate_stream(cfg, seed=13)
    labels = offline_labels(stream)
    sums: dict[int, list[int]] = defaultdict(list)
    for item in stream.items:
        if labels[item.item_id] == Label.SIGNAL and item.fact is not None:
            sums[item.task_index].append(int(item.fact.value.split("_")[1]))
    mean_first = sum(sums[0]) / len(sums[0])
    mean_last = sum(sums[3]) / len(sums[3])
    assert mean_last > mean_first + 3.0, (mean_first, mean_last)


def test_recency_bias_clusters_contradictions_toward_end() -> None:
    """High recency_bias pushes contradictions to later stream positions."""
    common = dict(
        n_tasks=1,
        items_per_task=300,
        n_subjects_per_task=64,
        n_attributes=4,
        label_mix=LabelMix(signal=0.7, distractor=0.2, noise=0.1),
        contradiction_rate=0.3,
    )
    low = generate_stream(_cfg(recency_bias=0.0, **common), seed=21)
    high = generate_stream(_cfg(recency_bias=0.95, **common), seed=21)
    low_orders = _contradiction_orders(low)
    high_orders = _contradiction_orders(high)
    assert low_orders and high_orders
    mean_low = sum(low_orders) / len(low_orders)
    mean_high = sum(high_orders) / len(high_orders)
    assert mean_high > mean_low, (mean_low, mean_high)
