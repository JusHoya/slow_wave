"""Tests for the WS2 datasheet emitter (Phase 1 exit criterion #5).

These tests are intentionally self-contained: rather than depend on WS1's
generator (developed concurrently), they hand-construct a small
:class:`~slow_wave.stream.schema.Stream` from schema types with known labels via
:class:`~slow_wave.stream.schema.GroundTruth`. An optional integration test
additionally exercises ``generate_stream`` but skips if WS1 is not yet importable.

Coverage:
* all seven Gebru sections present and non-empty;
* ``label_distribution`` sums to ``n_items`` and matches ``offline_labels``;
* ``n_contradictions`` counts re-asserted probed keys;
* determinism: byte-identical JSON across two builds from the same stream;
* schema validation: ``validate_datasheet`` round-trips an equal datasheet.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from slow_wave.stream.datasheet import (
    Datasheet,
    build_datasheet,
    datasheet_to_json,
    validate_datasheet,
)
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
    offline_labels,
)

# The seven canonical Gebru "Datasheets for Datasets" sections.
GEBRU_SECTIONS = (
    "motivation",
    "composition",
    "collection_process",
    "preprocessing",
    "uses",
    "distribution",
    "maintenance",
)


# --------------------------------------------------------------------------- #
# Hand-built fixture stream (no dependency on WS1's generator)
# --------------------------------------------------------------------------- #
def _make_stream() -> Stream:
    """Construct a tiny, fully-specified stream with known labels.

    Layout (6 items, 2 tasks):
      * task 0: signal, distractor, noise, signal-contradiction
      * task 1: signal, noise

    The two task-0 signals assert the SAME probed key ``(subj_0_0, attr_0)``
    with DIFFERENT values, so exactly one contradiction key exists. Label
    counts: signal=3, distractor=1, noise=2.
    """
    config = StreamGenConfig(
        scenario=CLScenario.TASK_INCREMENTAL,
        n_tasks=2,
        items_per_task=3,
        label_mix=LabelMix(signal=0.5, distractor=0.2, noise=0.3),
        contradiction_rate=0.25,
        drift=0.1,
        recency_bias=0.0,
    )

    def item(order: int, task_index: int, content: str, fact: Fact | None) -> StreamItem:
        return StreamItem(
            item_id=f"i{order:06d}",
            order=order,
            kind=ItemKind.INGEST,
            task_index=task_index,
            content=content,
            fact=fact,
        )

    items = [
        item(0, 0, "The attr_0 of subj_0_0 is val_1.", Fact(subject="subj_0_0", attribute="attr_0", value="val_1")),
        item(1, 0, "The attr_0 of dsubj_0 is val_2.", Fact(subject="dsubj_0", attribute="attr_0", value="val_2")),
        item(2, 0, "tok_3 tok_7 tok_1 tok_9", None),
        item(3, 0, "The attr_0 of subj_0_0 is val_3.", Fact(subject="subj_0_0", attribute="attr_0", value="val_3")),
        item(4, 1, "The attr_0 of subj_1_0 is val_0.", Fact(subject="subj_1_0", attribute="attr_0", value="val_0")),
        item(5, 1, "tok_2 tok_5 tok_8 tok_4", None),
    ]
    labels = {
        "i000000": Label.SIGNAL,
        "i000001": Label.DISTRACTOR,
        "i000002": Label.NOISE,
        "i000003": Label.SIGNAL,
        "i000004": Label.SIGNAL,
        "i000005": Label.NOISE,
    }
    return Stream(
        stream_id="task-incremental-s7-deadbeef",
        scenario=CLScenario.TASK_INCREMENTAL,
        seed=7,
        n_tasks=2,
        config=config,
        items=items,
        ground_truth=GroundTruth(labels=labels),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_build_datasheet_basic_fields() -> None:
    """Top-level computed statistics reflect the stream."""
    stream = _make_stream()
    ds = build_datasheet(stream)

    assert isinstance(ds, Datasheet)
    assert ds.scenario == CLScenario.TASK_INCREMENTAL
    assert ds.seed == 7
    assert ds.stream_id == "task-incremental-s7-deadbeef"
    assert ds.n_tasks == 2
    assert ds.n_items == 6
    assert ds.label_provenance.strip()


def test_all_gebru_sections_present_and_nonempty() -> None:
    """All seven Gebru sections exist with non-empty prose fields."""
    ds = build_datasheet(_make_stream())
    dumped = ds.model_dump(mode="json")

    for section in GEBRU_SECTIONS:
        assert section in dumped, f"missing Gebru section: {section}"
        body = dumped[section]
        assert isinstance(body, dict) and body, f"empty Gebru section: {section}"
        for field, value in body.items():
            if isinstance(value, str):
                assert value.strip(), f"empty prose field: {section}.{field}"


def test_label_distribution_sums_and_matches_offline_labels() -> None:
    """label_distribution sums to n_items and equals offline_labels counts."""
    stream = _make_stream()
    ds = build_datasheet(stream)

    assert sum(ds.label_distribution.values()) == ds.n_items == len(stream.items)

    expected = Counter(label.value for label in offline_labels(stream).values())
    for label in (Label.SIGNAL.value, Label.DISTRACTOR.value, Label.NOISE.value):
        assert ds.label_distribution[label] == expected.get(label, 0)

    # Concrete known counts for this fixture.
    assert ds.label_distribution == {"signal": 3, "distractor": 1, "noise": 2}


def test_label_proportions_and_requested_mix_rounded() -> None:
    """Proportions are counts/n_items at 6 dp; requested mix mirrors the config."""
    ds = build_datasheet(_make_stream())

    for label, count in ds.label_distribution.items():
        assert ds.label_proportions[label] == round(count / ds.n_items, 6)

    assert ds.requested_label_mix == {"signal": 0.5, "distractor": 0.2, "noise": 0.3}
    # Every float is rounded to <= 6 decimal places.
    for value in (*ds.label_proportions.values(), *ds.requested_label_mix.values()):
        assert round(value, 6) == value


def test_n_contradictions_counts_reasserted_keys() -> None:
    """Exactly the one re-asserted probed key counts as a contradiction."""
    ds = build_datasheet(_make_stream())
    assert ds.n_contradictions == 1


def test_regime_params_copied_from_config() -> None:
    """The regime block records the FR1.3 generation knobs."""
    stream = _make_stream()
    ds = build_datasheet(stream)
    cfg = stream.config

    assert ds.regime.contradiction_rate == round(cfg.contradiction_rate, 6)
    assert ds.regime.drift == round(cfg.drift, 6)
    assert ds.regime.recency_bias == round(cfg.recency_bias, 6)
    assert ds.regime.n_subjects_per_task == cfg.n_subjects_per_task
    assert ds.regime.n_attributes == cfg.n_attributes
    assert ds.regime.n_values == cfg.n_values
    assert ds.regime.probes_per_task == cfg.probes_per_task
    assert ds.regime.distractor_namespace_size == cfg.distractor_namespace_size
    assert ds.regime.noise_vocab_size == cfg.noise_vocab_size
    assert ds.regime.noise_tokens == cfg.noise_tokens


def test_config_override_argument() -> None:
    """An explicit config overrides stream.config for the reported mix/regime."""
    stream = _make_stream()
    override = StreamGenConfig(
        scenario=CLScenario.TASK_INCREMENTAL,
        n_tasks=2,
        items_per_task=3,
        label_mix=LabelMix(signal=0.7, distractor=0.1, noise=0.2),
        contradiction_rate=0.5,
    )
    ds = build_datasheet(stream, config=override)
    assert ds.requested_label_mix == {"signal": 0.7, "distractor": 0.1, "noise": 0.2}
    assert ds.regime.contradiction_rate == 0.5


def test_datasheet_json_is_deterministic() -> None:
    """datasheet_to_json is byte-identical across two builds from the same stream."""
    first = datasheet_to_json(build_datasheet(_make_stream()))
    second = datasheet_to_json(build_datasheet(_make_stream()))
    assert first == second
    # House style: sorted keys, 2-space indent, trailing newline.
    assert first.endswith("\n")
    assert "\n  " in first
    parsed = json.loads(first)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_validate_datasheet_round_trips() -> None:
    """The emitted JSON validates against the schema and round-trips equal."""
    ds = build_datasheet(_make_stream())
    data = json.loads(datasheet_to_json(ds))
    restored = validate_datasheet(data)
    assert isinstance(restored, Datasheet)
    assert restored == ds
    # And re-serializing the validated copy is byte-identical.
    assert datasheet_to_json(restored) == datasheet_to_json(ds)


def test_datasheet_forbids_extra_fields() -> None:
    """Datasheet uses extra='forbid' so unknown keys are rejected."""
    data = json.loads(datasheet_to_json(build_datasheet(_make_stream())))
    data["unexpected_field"] = "boom"
    with pytest.raises(Exception):
        validate_datasheet(data)


def test_generator_integration_if_available() -> None:
    """Optional: build a datasheet from a real generated stream (skip if WS1 absent)."""
    try:
        from slow_wave.stream.generator import generate_stream
    except Exception:  # pragma: no cover - depends on WS1 landing
        pytest.skip("WS1 generator not importable yet")

    config = StreamGenConfig(n_tasks=3, items_per_task=12)
    stream = generate_stream(config, seed=7)
    ds = build_datasheet(stream)

    assert ds.n_items == len(stream.items)
    assert sum(ds.label_distribution.values()) == ds.n_items
    # Determinism on a real stream too.
    assert datasheet_to_json(build_datasheet(stream)) == datasheet_to_json(
        build_datasheet(stream)
    )
    # Round-trips through the schema.
    assert validate_datasheet(json.loads(datasheet_to_json(ds))) == ds


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    raise SystemExit(pytest.main([__file__, "-v"]))
