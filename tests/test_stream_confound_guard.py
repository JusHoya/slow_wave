"""Tests for the FR1.6 confound guard (Phase 1 exit criterion #4).

These tests prove the guard is **non-vacuous**: alongside the clean-path checks
(a real stream's online view is label-free; the sanctioned offline accessor still
works) there are adversarial *positive controls* that deliberately poison objects
with labels — by banned field name, by reachable ``Label`` value, by nested
container, and by handing over a whole ``Stream`` — and assert the guard
**raises** ``ConfoundLeakError`` every time. A guard that always passed would
fail these.
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel, ConfigDict

from slow_wave.stream.guard import (
    BANNED_FIELD_NAMES,
    ConfoundLeakError,
    assert_no_label_leak,
    assert_online_view_is_clean,
    online_view,
)
from slow_wave.stream.schema import (
    CLScenario,
    Fact,
    GroundTruth,
    ItemKind,
    Label,
    Stream,
    StreamGenConfig,
    StreamItem,
    offline_labels,
)


# --------------------------------------------------------------------------- #
# Stream fixture: prefer the real WS1 generator, fall back to a hand-built one.
# --------------------------------------------------------------------------- #
def _hand_built_stream() -> Stream:
    """Construct a tiny, valid :class:`Stream` directly from schema types.

    Used when ``slow_wave.stream.generator`` is not yet importable (development
    in isolation). Produces 2 tasks of 3 ingest items each, with one of every
    label so :func:`offline_labels` is non-empty and varied.
    """
    items: list[StreamItem] = []
    labels: dict[str, Label] = {}
    items_per_task = 3
    n_items = 6
    for order in range(n_items):
        item_id = f"i{order:06d}"
        task_index = order // items_per_task
        bucket = order % 3
        if bucket == 0:
            fact = Fact(
                subject=f"subj_{task_index}_{order}",
                attribute="attr_0",
                value=f"val_{order}",
            )
            content = f"The attr_0 of subj_{task_index}_{order} is val_{order}."
            label = Label.SIGNAL
        elif bucket == 1:
            fact = Fact(
                subject=f"dsubj_{order}", attribute="attr_1", value=f"val_{order}"
            )
            content = f"The attr_1 of dsubj_{order} is val_{order}."
            label = Label.DISTRACTOR
        else:
            fact = None
            content = "tok_1 tok_2 tok_3 tok_4"
            label = Label.NOISE
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

    config = StreamGenConfig(n_tasks=2, items_per_task=items_per_task)
    return Stream(
        stream_id="task-incremental-s7-deadbeef",
        scenario=CLScenario.TASK_INCREMENTAL,
        seed=7,
        n_tasks=2,
        config=config,
        items=items,
        ground_truth=GroundTruth(labels=labels),
    )


@pytest.fixture
def stream() -> Stream:
    """A real generated stream if WS1 exists, else a hand-built fallback."""
    try:
        from slow_wave.stream.generator import generate_stream
    except ImportError:
        return _hand_built_stream()
    config = StreamGenConfig(n_tasks=2, items_per_task=6)
    return generate_stream(config, seed=7)


# --------------------------------------------------------------------------- #
# Poisoned objects for the positive controls.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _PoisonByName:
    """A dataclass that exposes a banned field name (``label``)."""

    item_id: str = "i000000"
    label: str = "signal"  # banned name, even though the value is a plain str


class _PoisonByValueModel(BaseModel):
    """A pydantic model hiding a real ``Label`` under an innocuous field name."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: object  # innocuous name; the *value* is a schema.Label


class _PlainHolder:
    """A plain object (``__dict__`` only) used for recursion controls."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


# --------------------------------------------------------------------------- #
# Clean path (structural separation holds for a real stream).
# --------------------------------------------------------------------------- #
def test_online_view_returns_only_stream_items(stream: Stream) -> None:
    """online_view returns a non-empty immutable tuple of StreamItems only."""
    view = online_view(stream)
    assert isinstance(view, tuple)
    assert len(view) > 0
    assert all(isinstance(item, StreamItem) for item in view)
    # The view is exactly the stream's items, in order.
    assert [it.item_id for it in view] == [it.item_id for it in stream.items]


def test_assert_online_view_is_clean_passes_for_real_stream(stream: Stream) -> None:
    """The full FR1.6 invariant holds for a well-formed stream (returns None)."""
    assert assert_online_view_is_clean(stream) is None


def test_individual_items_have_no_reachable_label(stream: Stream) -> None:
    """Each online item passes the recursive guard on its own."""
    for item in online_view(stream):
        assert assert_no_label_leak(item) is None


def test_stream_item_class_declares_no_banned_field() -> None:
    """StreamItem.model_fields contains none of the banned names (class level)."""
    assert not (set(StreamItem.model_fields) & BANNED_FIELD_NAMES)
    assert "label" not in StreamItem.model_fields
    assert "fact" in StreamItem.model_fields  # sanity: we inspected the right type


# --------------------------------------------------------------------------- #
# Sanctioned offline access still works (the guard does not break scoring).
# --------------------------------------------------------------------------- #
def test_offline_labels_still_reachable_via_accessor(stream: Stream) -> None:
    """offline_labels returns a non-empty dict[item_id, Label] (offline path)."""
    labels = offline_labels(stream)
    assert isinstance(labels, dict)
    assert len(labels) > 0
    assert set(labels) == {it.item_id for it in stream.items}
    assert all(isinstance(v, Label) for v in labels.values())


# --------------------------------------------------------------------------- #
# Positive controls — the guard MUST detect leaks (non-vacuous).
# --------------------------------------------------------------------------- #
def test_positive_control_banned_field_name_dataclass() -> None:
    """A dataclass with a field literally named ``label`` is rejected."""
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(_PoisonByName())


def test_positive_control_banned_field_name_pydantic() -> None:
    """A pydantic model declaring a banned field name is rejected."""

    class _PoisonModel(BaseModel):
        relevance: str = "high"

    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(_PoisonModel())


def test_positive_control_banned_mapping_key() -> None:
    """A mapping with a banned key is rejected even if the value is innocuous."""
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak({"ground_truth": "anything"})


def test_positive_control_label_value_under_innocuous_name() -> None:
    """A real Label hidden under an un-banned field name is still detected."""
    poisoned = _PoisonByValueModel(payload=Label.SIGNAL)
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(poisoned)


def test_positive_control_label_value_in_plain_object() -> None:
    """A plain ``__dict__`` object holding a Label is detected."""
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(_PlainHolder(note=Label.DISTRACTOR))


def test_positive_control_nested_container_recursion() -> None:
    """A Label nested inside a dict-of-list-of-object is found (recursion works)."""
    nested = {"items": [_PlainHolder(payload=Label.NOISE)]}
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(nested)


def test_positive_control_deeply_nested_label() -> None:
    """A Label buried in nested plain containers is found."""
    nested = {"a": {"b": [("c", [Label.SIGNAL])]}}
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(nested)


def test_positive_control_bare_label_value() -> None:
    """A bare Label passed directly is rejected (it subclasses str)."""
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(Label.SIGNAL)


def test_positive_control_bare_ground_truth() -> None:
    """A bare GroundTruth sidecar is rejected as a reachable label store."""
    gt = GroundTruth(labels={"i000000": Label.SIGNAL})
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(gt)


def test_positive_control_full_stream_trips_guard(stream: Stream) -> None:
    """Handing a whole Stream to the guard raises (it reaches GroundTruth/Label).

    A Stream legitimately owns its offline label sidecar, so it is *not* an
    online-safe object: passing it to assert_no_label_leak must trip the guard.
    This is the stricter behavior mandated by the contract.
    """
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(stream)


# --------------------------------------------------------------------------- #
# Class-level guard fires when a banned name coincides with a StreamItem field.
# --------------------------------------------------------------------------- #
def test_class_level_guard_fires_if_streamitem_field_were_banned(
    stream: Stream, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a real StreamItem field name were banned, the class-level check trips.

    Simulates "a future edit added a banned field to StreamItem" by injecting an
    existing StreamItem field (``item_id``) into BANNED_FIELD_NAMES, proving the
    class-level guard in assert_online_view_is_clean is non-vacuous.
    """
    import slow_wave.stream.guard as guard_mod

    monkeypatch.setattr(
        guard_mod, "BANNED_FIELD_NAMES", frozenset({"item_id"}), raising=True
    )
    with pytest.raises(ConfoundLeakError):
        assert_online_view_is_clean(stream)


# --------------------------------------------------------------------------- #
# Cycle safety.
# --------------------------------------------------------------------------- #
def test_cycle_safe_on_self_referential_clean_structure() -> None:
    """A self-referential clean structure does not recurse infinitely."""
    d: dict[str, object] = {"name": "ok"}
    d["self"] = d  # cycle
    assert assert_no_label_leak(d) is None


def test_cycle_safe_object_cycle_still_detects_label() -> None:
    """A cyclic structure that also hides a Label terminates and still raises."""
    holder = _PlainHolder(secret=Label.NOISE)
    holder.__dict__["loop"] = holder  # cycle
    with pytest.raises(ConfoundLeakError):
        assert_no_label_leak(holder)


# --------------------------------------------------------------------------- #
# Banned-name set sanity.
# --------------------------------------------------------------------------- #
def test_banned_field_names_cover_contract_minimum() -> None:
    """BANNED_FIELD_NAMES contains at least the names mandated by the contract."""
    required = {
        "label",
        "labels",
        "relevance",
        "ground_truth",
        "is_signal",
        "is_distractor",
        "is_noise",
        "gt",
        "ground_truth_label",
    }
    assert required <= BANNED_FIELD_NAMES
