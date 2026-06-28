"""Shared data model for the Slow Wave synthetic continual task stream (Phase 1).

This module is the **authoritative cross-module contract** for Phase 1 (see
``docs/PHASE1_CONTRACT.md``). The generator, datasheet emitter, probe-set
builder, and confound guard all build against the types defined here and must
not redefine them.

Design principles
-----------------
* **Confound separation by construction (FR1.6).** A :class:`StreamItem` is the
  *online-visible* unit the wake agent and retrieval policy will consume. It
  carries **no** ground-truth relevance label. The labels live in a physically
  separate :class:`GroundTruth` sidecar held on the :class:`Stream` and exposed
  **only** via :func:`offline_labels`. There is no attribute path from a
  ``StreamItem`` to its label, so labels cannot leak into any online
  retrieval/priority signal. The confound guard (``stream/guard.py``) asserts
  this structurally.
* **Determinism (FR1.4).** Every type is JSON-serializable with stable key order
  (pydantic ``model_dump(mode="json")`` + ``sort_keys=True``). All fields are
  strings/ints/enums or rounded floats so two builds from the same seed are
  byte-identical.
* **Exactly one scenario per stream (FR1.2).** :attr:`Stream.scenario` is a
  single :class:`CLScenario`; :func:`assert_same_scenario` refuses to mix
  scenarios across streams (no cross-scenario aggregation).

The synthetic domain is a *fact world*: items assert ``(subject, attribute,
value)`` triples in natural-language surface form. ``signal`` items assert facts
that the probe set queries (mission-relevant); ``distractor`` items assert
plausibly-formed facts about subjects/attributes that are never probed
(plausible but irrelevant); ``noise`` items are random token salads (irrelevant).
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Label(str, Enum):
    """Ground-truth relevance label for a stream item (FR1.1).

    OFFLINE-ONLY: this value is never attached to a :class:`StreamItem`; it lives
    in :class:`GroundTruth` and is reachable only via :func:`offline_labels`.
    """

    SIGNAL = "signal"
    DISTRACTOR = "distractor"
    NOISE = "noise"


class CLScenario(str, Enum):
    """Continual-learning scenario tag (van de Ven et al. 2022; FR1.2)."""

    TASK_INCREMENTAL = "task-incremental"
    DOMAIN_INCREMENTAL = "domain-incremental"
    CLASS_INCREMENTAL = "class-incremental"


class ItemKind(str, Enum):
    """Whether a stream item is an ingest item or a task marker."""

    INGEST = "ingest"
    TASK = "task"


# --------------------------------------------------------------------------- #
# Generation configuration
# --------------------------------------------------------------------------- #
class LabelMix(BaseModel):
    """Requested proportions of each relevance label in a stream (FR1.3).

    The three proportions should sum to ~1.0 (validated within a small
    tolerance). The generator allocates per-task label counts by rounding these
    proportions, so the *empirical* mix matches the request within tolerance
    (Phase 1 exit criterion #2).
    """

    model_config = ConfigDict(extra="forbid")

    signal: float = 0.5
    distractor: float = 0.3
    noise: float = 0.2

    @model_validator(mode="after")
    def _check_sum(self) -> "LabelMix":
        total = self.signal + self.distractor + self.noise
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"label_mix proportions must sum to 1.0 (got {total:.6f}: "
                f"signal={self.signal}, distractor={self.distractor}, "
                f"noise={self.noise})"
            )
        if min(self.signal, self.distractor, self.noise) < 0.0:
            raise ValueError("label_mix proportions must be non-negative")
        return self

    def as_dict(self) -> dict[str, float]:
        """Return the mix as a plain ``{label_value: proportion}`` dict."""
        return {
            Label.SIGNAL.value: self.signal,
            Label.DISTRACTOR.value: self.distractor,
            Label.NOISE.value: self.noise,
        }


class StreamGenConfig(BaseModel):
    """Parameters controlling synthetic stream generation (FR1.1-FR1.5).

    Fully determines a stream given a seed. ``extra="forbid"`` so a typo'd
    parameter fails loudly at load time. All numeric knobs have sensible,
    cheap defaults suitable for tests.
    """

    model_config = ConfigDict(extra="forbid")

    scenario: CLScenario = CLScenario.TASK_INCREMENTAL
    n_tasks: int = Field(default=5, ge=1)
    items_per_task: int = Field(default=40, ge=1)
    label_mix: LabelMix = Field(default_factory=LabelMix)

    # FR1.3 — distractor regime / temporal structure / contradictions.
    contradiction_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Fraction of *signal* items that re-assert an already-probed fact with a
    new value (a contradiction). Set to 0.0 for a stable, crisp probe set."""

    drift: float = Field(default=0.0, ge=0.0, le=1.0)
    """Strength of gradual value-distribution drift across tasks [0, 1]."""

    recency_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    """Tendency to cluster repeated/contradicting facts near the end of a task."""

    # Synthetic vocabulary sizes.
    n_subjects_per_task: int = Field(default=8, ge=1)
    n_attributes: int = Field(default=4, ge=1)
    n_values: int = Field(default=12, ge=2)
    probes_per_task: int = Field(default=6, ge=1)
    distractor_namespace_size: int = Field(default=64, ge=1)
    noise_vocab_size: int = Field(default=128, ge=1)
    noise_tokens: int = Field(default=8, ge=1)

    def canonical_dict(self) -> dict[str, Any]:
        """Return a JSON-canonical dict of this config (stable across runs)."""
        return json.loads(json.dumps(self.model_dump(mode="json"), sort_keys=True))


# --------------------------------------------------------------------------- #
# Core data types
# --------------------------------------------------------------------------- #
class Fact(BaseModel):
    """A synthetic ``(subject, attribute, value)`` knowledge triple.

    Frozen so a fact is hashable and cannot be mutated after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    attribute: str
    value: str

    def key(self) -> tuple[str, str]:
        """Return the ``(subject, attribute)`` identity of the fact."""
        return (self.subject, self.attribute)


class StreamItem(BaseModel):
    """A single, **online-visible** item in the stream.

    This is exactly what the wake agent / retrieval policy will see. It carries
    **no** ground-truth relevance label and **no** pointer to
    :class:`GroundTruth` — that separation is the structural half of the FR1.6
    confound guard. ``frozen=True`` makes items immutable.

    Attributes:
        item_id: Stable, zero-padded id, e.g. ``"i000007"``.
        order: 0-based position in the stream.
        kind: ingest item or task marker.
        task_index: Which task segment (``0 .. n_tasks-1``) the item belongs to.
        content: Natural-language surface form (for later embedding).
        fact: The asserted triple for ingest items that assert a fact; ``None``
            for pure-noise items that assert no structured fact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    order: int
    kind: ItemKind
    task_index: int
    content: str
    fact: Fact | None = None


class GroundTruth(BaseModel):
    """The OFFLINE-ONLY ground-truth sidecar for a stream (FR1.6).

    Holds the relevance label for every item, keyed by ``item_id``. This object
    is held on :class:`Stream` but is **never** reachable from a
    :class:`StreamItem`; the only sanctioned accessor is :func:`offline_labels`.

    Attributes:
        labels: ``item_id -> Label`` for **every** item (exactly one each).
    """

    model_config = ConfigDict(extra="forbid")

    labels: dict[str, Label]


class Stream(BaseModel):
    """A complete synthetic continual task stream.

    Attributes:
        stream_id: Reproducible id derived from scenario + seed + config hash.
        scenario: The single CL scenario this stream is tagged with (FR1.2).
        seed: The stream seed used to generate it.
        n_tasks: Number of task segments.
        config: The :class:`StreamGenConfig` that produced this stream.
        items: The ordered, online-visible items (no labels).
        ground_truth: The offline-only label sidecar.
    """

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    scenario: CLScenario
    seed: int
    n_tasks: int
    config: StreamGenConfig
    items: list[StreamItem]
    ground_truth: GroundTruth

    @model_validator(mode="after")
    def _check_label_coverage(self) -> "Stream":
        """Every item has exactly one label; no orphan labels (FR1.1)."""
        item_ids = [it.item_id for it in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("duplicate item_id in stream")
        item_id_set = set(item_ids)
        label_keys = set(self.ground_truth.labels)
        missing = item_id_set - label_keys
        if missing:
            raise ValueError(
                f"{len(missing)} item(s) have no ground-truth label "
                f"(e.g. {sorted(missing)[:3]})"
            )
        orphan = label_keys - item_id_set
        if orphan:
            raise ValueError(
                f"{len(orphan)} ground-truth label(s) reference no item "
                f"(e.g. {sorted(orphan)[:3]})"
            )
        return self


# --------------------------------------------------------------------------- #
# Probe set / accuracy matrix
# --------------------------------------------------------------------------- #
class Probe(BaseModel):
    """A held-out evaluation query with a known correct answer (FR1.5).

    The ``answer`` is ground-truth used for **offline scoring only**; the agent
    is asked the ``query``. ``task_id_visible`` reflects the scenario: in
    task-incremental the evaluator may reveal ``task_index``; in domain- and
    class-incremental it may not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str
    task_index: int
    subject: str
    attribute: str
    answer: str
    available_after_order: int
    query: str
    task_id_visible: bool


class ProbeSet(BaseModel):
    """The complete held-out probe set for a stream (FR1.5)."""

    model_config = ConfigDict(extra="forbid")

    scenario: CLScenario
    n_tasks: int
    probes: list[Probe]


class AccuracyMatrix(BaseModel):
    """The continual-learning accuracy matrix ``R[i][j]`` (FR1.5).

    ``R[i][j]`` is the accuracy on task *j*'s probes after the agent has
    processed the stream through the end of task *i*. Shape is
    ``n_tasks x n_tasks``; every entry is in ``[0, 1]``.
    """

    model_config = ConfigDict(extra="forbid")

    n_tasks: int
    scenario: CLScenario
    R: list[list[float]]

    @model_validator(mode="after")
    def _check_shape(self) -> "AccuracyMatrix":
        if len(self.R) != self.n_tasks:
            raise ValueError(
                f"R has {len(self.R)} rows, expected n_tasks={self.n_tasks}"
            )
        for r, row in enumerate(self.R):
            if len(row) != self.n_tasks:
                raise ValueError(
                    f"R row {r} has length {len(row)}, expected {self.n_tasks}"
                )
            for c, v in enumerate(row):
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"R[{r}][{c}]={v} outside [0, 1]")
        return self


# --------------------------------------------------------------------------- #
# Sanctioned accessors & scenario invariants
# --------------------------------------------------------------------------- #
def offline_labels(stream: Stream) -> dict[str, Label]:
    """Return the ground-truth labels for ``stream`` (OFFLINE scoring only).

    This is the **only** sanctioned way to read relevance labels. Online code
    (retrieval, priority, the wake agent) must never call it — the confound
    guard in ``stream/guard.py`` enforces that labels are unreachable from the
    online item view.

    Args:
        stream: The stream whose labels to read.

    Returns:
        A copy of the ``item_id -> Label`` mapping.
    """
    return dict(stream.ground_truth.labels)


def assert_single_scenario(stream: Stream) -> None:
    """Assert ``stream`` carries exactly one well-formed CL scenario (FR1.2).

    Raises:
        ValueError: If the scenario is not a valid :class:`CLScenario`.
    """
    if not isinstance(stream.scenario, CLScenario):
        raise ValueError(f"stream.scenario is not a CLScenario: {stream.scenario!r}")


def assert_same_scenario(streams: list[Stream]) -> CLScenario:
    """Assert all ``streams`` share one CL scenario; refuse mixing (FR1.2).

    No cross-scenario metric aggregation is permitted, so any attempt to combine
    streams of differing scenarios is rejected loudly.

    Args:
        streams: A non-empty list of streams.

    Returns:
        The single shared :class:`CLScenario`.

    Raises:
        ValueError: If ``streams`` is empty or contains more than one scenario.
    """
    if not streams:
        raise ValueError("assert_same_scenario requires at least one stream")
    scenarios = {s.scenario for s in streams}
    if len(scenarios) != 1:
        raise ValueError(
            "cross-scenario mixing is forbidden (no cross-scenario aggregation): "
            f"found scenarios {sorted(sc.value for sc in scenarios)}"
        )
    return next(iter(scenarios))
