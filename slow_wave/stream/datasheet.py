"""Datasheet emitter for the Slow Wave synthetic continual task stream (Phase 1).

This module implements **WS2** of the Phase 1 contract (see
``docs/PHASE1_CONTRACT.md``). It produces a :class:`Datasheet` describing a
generated :class:`~slow_wave.stream.schema.Stream`, following the structure of
**Gebru et al. (2021), "Datasheets for Datasets"** (Communications of the ACM
64(12)). The seven canonical Gebru sections are modeled as nested pydantic v2
models:

* :class:`Motivation`
* :class:`Composition`
* :class:`CollectionProcess`
* :class:`Preprocessing`
* :class:`Uses`
* :class:`Distribution`
* :class:`Maintenance`

On top of the prose sections, the datasheet carries the **computed statistics**
the PRD requires (label distribution / proportions, the requested label mix, the
distractor-regime parameters, label provenance, and the number of contradicting
probed keys). Everything is derived deterministically from the stream and its
config, and all floats are rounded to 6 decimal places, so two datasheets built
from the same stream serialize to **byte-identical** JSON.

This is OFFLINE scoring code: it reads relevance labels through the sanctioned
:func:`slow_wave.stream.schema.offline_labels` accessor, which is permitted for
offline/analysis code (the confound guard only forbids *online* code from
reaching labels).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict

from slow_wave.stream.schema import (
    CLScenario,
    Label,
    Stream,
    StreamGenConfig,
    offline_labels,
)

# Float precision for every numeric field in an emitted datasheet. Rounding to a
# fixed number of decimals is what makes the serialized JSON byte-stable.
_FLOAT_NDIGITS = 6

# Canonical label order used when materializing the label distribution. JSON is
# emitted with ``sort_keys=True`` so this order does not affect bytes, but it
# keeps every emitted distribution complete (all three labels always present).
_LABEL_ORDER: tuple[Label, ...] = (Label.SIGNAL, Label.DISTRACTOR, Label.NOISE)


# --------------------------------------------------------------------------- #
# Gebru sections (the seven standard "Datasheets for Datasets" headings)
# --------------------------------------------------------------------------- #
class Motivation(BaseModel):
    """Gebru §1 — why the dataset was created (purpose, creators, funding)."""

    model_config = ConfigDict(extra="forbid")

    purpose: str
    creators: str
    funding: str


class Composition(BaseModel):
    """Gebru §2 — what the instances are and how labels/noise are structured."""

    model_config = ConfigDict(extra="forbid")

    instances_represent: str
    n_instances: int
    sampling: str
    instance_data: str
    labels: str
    splits: str
    noise_sources: str
    self_contained: str
    sensitive_data: str


class CollectionProcess(BaseModel):
    """Gebru §3 — how the data were acquired (here: synthesized)."""

    model_config = ConfigDict(extra="forbid")

    acquisition: str
    mechanisms: str
    sampling_strategy: str
    timeframe: str
    ethical_review: str


class Preprocessing(BaseModel):
    """Gebru §4 — preprocessing / cleaning / labeling (here: none)."""

    model_config = ConfigDict(extra="forbid")

    preprocessing_done: str
    raw_data_saved: str
    software: str


class Uses(BaseModel):
    """Gebru §5 — current and intended uses (and what to avoid)."""

    model_config = ConfigDict(extra="forbid")

    current_uses: str
    suitable_tasks: str
    inappropriate_uses: str


class Distribution(BaseModel):
    """Gebru §6 — how the dataset is distributed and under what terms."""

    model_config = ConfigDict(extra="forbid")

    distribution_method: str
    license: str
    restrictions: str


class Maintenance(BaseModel):
    """Gebru §7 — who maintains the dataset and how it is versioned."""

    model_config = ConfigDict(extra="forbid")

    maintainer: str
    contact: str
    updates: str
    versioning: str


# --------------------------------------------------------------------------- #
# Computed-statistics sub-model
# --------------------------------------------------------------------------- #
class RegimeParams(BaseModel):
    """The distractor-regime / temporal-structure knobs copied from the config.

    These mirror the FR1.3 generation parameters so the datasheet records the
    exact regime under which the stream was produced. Float fields are rounded
    to 6 decimal places for byte-stable serialization.
    """

    model_config = ConfigDict(extra="forbid")

    contradiction_rate: float
    drift: float
    recency_bias: float
    n_subjects_per_task: int
    n_attributes: int
    n_values: int
    probes_per_task: int
    distractor_namespace_size: int
    noise_vocab_size: int
    noise_tokens: int


# --------------------------------------------------------------------------- #
# Top-level datasheet
# --------------------------------------------------------------------------- #
class Datasheet(BaseModel):
    """A Gebru-style datasheet plus computed statistics for one stream.

    The model *is* the schema: validating a JSON dict against
    :class:`Datasheet` (via :func:`validate_datasheet`) is the schema check
    required by Phase 1 exit criterion #5.

    Attributes:
        motivation: Gebru §1 prose.
        composition: Gebru §2 prose + instance count.
        collection_process: Gebru §3 prose.
        preprocessing: Gebru §4 prose.
        uses: Gebru §5 prose.
        distribution: Gebru §6 prose.
        maintenance: Gebru §7 prose.
        scenario: The single CL scenario the stream is tagged with.
        seed: The stream seed.
        stream_id: The reproducible stream identity.
        n_tasks: Number of task segments.
        n_items: Total number of stream items.
        label_distribution: ``label -> count`` over all items (all three labels
            always present, defaulting to 0).
        label_proportions: ``label -> count / n_items`` rounded to 6 dp.
        requested_label_mix: The proportions requested in the config's
            ``label_mix``, rounded to 6 dp.
        regime: The distractor-regime parameters used to generate the stream.
        label_provenance: How the labels were assigned (synthetic, by design).
        n_contradictions: Number of probed ``(subject, attribute)`` keys
            re-asserted with a new value among the signal items.
    """

    model_config = ConfigDict(extra="forbid")

    # The seven Gebru sections.
    motivation: Motivation
    composition: Composition
    collection_process: CollectionProcess
    preprocessing: Preprocessing
    uses: Uses
    distribution: Distribution
    maintenance: Maintenance

    # Computed statistics required by the PRD.
    scenario: CLScenario
    seed: int
    stream_id: str
    n_tasks: int
    n_items: int
    label_distribution: dict[str, int]
    label_proportions: dict[str, float]
    requested_label_mix: dict[str, float]
    regime: RegimeParams
    label_provenance: str
    n_contradictions: int


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def _count_contradictions(stream: Stream, labels: dict[str, Label]) -> int:
    """Count probed ``(subject, attribute)`` keys re-asserted with a new value.

    Among items whose relevance label is :attr:`Label.SIGNAL` and whose
    ``fact`` is not ``None``, group by :meth:`Fact.key`. A *contradiction key*
    is one asserted by at least two signal items that carry at least two
    distinct values.

    Args:
        stream: The stream to inspect.
        labels: The ``item_id -> Label`` mapping from :func:`offline_labels`.

    Returns:
        The number of contradiction keys.
    """
    counts_by_key: dict[tuple[str, str], int] = defaultdict(int)
    values_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in stream.items:
        if labels.get(item.item_id) is Label.SIGNAL and item.fact is not None:
            key = item.fact.key()
            counts_by_key[key] += 1
            values_by_key[key].add(item.fact.value)
    return sum(
        1
        for key, values in values_by_key.items()
        if counts_by_key[key] >= 2 and len(values) >= 2
    )


def _label_distribution(labels: dict[str, Label]) -> dict[str, int]:
    """Return ``label_value -> count`` over all items, all three labels present."""
    counts = Counter(labels.values())
    return {label.value: int(counts.get(label, 0)) for label in _LABEL_ORDER}


def _label_proportions(
    label_distribution: dict[str, int], n_items: int
) -> dict[str, float]:
    """Return ``label_value -> count / n_items`` rounded to 6 dp (0.0 if empty)."""
    if n_items <= 0:
        return {label: 0.0 for label in label_distribution}
    return {
        label: round(count / n_items, _FLOAT_NDIGITS)
        for label, count in label_distribution.items()
    }


# --------------------------------------------------------------------------- #
# Prose builders (deterministic, parameterized by stream statistics)
# --------------------------------------------------------------------------- #
def _build_motivation() -> Motivation:
    """Build the (stream-independent) motivation section."""
    return Motivation(
        purpose=(
            "This dataset is a programmatically-generated synthetic "
            "continual-learning stream for the Slow Wave memory-consolidation "
            "benchmark. It exists to measure how a sleep-inspired wake/sleep "
            "agent ingests an ordered fact stream, consolidates relevant "
            "('signal') knowledge, and resists interference from "
            "plausible-but-irrelevant 'distractor' facts and structureless "
            "'noise'."
        ),
        creators=(
            "Generated by the Slow Wave Phase 1 synthetic stream generator "
            "(slow_wave.stream.generator) as part of the Slow Wave research "
            "bench."
        ),
        funding=(
            "No external funding; produced internally as part of the Slow Wave "
            "open research bench."
        ),
    )


def _build_composition(
    *,
    n_items: int,
    n_tasks: int,
    label_distribution: dict[str, int],
    n_contradictions: int,
) -> Composition:
    """Build the composition section from the stream's computed statistics."""
    return Composition(
        instances_represent=(
            "Each instance is a single online-visible StreamItem asserting a "
            "synthetic (subject, attribute, value) fact in natural-language "
            "surface form, or -- for pure-noise items -- a structureless token "
            f"salad with no fact. Items are ordered and partitioned into "
            f"{n_tasks} contiguous task segments."
        ),
        n_instances=n_items,
        sampling=(
            "The dataset is not a sample of a larger population; it is "
            "exhaustively generated from a StreamGenConfig and a single integer "
            "seed. Every item the generator produces is included."
        ),
        instance_data=(
            "Each instance carries an item_id, a 0-based order, an item kind, a "
            "task_index, a natural-language content string, and an optional "
            "(subject, attribute, value) Fact (None for pure-noise items)."
        ),
        labels=(
            "Each item has exactly one offline-only relevance label drawn from "
            "{signal, distractor, noise}. Labels live in a physically separate "
            "GroundTruth sidecar and are never attached to a StreamItem, so they "
            "cannot leak into any online retrieval or priority signal (FR1.6). "
            f"Observed label counts: {label_distribution}."
        ),
        splits=(
            f"The stream is partitioned into {n_tasks} task segments by "
            "task_index rather than train/test splits; held-out evaluation is "
            "provided separately as a probe set."
        ),
        noise_sources=(
            "Noise is intentional and parameterized: a fraction of items are "
            "distractors (plausible facts over a never-probed namespace) and a "
            "fraction are noise (random tokens). Contradictions "
            f"({n_contradictions} probed key(s) re-asserted with a new value), "
            "value drift, and recency bias are deliberately injected per the "
            "regime parameters."
        ),
        self_contained=(
            "The dataset is fully self-contained: it depends on no external "
            "resources and is reproducible from its seed and config alone."
        ),
        sensitive_data=(
            "Contains no real-world, personal, confidential, or otherwise "
            "sensitive data; all subjects, attributes, and values are synthetic "
            "tokens."
        ),
    )


def _build_collection_process(scenario: CLScenario) -> CollectionProcess:
    """Build the collection-process section."""
    return CollectionProcess(
        acquisition=(
            "Data were not collected from any real-world source; they were "
            "synthesized deterministically by the Slow Wave stream generator."
        ),
        mechanisms=(
            "Generation uses NumPy PRNGs seeded via "
            "slow_wave.repro.seeding.derive_seed so that the same (config, "
            "seed) yields a byte-identical stream."
        ),
        sampling_strategy=(
            "Per-task label counts are allocated from the requested label mix "
            "(largest-remainder rounding) and placed in deterministic shuffled "
            "order; facts are sampled from synthetic subject/attribute/value "
            f"vocabularies under the '{scenario.value}' scenario's namespace "
            "policy."
        ),
        timeframe=(
            "Not applicable; the stream has no real-world collection timeframe. "
            "Generation is instantaneous and reproducible on demand."
        ),
        ethical_review=(
            "No ethical review process was required or conducted; no human "
            "subjects or real data are involved."
        ),
    )


def _build_preprocessing() -> Preprocessing:
    """Build the preprocessing section (synthetic streams need none)."""
    return Preprocessing(
        preprocessing_done=(
            "No cleaning, tokenization, or feature extraction is applied. Items "
            "are emitted in their final natural-language surface form; "
            "downstream embedding is the consumer's responsibility and is out "
            "of scope for this dataset."
        ),
        raw_data_saved=(
            "The generator output is itself the raw data; there is no separate "
            "raw form. The full stream and its offline GroundTruth sidecar are "
            "retained together."
        ),
        software=(
            "Produced by the open-source Slow Wave package "
            "(slow_wave.stream.generator); regeneration requires only the "
            "package, the StreamGenConfig, and the seed."
        ),
    )


def _build_uses() -> Uses:
    """Build the uses section."""
    return Uses(
        current_uses=(
            "Used by the Slow Wave bench to score memory consolidation: a "
            "held-out probe set queries each probed (subject, attribute) and an "
            "R[i][j] accuracy matrix measures retention and forgetting across "
            "tasks."
        ),
        suitable_tasks=(
            "Suitable for studying continual learning, catastrophic forgetting, "
            "retrieval/priority policies, and sleep-inspired consolidation "
            "under controlled signal/distractor/noise regimes."
        ),
        inappropriate_uses=(
            "Not suitable for training or evaluating real-world language "
            "understanding; the vocabulary is synthetic and carries no semantic "
            "content. Online code must never read the offline relevance labels."
        ),
    )


def _build_distribution() -> Distribution:
    """Build the distribution section."""
    return Distribution(
        distribution_method=(
            "Bundled with the Slow Wave benchmark repository; not distributed "
            "as a standalone artifact. Streams are regenerated on demand from a "
            "seed and config rather than shipped as a fixed blob."
        ),
        license=(
            "Distributed under the same license as the Slow Wave repository."
        ),
        restrictions=(
            "No third-party IP, export controls, or usage fees apply; the data "
            "are wholly synthetic."
        ),
    )


def _build_maintenance(stream_id: str) -> Maintenance:
    """Build the maintenance section, embedding the reproducible stream id."""
    return Maintenance(
        maintainer=(
            "Maintained by the Slow Wave bench maintainers as part of the "
            "slow_wave package."
        ),
        contact="Via the Slow Wave repository issue tracker.",
        updates=(
            "The dataset definition evolves only with the generator; any change "
            "to the generator or config is reflected in the stream_id and "
            "config hash. Streams are not patched in place -- they are "
            "regenerated from seed."
        ),
        versioning=(
            f"Reproducible identity: stream_id '{stream_id}' encodes the "
            "scenario, seed, and a config hash. An older version is recovered "
            "by regenerating from the same seed and config."
        ),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_datasheet(
    stream: Stream, config: StreamGenConfig | None = None
) -> Datasheet:
    """Build a Gebru-style :class:`Datasheet` for ``stream``.

    Reads relevance labels via the sanctioned offline accessor
    :func:`slow_wave.stream.schema.offline_labels` (allowed for offline scoring
    code) to compute the label distribution, proportions, and contradiction
    count. The result is fully determined by ``stream`` and ``config``, with all
    floats rounded to 6 decimal places, so two builds from the same stream
    serialize identically.

    Args:
        stream: The stream to describe.
        config: The generation config to report. Defaults to ``stream.config``.

    Returns:
        A populated :class:`Datasheet`.
    """
    cfg = config if config is not None else stream.config
    labels = offline_labels(stream)

    n_items = len(stream.items)
    label_distribution = _label_distribution(labels)
    label_proportions = _label_proportions(label_distribution, n_items)
    requested_label_mix = {
        label: round(float(value), _FLOAT_NDIGITS)
        for label, value in cfg.label_mix.as_dict().items()
    }
    n_contradictions = _count_contradictions(stream, labels)

    regime = RegimeParams(
        contradiction_rate=round(float(cfg.contradiction_rate), _FLOAT_NDIGITS),
        drift=round(float(cfg.drift), _FLOAT_NDIGITS),
        recency_bias=round(float(cfg.recency_bias), _FLOAT_NDIGITS),
        n_subjects_per_task=cfg.n_subjects_per_task,
        n_attributes=cfg.n_attributes,
        n_values=cfg.n_values,
        probes_per_task=cfg.probes_per_task,
        distractor_namespace_size=cfg.distractor_namespace_size,
        noise_vocab_size=cfg.noise_vocab_size,
        noise_tokens=cfg.noise_tokens,
    )

    return Datasheet(
        motivation=_build_motivation(),
        composition=_build_composition(
            n_items=n_items,
            n_tasks=stream.n_tasks,
            label_distribution=label_distribution,
            n_contradictions=n_contradictions,
        ),
        collection_process=_build_collection_process(stream.scenario),
        preprocessing=_build_preprocessing(),
        uses=_build_uses(),
        distribution=_build_distribution(),
        maintenance=_build_maintenance(stream.stream_id),
        scenario=stream.scenario,
        seed=stream.seed,
        stream_id=stream.stream_id,
        n_tasks=stream.n_tasks,
        n_items=n_items,
        label_distribution=label_distribution,
        label_proportions=label_proportions,
        requested_label_mix=requested_label_mix,
        regime=regime,
        label_provenance=(
            "Relevance labels (signal/distractor/noise) were assigned "
            "synthetically by the Slow Wave stream generator at generation "
            "time, by design: each item's label is determined by the requested "
            "label mix and the generator's deterministic allocation, not by any "
            "post-hoc human or model annotation. Labels are offline-only "
            "ground truth and are not part of the online item view (FR1.6)."
        ),
        n_contradictions=n_contradictions,
    )


def datasheet_to_json(ds: Datasheet) -> str:
    """Serialize ``ds`` to deterministic, pretty JSON.

    The JSON is produced with ``sort_keys=True`` and ``indent=2`` plus a
    trailing newline, matching the house style in
    :func:`slow_wave.repro.manifest.write_manifest`. Combined with the 6-dp
    float rounding in :func:`build_datasheet`, two datasheets built from the
    same stream yield byte-identical output.

    Args:
        ds: The datasheet to serialize.

    Returns:
        The JSON string (terminated by a single ``"\\n"``).
    """
    return json.dumps(ds.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def validate_datasheet(data: dict) -> Datasheet:
    """Validate ``data`` against the :class:`Datasheet` schema.

    This is the schema check for Phase 1 exit criterion #5: the model is the
    schema, so round-tripping a serialized datasheet back through
    :meth:`Datasheet.model_validate` proves the emitted JSON conforms.

    Args:
        data: A dict (e.g. ``json.loads`` of :func:`datasheet_to_json` output).

    Returns:
        The validated :class:`Datasheet`.

    Raises:
        pydantic.ValidationError: If ``data`` does not satisfy the schema.
    """
    return Datasheet.model_validate(data)
