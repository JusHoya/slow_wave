"""Synthetic continual task stream generator for the Slow Wave bench (Phase 1).

The public surface is re-exported here for convenience; see the per-module
docstrings and ``docs/PHASE1_CONTRACT.md`` for details:

* :mod:`slow_wave.stream.schema` — the shared data model (types + invariants).
* :mod:`slow_wave.stream.generator` — :func:`generate_stream` (deterministic).
* :mod:`slow_wave.stream.datasheet` — :func:`build_datasheet` (Gebru et al. 2021).
* :mod:`slow_wave.stream.probes` — probe set, oracle, ``R[i,j]`` accuracy matrix.
* :mod:`slow_wave.stream.guard` — the FR1.6 confound guard (label-leak detection).
* :mod:`slow_wave.stream.emit` — one-command end-to-end artifact emission.
"""

from __future__ import annotations

from slow_wave.stream.datasheet import (
    Datasheet,
    build_datasheet,
    datasheet_to_json,
)
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.guard import (
    BANNED_FIELD_NAMES,
    ConfoundLeakError,
    assert_no_label_leak,
    assert_online_view_is_clean,
    online_view,
)
from slow_wave.stream.probes import (
    OracleAgent,
    build_probe_set,
    compute_accuracy_matrix,
)
from slow_wave.stream.schema import (
    AccuracyMatrix,
    CLScenario,
    Fact,
    GroundTruth,
    ItemKind,
    Label,
    LabelMix,
    Probe,
    ProbeSet,
    Stream,
    StreamGenConfig,
    StreamItem,
    assert_same_scenario,
    assert_single_scenario,
    offline_labels,
)

__all__ = [
    # schema
    "AccuracyMatrix",
    "CLScenario",
    "Fact",
    "GroundTruth",
    "ItemKind",
    "Label",
    "LabelMix",
    "Probe",
    "ProbeSet",
    "Stream",
    "StreamGenConfig",
    "StreamItem",
    "assert_same_scenario",
    "assert_single_scenario",
    "offline_labels",
    # generator
    "generate_stream",
    # datasheet
    "Datasheet",
    "build_datasheet",
    "datasheet_to_json",
    # probes
    "OracleAgent",
    "build_probe_set",
    "compute_accuracy_matrix",
    # guard
    "BANNED_FIELD_NAMES",
    "ConfoundLeakError",
    "assert_no_label_leak",
    "assert_online_view_is_clean",
    "online_view",
]
