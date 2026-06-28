"""End-to-end emission + determinism test for the Phase 1 stream (exit #1).

This is the integration-level check that ties the four Phase 1 workstreams
together through :mod:`slow_wave.stream.emit`: two emissions from the same
config + seed must produce **byte-identical** stream and datasheet artifacts (and
the probe/accuracy companions), and the emitted stream must validate against the
shared schema. The canonical ``configs/stream_smoke.yaml`` is exercised exactly
as the one-command CLI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slow_wave.config import Config, load_config
from slow_wave.stream.emit import (
    ACCURACY_FILE,
    DATASHEET_FILE,
    PROBES_FILE,
    STREAM_FILE,
    emit_stream,
    run_emit,
)
from slow_wave.stream.schema import (
    AccuracyMatrix,
    CLScenario,
    ProbeSet,
    Stream,
    StreamGenConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM_SMOKE_YAML = REPO_ROOT / "configs" / "stream_smoke.yaml"

ARTIFACTS = [STREAM_FILE, DATASHEET_FILE, PROBES_FILE, ACCURACY_FILE]


def _files(root: Path) -> list[str]:
    """Sorted POSIX-relative paths of every file under ``root``."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_stream_smoke_config_loads() -> None:
    """The canonical stream smoke config loads and has a stream section."""
    cfg = load_config(STREAM_SMOKE_YAML)
    assert isinstance(cfg, Config)
    assert cfg.stream is not None
    assert cfg.stream.scenario == CLScenario.TASK_INCREMENTAL


def test_run_emit_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """Same config + seed -> byte-identical artifacts and identical layout (exit #1)."""
    cfg = load_config(STREAM_SMOKE_YAML)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    written_a = run_emit(cfg, out_a)
    written_b = run_emit(cfg, out_b)

    # Identical file layout under each output root.
    assert _files(out_a) == _files(out_b)

    # Every artifact is byte-for-byte identical.
    for name in ("stream", "datasheet", "probes", "accuracy_matrix"):
        bytes_a = Path(written_a[name]).read_bytes()
        bytes_b = Path(written_b[name]).read_bytes()
        assert bytes_a == bytes_b, f"{name} differs between two emissions"


def test_emitted_stream_validates_against_schema(tmp_path: Path) -> None:
    """The emitted stream.json round-trips into a valid Stream with the right scenario."""
    cfg = load_config(STREAM_SMOKE_YAML)
    written = run_emit(cfg, tmp_path)

    data = json.loads(Path(written["stream"]).read_text(encoding="utf-8"))
    stream = Stream.model_validate(data)
    assert stream.scenario == CLScenario.TASK_INCREMENTAL
    assert stream.n_tasks == cfg.stream.n_tasks
    assert len(stream.items) == cfg.stream.n_tasks * cfg.stream.items_per_task
    # Every item has exactly one label (Stream's own validator enforces this).
    assert len(stream.ground_truth.labels) == len(stream.items)


def test_emitted_probes_and_matrix_validate(tmp_path: Path) -> None:
    """The probe set and accuracy matrix artifacts validate and are well-formed."""
    cfg = load_config(STREAM_SMOKE_YAML)
    written = run_emit(cfg, tmp_path)

    probes = ProbeSet.model_validate(
        json.loads(Path(written["probes"]).read_text(encoding="utf-8"))
    )
    assert probes.probes, "expected a non-empty probe set"

    matrix = AccuracyMatrix.model_validate(
        json.loads(Path(written["accuracy_matrix"]).read_text(encoding="utf-8"))
    )
    n = cfg.stream.n_tasks
    assert matrix.n_tasks == n
    assert len(matrix.R) == n and all(len(row) == n for row in matrix.R)
    # contradiction_rate is 0.0 in the smoke config -> crisp lower-triangular ones.
    for i in range(n):
        for j in range(n):
            expected = 1.0 if j <= i else 0.0
            assert matrix.R[i][j] == expected, f"R[{i}][{j}]={matrix.R[i][j]}"


def test_emit_stream_direct_api(tmp_path: Path) -> None:
    """The lower-level emit_stream API writes all four artifacts under <out>/stream/."""
    cfg = StreamGenConfig(n_tasks=3, items_per_task=20)
    written = emit_stream(cfg, seed=123, out_dir=tmp_path)
    for name in ("stream", "datasheet", "probes", "accuracy_matrix"):
        assert Path(written[name]).is_file()
    assert {Path(p).name for p in written.values()} == set(ARTIFACTS)


def test_run_emit_without_stream_section_raises(tmp_path: Path) -> None:
    """A config lacking a stream section is a loud error, not a silent no-op."""
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")  # Phase 0 config: no stream
    assert cfg.stream is None
    with pytest.raises(ValueError, match="no `stream:` section"):
        run_emit(cfg, tmp_path)
