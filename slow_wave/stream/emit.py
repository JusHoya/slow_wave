"""End-to-end stream emission for the Slow Wave bench (Phase 1).

One command materializes a complete synthetic continual task stream and all of
its companion artifacts to disk, deterministically::

    python -m slow_wave.stream.emit --config configs/stream_smoke.yaml

Given a fixed config + seed, every emitted file is **byte-identical** across
runs (Phase 1 exit criterion #1). The emitter wires together the four Phase 1
workstreams:

* ``generator.generate_stream``     -> ``stream.json``           (online-safe items + offline labels)
* ``datasheet.build_datasheet``     -> ``datasheet.json``        (Gebru et al. 2021)
* ``probes.build_probe_set``        -> ``probes.json``           (held-out queries + known answers)
* ``probes.compute_accuracy_matrix``-> ``accuracy_matrix.json``  (R[i,j] vs. the trivial oracle)

As a belt-and-braces honesty check, the confound guard
(``guard.assert_online_view_is_clean``) runs before anything is written, so a
stream that could leak a ground-truth label into the online view can never be
emitted (FR1.6).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from slow_wave.config import Config, load_config
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream.datasheet import build_datasheet, datasheet_to_json
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.guard import assert_online_view_is_clean
from slow_wave.stream.probes import build_probe_set, compute_accuracy_matrix
from slow_wave.stream.schema import Stream, StreamGenConfig

logger = logging.getLogger(__name__)

# The artifact file names emitted under ``<out>/stream/``.
STREAM_FILE = "stream.json"
DATASHEET_FILE = "datasheet.json"
PROBES_FILE = "probes.json"
ACCURACY_FILE = "accuracy_matrix.json"


def _dump_json(model_json_obj, path: Path) -> Path:
    """Write ``model_json_obj`` to ``path`` as deterministic, pretty JSON.

    Serialized with ``sort_keys=True`` and ``indent=2`` plus a trailing newline
    so output is stable and diff-friendly across runs.

    Args:
        model_json_obj: A JSON-serializable object (typically the result of a
            pydantic ``model_dump(mode="json")``).
        path: Destination file path; parent directories are created if needed.

    Returns:
        The :class:`~pathlib.Path` written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model_json_obj, sort_keys=True, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def emit_stream(
    stream_config: StreamGenConfig,
    seed: int,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Generate a stream and write all Phase 1 artifacts to disk.

    Steps:
        1. ``generate_stream(stream_config, seed)``.
        2. Confound guard: assert the online view leaks no label (FR1.6).
        3. Build the datasheet, probe set, and accuracy matrix.
        4. Write ``stream.json``, ``datasheet.json``, ``probes.json`` and
           ``accuracy_matrix.json`` under ``<out_dir>/stream/`` as deterministic
           JSON.

    Args:
        stream_config: The :class:`StreamGenConfig` describing the stream.
        seed: The stream seed (a stable child seed; see :func:`run_emit`).
        out_dir: Output root; artifacts go under ``<out_dir>/stream/``.

    Returns:
        A mapping ``{"stream", "datasheet", "probes", "accuracy_matrix"}`` to the
        written paths.
    """
    stream: Stream = generate_stream(stream_config, seed)

    # Honesty by construction: refuse to emit a stream whose online view could
    # leak a ground-truth relevance label into a retrieval/priority path.
    assert_online_view_is_clean(stream)

    datasheet = build_datasheet(stream)
    probe_set = build_probe_set(stream)
    accuracy = compute_accuracy_matrix(stream, probe_set)

    base = Path(out_dir) / "stream"
    written = {
        "stream": _dump_json(stream.model_dump(mode="json"), base / STREAM_FILE),
        "datasheet": _write_text(datasheet_to_json(datasheet), base / DATASHEET_FILE),
        "probes": _dump_json(probe_set.model_dump(mode="json"), base / PROBES_FILE),
        "accuracy_matrix": _dump_json(
            accuracy.model_dump(mode="json"), base / ACCURACY_FILE
        ),
    }

    print(f"[stream] stream_id={stream.stream_id} scenario={stream.scenario.value}")
    print(
        f"[stream] n_items={len(stream.items)} n_tasks={stream.n_tasks} "
        f"n_probes={len(probe_set.probes)}"
    )
    for name, path in written.items():
        print(f"[stream] wrote {name}: {path}")
    return written


def _write_text(text: str, path: Path) -> Path:
    """Write already-serialized ``text`` to ``path`` verbatim (parents ok)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_emit(cfg: Config, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Emit a stream from a loaded :class:`Config`.

    The stream seed is derived from ``cfg.seed`` via
    ``derive_seed(cfg.seed, "stream")`` (the same namespaced derivation the
    Phase 0 smoke run uses), so the stream is reproducible from the config alone.

    Args:
        cfg: A loaded config whose ``stream`` section is populated.
        out_dir: Output root; defaults to ``cfg.output_dir``.

    Returns:
        The mapping of artifact name to written path (see :func:`emit_stream`).

    Raises:
        ValueError: If ``cfg.stream`` is ``None`` (no stream section configured).
    """
    if cfg.stream is None:
        raise ValueError(
            "config has no `stream:` section; add one (see "
            "configs/stream_smoke.yaml) to emit a stream."
        )
    set_global_seeds(cfg.seed)
    stream_seed = derive_seed(cfg.seed, "stream")
    return emit_stream(cfg.stream, stream_seed, out_dir or cfg.output_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.stream.emit``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-stream",
        description="Emit a synthetic continual task stream and its artifacts.",
    )
    parser.add_argument(
        "--config",
        default="configs/stream_smoke.yaml",
        help="Path to the experiment YAML config (default: configs/stream_smoke.yaml).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default: the config's output_dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    written = run_emit(cfg, args.out)
    print(written["stream"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
