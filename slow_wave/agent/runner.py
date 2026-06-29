"""One-command no-sleep wake run + manifest writer (Phase 2, WS-AGENT, EC1).

Mirrors :mod:`slow_wave.repro.smoke`: seed -> generate stream -> build probe set
-> embed -> run the :class:`~slow_wave.agent.wake.WakeAgent` -> assemble a run
manifest carrying the accuracy matrix ``R[i][j]`` *and* cost telemetry, then
write it to ``<out>/agent/manifest.json``. One command::

    python -m slow_wave.agent.runner --config configs/agent_smoke.yaml

The manifest satisfies Phase 2 exit criterion #1: it records the populated
``R[i,j]`` (``results.accuracy_matrix``), continual-learning metrics, the memory
footprint, and full cost telemetry (``cost.tokens.*``, ``cost.api_calls``,
``results.telemetry``). All non-LLM outputs are reproducible bit-for-bit given a
fixed config + seed (DX1); the LLM token/wall-clock fields are flagged
nondeterministic in the manifest.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from slow_wave.agent.wake import WakeAgent
from slow_wave.config import Config, load_config
from slow_wave.embeddings import get_embedder
from slow_wave.repro.gitinfo import git_info
from slow_wave.repro.manifest import new_manifest, write_manifest
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set

logger = logging.getLogger(__name__)


@dataclass
class _AggregateLLM:
    """A summed-cost LLM-result stand-in for the run's manifest cost block.

    The wake agent may make many (per-task) reasoning calls; the manifest carries
    a single LLM-cost record, so this aggregates their token totals into the
    duck-type :func:`slow_wave.repro.manifest.new_manifest` expects (exposing
    ``.text``, ``.model_id``, ``.input_tokens``, ``.output_tokens``, ``.mocked``,
    and ``.stop_reason``).

    Attributes:
        text: A short human-readable summary of the run (hashed/previewed into
            the manifest's ``llm`` block).
        model_id: The model id used for the reasoning calls.
        input_tokens: Summed prompt tokens across all reasoning calls.
        output_tokens: Summed completion tokens across all reasoning calls.
        mocked: ``True`` unless at least one genuine (non-mocked) call was made.
        stop_reason: Always ``"aggregate"`` (this is a synthetic roll-up).
    """

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    mocked: bool
    stop_reason: str = "aggregate"


def run_agent(cfg: Config, out_dir: str | Path | None = None) -> Path:
    """Run the no-sleep wake agent end-to-end and write its manifest (EC1).

    Steps:
        1. Require ``cfg.stream`` (raise :class:`ValueError` if ``None``, mirroring
           the stream emitter).
        2. Seed global RNGs; generate the stream from the ``"stream"``-derived seed.
        3. Build the probe set and the embedder.
        4. Run :class:`~slow_wave.agent.wake.WakeAgent` over the stream.
        5. Assemble an aggregate LLM-cost record and a manifest carrying ``R[i,j]``
           plus full cost telemetry.
        6. Write the manifest to ``<out>/agent/manifest.json`` and return its path.

    Args:
        cfg: The experiment configuration; ``cfg.stream`` must be set.
        out_dir: Output root; defaults to ``cfg.output_dir``. The manifest is
            written under ``<out_dir>/agent/manifest.json``.

    Returns:
        The path to the written manifest.

    Raises:
        ValueError: If ``cfg.stream`` is ``None`` (no stream to run).
    """
    start = time.perf_counter()

    # 1) A wake run needs a stream to process.
    if cfg.stream is None:
        raise ValueError(
            "run_agent requires cfg.stream to be set (no stream to process)."
        )

    # 2) Seeding + deterministic stream.
    set_global_seeds(cfg.seed)
    agent_seed = derive_seed(cfg.seed, "agent")
    stream_seed = derive_seed(cfg.seed, "stream")
    stream = generate_stream(cfg.stream, stream_seed)

    # 3) Probe set + embedder.
    probe_set = build_probe_set(stream)
    embedder = get_embedder(cfg)

    # 4) Run the wake agent.
    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set)

    wall = time.perf_counter() - start

    telemetry = result.telemetry
    matrix = result.accuracy_matrix
    metrics = result.metrics
    footprint = result.footprint

    # 5) Aggregate LLM-cost record + manifest assembly.
    summary = (
        f"no-sleep wake agent: {telemetry.api_calls} reasoning call(s) over "
        f"stream {stream.stream_id}; ACC={metrics.acc:.4f} BWT={metrics.bwt:.4f}"
    )
    aggregate_llm = _AggregateLLM(
        text=summary,
        model_id=cfg.model.id,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        mocked=agent.n_real_calls == 0,
    )

    # Probe coverage (DX2 honesty): R[i,j] is scored over the held-out probe set,
    # whose per-task size is capped at cfg.stream.probes_per_task by build_probe_set
    # (the cap is logged there). Surface the realized coverage in the manifest so a
    # manifest-only audit can see how many probed keys R was computed over.
    per_task_probes: dict[str, int] = {}
    for probe in probe_set.probes:
        key = str(probe.task_index)
        per_task_probes[key] = per_task_probes.get(key, 0) + 1
    probe_coverage = {
        "n_probes": len(probe_set.probes),
        "probes_per_task_cap": cfg.stream.probes_per_task,
        "per_task_probe_counts": per_task_probes,
    }

    deterministic_probe = {
        "accuracy_matrix": matrix.R,
        "n_tasks": matrix.n_tasks,
        "n_probes": len(probe_set.probes),
        "memory_footprint": footprint.model_dump(mode="json"),
        "retrieval_calls": telemetry.retrieval_calls,
        "n_evicted": telemetry.n_evicted,
        "n_items": telemetry.n_items_ingested,
    }
    results = {
        "accuracy_matrix": matrix.model_dump(mode="json"),
        "continual_metrics": metrics.model_dump(mode="json"),
        "memory_footprint": footprint.model_dump(mode="json"),
        "probe_coverage": probe_coverage,
        "telemetry": telemetry.as_dict(),
    }

    manifest = new_manifest(
        cfg=cfg,
        embedder=embedder,
        llm=aggregate_llm,
        seeds={"master": cfg.seed, "agent": agent_seed, "stream": stream_seed},
        deterministic_probe=deterministic_probe,
        wall_clock_s=wall,
        git=git_info(),
        api_calls=telemetry.api_calls,
        results=results,
    )

    # 6) Write + summarize.
    out_path = Path(out_dir or cfg.output_dir) / "agent" / "manifest.json"
    written = write_manifest(manifest, out_path)

    fallback_reason = getattr(embedder, "fallback_reason", None)
    if fallback_reason:
        logger.warning("Embedder fell back: %s", fallback_reason)

    print(f"[agent] manifest written to {written}")
    print(
        f"[agent] stream={stream.stream_id} ACC={metrics.acc:.4f} "
        f"BWT={metrics.bwt:.4f} api_calls={telemetry.api_calls} "
        f"total_tokens={telemetry.total_tokens}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.agent.runner``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-agent",
        description="Run the Phase 2 no-sleep wake agent and write its manifest.",
    )
    parser.add_argument(
        "--config",
        default="configs/agent_smoke.yaml",
        help="Path to the experiment YAML config (default: configs/agent_smoke.yaml).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default: the config's output_dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = run_agent(cfg, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
