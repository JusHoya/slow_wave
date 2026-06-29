"""One-command sleep-enabled dream run + manifest writer (Phase 3, EC7).

Mirrors :mod:`slow_wave.agent.runner`, but wires the Phase 3
:class:`~slow_wave.dream.engine.DreamEngine` into the wake loop via the
:meth:`~slow_wave.dream.engine.DreamEngine.sleep_hook`: seed -> generate stream
-> build probe set -> embed -> run the
:class:`~slow_wave.agent.wake.WakeAgent` *with the dream sleep hook* -> assemble a
run manifest carrying the accuracy matrix ``R[i][j]``, continual-learning
metrics, the memory footprint, **both** wake and dream telemetry, and a
provenance + archival audit (EC7), then write it to
``<out>/dream/manifest.json``. One command::

    python -m slow_wave.dream.runner --config configs/dream_smoke.yaml

Because consolidation is gated to the sleep windows (FR4.5/EC3), the SEMANTIC
store is written only inside dream cycles; signals consolidated there survive
episodic eviction, so the post-cycle evaluation can answer probes whose source
episodics have been demoted to the archival tier. Every dream cycle demotes
(never hard-deletes), so the archival audit in the manifest is recoverable
end-to-end (EC7). All non-LLM outputs are reproducible bit-for-bit given a fixed
config + seed (DX1) under the mock LLM; the LLM token/wall-clock fields are
flagged nondeterministic in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from slow_wave.agent.wake import WakeAgent
from slow_wave.config import Config, load_config
from slow_wave.dream.engine import DreamEngine
from slow_wave.embeddings import get_embedder
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.repro.gitinfo import git_info
from slow_wave.repro.manifest import new_manifest, write_manifest
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set

logger = logging.getLogger(__name__)


@dataclass
class _AggregateLLM:
    """A summed-cost LLM-result stand-in for the run's manifest cost block.

    The dream run makes wake reasoning calls *and* dream (transfer + augment)
    calls; the manifest carries a single LLM-cost record, so this aggregates the
    combined token totals into the duck-type
    :func:`slow_wave.repro.manifest.new_manifest` expects.

    Attributes:
        text: A short human-readable summary of the run.
        model_id: The model id used for the calls.
        input_tokens: Summed prompt tokens across wake + dream calls.
        output_tokens: Summed completion tokens across wake + dream calls.
        mocked: ``True`` unless at least one genuine (non-mocked) call was made.
        stop_reason: Always ``"aggregate"`` (a synthetic roll-up).
    """

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    mocked: bool
    stop_reason: str = "aggregate"


def provenance_archival_audit(substrate: MemorySubstrate) -> dict:
    """Return the EC7 provenance + archival integrity audit for a run.

    Verifies, over the live substrate, that the dream engine left audit trails
    intact and demoted-not-deleted:

    * every SEMANTIC entry's :attr:`~slow_wave.memory.schema.MemoryEntry.provenance`
      can be traced to a source that still exists somewhere (a live episodic
      entry or an archived one);
    * every archived entry is recoverable via
      :meth:`~slow_wave.memory.stores.ArchivalStore.recover` (no hard deletes).

    Args:
        substrate: The memory substrate after a dream run.

    Returns:
        A JSON-safe dict with per-tier counts, the traceable/recoverable counts,
        an example semantic-provenance trace, and the failure-event count.
    """
    semantic_entries = substrate.semantic.all_entries()
    n_traceable = 0
    example_trace: dict | None = None
    for entry in semantic_entries:
        traced_source: str | None = None
        for src_id in entry.provenance:
            if (
                substrate.episodic.get(src_id) is not None
                or substrate.archival.contains(src_id)
            ):
                traced_source = src_id
                break
        if traced_source is not None:
            n_traceable += 1
            if example_trace is None:
                example_trace = {
                    "semantic_entry": entry.entry_id,
                    "source": traced_source,
                    "fact": entry.fact.model_dump() if entry.fact is not None else None,
                }

    archival_entries = substrate.archival.all_entries()
    n_recoverable = sum(
        1
        for entry in archival_entries
        if substrate.archival.recover(entry.entry_id) is not None
    )

    return {
        "n_semantic": len(semantic_entries),
        "n_semantic_traceable": n_traceable,
        "n_archival": len(archival_entries),
        "n_archival_recoverable": n_recoverable,
        "example_semantic_provenance": example_trace,
        "n_failure_events": len(substrate.failure_events),
    }


def _dream_cycle_digest(dream_telemetry) -> str:
    """Return a stable sha256 digest of the run's dream-cycle structure.

    Folds the load-bearing deterministic identifiers of every cycle (operators
    run, replayed/written/pseudo ids) into one hash so a determinism check can
    compare whole runs with a single value.

    Args:
        dream_telemetry: The run's :class:`~slow_wave.dream.schema.DreamTelemetry`.

    Returns:
        A 64-char hex sha256 digest.
    """
    payload = []
    for cycle in dream_telemetry.cycles:
        payload.append(
            {
                "operators_run": cycle.operators_run,
                "replay": cycle.replay.sampled_ids() if cycle.replay else [],
                "written": cycle.transfer.written_entry_ids if cycle.transfer else [],
                "pseudo": cycle.augment.pseudo_entry_ids if cycle.augment else [],
                "demoted": cycle.conflict.demoted_entry_ids if cycle.conflict else [],
            }
        )
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_dream(cfg: Config, out_dir: str | Path | None = None) -> Path:
    """Run the sleep-enabled dream agent end-to-end and write its manifest (EC7).

    Steps:
        1. Require ``cfg.stream`` (raise :class:`ValueError` if ``None``).
        2. Seed global RNGs; generate the stream from the ``"stream"``-derived seed.
        3. Build the probe set and the embedder.
        4. Run :class:`~slow_wave.agent.wake.WakeAgent` over the stream with the
           :class:`~slow_wave.dream.engine.DreamEngine` sleep hook attached, so a
           dream cycle runs at each scheduled sleep window.
        5. Assemble an aggregate (wake + dream) LLM-cost record and a manifest
           carrying ``R[i,j]``, metrics, footprint, wake + dream telemetry, and
           the provenance/archival audit.
        6. Write the manifest to ``<out>/dream/manifest.json`` and return its path.

    Args:
        cfg: The experiment configuration; ``cfg.stream`` must be set and
            ``cfg.dream.enabled`` should be ``True`` (a disabled dream config runs
            but produces an empty dream telemetry — i.e. the no-sleep baseline).
        out_dir: Output root; defaults to ``cfg.output_dir``. The manifest is
            written under ``<out_dir>/dream/manifest.json``.

    Returns:
        The path to the written manifest.

    Raises:
        ValueError: If ``cfg.stream`` is ``None`` (no stream to run).
    """
    start = time.perf_counter()

    if cfg.stream is None:
        raise ValueError(
            "run_dream requires cfg.stream to be set (no stream to process)."
        )

    set_global_seeds(cfg.seed)
    agent_seed = derive_seed(cfg.seed, "agent")
    stream_seed = derive_seed(cfg.seed, "stream")
    stream = generate_stream(cfg.stream, stream_seed)

    probe_set = build_probe_set(stream)
    embedder = get_embedder(cfg)

    engine = DreamEngine(cfg)
    agent = WakeAgent(cfg, embedder)
    result = agent.run(stream, probe_set, sleep_hook=engine.sleep_hook)

    wall = time.perf_counter() - start

    wake_tel = result.telemetry
    dream_tel = engine.telemetry
    matrix = result.accuracy_matrix
    metrics = result.metrics
    footprint = result.footprint
    substrate = result.substrate

    total_input = wake_tel.input_tokens + dream_tel.input_tokens
    total_output = wake_tel.output_tokens + dream_tel.output_tokens
    total_api = wake_tel.api_calls + dream_tel.api_calls

    summary = (
        f"dream agent: {dream_tel.n_cycles} cycle(s), "
        f"{wake_tel.api_calls} wake + {dream_tel.api_calls} dream call(s) over "
        f"stream {stream.stream_id}; ACC={metrics.acc:.4f} BWT={metrics.bwt:.4f}"
    )
    aggregate_llm = _AggregateLLM(
        text=summary,
        model_id=cfg.model.id,
        input_tokens=total_input,
        output_tokens=total_output,
        mocked=agent.n_real_calls == 0,
    )

    audit = provenance_archival_audit(substrate)

    deterministic_probe = {
        "accuracy_matrix": matrix.R,
        "n_tasks": matrix.n_tasks,
        "n_probes": len(probe_set.probes),
        "memory_footprint": footprint.model_dump(mode="json"),
        "n_dream_cycles": dream_tel.n_cycles,
        "n_semantic_written": dream_tel.n_semantic_written,
        "n_pseudo": dream_tel.n_pseudo,
        "n_demoted_conflict": dream_tel.n_demoted_conflict,
        "n_evicted": wake_tel.n_evicted,
        "n_items": wake_tel.n_items_ingested,
        "dream_cycle_digest": _dream_cycle_digest(dream_tel),
        "provenance_archival_audit": audit,
    }
    results = {
        "accuracy_matrix": matrix.model_dump(mode="json"),
        "continual_metrics": metrics.model_dump(mode="json"),
        "memory_footprint": footprint.model_dump(mode="json"),
        "wake_telemetry": wake_tel.as_dict(),
        "dream_telemetry": dream_tel.model_dump(mode="json"),
        "provenance_archival_audit": audit,
    }

    manifest = new_manifest(
        cfg=cfg,
        embedder=embedder,
        llm=aggregate_llm,
        seeds={"master": cfg.seed, "agent": agent_seed, "stream": stream_seed},
        deterministic_probe=deterministic_probe,
        wall_clock_s=wall,
        git=git_info(),
        api_calls=total_api,
        results=results,
    )

    out_path = Path(out_dir or cfg.output_dir) / "dream" / "manifest.json"
    written = write_manifest(manifest, out_path)

    fallback_reason = getattr(embedder, "fallback_reason", None)
    if fallback_reason:
        logger.warning("Embedder fell back: %s", fallback_reason)

    print(f"[dream] manifest written to {written}")
    print(
        f"[dream] stream={stream.stream_id} ACC={metrics.acc:.4f} "
        f"BWT={metrics.bwt:.4f} cycles={dream_tel.n_cycles} "
        f"semantic_written={dream_tel.n_semantic_written} pseudo={dream_tel.n_pseudo} "
        f"api_calls={total_api} total_tokens={total_input + total_output}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.dream.runner``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-dream",
        description="Run the Phase 3 sleep-enabled dream agent and write its manifest.",
    )
    parser.add_argument(
        "--config",
        default="configs/dream_smoke.yaml",
        help="Path to the experiment YAML config (default: configs/dream_smoke.yaml).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default: the config's output_dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = run_dream(cfg, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
