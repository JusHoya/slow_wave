"""The Phase 0 "hello-bench" smoke run.

This is the end-to-end reproducibility check that every later phase builds on
(``docs/PHASE0_CONTRACT.md``). One command::

    python -m slow_wave.repro.smoke --config configs/smoke.yaml

seeds the RNGs, embeds a tiny deterministic corpus, draws a sampling order,
makes one (mock-by-default) Claude call, and writes a JSON run manifest
containing every FR6.1 field.

The non-LLM outputs (embedding hash, sampling order, file layout) are
reproducible bit-for-bit given a fixed config + seed; LLM-dependent fields are
flagged in the manifest's ``nondeterministic_fields``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from slow_wave.config import Config, load_config
from slow_wave.embeddings import embed_texts, embedding_sha256, get_embedder
from slow_wave.llm import complete
from slow_wave.repro.gitinfo import git_info
from slow_wave.repro.manifest import new_manifest, write_manifest
from slow_wave.repro.seeding import derive_seed, set_global_seeds

logger = logging.getLogger(__name__)


def run_smoke(cfg: Config, out_dir: str | Path | None = None) -> Path:
    """Run the hello-bench smoke and write its manifest.

    Steps (per the Phase 0 contract):
        1. Seed global RNGs from ``cfg.seed``; derive ``agent`` and ``stream`` seeds.
        2. Build the corpus (``cfg.smoke.texts`` or a deterministic default).
        3. Embed the corpus with the configured embedder.
        4. Draw a deterministic sampling order from the ``stream`` seed.
        5. Make one completion (mock by default; real if ``ANTHROPIC_API_KEY``).
        6. Assemble the run manifest with a deterministic probe + git/cost info.
        7. Write the manifest to ``<out>/smoke/manifest.json`` and return its path.

    Args:
        cfg: The experiment configuration.
        out_dir: Output root; defaults to ``cfg.output_dir``. The manifest is
            written under ``<out_dir>/smoke/manifest.json``.

    Returns:
        The path to the written manifest.
    """
    start = time.perf_counter()

    # 1) Seeding.
    set_global_seeds(cfg.seed)
    agent_seed = derive_seed(cfg.seed, "agent")
    stream_seed = derive_seed(cfg.seed, "stream")

    # 2) Deterministic corpus.
    texts = cfg.smoke.texts or [
        f"slow-wave smoke item {i}" for i in range(cfg.smoke.n_items)
    ]

    # 3) Embeddings.
    embedder = get_embedder(cfg)
    emb = embed_texts(embedder, texts)

    # 4) Deterministic sampling order from the stream seed.
    sampling_order = np.random.default_rng(stream_seed).permutation(len(texts)).tolist()

    # 5) One completion (mock unless an API key is present).
    llm = complete(cfg, cfg.smoke.prompt)

    wall = time.perf_counter() - start

    # 6) Manifest assembly.
    deterministic_probe = {
        "embedding_sha256": embedding_sha256(emb),
        "sampling_order": sampling_order,
        "n_items": len(texts),
        "embedder_backend": embedder.backend,
    }
    manifest = new_manifest(
        cfg=cfg,
        embedder=embedder,
        llm=llm,
        seeds={"master": cfg.seed, "agent": agent_seed, "stream": stream_seed},
        deterministic_probe=deterministic_probe,
        wall_clock_s=wall,
        git=git_info(),
    )

    # 7) Write.
    out_path = Path(out_dir or cfg.output_dir) / "smoke" / "manifest.json"
    written = write_manifest(manifest, out_path)

    # Honesty by construction: surface an embedder fallback if one happened.
    fallback_reason = getattr(embedder, "fallback_reason", None)
    if fallback_reason:
        logger.warning("Embedder fell back: %s", fallback_reason)

    print(f"[smoke] manifest written to {written}")
    print(
        f"[smoke] llm={'mocked' if llm.mocked else 'real'} "
        f"embedder_backend={embedder.backend} n_items={len(texts)}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.repro.smoke`` / ``slow-wave-smoke``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-smoke",
        description="Run the Phase 0 hello-bench smoke and write a run manifest.",
    )
    parser.add_argument(
        "--config",
        default="configs/smoke.yaml",
        help="Path to the experiment YAML config (default: configs/smoke.yaml).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default: the config's output_dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = run_smoke(cfg, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
