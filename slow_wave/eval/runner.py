"""One-command Phase 4 evaluation run + manifest writer (FR5.x).

Runs the nine-arm control battery on one shared stream per seed, matches budgets,
computes the metric + statistics suites, the A/A noise floor, the preregistered
primary endpoint, and the temperature-0 stability + memory-drift bias controls,
then writes a single experiment manifest. One command::

    python -m slow_wave.eval.runner --config configs/eval_smoke.yaml

The manifest (``<out>/eval/manifest.json``) satisfies the Phase 4 exit criteria:
every arm ran on the same stream via one harness (EC1), the A/A control's noise
floor (EC2), the oracle prune ceiling (EC3), the matched-budget verdict + Pareto
frontier (EC4), the decoupled metric suite (EC5), the statistics suite (EC6), the
prereg primary endpoint computed exactly as registered (EC7), and the bias
controls (EC8). All non-LLM outputs are reproducible bit-for-bit given a fixed
config + seeds under the mock LLM (DX1).
"""

from __future__ import annotations

import argparse

from slow_wave.config import load_config
from slow_wave.eval.harness import run_experiment


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.eval.runner``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-eval",
        description="Run the Phase 4 nine-arm control battery and write its manifest.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval_smoke.yaml",
        help="Path to the experiment YAML config (default: configs/eval_smoke.yaml).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default: the config's output_dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = run_experiment(cfg, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
