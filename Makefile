# Slow Wave — developer task runner (POSIX make; used by Linux CI).
#
# Windows has no `make` by default. Run the equivalent commands directly, e.g.
# the canonical one-command smoke reproduction:
#
#     python -m slow_wave.repro.smoke --config configs/smoke.yaml
#
# (Recipe lines below are TAB-indented, as POSIX make requires.)

.PHONY: setup test repro-smoke repro-stream repro-agent repro-dream repro-eval repro-phase5 repro-figures repro-numbers repro-paper lint

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

repro-smoke:
	python -m slow_wave.repro.smoke --config configs/smoke.yaml

# Phase 1: emit a synthetic continual task stream + datasheet + probe set + R[i,j].
repro-stream:
	python -m slow_wave.stream.emit --config configs/stream_smoke.yaml

# Phase 2: run the no-sleep wake agent end-to-end; writes R[i,j] + cost telemetry.
repro-agent:
	python -m slow_wave.agent.runner --config configs/agent_smoke.yaml

# Phase 3: run the sleep-enabled dream agent; writes R[i,j] + wake/dream telemetry
# + a provenance/archival audit. (configs/dream_full.yaml is the treatment arm.)
repro-dream:
	python -m slow_wave.dream.runner --config configs/dream_smoke.yaml

# Phase 4: run the nine-arm control battery on one shared stream per seed at
# matched budgets; writes metrics + statistics + A/A noise floor + the prereg
# primary endpoint + bias controls into a single experiment manifest.
# (configs/eval_full.yaml is the science-scale >=5-seed grid.)
repro-eval:
	python -m slow_wave.eval.runner --config configs/eval_smoke.yaml

# Phase 5: run the full preregistered experiment grid (arm x distractor-regime x
# seed), the stream-length sweep (long-context crossover), and the sim-vs-real
# long-horizon runs; then compute the analysis (primary-endpoint verdict,
# crossover, TMR, power, negative-result mapping) and regenerate every figure.
# Writes the committed artifacts under paper/data/ + paper/figures/.
# (configs/phase5_full.yaml is the science-scale 8-seed grid.)
repro-phase5:
	python -m slow_wave.eval.grid --config configs/phase5_full.yaml --out .
	python -m slow_wave.eval.analysis --result phase5/phase5_result.json --out .
	python -m slow_wave.paper.figures --result phase5/phase5_result.json --out paper/figures

# Phase 5 headline-figure reproduction (FR6.2/DX1/EC4): regenerate EVERY figure
# from the COMMITTED Phase 5 result manifest in one command — no re-run needed.
repro-figures:
	python -m slow_wave.paper.figures --result phase5/phase5_result.json --out paper/figures

# Phase 6: regenerate the manuscript's in-text number macros + per-arm results
# table from the COMMITTED Phase 5 result (every paper number is regenerable;
# EC2/EC8). Writes paper/generated/numbers.tex + paper/generated/arm_metrics_table.tex.
repro-numbers:
	python -m slow_wave.paper.numbers --result phase5/phase5_result.json --out paper/generated

# Phase 6: one-command manuscript build -- regenerate numbers + figures from the
# committed result, then compile paper/main.tex -> paper/main.pdf (needs a LaTeX
# toolchain: latexmk + pdflatex + bibtex). See scripts/build_paper.sh.
repro-paper:
	bash scripts/build_paper.sh

lint:
	python -c "import slow_wave; print(slow_wave.__version__)"
