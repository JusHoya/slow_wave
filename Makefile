# Slow Wave — developer task runner (POSIX make; used by Linux CI).
#
# Windows has no `make` by default. Run the equivalent commands directly, e.g.
# the canonical one-command smoke reproduction:
#
#     python -m slow_wave.repro.smoke --config configs/smoke.yaml
#
# (Recipe lines below are TAB-indented, as POSIX make requires.)

.PHONY: setup test repro-smoke repro-stream repro-agent repro-dream repro-eval lint

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

lint:
	python -c "import slow_wave; print(slow_wave.__version__)"
