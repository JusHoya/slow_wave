# Slow Wave — developer task runner (POSIX make; used by Linux CI).
#
# Windows has no `make` by default. Run the equivalent commands directly, e.g.
# the canonical one-command smoke reproduction:
#
#     python -m slow_wave.repro.smoke --config configs/smoke.yaml
#
# (Recipe lines below are TAB-indented, as POSIX make requires.)

.PHONY: setup test repro-smoke repro-stream repro-agent lint

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

lint:
	python -c "import slow_wave; print(slow_wave.__version__)"
