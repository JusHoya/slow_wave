# Slow Wave — developer task runner (POSIX make; used by Linux CI).
#
# Windows has no `make` by default. Run the equivalent commands directly, e.g.
# the canonical one-command smoke reproduction:
#
#     python -m slow_wave.repro.smoke --config configs/smoke.yaml
#
# (Recipe lines below are TAB-indented, as POSIX make requires.)

.PHONY: setup test repro-smoke lint

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

repro-smoke:
	python -m slow_wave.repro.smoke --config configs/smoke.yaml

lint:
	python -c "import slow_wave; print(slow_wave.__version__)"
