#!/usr/bin/env bash
# Build the Slow Wave manuscript end-to-end from committed data (Phase 6).
#
# Regenerates every in-text number macro and every figure from the committed
# Phase 5 result, then compiles the PDF. This is the one-command reproduction
# of the manuscript (EC2/EC8): nothing in the paper is hand-transcribed.
#
# Usage (from the repo root):
#   bash scripts/build_paper.sh
#
# Requires: the package installed (pip install -e ".[viz]") for matplotlib,
# and a LaTeX toolchain providing latexmk + pdflatex + bibtex (TeX Live / MiKTeX).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/3] Regenerating result-number macros + per-arm table ..."
python -m slow_wave.paper.numbers --result phase5/phase5_result.json --out paper/generated

echo "[2/3] Regenerating figures from the committed Phase 5 result ..."
python -m slow_wave.paper.figures --result phase5/phase5_result.json --out paper/figures

echo "[3/3] Compiling paper/main.tex -> paper/main.pdf ..."
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

echo "Done: paper/main.pdf"
