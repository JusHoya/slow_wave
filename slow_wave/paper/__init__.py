"""Paper-figure generation for the Slow Wave bench (Phase 5, WS3).

Regenerates the seven Phase 5 deliverable figures from a committed
:class:`~slow_wave.eval.phase5_schema.Phase5Result` (with its ``analysis``
filled) so the whole figure set reproduces from one command (EC4). See
:mod:`slow_wave.paper.figures` and ``docs/PHASE5_CONTRACT.md``.

This package imports cleanly **without matplotlib** — matplotlib is imported
lazily inside the rendering function bodies (the matplotlib/CI rule), so importing
:data:`slow_wave.paper.FIGURES` never requires the plotting stack.
"""

from __future__ import annotations

from slow_wave.paper.figures import FIGURES, generate_all_figures
from slow_wave.paper.numbers import generate_all_numbers

__all__ = ["FIGURES", "generate_all_figures", "generate_all_numbers"]
