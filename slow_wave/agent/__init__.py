"""No-sleep wake agent for the Slow Wave bench (Phase 2).

The public surface is re-exported here for convenience; see the per-module
docstrings and ``docs/PHASE2_CONTRACT.md`` for details:

* :mod:`slow_wave.agent.budget` — the :class:`~slow_wave.agent.budget.TokenBudgetController`
  the wake loop self-moderates against (FR3.3).
* :mod:`slow_wave.agent.wake` — the :class:`~slow_wave.agent.wake.WakeAgent`
  (single-pass, evaluate-after-each-task no-sleep loop), its telemetry, and result.
* :mod:`slow_wave.agent.runner` — :func:`~slow_wave.agent.runner.run_agent`,
  the one-command end-to-end baseline run that writes ``R[i,j]`` + cost telemetry
  to a manifest. Run it as a module (``python -m slow_wave.agent.runner``); it is
  intentionally *not* re-exported here so ``-m`` execution does not double-import
  the module (mirrors how :mod:`slow_wave.stream.emit` is handled).
"""

from __future__ import annotations

from slow_wave.agent.budget import TokenBudgetController
from slow_wave.agent.wake import WakeAgent, WakeResult, WakeTelemetry

__all__ = [
    "TokenBudgetController",
    "WakeAgent",
    "WakeResult",
    "WakeTelemetry",
]
