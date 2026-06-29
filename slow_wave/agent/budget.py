"""Token-budget controller for the no-sleep wake agent (Phase 2, WS-AGENT, FR3.3).

The wake agent self-moderates its (optional) per-task Claude reasoning calls
against a total token ceiling. :class:`TokenBudgetController` is the small,
stateful accountant that tracks how many input/output tokens have been spent,
answers whether a prospective call can still be afforded, and counts the calls
that were *skipped* because they would have blown the budget.

Skipping is never silent (DX2): :meth:`TokenBudgetController.skip` logs an INFO
line so a budget-truncated run is auditable from the logs alone. A ``None``
budget means "unbounded" — every call is afforded and nothing is ever skipped.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TokenBudgetController:
    """Track and enforce a total (input+output) LLM token budget (FR3.3, DX2).

    The controller is a pure accountant: it never makes LLM calls itself, it only
    records what was spent and answers whether the next call fits. A ``None``
    ``max_tokens`` disables the ceiling (everything is affordable, nothing is
    skipped); a finite ceiling causes the wake loop to skip — and log — any
    reasoning call whose estimate would exceed the remaining budget.

    Attributes:
        max_tokens: The total token ceiling, or ``None`` for unbounded.
        spent_input: Cumulative input tokens recorded so far.
        spent_output: Cumulative output tokens recorded so far.
        n_skipped: Number of calls skipped because the budget could not afford
            them.
    """

    def __init__(self, max_tokens: int | None) -> None:
        """Initialize the controller.

        Args:
            max_tokens: The total (input+output) token ceiling for the whole run,
                or ``None`` for an unbounded budget.
        """
        self.max_tokens: int | None = max_tokens
        self.spent_input: int = 0
        self.spent_output: int = 0
        self.n_skipped: int = 0

    @property
    def total_spent(self) -> int:
        """Return the total tokens spent so far (``spent_input + spent_output``)."""
        return self.spent_input + self.spent_output

    @property
    def remaining(self) -> int | None:
        """Return tokens left before the ceiling, or ``None`` if unbounded.

        Returns:
            ``max_tokens - total_spent`` when a finite budget is set (may go
            negative if a single recorded call overshot the estimate), else
            ``None``.
        """
        if self.max_tokens is None:
            return None
        return self.max_tokens - self.total_spent

    @property
    def exhausted(self) -> bool:
        """Return whether a finite budget has been fully spent.

        Returns:
            ``True`` only when ``max_tokens`` is set and ``total_spent`` has
            reached or exceeded it; always ``False`` for an unbounded budget.
        """
        return self.max_tokens is not None and self.total_spent >= self.max_tokens

    def can_afford(self, est_tokens: int) -> bool:
        """Return whether a call estimated at ``est_tokens`` still fits.

        Args:
            est_tokens: The estimated input+output tokens of a prospective call.

        Returns:
            ``True`` if the budget is unbounded or ``total_spent + est_tokens``
            stays within ``max_tokens``; ``False`` otherwise.
        """
        if self.max_tokens is None:
            return True
        return self.total_spent + est_tokens <= self.max_tokens

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record the actual token cost of a completed call.

        Args:
            input_tokens: Prompt tokens consumed by the call.
            output_tokens: Completion tokens produced by the call.
        """
        self.spent_input += input_tokens
        self.spent_output += output_tokens

    def skip(self) -> None:
        """Record (and log) a call skipped because the budget could not afford it.

        Increments :attr:`n_skipped` and emits an INFO line so a budget-truncated
        run is never silent (DX2).
        """
        self.n_skipped += 1
        logger.info(
            "token budget exhausted: skipping reasoning call "
            "(spent=%d, max=%s, skipped=%d)",
            self.total_spent,
            self.max_tokens,
            self.n_skipped,
        )
