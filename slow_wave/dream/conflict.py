"""CONFLICT / unlearning step for the dream engine (Phase 3, FR4.7, EC7).

The optional fifth dream operator is the Crick-Mitchison "reverse learning"
analogue: at the tail of the NREM phase it sweeps the *active* memory for
same-key contradictions (two consolidated assertions of the same
``(subject, attribute)`` with different values) and resolves each by keeping a
single survivor and **demoting** the losers to the auditable archival tier. It
demotes, never destroys (EC7) — a demoted entry is removed from its active store
but stays recoverable via :meth:`MemorySubstrate.archival.recover`.

The step only ever reads confound-free :class:`~slow_wave.memory.schema.MemoryEntry`
graphs (the ``fact`` triple and ``salience``, never a relevance label — FR1.6)
and uses no randomness, so it is a pure deterministic function of
``(substrate state, dream_cfg, now_order)`` (DX1): group iteration, survivor
selection, and the demoted-id order are all stably sorted.
"""

from __future__ import annotations

import logging

from slow_wave.config import DreamConfig
from slow_wave.dream.schema import ConflictResult
from slow_wave.memory.schema import MemoryEntry
from slow_wave.memory.stores import MemorySubstrate

logger = logging.getLogger(__name__)


def resolve_conflicts(
    substrate: MemorySubstrate,
    *,
    dream_cfg: DreamConfig,
    now_order: int,
) -> ConflictResult:
    """Detect & resolve same-key contradictions; demote, don't destroy (FR4.7).

    Gathers every **active** entry (EPISODIC + SEMANTIC) that carries a fact and
    groups them by :meth:`~slow_wave.memory.schema.MemoryEntry.key` (the
    ``(subject, attribute)`` identity). A group is a *conflict* iff it holds
    ``>= 2`` distinct ``fact.value`` s. For each conflicting group exactly one
    survivor is kept and every other member is demoted via
    :meth:`~slow_wave.memory.stores.MemorySubstrate.demote_entry` with
    ``reason="conflict_unlearning"`` — demote-not-delete (EC7): the loser leaves
    its active store but stays recoverable from the archival tier.

    Survivor selection follows ``dream_cfg.conflict_demote_strategy``:

    * ``"older"`` (default): keep the entry with the **largest**
      ``(created_order, entry_id)`` — i.e. the latest asserted value. This
      matches the wake agent's latest-wins exact-key ``answer()``, so resolving
      conflicts never changes a probe answer (only the superseded values are
      retired).
    * ``"lower_salience"``: keep the entry with the **highest**
      ``salience.importance`` (tie-break: larger ``created_order``, then larger
      ``entry_id`` for full determinism).

    Group iteration order, in-group member order, and the resulting
    ``demoted_entry_ids`` order are all stably sorted, so identical
    ``(substrate state, dream_cfg, now_order)`` inputs yield byte-identical
    results (DX1). All active entries are gathered up front, so the per-group
    demotions never mutate a store mid-scan. When no group contradicts, the pass
    is a no-op returning all-zero counters.

    Args:
        substrate: The memory substrate whose active EPISODIC + SEMANTIC entries
            are scanned for same-key contradictions; losers are demoted to its
            archival tier in place.
        dream_cfg: The dream configuration supplying ``conflict_demote_strategy``.
        now_order: The current stream order (the sleep window), recorded as the
            demotion order on each archived loser.

    Returns:
        A :class:`~slow_wave.dream.schema.ConflictResult` recording the number of
        conflicting groups detected, the number of entries demoted, and the ids
        of the demoted entries (the archival audit trail).
    """
    # Gather every active fact-bearing entry up front so per-group demotions
    # (which mutate the active stores) never disturb the scan.
    groups: dict[tuple[str, str], list[MemoryEntry]] = {}
    for store in (substrate.episodic, substrate.semantic):
        for entry in store.all_entries():
            key = entry.key()
            if key is None:  # noise / fact-free entries assert nothing to clash
                continue
            groups.setdefault(key, []).append(entry)

    strategy = dream_cfg.conflict_demote_strategy
    n_conflicts_detected = 0
    demoted_entry_ids: list[str] = []

    # Deterministic group order: sort keys (tuples of strings).
    for key in sorted(groups):
        members = groups[key]
        # A conflict requires >= 2 DISTINCT values under the same key.
        if len({m.fact.value for m in members}) < 2:
            continue

        n_conflicts_detected += 1

        # Stable in-group order so survivor choice and demote order are fixed.
        members_sorted = sorted(members, key=lambda m: (m.created_order, m.entry_id))
        if strategy == "lower_salience":
            survivor = max(
                members_sorted,
                key=lambda m: (m.salience.importance, m.created_order, m.entry_id),
            )
        else:  # "older" (default): latest-wins survivor.
            survivor = max(members_sorted, key=lambda m: (m.created_order, m.entry_id))

        for member in members_sorted:
            if member.entry_id == survivor.entry_id:
                continue
            if substrate.demote_entry(
                member.entry_id,
                reason="conflict_unlearning",
                at_order=now_order,
            ):
                demoted_entry_ids.append(member.entry_id)

    if n_conflicts_detected:
        logger.info(
            "conflict[%s]: resolved %d conflicting key(s), demoted %d entries "
            "to archival at order %d (survivors kept active)",
            strategy,
            n_conflicts_detected,
            len(demoted_entry_ids),
            now_order,
        )

    return ConflictResult(
        n_conflicts_detected=n_conflicts_detected,
        n_demoted=len(demoted_entry_ids),
        demoted_entry_ids=demoted_entry_ids,
    )
