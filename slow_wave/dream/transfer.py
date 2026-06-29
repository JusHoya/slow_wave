"""TRANSFER operator + CLS interleaving for the dream engine (Phase 3, FR4.2, EC4).

TRANSFER is the NREM "distill episodic into semantic" step of a dream cycle: the
episodics sampled this cycle (the REPLAY set, or the whole recent pool when
REPLAY is off) are batched and summarized by a single Claude "dream summary" call
per batch, and each fact-bearing source is consolidated into a durable
:class:`~slow_wave.memory.schema.MemoryEntry` in the SEMANTIC store.

Two design rules carry forward from Phase 2 and are load-bearing here:

* **Preserve the structured fact (latest-wins).** Probe answering is exact-key
  lookup over active memory, so a consolidated entry preserves the source
  episodic's ``(subject, attribute, value)`` triple *verbatim*; the LLM summary
  becomes the entry's natural-language :attr:`~MemoryEntry.content` (and feeds
  generator-fidelity tracking), and the semantic entry's ``created_order`` is set
  to the source's ``created_order`` so latest-wins ordering survives episodic
  eviction.
* **CLS interleaving (EC4).** When enabled, each batch additionally pulls a
  deterministic sample of *prior* consolidated entries from the semantic store
  and mixes them into the summarization prompt as context — the
  complementary-learning-systems antidote to catastrophic interference. Those
  prior memories are never re-written; they are counted in ``n_interleaved_items``.
  Disabling interleaving (``n_interleaved_items == 0``) is the on-purpose
  catastrophic-interference condition.

The operator reads only confound-free entries (never a relevance label, FR1.6),
records everything it bounds — refused protected overwrites and batches skipped
by ``transfer_max_calls`` (DX2) — and is a deterministic function of
``(cfg, sources, rng)`` under the mock LLM (DX1): all randomness flows through the
injected ``rng``.
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.config import Config, DreamConfig
from slow_wave.dream.schema import TransferResult
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate

logger = logging.getLogger(__name__)


def _batched(items: list[MemoryEntry], size: int) -> list[list[MemoryEntry]]:
    """Partition ``items`` into consecutive batches of at most ``size`` entries.

    Args:
        items: The entries to partition (order preserved).
        size: Maximum batch size (``>= 1``).

    Returns:
        A list of batches; the final batch may be shorter than ``size``.
    """
    return [items[start : start + size] for start in range(0, len(items), size)]


def _sample_interleaved(
    prior: list[MemoryEntry], k: int, rng: np.random.Generator
) -> list[MemoryEntry]:
    """Deterministically sample up to ``k`` prior consolidated entries (CLS).

    Args:
        prior: The pool of prior consolidated (semantic) entries to draw context
            from (stable order).
        k: Target number to sample; ``<= 0`` selects nothing.
        rng: The cycle's seeded generator (the only source of randomness, DX1).

    Returns:
        Up to ``min(k, len(prior))`` entries sampled without replacement. The
        ``rng`` is left untouched when there is nothing to sample, so disabled
        interleaving never perturbs the deterministic draw sequence.
    """
    if k <= 0 or not prior:
        return []
    n = min(k, len(prior))
    idx = rng.choice(len(prior), size=n, replace=False)
    return [prior[int(i)] for i in idx]


def _build_summary_prompt(
    batch: list[MemoryEntry], interleaved: list[MemoryEntry]
) -> str:
    """Build the deterministic "dream summary" prompt for one batch.

    The new observations and any interleaved prior memories are rendered in a
    stable, numbered order so the prompt — and therefore the mock LLM's response
    — is byte-identical for identical inputs (DX1).

    Args:
        batch: The fact-bearing source episodics being distilled this batch.
        interleaved: Prior consolidated entries mixed in as context only (CLS);
            empty when interleaving is off.

    Returns:
        The prompt string passed to ``llm_complete``.
    """
    lines = [
        "You are consolidating recent episodic observations into durable, "
        "generalizable semantic memory during a sleep cycle.",
        "",
        "New observations to consolidate:",
    ]
    for i, entry in enumerate(batch, start=1):
        lines.append(f"  {i}. {entry.content}")
    if interleaved:
        lines.append("")
        lines.append("Prior consolidated knowledge (context only — do not rewrite):")
        for i, entry in enumerate(interleaved, start=1):
            lines.append(f"  {i}. {entry.content}")
    lines.append("")
    lines.append(
        "Write one concise paragraph that distills the new observations, "
        "integrating them with the prior knowledge without contradicting it."
    )
    return "\n".join(lines)


def transfer(
    substrate: MemorySubstrate,
    sources: list[MemoryEntry],
    *,
    cfg: Config,
    dream_cfg: DreamConfig,
    embedder,
    llm_complete,
    rng: np.random.Generator,
    now_order: int,
) -> TransferResult:
    """Distill sampled episodics into durable SEMANTIC entries (FR4.2, EC4).

    The four-step WS-TRANSFER semantics:

    1. **Filter + batch.** Keep only ``sources`` that carry a fact (noise has
       ``fact=None`` and is never consolidated), then partition them into batches
       of ``dream_cfg.transfer_batch_size``.
    2. **CLS interleaving (EC4).** When ``dream_cfg.cls_interleave`` is ``True``,
       each processed batch also pulls up to
       ``round(dream_cfg.cls_interleave_ratio * dream_cfg.transfer_batch_size)``
       prior consolidated entries from ``substrate.semantic`` (sampled
       deterministically via ``rng`` from a snapshot taken before any write this
       cycle) and mixes them into the summarization prompt as **context only** —
       they are not re-written and are counted in ``n_interleaved_items``. When
       ``False`` no prior memory is mixed (``n_interleaved_items == 0`` — the
       catastrophic-interference condition). On the first-ever cycle the semantic
       store is empty, so ``n_interleaved_items`` is ``0`` even when on.
    3. **Summarize.** One ``llm_complete(cfg, prompt)`` "dream summary" call per
       batch, accumulating ``api_calls`` / ``input_tokens`` / ``output_tokens``.
       The per-cycle ceiling ``dream_cfg.transfer_max_calls`` (``None`` =>
       unbounded) is honored: once it is reached the remaining batches are skipped
       (their sources are not written), counted in ``n_skipped_calls``, and logged
       (DX2).
    4. **Consolidate.** For each fact-bearing source in a processed batch, upsert
       a semantic ``MemoryEntry`` whose ``entry_id`` is ``f"s{created_order:06d}"``,
       whose ``content`` is the batch's dream summary, and which **preserves the
       source fact** and ``created_order`` (latest-wins) with a provenance pointer
       back to the source episodic (EC3/EC7). The write goes through
       ``substrate.semantic.upsert(..., failure_sink=substrate.failure_events)``,
       so an attempt to clobber a *protected* same-key fact is refused (FR2.5),
       counted in ``n_refused``, and the protected value is preserved.

    ``n_consolidated`` counts every fact-bearing source distilled in a processed
    batch (one upsert attempt each), so
    ``n_consolidated == n_semantic_written + n_refused``.

    Args:
        substrate: The memory substrate whose SEMANTIC store is written and whose
            ``failure_events`` sink receives protection refusals.
        sources: The candidate source episodics to consolidate (the REPLAY set, or
            the recent pool when REPLAY is off); non-fact sources are ignored.
        cfg: The top-level config passed through to ``llm_complete``.
        dream_cfg: The dream configuration (batch size, interleaving toggle/ratio,
            call ceiling).
        embedder: The embedder duck-type; ``encode([str]) -> (n, dim)`` float32,
            L2-normalized rows.
        llm_complete: The injected completion fn with the
            ``complete(cfg, prompt, system=None) -> LLMResult`` signature.
        rng: The cycle's seeded generator — the sole source of randomness (DX1).
        now_order: The current stream order (the sleep window); stamped onto each
            new semantic entry's ``recency_order``.

    Returns:
        A :class:`~slow_wave.dream.schema.TransferResult` recording the batch
        count, consolidation/write/refusal counts, the interleaving flag and
        item count, the written semantic entry ids, the accumulated call/token
        totals, and any skipped batches.
    """
    fact_sources = [s for s in sources if s.fact is not None]
    n_noise = len(sources) - len(fact_sources)
    if n_noise:
        logger.info(
            "transfer: ignoring %d non-fact source(s) at order %d (not consolidated)",
            n_noise,
            now_order,
        )

    result = TransferResult(interleaved=dream_cfg.cls_interleave)

    if not fact_sources:
        logger.info(
            "transfer: no fact-bearing sources to consolidate at order %d", now_order
        )
        return result

    # CLS context is drawn from the entries that were already consolidated
    # *before* this cycle began (a snapshot), so this cycle's own writes never
    # masquerade as "prior" knowledge.
    if dream_cfg.cls_interleave:
        prior_semantic = list(substrate.semantic.all_entries())
        interleave_k = round(
            dream_cfg.cls_interleave_ratio * dream_cfg.transfer_batch_size
        )
    else:
        prior_semantic = []
        interleave_k = 0

    batches = _batched(fact_sources, dream_cfg.transfer_batch_size)
    result.n_batches = len(batches)

    for batch in batches:
        # DX2: honor the per-cycle call ceiling — skip + record + log, never drop
        # silently. Once the cap is hit every later batch is skipped too.
        if (
            dream_cfg.transfer_max_calls is not None
            and result.api_calls >= dream_cfg.transfer_max_calls
        ):
            result.n_skipped_calls += 1
            logger.warning(
                "transfer: skipping batch of %d source(s) at order %d "
                "(transfer_max_calls=%d reached)",
                len(batch),
                now_order,
                dream_cfg.transfer_max_calls,
            )
            continue

        interleaved = _sample_interleaved(prior_semantic, interleave_k, rng)
        result.n_interleaved_items += len(interleaved)

        prompt = _build_summary_prompt(batch, interleaved)
        llm_result = llm_complete(cfg, prompt)
        result.api_calls += 1
        result.input_tokens += llm_result.input_tokens
        result.output_tokens += llm_result.output_tokens
        summary = llm_result.text

        # All entries distilled in this batch share the one dream summary as their
        # content (and therefore one embedding); each preserves its own fact.
        embedding = embedder.encode([summary])[0]

        for src in batch:
            result.n_consolidated += 1
            entry = MemoryEntry(
                entry_id=f"s{src.created_order:06d}",
                tier=MemoryTier.SEMANTIC,
                content=summary,
                fact=src.fact,  # PRESERVE the structured fact verbatim
                created_order=src.created_order,  # latest-wins ordering
                salience=SalienceMeta(
                    importance=src.salience.importance,
                    recency_order=now_order,
                    novelty=src.salience.novelty,
                ),
                provenance=(src.entry_id,),  # EC3/EC7 trace
            )
            applied = substrate.semantic.upsert(
                entry,
                embedding,
                now_order,
                failure_sink=substrate.failure_events,
            )
            if applied:
                result.n_semantic_written += 1
                result.written_entry_ids.append(entry.entry_id)
            else:
                result.n_refused += 1

    logger.info(
        "transfer: %d batch(es) -> %d consolidated (%d written, %d refused), "
        "%d interleaved item(s) [%s], %d call(s), %d skipped at order %d",
        result.n_batches,
        result.n_consolidated,
        result.n_semantic_written,
        result.n_refused,
        result.n_interleaved_items,
        "on" if result.interleaved else "off",
        result.api_calls,
        result.n_skipped_calls,
        now_order,
    )

    return result
