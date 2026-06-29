"""GENERATIVE-AUGMENT operator for the dream engine (Phase 3, FR4.4, EC5).

GENERATIVE-AUGMENT is the REM-like phase of a dream cycle: a small batch of
recent episodics is *re-imagined* into free-text **pseudo-episodes**
(paraphrases, abstractions, counterfactuals) that are written back into the
EPISODIC tier to broaden coverage the way REM sleep is hypothesized to recombine
the day's experience. Unlike TRANSFER, the generator never asserts a structured
``fact``: a pseudo-episode is a generalization, so it carries ``fact=None`` and
can never overwrite a probed key's exact-lookup answer (no R-corruption).

Because repeated generative summarization can *degrade* rather than distill
memory, every pass logs a :class:`~slow_wave.dream.schema.FidelityScore`: the
cosine similarity between each pseudo-episode's embedding and its source
episodic's embedding (fidelity in ``[0, 1]``; drift is ``1 - fidelity``). This is
the FR5.6 generator-drift seed the bench uses to flag a collapsing generator.

The operator draws all randomness from the injected ``rng`` and, under the mock
LLM, is byte-identical run-to-run (DX1). It reads only confound-free
:class:`~slow_wave.memory.schema.MemoryEntry` content/salience — never a relevance
label (FR1.6) — and the pseudo-episodes it writes carry no banned field.
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.config import Config, DreamConfig
from slow_wave.dream.schema import AugmentResult, FidelityScore
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.memory.stores import MemorySubstrate

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity of two vectors, guarding against zero norm.

    The bench's embeddings are already L2-normalized (so a dot product would
    suffice), but the full ``dot / (||a|| ||b||)`` form is used with an explicit
    zero-norm guard so a degenerate (all-zero) embedding — e.g. a pseudo text
    with no alphanumeric tokens — yields ``0.0`` rather than ``nan``.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cosine similarity as a float, or ``0.0`` if either vector has zero
        norm.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _source_vector(
    substrate: MemorySubstrate, src: MemoryEntry, embedder
) -> np.ndarray:
    """Return the source episodic's stored embedding, re-encoding if missing.

    Looks up ``src``'s vector in the store that currently owns it (selected by
    ``src.tier`` — a source evicted/demoted mid-pass keeps its preserved vector
    in the archival tier). Falls back to ``embedder.encode([src.content])[0]``
    when no stored vector is found (e.g. archival disabled), which is identical
    to the originally stored vector for the deterministic hash embedder, so
    fidelity stays reproducible either way.

    Args:
        substrate: The memory substrate holding the source.
        src: The source episodic the pseudo-episode was synthesized from.
        embedder: The embedder duck-type (``.encode(list[str]) -> (n, dim)``).

    Returns:
        The source's ``(dim,)`` L2-normalized embedding.
    """
    store = {
        MemoryTier.EPISODIC: substrate.episodic,
        MemoryTier.SEMANTIC: substrate.semantic,
        MemoryTier.ARCHIVAL: substrate.archival,
    }.get(src.tier)
    vec = store.vector(src.entry_id) if store is not None else None
    if vec is None:
        vec = embedder.encode([src.content])[0]
    return np.asarray(vec)


def augment(
    substrate: MemorySubstrate,
    sources: list[MemoryEntry],
    *,
    cfg: Config,
    dream_cfg: DreamConfig,
    embedder,
    llm_complete,
    rng: np.random.Generator,
    now_order: int,
) -> AugmentResult:
    """Synthesize pseudo-episodes from sampled episodics (FR4.4, REM-like).

    Up to ``dream_cfg.augment_per_cycle`` sources are selected from ``sources``
    deterministically via ``rng``, with **fact-bearing** sources preferred over
    fact-free ones (so the generator re-imagines signal before noise). For each
    selected source ``i`` (kind ``dream_cfg.augment_kinds[i % len(kinds)]``):

    * ``llm_complete(cfg, f"{kind} of: {src.content}")`` produces the pseudo text
      (its ``api_calls``/token usage is accumulated);
    * a pseudo-episode is written to the EPISODIC tier via
      :meth:`~slow_wave.memory.stores.MemorySubstrate.observe` with
      ``fact=None`` (never corrupt an exact-key answer), ``created_order ==
      now_order``, importance inherited from the source, and a ``provenance``
      pointer back to ``src.entry_id`` (EC7 trace; ``observe`` demotes any
      eviction to archival, so the audit stays intact);
    * a fidelity score ``max(0.0, cosine(pseudo_emb, source_emb))`` is recorded,
      where ``source_emb`` is the source's stored vector (re-encoded if missing).

    A :class:`~slow_wave.dream.schema.FidelityScore` is logged per cycle
    (``n_pseudo``, ``mean_fidelity``, ``min_fidelity``, ``mean_drift =
    1 - mean_fidelity``; all ``0.0`` when nothing is synthesized). An empty
    ``sources`` list or ``augment_per_cycle == 0`` returns an empty result
    without raising. Given fixed ``(cfg, sources, rng)`` under the mock LLM the
    result is byte-identical run-to-run (DX1).

    Args:
        substrate: The memory substrate the pseudo-episodes are written into.
        sources: The candidate episodics to re-imagine (the transfer source /
            candidate pool selected by the engine).
        cfg: The top-level config passed through to ``llm_complete``.
        dream_cfg: The dream configuration supplying ``augment_per_cycle`` and
            ``augment_kinds``.
        embedder: The embedder duck-type (``.encode(list[str]) -> (n, dim)``
            float32, L2-normalized rows).
        llm_complete: The injected completion function
            (``complete(cfg, prompt) -> LLMResult``); the deterministic mock when
            no API key is set.
        rng: The cycle's :class:`numpy.random.Generator` (all randomness flows
            through it; DX1).
        now_order: The current stream order (the sleep window); the pseudo ids,
            ``created_order``, and recency stamps are taken from it.

    Returns:
        An :class:`~slow_wave.dream.schema.AugmentResult` recording the number of
        pseudo-episodes, their ids, the per-cycle fidelity/drift score, and the
        accumulated Claude call/token counters.
    """
    n_sources = len(sources)
    n_take = min(dream_cfg.augment_per_cycle, n_sources)
    kinds = dream_cfg.augment_kinds

    # DX2: anything that bounds output to nothing is logged, never silently swallowed.
    if n_take <= 0 or not kinds:
        if n_take > 0 and not kinds:
            logger.warning(
                "augment: augment_kinds is empty; no pseudo-episodes synthesized "
                "at order %d",
                now_order,
            )
        else:
            logger.info(
                "augment: no pseudo-episodes at order %d "
                "(augment_per_cycle=%d, n_sources=%d)",
                now_order,
                dream_cfg.augment_per_cycle,
                n_sources,
            )
        return AugmentResult(fidelity=FidelityScore())

    # Deterministic, fact-bearing-first source selection: shuffle each group via
    # rng, keep fact-bearing sources ahead of fact-free ones, then take n_take.
    fact_indices = [i for i, s in enumerate(sources) if s.fact is not None]
    other_indices = [i for i, s in enumerate(sources) if s.fact is None]
    fact_order = [fact_indices[j] for j in rng.permutation(len(fact_indices))]
    other_order = [other_indices[j] for j in rng.permutation(len(other_indices))]
    selection = (fact_order + other_order)[:n_take]
    chosen = [sources[idx] for idx in selection]

    n_kinds = len(kinds)
    pseudo_entry_ids: list[str] = []
    fidelities: list[float] = []
    api_calls = 0
    input_tokens = 0
    output_tokens = 0

    for i, src in enumerate(chosen):
        kind = kinds[i % n_kinds]
        result = llm_complete(cfg, f"{kind} of: {src.content}")
        api_calls += 1
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        pseudo_text = result.text

        emb = embedder.encode([pseudo_text])[0]
        entry_id = f"g{now_order:06d}_{i}"
        entry = MemoryEntry(
            entry_id=entry_id,
            tier=MemoryTier.EPISODIC,
            content=pseudo_text,
            fact=None,  # never corrupt exact-key answers (no R-corruption)
            created_order=now_order,
            salience=SalienceMeta(
                importance=src.salience.importance,
                recency_order=now_order,
            ),
            provenance=(src.entry_id,),  # EC7 trace back to the source episodic
        )
        substrate.observe(entry, emb, now_order)
        pseudo_entry_ids.append(entry_id)

        source_emb = _source_vector(substrate, src, embedder)
        fidelities.append(max(0.0, _cosine(emb, source_emb)))

    n_pseudo = len(pseudo_entry_ids)
    if n_pseudo > 0:
        mean_fidelity = float(np.mean(fidelities))
        min_fidelity = float(np.min(fidelities))
        mean_drift = 1.0 - mean_fidelity
    else:  # pragma: no cover - guarded by the n_take<=0 early return above
        mean_fidelity = min_fidelity = mean_drift = 0.0

    fidelity = FidelityScore(
        n_pseudo=n_pseudo,
        mean_fidelity=mean_fidelity,
        min_fidelity=min_fidelity,
        mean_drift=mean_drift,
    )

    logger.info(
        "augment: synthesized %d/%d pseudo-episodes at order %d "
        "(mean_fidelity=%.6f, min_fidelity=%.6f, mean_drift=%.6f)",
        n_pseudo,
        n_sources,
        now_order,
        mean_fidelity,
        min_fidelity,
        mean_drift,
    )

    return AugmentResult(
        n_pseudo=n_pseudo,
        pseudo_entry_ids=pseudo_entry_ids,
        fidelity=fidelity,
        api_calls=api_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
