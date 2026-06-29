"""Bias controls for the dream summarizer (Phase 4, FR5.6, EC8).

Two cheap, deterministic controls quantify failure modes that an
accuracy-only evaluation would miss:

* **Temperature-0 stability** (:func:`temperature_zero_stability`) — run the
  summarizer ``n_repeats`` times on the *same* input and measure run-to-run
  variance of the output (distinct texts, pairwise embedding similarity, and the
  coefficient of variation of output token counts). Under a genuine
  temperature-0 model the summary may still drift; this number captures it. Under
  the deterministic mock LLM the repeats are byte-identical, so the control
  reports ``identical=True``, ``distinct_outputs=1``,
  ``mean_pairwise_similarity=1.0`` and ``token_cv=0.0``.

* **Memory drift** (:func:`memory_drift`) — re-summarize a memory's *own* summary
  repeatedly and track each round's fidelity to the **original** source. A
  monotonic decline flags silent corruption: repeated summarization degrading
  rather than distilling memory.

Both controls are deterministic functions of ``(cfg, injected llm/embedder)``
under the mock LLM (DX1): randomness only enters through a real Claude call, and
neither control calls one when no API key is present. The summarizer is the same
"dream summary" completion the TRANSFER operator uses
(:func:`slow_wave.dream.transfer.transfer`): a single
``llm_complete(cfg, prompt)`` call over a rendered observation.
"""

from __future__ import annotations

import logging

import numpy as np

from slow_wave.eval.schema import DriftResult, StabilityResult

logger = logging.getLogger(__name__)

# Default fixed synthetic episode the controls summarize when the caller does not
# supply one. It carries a concrete fact so a real summarizer has something to
# distill; under the mock LLM its exact content only needs to be stable (DX1).
_DEFAULT_EPISODE_TEXT = (
    "On day 12 the field team logged that reservoir B's intake valve was "
    "replaced after a sustained pressure drop, and the maintenance lead "
    "signed off the repair before the evening shift handover."
)

# Fidelity-drop threshold above which memory drift is flagged as silent
# corruption (DriftResult.degraded). Documented default per the Phase 4 contract.
_DRIFT_THRESHOLD = 0.15


def _summary_prompt(text: str) -> str:
    """Render the deterministic "dream summary" prompt for one observation.

    Mirrors the TRANSFER operator's consolidation prompt so the controls
    exercise the same summarization path the bench actually uses. The single
    observation is rendered in a stable layout, so the prompt -- and therefore
    the mock LLM's response -- is byte-identical for identical input (DX1).

    Args:
        text: The observation (or prior-round summary) to consolidate.

    Returns:
        The prompt string passed to ``llm_complete``.
    """
    return (
        "You are consolidating a recent episodic observation into durable, "
        "generalizable semantic memory during a sleep cycle.\n"
        "\n"
        "Observation to consolidate:\n"
        f"  {text}\n"
        "\n"
        "Write one concise paragraph that distills the observation into a "
        "durable, generalizable summary."
    )


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Return the mean pairwise cosine similarity over L2-normalized rows.

    The rows produced by the bench's embedders are already unit-norm, so cosine
    similarity is the row dot product. With fewer than two rows there are no
    pairs, so the result is the trivially-stable ``1.0``.

    Args:
        embeddings: An ``(n, dim)`` array of L2-normalized row vectors.

    Returns:
        The mean of the upper-triangular pairwise cosine similarities, clipped to
        ``[0, 1]`` (the documented stability range).
    """
    mat = np.asarray(embeddings, dtype=np.float64)
    n = mat.shape[0]
    if n < 2:
        return 1.0
    sims = [
        float(np.dot(mat[i], mat[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def temperature_zero_stability(
    cfg,
    *,
    llm_complete=None,
    embedder=None,
    n_repeats: int = 3,
    source_text: str | None = None,
) -> StabilityResult:
    """Quantify run-to-run variance of the dream summarizer (FR5.6, EC8).

    Calls the summarizer ``n_repeats`` times on the **same** input and measures
    how much the output varies: how many distinct texts appear, the mean pairwise
    cosine similarity of their embeddings, and the coefficient of variation of
    the output token counts. Under the deterministic mock LLM every repeat is
    byte-identical, yielding ``identical=True``, ``distinct_outputs=1``,
    ``mean_pairwise_similarity=1.0`` and ``token_cv=0.0``.

    Args:
        cfg: A :class:`slow_wave.config.Config` (sampling params + embedding
            backend).
        llm_complete: Completion fn with the
            ``complete(cfg, prompt, system=None) -> LLMResult`` signature.
            Defaults to :func:`slow_wave.llm.complete` (the deterministic mock
            when no API key is set).
        embedder: An embedder exposing ``encode(list[str]) -> np.ndarray`` with
            L2-normalized rows. Defaults to
            :func:`slow_wave.embeddings.get_embedder` for ``cfg``.
        n_repeats: Number of repeated summarization calls on the same input.
        source_text: The fixed input to summarize; defaults to a synthetic
            episode text.

    Returns:
        The populated :class:`StabilityResult`.
    """
    if llm_complete is None:
        from slow_wave.llm import complete

        llm_complete = complete
    if embedder is None:
        from slow_wave.embeddings import get_embedder

        embedder = get_embedder(cfg)

    text = source_text if source_text is not None else _DEFAULT_EPISODE_TEXT
    prompt = _summary_prompt(text)

    outputs: list[str] = []
    token_counts: list[int] = []
    for _ in range(max(0, n_repeats)):
        result = llm_complete(cfg, prompt)
        outputs.append(result.text)
        token_counts.append(int(result.output_tokens))

    distinct_outputs = len(set(outputs))
    identical = distinct_outputs <= 1

    if identical or len(outputs) < 2:
        # Byte-identical texts have identical embeddings; cosine is exactly 1.0.
        mean_pairwise_similarity = 1.0
    else:
        mean_pairwise_similarity = _mean_pairwise_cosine(embedder.encode(outputs))

    counts = np.asarray(token_counts, dtype=np.float64)
    mean_tokens = float(counts.mean()) if counts.size else 0.0
    token_cv = float(counts.std() / mean_tokens) if mean_tokens > 0.0 else 0.0

    stability = StabilityResult(
        n_repeats=n_repeats,
        distinct_outputs=distinct_outputs,
        identical=identical,
        mean_pairwise_similarity=round(mean_pairwise_similarity, 6),
        token_cv=round(token_cv, 6),
    )
    logger.info(
        "temperature_zero_stability: %d repeat(s) -> %d distinct, identical=%s, "
        "similarity=%.6f, token_cv=%.6f",
        stability.n_repeats,
        stability.distinct_outputs,
        stability.identical,
        stability.mean_pairwise_similarity,
        stability.token_cv,
    )
    return stability


def memory_drift(
    cfg,
    *,
    llm_complete=None,
    embedder=None,
    n_rounds: int = 3,
    source_text: str | None = None,
) -> DriftResult:
    """Detect degradation from repeated re-summarization (FR5.6, EC8).

    Starting from a source observation, each round re-summarizes the *previous*
    round's output, and the round's fidelity is the cosine similarity of its
    output's embedding to the **original** source embedding. A non-increasing
    fidelity sequence (``monotonic_decline``) whose total drop exceeds the drift
    threshold flags silent corruption (``degraded``) -- repeated summarization
    eroding rather than distilling the memory.

    Args:
        cfg: A :class:`slow_wave.config.Config`.
        llm_complete: Completion fn with the
            ``complete(cfg, prompt, system=None) -> LLMResult`` signature.
            Defaults to :func:`slow_wave.llm.complete`.
        embedder: An embedder exposing ``encode(list[str]) -> np.ndarray`` with
            L2-normalized rows. Defaults to
            :func:`slow_wave.embeddings.get_embedder` for ``cfg``.
        n_rounds: Number of re-summarization rounds.
        source_text: The original source observation; defaults to a synthetic
            episode text.

    Returns:
        The populated :class:`DriftResult`.
    """
    if llm_complete is None:
        from slow_wave.llm import complete

        llm_complete = complete
    if embedder is None:
        from slow_wave.embeddings import get_embedder

        embedder = get_embedder(cfg)

    text = source_text if source_text is not None else _DEFAULT_EPISODE_TEXT
    source_emb = np.asarray(embedder.encode([text])[0], dtype=np.float64)

    fidelity_per_round: list[float] = []
    current = text
    for _ in range(max(0, n_rounds)):
        result = llm_complete(cfg, _summary_prompt(current))
        current = result.text
        out_emb = np.asarray(embedder.encode([current])[0], dtype=np.float64)
        fidelity_per_round.append(round(float(np.dot(out_emb, source_emb)), 6))

    if fidelity_per_round:
        faithfulness = fidelity_per_round[-1]
        monotonic_decline = all(
            fidelity_per_round[i] >= fidelity_per_round[i + 1]
            for i in range(len(fidelity_per_round) - 1)
        )
        degraded = (fidelity_per_round[0] - fidelity_per_round[-1]) > _DRIFT_THRESHOLD
    else:
        faithfulness = 0.0
        monotonic_decline = True
        degraded = False

    drift = DriftResult(
        n_rounds=n_rounds,
        fidelity_per_round=fidelity_per_round,
        faithfulness=faithfulness,
        degraded=degraded,
        monotonic_decline=monotonic_decline,
        drift_threshold=_DRIFT_THRESHOLD,
    )
    logger.info(
        "memory_drift: %d round(s) -> fidelities=%s, faithfulness=%.6f, "
        "monotonic=%s, degraded=%s",
        drift.n_rounds,
        drift.fidelity_per_round,
        drift.faithfulness,
        drift.monotonic_decline,
        drift.degraded,
    )
    return drift
