"""Phase 5 sweep runner: the arm x regime x seed grid + length & sim/real sweeps.

This is WS1 of Phase 5 (PRD §8, Phase 5; see ``docs/PHASE5_CONTRACT.md``). It
layers three *across-cell* sweeps on top of the Phase 4 nine-arm harness
(:func:`slow_wave.eval.harness.build_experiment`) and serializes them into one
:class:`~slow_wave.eval.phase5_schema.Phase5Result`:

* :func:`run_regime_sweep` — the **arm x distractor-regime x seed** grid (EC1).
  Each regime is one full nine-arm experiment on a stream whose ``label_mix`` is
  varied; every cell writes a per-cell manifest (cost + git hash) and rolls its
  summary arrays + the regime's preregistered primary endpoint into a
  :class:`~slow_wave.eval.phase5_schema.RegimeCell`.
* :func:`run_length_sweep` — the **stream-length** sweep feeding the long-context
  crossover (EC6 data). For each length ``L`` and seed, one stream of
  ``n_tasks=L`` is generated and each key arm is run via
  :func:`slow_wave.eval.harness.run_arm`.
* :func:`run_sim_vs_real` — the **accelerated-sim vs real-long-horizon**
  comparison (EC3): a short, high-compression "sim" stream vs a long,
  low-compression "real" stream per key arm, with numpy-only agreement scalars.

:func:`run_phase5` runs all three and writes the committed artifact. This module
does **no** verdict inference (that is WS2,
:mod:`slow_wave.eval.analysis`); it only produces the raw sweep data.

Design principles (inherited from Phase 0-4 — non-negotiable)
-------------------------------------------------------------
* **Determinism is sacred (DX1).** All randomness flows through the harness /
  ``run_arm``, which seed every RNG via ``numpy.random.default_rng(derive_seed(
  seed, "<name>"))``; the stream seed for every run is ``derive_seed(seed,
  "stream")`` (identical to the harness so pairing is consistent). The global
  RNGs are reset via :func:`~slow_wave.repro.seeding.set_global_seeds` at the top
  of each sweep so a sweep is reproducible regardless of what ran before. The
  assembled :class:`Phase5Result` carries **no** wall-clock or timestamp, so two
  :func:`run_phase5` calls on the same config produce a byte-identical
  ``phase5_result.json``.
* **Confound guard (FR1.6).** Relevance labels are read **only** transitively via
  the already-computed :class:`~slow_wave.eval.schema.ArmResult.prune_quality`
  (precision/recall/F1/signal_retention); this module never touches
  ``offline_labels`` / ``ground_truth`` / ``probe.answer`` directly.
* **No silent caps (DX2).** Every sweep logs its coverage (regimes / lengths /
  cells run, 0 dropped) into both :data:`logger` and the result's
  ``coverage_notes``.
* **No base mutation.** Configs are varied on deep copies via
  ``cfg.model_copy(deep=True)``; the base config is never touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from slow_wave.config import Config, load_config
from slow_wave.embeddings import get_embedder
from slow_wave.eval.harness import build_experiment, run_arm
from slow_wave.eval.phase5_schema import (
    GridResult,
    LengthPoint,
    LengthSweep,
    Phase5Result,
    RegimeCell,
    RegimeMix,
    RetentionCurve,
    SimRealArm,
    SimRealResult,
)
from slow_wave.eval.schema import ArmResult, ExperimentResult
from slow_wave.llm import complete
from slow_wave.repro.gitinfo import git_info
from slow_wave.repro.manifest import new_manifest, read_manifest, write_manifest
from slow_wave.repro.seeding import derive_seed, set_global_seeds
from slow_wave.stream.generator import generate_stream
from slow_wave.stream.probes import build_probe_set
from slow_wave.stream.schema import LabelMix

logger = logging.getLogger(__name__)

#: Default key arms for the long-context crossover length sweep (EC6): the
#: consolidation treatment, the unbounded-memory ceiling, and the forgetting
#: baseline.
_DEFAULT_LENGTH_ARMS: list[str] = ["full_dream", "long_context", "no_sleep"]

#: Default key arms for the sim-vs-real agreement check (EC3).
_DEFAULT_SIMREAL_ARMS: list[str] = ["full_dream", "no_sleep", "long_context"]


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _stream_seed(seed: int) -> int:
    """Return the stream-generation seed for a master ``seed`` (mirrors harness).

    The length and sim/real sweeps call :func:`slow_wave.eval.harness.run_arm`
    directly, so they must derive each seed's stream exactly as
    :func:`slow_wave.eval.harness.build_experiment` does — via
    ``derive_seed(seed, "stream")`` — so the per-seed pairing is consistent.

    Args:
        seed: The master run seed.

    Returns:
        The derived stream seed.
    """
    return derive_seed(seed, "stream")


def _by_arm_seed(results: list[ArmResult]) -> dict[tuple[str, int], ArmResult]:
    """Index a flat list of :class:`ArmResult` by ``(arm_name, seed)``."""
    return {(r.arm_name, r.seed): r for r in results}


def _prune_attr(result: ArmResult, attr: str) -> float:
    """Return one prune-quality scalar via the sanctioned transitive label read.

    Reads ``precision`` / ``recall`` / ``f1`` / ``signal_retention`` **only** off
    the already-computed :class:`~slow_wave.eval.schema.ArmResult.prune_quality`
    (FR1.6: never ``offline_labels`` / ``ground_truth`` directly); returns ``0.0``
    when the arm pruned nothing and has no prune quality.

    Args:
        result: The arm result to read.
        attr: The :class:`~slow_wave.eval.schema.PruneQuality` field name.

    Returns:
        The metric as a float (``0.0`` if ``prune_quality`` is ``None``).
    """
    pq = result.prune_quality
    return float(getattr(pq, attr)) if pq is not None else 0.0


def _final_row(result: ArmResult) -> list[float]:
    """Return the final row ``R[T-1][:]`` of an arm's accuracy matrix.

    The final row is accuracy on each task *j* after the whole stream — the
    retention curve the figures layer draws as a seed band.

    Args:
        result: The arm result whose accuracy matrix to read.

    Returns:
        The last row of ``R`` as a list of floats (empty for an empty matrix).
    """
    rows = result.accuracy_matrix.R
    return [float(v) for v in rows[-1]] if rows else []


def _retention_curve(
    arm_name: str, results_in_seed_order: list[ArmResult], seeds: list[int]
) -> RetentionCurve:
    """Assemble a per-seed :class:`RetentionCurve` from aligned arm results.

    Args:
        arm_name: The arm the curve belongs to.
        results_in_seed_order: The arm's results, one per seed, in ``seeds`` order.
        seeds: The seeds aligned with ``results_in_seed_order``.

    Returns:
        A :class:`RetentionCurve` whose ``final_row_per_seed`` holds each seed's
        ``R[T-1][:]``.
    """
    n_tasks = (
        results_in_seed_order[0].accuracy_matrix.n_tasks
        if results_in_seed_order
        else 0
    )
    return RetentionCurve(
        arm_name=arm_name,
        n_tasks=n_tasks,
        final_row_per_seed=[_final_row(r) for r in results_in_seed_order],
        seeds=list(seeds),
    )


def _write_cell_manifest(
    cfg: Config, embedder, experiment: ExperimentResult, manifest_path: Path, wall: float
) -> Path:
    """Write one regime cell's :class:`ExperimentResult` into a run manifest.

    Mirrors :func:`slow_wave.eval.harness._write_manifest` (same deterministic
    probe + summed-cost LLM stand-in) but to an explicit per-cell path so EC1 is
    satisfied: every grid run has a manifest with **cost + git hash**.

    Args:
        cfg: The (regime-varied) effective config the cell ran under.
        embedder: The shared embedder (for embedding provenance).
        experiment: The cell's assembled nine-arm experiment result.
        manifest_path: The exact destination path for the manifest JSON.
        wall: Measured wall-clock duration of the cell, in seconds.

    Returns:
        The path the manifest was written to.
    """
    from slow_wave.eval.harness import _aggregate_llm

    llm, _ = _aggregate_llm(cfg, experiment)
    total_api = sum(r.cost.api_calls for r in experiment.arm_results)
    pe = experiment.primary_endpoint
    deterministic_probe = {
        "scenario": experiment.scenario,
        "arms_run": experiment.arms_run,
        "seeds": experiment.seeds,
        "acc_by_arm": {
            a: [
                round(r.continual_metrics.acc, 6)
                for r in experiment.arm_results
                if r.arm_name == a
            ]
            for a in experiment.arms_run
        },
        "prune_f1_by_arm": {
            a: [
                round(r.prune_quality.f1, 6)
                for r in experiment.arm_results
                if r.arm_name == a and r.prune_quality is not None
            ]
            for a in experiment.arms_run
        },
        "budget_matched": experiment.budget_report.matched,
        "aa_abs_difference": round(experiment.aa.abs_difference, 6),
        "aa_significant": experiment.aa.significant,
        "primary_endpoint": None
        if pe is None
        else {"name": pe.name, "value": round(pe.value, 6), "verdict": pe.verdict},
    }
    manifest = new_manifest(
        cfg=cfg,
        embedder=embedder,
        llm=llm,
        seeds={"master": cfg.seed, **{f"seed_{s}": s for s in experiment.seeds}},
        deterministic_probe=deterministic_probe,
        wall_clock_s=wall,
        git=git_info(),
        api_calls=total_api,
        results={"experiment": experiment.model_dump(mode="json")},
    )
    return write_manifest(manifest, manifest_path)


# --------------------------------------------------------------------------- #
# WS1 public API
# --------------------------------------------------------------------------- #
def run_regime_sweep(cfg: Config, out_dir: str | Path) -> GridResult:
    """Run the arm x distractor-regime x seed grid (EC1).

    For each regime declared in ``cfg.hyperparameters["phase5"]["regimes"]`` (>= 3),
    a deep copy of ``cfg`` is taken, its ``cfg.stream.label_mix`` is set to the
    regime's mix, and :func:`slow_wave.eval.harness.build_experiment` runs the
    full nine-arm x seed experiment on that stream. The cell's
    :class:`~slow_wave.eval.schema.ExperimentResult` is written to
    ``<out>/phase5/regime_<name>/manifest.json`` (a manifest with cost + git hash),
    and its summary arrays are rolled into a
    :class:`~slow_wave.eval.phase5_schema.RegimeCell`: per-arm per-seed ACC, prune
    precision/recall/F1, signal_retention, total tokens and memory vectors; the
    **full** preregistered primary-endpoint object copied from the cell plus the
    mirrored scalar value/verdict/CI fields; the A/A noise floor; and the
    matched-budget verdict.

    The registered ``distractor_heavy`` regime (``cfg.hyperparameters["phase5"]
    ["primary_regime"]``) is the headline ``primary_regime``; its label mix equals
    the base ``cfg.stream.label_mix`` so this grid's primary column reproduces the
    ``eval_full`` result.

    Args:
        cfg: The base experiment config; ``cfg.stream`` must be set and
            ``cfg.hyperparameters["phase5"]`` must carry the regimes + primary
            regime.
        out_dir: Output root; per-cell manifests go under ``<out>/phase5``.

    Returns:
        The assembled :class:`~slow_wave.eval.phase5_schema.GridResult`.

    Raises:
        KeyError: If ``cfg.hyperparameters["phase5"]`` (or its ``regimes`` /
            ``primary_regime`` keys) is missing.
        ValueError: If ``cfg.stream`` is ``None`` (raised by the harness).
    """
    phase5 = cfg.hyperparameters["phase5"]
    regimes = phase5["regimes"]
    primary_regime = phase5["primary_regime"]
    out = Path(out_dir)

    embedder = get_embedder(cfg)
    arm_names = list(cfg.eval.arms)
    seeds = list(cfg.eval.seeds)

    cells: list[RegimeCell] = []
    for regime in regimes:
        mix = RegimeMix(
            name=regime["name"],
            signal=regime["signal"],
            distractor=regime["distractor"],
            noise=regime["noise"],
        )
        cell_cfg = cfg.model_copy(deep=True)
        cell_cfg.stream.label_mix = LabelMix(
            signal=mix.signal, distractor=mix.distractor, noise=mix.noise
        )

        start = time.perf_counter()
        experiment = build_experiment(cell_cfg)
        wall = time.perf_counter() - start

        manifest_path = out / "phase5" / f"regime_{mix.name}" / "manifest.json"
        written = _write_cell_manifest(
            cell_cfg, embedder, experiment, manifest_path, wall
        )

        by = _by_arm_seed(experiment.arm_results)
        ordered = {a: [by[(a, s)] for s in seeds] for a in arm_names}

        pe = experiment.primary_endpoint
        cell = RegimeCell(
            regime=mix,
            manifest_path=written.as_posix(),
            arms=list(arm_names),
            seeds=list(seeds),
            acc_by_arm={
                a: [float(r.continual_metrics.acc) for r in ordered[a]]
                for a in arm_names
            },
            prune_precision_by_arm={
                a: [_prune_attr(r, "precision") for r in ordered[a]] for a in arm_names
            },
            prune_recall_by_arm={
                a: [_prune_attr(r, "recall") for r in ordered[a]] for a in arm_names
            },
            prune_f1_by_arm={
                a: [_prune_attr(r, "f1") for r in ordered[a]] for a in arm_names
            },
            signal_retention_by_arm={
                a: [_prune_attr(r, "signal_retention") for r in ordered[a]]
                for a in arm_names
            },
            total_tokens_by_arm={
                a: [float(r.cost.total_tokens) for r in ordered[a]] for a in arm_names
            },
            memory_vectors_by_arm={
                a: [float(r.cost.memory_vectors) for r in ordered[a]]
                for a in arm_names
            },
            primary_endpoint=pe.model_copy(deep=True) if pe is not None else None,
            primary_value=(pe.value if pe is not None else None),
            primary_verdict=(pe.verdict if pe is not None else None),
            primary_ci_lo=(pe.difference_ci.lo if pe is not None else None),
            primary_ci_hi=(pe.difference_ci.hi if pe is not None else None),
            aa_abs_difference=experiment.aa.abs_difference,
            aa_significant=experiment.aa.significant,
            budget_matched=experiment.budget_report.matched,
        )
        cells.append(cell)

    n_cells = len(cells) * len(arm_names) * len(seeds)
    coverage = [
        f"regime sweep: ran {len(cells)} regimes x {len(arm_names)} arms x "
        f"{len(seeds)} seeds = {n_cells} cells; 0 dropped",
        f"regimes={[c.regime.name for c in cells]}; primary_regime={primary_regime}",
    ]
    for note in coverage:
        logger.info(note)

    return GridResult(
        regimes=cells,
        arms=list(arm_names),
        seeds=list(seeds),
        primary_regime=primary_regime,
        coverage_notes=coverage,
    )


def run_length_sweep(
    cfg: Config,
    out_dir: str | Path,
    *,
    lengths: list[int],
    key_arms: list[str] | None = None,
) -> LengthSweep:
    """Stream-length sweep feeding the long-context crossover (EC6 data).

    For each ``L`` in ``lengths`` (sorted ascending) and each seed in
    ``cfg.eval.seeds``, one stream of ``n_tasks=L`` is generated from a deep copy
    of ``cfg.stream`` (``items_per_task`` held fixed) with the harness stream seed
    ``derive_seed(seed, "stream")``, its probe set is built, and each ``key_arm``
    is run via :func:`slow_wave.eval.harness.run_arm`. Per-(arm, seed) ACC, total
    tokens and final memory-vector count are recorded into a
    :class:`~slow_wave.eval.phase5_schema.LengthPoint`. As ``L`` grows the
    unbounded ``long_context`` arm's memory is non-decreasing (it never evicts)
    while ``full_dream`` stays bounded — the data the crossover verdict reads.

    Args:
        cfg: The base experiment config (``cfg.stream`` must be set).
        out_dir: Output root (unused for I/O here; kept for signature symmetry).
        lengths: The stream lengths ``L`` (``n_tasks``) to sweep.
        key_arms: Arms measured at each length; defaults to
            ``['full_dream', 'long_context', 'no_sleep']``.

    Returns:
        The assembled :class:`~slow_wave.eval.phase5_schema.LengthSweep`, its
        points ascending in ``L``.
    """
    arms = list(key_arms) if key_arms is not None else list(_DEFAULT_LENGTH_ARMS)
    swept = sorted(lengths)
    seeds = list(cfg.eval.seeds)

    set_global_seeds(cfg.seed)
    embedder = get_embedder(cfg)

    points: list[LengthPoint] = []
    for length in swept:
        var_cfg = cfg.model_copy(deep=True)
        var_cfg.stream.n_tasks = length

        acc_by_arm: dict[str, list[float]] = {a: [] for a in arms}
        tokens_by_arm: dict[str, list[float]] = {a: [] for a in arms}
        vectors_by_arm: dict[str, list[float]] = {a: [] for a in arms}
        for seed in seeds:
            stream = generate_stream(var_cfg.stream, _stream_seed(seed))
            probe_set = build_probe_set(stream)
            for arm in arms:
                res = run_arm(arm, var_cfg, stream, probe_set, embedder, seed)
                acc_by_arm[arm].append(float(res.continual_metrics.acc))
                tokens_by_arm[arm].append(float(res.cost.total_tokens))
                vectors_by_arm[arm].append(float(res.cost.memory_vectors))

        points.append(
            LengthPoint(
                n_tasks=length,
                items_per_task=var_cfg.stream.items_per_task,
                arms=list(arms),
                seeds=list(seeds),
                acc_by_arm=acc_by_arm,
                total_tokens_by_arm=tokens_by_arm,
                memory_vectors_by_arm=vectors_by_arm,
            )
        )

    coverage = [
        f"length sweep: ran {len(swept)} lengths {swept} x {len(seeds)} seeds x "
        f"{len(arms)} arms = {len(swept) * len(seeds) * len(arms)} runs; 0 dropped",
        f"length sweep arms={arms}",
    ]
    for note in coverage:
        logger.info(note)

    return LengthSweep(
        treatment_arm="full_dream",
        baseline_arm="long_context",
        points=points,
        coverage_notes=coverage,
    )


def run_sim_vs_real(
    cfg: Config,
    out_dir: str | Path,
    *,
    key_arms: list[str] | None = None,
    sim_n_tasks: int,
    sim_compression: float,
    real_n_tasks: int,
    real_compression: float,
) -> SimRealResult:
    """Accelerated-sim vs real-long-horizon runs per key arm (EC3).

    Builds a "sim" config (``n_tasks=sim_n_tasks``, ``sim_time.compression_factor=
    sim_compression`` — a short, accelerated stream) and a "real" config
    (``n_tasks=real_n_tasks``, ``compression=real_compression`` — a long,
    low-compression stream). For each ``key_arm`` every seed in ``cfg.eval.seeds``
    is run via :func:`slow_wave.eval.harness.run_arm` on both, collecting per-seed
    ACC and the final-row retention curve. The numpy-only agreement scalars —
    Pearson r and Spearman rank correlation between the per-arm **mean-ACC**
    vectors (sim vs real), whether the arm ranking is preserved, any inversions,
    and the worst-case mean-ACC divergence — quantify whether time-compression
    distorts retention (a documented inversion at scale is a *finding*, EC3).

    With a single arm or zero-variance mean vector the correlations are defined as
    ``1.0`` with an honest note (never raised).

    Args:
        cfg: The base experiment config (``cfg.stream`` must be set).
        out_dir: Output root (unused for I/O here; kept for signature symmetry).
        key_arms: Arms compared; defaults to
            ``['full_dream', 'no_sleep', 'long_context']``.
        sim_n_tasks: Stream length for the accelerated sim runs.
        sim_compression: ``sim_time.compression_factor`` for the sim runs.
        real_n_tasks: Stream length for the real long-horizon runs.
        real_compression: ``sim_time.compression_factor`` for the real runs.

    Returns:
        The assembled :class:`~slow_wave.eval.phase5_schema.SimRealResult`.
    """
    arms = list(key_arms) if key_arms is not None else list(_DEFAULT_SIMREAL_ARMS)
    seeds = list(cfg.eval.seeds)

    set_global_seeds(cfg.seed)
    embedder = get_embedder(cfg)

    sim_cfg = cfg.model_copy(deep=True)
    sim_cfg.stream.n_tasks = sim_n_tasks
    sim_cfg.sim_time.compression_factor = sim_compression

    real_cfg = cfg.model_copy(deep=True)
    real_cfg.stream.n_tasks = real_n_tasks
    real_cfg.sim_time.compression_factor = real_compression

    def _run_side(side_cfg: Config) -> dict[str, list[ArmResult]]:
        """Run every key arm over every seed on one (sim or real) config."""
        per_arm: dict[str, list[ArmResult]] = {a: [] for a in arms}
        for seed in seeds:
            stream = generate_stream(side_cfg.stream, _stream_seed(seed))
            probe_set = build_probe_set(stream)
            for arm in arms:
                per_arm[arm].append(
                    run_arm(arm, side_cfg, stream, probe_set, embedder, seed)
                )
        return per_arm

    sim_results = _run_side(sim_cfg)
    real_results = _run_side(real_cfg)

    sim_arms: list[SimRealArm] = []
    sim_means: list[float] = []
    real_means: list[float] = []
    for arm in arms:
        acc_sim = [float(r.continual_metrics.acc) for r in sim_results[arm]]
        acc_real = [float(r.continual_metrics.acc) for r in real_results[arm]]
        sim_means.append(float(np.mean(acc_sim)) if acc_sim else 0.0)
        real_means.append(float(np.mean(acc_real)) if acc_real else 0.0)
        sim_arms.append(
            SimRealArm(
                arm_name=arm,
                acc_sim_per_seed=acc_sim,
                acc_real_per_seed=acc_real,
                retention_sim=_retention_curve(arm, sim_results[arm], seeds),
                retention_real=_retention_curve(arm, real_results[arm], seeds),
            )
        )

    pearson, spearman, ranking_preserved, inversions, max_div, note = _agreement(
        arms, np.asarray(sim_means), np.asarray(real_means)
    )

    coverage = (
        f"sim/real: {len(arms)} arms x {len(seeds)} seeds on sim(n_tasks="
        f"{sim_n_tasks}, c={sim_compression}) and real(n_tasks={real_n_tasks}, "
        f"c={real_compression}) = {2 * len(arms) * len(seeds)} runs; 0 dropped"
    )
    logger.info(coverage)
    logger.info("sim/real agreement: %s", note)

    return SimRealResult(
        arms=sim_arms,
        sim_n_tasks=sim_n_tasks,
        sim_compression=sim_compression,
        real_n_tasks=real_n_tasks,
        real_compression=real_compression,
        seeds=list(seeds),
        pearson_agreement=pearson,
        spearman_agreement=spearman,
        ranking_preserved=ranking_preserved,
        inversions=inversions,
        max_abs_acc_divergence=max_div,
        note=note,
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return ascending ordinal ranks of ``values`` via a stable argsort."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation of two equal-length vectors, or ``None`` if undefined.

    Returns ``None`` when either vector has fewer than two points or zero
    variance (the correlation is undefined), so the caller can substitute the
    documented ``1.0`` agreement default with a note.
    """
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r):
        return None
    return float(np.clip(r, -1.0, 1.0))


def _agreement(
    arms: list[str], sim_means: np.ndarray, real_means: np.ndarray
) -> tuple[float, float, bool, list[str], float, str]:
    """Compute the sim-vs-real agreement scalars + honest note (EC3).

    Args:
        arms: The compared arm names (ordering matches the mean vectors).
        sim_means: Per-arm mean sim ACC.
        real_means: Per-arm mean real ACC.

    Returns:
        ``(pearson, spearman, ranking_preserved, inversions, max_abs_acc_
        divergence, note)``. Pearson/Spearman default to ``1.0`` with a note when
        undefined (single arm / zero variance); never raises.
    """
    notes: list[str] = []
    pearson_opt = _corr(sim_means, real_means)
    spearman_opt = (
        _corr(_ranks(sim_means), _ranks(real_means)) if sim_means.size >= 2 else None
    )
    if pearson_opt is None or spearman_opt is None:
        notes.append(
            "single arm or zero-variance mean vector; agreement defined as 1.0"
        )
    pearson = 1.0 if pearson_opt is None else pearson_opt
    spearman = 1.0 if spearman_opt is None else spearman_opt

    inversions: list[str] = []
    n = len(arms)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(np.sign(sim_means[i] - sim_means[j]))
            r = float(np.sign(real_means[i] - real_means[j]))
            if s != 0.0 and r != 0.0 and s != r:
                inversions.append(f"{arms[i]} vs {arms[j]}")
    ranking_preserved = not inversions

    max_div = (
        float(np.max(np.abs(sim_means - real_means))) if sim_means.size else 0.0
    )

    if not arms:
        headline = "no arms compared"
    elif inversions:
        headline = (
            f"documented ranking inversion(s) under time-compression: "
            f"{', '.join(inversions)}"
        )
    else:
        headline = "sim preserves the real arm ranking (no inversions)"
    note = (
        f"{headline}; pearson={pearson:.3f}, spearman={spearman:.3f}, "
        f"max|mean ACC sim-real|={max_div:.3f}"
    )
    if notes:
        note = f"{note} [{'; '.join(notes)}]"
    return pearson, spearman, ranking_preserved, inversions, max_div, note


def _retention_from_manifest(
    manifest_path: str, arms: list[str], seeds: list[int]
) -> list[RetentionCurve]:
    """Build per-arm retention curves from a written per-cell manifest.

    Reads the regime cell's manifest (its ``results.experiment.arm_results`` holds
    each arm/seed accuracy matrix) and extracts the final row ``R[T-1][:]`` per
    seed for every arm, so the retention figure can draw seed bands tied to the
    committed manifest. Labels are not read — only the accuracy matrices.

    Args:
        manifest_path: Path to the per-cell manifest JSON.
        arms: The arms (report order) to build curves for.
        seeds: The seeds aligned with each curve's per-seed rows.

    Returns:
        One :class:`RetentionCurve` per arm.
    """
    manifest = read_manifest(manifest_path)
    arm_results = manifest.results["experiment"]["arm_results"]
    by: dict[tuple[str, int], dict] = {
        (ar["arm_name"], ar["seed"]): ar for ar in arm_results
    }
    curves: list[RetentionCurve] = []
    for arm in arms:
        rows: list[list[float]] = []
        n_tasks = 0
        for s in seeds:
            ar = by[(arm, s)]
            matrix = ar["accuracy_matrix"]
            n_tasks = matrix["n_tasks"]
            r_rows = matrix["R"]
            rows.append([float(v) for v in r_rows[-1]] if r_rows else [])
        curves.append(
            RetentionCurve(
                arm_name=arm,
                n_tasks=n_tasks,
                final_row_per_seed=rows,
                seeds=list(seeds),
            )
        )
    return curves


def run_phase5(cfg: Config, out_dir: str | Path | None = None) -> Path:
    """Run all three Phase 5 sweeps and write the committed artifact (one command).

    Runs :func:`run_regime_sweep` (EC1), :func:`run_length_sweep` (EC6) and
    :func:`run_sim_vs_real` (EC3) — reading their knobs from
    ``cfg.hyperparameters["phase5"]`` — then assembles a
    :class:`~slow_wave.eval.phase5_schema.Phase5Result` (``analysis=None``; WS2
    fills it later) and writes it to ``<out>/phase5/phase5_result.json`` as pretty,
    key-sorted JSON. The result carries no wall-clock/timestamp, so two calls on
    the same config produce a byte-identical file (DX1).

    Args:
        cfg: The fully-loaded experiment config (with the ``phase5`` hyperparameter
            block and ``cfg.stream`` set).
        out_dir: Output root; defaults to ``cfg.output_dir``.

    Returns:
        The path to the written ``phase5_result.json``.
    """
    out = Path(out_dir or cfg.output_dir)
    phase5 = cfg.hyperparameters["phase5"]
    set_global_seeds(cfg.seed)

    grid = run_regime_sweep(cfg, out)

    length_cfg = phase5["length_sweep"]
    length_sweep = run_length_sweep(
        cfg,
        out,
        lengths=list(length_cfg["lengths"]),
        key_arms=list(length_cfg.get("key_arms", _DEFAULT_LENGTH_ARMS)),
    )

    sim_cfg = phase5["sim_real"]
    sim_real = run_sim_vs_real(
        cfg,
        out,
        key_arms=list(sim_cfg.get("key_arms", _DEFAULT_SIMREAL_ARMS)),
        sim_n_tasks=int(sim_cfg["sim_n_tasks"]),
        sim_compression=float(sim_cfg["sim_compression"]),
        real_n_tasks=int(sim_cfg["real_n_tasks"]),
        real_compression=float(sim_cfg["real_compression"]),
    )

    # Retention seed-bands for the primary regime, read from its committed manifest.
    manifest_paths = [cell.manifest_path for cell in grid.regimes]
    primary_cell = next(
        (c for c in grid.regimes if c.regime.name == grid.primary_regime),
        grid.regimes[0] if grid.regimes else None,
    )
    retention: dict[str, list[RetentionCurve]] = {}
    if primary_cell is not None:
        retention[grid.primary_regime] = _retention_from_manifest(
            primary_cell.manifest_path, grid.arms, grid.seeds
        )

    # model_mocked from a probe LLM result (mock => True under no API key).
    probe = complete(cfg, "phase5 provenance probe")

    coverage_notes = list(grid.coverage_notes)
    coverage_notes.extend(length_sweep.coverage_notes)
    coverage_notes.append(sim_real.note)
    coverage_notes.append(
        f"phase5: {len(grid.regimes)} regimes, {len(length_sweep.points)} lengths, "
        f"{len(sim_real.arms)} sim/real arms; {len(manifest_paths)} per-cell "
        f"manifests written; model_mocked={probe.mocked}"
    )

    result = Phase5Result(
        experiment=cfg.experiment,
        scenario=cfg.stream.scenario.value,
        git_commit=git_info().get("commit"),
        model_id=cfg.model.id,
        model_mocked=probe.mocked,
        grid=grid,
        length_sweep=length_sweep,
        sim_real=sim_real,
        retention=retention,
        analysis=None,
        coverage_notes=coverage_notes,
        manifest_paths=manifest_paths,
    )

    out_path = out / "phase5" / "phase5_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    logger.info("phase5 result written to %s", out_path)
    print(f"[phase5] result written to {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.eval.grid`` (mirrors the runner).

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-phase5",
        description=(
            "Run the Phase 5 regime x arm x seed grid + length & sim/real sweeps "
            "and write phase5_result.json."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/phase5_smoke.yaml",
        help="Path to the Phase 5 YAML config (default: configs/phase5_smoke.yaml).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default: the config's output_dir).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = run_phase5(cfg, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
