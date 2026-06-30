"""Phase 5 deliverable figures, regenerated from a committed ``Phase5Result``.

This is the **figures layer** (WS3; PRD §8 Phase 5, ``docs/PHASE5_CONTRACT.md``).
It reads a committed :class:`~slow_wave.eval.phase5_schema.Phase5Result` (with its
:attr:`~slow_wave.eval.phase5_schema.Phase5Result.analysis` filled by WS2) and
regenerates the seven deliverable figures so the whole figure set reproduces from
one command (EC4). It introduces **no new science** — it only visualizes the raw
per-seed arrays WS1 measured and the verdicts WS2 computed.

The seven figures
-----------------
#. ``retention_curves`` — per-task retention with seed-variability bands.
#. ``ablation_table`` — ablation summary incl. the replay×downscale 2×2.
#. ``mechanism_pr`` — prune precision / recall / F1 per arm.
#. ``cost_pareto`` — accuracy vs compute with the Pareto frontier highlighted.
#. ``long_context_crossover`` — acc-per-token vs L, marking the crossover (EC6).
#. ``tmr_targeting`` — TMR-style signal-retention lift vs Hu et al. 2020 g≈0.29.
#. ``sim_vs_real`` — accelerated-sim vs real-long-horizon agreement (EC3).

Plots a reviewer trusts (DX4)
-----------------------------
Every panel that draws per-seed data carries a **seed-variability band** (shaded
95% normal-approximation CI), uses a **colorblind-safe** Okabe–Ito palette, avoids
chart-junk, and is written as a **vector PDF** (plus a ``.png`` sibling for quick
view). Every figure's caption states **n** (the number of seeds) and the **CI
method**, and carries the mandatory **mock-LLM caveat** (DX5); captions live both
on the figure (a small wrapped text box) and in ``figures_manifest.json``. All
means and intervals are computed from the raw per-seed arrays with numpy — no data
is invented.

The matplotlib/CI rule (load-bearing)
-------------------------------------
CI installs no matplotlib, so this module **never imports matplotlib at module
scope** — it imports it *lazily inside the function bodies* (after calling
``matplotlib.use("Agg")`` for headless rendering). The module therefore imports
cleanly without matplotlib; only the rendering functions need it. Its test guards
every rendering test with ``pytest.importorskip("matplotlib")``.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path

import numpy as np

from slow_wave.eval.phase5_schema import Phase5Result, RegimeCell

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #

#: ``{figure_key -> output PDF filename}`` for the seven Phase 5 deliverables.
#: ``generate_all_figures`` renders each key to ``out_dir / FIGURES[key]`` (a
#: vector PDF) plus a ``.png`` sibling, in this order.
FIGURES: dict[str, str] = {
    "retention_curves": "retention_curves.pdf",
    "ablation_table": "ablation_table.pdf",
    "mechanism_pr": "mechanism_pr.pdf",
    "cost_pareto": "cost_pareto.pdf",
    "long_context_crossover": "long_context_crossover.pdf",
    "tmr_targeting": "tmr_targeting.pdf",
    "sim_vs_real": "sim_vs_real.pdf",
}

#: Okabe–Ito colorblind-safe qualitative palette (8 hues + grey).
_CB_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#999999",  # grey
)

#: ~95% two-sided normal quantile (matches ``slow_wave.eval.stats``).
_Z95: float = 1.959964

#: The CI-method token every caption advertises (DX4).
_CI_METHOD: str = "95% CI (normal approx, mean +/- 1.96*SE over seeds)"

#: Fallback mock-LLM caveat (DX5) if the analysis somehow carries an empty one.
_MOCK_CAVEAT_FALLBACK: str = (
    "MOCK-LLM CAVEAT: all numbers are a mechanism demonstration in the synthetic + "
    "deterministic mock-LLM regime, NOT a scientific claim about a real Claude model."
)

# Which seed pool each figure's n refers to.
_SEED_SOURCE: dict[str, str] = {
    "retention_curves": "grid",
    "ablation_table": "grid",
    "mechanism_pr": "grid",
    "cost_pareto": "grid",
    "long_context_crossover": "length",
    "tmr_targeting": "grid",
    "sim_vs_real": "simreal",
}

# The four cells of the replay×downscale 2×2 ablation.
_NO_SLEEP = "no_sleep"
_REPLAY_ONLY = "replay_only"
_DOWNSCALE_ONLY = "downscale_only"
_FULL_DREAM = "full_dream"


# --------------------------------------------------------------------------- #
# numpy aggregation helpers (no matplotlib here)
# --------------------------------------------------------------------------- #
def _mean_ci(values) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` for a 95% normal-approximation CI of the mean.

    Args:
        values: A 1-D iterable of per-seed scalars.

    Returns:
        ``(mean, lo, hi)``. With fewer than two finite values the band collapses
        to the point estimate (``lo == hi == mean``); an empty input yields zeros.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1)) / np.sqrt(arr.size)
    return mean, mean - _Z95 * se, mean + _Z95 * se


def _column_mean_ci(rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column ``(mean, lo, hi)`` over a ``(n_seeds, n_cols)`` matrix."""
    mat = np.asarray(rows, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    means, los, his = [], [], []
    for col in mat.T:
        m, lo, hi = _mean_ci(col)
        means.append(m)
        los.append(lo)
        his.append(hi)
    return np.asarray(means), np.asarray(los), np.asarray(his)


def _arm_means(by_arm: dict[str, list[float]], arms: list[str]):
    """Return aligned ``(means, lo, hi)`` arrays for ``arms`` from a per-arm dict."""
    means, los, his = [], [], []
    for arm in arms:
        m, lo, hi = _mean_ci(by_arm.get(arm, []))
        means.append(m)
        los.append(lo)
        his.append(hi)
    return np.asarray(means), np.asarray(los), np.asarray(his)


def _primary_cell(result: Phase5Result) -> RegimeCell:
    """Return the registered primary-regime :class:`RegimeCell` (or the first)."""
    name = result.grid.primary_regime
    for cell in result.grid.regimes:
        if cell.regime.name == name:
            return cell
    if result.grid.regimes:
        return result.grid.regimes[0]
    raise ValueError("Phase5Result.grid has no regime cells to plot.")


def _seed_count(result: Phase5Result, key: str) -> int:
    """Number of seeds behind ``key``'s figure (its caption's ``n``)."""
    source = _SEED_SOURCE.get(key, "grid")
    if source == "grid":
        return len(_primary_cell(result).seeds)
    if source == "length":
        pts = result.length_sweep.points
        return len(pts[0].seeds) if pts else 0
    if source == "simreal":
        return len(result.sim_real.seeds)
    return 0


def _pareto_frontier(accs: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-non-dominated points (max accuracy, min tokens)."""
    n = len(accs)
    on = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = (
                accs[j] >= accs[i]
                and tokens[j] <= tokens[i]
                and (accs[j] > accs[i] or tokens[j] < tokens[i])
            )
            if dominates:
                on[i] = False
                break
    return on


# --------------------------------------------------------------------------- #
# Captions (single source of truth for figure text + manifest)
# --------------------------------------------------------------------------- #
def _caveat(result: Phase5Result) -> str:
    """The mock-LLM caveat to stamp on every figure (DX5)."""
    if result.analysis is not None and result.analysis.mock_llm_caveat:
        text = result.analysis.mock_llm_caveat.strip()
        prefix = "" if text.upper().startswith("MOCK-LLM CAVEAT") else "MOCK-LLM CAVEAT: "
        return f"{prefix}{text}"
    return _MOCK_CAVEAT_FALLBACK


def _build_caption(result: Phase5Result, key: str) -> str:
    """Build the caption for one figure (states n, the CI method, and the caveat).

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        key: One of the :data:`FIGURES` keys.

    Returns:
        A single-line caption guaranteed to contain an ``n=`` token, a CI-method
        token (``95% CI``), and the mock-LLM caveat (DX4/DX5).
    """
    n = _seed_count(result, key)
    regime = result.grid.primary_regime
    a = result.analysis
    bodies = {
        "retention_curves": (
            f"Per-task retention (final-row accuracy R[T-1,j] after the full stream) "
            f"per arm in the '{regime}' regime."
        ),
        "ablation_table": (
            "Ablation summary in the primary regime: the replay x downscale 2x2 "
            "(no_sleep / replay_only / downscale_only / full_dream) as a mean-ACC "
            "heatmap, beside per-arm mean ACC."
        ),
        "mechanism_pr": (
            "Mechanism-level prune quality (precision / recall / F1) per arm in the "
            "primary regime, decoupled from downstream accuracy."
        ),
        "cost_pareto": (
            "Accuracy vs compute (total tokens) per arm in the primary regime; the "
            "Pareto-non-dominated frontier (max ACC, min tokens) is highlighted."
        ),
        "long_context_crossover": (
            "Long-context crossover (EC6): acc-per-token (mean ACC / mean tokens, the "
            "preregistered metric) for treatment vs baseline across stream length L."
        ),
        "tmr_targeting": (
            "TMR-style targeting: signal-retention lift for replay arms over no-replay "
            "arms (a replay-targeting analogue, NOT a literal cued-TMR protocol), "
            "vs the Hu et al. 2020 benchmark Hedges' g=0.29."
        ),
        "sim_vs_real": (
            "Sim-vs-real agreement (EC3): per-arm mean ACC on the accelerated "
            "high-compression sim stream vs the real low-compression long-horizon "
            "stream; annotated with the rank agreement and any inversion."
        ),
    }
    extras: list[str] = []
    if key == "long_context_crossover" and a is not None:
        if a.crossover.crossover_found and a.crossover.crossover_length is not None:
            extras.append(f"crossover at L={a.crossover.crossover_length}")
        else:
            extras.append("no cost-adjusted crossover in the swept range")
    if key == "tmr_targeting" and a is not None:
        extras.append(
            f"pooled obs: replay={len(a.tmr.signal_retention_replay)}, "
            f"no_replay={len(a.tmr.signal_retention_no_replay)}; "
            f"g={a.tmr.hedges_g:.2f}"
        )
    if key == "sim_vs_real":
        extras.append(
            f"Spearman={result.sim_real.spearman_agreement:.2f}, "
            f"ranking_preserved={result.sim_real.ranking_preserved}"
        )
    extra = (" " + "; ".join(extras) + ".") if extras else ""

    body = bodies[key]
    ci_method = _CI_METHOD
    if key == "tmr_targeting":
        # The right-panel Hedges' g error bar is a bootstrap CI, not the per-seed
        # normal-approx band used elsewhere — state both methods accurately (DX4).
        ci_method = f"{_CI_METHOD}; Hedges' g error bar = bootstrap 95% CI"
    return (
        f"{body}{extra} n={n} seeds; bands/error bars = {ci_method}. {_caveat(result)}"
    )


# --------------------------------------------------------------------------- #
# Lazy matplotlib plumbing (imported only inside function bodies)
# --------------------------------------------------------------------------- #
def _pyplot():
    """Import pyplot lazily on the headless ``Agg`` backend.

    Imported inside the function body (never at module scope) so the module
    imports cleanly where matplotlib is absent (CI). Returns the ``pyplot``
    module after forcing the non-interactive ``Agg`` backend.
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, plt, result: Phase5Result, out_dir: Path, key: str, title: str) -> Path:
    """Stamp the title + caption, save vector PDF + PNG sibling, and close.

    Args:
        fig: The matplotlib figure.
        plt: The pyplot module from :func:`_pyplot`.
        result: The artifact (for the caption).
        out_dir: Destination directory (created if needed).
        key: The :data:`FIGURES` key (selects the output filename).
        title: The figure's bold suptitle.

    Returns:
        Path to the written PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    caption = _build_caption(result, key)
    wrapped = "\n".join(textwrap.wrap(caption, width=118))
    n_lines = wrapped.count("\n") + 1
    bottom = min(0.30, 0.06 + 0.022 * n_lines)
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.95))
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=7, wrap=True)

    pdf_path = out_dir / FIGURES[key]
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# --------------------------------------------------------------------------- #
# Figure 1 — retention curves
# --------------------------------------------------------------------------- #
def figure_retention_curves(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render per-task retention curves with seed-variability bands (DX4).

    For each arm's :class:`RetentionCurve` in the primary regime, plots the
    per-task mean of the final accuracy row R[T-1, j] with a shaded 95% CI band
    over seeds, so a reviewer sees both the retention profile and its variability.

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    regime = result.grid.primary_regime
    curves = result.retention.get(regime) or next(iter(result.retention.values()), [])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, curve in enumerate(curves):
        color = _CB_PALETTE[i % len(_CB_PALETTE)]
        means, los, his = _column_mean_ci(curve.final_row_per_seed)
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means, marker="o", color=color, label=curve.arm_name, linewidth=1.8)
        ax.fill_between(x, los, his, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("task index j (presentation order)")
    ax.set_ylabel("retention accuracy R[T-1, j]")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower left")
    return _finish(
        fig, plt, result, Path(out_dir), "retention_curves",
        f"Retention curves ({regime} regime)",
    )


# --------------------------------------------------------------------------- #
# Figure 2 — ablation table (incl. replay x downscale 2x2)
# --------------------------------------------------------------------------- #
def figure_ablation_table(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render the ablation summary including the replay×downscale 2×2 (DX4).

    Left panel: the 2×2 mean-ACC heatmap whose four cells are
    ``no_sleep`` / ``replay_only`` / ``downscale_only`` / ``full_dream`` (each
    annotated with its mean ACC and 95% CI). Right panel: every arm's mean ACC
    with a 95% CI error bar.

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    cell = _primary_cell(result)
    acc = cell.acc_by_arm

    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(10.4, 4.8), width_ratios=[1.0, 1.25])

    # --- 2x2 heatmap: rows = downscale off/on, cols = replay off/on ---------- #
    layout = [[_NO_SLEEP, _REPLAY_ONLY], [_DOWNSCALE_ONLY, _FULL_DREAM]]
    grid_mean = np.full((2, 2), np.nan)
    annot = [["", ""], ["", ""]]
    for r in range(2):
        for c in range(2):
            arm = layout[r][c]
            m, lo, hi = _mean_ci(acc.get(arm, []))
            grid_mean[r, c] = m
            annot[r][c] = f"{arm}\n{m:.3f}\n[{lo:.3f}, {hi:.3f}]"
    im = ax_h.imshow(grid_mean, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax_h.set_xticks([0, 1], ["replay off", "replay on"])
    ax_h.set_yticks([0, 1], ["downscale off", "downscale on"])
    for r in range(2):
        for c in range(2):
            val = grid_mean[r, c]
            txt_color = "white" if (np.isnan(val) or val < 0.55) else "black"
            ax_h.text(c, r, annot[r][c], ha="center", va="center",
                      fontsize=8, color=txt_color)
    ax_h.set_title("replay x downscale 2x2 (mean ACC)", fontsize=10)
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="mean ACC")

    # --- right: all-arm mean ACC bars with CI -------------------------------- #
    arms = list(cell.arms) if cell.arms else list(acc.keys())
    means, los, his = _arm_means(acc, arms)
    yerr = np.vstack([means - los, his - means])
    x = np.arange(len(arms))
    colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(len(arms))]
    ax_b.bar(x, means, yerr=yerr, capsize=3, color=colors, alpha=0.9,
             error_kw={"linewidth": 1.0})
    ax_b.set_xticks(x, arms, rotation=40, ha="right", fontsize=8)
    ax_b.set_ylabel("mean ACC")
    ax_b.set_ylim(0.0, 1.02)
    ax_b.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_b.set_title("per-arm mean ACC", fontsize=10)

    return _finish(
        fig, plt, result, Path(out_dir), "ablation_table",
        f"Ablation summary ({result.grid.primary_regime} regime)",
    )


# --------------------------------------------------------------------------- #
# Figure 3 — mechanism precision/recall/F1
# --------------------------------------------------------------------------- #
def figure_mechanism_pr(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render per-arm prune precision / recall / F1 with seed bands (DX4).

    Grouped bars (precision, recall, F1) per arm in the primary regime, each with
    a 95% CI error bar over seeds — the bench's decoupled mechanism metric.

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    cell = _primary_cell(result)
    arms = list(cell.arms) if cell.arms else list(cell.prune_f1_by_arm.keys())

    metrics = [
        ("precision", cell.prune_precision_by_arm, _CB_PALETTE[0]),
        ("recall", cell.prune_recall_by_arm, _CB_PALETTE[1]),
        ("F1", cell.prune_f1_by_arm, _CB_PALETTE[2]),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(arms))
    width = 0.26
    for k, (label, by_arm, color) in enumerate(metrics):
        means, los, his = _arm_means(by_arm, arms)
        yerr = np.vstack([means - los, his - means])
        ax.bar(x + (k - 1) * width, means, width, yerr=yerr, capsize=2.5,
               label=label, color=color, alpha=0.9, error_kw={"linewidth": 0.9})

    ax.set_xticks(x, arms, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper right")
    return _finish(
        fig, plt, result, Path(out_dir), "mechanism_pr",
        f"Prune quality per arm ({result.grid.primary_regime} regime)",
    )


# --------------------------------------------------------------------------- #
# Figure 4 — cost / accuracy Pareto
# --------------------------------------------------------------------------- #
def figure_cost_pareto(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render accuracy vs compute with the Pareto frontier highlighted (DX4).

    Per-arm mean ACC vs mean total tokens (primary regime), with a vertical 95%
    CI error bar on accuracy. The Pareto-non-dominated arms (max ACC, min tokens)
    are connected by a frontier line and outlined.

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    cell = _primary_cell(result)
    arms = list(cell.arms) if cell.arms else list(cell.acc_by_arm.keys())

    accs, alos, ahis, toks = [], [], [], []
    for arm in arms:
        m, lo, hi = _mean_ci(cell.acc_by_arm.get(arm, []))
        accs.append(m)
        alos.append(lo)
        ahis.append(hi)
        toks.append(_mean_ci(cell.total_tokens_by_arm.get(arm, []))[0])
    accs = np.asarray(accs)
    toks = np.asarray(toks)
    yerr = np.vstack([accs - np.asarray(alos), np.asarray(ahis) - accs])
    on = _pareto_frontier(accs, toks)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.errorbar(toks, accs, yerr=yerr, fmt="none", ecolor="#999999",
                elinewidth=0.9, capsize=3, zorder=1)
    for i, arm in enumerate(arms):
        color = _CB_PALETTE[i % len(_CB_PALETTE)]
        edge = "#000000" if on[i] else "none"
        ax.scatter(toks[i], accs[i], s=90 if on[i] else 55, color=color,
                   edgecolors=edge, linewidths=1.4, zorder=3, label=arm)
        ax.annotate(arm, (toks[i], accs[i]), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    if on.any():
        order = np.argsort(toks[on])
        ax.plot(toks[on][order], accs[on][order], color="#000000",
                linewidth=1.3, linestyle="--", alpha=0.7,
                label="Pareto frontier", zorder=2)

    ax.set_xlabel("mean compute (total tokens)")
    ax.set_ylabel("mean ACC")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    return _finish(
        fig, plt, result, Path(out_dir), "cost_pareto",
        f"Cost / accuracy Pareto ({result.grid.primary_regime} regime)",
    )


# --------------------------------------------------------------------------- #
# Figure 5 — long-context crossover (EC6)
# --------------------------------------------------------------------------- #
def figure_long_context_crossover(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render the long-context crossover (EC6) from ``analysis.crossover``.

    Plots the preregistered acc-per-token metric (mean ACC / mean tokens) for the
    treatment vs baseline arm across stream length L, marks the cost-adjusted
    crossover (or annotates its absence), and overlays per-seed variability bands
    recomputed from the length sweep where the arms are present.

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    cx = result.analysis.crossover
    lengths = np.asarray(cx.lengths, dtype=float)

    # per-seed acc-per-token bands from the raw length sweep, keyed by L.
    def _band(arm: str) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for pt in result.length_sweep.points:
            acc = np.asarray(pt.acc_by_arm.get(arm, []), dtype=float)
            tok = np.asarray(pt.total_tokens_by_arm.get(arm, []), dtype=float)
            if acc.size == 0 or tok.size != acc.size:
                continue
            ratio = np.divide(acc, tok, out=np.zeros_like(acc), where=tok > 0)
            _, lo, hi = _mean_ci(ratio)
            out[pt.n_tasks] = (lo, hi)
        return out

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    series = [
        (cx.treatment_arm, cx.acc_per_token_treatment, _CB_PALETTE[0]),
        (cx.baseline_arm, cx.acc_per_token_baseline, _CB_PALETTE[3]),
    ]
    for arm, vals, color in series:
        y = np.asarray(vals, dtype=float)
        ax.plot(lengths, y, marker="o", color=color, linewidth=1.9, label=arm)
        band = _band(arm)
        los = np.array([band.get(int(L), (yi, yi))[0] for L, yi in zip(cx.lengths, y)])
        his = np.array([band.get(int(L), (yi, yi))[1] for L, yi in zip(cx.lengths, y)])
        ax.fill_between(lengths, los, his, color=color, alpha=0.16, linewidth=0)

    if cx.crossover_found and cx.crossover_length is not None:
        ax.axvline(cx.crossover_length, color="#009E73", linestyle="--", linewidth=1.4)
        ax.annotate(f"crossover L={cx.crossover_length}",
                    (cx.crossover_length, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=9, color="#009E73")
    else:
        ax.text(0.5, 0.94, "no cost-adjusted crossover in swept range",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="#D55E00")

    ax.set_xlabel("stream length L (n_tasks)")
    ax.set_ylabel("acc per token (mean ACC / mean tokens)")
    ax.set_xticks(cx.lengths)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=9, frameon=False, loc="best")
    return _finish(
        fig, plt, result, Path(out_dir), "long_context_crossover",
        "Long-context crossover (EC6)",
    )


# --------------------------------------------------------------------------- #
# Figure 6 — TMR targeting (FR5.3)
# --------------------------------------------------------------------------- #
def figure_tmr_targeting(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render the TMR-style signal-retention lift vs the Hu 2020 g=0.29 line.

    Left panel: per-(arm,seed) signal retention for replay vs no-replay groups
    (raw points + group mean and 95% CI). Right panel: the standardized lift
    (Hedges' g) with its bootstrap CI and the Hu et al. (2020) benchmark g=0.29
    reference line. This is a replay-targeting *analogue*, not a literal cued-TMR
    protocol (DX5).

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    tmr = result.analysis.tmr
    replay = np.asarray(tmr.signal_retention_replay, dtype=float)
    no_replay = np.asarray(tmr.signal_retention_no_replay, dtype=float)

    fig, (ax_g, ax_e) = plt.subplots(1, 2, figsize=(10.2, 4.8), width_ratios=[1.25, 1.0])

    # --- left: grouped signal retention -------------------------------------- #
    rng = np.random.default_rng(0)
    for pos, (data, color, name) in enumerate([
        (replay, _CB_PALETTE[2], "replay"),
        (no_replay, _CB_PALETTE[4], "no_replay"),
    ]):
        if data.size:
            jitter = (rng.random(data.size) - 0.5) * 0.18
            ax_g.scatter(np.full(data.size, pos) + jitter, data, color=color,
                         alpha=0.6, s=32, zorder=2)
            m, lo, hi = _mean_ci(data)
            ax_g.errorbar(pos, m, yerr=[[m - lo], [hi - m]], fmt="D", color=color,
                          capsize=5, markersize=8, zorder=3, elinewidth=1.4)
    ax_g.set_xticks([0, 1], ["replay arms", "no-replay arms"], fontsize=9)
    ax_g.set_ylabel("signal retention")
    ax_g.set_ylim(-0.02, 1.02)
    ax_g.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_g.set_title(f"mean lift = {tmr.mean_lift:+.3f}", fontsize=10)

    # --- right: hedges g vs benchmark ---------------------------------------- #
    g, glo, ghi = tmr.hedges_g, tmr.g_ci_lo, tmr.g_ci_hi
    ax_e.errorbar(0, g, yerr=[[max(g - glo, 0.0)], [max(ghi - g, 0.0)]], fmt="o",
                  color=_CB_PALETTE[0], capsize=6, markersize=10, elinewidth=1.6)
    ax_e.axhline(tmr.benchmark_g, color="#D55E00", linestyle="--", linewidth=1.5,
                 label=f"Hu et al. 2020 g={tmr.benchmark_g:.2f}")
    ax_e.axhline(0.0, color="#999999", linewidth=0.8)
    ax_e.set_xticks([0], ["replay vs no-replay"], fontsize=9)
    ax_e.set_ylabel("Hedges' g (signal-retention lift)")
    verdict = "exceeds" if tmr.exceeds_benchmark else "below"
    ax_e.set_title(f"g={g:.2f} ({verdict} benchmark)", fontsize=10)
    ax_e.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_e.legend(fontsize=9, frameon=False, loc="best")

    return _finish(
        fig, plt, result, Path(out_dir), "tmr_targeting",
        "TMR-style targeting effect (FR5.3)",
    )


# --------------------------------------------------------------------------- #
# Figure 7 — sim vs real (EC3)
# --------------------------------------------------------------------------- #
def figure_sim_vs_real(result: Phase5Result, out_dir: str | Path) -> Path:
    """Render the accelerated-sim vs real-long-horizon agreement (EC3).

    Left panel: per-arm mean ACC on the sim (short, high-compression) stream vs
    the real (long, low-compression) stream, with 95% CI error bars; annotated
    with the Pearson/Spearman agreement and any ranking inversion. Right panel:
    the sim-vs-real retention-curve overlay for the treatment arm.

    Args:
        result: The Phase 5 artifact (``analysis`` must be populated).
        out_dir: Destination directory.

    Returns:
        Path to the written PDF.
    """
    _require_analysis(result)
    plt = _pyplot()
    sr = result.sim_real
    arms = [a.arm_name for a in sr.arms]

    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.8), width_ratios=[1.3, 1.0])

    # --- left: grouped sim vs real mean ACC ---------------------------------- #
    x = np.arange(len(arms))
    width = 0.38
    sim_m, sim_lo, sim_hi, real_m, real_lo, real_hi = ([] for _ in range(6))
    for a in sr.arms:
        m, lo, hi = _mean_ci(a.acc_sim_per_seed)
        sim_m.append(m)
        sim_lo.append(lo)
        sim_hi.append(hi)
        m, lo, hi = _mean_ci(a.acc_real_per_seed)
        real_m.append(m)
        real_lo.append(lo)
        real_hi.append(hi)
    sim_m = np.asarray(sim_m)
    real_m = np.asarray(real_m)
    sim_err = np.vstack([sim_m - np.asarray(sim_lo), np.asarray(sim_hi) - sim_m])
    real_err = np.vstack([real_m - np.asarray(real_lo), np.asarray(real_hi) - real_m])
    ax_b.bar(x - width / 2, sim_m, width, yerr=sim_err, capsize=3, label="sim",
             color=_CB_PALETTE[0], alpha=0.9, error_kw={"linewidth": 0.9})
    ax_b.bar(x + width / 2, real_m, width, yerr=real_err, capsize=3, label="real",
             color=_CB_PALETTE[1], alpha=0.9, error_kw={"linewidth": 0.9})
    ax_b.set_xticks(x, arms, rotation=25, ha="right", fontsize=9)
    ax_b.set_ylabel("mean ACC")
    ax_b.set_ylim(0.0, 1.02)
    ax_b.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    inv = ", ".join(sr.inversions) if sr.inversions else "none"
    ax_b.set_title(
        f"Pearson={sr.pearson_agreement:.2f}  Spearman={sr.spearman_agreement:.2f}  "
        f"inversions: {inv}", fontsize=9,
    )
    ax_b.legend(fontsize=9, frameon=False, loc="lower right")

    # --- right: retention overlay for the treatment arm ---------------------- #
    target = next((a for a in sr.arms if a.arm_name == _FULL_DREAM), sr.arms[0] if sr.arms else None)
    if target is not None:
        sm, slo, shi = _column_mean_ci(target.retention_sim.final_row_per_seed)
        rm, rlo, rhi = _column_mean_ci(target.retention_real.final_row_per_seed)
        xs = np.arange(1, len(sm) + 1)
        xr = np.arange(1, len(rm) + 1)
        ax_r.plot(xs, sm, marker="o", color=_CB_PALETTE[0], label="sim", linewidth=1.8)
        ax_r.fill_between(xs, slo, shi, color=_CB_PALETTE[0], alpha=0.16, linewidth=0)
        ax_r.plot(xr, rm, marker="s", color=_CB_PALETTE[1], label="real", linewidth=1.8)
        ax_r.fill_between(xr, rlo, rhi, color=_CB_PALETTE[1], alpha=0.16, linewidth=0)
        ax_r.set_title(f"{target.arm_name} retention (sim vs real)", fontsize=10)
    ax_r.set_xlabel("task index j")
    ax_r.set_ylabel("retention R[T-1, j]")
    ax_r.set_ylim(-0.02, 1.02)
    ax_r.grid(True, alpha=0.25, linewidth=0.5)
    ax_r.legend(fontsize=9, frameon=False, loc="lower left")

    return _finish(
        fig, plt, result, Path(out_dir), "sim_vs_real",
        "Sim vs real agreement (EC3)",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _require_analysis(result: Phase5Result) -> None:
    """Raise a clear :class:`ValueError` if the analysis is missing."""
    if result.analysis is None:
        raise ValueError(
            "Phase5Result.analysis is None; the figures need the populated "
            "AnalysisReport for captions, the crossover, and the TMR effect. "
            "Run the WS2 analysis (slow_wave.eval.analysis) before generating figures."
        )


_BUILDERS = {
    "retention_curves": figure_retention_curves,
    "ablation_table": figure_ablation_table,
    "mechanism_pr": figure_mechanism_pr,
    "cost_pareto": figure_cost_pareto,
    "long_context_crossover": figure_long_context_crossover,
    "tmr_targeting": figure_tmr_targeting,
    "sim_vs_real": figure_sim_vs_real,
}


def generate_all_figures(result_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Render all seven Phase 5 figures from a committed result (EC4).

    Loads the :class:`Phase5Result` at ``result_path``, renders each of the seven
    deliverable figures as a vector PDF (+ a PNG sibling) into ``out_dir``, and
    writes ``figures_manifest.json`` mapping ``figure_key -> {pdf, png, caption}``.
    Every caption states n (seeds) and the CI method and carries the mock-LLM
    caveat (DX4/DX5).

    Args:
        result_path: Path to the committed ``phase5_result.json``.
        out_dir: Destination directory (created if needed).

    Returns:
        The written paths: the seven PDFs, their seven PNG siblings, and the
        ``figures_manifest.json`` path (the PDFs first, in :data:`FIGURES` order).

    Raises:
        ValueError: If the loaded result's ``analysis`` is ``None`` (the figures
            need it for captions, the crossover, and the TMR effect).
    """
    result = Phase5Result.model_validate_json(Path(result_path).read_text(encoding="utf-8"))
    _require_analysis(result)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs: list[Path] = []
    pngs: list[Path] = []
    manifest: dict[str, dict[str, str]] = {}
    for key, builder in _BUILDERS.items():
        pdf = builder(result, out_dir)
        png = pdf.with_suffix(".png")
        pdfs.append(pdf)
        pngs.append(png)
        manifest[key] = {
            "pdf": pdf.name,
            "png": png.name,
            "caption": _build_caption(result, key),
        }
        logger.info("rendered figure '%s' -> %s", key, pdf.name)

    manifest_path = out_dir / "figures_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote %d figures + manifest to %s", len(pdfs), out_dir)
    return [*pdfs, *pngs, manifest_path]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.paper.figures``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="slow-wave-figures",
        description="Regenerate the seven Phase 5 deliverable figures (EC4).",
    )
    parser.add_argument(
        "--result",
        default="phase5/phase5_result.json",
        help="Path to the committed phase5_result.json (with analysis filled).",
    )
    parser.add_argument(
        "--out",
        default="paper/figures",
        help="Output directory for the figures (default: paper/figures).",
    )
    args = parser.parse_args(argv)

    paths = generate_all_figures(args.result, args.out)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
