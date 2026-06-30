"""Regenerate the paper's in-text numbers and results table from committed data.

Phase 6 (the scientific paper) requires that **every results number** in the
manuscript be regenerable from a committed manifest + script (PRD Phase 6 EC2;
DX1). This module is that script. It reads the committed Phase 5 artifacts
(``phase5/phase5_result.json`` with its embedded ``analysis`` block, plus the
per-regime experiment manifests under ``phase5/regime_*/manifest.json``) and
emits two LaTeX files the manuscript ``\\input``s:

* ``paper/generated/numbers.tex`` -- one ``\\newcommand`` per headline number
  (primary endpoint, TMR effect, power, sim-vs-real agreement, crossover, drift
  / stability controls, grid dimensions, model + git provenance). The body never
  hard-codes a results number; it cites ``\\SWprimaryDiff`` etc., so re-running
  this script after a re-run of the grid updates the paper mechanically.
* ``paper/generated/arm_metrics_table.tex`` -- a ``booktabs`` table of per-arm
  mean ACC / BWT / FWT / prune precision / recall / F1 / memory-vector count /
  total tokens in the primary regime, aggregated over the seed set.

Pure standard library (no numpy / matplotlib), so it imports and runs anywhere
the repo does and stays CI-safe. Deterministic given the committed inputs.

MOCK-LLM CAVEAT (DX5): the committed Phase 5 numbers were produced under the
deterministic mock LLM, so these macros describe a *mechanism demonstration in
the synthetic + mock-LLM regime*, never a claim about a real Claude model.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Display config                                                              #
# --------------------------------------------------------------------------- #

# Canonical arm display order + labels for the per-arm metrics table.
ARM_ORDER: list[str] = [
    "no_sleep",
    "replay_only",
    "downscale_only",
    "full_dream",
    "reflection",
    "random_pruning",
    "oracle",
    "long_context",
    "aa",
]
ARM_LABEL: dict[str, str] = {
    "no_sleep": "no-sleep",
    "replay_only": "replay-only",
    "downscale_only": "downscale-only",
    "full_dream": "full-dream",
    "reflection": "reflection",
    "random_pruning": "random-pruning",
    "oracle": "oracle",
    "long_context": "long-context",
    "aa": "A/A",
}

REGIME_DISPLAY: dict[str, str] = {
    "signal_rich": "signal-rich",
    "balanced": "balanced",
    "distractor_heavy": "distractor-heavy",
}


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #


def _f(x: Any, nd: int = 3) -> str:
    """Format a float to ``nd`` decimals; pass through ``--`` for ``None``."""
    if x is None:
        return "--"
    return f"{float(x):.{nd}f}"


def _signed(x: Any, nd: int = 3) -> str:
    """Format with an explicit leading sign (``+0.379`` / ``-0.875``)."""
    if x is None:
        return "--"
    return f"{float(x):+.{nd}f}"


def _pval(p: Any) -> str:
    """Format a p-value, collapsing the tiny tail to ``<0.001``."""
    if p is None:
        return "--"
    p = float(p)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


_LATEX_ESCAPES = {
    "%": "\\%",
    "&": "\\&",
    "_": "\\_",
    "#": "\\#",
    "$": "\\$",
    "<": "\\textless{}",
    ">": "\\textgreater{}",
}


def _latex_escape(s: str) -> str:
    """Escape the LaTeX-special characters that appear in our stat strings."""
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _regime_manifest_path(result_path: Path, regime: str) -> Path:
    """Locate a per-regime manifest relative to the result file's repo root."""
    # phase5_result.json lives at phase5/phase5_result.json; manifests are
    # siblings at phase5/regime_<name>/manifest.json. Resolve via the parent.
    base = result_path.parent
    return base / f"regime_{regime}" / "manifest.json"


# --------------------------------------------------------------------------- #
# Macro extraction                                                            #
# --------------------------------------------------------------------------- #


def build_macros(result: dict[str, Any], result_path: Path) -> dict[str, str]:
    """Return the ordered macro-name -> formatted-value map for numbers.tex."""
    an = result["analysis"]
    grid = result["grid"]
    sim = result["sim_real"]
    cross = an["crossover"]
    tmr = an["tmr"]
    power = an["power"]
    neg = an["negative"]

    n_seeds = int(an["n_seeds"])
    arms = grid["arms"]
    regimes = grid["regimes"]
    n_arms = len(arms)
    n_regimes = len(regimes)

    # Actual cell count from per-regime manifests (no silent caps: count what
    # was really run rather than assuming arms x seeds x regimes).
    actual_cells = 0
    for r in regimes:
        acc_by_arm = r.get("acc_by_arm", {})
        actual_cells += sum(len(v) for v in acc_by_arm.values())
    expected_cells = n_arms * n_regimes * n_seeds
    dropped = expected_cells - actual_cells

    per_regime = an["per_regime_verdicts"]

    # Secondary contrast string, e.g.
    # "full_dream > replay_only (delta=+0.371, 95% CI [0.312, 0.425], wilcoxon p=0.014)"
    sec = neg.get("secondary_contrasts", {}).get("full_dream vs replay_only", "")

    macros: dict[str, str] = {}

    # --- provenance --------------------------------------------------------- #
    macros["SWmodelID"] = str(result["model_id"])
    macros["SWmodelMocked"] = "true" if result["model_mocked"] else "false"
    macros["SWgitCommit"] = str(result["git_commit"])[:12]
    macros["SWgitCommitFull"] = str(result["git_commit"])
    macros["SWscenario"] = str(result["scenario"])
    macros["SWexperiment"] = str(result["experiment"])

    # --- grid dimensions ---------------------------------------------------- #
    macros["SWnSeeds"] = str(n_seeds)
    macros["SWnArms"] = str(n_arms)
    macros["SWnRegimes"] = str(n_regimes)
    macros["SWnCells"] = str(actual_cells)
    macros["SWnCellsExpected"] = str(expected_cells)
    macros["SWnDropped"] = str(dropped)
    macros["SWprimaryRegime"] = REGIME_DISPLAY.get(
        grid["primary_regime"], grid["primary_regime"]
    )

    # --- primary endpoint (EC5 of Phase 5 / headline) ----------------------- #
    # The endpoint name carries underscores; escape it since the body renders it
    # in text (e.g. inside \texttt{...}) where a raw "_" is a math-mode error.
    macros["SWprimaryEndpoint"] = _latex_escape(str(an["primary_endpoint_name"]))
    macros["SWprimaryDiff"] = _signed(an["primary_value"], 3)
    macros["SWprimaryCIlo"] = _f(an["primary_ci_lo"], 3)
    macros["SWprimaryCIhi"] = _f(an["primary_ci_hi"], 3)
    macros["SWprimaryCImethod"] = str(an["primary_ci_method"])
    macros["SWprimaryTest"] = str(an["primary_test_name"]).replace("_", " ")
    macros["SWprimaryP"] = _pval(an["primary_test_p"])
    macros["SWprimaryEffectName"] = str(an["primary_effect_name"]).replace("_", " ")
    macros["SWprimaryD"] = _f(an["primary_effect_value"], 2)
    macros["SWprimaryEffectMag"] = str(an["primary_effect_magnitude"])
    macros["SWprimaryVerdict"] = str(an["primary_verdict"])
    macros["SWnoiseFloor"] = _f(an["noise_floor"], 4)
    macros["SWexceedsNoiseFloor"] = "true" if an["exceeds_noise_floor"] else "false"

    # --- per-regime primary diffs + verdicts -------------------------------- #
    for r in regimes:
        mp = r.get("manifest_path", "")
        regime = (
            mp.split("regime_", 1)[1].split("/", 1)[0] if "regime_" in mp else ""
        )
        if not regime:
            continue
        camel = "".join(p.capitalize() for p in regime.split("_"))
        macros[f"SWdiff{camel}"] = _signed(r["primary_value"], 3)
        macros[f"SWciLo{camel}"] = _f(r["primary_ci_lo"], 3)
        macros[f"SWciHi{camel}"] = _f(r["primary_ci_hi"], 3)
        macros[f"SWverdict{camel}"] = str(per_regime.get(regime, r["primary_verdict"]))

    # --- secondary contrast (full_dream vs replay_only) --------------------- #
    # Keep a fully-escaped descriptive macro AND parse clean numeric macros out
    # of the committed analysis string (format fixed by slow_wave.eval.analysis).
    macros["SWsecondaryContrast"] = _latex_escape(sec)
    m_delta = re.search(r"delta=([+-]?[0-9.]+)", sec)
    m_ci = re.search(r"CI \[([0-9.]+), ([0-9.]+)\]", sec)
    m_p = re.search(r"p=([0-9.]+)", sec)
    if m_delta:
        macros["SWdreamVsReplayDelta"] = _signed(float(m_delta.group(1)), 3)
    if m_ci:
        macros["SWdreamVsReplayCIlo"] = _f(float(m_ci.group(1)), 3)
        macros["SWdreamVsReplayCIhi"] = _f(float(m_ci.group(2)), 3)
    if m_p:
        macros["SWdreamVsReplayP"] = _pval(float(m_p.group(1)))

    # --- TMR-style targeting ------------------------------------------------ #
    macros["SWtmrG"] = _f(tmr["hedges_g"], 2)
    macros["SWtmrCIlo"] = _f(tmr["g_ci_lo"], 2)
    macros["SWtmrCIhi"] = _f(tmr["g_ci_hi"], 2)
    macros["SWtmrBench"] = _f(tmr["benchmark_g"], 2)
    macros["SWtmrLift"] = _signed(tmr["mean_lift"], 3)
    macros["SWtmrExceeds"] = "true" if tmr["exceeds_benchmark"] else "false"

    # --- power -------------------------------------------------------------- #
    macros["SWpowerFloor"] = str(power["floor"])
    macros["SWpowerN"] = str(power["n_seeds"])
    macros["SWpowerObsD"] = _f(power["observed_effect_d"], 2)
    macros["SWpowerReqN"] = str(power["required_n_for_observed"])
    macros["SWpowered"] = "true" if power["powered_for_observed"] else "false"

    # --- sim-vs-real agreement ---------------------------------------------- #
    macros["SWsimPearson"] = _f(sim["pearson_agreement"], 3)
    macros["SWsimSpearman"] = _f(sim["spearman_agreement"], 2)
    macros["SWsimMaxDiv"] = _f(sim["max_abs_acc_divergence"], 3)
    macros["SWsimInversions"] = str(len(sim.get("inversions", [])))
    macros["SWsimCompression"] = _f(sim["sim_compression"], 0)
    macros["SWsimNtasksSim"] = str(sim["sim_n_tasks"])
    macros["SWsimNtasksReal"] = str(sim["real_n_tasks"])
    macros["SWsimRankPreserved"] = "true" if sim["ranking_preserved"] else "false"

    # --- long-context crossover (EC6 of Phase 5) ---------------------------- #
    lengths = cross["lengths"]
    macros["SWcrossoverLengths"] = ", ".join(str(x) for x in lengths)
    macros["SWcrossoverLmin"] = str(min(lengths))
    macros["SWcrossoverLmax"] = str(max(lengths))
    macros["SWcrossoverFound"] = "true" if cross["crossover_found"] else "false"
    macros["SWcrossoverMetric"] = str(cross["metric"]).replace("_", " ")
    macros["SWcrossoverBaseline"] = ARM_LABEL.get(
        cross["baseline_arm"], cross["baseline_arm"]
    )
    macros["SWcrossoverTreatment"] = ARM_LABEL.get(
        cross["treatment_arm"], cross["treatment_arm"]
    )

    # --- bias controls: drift + temperature-0 stability --------------------- #
    # Read from the primary-regime manifest (per-cell, where these live).
    primary_regime = grid["primary_regime"]
    man_path = _regime_manifest_path(result_path, primary_regime)
    if man_path.exists():
        man = _load_json(man_path)
        exp = man["results"]["experiment"]
        drift = exp.get("drift", {})
        stab = exp.get("stability", {})
        macros["SWdriftFaithfulness"] = _f(drift.get("faithfulness"), 3)
        macros["SWdriftThreshold"] = _f(drift.get("drift_threshold"), 2)
        macros["SWdriftDegraded"] = "true" if drift.get("degraded") else "false"
        macros["SWdriftRounds"] = str(drift.get("n_rounds", ""))
        macros["SWstabilityCV"] = _f(stab.get("token_cv"), 3)
        macros["SWstabilityIdentical"] = "true" if stab.get("identical") else "false"
        macros["SWstabilityRepeats"] = str(stab.get("n_repeats", ""))

    return macros


def render_numbers_tex(macros: dict[str, str]) -> str:
    """Render the macro map as a ``\\newcommand`` block."""
    lines = [
        "% AUTO-GENERATED by slow_wave.paper.numbers -- DO NOT EDIT BY HAND.",
        "% Regenerate with:  python -m slow_wave.paper.numbers",
        "% Source: committed phase5/phase5_result.json + phase5/regime_*/manifest.json",
        "% MOCK-LLM mechanism-demonstration regime (DX5): not a real-model claim.",
        "",
    ]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Per-arm metrics table                                                       #
# --------------------------------------------------------------------------- #


def build_arm_table(result: dict[str, Any], result_path: Path) -> str:
    """Render the per-arm metrics ``booktabs`` table for the primary regime."""
    grid = result["grid"]
    primary_regime = grid["primary_regime"]
    man_path = _regime_manifest_path(result_path, primary_regime)
    man = _load_json(man_path)
    exp = man["results"]["experiment"]
    arm_results = exp["arm_results"]
    acc_by_arm = man["deterministic_probe"]["acc_by_arm"]

    # Aggregate per arm across seeds.
    agg: dict[str, dict[str, list[float]]] = {}
    for ar in arm_results:
        arm = ar["arm_name"]
        bucket = agg.setdefault(
            arm,
            {"bwt": [], "fwt": [], "prec": [], "rec": [], "f1": [], "vec": [], "tok": []},
        )
        cm = ar.get("continual_metrics", {})
        pq = ar.get("prune_quality", {})
        cost = ar.get("cost", {})
        bucket["bwt"].append(cm.get("bwt"))
        bucket["fwt"].append(cm.get("fwt"))
        bucket["prec"].append(pq.get("precision"))
        bucket["rec"].append(pq.get("recall"))
        bucket["f1"].append(pq.get("f1"))
        bucket["vec"].append(cost.get("memory_vectors"))
        bucket["tok"].append(cost.get("total_tokens"))

    header = (
        "% AUTO-GENERATED by slow_wave.paper.numbers -- DO NOT EDIT BY HAND.\n"
        "% Per-arm metrics, primary regime, mean over seeds. Regenerate with:\n"
        "%   python -m slow_wave.paper.numbers\n"
        "\\begin{tabular}{l rrr rrr rr}\n"
        "\\toprule\n"
        "Arm & ACC & BWT & FWT & Prec. & Rec. & F1 & Vec. & kTok \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for arm in ARM_ORDER:
        if arm not in agg and arm not in acc_by_arm:
            continue
        b = agg.get(arm, {})
        acc = _mean(acc_by_arm.get(arm, []))
        vec = _mean(b.get("vec", []))
        tok = _mean(b.get("tok", []))
        ktok = f"{tok / 1000.0:.1f}" if tok is not None else "--"
        rows.append(
            f"{ARM_LABEL.get(arm, arm)} & "
            f"{_f(acc, 3)} & {_signed(_mean(b.get('bwt', [])), 3)} & "
            f"{_signed(_mean(b.get('fwt', [])), 3)} & "
            f"{_f(_mean(b.get('prec', [])), 3)} & {_f(_mean(b.get('rec', [])), 3)} & "
            f"{_f(_mean(b.get('f1', [])), 3)} & "
            f"{_f(vec, 0)} & {ktok} \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(rows) + "\n" + footer


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def generate_all_numbers(result_path: str, out_dir: str) -> list[Path]:
    """Write numbers.tex + arm_metrics_table.tex; return the written paths."""
    rp = Path(result_path)
    result = _load_json(rp)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    macros = build_macros(result, rp)
    numbers_path = out / "numbers.tex"
    numbers_path.write_text(render_numbers_tex(macros), encoding="utf-8")

    table_path = out / "arm_metrics_table.tex"
    table_path.write_text(build_arm_table(result, rp), encoding="utf-8")

    return [numbers_path, table_path]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m slow_wave.paper.numbers``."""
    parser = argparse.ArgumentParser(
        prog="slow-wave-numbers",
        description=(
            "Regenerate the paper's in-text number macros + per-arm results "
            "table from the committed Phase 5 result (EC2)."
        ),
    )
    parser.add_argument(
        "--result",
        default="phase5/phase5_result.json",
        help="Path to the committed phase5_result.json (with analysis filled).",
    )
    parser.add_argument(
        "--out",
        default="paper/generated",
        help="Output directory for the .tex files (default: paper/generated).",
    )
    args = parser.parse_args(argv)

    for path in generate_all_numbers(args.result, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
