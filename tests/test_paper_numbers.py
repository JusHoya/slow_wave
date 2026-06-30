"""Golden-artifact + contract tests for slow_wave.paper.numbers (Phase 6, EC2).

The manuscript's in-text results numbers and the per-arm results table are not
hand-typed: they are generated from the committed Phase 5 result by
``slow_wave.paper.numbers`` and ``\\input`` into the LaTeX. This module pins that
generator so a future change that silently moves a shipped paper number (or
breaks the LaTeX-escaping of an identifier macro) is caught in CI.

The generator reads the committed ``phase5/phase5_result.json`` plus the
per-regime manifests under ``phase5/regime_*/manifest.json`` (the per-arm table
needs the manifest's ``arm_results``). The tests self-skip if those committed
artifacts are absent (sparse checkout), mirroring
``test_phase5_committed_artifacts``.

Every number here is the deterministic mock-LLM mechanism-demonstration value
(DX5 -- not a claim about a real Claude model).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slow_wave.paper import numbers

_REPO = Path(__file__).resolve().parents[1]
_RESULT = _REPO / "phase5" / "phase5_result.json"
_PRIMARY_MANIFEST = _REPO / "phase5" / "regime_distractor_heavy" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not (_RESULT.exists() and _PRIMARY_MANIFEST.exists()),
    reason="committed phase5 artifacts not present in this checkout",
)


def test_module_imports_without_matplotlib() -> None:
    """numbers.py is pure-stdlib; importing it must not require matplotlib."""
    assert hasattr(numbers, "generate_all_numbers")
    assert hasattr(numbers, "build_macros")
    assert hasattr(numbers, "build_arm_table")


def test_generate_writes_both_files(tmp_path: Path) -> None:
    """generate_all_numbers writes numbers.tex + arm_metrics_table.tex, non-empty."""
    written = numbers.generate_all_numbers(str(_RESULT), str(tmp_path))
    names = {p.name for p in written}
    assert names == {"numbers.tex", "arm_metrics_table.tex"}
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_numbers_tex_golden_macros(tmp_path: Path) -> None:
    """The shipped headline numbers appear as the expected \\newcommand macros."""
    numbers.generate_all_numbers(str(_RESULT), str(tmp_path))
    tex = (tmp_path / "numbers.tex").read_text(encoding="utf-8")
    # Golden values (mock-LLM mechanism demo; mirror test_phase5_committed_artifacts).
    expected = {
        r"\newcommand{\SWnSeeds}{8}",
        r"\newcommand{\SWnArms}{9}",
        r"\newcommand{\SWnRegimes}{3}",
        r"\newcommand{\SWnCells}{216}",
        r"\newcommand{\SWnDropped}{0}",
        r"\newcommand{\SWprimaryDiff}{+0.379}",
        r"\newcommand{\SWprimaryCIlo}{0.329}",
        r"\newcommand{\SWprimaryCIhi}{0.429}",
        r"\newcommand{\SWprimaryD}{4.89}",
        r"\newcommand{\SWprimaryVerdict}{confirmed}",
        r"\newcommand{\SWnoiseFloor}{0.0625}",
        r"\newcommand{\SWtmrG}{1.70}",
        r"\newcommand{\SWtmrBench}{0.29}",
        r"\newcommand{\SWcrossoverFound}{false}",
        r"\newcommand{\SWsimSpearman}{1.00}",
    }
    missing = {m for m in expected if m not in tex}
    assert not missing, f"missing/changed golden macros: {missing}"


def test_endpoint_macro_is_latex_safe(tmp_path: Path) -> None:
    """The endpoint identifier macro must escape underscores (raw '_' breaks text)."""
    numbers.generate_all_numbers(str(_RESULT), str(tmp_path))
    tex = (tmp_path / "numbers.tex").read_text(encoding="utf-8")
    line = next(ln for ln in tex.splitlines() if "SWprimaryEndpoint" in ln)
    assert "acc\\_diff" in line, line  # escaped, not a raw underscore
    assert "acc_diff" not in line.replace("\\_", "")  # no surviving raw underscore


def test_secondary_contrast_has_no_raw_percent(tmp_path: Path) -> None:
    """The secondary-contrast macro escapes '%' (a raw '%' comments out the line)."""
    numbers.generate_all_numbers(str(_RESULT), str(tmp_path))
    tex = (tmp_path / "numbers.tex").read_text(encoding="utf-8")
    line = next(ln for ln in tex.splitlines() if "SWsecondaryContrast" in ln)
    # Every '%' in the value must be backslash-escaped.
    idx = 0
    while (idx := line.find("%", idx)) != -1:
        assert line[idx - 1] == "\\", f"unescaped %% in: {line}"
        idx += 1


def test_arm_table_has_all_nine_arms(tmp_path: Path) -> None:
    """The per-arm table is a booktabs table with a row for every arm."""
    numbers.generate_all_numbers(str(_RESULT), str(tmp_path))
    tbl = (tmp_path / "arm_metrics_table.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tbl and "\\bottomrule" in tbl
    assert "\\begin{tabular}" in tbl and "\\end{tabular}" in tbl
    for label in [
        "no-sleep", "replay-only", "downscale-only", "full-dream", "reflection",
        "random-pruning", "oracle", "long-context", "A/A",
    ]:
        assert label in tbl, f"arm row missing from table: {label}"
    # Ceiling sanity: long-context retains everything (no pruning) so prune cols are 0.000.
    lc_row = next(ln for ln in tbl.splitlines() if ln.startswith("long-context"))
    assert "1.000" in lc_row  # ACC ceiling
