"""FR1.6 confound-guard firewall for the Phase 5 modules (regression robustness).

The confound guard is sacred (PHASE5_CONTRACT.md): only the ``oracle`` arm and
offline scoring may read ground-truth labels, and only via
``slow_wave.stream.schema.offline_labels``. The Phase 5 modules
(``eval/grid.py``, ``eval/analysis.py``, ``paper/figures.py``) must read labels
**only transitively** through the already-built ``ArmResult.prune_quality`` /
``ExperimentResult`` — never ``offline_labels`` / ``ground_truth`` /
``probe.answer`` directly.

Upstream (Phase 1-4) this is enforced with monkeypatch-spy tests; the three
Phase-5 modules had only architecture + manual review guarding them. This test
parses each module's AST and fails if any *executable* node imports, names, or
attribute-accesses a forbidden label symbol — so a future direct label read is
caught automatically. Docstrings/comments that merely *mention* the guard are
``ast.Constant`` nodes and are correctly ignored.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# Modules that must NOT read labels directly.
_GUARDED = [
    _REPO / "slow_wave" / "eval" / "grid.py",
    _REPO / "slow_wave" / "eval" / "analysis.py",
    _REPO / "slow_wave" / "paper" / "figures.py",
]

# Symbols whose direct use signals a ground-truth read.
_FORBIDDEN_IMPORT = {"offline_labels", "ground_truth"}
_FORBIDDEN_NAME = {"offline_labels", "ground_truth"}
_FORBIDDEN_ATTR = {"offline_labels", "ground_truth", "answer"}


def _label_reads(path: Path) -> list[str]:
    """Return executable label-read sites (empty == clean). Docstrings ignored."""
    tree = ast.parse(path.read_text("utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORT:
                    hits.append(f"L{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAME:
            hits.append(f"L{node.lineno}: name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTR:
            hits.append(f"L{node.lineno}: attr .{node.attr}")
    return hits


@pytest.mark.parametrize("path", _GUARDED, ids=lambda p: p.name)
def test_phase5_module_has_no_direct_label_read(path: Path) -> None:
    """No Phase-5 module imports/names/attribute-reads a ground-truth label (FR1.6)."""
    assert path.exists(), path
    hits = _label_reads(path)
    assert not hits, (
        f"{path.name} reads ground-truth labels directly (FR1.6 violation): {hits}. "
        "Phase 5 must read prune quality transitively via ArmResult.prune_quality."
    )


def test_firewall_detects_a_planted_violation(tmp_path: Path) -> None:
    """Meta-test: the detector is non-vacuous (a planted label read is caught)."""
    bad = tmp_path / "bad_module.py"
    bad.write_text(
        "from slow_wave.stream.schema import offline_labels\n"
        "def leak(probe):\n"
        "    return offline_labels(probe), probe.answer\n",
        encoding="utf-8",
    )
    hits = _label_reads(bad)
    assert any("import offline_labels" in h for h in hits)
    assert any("name offline_labels" in h for h in hits)
    assert any("attr .answer" in h for h in hits)
