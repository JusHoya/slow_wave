"""The Phase 0 determinism test (exit criterion #3).

Two smoke runs with the same config + seeds must produce identical non-LLM
outputs: the embedding hash, the sampling order, the item count, and the file
layout. LLM-dependent fields must be flagged in ``nondeterministic_fields``. The
LLM is forced onto its deterministic mock path by removing the API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slow_wave.config import load_config
from slow_wave.repro.smoke import run_smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_YAML = REPO_ROOT / "configs" / "smoke.yaml"


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _relative_files(root: Path) -> list[str]:
    """Sorted POSIX-style relative paths of every file under ``root``."""
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def test_two_smoke_runs_are_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same config + seeds -> identical deterministic probe and file layout."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # force mock LLM
    cfg = load_config(SMOKE_YAML)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    path_a = run_smoke(cfg, out_a)
    path_b = run_smoke(cfg, out_b)

    man_a = _read_json(path_a)
    man_b = _read_json(path_b)

    probe_a = man_a["deterministic_probe"]
    probe_b = man_b["deterministic_probe"]

    # Non-LLM outputs are identical.
    assert probe_a["embedding_sha256"] == probe_b["embedding_sha256"]
    assert probe_a["sampling_order"] == probe_b["sampling_order"]
    assert probe_a["n_items"] == probe_b["n_items"]

    # File layout under each run dir matches.
    assert _relative_files(out_a) == _relative_files(out_b)

    # LLM-dependent fields are flagged as nondeterministic. The manifest may
    # flag the whole "llm" block or granular "llm.*" / "cost.tokens.output"
    # paths; require at least one LLM-dependent entry to be present.
    nd = man_a["nondeterministic_fields"]
    assert isinstance(nd, list) and nd
    llm_flagged = [f for f in nd if f == "llm" or f.startswith("llm.")]
    assert llm_flagged, f"expected LLM-dependent fields flagged, got {nd}"

    # This no-key run used the deterministic mock.
    assert man_a["model"]["mocked"] is True
    assert man_b["model"]["mocked"] is True
