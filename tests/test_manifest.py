"""Schema test for the run manifest (Phase 0 exit criterion #2).

This is the artifact a red-team inspects: it explicitly enumerates every
PRD §5.6 / FR6.1 field as a dotted JSON path, builds a real manifest via
``new_manifest`` with small fakes, and asserts each required field is present
and correctly typed in the serialized manifest. It also checks the
write/read round-trip and that nondeterministic fields are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from slow_wave.config import load_config
from slow_wave.repro.manifest import (
    Manifest,
    new_manifest,
    read_manifest,
    write_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_CONFIG = _REPO_ROOT / "configs" / "smoke.yaml"


# --- FR6.1 checklist -------------------------------------------------------
# Each entry maps an FR6.1 requirement to a concrete dotted path in the
# serialized manifest, the accepted Python type(s), and whether None is a
# legal value for that field. A red-team can read this table top-to-bottom and
# confirm the manifest carries every field the PRD demands.
#
#   (dotted_path, accepted_types, none_ok)
FR61_REQUIRED_PATHS = [
    # exact model id + sampling params
    ("model.id", (str,), False),
    ("model.sampling.temperature", (float, int), False),
    ("model.sampling.max_tokens", (int,), False),
    ("model.sampling.top_p", (float, int), True),
    ("model.sampling.effort", (str,), True),
    ("model.mocked", (bool,), False),
    # embedding model + version + dim (+ backend)
    ("embedding.model", (str,), False),
    ("embedding.version", (str,), False),
    ("embedding.dim", (int,), False),
    ("embedding.backend", (str,), False),
    # all hyperparameters + search ranges
    ("hyperparameters", (dict,), False),
    ("search_ranges", (dict,), False),
    # seed list (+ named seed map)
    ("seed_list", (list,), False),
    ("seeds", (dict,), False),
    # git commit hash
    ("git.commit", (str,), True),
    # wall-clock + token/compute cost
    ("cost.wall_clock_s", (float, int), False),
    ("cost.tokens.input", (int,), False),
    ("cost.tokens.output", (int,), False),
    ("cost.tokens.total", (int,), False),
    ("cost.api_calls", (int,), False),
    # sim-time compression factor
    ("sim_time.compression_factor", (float, int), False),
]


# --- fakes -----------------------------------------------------------------
@dataclass
class FakeEmbedder:
    """Stand-in for the WS4 embedder duck-type (.backend/.model/.version/.dim)."""

    backend: str = "hash"
    model: str = "hash-bow-v1"
    version: str = "1.0"
    dim: int = 384


@dataclass
class FakeLLM:
    """Stand-in for the WS4 LLMResult duck-type."""

    model_id: str = "claude-opus-4-8"
    input_tokens: int = 17
    output_tokens: int = 23
    mocked: bool = True
    text: str = "[MOCK:deadbeef] Memory consolidation is the offline stabilization of traces."
    stop_reason: str | None = "end_turn"


_MISSING = object()


def _get_path(data: dict, dotted: str):
    """Walk a dotted path through nested dicts; return _MISSING if absent."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _build_manifest() -> Manifest:
    cfg = load_config(_SMOKE_CONFIG)
    seeds = {"master": cfg.seed, "agent": 111, "stream": 222}
    return new_manifest(
        cfg=cfg,
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        seeds=seeds,
        deterministic_probe={
            "embedding_sha256": "0" * 64,
            "sampling_order": [3, 1, 0, 2],
            "n_items": 8,
        },
        wall_clock_s=0.125,
        # Pass git explicitly so the test is hermetic (no dependency on git
        # being installed in CI); the field is still exercised and present.
        git={"commit": "abc1234", "dirty": False, "branch": "main"},
    )


def test_manifest_has_every_fr61_field() -> None:
    """Every FR6.1 path is present in the dumped manifest and correctly typed."""
    dumped = _build_manifest().model_dump(mode="json")

    missing = []
    bad_type = []
    for path, types, none_ok in FR61_REQUIRED_PATHS:
        value = _get_path(dumped, path)
        if value is _MISSING:
            missing.append(path)
            continue
        if value is None:
            if not none_ok:
                bad_type.append((path, "None not allowed"))
            continue
        if not isinstance(value, types):
            bad_type.append((path, f"{type(value).__name__} not in {types}"))

    assert not missing, f"Manifest is missing FR6.1 fields: {missing}"
    assert not bad_type, f"Manifest FR6.1 fields have wrong types: {bad_type}"


def test_seed_list_is_list_of_ints() -> None:
    """seed_list is a sorted, de-duplicated list of ints; seeds is a dict[str,int]."""
    dumped = _build_manifest().model_dump(mode="json")

    seed_list = dumped["seed_list"]
    assert isinstance(seed_list, list)
    assert seed_list == sorted(set(seed_list)), "seed_list must be sorted + de-duplicated"
    assert all(isinstance(s, int) for s in seed_list)

    seeds = dumped["seeds"]
    assert isinstance(seeds, dict)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in seeds.items())
    # seed_list must be exactly the distinct seed values.
    assert set(seed_list) == set(seeds.values())


def test_cost_tokens_total_is_consistent() -> None:
    """cost.tokens.total equals input + output."""
    dumped = _build_manifest().model_dump(mode="json")
    tokens = dumped["cost"]["tokens"]
    assert tokens["total"] == tokens["input"] + tokens["output"]


def test_meta_fields_present() -> None:
    """Run metadata (manifest_version, run_id, created_at, hashes, env) is present."""
    dumped = _build_manifest().model_dump(mode="json")
    for key in (
        "manifest_version",
        "run_id",
        "experiment",
        "created_at",
        "config_hash",
        "package_version",
        "python_version",
        "platform",
        "deterministic_probe",
        "llm",
        "nondeterministic_fields",
    ):
        assert key in dumped, f"missing meta field: {key}"
    assert dumped["manifest_version"] == "1.0"
    assert dumped["experiment"] == "hello-bench-smoke"
    # run_id is reproducible: derived from experiment + config hash, no randomness.
    assert dumped["run_id"].startswith("hello-bench-smoke-")


def test_nondeterministic_fields_flagged() -> None:
    """The manifest flags created_at, wall-clock and at least one llm.* path."""
    dumped = _build_manifest().model_dump(mode="json")
    flagged = dumped["nondeterministic_fields"]
    assert isinstance(flagged, list)
    assert "created_at" in flagged
    assert "cost.wall_clock_s" in flagged
    assert any(p.startswith("llm.") for p in flagged), flagged


def test_run_id_is_reproducible() -> None:
    """Two manifests from the same config get the same derived run_id."""
    assert _build_manifest().run_id == _build_manifest().run_id


def test_write_read_round_trip(tmp_path) -> None:
    """write_manifest then read_manifest reproduces the manifest exactly."""
    manifest = _build_manifest()
    out = write_manifest(manifest, tmp_path / "nested" / "manifest.json")
    assert out.exists()
    assert out.read_text(encoding="utf-8").endswith("\n")  # trailing newline

    loaded = read_manifest(out)
    assert loaded.model_dump(mode="json") == manifest.model_dump(mode="json")


def test_write_manifest_is_sorted_and_indented(tmp_path) -> None:
    """Serialized JSON is deterministic: indent=2 and sort_keys=True."""
    out = write_manifest(_build_manifest(), tmp_path / "manifest.json")
    text = out.read_text(encoding="utf-8")
    # sort_keys=True: top-level keys appear in alphabetical order.
    assert text.index('"config_hash"') < text.index('"run_id"')
    # indent=2: nested values are indented.
    assert "\n  " in text


def test_git_passthrough_used() -> None:
    """An explicitly provided git dict flows into git.* fields."""
    dumped = _build_manifest().model_dump(mode="json")
    assert dumped["git"]["commit"] == "abc1234"
    assert dumped["git"]["dirty"] is False
    assert dumped["git"]["branch"] == "main"


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    raise SystemExit(pytest.main([__file__, "-v"]))
