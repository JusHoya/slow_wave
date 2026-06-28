"""Run-manifest schema and I/O for the Slow Wave bench (Phase 0).

A *manifest* is the JSON provenance record emitted by every run. It captures
**every** field required by PRD §5.6 / FR6.1 so that a result can be audited
and reproduced from a clean checkout:

* exact model id and sampling params (``model.id``, ``model.sampling.*``);
* embedding model + version + dim + backend (``embedding.*``);
* all hyperparameters and search ranges (``hyperparameters``, ``search_ranges``);
* the seed list and the named seed map (``seed_list``, ``seeds``);
* the git commit hash (``git.commit``) and dirty/branch state;
* wall-clock and token/compute cost (``cost.wall_clock_s``, ``cost.tokens.*``,
  ``cost.api_calls``);
* the sim-time compression factor (``sim_time.compression_factor``).

Plus run metadata (``manifest_version``, ``run_id``, ``experiment``,
``created_at``, ``config_hash``, ``package_version``, ``python_version``,
``platform``), a ``deterministic_probe`` block (the values two runs must match),
an ``llm`` block (LLM call provenance), and an explicit
``nondeterministic_fields`` list flagging which paths may legitimately vary
run-to-run.

The canonical field<->FR6.1 mapping is enumerated and asserted by
``tests/test_manifest.py`` (the schema test that satisfies Phase 0 exit
criterion #2).
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

import slow_wave
from slow_wave.repro import gitinfo

MANIFEST_VERSION = "1.0"


class SamplingParams(BaseModel):
    """LLM sampling parameters recorded for the run."""

    temperature: float
    max_tokens: int
    top_p: float | None = None
    effort: str | None = None


class ModelManifest(BaseModel):
    """The exact model id, its sampling params, and whether it was mocked."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    sampling: SamplingParams
    mocked: bool


class EmbeddingManifest(BaseModel):
    """The embedding model identity: name, version, dimension, and backend."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    version: str
    dim: int
    backend: str


class TokenCost(BaseModel):
    """Token accounting for the run's LLM call(s)."""

    input: int
    output: int
    total: int


class CostInfo(BaseModel):
    """Wall-clock, token, and API-call cost of the run."""

    wall_clock_s: float
    tokens: TokenCost
    api_calls: int


class GitManifest(BaseModel):
    """Git provenance: commit hash, dirty state, and branch."""

    commit: str | None = None
    dirty: bool | None = None
    branch: str | None = None


class SimTimeManifest(BaseModel):
    """Sim-time configuration: the sim-time / wall-time compression factor."""

    compression_factor: float


class Manifest(BaseModel):
    """The complete run manifest (every FR6.1 field + run metadata)."""

    model_config = ConfigDict(protected_namespaces=())

    manifest_version: str = MANIFEST_VERSION
    run_id: str
    experiment: str
    created_at: str  # ISO8601 UTC
    config_hash: str
    package_version: str
    python_version: str
    platform: str
    model: ModelManifest
    embedding: EmbeddingManifest
    seeds: dict[str, int]
    seed_list: list[int]
    hyperparameters: dict = Field(default_factory=dict)
    search_ranges: dict = Field(default_factory=dict)
    git: GitManifest
    cost: CostInfo
    sim_time: SimTimeManifest
    deterministic_probe: dict = Field(default_factory=dict)
    llm: dict = Field(default_factory=dict)
    nondeterministic_fields: list[str] = Field(default_factory=list)


# Fields that may legitimately differ between two runs of the same config+seeds.
# When the LLM is mocked these ``llm.*``/``cost`` values are in fact
# deterministic, but they are still listed because they vary once a real
# (non-mocked) Claude call is made. The determinism check (exit criterion #3)
# compares only ``deterministic_probe`` and the set of output files.
NONDETERMINISTIC_FIELDS: list[str] = [
    "created_at",
    "cost.wall_clock_s",
    "cost.tokens.output",
    "cost.tokens.total",
    "llm.text_sha256",
    "llm.text_preview",
    "llm.output_tokens",
    "git.commit",
    "git.dirty",
]


def new_manifest(
    *,
    cfg,
    embedder,
    llm,
    seeds: dict[str, int],
    deterministic_probe: dict,
    wall_clock_s: float,
    git: dict | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Manifest:
    """Assemble a :class:`Manifest` from a config, embedder, LLM result, etc.

    Args:
        cfg: A loaded ``Config`` exposing ``model.{id,temperature,max_tokens,
            top_p,effort}``, ``embedding.*``, ``sim_time.compression_factor``,
            ``hyperparameters``, ``search_ranges``, ``experiment`` and
            ``content_hash()``.
        embedder: An embedder duck-type exposing ``.backend``, ``.model``,
            ``.version`` and ``.dim``.
        llm: An LLM result exposing ``.model_id``, ``.input_tokens``,
            ``.output_tokens``, ``.mocked``, ``.text`` and (optionally)
            ``.stop_reason``.
        seeds: Mapping of named seeds (e.g. ``{"master", "agent", "stream"}``).
        deterministic_probe: The probe block that two runs of the same
            config+seeds must reproduce exactly.
        wall_clock_s: Measured wall-clock duration of the run, in seconds.
        git: Optional pre-collected git info dict; if ``None``,
            :func:`slow_wave.repro.gitinfo.git_info` is called.
        run_id: Optional explicit run id; if ``None``, a reproducible id is
            derived from ``cfg.experiment`` + the config content hash.
        created_at: Optional explicit ISO8601 UTC timestamp; if ``None``, the
            current UTC time is used.

    Returns:
        A fully-populated :class:`Manifest`.
    """

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    if run_id is None:
        # Reproducible given config (and, transitively, seeds baked into config):
        # no randomness or wall-clock involved.
        run_id = f"{cfg.experiment}-{cfg.content_hash()[:8]}"

    if git is None:
        git = gitinfo.git_info()

    sampling = SamplingParams(
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        top_p=cfg.model.top_p,
        effort=cfg.model.effort,
    )
    model = ModelManifest(id=cfg.model.id, sampling=sampling, mocked=llm.mocked)

    embedding = EmbeddingManifest(
        model=embedder.model,
        version=embedder.version,
        dim=embedder.dim,
        backend=embedder.backend,
    )

    input_tokens = llm.input_tokens
    output_tokens = llm.output_tokens
    tokens = TokenCost(
        input=input_tokens,
        output=output_tokens,
        total=input_tokens + output_tokens,
    )
    cost = CostInfo(wall_clock_s=wall_clock_s, tokens=tokens, api_calls=1)

    llm_block = {
        "model_id": llm.model_id,
        "mocked": llm.mocked,
        "text_sha256": hashlib.sha256(llm.text.encode("utf-8")).hexdigest(),
        "text_preview": llm.text[:120],
        "output_tokens": llm.output_tokens,
        "stop_reason": getattr(llm, "stop_reason", None),
    }

    git_manifest = GitManifest(
        commit=git.get("commit"),
        dirty=git.get("dirty"),
        branch=git.get("branch"),
    )

    sim_time = SimTimeManifest(
        compression_factor=cfg.sim_time.compression_factor,
    )

    return Manifest(
        manifest_version=MANIFEST_VERSION,
        run_id=run_id,
        experiment=cfg.experiment,
        created_at=created_at,
        config_hash=cfg.content_hash(),
        package_version=slow_wave.__version__,
        python_version=_platform.python_version(),
        platform=_platform.platform(),
        model=model,
        embedding=embedding,
        seeds=dict(seeds),
        seed_list=sorted(set(seeds.values())),
        hyperparameters=dict(cfg.hyperparameters),
        search_ranges=dict(cfg.search_ranges),
        git=git_manifest,
        cost=cost,
        sim_time=sim_time,
        deterministic_probe=dict(deterministic_probe),
        llm=llm_block,
        nondeterministic_fields=list(NONDETERMINISTIC_FIELDS),
    )


def write_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Write ``manifest`` to ``path`` as deterministic, pretty JSON.

    Parent directories are created if needed. The JSON is serialized with
    ``indent=2`` and ``sort_keys=True`` (plus a trailing newline) so it is
    stable and diff-friendly across runs.

    Args:
        manifest: The manifest to serialize.
        path: Destination file path.

    Returns:
        The :class:`~pathlib.Path` the manifest was written to.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    return out_path


def read_manifest(path: str | Path) -> Manifest:
    """Load and validate a :class:`Manifest` from a JSON file.

    Args:
        path: Path to a manifest JSON file.

    Returns:
        The validated :class:`Manifest`.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Manifest.model_validate(data)
