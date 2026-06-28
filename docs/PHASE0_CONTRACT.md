# Phase 0 — Shared Interface Contract (authoritative)

This file pins the cross-module interfaces for Phase 0 so independently-authored
modules integrate without guessing. **Do not deviate from these signatures.**

- Python: target **3.12** (must support 3.11+). Package name: **`slow_wave`** (already created, `__version__ = "0.1.0"`).
- Pinned model ids: agent/dream default **`claude-opus-4-8`** (Opus 4.8); bulk **`claude-haiku-4-5`**; fallback **`claude-sonnet-4-6`**.
- Default real embedding model: **`BAAI/bge-small-en-v1.5`** (dim 384). Smoke default backend: **`hash`** (dependency-free, deterministic).
- Manifest version: **`"1.0"`**.
- pydantic v2. For any model that has a field literally named `model`, set
  `model_config = ConfigDict(protected_namespaces=())` to silence the protected-namespace warning.

## `slow_wave/config.py`  (WS2 owns)

```python
class ModelConfig(BaseModel):
    id: str = "claude-opus-4-8"
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float | None = None
    effort: str | None = None            # adaptive-thinking effort knob, e.g. "low"/"medium"/"high"

class EmbeddingConfig(BaseModel):        # set protected_namespaces=() (has `model` field)
    backend: Literal["hash", "sentence-transformers"] = "hash"
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384

class SimTimeConfig(BaseModel):
    compression_factor: float = 1.0      # sim-time / wall-time

class SmokeConfig(BaseModel):
    prompt: str = "In one sentence, state what memory consolidation is."
    n_items: int = 8
    texts: list[str] | None = None       # if None, smoke generates deterministic corpus from n_items

class Config(BaseModel):                 # extra="forbid"; protected_namespaces=()
    experiment: str
    description: str = ""
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sim_time: SimTimeConfig = Field(default_factory=SimTimeConfig)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    search_ranges: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "runs"

    def content_hash(self) -> str: ...   # sha256 of canonical json (model_dump, sort_keys), stable across runs

def load_config(path: str | Path) -> Config: ...   # read YAML -> validate -> Config
```

## `slow_wave/repro/seeding.py`  (WS3 owns)

```python
def set_global_seeds(seed: int) -> None:          # seeds python `random` and numpy global RNG
def derive_seed(master: int, name: str) -> int:   # deterministic, stable: blake2b(f"{master}:{name}") -> uint32
```

## `slow_wave/repro/gitinfo.py`  (WS3 owns)

```python
def git_commit_hash(short: bool = False) -> str | None   # None if not a git repo / git missing
def git_is_dirty() -> bool | None
def git_info() -> dict     # {"commit": str|None, "dirty": bool|None, "branch": str|None}
```

## `slow_wave/repro/manifest.py`  (WS3 owns) — must contain EVERY FR6.1 field

```python
MANIFEST_VERSION = "1.0"

# Nested models -> Manifest. Required JSON paths (FR6.1 checklist the schema test asserts):
#   model.id                              (exact model id)
#   model.sampling.{temperature,max_tokens,top_p,effort}   (sampling params)
#   model.mocked                          (whether the LLM call was mocked)
#   embedding.{model,version,dim,backend} (embedding model + version + dim)
#   hyperparameters                       (all hyperparameters)
#   search_ranges                         (search ranges)
#   seed_list  (list[int])  AND  seeds (dict {master,agent,stream})
#   git.commit                            (git commit hash)
#   cost.wall_clock_s                     (wall clock)
#   cost.tokens.{input,output,total}, cost.api_calls   (token/compute cost)
#   sim_time.compression_factor           (sim-time compression factor)
# Plus meta: manifest_version, run_id, experiment, created_at (ISO8601 UTC),
#   config_hash, package_version, python_version, platform,
#   deterministic_probe (dict), llm (dict), nondeterministic_fields (list[str]).

def new_manifest(*, cfg: "Config", embedder, llm, seeds: dict[str, int],
                 deterministic_probe: dict, wall_clock_s: float,
                 git: dict | None = None, run_id: str | None = None,
                 created_at: str | None = None) -> "Manifest": ...
#   - reads cfg.model.{id,temperature,max_tokens,top_p,effort}, cfg.embedding.*,
#     cfg.sim_time.compression_factor, cfg.hyperparameters, cfg.search_ranges, cfg.content_hash()
#   - embedder exposes: .backend, .model, .version, .dim
#   - llm exposes: .model_id, .input_tokens, .output_tokens, .mocked, .text
#   - nondeterministic_fields MUST flag the time + LLM-dependent fields. Either a
#     bare "llm" entry OR granular paths ("llm.text_sha256", "llm.output_tokens",
#     ...) are acceptable; the implementation uses the granular form (more precise,
#     since llm.model_id/llm.mocked are deterministic). MUST include at least
#     "created_at", "cost.wall_clock_s", "cost.tokens.output", and one "llm.*"
#     path (and "git.commit"/"git.dirty" may vary across checkouts).

def write_manifest(manifest: "Manifest", path: str | Path) -> Path:   # JSON, indent=2, sort_keys=True, parents ok
def read_manifest(path: str | Path) -> "Manifest":
```

## `slow_wave/embeddings.py`  (WS4 owns)

```python
# Embedder duck-type: attributes .backend:str .model:str .version:str .dim:int ; method .encode(texts)->np.ndarray
class HashEmbedder:        # backend="hash", model="hash-bow-v1", version="1.0"; deterministic, no heavy deps
    def encode(self, texts: list[str]) -> np.ndarray   # (n, dim) float32, L2-normalized, stable cross-run/platform
class SentenceTransformerEmbedder:   # lazy import sentence_transformers; version = st.__version__; dim from model
def get_embedder(cfg: "Config"):     # requested backend; if ST/torch import fails -> HashEmbedder, set .fallback_reason
def embed_texts(embedder, texts: list[str]) -> np.ndarray
def embedding_sha256(arr: np.ndarray) -> str   # stable: round to 6 dp -> bytes -> sha256 hex
```

## `slow_wave/llm.py`  (WS4 owns)

```python
@dataclass
class LLMResult:
    text: str; model_id: str; input_tokens: int; output_tokens: int; mocked: bool
    stop_reason: str | None = None

def complete(cfg: "Config", prompt: str, system: str | None = None) -> LLMResult:
    # if os.environ.get("ANTHROPIC_API_KEY"): real anthropic SDK call with cfg.model.id, temperature,
    #     max_tokens, top_p; parse resp.usage.input_tokens/output_tokens, text from content blocks; mocked=False
    # else: deterministic MOCK -> text=f"[MOCK:{sha256(prompt).hex[:8]}] ...", token counts derived
    #     deterministically from text length; mocked=True. NEVER raises on missing key.
```

## `slow_wave/repro/smoke.py`  (WS4 owns) — the hello-bench

```python
def run_smoke(cfg: "Config", out_dir: str | Path | None = None) -> Path:
    # 1) set_global_seeds(cfg.seed); agent_seed=derive_seed(cfg.seed,"agent"); stream_seed=derive_seed(cfg.seed,"stream")
    # 2) texts = cfg.smoke.texts or [f"slow-wave smoke item {i}" for i in range(cfg.smoke.n_items)]
    # 3) embedder=get_embedder(cfg); emb=embed_texts(embedder, texts)
    # 4) sampling_order = np.random.default_rng(stream_seed).permutation(len(texts)).tolist()
    # 5) llm = complete(cfg, cfg.smoke.prompt)
    # 6) manifest = new_manifest(cfg=, embedder=, llm=, seeds={"master","agent","stream"},
    #       deterministic_probe={"embedding_sha256": embedding_sha256(emb),
    #                            "sampling_order": sampling_order, "n_items": len(texts)},
    #       wall_clock_s=, git=git_info())
    # 7) return write_manifest(manifest, Path(out_dir or cfg.output_dir)/"smoke"/"manifest.json")
# __main__ : argparse --config (default "configs/smoke.yaml"), --out ; print manifest path
```

**One command (canonical):** `python -m slow_wave.repro.smoke --config configs/smoke.yaml`
Makefile target `repro-smoke` wraps exactly this.

## Determinism rule (exit criterion #3)
`deterministic_probe` carries `{embedding_sha256, sampling_order, n_items}` and may include
the extra deterministic key `embedder_backend`. Two smoke runs with the same config+seeds must
produce identical: `deterministic_probe.embedding_sha256`, `deterministic_probe.sampling_order`,
`deterministic_probe.n_items`, and the set of output files.
Everything in `nondeterministic_fields` (LLM text/output-tokens, wall-clock, created_at) is exempt and flagged.
