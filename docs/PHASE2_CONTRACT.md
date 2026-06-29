# Phase 2 — Shared Interface Contract (authoritative)

This file pins the cross-module interfaces for **Phase 2 (Memory Substrate +
Baseline No-Sleep Wake Agent)** so independently-authored modules integrate
without guessing. **Do not deviate from these signatures.** Import shared Phase 0
/ Phase 1 types from their modules — do not redefine them.

## Objective & exit criteria (PRD §8, Phase 2)
Dual-store memory + a no-sleep wake agent that runs a Phase 1 stream end-to-end
and emits the full metric stream **without any dream cycle** (the
catastrophic-forgetting reference). The work must satisfy, verbatim:

- **EC1** No-sleep agent runs an entire stream end-to-end and writes a populated
  `R[i,j]` **plus cost telemetry** to a manifest.
- **EC2** EPISODIC and SEMANTIC are **physically separate**; per-store retrieval,
  footprint, and forgetting are independently queryable (test asserts separation).
- **EC3** Every consolidated/episodic item exposes a **provenance pointer** to its
  source(s); a test traces ≥1 item back to its origin.
- **EC4** Archival eviction **demotes** (does not delete): an evicted item is
  recoverable from the archival tier (test).
- **EC5** A write-protection violation (distractor overwriting a protected fact)
  produces a **logged failure event** (test injects the condition).
- **EC6** The baseline demonstrably **forgets**: on a noisy stream, BWT is
  measurably negative / forgetting > 0.

Maps to FR2.1–FR2.5 (memory), FR3.1–FR3.3 (wake agent), FR5.3 (BWT/forgetting).

## Ground rules (inherited from Phase 0/1 — non-negotiable)
- Python **3.12** target (support 3.11+). Run tests with
  `.venv/Scripts/python.exe -m pytest`.
- pydantic **v2**. `extra="forbid"` on config-like / structured models; `frozen=True`
  only for immutable value objects (`MemoryEntry` is **mutable** — salience/tier
  change over time). For any model with a field literally named `model`, set
  `model_config = ConfigDict(protected_namespaces=())` (none of the Phase 2 models
  need it).
- **Determinism is sacred (DX1).** All randomness via
  `numpy.random.default_rng(derive_seed(seed, "<name>"))`. Never Python `hash()`
  for anything reaching an artifact; use `hashlib`. JSON with `sort_keys=True`.
  Given a fixed config+seed, the agent's `R[i,j]`, memory footprint, eviction
  counts, and retrieval counts are **byte-identical** across two runs (LLM token
  counts/wall-clock are the only flagged-nondeterministic outputs).
- **Lean core deps only**: `pydantic`, `pyyaml`, `numpy`. No torch / scipy /
  sentence-transformers in Phase 2 code. Embeddings come from the existing
  `slow_wave.embeddings` (hash backend is the default, dependency-free).
- **Confound guard (FR1.6) is sacred.** The wake agent consumes **only**
  `slow_wave.stream.guard.online_view(stream)` (label-free `StreamItem`s). It must
  **never** call `offline_labels` or touch `stream.ground_truth`. No memory object
  carries a relevance label or any `guard.BANNED_FIELD_NAMES` field name.
- **Import by full module path** (`from slow_wave.memory.stores import ...`), NOT
  via the package root. The orchestrator owns `slow_wave/memory/__init__.py`,
  `slow_wave/agent/__init__.py`, `slow_wave/eval/__init__.py`, `config.py`,
  `repro/manifest.py`, the `configs/*.yaml`, and the cross-module integration
  tests — **do not edit those**.
- Google-style docstrings on every public function/class, matching the density
  and tone of the existing Phase 0/1 modules.

## Already scaffolded by the orchestrator (consume as-is — do not edit)
- `slow_wave/config.py` now exposes `MemoryConfig` and `AgentConfig`, wired onto
  `Config` as `cfg.memory` and `cfg.agent`. Field names are authoritative:
  - `MemoryConfig`: `episodic_capacity:int=0` (0 = unbounded), `archival_enabled:bool=True`,
    `retrieval_policy:str="recency_importance_relevance"`, `retrieval_top_k:int=8`,
    `recency_half_life:float=64.0`, `weight_recency:float=1.0`,
    `weight_importance:float=1.0`, `weight_relevance:float=1.0`,
    `base_salience:float=1.0`, `novelty_enabled:bool=True`.
  - `AgentConfig`: `token_budget:int|None=None`, `reasoning_calls:Literal["off","per_task"]="per_task"`,
    `reasoning_prompt:str=...`. (Probe answering is exact-key lookup over active
    memory, so there is no eval top-k knob.)
- `slow_wave/repro/manifest.py`: `Manifest` gained an optional `results:dict={}`
  field; `new_manifest(...)` gained `api_calls:int=1` and `results:dict|None=None`
  kwargs. Use these to record `R[i,j]` + telemetry (EC1).
- Embedder duck-type (`slow_wave.embeddings.get_embedder(cfg)`): `.backend`,
  `.model`, `.version`, `.dim`, `.encode(list[str])->np.ndarray` (`(n,dim)` float32,
  **L2-normalized** rows; so cosine == dot product).
- LLM: `slow_wave.llm.complete(cfg, prompt, system=None) -> LLMResult` with
  `.text/.model_id/.input_tokens/.output_tokens/.mocked/.stop_reason`. Deterministic
  mock when no `ANTHROPIC_API_KEY` (the dev/CI default).

---

## WS-MEM — Memory substrate (`slow_wave/memory/{schema,salience,stores,retrieval}.py`)
Owns the **entire** `slow_wave/memory/` package implementation + its tests. Depends
only on `numpy`, `pydantic`, `slow_wave.stream.schema` (for `Fact`), and
`slow_wave.config.MemoryConfig`. **Tests:** `tests/test_memory_stores.py`,
`tests/test_memory_retrieval.py`, `tests/test_memory_protection.py`.

### `schema.py` — data model (exact)
```python
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from slow_wave.stream.schema import Fact

class MemoryTier(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVAL = "archival"

class SalienceMeta(BaseModel):            # mutable (updated on access)
    model_config = ConfigDict(extra="forbid")
    importance: float                      # weight; starts at MemoryConfig.base_salience
    recency_order: int                     # stream order of last observation/access
    access_count: int = 0
    novelty: float = 0.0                   # embedding distance to consolidated memory at insert, [0,1]
    surprise: float = 0.0                  # reserved (reward/surprise); default 0

class MemoryEntry(BaseModel):             # mutable (tier flips on demotion)
    model_config = ConfigDict(extra="forbid")
    entry_id: str                          # stable id; episodic uses "e{order:06d}"
    tier: MemoryTier
    content: str
    fact: Fact | None = None
    created_order: int                     # stream order at creation (latest-wins tie-break for answers)
    salience: SalienceMeta
    provenance: tuple[str, ...]            # source ids: stream item_id(s) (episodic) / source entry_id(s) (semantic)
    protected: bool = False                # EWC-spirit write-protection flag (FR2.5)
    def key(self) -> tuple[str, str] | None: ...   # self.fact.key() or None

class FailureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str                              # e.g. "protected_overwrite"
    at_order: int
    key: tuple[str, str] | None
    detail: str
    old_value: str | None = None
    new_value: str | None = None
    source: str | None = None              # id of the offending write (e.g. distractor entry/item id)

class StoreFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tier: MemoryTier
    n_entries: int
    n_vectors: int
    dim: int
    bytes: int                             # n_vectors * dim * 4  (float32)

class MemoryFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodic: StoreFootprint
    semantic: StoreFootprint
    archival: StoreFootprint
    total_bytes: int
```
**No banned field names** (`label`, `relevance`, `ground_truth`, ...) anywhere —
the integration confound test runs `guard.assert_no_label_leak` over live entries.

### `salience.py` — scoring primitives (exact)
```python
import numpy as np

def recency_factor(now_order: int, last_order: int, half_life: float) -> float:
    """0.5 ** (max(0, now_order-last_order) / half_life); in (0, 1]."""

def novelty_score(vector: np.ndarray, reference: np.ndarray | None) -> float:
    """1 - max cosine similarity of `vector` to any row of `reference`
    (the consolidated/semantic matrix), clipped to [0, 1]. Returns 1.0 when
    `reference` is None or empty (everything is novel vs. an empty store)."""

def retrieval_score(*, recency: float, importance: float, relevance: float,
                    w_recency: float, w_importance: float, w_relevance: float) -> float:
    """Park et al. (2023) weighted-sum memory-stream score:
    w_recency*recency + w_importance*importance + w_relevance*relevance.
    `recency` in (0,1], `relevance` = max(0, cosine) in [0,1], `importance`>=0."""

def eviction_score(*, recency: float, importance: float,
                   w_recency: float, w_importance: float) -> float:
    """Query-free priority for eviction (lowest is evicted first):
    w_recency*recency + w_importance*importance."""
```

### `stores.py` — the three physical stores + substrate (exact)
Each store keeps entries **and** a parallel float32 vector index (so footprint =
vector count × dim × 4 bytes). Vectors are passed in by the caller (the agent
computes embeddings); stores never embed.
```python
class _VectorStore:  # private base is fine; not part of the public contract
    ...

class EpisodicStore:
    def __init__(self, dim: int, capacity: int = 0, half_life: float = 64.0,
                 w_recency: float = 1.0, w_importance: float = 1.0) -> None: ...
    def __len__(self) -> int: ...
    def append(self, entry: MemoryEntry, embedding: np.ndarray,
               now_order: int) -> list[MemoryEntry]:
        """Append entry+vector (tier must be EPISODIC). If capacity>0 and the
        store exceeds it, evict lowest-`eviction_score` live entries (computed at
        `now_order`) until within capacity; return the evicted entries (callers
        demote them). capacity==0 means unbounded (returns [])."""
    def touch(self, entry_id: str, now_order: int) -> None:    # ++access_count, recency_order=now_order
    def get(self, entry_id: str) -> MemoryEntry | None: ...
    def vector(self, entry_id: str) -> np.ndarray | None: ...
    def all_entries(self) -> list[MemoryEntry]: ...            # live entries, insertion order
    def find_by_key(self, key: tuple[str, str]) -> list[MemoryEntry]: ...
    def snapshot(self) -> tuple[list[MemoryEntry], np.ndarray]:  # (entries, (n,dim) matrix) aligned
    def footprint(self) -> StoreFootprint: ...

class SemanticStore:
    def __init__(self, dim: int, half_life: float = 64.0,
                 w_recency: float = 1.0, w_importance: float = 1.0) -> None: ...
    def upsert(self, entry: MemoryEntry, embedding: np.ndarray, now_order: int,
               failure_sink: list[FailureEvent] | None = None) -> bool:
        """Write a SEMANTIC entry (tier must be SEMANTIC). FR2.5 write-protection:
        if an existing entry with the same fact.key() is `protected` and the
        incoming fact.value differs, append a FailureEvent(kind="protected_overwrite",
        ...) to failure_sink, log a WARNING, DO NOT overwrite (preserve protected
        entry), and return False. Otherwise upsert by key (latest-wins for an
        unprotected same-key entry; new key => append) and return True."""
    # same query surface as EpisodicStore: touch/get/vector/all_entries/find_by_key/snapshot/footprint

class ArchivalStore:
    def __init__(self, dim: int) -> None: ...
    def demote(self, entry: MemoryEntry, embedding: np.ndarray | None,
               reason: str, at_order: int) -> None:
        """Record `entry` in the archival tier with tier set to ARCHIVAL,
        preserving entry_id/content/fact/provenance/salience. Auditable: stores
        (reason, at_order) per archived entry. Never raises on duplicate."""
    def recover(self, entry_id: str) -> MemoryEntry | None:   # the archived entry (content/fact intact)
    def contains(self, entry_id: str) -> bool: ...
    def find_by_key(self, key: tuple[str, str]) -> list[MemoryEntry]: ...
    def all_entries(self) -> list[MemoryEntry]: ...
    def footprint(self) -> StoreFootprint: ...

class MemorySubstrate:
    def __init__(self, mem_cfg: "MemoryConfig", dim: int) -> None:
        """Build episodic (with mem_cfg.episodic_capacity + weights/half_life),
        semantic, archival, and failure_events:list[FailureEvent]=[]."""
    episodic: EpisodicStore
    semantic: SemanticStore
    archival: ArchivalStore
    failure_events: list[FailureEvent]
    archival_enabled: bool
    def observe(self, entry: MemoryEntry, embedding: np.ndarray, now_order: int) -> list[MemoryEntry]:
        """Append `entry` to EPISODIC; demote any evicted entries to ARCHIVAL when
        archival_enabled, else drop them with a logged INFO line (DX2: never drop
        silently). Return the evicted entries (for telemetry/eviction count)."""
    def footprint(self) -> MemoryFootprint: ...               # total_bytes = sum of the three
```

### `retrieval.py` — pluggable policy + registry (exact; FR2.3, DX6)
```python
from typing import Protocol
import numpy as np

class RetrievalPolicy(Protocol):
    name: str
    def score(self, query_vec: np.ndarray, entries: list["MemoryEntry"],
              vectors: np.ndarray, now_order: int, mem_cfg: "MemoryConfig") -> np.ndarray:
        """Return a (len(entries),) float array of scores (higher = better)."""

REGISTRY: dict[str, RetrievalPolicy]
def register_policy(policy: RetrievalPolicy) -> None: ...      # by policy.name
def get_policy(name: str) -> RetrievalPolicy: ...             # KeyError if unknown

class RecencyImportanceRelevancePolicy:
    name = "recency_importance_relevance"
    def score(self, query_vec, entries, vectors, now_order, mem_cfg) -> np.ndarray:
        # per entry: recency=recency_factor(now, e.salience.recency_order, mem_cfg.recency_half_life);
        #            relevance=max(0, cosine(query_vec, vectors[i]));  importance=e.salience.importance;
        #            retrieval_score(...weights from mem_cfg...). Vectorize where possible.

# RecencyImportanceRelevancePolicy() is auto-registered at import (REGISTRY default).

def retrieve(substrate: "MemorySubstrate", query_vec: np.ndarray, top_k: int, *,
             now_order: int, mem_cfg: "MemoryConfig",
             tiers: tuple["MemoryTier", ...] = (MemoryTier.EPISODIC, MemoryTier.SEMANTIC),
             policy: RetrievalPolicy | None = None,
             touch: bool = False) -> list[tuple["MemoryEntry", float]]:
    """Gather (entries, vectors) from the requested tiers (per-store retrieval =
    pass a single tier), score with `policy` (default: mem_cfg.retrieval_policy
    via get_policy), sort by score **descending with a stable tie-break on
    entry_id**, return the top_k (entry, score) pairs. If touch=True, call the
    owning store's .touch(entry_id, now_order) for each returned entry (wake-time
    retrieval); eval-time retrieval passes touch=False (read-only, non-mutating)."""
```
ARCHIVAL is **never** retrieved by default (it is the forgotten tier) — passing
`tiers=(MemoryTier.ARCHIVAL,)` is allowed for audit/tests only.

### WS-MEM tests must cover
- **EC2 separation:** episodic and semantic are distinct objects; writing one does
  not appear in the other; `retrieve(..., tiers=(EPISODIC,))` vs `(SEMANTIC,)`
  return disjoint results; `substrate.footprint()` reports each tier independently.
- **EC3 provenance:** an episodic entry's `provenance` contains its source stream
  `item_id`; trace ≥1 entry back to its origin. (Also build a semantic entry from
  episodic sources and trace it.)
- **EC4 demote-not-delete:** fill an `EpisodicStore(capacity=N)` past N via
  `substrate.observe`; assert evicted entry is **absent** from episodic but
  `substrate.archival.recover(entry_id)` returns it with content/fact intact.
- **EC5 protection:** `semantic.upsert` a `protected` entry for key (s,a)=v1; then
  `upsert` a different-value (s,a)=v2 entry tagged with a distractor source →
  assert a `FailureEvent(kind="protected_overwrite")` is recorded in the sink, the
  protected value is preserved, and `upsert` returns False. Positive control: an
  unprotected same-key upsert **does** overwrite and returns True.
- Retrieval: baseline policy ranks an exact subject+attribute match above a
  same-attribute-different-subject distractor and above noise; registry
  `register_policy`/`get_policy` round-trips a trivial custom policy (DX6).
- Determinism: identical inputs → identical retrieval order and footprint.

---

## WS-EVAL — Continual-learning metrics (`slow_wave/eval/metrics.py`)
Owns `slow_wave/eval/metrics.py` + `tests/test_eval_metrics.py`. Depends only on
`pydantic` and `slow_wave.stream.schema.AccuracyMatrix`. Pure, deterministic, no
heavy deps. (Phase 4 adds the statistics suite **on top** of these point metrics.)

```python
from pydantic import BaseModel, ConfigDict
from slow_wave.stream.schema import AccuracyMatrix

class ContinualMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    acc: float                       # average accuracy
    bwt: float                       # backward transfer (negative = forgetting)
    fwt: float                       # forward transfer
    average_forgetting: float        # mean per-task forgetting (>=0 typical when forgetting)
    per_task_forgetting: list[float] # length n_tasks-1 (last task excluded)

# All take R as list[list[float]] (n_tasks x n_tasks), 0-based, T = n_tasks.
def average_accuracy(R) -> float:
    """ACC = mean over j of R[T-1][j]  (accuracy on every task after the last).
    T==0 -> 0.0."""
def backward_transfer(R) -> float:
    """BWT = (1/(T-1)) * sum_{i=0}^{T-2} (R[T-1][i] - R[i][i]).  T<=1 -> 0.0.
    (Lopez-Paz & Ranzato 2017.) Negative => the baseline forgot earlier tasks."""
def forward_transfer(R, baseline: list[float] | None = None) -> float:
    """FWT = (1/(T-1)) * sum_{i=1}^{T-1} (R[i-1][i] - baseline[i]).
    baseline defaults to zeros (random pre-training accuracy) -> FWT = mean of the
    just-above-diagonal entries. T<=1 -> 0.0."""
def per_task_forgetting(R) -> list[float]:
    """For j in 0..T-2: max_{l in 0..T-2} R[l][j] - R[T-1][j]  (Chaudhry et al.
    2018). Length T-1 (the last task cannot be forgotten yet). T<=1 -> []."""
def average_forgetting(R) -> float:
    """Mean of per_task_forgetting(R); 0.0 when that list is empty."""
def compute_continual_metrics(matrix: AccuracyMatrix) -> ContinualMetrics:
    """Bundle all of the above from matrix.R."""
```

### WS-EVAL tests must cover
- Lower-triangular-ones matrix (the oracle / no-forgetting skeleton): ACC==1.0,
  BWT==0.0, average_forgetting==0.0, per_task_forgetting all 0.
- A hand-crafted forgetting matrix with known values: assert BWT < 0 and
  average_forgetting > 0 equal the hand-computed numbers (exact).
- Edge cases: T==1 and T==0 return zeros without raising; entries respected in [0,1].

---

## WS-AGENT — No-sleep wake agent (`slow_wave/agent/{budget,wake,runner}.py`)
Owns `slow_wave/agent/{budget,wake,runner}.py` + `tests/test_agent_wake.py`,
`tests/test_agent_budget.py`, `tests/test_agent_runner.py`. **Wave 2** — develop
against the *already-landed* `slow_wave.memory.*` and `slow_wave.eval.metrics`
modules (import them directly). Depends on: memory (all), retrieval, eval.metrics,
`slow_wave.embeddings`, `slow_wave.llm`, `slow_wave.config`, `slow_wave.repro`
(seeding, manifest, gitinfo), `slow_wave.stream` (generator, guard.online_view,
probes.build_probe_set, schema).

### `budget.py` — token-budget controller (FR3.3, exact)
```python
class TokenBudgetController:
    def __init__(self, max_tokens: int | None) -> None: ...
    spent_input: int
    spent_output: int
    n_skipped: int                                   # calls skipped due to budget (DX2)
    @property
    def total_spent(self) -> int: ...                # spent_input + spent_output
    @property
    def remaining(self) -> int | None: ...           # None if max_tokens is None
    @property
    def exhausted(self) -> bool: ...                 # max not None and total_spent >= max
    def can_afford(self, est_tokens: int) -> bool:   # True if max None or total_spent+est <= max
    def record(self, input_tokens: int, output_tokens: int) -> None:
    def skip(self) -> None:                          # ++n_skipped (logs an INFO line)
```

### `wake.py` — the wake loop (FR3.1–FR3.3, exact)
```python
from dataclasses import dataclass, field

@dataclass
class WakeTelemetry:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls_made: int = 0
    reasoning_calls_skipped: int = 0
    retrieval_calls: int = 0
    n_items_ingested: int = 0
    n_evicted: int = 0
    step_latencies_s: list[float] = field(default_factory=list)
    @property
    def total_tokens(self) -> int: ...
    @property
    def p95_latency_s(self) -> float: ...            # 0.0 if no steps
    def as_dict(self) -> dict: ...                    # JSON-safe (sorted), for the manifest

@dataclass
class WakeResult:
    accuracy_matrix: "AccuracyMatrix"
    metrics: "ContinualMetrics"
    footprint: "MemoryFootprint"
    telemetry: WakeTelemetry
    substrate: "MemorySubstrate"

class WakeAgent:
    def __init__(self, cfg: "Config", embedder, *, llm_complete=None) -> None:
        """Build MemorySubstrate(cfg.memory, dim=embedder.dim) and a
        TokenBudgetController(cfg.agent.token_budget). llm_complete defaults to
        slow_wave.llm.complete (injectable for tests)."""

    def run(self, stream: "Stream", probe_set: "ProbeSet") -> WakeResult:
        """No-sleep wake pass (single online pass, evaluate after each task):
        1. items = guard.online_view(stream)            # LABEL-FREE; never read labels
        2. batch-embed item contents and probe queries (embedder.encode).
        3. for task t in 0..n_tasks-1:
             for each item with task_index==t (ascending order):
               - build a MemoryEntry(tier=EPISODIC, entry_id=f"e{order:06d}",
                 content, fact=item.fact, created_order=order,
                 salience=SalienceMeta(importance=cfg.memory.base_salience,
                                       recency_order=order,
                                       novelty=novelty vs semantic store
                                               if cfg.memory.novelty_enabled else 0.0),
                 provenance=(item.item_id,))
               - substrate.observe(entry, embedding, now_order=order)  # appends + demotes
             - if cfg.agent.reasoning_calls=="per_task": budget-gated reasoning step
               (FR3.1: retrieve from memory, then call Claude). If budget can't
               afford it, skip + telemetry.reasoning_calls_skipped++. Otherwise do
               ONE read-only `retrieve(..., touch=False)` over active memory to
               ground the prompt (retrieval_calls++), then llm_complete. **No
               semantic-store writes** (gating). touch=False because selective
               re-potentiation of accessed memories is a *dream* operator (FR4.3),
               not baseline wake — so eviction stays a clean function of
               observation recency (the catastrophic-forgetting reference).
             - evaluate EVERY probe against current memory via self.answer(...,
               touch=False) -> row R[t][j] = fraction correct for task j.
        4. R = AccuracyMatrix(n_tasks, scenario, rows); metrics =
           compute_continual_metrics(R); footprint = substrate.footprint().
        Returns WakeResult. Deterministic given (cfg, stream, probe_set) under the
        mock LLM."""

    def answer(self, probe: "Probe", query_vec: np.ndarray, now_order: int) -> str:
        """Answer a probe from **active** memory by exact-key lookup (read-only).
        Gather entries whose fact.key()==(probe.subject, probe.attribute) from the
        ACTIVE tiers (EPISODIC + SEMANTIC, never ARCHIVAL) via each store's
        find_by_key; return the fact.value of the one with the largest
        created_order (latest-wins, so contradictions resolve to the final value);
        else "". This makes the accuracy matrix crisp and deterministic
        (present ⇒ correct, evicted/forgotten ⇒ wrong) and immune to
        embedding-similarity fuzz. The agent sees only probe.query/subject/attribute
        — NEVER probe.answer. (The recency×importance×relevance retrieval policy is
        exercised by the read-only wake-time context retrieve above and is tested
        directly in WS-MEM; query_vec is accepted for signature symmetry and future
        similarity-gated answering.)"""
```
**Gating invariant (FR3.1):** the wake loop performs **no** `semantic.upsert` /
semantic writes — the SEMANTIC store is empty after a baseline run. (Phase 3's
dream engine is the only writer of the semantic store.)

### `runner.py` — one-command run + manifest (EC1, exact)
```python
def run_agent(cfg: "Config", out_dir: str | Path | None = None) -> Path:
    """End-to-end no-sleep run:
    - require cfg.stream (raise ValueError if None, mirroring stream.emit).
    - set_global_seeds(cfg.seed); stream = generate_stream(cfg.stream,
      derive_seed(cfg.seed, "stream")).
    - probe_set = build_probe_set(stream); embedder = get_embedder(cfg).
    - result = WakeAgent(cfg, embedder).run(stream, probe_set).
    - assemble an aggregate llm-cost object (summed input/output tokens,
      mocked=True unless a real call happened, model_id=cfg.model.id) and call
      new_manifest(..., wall_clock_s, api_calls=telemetry.api_calls,
        deterministic_probe={ "accuracy_matrix": R.R, "n_tasks": R.n_tasks,
          "memory_footprint": footprint.model_dump(mode="json"),
          "retrieval_calls": telemetry.retrieval_calls,
          "n_evicted": telemetry.n_evicted, "n_items": telemetry.n_items_ingested },
        results={ "accuracy_matrix": R.model_dump(mode="json"),
          "continual_metrics": metrics.model_dump(mode="json"),
          "memory_footprint": footprint.model_dump(mode="json"),
          "telemetry": telemetry.as_dict() }).
    - write_manifest(manifest, Path(out or cfg.output_dir)/"agent"/"manifest.json")
      and return the path. Print a one-line summary (stream_id, ACC, BWT,
      api_calls, total_tokens)."""

def main(argv: list[str] | None = None) -> int:
    # argparse --config (default "configs/agent_smoke.yaml") --out ; print manifest path
```
The manifest therefore carries **R[i,j] (results.accuracy_matrix) + cost telemetry
(cost.tokens.*, cost.api_calls, results.telemetry)** — EC1.

### WS-AGENT tests must cover
- `test_agent_wake.py`: a WakeAgent run on a small generated stream returns a
  well-formed `AccuracyMatrix` (square, [0,1]); the SEMANTIC store is **empty**
  after the run (gating FR3.1); telemetry counts are populated; the wake loop never
  reads labels (assert it works from `online_view` only — e.g. monkeypatch
  `offline_labels` to raise and confirm `run` still succeeds).
- `test_agent_budget.py`: tiny `token_budget` => reasoning calls are skipped
  (`reasoning_calls_skipped>0`, total tokens within budget); generous/None budget
  => all per-task calls made; `can_afford/record/remaining/exhausted` behave.
- `test_agent_runner.py`: `run_agent(cfg)` writes a manifest whose
  `results.accuracy_matrix` is populated, `cost.api_calls>=1` (or 0 if
  reasoning_calls="off"), `cost.tokens.total>0` when reasoning on, and `results`
  round-trips via `read_manifest`. Use `configs/agent_smoke.yaml`.

---

## Integration (orchestrator-owned — do NOT implement in a workstream)
After WS-MEM + WS-EVAL (wave 1) and WS-AGENT (wave 2) land, the orchestrator
wires the three package `__init__.py` exports, adds `configs/agent_smoke.yaml` +
`configs/agent_forgetting.yaml`, the cross-module tests
`tests/test_agent_forgetting.py` (EC6: BWT<0 / forgetting>0 on the bounded noisy
config) and `tests/test_agent_determinism.py` (byte-identical R + footprint across
two runs), a confound re-check over live memory entries, the `repro-agent` Makefile
target, and a README/CONTRACT note. Keep modules import-clean so this glue is
mechanical.
