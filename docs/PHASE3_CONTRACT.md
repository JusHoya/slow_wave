# Phase 3 — Shared Interface Contract (authoritative)

This file pins the cross-module interfaces for **Phase 3 (The Dream Engine —
four ablatable operators)** so independently-authored modules integrate without
guessing. **Do not deviate from these signatures.** Import shared Phase 0/1/2
types from their modules — do not redefine them.

## Objective & exit criteria (PRD §8, Phase 3)
Implement REPLAY, TRANSFER, DOWNSCALE, GENERATIVE-AUGMENT as **independent
toggles** in a two-phase (NREM→REM) cycle, plus gating and sleep-pressure
control. The work must satisfy, verbatim:

- **EC1** Each of the four operators can be **independently enabled/disabled by
  config**; a 2×2 (replay-module × downscale-module) and 4-way on/off matrix all
  instantiate and run (test enumerates the **2⁴ = 16** combinations and confirms
  each executes).
- **EC2** **DOWNSCALE** verifiably applies global decay then re-potentiates only
  replayed items: after a cycle, replayed-item salience **>** non-replayed under
  the same decay (test).
- **EC3** **TRANSFER** writes semantic entries **only** inside a scheduled sleep
  window (gating test: no semantic writes occur during wake).
- **EC4** **CLS interleaving** is enforced when enabled and *removable* (the
  catastrophic-interference condition): a test confirms batches mix new + prior
  consolidated memories when on, and don't when off.
- **EC5** **GENERATIVE-AUGMENT** produces pseudo-episodes and logs a
  generator-fidelity/drift score per cycle.
- **EC6** Decay function is swappable among exponential / Weibull / ACT-R (test
  runs all three).
- **EC7** A full dream cycle runs on a Phase-1 stream and leaves provenance +
  archival audit intact (**no hard deletes**).

Maps to FR4.1–FR4.7, FR5.6 (drift detector seed), DX2 (no silent caps), DX6
(reusability).

## Ground rules (inherited from Phase 0/1/2 — non-negotiable)
- Python **3.12** target (support 3.11+). Run tests with
  `.venv/Scripts/python.exe -m pytest`.
- pydantic **v2**. `extra="forbid"` on every structured model. Reuse the
  **mutable** `MemoryEntry` / `SalienceMeta` (salience changes over time) and the
  **frozen** `Fact` — do not redefine them. For any model with a field literally
  named `model`, set `model_config = ConfigDict(protected_namespaces=())` (none of
  the Phase 3 models need it).
- **Determinism is sacred (DX1).** All randomness via
  `numpy.random.default_rng(derive_seed(seed, "<name>"))` — never Python `hash()`
  or `np.random` global. Given a fixed config+seed **under the mock LLM**, every
  dream operator's structured result (sample ids, written semantic ids,
  salience values, pseudo ids, demoted ids, counts) is **byte-identical** across
  two runs. The only nondeterministic outputs are wall-clock timing and (once a
  real Claude call is made) token counts / generated text.
- **Lean core deps only**: `pydantic`, `pyyaml`, `numpy`. No torch / scipy /
  sentence-transformers in Phase 3 code. Embeddings come from the embedder
  duck-type passed in (`.dim`, `.encode(list[str]) -> np.ndarray` → `(n, dim)`
  float32, **L2-normalized** rows, so cosine == dot product).
- **Confound guard (FR1.6) is sacred.** Dream operators consume only
  confound-free `MemoryEntry` graphs and the embedder. They must **never** read
  `offline_labels`, `stream.ground_truth`, `probe.answer`, or any
  `guard.BANNED_FIELD_NAMES` field. No dream-created entry (semantic / pseudo)
  may carry a relevance label or a banned field name — the integration confound
  test walks live entries with `guard.assert_no_label_leak`.
- **Honesty by construction (DX2).** Anything an operator bounds (top-N replay,
  a per-cycle call ceiling, a refused/ skipped write) is recorded in its result
  model **and** `logger.info/​warning`-logged. Never drop silently.
- **Import by full module path** (`from slow_wave.dream.decay import get_decay`),
  NOT via the package root. The orchestrator owns
  `slow_wave/dream/__init__.py`, `slow_wave/dream/runner.py`,
  `slow_wave/dream/engine.py` *integration glue*, `slow_wave/config.py`,
  `slow_wave/dream/schema.py`, `slow_wave/dream/decay.py`, `slow_wave/agent/wake.py`,
  `slow_wave/memory/stores.py`, the `configs/*.yaml`, and the cross-module
  integration tests — **do not edit those** (engine.py *is* a workstream; see
  WS-ENGINE).
- Google-style docstrings on every public function/class, matching the density
  and tone of the existing Phase 0/1/2 modules.

## Already scaffolded by the orchestrator (consume as-is — do NOT edit)
- **`slow_wave/config.py`** now exposes **`DreamConfig`**, wired onto `Config` as
  `cfg.dream`. Field names are authoritative (see the class for full docs):
  `enabled:bool=False`; operator toggles `replay_enabled/transfer_enabled/
  downscale_enabled/augment_enabled:bool=True`, `conflict_enabled:bool=False`;
  scheduling `sleep_every_n_tasks:int=1`, `sleep_pressure_mode:Literal["fixed",
  "adaptive"]="fixed"`, `sleep_pressure_churn_threshold:int=0`; REPLAY
  `replay_sample_size:int=16`, `replay_strategy:Literal["uniform","prioritized"]
  ="prioritized"`, `replay_priority_alpha:float=1.0`, `replay_priority_eps:float
  =1e-6`; TRANSFER `transfer_batch_size:int=8`, `cls_interleave:bool=True`,
  `cls_interleave_ratio:float=0.5`, `transfer_max_calls:int|None=None`; DOWNSCALE
  `decay_function:Literal["exponential","weibull","act_r"]="exponential"`,
  `repotentiate_boost:float=1.5`, `decay_exponential_rate:float=0.1`,
  `decay_weibull_scale:float=10.0`, `decay_weibull_k:float=1.5`,
  `decay_act_r:float=0.5`; AUGMENT `augment_per_cycle:int=4`,
  `augment_kinds:list[str]=["paraphrase","abstraction","counterfactual"]`;
  CONFLICT `conflict_demote_strategy:Literal["older","lower_salience"]="older"`.
- **`slow_wave/dream/schema.py`** — the shared result models (consume; do not
  redefine): `DreamPhase`, `ReplaySample`, `ReplayResult`, `TransferResult`,
  `DownscaleResult`, `FidelityScore`, `AugmentResult`, `ConflictResult`,
  `DreamCycleResult`, `DreamTelemetry`. Every operator **returns** its result
  model; the engine rolls them into `DreamCycleResult`; the runner serializes
  `DreamTelemetry` into the manifest. Read the field docstrings — they define the
  exact counters you must populate (incl. the DX2 `n_dropped`/`n_refused`/
  `n_skipped*` fields).
- **`slow_wave/dream/decay.py`** — the swappable decay registry (EC6). Use it:
  `exponential_decay(age,*,rate)`, `weibull_decay(age,*,scale,k)`,
  `act_r_decay(age,*,d)`, all `f(0)=1`, non-increasing, in `(0,1]`;
  `DECAY_REGISTRY`, `register_decay(name,fn)`, `get_decay(name)`,
  `decay_factor(name, age, params: dict|None)`, and **`params_for(name, cfg)`**
  which maps `DreamConfig`'s flat decay knobs to each curve's kwargs.
- **`slow_wave/memory/stores.py`** gained two public methods (consume as-is):
  `_VectorStore.pop(entry_id) -> (entry, vector)` and
  **`MemorySubstrate.demote_entry(entry_id, *, reason, at_order) -> bool`**
  (removes an active entry from episodic/semantic and demotes it to archival, or
  drops-with-log when `archival_enabled=False`). Use `demote_entry` for
  conflict/unlearning (demote-not-delete).
- **`slow_wave/agent/wake.py`** — `WakeAgent.run(stream, probe_set, *,
  sleep_hook=None)`. The orchestrator added the optional **`sleep_hook`**: it is
  invoked once per task segment, **after** ingest + reasoning and **before**
  evaluation, with the keyword signature
  `sleep_hook(substrate, *, embedder, llm_complete, now_order, task_index)`. This
  is the *only* place semantic writes are allowed (gating, EC3). `None` ⇒
  byte-identical Phase 2 baseline.
- **`configs/dream_smoke.yaml`** (all operators on, cycle every task, bounded
  episodic so eviction→archival is live) and **`configs/dream_full.yaml`** (the
  treatment over the forgetting stream).
- Reuse from earlier phases: `slow_wave.memory.{schema,stores,salience,retrieval}`,
  `slow_wave.eval.metrics`, `slow_wave.repro.seeding.derive_seed`,
  `slow_wave.llm.complete` (signature `complete(cfg, prompt, system=None) ->
  LLMResult` with `.text/.model_id/.input_tokens/.output_tokens/.mocked/
  .stop_reason`; deterministic mock when no key), `slow_wave.stream.schema.Fact`.

## Canonical cycle structure (the engine assembles this; operators are pieces)
A dream cycle is two phases (PRD §7.1 diagram):
- **NREM:** `REPLAY → TRANSFER → DOWNSCALE → (CONFLICT, optional)`
- **REM:** `GENERATIVE-AUGMENT`

The engine selects the **recent candidate pool** = episodic entries created since
the previous cycle (`created_order > last_cycle_order`; all episodics on the
first cycle). REPLAY samples that pool. The **replayed set** = sampled entries
(empty if REPLAY is off). TRANSFER consolidates the **transfer source** = the
replayed set if REPLAY is on, else the whole candidate pool. DOWNSCALE
re-potentiates the replayed set (pure global decay if REPLAY is off). AUGMENT
draws sources from the transfer source / candidate pool. Every operator is an
independent toggle, so **all 16 on/off combinations must instantiate and run**
(EC1) — including all-off (an empty cycle that records `operators_run=[]`).

**TRANSFER design rule (carry forward from Phase 2):** probe answering is
exact-key lookup over active memory (latest-wins by `created_order`). So a
semantic entry **preserves the source episodic's structured `fact`** verbatim
(same `(subject, attribute, value)`); the Claude "dream summary" becomes the
entry's natural-language `content` (and feeds generator-fidelity tracking), but
the `fact` is carried deterministically so answers stay crisp and a consolidated
signal survives episodic eviction. Set the semantic entry's `created_order` to
the source's `created_order` so latest-wins ordering is preserved.

---

## WS-REPLAY — REPLAY operator (`slow_wave/dream/replay.py`)
Owns `slow_wave/dream/replay.py` + `tests/test_dream_replay.py`. Depends only on
`numpy`, `slow_wave.config.DreamConfig`, `slow_wave.memory.schema` (+ optionally
`slow_wave.memory.salience.recency_factor`), `slow_wave.dream.schema`.

```python
import numpy as np
from slow_wave.config import DreamConfig
from slow_wave.memory.schema import MemoryEntry
from slow_wave.dream.schema import ReplaySample, ReplayResult

def replay(
    candidates: list[MemoryEntry],
    *,
    dream_cfg: DreamConfig,
    rng: np.random.Generator,
    now_order: int,
    recency_half_life: float = 64.0,
) -> ReplayResult:
    """Sample up to dream_cfg.replay_sample_size entries from `candidates`.

    - strategy == "uniform": every candidate has priority 1.0, probability 1/N,
      is_weight 1.0; sample WITHOUT replacement (rng.choice, replace=False).
    - strategy == "prioritized" (Schaul et al. 2016): per-entry priority =
      ( recency_factor(now_order, e.salience.recency_order, recency_half_life)
        * max(e.salience.importance, 0)
        * (e.salience.novelty + replay_priority_eps)
        * (1.0 + max(e.salience.surprise, 0)) + replay_priority_eps
      ) ** replay_priority_alpha ; probability = priority / sum(priority);
      sample WITHOUT replacement with p=probability; importance-sampling weight
      = (1/(N*probability))**1.0 normalized so max is 1.0 (log it, never drop).
    n_sampled = min(replay_sample_size, len(candidates)); n_dropped = the rest
    (DX2 — log it). Empty candidates or replay_sample_size==0 → an empty
    ReplayResult (n_candidates set, n_sampled=0). Deterministic given rng."""
```
**Tests must cover:** uniform vs prioritized both return ≤ sample_size samples
with valid probabilities (sum≈1 over the pool) and is_weights in (0,1];
prioritized ranks a high-recency/high-importance entry's *expected* selection
above a stale low-importance one (e.g. with a degenerate pool where one entry
dominates, it is always picked); `n_dropped == n_candidates - n_sampled` and is
logged; determinism (same rng seed → identical sampled ids); `replay_sample_size
== 0` and empty pool are safe.

---

## WS-DOWNSCALE — DOWNSCALE operator + decay test (`slow_wave/dream/downscale.py`)
Owns `slow_wave/dream/downscale.py` + `tests/test_dream_downscale.py` +
`tests/test_dream_decay.py` (EC6 lives here — you exercise the swap). Depends on
`slow_wave.config.DreamConfig`, `slow_wave.memory.{schema,stores}`,
`slow_wave.dream.{decay,schema}`.

```python
from slow_wave.config import DreamConfig
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.memory.schema import MemoryTier
from slow_wave.dream.decay import decay_factor, params_for
from slow_wave.dream.schema import DownscaleResult

def downscale(
    substrate: MemorySubstrate,
    *,
    dream_cfg: DreamConfig,
    replayed_ids: set[str],
    now_order: int,
) -> DownscaleResult:
    """Global salience decay, then re-potentiate ONLY replayed items (FR4.3).

    For every live entry in EPISODIC and SEMANTIC:
      age = max(0, now_order - e.salience.recency_order)
      factor = decay_factor(dream_cfg.decay_function, age,
                            params_for(dream_cfg.decay_function, dream_cfg))
      e.salience.importance *= factor                      # decay all
      if e.entry_id in replayed_ids:                       # protect signal
          e.salience.importance *= dream_cfg.repotentiate_boost   # boost (>=1)
          e.salience.recency_order = now_order             # reset age
    Record decay_function, n_decayed, n_repotentiated, mean_salience_before/after
    (means over the decayed entries; 0.0 if none). Pure-decay when replayed_ids
    is empty. Does not add/remove/demote entries (only mutates salience)."""
```
**Tests must cover (EC2 + EC6):** **EC2** — two episodic entries with *identical*
importance and recency_order; put one id in `replayed_ids`; after `downscale`,
the replayed entry's importance is **strictly greater** than the non-replayed
one (same decay, boost wins) and the replayed entry's `recency_order ==
now_order`. **EC6** — `tests/test_dream_decay.py` exercises all three curves via
`decay_factor`: each is `1.0` at age 0, strictly `< 1.0` and decreasing for
ages `0 < a1 < a2`, in `(0,1]`; and `downscale` runs end-to-end with
`decay_function` set to each of `"exponential"/"weibull"/"act_r"` producing
distinct post-decay salience. Determinism: identical substrate+cfg → identical
salience.

---

## WS-TRANSFER — TRANSFER operator + CLS interleaving (`slow_wave/dream/transfer.py`)
Owns `slow_wave/dream/transfer.py` + `tests/test_dream_transfer.py`. Depends on
`numpy`, `slow_wave.config.{Config,DreamConfig}`, `slow_wave.memory.{schema,
stores}`, `slow_wave.stream.schema.Fact`, `slow_wave.dream.schema`, and the
injected `embedder` + `llm_complete`.

```python
import numpy as np
from slow_wave.config import Config, DreamConfig
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.dream.schema import TransferResult

def transfer(
    substrate: MemorySubstrate,
    sources: list[MemoryEntry],
    *,
    cfg: Config,
    dream_cfg: DreamConfig,
    embedder,
    llm_complete,
    rng: np.random.Generator,
    now_order: int,
) -> TransferResult:
    """Distill sampled episodics into durable SEMANTIC entries (FR4.2).

    1. Keep only `sources` that carry a fact (noise has fact=None → not
       consolidated). Partition into batches of dream_cfg.transfer_batch_size.
    2. CLS interleaving (EC4): when dream_cfg.cls_interleave is True, each batch
       additionally PULLS prior consolidated entries from substrate.semantic
       (sample up to round(cls_interleave_ratio * batch_size), deterministically
       via rng) and includes them as context in the summarization prompt — these
       are NOT re-written; count them in n_interleaved_items. When False, no
       prior memory is mixed (n_interleaved_items == 0) — the
       catastrophic-interference condition. (On the first-ever cycle the semantic
       store is empty, so n_interleaved_items is 0 even when on; the EC4 test
       seeds/uses a populated semantic store — see below.)
    3. Per batch: ONE llm_complete(cfg, prompt) "dream summary" call (respect
       dream_cfg.transfer_max_calls across the cycle — skip + count
       n_skipped_calls + log when exceeded). Accumulate api_calls/input/output
       tokens.
    4. For each NEW (non-interleaved) source with a fact, upsert a semantic entry:
         MemoryEntry(entry_id=f"s{src.created_order:06d}", tier=SEMANTIC,
           content=<the dream summary text>, fact=src.fact,           # PRESERVE fact
           created_order=src.created_order,                           # latest-wins
           salience=SalienceMeta(importance=src.salience.importance,
                                 recency_order=now_order,
                                 novelty=src.salience.novelty),
           provenance=(src.entry_id,))                                # EC3/EC7 trace
       via substrate.semantic.upsert(entry, embedder.encode([content])[0],
           now_order, failure_sink=substrate.failure_events). Count
       n_semantic_written (upsert True) and n_refused (upsert False — protected
       overwrite). Record written_entry_ids and n_consolidated.
    Deterministic given (cfg, sources, rng) under the mock LLM."""
```
**Tests must cover (EC4 + gating support):** a transfer over signal-bearing
sources writes semantic entries that **preserve the source fact** and a
provenance pointer to the source episodic id; `n_refused` increments when a
source would overwrite a **protected** same-key semantic entry (and the protected
value is preserved — reuse the FR2.5 path); **EC4** — with a pre-populated
semantic store, `cls_interleave=True` ⇒ `n_interleaved_items > 0`, and
`cls_interleave=False` ⇒ `n_interleaved_items == 0` on the same inputs; noise
(fact=None) sources are not consolidated; `transfer_max_calls` caps calls and
logs/records `n_skipped_calls`; determinism.

---

## WS-AUGMENT — GENERATIVE-AUGMENT operator (`slow_wave/dream/augment.py`)
Owns `slow_wave/dream/augment.py` + `tests/test_dream_augment.py`. Depends on
`numpy`, `slow_wave.config.{Config,DreamConfig}`, `slow_wave.memory.{schema,
stores}`, `slow_wave.dream.schema`, and the injected `embedder` + `llm_complete`.

```python
import numpy as np
from slow_wave.config import Config, DreamConfig
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.memory.schema import MemoryEntry, MemoryTier, SalienceMeta
from slow_wave.dream.schema import AugmentResult, FidelityScore

def augment(
    substrate: MemorySubstrate,
    sources: list[MemoryEntry],
    *,
    cfg: Config,
    dream_cfg: DreamConfig,
    embedder,
    llm_complete,
    rng: np.random.Generator,
    now_order: int,
) -> AugmentResult:
    """Synthesize pseudo-episodes from sampled episodics (FR4.4, REM-like).

    Pick up to dream_cfg.augment_per_cycle sources (deterministically via rng;
    prefer fact-bearing sources). For pseudo i (kind = augment_kinds[i % len]):
      - llm_complete(cfg, f"<kind> of: {src.content}") → pseudo text (accumulate
        api_calls/tokens).
      - write a pseudo-episode to EPISODIC via substrate.observe(entry, emb,
        now_order) where
          entry = MemoryEntry(entry_id=f"g{now_order:06d}_{i}", tier=EPISODIC,
            content=<pseudo text>, fact=None,            # never corrupt exact-key answers
            created_order=now_order,
            salience=SalienceMeta(importance=src.salience.importance,
                                  recency_order=now_order),
            provenance=(src.entry_id,))                  # EC7 trace
        emb = embedder.encode([pseudo text])[0]. (observe demotes any eviction to
        archival — EC7-safe.)
      - fidelity_i = max(0, cosine(emb, source_emb)) where source_emb =
        substrate.<owning store>.vector(src.entry_id) or embedder.encode(
        [src.content])[0] if missing.
    Log a FidelityScore per cycle: n_pseudo, mean_fidelity, min_fidelity,
    mean_drift = 1 - mean_fidelity (all 0.0 if n_pseudo==0). Record
    pseudo_entry_ids. Deterministic given (cfg, sources, rng) under the mock LLM."""
```
**Tests must cover (EC5):** an augment pass with sources produces
`n_pseudo == min(augment_per_cycle, len(sources))` pseudo-episodes that live in
the episodic tier, carry `fact=None`, and trace via provenance to a source id; a
`FidelityScore` is produced with `mean_fidelity` in `[0,1]` and `mean_drift ==
1 - mean_fidelity`; `augment_per_cycle == 0` or empty sources → empty result
(n_pseudo 0, fidelity zeros) without raising; determinism (same rng → same
pseudo ids/text/fidelity under mock LLM).

---

## WS-CONFLICT — conflict / unlearning step (`slow_wave/dream/conflict.py`)
Owns `slow_wave/dream/conflict.py` + `tests/test_dream_conflict.py`. Depends on
`slow_wave.config.DreamConfig`, `slow_wave.memory.{schema,stores}`,
`slow_wave.dream.schema`.

```python
from slow_wave.config import DreamConfig
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.dream.schema import ConflictResult

def resolve_conflicts(
    substrate: MemorySubstrate,
    *,
    dream_cfg: DreamConfig,
    now_order: int,
) -> ConflictResult:
    """Detect & resolve same-key contradictions; demote, don't destroy (FR4.7).

    Gather ACTIVE entries (episodic + semantic) that carry a fact, group by
    fact.key(). A group is a CONFLICT iff it holds >= 2 DISTINCT fact.value's.
    For each conflicting group keep ONE survivor and demote the rest via
    substrate.demote_entry(entry_id, reason="conflict_unlearning",
    at_order=now_order) — demote-not-delete (EC7). Survivor selection:
      - "older"  (default): keep the entry with the LARGEST (created_order,
        entry_id) — i.e. the latest value, matching the wake agent's
        latest-wins answer() so no probe answer changes.
      - "lower_salience": keep the entry with the HIGHEST salience.importance
        (tie-break larger created_order).
    Record n_conflicts_detected (number of conflicting groups), n_demoted, and
    demoted_entry_ids. No-op (zeros) when there are no conflicts. Deterministic."""
```
**Tests must cover (FR4.7):** seed an episodic store with two entries for the
same key but different values; `resolve_conflicts` reports 1 conflict, demotes 1
entry, and that entry is **recoverable from `substrate.archival`** (not deleted)
while the survivor stays active; a non-conflicting store (all distinct keys, or a
same-key/same-value pair) yields zeros; the `"older"` survivor equals the
latest-`created_order` entry; determinism.

---

## WS-ENGINE — two-phase cycle, gating, sleep-pressure (`slow_wave/dream/engine.py`)
Owns `slow_wave/dream/engine.py` + `tests/test_dream_engine.py`. **Wave 2** —
develop against the *already-landed* operator modules (import them directly).
Depends on: all five operator modules, `slow_wave.config.Config`,
`slow_wave.memory.{stores,schema}`, `slow_wave.dream.schema`,
`slow_wave.repro.seeding.derive_seed`, `slow_wave.llm.complete`.

```python
class DreamEngine:
    def __init__(self, cfg: "Config", *, llm_complete=None) -> None:
        """Hold cfg + the injectable llm_complete (defaults to slow_wave.llm.
        complete). Initialize self.telemetry = DreamTelemetry() and an internal
        last-cycle-order cursor (start -1) and churn accumulator."""
    cfg: "Config"
    telemetry: "DreamTelemetry"

    def sleep_hook(self, substrate, *, embedder, llm_complete, now_order,
                   task_index) -> "DreamCycleResult | None":
        """The WakeAgent sleep-window callback (matches the wake.py signature).
        Returns None (no cycle) unless gating says sleep. Gating (FR4.5/4.6):
          - if not cfg.dream.enabled: return None.
          - fixed: sleep iff (task_index + 1) % cfg.dream.sleep_every_n_tasks == 0.
          - adaptive: the fixed condition OR (sleep_pressure_churn_threshold > 0
            and churn-since-last-cycle >= threshold), churn = now_order -
            last_cycle_order. (Use the passed llm_complete if given, else the
            engine's.) On a sleep, call run_cycle(...) and return its result."""

    def run_cycle(self, substrate, *, embedder, llm_complete, now_order,
                  task_index) -> "DreamCycleResult":
        """Run ONE two-phase cycle and record it into self.telemetry.

        cycle_index = self.telemetry.n_cycles
        rng = np.random.default_rng(derive_seed(cfg.seed, f"dream_cycle_{cycle_index}"))
        candidates = episodic entries with created_order > last_cycle_order
                     (all episodics if first cycle).
        NREM:
          replay_res = replay(candidates, dream_cfg, rng, now_order,
                              recency_half_life=cfg.memory.recency_half_life)
                       if replay_enabled else None
          replayed = [episodic.get(id) for id in replay_res.sampled_ids()]
                     if replay_enabled else []
          transfer_source = replayed if replay_enabled else candidates
          transfer_res = transfer(substrate, transfer_source, ...) if transfer_enabled else None
          downscale_res = downscale(substrate, replayed_ids={e.entry_id for e in replayed},
                              now_order=now_order, dream_cfg=...) if downscale_enabled else None
          conflict_res = resolve_conflicts(substrate, ...) if conflict_enabled else None
        REM:
          augment_res = augment(substrate, (transfer_source or candidates), ...)
                        if augment_enabled else None
        Assemble DreamCycleResult(cycle_index, at_order=now_order, task_index,
          operators_run=[names in run order], per-op results, summed
          api_calls/input_tokens/output_tokens across transfer+augment). Update
          last_cycle_order=now_order, reset churn. self.telemetry.record(result).
          ALL 16 on/off combinations must run without error (EC1)."""
```
**Tests must cover (EC1 + gating + sleep-pressure):** **EC1** — enumerate all
**2⁴ = 16** combinations of (replay, transfer, downscale, augment) toggles;
build a tiny substrate (a handful of fact-bearing episodics), run `run_cycle`
for each combo, and assert it returns a `DreamCycleResult` whose `operators_run`
exactly matches the enabled set (and the all-off combo yields `operators_run ==
[]`); also explicitly run the 2×2 (replay × downscale) sub-matrix. Gating:
`sleep_hook` returns None when `(task_index+1) % sleep_every_n_tasks != 0` and a
cycle when it divides; `enabled=False` ⇒ always None; adaptive mode triggers on
churn ≥ threshold. Determinism: two `run_cycle`s on identical fresh substrates →
identical sampled/written/pseudo ids and salience.

---

## Integration (orchestrator-owned — do NOT implement in a workstream)
After Wave 1 (operators) + Wave 2 (engine) land, the orchestrator writes
`slow_wave/dream/runner.py` (`run_dream(cfg, out_dir=None) -> Path`: seed →
stream → probes → embedder → `WakeAgent(cfg, embedder).run(stream, probe_set,
sleep_hook=DreamEngine(cfg).sleep_hook)` → manifest carrying `R[i,j]`,
continual metrics, footprint, **wake + dream telemetry**, and a provenance/
archival audit; `cost.api_calls`/`cost.tokens` summed over wake+dream; one
command `python -m slow_wave.dream.runner --config configs/dream_smoke.yaml`),
the `slow_wave/dream/__init__.py` exports, the cross-module tests
`tests/test_dream_matrix.py` (EC1 over the engine via a real Phase-1 stream),
`tests/test_dream_integration.py` (EC3 gating end-to-end: semantic empty
mid-wake / populated post-sleep; EC7 full cycle on a Phase-1 stream leaves
provenance + archival intact with no hard deletes; confound re-check over live
entries), the `repro-dream` Makefile target, and the README/CONTRACT note. Keep
modules import-clean so this glue is mechanical.
