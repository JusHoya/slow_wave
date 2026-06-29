# Phase 4 — Shared Interface Contract (authoritative)

This file pins the cross-module interfaces for **Phase 4 (Evaluation Harness,
Control Battery & Preregistration)** so independently-authored modules integrate
without guessing. **Do not deviate from these signatures.** Import shared
Phase 0–3 types from their modules — do not redefine them.

## Objective & exit criteria (PRD §8, Phase 4)
Wire all control arms into one harness at matched budgets, implement the full
metric + statistics suite, and commit the preregistration. The work must
satisfy, verbatim:

- **EC1** **All nine arms** (`no_sleep`, `replay_only`, `downscale_only`,
  `random_pruning`, `full_dream`, `reflection`, `oracle`, `long_context`, `aa`)
  instantiate and run on the same stream via one harness (test runs each ≥ once).
- **EC2** The **A/A control** yields no significant difference between two
  identical configs (different seeds) on the primary endpoint — the noise floor.
- **EC3** The **oracle** arm achieves higher prune precision/recall than any
  non-oracle arm on a distractor-heavy stream (sanity ceiling check).
- **EC4** **Matched-budget controller** equalizes tokens/retrieval/memory-size
  across arms within tolerance and **records actuals**; where unmatched, a
  **Pareto frontier** is produced instead (test).
- **EC5** Metrics module computes ACC, BWT, FWT, per-task FM from `R[i,j]`,
  **plus** prune precision/recall/F1 and the decay-vs-relevance **calibration
  curve** — all decoupled and emitted as JSON (schema test).
- **EC6** Statistics module produces bootstrap 95% CIs, rliable
  IQM/performance-profiles/probability-of-improvement, Wilcoxon + Friedman/
  post-hoc, and standardized effect sizes with CIs and multiple-comparison
  correction (unit tests on synthetic inputs with **known answers**).
- **EC7** **Preregistration committed with a git hash** *before* any real long
  run: it names H1, H0, the single primary endpoint, seed plan + power analysis,
  tests, and explicit rejection criteria. Analysis code **refuses** to compute a
  non-preregistered primary endpoint (test).
- **EC8** **Temperature-0 stability control** quantifies run-to-run variance of
  the dream summarizer; **memory-drift detector** flags degradation from repeated
  summarization (both produce numbers on a tiny run).

Maps to FR5.1–FR5.6, DX2 (no silent caps), DX3 (falsifiability in code), DX6
(reusability).

## Ground rules (inherited from Phase 0–3 — non-negotiable)
- Python **3.12** target (support 3.11+). Run tests with
  `.venv/Scripts/python.exe -m pytest`.
- pydantic **v2**. `extra="forbid"` on every structured model. Reuse the shared
  Phase 0–3 types (`Config`/`EvalConfig`, `Stream`/`AccuracyMatrix`/`Probe`,
  `MemoryEntry`/`MemoryFootprint`/`MemorySubstrate`, `ContinualMetrics`,
  `DreamEngine`/`DreamTelemetry`, `LLMResult`) — **do not redefine them**.
- **Determinism is sacred (DX1).** All randomness via
  `numpy.random.default_rng(derive_seed(seed, "<name>"))` or an explicitly-seeded
  `default_rng(stats_seed)` — never Python `hash()` or the `np.random` global.
  Given a fixed config+seed **under the mock LLM**, every structured result
  (arm metrics, prune counts, bootstrap CIs, test statistics, control numbers) is
  **byte-identical** across two runs. The only nondeterministic outputs are
  wall-clock timing and (once a real Claude call is made) token counts /
  generated text.
- **LEAN CORE, NUMPY-ONLY STATS (load-bearing for Phase 4).** CI installs only
  `.[dev]` = `pydantic`, `pyyaml`, `numpy`, `anthropic`, `pytest`. **`scipy`,
  `statsmodels`, `rliable`, and `matplotlib` are NOT available in CI** (and
  rliable/statsmodels are not in the dev venv). Therefore **every Phase 4 module
  imports only `numpy` (+ stdlib + pydantic + slow_wave)**. Implement bootstrap
  CIs, IQM, performance profiles, probability of improvement, Wilcoxon
  signed-rank, Friedman, post-hoc, Holm correction, and effect sizes **from
  scratch in numpy**, citing the source papers (Agarwal et al. 2021 / Demšar 2006
  / Holm 1979 / Lopez-Paz & Ranzato 2017) — exactly as `eval/metrics.py` already
  implements CL metrics in plain Python citing its papers. **Never `import scipy`,
  `import statsmodels`, `import rliable`, or `import matplotlib` in module code or
  in a test that is not guarded by `pytest.importorskip`.** A separate *optional*
  cross-check test MAY `pytest.importorskip("scipy")` to validate against scipy,
  but the suite must pass green with scipy absent.
- **Confound guard (FR1.6) is sacred.** Only the **`oracle`** arm and the
  **offline scoring** functions (`prune_metrics`, the probe scorer) may read
  ground-truth labels, and only via `slow_wave.stream.schema.offline_labels`.
  Every other arm's online path must never read `offline_labels`,
  `stream.ground_truth`, `probe.answer`, or any `guard.BANNED_FIELD_NAMES` field.
  `ArmSpec.uses_labels` is `True` **only** for `oracle`; the harness asserts it.
- **Honesty by construction (DX2).** Anything bounded (a Pareto-dropped arm, a
  mismatched budget, a coverage cap, a skipped cell) is recorded in a result
  field **and** `logger.info/warning`-logged. Never drop silently.
- **Import by full module path** (`from slow_wave.eval.stats import bootstrap_ci`),
  NOT via the package root. The orchestrator owns `slow_wave/eval/__init__.py`,
  `slow_wave/eval/harness.py`, `slow_wave/eval/runner.py`,
  `slow_wave/eval/schema.py`, `slow_wave/config.py`, the `configs/eval_*.yaml`,
  `prereg/preregistration.yaml`, and the cross-module integration tests —
  **do not edit those**.
- Google-style docstrings on every public function/class, matching the density
  and tone of the existing Phase 0–3 modules.

## Already scaffolded by the orchestrator (consume as-is — do NOT edit)
- **`slow_wave/config.py`** now exposes **`EvalConfig`**, wired onto `Config` as
  `cfg.eval`. Field names are authoritative (read the class docstring for full
  docs): `arms:list[str]` (the nine names), `seeds:list[int]=[0,1,2,3,4]`,
  `match_budget:bool=True`, `budget_tolerance:float=0.15`,
  `token_budget:int|None=None`, `bootstrap_resamples:int=10000`,
  `ci_level:float=0.95`, `stats_seed:int=0`,
  `prereg_path:str="prereg/preregistration.yaml"`,
  `primary_endpoint:str="acc_diff_full_dream_vs_no_sleep"`,
  `stability_repeats:int=3`, `drift_rounds:int=3`,
  `aa_reference_arm:str="no_sleep"`, `treatment_arm:str="full_dream"`,
  `baseline_arm:str="no_sleep"`.
- **`slow_wave/eval/schema.py`** — the shared result models (consume; do not
  redefine): `ArmSpec`, `ArmCost`, `LabelCount`, `PruneQuality`,
  `CalibrationBin`, `CalibrationCurve`, `ArmResult`, `ArmBudgetActuals`,
  `ParetoPoint`, `BudgetReport`, `BootstrapCI`, `EffectSize`, `TestResult`,
  `Comparison`, `RobustAggregate`, `StatsReport`, `Preregistration`,
  `StabilityResult`, `DriftResult`, `AAResult`, `PrimaryEndpoint`,
  `ExperimentResult`. **Read the field docstrings — they define the exact
  fields/counters you must populate.**
- **`prereg/preregistration.yaml`** — the committed registered-report artifact.
  Its `primary_endpoint` is `acc_diff_full_dream_vs_no_sleep`. It parses into
  `Preregistration`.
- **`configs/eval_smoke.yaml`** (tiny 3-task / 3-seed grid) and
  **`configs/eval_full.yaml`** (5-task / 5-seed). Both distractor-heavy
  (`label_mix: signal .34 / distractor .40 / noise .26`) with a bounded episodic
  store so the no-sleep baseline forgets and arms diverge.
- Reuse from earlier phases: `slow_wave.stream.{generator,probes,schema,guard}`,
  `slow_wave.memory.{schema,stores,salience,retrieval}`,
  `slow_wave.agent.wake.WakeAgent` (`.run(stream, probe_set, *, sleep_hook=None)
  -> WakeResult` with `.accuracy_matrix/.metrics/.footprint/.telemetry/
  .substrate`), `slow_wave.dream.engine.DreamEngine` (`.sleep_hook`, `.telemetry`
  with `.n_cycles/.input_tokens/.output_tokens/.api_calls/.n_pseudo` and per-cycle
  `.cycles[i].augment.fidelity.mean_fidelity` — there is NO single roll-up
  fidelity field, so mean generator fidelity = mean of
  `c.augment.fidelity.mean_fidelity` over cycles where `c.augment is not None and
  c.augment.n_pseudo > 0`, else `None`),
  `slow_wave.eval.metrics.compute_continual_metrics`,
  `slow_wave.repro.seeding.derive_seed`, `slow_wave.embeddings.get_embedder`,
  `slow_wave.llm.complete`.

## The nine arms — authoritative toggle mapping (WS-ARMS implements)
Each arm is a named `ArmSpec` whose `config_overrides` deep-merge onto the base
config. Three arms (`random_pruning`, `oracle`) attach a **custom sleep hook**
instead of the `DreamEngine`; `long_context` only changes memory. Pairing in the
experiment is **by seed**: for each seed the harness generates ONE stream from
`derive_seed(seed,"stream")` and runs every arm on it (same stream within a seed;
both stream and sampling vary across seeds, FR5.4).

| arm | family | overrides / behavior |
|---|---|---|
| `no_sleep` | baseline | `dream.enabled=False`. Wake-only; the forgetting reference. |
| `replay_only` | ablation | `dream.enabled=True; replay=on, downscale=on, transfer=off, augment=off, conflict=off`. "Mere re-exposure": replay samples recent episodics, downscale re-potentiates the sampled (decay all, protect replayed) so they survive eviction — but NO LLM summarization to semantic and NO REM augment (§9.1#1). |
| `downscale_only` | ablation | `dream.enabled=True; replay=off, transfer=off, downscale=on, augment=off`. Pure SHY homeostasis: global decay, re-potentiate nothing. |
| `random_pruning` | control | `dream.enabled=True; all four operators off` + **`make_random_prune_hook`**: each sleep window demotes a random fraction (default 0.5) of active entries, blind to salience AND labels (the ground-truth-blind negative control). |
| `full_dream` | treatment | `dream.enabled=True; replay+transfer+downscale+augment all on` (the base config defaults). The treatment. |
| `reflection` | control | `dream.enabled=True; replay=on (strategy=uniform), transfer=on (cls_interleave=off), downscale=off, augment=off`. Generative-Agents shallow synthesis (Park et al. 2023): uniformly sample recent, summarize into semantic, no homeostasis/interleave/REM. |
| `oracle` | ceiling | `dream.enabled=True; all four operators off`, **`uses_labels=True`** + **`make_oracle_prune_hook`**: each sleep window demotes EXACTLY the active entries whose source item's ground-truth label is `distractor`/`noise` (reads `offline_labels` — the one sanctioned label use; the prune-quality ceiling). |
| `long_context` | ceiling | `dream.enabled=False; memory.episodic_capacity=0` (unbounded → never forgets → whole stream effectively in context; max footprint/cost — the key validity threat). |
| `aa` | control | builds the `aa_reference_arm` (default `no_sleep`) config verbatim. The harness runs it under two seeds for the noise floor (EC2). |

---

## WS-STATS — statistics suite (`slow_wave/eval/stats.py`)
Owns `slow_wave/eval/stats.py` + `tests/test_eval_stats.py`. **numpy + stdlib +
`slow_wave.eval.schema` only — NO scipy/statsmodels/rliable.** Pure, deterministic
functions; every randomized routine takes an explicit `rng:
np.random.Generator`.

```python
import numpy as np
from slow_wave.eval.schema import (BootstrapCI, EffectSize, TestResult,
    Comparison, RobustAggregate)

def bootstrap_ci(samples, *, statistic="mean", level=0.95, n_resamples=10000,
                 rng, name=None) -> BootstrapCI:
    """Percentile bootstrap CI of a statistic over `samples` (Efron). statistic
    in {"mean","median","iqm"}. Deterministic given rng. n<2 -> degenerate CI
    (lo==hi==point)."""

def paired_diff_ci(a, b, *, level=0.95, n_resamples=10000, rng) -> BootstrapCI:
    """Percentile bootstrap CI of mean(a-b) over paired samples (statistic
    'paired_mean_diff'); resample pair indices jointly."""

def iqm(samples) -> float:
    """Interquartile mean: mean of the middle 50% (Agarwal et al. 2021)."""

def robust_aggregate(arm_name, samples, *, level=0.95, n_resamples=10000, rng
                     ) -> RobustAggregate:
    """IQM, median, mean — each with a bootstrap_ci."""

def cohens_d(a, b, *, paired=True) -> float: ...
def effect_size_cohens_d(a, b, *, paired=True, level=0.95, n_resamples=10000,
                         rng) -> EffectSize:
    """Standardized mean difference with bootstrap CI; magnitude by |d|<0.2
    negligible / <0.5 small / <0.8 medium / else large."""

def cliffs_delta(a, b) -> float: ...
def effect_size_cliffs_delta(a, b, *, level=0.95, n_resamples=10000, rng
                             ) -> EffectSize:
    """Non-parametric effect; magnitude by |d|<.147 negligible/.33 small/.474
    medium/else large (Romano et al.)."""

def wilcoxon_signed_rank(a, b) -> TestResult:
    """Paired Wilcoxon signed-rank (zero-differences dropped; normal
    approximation with tie+continuity correction; two-sided p). test=
    "wilcoxon_signed_rank"."""

def friedman(groups) -> TestResult:
    """Friedman repeated-measures omnibus over k>=3 aligned arms (chi-square
    approx with ties correction; detail has df). test="friedman"."""

def holm_correction(p_values, *, alpha=0.05) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down: return (adjusted_p, reject) per input, order
    preserved (Holm 1979)."""

def probability_of_improvement(a, b) -> float:
    """rliable P(a>b) over all cross pairs, ties=0.5 (Agarwal 2021)."""

def performance_profile(samples, taus) -> list[list[float]]:
    """[[tau, fraction of runs with score>=tau], ...] over the given taus."""
```
**Tests must cover (EC6):** bootstrap mean CI brackets the true mean on a known
sample and is deterministic given a seeded rng; `iqm` equals a hand value on a
small array (e.g. `[1..8]` -> mean of `3,4,5,6` = 4.5); `cohens_d` matches a
hand-computed standardized difference; `wilcoxon_signed_rank` matches a tiny
hand/known case (and, in a separate `importorskip("scipy")` test only, agrees
with `scipy.stats.wilcoxon` within tolerance); `friedman` matches a known case;
`holm_correction` matches the step-down ordering on a known p-vector;
`probability_of_improvement` is 1.0 when a strictly dominates b, 0.5 on ties;
`performance_profile` is non-increasing in tau; all randomized fns are
byte-identical across two runs with the same `stats_seed`.

---

## WS-METRICS — mechanism-level prune metrics (`slow_wave/eval/prune_metrics.py`)
Owns `slow_wave/eval/prune_metrics.py` + `tests/test_eval_prune_metrics.py`.
Depends on `slow_wave.stream.schema` (incl. `offline_labels`),
`slow_wave.memory.{schema,stores}`, `slow_wave.eval.schema`, `numpy`. This is
**offline scoring** — it MAY read labels via `offline_labels` only.

```python
from slow_wave.stream.schema import Stream, Label, offline_labels
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.eval.schema import PruneQuality, CalibrationCurve, LabelCount

def retained_item_ids(stream, substrate) -> set[str]:
    """Item_ids with a LIVE representation in active memory (episodic OR
    semantic). Rule (deterministic):
      valid = {it.item_id for it in stream.items}
      order_to_item = {it.order: it.item_id for it in stream.items}
      for e in substrate.episodic.all_entries():
          if e.provenance and e.provenance[0] in valid: add e.provenance[0]
              # excludes augment pseudo-episodes (provenance[0] is an entry_id)
      for s in substrate.semantic.all_entries():
          iid = order_to_item.get(s.created_order)   # transfer preserves source order
          if iid is not None: add iid
    """

def prune_quality(stream, substrate) -> PruneQuality:
    """Offline consolidation quality vs ground truth (FR5.3). 'Positive' class =
    PRUNED; a correct prune targets a distractor/noise item, a signal should be
    retained. Over ALL stream items (one label each):
      retained = retained_item_ids(stream, substrate); pruned = the rest.
      tp = pruned & (distractor|noise); fp = pruned & signal;
      fn = retained & (distractor|noise); tn = retained & signal.
      precision = tp/(tp+fp) or 0; recall = tp/(tp+fn) or 0;
      f1 = 2PR/(P+R) or 0; signal_retention = tn/(tn+fp) or 0.
    Populate retained_by_label/pruned_by_label (LabelCount) and all counts."""

def calibration_curve(stream, substrate, *, n_bins=10) -> CalibrationCurve:
    """Decay/salience-vs-true-relevance reliability curve (FR5.3). Each item's
    score = max salience.importance over its active representations (0 if none),
    min-max normalized to [0,1] across items. Bin into n_bins equal-width [0,1]
    bins; per bin report n, mean_salience, frac_signal. expected_calibration_error
    = sum_b (n_b/N)*|mean_salience_b - frac_signal_b|. Empty/degenerate -> a
    well-formed zero curve (no raise)."""
```
**Tests must cover (EC5):** a hand-built substrate where exactly the distractor/
noise items are pruned and all signals retained yields `precision==recall==f1==
1.0` and `signal_retention==1.0`; the opposite (signals pruned, noise kept)
yields `recall==0`; counts satisfy `tp+fp+fn+tn == n_items` and
`tp+fp==n_pruned`, `tp+fn==(#distractor+#noise)`; the prune metrics are computed
over `offline_labels` only (a probe of the code path); `calibration_curve` bins
sum to `n_items`, ECE in `[0,1]`, and is higher-`frac_signal` in higher-salience
bins for a well-calibrated synthetic substrate; everything is JSON-dumpable and
deterministic; empty substrate is safe.

---

## WS-ARMS — the nine control arms (`slow_wave/eval/arms.py` + `arm_ops.py`)
Owns `slow_wave/eval/arms.py`, `slow_wave/eval/arm_ops.py`,
`tests/test_eval_arms.py`. Depends on `slow_wave.config`,
`slow_wave.dream.engine.DreamEngine`, `slow_wave.agent.wake.WakeAgent`,
`slow_wave.memory.{stores,schema}`, `slow_wave.stream.schema` (+ `offline_labels`
for the oracle hook only), `slow_wave.repro.seeding.derive_seed`,
`slow_wave.eval.schema`, `numpy`.

### `arm_ops.py` — the two custom sleep-window operators
```python
from dataclasses import dataclass, field
import numpy as np
from slow_wave.config import Config
from slow_wave.memory.stores import MemorySubstrate
from slow_wave.stream.schema import Stream, Label, offline_labels

@dataclass
class PruneTelemetry:
    n_cycles: int = 0
    n_demoted: int = 0
    demoted_ids: list[str] = field(default_factory=list)

def make_random_prune_hook(cfg: Config, *, seed: int, telemetry: PruneTelemetry,
                           prune_fraction: float = 0.5):
    """Return a WakeAgent sleep hook (signature
    `(substrate, *, embedder, llm_complete, now_order, task_index)`) that, at each
    SCHEDULED window ((task_index+1) % cfg.dream.sleep_every_n_tasks == 0),
    demotes round(prune_fraction * n_active) active entries chosen UNIFORMLY AT
    RANDOM (blind to salience and labels) via substrate.demote_entry(reason=
    "random_pruning"). Deterministic: rng = default_rng(derive_seed(seed,
    f"random_prune_{telemetry.n_cycles}")). Increments telemetry. Demote-not-delete
    (no hard deletes). Logs the count (DX2). Never reads labels."""

def make_oracle_prune_hook(cfg: Config, stream: Stream, *,
                           telemetry: PruneTelemetry):
    """Return a WakeAgent sleep hook that, at each scheduled window, demotes
    EXACTLY the active entries whose SOURCE item's ground-truth label is
    distractor/noise. labels = offline_labels(stream); for each active episodic
    entry e with e.provenance[0] a stream item_id, if labels[item_id] in
    {DISTRACTOR, NOISE}: substrate.demote_entry(reason="oracle_prune"). Reads
    labels — the ONE sanctioned use (ArmSpec.uses_labels=True). Deterministic
    (iterate entries in insertion order). Increments telemetry."""
```

### `arms.py` — registry + builder
```python
from dataclasses import dataclass
from typing import Callable
from slow_wave.config import Config
from slow_wave.dream.engine import DreamEngine
from slow_wave.stream.schema import Stream
from slow_wave.eval.schema import ArmSpec

ARM_REGISTRY: dict[str, ArmSpec]   # all nine specs (see toggle table above)

@dataclass
class RunnableArm:
    spec: ArmSpec
    cfg: Config                      # effective config (overrides applied; .seed=seed)
    sleep_hook: Callable | None      # WakeAgent sleep-window callback, or None
    def n_cycles(self) -> int: ...           # dream cycles or prune cycles run
    def dream_tokens(self) -> tuple[int,int,int]: ...  # (input,output,api_calls); (0,0,0) for custom-hook/no-dream arms
    def generator_fidelity(self) -> float | None: ...  # mean augment fidelity, or None

def build_arm(name: str, base_cfg: Config, stream: Stream, seed: int
              ) -> RunnableArm:
    """Materialize arm `name`: deep-merge ARM_REGISTRY[name].config_overrides onto
    base_cfg (a copy), set effective.seed=seed, then attach the sleep machinery:
      - dream-driven arms: engine=DreamEngine(effective_cfg); sleep_hook=
        engine.sleep_hook if effective_cfg.dream.enabled else None; n_cycles/
        dream_tokens/generator_fidelity read engine.telemetry.
      - random_pruning: PruneTelemetry + make_random_prune_hook; dream_tokens=(0,0,0).
      - oracle: PruneTelemetry + make_oracle_prune_hook(effective_cfg, stream);
        dream_tokens=(0,0,0); asserts spec.uses_labels.
      - long_context / no_sleep: dream disabled -> sleep_hook=None.
      - aa: build_arm(aa_reference_arm,...) config verbatim.
    Deep-merge must not mutate base_cfg. Unknown name -> KeyError (caller guards)."""
```
**Tests must cover (EC1 + EC3 support):** `ARM_REGISTRY` has exactly the nine
names; only `oracle` has `uses_labels=True`; `build_arm` for every arm returns a
`RunnableArm` whose `cfg` validates and whose toggles match the table; **each arm
runs end-to-end** — `WakeAgent(arm.cfg, embedder).run(stream, probe_set,
sleep_hook=arm.sleep_hook)` returns a `WakeResult` without error on a Phase-1
stream (this is the WS-local EC1 check); `random_pruning` and `oracle` demote
entries to archival (no hard deletes; recoverable) and `oracle` demotes only
distractor/noise sources (a seeded distractor-heavy stream: every demoted item's
label ∈ {distractor,noise}); `long_context` never evicts (archival empty,
episodic holds all fact items); determinism (same seed → identical sampled/
demoted ids). Do NOT compute prune precision/recall here (the harness does).

---

## WS-BUDGET — matched-budget controller + Pareto (`slow_wave/eval/budget_match.py`)
Owns `slow_wave/eval/budget_match.py` + `tests/test_eval_budget_match.py`.
Depends on `slow_wave.eval.schema`, `numpy`, stdlib.

```python
from slow_wave.eval.schema import ArmCost, ArmBudgetActuals, ParetoPoint, BudgetReport

def pareto_frontier(points) -> list[ParetoPoint]:
    """points: list of (arm_name, accuracy, compute_tokens, memory_vectors). A
    point is on the frontier iff no other point has accuracy >= it AND
    compute_tokens <= it with at least one strict (maximize accuracy, minimize
    compute). Return ParetoPoint per input with on_frontier set; deterministic
    (stable order)."""

def match_budget(arm_costs, *, accuracies, tolerance=0.15, target_tokens=None,
                 exclude=("oracle","long_context","aa")) -> BudgetReport:
    """arm_costs: dict[arm_name -> list[ArmCost]] (one per seed). accuracies:
    dict[arm_name -> list[float]] (per-seed ACC) for the Pareto axis.
      - target_tokens: if None, the MEDIAN over included arms of each arm's mean
        total_tokens (robust common target); record target_retrieval/
        target_memory_vectors analogously (median of means).
      - per_arm: ArmBudgetActuals for EVERY arm (including excluded; their
        *_within_tolerance reflect the comparison but they don't gate `matched`).
        within_tolerance := |mean - target| <= tolerance * max(target, 1).
      - matched := all INCLUDED (non-excluded) arms matched on all three axes.
      - pareto := pareto_frontier over (arm, mean ACC, mean tokens, mean vectors)
        for ALL arms — ALWAYS produced (FR5.2: the reported artifact when matching
        is infeasible). notes: DX2 lines naming excluded ceilings."""
```
**Tests must cover (EC4):** with hand-built `ArmCost`s where every included arm
is within tolerance of the target, `matched is True` and `target_tokens` equals
the median-of-means; perturbing one included arm beyond tolerance flips
`matched` to False and that arm's `tokens_within_tolerance` to False while
`per_arm` still records its actuals; `pareto_frontier` on a known set marks the
dominating points on the frontier and dominated ones off it (e.g. a high-acc
low-cost point dominates a low-acc high-cost one); excluded ceilings appear in
`per_arm`/`pareto` but never flip `matched`; deterministic + JSON-dumpable.

---

## WS-PREREG — preregistration guard + bias controls (`prereg.py` + `controls.py`)
Owns `slow_wave/eval/prereg.py`, `slow_wave/eval/controls.py`,
`tests/test_eval_prereg.py`, `tests/test_eval_controls.py`. Depends on
`slow_wave.eval.{schema,stats}`, `slow_wave.config`, `slow_wave.llm.complete`,
`slow_wave.embeddings`, `numpy`, `pyyaml`, stdlib.

### `prereg.py`
```python
from slow_wave.eval.schema import (Preregistration, PrimaryEndpoint, AAResult,
    ArmResult, BootstrapCI, EffectSize, TestResult)

class NonPreregisteredEndpointError(ValueError): ...

def load_preregistration(path) -> Preregistration:
    """Parse the committed YAML artifact into a Preregistration (extra='forbid')."""

def assert_primary_endpoint(prereg, requested_name) -> None:
    """Raise NonPreregisteredEndpointError if requested_name != prereg
    .primary_endpoint (DX3: analysis refuses a non-preregistered endpoint)."""

def compute_primary_endpoint(prereg, arm_results, *, requested_name, aa,
                             ci_level=0.95, n_resamples=10000, rng
                             ) -> PrimaryEndpoint:
    """FIRST call assert_primary_endpoint(prereg, requested_name). Pair
    treatment_arm vs baseline_arm ArmResults BY SEED on final ACC
    (continual_metrics.acc); value = mean paired diff (treatment-baseline);
    difference_ci = stats.paired_diff_ci; effect = stats.effect_size_cohens_d
    (paired); test = stats.wilcoxon_signed_rank; noise_floor = aa.abs_difference;
    exceeds_noise_floor = |value|>noise_floor; verdict = "confirmed" iff value>0
    and CI excludes 0 and exceeds_noise_floor else "refuted" if CI includes 0,
    else "inconclusive". Deterministic given rng."""
```

### `controls.py`
```python
from slow_wave.eval.schema import StabilityResult, DriftResult

def temperature_zero_stability(cfg, *, llm_complete=None, embedder=None,
                               n_repeats=3, source_text=None) -> StabilityResult:
    """Call the summarizer n_repeats times on the SAME input (default a fixed
    synthetic episode text). distinct_outputs/identical over the texts;
    mean_pairwise_similarity = mean cosine over repeat embeddings; token_cv =
    std/mean of output token counts. Under the mock LLM: identical=True,
    distinct_outputs=1, similarity=1.0, token_cv=0.0 (EC8 numbers)."""

def memory_drift(cfg, *, llm_complete=None, embedder=None, n_rounds=3,
                 source_text=None) -> DriftResult:
    """Re-summarize the previous round's output n_rounds times; fidelity_per_round
    = cosine(round_output_emb, ORIGINAL source_emb); faithfulness = last round's
    fidelity; monotonic_decline = fidelities non-increasing; degraded =
    (fidelity[0]-fidelity[-1]) > drift_threshold (default 0.15). Produces numbers
    on a tiny run under the mock LLM (EC8)."""
```
**Tests must cover (EC7 + EC8):** `load_preregistration` parses the committed
artifact and its `primary_endpoint=='acc_diff_full_dream_vs_no_sleep'`;
`assert_primary_endpoint` raises `NonPreregisteredEndpointError` for any other
name and is silent for the right one; `compute_primary_endpoint` on synthetic
ArmResults where treatment strictly beats baseline yields `verdict=="confirmed"`
with `value>0`, and on equal arms yields a non-confirmed verdict with a CI
spanning 0; it RAISES for a wrong `requested_name` (DX3); `temperature_zero_
stability` and `memory_drift` return well-formed results with the mock-LLM
numbers above and are deterministic + JSON-dumpable.

---

## Integration (orchestrator-owned — do NOT implement in a workstream)
After the five workstreams land, the orchestrator writes
`slow_wave/eval/harness.py` (`run_arm(...) -> ArmResult` building each arm via
`build_arm`, running the `WakeAgent`, computing continual metrics + `prune_quality`
+ `calibration_curve` + `ArmCost` from wake+dream telemetry; `run_experiment(cfg,
out_dir) -> Path` generating one stream per seed, running the nine-arm grid,
matching budgets, computing the stats suite, the A/A control, the prereg primary
endpoint, and the stability + drift controls, then writing a single
`ExperimentResult` into a run manifest under `<out>/eval/manifest.json`),
`slow_wave/eval/runner.py` (one command `python -m slow_wave.eval.runner --config
configs/eval_smoke.yaml`), `slow_wave/eval/__init__.py` exports, the `repro-eval`
Makefile target, the cross-module tests `tests/test_eval_harness.py` (EC1 nine
arms run; EC2 A/A noise floor; EC3 oracle ceiling; EC4 matched budget/Pareto; the
ExperimentResult round-trips to JSON) and `tests/test_eval_matrix.py` if needed,
and the README/CONTRACT note. Keep modules import-clean so this glue is
mechanical.
