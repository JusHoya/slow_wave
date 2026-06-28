# Phase 1 — Shared Interface Contract (authoritative)

This file pins the cross-module interfaces for **Phase 1 (Synthetic Continual
Task Stream Generator)** so independently-authored modules integrate without
guessing. **Do not deviate from these signatures.** The shared data model is
already implemented in `slow_wave/stream/schema.py` — import from there; do not
redefine types.

## Ground rules (inherited from Phase 0)
- Python **3.12** target (support 3.11+). Run tests with `.venv/Scripts/python.exe -m pytest`.
- pydantic **v2**. For any model with a field literally named `model`, set
  `model_config = ConfigDict(protected_namespaces=())`. Use `extra="forbid"` on
  config-like models; use `frozen=True` for immutable value objects.
- **Determinism is sacred.** Seed every RNG explicitly via
  `numpy.random.default_rng(derive_seed(seed, "<name>"))` from
  `slow_wave.repro.seeding`. Never use Python's salted `hash()` for anything that
  reaches an artifact; use `hashlib` (blake2b/sha256). Serialize JSON with
  `sort_keys=True`. All emitted artifacts must be **byte-identical** across two
  runs with the same seed + config.
- **Lean core deps only**: `pydantic`, `pyyaml`, `numpy`. No torch / sentence-
  transformers / scipy in Phase 1 (stream items carry text, not embeddings).
- **Import by full module path** (`from slow_wave.stream.generator import ...`),
  NOT via the package root — the orchestrator owns `slow_wave/stream/__init__.py`
  and will wire exports at integration time. Do not edit `__init__.py`,
  `config.py`, or `schema.py`; they are orchestrator-owned.
- Add Google-style docstrings to every public function/class, matching the
  density and tone of existing Phase 0 modules.

## The synthetic domain (shared mental model)
A stream is an ordered sequence of `StreamItem`s partitioned into `n_tasks`
contiguous **task segments** (`task_index` 0..n_tasks-1, `items_per_task` each).
Items assert synthetic `(subject, attribute, value)` facts in natural-language
surface form:
- **signal** — asserts a fact about a *probed* `(subject, attribute)` (mission-
  relevant; the probe set queries it).
- **distractor** — asserts a plausibly-formed fact about a subject/attribute from
  a disjoint "distractor namespace" that is **never** probed (plausible but
  irrelevant). Surface form is indistinguishable from signal.
- **noise** — a random token salad; `fact=None` (irrelevant, structureless).

Relevance labels (`signal`/`distractor`/`noise`) are **OFFLINE-ONLY**: they live
in `Stream.ground_truth` and are read only via `schema.offline_labels(stream)`.
A `StreamItem` has **no** label field by construction (FR1.6).

## Shared types (already in `slow_wave/stream/schema.py` — DO NOT redefine)
- Enums: `Label{SIGNAL,DISTRACTOR,NOISE}`, `CLScenario{TASK_INCREMENTAL,
  DOMAIN_INCREMENTAL,CLASS_INCREMENTAL}`, `ItemKind{INGEST,TASK}`.
- `LabelMix(signal, distractor, noise)` — proportions summing to ~1.0 (validated).
- `StreamGenConfig` — all generation knobs (scenario, n_tasks, items_per_task,
  label_mix, contradiction_rate, drift, recency_bias, n_subjects_per_task,
  n_attributes, n_values, probes_per_task, distractor_namespace_size,
  noise_vocab_size, noise_tokens). Has `.canonical_dict()`.
- `Fact(subject, attribute, value)` (frozen) with `.key() -> (subject, attribute)`.
- `StreamItem(item_id, order, kind, task_index, content, fact|None)` (frozen, **no
  label**).
- `GroundTruth(labels: dict[item_id, Label])` (offline-only sidecar).
- `Stream(stream_id, scenario, seed, n_tasks, config, items, ground_truth)` —
  validates every item has exactly one label.
- `Probe(probe_id, task_index, subject, attribute, answer, available_after_order,
  query, task_id_visible)` (frozen).
- `ProbeSet(scenario, n_tasks, probes)`.
- `AccuracyMatrix(n_tasks, scenario, R: list[list[float]])` — validates square
  shape and entries in [0,1].
- Helpers: `offline_labels(stream)`, `assert_single_scenario(stream)`,
  `assert_same_scenario(streams) -> CLScenario` (raises on cross-scenario mixing).

---

## WS1 — Generator  (`slow_wave/stream/generator.py` + `tests/test_stream_generator.py`)

Implement the deterministic generator. **Public API (exact signatures):**

```python
def generate_stream(config: StreamGenConfig, seed: int) -> Stream: ...
```

Behavior:
1. **Deterministic.** All randomness from `numpy.random.default_rng(derive_seed(
   seed, "<purpose>"))` (e.g. purposes `"labels"`, `"subjects"`, `"values"`,
   `"order"`, `"noise"`). Same `(config, seed)` → identical `Stream` (and thus
   identical JSON). `stream_id = f"{config.scenario.value}-s{seed}-{h}"` where
   `h` = first 8 hex of `sha256` over `json.dumps({"seed":seed, "config":
   config.canonical_dict()}, sort_keys=True)`.
2. **Items & tasks.** Produce `n_tasks * items_per_task` ingest items
   (`kind=ItemKind.INGEST`), `order` = global 0-based index, `task_index` =
   `order // items_per_task`. `item_id = f"i{order:06d}"`.
3. **Labels (FR1.1, exit #2).** Per task, allocate label counts by rounding
   `label_mix` proportions to `items_per_task` (largest-remainder method so they
   sum exactly), then place them in a deterministic shuffled order within the
   task. Every item gets **exactly one** label, stored in
   `GroundTruth.labels[item_id]`. Empirical mix must match requested within
   tolerance (your test checks ≥3 distractor ratios, e.g. 0.1/0.3/0.5).
4. **Facts.**
   - signal → `Fact` over a **probed** namespace: subjects `f"subj_{task_index}_{k}"`
     (k in 0..n_subjects_per_task-1), attributes `f"attr_{a}"`
     (a in 0..n_attributes-1), values `f"val_{v}"` (v in 0..n_values-1).
   - distractor → `Fact` over a **disjoint distractor namespace**: subjects
     `f"dsubj_{d}"` (d in 0..distractor_namespace_size-1) — never overlaps probed
     subjects. Same attribute/value vocab is fine (plausible surface form).
   - noise → `fact=None`; `content` = `noise_tokens` random tokens drawn from
     `[f"tok_{t}" for t in range(noise_vocab_size)]`.
   - `content` for fact items: a fixed template, e.g.
     `f"The {attribute} of {subject} is {value}."`
5. **FR1.3 regime knobs (must visibly take effect):**
   - **contradiction_rate**: a fraction of *signal* items re-assert an
     already-introduced probed `(subject,attribute)` with a **different** value
     (a contradiction). The later value wins (canonical). Record nothing extra on
     the item — contradictions are detectable as repeat `(subject,attribute)`
     signal facts with changed value. With `contradiction_rate=0.0` every probed
     `(subject,attribute)` is asserted exactly once (stable probe set).
   - **drift**: as `task_index` grows, bias the sampled value index upward (a
     gradual value-distribution shift). At `drift=0.0` values are uniform.
   - **recency_bias**: cluster repeated/contradicting facts toward the end of a
     task when > 0. At `recency_bias=0.0` placement is uniform.
6. **CL scenario tagging (FR1.2, exit #3).** Set `Stream.scenario = config.scenario`
   (exactly one). Scenario shapes the probed namespace policy:
   - task-incremental: probed subjects are disjoint per task (as above).
   - domain-incremental: same `attr_*` set every task, but subjects use a per-task
     "domain" prefix `f"dom{task_index}_subj_{k}"` (input shift, shared answer
     space).
   - class-incremental: each task introduces **new** attribute "classes"
     `f"attr_t{task_index}_{a}"` so the set of classes grows over tasks.
   (Probe-side handling of these lives in WS3; you only need to tag + namespace.)

**Tests (`tests/test_stream_generator.py`) must cover exit criteria 1, 2, 3:**
- determinism: two `generate_stream(cfg, seed)` calls produce equal `Stream`
  objects AND byte-identical `json.dumps(stream.model_dump(mode="json"),
  sort_keys=True)`; a different seed changes the stream.
- labels: every item has exactly one label; empirical label proportions match the
  requested `label_mix` within tolerance across **≥3 distractor ratios**.
- scenario: `Stream.scenario` is exactly the requested one; `assert_same_scenario`
  raises when two streams of different scenarios are combined (cross-scenario
  mixing rejected).
- regime knobs visibly take effect (contradictions appear iff rate>0; etc.).

---

## WS2 — Datasheet  (`slow_wave/stream/datasheet.py` + `tests/test_stream_datasheet.py`)

Emit a Gebru et al. (2021) **Datasheet for Datasets** describing a stream (exit
#5). **Public API:**

```python
class Datasheet(BaseModel):  # pydantic v2, extra="forbid"
    # The seven Gebru sections as nested models or richly-typed fields:
    #   motivation, composition, collection_process, preprocessing,
    #   uses, distribution, maintenance
    # PLUS computed statistics needed by the PRD:
    #   - scenario, seed, stream_id, n_tasks, n_items
    #   - label_distribution: dict[str, int]   (counts per label)
    #   - label_proportions: dict[str, float]  (rounded to 6 dp)
    #   - requested_label_mix: dict[str, float]
    #   - distractor regime params (contradiction_rate, drift, recency_bias, etc.)
    #   - label_provenance: str  (how labels were assigned — synthetic, by design)
    #   - n_contradictions: int  (count of probed (subj,attr) reasserted w/ new value)
    ...

def build_datasheet(stream: Stream, config: StreamGenConfig | None = None) -> Datasheet: ...
def datasheet_to_json(ds: Datasheet) -> str:  # sort_keys=True, indent=2, trailing \n
```

Behavior:
- `build_datasheet` reads labels via `schema.offline_labels(stream)` (it is
  offline scoring code — allowed). `config` defaults to `stream.config`.
- **Deterministic**: same stream → byte-identical datasheet JSON. Round all floats
  to 6 dp.
- Fill every Gebru section with synthetic-appropriate prose (motivation = why this
  synthetic stream exists; composition = item/label structure; collection =
  programmatic generation; preprocessing = none; uses = consolidation bench;
  distribution = bundled with the bench; maintenance = regenerated from seed).
- Schema validation: the model itself is the schema; provide a `validate_datasheet
  (data: dict) -> Datasheet` that round-trips through `Datasheet.model_validate`.

**Tests** must: build a datasheet, assert all Gebru sections present and non-empty,
assert `label_distribution` sums to `n_items` and matches `offline_labels`
counts, assert determinism (byte-identical JSON over two builds), and assert the
JSON validates against the schema (round-trip `model_validate`).

---

## WS3 — Probe set / R[i,j] / oracle  (`slow_wave/stream/probes.py` + `tests/test_stream_probes.py`)

Build the held-out probe set and compute a well-formed accuracy matrix against a
trivial oracle (exit #6). **Public API:**

```python
def build_probe_set(stream: Stream) -> ProbeSet: ...

class OracleAgent:
    """Trivial perfect-memory oracle for the probe-set skeleton check."""
    def reset(self) -> None: ...
    def observe(self, item: StreamItem) -> None: ...        # remember signal-form facts it sees
    def answer(self, probe: Probe) -> str: ...              # latest value seen for (subject,attribute), else ""

def compute_accuracy_matrix(stream: Stream, probe_set: ProbeSet,
                            agent: OracleAgent | None = None) -> AccuracyMatrix: ...
```

Behavior:
- **build_probe_set**: derive one probe per probed `(subject, attribute)` whose
  canonical answer is the **latest signal value** for that key over the whole
  stream. `available_after_order` = order of the **first** signal assertion of
  that key. `query = f"What is the {attribute} of {subject}?"`.
  `task_index` = the task in which the key first appears. `probe_id =
  f"p{idx:06d}"`. `task_id_visible = (stream.scenario == TASK_INCREMENTAL)`.
  Identify "probed" keys structurally (signal items assert probed-namespace
  facts) — but you must read labels via `offline_labels` to know which items are
  signal (building probes is offline). Aim for ~`probes_per_task * n_tasks`
  probes (cap per task at `probes_per_task` deterministically if more keys exist).
- **OracleAgent**: `observe` records `(subject,attribute) -> value` for **every
  fact-bearing item it sees** (it does not get labels — it just memorizes facts,
  which is fine for a skeleton oracle). `answer` returns the latest value for the
  probe's `(subject,attribute)` or `""` if unseen.
- **compute_accuracy_matrix**: R is `n_tasks x n_tasks`. For each cutoff task
  `i` (0..n_tasks-1): reset the agent, replay all items with `task_index <= i` in
  order (`observe` each), then for each task `j` compute the fraction of task
  `j`'s probes the agent answers correctly (`answer(probe) == probe.answer`).
  Entry with zero probes for task `j` → define as `0.0` (and your test should use
  a config where every task has probes). Return `AccuracyMatrix`.
- **Well-formed known answers**: with `contradiction_rate=0.0`, the oracle yields
  `R[i][j] == 1.0` for `j <= i` (facts seen) and `0.0` for `j > i` (not yet
  introduced) — a lower-triangular ones matrix. Your test asserts this exact
  pattern (the "known answers" skeleton), plus square shape and entries in [0,1].

**Tests** must: build a probe set (assert non-empty, every probe has a known
answer, shapes), build the oracle, compute R with `contradiction_rate=0.0`, and
assert the lower-triangular-ones pattern + `AccuracyMatrix` shape validation.

---

## WS4 — Confound guard  (`slow_wave/stream/guard.py` + `tests/test_stream_confound_guard.py`)

Enforce and prove FR1.6: ground-truth relevance labels are **not reachable** from
any online retrieval/priority code path (exit #4). **Public API:**

```python
# The frozen, label-free projection the wake agent / retrieval will consume.
def online_view(stream: Stream) -> tuple[StreamItem, ...]: ...
    # returns stream.items as an immutable tuple after guard-checking that no item
    # exposes a label-bearing field or a path to GroundTruth.

BANNED_FIELD_NAMES: frozenset[str]  # {"label","labels","relevance","ground_truth",
                                    #  "is_signal","is_distractor","is_noise","gt", ...}

def assert_no_label_leak(obj, *, _seen=None) -> None: ...
    # Recursively traverse obj's pydantic fields / dataclass fields / __dict__ /
    # mapping & sequence members. Raise ConfoundLeakError if:
    #   - any reachable value is an instance of schema.Label, OR
    #   - any reachable value is a schema.GroundTruth, OR
    #   - any field/key name is in BANNED_FIELD_NAMES.
    # Cycle-safe via the _seen id-set. Do NOT descend into a full Stream (which
    # legitimately holds GroundTruth); this is for the ONLINE view objects.

class ConfoundLeakError(AssertionError): ...

def assert_online_view_is_clean(stream: Stream) -> None: ...
    # Convenience: run assert_no_label_leak over every item in online_view(stream)
    # AND assert StreamItem has no banned field at the class level.
```

Behavior:
- `online_view` must return only `StreamItem`s (which by construction have no
  label) and must itself call `assert_no_label_leak` on each before returning, so
  it is impossible to obtain an online view that leaks a label.
- The guard must also assert at the **class/type** level that `schema.StreamItem`
  declares no banned field name (so a future edit that adds a `label` field to
  `StreamItem` is caught by the test).
- Provide a positive control in tests: construct a deliberately-poisoned object
  (e.g. a small pydantic/dataclass holding a `Label` or a field named `label`)
  and assert `assert_no_label_leak` **raises** `ConfoundLeakError` — proving the
  guard actually detects leaks (not a vacuous pass).

**Tests** must cover exit #4: (a) `online_view(stream)` is clean for a real
generated stream (use WS1's `generate_stream`); (b) `StreamItem` has no
label-bearing field; (c) `offline_labels(stream)` still works (labels ARE
reachable via the sanctioned offline accessor); (d) the poisoned positive control
raises `ConfoundLeakError`.

> WS4 may import `slow_wave.stream.generator.generate_stream` in its test (it will
> exist at integration time). If you want to develop in isolation first, you can
> also hand-construct a tiny `Stream` directly from `schema` types.

---

## Integration (orchestrator-owned — do not implement in a WS)
After WS1-WS4 land, the orchestrator will: wire `slow_wave/stream/__init__.py`
exports, add `slow_wave/stream/emit.py` (writes stream + datasheet + probe set +
R[i,j] to disk deterministically) with a `python -m slow_wave.stream.emit` CLI,
add `configs/stream_smoke.yaml`, update the Makefile/README, and add an
end-to-end byte-identical determinism test. Keep your modules import-clean so this
glue is mechanical.
