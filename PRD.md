# PRD — *Slow Wave*: A Test Bench for Sleep-Inspired Memory Consolidation in Autonomous AI Agents

**Document status:** v1.0 (kickoff) · **Owner:** lead product/spec author · **Date:** 2026-06-28
**Execution model:** spec-driven; each numbered phase below is executable via `/sprint Phase N` and red-teamed against its Exit Criteria.
**Working directory:** `C:\Users\hoyer\WorkSpace\Projects\slow_wave`

> This PRD is grounded in a citation-audited literature review across six domains (sleep neuroscience, complementary learning systems / replay, LLM-agent memory, Google DeepMind prior art, unbiased evaluation methodology, and scientific-paper conventions) plus a verified survey of Anthropic's memory/context features. Every reference in Appendix A was checked for existence; refuted citations were dropped, uncertain ones flagged.

---

## 1. Overview & Vision

### 1.1 What we are building
*Slow Wave* is a **research test bench** that treats AI memory consolidation as a precursor branch of neuroscience for artificial minds. Long-running autonomous agents ingest large volumes of data, most of which never affects mission performance. *Slow Wave* gives such agents a periodic, offline **"sleep/dream" cycle** that — by analogy to biological slow-wave and REM sleep — **replays** recent experience, **transfers** episodic traces into durable semantic memory, **downscales and prunes** low-value (noise/distractor) memories while protecting mission-relevant ones, and optionally **generatively augments** memory to improve generalization. The bench *instruments* every one of these operations so they can be measured against **known ground-truth relevance**, and it *generates the experimental data* that a final scientific paper reports.

### 1.2 Why it matters
As world models and persistent "second brain" agents proliferate, continuous ingestion will swamp working memory with data that does not improve mission performance. Biology solved an analogous problem with sleep: consolidation is not one process but a coordinated sequence — replay (sharp-wave ripples), systems consolidation (hippocampus→neocortex transfer), and synaptic downscaling (the Synaptic Homeostasis Hypothesis) — that strengthens signal and prunes noise. The central scientific bet of this project is that an **engineered analogue of this cycle improves long-horizon agent performance and memory efficiency**. *Slow Wave* is built to test that bet **falsifiably** — and to treat a negative result as a first-class, publishable insight.

### 1.3 The experience / feeling it should deliver
- **For the researcher running it:** the feeling of operating a *scientific instrument*, not a demo — every run is reproducible, every claim is traceable to a tool result, and every metric has a confidence interval.
- **For the reader of the paper:** the sense that a genuinely novel question ("does sleep-like consolidation transfer to frozen-weights LLM agents?") was asked rigorously, with controls a skeptic would design, and answered honestly whichever way the data fell.
- **For the field:** a reusable, open apparatus and protocol that others can extend — the first mechanism-decomposed continual-learning bench for LLM-agent memory with controlled distractors and ground-truth relevance.

---

## 2. Target User & Context

| User | Skill level | Context of use | What they need from *Slow Wave* |
|---|---|---|---|
| **AI/ML researcher (primary)** | Expert in Python + ML; comfortable with LLM APIs, embeddings, continual-learning metrics | Designing/running experiments; iterating on the dream engine; producing the paper | A clean, configurable, reproducible experiment harness; sensible defaults; trustworthy metrics & plots |
| **Reviewer / red-team (secondary)** | Expert; adversarial | Verifying that exit criteria are met and that claims are defensible | Falsifiable criteria, preregistration, control battery, raw artifacts to re-analyze |
| **Practitioner building persistent-memory agents (tertiary)** | Intermediate–advanced engineer | Deciding whether to adopt a consolidation cycle in production | A clear statement of *when* consolidation beats "just stuff it in context" (the long-context crossover), at what cost |

**Situation:** the project is run by a solo/small team with finite compute. It must iterate fast (accelerated "sim-time" cycles, many seeds) and validate occasionally (a few genuine long-horizon runs). The deliverable is *both* a working bench and a peer-review-quality paper, weighted equally.

---

## 3. Goals & Non-Goals

### 3.1 Goals
1. **G1 — Apparatus.** Build an instrumented, reproducible test bench: a synthetic continual task stream with injected noise and ground-truth relevance, a dual-store memory substrate, and a four-operator "dream" engine whose operators are independently ablatable.
2. **G2 — Falsifiable test.** Pre-register and execute an unbiased experiment with a full control battery that can confirm *or refute* the hypothesis that sleep-like consolidation improves long-horizon performance/efficiency.
3. **G3 — Measurement.** Quantify consolidation *quality* directly (precision/recall of pruning signal vs. noise) — something neither neuroscience nor existing LLM-memory benchmarks can do — in addition to downstream task accuracy.
4. **G4 — Paper.** Produce a rigorous, properly-cited scientific paper (IMRaD; ScienceDirect/arXiv style; author-date citations) drawing on Anthropic and DeepMind work, with figures generated from the bench's own data and a reproducibility appendix.
5. **G5 — Negative-result discipline.** If the hypothesis is wrong, document *why* — with the same rigor as a positive result — as an insight for the field.

### 3.2 Non-Goals
- **NG1 — No weight updates.** We study **memory** consolidation for **frozen-weights** LLM agents (the realistic deployment regime). We do *not* fine-tune model weights, do RLHF, or train new models.
- **NG2 — Not a production memory product.** No multi-tenant service, no UI beyond plots/reports, no SLA. (A practitioner takeaway is a goal; a shipped product is not.)
- **NG3 — No claim of biological fidelity.** Neuroscience is the source of *testable mechanisms and metaphors*, not evidence of efficacy. We do not claim the bench models the brain.
- **NG4 — Not a general agent framework.** The agent is a measurement vehicle scoped to the synthetic stream (and optional secondary domains), not a general-purpose autonomous agent.
- **NG5 — No security/biology-sensitive content** in task streams (keeps Claude usage clean and avoids classifier refusals).

---

## 4. Key Assumptions & Decisions

### 4.1 Decisions confirmed with the project owner
| # | Decision | Rationale |
|---|---|---|
| D1 | **Integrated bench ↔ paper (equal weight).** The bench *generates* the paper's data. | Avoids a demo-with-write-up; keeps results falsifiable and reproducible. |
| D2 | **Hybrid stack:** Claude API for agent reasoning + "dream" summarization; **local open-weight embedding models** for memory vectors. | Best capability/cost mix; mirrors the published Letta "sleep-time compute" split (Lin et al. 2025). |
| D3 | **Synthetic continual task stream** with injected noise + **ground-truth relevance labels** as the primary testbed. | The only substrate where pruning precision/recall is directly measurable and confounds are controllable. |
| D4 | **Accelerated "sim-time"** wake/sleep cycles for cheap iteration + **a few real long-horizon runs** to validate that compression does not distort retention curves. | Fast science, controlled cost; sim-vs-real agreement becomes an explicit validity check. |

### 4.2 Labeled assumptions (made to proceed; owner may override)
- **A1 — Model defaults.** Agent reasoning + dream summarization default to **Claude Opus 4.8** (`claude-opus-4-8`, 1M context, $5/$25 per MTok); cheap/bulk operations (e.g., scoring, salience tagging) may use **Claude Haiku 4.5** (`claude-haiku-4-5`, $1/$5) to control cost; **Claude Sonnet 4.6** (`claude-sonnet-4-6`, $3/$15) is the mid-tier fallback. Exact model is a config knob and is recorded in every run manifest. *(All model IDs are pinned snapshots; the 4.6+ family IDs are dateless but still pinned.)*
- **A2 — Embeddings.** Local `sentence-transformers` model (default `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`), CPU-friendly, version-pinned. Vector index: in-process (NumPy/FAISS) — scale is modest.
- **A3 — Language/runtime.** **Python 3.11+**; `anthropic` SDK; `numpy`/`scipy`/`statsmodels`/`rliable` for stats; `matplotlib` for plots; `pydantic`+YAML for config; `pytest` for tests. (See §7.)
- **A4 — Compute budget.** Sim-time runs dominate; ≥5 seeds per arm minimum (N justified by power analysis, not convenience). A token-budget controller equalizes spend across arms and aborts runaway runs. Hard ceilings configurable.
- **A5 — Paper venue.** Target **arXiv (cs.LG) first**, NeurIPS/ICLR-style LaTeX with **author-date (natbib)** citations, portable to an Elsevier cognitive/neural journal under "Your Paper Your Way." Switch to IEEE numbered only if a final venue requires it.
- **A6 — Determinism reality.** LLM calls are non-deterministic even at temperature 0; we mitigate (pinned model id, multi-seed, reported dispersion, a temperature-0 stability control) rather than assume reproducibility.

---

## 5. Functional Requirements

Each module below is a concrete component with defined behavior. (Phase mapping in §8.)

### 5.1 Synthetic Continual Task Stream Generator (`stream/`)
- **FR1.1** Generate a long, ordered stream of *tasks* and *ingest items*. Each item carries a **ground-truth relevance label** w.r.t. the agent's mission(s): `signal` (mission-relevant), `distractor` (plausible but irrelevant), or `noise` (random).
- **FR1.2** Support **continual-learning scenarios** explicitly tagged: task-incremental, domain-incremental, class-incremental (van de Ven et al. 2022). No cross-scenario metric aggregation.
- **FR1.3** Inject controllable **distractor regimes** (e.g., distractor ratio sweeps), **temporal structure** (recency, drift), and **contradictions** (items that conflict with earlier signal, to test conflict resolution).
- **FR1.4** Be **fully deterministic given a stream seed**; emit a machine-readable **datasheet** (Gebru et al. 2021) describing distribution, label provenance, and regime parameters.
- **FR1.5** Provide an **evaluation probe set** (held-out queries/tasks) with known correct answers, decoupled from the online stream, to compute the accuracy matrix `R[i,j]` (accuracy on task *j* after learning through task *i*).
- **FR1.6 (Critical confound guard).** Ground-truth relevance labels are available to **offline scoring only** — they must **never** enter the online retrieval/priority signal (prevents the "oracle-replay" leak). Enforced in code and asserted in tests.

### 5.2 Memory Substrate (`memory/`)
- **FR2.1 — Dual physical store.** A fast **EPISODIC** buffer (append-only, timestamped, local-embedding-indexed raw wake traces) and a slow **SEMANTIC/SKILL** store (compressed, abstracted entries) — physically separate so retrieval, provenance, and forgetting are measurable per store. (Complementary Learning Systems; Kumaran, Hassabis & McClelland 2016.)
- **FR2.2 — Salience model.** Each memory item carries a salience/weight and metadata: recency, access count, novelty (embedding distance to consolidated memory), surprise/reward, and a **provenance pointer** to source episodes.
- **FR2.3 — Retrieval policy.** Baseline retrieval = recency × importance × relevance over the stores (Generative Agents memory-stream policy; Park et al. 2023), pluggable.
- **FR2.4 — Archival tier.** Forgetting **demotes** items to an auditable archival tier rather than hard-deleting, so eviction can be scored and unlearning validated as non-destructive.
- **FR2.5 — Write-protection.** Consolidated high-importance memories carry an EWC-spirit protection flag (Kirkpatrick et al. 2017); a later distractor silently overwriting a protected fact is a **logged failure event**.

### 5.3 Wake Agent (`agent/`)
- **FR3.1** A **WAKE loop**: ingest stream items, attempt tasks, append episodic traces, retrieve from memory, call Claude for reasoning. No semantic-store writes during wake (writes are gated to sleep).
- **FR3.2** Emit, per task, the data needed to populate `R[i,j]`, plus token/compute/latency telemetry.
- **FR3.3** Configurable model, effort, and a **token-budget controller** that the agent self-moderates against (and that the harness enforces as a ceiling).

### 5.4 The Dream Engine — four independently ablatable operators (`dream/`)
The heart of the bench. Each operator is a toggle; the engine runs a **two-phase cycle** (NREM-like → REM-like) per the sequential hypothesis, and phases are independently ablatable and reorderable.
- **FR4.1 — REPLAY.** Re-sample, re-embed, re-score a subset of recent episodics in compressed batches ("ripple-analogue"). Sampling supports **uniform** (DQN baseline; Mnih et al. 2015) and **prioritized** (recency × relevance × novelty × surprise; Schaul et al. 2016, with importance-sampling weights logged).
- **FR4.2 — TRANSFER.** Claude "dream" summarization distills sampled episodics into compressed, durable **semantic** entries (episodic→semantic promotion), enforcing **CLS interleaving** (each batch mixes new episodes with sampled prior consolidated memories; removing interleaving is the on-purpose catastrophic-interference condition). Provenance preserved.
- **FR4.3 — DOWNSCALE.** Each cycle, multiply **every** item's salience by a factor <1 (global renormalization, SHY analogue; Tononi & Cirelli 2006/2014/2020), then **re-potentiate only replayed items** → "decay all, protect signal." Decay function ablatable: exponential / Weibull / ACT-R activation decay.
- **FR4.4 — GENERATIVE-AUGMENT (REM-like).** Claude synthesizes paraphrases/counterfactuals/abstractions (pseudo-episodes) from episodics (LLM analogue of generative replay, Shin et al. 2017; world-model "dreaming," Ha & Schmidhuber 2018). Tested on held-out task variants for generalization; generator fidelity/drift is tracked.
- **FR4.5 — GATING / scheduling.** Semantic-store writes occur **only** inside scheduled sleep windows; "continuous online writes" is a comparison condition.
- **FR4.6 — Sleep-pressure controller.** Cycle frequency/duration and downscaling magnitude can scale to accumulated wake "memory churn" (SWA-homeostasis analogue); adaptive-vs-fixed is an experiment.
- **FR4.7 — Conflict/unlearning step (optional).** Detect and resolve contradictions among entries (Crick–Mitchison "reverse learning" analogue; demote, don't destroy).

### 5.5 Evaluation Harness & Control Battery (`eval/`)
- **FR5.1 — Control arms** (all share one harness at matched budgets): **no-sleep** (wake-only baseline), **replay-only**, **downscale/prune-only**, **random-pruning** (ground-truth-blind negative control), **full-dream** (treatment), **reflection** (Generative Agents shallow-synthesis control), **oracle/upper-bound** (prune exactly the injected distractors using known labels), **long-context** (entire stream stuffed into a frontier long-context window, no consolidation — the key validity threat), and an **A/A control** (two identical configs, different seeds — the noise floor any claimed effect must exceed).
- **FR5.2 — Matched-budget rule.** A budget controller equalizes LLM tokens (agent + dream), retrieval calls, and final memory vector count across arms; actuals are recorded. Where matching is impossible, report a **Pareto frontier** of accuracy vs. compute instead of a single matched point.
- **FR5.3 — Metrics (emit machine-readable JSON per run):**
  - From `R[i,j]`: **Average Accuracy (ACC)**, **Backward Transfer / forgetting (BWT)**, **Forward Transfer (FWT)** (Lopez-Paz & Ranzato 2017), per-task **Forgetting Measure** (Chaudhry et al. 2018).
  - **Mechanism-level (the bench's superpower):** precision / recall / F1 of consolidation at *discarding distractors vs. retaining signal*, plus a calibration curve of embedding-decay vs. true relevance — reported **decoupled** from downstream accuracy (the two can diverge).
  - **Cost:** memory footprint (vector count, bytes), test-time tokens, p95 latency, compute + wall-clock, sim-time compression factor (Díaz-Rodríguez et al. 2018 conventions).
  - **Generator fidelity:** hallucination/drift of dream summaries vs. raw episodes.
  - **TMR-style targeting:** retention lift for a force-replayed labeled subset (benchmark vs. Hu et al. 2020 meta-analytic g≈0.29).
- **FR5.4 — Statistics.** ≥5 seeds (vary **both** agent/LLM sampling seed and stream-generation seed), N justified by **power analysis** (Colas et al. 2018); **bootstrap 95% CIs** and **rliable** robust aggregates (IQM, performance profiles, probability of improvement; Agarwal et al. 2021); Wilcoxon signed-rank (paired arm-vs-control) and Friedman + post-hoc (multi-arm; Demšar 2006); **standardized effect sizes with CIs**, not just p-values; **multiple-comparison correction** across the metric/arm/regime grid; a pre-specified **primary endpoint**.
- **FR5.5 — Preregistration.** A registered-report-style document (hypothesis, primary endpoint, seeds, tests, rejection criteria) **committed with a timestamped git hash before any real long run** (Pineau et al. 2021; Karl et al. 2024).
- **FR5.6 — Bias controls.** Blind metric computation (labels offline only); LLM-nondeterminism mitigation (pinned dated/static model id in every manifest; a temperature-0 stability control); scenario tagging with no cross-scenario aggregation; memory-drift/silent-corruption detection (a ground-truth-faithfulness metric flags when repeated summarization degrades rather than distills memory).

### 5.6 Reproducibility & Run-Manifest System (`repro/`)
- **FR6.1** Every run emits a JSON **manifest**: exact model id, sampling params, embedding model+version+dim, all hyperparameters + search ranges, seed list, **git commit hash**, wall-clock + token/compute cost, sim-time compression factor.
- **FR6.2** Every figure is generated by a script that reads manifests; captions state *n* and CI method. **One-command reproduction** of the headline figure.
- **FR6.3** Raw Claude API transcripts archived as a frozen dataset (to survive model-snapshot deprecation).
- **FR6.4** A **Datasheet** (synthetic stream) and **Model Card** (Claude snapshot + sampling + embedding model) ship as appendices.

### 5.7 Scientific Paper (`paper/`)
- **FR7.1** IMRaD-for-ML manuscript (Title → Abstract+keywords → Intro+contributions → Related Work → Test-Bench Design → Experimental Protocol → Results+inline Discussion → Limitations → Reproducibility → Conclusion → Appendices).
- **FR7.2** Results populated **from the bench's own data**; every figure regenerable from manifests.
- **FR7.3** Related work cites **Anthropic** (memory tool, context editing/management, Agent SDK, Letta sleep-time compute as the closest precedent) and **DeepMind** (DQN replay, prioritized replay, MERLIN, CLS-for-AI, world models) plus neuroscience and continual-learning anchors (Appendix A).
- **FR7.4** **Negative results are first-class** and pre-specified (see §8 Phase 6 and §9).

---

## 6. Design & Experience Requirements

These are testable expectations for *how the artifact behaves and reads*, not just what it computes.

- **DX1 — Instrument, not demo.** Any experiment is launchable from a single config file + one command; re-running with the same config + seeds reproduces results bit-for-bit *except* for documented LLM nondeterminism, whose magnitude is itself reported. **Testable:** `repro` script regenerates the headline figure from committed manifests with no manual steps.
- **DX2 — Honesty by construction.** No silent caps. If a run bounds coverage (top-N replay, sampling, no-retry), it `log()`s what was dropped. **Testable:** grep the harness for truncation points; each has a corresponding log line and a manifest field.
- **DX3 — Falsifiability visible in the code.** The null hypothesis, primary endpoint, and rejection criteria exist as a committed, parseable preregistration artifact referenced by the analysis code. **Testable:** analysis fails loudly if it computes a non-preregistered primary endpoint.
- **DX4 — Plots that a reviewer trusts.** Every figure shows seed-variability bands (shaded CIs), states *n* and the CI method in the caption, and is produced by a deterministic script. Style: clean, colorblind-safe palette, no chart-junk, publication-grade (vector PDF/SVG). **Testable:** each figure file has a sibling `*.py` and a caption with *n* + CI method.
- **DX5 — Tone of the paper.** Precise, measured, and decisive; claims are scoped to the synthetic regime and the frozen-weights setting; neuroscience framed as motivation, not proof. **Testable:** red-team checklist confirms no overclaiming (e.g., no "models the brain," no efficacy claim beyond tested regimes).
- **DX6 — Reusability.** A new control arm or dream operator can be added by implementing one interface and registering it, without touching the harness core. **Testable:** adding a trivial no-op arm requires changes only under `eval/arms/` (or equivalent) and a registry entry.

---

## 7. Architecture & Tech Stack

### 7.1 Component diagram (data flow)
```
                ┌─────────────────────────────────────────────────────────┐
                │                    Experiment Harness                    │
                │   (config → arms → seeds → matched-budget controller)    │
                └───────────────┬───────────────────────────┬─────────────┘
                                │                            │
                   ┌────────────▼───────────┐    ┌───────────▼────────────┐
   Stream Gen ───▶ │  WAKE loop (Claude API) │    │   Metrics + Stats      │
   (signal/        │  • ingest + attempt task│    │   ACC/BWT/FWT/FM       │
    distractor/    │  • append EPISODIC      │    │   prune precision/recall│
    noise +        │  • retrieve (rec×imp×rel)│    │   cost / Pareto        │
    GT labels)     └───────────┬─────────────┘    │   bootstrap CI, rliable │
        │                      │ (scheduled        └───────────┬────────────┘
        │ datasheet            │  sleep window)                 │
        ▼                      ▼                                ▼
  ┌───────────┐   ┌──────────────────────────────┐    ┌──────────────────┐
  │ Probe set │   │   DREAM ENGINE (2-phase)      │    │  Plots + Paper    │
  │ (held-out)│   │  NREM: REPLAY→TRANSFER→DOWNSCALE│   │  (figures from    │
  └───────────┘   │  REM:  GENERATIVE-AUGMENT      │    │   manifests)      │
                  │  + GATING + sleep-pressure     │    └──────────────────┘
                  └───────────┬──────────────────┘
                              │ writes (gated)
            ┌─────────────────▼───────────────────┐
            │   MEMORY SUBSTRATE                    │
            │   EPISODIC (fast, local embeddings)   │
            │   SEMANTIC/SKILL (slow, compressed)   │
            │   ARCHIVAL tier (auditable eviction)  │
            │   + salience, provenance, protection  │
            └───────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Run Manifest (JSON)│  ← model id, seeds, git hash,
                    │  + frozen transcripts│    cost, sim-time factor
                    └─────────────────────┘
```

### 7.2 Stack
| Layer | Choice | Notes |
|---|---|---|
| Language/runtime | Python 3.11+ | Matches A3 |
| Agent + dream LLM | `anthropic` SDK; **Opus 4.8** default (`claude-opus-4-8`), Haiku 4.5 for bulk scoring, Sonnet 4.6 fallback | Adaptive thinking; `effort` configured per call; token-budget controller; pinned model id in manifest |
| Embeddings | `sentence-transformers` (local, version-pinned) | CPU-friendly; deterministic given input |
| Vector index | NumPy / FAISS (in-process) | Modest scale; exact provenance |
| Config | `pydantic` + YAML | One file per experiment |
| Stats | `numpy`, `scipy`, `statsmodels`, `rliable` | CIs, IQM, performance profiles, significance |
| Plotting | `matplotlib` (+ seaborn optional) | Vector output, colorblind-safe |
| Testing | `pytest` | Confound guards (FR1.6) are tests |
| Paper | LaTeX (NeurIPS/ICLR-style), `natbib` author-date | arXiv-first; Elsevier-portable |

### 7.3 Relationship to Anthropic's production features (for Related Work & framing)
Anthropic ships **online, automatic** context-management primitives that map onto *individual* operations *Slow Wave* performs offline and ablatably:
- **Memory tool** (`memory_20250818`; GA on the Messages API, Claude 4+): a client-side `/memories` file store that persists across sessions and is read back "just in time" → the **persistence/replay substrate**.
- **Context editing** (beta `context-management-2025-06-27`; `clear_tool_uses_20250919`): server-side *clearing* of stale tool results at a token threshold (reported ~84% token reduction; memory + context editing reported ~39% agentic-search improvement) → the online analogue of **prune/downscale**.
- **Claude Agent SDK** (`CLAUDE.md` project memory, subagents, resumable sessions) → the long-horizon harness whose transcripts one would consolidate.
- **Letta "sleep-time compute"** (Lin et al. 2025) → the closest published precedent for the wake/sleep split; *Slow Wave* differs by optimizing **consolidation quality and forgetting fidelity under known ground-truth relevance**, not anticipated-query compute.

**The paper's differentiation:** Anthropic's features fire *reactively at token thresholds* with *bundled, non-ablatable* behavior and *no ground-truth relevance signal*; *Slow Wave* is a *scheduled, offline, mechanism-decomposed* study that measures consolidation quality against known relevance.

---

## 8. Phased Roadmap

Phases are numbered, independently shippable, and ordered so each unblocks the next. Every phase has an **objective**, **scoped deliverables**, and **observable, falsifiable Exit / Acceptance Criteria** a red-team can check. Execute any phase with `/sprint Phase N`.

---

### Phase 0 — Foundations & Reproducibility Scaffolding
**Objective:** stand up the repo, config, dependency pinning, and the run-manifest + reproducibility harness that every later phase depends on.
**Deliverables:** repo layout (`stream/`, `memory/`, `agent/`, `dream/`, `eval/`, `repro/`, `paper/`, `tests/`, `configs/`); pinned dependencies; `pydantic`+YAML config loader; run-manifest writer; seeding utilities; CI that runs `pytest`; a "hello-bench" smoke run that exercises a Claude call + a local embedding and writes a manifest.

**Exit / Acceptance Criteria (all must hold):**
- [ ] `pip install` (or lockfile) reproduces the environment from a clean checkout; `pytest` runs green in CI.
- [ ] A smoke run produces a JSON manifest containing **every** FR6.1 field (model id, sampling params, embedding model+version+dim, seeds, git hash, wall-clock, token cost, sim-time factor) — verified by a schema test.
- [ ] Given a fixed config + seeds, two smoke runs produce **identical** non-LLM outputs (embeddings, sampling order, file layout); any LLM-dependent field is flagged as such in the manifest.
- [ ] One command (`make repro-smoke` or equivalent) reproduces the smoke artifact end-to-end with no manual steps.

---

### Phase 1 — Synthetic Continual Task Stream Generator
**Objective:** build the measurement substrate — labeled streams with ground-truth relevance, distractor regimes, CL-scenario tags, and a held-out probe set.
**Deliverables:** stream generator (FR1.1–FR1.5); datasheet emitter; probe-set builder; the **confound guard** (FR1.6) as enforced code + tests.

**Exit / Acceptance Criteria:**
- [ ] Generator is **deterministic given a seed**: same seed → byte-identical stream + datasheet (test-verified).
- [ ] Every stream item has exactly one ground-truth label ∈ {`signal`,`distractor`,`noise`}; label distribution matches the requested regime within tolerance (test over ≥3 distractor ratios).
- [ ] Each stream is tagged with exactly one CL scenario (task/domain/class-incremental); a test rejects cross-scenario mixing.
- [ ] **Confound test passes:** an assertion/test proves ground-truth labels are *not* reachable from any online retrieval/priority code path (FR1.6).
- [ ] A datasheet (Gebru et al. 2021 fields) is emitted per stream and validated against a schema.
- [ ] The probe set computes a well-formed `R[i,j]` skeleton (correct shape, known answers) against a trivial oracle agent.

---

### Phase 2 — Memory Substrate + Baseline (No-Sleep) Wake Agent
**Objective:** dual-store memory + a wake-loop agent that runs the stream and emits the full metric stream — *without* any dream cycle (the no-sleep baseline / catastrophic-forgetting reference).
**Deliverables:** EPISODIC + SEMANTIC stores, salience model, provenance, archival tier, write-protection (FR2.x); wake agent (FR3.x); retrieval policy; `R[i,j]` population + cost telemetry.

**Exit / Acceptance Criteria:**
- [ ] The no-sleep agent runs an entire stream end-to-end and writes a populated `R[i,j]` plus cost telemetry to a manifest.
- [ ] EPISODIC and SEMANTIC are physically separate; per-store retrieval, footprint, and forgetting are independently queryable (test asserts separation).
- [ ] Every consolidated/episodic item exposes a provenance pointer to its source(s); a test traces ≥1 item back to its origin.
- [ ] Archival eviction **demotes** (does not delete): an evicted item is recoverable from the archival tier (test).
- [ ] A write-protection violation (distractor overwriting a protected fact) produces a **logged failure event** (test injects the condition).
- [ ] The baseline demonstrably **forgets**: on a noisy stream, BWT is measurably negative / forgetting > 0 (sanity check that the phenomenon under study exists).

---

### Phase 3 — The Dream Engine (four ablatable operators)
**Objective:** implement REPLAY, TRANSFER, DOWNSCALE, GENERATIVE-AUGMENT as independent toggles in a two-phase (NREM→REM) cycle, plus gating and sleep-pressure control.
**Deliverables:** the four operators (FR4.1–FR4.4); gating (FR4.5); sleep-pressure controller (FR4.6); optional conflict/unlearning step (FR4.7); CLS-interleaving enforcement; generator-fidelity tracking.

**Exit / Acceptance Criteria:**
- [ ] Each of the four operators can be **independently enabled/disabled by config**; a 2×2 (replay-module × downscale-module) and 4-way on/off matrix all instantiate and run (test enumerates the 2⁴ combinations and confirms each executes).
- [ ] **DOWNSCALE** verifiably applies global decay then re-potentiates only replayed items: after a cycle, replayed-item salience > non-replayed under the same decay (test).
- [ ] **TRANSFER** writes semantic entries **only** inside a scheduled sleep window (gating test: no semantic writes occur during wake).
- [ ] **CLS interleaving** is enforced when enabled and *removable* (the catastrophic-interference condition): a test confirms batches mix new + prior consolidated memories when on, and don't when off.
- [ ] **GENERATIVE-AUGMENT** produces pseudo-episodes and logs a generator-fidelity/drift score per cycle.
- [ ] Decay function is swappable among exponential / Weibull / ACT-R (test runs all three).
- [ ] A full dream cycle runs on a Phase-1 stream and leaves provenance + archival audit intact (no hard deletes).

---

### Phase 4 — Evaluation Harness, Control Battery & Preregistration
**Objective:** wire all control arms into one harness at matched budgets, implement the full metric + statistics suite, and commit the preregistration.
**Deliverables:** the nine arms (FR5.1); matched-budget controller + Pareto reporting (FR5.2); metrics (FR5.3); statistics (FR5.4); preregistration artifact (FR5.5); bias controls incl. temperature-0 stability control and memory-drift detector (FR5.6).

**Exit / Acceptance Criteria:**
- [ ] **All nine arms** (no-sleep, replay-only, downscale-only, random-pruning, full-dream, reflection, oracle, long-context, A/A) instantiate and run on the same stream via one harness (test runs each at least once).
- [ ] The **A/A control** yields no significant difference between two identical configs (different seeds) on the primary endpoint — establishing the noise floor (test/assertion).
- [ ] The **oracle** arm achieves higher prune precision/recall than any non-oracle arm on a distractor-heavy stream (sanity ceiling check).
- [ ] Matched-budget controller equalizes tokens/retrieval/memory-size across arms within tolerance and **records actuals**; where unmatched, a Pareto frontier is produced instead (test).
- [ ] Metrics module computes ACC, BWT, FWT, per-task FM from `R[i,j]`, **plus** prune precision/recall/F1 and the decay-vs-relevance calibration curve — all decoupled and emitted as JSON (schema test).
- [ ] Statistics module produces bootstrap 95% CIs, rliable IQM/performance-profiles/probability-of-improvement, Wilcoxon + Friedman/post-hoc, and standardized effect sizes with CIs and multiple-comparison correction (unit tests on synthetic inputs with known answers).
- [ ] **Preregistration committed with a git hash** *before* any real long run: it names H1, H0, the single primary endpoint, seed plan + power analysis, tests, and explicit rejection criteria. Analysis code refuses to compute a non-preregistered primary endpoint (test).
- [ ] **Temperature-0 stability control** quantifies run-to-run variance of the dream summarizer; **memory-drift detector** flags degradation from repeated summarization (both produce numbers on a tiny run).

---

### Phase 5 — Experiments, Analysis & Plots
**Objective:** execute the preregistered experiment grid in accelerated sim-time (≥5 seeds), validate with a few real long runs, generate all figures, and record the H1/H0 decision with its statistics.
**Deliverables:** results dataset (manifests + JSON metrics); figures (retention curves with seed bands, ablation table incl. 2×2, mechanism precision/recall, cost Pareto, long-context crossover, TMR targeting, sim-vs-real agreement); a written analysis with the primary-endpoint verdict.

**Exit / Acceptance Criteria:**
- [ ] The full arm × distractor-regime × seed grid completes; every run has a manifest with cost + git hash; coverage (and any dropped cells) is logged — **no silent caps**.
- [ ] **≥5 seeds per arm** (varying both seed types), with N consistent with the committed power analysis.
- [ ] At least **one real long-horizon run per key arm** is executed, and a **sim-vs-real agreement** figure quantifies whether time-compression distorts retention curves (a documented inversion at scale counts as a finding, not a failure).
- [ ] Every figure is regenerated by `repro` from committed manifests in one command; each caption states *n* and the CI method (red-team re-runs it).
- [ ] The **primary endpoint** is computed exactly as preregistered, with effect size + CI and the preregistered test; the H1/H0 verdict is stated unambiguously (confirmed / refuted / inconclusive-with-reason).
- [ ] The **long-context crossover** (stream length L beyond which consolidation beats stuff-it-in-context, if any) is reported, or its absence stated.
- [ ] If the result is **negative**, the analysis maps it to one of the preregistered negative forms (§9) and ties it to the datasheet regime that produced it.

---

### Phase 6 — Scientific Paper
**Objective:** write the peer-review-quality paper from the bench's data, with verified citations and a reproducibility appendix; document negative results as insights.
**Deliverables:** complete IMRaD manuscript (FR7.x) in LaTeX; figures sourced from Phase 5; Related Work citing Anthropic + DeepMind + neuroscience + CL anchors; Datasheet + Model Card + hyperparameter/prompt appendices; reproducibility statement anchored to the NeurIPS checklist / Pineau et al. 2021.

**Exit / Acceptance Criteria:**
- [ ] Manuscript contains all IMRaD-for-ML sections (§5.7 FR7.1) and an explicit contributions list (apparatus + protocol + ablation results incl. negatives).
- [ ] **Every results figure/number is regenerable** from a committed manifest+script (red-team spot-checks ≥2 figures end-to-end).
- [ ] **Every citation resolves** to a real, locatable source (red-team checks a random sample against Appendix A / live URLs); no refuted citation appears; uncertain ones are flagged as such in text.
- [ ] Related Work cites **Anthropic** (memory tool, context editing, Agent SDK, Letta sleep-time compute) **and DeepMind** (DQN/PER/MERLIN/CLS/world models) with correct attributions.
- [ ] Title + abstract front-load the discoverability terms (memory consolidation, continual learning, LLM agent, sleep/dream) per A5.
- [ ] A **Limitations** section states the frozen-weights scope, sim-time-compression validity caveat, proprietary-model reproducibility caveat, and scenario scope — and the paper makes **no efficacy/biological-fidelity overclaim** (red-team checklist DX5).
- [ ] If the hypothesis was not confirmed, the paper presents the **negative result as a first-class finding** with mechanism-level explanation (§9), not a footnote.
- [ ] Reproducibility appendix: pinned model id, full hyperparameters + search ranges, prompts, per-run manifest schema; one-command headline-figure reproduction documented.

---

## 9. Risks & Open Questions

### 9.1 Pre-specified negative results (each is a publishable insight)
1. **"Dreaming ≯ replay-only at matched budget."** Effect-size CI over replay-only spans zero → isolates the effect to mere re-exposure; LLM summarization adds nothing in this regime.
2. **"Neither biological metaphor carries the effect."** In the 2×2 (replay × downscale), both main effects' CIs include zero → clean falsification of SHY/active-consolidation transfer to frozen-weights LLM agents.
3. **"Generative augmentation degrades precise retrieval."** REM-like arm improves generalization but lowers needle-in-haystack recall past an augmentation-volume threshold → maps the fidelity/compression Pareto.
4. **"Long-context wins below stream length L."** Consolidation only beats stuff-it-in-context beyond a crossover → quantifies when external consolidation is worth its cost.
5. **"Gating is unnecessary."** Continuous online writes match gated writes once decay+protection is present → a clean simplification.
6. **"Adaptive sleep pressure ≯ fixed cycles."** → the added controller complexity is unjustified.

### 9.2 Risks & mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| **Oracle-replay leak** (labels bleed into online priority) | Invalidates all results | FR1.6 confound guard as enforced code + test; blind metric computation |
| **LLM nondeterminism** masquerading as effect | False positives/negatives | Pinned model id, multi-seed, temperature-0 stability control, reported dispersion; A/A noise floor |
| **Sim-time compression distorts retention** | External-validity threat | A few real long runs + sim-vs-real agreement figure; pre-register that an inversion at scale is publishable |
| **Long-context baseline beats us everywhere** | Undercuts the premise | Frame premise around the *crossover* and cost; report it honestly (§9.1 #4) |
| **Memory drift from repeated summarization** | Silent corruption | Ground-truth-faithfulness metric + drift detector (FR5.6) |
| **Cost overrun** on Claude calls | Budget | Token-budget controller, Haiku for bulk scoring, matched-budget caps, sim-time first |
| **Hallucinated citations in the paper** | Credibility | Citations are pre-audited (Appendix A); red-team re-checks a sample at Phase 6 |
| **Overclaiming** (brain fidelity / efficacy) | Reviewer rejection | DX5 red-team checklist; neuroscience framed as motivation only |

### 9.3 Open questions (resolve during execution)
- **OQ1** — Exact **primary endpoint**: final ACC at equal retrieval budget vs. a cost-adjusted *accuracy-per-retrieval-token*? (Decide and **pre-register before Phase 5**; it drives the power analysis.)
- **OQ2** — Should we run a **secondary domain** (e.g., "second-brain" doc-QA) for external validity, or keep scope to the synthetic stream for v1? (Owner chose single-domain; revisit if time allows.)
- **OQ3** — Embedding model choice and dimension (bge-small vs. MiniLM) — does the conclusion depend on it? (Add as a small robustness ablation if cheap.)
- **OQ4** — Target journal after arXiv (Neural Networks / Cognitive Systems Research / Neurocomputing) — affects final citation style only.
- **OQ5** — Do we expose memory stores to the agent via Anthropic's **memory tool / MCP** for a "realism" variant, or keep the substrate fully in-process for control? (Default: in-process for control; note as future work.)

---

## Appendix A — Master Reference List (citation-audited)

Confidence tags: **[C]** confirmed · **[C-corr]** confirmed with audit correction · **[U]** uncertain — verify before final cite.

**A. Sleep & memory neuroscience**
1. [C] McClelland JL, McNaughton BL, O'Reilly RC. *Why there are complementary learning systems in the hippocampus and neocortex.* Psychol Rev. 1995;102(3):419-457.
2. [C] Buzsáki G. *Two-stage model of memory trace formation.* Neuroscience. 1989;31(3):551-570.
3. [C] Buzsáki G. *Hippocampal sharp wave-ripple.* Hippocampus. 2015;25(10):1073-1188.
4. [C] Diekelmann S, Born J. *The memory function of sleep.* Nat Rev Neurosci. 2010;11(2):114-126.
5. [C] Rasch B, Born J. *About sleep's role in memory.* Physiol Rev. 2013;93(2):681-766.
6. [C] Klinzing JG, Niethard N, Born J. *Mechanisms of systems memory consolidation during sleep.* Nat Neurosci. 2019;22(10):1598-1610.
7. [C] Tononi G, Cirelli C. *Sleep function and synaptic homeostasis.* Sleep Med Rev. 2006;10(1):49-62.
8. [C] Tononi G, Cirelli C. *Sleep and the price of plasticity.* Neuron. 2014;81(1):12-34.
9. [C] Tononi G, Cirelli C. *Sleep and synaptic down-selection.* Eur J Neurosci. 2020;51(1):413-421.
10. [C] Hu X, Cheng LY, Chiu MH, Paller KA. *Promoting memory consolidation during sleep: a meta-analysis of TMR.* Psychol Bull. 2020;146(3):218-244.
11. [C] Crick F, Mitchison G. *The function of dream sleep.* Nature. 1983;304:111-114.
12. [C] Poe GR. *Sleep is for forgetting.* J Neurosci. 2017;37(3):464-473.
13. [C] Hoel E. *The overfitted brain: dreams evolved to assist generalization.* Patterns. 2021;2(5):100244.

**B. CLS, replay, world models, continual learning (AI)**
14. [C] Kumaran D, Hassabis D, McClelland JL. *What learning systems do intelligent agents need? CLS theory updated.* Trends Cogn Sci. 2016;20(7):512-534.
15. [C] Hassabis D, Kumaran D, Summerfield C, Botvinick M. *Neuroscience-inspired artificial intelligence.* Neuron. 2017;95(2):245-258.
16. [C] Mnih V, et al. *Human-level control through deep reinforcement learning.* Nature. 2015;518:529-533.
17. [C] Schaul T, Quan J, Antonoglou I, Silver D. *Prioritized experience replay.* ICLR 2016. arXiv:1511.05952.
18. [C] Shin H, Lee JK, Kim J, Kim J. *Continual learning with deep generative replay.* NeurIPS 2017. arXiv:1705.08690.
19. [C] Kirkpatrick J, et al. *Overcoming catastrophic forgetting in neural networks (EWC).* PNAS. 2017;114(13):3521-3526.
20. [C] Huszár F. *Note on the quadratic penalties in EWC.* PNAS. 2018;115(11):E2496-E2497.
21. [C] Rolnick D, Ahuja A, Schwarz J, Lillicrap TP, Wayne G. *Experience replay for continual learning (CLEAR).* NeurIPS 2019. arXiv:1811.11682.
22. [C] Wayne G, et al. *Unsupervised predictive memory in a goal-directed agent (MERLIN).* 2018. arXiv:1803.10760.
23. [C-corr] Ha D, Schmidhuber J. *World Models* (NeurIPS 2018 camera-ready: "Recurrent World Models Facilitate Policy Evolution"). arXiv:1803.10122.
24. [C] Hafner D, Lillicrap T, Ba J, Norouzi M. *Dream to Control: learning behaviors by latent imagination.* ICLR 2020. arXiv:1912.01603.
25. [C-corr] Hafner D, Pasukonis J, Ba J, Lillicrap T. *Mastering diverse control tasks through world models (DreamerV3).* Nature. 2025;640:647-653.
26. [C] Bruce J, et al. *Genie: Generative Interactive Environments.* ICML 2024. arXiv:2402.15391.
27. [C-corr] Gemini Team, Google. *Gemini 1.5.* 2024. arXiv:2403.05530.
28. [C-corr] van de Ven GM, Siegelmann HT, Tolias AS. *Brain-inspired replay for continual learning.* Nat Commun. 2020;11:4069.

**C. LLM-agent memory / persistent-memory systems**
29. [C] Park JS, et al. *Generative agents: interactive simulacra of human behavior.* UIST 2023. arXiv:2304.03442.
30. [C] Packer C, et al. *MemGPT: towards LLMs as operating systems.* 2023. arXiv:2310.08560.
31. [C] Lin K, et al. *Sleep-time compute: beyond inference scaling at test-time.* 2025. arXiv:2504.13171.
32. [C] Chhikara P, et al. *Mem0: building production-ready AI agents with scalable long-term memory.* ECAI 2025. arXiv:2504.19413.
33. [C] Maharana A, et al. *Evaluating very long-term conversational memory of LLM agents (LoCoMo).* ACL 2024. arXiv:2402.17753.
34. [C] Wu D, et al. *LongMemEval.* ICLR 2025. arXiv:2410.10813.
35. [C] Zhang Z, et al. *A survey on the memory mechanism of LLM-based agents.* ACM TOIS. 2025. arXiv:2404.13501.
36. [U] Xie Y. *Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in LLMs.* 2026. arXiv:2603.14517. — **Closest direct prior art; future-dated, near-uncited single-author preprint. Cite as prior-art signal only; verify before relying on claims.**

**D. Anthropic (memory/context/agents) — verified live**
37. [C] Anthropic. *Memory tool.* Claude Developer Platform Docs. `platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool` (tool type `memory_20250818`; GA on Messages API; Claude 4+).
38. [C] Anthropic. *Context editing.* `platform.claude.com/docs/en/build-with-claude/context-editing` (beta `context-management-2025-06-27`; `clear_tool_uses_20250919`; ~84% token reduction).
39. [C] Anthropic. *Managing context on the Claude Developer Platform.* 2025-09-29. `claude.com/blog/context-management` (memory + context editing ≈ 39% agentic-search improvement).
40. [C] Anthropic. *Effective context engineering for AI agents.* 2025-09-29. `anthropic.com/engineering/effective-context-engineering-for-ai-agents`.
41. [C] Schluntz E, Zhang B (Anthropic). *Building effective agents.* 2024-12-19. `anthropic.com/engineering/building-effective-agents`.
42. [C] Hadfield J, et al. (Anthropic). *How we built our multi-agent research system.* 2025-06-13. `anthropic.com/engineering/multi-agent-research-system`.
43. [C] Anthropic. *Claude Agent SDK — overview.* `code.claude.com/docs/en/agent-sdk/overview` (CLAUDE.md project memory; subagents; resumable sessions).
44. [C] Anthropic. *Introducing the Model Context Protocol.* 2024-11-25. `anthropic.com/news/model-context-protocol`.
45. [C] Bai Y, et al. (Anthropic). *Constitutional AI: Harmlessness from AI Feedback.* 2022. arXiv:2212.08073. (tangential framing)
46. [U] Anthropic. *Effective harnesses for long-running agents.* `anthropic.com/engineering/effective-harnesses-for-long-running-agents` — referenced from memory-tool docs; confirm page before citing.

**E. Evaluation, statistics, reproducibility, documentation**
47. [C] Lopez-Paz D, Ranzato M. *Gradient episodic memory for continual learning.* NeurIPS 2017. arXiv:1706.08840.
48. [C] Chaudhry A, Dokania PK, Ajanthan T, Torr PHS. *Riemannian walk for incremental learning.* ECCV 2018. arXiv:1801.10112.
49. [C] Díaz-Rodríguez N, Lomonaco V, Filliat D, Maltoni D. *Don't forget, there is more than forgetting: new metrics for continual learning.* NeurIPS 2018 CL Workshop. arXiv:1810.13166.
50. [C] van de Ven GM, Tuytelaars T, Tolias AS. *Three types of incremental learning.* Nat Mach Intell. 2022;4(12):1185-1197.
51. [C] Henderson P, et al. *Deep reinforcement learning that matters.* AAAI 2018. arXiv:1709.06560.
52. [C] Colas C, Sigaud O, Oudeyer P-Y. *How many random seeds?* 2018. arXiv:1806.08295.
53. [C] Agarwal R, et al. *Deep RL at the edge of the statistical precipice (rliable).* NeurIPS 2021. arXiv:2108.13264.
54. [C] Demšar J. *Statistical comparisons of classifiers over multiple data sets.* JMLR. 2006;7:1-30.
55. [C-corr] Karl F, Kemeter LM, Dax G, Sierak P. *Position: embracing negative results in machine learning.* ICML 2024. arXiv:2406.03980.
56. [C] Pineau J, et al. *Improving reproducibility in ML research (NeurIPS 2019 Reproducibility Program).* JMLR. 2021;22(164).
57. [C] Gebru T, et al. *Datasheets for datasets.* Commun ACM. 2021;64(12):86-92. (arXiv:1803.09010)
58. [C] Mitchell M, et al. *Model cards for model reporting.* FAT* 2019:220-229.

> Items marked **[U]** must be re-verified before final citation (notably #36 Xie 2026 and #46). Low-authority/SEO pages and trade books surfaced in research were dropped per audit; use primary sources above for empirical claims.

---

## Summary (for the owner)

**Project:** *Slow Wave* — a reproducible test bench that gives long-running AI agents an offline "sleep/dream" consolidation cycle, measures whether it improves long-horizon performance and memory efficiency against ground-truth relevance, and reports the result (positive **or** negative) in a peer-review-quality paper.

**Phase list (one-line objectives):**
- **Phase 0 — Foundations:** repo, config, run-manifest + reproducibility harness.
- **Phase 1 — Stream generator:** labeled continual task stream with distractors + ground-truth relevance.
- **Phase 2 — Memory + baseline agent:** dual-store memory; no-sleep wake agent emitting `R[i,j]`.
- **Phase 3 — Dream engine:** four ablatable operators (replay / transfer / downscale / generative-augment) in a 2-phase cycle.
- **Phase 4 — Eval harness + preregistration:** nine control arms, matched budgets, CL metrics + statistics, committed prereg.
- **Phase 5 — Experiments + plots:** run the grid (sim-time + a few real runs), generate figures, decide H1/H0.
- **Phase 6 — Paper:** IMRaD manuscript from the data, verified Anthropic + DeepMind citations, negative-results-as-insight.

**Assumptions to confirm or override:** model defaults (Opus 4.8 agent + dream, Haiku 4.5 for bulk scoring — **A1**); local embeddings (bge-small/MiniLM — **A2**); Python/anthropic/rliable stack (**A3**); ≥5 seeds + budget controller (**A4**); arXiv-first author-date paper (**A5**). Biggest open call to lock early: the **primary endpoint** (OQ1) — final ACC at equal retrieval budget vs. accuracy-per-retrieval-token — since it must be pre-registered before Phase 5 and drives the power analysis. Also open: optional second domain (OQ2) and whether to expose memory via Anthropic's memory tool/MCP for a realism variant (OQ5).

Execute any phase with **`/sprint Phase N`** (start with `/sprint Phase 0`).
