# Slow Wave

*Slow Wave* is a reproducible research test bench for **sleep-inspired memory
consolidation in autonomous AI agents**. Long-running agents ingest far more
data than ever affects mission performance; biology solves the analogous
problem with sleep. *Slow Wave* gives a frozen-weights LLM agent a periodic,
offline "sleep/dream" cycle that replays recent experience, transfers episodic
traces into durable semantic memory, downscales and prunes low-value memories
while protecting mission-relevant ones, and optionally augments memory
generatively — then **instruments every operation against known ground-truth
relevance** so the effect can be measured falsifiably. The bench generates the
data a final scientific paper reports. See [`PRD.md`](PRD.md) for the full
vision, requirements, and phased roadmap.

## Status: Phase 3 — The Dream Engine (four ablatable operators)

Phase 0 stood up the repo, configuration, pinned dependencies, the run-manifest
+ reproducibility harness, CI, and a one-command "hello-bench" smoke run
([`docs/PHASE0_CONTRACT.md`](docs/PHASE0_CONTRACT.md)).

Phase 1 built the measurement substrate: a deterministic **synthetic continual
task stream generator** that emits labeled streams with ground-truth relevance
(`signal`/`distractor`/`noise`), controllable distractor regimes, temporal
structure and contradictions, an explicit continual-learning scenario tag
(task/domain/class-incremental), a Gebru et al. (2021) **datasheet**, and a
held-out **probe set** that computes the `R[i,j]` accuracy matrix against a
trivial oracle. A **confound guard** (FR1.6) enforces — and proves by test —
that ground-truth labels can never reach an online retrieval/priority code path.
The authoritative cross-module interface spec is
[`docs/PHASE1_CONTRACT.md`](docs/PHASE1_CONTRACT.md).

Phase 3 adds the **dream engine**: four independently-ablatable operators run in
a two-phase (NREM→REM) cycle at scheduled sleep windows. **REPLAY** re-samples
recent episodics (uniform or prioritized, with importance-sampling weights
logged); **TRANSFER** distills sampled episodics into durable **semantic** entries
(preserving the structured fact + provenance) and enforces removable **CLS
interleaving**; **DOWNSCALE** applies a swappable decay curve (exponential /
Weibull / ACT-R) to all salience and re-potentiates only replayed items
("decay all, protect signal"); **GENERATIVE-AUGMENT** synthesizes pseudo-episodes
and logs a generator-fidelity/drift score. An optional conflict/unlearning step
demotes contradictions (never deletes). Each operator is a config toggle (all 2⁴
combinations run), consolidation is **gated to sleep** (no semantic writes during
wake), and a full cycle leaves the provenance + archival audit intact. The
authoritative interface spec is [`docs/PHASE3_CONTRACT.md`](docs/PHASE3_CONTRACT.md).

Phase 2 adds the **dual-store memory substrate** and the **no-sleep wake agent**
— the catastrophic-forgetting reference the whole study is measured against. The
substrate has physically separate **EPISODIC**, **SEMANTIC/SKILL**, and auditable
**ARCHIVAL** tiers with a salience model (recency, access, novelty, provenance),
a pluggable recency×importance×relevance retrieval policy, eviction that
**demotes** rather than deletes, and EWC-spirit **write-protection** (a distractor
overwriting a protected fact is a logged failure event). The wake agent runs a
Phase 1 stream end-to-end **without any dream cycle** (no semantic writes during
wake — those are gated to sleep), populates `R[i,j]` from held-out probes, and
emits cost/footprint telemetry to a manifest. With a capacity-bounded episodic
store on a noisy stream the baseline **demonstrably forgets** (backward transfer
< 0). The authoritative interface spec is
[`docs/PHASE2_CONTRACT.md`](docs/PHASE2_CONTRACT.md).

## Repository layout

| Path | Purpose |
|---|---|
| `slow_wave/stream/` | Synthetic continual task stream generator (labels, distractor regimes, probe sets). |
| `slow_wave/memory/` | Dual-store memory substrate (episodic + semantic/skill + archival tier). |
| `slow_wave/agent/` | Wake-loop agent (ingest, attempt tasks, retrieve, call Claude). |
| `slow_wave/dream/` | The dream engine: four ablatable operators (replay / transfer / downscale / generative-augment). |
| `slow_wave/eval/` | Evaluation harness, control battery, metrics, and statistics. |
| `slow_wave/repro/` | Reproducibility: config-driven seeding, git info, run manifests, and the smoke bench. |
| `configs/` | One YAML config per experiment (including `smoke.yaml`). |
| `tests/` | Pytest suite, including confound guards and the manifest schema test. |
| `docs/` | Design docs and interface contracts. |
| `paper/` | The scientific paper (LaTeX, figures sourced from manifests). |
| `runs/` | Run outputs (manifests, transcripts); git-ignored except `.gitkeep`. |

## Quickstart

Requires **Python 3.12** (3.11+ supported).

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install the package + dev (test) dependencies — no heavy ML deps needed
pip install -e ".[dev]"

# 3. Run the test suite
python -m pytest

# 4. Reproduce the smoke artifact end-to-end (the canonical one command)
python -m slow_wave.repro.smoke --config configs/smoke.yaml
# or, with POSIX make:
make repro-smoke
```

The smoke run writes a manifest to `runs/smoke/manifest.json`.

### Emit a synthetic continual task stream (Phase 1)

```bash
# Deterministic; no LLM and no heavy ML deps required
python -m slow_wave.stream.emit --config configs/stream_smoke.yaml
# or, with POSIX make:
make repro-stream
```

This writes four byte-reproducible artifacts under `runs/stream/`:
`stream.json` (online-safe items + an offline-only label sidecar),
`datasheet.json` (Gebru et al. 2021), `probes.json` (held-out queries + known
answers), and `accuracy_matrix.json` (the `R[i,j]` skeleton vs. the trivial
oracle). Given the same config + seed, every file is byte-identical across runs.

### Run the no-sleep baseline wake agent (Phase 2)

```bash
# Deterministic; hash embeddings + mock LLM by default (no key / heavy deps needed)
python -m slow_wave.agent.runner --config configs/agent_smoke.yaml
# or, with POSIX make:
make repro-agent

# The capacity-bounded, noisy config in which the baseline demonstrably forgets:
python -m slow_wave.agent.runner --config configs/agent_forgetting.yaml
```

This runs a Phase 1 stream end-to-end through the dual-store memory substrate and
writes `runs/agent/manifest.json` carrying the populated `R[i,j]`, the
continual-learning metrics (ACC, BWT, forward transfer, per-task forgetting), the
per-tier memory footprint, and cost/latency telemetry. `agent_smoke.yaml` uses an
unbounded store (clean lower-triangular `R`, BWT≈0); `agent_forgetting.yaml`
bounds the episodic store so older tasks are evicted (demoted to the archival
tier) and backward transfer goes clearly negative.

### Run the dream engine (Phase 3)

```bash
# Deterministic; hash embeddings + mock LLM by default (no key / heavy deps needed)
python -m slow_wave.dream.runner --config configs/dream_smoke.yaml
# or, with POSIX make:
make repro-dream

# The "full-dream" treatment over the forgetting stream:
python -m slow_wave.dream.runner --config configs/dream_full.yaml
```

This runs the same Phase 1 stream through the wake loop but attaches the dream
engine at each scheduled **sleep window**, then writes `runs/dream/manifest.json`
carrying `R[i,j]`, the continual-learning metrics, the per-tier footprint, **both**
wake and dream telemetry (per-cycle operator logs, replay/IS weights, transfer
counts, generator-fidelity/drift), and a **provenance + archival audit**. On the
same bounded, noisy stream where the no-sleep baseline strongly forgets,
consolidation transfers sampled signal facts into the (unbounded) semantic store
so they survive episodic eviction — measurably reducing backward transfer. (Whether
dreaming beats the baseline at *matched budget* is a Phase 4/5 question, not
asserted here.)

### LLM: real call vs. deterministic mock

The smoke run makes a **real Claude API call when `ANTHROPIC_API_KEY` is set**
in the environment, and otherwise falls back to a **deterministic, flagged
mock** so the bench runs (and CI stays green) with no key. Either way the
manifest records whether the call was mocked, and all non-LLM outputs
(embeddings, sampling order, file layout) are reproducible bit-for-bit across
runs with the same config and seeds.

## Heavy / optional dependencies

Phase 0 needs only the pinned core deps. Embeddings, statistics, plotting, and
the vector index are installed via extras when later phases need them:

```bash
pip install -e ".[embeddings,stats,viz,vector]"   # or ".[all]"
# equivalently:
pip install -r requirements-optional.txt
```

## More

- Phase 0 interface contract: [`docs/PHASE0_CONTRACT.md`](docs/PHASE0_CONTRACT.md)
- Phase 1 interface contract: [`docs/PHASE1_CONTRACT.md`](docs/PHASE1_CONTRACT.md)
- Phase 2 interface contract: [`docs/PHASE2_CONTRACT.md`](docs/PHASE2_CONTRACT.md)
- Phase 3 interface contract: [`docs/PHASE3_CONTRACT.md`](docs/PHASE3_CONTRACT.md)
- Product requirements & roadmap: [`PRD.md`](PRD.md)
