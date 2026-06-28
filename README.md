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

## Status: Phase 1 — Synthetic Continual Task Stream Generator

Phase 0 stood up the repo, configuration, pinned dependencies, the run-manifest
+ reproducibility harness, CI, and a one-command "hello-bench" smoke run
([`docs/PHASE0_CONTRACT.md`](docs/PHASE0_CONTRACT.md)).

Phase 1 builds the measurement substrate: a deterministic **synthetic continual
task stream generator** that emits labeled streams with ground-truth relevance
(`signal`/`distractor`/`noise`), controllable distractor regimes, temporal
structure and contradictions, an explicit continual-learning scenario tag
(task/domain/class-incremental), a Gebru et al. (2021) **datasheet**, and a
held-out **probe set** that computes the `R[i,j]` accuracy matrix against a
trivial oracle. A **confound guard** (FR1.6) enforces — and proves by test —
that ground-truth labels can never reach an online retrieval/priority code path.
The authoritative cross-module interface spec is
[`docs/PHASE1_CONTRACT.md`](docs/PHASE1_CONTRACT.md).

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
- Product requirements & roadmap: [`PRD.md`](PRD.md)
