# Slow Wave — Phase 5 Results

> **MOCK-LLM CAVEAT (DX5).** MECHANISM DEMONSTRATION ONLY. Every number in this analysis was produced in the synthetic fact-world stream under the deterministic mock LLM (no ANTHROPIC_API_KEY on this box), so it is reproducible bit-for-bit. The H1/H0 verdict is therefore a statement about the bench mechanism in this synthetic + mock-LLM regime, NOT a scientific claim about a real Claude model. Neuroscience (sleep, replay, TMR) is motivation and analogy only, never a proof of biological fidelity or real-model efficacy (DX5).

Experiment `phase5-full` · scenario `task-incremental` · git `05add10379a5efcbdba95501bcf9ad7a83b86465` · model `claude-opus-4-8` (mocked=True).

## Headline

Mechanism demonstration under the deterministic mock LLM: CONFIRMED H1 in the 'distractor_heavy' regime (mean paired ACC diff full_dream - no_sleep = +0.379, clears the A/A noise floor). On the long-horizon sweep no long-context crossover appears in the swept range (long_context keeps the raw-accuracy lead at maximum cost); the TMR-style replay-targeting effect is g=1.702 versus the 0.29 benchmark (exceeds); realized N=8 (floor 5 met); the observed effect needs N>=2. These are synthetic + mock-LLM mechanism results, not claims about a real Claude model.

## Primary endpoint (EC5)

The single preregistered **primary endpoint** is `acc_diff_full_dream_vs_no_sleep` (full_dream - no_sleep, paired by seed). Verdict: **CONFIRMED** (n=8 paired seeds).

| Field | Value |
| --- | --- |
| Point estimate (paired mean ACC diff) | 0.3792 |
| 95% CI | [0.3292, 0.4292] (percentile) |
| Test | wilcoxon_signed_rank (p=0.0141) |
| Effect | cohens_d = 4.8902 (large) |
| A/A noise floor | 0.0625 (exceeded=True) |

## Per-regime verdicts

| Regime | Verdict |
| --- | --- |
| signal_rich | confirmed |
| balanced | confirmed |
| distractor_heavy (primary) | confirmed |

## Long-context crossover (EC6)

no cost-adjusted crossover in the swept range [2, 3, 5, 8, 12]; long_context dominates on raw accuracy at maximum cost (EC6: absence stated).

Cost-adjusted metric: `acc_per_token` for `full_dream` vs `long_context`.

| L | acc/token (treatment) | acc/token (baseline) | ACC gap (base - treat) |
| --- | --- | --- | --- |
| 2 | 0.000403877 | 0.00238166 | +0.1875 |
| 3 | 0.000218006 | 0.00160707 | +0.2847 |
| 5 | 0.000112338 | 0.000909815 | +0.3500 |
| 8 | 6.78154e-05 | 0.000582072 | +0.3568 |
| 12 | 4.32732e-05 | 0.000380662 | +0.3785 |

Cost-adjusted crossover length: none in range; raw-accuracy crossover length: none in range.

## TMR-style targeting effect (FR5.3)

Replay arms ['full_dream', 'replay_only', 'reflection'] vs no-replay arms ['no_sleep', 'downscale_only']: mean signal-retention lift = +0.2810; bias-corrected Hedges' **g = 1.7017** (95% CI [1.1399, 2.7388]). Benchmark (Hu et al. 2020) **g = 0.29** — exceeds.

replay arms ['full_dream', 'replay_only', 'reflection'] vs no-replay ['no_sleep', 'downscale_only']: mean signal-retention lift=+0.281, bias-corrected unpaired Hedges' g=1.702 (95% CI [1.140, 2.739], bootstrap on unpaired Cohen's d x Hedges factor). Benchmark Hu et al. (2020) g=0.29: exceeds it. Proxy caveat: this is a prioritized-replay targeting ANALOGUE, not a literal cued-TMR protocol.

## Realized power (EC2)

Realized **power**: n_seeds = 8 (floor 5, met=True); observed |d| = 4.8902; required n for the observed effect = 2 (powered=True).

to detect the observed paired d=4.890 at alpha=0.05 (two-sided), power=0.8, Colas et al. (2018) requires n>=2 paired seeds; the run used n=8.

## Sim-vs-real agreement (EC3)

sim preserves the real arm ranking (no inversions); pearson=0.981, spearman=1.000, max|mean ACC sim-real|=0.398

## Negative-result mapping (EC7)

Applicable: **False**. Regime tie: distractor_heavy (signal=0.34, distractor=0.4, noise=0.26).

Matched preregistered negative-result forms:

- Long-context wins below stream length L (consolidation only beats stuff-it-in-context beyond a crossover).

Registered secondary contrasts:

- **full_dream vs replay_only**: full_dream > replay_only (delta=+0.371, 95% CI [0.312, 0.425], wilcoxon p=0.014)

no negative-result form OVERTURNS H1: the primary endpoint is confirmed and the registered secondary contrast is positive. However, the following preregistered negative pattern(s) ARE observed and reported (see the crossover / EC6 section): 'Long-context wins below stream length L (consolidation only beats stuff-it-in-context beyond a crossover).'. These concern secondary cost/length trade-offs, not the matched-budget primary contrast.
