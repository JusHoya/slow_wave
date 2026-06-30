# Canonical citation keys — Slow Wave manuscript (Phase 6)

**This file is the contract between the manuscript body (W1) and the
bibliography (W2).** The body cites with `\citep{key}` / `\citet{key}`; the
`references.bib` file defines an entry for **exactly** these keys. Do not invent
keys outside this list; if a new source is genuinely needed, add it here first.

Keys map 1:1 to **PRD Appendix A** (the citation-audited master reference list).
The **confidence tag** carries the audit verdict and governs how the source may
be used in text:

- **[C]** confirmed — cite normally.
- **[C-corr]** confirmed with an audit correction — cite normally; the `.bib`
  entry must reflect the *corrected* title/venue (see notes).
- **[U]** uncertain — **must be flagged in the prose as unverified** (e.g.
  "an unverified preprint", "page not re-confirmed") wherever cited, per Phase 6
  EC3. Never present a `[U]` source as established fact.

> Phase 6 EC3: every citation must resolve to a real, locatable source; **no
> refuted citation may appear**; uncertain ones are flagged as such in text.
> The audit already dropped refuted/low-authority items — do not re-add them.

## A. Sleep & memory neuroscience
| key | tag | source (Appendix A #) |
|---|---|---|
| `mcclelland1995cls` | C | #1 McClelland, McNaughton & O'Reilly 1995, *Psychol Rev* — why there are CLS |
| `buzsaki1989twostage` | C | #2 Buzsáki 1989, *Neuroscience* — two-stage model |
| `buzsaki2015ripples` | C | #3 Buzsáki 2015, *Hippocampus* — sharp wave-ripple |
| `diekelmann2010memory` | C | #4 Diekelmann & Born 2010, *Nat Rev Neurosci* |
| `rasch2013sleep` | C | #5 Rasch & Born 2013, *Physiol Rev* |
| `klinzing2019systems` | C | #6 Klinzing, Niethard & Born 2019, *Nat Neurosci* |
| `tononi2006shy` | C | #7 Tononi & Cirelli 2006, *Sleep Med Rev* — SHY |
| `tononi2014price` | C | #8 Tononi & Cirelli 2014, *Neuron* — price of plasticity |
| `tononi2020downselection` | C | #9 Tononi & Cirelli 2020, *Eur J Neurosci* — down-selection |
| `hu2020tmr` | C | #10 Hu, Cheng, Chiu & Paller 2020, *Psychol Bull* — TMR meta-analysis (g≈0.29) |
| `crick1983dreamsleep` | C | #11 Crick & Mitchison 1983, *Nature* — reverse learning |
| `poe2017forgetting` | C | #12 Poe 2017, *J Neurosci* — sleep is for forgetting |
| `hoel2021overfitted` | C | #13 Hoel 2021, *Patterns* — overfitted brain |

## B. CLS, replay, world models, continual learning (AI)
| key | tag | source (Appendix A #) |
|---|---|---|
| `kumaran2016cls` | C | #14 Kumaran, Hassabis & McClelland 2016, *Trends Cogn Sci* — CLS updated |
| `hassabis2017neuroai` | C | #15 Hassabis et al. 2017, *Neuron* — neuroscience-inspired AI |
| `mnih2015dqn` | C | #16 Mnih et al. 2015, *Nature* — DQN (experience replay) |
| `schaul2016per` | C | #17 Schaul, Quan, Antonoglou & Silver 2016, ICLR — prioritized experience replay |
| `shin2017generativereplay` | C | #18 Shin et al. 2017, NeurIPS — deep generative replay |
| `kirkpatrick2017ewc` | C | #19 Kirkpatrick et al. 2017, *PNAS* — EWC |
| `huszar2018ewcnote` | C | #20 Huszár 2018, *PNAS* — note on EWC quadratic penalties |
| `rolnick2019clear` | C | #21 Rolnick et al. 2019, NeurIPS — CLEAR |
| `wayne2018merlin` | C | #22 Wayne et al. 2018, arXiv — MERLIN |
| `ha2018worldmodels` | C-corr | #23 Ha & Schmidhuber 2018 — camera-ready title "Recurrent World Models Facilitate Policy Evolution", NeurIPS 2018 (arXiv:1803.10122) |
| `hafner2020dreamer` | C | #24 Hafner et al. 2020, ICLR — Dreamer (latent imagination) |
| `hafner2025dreamerv3` | C-corr | #25 Hafner et al. 2025, *Nature* 640:647-653 — DreamerV3 |
| `bruce2024genie` | C | #26 Bruce et al. 2024, ICML — Genie |
| `gemini2024` | C-corr | #27 Gemini Team, Google 2024, arXiv:2403.05530 — Gemini 1.5 |
| `vandeven2020brainreplay` | C-corr | #28 van de Ven, Siegelmann & Tolias 2020, *Nat Commun* — brain-inspired replay |

## C. LLM-agent memory / persistent-memory systems
| key | tag | source (Appendix A #) |
|---|---|---|
| `park2023generativeagents` | C | #29 Park et al. 2023, UIST — generative agents (memory-stream rec×imp×rel) |
| `packer2023memgpt` | C | #30 Packer et al. 2023, arXiv — MemGPT |
| `lin2025sleeptime` | C | #31 Lin et al. 2025, arXiv:2504.13171 — sleep-time compute (closest precedent) |
| `chhikara2025mem0` | C | #32 Chhikara et al. 2025, ECAI — Mem0 |
| `maharana2024locomo` | C | #33 Maharana et al. 2024, ACL — LoCoMo |
| `wu2025longmemeval` | C | #34 Wu et al. 2025, ICLR — LongMemEval |
| `zhang2025memorysurvey` | C | #35 Zhang et al. 2025, ACM TOIS — survey of LLM-agent memory |
| `xie2026forget` | **U** | #36 Xie 2026, arXiv:2603.14517 — sleep-inspired consolidation for LLMs. **FLAG IN TEXT as an unverified, future-dated single-author preprint; cite as prior-art signal only.** |

## D. Anthropic (memory / context / agents)
| key | tag | source (Appendix A #) |
|---|---|---|
| `anthropic_memorytool` | C | #37 Anthropic, *Memory tool* docs (`memory_20250818`; GA Messages API; Claude 4+) |
| `anthropic_contextediting` | C | #38 Anthropic, *Context editing* docs (beta `context-management-2025-06-27`; `clear_tool_uses_20250919`; ~84% token reduction) |
| `anthropic_contextmgmt` | C | #39 Anthropic blog 2025-09-29, *Managing context* (memory+editing ≈39% agentic-search improvement) |
| `anthropic_contextengineering` | C | #40 Anthropic 2025-09-29, *Effective context engineering for AI agents* |
| `schluntz2024agents` | C | #41 Schluntz & Zhang (Anthropic) 2024-12-19, *Building effective agents* |
| `hadfield2025multiagent` | C | #42 Hadfield et al. (Anthropic) 2025-06-13, *How we built our multi-agent research system* |
| `anthropic_agentsdk` | C | #43 Anthropic, *Claude Agent SDK — overview* (CLAUDE.md memory; subagents; resumable sessions) |
| `anthropic_mcp` | C | #44 Anthropic 2024-11-25, *Introducing the Model Context Protocol* |
| `bai2022cai` | C | #45 Bai et al. (Anthropic) 2022, arXiv:2212.08073 — Constitutional AI (tangential framing) |
| `anthropic_harnesses` | **U** | #46 Anthropic, *Effective harnesses for long-running agents*. **FLAG/verify page before citing; cite only if confirmed live, else omit.** |

## E. Evaluation, statistics, reproducibility, documentation
| key | tag | source (Appendix A #) |
|---|---|---|
| `lopezpaz2017gem` | C | #47 Lopez-Paz & Ranzato 2017, NeurIPS — GEM (ACC/BWT/FWT) |
| `chaudhry2018riemannian` | C | #48 Chaudhry et al. 2018, ECCV — Riemannian walk (Forgetting Measure) |
| `diazrodriguez2018metrics` | C | #49 Díaz-Rodríguez et al. 2018, NeurIPS CL Workshop — new CL metrics |
| `vandeven2022threetypes` | C | #50 van de Ven, Tuytelaars & Tolias 2022, *Nat Mach Intell* — three types of incremental learning |
| `henderson2018rldrl` | C | #51 Henderson et al. 2018, AAAI — deep RL that matters |
| `colas2018seeds` | C | #52 Colas, Sigaud & Oudeyer 2018, arXiv — how many random seeds? (power) |
| `agarwal2021rliable` | C | #53 Agarwal et al. 2021, NeurIPS — rliable (IQM, perf profiles, PoI) |
| `demsar2006statistical` | C | #54 Demšar 2006, JMLR — statistical comparisons (Friedman + post-hoc) |
| `karl2024negative` | C-corr | #55 Karl, Kemeter, Dax & Sierak 2024, ICML — embracing negative results |
| `pineau2021reproducibility` | C | #56 Pineau et al. 2021, JMLR — improving reproducibility (NeurIPS checklist) |
| `gebru2021datasheets` | C | #57 Gebru et al. 2021, *Commun ACM* — datasheets for datasets |
| `mitchell2019modelcards` | C | #58 Mitchell et al. 2019, FAT* — model cards |

## Required-citation crosswalk (Phase 6 EC4)
Related Work **must** cite, with correct attributions:
- **Anthropic:** memory tool (`anthropic_memorytool`), context editing
  (`anthropic_contextediting`), Agent SDK (`anthropic_agentsdk`), Letta
  sleep-time compute (`lin2025sleeptime`).
- **DeepMind:** DQN (`mnih2015dqn`), prioritized experience replay
  (`schaul2016per`), MERLIN (`wayne2018merlin`), CLS-for-AI
  (`kumaran2016cls` / `hassabis2017neuroai`), world models
  (`ha2018worldmodels` / `hafner2020dreamer` / `hafner2025dreamerv3`).
