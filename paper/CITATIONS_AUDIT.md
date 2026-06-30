# Citation audit — Slow Wave manuscript (Phase 6, EC3)

Every key in `paper/references.bib` is listed below with its confidence tag
(from PRD Appendix A, which was itself citation-audited), a resolvable locator,
and a resolution note. Confidence tags: **C** confirmed · **C-corr** confirmed
with an audit correction (the `.bib` carries the corrected title/venue) ·
**U** uncertain — **flagged as unverified in the manuscript prose**.

No refuted or low-authority source appears: the PRD audit dropped those, and
this bibliography contains exactly the 58 keyed sources and nothing else. The
two **U** items are cited only as unverified prior-art signals and are flagged
in text (`02_related_work.tex` for `xie2026forget`; the `.bib` `note` field for
both).

## A. Sleep & memory neuroscience
| key | tag | locator | status |
|---|---|---|---|
| mcclelland1995cls | C | doi:10.1037/0033-295X.102.3.419 | resolves (Psychol Rev 102(3)) |
| buzsaki1989twostage | C | doi:10.1016/0306-4522(89)90423-5 | resolves (Neuroscience 31(3)) |
| buzsaki2015ripples | C | doi:10.1002/hipo.22488 | resolves (Hippocampus 25(10)) |
| diekelmann2010memory | C | doi:10.1038/nrn2762 | resolves (Nat Rev Neurosci 11(2)) |
| rasch2013sleep | C | doi:10.1152/physrev.00032.2012 | resolves (Physiol Rev 93(2)) |
| klinzing2019systems | C | doi:10.1038/s41593-019-0467-3 | resolves (Nat Neurosci 22(10)) |
| tononi2006shy | C | doi:10.1016/j.smrv.2005.05.002 | resolves (Sleep Med Rev 10(1)) |
| tononi2014price | C | doi:10.1016/j.neuron.2013.12.025 | resolves (Neuron 81(1)) |
| tononi2020downselection | C | doi:10.1111/ejn.14335 | resolves (Eur J Neurosci 51(1)) |
| hu2020tmr | C | doi:10.1037/bul0000223 | resolves (Psychol Bull 146(3)); g≈0.29 benchmark |
| crick1983dreamsleep | C | doi:10.1038/304111a0 | resolves (Nature 304) |
| poe2017forgetting | C | doi:10.1523/JNEUROSCI.0820-16.2017 | resolves (J Neurosci 37(3)) |
| hoel2021overfitted | C | doi:10.1016/j.patter.2021.100244 | resolves (Patterns 2(5)) |

## B. CLS, replay, world models, continual learning (AI)
| key | tag | locator | status |
|---|---|---|---|
| kumaran2016cls | C | doi:10.1016/j.tics.2016.05.004 | resolves (Trends Cogn Sci 20(7)) |
| hassabis2017neuroai | C | doi:10.1016/j.neuron.2017.06.011 | resolves (Neuron 95(2)) |
| mnih2015dqn | C | doi:10.1038/nature14236 | resolves (Nature 518); DeepMind / DQN |
| schaul2016per | C | arXiv:1511.05952 (ICLR 2016) | resolves; DeepMind / PER |
| shin2017generativereplay | C | arXiv:1705.08690 (NeurIPS 2017) | resolves |
| kirkpatrick2017ewc | C | doi:10.1073/pnas.1611835114 | resolves (PNAS 114(13)); EWC |
| huszar2018ewcnote | C | doi:10.1073/pnas.1717042115 | resolves (PNAS 115(11)) |
| rolnick2019clear | C | arXiv:1811.11682 (NeurIPS 2019) | resolves; CLEAR |
| wayne2018merlin | C | arXiv:1803.10760 | resolves; DeepMind / MERLIN |
| ha2018worldmodels | C-corr | arXiv:1803.10122 | resolves; **corrected** camera-ready title "Recurrent World Models Facilitate Policy Evolution" (NeurIPS 2018) |
| hafner2020dreamer | C | arXiv:1912.01603 (ICLR 2020) | resolves; Dreamer |
| hafner2025dreamerv3 | C-corr | doi:10.1038/s41586-025-08744-2; arXiv:2301.04104 | **corrected** to Nature 2025;640:647-653 (DreamerV3) |
| bruce2024genie | C | arXiv:2402.15391 (ICML 2024) | resolves; DeepMind / Genie |
| gemini2024 | C-corr | arXiv:2403.05530 | resolves; corp-author "Gemini Team, Google" |
| vandeven2020brainreplay | C-corr | doi:10.1038/s41467-020-17866-2 | resolves (Nat Commun 11:4069) |

## C. LLM-agent memory
| key | tag | locator | status |
|---|---|---|---|
| park2023generativeagents | C | arXiv:2304.03442 (UIST 2023) | resolves; rec×imp×rel memory stream |
| packer2023memgpt | C | arXiv:2310.08560 | resolves |
| lin2025sleeptime | C | arXiv:2504.13171 | resolves; **Letta sleep-time compute** (closest precedent) |
| chhikara2025mem0 | C | arXiv:2504.19413 (ECAI 2025) | resolves |
| maharana2024locomo | C | arXiv:2402.17753 (ACL 2024) | resolves; LoCoMo |
| wu2025longmemeval | C | arXiv:2410.10813 (ICLR 2025) | resolves; LongMemEval |
| zhang2025memorysurvey | C | arXiv:2404.13501 (ACM TOIS) | resolves |
| **xie2026forget** | **U** | arXiv:2603.14517 | **UNCERTAIN — verify before final cite.** The arXiv id resolves (a recent single-author preprint, "SleepGate"); web checks indicate the author is "Y. Xie" (the `.bib` uses the initial form rather than asserting a full first name). It remains near-uncited and its strong claims are unverified, so it is cited in prose as an *unverified* prior-art signal only, never relied on for a claim. |

## D. Anthropic
| key | tag | locator | status |
|---|---|---|---|
| anthropic_memorytool | C | platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool | resolves (docs); `memory_20250818`, GA, Claude 4+ |
| anthropic_contextediting | C | platform.claude.com/docs/en/build-with-claude/context-editing | resolves (docs); beta `context-management-2025-06-27` |
| anthropic_contextmgmt | C | claude.com/blog/context-management | resolves (blog, 2025-09-29) |
| anthropic_contextengineering | C | anthropic.com/engineering/effective-context-engineering-for-ai-agents | resolves (engineering blog) |
| schluntz2024agents | C | anthropic.com/engineering/building-effective-agents | resolves (2024-12-19) |
| hadfield2025multiagent | C | anthropic.com/engineering/multi-agent-research-system | resolves (2025-06-13) |
| anthropic_agentsdk | C | code.claude.com/docs/en/agent-sdk/overview | resolves (docs); CLAUDE.md, subagents, resumable sessions |
| anthropic_mcp | C | anthropic.com/news/model-context-protocol | resolves (2024-11-25) |
| bai2022cai | C | arXiv:2212.08073 | resolves; Constitutional AI (tangential framing) |
| **anthropic_harnesses** | **U** | anthropic.com/engineering/effective-harnesses-for-long-running-agents | **UNCERTAIN — verify page before final cite.** Referenced from the memory-tool docs but not re-confirmed live at audit time. **Not cited in the manuscript body**; retained in the `.bib` with a `note` flag only. If unconfirmed at submission, omit. |

## E. Evaluation, statistics, reproducibility, documentation
| key | tag | locator | status |
|---|---|---|---|
| lopezpaz2017gem | C | arXiv:1706.08840 (NeurIPS 2017) | resolves; GEM (ACC/BWT/FWT) |
| chaudhry2018riemannian | C | arXiv:1801.10112 (ECCV 2018) | resolves; Forgetting Measure |
| diazrodriguez2018metrics | C | arXiv:1810.13166 | resolves; new CL metrics |
| vandeven2022threetypes | C | doi:10.1038/s42256-022-00568-3 | resolves (Nat Mach Intell 4(12)) |
| henderson2018rldrl | C | arXiv:1709.06560 (AAAI 2018) | resolves |
| colas2018seeds | C | arXiv:1806.08295 | resolves; power analysis |
| agarwal2021rliable | C | arXiv:2108.13264 (NeurIPS 2021) | resolves; rliable |
| demsar2006statistical | C | jmlr.org/papers/v7/demsar06a.html | resolves (JMLR 7:1-30) |
| karl2024negative | C-corr | arXiv:2406.03980 (ICML 2024) | resolves; "Position: Embracing Negative Results in ML" |
| pineau2021reproducibility | C | jmlr.org/papers/v22/20-303.html | resolves (JMLR 22(164)) |
| gebru2021datasheets | C | doi:10.1145/3458723 | resolves (Commun ACM 64(12)) |
| mitchell2019modelcards | C | doi:10.1145/3287560.3287596 | resolves (FAT* 2019:220-229) |

## Summary
- 58/58 keys present; key set matches `paper/KEYS.md` exactly (no missing, no extra).
- 0 refuted/dropped sources present.
- 2 uncertain (`xie2026forget`, `anthropic_harnesses`), both flagged; `anthropic_harnesses` is additionally kept out of the manuscript body.
- DOIs/arXiv ids/stable URLs are provided for every entry as the resolution locator.
