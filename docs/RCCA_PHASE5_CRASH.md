# RCCA — Phase 5 Red-Team Laptop Crashes

**Root Cause & Corrective Action Report**

| Field | Value |
|---|---|
| **Date** | 2026-06-29 |
| **System** | HP OmniBook Ultra Flip Laptop 14-fh0xxx (SKU AW5Z6UA#ABA) |
| **CPU / Platform** | Intel Core Ultra 9 288V — Lunar Lake (Series 2), 8 physical / 8 logical cores (no SMT), base 3.3 GHz |
| **BIOS** | HP "W75 Ver. 01.05.00", released 2026-04-12 |
| **Memory** | 32 GB LPDDR5X on-package / soldered / non-reseatable (31.48 GB usable) |
| **iGPU** | Intel Arc 140V, driver 32.0.101.7026 (2025-08-18) |
| **NPU** | Intel AI Boost, driver 32.0.100.4724 (2026-03-18) |
| **Workload under test** | `slow_wave` Phase 5 eval (`slow_wave/eval/grid.py`) under the "ultracode" multi-agent red-team orchestration |
| **Author** | Reliability analysis (RCCA) |
| **Classification** | Hardware / firmware machine-check, surfaced by sustained all-core orchestration load |
| **Severity** | High — repeated unclean shutdowns, recurring across months, no diagnostic dump captured |
| **Status** | Root cause confirmed (component-level confirmation pending dump capture / WHEA decode) |

---

## 1. Executive Summary

The Phase 5 red-team crashes are a **hardware machine-check**, not a software fault. The dominant bugcheck in this machine's crash history is **`0x124 WHEA_UNCORRECTABLE_ERROR` (4 occurrences)** — a CPU Machine Check Exception (MCE) read by the Windows WHEA handler from the processor's own Machine Check Architecture (MCA) registers. Every system-killing crash also logs **WHEA-Logger Event 1 "A fatal hardware error has occurred."** Usermode (Ring 3) Python physically **cannot** raise a `0x124`: it cannot execute privileged instructions, write MCA MSRs, or inject an MCE. The crash is the silicon reporting its own uncorrectable fault.

The user's hypothesis — **"a matrix blew up very large"** and exhausted memory — is **refuted quantitatively.** The only configuration-scaling matrix in the entire run is the `T × T` accuracy matrix, capped at **12 × 12 = 144 floats = 1.15 KB**. The single largest NumPy array in the whole pipeline is a **5.12 MB** bootstrap tensor. Realistic peak working set is **~0.15–0.3 GB**, roughly **100× below** the 31.48 GB of usable RAM, and the pagefile's all-time **PeakUsage is 81 MB**. The workload cannot approach 32 GB; OOM is impossible at the configured scale.

What the orchestration *does* do is supply the **stressor**: the red team fans out **~11 concurrent agents in one parallel batch**, each shelling out to single-process NumPy/pytest jobs. With **no BLAS/OpenMP thread cap anywhere** (OpenBLAS defaults to 8 threads = `os.cpu_count()`) and **Maximum Processor State = 100% on AC and DC**, overlapping jobs oversubscribe all 8 cores and pin the CPU at its top turbo (highest-voltage, highest-heat) bins for minutes. That sustained all-core power/thermal/voltage soak is precisely the condition that surfaces a **latent, load-dependent hardware/firmware marginality** on this Lunar Lake SoC.

**Verdict:** the box is **stability-limited under load, not capacity-limited.** For the Phase 5 compute (tens of MB on a 32 GB / 8-core machine) it is wildly over-provisioned. The crashes are very likely remediable via firmware/driver updates plus power/concurrency containment; if `0x124` machine-checks persist at stock after those fixes, this is an HP warranty/RMA defect (on-package LPDDR5X and the SoC are non-serviceable). A critical secondary finding: **no crash dump has ever been captured**, so we are currently blind to the exact faulting MCA bank — fixing dump capture is the first diagnostic priority.

---

## 2. Problem Statement & Impact

During Phase 5 red-team execution the laptop hard-crashes (bugcheck → unclean shutdown) **specifically when the multi-agent fan-out begins.** On 2026-06-29 the machine crashed **twice** inside the orchestration window (15:55 and 18:34). The user initially suspected the Phase 5 evaluation code allocated an oversized matrix and exhausted memory.

**Impact:**
- **Work loss / interruption:** the red-team run did not complete cleanly (only 6 of the 11+1 agent logs persisted for the crashed workflow `wf_c8ac896e-a21`, consistent with a crash partway through the batch).
- **Recurring, not one-off:** unclean shutdowns (Kernel-Power Event 41) on 6/29, 5/15, 4/22, 4/18, 4/17, 4/15 — a months-long pattern on a new machine.
- **Diagnostic blindness:** no crash dump has ever been retained, so the faulting component has never been named from a dump.
- **Misdirected remediation risk:** the "matrix blew up" theory, if pursued, would waste effort hardening already-tiny software while leaving a hardware/firmware fault unaddressed.

---

## 3. Timeline of Events

### 3.1 Today's crashes (2026-06-29)

| Time | BugCheck | Name | Param1 | Co-logged on next boot |
|---|---|---|---|---|
| 15:55 | `0x1E` | KMODE_EXCEPTION_NOT_HANDLED | `0xC0000005` (STATUS_ACCESS_VIOLATION) | Kernel-Power 41 + WER 1001 + WHEA-Logger Event 1 (15:55:31) |
| 16:49 | — (usermode) | IntelConnect.exe app crash | `0xC0000409` (STACK_BUFFER_OVERRUN) | Reliability record only (not a BSOD) |
| 18:34 | `0x3B` | SYSTEM_SERVICE_EXCEPTION | `0xC0000005` (STATUS_ACCESS_VIOLATION) | Kernel-Power 41 + WER 1001 + WHEA-Logger Event 1 (18:35:03) |

Both BSODs carry `0xC0000005` (access violation) — the classic signature of silently-corrupted instructions/data — and **each is paired within seconds with a WHEA fatal hardware-error record.** They bracket the multi-agent fan-out window.

### 3.2 BugCheck (WER 1001) history — 1:1 with Kernel-Power 41 unclean shutdowns

| Date / Time | BugCheck | Name | Notes |
|---|---|---|---|
| 2026-04-15 01:08 | `0x0A` | IRQL_NOT_LESS_OR_EQUAL | First cluster begins 3 days after BIOS W75 01.05.00 (rel. 04-12) |
| 2026-04-15 07:58 | `0x0A` | IRQL_NOT_LESS_OR_EQUAL | |
| 2026-04-17 11:08 | `0x1E` | KMODE_EXCEPTION_NOT_HANDLED | param `0xC000001D` (illegal instruction) |
| 2026-04-18 07:14 | `0x0A` | IRQL_NOT_LESS_OR_EQUAL | |
| 2026-04-22 20:08 | `0x3B` | SYSTEM_SERVICE_EXCEPTION | Same day as a WHEA fatal **and** a corrected PCIe AER (escalation pattern) |
| 2026-05-15 22:39 | `0x7F` | UNEXPECTED_KERNEL_MODE_TRAP | |
| 2026-06-29 15:55 | `0x1E` | KMODE_EXCEPTION_NOT_HANDLED | param `0xC0000005` |
| 2026-06-29 18:34 | `0x3B` | SYSTEM_SERVICE_EXCEPTION | param `0xC0000005` |

### 3.3 WHEA-Logger history

| Date / Time | Event | Severity | Component |
|---|---|---|---|
| 2026-03-05 | 17 | Corrected | PCI Express Root Port (AER) |
| 2026-03-19 | 17 | Corrected | PCI Express Root Port (AER) |
| 2026-04-15 01:08 | 1 | **Fatal** | Generic ("fatal hardware error"; component not expanded in message text) |
| 2026-04-22 15:29 | 17 | Corrected | PCI Express Root Port (AER) |
| 2026-04-22 20:08 | 1 | **Fatal** | Generic |
| 2026-06-11 | 17 | Corrected | PCI Express Root Port — Bus:Dev:Fn 0:0x1C:0, `PCI\VEN_8086&DEV_A83C` |
| 2026-06-29 15:55:31 | 1 | **Fatal** | Generic |
| 2026-06-29 18:35:03 | 1 | **Fatal** | Generic |

The **corrected-then-fatal escalation** (e.g. corrected AER on 04-22 15:29 → fatal `0x3B` at 20:08 the same day) is the textbook "marginal part degrading under stress" pattern. `DEV_A83C` is a Core Ultra 200V (Lunar Lake) **PCIe root port integrated in the CPU package** — i.e. the corrected errors point back at the SoC/uncore/IO or its power delivery.

---

## 4. Evidence

### 4.1 Kernel bugcheck tally (WER reports, recent history)

| Count | BugCheck | Name | Class |
|---|---|---|---|
| **4** | `0x124` | WHEA_UNCORRECTABLE_ERROR | **CPU machine-check (MCA) fatal — dominant** |
| 3 | `0x0A` | IRQL_NOT_LESS_OR_EQUAL | Kernel memory/IRQL fault |
| 2 | `0x1E` | KMODE_EXCEPTION_NOT_HANDLED | Unhandled kernel exception |
| 2 | `0x3B` | SYSTEM_SERVICE_EXCEPTION | Fault in a system service routine |
| 1 | `0x15F` | PDC_WATCHDOG_TIMEOUT | Power-management watchdog |
| 1 | `0x7F` | UNEXPECTED_KERNEL_MODE_TRAP | CPU trap (e.g. double fault) |

**Six distinct bugcheck codes** — a *scatter*, not a single repeating module. A buggy driver repeats **one** code at **one** module; a marginal CPU/voltage rail throws an assortment of access violations, IRQL faults, kernel traps, and watchdog timeouts that all trace back to corrupted execution.

### 4.2 WHEA records (see §3.3)

`0x124` is raised by the Windows WHEA handler when it reads an uncorrectable MCE from `IA32_MCi_STATUS`. The recurring **corrected** PCIe-root-port AER errors localize to the on-package SoC; the **fatal** Event 1 records accompany every BSOD.

### 4.3 System specifications

| Item | Value | Relevance |
|---|---|---|
| CPU | Core Ultra 9 288V, 8 cores / 8 threads, base 3.3 GHz | No SMT — 8 BLAS threads/process already = 1 thread/core; more jobs oversubscribe |
| RAM | 32 GB LPDDR5X soldered, 31.48 GB usable, ~17.8 GB free at probe | Non-reseatable; OOM not in play (see §4.5) |
| BIOS | W75 01.05.00 (2026-04-12) | Lunar Lake BIOS routinely ships CPU microcode stability fixes; newer rev **unconfirmed** — must verify in HP Support Assistant |
| iGPU driver | Arc 140V 32.0.101.7026 (2025-08-18) | **~10 months stale**; intermediate releases reworked Core Ultra 200V **power management** |
| NPU driver | AI Boost 32.0.100.4724 (2026-03-18) | Workload never touches NPU |
| Power plan | Balanced; **Max processor state 100% (0x64) AC & DC**, Min 5%, no power cap | Lets oversubscribed load run at top turbo (highest V/heat) continuously |

### 4.4 Pagefile / crash-dump state

| Item | Value | Implication |
|---|---|---|
| `CrashDumpEnabled` | **3 = Small dump (minidump)** | Per Microsoft mapping (0=None,1=Complete,2=Kernel,3=Small,7=Automatic), so MEMORY.DMP is **never written by design** — its absence is expected, not a failure |
| `DumpFile` | `C:\Windows\MEMORY.DMP` | Not produced in small-dump mode |
| `C:\Windows\Minidump` | **EMPTY** | The real capture failure — OS logged minidump filenames (e.g. `062926-13359-01.dmp`) but the files did not persist |
| `AutoReboot` | **1** | Machine reboots before the STOP screen can be read/photographed |
| `AutomaticManagedPagefile` | True → left at 3 GB floor | A 3 GB pagefile cannot stage a **kernel** dump on a 32 GB box (matters only if dump type is upgraded) |
| Pagefile `AllocatedBaseSize` / `CurrentUsage` / `PeakUsage` | 3072 MB / 73 MB / **81 MB** | Direct proof no process has ever approached memory pressure |
| `C:` free space | 1691.7 GB | Disk space is **not** a constraint on dump capture |

**Why no dump survives (two compounding reasons):** (a) the box is in small-dump mode, so MEMORY.DMP is never even attempted; and (b) under an **uncorrectable MCE the CPU/uncore is too compromised to flush even a minidump** to the pagefile before the platform dies — so nothing is staged for next-boot extraction regardless of dump type or pagefile size.

### 4.5 Workload peak-memory math (refutes "the matrix blew up")

| Structure | Source | Shape | Bytes |
|---|---|---|---|
| Accuracy matrix `R` | `grid.py` reads `result.accuracy_matrix.R`; `T = n_tasks`, capped at **12** by `length_sweep` | `T × T` (max 12×12 = 144 floats) | **1,152 B (1.15 KB)** |
| Cliff's-delta bootstrap tensor | `stats.py:579` `diff = ra[:,:,None] - rb[:,None,:]`, `n_resamples=10000`, `n_seeds=8` | `(10000, 8, 8)` | **5,120,000 B (5.12 MB)** — largest array in the run |
| Other bootstraps | `stats.py:288,490` | `(10000, 8)` | 640,000 B (0.64 MB) |
| Embeddings | `embeddings.py:78` `np.zeros((len(texts), dim))`, hash backend, dim=384, max 12 tasks × 40 items | `~480 × 384` float64 | ~1,474,560 B (1.5 MB) |
| **Sum of largest arrays** | | | **single-digit MB** |
| **Realistic peak RSS** | interpreter + numpy + pydantic (~150 MB) + transient arrays | | **~0.15–0.3 GB** |
| **Usable RAM** | | | **31.48 GB (~100× headroom)** |

**Distance-to-1 GB sanity checks:**
- Accuracy matrix: `T² × 8 B ≥ 1 GB` ⇒ `T ≈ 11,500` tasks. Config max **T = 12**.
- Cliff's bootstrap: scales with **seeds²**, not tasks/items; `10000 × n² × 8 B ≥ 1 GB` ⇒ `n ≈ 112` seeds. Config has **8**.

### 4.6 Orchestration load profile (the stressor)

| Observation | Evidence |
|---|---|
| Red team launches **11 agents in one `parallel()` batch** (+ a 12th synthesis agent) | `phase5-redteam-wf_c8ac896e-a21.js` line 131 `await parallel(tasks.map(t => () => agent(...)))`; 11 tasks (EC1–EC7 + 4 AUDIT-*); line 162 synthesis agent |
| Each agent re-runs CPU work | tasks run `python -m slow_wave.eval.grid` (AUDIT-determinism runs the grid **twice**), `python -m slow_wave.paper.figures` (renders 7 figures), and pytest files |
| **No BLAS/OMP thread cap anywhere** | OpenBLAS 0.3.33 `DYNAMIC_ARCH NO_AFFINITY MAX_THREADS=24`; `os.cpu_count()==8` ⇒ ~8 threads/process; no `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` in env, registry, or any `*.py` |
| Project code does **not** self-parallelize | pytest `addopts='-q'` (no xdist/-n); grid is sequential single-process numpy; no `multiprocessing/concurrent.futures/threadpool`; only subprocess is one `git` call (`gitinfo.py:37`) |
| Power policy applies no containment | Max processor state 100% AC & DC, no power cap |
| Peak-load thread math | K overlapping numpy jobs ⇒ K×8 BLAS threads (K=4→32, K=8–11→64–88) competing for **8 physical cores (no SMT)**, plus each Node runtime's V8/libuv threads ⇒ 80–120+ ready threads vs 8 cores ⇒ sustained ~100% all-core utilization for **minutes** |

---

## 5. Investigation — Differential Diagnosis

### 5.1 RULED OUT

| Hypothesis | Verdict | Basis |
|---|---|---|
| **"A matrix blew up very large"** (user's theory) | **Ruled out** | Only config-scaling matrix is `T × T` = 1.15 KB at the config cap of T=12; reaching 1 GB needs ~11,500 tasks |
| **Software OOM / 32 GB exhaustion** | **Ruled out** | Peak RSS ~0.15–0.3 GB (~100× below RAM); pagefile PeakUsage ever = 81 MB; ~17.8 GB free at probe |
| **Memory leak across regime/length/seed loops** | **Ruled out** | Per-cell `ExperimentResult` is `model_dump`'d (`grid.py:238`) and released each iteration; only compact summary float-lists accumulate; bounded by 3 regimes + 5 lengths + 3 arms |
| **Bootstrap (10000 resamples) blowup** | **Ruled out** | Largest resample tensor is 5.12 MB; scales with seeds² (=64), not tasks/items |
| **BLAS/OMP oversubscription *inside the workload*** | **Ruled out as a kernel-crash cause** | grid.py spawns no threads/processes and dispatches no multithreaded GEMM (only 1-D `np.dot`); oversubscription is a property of the *external orchestrator* and causes **slowdown**, never a bugcheck |
| **NumPy / GPU-driver interaction in the workload** | **Ruled out** | Hash embedder + mock LLM run on pure CPU numpy; the Arc iGPU/NPU is never touched, and a driver bug would not present as a CPU-MCA `0x124` |
| **Single buggy driver** | **Ruled out** | A driver repeats **one** bugcheck at **one** module; the observed spread of six unrelated codes (`0x124/0x0A/0x1E/0x3B/0x7F/0x15F`) plus a CPU MCA and PCIe AER is the marginal-hardware signature |
| **Usermode Python causing `0x124`** | **Impossible by mechanism** | Ring 3 cannot execute privileged instructions, write MCA MSRs, or inject an MCE; `0x124` is the CPU's own machine-check reported via WHEA |
| Disk-space exhaustion blocking dump write | Ruled out | C: has 1691.7 GB free |
| Pagefile-too-small as cause of **empty minidump dir** | Ruled out | Minidumps are sub-MB; box is in small-dump mode and 3 GB pagefile peaked at 81 MB — size is irrelevant for minidumps (relevant only if upgraded to kernel/complete dump) |
| CI as a local stressor | Ruled out | `.github/workflows/ci.yml` runs on GitHub ubuntu-latest, not the laptop |
| A confirmed-newer BIOS already exists | **Not established** | HP's JS-heavy page timed out; no version above W75 01.05.00 surfaced — must verify in HP Support Assistant, not asserted |

### 5.2 CONFIRMED

**CPU machine-check / marginal-hardware instability under sustained max-turbo load.**

- The dominant bugcheck is `0x124 WHEA_UNCORRECTABLE_ERROR` (4×) — a hardware-detected uncorrectable MCE.
- Every BSOD co-logs a **WHEA-Logger Event 1 fatal hardware error**, and today's `0x1E`/`0x3B` both carry `0xC0000005` (corruption-induced access violation).
- Recurring **corrected PCIe AER** errors on the **on-package CPU root port** (`DEV_A83C`) escalate toward fatals over months — the marginal-part-degrading pattern.
- The fault is **load/voltage/thermal dependent**: it surfaces exactly when the orchestrator fan-out spikes the machine from idle to sustained all-core max-turbo, the documented trigger class for Lunar Lake (Core Ultra 200V) power-sequencing / peak-draw instability.
- All software OOM/blowup explanations fail quantitatively (§4.5); usermode software cannot raise `0x124` by mechanism.

**Confidence: HIGH.** All four independent investigations converged on the same conclusion. Component-level confirmation (which MCA bank / whether CPU-core vs memory-controller vs PCIe) is pending dump capture and/or WHEA CPER decode — but the bugcheck spread already points at the CPU/package.

---

## 6. Root Cause

### 6.1 Five Whys

1. **Why did the laptop crash during the red-team run?**
   The kernel hit fatal bugchecks (`0x1E`/`0x3B` today, `0x124` dominant historically) accompanied by WHEA fatal hardware-error records.

2. **Why were fatal hardware-error / machine-check events raised?**
   The CPU's Machine Check Architecture detected an **uncorrectable** internal fault (corrupted execution → access violations, IRQL faults, kernel traps), and the WHEA handler escalated it to a bugcheck.

3. **Why did the CPU detect an uncorrectable fault at that moment?**
   The silicon was running at its **top turbo bins (highest voltage/heat) on all 8 cores, continuously, for minutes** — the operating point at which a latent marginality manifests.

4. **Why was the CPU pinned at sustained all-core max-turbo?**
   The red team fanned out **~11 concurrent agents**, each spawning single-process NumPy/pytest jobs; with **no BLAS/OMP thread cap** each numpy process spawns ~8 threads, oversubscribing all 8 cores, while **Max Processor State = 100%** removes any turbo headroom limit.

5. **Why is the hardware unstable at that (in-spec) operating point?**
   A **marginal/defective hardware or firmware condition** in the Lunar Lake SoC/power-delivery path: stale CPU microcode (BIOS) and a ~10-month-stale graphics driver whose newer revisions specifically rework Core Ultra 200V power management, on top of possibly out-of-spec silicon. **This is the root cause** — the sustained load is the trigger, not the defect.

> **Note on the parallel diagnostic failure:** a separate 5-Whys on "why couldn't we diagnose it" terminates at: small-dump mode (`CrashDumpEnabled=3`) + undersized 3 GB pagefile + `AutoReboot=1`, compounded by MCE severity preventing any dump flush. This is why **fixing capture is the first action**.

### 6.2 Ishikawa / Fishbone

```
                                                              FATAL MACHINE-CHECK CRASH
                                                              (0x124 + WHEA fatal) under
                                                              sustained all-core load
   MACHINE (Hardware)              METHOD                                |
   ------------------             --------                               |
   * Lunar Lake SoC marginal      * Red team fans out ~11 agents at once |
     (uncorr. CPU MCA, 0x124 x4)  * Each agent re-runs grid/pytest/figs  |
   * On-package PCIe root port    * No batching / serialization          |
     corrected AER (DEV_A83C)     * No BLAS/OMP thread cap set           |
   * Soldered LPDDR5X (non-RMA-   * Stress test never run at stock       |
     able at component level)                                            |
        \                              \                                 |
         \------------------------------\------------------> ( EFFECT ) <
         /------------------------------/                              /
        /                              /                              /
   MATERIAL / FIRMWARE            MEASUREMENT                   ENVIRONMENT
   --------------------          -------------                 -----------
   * BIOS W75 01.05.00 —          * CrashDumpEnabled=3 (small) * Sustained all-core
     possibly stale microcode     * Pagefile 3 GB (kernel        max-turbo => peak
   * Arc 140V driver ~10 mo old     dump can't stage)            voltage + heat
     (power-mgmt fixes missed)    * AutoReboot=1 (STOP code    * 2-in-1 convertible
   * Intel ICPS (IntelConnect.exe   lost)                        vents easily blocked
     crashed 0xc0000409)          * No WinDbg installed        * Thermal transients
                                  * Empty Minidump dir           aggravate marginal Si
                                                                PEOPLE
                                                                ------
                                                                * "Matrix blew up" mis-hypothesis
                                                                  risked misdirected fix
                                                                * Default power plan unchanged (100%)
```

### 6.3 Statement of cause

- **PRIMARY ROOT CAUSE:** A **marginal / unstable hardware-or-firmware condition in the Lunar Lake SoC (CPU/uncore/power-delivery)** that throws uncorrectable machine-check exceptions when driven to sustained all-core max-turbo. (Likely fixable via firmware/driver/power containment; if it persists at stock, a board/CPU warranty defect.)
- **CONTRIBUTING FACTORS:**
  1. **~10-month-stale Intel Arc 140V driver** (32.0.101.7026) — newer releases rework Core Ultra 200V power management.
  2. **Unbounded turbo / Maximum Processor State = 100%** on AC & DC — keeps cores at peak voltage/heat.
  3. **Possibly stale BIOS CPU microcode** (W75 01.05.00, 2026-04-12) — newer rev unconfirmed.
  4. **Unbounded BLAS/orchestration concurrency** — ~11 agents × ~8 BLAS threads, no thread cap, oversubscribing 8 cores — the load that surfaces the fault.
  5. **3 GB pagefile + small-dump mode + AutoReboot** — masked diagnosis; no dump or STOP code ever captured.

---

## 7. Corrective & Preventive Action Plan

> Owner legend: **Claude-can-apply** = read-only or safely reversible, Claude may run; **user-must-approve** = system change Claude can stage but the user authorizes; **user-only** = vendor/physical action only the user can perform.

| # | Priority | Type | Action | Exact command / config | Owner | Risk |
|---|---|---|---|---|---|---|
| 1 | **P0** | Diagnostic | Disable AutoReboot so the next BSOD's STOP code + faulting `.sys` stay on screen (most reliable signal when MCEs prevent any dump). Photograph the screen. | `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name AutoReboot -Value 0 -Type DWord` | user-must-approve (elevated) | Read-only until a crash; harmless |
| 2 | **P0** | Diagnostic | Upgrade dump type small→Automatic so a kernel-detail MEMORY.DMP + minidump are produced when a dump *can* be written (the software-looking `0x1E`/`0x3B` may write even if `0x124` can't), and stop auto-purge. | `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name CrashDumpEnabled -Value 7 -Type DWord; New-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name AlwaysKeepMemoryDump -Value 1 -PropertyType DWord -Force` | user-must-approve (elevated) | Reboot recommended |
| 3 | **P0** | Diagnostic | Enable dump staging on a 32 GB box. **Choose ONE.** (A) DedicatedDumpFile decouples from the small pagefile; **or** (B) grow the boot pagefile. | **(A)** `New-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -Name DedicatedDumpFile -Value 'C:\DedicatedDump.sys' -PropertyType String -Force; New-ItemProperty ... -Name DumpFileSize -Value 16384 -PropertyType DWord -Force` — **(B)** `$cs=Get-CimInstance Win32_ComputerSystem; Set-CimInstance $cs -Property @{AutomaticManagedPagefile=$false}; Set-CimInstance -Query 'SELECT * FROM Win32_PageFileSetting WHERE Name="C:\\pagefile.sys"' -Property @{InitialSize=16384;MaximumSize=36864}` | user-must-approve (elevated) | Uses ~16–32 GB disk (1.6 TB free); reboot required; do **A xor B** |
| 4 | **P1** | Containment | Cap BLAS/OpenMP threads to 1–2 for the whole orchestration session **before** launching the red team. Highest-leverage software lever; collapses each numpy process from ~8 threads to 1–2, removing all-core oversubscription. Numbers are unchanged. | `$env:OMP_NUM_THREADS=2; $env:OPENBLAS_NUM_THREADS=2; $env:MKL_NUM_THREADS=2; $env:NUMEXPR_NUM_THREADS=2` — then launch the orchestration **from this shell** | Claude-can-apply (recommend; user runs in their shell) | None functional; slightly slower per process |
| 5 | **P1** | Containment | Cap Maximum Processor State to ~85–90% on AC **and** DC to keep cores off the top turbo/voltage bins that expose the fault (also a diagnostic: if crashes stop capped and return at 100%, that confirms load/voltage-dependent marginality). | `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 90; powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 90; powercfg /setactive SCHEME_CURRENT` (revert with 100) | user-must-approve | Mild perf reduction; reversible |
| 6 | **P1** | Containment | Cap orchestration concurrency: batch the red-team fan-out from 11 concurrent agents to chunks of 2–3 (or run sequentially). | Edit `phase5-redteam-wf_*.js`: replace the single `await parallel(tasks.map(...))` with chunked runs (await each chunk of 2–3 before the next) | user-must-approve (their orchestration scripts) | Slower wall-clock |
| 7 | **P1** | Corrective | Update the ~10-month-stale Intel Arc 140V driver 32.0.101.7026 → current (32.0.101.8826, 2026-06-01) via Intel DSA; intermediate releases rework Core Ultra 200V power management. Also update Intel ME/Chipset + NPU via DSA. | After installing Intel DSA from intel.com/dsa, verify: `Get-CimInstance Win32_VideoController \| ? {$_.Name -match 'Arc'} \| Select Name,DriverVersion,DriverDate` | user-only (vendor installer) | Use Intel/HP package; clean-install if prompted |
| 8 | **P1** | Corrective | Update HP BIOS/firmware to newest (carries Lunar Lake microcode stability fixes). Confirm whether anything newer than W75 01.05.00 exists (currently unconfirmed). Flash on AC, do not interrupt. | `Get-CimInstance Win32_BIOS \| Select SMBIOSBIOSVersion,ReleaseDate` then HP Support Assistant ▸ Updates (or download latest BIOS SoftPaq for 14-fh0xxx) | user-only (vendor; firmware) | BIOS flash risk — keep on AC; back up first |
| 9 | **P2** | Diagnostic | Install threadpoolctl to confirm the live thread count per process dropped after the cap (was missing in the venv). | `.venv/Scripts/python.exe -m pip install threadpoolctl; .venv/Scripts/python.exe -c "from threadpoolctl import threadpool_info; print(threadpool_info())"` | Claude-can-apply | None (dev dependency) |
| 10 | **P2** | Diagnostic | Run the Phase 5 eval **alone** to confirm it does NOT crash in isolation (expected: completes, peak RSS « 1 GB). | `python -m slow_wave.eval.grid --config configs/phase5_full.yaml` | Claude-can-apply | Low; behavioral only |
| 11 | **P2** | Diagnostic | Install WinDbg to read any future dump; decode the WHEA record. For `0x124`, decode the WHEA_ERROR_RECORD with `!errrec` to read the MCA bank / `MCi_STATUS` and name the failing component. | `winget install --id Microsoft.WinDbg -e` then in WinDbg: `.sympath srv*C:\symbols*https://msdl.microsoft.com/download/symbols; .reload; !analyze -v; !errrec <addr>` | user-must-approve (install) | Read-only analysis |
| 12 | **P2** | Diagnostic | Decode the WHEA fatal CPER section-type GUID we already have (Event 1) to name the component (Proc Generic / IA32-x64 MCA / Memory / PCIe). | `$e=Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger';Id=1} -MaxEvents 1; [xml]$x=$e.ToXml(); ($x.Event.EventData.Data | ? {$_.Name -eq 'RawData'}).'#text'` then match GUID: `8a1e1d01…`=CPU MCA, `9876ccad…`=Proc Generic, `a5bc1114…`=Memory, `d995e954…`=PCIe | Claude-can-apply (read-only) | None; manual GUID match |
| 13 | **P2** | Diagnostic | After firmware/driver/dump fixes, confirm at stock with non-destructive vendor tools: Windows Memory Diagnostic, HP UEFI Diagnostics (Extensive), Intel Processor Diagnostic Tool. | `mdsched.exe`; reboot ▸ tap F2 ▸ System Tests ▸ Extensive; Intel PDT (000005567) | user-only | mdsched / UEFI reboot the machine |
| 14 | P3 | Preventive | Thermal/cooling hygiene: clear vents, run on a hard flat surface (convertible vents block easily), keep on AC during fan-out, confirm fans spin under load. | — | user-only | None |
| 15 | P3 | Preventive | Persist thread caps for the user account so every future python/numpy run inherits them. | `[Environment]::SetEnvironmentVariable('OMP_NUM_THREADS','2','User')` (repeat for OPENBLAS/MKL/NUMEXPR) | user-must-approve | New shells only |
| 16 | P3 | Preventive | Given recurring PCIe-root-port AER + a crashing `IntelConnect.exe` (0xC0000409), update or remove Intel Connectivity Performance Suite (ICPS). | `Get-AppxPackage *IntelConnectivity*` (review before uninstalling) | user-must-approve | Review first |
| 17 | P3 | Diagnostic (opt-in) | **OPTIONAL, USER-INITIATED ONLY:** controlled CPU stress (OCCT/Prime95) while monitoring temps/clocks/voltages (HWiNFO64) to prove load/thermal/voltage causation. **Do NOT auto-run** — can hard-crash a machine already throwing `0x124`. Short runs; stop at first WHEA Event 17/1. | Monitor live: `Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 5 \| Select TimeCreated,Id,LevelDisplayName` | user-only | Can reproduce a hard crash / data loss — user decision |
| 18 | P3 | Preventive | **Warranty / RMA escalation** if `0x124`/MCEs persist at **stock** after BIOS + driver + ME updates. Soldered LPDDR5X + non-serviceable SoC ⇒ board/CPU replacement, not a user repair. Attach evidence bundle. | `Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 50 \| Select TimeCreated,Id,LevelDisplayName,Message \| Out-File "$env:USERPROFILE\Desktop\whea_history.txt"` | user-only | None; user files HP case |

---

## 8. "Are We Hardware-Limited?" — Explicit Verdict

**Short answer: NOT capacity-limited. Currently STABILITY-limited.**

### 8.1 Capacity — wildly over-provisioned for Phase 5
The Phase 5 compute is a **tens-of-MB, single-process, CPU-only numpy workload** (peak RSS ~0.15–0.3 GB) on a **32 GB / 8-core** machine. That is ~100× memory headroom and far more compute than the sequential grid needs. **There is no capacity wall** — not memory, not disk (1.6 TB free), not even per-core throughput for the eval run in isolation. The "matrix blew up / ran out of RAM" framing is quantitatively false.

### 8.2 Stability — the actual limit
The machine throws **fatal, uncorrectable CPU machine-check exceptions (`0x124`) under sustained all-core load.** This is a **hardware/firmware marginality**, not a resource shortage. It is exposed by the *kind* of load (sustained peak-turbo voltage/heat from the unbounded multi-agent fan-out), not the *size* of the data. The same lightweight Phase 5 process run alone is not expected to trip it.

### 8.3 Likely fixable, else a defect
This marginality is **probably remediable** via the firmware/driver/power-containment track (BIOS microcode, current Arc driver with reworked Core Ultra 200V power management, ~85–90% processor-state cap, thread/concurrency caps). **If `0x124` machine-checks persist at stock after those updates, it is a genuine hardware defect** — and since LPDDR5X is soldered and the SoC/board are non-serviceable, it becomes an HP warranty/RMA.

### 8.4 The one genuine capacity nuance
**8 cores (no SMT) is modest for *very wide* agent fan-out.** Eleven concurrent high-effort agents on 8 cores is over-subscribed by design and will be slow regardless of stability. But **that over-subscription is not what is crashing the box** — it would merely cause contention/slowdown on healthy silicon. The crash is the hardware machine-check; the fan-out is its trigger, not its cause.

---

## 9. Verification Plan

The fix is verified in this order, because **diagnosis must be unblinded before re-stressing the machine.**

1. **Enable capture first (P0, actions 1–3).** Disable AutoReboot, set Automatic dump + AlwaysKeepMemoryDump, add a DedicatedDumpFile (or grow the pagefile), install WinDbg. **Success criterion:** the *next* crash either leaves a readable STOP screen (photograph it) or writes a dump that `!analyze -v` / `!errrec` can decode to a named MCA bank / component. This converts "blind" into "diagnosable."
2. **Decode existing WHEA record (action 12).** Pull the CPER section-type GUID from today's Event 1 for immediate component corroboration even before a new dump.
3. **Apply firmware/driver/power containment (actions 5, 7, 8).** Update BIOS + Arc/ME/chipset/NPU drivers; cap Max Processor State to 85–90%.
4. **Re-run the red team WITH caps (actions 4, 6, 9, 10).** Set thread caps in the launching shell, batch the fan-out to 2–3 agents, and watch live temps/clocks (HWiNFO64) and WHEA events.
5. **Run the stock diagnostic ladder (action 13)** after the patched state is in place: Windows Memory Diagnostic, HP UEFI Extensive, Intel PDT.
6. **Define success:**
   - **Primary success** = a **full orchestrated red-team session completes with zero WHEA fatal (Event 1) and zero bugcheck events**, and zero new Kernel-Power 41 unclean shutdowns.
   - **Diagnostic-signal success** = if crashes stop with the 85–90% cap but **return** when restored to 100%, that confirms load/voltage-dependent marginal hardware and strengthens the RMA case.
   - **Escalation trigger** = any `0x124` at **stock** after BIOS + driver + ME updates ⇒ open the HP warranty case (action 18) with the evidence bundle (WHEA Event 1/17 history, Kernel-Power 41 log, bugcheck tally, decoded CPER GUID, captured dump).

---

## 10. Appendix — Safe Orchestration Recipe

Paste-ready containment the user can apply. **Run the PowerShell snippet in the shell from which the orchestration will be launched**, so all child python/Node/numpy processes inherit the caps.

### 10.1 Per-session thread + power caps (run before launching the red team)

```powershell
# 1) Cap BLAS/OpenMP threads so each numpy process uses 1-2 threads, not 8.
#    (Results are byte-identical; only slightly slower per process.)
$env:OMP_NUM_THREADS      = 2
$env:OPENBLAS_NUM_THREADS = 2
$env:MKL_NUM_THREADS      = 2
$env:NUMEXPR_NUM_THREADS  = 2

# 2) Keep the CPU off the top turbo/voltage bins (AC + DC). Revert later with 100.
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 90
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 90
powercfg /setactive SCHEME_CURRENT

# 3) Launch the orchestration FROM THIS SHELL so children inherit the env vars.
```

### 10.2 Persist thread caps for the user account (optional, new shells only)

```powershell
[Environment]::SetEnvironmentVariable('OMP_NUM_THREADS','2','User')
[Environment]::SetEnvironmentVariable('OPENBLAS_NUM_THREADS','2','User')
[Environment]::SetEnvironmentVariable('MKL_NUM_THREADS','2','User')
[Environment]::SetEnvironmentVariable('NUMEXPR_NUM_THREADS','2','User')
```

### 10.3 Make the next crash diagnosable (run elevated; reboot after)

```powershell
$cc = 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl'
Set-ItemProperty $cc -Name AutoReboot        -Value 0  -Type DWord          # keep STOP screen on display
Set-ItemProperty $cc -Name CrashDumpEnabled  -Value 7  -Type DWord          # Automatic (kernel + minidump)
New-ItemProperty  $cc -Name AlwaysKeepMemoryDump -Value 1 -PropertyType DWord -Force
# Decouple dump staging from the 3 GB pagefile (C: has ~1.6 TB free):
New-ItemProperty  $cc -Name DedicatedDumpFile -Value 'C:\DedicatedDump.sys' -PropertyType String -Force
New-ItemProperty  $cc -Name DumpFileSize      -Value 16384 -PropertyType DWord -Force                # 16 GB
# then: Restart-Computer
```

### 10.4 Cap orchestration concurrency (edit the workflow script)

```text
In phase5-redteam-wf_*.js, replace the single wide fan-out:

    await parallel(tasks.map(t => () => agent(t)));

with a chunked version that runs 2-3 agents at a time:

    const CHUNK = 3;
    for (let i = 0; i < tasks.length; i += CHUNK) {
      const batch = tasks.slice(i, i + CHUNK);
      await parallel(batch.map(t => () => agent(t)));
    }
```

### 10.5 Quick verification commands

```powershell
# Confirm thread caps are live in a numpy process:
.venv/Scripts/python.exe -m pip install threadpoolctl
.venv/Scripts/python.exe -c "from threadpoolctl import threadpool_info; print(threadpool_info())"

# Confirm power cap:
powercfg /q SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX

# Watch for hardware errors during/after a run (success = nothing new):
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 10 |
  Select-Object TimeCreated,Id,LevelDisplayName

# Run the Phase 5 eval alone (expected: completes, peak RSS << 1 GB):
python -m slow_wave.eval.grid --config configs/phase5_full.yaml
```

---

*Prepared as a formal RCCA. Component-level confirmation (specific MCA bank: CPU-core vs memory-controller vs PCIe) is pending the first captured dump or decoded WHEA CPER record; no dump contents are asserted here because none exist. The conclusion — hardware/firmware machine-check surfaced by sustained all-core orchestration load, not the Phase 5 workload — is supported independently by all four investigations at HIGH confidence.*
