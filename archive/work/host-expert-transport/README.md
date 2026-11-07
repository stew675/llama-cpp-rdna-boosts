# host-expert-transport (archived 2026-11-07) — the prefill expert transport, its two tiers and its engines

**CLOSED with r234.**  This directory is the record of `TODO` **#271** (resolved: the prefill staging
decline is now named, and was measured not to be a cost) and `TODO` **#272** (closed as documented: the
transport band was re-measured, the 8192 "stall" was refuted, and no default was changed).  The only
delivery change is one decision line and one census split in `moe-expert-cache.cu` (block 12);
`RESOLUTION.md` has the conclusions, this file the configuration and the raw numbers.

## 1. What was asked (the two TODO items, verbatim intent)

* **#271.**  On a pinned host-expert master, **54 of 144** tables declined the prefill staging path on
  every ubatch (`prefill cache staging: ok=783 unusable=477`) and were served by the scheduler's per-tensor
  copies instead — a silent prefill cost, with no log line naming the tables.  The work was "an INFO/WARN
  line naming table/layer/reason once per run, then decide per reason".
* **#272.**  The prefill expert-transport engine/stream choice is width- and device-count shaped and the
  shipped switch combination is wrong on one shape.  The work was "name the 8192 stall, then the sweep";
  a config-only / documentation outcome was explicitly legitimate, and was what
  [`../host-expert-mirror/RESOLUTION.md`](../host-expert-mirror/RESOLUTION.md) recommended.

## 2. Config (every run below)

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (shard 1, 83.6 GB), 1 × R9700 (gpu0, **PCIe Gen5 x8** — see
`rocm-smi --showclocks`), ROCm 10.1.0, gfx1201, `--fit off -ngl 99 --n-cpu-moe 48`,
`-c 204800 -fa on -ctk/-ctv q8_0 -sm layer --load-mode mmap --lazy-mode auto -t 7 --cache-ram 1024`,
prompt `prompt_52k.txt` (52 512 tokens), `-n 1`, greedy, `MOE_HOST_POOL_RAM_PCT=0`,
`LLAMA_KV_REGION_POOL=0`, `GGML_CUDA_SLAB_HEADROOM_MIB=1536`, `-ub` as noted.  Two host tiers:

* **pinned** — `--host-experts pinned`: the whole 47 962.5 MiB host set in `ROCm_Host`, 0 MiB pageable.
* **mirror pool** — `--limit-host-experts 47`: the whole host set as a pinned per-expert **pool**.

## 3. Protocol: ABBA, and why (this cost the session two wrong conclusions)

Two effects on this box are larger than the differences under test, and both are **position-in-sequence**,
not configuration:

1. A block-ordered set of reps drifts monotonically (up to 16 %); only interleaved reps are trustworthy.
   That is `../host-expert-mirror/RESOLUTION.md` §4, and it reproduced.
2. **Identical consecutive runs alternate between two levels with period 2** — measured on one unchanged
   arm, four runs in a row: **1873 / 2177 / 1865 / 2180 t/s**.  Amplitude by arm: pinned bulk 16 %, pool
   per-expert copies 17-32 %, pool gather **0 %** (1791 ± 2 over four runs).  The cause is **not**
   identified: not paging (`pswpin` +≤200 pages per run, `pgmajfault` ~2 400 per run), not
   `MemAvailable` (flat), not the arena (`72 796.2 MiB`, `31 968` slots and the plan line identical in
   every run), and `AnonHugePages` is 0 GiB after every teardown in both alloc shapes.  So: run **each**
   arm in **both** orders (`A B B A`), report the **spread**, and never quote a single run.

A plain `A B A B` reads as "B is 6 % faster" whatever B is (measured: base 1757 at position 1 against
base 1863 at position 2).  The harnesses are `abba2.sh` (two binaries), `abba2env.sh` / `abba3env.sh`
(two/three env arms) and `sweep.sh` (a block-ordered arm — do not quote it).

## 4. #271 — which tables decline, and why

Runs: `runs/diag-pinned-6144` (instrumented predicate), `runs/diag2-pinned-6144` (per-table census),
`runs/diag3-pinned`, `runs/fix1-pinned-6144`, `runs/fix2-pinned-6144`.

* **Which**: 50 of 144 tables on a pinned master, always the same ones, declining on **every** ubatch of
  the prefill (9 of 9 at `-ub 6144`); 141 tables appear in the prefill graph per ubatch, 3 never do.
  Layers 0, 27, 28 (gate+up), 29, 31, 32, 33, 35, 36 (gate+up), 37, 39, 40, 41, 43, 44, 45, 46 — the
  whole-layer blocks are the tables whose arenas the slab reclaimed first.
* **Why**: `arena == nullptr && slots == 0` for all 50 — the other three predicates in
  `t.host_pinned && (arena || slots || host_dev || cold_safe)` never fire on this tier (`host_dev` is
  non-null on any table that reports `host_pinned`, `cold_safe` is true for every unfused table).  The
  sequence in the log is
  `moe_cache_evict_slab_range: evicted 7536.0 MiB … (arena now 13308.6 MiB)` (then another 150 MiB) —
  the prefill's compute layout (a 2 560 MiB compute buffer plus 32 context checkpoints from
  `--cache-ram`) took the slab chunks those arenas lived in — after which the tables are **stood down
  for the rest of the prefill**.  They are re-armed by `moe_cache_rearm()` at the compute-buffer drop
  (exit census: 144 tables, 31 968 slots = 222 each), i.e. **the state is prefill-only and legal**.
* **What it costs**: nothing measurable.  ABBA, `runs/abba-c*`: the 50 declined (default) **1826 t/s**
  against the same 50 staged with one bulk copy each **1824 t/s** (±0.1 %).  The scheduler's own
  whole-table upload is a single bulk async copy on the **same copy stream**, so it was never "per-tensor
  copies serialised on the compute stream" (that reading came from a trace of a *different* arm).
* **What it costs if you stage it the obvious way**: staging the 50 through `moe_cache_stage_kernel`
  (the pinned tier's normal engine, one 512-block launch per table) measures **1572 / 1595 t/s** against
  **1757 / 1765** for the decline — i.e. **-10 %**.  The single-launch kernel is a *worse* engine than one
  bulk DMA for a whole-table copy, which is the same CU-monopolisation mechanism
  `../host-expert-mirror/LEAD-pool-staging-gap.md` §2 traced on the pool tier.

**So the r234 change is visibility only** (`stage_decline_note` + one WARN per run + the `unsized=` split
in the census).  Both tempting "fixes" — relax the arena requirement and stage via the kernel, or stage
via the copy engine — were built and measured, and are **worse or neutral**; neither shipped.

## 5. #272 — the band, re-measured (1 GPU, mirror pool, ABBA × 2 cycles)

`gather` = the default (`GGML_SCHED_DEVGATHER` unset); `CE` = `GGML_SCHED_DEVGATHER=0
MOE_HOST_POOL_STAGE_BATCH=0` (per-expert copy-engine staging).  Runs: `runs/x-*`.

| `-ub` | gather (4 runs) | CE (4 runs) | verdict |
|---|---|---|---|
| 2048 | 1066 / 1065 / 1066 / 1066 | 786 / 834 / 970 / 973 | **gather +20 %** |
| 5120 | 1564 / 1565 / 1562 / 1564 | 1837 / 1849 / 1842 / 1843 | **CE +18 %** (stable) |
| 6144 | 1674 / 1671 / 1671 / 1675 | 1721 / 2026 / 2027 / 2028 | **CE +17 %** (bimodal) |
| 8192 | 1791 / 1793 / 1790 / 1788 | 2133 / 1838 / 2045 / 1829 | **CE +10 %** (bimodal) |
| 16384 | 1987 / 1985 / 1988 / 1987 | 2029 / 1959 / 2023 / 1996 | tie (±1 %) |

* The **gather is the stable transport** (±0.1 % in every cell); every copy-engine arm is bimodal.
* **The archived "the gather wins at 8192" is refuted**: CE was ahead of the gather in **4 of 4** runs at
  8192 and the upper edge of the band is at ≥16 384 (a tie).  The lower edge is between 2048 and 5120 and
  was not measured.
* **The 8192 "stall" is not a stall.**  The suspect was ring depth `GGML_SCHED_STAGE_SLOTS` (8) against the
  split count: `=16` (the compile-time maximum) changes nothing (`runs/x-c*`: s8 2058/2095/1606/2141
  against s16 1679/2086/2134/1838 — the spread is the bimodality, not the depth).  What the archived trace
  saw (kernel/wall 0.80) is the low phase of the two-level behaviour above.

## 6. #272 — the pinned tier's engines (the tier `-lme auto` selects by default)

1 GPU, `-ub 6144`, ABBA (runs `runs/y-*`, `runs/p*`):

| arm | runs | phase-balanced |
|---|---|---|
| default: staging kernel for the 91 sized tables, decline for the 50 stood down | 1764.5 / 1764.9 / 1761.0 (low) · 1879.8 (high) | **1 793** |
| `GGML_MOE_CACHE_STAGE=0` (no cache staging: one bulk H2D per table) | 1862.5 / 1873.1 / 1865.0 (low) · 2183.3 / 2177.3 / 2180.0 (high) | **2 023 (+12.8 %)** |
| `GGML_SCHED_DEVGATHER=1` (force the gather on a pinned master) | 1490.5 / 1494.9 | **1 493 (-16 %)** |
| per-expert copy-engine staging for the pinned tier (prototype, not shipped) | 1620.5 (low) / 2142.4 (high) | not established |

So on a pinned master the r114 single-launch staging kernel — the shipped default — is the **slowest**
engine reachable today, and "do not stage at all" (one bulk DMA per table, at the cost of sending the
arena-resident bytes over PCIe too) is **+12.8 %**.  This is a real finding, but it is **one shape on one
box** (1 GPU, `-ub 6144`, a Gen5 **x8** link, a 47 962 MiB host tier) and the copy-engine arms are
bimodal, so it was recorded, not defaulted (see `RESOLUTION.md` §3).

## 6b. Addendum (2026-11-07, after r234 shipped): where the staging copy actually pays

`GGML_MOE_CACHE_STAGE=0` is the kill switch for the whole feature, so "does it pay" decides the default.
Four runs per arm, balanced ABBA, same config as §2:

| tier / shape | staging (default) | `GGML_MOE_CACHE_STAGE=0` | verdict |
|---|---|---|---|
| **pageable master** (pool, `-sm layer`, `GGML_SCHED_DEVGATHER=0`) | 1052-1556 t/s | **103-281** | staging is **5-10x** faster: without it every byte goes through `moe_cache_upload_pageable` (a host `memcpy` into a pinned bounce + a pinned H2D per expert) instead of the arena's D2D |
| pinned, **2 devices `-sm layer`, 99.1 % residency** (arena 47 534.9 MiB of 47 962.5 MiB) | **1961** (1933-1982, stable) | 1885 (1811-1961) | staging **+4.0 %**; with the arena covering the set it moves `arena=322 875 MiB host=0.0 MiB` (H2D 0.0 %), so nothing crosses PCIe |
| pinned, **1 device, 43.4 % residency** (arena 20 796.2 MiB) | 1793 | **2023** | staging **-12.8 %**: the one-launch fill kernel is the slow engine and the arena saves too few bytes |

A `-ncmoe 24` arm cannot reach high residency on one 32 GiB card: the resident layers plus the auto arena
over-commit and the context fails to initialise (`cudaMalloc failed: out of memory` on the pp compute
buffer) — use two devices for that shape, as above.

`GGML_MOE_CACHE_STAGE` is read only in `ggml/src/ggml-cuda/moe-expert-cache.cu` (the CUDA/HIP backend is
the only one that fills the MoE iface hooks).  A **CPU-only** run never loads that backend, so the switch
cannot change its throughput.

## 6c. Addendum 2: the engine matrix, and the box's copy-engine phase (2026-11-07, evening)

Measured while evaluating "one engine instead of two" (which would delete `moe_cache_stage_kernel`).  Same
config as §2, 1 GPU unless stated, balanced ABBA, 4 runs per arm, `[ Prompt: ... ]` footer as the figure.

### The engine ranking reverses with a run-to-run PHASE, it does not have a winner

Pinned master, `-ub 6144`, 43.4 % residency — the cleanest dataset, because each arm appears in *both* phases:

| position in the cycle | 1 (low phase) | 2 (high phase) | 3 (low) | 4 (high) |
|---|---|---|---|---|
| kernel (`moe_cache_stage_kernel`, the shipped default) | 1760.7 | — | — | 1876.1 |
| per-expert copy-engine fill | — | 2146.4 | 1623.5 | — |

Phase-matched: **low → kernel 1761 vs CE 1624 (kernel +8.4 %); high → kernel 1876 vs CE 2147 (CE +14.4 %).**
Both arms swing with the same period; the amplitude is a property of the transport:

| arm | DMA ops per table | low / high | amplitude |
|---|---|---|---|
| device gather (a KERNEL read, no DMA) | 0 | 1791 / 1793 / 1790 / 1788 | **±0.1 %** |
| staging kernel (1 launch per table) | 1 launch | 1761 / 1876 | **6.5 %** |
| whole-table bulk upload (`GGML_MOE_CACHE_STAGE=0`) | 1 DMA per table | 1866 / 2180 | **17 %** |
| per-expert copy engine | 512 DMA per table | 1624 / 2147 | **32 %** |

So the *cost* of the phase is proportional to the number of DMA operations the transport issues, and a
kernel-read transport is immune.  The whole-table bulk arm is the only one that beat the shipped kernel in
**both** phases (+6 % low, +16 % high), at the price of sending the arena-resident bytes over PCIe too.

### What the phase is, and what it is not (each ruled out by measurement)

* **Not PLE page cache / lazy mode.**  `--lazy-mode auto` (default), `off` (the 27 465 MiB PLE table read
  into host RAM) and `mlock` (populated and pinned) all oscillate with the same ~1.30 ratio
  (`off`: 1838/1456/1836/1462; `mlock`: 1832/1460/1834/1465).  On THIS box `auto` is also the fastest of the
  three — with a 47 962.5 MiB host-expert tier there is no room for a 26.8 GiB PLE pin (`MemAvailable` falls
  to 7.7 GiB and the model's non-PLE tensors lose the page cache), so `mlock`/`off` cost ~15 % here even
  though they are the right answer when the PLE table is the only large host consumer.
* **Not memory pressure**: `MemAvailable` minimum is identical within a lazy-mode series (31.2-31.5 GiB on
  `auto`), and the amplitude does not grow when it is pushed to 7.7 GiB.
* **Not the GPU clock**: `rocm-smi` sclk max is ~3220 MHz in both phases.
* **Not the work**: arena 20 796.2 MiB (43.4 %), staged bytes `arena=113 363.1 MiB host=148 086.9 MiB
  (H2D 56.6 %)`, evictions (7 536.0 + 150.0 MiB), `hits=768 fills=2202` — identical in both phases, as are the
  kernel dispatch counts (21 055 `copyBuffer`) in the traces.
* **Not the ring depth**: `GGML_SCHED_STAGE_SLOTS=16` (the compile-time maximum) vs 8 → +1 % (noise).
* **Not the GPU slot**: the same pattern on gpu0 (Gen5 x8) and gpu2 (1630/2163/1642/2164).
* **It IS the SDMA copy-engine path.**  A `rocprofv3` kernel trace of a low-phase vs a high-phase run shows
  the low phase at **67 % GPU-busy** against 85 %, with the copy engine *less* busy (0.16 s vs 0.36 s) in a
  longer wall — i.e. the copies stall instead of transferring, and the GPU waits.  The extra time is many
  small GPU-idle gaps (< 5 ms each; the > 5 ms gap structure is identical in both phases).  And with
  `HSA_ENABLE_SDMA=0` (copies issued as blit kernels instead of SDMA) the oscillation **disappears**:
  1635 / 1557 / 1541 / 1566 t/s — stable, but pinned at the *low* level.

**Conclusion: a ROCm/SDMA dispatch-latency effect on this box, not a property of the fill code.**  A consumer
cannot be asked to set `HSA_ENABLE_SDMA=0` (it costs the fast phase), so the delivery keeps the *low-DMA-count*
engine (the one-launch kernel on the pinned tier, the device gather on the pool tier) and the per-expert copy
engine stays an A/B arm (`MOE_HOST_POOL_STAGE_BATCH=0`).  The engine merge that prompted this
(copy-engine for both tiers, kernel deleted) was **built, measured and not shipped**: it is better in the high
phase, worse in the low one, and its mean is not separable from the phase.

### The engine matrix (all cells, for the record)

Pinned tier, kernel vs per-expert copy engine, 4 runs per arm:

| cell | kernel | CE | verdict |
|---|---|---|---|
| 1 GPU, `-ub 6144`, 43.4 % res | 1887/1860/1745/1871 | 2138/2128/2139 (one 1613) | CE ahead in the high phase, behind in the low one (see above) |
| 1 GPU, `-ub 2048`, 43.4 % | 1051/1083/1084/1057 | 1115/1114/1116 (one 833) | CE +4.3 % on the consistent runs |
| 2 dev `-sm layer`, 99.1 % res | 1953/1985/1928/1973 | 2104/1952/2086/1937 | CE +3.1 %, and the kernel is the stable arm |
| 2 dev `-sm tensor` | 2406/2369/2350/2374 | 2387/2378/2363/2361 | tie (±1 %) |

Text is byte-identical between the engines on every shape (`d3d554954cba` 1 GPU / `301b143c1cf5` 2-device
tensor), and the prefill-logit A/B (staged vs `GGML_MOE_CACHE_STAGE=0`, Flash-Next `-ncmoe 44`) is
**KLD 0.000000 / same-top-p 100.000 %** — so nothing here is a purity question, only a transport one.

### The pool tier, re-measured with the same protocol (palindrome, not ABAB)

| `-ub` | gather | CE staging | gather's lead |
|---|---|---|---|
| 4096 | 1260.6 / 1261.4 | 1121.5 / 1122.9 | +12.3 % |
| 4608 | 1322.4 / 1321.7 | 1225.1 / 1216.6 | +8.3 % |
| 5120 | 1389.5 / 1384.2 | 1331.3 / 1302.8 | +5.3 % |

The r234 "CE wins from 5120 up (+18 %)" **does not reproduce** in this session state (the CE arm moved ~29 %,
the gather ~11 %).  So there is no width threshold to encode: the gather is at least as good at every width
measured, and much better at 2048/3072.  The pool tier keeps the gather as its default — no rule needed.

## 7. Reproduce

```sh
cd archive/work/host-expert-transport
PINNED=1 ./abba3env.sh 6144 "kern " "ce MOE_HOST_POOL_STAGE_BATCH=0" "nostage GGML_MOE_CACHE_STAGE=0"
./abba2env.sh 5120 "g5 " "c5 GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0"
PINNED=1 ./abba2.sh 2 6144 base fix          # two prebuilt trees: build-rocm/bin-<tag>
```

`runs/` keeps every run's log, `mem.txt`/`mem_after.txt`, `vm_before/after.txt` and `clk.txt`
(`rocm-smi --showclocks` sampled every 2 s).
