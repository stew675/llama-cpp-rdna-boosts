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

## 7. Reproduce

```sh
cd archive/work/host-expert-transport
PINNED=1 ./abba3env.sh 6144 "kern " "ce MOE_HOST_POOL_STAGE_BATCH=0" "nostage GGML_MOE_CACHE_STAGE=0"
./abba2env.sh 5120 "g5 " "c5 GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0"
PINNED=1 ./abba2.sh 2 6144 base fix          # two prebuilt trees: build-rocm/bin-<tag>
```

`runs/` keeps every run's log, `mem.txt`/`mem_after.txt`, `vm_before/after.txt` and `clk.txt`
(`rocm-smi --showclocks` sampled every 2 s).
