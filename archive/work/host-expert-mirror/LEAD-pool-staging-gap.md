# LEAD: the mirror's prefill transport — the single-launch staging kernel is the worst option, and on one GPU the copy-engine path beats the gather by 13 %

**Status: OPEN lead, measured (2026-11-05, r221 tree).**  Nothing in `patches/`, nothing committed to the fork.
Follow-up to [`spike-261105/SPIKE.md`](spike-261105/SPIKE.md) §4 ("a new lead").  The campaign's original target
(hand the op the pool alias, `MOE_HOST_POOL_ALIAS`) is now **the wrong lever**: the recoverable loss is not the
bytes and not the in-place read, it is **which stream carries the transfer**.

Every number below, same config: 1 x R9700 unless stated, ROCm 10.1.0, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`,
`-lme 47` + `MOE_HOST_POOL_RAM_PCT=0` (plan line: "the whole host expert set (a MIRROR)"), `-c 204800 -ctk/-ctv
q8_0 -fa on -t 7 --cache-ram 1024 --fit off -ngl 99 --n-cpu-moe 48 --load-mode mmap --lazy-mode auto`, prompt
`spike-261105/prompt_52k.txt` (52 512 tokens), `-n 1`, no cgroup (MemAvailable ~80 GiB, recorded per run).
Transport arms are the three that exist today: **gather** (the default), **staging kernel**
(`GGML_SCHED_DEVGATHER=0`, the r114 single-launch `moe_cache_stage_kernel`), **staging CE**
(`GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0`, one `cudaMemcpyAsync` per expert on the copy stream).

## 1. The three transports, interleaved ABAB (`spike-261105/abba.sh`)

| 1 GPU, `-ub 6144` | c1 | c2 | c3 | verdict |
|---|---|---|---|---|
| gather (default) | 1626.6 | 1629.2 | 1616-1625 (earlier) | stable |
| staging kernel | 1425.3 | 1411.8 | | **-12 % vs gather** |
| **staging CE** | **1840.5** | **1844.7** | | **+13 % vs gather** |

| 1 GPU, `-ub 2048` | c1 | c2 |
|---|---|---|
| gather | 1016.4 | 1020.6 |
| staging kernel | 875.0 | 871.5 |
| staging CE | 741.7 | 907.4 (noisy) |

### The `-ub` band (1 GPU, one run each, interleaved where repeated)

| width | gather | staging kernel | staging CE | winner |
|---|---|---|---|---|
| 2048 | 1016.4 / 1020.6 | 875.0 / 871.5 | 741.7 / 907.4 | gather (CE noisy and last) |
| 5120 | 1510.6 | - | **1693.8** | CE +12 % |
| 6144 | 1626.6 / 1629.2 | 1425.3 / 1411.8 | **1840.5 / 1844.7** | CE +13 % |
| 7168 | 1684.6 | - | **1889.2** | CE +12 % |
| 8192 | **1747.8** | 1546.1 | 1695.6 | gather +3 % |
| 16384 | **1945.1** | 1736.5 | 1788.4 | gather +9 % |

So the CE win is a **width band (5120-7168)**, not a trend, and it is gone by 8192.  The 8192 trace pair says
why: `t8k-gather` wall 30.20 s / total kernel 28.86 s (0.96x, the 9.20 s gather serialised), `t8k-ce` wall
31.42 s / total kernel **25.01 s (0.80x)** — the CE arm does *less* GPU work but 20 % of its wall is spent with
no kernel running, i.e. the staging pipeline stalls (the ring/graph-input hand-off is waiting) instead of
overlapping.  At 6144 it was 0.93x with a shorter wall.  The stall's exact trigger (ring depth
`GGML_SCHED_STAGE_SLOTS` vs the split count, or the graph-input hand-off) is **not** identified, and it is the
next probe if this lead is ever promoted.

| 3 GPU, `-sm tensor`, `-ub 6144` (per-device ubatch 2048) | t/s |
|---|---|
| **gather** (default) | **2358.6** |
| staging kernel | 2074.5 |
| staging CE | 2053.4 |

So: **CE > gather > kernel on one GPU at a large ubatch; gather > both on a tensor split and at a small
ubatch; the single-launch kernel is last everywhere.**  No #131 collapse reproduced on the 3-device mirror
(`ok=3807` staging calls, all three arms in the 2.0-2.4 k t/s range) — that collapse is the 122B/67 GiB shape.

**The `-ub 6144` row is the campaign's own consumer shape** (`bench-56gb/runme-56gb-mirror.sh` ships
`--ubatch-size 6144`, one GPU), i.e. the +13 % is available on the exact config the 1802-vs-1949 gap was
measured on, with two documented environment variables and no code change.

## 2. Why (`rocprofv3 --kernel-trace`, three transcripts on the 52k prompt)

| arm | wall | total GPU kernel | kernel/wall | transport's own cost |
|---|---|---|---|---|
| gather | 32.74 s | 31.22 s | 0.95 — **serialised** | `moe_cache_gather_kernel` **10.91 s** on the compute stream |
| staging kernel | 37.94 s | **43.55 s** | 1.15 — overlapped | `moe_cache_stage_kernel` 11.22 s + `copyBuffer` 2.56 s, **and the compute kernels are starved** |
| staging CE | **29.75 s** | 27.57 s | 0.93 — overlapped | `copyBuffer` 5.92 s under 21.65 s of compute |

`spike-261105/kernel-diff.py` on the three CSVs, same-count kernels only:

| kernel (count) | gather | staging kernel | staging CE |
|---|---|---|---|
| `mm_ids_helper_512_10` (1410) | 0.051 s | **2.081 s (41x)** | 0.055 s |
| `quantize_mmq_q8_1<…, true>` (940) | 0.336 s | **1.657 s (4.9x)** | 0.401 s |
| `mul_mat_q_routed_compact<20>` (342) | 1.291 s | **4.438 s (3.4x)** | 1.476 s |
| `mul_mat_q_routed_compact<42>` (81) | 0.304 s | **1.100 s (3.6x)** | 0.369 s |
| `hc_combine_norm_f32` (940) | 1.559 s | 1.851 s | 1.563 s |
| **TOTAL** | 31.22 s | 43.55 s | 27.57 s |

Three mechanisms, all measured:

1. **The gather is serialised**: it is a node in the compute graph, so its 10.9 s is added to the prefill
   (kernel/wall 0.95, no cross-stream overlap).  Its *rate* is already at the link's ceiling (33 GB/s of pool
   reads, `SPIKE.md` §2) — the bytes are not the problem, the stream is.
2. **The single-launch staging kernel monopolises the CUs**: 144 blocks x 256 threads streaming for ~7-9 ms per
   table while short kernels wait for a free CU on the other stream.  The same work, same stream, same bytes,
   takes **3-41x longer** in exactly the arm where that kernel is resident.  This is the regression r114
   introduced: it replaced per-expert copies for +34 % prefill *when the alternative was a pageable master that
   paid a host memcpy into a pinned bounce per copy*.  With the pool alias readable in place (r215/r219) there
   is no bounce to amortise, so the kernel buys nothing and costs CU time.  (`INVARIANTS.md` r215 item 7's rule,
   one knob over: re-measure, do not assume.)
3. **Many short copies interleave**: ROCclr's per-copy blit (`__amd_rocclr_copyBuffer`, 182 873 dispatches of
   ~32 us) runs *under* compute without monopolising a CU group, so the copy hides inside the compute time
   instead of inflating it.

## 3. The sub-experiments that close the alternative explanations

| question | answer |
|---|---|
| Is it the pool's alloc shape? (`MOE_HOST_POOL_ALLOC=thp` vs `portable`) | **No.** CE: thp 1840.5/1844.7, portable 1833.6/1843.6. Kernel: thp 1425/1412, portable 1433.7/1421.9. The alloc shape does not move either arm (and the gather measured identical on both in `SPIKE.md` §2). |
| Is it the copy *kind*? (D2D from the mapped alias vs H2D from the host pointer, `MOE_HOST_POOL_H2D=1`) | **No.** thp+CE+H2D 1842.4, portable+CE+H2D 1829.0 — the same as D2D. |
| AnonHugePages present? | 0 GiB after every run in both alloc shapes (the r215 THP accounting is unchanged); the win is not a THP artefact. |
| Is the pinned tier the same? (`--host-experts pinned`, a real 47 962.5 MiB `ROCm_Host` master, 0 MiB pageable) | **Opposite**: staging kernel **1820.1 / 1825.1** > staging CE **1544.9 / 1547.5** (interleaved).  Confounded — 54 of 144 tables decline staging on that tier (`ok=783 unusable=477`) and are served by per-tensor copies on the compute stream; see §5.2.  Its best number (1825) equals the pool's CE arm (1842) **without** a 46.84 GiB duplicate, which is the campaign's whole goal. |

Protocol note (it cost two rounds): a **block-ordered** set of "reps" produced a monotonic 16 % drift in this
box state (kernel 1241 -> 1434, CE 1549 -> 1844 as the session went on) while the gather stayed at 1616-1629
throughout.  Only the interleaved ABAB sequence is trustworthy here (`abba.sh`); the drift's cause is not
identified (page-cache warmth of the 83.6 GB model is the leading candidate) and it is why `SPIKE.md`'s
first-pass staging numbers are superseded by §1 above.

## 4. What this means for the campaign

* **The alias (`MOE_HOST_POOL_ALIAS`) is not the lever.**  It moves the same bytes over the same link at the
  same rate and puts a host pointer into the prefill kernels' fusion guards for a prize (overlap) that the
  copy-engine path already delivers, at no risk, with two existing kill switches.  Recommend dropping step 3 as
  scoped and recording that here.
* **The remaining gap is bounded and now attributable**: at `-ub 6144` on one GPU the transport takes 10.9 s of
  a 32.7 s prefill because it is serialised; the best measured transport is 29.75 s.  What is left after that is
  compute, not transfer.
* **The single-launch staging kernel is the concrete, unambiguous defect** and it is not width- or
  device-dependent: it is last at every width measured (2048, 6144, 8192, 16384) and on the tensor split, and
  the trace says why (CU monopolisation, compute kernels 3-41x slower at identical dispatch counts).  Whatever
  is decided about the gather, this arm should not be the fallback anybody lands on.
* **A default change is NOT justified by this record.**  The CE win is a 1-GPU, 5120-7168-token band, and it
  reverses on a tensor split (+14 % for the gather) and above ~8k.  A width- and device-count-shaped rule in
  `sched_input_gatherable` / the staging batch default is the only shape a promotion could take, and it needs
  the crossover's cause first (§5.3).
* **A pure-documentation outcome is legitimate and is what I recommend for this session**:
  `GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0` is two documented kill switches, so a 1-GPU consumer at
  `-ub` 5120-7168 can have +12-13 % today; `ENVIRONMENT.md` describes neither arm's effect on prefill.  That is
  an `optional/`-style tuning note at most, not a block.

## 5. Open, and what would have to be proved before any promotion

1. **Purity.**  Same bytes, so a transport change.  Measured so far: **same-seed greedy text is byte-identical
   across all three transports** (`runs/txt-*/text.clean`, sha256 `68704e68…`, 135 chars).  Still owed before a
   delivery change: `scripts/gate-prefill-logits.sh --ab "GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0"`
   (0.000000 / 100.000 % is the requirement — note the gate's default model/config may not route host experts at
   all, in which case the A/B is vacuous and must be run in the mirror config), `scripts/gate-host-pool.sh`,
   `test-backend-ops -o MUL_MAT_ID`, and the `-sm layer` arm.
2. **The pinned tier's 54 declined tables.**  `ok=783 unusable=477`: 54 of 144 tables never enter the staging
   kernel on a pinned master and are served by per-tensor copies on the compute stream.  The decline predicate
   is `t.host_pinned && (arena==null || slots<=0 || host_dev==null || !cold_safe)` and nothing logs which tables
   or why — a WARN-level decision line that does not exist, on a tier that is otherwise the fastest measured.
   `TODO`-candidate; needs an instrument before it can be filed.
3. **The 8192 crossover's cause**, then a cross-shape sweep: 1/2/3 GPU x {`tensor`, `layer`} x {mirror,
   sub-complement pool} x `-ub` {2048, 5120, 6144, 7168, 8192, 16384}.  The width band alone rules out a
   blanket default; the 0.80x kernel/wall at 8192 says the CE arm's overlap is *stalling*, and a stall that
   depends on ring depth is a fixable thing, but only once it is named.
4. **`MOE_HOST_POOL_ALLOC=portable` is not needed for this** — keep the r215 THP shape (its fill/pin win,
   `INVARIANTS.md` r215 items 3-5, is unrelated).

## 6. Reproduce

```sh
cd wip/host-expert-mirror/spike-261105
./abba.sh 2 6144                 # gather / staging kernel / staging CE, interleaved (the only trustworthy form)
./abba.sh 2 2048
DEV=0,1,2 SM=tensor ./arm.sh t3-gather 6144      # the tensor-split control
./trace-arm.sh trace-pool-staging-nobatch-52k 6144 GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0
./kernel-diff.py runs/trace-ub6144-52k-instr runs/trace-pool-staging-52k runs/trace-pool-staging-nobatch-52k
./text-arm.sh txt-ce GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0
```
