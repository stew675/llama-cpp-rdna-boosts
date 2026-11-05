# Step 1 spike — measured 2026-11-05 (r221 tree, 1 x R9700, no cgroup)

**Read this before writing any code.**  It answers `../README.md` §3 step 1 (and step 2's first half) with
measurements, and it names the mechanism correctly: on a mirror the prefill transport is the **device gather**,
not the staging engine.

## Config and caveats

`runme-56gb-mirror.sh` translated to `llama-cli`: 1 x R9700 (gfx1201), `-lme 47` (a 46 962.5 MiB host expert
set), `MOE_HOST_POOL_RAM_PCT=0` (no ceiling, so the tier cannot be shaved), `LLAMA_KV_REGION_POOL=0`,
`GGML_CUDA_SLAB_HEADROOM_MIB=1536`, `--load-mode mmap --lazy-mode auto -c 204800 -ctk/-ctv q8_0 -fa on
-t 7 --cache-ram 1024`, `--fit off -ngl 99 --n-cpu-moe 48`, prompt `prompt_52k.txt` (10 x
`prompts/prose-rdna-boosts.txt`, 160 740 B, sha256 `5e4849fe…`, 52 512 tokens), `-n 1`.

Every run's plan line: `-lme 47 -> a pinned per-EXPERT pool of 48128.0 MiB of 47962.5 MiB: the whole host
expert set (a MIRROR)`.  **No cgroup/ballast**: this box has no passwordless sudo for `systemd-run`, so
MemAvailable is ~80 GiB rather than the rig's 57 GiB (recorded per run in `runs/*/mem_before.txt`). The tier is
still a mirror; the absolute t/s therefore sits ~15-20 % below `wip/bench-56gb/RESULTS.md` row D (that row was
also measured on the block-15 tip `790686b53`, not on r221). **What this record is for is a SHARE and a rate,
not an absolute t/s.**

## 1. Which copy runs: the gather, and only the gather

| arm | -ub | prefill t/s | gather calls | staging census |
|---|---|---|---|---|
| mirror, default | 6144 | **1618.8 / 1624.9** | 1410 (1410 pool-backed), 0 neither-tier | **absent** (all counters 0) |
| mirror, default | 2048 | **1018.4 / 1020.1** | 3807 (3807 pool-backed), 0 neither-tier | absent |
| `GGML_MOE_CACHE_STAGE=0` | 6144 | **1616.2** | 1410 | absent |
| `GGML_SCHED_DEVGATHER=0` (staging) | 6144 | **1241.5 / 1352.3** | **0** | ok=1269, arena 115.9 GiB + host 304.1 GiB, H2D 72.4 % |
| `GGML_SCHED_DEVGATHER=0` (staging) | 2048 | **845.1** | 0 | ok=3666, arena 393.9 GiB + host 819.3 GiB, H2D 67.5 % |

So: `GGML_MOE_CACHE_STAGE=0` is a **no-op** (-0.2 %, noise) and `GGML_SCHED_DEVGATHER=0` replaces the gather
with whole-table staging.  **The staging rows in this table are block-ordered and were later found to be
drift-confounded** (16 % monotonic drift over the session; see [`../LEAD-pool-staging-gap.md`](../LEAD-pool-staging-gap.md)
§3 "protocol note"): the interleaved ABAB values are gather 1627, staging kernel 1418, staging CE 1842.  The
*naming* conclusion of this section (the copy is the gather) is unaffected -- those rows are the switch, not a
ranking. The copy at `-ub 6144` is `moe_cache_gather_kernel`, as suspected; the campaign's
"46.84 GiB per ubatch through `moe_cache_upload_pageable`" was a misattribution (the *bytes* were right, the
mechanism was not — and so is the conclusion drawn from it, see §4).

## 2. The share of prefill wall time, and the rate (`rocprofv3 --kernel-trace`)

Traced, so quote durations only, never a t/s from a trace (the archive's own rule).

| prompt / ub | wall (prompt eval) | total GPU kernel | `moe_cache_gather_kernel` | share |
|---|---|---|---|---|
| 21 036 tok / 2048 | 21.25 s | 20.17 s | **11.09 s** (1692 calls, 6.56 ms each) | **55.0 %** |
| 21 036 tok / 6144 | 13.88 s | 13.03 s | 4.93 s (705 calls, 6.99 ms) | **37.8 %** |
| 52 512 tok / 6144 | 32.74 s | 31.22 s | **10.91 s** (1410 calls, 7.52 ms each) | **34.9 %** |

GPU-busy is 94-95 % of the wall, so the copy is **serialised with the compute** (it is a node on the compute
stream), not overlapped: sum-of-kernels ≈ wall. That is the opposite of the "if the copy is overlapped and its
wall-time share is small, close it" condition in step 1.

**Exact bytes** (a temporary census counter, since removed: "gather bytes offered", every expert in the map
counted once per call, an upper bound that is tight at these widths — 6144 x 10 draws over 512 experts routes
every expert):

| prompt / ub | pool (host, over PCIe) | arena (VRAM, D2D) | pool bytes per call | pool read rate |
|---|---|---|---|---|
| 52 512 tok / 6144 | 329.19 GiB | 126.50 GiB | 250.7 MB | **32.6-33.3 GB/s** |
| 52 512 tok / 2048 | — (same rate; 3x the ubatches) | | | |

33 GB/s against this stack's **28.6 GB/s** bulk H2D calibration (`P4-ROOT-CAUSE.md` §5d-ter): the gather's
kernel reads of the pinned pool alias **already outrun the copy engine**. There is no latency/issue tax left to
reclaim inside the copy.

## 3. The two cheap "trim the transfer" levers, measured

* **`-ub` is a bytes-per-token lever.**  Every mirror ubatch routes essentially the whole table, so the copy is
  ~46.8 GiB *per ubatch* whatever the width, i.e. bytes/token ∝ 1/ub while the ubatch count ∝ ub:

  | -ub | prefill t/s | pool GiB | arena GiB | arena MiB (sized) |
  |---|---|---|---|---|
  | 2048 | 1018-1020 | | | 21 545.7 |
  | 6144 | 1618.8 / 1624.9 | 329.19 | 126.50 | 20 796.2 |
  | 8192 | 1737.5 | 278.12 | 86.43 | 19 859.5 |
  | 12288 | 1862.7 | 225.71 | 47.70 | 18 079.6 |
  | 16384 | **1942.2** | 196.26 | 31.59 | 16 393.4 |

  +20 % prefill from 6144 -> 16384 on the same config, no code. The cost is visible in the same table: the
  compute buffers grow and the arena shrinks (20.8 -> 16.4 GiB), so it trades transport bytes for residency and
  needs a sweep at the consumer's shape rather than a blanket recommendation.
* **A bigger arena alone is nearly inert.**  `-c 65536` (arena 23 044.5 MiB, 48 %): pool 312.0 GiB (vs 329.19),
  prefill 1638.5 t/s (+1.2 %). The arena is *sized* to 43-50 % of the expert bytes but only ~28 % of the routed
  offers are served from it (126.5 of 455.7 GiB) because the slots are filled on demand during the pass
  (`expert access: hits=951 fills=2019`). Sizing the arena without pre-populating it buys little.

## 4. What this means for the campaign (the honest arithmetic)

1. **The bytes are irreducible and already moved at the link's rate.**  An in-place alias reads the same bytes
   over the same PCIe link at the same ~33 GB/s. It saves a VRAM write + a VRAM re-read of the same bytes (local,
   ~0.1 ms of the 7.5 ms per call) and it removes the double buffer — it does **not** move fewer bytes.
2. **So the whole prize is overlap.**  At `-ub 6144` the copy is 34.9 % of prefill and is serialised; the
   theoretical best case for fusing the read into `MMQ` is `max(compute, read)` = 20.3 s instead of 31.2 s
   (+53 %), and the same best case at `-ub 2048` is +122 %. The one published in-place prefill on this hardware
   (`briansp2020`'s branch: 1949 vs 1802 t/s = **+8 %**) recovers roughly a fifth of that window, which is the
   realistic order of magnitude, not the ceiling.
3. **The staging engine is not the thing to trim.**  At `-ub 6144` staging moves 410-420 GiB per pass (304 GiB
   H2D + 116 GiB D2D) against the gather's 456 GiB offered (329 pool + 126 arena) — `i.e. it already moves the
   same bytes` — and it is 20-28 % slower because every staged byte is host-memcpy'd into a bounce and issued as
   H2D, while the gather's kernel reads the pinned alias directly. Staging cannot be trimmed without *becoming*
   the gather; the next trim is "don't copy at all".
4. **Step 2's first half is already proven by shipped code.**  A kernel reads the pool alias in place today:
   1410 calls, `0` experts served from the neither-tier path, at ~33 GB/s, under both `MOE_HOST_POOL_ALLOC`
   shapes (the `portable` arm reproduced 1623.3 t/s and byte-identical census). No one-off test is needed to
   show the alias is readable; what does not exist is handing the *op* the alias instead of a copy.
5. **The seam for step 3 already exists in the tree and is dead.**  `mmq_routed_src` (`mmq.cuh`) is an
   in-kernel, per-expert source switch (`arena` slot / `cold` host alias / `tail`) in
   `mul_mat_q_routed_compact`, i.e. exactly the shape step 3 recommends — but `routed_src` is initialised to
   `{}` and never filled, so `src.on == 0` and the kernel always reads the staged `x`.  The caller was dropped
   in the r50 rebase (`archive/work/r50-rebase/README.md` §"kept the `mmq_routed_src` struct").  A future
   attempt starts there, not from scratch.

## 5. The risk that decides it

**A per-expert in-place read replaces "one deduplicated copy per ubatch" with a read inside the tile loop.**  The
gather's whole point is that each routed expert's bytes cross PCIe **once per ubatch**; `mul_mat_q_routed_compact`
launches one block per (expert, J-tile), so an alias is only byte-neutral if L2 absorbs the repeats for every
expert whose token columns span more than one J tile. That is not derivable from the gather's numbers and it is
the -22 % precedent's class of change (`P4-ROOT-CAUSE.md` §5d: a registered mapping whose fills became
synchronous DMA). It has to be measured before it is believed.

## 5b. The follow-up lead (measured): the transport's STREAM, not its bytes

`SPIKE.md` §4's "a cheap lead" was followed to a result in
[`../LEAD-pool-staging-gap.md`](../LEAD-pool-staging-gap.md).  Summary, because it changes this note's
conclusion: the gather is serialised on the compute stream (10.9 s of a 32.7 s prefill at `-ub 6144`), the
r114 single-launch staging kernel *starves* the compute kernels (same-count kernels 3-41x slower), and the
per-expert copy-engine staging (`GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0`) overlaps the same bytes
under compute for **1842 t/s vs the gather's 1627 (+13 %)** on one GPU at `-ub 6144`.  On a 3-device tensor
split the ordering reverses (gather 2359 > kernel 2075 > CE 2053), so the gather default is right there.  The
in-place alias is therefore **not** the lever: it moves the same bytes at the same rate for a prize (overlap)
that the copy-engine path already delivers without touching a prefill kernel.

## 6. What would reopen / what this does not cover

* `-sm tensor` at 2/3 devices (the mirror rows the rig still owes): the per-device slice offset is the r221
  cold-id geometry, and the gather's per-device share/rate are unmeasured here.
* The 179.5k second pass at the rig's own shape (row H's 1802 t/s) — the share at 52k is what this spike has.
* A pipelined/double-buffered gather (the archived `host-pool-speed/PROPOSAL-pipelined-staging.md`): it attacks
  the same 34.9 % by overlapping the copy with compute without putting a host pointer in the prefill kernels.
  It was measured neutral in the *pageable* world; it is untested in this gather-at-link-speed world, and on
  these numbers it is the lower-risk way to chase the overlap.

## Reproduce

```sh
cd wip/host-expert-mirror/spike-261105
./arm.sh ub6144-gather 6144                     # 1618-1625 t/s, gather census
./arm.sh ub6144-stage 6144 GGML_SCHED_DEVGATHER=0  # 1241-1352 t/s, staging census
./trace-arm.sh trace-ub6144 6144                # durations only (kernel trace)
```

The recorded logs are in `runs/*/run.log` (+ `runs/trace-*/trace_kernel_trace.csv` for the two traces, ~20 MB
each — not for commit).
