# HANDOVER — `wip/host-pool-speed` (cold start, 2026-11-02)

**Read this first; the campaign charter is in [`README.md`](README.md).**  Nothing here is in `patches/` and
nothing is applied to the fork (`AGENTS.md`, WIP rules).  `README.md` holds the goals, the scope decisions and
the plan; this file holds the state, the measurements taken and the next actions.

## 0. Status, and the ONE thing to do next

> **Superseded for the *next* session (2026-11-02, session 3): if you are starting on the MULTI-GPU box, read
> [`HANDOVER-MULTIGPU.md`](HANDOVER-MULTIGPU.md) instead — it carries the current state (four campaign patches,
> the pool-aware gather, P1 closed on evidence, the remap disposition in `TODO` #269) and the ordered work list
> (W1 multi-device arms → W2 tensor-split-aware remap → W3 the `RAM_PCT` clamp → W4 the head-to-head).**
> This file remains the **1-GPU host's session-2 record**: its §3 baseline numbers are `soar`'s own, and its
> §4/F7 (no expert cache for `pinned` under `-sm tensor`) is the first blocker the multi-GPU session inherits.
> Session 3's measured record is `README.md` §14-§17.

**Session 2 (2026-11-02, overnight) — the blocking finding was a false alarm, the baseline is measured, and the
decode lever is identified.**

1. **F3 is RESOLVED and there is no delivery bug** (§4/F3): `--host-experts pool` activates on the 122B exactly as
   on 35B-A3B (144 host tables, `ROCm_HostMap`, a 204/256-slot pool, clamped, prepopulated, re-ranked).  Session 1
   mis-read three artifacts: `LLAMA_LOG_INFO` needs `-lv 4` (so `-lv 3` cannot show any of the lines F3 looked
   for), `moe_cache_report` is an `atexit` hook that prints **after** the cache is un-sized (so
   `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned` is printed by every run in every mode — the 35B "control"
   reads identically), and the 22.34 t/s row labelled "pool" had no pool in it.
2. **The pool baseline is measured** (§3): `pinned` 22.33 +/- 0.06, `pool`/`O_DIRECT` **17.76 +/- 0.03
   (-20.5 %)**, `pool`/buffered 15.22-18.10 (unstable).  Prefill -7.5 %.  The `DIO` boolean is *free* here (§5.1).
3. **The decode lever is pool COVERAGE, through demand disk reads** (§3.2): at `FRAC` 50 / 80 / 100 decode is
   11.75 / 17.76 / **20.59** t/s with 366 / 165 / 100 GiB read per run (~163 / ~53 / ~13 MiB per decode token).
   A *full* pool still trails `pinned` by **7.8 %** (the maintainer's observation, with a number), and the shipped
   `FRAC 80` default costs **16 %** against full coverage on this box.  The copy kind is excluded (D2D == H2D,
   §3.1).
4. **F2 is RESOLVED, and the two-tier model is now calibrated against the disk tier** (§4/F2): the runtime's `h=`
   is decode-band only (`--score-band decode` is now the default; +0.0004 / -0.0030 error), and two pool-model
   defects are fixed so the model predicts 160 / 49 / 0 MiB of L3 per token at `FRAC` 50 / 80 / 100 against
   ~163 / ~53 / ~13 measured.
5. **The first screening-passed candidate**: README §4.2's "size the pool by heat mass" is *degenerate* (every
   table routes `n_used` experts per token, so its mass is a constant), but a **concentration**-based sizing
   (`cov90`) separates at **-42 % disk bytes/token** at `FRAC 80` (§4/F6).
6. **The maintainer's three calls are implemented, gated and measured** (§3.3, `patches/0001-0002`):
   `cov90` per-table pool sizing (**+7.5 %** decode at `FRAC 80`, **+16.4 %** at `FRAC 60`, identical pinned
   bytes), **full coverage as the default** (`MOE_HOST_POOL_FRAC=100` -> decode 20.62 vs 17.80, which also makes
   the pool's raw overhead measurable at **-7.9 %**), and the two diagnostic fixes (the teardown report now
   describes the run, and a forced `DIO` logs its mode).  All gates green; the one trade is prefill (-13/-15 %).

**Next action, exactly:** re-measure `cov90` on the **2-GPU x8 `-sm tensor`** arm and under the 64 GiB cgroup
(the field case, where it should pay most), and decide on the prefill trade (§3.3 item 3) before promoting
`patches/0001-0002` into `patches/` — nothing here is in the delivery set yet.

## 1. What the campaign is (do not re-derive)

Goal: make `--host-experts pool` **simpler, more automatic and faster**, so that PR #267's
`--host-experts mmap` + sparse-registration has **no performance advantage** and stays unadopted.  Pool mode is
the keeper because every host read is staged through an explicit pinned intermediary, which works for every
split type; #267 reads the pageable master directly, which needs `n_slices == 0`, POSIX-only aliases and still
declines under `-sm tensor`.  Full reasoning and the #267 mechanism-by-mechanism mapping: `README.md` §1/§8.

Scope decisions already taken by the maintainer (do not relitigate):

- **Phase 1** = one GPU **and** `-sm tensor` on the two PCIe5 **x8** GPUs; 3-GPU arms and Flash-Next are Phase 2.
- **`-ncmoe 48` is the primary arm** (49 blocks → ~62 GiB of host experts; best arena hit rate, the VRAM-limited
  shape, and the tightest pool case).
- **Decode runs ≥ 2000 tokens** (1000 floor).  Short runs are correctness smoke only.
- **`-ub 2048 -b 2048` always** (the default `-ub 512` under-stresses the system; the maintainer's explicit call).
- **Reference model = 122B-A10B UD-Q4_K_XL**; Flash-Next is a final-validation model (its `ple.*` engine is a
  second stream subsystem and confounds everything — verified in its GGUF).
- Candidates are **screened on a real routing trace before implementation** (`README.md` §9.6).
- Avoid `GGML_MOE_DEVREMAP` in the tuning (the maintainer's call; it also declines on pool tables by
  construction, so the pool arms are unaffected either way).

## 2. Hard facts about the setup (verified, do not re-check)

**Box** (`primary`): 96 GiB RAM (135 GiB swap), 32 threads, 3 × R9700 gfx1201 + iGPU gfx1036 = 4 HIP devices.

| HIP index | PCI | CPU root port | link | use |
|---|---|---|---|---|
| **0** | `03:00.0` | `00:01.1` | **x8** | Phase-1 pair (first) |
| 1 | `06:00.0` | `00:01.2` | **x4** | **never** |
| **2** | `09:00.0` | `00:01.3` | **x8** | Phase-1 pair (second) |
| 3 | `1a:00.0` | — | iGPU | never (gfx1201-only build → `invalid kernel file`) |

Established with `rocm-smi --showbus` + the **CPU root ports'** `current_link_width` (8/4/8).  The GPU's own
`current_link_width` reads 16 behind the switch — **not** a valid signal.  With `HIP_VISIBLE_DEVICES=0,2` the
in-process indices are 0 and 1, so `--device 0,2` under that pin silently selects one device.

**Model** `/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-0000{1,2,3}-of-00003.gguf`,
`qwen35moe`, **73.2 GiB**, 49 blocks × 256 experts (8 used), hybrid SSM (`full_attention_interval=4`),
`nextn_predict_layers=1` (embedded MTP head — only loaded when a speculator is enabled; with `--spec-type none`
the loader logs `model has unused tensor blk.48.*` and ignores it, which is expected), **no PLE**.  Host expert
set at `-ncmoe 48` = **67,110 MiB**.  Expert weights ≈ 1.4 GB per layer, so `-ncmoe N` ≈ `1.4 × N` GB on host.
Load ≈ 19 s (`/llm` is on the fast mount).  Sidecar prior: `<base>.moe-heatmap` next to the shards.

**Build**: `~/llama.cpp` at `efaa60db0` (r206 tip), binaries in `build-rocm/`,
`LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`.

## 3. Measured — the pool-vs-pinned baseline, 1 GPU (`-ncmoe 48 -ngl 99 -c 8192 -ctk/ctv q8_0 -fa auto
-b 2048 -ub 2048`), 5258-token prose prompt, `-n 2000`, greedy, arms **alternated** in one run
(session 2, `data/20261102-session2-arms/`, `tools/run-arms-1gpu.sh`)

| arm | decode t/s | prompt t/s | arena `h=` | effective `DIO` | disk read | wall |
|---|---|---|---|---|---|---|
| `pinned` #1 | 22.27 | 679.3 | 0.7543 | — | 72.1 GiB | 230 s |
| **`pool`, DIO auto** | **17.73** | 632.8 | 0.7623 | `O_DIRECT` | 170.4 GiB | 147 s |
| `pinned` #2 | 22.37 | 681.2 | 0.7543 | — | 70.5 GiB | 143 s |
| `pool`, `MOE_HOST_POOL_DIO=0` (buffered) | **15.22** | 542.7 | 0.7623 | buffered | 161.5 GiB | 168 s |
| `pool`, `MOE_HOST_POOL_DIO=1` (forced) | **17.77** | 626.4 | 0.7623 | `O_DIRECT` | 161.7 GiB | 145 s |
| `pinned` #3 | 22.36 | 679.4 | 0.7543 | — | 62.7 GiB | 125 s |
| `pool`, `MOE_HOST_POOL_DIO=0` (buffered) | **18.10** | 587.6 | 0.7623 | buffered | 152.5 GiB | 146 s |
| `pool`, DIO auto | **17.78** | 628.4 | 0.7623 | `O_DIRECT` | 161.9 GiB | 146 s |

Aggregates: **`pinned` 22.33 +/- 0.06** (three arms, 0.3 % spread), **`pool`/`O_DIRECT` 17.76 +/- 0.03**
(**-20.5 %**), `pool`/buffered **15.22-18.10** (unstable, mean 16.66, -25 %).  Prefill: `pinned` 680 +/- 1,
`pool`/`O_DIRECT` 629 +/- 3 (**-7.5 %**), buffered 543-588.

The pool arm reads **152-170 GiB from disk per run against the pinned arm's 63-72 GiB** (which is only the
model load): +90 GiB, of which ~53.7 GiB is the one-shot pool prepopulate and ~36 GiB is **demand reads on the
arena-fill path** -- the maintainer's "streamed from disk over and over" effect, quantified at this arm.

Cumulative cache counters (whole session, from the teardown report; the run's own `h=` is decode-band only,
see F2):

| arm | hits | fills | colds | hit/fill/cold | H2D fills | H2D cold reads |
|---|---|---|---|---|---|---|
| `pinned` #3 | 1,618,544 | 71,790 | 455,482 | 75.43 / 3.35 / 21.23 % | 130,497 MiB | 827,473 MiB |
| `pool` DIO=1 | 1,639,383 | 103,148 | 408,157 | 76.23 / 4.80 / 18.98 % | 187,558 MiB | 741,056 MiB |

So the pool arm **does not** move more bytes over the host link (929 GiB vs 958 GiB total) — it moves
slightly fewer, in slightly different pieces.  The -20.5 % is therefore not fill bandwidth; the copy *kind*
is the next suspect (§5.2).

Read-outs: **auto arena = 22.5 GiB** (`alloc_all_locked` sizes from what is free after `--fit`, the graph
reserve and the KV cache, and takes the slab's 23 GiB arena region) — "MIB unset" is *not* small.  Restricting
it to 4 GiB costs **4.3× the prefill and 42 % of the decode**.  The pinned arm holds **65.5 GiB of
non-swappable RAM** (`ROCm_Host model buffer size = 67110.00 MiB`).  The remap (DEVREMAP) is worth **nothing
measurable at 22 t/s** on this arm — its per-op sync is buried at 44.7 ms/token (on 35B-A3B at 60 t/s it was
+8.4 %), so do not use the 122B to judge it.  `pinned` reaches 22.3-22.4 t/s, so **the pool deficit to explain
is 4.6 ms/token**.

Session 1's table is superseded (two of its rows were mis-labelled, F3); its re-read form is in §4/F3.

### 3.1 The copy kind (`MOE_HOST_POOL_H2D`) — tested, **no effect**

`MOE_HOST_POOL_H2D=1` (fork-local, `patches/h2d-experiment.patch`) makes a pool hit/fill a
`cudaMemcpyHostToDevice` from the pool's pinned host pointer instead of the historical
`cudaMemcpyDeviceToDevice` from its mapped UVA alias (what the pinned arm does on its own master, §5.2).
`data/20261102-h2d/`:

| arm | decode t/s | prompt t/s | disk read / run |
|---|---|---|---|
| `pinned` (3 arms) | 22.36 / 22.35 / 22.37 | 682.0 / 679.3 / 682.4 | 61.5-67.8 GiB |
| `pool`, D2D from the alias (shipped) | 17.76 / 17.80 | 626.3 / 635.4 | 148.5-165.0 GiB |
| `pool`, H2D from the pinned pointer | 17.79 / 17.78 | 636.5 / 631.3 | 150.7-161.4 GiB |

**0.1 % apart, inside the arm-to-arm noise: the copy kind is not the deficit.**  The switch is reverted from
the fork's tree (the patch is kept in the campaign for the record), so no delivery decision is implied.

What the same table *does* show is where the gap is, by wall clock:

| phase | `pinned` | `pool` | delta |
|---|---|---|---|
| prefill (5258 tok) | 7.71 s | 8.36 s | **+0.65 s** |
| decode (1862 tok) | 83.4 s | 104.8 s | **+21.4 s** |
| disk read per run | ~69 GiB (the model load) | ~169 GiB | **+100 GiB** |

Of that +100 GiB, ~53.7 GiB is the one-shot pool prepopulate and **~46 GiB is demand reads on the
arena-fill path**.  At this box's ~2 GB/s that is **~23 s**, i.e. the +21.4 s decode deficit almost exactly.
So the pool's decode cost on this arm is **not** a copy, a sync or a bandwidth problem — it is **disk reads
caused by the arena drifting off the pool's complement** (an expert the arena evicts is only in the pool with
probability `FRAC`, so ~20 % of arena misses at FRAC 80 find nothing in the pool and go to the file).  §3.2
tests that directly by moving `FRAC`.

### 3.2 Pool coverage (`MOE_HOST_POOL_FRAC`) is the decode lever — and the model now predicts it

Same arm, `FRAC` moved (all `DIO auto` = `O_DIRECT`, `data/20261102-frac/`):

| `FRAC` | pool slots/table | pinned pool | decode t/s | prefill t/s | disk read / run | model's L3 MiB/token |
|---|---|---|---|---|---|---|
| **50** | 128/256 | 33.5 GiB | **11.75** | 330.1 | 366 GiB | 160.1 |
| **80** (default) | 204/256 | 53.7 GiB | **17.76 +/- 0.03** | 629 +/- 3 | 152-170 GiB | 48.7 |
| **100** | 256/256 | 67.1 GiB | **20.59 / 20.61** | 620.4 | 93-108 GiB | 0.0 |
| `pinned` | — | 67.1 GiB master | **22.34 +/- 0.01** | 680 +/- 1 | 61-71 GiB | — |

Three things fall out of it:

1. **Decode and prefill are monotone in pool coverage, and the disk reads are the mechanism**: 366 -> 165 ->
   100 GiB per run, i.e. ~163 / ~53 / ~13 MiB of demand read per decode token (measured minus the ~70 GiB
   model load), and the replayer's L3 prediction is 160 / 49 / 0.0 MiB per token on the same trace.  That
   agreement is the pool-side calibration of the tool (§4/F2).
2. **A FULL pool still trails `pinned` by 7.8 %** (20.60 vs 22.34) — the maintainer's "a pool big enough for
   the whole set still trails pinned", now with a number.  Note the full pool pins **67.1 GiB**, i.e. exactly
   the RAM the pinned master uses, so at `FRAC 100` the pool has no memory advantage left; the 7.8 % is the
   residual fill/cold-read path, and it is the *only* decode item left on this arm.
3. **The shipped default `FRAC 80` costs 16 % decode against a full pool on this box**, and the
   `MOE_HOST_POOL_RAM_PCT` clamp did *not* bind at `FRAC 100` (256/256 slots were granted; MemAvailable was
   ~80 GiB so the 85 % ceiling was ~68 GiB >= 67.1 GiB).  So on the 122B at `-ncmoe 48` the pool *can* hold the
   whole set and the default wastes it.  That is a **default-policy finding** for README §5's "one automatic
   decision": when the host set fits under the RAM ceiling, ask for full coverage (the memory cost is the same
   as `pinned`, and the alternative is `pinned`).  It needs the field boxes (a 30 GiB gfx1100 cannot) and the
   3-GPU arms before it is a shipped default, so it is recorded, not applied.

No swap anywhere (`SwapFree` 133.03 -> 133.05 GiB across the whole matrix), and no `oom`/`gpu fault` in the
journal.

**Greedy text across modes (first hashed check on the 122B in this campaign):** pinned / `pool` D2D /
`pool` H2D all produce `270 chars sha=cacea85d5705` at `-n 64` (`data/20261102-text-purity/`, logs without
`-lv`).  The same flags at `-lv 4` build a pool (`host_experts_mode=POOL`, `table_pool_locked ... 204/256`), so
this is a real cross-mode purity result, not three pinned runs: at default verbosity the cache's whole report
is WARN-and-above, so the activation was proved by a separate `-lv 4` control.  (`-lv 3` is *not* usable for a
text hash: the WARN lines that appear during the first decode land inside the extracted span and change it —
1500 vs 1621 chars in this run — so the hash must be taken with no `-lv` at all.)

### 3.3 The two sizing changes, implemented and measured (2026-11-02, later session)

The maintainer's calls: implement `cov90`, make **full coverage the default** (it is what makes the pool's own
overhead measurable), and land the two diagnostic fixes.  All three are implemented
(`patches/0001-*`, `patches/0002-*`), gated, and measured on the reference arm — 7 arms, alternating, 2000-token
decodes, `data/20261102-sizing/`:

| arm | decode t/s | prefill t/s | arena `h` | pool slots (last table) | pool MiB pinned | disk read | pool hit (teardown) |
|---|---|---|---|---|---|---|---|
| `pinned` #1 | **22.39** | 677.2 | 0.7543 | — | — | 65.1 GiB | — |
| **`pool` default = FRAC 100** | **20.62** | 613.4 | 0.7596 | 256/256 | 67,110.0 | 111.6 GiB | **0.9169** |
| `pool` FRAC 80 uniform (the old default) | 17.80 | 637.0 | 0.7623 | 204/256 | 53,478.3 | 153.7 GiB | 0.6200 |
| `pool` FRAC 80 **cov90** | **19.14** | 554.3 | 0.7601 | 160/256 | 53,477.2 | 131.1 GiB | 0.6785 |
| `pool` FRAC 60 uniform (sub-full / field-like) | 13.60 | 443.2 | 0.7664 | 153/256 | 40,108.7 | 271.8 GiB | 0.3405 |
| `pool` FRAC 60 **cov90** | **15.83** | 377.3 | 0.7600 | 112/256 | 40,108.3 | 210.4 GiB | **0.4416** |
| `pinned` #2 | 22.38 | 689.4 | 0.7543 | — | — | 61.7 GiB | — |

**1. Full coverage by default measures the pool's raw overhead at -7.9 %** (20.62 vs 22.38): at `FRAC 100` the
pool pins 67,110 MiB — exactly what the pinned master occupies — and its own hit rate is 0.9169.  That -7.9 % is
now the number the mode reports about itself, and the only decode item left on this arm.

**2. `cov90` sizing wins wherever the budget is not already full, at identical pinned bytes**
(53,478.3 vs 53,477.2 MiB; 40,108.7 vs 40,108.3 MiB):

| `FRAC` | uniform | cov90 | decode gain | disk read | pool hit |
|---|---|---|---|---|---|
| 60 | 13.60 | **15.83** | **+16.4 %** | 271.8 -> 210.4 GiB (-23 %) | 0.3405 -> 0.4416 |
| 80 | 17.80 | **19.14** | **+7.5 %** | 153.7 -> 131.1 GiB (-15 %) | 0.6200 -> 0.6785 |
| 100 | 20.62 | (identical: full coverage leaves no split to make) | — | — | — |

The fit itself: `144 host tensors, budget 40108.7 MiB, alpha=1.108, 22059 slots vs 22032 uniform (100.1 %)` — i.e.
the same budget, redistributed by concentration.  The gain grows as the budget tightens, which is the field case
(a 30 GiB box cannot hold a 67 GiB set), so this is the campaign's first *measured* pool win that is not a bug fix.

**3. The trade to record: cov90 costs prefill.**  637 -> 554 t/s at FRAC 80 (-13 %), 443 -> 377 (-15 %) at FRAC 60.
Mechanism (hypothesis, not yet isolated): the pool's content is ranked by the *decode* prior, and concentration
sizing gives each table fewer slots, so a prefill ubatch — whose routing is nearly uniform over all 256 experts —
finds less of what it needs.  In wall clock it is still a clear net win for a 2000-token request (FRAC 60: prefill
+2.0 s, decode -19.3 s; FRAC 80: prefill +1.2 s, decode -7.3 s).  A prefill-friendly alternative (stage from the
mapping rather than the pool, or reserve a prefill share) is a candidate for the next session, not a fix.

**Gates (all green):** greedy text byte-identical across `pinned` / FRAC 60 uniform / FRAC 60 cov90 / FRAC 100
default (`270 chars sha=cacea85d5705`, hashed with no `-lv`, `data/20261102-sizing-purity/`);
`MOE_HOST_POOL_VERIFY=1` on a cov90 arm: **0** `POOLREAD MISMATCH`; `scripts/check-gpu-faults.sh` clean on all 7
arms; no swap growth (`SwapFree` 132.91 -> 132.90 GiB).

**Diagnostics (patches/0001):** the teardown report now snapshots its structural aggregates before
`moe_cache_forget_all` un-sizes the cache, so it reads the RUN (`144 tensors, 36864 slots, 67110.0 MiB pinned;
h=0.9169`) instead of zeros — which is why the table above has a pool hit rate at all.  A forced
`MOE_HOST_POOL_DIO=0|1` now logs its effective mode (the `(forced by ...)` branch was unreachable dead code).

## 4. Findings

**F1 — a decode routing trace requires `GGML_MOE_DEVREMAP=0`.**  The `D` records are emitted by the eager
`moe_cache_update_host` hook; the remap never reads the routing back and the scheduler returns before the eager
path.  Shipped default → the trace stops after the two sizing passes (**288 `D` records for 1864 decode steps**);
with the remap off → **268,416 `D` records** (38 MB, 1864 × 144 tables ✓).  `ENVIRONMENT.md`'s
`MOE_EXPERT_TRACE` row and the replayer's docstring document the identity-table caveat but not this one — a
one-line doc fix when the campaign next touches docs.

**F2 — RESOLVED 2026-11-02 (session 2): the replayer now calibrates, and the cause was the score band.**
The runtime's logged `h=` is a **decode-band-only** number, and the replayer was scoring the prefill band with
it.  Proof from the run's own counters (no model involved):

* `takeover=268272` = 144 tables x 1,863 decode steps (the per-op remap hook fires once per layer per token).
* `expert access: hits+fills+colds` = **2,145,816** (`pinned`) / **2,150,688** (`pool`) reaches, and
  144 x 1,863 x 8 experts = **2,146,176** — i.e. `h=` counts one reach per expert per *decode* step and
  **exactly zero** prefill reaches.
* The trace carries 423 `P` records holding **740,814 prefill expert positions**
  (`n_tok` 1162/2044/2048).  `moe_cache_update_host` returns before `access_locked` on the prefill band
  (whole-table staging, or a decline), so none of those positions can ever reach `hits`/`misses`.  Scoring them
  through the decode policy is what pulled the model's `shipped` h down by 0.15.

Calibration, same trace (`data/traces/trace-122b-ncmoe48-devremap0.txt.gz`), `--seed sidecar:<the model's
.moe-heatmap>`, `--devices 1`:

| arena | run's `h=` | replayer `--score-band all` (old default) | replayer `--score-band decode` | decode error |
|---|---|---|---|---|
| 3,932 MiB / 15 slots (the run's own sizing) | 0.3488 | 0.1956 (**-0.153**) | **0.3492** | **+0.0004** |
| 22,517 MiB / 85 slots (auto arena) | 0.7543 | 0.6328 (**-0.122**) | **0.7513** | **-0.0030** |

**Fix applied to `scripts/moe-trace-replay.py`** (tool only, no `patches/` change): `--score-band` now defaults
to `decode`, the help/docstring say why (with the reach arithmetic above), and the calibration note points at
`--seed sidecar` + `--slots`.  `all` stays available as a diagnostic.  Re-verified through the delivery script
itself: `calibration: model 'shipped' h=0.3492 vs the run's logged h=0.3488 (error +0.0004)` and
`h=0.7513 vs 0.7543 (error -0.0030)`.

Two candidate causes were tested and **killed**: the `ghost` reset on admission (`access_locked` zeroes
`ghost[e]` when it admits; the replayer keeps a lifetime counter) moves the decode-band h by **0.0002**, and
`--churn off` does not fit either band.  Two practical notes: the slot count still matters (`--slots 14` gives
0.3390 vs 15 -> 0.3492, so pass the run's own count when the log has it), and the sidecar seed is nearly
irrelevant *at 4 GiB* (sidecar 0.3492 / heat 0.3492 / none 0.3395) because 232k evictions wash it out — the
`--churn off` arm is where the seed shows (sidecar 0.2757 vs none 0.0670).

Screening tool: `tools/replayer-calib.py` (sweeps band x churn x seed x ghost-reset on one trace).

**F2, part 2 (same session): the two-tier pool model had two real defects, and it now predicts the disk
tier.**  Both were found while screening README §4.2's sizing candidate, and both are fixed in
`scripts/moe-trace-replay.py`:

1. **`p_map` leak in `pool-shipped`.**  The hand-over evicted the weakest pool entry with
   `self.p_slot[w] = -1`, which freed the slot but left the expert in `p_map` forever.  `p_map` then grew
   past `slots_p` (measured: 26,769 "resident" experts in 9,216 slots at `--pool-frac 25`) and a phantom
   entry counted as a pool hit.  Fixed to `self.p_drop(...)`, and `p_put` now also mirrors
   `pool_slot_assign_locked`'s "never two indices for one expert".
2. **A cold (admission-rejected) miss was counted as an L3/disk read even when the expert was pool-resident.**
   With `MOE_HOST_POOL_COLD` (default 1) the kernel reads a pool-resident expert **in place** from the pool's
   device alias — a PCIe read of pinned RAM, not a disk read.  `TwoTier.cold_read()` now separates the two
   (`pool_cold` vs `l3`), which is what makes `l3` comparable to `/proc/diskstats`.  Before the fix the model
   predicted ~460 MiB/token of L3 at *every* `FRAC`; after it, on the reference trace at `--arena-mib 22517`
   and `heat-complement` init:

   | pool `FRAC` | model L3 MiB/token | measured demand read/decode token | run decode |
   |---|---|---|---|
   | 50 | **160.1** | ~163 | 11.75 t/s |
   | 80 | **48.7** | ~53 | 17.76 t/s |
   | 100 | **0.0** | ~13 | 20.59 t/s |

   The 100 % residual is page-cache warmth in the measured arm (the pool prepopulate is `O_DIRECT`, the model
   load is not); the 50/80 agreement is within ~10 %.  The single-tier `shipped` calibration is unchanged by
   either fix (0.3492 / 0.7513).  **Use `--pool-init heat-complement`** for a calibration: it is the real layout
   (the arena is heat-seeded, the pool gets the complement).  With plain `heat` the model puts the hottest 204
   experts in the pool *and* lets the arena refill from the top, so L3 reads 34.2 MiB/token at `FRAC 80` instead
   of 48.7 — still the right order, but the wrong layout.

Neither change touches `patches/`; both are recorded in `WORKLOG.md` 261102.001.

**F3 — RESOLVED 2026-11-02 (session 2): the pool activates on the 122B; the diagnosis was three reporting
artifacts, and there is no delivery bug.**  A `-lv 4` re-run of the same arm (`logs/f3-122b-pool-lv4.log`) is
unequivocal:

```
0.00.226.771 I llama_model_loader: host_experts_mode=POOL: NO pinned master; ...
0.00.375.294 I load_tensors: ROCm_HostMap model buffer size =   528.00 MiB      (x144 host tables)
0.02.259.608 I bind_host_dev_locked: the host expert master is pageable (pool-backed `--host-experts pool`) ...
0.04.263.060 I pool_decide_dio_locked: host pool fill mode: O_DIRECT (page cache bypassed) -- MemAvailable
             77873.8 MiB, planned pool 53688.0 MiB, host expert set 67110.0 MiB (the set does not fit beside the pool ...)
0.04.263.070 I pool_clamp_slots_to_ram: host pool RAM ceiling: 85% of MemAvailable = 66192.7 MiB (MOE_HOST_POOL_RAM_PCT)
0.04.263.072 I table_pool_locked: host pool (MOE_HOST_POOL_FRAC of the host expert bytes): 144 host tensors,
             67110 MiB host expert bytes; layer=2 role=blk.2.ffn_down_exps.weight -> 204/256 slots of 2162688 B
0.04.425.518 I pool_prepopulate_locked: prewarmed host pool ... 204/204 slots (ranked by the persisted heat prior)
0.04.465.045 I pool_rerank_from_heat_locked: host pool laid out as the arena's complement ... 51 filled, 0 freed,
             170 wanted non-resident
```

So the pool is built (144 tables x 204/256 slots = **53.7 GiB pinned**), clamped, prewarmed from the sidecar and
ranked as the arena's complement.  Three separate artifacts made session 1 read it as "inactive":

1. **`-lv 3` cannot print a single libllama INFO line.**  `common/log.cpp:551` maps
   `GGML_LOG_LEVEL_INFO -> LOG_LEVEL_TRACE (4)`, and the filter is `verbosity <= thold`.  Every `LLAMA_LOG_INFO`
   therefore needs **`-lv 4`**: `host_experts_mode=POOL`, `bind_host_dev_locked`, `pool_decide_dio_locked`,
   `pool_clamp_slots_to_ram`, `table_pool_locked`, `pool_rerank_*`, *and* `moe_cache_report: MoE expert cache:
   h=` are all invisible at `-lv 3`.  `-lv 3` only shows `COM_INF`/`COM_WRN` from `common/` and `srv`
   (`common_params_print_info`, `slot print_timing`, `alloc_all_locked`, `SLAB_INIT`, the `moe_cache_*` WARNs).
   Session 1's pool arms ran at `-lv 3`, so the lines F3 was told to look for could not appear, in either mode.
   **The campaign protocol must say `-lv 4` for anything that comes out of libllama** (README §9.5 corrected).
2. **`moe_cache_report` is an `atexit` hook that runs after the cache has been un-sized.**  It is registered at
   `moe-expert-cache.cu:6903` (`atexit(moe_cache_report)`), and model teardown calls `moe_cache_forget_all`
   first ("dropped 144 expert-cache table(s) ... and un-sized the cache").  The structural counters live in the
   dropped tables, so **every** run ends with
   `MoE expert cache: h=0.0000 (0/0 reaches), ... tables=0, slots=0, arena=0.0 MiB` and
   `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned; h=0.0000 (0/0)` — pinned and pool alike.  The 35B
   control that "worked" (`logs/poolmode-probe.log:452-453`) reads **exactly** the same.  Only the cumulative
   counters survive the teardown (`takeover=`, `expert access: hits/fills/colds`, `expert H2D traffic`).  This is
   a diagnostic defect worth a delivery fix (report before the forget, or snapshot in `forget_all`), because the
   line reads like decisive evidence of a dead pool and is quoted as such in every run.
3. **The 22.34 t/s row in §3 was not a pool run.**  `logs/base-pool-default.log` contains **zero**
   `host pool (L2)` lines, while every genuine pool run (including a `-lv 3` control I ran to check,
   `logs/f3-122b-pool-lv3.log`) contains exactly one — the line is printed whenever a pool exists, at any
   `-lv`.  That log also has no `ROCm_HostMap` buffer and `ROCm_Host model buffer size = 67110 MiB`, i.e. it is a
   **pinned** run: the invocation did not carry `--host-experts pool` (a command-line slip, or a log/arm mix-up).

Corrected session-1 table (same logs, re-read; 1 GPU, `-ncmoe 48`, 1862-token decode, 5258-token prompt):

| arm | arena `h=` | prompt t/s | decode t/s |
|---|---|---|---|
| `pinned`, shipped default | 0.7543 | 681.2 | **22.40** |
| `pinned`, `GGML_MOE_DEVREMAP=0` | 0.7538 | 684.5 | 22.27 |
| `pool`, `MOE_HOST_POOL_DIO` auto -> **O_DIRECT** | 0.7623 | 624.7 | **17.73** |
| `pool`, `MOE_HOST_POOL_DIO=0` (buffered) | 0.7623 | 493.8 | **18.21** |
| ~~`pool`, "shipped default"~~ **actually pinned** (no pool in the log) | 0.7543 | 681.0 | 22.34 |

The pool costs **-20.8 %** decode against pinned on the 122B at `-ncmoe 48`, and the DIO arm only accounts for
2.4 points of it — so "one extra host memcpy per staged byte" plus the arena's now-pageable fill source is the
bulk, not `O_DIRECT`.  Session 2's matrix (§3) re-measures this properly.

For the record, the control that *works* (35B-A3B Q4_K_M, `-ncmoe 8`, `--host-experts pool`, `-lv 4`,
`logs/poolmode-probe.log`) — note that it is byte-for-byte the same set of lines, only smaller:

```
llama_model_loader: host_experts_mode=POOL: NO pinned master; the host experts stay in the model mapping ...
bind_host_dev_locked: the host expert master is pageable (pool-backed `--host-experts pool`); ...
pool_decide_dio_locked: host pool fill mode: buffered (page cache as L3) -- MemAvailable 79172.9 MiB,
                       planned pool 3051.2 MiB, host expert set 3814.0 MiB ...
pool_clamp_slots_to_ram: host pool RAM ceiling: 85% of MemAvailable = 67296.9 MiB (MOE_HOST_POOL_RAM_PCT)
table_pool_locked: host pool (... FRAC of the host expert bytes): 24 host tensors, 3814 MiB ... -> 204/256 slots
```

Relevant code (checked, and it is *not* declining anything): `llama_model_loader::host_expert_buft()` returns
the master buft when `host_experts_mode == POOL && use_mmap && !LLAMA_HOST_MASTER_PINNED` and
`ggml_backend_dev_moe_host_master_buft(dev) != nullptr` (NULL only for an **integrated** device; the R9700 is
discrete), and the 122B trace shows the `ROCm_HostMap` master being taken for all 144 tables.

**F4 — the pinned arm's price is now measured**: 65.5 GiB of non-swappable RAM for 22.4 t/s on one card.  That,
not a throughput ratio, is what pool mode has to be judged against.

**F5 — `DIO` is a per-box boolean applied to a per-read problem** (the maintainer's question, 2026-11-02).
See §5.

**F7 — the 2-GPU `-sm tensor` shape on the 122B is a different regime: `pinned` gets NO expert cache at all,
and a full pool can stall the load (2026-11-02, preliminary — short decodes, single samples).**  Running the
campaign's Phase-1 pair arm (`HIP_VISIBLE_DEVICES=0,2`, `-sm tensor`, `-ncmoe 48`, `data/20261102-pair-smoke/`):

* **`pinned`: the cache never sizes.**  The log has **no** `alloc_all_locked` line and no `moe_cache_report`
  output at all, and the memory breakdown after load shows `Meta() 65248 = 51424 free` — i.e. ~51 GiB of the
  65 GiB of device memory is untouched.  So on this shape `pinned` decode (14.46 t/s, 8-token run) is the
  *no-cache* rate, and the earlier claim that pool mode works on 1/2/3-GPU tensor splits has to be re-checked
  against a model whose MoE op registers per-device tables under the Meta wrapper.  First thing to check next:
  `llama_model_moe_host_expert_dev_count(model)` under `-sm tensor` (the cache is allocated only when it is
  non-zero), then the `MUL_MAT_ID` device list the registration sees.
* **`pool` builds (144 tensors, 256/256 slots, 67110 MiB pinned) and the first attempt STALLED** in the pool
  prepopulate (rc=137 after 457 s, last log line `table_pool_locked` #125/144, the process sleeping in a
  nanosleep loop) while the box carried a full model page cache (available RAM had fallen to 54 GiB with 67.1 GiB
  of pool to pin).  With a clean page cache the same arm **completed** (`112 GiB` read, `rc=0`), so this is
  over-commitment pressure during the prepopulate, not a pool/layout bug — see the `RAM_PCT` critique (the
  ceiling does not reserve the model's own page cache).
* Both pair pool runs *did* get an arena (`MoE arena = 0.8323`) where `pinned` did not, which is why pool mode
  looks faster than `pinned` on this shape (17.69 vs 14.46, **8-token runs — not a policy number**).

> **F7 CORRECTED 2026-11-03 (multi-GPU box, `soar`): the whole of F7 was an ENVIRONMENT ARTEFACT and there is no
> `-sm tensor` registration gap.**  Those arms ran with **`MOE_EXPERT_CACHE_MIB=0`**, which disables the cache; with
> the cache off, `offload_op` falls back to the batch-size test and a 1-token decode leaves the routed `MUL_MAT_ID`
> on the CPU — so 98 decode splits instead of 50, no slab, no `alloc_all_locked`, no arena, and a "14.46 t/s
> no-cache rate".  Reproduced exactly (14.40 t/s, prefill 887.2, splits 145/98).  With the shipped default
> (AUTO), on the same pair and the same arm: `alloc_all_locked: device 0/1 sized 190 slots/table`, arena
> **49808.2 MiB**, `h=0.9477`, 39.72-39.87 t/s decode / 1604-1648 prefill.  The split accounting is healthy and
> instrumented: `moe_host_expert_dev_count=2`, 33555 MiB per device, `layer_dev=Meta() meta=1` distributed to
> ROCm0/ROCm1.  **The two bullet points above remain valid as descriptions of what the `MIB=0` arm looked like;
> they are not a defect in the split.**  Full record: [`README.md`](README.md) §18.1/§18.2.

**Consequence for the new default:** `MOE_HOST_POOL_FRAC=100` is measured on **one** host-expert device (§3.3) and
should stay the default there; on a multi-device tensor split it over-commits (pool 67.1 GiB + a 73 GiB model on
96 GiB) and needs the page-cache budget, not a bigger FRAC.  The 2-GPU arms must be re-run with 2000-token
decodes before any conclusion, and the `-sm tensor` registration gap above is the real blocker for that arm.

**F6 — README §4.2's per-table pool sizing candidate, as specified, is DEGENERATE; the right signal is
concentration.**  The candidate is "allocate each table's pool slots by its decode heat mass".  A top-k router
hands every table exactly `n_used` experts per token, so a table's *total* decode routed mass is a constant
(`n_used x steps` — 8 x 1899 on this trace) for every table: allocating by mass reproduces the uniform rule
**exactly** (verified: identical slot vectors at `FRAC` 30/80, `tools/pool-sizing-sim.py`).  What does differ
between tables is the *shape* of their distribution, so the candidate must be re-specified as
**`cov_t`: the number of slots a table needs to hold `t` of its own routed mass** (t = 0.90 below), fitted to
the same byte budget.  On the reference trace (arena 22,517 MiB, `heat-complement` pool init, `pool-shipped`
policy, decode band):

| sizing | `FRAC` 50 L3 MiB/tok | `FRAC` 80 L3 MiB/tok | h |
|---|---|---|---|
| uniform (shipped) | 160.1 | 48.7 | 0.7515 |
| heat-mass (as specified) | 160.1 (= uniform) | 48.7 (= uniform) | 0.7515 |
| equal pinned BYTES per table | 160.7 (+0.4 %) | 51.2 (+5 %) | 0.7515 |
| **`cov90`** | **114.0 (-29 %)** | **28.5 (-42 %)** | 0.7515 |

`h` is unchanged by construction (the arena policy does not move), so the whole effect is in the disk tier:
-20 MiB/token x 1862 tokens is ~37 GiB less read per run, i.e. ~18 s of the 21 s decode deficit at `FRAC` 80.
That is a sizeable, cheap, **screening-passed** candidate for the next session (implement from `heat_dec`, same
`RAM_PCT` ceiling, env-gated first).  Caveat: it is a model result, calibrated against the measured disk tier to
~10 % on this arm (§4/F2 part 2) — the run is the gate, not the simulation.

Screening tools: `tools/pool-sizing-sim.py` (sizing candidates), `tools/replayer-calib.py` (model calibration).
Every number above is reproducible from `data/traces/trace-122b-ncmoe48-devremap0.txt.gz` with no GPU.

**F6 is now IMPLEMENTED and MEASURED** (`patches/0002`, `MOE_HOST_POOL_SIZING=1`, §3.3): **+16.4 %** decode at
`FRAC 60` (13.60 -> 15.83 t/s, disk 272 -> 210 GiB) and **+7.5 %** at `FRAC 80` (17.80 -> 19.14, disk 154 -> 131
GiB) at *identical* pinned bytes, with the pool's own hit rate up 10 points at the tight budget.  The measured
gain is smaller than the -42 % disk-bytes screen — the screen assumed a `heat-complement` pool init and reported
its L3 number (48.7 MiB/token) while the run's total disk read also carries the ~70 GiB model load — but the
direction and the budget dependence match.  The one cost: prefill -13/-15 %, from a pool content ranked for
decode serving a nearly uniform prefill routing (§3.3, item 3).

## 5. `O_DIRECT` vs buffered — the analysis and the proposed shape

The shipped rule (`pool_decide_dio_locked`) decides **once per box, at the first pool build**: buffered iff
`planned pool + the whole host expert set fits in MemAvailable`, else `O_DIRECT` for *every* fill.  Why that is
the wrong axis:

- The two phases want opposite things.  **Wide prefill staging** reads each expert once, sequentially, in bulk:
  caching those pages is pure pollution, so `O_DIRECT` is right when the set does not fit.  **Decode-band
  fills** are small (one expert = 1.7 MiB here), latency-critical and repeated over thousands of steps: they
  want the page cache, which is a free L3.  r114 measured exactly this split on 35B-A3B: buffered **2103/46.4**
  (prefill/decode) vs forced `O_DIRECT` **2088/22.7** — prefill identical, **decode halved**.  The all-or-nothing
  switch throws that decode away on exactly the boxes that need the mode.
- The right decision is **per read**, and the information is available:
  1. **Residency check** — `mincore()` on the 1.7 MiB expert range: resident → buffered `memcpy` from the page
     cache (free); not resident → `pread` with `POSIX_FADV_NOREUSE`/`O_DIRECT` (no pollution).  This replaces
     the per-box guess with a per-byte fact.
  2. **A page-cache budget** — `MOE_HOST_POOL_PGCACHE_MIB` (default: `min(25 % MemAvailable, MemAvailable −
     pool − reserve)`), enforced with `posix_fadvise(POSIX_FADV_DONTNEED)` on the coldest expert ranges, which
     the existing heat prior already ranks.  That gives the page cache a *bounded* tier-2 role (the "L3" the
     docs promise) instead of an unbounded thrash source, and it composes with `RAM_PCT` rather than
     contradicting it.
  3. **Readahead on the next fills** — `MADV_WILLNEED`/`fadvise` for the experts the routing is about to need.
     `MOE_HOST_POOL_PREFETCH` already exists (default 0); re-measure it against the r113 layout first.
- Keep `MOE_HOST_POOL_DIO` as the A/B/bisect kill switch (`0`/`1` force either arm), but stop letting the
  *default* be a box-wide verdict.  This is the same "one automatic decision" theme as `README.md` §5.
- Measure it with `/proc/diskstats` (bytes read per decode token), a `mincore` page-cache sampler over the
  expert file ranges, and `VmSwap`/`pswpout` — not throughput alone.

Note for this box: with 67 GiB of host experts and a 53.7 GiB planned pool, `planned pool + set = 119 GiB`
against ~76 GiB `MemAvailable`, so `auto` **picks `O_DIRECT` here** even though the box is RAM-rich.  That is
the phenomenon to fix, and it is why the 122B pool arm must be measured before and after.

### 5.1 Measured (session 2, §3): the boolean is **free** on this arm, and the *buffered* arm is the unstable one

| arm | decode | prefill | disk read |
|---|---|---|---|
| `pool`, `DIO` auto (= `O_DIRECT` here) | 17.73 / 17.78 | 632.8 / 628.4 | 170.4 / 161.9 GiB |
| `pool`, `DIO=1` forced | 17.77 | 626.4 | 161.7 GiB |
| `pool`, `DIO=0` buffered | **15.22 / 18.10** (mean 16.66) | **542.7 / 587.6** | 161.5 / 152.5 GiB |
| `pinned` | 22.33 +/- 0.06 | 680 +/- 1 | 62.7-72.1 GiB |

Three conclusions, and they *change the plan*:

1. **Auto is picking the right arm on this box, and it costs nothing.**  Forced `O_DIRECT` is within 0.3 % of
   auto, so a per-read residency check cannot win 2 % here — it can only win where the pages really are
   resident, which at `-ncmoe 48` on 96 GiB they are not (the pool pins 53.7 GiB and the arena 22.5 GiB, so a
   73 GiB model has almost no page cache left).
2. **Buffered is not the safe arm either — it is the *unstable* arm (15.2-18.1 t/s, 20 % spread)**, because it
   competes with the pinned pool for the same RAM and its prefill loses 13-20 %.  Session 1 saw 18.21 for a
   buffered arm; the variance is the page-cache state, not the build.  So "just buffer the decode band" (the
   naive reading of §5) is disproved on this arm: an unbounded page cache is a thrash source exactly as the
   code comment says, and a *budget* is the part that matters (item 2 of the design, not item 1).
3. **The disk is not where the -20.5 % decode goes.**  The pool arm streams 152-170 GiB per run against the
   pinned arm's 63-72 GiB, but the *host-link* traffic is the same or lower (929 vs 958 GiB total, §3), and the
   decode deficit survives at a constant 17.7 whatever `DIO` does.  So F5's per-read residency is a **disk-tier
   feature for the sub-full-pool / cgroup case** (README §9.2, Phase 2), not the lever on the reference arm.

**Latent bug found while measuring (delivery, tiny, not fixed here):** a forced `MOE_HOST_POOL_DIO=0|1` is
decided in the env parse (`g_pool_dio_decided = true`), so `pool_decide_dio_locked` returns on its first line and
its `(forced by MOE_HOST_POOL_DIO=%d)` branch is **dead code** — a forced arm logs no effective mode at all,
while the env-init line keeps saying "fill mode AUTO".  An A/B arm is therefore invisible in the log (it was
invisible in session 2's own matrix: `pool_decide_dio_locked` appears only in the `auto` arms).

### 5.2 Where the remaining -20.5 % decode is — answered by §3.1

Three suspects were eliminated in session 2, in this order:

1. **The copy kind out of the pool** (`D2D` from the mapped alias vs the pinned arm's `H2D`): **no effect**
   (17.76/17.80 vs 17.79/17.78, §3.1).  The current D2D choice stays; its cross-device rationale is intact.
2. **Host-link bandwidth**: the pool arm moves *fewer* bytes over the host link than `pinned` (929 vs 958 GiB,
   §3), and its arena hit rate is slightly *higher* (0.7623 vs 0.7543).
3. **The `DIO` boolean** (§5.1): free on this arm.

What is left is **disk reads**: ~46 GiB of demand reads per run (≈23 s at ~2 GB/s), against a +21.4 s decode
delta.  The mechanism is the r113 design decision that the demand path never evicts and never copies the arena
victim into the pool: with the pool sized to the arena's *sizing-time* complement, an expert the F-LRU arena
evicts is in the pool only `FRAC` of the time, and the rest are file reads.  §3.2 measures that dependence.
If it holds, the campaign's decode lever is **pool coverage/re-fill policy**, not a copy path, and the
candidates are (a) a higher effective `FRAC` (the `RAM_PCT` clamp allows ~252/256 slots here), (b) filling an
empty pool slot with the arena victim on eviction (r113 measured the *background-worker* version of that at 2x
decode; the synchronous version has never been measured), (c) `MOE_HOST_POOL_PREFETCH` against the r113 layout.
All three are screenable in the replayer (`pool-*` policies) now that F2 calibrates.

### 5.3 Refined F5 design note (what to keep)

1. **Per-read residency (`mincore`) — keep, but as the *sub-full-pool* fix, not the reference-arm fix.**  It is
   the honest answer to "streamed from disk even though it should be in the page cache", and it is what makes
   `DIO=auto` a per-byte decision instead of a per-box verdict.  Gate it on the arm where it can show: a
   sub-full pool (`MOE_HOST_POOL_FRAC=25`) and the 64 GiB cgroup, measured with `/proc/diskstats` bytes/token
   and `MOE_HOST_POOL_VERIFY=1`.
2. **The page-cache budget is the load-bearing half** (§5.1 conclusion 2): the buffered arm is unstable
   without it, so `MOE_HOST_POOL_PGCACHE_MIB` + `posix_fadvise(DONTNEED)` on the coldest ranges should land
   *with* the residency check, not after it.
3. **Readahead (`MADV_WILLNEED` on the next fills) stays behind measurement** — `MOE_HOST_POOL_PREFETCH` is
   still default 0 and unmeasured against the r113 layout.

## 6. Next steps, in order

**Session 2 closed:** F3 resolved (§4, no delivery bug), the pool-vs-pinned baseline + the `DIO` A/B measured
(§3), the coverage gradient measured (§3.2), F5's boolean measured and its design note re-scoped (§5.1-5.3),
F2 reconciled and the two-tier model calibrated against the disk tier (§4), and README §4.2's sizing candidate
re-specified into a screening-passed one (§4/F6).  What is left, in priority order:

1. ~~`cov90` per-table pool sizing~~ **DONE** (§3.3): implemented as `MOE_HOST_POOL_SIZING=1` (`patches/0002`),
   measured +7.5 % (FRAC 80) / +16.4 % (FRAC 60) decode at identical pinned bytes.  Next: gate it on the 2-GPU
   x8 `-sm tensor` arm and the 64 GiB cgroup, then flip its default (default-on policy) if it holds.
2. ~~Full-coverage default~~ **DONE** (§3.3): `MOE_HOST_POOL_FRAC` now defaults to 100 (ceiling-clamped), decode
   20.62 vs 17.80, and the pool's raw overhead is measurable at -7.9 %.  Next: the field boxes (a 30 GiB box
   clamps, which is the interesting arm) and the 3-GPU arms before it ships in `patches/`.
3. **The cov90 prefill trade** (§3.3 item 3): -13/-15 % prefill for +7.5/+16.4 % decode.  Isolate it (is it the
   content ranking or the smaller slot count?) and decide: accept, or give prefill its own staging path.
4. **The 2-GPU `-sm tensor` arm** (§4/F7): `pinned` gets no expert cache there at all and a full pool can stall
   the prepopulate under page-cache pressure — so the *default* is only validated on one host-expert device, and
   the tensor-split registration gap (`llama_model_moe_host_expert_dev_count` under the Meta wrapper) is the
   blocker to fix before the pair arm means anything.  Re-run with 2000-token decodes.
5. **The residual 7.9 % at a full pool** (§3.3): the only decode item left on this arm once coverage is
   complete, and the pool's own overhead by definition.  Candidates: the per-fill event/`bounce_acquire`
   bookkeeping (pool-only), the eager path's per-op sync (present in `pinned` too), and P0-a's remap (which is
   the +8 % mechanism on 35B-A3B).  Instrument before changing: the copy kind is already excluded (§3.1).
6. **`mincore` per-read residency + a page-cache budget** (§5.3), on the sub-full-pool / 64 GiB cgroup arms —
   now measurable against a calibrated L3 model, and still the honest answer to the maintainer's
   "re-streamed from disk" report for boxes where the set cannot be covered.
7. **P0-a — the remap on pool-backed tables** (`README.md` §3.1): still blocked on freshness (a pool layout
   epoch in the cold id, or a decode-band layout freeze).  Judge it on 35B-A3B where its ceiling was measured;
   on the 122B the remap is invisible (§3).
8. Then the remaining P1 items in `README.md`: demand precopy (§4.3) and the two standalone #267 borrows — the
   bounce-ring cap (0010) and arena-across-prefill (0003) (§4.4).
9. Only then the remaining Phase-2 items: 2-GPU x8 `-sm tensor` arms, Flash-Next validation, and the `RAM_PCT`/swap soak
   (which the maintainer places at the very end, as a Flash-Next + PLE effect).

## 7. Traps (all bitten today or previously)

- **`-lv 4` for anything out of libllama; `-lv 3` only covers `common/` and `srv`.**  `common/log.cpp` maps a
  raw `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4), so `LLAMA_LOG_INFO` — `host_experts_mode=POOL`,
  `table_pool_locked`, `pool_decide_dio_locked`, `pool_clamp_slots_to_ram`, `apply_heat_seed_locked`,
  `moe_cache_report: MoE expert cache: h=` — needs `-lv 4`.  At `-lv 3` you get `COM_INF`/`COM_WRN` only
  (`common_params_print_info`, `slot print_timing: MoE arena =`, `alloc_all_locked`, the `moe_cache_*` WARNs).
  The `-lv` default is **1**.  Never `-lv` when hashing generated text (`prompts/README.md`).
- **The teardown cache report is always zero, in every mode.**  `moe_cache_report` is an `atexit` hook that runs
  after `moe_cache_forget_all`, so `tables=0/slots=0/arena=0.0 MiB` and `host pool (L2): 0 tensors, 0 slots, 0.0
  MiB pinned` say nothing about whether a pool existed.  Use the per-turn `slot print_timing: MoE arena =` line
  for `h=`, and the cumulative `takeover=`/`expert access:`/`expert H2D traffic:` lines for the counters.
- `llama-cli` in this build is the **server-based** CLI: chat template on, so pass `--reasoning off` for prose
  and `--single-turn --no-display-prompt`.  It reuses the server's `print_timing` / `slot` log lines.
- Always pass `-c`: the model's training context is 262144 and an unset `-c` spends VRAM on context instead of
  arena (KV is cheap here: ~27 KB/token, ~220 MiB at `-c 8192`).
- `MOE_EXPERT_CACHE_MIB=4096` changes the regime by 2×; use it only to make tables traceable.
- Never run arms in parallel; alternate, and keep the `pinned` arm in the same run.
- Pin `HIP_VISIBLE_DEVICES` on every arm (the iGPU trap), and remember the post-pin index remap for the pair.

## 8. Open questions and decisions (session 2)

**Decisions taken autonomously overnight (the maintainer reviews in the morning):**

- **D1 — `-lv 4` is now the campaign's log level** for any arm whose numbers are read from libllama lines
  (the protocol in `README.md` §9.5 said `-lv 3`; `-lv 3` cannot print a single `LLAMA_LOG_INFO`).  Corrected in
  the protocol, not just in the handover.
- **D2 — the replayer's `--score-band` default is now `decode`** (F2).  This is a `scripts/` change with no
  `patches/` change; `all` remains a diagnostic.
- **D3 — F5 is re-scoped:** per-read residency is a *sub-full-pool / cgroup* feature, not a reference-arm one.
  The reference arm's lever is the copy kind (§5.2), so that is what session 3 chases first.
- **D4 — `wip/host-pool-speed/data/20261102-*/` keeps every arm log and the gzipped calibration trace
  (9.1 MB)** so the numbers above are reproducible without a GPU run.  /tmp is treated as scratch.
- **D5 — the `MOE_HOST_POOL_H2D` experiment is finished and reverted** (no effect, §3.1).  The fork's tree is
  clean at `efaa60db0` and `build-rocm` is rebuilt from it (post-revert `pinned` arm: 22.36 t/s ✓).  The patch
  is kept as `patches/h2d-experiment.patch` for the record only.
- **D6 — two more `scripts/moe-trace-replay.py` fixes** (the `pool-shipped` `p_map` leak and the pool-resident
  cold-read accounting, §4/F2 part 2).  Both are model-fidelity bugs, both validated against measured disk bytes,
  both recorded in `WORKLOG.md` 261102.001; neither changes the single-tier calibration.
- ~~**D7 — `cov90` sizing is NOT implemented**~~ **Superseded by the maintainer's call**: it is implemented,
  gated and measured (§3.3), as is the full-coverage default and both diagnostic fixes.  The changes live on the
  fork's local branch `wip/host-pool-speed-pool-sizing` and are exported as `patches/0001-*` and
  `patches/0002-*`; **nothing is in the delivery set** (`patches/` and `release.json` untouched), and the
  `--host-experts pool` help text, `ENVIRONMENT.md` and `patches/README.md` notes for `MOE_HOST_POOL_SIZING` /
  `MOE_HOST_POOL_COV_PCT` / the `FRAC` default must ride with the promotion, not before it.
- **D9 — the work is SAVED, not promoted** (maintainer's call, 2026-11-02): the two patches live in
  [`patches/`](patches/README.md) with their own README, the campaign docs are current, and
  [`HEAD-TO-HEAD.md`](HEAD-TO-HEAD.md) is written so this approach can be pitted against the parallel session's
  `mmap` / pageable-master one before anything is promoted.  Nothing in the delivery set changed.
- **D8 — the fork is left on `wip/host-pool-speed-pool-sizing`** with `build-rocm` built from it, because the
  next step is to gate these changes.  `git checkout rdna-boosts` + `build-llama-rocm` restores the r206 binary
  (the campaign's earlier numbers were all taken on it; note that on this branch `MOE_HOST_POOL_FRAC` defaults to
  100, so a re-measure of the old default needs `MOE_HOST_POOL_FRAC=80` explicitly).

**Questions for the maintainer:**

1. **The teardown cache report is a trap** (`moe_cache_report` = `atexit`, and `moe_cache_forget_all` runs
   first, so `tables=0/slots=0/arena=0.0 MiB` and `host pool (L2): 0 tensors` print in *every* run, in both
   modes).  It cost session 1 a whole evening and a false "blocking" finding.  Suggested delivery fix: call
   `moe_cache_report()` from `moe_cache_forget_all` before the drop (or snapshot the per-table aggregates into
   statics at forget time and prefer them in the report), gated so it prints once.  Small, diagnostic-only, no
   behaviour change — say whether to carry it as its own change or fold it into the next block amendment.
2. **A forced `MOE_HOST_POOL_DIO=0|1` logs no effective mode** (§5.1): `g_pool_dio_decided` is set in the env
   parse, so `pool_decide_dio_locked`'s `(forced by MOE_HOST_POOL_DIO=%d)` branch is dead code and the
   env-init line still says "fill mode AUTO".  One-line diagnostic fix; same question as (1).
3. **The copy kind (§5.2)**: if `MOE_HOST_POOL_H2D=1` wins on 1 GPU, is the per-device rule (H2D when the
   copying device owns the pool) acceptable as the shipped behaviour, with D2D kept for the cross-device case
   (which is the constraint the current code documents)?  That is a *code* change in block 12's territory, so
   it needs your call before it is cut as a patch.
4. The cgroup-aware `RAM_PCT` denominator (`README.md` §9.2/§3.2): own small change now, or ride with Phase 2?
5. `TODO` handle for the swap-thrash report (still only recorded in `wip/`) — still open from session 1.
6. The knob-collapse target (`README.md` §5) and whether `DIO` becomes part of it (§5 here).


## 9. Files

- `README.md` — charter, scope, plan, P0-P3 items, measurement protocol, open questions.
- **Campaign data (in-repo, survives /tmp):**
  - `data/20261102-session1/` — session 1's arm logs (`base-*.log`), the working 35B pool probe
    (`poolmode-probe.log`), the trace/sizing logs.  The *evidence* behind §4/F3 and the corrected table.
  - `data/20261102-session2-f3/` — the `-lv 3` vs `-lv 4` 122B pool controls that resolved F3.
  - `data/20261102-session2-arms/` — the 8-arm pool-vs-pinned + `DIO` matrix (§3) and `summary.tsv`.
  - `data/20261102-h2d/` — the copy-kind A/B (§3.1): `pinned`, `pool` D2D, `pool` H2D (`MOE_HOST_POOL_H2D=1`).
  - `data/20261102-frac/` — the coverage gradient (§3.2): `FRAC` 50 / 80 / 100 against `pinned`.
  - `data/20261102-text-purity/` and `-lv3/` — the greedy-text hashes across modes (§3.2), plus the
    `pool-control-lv4` arm that proves the pool built for them.
  - `data/traces/trace-122b-ncmoe48-devremap0.txt.gz` — session 1's 122B routing trace (gzipped, 9.1 MB), the
    F2 calibration input; regenerate with the `README.md` §9.6 capture command at `MOE_EXPERT_CACHE_MIB=4096`,
    `GGML_MOE_DEVREMAP=0`.
- `tools/run-arms-1gpu.sh`, `tools/run-arms-1gpu-h2d.sh` (+ `tools/lib-arms.sh`) — the arm runners (pin
  `HIP_VISIBLE_DEVICES=0`, `-lv 4`, 2000-token decode, disk-read + memory accounting per arm, alternating arms).
- `tools/replayer-calib.py` — the F2 sweep (band x churn x seed x ghost-reset) over one trace, importing the
  delivery replayer.
- `tools/pool-sizing-sim.py` — the §4.2/F6 sizing screen (uniform / heat-mass / equal-bytes / `cov90`) through
  the delivery's two-tier model.
- `tools/run-arms-1gpu-frac.sh`, `tools/run-text-purity-1gpu.sh` — the coverage gradient and the no-`-lv`
  text-hash runners.
- `data/20261102-sizing/`, `data/20261102-sizing-purity/`, `data/20261102-sizing-verify/`, `data/20261102-diag*` —
  the sizing matrix (§3.3), the no-`-lv` text hashes, the `MOE_HOST_POOL_VERIFY=1` arm, and the diagnostics
  controls.
- `tools/run-arms-1gpu-sizing.sh` — the sizing/default matrix runner.
- `patches/h2d-experiment.patch` — the fork-local `MOE_HOST_POOL_H2D` switch (§3.1).  **Not in `patches/`**: the
  experiment is finished and the fork's tree is reverted, so this file is the record, not a queued change.
- **[`patches/README.md`](patches/README.md)** — the campaign patch set: what `0001`/`0002` do, the apply order
  (on the r206 tip `efaa60db0`), the validation (trees match `0b8cc5a8b`), the knobs, the expected numbers and the
  caveats.  Read it before rebuilding this work anywhere.
- **[`HEAD-TO-HEAD.md`](HEAD-TO-HEAD.md)** — the prepared protocol for pitting this approach against the
  `--host-experts mmap` / pageable-master one: the fixed arm matrix, metrics, gates, scoring rules and the traps
  that would silently make the two sides incomparable.  **Not run yet.**
- `patches/0001-moe-cache-report-the-RUN-*.patch` — the diagnostics fixes (§3.3), `git format-patch` from the
  fork's `wip/host-pool-speed-pool-sizing` branch.
- `patches/0002-host-pool-full-coverage-default-*.patch` — full-coverage default + `cov90` sizing (§3.3).
  Both apply on the r206 tip (`efaa60db0`).  **Not in `patches/`** until they are gated on the arms §6.1-6.2
  names.
- PR #267 (not adopted): the submitter's branch `brian/wip-mmap-host-experts-r204`, directory
  `wip/mmap-host-experts-r204/`.
