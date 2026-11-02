# HANDOVER — `wip/host-pool-speed` (cold start, 2026-11-02)

**Read this first; the campaign charter is in [`README.md`](README.md).**  Nothing here is in `patches/` and
nothing is applied to the fork (`AGENTS.md`, WIP rules).  `README.md` holds the goals, the scope decisions and
the plan; this file holds the state, the measurements taken and the next actions.

## 0. Status, and the ONE thing to do next

**Session 2 (2026-11-02, overnight) — F3 is RESOLVED, and it was not a delivery bug.**  `--host-experts pool`
activates on the 122B exactly as it does on 35B-A3B (144 host tables, `ROCm_HostMap` buffers, a 204/256-slot
pool = 53.7 GiB pinned, `pool_decide_dio_locked`/`pool_clamp_slots_to_ram`/`table_pool_locked`/
`pool_prepopulate_locked`/`pool_rerank_from_heat_locked` all print).  Session 1's three pool measurements were
mis-read: (1) `-lv 3` cannot print **any** libllama INFO line (`LLAMA_LOG_INFO` maps to verbosity **4**), so the
lines F3 said to look for were unreachable; (2) `moe_cache_report` is an `atexit` hook that runs **after**
`moe_cache_forget_all`, so `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned` is printed by *every* run in
every mode — the 35B control log reads identically; (3) the 22.34 t/s row labelled `--host-experts pool` had no
pool in it at all (zero `host pool (L2)` lines vs one for every real pool run), i.e. that invocation did not
carry the flag.  Full evidence and the corrected session-1 table: §4/F3.

**Next action, exactly:** the pool-vs-pinned baseline at the real auto arena is now running (`tools/
run-arms-122b.sh`, §6 step 2); then F5's per-read residency (§5) and F2's replayer reconciliation (§4).

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

### 5.2 Where the remaining -20.5 % decode is *not* — and the next suspect (copy kind)

The pool arm and the pinned arm move the same host-link bytes, so the gap is not bandwidth.  What differs is
the **copy that fills an arena slot**:

* `pinned`: `cudaMemcpyHostToDevice` from the pinned master (`expert_host_src`, `table_master_pageable`
  false → `out->kind = H2D`).
* `pool`: `cudaMemcpyDeviceToDevice` **from the pool's mapped UVA alias** (`out->kind = D2D`,
  `out->base = pool->base_dev + slot`) — chosen deliberately because a *cross-device* H2D from ROCm pinned
  memory falls back to a pageable staging memmove (comment at the pool allocation, `moe-expert-cache.cu:1288`).
  On a **single-GPU** arm the copy is same-device, so that constraint does not apply and the D2D-from-alias may
  simply be the slower path (it is the one that costs `pool->base_dev` to be a zero-copy mapping).

Session 2 tested it: `MOE_HOST_POOL_H2D=1` (fork-local experiment, not in `patches/`) switches a pool hit/fill to
`cudaMemcpyHostToDevice` from `pool->base`.  Result in §3.1 (below).  If it recovers a large part of the gap
the fix for the delivery is a per-device rule (H2D when the copying device owns the pool, D2D otherwise), not a
new mode.

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
(§3), F5's boolean measured and its design note re-scoped (§5.1-5.3), and F2 reconciled so the replayer
calibrates (§4).  What is left, in priority order:

1. **The -20.5 % decode gap, and its next suspect is the copy kind** (§5.2): `MOE_HOST_POOL_H2D=1` (a
   fork-local switch, already built) makes a pool hit/fill an H2D copy from the pool's pinned host pointer
   instead of a D2D copy from its mapped UVA alias, which is what the pinned arm does.  If it wins, the
   delivery fix is a per-device rule (H2D when the copying device owns the pool, D2D otherwise) — it needs no
   new mode and it applies to every split type.  Result: §3.1.
2. **`mincore` per-read residency + a page-cache budget** (§5.3), on the arm where it can show: a sub-full
   pool (`MOE_HOST_POOL_FRAC=25`) and/or the 64 GiB cgroup, not the reference arm (`DIO` is already free
   there).  This is also the honest instrument for the maintainer's "re-streamed from disk" report.
3. **P0-a — the remap on pool-backed tables** (`README.md` §3.1): still the +8 % mechanism on 35B-A3B, still
   blocked on freshness (a pool layout epoch in the cold id, or a decode-band layout freeze).  On the 122B the
   remap is invisible (§3), so judge it on 35B-A3B where its ceiling was measured.
4. Then the P1 items in `README.md`: per-table pool sizing from heat mass (§4.2) — now screenable, the
   replayer calibrates — demand precopy (§4.3), and the two standalone #267 borrows: the bounce-ring cap
   (0010) and arena-across-prefill (0003) (§4.4).
5. Only then the Phase-2 items: 2-GPU x8 `-sm tensor` arms, Flash-Next validation, and the `RAM_PCT`/swap soak
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
- **D4 — `wip/host-pool-speed/data/20261102-session2-*/` keeps every arm log and the gzipped calibration trace
  (9.1 MB)** so the numbers above are reproducible without a GPU run.  /tmp is treated as scratch.

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
  - `data/traces/trace-122b-ncmoe48-devremap0.txt.gz` — session 1's 122B routing trace (gzipped, 9.1 MB), the
    F2 calibration input; regenerate with the `README.md` §9.6 capture command at `MOE_EXPERT_CACHE_MIB=4096`,
    `GGML_MOE_DEVREMAP=0`.
- `tools/run-arms-1gpu.sh`, `tools/run-arms-1gpu-h2d.sh` (+ `tools/lib-arms.sh`) — the arm runners (pin
  `HIP_VISIBLE_DEVICES=0`, `-lv 4`, 2000-token decode, disk-read + memory accounting per arm, alternating arms).
- `tools/replayer-calib.py` — the F2 sweep (band x churn x seed x ghost-reset) over one trace, importing the
  delivery replayer.
- PR #267 (not adopted): the submitter's branch `brian/wip-mmap-host-experts-r204`, directory
  `wip/mmap-host-experts-r204/`.
