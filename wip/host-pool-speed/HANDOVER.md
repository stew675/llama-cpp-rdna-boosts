# HANDOVER — `wip/host-pool-speed` (cold start, 2026-11-02)

**Read this first; the campaign charter is in [`README.md`](README.md).**  Nothing here is in `patches/` and
nothing is applied to the fork (`AGENTS.md`, WIP rules).  `README.md` holds the goals, the scope decisions and
the plan; this file holds the state, the measurements taken and the next actions.

## 0. Status, and the ONE thing to do next

Session 1 (2026-11-02) established the baselines on the reference model and produced **five findings**, one of
which blocks everything else:

> **F3 — `--host-experts pool` did not activate on the 122B**: the run reports `host pool (L2): 0 tensors, 0
> slots, 0.0 MiB pinned` and never prints `host_experts_mode=POOL`.  The same flag on 35B-A3B builds a pool
> normally.  Until this is explained, **no pool number on the 122B means anything** (the run silently took the
> per-table pinned bounce path and measured 17.7 t/s against pinned's 22.4).

**Next action, exactly:** re-run the 122B pool arm with `-lv 4` and diff the loader lines against the 35B
probe in §4/F3.  The two candidate causes are (a) the pool-backed master buft was declined for a 67 GiB host
expert set (then the loader falls back to a pinned master with no warning), or (b) the `MOE_HOST_POOL_RAM_PCT`
clamp zeroed every table.  The lines to look for are `host_experts_mode=POOL`, `pool_decide_dio_locked`,
`pool_clamp_slots_to_ram` and `table_pool_locked`.

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

## 3. Measured, 1 GPU, `-ncmoe 48 -ngl 99 -c 8192 -ctk/ctv q8_0 -fa auto -b 2048 -ub 2048`, 5258-token prose
prompt, `-n 2000`, greedy, two alternating passes unless noted (logs `/tmp/base-*.log`, `/tmp/run-*.log`)

| arm | arena | residency | `h=` | prompt | decode |
|---|---|---|---|---|---|
| **`pinned`, shipped default** | 22,517.6 MiB | 33.6 % | 0.7543 | **681.2 t/s** | **22.4 t/s** |
| `pinned`, `GGML_MOE_DEVREMAP=0` | 22,517.6 MiB | 33.6 % | 0.7538 | 684.5 t/s | 22.3 t/s |
| `--host-experts pool` (pool **did not activate**, F3) | 22,517.6 MiB | 33.6 % | 0.7623 | 624.7 t/s | **17.7 t/s** |
| same, `MOE_HOST_POOL_DIO=0` (no pool → no effect) | 22,517.6 MiB | 33.6 % | 0.7623 | 493.8 t/s | 18.2 t/s |
| `MOE_EXPERT_CACHE_MIB=4096` (trace arm) | 3,932.2 MiB | 5.5 % | 0.3488 | 157.2 t/s | 12.9 t/s |

Read-outs: **auto arena = 22.5 GiB** (`alloc_all_locked` sizes from what is free after `--fit`, the graph
reserve and the KV cache, and takes the slab's 23 GiB arena region) — "MIB unset" is *not* small.  Restricting
it to 4 GiB costs **4.3× the prefill and 42 % of the decode**.  The pinned arm holds **65.5 GiB of
non-swappable RAM**.  The remap (DEVREMAP) is worth **nothing measurable at 22 t/s** on this arm — its per-op
sync is buried at 44.7 ms/token (on 35B-A3B at 60 t/s it was +8.4 %), so do not use the 122B to judge it.

## 4. Findings

**F1 — a decode routing trace requires `GGML_MOE_DEVREMAP=0`.**  The `D` records are emitted by the eager
`moe_cache_update_host` hook; the remap never reads the routing back and the scheduler returns before the eager
path.  Shipped default → the trace stops after the two sizing passes (**288 `D` records for 1864 decode steps**);
with the remap off → **268,416 `D` records** (38 MB, 1864 × 144 tables ✓).  `ENVIRONMENT.md`'s
`MOE_EXPERT_TRACE` row and the replayer's docstring document the identity-table caveat but not this one — a
one-line doc fix when the campaign next touches docs.

**F2 — the replayer does not calibrate; this blocks "simulate before implementing".**  Same trace
(`/tmp/trace-122b-ncmoe48-devremap0.txt`), `--devices 1`, `--seed sidecar:<the model's .moe-heatmap>`:

| arena | slots/table | replayer `shipped` | run's `h=` | error |
|---|---|---|---|---|
| 3,932 MiB | 14-16 | 0.19-0.20 (0.2855 with `--churn off`) | 0.349 | **−0.16** |
| 22,518 MiB | 87-92 | 0.642 | 0.754 | **−0.11** |

The delivered placement is the shape the replayer models — a `-lv 4` run shows a **uniform 15 slots/table**
seeded from the persisted map (`apply_heat_seed_locked: ... heat-seeded 15/15 slots from the persisted map
(256 experts, ranked by the DECODE band)`) — so the gap is in the *dynamics*.  `--seed sidecar` reports only
31.4 % of the prior's mass in the seeded slots while the run's `h` is higher than that, i.e. the real cache
loses far less of its seed to eviction than F-LRU-with-`count=0` predicts.  Candidate causes to test: the
seeded-resident eviction order, the `MOE_EXPERT_CACHE_PERIOD`/decay interaction, and whether `access_locked`'s
`t.hits`/`t.misses` count the band-wide union of a multi-token pass differently from the replayer.

**F3 — the pool did not activate on the 122B (BLOCKING).**  `moe_cache_report: host pool (L2): 0 tensors,
0 slots, 0.0 MiB pinned`, no `host_experts_mode=POOL` line, no clamp warning, and `pool_decide_dio_locked` never
prints — i.e. the loader never selected the pool-backed master buft, so the host experts stayed in a **pinned**
master and the run took the per-table pinned bounce path (17.7 t/s vs pinned 22.4).  Control that *works*
(35B-A3B Q4_K_M, `-ncmoe 8`, `--host-experts pool`, `-lv 4`, `/tmp/poolmode-probe.log`):

```
llama_model_loader: host_experts_mode=POOL: NO pinned master; the host experts stay in the model mapping ...
bind_host_dev_locked: the host expert master is pageable (pool-backed `--host-experts pool`); ...
pool_decide_dio_locked: host pool fill mode: buffered (page cache as L3) -- MemAvailable 79172.9 MiB,
                       planned pool 3051.2 MiB, host expert set 3814.0 MiB ...
pool_clamp_slots_to_ram: host pool RAM ceiling: 85% of MemAvailable = 67296.9 MiB (MOE_HOST_POOL_RAM_PCT)
table_pool_locked: host pool (... FRAC of the host expert bytes): 24 host tensors, 3814 MiB ... -> 204/256 slots
```

Relevant code: `llama_model_loader::host_expert_buft()` returns the master buft only when
`host_experts_mode == POOL && use_mmap && !LLAMA_HOST_MASTER_PINNED` **and**
`ggml_backend_dev_moe_host_master_buft(dev) != nullptr` (NULL for integrated devices; the R9700 is discrete, so
it should be offered).  Compare the 122B `-lv 4` run line by line against the probe.

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

## 6. Next steps, in order

1. **F3 first** (§0).  No pool numbers until the pool activates on the 122B.
2. **P0-a — the remap on pool-backed tables** (`README.md` §3.1): the pool arm cannot use
   `GGML_MOE_DEVREMAP` (`devremap_ok_locked` needs `table_cold_ok` = a pinned master), measured as the whole
   remaining decode gap on 35B-A3B (pool 60.4 vs pinned+remap 65.7).  The blocker is freshness — the lagging
   map vs `pool_rerank_from_heat_locked`, whose only two call sites are `alloc_table_locked` and the first
   decode-band pass — so the design is a **pool layout epoch** carried in the cold id (or a decode-band freeze).
   On the 122B this needs to be re-measured at 22 t/s first: F3's table shows the remap is invisible there.
3. **F5's per-read residency + page-cache budget** (§5), with the `DIO` A/B on the 122B pool arm.
4. **F2 — reconcile the replayer** (§4), because §9.6's simulate-before-implement rule depends on it.
5. Then the P1 items in `README.md`: per-table pool sizing from heat mass (§4.2), demand precopy (§4.3), and
   the two standalone #267 borrows — the bounce-ring cap (0010) and arena-across-prefill (0003) (§4.4).
6. Only then the Phase-2 items: 2-GPU x8 `-sm tensor` arms, Flash-Next validation, and the `RAM_PCT`/swap soak
   (which the maintainer places at the very end, as a Flash-Next + PLE effect).

## 7. Traps (all bitten today or previously)

- `-lv` default is **1**; the cache's `h=` report is **INFO** → pass `-lv 3` (or `4` for the sizing lines).
  Never `-lv` when hashing generated text (`prompts/README.md`).
- `llama-cli` in this build is the **server-based** CLI: chat template on, so pass `--reasoning off` for prose
  and `--single-turn --no-display-prompt`.  It reuses the server's `print_timing` / `slot` log lines.
- Always pass `-c`: the model's training context is 262144 and an unset `-c` spends VRAM on context instead of
  arena (KV is cheap here: ~27 KB/token, ~220 MiB at `-c 8192`).
- `MOE_EXPERT_CACHE_MIB=4096` changes the regime by 2×; use it only to make tables traceable.
- Never run arms in parallel; alternate, and keep the `pinned` arm in the same run.
- Pin `HIP_VISIBLE_DEVICES` on every arm (the iGPU trap), and remember the post-pin index remap for the pair.

## 8. Open questions for the maintainer

1. F3: if the pool-backed master buft is being silently declined for a 67 GiB host set, is failing over to
   pinned the behaviour you want, or should it be a hard error (a user who asked for `pool` and got a 65 GiB
   pinned master has been silently moved into a different memory regime)?
2. The cgroup-aware `RAM_PCT` denominator (`README.md` §9.2/§3.2): own small change now, or ride with Phase 2?
3. `TODO` handle for the swap-thrash report (still only recorded in `wip/`).
4. The knob-collapse target (`README.md` §5) and whether `DIO` becomes part of it (§5 here).

## 9. Files

- `README.md` — charter, scope, plan, P0-P3 items, measurement protocol, open questions.
- Campaign data today: `/tmp/base-*.log`, `/tmp/run-trace-1gpu*.log`, `/tmp/poolmode-probe.log`,
  `/tmp/trace-122b-ncmoe48-devremap0.txt` (38 MB routing trace), `/tmp/sizing-122b.log`.  **These are in /tmp
  and will not survive a reboot** — copy anything worth keeping into `benchmarks/` or a campaign directory.
- PR #267 (not adopted): the submitter's branch `brian/wip-mmap-host-experts-r204`, directory
  `wip/mmap-host-experts-r204/`.
