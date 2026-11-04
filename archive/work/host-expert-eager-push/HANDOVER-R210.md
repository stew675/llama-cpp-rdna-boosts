# HANDOVER — finish r210: resolve the `llama-batched-bench -npl 8` decode collapse, then run the release gates

**For a fresh session with no memory of the campaign.**  Everything below is the state as of 2026-11-04, the exact
open question, the measurements that exist, the commands to run, and the traps.  Read
[`P5-RESULTS.md`](P5-RESULTS.md) and [`P6-POOL-REGION.md`](P6-POOL-REGION.md) for the engineering narrative, and
`AGENTS.md` at the repo root for the delivery policy.

**Terminology — read this first, it is the one thing that went wrong repeatedly in the previous session.**
There were *two* different "mirror" ideas and the earlier one was **superseded**:

| name | what it was | status |
|---|---|---|
| "mirror" (patch `0012`, P6 §8) | a hard per-tensor boolean: `slots >= n_experts` → keep every expert's pool copy, else the historical complement | **SUPERSEDED — the boolean no longer exists in the tree** |
| **surplus / expanded pool** (patch `0013`, P6 §10) | ONE rule at every budget: the pool is laid out as **the arena's complement first**, then the **arena-resident experts in heat order while capacity remains** | **this is what ships** |

So when this document says "the pool holds the full weights", it means *the surplus rule filled the pool with the
complement **and** the arena-resident experts because the budget was big enough* — not the old `slots>=n_experts`
switch.  If you `grep -n mirror ggml/src/ggml-cuda/moe-expert-cache.cu` you will find only comments/`"complement +
mirrored residents"` strings, never a `mirror` flag.

---

## 1. State of the world (verify this before touching anything)

| what | where / value |
|---|---|
| fork | `~/llama.cpp`, branch **`promo-block16`** (misleading name: it is the folded 16-block chain, see below), tree clean |
| **delivery** | `~/llama-cpp-rdna-boosts`, HEAD **`64b08688`**, `release.json` = `release: v16-6050b467c-r210`, `n_blocks: 16`, `base: 6050b467c`, **`tip: 489c2b4bc`**, **`tree: d2298e6d`** |
| canonical 16-block chain | rebuilt in a worktree: **`/tmp/canon`** (from `scripts/apply-all.sh`, base `6050b467c`).  It will not survive a reboot — re-create it with `git -C ~/llama.cpp worktree add -f /tmp/canon 6050b467c && (cd /tmp/canon && git branch -D rdna-boosts; bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .)` |
| build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1.0, `build-rocm`); fast loop `cmake --build build-rocm --target llama-cli llama-batched-bench -j 16` |
| hardware | 3x R9700 (gfx1201, 32 GiB).  **devices 0 and 2 are PCIe x8, device 1 is x4** — use `HIP_VISIBLE_DEVICES=0,2` for any pair |
| campaign harness | `wip/host-expert-eager-push/p3-matrix.sh` (env knobs: `GPU`, `SM`, `N`, `EOS`, `NCMOE`, `CACHE_MIB`, `LME`, `POOL_FRAC`, `HOSTMODE`, `MODEL`, `LAZY`, `PROMPT_FILE`, `LV`); `MOE_*`/`GGML_*` env is passed through verbatim |

**The fold is done and proved content-preserving.**  The change lives in **block 09** (host-resident MoE) — *not* a new
block — and blocks 10-15 were re-cut on top.  The proof is that the 16-block tip tree is still **`d2298e6d`**
(`git -C /tmp/canon rev-parse HEAD^{tree}`), and `scripts/validate-set.sh` is GREEN (16 patches, checksums,
fresh-tarball strict `git am`, tree match, no conflict markers, Windows-portability clean).  The fold needed a
**6 file / 16 hunk** merge because **block 12** ("campaign remainder — backend iface fills") and **block 14**
(relaxed-tensor split) touch the same host-expert and loader regions; block 12's own machinery
(`expert_host_src`, the loader's ~382-line host-expert block) was deliberately kept **out** of block 09 so
attribution stays correct.

**Working tree caveat:** `~/llama.cpp`'s `build-rocm` binary is built from the `promo-block16` content, which is
tree `d2298e6d` — i.e. byte-identical to the delivery.  Any rebuild of `d2298e6d` must reproduce every number here.

---

## 2. What is finished and green

* **Delivery**: 16 blocks, `v16-6050b467c-r210`, tree `d2298e6d`, `validate-set.sh` green.
* **Docs**: `ENVIRONMENT.md` (+`MOE_HOST_LIMIT_FRAC`, `MOE_HOST_TIER`, `MOE_HOST_POOL_COLD_MULTIDEV`),
  `INVARIANTS.md` (8 x r210 entries), `WORKLOG.md` **261104.001** (newest first; says "block 09 amended", 16
  blocks, `v16-6050b467c-r210`, tip `489c2b4bc`, tree `d2298e6d`), `patches/README.md` (block-09 amendment note).
* **Coherence gate** (on the final tree): 2 devices **1318 prefill / 69.8 decode**, `sha=231512fc77e5`; 1 device
  **52.2**, `sha=4ae33f7bd65c`; `scripts/check-gpu-faults.sh` clean everywhere.
* **Behaviour** (2 devices, 35B-A3B Q4_K_M `-ncmoe 40`, `-n 300`, prose 5246 for prefill, byte-identical text):

  | arm | prefill | decode | pool census |
  |---|---|---|---|
  | `-lme auto` (fit → surplus/expanded pool) | **1318** | 69.8 | `h=1.0000`, 30 960 fills = prepopulate only, 0 evictions, 18 662 MiB |
  | `--host-experts pinned` (whole-tensor `hipHostMalloc` masters) | 1105.4 | **71.9** | no pool |
  | `-lme 16` (86 % of the set) | — | 66.5 | `h=0.8368`, 16 329 MiB |
  | `-lme 8` (43 %, a genuine sub-complement budget) | — | 54.1 | `h=0.3631`, 8 165 MiB |

---

## 3. THE OPEN BLOCKER — `llama-batched-bench -npl 8` decode collapses

### 3.1 What was measured (reproduced twice, on tree `d2298e6d`)

Command:

```bash
cd ~/llama.cpp && env HIP_VISIBLE_DEVICES=0,2 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib \
  ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_CACHE_MIB=2048 \
  ./build-rocm/bin/llama-batched-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    -ngl 99 -sm tensor -ncmoe 40 -c 16384 -npp 512 -ntg 128 -npl 1,4,8
```

| npl | pp t/s | tg t/s | note |
|---|---|---|---|
| 1 | 130 | 73.3 | pp 130 is a cold first-pass artefact (see 3.3) |
| 4 | 1762.8 | 191.3 | |
| **8** | 649.7 | **49.7** | **reproduced: a second run of `-npl 8` alone gave tg 49.72 / pp 649.69** |

The comparison point: the **pre-r210** tree (delivery tip `27271aa91`) measured **pp 1243/1243/1263, tg
74.9/192.2/269.2** for `-npl 1,4,8` with the same command.  So at 8 parallel slots r210's tg is **~50 against ~269**,
i.e. a ~5x collapse in the multi-slot decode band — while every single-sequence gate (1 and 2 devices,
`-n 300`/`-n 2000`) is green and *better* than before.

**Do not assume the regression is real until you have validated the instrument (3.3).**  `llama-batched-bench` has
been unreliable in this repo before, and the previous session's own notes say so.

### 3.1a The matrix is ALREADY RUN — the instrument is fine and the cause is the POOL TIER, not the expansion

Same box, same command, same moment, `-npp 512 -ntg 128 -npl 8`:

| arm | pp t/s | tg t/s | total t/s |
|---|---|---|---|
| **old tree** `27271aa91` (pre-r210, default = pinned masters) | 1094.4 | **255.9** | 661.1 |
| r210 `--host-experts pinned` (whole-tensor `hipHostMalloc` masters) | 1096.8 | **260.9** | 668.5 |
| r210 `auto` (expanded pool, surplus rule) | 637.2 | **50.8** | 192.6 |
| r210 bounded pool `MOE_HOST_LIMIT_FRAC=1` (FRAC 50, ~9.3 GiB) | 356.7 | **50.1** | 160.4 |
| r210 full coverage, pageable master, explicit 18 662 MiB pool | 357.2 | **50.1** | 160.4 |

**Conclusions, all measured — do not re-derive them:**

1. **The instrument is fine.** The old tree on today's box reproduces its old ~256 tg exactly.
2. **The regression is the POOL tier, at every coverage.** Both the bounded complement and the full-coverage pool
   collapse to ~50 tg; the whole-tensor pinned-master tier is unchanged from the old tree (260.9 vs 255.9).  So the
   expansion/surplus rule (P6 §10) is **not** the cause — a 43 %-coverage bounded pool is just as bad.
3. **The mechanism is the pool-backed PRE-FILL STAGING, not the fill transport.** Prefill collapses 3x as well
   (1096.8 → 356-637), which is `moe_cache_stage_table`'s two-tier/per-expert path for a table whose master is the
   pageable model mapping, against the pinned master's **single-launch** staging.  With many short per-sequence
   ubatches (`-npp 512` over 8 sequences) that per-op staging cost dominates the whole run, which is why the single-
   sequence `llama-cli` gates (one wide 5246-token prefill) show the opposite ordering (mirror pool 1318 vs pinned
   1105 — the pool wins there because the two-tier path amortises over a wide ubatch).
4. Note the *shape* difference from `llama-cli`: r209's default was the **pinned-master** tier, so block 09's
   shipped numbers were never exposed to the pool on this workload.  r210 changes the default, so the multi-slot
   band regressed *by policy*, not by a bug in the surplus rule.

**Therefore the fix is a policy/engineering choice, in this order of preference:**

* **(a) Make the pool's staging single-launch when the pool covers the whole table.**  The pool is a contiguous
  per-tensor pinned region, so a fully resident table can be staged exactly like a pinned master — this is the
  "a 100 %-covered pool becomes the master" step already recorded as *not attempted* in `P6-POOL-REGION.md` §5, and
  it now has a concrete motive.  The decision points are `moe_cache_stage_table()` and the `t.host_pinned`
  derivation in `ggml/src/ggml-cuda/moe-expert-cache.cu` (see also `bind_host_dev_locked`); do it with a
  `region`/`region_dev` field on the table, **never** by rebinding `t.host` (it keys `g_host_pools`/`g_host_src`
  and is shared by every per-device table of one tensor).
* **(b) Cheap and safe: keep the multi-slot band on the pinned tier.**  Make the *default* the pinned-master tier
  again and reach for the pool only where it wins (one wide prefill, or an explicit `-lme N` / `-lme pool`).  That
  is a one-line policy change in `plan_host_expert_placement()` (`src/llama-model-loader.cpp`, the
  `pinned_masters` / budget branch), and it restores 256 tg at `-npl 8` while giving up the +19 % prefill.
* **(c) Do not** chase the pool's fill transport: the fill census at 2 devices was already at parity and the
  read-ahead experiment was measured as a loss (§9 of `P6-POOL-REGION.md`).

Whichever is chosen, re-run this exact table (it is the acceptance bar) plus the single-sequence arms, then re-cut
and re-release.

### 3.2 Hypotheses, in the order worth testing (kept for completeness — mostly settled by 3.1a)

1. **The expanded pool has no spare capacity under a large per-step expert set.**  At `-lme auto` the pool is
   `slots == n_experts` per tensor (100 %, 18 662 MiB) — i.e. the *full* set, complement **and** the arena-resident
   experts.  With 8 concurrent sequences one decode step routes vastly more distinct experts, so the VRAM arena
   churns hard; in the expanded layout `pool_free_slot_locked()` finds **no free slot** (every slot is a valid
   resident), so a re-admission cannot be homed in the pool and falls through to the **bounce ring** — the
   synchronous 288 KiB page-cache `memcpy` + H2D that P6 §8/§9 measured as the worst of both worlds.  Under the
   *complement* layout there are always free slots (the arena-resident experts' slots are free), so a churn-heavy
   batch has somewhere to go.  This is the single most likely mechanism and it also explains why the single-sequence
   gates are clean: they churn far less.
2. **`llama-batched-bench` itself** (3.3) — including that its `-npl 8` arm may have been measuring something
   different in the old run (e.g. a different default arena size, a warm/cold page cache, or an arena that fell
   back to `cudaMalloc`).
3. **A different code path at high batch**: the prefill/verify band (`n_tok > MOE_EXPERT_CACHE_MAX_TOK`) changes the
   staging path, and `GGML_SCHED_DEVGATHER` is on by default for the prefill band for a pool-backed table — see
   `TODO` #131/#136.  `-npl 8` with `-npp 512` alternates prompt and generation batches, so the band machinery
   (slab narrow/wide, staging ring) is exercised far more than in `llama-cli`.
4. **Memory pressure on the host**: expanded pool 18 662 MiB pinned **plus** the 18.6 GiB page-cache copy of the same
   bytes plus per-table bounce rings.  If the box is near its limit, the mapping's re-reads go to disk.  Check
   `MemAvailable` during the run (`wip/host-expert-eager-push/p3-mem.sh`, `p3-busy.sh`).

### 3.3 Validate the instrument FIRST (`llama-batched-bench` has lied here before)

1. **Re-measure the old tree with the same binary/command**: build `27271aa91` (delivery tip, pre-r210) in a second
   build dir or worktree and run the *same* batched-bench command.  If it also gives ~50 t/s now, the "regression"
   is the instrument/box state (page cache, thermal, `MemAvailable`), not r210.  **This is the cheapest decisive
   test and it should be the first thing you run.**
2. **Cross-check with an arm the tool cannot get wrong**: `-npl 8` at npl 1 must equal `llama-cli -n 128` within a
   few percent, and the *all-VRAM* control (`-ncmoe 0`, experts in VRAM) must be unchanged between the two trees.
   If the control moves too, the harness/box is the variable.
3. **Check the census, don't trust throughput**: run the `-npl 8` arm with `LV=4`/`-lv 4` and read
   `moe_cache_report`:
   * `pool fills: ... mapping memcpys N ms` — a large number means page-cache re-staging (hypothesis 1);
   * `host pool (L2): ... h=<hit rate>` — an expanded pool at 100 % should report `h≈1.0` and **0 evictions**;
     a churn-heavy batch that bounces will show up as `expert access: fills=… colds=…` shifting toward fills and
     `fills` bytes far above the prepopulate;
   * `MOE_HOST_PUSH_PROBE=1` prints the fill-copy split (`H2D` = bounce, `D2D` = from the pool).  **A large H2D count
     at `-npl 8` confirms hypothesis 1 directly.**
4. Remember the census token accounting is weird (`over ~3 tokens`): trust the *ratios* and the byte counters, not
   the implied token count.

### 3.4 The attribution matrix (run serially, never two arms at once)

`HOSTMODE` must be on the **script invocation line**, never after the tag (trap 6.1 of `IMPLEMENTATION-PLAN.md` —
it silently made every early "pinned" run a pool run).

```bash
cd wip/host-expert-eager-push
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
bench() { env HIP_VISIBLE_DEVICES=0,2 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib ROCBLAS_USE_HIPBLASLT=0 \
  MOE_EXPERT_CACHE_MIB=2048 "$@" ~/llama.cpp/build-rocm/bin/llama-batched-bench -m $M \
  -ngl 99 -sm tensor -ncmoe 40 -c 16384 -npp 512 -ntg 128 -npl 1,4,8; }

bench                                   # A: auto -> expanded pool (the reported 49.7)
bench MOE_HOST_TIER=tensor              # B: whole-tensor pinned masters
bench MOE_HOST_POOL_MIB=18662 MOE_HOST_LIMIT_FRAC=1   # C: pool (pageable masters), full coverage, explicit budget
bench MOE_HOST_POOL_FRAC=50  MOE_HOST_LIMIT_FRAC=1    # D: BOUNDED complement, ~9.3 GiB (hypothesis 1's control)
bench MOE_EXPERT_CACHE_MIB=0            # E: cache off — the pure path, same in both trees
```

| outcome | reading | fix |
|---|---|---|
| B ≈ 269, A ≈ 50 | the expanded pool is the cause | make the expansion **concurrency-aware**: leave headroom (or fall back to the complement layout) when the routed set per step exceeds what the arena can hold.  The decision point is `pool_rerank_from_heat_locked()` (~line 2049) and `fill_slot_from_host()`'s `keep_all` (~line 2404) in `ggml/src/ggml-cuda/moe-expert-cache.cu`; a clean rule is *expand only when `slots > |complement| + (arena churn headroom)`*, or gate the expansion on the measured band (`g_band_ops`/`n_tok`). |
| D ≈ 269, C ≈ 50 | full coverage is fine, so it is not "no free slots" but the *pageable master* | then the interaction is the pool's staging branch at high batch; check `moe_cache_stage_table` and `GGML_SCHED_DEVGATHER`. |
| B ≈ D ≈ C ≈ 50 | not the tier at all | bisect the block-09 amendment (below) or suspect the instrument. |
| all arms ≈ 50 **and** old tree ≈ 50 | the instrument/box | `llama-batched-bench` is unreliable; use `llama-cli` with `-np`/`-b` arms or `llama-server` with N slots as the gate instead, and record why in `MANIFESTS.md`. |

If it turns out to be r210, bisect by rebuilding the block-09 *pre-amendment* commit (`3abfce47a`) and the folded one
(`f4e1d4180` in the pre-fold worktree, or simply `--host-experts pinned`, which bypasses the whole tier).

### 3.5 If a fix is needed, it must not regress the single-sequence win

The acceptance bar for any fix: **2-device `-lme auto` keeps prefill ≥ 1300 and decode ≥ 69 at `-n 300`**, the
`-npl 8` arm recovers to the old tree's ~269, byte-identical text in every arm, `check-gpu-faults` clean, and
`validate-set.sh` green with **tip tree still `d2298e6d` or a *new* tree that `make-release.sh` records** (if content
changes, the tree changes and that is fine — but then re-run the whole gate list and re-cut).

---

## 4. Then finish the release (in this order)

1. **MTP four-axis protocol** — `benchmarks/mtp-adaptive-methodology.md`.  Mandatory before shipping any
   decode/fusion/host-expert change.  Not yet run for r210.
2. **`scripts/gate-per-block-build.sh --changed-from 27271aa91`** — one clean build per block commit (block 09 is the
   one that changed, but the ladder builds every block).  ~20-30 min with ccache; `--fast` for the release profile.
3. **`scripts/validate-set.sh`** (already green; re-run after any change).
4. **`scripts/fork-refresh.sh --build --gates --push`** — re-cuts the personal fork's `rdna-boosts` branch at
   `release.json.base`, verifies tree **and** tip against `release.json`, and pushes **only** to
   `git@github.com:stew675/llama.cpp.git`.  Never push to `ggml-org/llama.cpp`.  This also needs the `/tmp/canon`
   chain to be the same content — regenerate from `apply-all.sh` if in doubt.
5. **Container images** — off-host: the worker watcher (`scripts/container-watch.sh` on `radiance`) polls for the new
   tag and runs `scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune`.  **Do not** run the two-line image
   build on the primary box during a release.
6. **Tag** `v16-6050b467c-r210` (the `v16` is required: 16 blocks, no new block — the maintainer's call).  `release.json`
   already carries that name; if content changes after step 1-3, bump the `rNN` (e.g. `r211`) via
   `scripts/make-release.sh --release v16-6050b467c-r211`.

---

## 5. Traps that cost this campaign real time

1. **`HOSTMODE` on the invocation line** (`HOSTMODE=pinned ./p3-matrix.sh …`), never in `"$@"` after the tag.
2. **ggml's `GGML_LOG_WARN`/`INFO` is filtered at llama-cli's default level** — user-facing warnings use
   `fprintf(stderr, …)`; the census needs `LV=4`/`-lv 4`.
3. **A `-lv 4` run has no usable text hash** (stats interleave into the generated text).  Hash from a clean run,
   census from an `LV=4` run — `prompts/README.md` + `scripts/extract-generated.py`.
4. **`--ignore-eos -n 300` for throughput** (an EOS-stopped 150-token answer averages in the cold head), and use
   `-n 2000` when comparing *steady-state* rates: both arms warm up, and at `-n 300` the pool still shows a ~2 %
   deficit that is gone by `-n 2000`.
5. **Serial runs only**; pin `HIP_VISIBLE_DEVICES`; never `0,1` for a pair (device 1 is x4); a 3-device `-sm tensor`
   split is `TODO` #135's degenerate (mirrored) shape, warned at sizing.
6. **The heat prior is a persisted sidecar** (`<model>.moe-heatmap`).  It *evolves* between runs; `chmod 444` it to
   freeze both arms onto the same prior when A/B-ing placement (that is how the P6 numbers were taken).
7. **A worker thread must not be the first to touch the ROCm runtime** (its lazily created `AsyncEventsLoop` aborts
   the process: `std::system_error: Operation not permitted`).  The read-ahead experiment hit this; the code was
   **dropped** (`a7dfe8871`, reverted) — do not resurrect it.
8. **`-ncmoe`'s tail bound is `hparams.n_layer()`, not `n_layer_all`** (the latter counts the nextn/MTP head).
9. **`llama-batched-bench` needs its own binary rebuilt** when `llama_model_params` changes — a stale tool segfaults
   at startup with no message (cost 10 minutes in the previous session).  Build `llama-cli` **and**
   `llama-batched-bench` together.

---

## 6. Where the knowledge lives

| topic | file |
|---|---|
| the tier, its measurements, the surplus rule | [`P5-RESULTS.md`](P5-RESULTS.md), [`P6-POOL-REGION.md`](P6-POOL-REGION.md) (§8 complement-vs-mirror, §9 the dropped read-ahead, §10 the surplus rule) |
| root cause of the `-sm tensor` loss (pageable staging) | [`P4-ROOT-CAUSE.md`](P4-ROOT-CAUSE.md) |
| the ruling-out list + traps | [`HANDOVER.md`](HANDOVER.md), `P3-MULTIGPU-RESULTS.md` |
| the executable plan (steps a-e) | [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md) |
| "do not re-derive" facts | `INVARIANTS.md` (r210 block), `TODO.md` #139/#140 |
| delivery contract | `MANIFESTS.md` (gates), `BASELINE.md` (provenance), `release.json` (source of truth) |
| every env var | `ENVIRONMENT.md` |
| the campaign patches (not in the delivery) | `patches/0001`-`0013` in this directory — `0010`-`0013` are the ones now folded into block 09 |

**Nothing is pushed and no tag exists.**  The delivery at `64b08688` is self-consistent (16 blocks, tree `d2298e6d`,
`validate-set.sh` green) and fully revertible; the only reason not to tag is §3.
