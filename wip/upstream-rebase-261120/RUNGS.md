# The ladder — 18 rungs from `4f9dfa0e3` onto `919794500`

One rung per delivery block, in release order.  Each rung is **one `git cherry-pick`** of the block commit
onto `rebase-r328` in `~/llama.cpp`, its conflicts resolved by hand, then `~/bin/build-llama-rocm`, then the
coherence run, then the rung's log line written **here** before the next rung starts.

Method, exit criteria and traps: [`README.md`](README.md) §1.  Decision points: [`DECISIONS.md`](DECISIONS.md).
Conflict evidence per rung: [`CONFLICT-MAP.md`](CONFLICT-MAP.md).

**Reference points for a rung**

| what | value |
|---|---|
| ladder branch | `rebase-r328` in `~/llama.cpp`, created at `919794500` (never commit to `master`) |
| r327 binaries (the shipping reference) | `~/llama.cpp/build-rocm/bin-r327` |
| pristine-base oracle | `~/llama.cpp/build-base-novmm/bin/llama-cli` (`GGML_HIP_NO_VMM=ON`, i.e. the base **without** the VMM pool) |
| pristine-base text hash | **`100 chars sha=1c5d32ac537d`** (`--seed 42 --temp 0 -n 20`, `Prompt 247.7 / Gen 92.2 t/s`) — the smoke reference for a rung that touches no numerics |
| the ladder's own build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` (`build-rocm`, VMM **OFF**, ROCm 10.1.0) |
| the base blocker | [`BASE-BLOCKER.md`](BASE-BLOCKER.md) — R00 must carry the `ac931b5bc` revert or no rung can be verified |

**Universal per-rung exit criteria**

1. Conflicts resolved by hand; **every** resolution written into the rung's log below.
2. `git grep -n '<<<<<<<'` empty; no `TODO(rebase)` left in the code (list it in the log instead).
3. `~/bin/build-llama-rocm` green, no new warnings in touch-path files.
4. Coherence: `HIP_VISIBLE_DEVICES=0 llama-cli -m Qwen3.5-4B-Q8_0 -ngl 99 -sm layer -p "The capital of
   France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn`, hashed with
   `scripts/extract-generated.py`; plus the block's own census (`-lv 4`) or kill-switch A/B where it has one.
5. Rung commit id recorded.

## Ledger

| rung | blk | source commit | conflicts | decisions | status |
|---|---|---|---|---|---|
| R00 | 00 | `de99fa2d0` | 4 (7 regions) | D1 (take upstream's), D2 (carry), D15 (top-k fix in 00) | **DONE 2026-11-20** — `4cba0ad32` (tree `a3824ea0e`) + repair `3ea8b0212` (tree `e12a34bad`); coherence `1c5d32ac537d`; MUL_MAT_ID 1078/0 (16 dup); Flash-Next prefill 1692 t/s after the repair; MMID_512 A/B +0.7 % (text identical) |
| R01 | 01 | `a94e2214c` | 0 | — | **DONE 2026-11-20** — `14a0414bb`, tree `6566ca3ac`; one rung repair (HIP's 64-bit warp mask); 3-device tensor text `1c5d32ac537d` == R00; pipeline census 3 GPUs |
| R02 | 02 | `cef19b032` | 5 | D5 (**resolved: MERGE**, no code dropped) | **DONE 2026-11-20** — `3941bad4e`, tree `0799ebfad`; all 5 conflicts are unions; adaptive arm 29.3 -> 65.2 t/s with identical text; enum count 13 -> 14 |
| R03 | 03 | `36da15029` | 1 | — | **DONE 2026-11-20** — `e612388ba`, tree `971953b99`; test-file union; GATED_DELTA_NET 76/76; chunked prefill +8.8 % with identical text |
| R04 | 04 | `cfe40b5f9` | 2 | **D3 (resolved: union + drop upstream's guard)** | **DONE 2026-11-20** — `768d27561`, tree `0866d4229`; 7 rope arms; fusion worth +5.8 % pp / +16.5 % tg at bf16 KV, identical text; ROPE 515/515 || R05 | 05 | `7362b5958` | 7 (22 regions) | **D16 — decided: the delivery's iteration model wins** | **DONE 2026-11-20** — `1c9bfa37c`, tree `13683655d`; FA op matrix 7428/0; native q8_0 K/V +5.6 % decode; rotation not carried, verified absent |
| R06 | 06 | `f0564df7d` | 2 (2 regions) | D11 | **DONE 2026-11-21** — `ee377ec45`, tree `b5dc56996`; both conflicts are unions (upstream's few-row MMQ keeps precedence, MMB follows; `expert_bounds` re-ported onto the R00 store helper); coherence `1c5d32ac537d`; MUL_MAT_ID 1078/0 (16 dup); MMB default +0.33 % pp2048 / +0.62 % pp16384, MMF32 +0.9 %/+1.0 %; **finding: block 06's MMB cache/slot buffers are pool-unsafe until block 10's `pool_stack` (R06.2)** |
| R07 | 07 | `f5bdbefda` | 1 (2 regions) | D11 | **DONE 2026-11-21** — `1435e7873`, tree `d3b004581`; both conflicts in `mmvq.cu` are unions (enum `ORIN` + `NAVI21`, `calc_nwarps` both special cases); upstream's RDNA3_5 `ne11` gate adopted (dispatch threshold, not an alternative to our VDR tables, unreachable on gfx1201); coherence `1c5d32ac537d`; MUL_MAT (q6_K/q8_0/q1_0/iq4_xs) 335/0; 27B Q6_K pp2048 1154.5 — the delivery's `MMQ_J_MAX` cap only hurts (64 → 977.5), the grid pad is +1.4 % with identical text |
| R08 | 08 | `fe283f27c` | 8 (13 regions) | **D4 (resolved: MERGE, no precedence change)**, D3, D11 | **DONE 2026-11-21** — `510769147`, tree `2ed6cb541`; repair list in R08.3 (5 raw warp-mask sites in block 08's own code, the `k_set_rows_q8_0_group` F32-source gate, a `force_tiling`→`has_gate` template-slot artifact, a lost `#if` guard); coherence `1c5d32ac537d` on the default and on all three kill switches; NORM/RMS_NORM/GROUP_NORM/SET_ROWS 368/0; MUL_MAT_ID 1079/0 (16 dup); fusion off costs −5.4 % pp2048 / −18.7 % tg128 with identical text; `DISABLE_NORM_ROWS=1` is **inert** here (its dispatch is block 10's) |
| R09 | 09 | `123d53419` | 4 (7 regions) | D8, D9 (both MERGE) | **DONE 2026-11-21** — `24cecfb51`, tree `b05ee8543`; all four conflicts resolved as merges (the two extracted functions moved to `mmvq.cu` with upstream's two refinements ported in, `seq_rm` adopted, `split_has_nodes` **not** taken — see R09.3); build green, coherence `1c5d32ac537d`, MUL_MAT_ID 1079/0; **finding: the arena-off arm diverges in text, and the cause is block 09's own host-offload relaxation** (control `GGML_MOE_HOST_OFFLOAD=0` restores identity; r327 agrees) |
| R10 | 10 | `7bbf540d5` | 7 (4+3+2+1 regions) | D8, D12 + the four carried obligations | **DONE 2026-11-21** — `f5a090a31`, tree `394e0b708`; the dispatcher reconciled into **one** TU (upstream's two `#32454` NORM+MUL arms ported in = D4's arm), `top_k_gather_rows` de-duplicated by construction; **R06.2 dense half CLOSED** (27B IQ3_S prefills, `ebf2a5a9b00a` = r327) and **R09.3 CLOSED** (arena == arena-off now); a `needs_sync`-vs-dispatch mismatch my own union created was caught by the op test and fixed; **new finding**: the arena arm's decode fell 54.4 -> 20.4 t/s and no switch recovers it (owned by blocks 14/15) |
| R11 | 11 | `61168c60b` | 5 | D9, D13 (both MERGE, carried) | **DONE 2026-11-21** — `dbe4fdd40`, tree `2b6aec1f0`; op count 100+6=106 (rpc patch 7), `has_recr && n_pos == 1` union, qwen4exp union carries upstream's FP8 PLE scale *and* the lazy reader, 4th 64-bit-mask recurrence (fattn-qsa/indexer-score/hc-mix); coherence `1c5d32ac537d`, quant-coherence PASS (Q4_K_M gather transparent, no `////`); **`gate-hc-fusions.sh` RED** — 3 numeric arms left their anchors (see R11.2) |
| R12 | 12 | `dad1442b3` | 10 | D5, D6, D7 | **PAUSED 2026-11-21** — ten conflicts resolved and captured in `r12-wip/r12-resolved-but-unbuildable.diff`; **blocked on `common_speculative_impl_draft_dflash`**, which is two implementations of one class (upstream's `dflash_options`/`init()` refactor vs the delivery's inline ctor + DSpark/DFlash2/`liliCorr`/ingest state).  Ladder reverted to `dbe4fdd40` (R11) — a rung must build, and this one did not.  See `r12-wip/README.md` |
| R13 | 13 | `1014e6bf5` | 0 | — | not started |
| R14 | 14 | `e88d45d9e` | 3 | — | not started |
| R15 | 15 | `35a9ae985` | 0 | — | not started |
| R16 | 16 | `286191b31` | 0 | D7 | not started |
| R17 | T | `e159adfc2` | 0 | D10, **`TODO` #351 (the flint `-sm tensor` re-test)** | not started |
| R18 | — | release | — | **`TODO` #352 (container `web` stage)** | not started |

---

# R00 — block 00, foundation  ·  **DONE 2026-11-20** — `4cba0ad32`, tree `a3824ea0e`

```
git cherry-pick de99fa2d0        # 37 files, +1145/-358  ->  lands as 34 files, +916/-260
```

| exit criterion | result |
|---|---|
| conflicts resolved by hand | 4 files / 7 conflict regions (below), plus the shader |
| no markers left | yes (`git grep '<<<<<<<'` clean; the only hit is upstream's own ASCII art in `vendor/miniaudio`) |
| build | green, `~/bin/build-llama-rocm` (`all` targets), **zero** `error:` lines |
| coherence 4B | **`100 chars sha=1c5d32ac537d`**, `Prompt 242.0 / Generation 90.7 t/s` — identical to the pristine-base oracle |
| the base blocker | **gone on the delivery build**: the GPU arm runs with `GGML_HIP_NO_VMM=OFF` (the unpatched base hangs; `BASE-BLOCKER.md`) |
| `test-backend-ops -o MUL_MAT_ID` | **1078 OK / 0 FAIL**, of which **16 are `dup=1`** — upstream's path handles duplicates |
| Vulkan shader | merged and **compiled with `glslc` in 5 variants** (f32 / f16 / bf16 / f32+FA_QUANTS / f16+FA_QUANTS), all rc=0 |
| known-red | **one**, recorded below (R00.6): upstream's qwen4exp indexer top-k, which **block 10 already fixes** |
| deferred | the `mm_ids_helper_512_10` A/B (R00.6 gives the reason) |

## R00.0 The four resolutions

| path | ours | upstream | resolution |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | deleted a `TODO` comment in the `MUL_MAT_ID` host fallback | kept it, with the note that removing the `break` costs ~10 % | **upstream's** (their note about their own path; `#32505` then made it moot) |
| `ggml/src/ggml-cuda/mmf.cuh` | a 3-line comment explaining why every `MUL_MAT_ID` takes the compact-ids kernel | restructured the same kernel (`#32505`) | **upstream's** — the file is now **byte-identical to upstream**: the comment's claim became upstream's reality, and `ggml.h`'s `[TAG_MUL_MAT_ID_DUPLICATE_ID]` caveat is gone.  Recorded because it means **this block no longer touches `mmf.cuh` at all** |
| `ggml/src/ggml-cuda/mmid.cu` | the whole dup-ids port (`mm_ids_write_tile_spec`, no shared store) | `#32505`'s `mm_ids_helper_store` + `check_experts`/`dump_store` + `ne_store` | **upstream's file**, with **our `mm_ids_helper_512_10` re-ported** (R00.3) |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp` | `col_live[Bc]` per-column liveness | `#31323`'s `PV_IN_SFSH` row-tiled PV staging + `pvsh_load` | **both** — independent prologue declarations; `col_live` never touches `pvsh` (verified: every `pvsh` use is upstream's and goes through the accessor) |

## R00.3 `mmid.cu`: what the re-port actually is

Upstream's `#32505` design: each expert block fills a shared-memory `mm_ids_helper_store` and flushes it with
`dump_store`, then thread 0 writes that expert's `tile_spec`.  The delivery's fast path is orthogonal — it
replaces **the whole 512-block launch** for one shape — so the port is:

1. **Extracted** upstream's inline tile-spec emission (the `col_low`/`col_high` while-loop plus the
   between-the-seams entry) into `mm_ids_write_tile_spec(tile_spec, expert, nex_prev, n_dumped, tile_width,
   n_tokens, n_expert_used)`, called by `mm_ids_helper` at the point where it used to be inline.  One helper,
   two callers, no duplicated seam logic.
2. **`mm_ids_helper_512_10`** — unchanged algorithm (per-(token,iex) bucket sort with `counts`/`cursors` in
   shared memory, `__syncthreads()` between phases), now writing `tile_spec` through the extracted helper with
   `nex_prev = counts[e]`, `n_dumped = cursors[e]`.  `__launch_bounds__(1024)` added.
3. **Dispatch** at the top of `ggml_cuda_launch_mm_ids_helper`: RDNA3_5/RDNA4 + `n_experts == 512` +
   `n_expert_used == 10` + `GGML_CUDA_DISABLE_MMID_512` unset, before the `switch (n_expert_used)`.

`ids_dst`/`ids_src1` semantics are identical to upstream's `dump_store` (`it*n_expert_used + iex_used` and
`it*sis1 + iex_used % nchannels_y`), which is what makes the shared writer safe.

## R00.5 The `#30542` revert (D2)

Re-applied mechanically (upstream's 13 `ggml-cuda.cu` commits do not touch the pool) and **verified
behaviourally**, which is the point: the unpatched base cannot complete a GPU decode on this box, and the
R00 build runs the 4B arm at 242/90.7 t/s with VMM **OFF**.  `ggml_cuda_pool_vmm` here is upstream's plus the
revert, and *not* the delivery's later pool work (`g_pool_mapped`, `ggml_cuda_pool_trace*`), which belongs to
blocks 09/15 and is therefore absent at this rung as expected.

## R00.6 RESOLVED at R00: the VMM pool refuses upstream's strided 2-D top-k copy

**Symptom (before the repair).** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (2 devices, `-sm tensor`, `-c 8192
-ub 1024`) aborted once prefill started:

```
ROCm error: invalid argument
  current device: 0, in function top_k_argsort_cuda at ggml/src/ggml-cuda/top-k.cu:313
  hipMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows, ...)
```

**Attribution: it is the delivery's required allocator, not any delivery source change.**  A build of the
**pristine base with nothing but the `#30542` revert** — zero delivery code — aborts identically:

| build | delivery code | VMM pool | result |
|---|---|---|---|
| `build-base-novmm` (pristine base, `GGML_HIP_NO_VMM=ON`) | none | no | prefill **1594.3 t/s** |
| `build-revert-only` (pristine base + **only** the `#30542` revert) | **none** | yes | **ABORT** |
| `build-r00-novmm` (R00, `GGML_HIP_NO_VMM=ON`) | block 00 | no | prefill **1683.2 t/s** |
| R00 before the repair (`build-rocm`) | block 00 | yes | **ABORT** |
| r327 (`bin-r327`) | all 18 | yes | prefill 2101.1 t/s — block 10 already fixed the copy |

The refused call is a **strided D2D 2-D copy with `spitch != dpitch` into a VMM-backed destination**; the
equal-pitch call (`ncols == k`) just before it succeeds, and the instrumented run shows
`err_after_sort=no error`, so the argsort was fine and only the copy was refused.  A standalone HIP repro of
the pool's exact layout does **not** reproduce it, so the remaining detail is a ROCm-mapping question
(mapping reuse / `hipMemSetAccess` history) — recorded as such, not as a delivery defect.  The same refusal
class is already on the record twice: `ggml_cuda_cpy_as_memcpy_2d`'s guard, and the windowed KV's 2-D state
copy (issue #313, block 14).

**Decision (maintainer, 2026-11-20): put the fix in the first block that needs it — block 00 — not in block
10.**  Block 00's own revert is what enables the pool, so block 00 is the block that needs the workaround.
Rejected alternatives: leaving it red to R10 (loses the field-model oracle for R02-R09 and the `MMID_512`
A/B); a `#define`'d minimal fix in 00 that block 10 deletes (a switch that exists only to be removed, and
more code than the kernel it replaces); and moving the **revert** to block 09 to dodge it (that would build
R00-R08 on a different allocator than the release — the r278 trap — and we now know the allocator changes op
behaviour, which is exactly what a rung must not hide).

**The repair (`3ea8b0212`)** adopts block 10's `top_k_gather_rows` / `top_k_gather_cuda` **verbatim** and
calls it from `top_k_argsort_cuda`.  Verification:

| arm | result |
|---|---|
| the red arm: Flash-Next IQ3_S, 2 devices, `-sm tensor`, 5298-token prompt | **prefill 1692.5 t/s**, no abort |
| 4B coherence (the sampler's top-k runs through the same gather) | `100 chars sha=1c5d32ac537d` — **unchanged** |
| `mm_ids_helper_512_10` A/B, now reachable | **ON 1642.7 / 1638.7 vs OFF 1635.4 / 1624.2 t/s (+0.7 %)**, text identical `52 chars sha=b3de97c70c88` |

**Obligations this creates.**
1. **R10:** when block 10 is cherry-picked, **drop its `top_k_gather_rows` hunk** — the tree already has it
   (keep-local on that file's conflict).  Block 10 keeps its `ggml_cuda_cpy_as_memcpy_2d` guard, which is a
   different site.
2. **R18:** squash `3ea8b0212` into block 00 (the r278 rung-repair pattern).
3. **R18, measurement:** the `MMID_512` claim of record is **+4.5 % pp16384 / +5.2 % pp2048**; at this base
   it reads **+0.7 %** on a 5298-token prompt at 2 devices (2 reps per arm, text identical).  Re-measure with
   the delivery's own pp16384 arm before any number is quoted; if it does not reproduce, the block-00
   per-block note and `ENVIRONMENT.md`/`patches/README.md` must say so rather than repeat the old figures.

## R00.7 The rung's own trap: `GGML_CUDA_FA_QUANTS`

The configure aborted with

```
GGML_CUDA_FA_QUANTS: unknown type "iq4_nl" in "iq4_nl-iq4_nl", must be one of: q4_0;q4_1;q5_0;q5_1;q8_0;bf16;f16;f8_e4m3
```

`iq4_nl` is **block 05's** FA instance, and this variable is set by hand in `build-rocm/CMakeCache.txt` on
this box.  It was set to `q4_0-q4_0;q4_1-q4_1;q5_0-q5_0;q5_1-q5_1;q8_0-q8_0;bf16-bf16;f16-f16;f8_e4m3-f8_e4m3`
for R00 and **`iq4_nl-iq4_nl` must be added back when R05 lands** (`AGENTS.md` documents this exact trap).

## R00.8 Also recorded

- `LLAMA_USE_PREBUILT_UI` is now `OFF` in `build-rocm`'s cache (the new build script default), so a rung build
  fetches no UI: `-- UI: embedded 0 assets`.
- A `-ncmoe 24` Flash-Next arm crashed the same way as R00.6 — **one root cause, not a placement bug**.
- The `-ncmoe 47` arms did run (14 s load, 39/13.5 t/s) because their prefill was 1-19 tokens, i.e. an
  `n_pool` small enough for `ncols == k`.  No conclusion should be drawn from those rates.

---

# R01 — block 01, hybrid HIP all-reduce  ·  **DONE 2026-11-20** — `14a0414bb`, tree `6566ca3ac`

```
git cherry-pick a94e2214c        # 4 files, +2900/-49, zero conflicts
```

| exit criterion | result |
|---|---|
| conflicts | **none** (the cherry-pick applied clean) |
| **one rung repair** | `ggml_cuda_ce_xor`'s raw `__shfl_xor_sync(0xffffffff, v, m, 32)` does not compile any more — see below |
| build | green (`llama-cli`, `llama-server`, `llama-bench`, `test-backend-ops`) |
| coherence, 1 GPU `-sm layer` | `100 chars sha=1c5d32ac537d` |
| the arm that matters, 3 devices `-sm tensor` | `100 chars sha=1c5d32ac537d` — **identical to R00** (`build-r00-novmm`, same arm), `88.0 / 104.6` t/s against R00's `86.2 / 66.2` (single runs, not a benchmark) |
| non-vacuity | `-lv 4`: `ggml_cuda_ar_pipeline_init: initialized HIP AllReduce pipeline: 3 GPUs, 1024 KB chunked kernel staging + 32 MB copy-engine staging per GPU` — the arm is reached |

## R01.1 The repair: a HIP mask is 64-bit now

```
/opt/rocm-10.1.0-gfx120X/include/hip/amd_detail/amd_warp_sync_functions.h:307:62:
  error: static assertion failed due to requirement 'sizeof(unsigned int) == 8':
  The mask must be a 64-bit integer. Implicitly promoting a smaller integer is almost always an error.
  note: in instantiation of '__shfl_xor_sync<unsigned int, float>' requested here   (ggml-cuda.cu:1806)
  note: in instantiation of 'ggml_cuda_ce_xor<1>' requested here                     (ggml-cuda.cu:1812)
```

Upstream `278a89cb7` (`#31617`) introduced the typed `ggml_cuda_thread_mask_t` and the wrappers
`ggml_cuda_shfl_xor` / `ggml_cuda_ballot` / `GGML_CUDA_FULL_THREAD_MASK`; HIP's header now refuses a 32-bit
mask literal.  Block 01's clean-side reduction (the q8 copy-engine arm) called the intrinsic directly.
**Resolution:** `ggml_cuda_shfl_xor(v, m, 32)` — upstream's wrapper, default mask
`GGML_CUDA_FULL_THREAD_MASK` (all lanes at the physical warp size), width 32 preserved, so the reduction's
lane mapping and its result are unchanged.  This is the first place the "upstream refactored the primitive"
class of drift (14 of the 61 conflicted paths are behind that one commit) bites a *delivery-only* file that
the probe could not see — a rung repair, not a decision.

## R01.2 Note for R02 and later

Block 01 adds `allreduce-hip.cu` (1795 lines) and a dispatch in `ggml-cuda.cu`.  Nothing upstream refactored
inside it beyond the mask type, so later FA/MMQ rungs should not re-open it; if `GGML_CUDA_FA_QUANTS` or a
fusion arm changes the *tensor sizes* the all-reduce sees, `gate-prefill-logits.sh` (R18) is the check, not a
rung.

---

# R02 — block 02, adaptive MTP draft depth  ·  **DONE 2026-11-20** — `3941bad4e`, tree `0799ebfad`

```
git cherry-pick cef19b032        # 23 files -> 53 files changed, +5290/-420
```

**All 5 conflicts are the DFlash spec refactor (`#32016`, `32bdc3a89`) and every one is a union** — no
delivery code was dropped, so `DECISIONS.md` **D5 resolves as MERGE, not as a decision**.

| exit criterion | result |
|---|---|
| conflicts | 5 files, 9 regions (below) |
| build | green, `~/bin/build-llama-rocm` (`all` targets), **0** errors |
| coherence 4B, plain path | `100 chars sha=1c5d32ac537d` — unchanged |
| **the adaptive arm** | `--spec-type draft-mtp-adaptive` on Qwen3.8-27B UD-Q4_K_XL: **29.3 -> 65.2 t/s** generation (**+122 %**) |
| text identity | `none` and `draft-mtp-adaptive` both read **`138 chars sha=55824e640aa0`** — the adaptive controller changes speed, not output |
| census (non-vacuity) | `adaptive draft depth enabled (draft-mtp-adaptive)`; `draft acceptance = 0.95238 (20/21), mean len = 3.86`; `#acc rate/pos = (1.000, 1.000, 0.857)` |

## R02.1 The five resolutions (all unions)

| region | HEAD (upstream `#32016`) | ours (block 02) | resolution |
|---|---|---|---|
| `common/common.h` enum | added `DRAFT_DFLASH2`, `DRAFT_LILICORR` after DSPARK | added `DRAFT_MTP_ADAPTIVE` after `DRAFT_MTP` | **both**, upstream's order, ours inserted after `MTP` |
| `common/common.h` `need_n_rs_seq()` | explicit list: MTP, EAGLE3, DFLASH, DSPARK, **DFLASH2, LILICORR** | `has_mtp()` + EAGLE3/DFLASH/DSPARK | **both**: `has_mtp() || (EAGLE3, DFLASH, DSPARK, DFLASH2, LILICORR)`.  `has_mtp()` already covers `MTP_ADAPTIVE` (common.h:424) |
| `common/speculative.cpp` name map | upstream's map incl. `draft-dflash2`, `draft-lilicorr` | ours incl. `draft-mtp-adaptive` | **both unioned** |
| `common/speculative.cpp` **420-line region** | `#32016` inserted the DSpark / DFlash2 / LiliCorr classes (derived from `impl_draft_dflash`) at the same point we inserted the `mtp_dev_rows_enabled()` helper | 13 lines | **both**: upstream's classes first, then our helper, immediately before `struct common_speculative_impl_draft_mtp` |
| `common/speculative.cpp` `type_to_str()` / `static_assert(COUNT == 13)` / `add_config_if_enabled()` | 13 types | 12 + ours | **both**; the count is now **14** |

**D5's actual question — does the adaptive-depth control still have a seam?**  Yes.  `#32016` refactored by
*adding sibling classes* (`DSpark`/`DFlash2`/`LiliCorr` derive from `impl_draft_dflash`) and left
`common_speculative_impl_draft_mtp` as its own class, which is where every block-02 hunk (`adaptive`,
`n_cap`/`n_hard`/`n_last`, `adaptive_ctrl`, `align_draft_caps()`, `set_prompt_end()`,
`prompt_batch_unneeded()`, the rewritten `accept()`) landed **cleanly**.  So the delivery's controller is
orthogonal to upstream's new speculators — the one thing to watch for is `need_n_rs_seq()`, which upstream
made explicit and now needs to stay in step with the enum (a new upstream type that needs rs_seq must be
added there).

## R02.2 The generated docs: regenerate, never hand-merge

`tools/cli/README.md`, `tools/server/README.md` and `docs/speculative.md` are generated from
`common/arg.cpp` by `llama-gen-docs`.  The two tables were unioned by hand first, then **regenerated**
(`./build-rocm/bin/llama-gen-docs`), which is the canonical form and is what R18 will do anyway.

**That regeneration found a pre-existing defect, not caused by this re-base:** block 00's `arg.cpp` hunk adds
the `--lazy-mode mlockr` help text (and is the only delivery block that does — the delivery's blocks were
re-cut from the tip, so a later feature's help text sits in the foundation block), but neither
`tools/cli/README.md` nor `docs/speculative.md` at **r327** lists `mlockr` at all
(`git show e159adfc2:tools/cli/README.md | grep -c mlockr` = **0**, `arg.cpp` = **2**).  The docs are stale at
the release.  The rung regeneration fixes it as a side effect; **R18 must regenerate and assert no drift** so
the release cannot go out stale again.

## R02.3 Notes for later rungs

- The 27B single-GPU arm needs `--fit false -c 8192 -fa auto`; the default `--fit` tried a 16 GiB KV
  allocation on a 32 GiB card and died in `llama_init_from_model`.
- `tests/test-speculative-adaptive.cpp` and `tests/test-recurrent-state-depth.cpp` arrive with this block and
  build clean; they are not part of `--fast`, so they are first compiled here (the full build compiles them).
- `spec_draft_align_cache()` / `spec_align_enabled()` / `spec_align_width_band()` (block 02's draft-KV align
  helpers) merged cleanly; their gate is `scripts/gate-spec-align.sh` at R18.

---

# R03 — block 03, chunked gated-delta-net prefill  ·  **DONE 2026-11-20** — `e612388ba`, tree `971953b99`

```
git cherry-pick 36da15029        # 25 files (702 + 996 + 1015 lines of chunked-GDN kernels)
```

| exit criterion | result |
|---|---|
| conflicts | **1 file, 2 regions** — `tests/test-backend-ops.cpp` |
| build | green (`all` targets), 0 errors |
| coherence 4B (a GDN hybrid, so the chunked path runs) | `100 chars sha=1c5d32ac537d` — unchanged |
| `test-backend-ops -o GATED_DELTA_NET` | **76 OK / 0 FAIL**, 2/2 backends (this is the file the conflict was in) |
| **the block's own switch moves its own measurement** | chunked prefill **ON 6870.1 t/s** vs **OFF 6314.6 t/s** on a 5298-token prompt (**+8.8 %**), decode 87.8 both, text **identical** (`43 chars sha=aeef162bc0ce`) |

## R03.1 The two resolutions — both are unions of independent additions

**Region 1 (`test_gated_delta_net::build_graph`).**  Upstream added `padded_q` / `permuted_k` layout
variants and calls `ggml_gated_delta_net(...)` with 8 arguments; the delivery's block 03 adds a ninth,
`n_rs_batch`.  Resolution: keep upstream's two layout blocks and pass `/*n_rs_batch=*/1` — the test now
exercises the delivery's parameter *in* upstream's new layouts, which is strictly more coverage than either
side had.

**Region 2 (`initialize_tensors`, the `g` gate).**  Upstream's new `correlated` case initialises the gate at a
constant `-1e-4`; the delivery replaced the general `-20 .. -1e-4` range with `-0.5 .. -1e-4` because the old
range decayed the state to zero and hid `n_seqs > 1` bugs.  Resolution: `correlated ? -1e-4f : -0.5f`, i.e.
upstream's near-no-decay stress case plus the delivery's realistic floor, with the delivery's comment kept
(it explains why the floor exists).

## R03.2 Later-rung notes

- **Upstream still has no chunked GDN on CUDA**: `ggml/src/ggml-cuda/gated_delta_net.cu` at `919794500` carries
  `//TODO: Add chunked kernel for even faster pre-fill`.  The only upstream "chunked GDN" in the range is
  **Vulkan** (`16f80bd0c`, `#30207`), a different backend — so this block is not subsumed and R03 needed no
  `DECISIONS.md` entry.
- The block builds three new kernel translation units (`gated_delta_net_chunked{,_bf16,_bf16_gfx11}.cu`), so
  the build time steps up here; the per-rung `--fast` profile does not build them until the release ladder.
- Knobs seen at this rung (all pre-existing): `GGML_CUDA_GDN_CHUNKED`, `GGML_CUDA_GDN_CHUNKED_BF16`,
  `GDN_CHUNKED_MIN_TOKENS`, `GDN_CHUNKED_NTHREADS`, `GDN_CHUNKED_CS`, `GDN_CHUNKED_VT`,
  `GDN_CHUNKED_STATE_PIPE`.  `GDN_CHUNKED_MIN_TOKENS` and `n_rs_batch`'s use as a kernel input are what
  `scripts/gate-spec-gdn-split.sh` grades at R18.

---

# R04 — block 04, BF16 KV cache + native BF16 flash attention  ·  **DONE 2026-11-20** — `768d27561`, tree `0866d4229`

```
git cherry-pick cfe40b5f9        # 8 files -> 8 files changed, +606/-195
```

**This is `D3`, and it was a real collision — in the fusion *decision*, not in the text.**  Both sides added
rope dispatch arms; only one side was allowed to win the predicate, and the delivery's side is the feature.

| exit criterion | result |
|---|---|
| conflicts | 2 (`rope.cu`, `ggml-cuda.cu`), 1 region each |
| build | green (`all` targets), 0 errors |
| coherence 4B, f16 KV | `100 chars sha=1c5d32ac537d` — unchanged |
| **bf16 KV arm** (the block's feature), f16 vs bf16 | prefill 6843.4 vs 6795.8, decode 89.8 vs 87.7, text **identical** both (`71 chars sha=1de99b7e6ee6`) |
| **the fusion is taken AND worth having** | `GGML_CUDA_DISABLE_FUSION=0`: **6786.5 / 88.8** against **6417.0 / 76.2** with it off — **+5.8 % prefill, +16.5 % decode**, identical text |
| `test-backend-ops -o ROPE` | **515 OK / 0 FAIL**, 2/2 backends |

## R04.1 `rope.cu` — a pure union of complementary arms

| arm | who added it |
|---|---|
| `float -> half`, `float -> float`, `half -> half` | pre-existing |
| `float -> nv_bfloat16` | block 04 (merged cleanly) |
| **`nv_bfloat16 -> nv_bfloat16`**, **`nv_bfloat16 -> half`** | upstream `#32387` |
| **`half -> nv_bfloat16`** | block 04 (the conflict) |

All seven arms are kept, so every pair of {F32, F16, BF16} now has a kernel.  That is what makes the next
decision safe.

## R04.2 `ggml-cuda.cu` — the actual `D3` decision

`ggml_cuda_should_fuse_rope_set_rows()` decides whether `ROPE + VIEW + SET_ROWS` runs as one kernel.  Upstream
`#32387` added:

```cpp
// BF16 rope fuses into an F16 or BF16 cache, other ropes not into a BF16 cache
if (rope->type == GGML_TYPE_BF16 ? set_rows->type == GGML_TYPE_F32 : set_rows->type == GGML_TYPE_BF16) {
    return false;
}
```

i.e. **every non-BF16 rope into a BF16 cache is refused** — which is precisely the delivery's bf16-KV path
(an f32/f16 rope writing into a bf16 cache).  Upstream refused it because *its* matrix had no `F32/F16 -> BF16`
arm; the delivery's block 04 supplies them, so with the unioned matrix the refusal is the only thing that
would disable the feature.  **Resolution: the guard is dropped and the outer cache-type list is the test.**
That is not a delivery-code deletion — it removes a freshly added upstream guard whose premise the delivery's
own kernels invalidate — and the measurement above is what justifies it (+5.8 % pp / +16.5 % tg at bf16 KV,
identical text).

If the maintainer would rather keep upstream's guard for non-delivery configurations, the alternative is to
keep it and let the bf16-cache fusion go — that costs the numbers in the table above on the delivery's own
bf16-KV path, so it is recorded here rather than taken silently.

## R04.3 Notes for later rungs

- `GGML_CUDA_DISABLE_FUSION` is the switch that grades any fusion decision; R08's norm/MMQ fusions and R10's
  dispatcher will use the same arm shape (fusion ON vs OFF, text identical, rate moved).
- The bf16 cache costs ~0.7 % prefill and ~2 % decode against f16 on this model at this size — i.e. it is
  *not* a decode win here; the value of the block is the native (non-scratch) bf16 attention path, which the
  prefill-logit gate and the FA op test grade, not a rate.
- The `-ctk bf16 -ctv bf16` arm needs `-fa on`; with FA off the cache type falls back and the arm is vacuous.

---

# R05 — block 05, FA band + native quantized K/V  ·  **BLOCKED ON A DECISION (D16)** — analysed, not resolved

```
git cherry-pick 7362b5958     # 341 files, 7 conflicted paths, 22 conflict regions
```

The cherry-pick was taken and **aborted** on 2026-11-20 so the ladder stays clean at `768d27561`.  Nothing is
half-merged.

## R05.0 Why this rung is different from R00-R04

Upstream `054a1babf` (`#32407`, *Rotate K/V read order for better L2 cache hit rates*) and block 05 **rewrite the
same iteration model** in `flash_attn_ext_f16_process_tile`, in incompatible ways:

| | upstream `#32407` | block 05 |
|---|---|---|
| loop shape | `if constexpr (nstages > 1)` + a contiguous run and a rotated remainder, `kb0_next = kb0 + 1 == kb0_max ? 0 : kb0 + 1`, `nkb = nkb_pre + nkb_post`, `kb0 = ... ? 0 : kb0_start` | `if (kb0 < kb0_stop)` then `for (; kb0 < kb0_stop; kb0 += kb0_step)`, `kb0_step > 1` = the round-robin KV split of a band block |
| extra state | `kb0_max` (skip fully-masked blocks past it) | `kq_blocks` bitmap skip (issue #48), a **neutral partial** written by a band block that owns no iteration, and the RDNA4 fetch-ahead ring (`band_ring_v`, `band_q8`, `band_kb0_next`) |
| `nbatch_*` / `Q_in_reg` / `nstages` / `swz` | unchanged | parameterised on `ncols2`, `band_wide`, `band_fetch_ahead` |

`#32407` is **not** a prefetch hint: it changes *which* blocks a tile visits and in what order, and its
bookkeeping (`nkb_pre`/`nkb_post`) must account for every block exactly once.  Block 05's band model is a
different partition of the same work.  A merge is therefore a **third iteration model** that supports a strided
band *and* the rotation — inside a ~3000-line template kernel, where a subtle off-by-one silently corrupts
attention on every MMA-path model.

## R05.1 The 22 regions, classified

| class | count | regions |
|---|---|---|
| mechanical — parameter types / appended parameters (`int32_t nb32` -> `int64_t`, plus the delivery's `derived`/`kv_native`/band parameters) | 5 | `fattn-common.cuh` r1, `fattn-vec.cuh` r1, `fattn-tile.cuh` r1, `fattn-mma-f16.cuh` r1, r4, r10 |
| mechanical — call sites that only need upstream's renamed argument (`kb0_next*nbatch_fa`, `kb0, kb0_next`) plus the delivery's extras | 4 | `fattn-mma-f16.cuh` r2, r3, r6, r7 |
| small semantic union — the per-head-mask fix (`#32217`) vs the delivery's derived-mask predicate (`ggml_cuda_flash_attn_ext_has_mask(dst)`) | 3 | `fattn-tile.cuh` r2, r3, `fattn.cu` r1, r2 (r2 = the `maskh` head term `nb32*(head0 % ne32)`) |
| mechanical — test parameters (`derived`/`derived_hole` + upstream's `n_kv_max`/`q_head_padding`/`mask_heads`) | 3 | `tests/test-backend-ops.cpp` r1, r2, r3 |
| small port — upstream refactored the kq-mask builder into `build_attn_inp_attn_mask` (+38 lines) and the delivery threads a `derived` flag through `build_attn_inp_kq_mask` | 1 | `src/llama-graph.cpp` r1 |
| **the iteration model** | **5** | `fattn-mma-f16.cuh` r5, r8, r9, r11, r12 |

So ~17 of the 22 are ordinary union work; **5 regions are a design decision**, and they are the core of the
block.

## R05.2 The options (a maintainer decision — `DECISIONS.md` D16)

| option | what | cost / risk |
|---|---|---|
| **A — combined model** | design a third iteration model: a strided band that also honours the rotation, `nkb_pre`/`nkb_post` and `kb0_max` | highest fidelity, highest risk. The oracle is `test-backend-ops -o FLASH_ATTN_EXT` (**7381 cases, ~25 min per iteration**) plus the prefill-logit KLD gate and text arms. Estimate 2-4 focused sessions, and it should not be attempted without that gate green at every step |
| **B — keep block 05's model, drop `#32407`** (recommended) | the 17 mechanical/small regions still get the union treatment, so upstream's per-head-mask fix (`#32217`) and the type changes survive; only the rotation is not carried | loses an upstream L2 micro-optimisation with **no recorded number** in its own commit message or in this repo's benchmarks; the delivery's band/stream-K split is a shipped, gated feature. Reversible: the rotation can come back as its own rung-repair or an `upstream/` candidate once the band model is stable |
| **C — take upstream's model, re-band later** | keep `#32407`, re-implement the band/stream-K split on top | equivalent to redoing block 05; the delivery's FA band is one of its headline features |

**Not attempted: a blend of A and B without the decision**, because the two models disagree about which blocks
a tile owns, and a wrong choice is invisible to text on a short prompt while silently wrong at depth.

---

# R05 — block 05, FA band + native quantized K/V + derived kq mask  ·  **DONE 2026-11-20** — `1c9bfa37c`, tree `13683655d`

```
git cherry-pick 7362b5958     # 341 files, 7 conflicted files, 22 conflict regions
```

**`D16` decided (maintainer, 2026-11-20): option 2 — the delivery's FA iteration model wins, upstream's
`#32407` K/V read-order rotation is not carried.**  The standing bias stated with the decision: across every
re-base so far, upstream has never beaten the delivery on a shared mechanism, so where the two are
alternatives the delivery's side is taken and the loss is recorded and measured rather than assumed absent.

| exit criterion | result |
|---|---|
| conflicts | 7 files / **22 regions** (classification below) |
| build | green, `all` targets, 0 errors |
| coherence 4B | `100 chars sha=1c5d32ac537d` — unchanged |
| **`test-backend-ops -o FLASH_ATTN_EXT`** | **7428 OK / 0 FAIL**, 2/2 backends — the whole FA matrix, and precisely the oracle a combined iteration model would have needed at every step |
| the block's own switch, on the path it exists for | q8_0 KV: `GGML_CUDA_FA_KV_NATIVE=1` **88.6 t/s** decode vs **83.9** staged (**+5.6 %**), identical text `71 chars sha=1de99b7e6ee6`; f16 KV is a wash (staging only costs anything for a *quantized* cache, so the f16 arm was vacuous and is recorded as such) |

## R05.1 What was decided and what it cost

| | upstream `#32407` | block 05 (kept) |
|---|---|---|
| iteration | `nkb = nkb_pre + nkb_post` — contiguous run + rotated remainder, `kb0 + 1`, wrap at `kb0_max` | strided band `for (; kb0 < kb0_stop; kb0 += kb0_step)` |
| skip | `kb0 >= kb0_max` | `kq_blocks` bitmap (issue #48) |
| empty block | n/a | a neutral partial (`max -FLT_MAX/2`, rowsum 0, VKQ 0) the fixup ignores |
| extra | — | RDNA4 fetch-ahead ring (`band_ring_v`, `band_q8`, `band_kb0_next`) |

**Cost of the choice:** the rotation is gone.  It is a pure L2-locality reordering with **no number in its own
commit message**; the band/stream-K split it would have displaced is a shipped, gated feature.  Reversible: it
can return as a rung-repair or an `upstream/` candidate once the band model is stable — and if it does, it must
arrive as a *combined* model with the FA op matrix green, which is the work option 1 would have been.

**Verified absent** (so nothing of it is silently half-wired): no `nkb_pre`, `nkb_post`, `kb0_max` or `kb0_rot`
anywhere in the FA sources; every `kb0_next` in the tree is the delivery's own fetch-ahead.  `KV_max` *is*
present and is the delivery's own, pre-existing — the merge did not import it (the `fattn_kernel_t` typedef and
its parameter list are byte-identical between block 05's commit and the merged tree).

## R05.2 The 22 regions

| class | n | what |
|---|---|---|
| iteration model | 5 | `fattn-mma-f16.cuh` r5, r8, r9, r11, r12 — **ours** |
| rotation plumbing in signatures / call sites | 7 | `fattn-mma-f16.cuh` r1, r2, r3, r4, r6, r7, r10 (`kb0_next`/`kb0_max` params and their call sites) — **ours** |
| per-head mask (`#32217`), taken from upstream | 4 | `fattn-tile.cuh` r2 (`+ nb32*(head0 % ne32)`) and r3, `fattn.cu` r1, r2 — unioned into our derived-mask predicate |
| signature types | 3 | `fattn-common.cuh` r1, `fattn-vec.cuh` r1, `fattn-tile.cuh` r1 — ours' parameters, `nb32` kept `int32_t` (see below) |
| graph refactor | 1 | `src/llama-graph.cpp` — upstream's `build_inp_mem_hybrid_iswa_rel` + `build_attn_inp_attn_mask`, called through the delivery's derived-mask signatures |
| test parameters | 3 | `tests/test-backend-ops.cpp` — `derived`/`derived_hole`/`mask_hole` **and** `n_kv_max`/`q_head_padding`/`mask_heads` |

## R05.3 Three fixes the compiler forced, all worth recording

1. **`nb32` is `int64_t` — and my first answer here was WRONG, corrected by the maintainer's question.**  I
   initially kept the delivery's `int32_t` (a half-widened typedef does not link against an int32 kernel, and
   `nb32` on its own looked small: `mask->nb[2]` = `n_kv*xn_tokens*2` bytes, 512 MiB at 128k context with a
   2048-token ubatch).  **But `nb32` is not used on its own.**  The per-head term this rung *adopted from
   upstream* is `nb32*(zt_Q % ne32)`, an `int` multiplication with a head index up to 71: 512 MiB x 71 =
   **35.5 GiB**, so the product wraps and the mask pointer is wrong.  Overflow begins at `nb32` > 2^31/71 =
   **28.8 MiB**, i.e. ~15.1M cell-tokens — e.g. a 64k context with a 16-token ubatch.  Upstream widened it in
   the typedef **and all four kernel definitions together**; that is now done here in the same four places
   (`fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh`); `nb31` stays `int32_t`, as
   upstream leaves it.  Repair `c3860af1e`; FA matrix re-verified 7428/7428.
   **Honest gap:** the op matrix exercises **zero** cases with `mask_heads > 1`, so the per-head path this
   serves has no automated coverage here.  The maintainer then placed `Inkling-2-Mini` on the box to close it
   (`/llm/models/Inkling-2-Mini/Q8_K_XL/…`, `arch = inkling2`, `block_count = 48` with **24 physical blocks**
   plus `loop.core_start/end` and `num_loops = 3`) — and **that model cannot load at any base in this ladder**:
   `src/models/inkling2.cpp` implements no loop, so it asks for `blk.24.*`.  The pristine `919794500` base fails
   identically and r327 does not know the arch, so no delivery change is involved.  Recorded as **`TODO` #353**
   with the diagnosis, what it blocks (this per-head path and the `nb32` widening) and its two re-entry
   conditions.  So the widening is correct by construction and matches upstream, and is still **not locally
   validated** — which is a property of the model support, not of this rung.
2. **`#31617`'s typed masks reach block 05's own code.**  HIP now static_asserts a 64-bit warp mask, so the
   seven raw `__shfl[_xor]_sync(0xFFFFFFFF, ...)` calls in the band/fetch-ahead code had to become
   `ggml_cuda_shfl_xor` / `ggml_cuda_shfl` — the same class of repair as R01, but inside block 05's new code
   rather than in an untouched file.
3. **`fattn-tile.cuh` needed upstream's `mask` tensor declaration.**  The delivery's launcher predates `#32217`,
   so `mask->ne[2] == 1` had no `mask` to test: `const ggml_tensor * mask = dst->src[3];` is restored (null for
   the derived form, which is exactly what the predicate wants).

## R05.4 Notes for later rungs

- `GGML_CUDA_FA_QUANTS` was restored to include `iq4_nl-iq4_nl` at this rung, as `RUNGS.md` R00.7 required —
  block 05 adds `iq4_nl` to `FA_TYPES`, and the configure fails without it before R05 and with it after.
- The FA op matrix is the gate for anything that touches this file; it took ~4 min here (7428 cases), not the
  ~25 min the cost note warns about, because `-o FLASH_ATTN_EXT` on one device is much cheaper than the full
  sweep.
- `-fa on` is required for the native-K/V arms; with FA off the cache type is not used by the attention kernel
  and both arms collapse to the same path.

---

# R06 — block 06, MMB quantized prefill GEMM  ·  **DONE 2026-11-21** — `ee377ec45`, tree `b5dc56996`

```
git cherry-pick f0564df7d        # 11 files, +3000/-16 (mmb.cu 2436 lines, mmf32.cu 201, a new test)
```

| exit criterion | result |
|---|---|
| conflicts | 2 files / 2 regions (`ggml-cuda.cu`, `mmid.cu`) — both unions, no delivery code dropped |
| no markers left | yes (the `git grep` hits are the binary vocab files, as at R00) |
| build | green, `all` targets, 0 errors; the only warning in a touch-path file is block 06's own `cudaFree` nodiscard at `mmb.cu:1953-1954`, **pre-existing** (identical lines at r327) |
| coherence 4B | **`100 chars sha=1c5d32ac537d`** — unchanged |
| `test-backend-ops -o MUL_MAT_ID` | **1078 OK / 0 FAIL**, 16 `dup=1` — the compaction whose signature this rung changes |
| the block's own switch | MMB default vs `GGML_CUDA_MMB=0` on 4B Q8_0: pp2048 7793.1 vs 7767.3 (**+0.33 %**), pp16384 6768.4 vs 6727.0 (**+0.62 %**); the F32 part alone (`GGML_CUDA_MMF32=0`) is 7722.9 / 6699.9 (**+0.9 % / +1.0 %**) |
| text identity | 35B-A3B Q8_0, 2 devices `-sm tensor`, 5298-token prompt: default `73 chars sha=86b7f9b5aa80`, `GGML_CUDA_MMB=0` **the same** `86b7f9b5aa80` |
| **one finding** | R06.2: block 06's MMB activation cache is not pool-safe; block 10's `pool_stack` is the fix.  Inert on the default RDNA4 *routed* arm (`routed=0`), reachable on a default *dense* type (IQ3_S) |

## R06.1 The two resolutions (both unions)

**`ggml-cuda.cu` — `ggml_cuda_mul_mat_id`, the dispatch after the MMVQ band.**  Upstream `#32065` added
`use_mmq_few` → `ggml_cuda_mul_mat_q_few()` at exactly the point block 06 inserts its MMB check.  Resolution:
**keep both, upstream's first**, because they are disjoint by *backend* and by *shape*: `mmq-few.cu` is
compiled out under HIP (`ggml_cuda_should_use_mmq_few()` returns `false` unconditionally on HIP/MUSA), and
MMB is RDNA-only and needs `T >= mmb_min_t()` (512) while few-row MMQ claims 2-24 columns.  So the NVIDIA
path keeps upstream's dispatch untouched and AMD reaches MMB exactly as before.

**`mmid.cu` — `ggml_cuda_launch_mm_ids_helper` gained `int32_t * expert_bounds`** (block 06's second reader of
the same compaction: MMQ/MMF consume `tile_spec`, MMB consumes per-expert bounds).  The rung has to re-port
it onto the R00 tree, where the helper is **upstream `#32505`'s store design** (`check_experts`/`dump_store`/
`ne_store`) plus our `mm_ids_helper_512_10`, not the pre-rebase body block 06's diff assumes.  Three places:

| site | resolution |
|---|---|
| `mm_ids_helper` signature + tail | add the parameter; after `mm_ids_write_tile_spec` write `expert_bounds[expert] = nex_prev` and, in the block whose `expert == gridDim.x - 1`, `expert_bounds[gridDim.x] = nex_prev + n_dumped` (**`n_dumped`**, upstream's running total — block 06's pre-rebase body used `it_compact`, which the store design resets at every dump) |
| `mm_ids_helper_512_10` tail | add `expert_bounds[tid] = counts[tid]` under `tid < n_experts` and `expert_bounds[n_experts] = counts[n_experts-1] + cursors[n_experts-1]` from the last thread (`counts[]` holds the compact-row start after the in-place cumulative pass, `cursors[]` the occurrence count) |
| `launch_mm_ids_helper` / `ggml_cuda_launch_mm_ids_helper` | thread the parameter through both, keep the **R00 store launch** (`nbytes_shared`/`ne_store`) and the **R00 512/10 fast path** with its `GGML_CUDA_DISABLE_MMID_512` switch.  The whole pre-rebase `mm_ids_helper` body that block 06's diff carries inside the conflict region is **dropped** (upstream's store version is already above it in the file — keeping both would duplicate the kernel) |

The `mmq.cu` / `mmf.cu` call sites are auto-merged to pass `nullptr`, which is what they carry at r327 too.
`mmid.cuh`'s declaration merged cleanly.  Bounds semantics are bit-for-bit block 06's (`mmb_build_desc2`
reads `bounds[e]`/`bounds[e+1]`, which consecutive blocks fill).

## R06.2 FINDING — block 06's MMB scratch buffers are not pool-safe until block 10's `pool_stack`

Block 06 keeps its BF16-activation cache and its pinned producer slots in process-wide globals
(`g_mmb_cache`, `g_mmb_slots[]`, `g_mmb_slot_cap[]`) and **frees them in an order the delivery's VMM pool
cannot accept**: `mmb_cache_insert()` deletes the *oldest* entry (`delete g_mmb_cache.front().buf`) and
`ggml_cuda_mmb_slot_reserve()` deletes on growth (`delete e.buf`), while the VMM pool is a strict stack
(`ggml-cuda.cu:808`: `GGML_ASSERT(ptr == (char *) pool_addr + pool_used)`).  Block 10 is what makes this
legal — `7bbf540d5` rewrote exactly these functions onto `ctx.pool_stack_push`/`pool_stack_kill` with the
comment *"these buffers are evicted, grown and replaced in an order the pool cannot accept"*.  So the defect
is the **block-06-without-block-10 intermediate tree**, not a resolution of this rung:

| arm (this rung's build unless named) | result |
|---|---|
| 27B **UD-IQ3_S** (a default-on RDNA4 dense type), 1 GPU, 6 k-token prompt, MMB default | **ABORT** at `ggml-cuda.cu:808` after 6 `MMB_DENSE` lines |
| same model, `GGML_CUDA_MMB=0` | OK — prefill **1175.9** / gen 32.9 t/s |
| same model, **r327** (`bin-r327`, all 18 blocks) | OK — prefill **1162.4** / gen 37.8, `50 chars sha=ebf2a5a9b00a` |
| 35B-A3B Q8_0, 2 dev `-sm tensor`, `GGML_CUDA_MMB_TYPES=q8_0` (dense forced on), routed off | OK — 80 `MMB_DENSE` hits, prefill 4915.1 t/s |
| 35B-A3B Q8_0, `…_TYPES=q8_0 …_ROUTED=1` (routed forced on) | **ABORT** — backtrace: `ggml_cuda_pool_vmm::free` ← `ggml_cuda_mul_mat_id_mmb_glu` (sometimes the non-GLU `…_id_mmb`) |
| the same, `GGML_CUDA_MMB_GLU=0` / `…_CACHE=1` / `GGML_CUDA_POOL_DEBUG=1` | still aborts (or dies on `invalid resource handle` for the debug pool) |

**Why the rung still passes:** (1) the *shipped default* on this box is `routed=0`, and the RDNA4 dense set is
`{IQ3_S, Q4_1, Q5_0}` — a Q8_0 model is untouched, which is what the coherence/text arms measure; (2) the
fix is a block already in the ladder; (3) a rung is not shippable.  **Owed:**

1. **R10** — after block 10 lands, re-run the 27B IQ3_S arm (must prefill, not abort) **and** the forced
   routed arm (`GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1`, 35B-A3B, 5298-token prompt) and require the
   text to equal the `GGML_CUDA_MMB=0` hash.  That arm is also the **only** coverage of this rung's
   `expert_bounds` re-port, so it must be run before R18.
2. **R18** — nothing to add to the gates beyond that: the block-06 arm in the release ladder is the default
   RDNA4 config, which is clean here.

## R06.3 D11, the knobs that belong to this rung

`DECISIONS.md` D11 asks for "the better of each knob", measured on one tree.  What R06 owns:

| knob | arms | result |
|---|---|---|
| MMB (block 06 itself) | default vs `GGML_CUDA_MMB=0`, 4B Q8_0, `llama-bench -p 2048,16384 -r 3` | **7793.1 / 6768.4 vs 7767.3 / 6727.0** → MMB stays, +0.33 % / +0.62 % (pp16384 repeatability ±0.15 %, so the direction is real and the size is small on this model because Q8_0 is *not* in the RDNA4 dense set — only the F32 router/HC paths fire) |
| F32 skinny prefill GEMM (block 06's `mmf32.cu`) | default vs `GGML_CUDA_MMF32=0` | **7722.9 / 6699.9** → +0.9 % / +1.0 %, i.e. most of the default arm's gain is this kernel, not the quantized dense tile |
| few-row MMQ for `MUL_MAT_ID` (upstream `#32065`) | upstream's dispatch vs ours | **no AMD A/B exists**: `ggml_cuda_should_use_mmq_few()` is `false` for every HIP build, so upstream's new path is inert here and our dispatch is unopposed.  Recorded, not assumed |
| MMQ config table (`mmq-config-gcn.cuh`) vs the delivery's table | upstream's vs ours | **not this rung** — block 06 touches `mmq.cu` only to pass `expert_bounds=nullptr`; the table collision is R07/R08's |

## R06.4 Notes for later rungs

- `GGML_CUDA_FA_QUANTS` still carries `iq4_nl-iq4_nl` from R05 — no configure change needed here.
- The new `tests/test-logits-width-probe.cpp` is added to `tests/CMakeLists.txt`; the incremental build
  compiles it, so a later `--fast` ladder rung will not be surprised by it.
- Block 06 also registers MMB in `ggml_backend_cuda_context`'s destructor (`ggml_cuda_mmb_release_all()`) and
  calls `ggml_cuda_mmb_begin_graph()` at the top of `graph_compute`; both merged cleanly and are what R06.2's
  ordering depends on.  When block 10 rewrites them, this rung's comments are the reference for what changed.

---

# R07 — block 07, k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad)  ·  **DONE 2026-11-21** — `1435e7873`, tree `d3b004581`

```
git cherry-pick f5bdbefda        # 14 files, +1468/-135 (mmvq-wmma.cu 433 lines new, vecdotq.cuh +542/-…)
```

| exit criterion | result |
|---|---|
| conflicts | 1 file / 2 regions (`mmvq.cu`), both unions |
| no markers left | yes; no `TODO(rebase)` |
| build | green, `all` targets, 0 errors, **0** warnings in the touch-path files |
| coherence 4B | **`100 chars sha=1c5d32ac537d`** — unchanged, and **identical with `GGML_CUDA_GRID_STALL_PAD=0`** |
| `test-backend-ops -o MUL_MAT -p type_a=(q6_K\|q8_0\|q1_0\|iq4_xs)` | **335 OK / 0 FAIL** — the `vec_dot_*` family and the MMQ load tiles this block rewrites |
| the block's own switch (pad) | 27B Q6_K, `llama-bench -p 2048 -n 128 -r 3`: default **1154.45 / 24.23** vs `GGML_CUDA_GRID_STALL_PAD=0` **1138.75 / 24.18** → the pad is +1.37 % prefill, +0.2 % decode, text identical |
| the delivery's tile knob | `GGML_CUDA_MMQ_J_MAX=64` → **977.53 / 24.20** (**−15.3 %** prefill): capping the MMQ tile width only hurts at this base, so the default (up to 128) stands |

## R07.1 The two resolutions — both unions in `mmvq.cu`

**Region 1, `mmvq_parameter_table_id`.**  Upstream added `MMVQ_PARAMETERS_ORIN` for Jetson Orin; block 07 adds
`MMVQ_PARAMETERS_NAVI21` for real RDNA2 (gfx103x).  Both kept, upstream's first.  Nothing indexes the enum
positionally, so the order is cosmetic; what matters is that `get_device_table_id()` (device and host overloads)
route to the right id.

**Region 2, `calc_nwarps`.**  Upstream's `ORIN` case (`→ GENERIC`) and the delivery's `NAVI21` case (4 waves at
`ncols_dst == 1`, 2 at 2-4, 1 above) insert at the same point.  Both kept, back to back — they test different
ids.

**The table-id split survived the auto-merge and was checked, not assumed.**  The delivery's own split (RDNA3_5
keeps its own table rather than upstream's "RDNA2 || RDNA3_5 → RDNA2", real RDNA2 gets NAVI21) is in place on
both paths, and upstream's new `RDNA4_M` family is folded into the NAVI21/RDNA2 table on both.  That ordering is
sound: `GGML_CUDA_CC_IS_RDNA2(cc)` is `[RDNA2, RDNA3)` and `GGML_CUDA_CC_IS_RDNA4_M(cc)` is `[RDNA4_M, RDNA4)`,
i.e. disjoint ranges, so neither branch can shadow the other.

## R07.2 D11 — the knobs this rung owns, measured (`per knob, not per file`)

| knob | arms | result |
|---|---|---|
| MMVQ `ne11` gate for RDNA3_5 (upstream `c7518dbec`, `Q8_0: ne11 <= 6`) | upstream's vs ours | **adopted, no A/B possible here**: it merged cleanly and sits in `ggml_cuda_should_use_mmvq` *before* the delivery's tables, but it is a **dispatch threshold**, not an alternative to block 07's VDR/`nwarps` tables (which are the kernel shape) — and it is RDNA3_5-only, so gfx1201 never reaches it.  It stays; the gfx1151 box is where it would be measured |
| MMQ tile width (`GGML_CUDA_MMQ_J_MAX`, block 07's own A/B knob) | default (≤128) vs 64 vs 32 vs 16 | **default wins**: pp2048 1154.45 (128) vs 977.53 (64); the short-prompt arm ranks them the same way (674.25 / 575.58 / 450.77 t/s at pp64).  No delivery override is warranted at this base |
| grid-stall pad (block 07, default ON under HIP) | default vs `GGML_CUDA_GRID_STALL_PAD=0` | **+1.37 % pp2048**, +0.2 % tg128, text identical → stays on (its own gate is `scripts/gate-grid-stall-pad.sh`, run at R18) |
| wide-VDR MoE (`vec_dot_*_moe`) | on vs off | **no switch to turn it off** — the MoE VDR is a per-arch default inside the kernel association.  Its own D11 row targets "the 1-GPU host-expert decode on RDNA3_5", which this box cannot measure (gfx1201 uses the dense-VDR association).  Recorded, and graded by the op matrix above |
| verify-band WMMA iu8 family (r290, `--wmma-verify` / `GGML_CUDA_MMVQ=…`) | — | **opt-in and default OFF** (`common_params::wmma_verify = 0`, `g_min_cols` unresolved → K-split kernel), so nothing on this rung's default path reaches it; with it off the byte-identical contract is unchanged.  Its A/B is the four-axis MTP protocol + `llama-batched-bench`, owed at R18 if the default is ever flipped |

## R07.3 A pre-existing trap in the delivery's own knob (attributed, not a rung regression)

`GGML_CUDA_MMQ_J_MAX=8` **aborts** on a Q6_K model: `GGML_ASSERT(J_best > 0) failed` (`mmq.cu:271`), because no
`J = 8` config exists for that type at this base (upstream's per-arch `mmq-config-*.cuh` tables).  The **same
abort happens at r327** (`bin-r327`, same arm, `mmq.cu:615`) — so it is the knob's floor, not this rung:
**usable values are ≥ 16** on gfx1201/Q6_K.  Recorded because `ENVIRONMENT.md` documents the knob as "a value
< 8 is ignored", which is true but unhelpful: 8 is accepted and then asserts.  Not fixed here (that would be a
delivery change mid-climb); it is a candidate for a one-line clamp at R18 if the maintainer wants it.

---

# R08 — block 08, fused kernels (old-08 fusions + old-13 fused MoE + old-15 PR57/63)  ·  **DONE 2026-11-21** — `510769147`, tree `2ed6cb541`

```
git cherry-pick fe283f27c        # 38 files, +3.0k/-… (norm-gated 128, gdn-conv, ple-conv, mmvq-rdna new)
```

The climb's worst rung (9 of 38 paths in the probe; 8 conflicted files, 13 regions) plus **three** fixes the
compiler forced and **one** auto-merge artifact that would have compiled and mis-dispatched silently.

| exit criterion | result |
|---|---|
| conflicts | 8 files / 13 regions (`common.cuh`, `mmf.cuh`, `mmq.cuh`×4, `norm.cu`×2, `quantize.cu`×2, `set-rows.cu`, `topk-moe.cu`×7, `tests/test-backend-ops.cpp`×2) |
| no markers left | yes; no `TODO(rebase)`; both `#if`/`#endif` guards in `topk-moe.cu` balanced (see R08.3) |
| build | green, `all` targets, 0 errors, **0** warnings in the touch-path files |
| coherence 4B | **`100 chars sha=1c5d32ac537d`** — unchanged, and **identical** with `GGML_CUDA_DISABLE_FUSION=1`, `GGML_CUDA_DISABLE_NORM_ROWS=1` and `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` |
| `test-backend-ops -o NORM,RMS_NORM,GROUP_NORM,SET_ROWS` | **368 OK / 0 FAIL** — the norm epilogue union and the `SET_ROWS` q8_0 group kernel |
| `test-backend-ops -o MUL_MAT_ID` | **1079 OK / 0 FAIL**, 16 `dup=1` |
| the block's own surface | 4B Q8_0, `llama-bench -p 2048 -n 128 -r 3`: default **7409.28 / 97.67**; `DISABLE_FUSION=1` **7027.45 / 82.26** (−5.4 % pp, −18.7 % tg) — the fusion surface is live and worth having, text unchanged |
| `GGML_CUDA_DISABLE_NORM_ROWS=1` | **7388.04 / 97.48** — inside the arm's noise, i.e. **the switch is inert at this rung**, exactly as block 08's own commit says ("the norm-rows gates stay inert until block 10's fusion dispatcher exists") |
| **the block's headline path** (fused routed gate/up+GLU, 35B-A3B Q8_0, 2 dev `-sm tensor`, 5298-token prompt) | default **4850.8 / 80.6** `73 chars sha=86b7f9b5aa80`; `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` **4731.2 / 81.1**, **same hash**; `GGML_CUDA_DISABLE_FUSION=1` **4508.9 / 62.2**, **same hash** — the fused MoE is ~+2.5 % prefill and the whole fusion surface +29 % decode, and all three arms are **bit-identical to each other *and* to R06's** `86b7f9b5aa80` (the fusion is exact, not approximate) |

## R08.1 The eight resolutions

| path | ours (upstream) | theirs (block 08) | resolution |
|---|---|---|---|
| `common.cuh` | `#include "ggml-common.h"` | `ggml_cuda_rms_norm_vec()` (the issue-#229 numerics selector) | **both** |
| `mmf.cuh` | `ncols_case = ids ? 16 : ncols_dst` | `compact_ids ? MMF_IDS_COLS : (ids && ncols_dst > 16) ? 16 : ncols_dst` | **union**: `compact_ids ? MMF_IDS_COLS : (ids ? 16 : ncols_dst)`.  Upstream's rework tiles the whole ids path at 16 and `MMF_IDS_COLS == 16`, so the two agree; the compact branch is kept explicit because its reason is different (one row per (token, slot) occurrence) |
| `norm.cu` (1) | `norm_f32<block_size, do_multiply, do_add>` + `mul`/`add` params (`#32454`) | plain `norm_f32<block_size>` + `norm_f2bf()` | **both**: upstream's parameterised `norm_f32` kept (the body that follows is the union and uses `do_multiply`/`do_add`) **and** the delivery's `norm_f2bf` helper kept — the merged `rms_norm_f32` body writes the HC stream's BF16 copy through it |
| `norm.cu` (2) | `ggml_cuda_op_norm(ctx, dst, mul, add)` | `l2_norm_pair_f32` + `_cuda` (the SSM q/k pair) | **both**: the delivery's kernels first, then upstream's 3-arg signature (the merged body is upstream's fused form; `norm.cuh` already defaults `mul`/`add` to `nullptr`, so the 2-arg callers still work) |
| `quantize.cu` (1) | upstream's pre-restructure body fragment | the chunked/scatter loop header | **theirs** — the body after the header is the union and already carries both (`if constexpr (glu)` + `ggml_cuda_shfl_xor`); this is the same call block 08's own commit recorded at r278 |
| `quantize.cu` (2) | `quantize_mmq_few_id_q8_1_cuda` (`#32065`) | `quantize_mmq_q8_1_glu_cuda` + `quantize_mmq_q8_1_swiglu` | **both** — different functions, no overlap (upstream's is the NVIDIA few-row quantizer, block 08's is the fused-GLU feed) |
| `set-rows.cu` | `k_set_rows_quant<src_t, idx_t, ...>` (`src_t` added by `#32387`) | a `block_q8_0` group fast path + the old signature | **both** with the fallback given upstream's `src_t` first argument (R08.3 explains the second half of this fix) |
| `mmq.cuh` (×4) | `force_tiling` template param, `has_fusion`-only launches | `has_gate` param, the routed-compact dispatch, the fused args | **union**: the template is now `<…, has_fusion, has_gate = false, force_tiling = false>`; the delivery's routed-compact + fused launch is kept with upstream's `rows_innermost`-aware `gridDim.y` clamp |
| `topk-moe.cu` (×7) | plain `ggml_cuda_shfl_xor` butterflies, plain reduce | `topk_moe_argmax_stage*` (DPP/key sort), bit-exact norm reduce, `calc_experts_per_thread` | **both**: the delivery's bit-identical softmax/argmax/reduce bodies win (that is the delivery's stated contract for this kernel), upstream's `calc_experts_per_thread` is kept (the kernel calls it), and the delivery's key/sort helpers were re-added |
| `tests/test-backend-ops.cpp` | the duplicate-id cases stay commented | the delivery enables them (`test_mul_mat_id_dup`) | **theirs** — upstream's `#if 0` predates `#32505`; the path is implemented on both sides now, so the delivery's standing oracle stays enabled |

## R08.2 **D4 — resolved as MERGE: the two norm fusions are additive, and the shipped order already decides precedence**

`DECISIONS.md` D4 asked which mechanism owns "norm followed by a MUL".  The answer is in the r327 tree, not in a
choice: the delivery's narrow-row matcher `ggml_cuda_norm_rows_try_fuse()` sits **~1000 lines ahead** of every
generic `can_fuse(RMS_NORM, MUL[, ADD])` branch in the dispatcher (line 2199 vs 3314/3319 of
`ggml-cuda-try-fuse.cu`), so where it matches it wins, and upstream's generic branch is the fall-through.

Both delivery matchers bail on `rms->op != GGML_OP_RMS_NORM` and claim only narrow rows (ncols <= 256,
>= 4096 rows, F32, single-use).  `#32454` adds (a) the **layer-norm** `GGML_OP_NORM` variant — a node type the
delivery never claims — and (b) a `mul`/`add` epilogue on the general `norm.cu` kernel.  **Nothing is deleted and
no precedence changes**; at this rung the delivery's matcher is not even reachable (its dispatch is block 10's
TU), which is why `GGML_CUDA_DISABLE_NORM_ROWS=1` is inert above.

**Obligation (R10):** when block 10 puts the matcher into `ggml-cuda-try-fuse.cu`, it must stay **ahead** of
upstream's generic norm branches — that position *is* the D4 resolution — and `scripts/gate-norm-rows-alias.sh`
must show both arms still firing at R18.

## R08.3 Four fixes the compiler or the merge forced (all recorded, none silent)

1. **The 64-bit warp mask, now inside block 08's own code** (third time in this climb: R01's all-reduce, R05's
   band code).  Five sites: `set-rows.cu`'s `k_set_rows_q8_0_group`, `quantize.cu`'s chunked q8_1 reductions
   (×3), `mmvq.cu`'s new kernels (×4), `topk-moe.cu`'s `topk_moe_xor` fallback.  All converted to
   `ggml_cuda_shfl_xor` / `ggml_cuda_shfl` (same lane mapping, `GGML_CUDA_FULL_THREAD_MASK` default), so no
   numerics change — confirmed by the unchanged coherence hash and the op matrices.
2. **`k_set_rows_q8_0_group` needs the F32-source gate.**  `#32387` added **BF16** to `SET_ROWS`, so
   `set_rows_cuda_quant` is now instantiated with `src_t = nv_bfloat16`; the delivery's group kernel takes
   `const float *`.  The `if constexpr` now requires `std::is_same_v<block_type, block_q8_0> &&
   std::is_same_v<src_t, float>`, so a BF16 source keeps the generic kernel (which is what the delivery's own
   path did).
3. **An auto-merge artifact that would have mis-dispatched silently.**  In `mmq.cuh` upstream's
   `mul_mat_q<…, has_fusion, /*force_tiling=*/true>` launch survived the auto-merge **into the delivery's 6th
   template slot**, which is `has_gate` — i.e. it compiled and would have launched the gated arm with
   `has_gate=true, force_tiling=false`.  Fixed by giving the template **both** parameters in a documented order
   and re-pointing that launch at `<…, has_fusion, has_gate, true>`.  **Class of trap:** an auto-merge inside a
   template *parameter list* re-binds positional bools without any conflict marker — check every instantiation
   whose argument list is identical in shape but differs in meaning.
4. **`topk-moe.cu` lost its `#if` guard.**  Block 08 wraps `group_mask_warp_inplace` in
   `#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000`; the merge took the `#endif` (it sat inside the
   conflicting region) and dropped the `#if` (which sat in common context before it).  Restored before
   `group_mask_warp_inplace`, with upstream's `calc_experts_per_thread` placed after it (unguarded, as upstream
   has it).

## R08.4 D11 and later-rung notes

- **The few-row quantizer (`#32065`) is inert on HIP**, exactly like the few-row MMQ it feeds:
  `ggml_cuda_should_use_mmq_few()` returns `false` for every HIP build, so `quantize_mmq_few_id_q8_1_cuda` is
  never reached here and block 08's chunked/scatter quantizer is unopposed.  Recorded, not assumed.
- The routed-compact MoE MMQ dispatch (`GGML_CUDA_DISABLE_MMQ_ROUTED`) and the fused gate arm
  (`has_gate`) are block 08's own; both are graded by `MUL_MAT_ID` above and by the 35B-A3B arms later.  Their
  fusion-aliasing gate (`gate-norm-rows-alias.sh`) stays inert until R10, as block 08 says.
- `norm.cu` now carries three parallel norm surfaces (upstream's fused `norm_f32`, the delivery's `rms_norm_*`
  decode-band fusions, and `norm-gated.cu`'s narrow-row kernel).  R10's dispatcher is where the precedence
  between them lives; do not "simplify" one away without the prefill-logit gate.

---

# R09 — block 09, host-resident MoE (iface, scheduler, Meta, slab and cache engine)  ·  **DONE 2026-11-21** — `24cecfb51`, tree `b05ee8543`

```
git cherry-pick 123d53419        # 70 files, +18251/-540 (moe-expert-cache, moe-cache iface, loader, bench)
```

The climb's highest *exposure* (136 upstream commits on the block's files) but only 4 conflicted paths — the
block adds new TUs rather than rewriting upstream's, so most of the 70 files merged.  The two `DECISIONS.md`
items (**D8**, **D9**) are both `MERGE` and neither needed a choice.

| exit criterion | result |
|---|---|
| conflicts | 4 files / 7 regions (`ggml-backend.cpp`×2, `ggml-cuda/common.cuh`×2, `ggml-cuda/ggml-cuda.cu`, `src/llama-kv-cache.cpp`) |
| no markers left | yes; no `TODO(rebase)` |
| build | green, `all` targets, 0 errors, **0** warnings in the touch-path files |
| coherence 4B | **`100 chars sha=1c5d32ac537d`** — unchanged, and unchanged again after the `ggml-backend.cpp` split-condition change recorded in R09.3 |
| `test-backend-ops -o MUL_MAT_ID` | **1079 OK / 0 FAIL** |
| the block's headline arm (35B-A3B Q8_0, 1 GPU, `-ncmoe 20 -c 8192`, VMM build) | arena default **1268.8 / 54.4** `73 chars sha=86b7f9b5aa80`; that hash is **identical to r327's** on the same arm |
| **one finding** | R09.3: with the arena off (`MOE_EXPERT_CACHE_MIB=0`) this tree reads different text (`34 chars sha=22584470ac60`) where r327 reads the same text as its arena arm; the cause is block 09's **host-offload relaxation**, proven by the `GGML_MOE_HOST_OFFLOAD=0` control |

## R09.1 The four resolutions

| path | upstream | delivery (block 09) | resolution |
|---|---|---|---|
| `ggml-backend.cpp` (1) | `bool split_has_nodes = false;` | `int cur_moe_layer = -1;` (+ comment) | **both** |
| `ggml-backend.cpp` (2) | `… && split_has_nodes` | `… && split->n_inputs > 0` + `int next_moe_layer = -1;` | **the delivery's condition**, plus `next_moe_layer`; upstream's `split_has_nodes` dropped.  Reason: r327 — the validated reference — uses `split->n_inputs > 0`, and the delivery's own note (issue #129, next to this code) says the host-weight pass may only start a new split *once the split already has an input to reuse*.  Upstream's test is true one node earlier (set as soon as a non-view node joins), so it moves the split boundary the host-expert staging keys on.  Measured: the choice does not change the text of either arm (both readings are identical under the two conditions), so this is the conservative, reference-matching choice — see R09.3 for what *did* move the text |
| `ggml-cuda/common.cuh` (1) | `#include "ggml-common.h"` | two read-once diagnostic-flag helpers | **both** |
| `ggml-cuda/common.cuh` (2) | `__reduce_add_sync(GGML_CUDA_FULL_THREAD_MASK, x)` | `__reduce_add_sync(0xffffffff, x)` + the `GGML_CUDA_DPP_XOR`/`ggml_cuda_xor_lane` branch | **both**: upstream's typed mask kept, the delivery's DPP butterfly kept as the `#elif` arm (same association order, so same result) |
| `ggml-cuda/ggml-cuda.cu` | `ggml_cuda_should_fuse_mul_mat_vec_q()` + `ggml_cuda_match_shared_expert()` inline | both **extracted to `mmvq.cu`** | **theirs** (they live in `mmvq.cu`), **but the two functions were then re-ported**: the moved copies predate upstream's `cc < GGML_CUDA_CC_VOLTA` floor (they had `cc <= GGML_CUDA_CC_PASCAL`, which lets Pascal-61 through) and upstream's few-row-MMQ early-out (`#32065`; inert on HIP but needed for cross-backend consistency).  `mmvq.cu` now carries both, plus the `mmq-few.cuh` include |
| `src/llama-kv-cache.cpp` | `cells.seq_rm(seq_id, p0, p1)` (`#32380`) | a manual scan of `used_min()..used_max_p1()` | **upstream's**: `seq_rm` walks the per-sequence position index `seq_pos[seq_id]`, so it is the same answer for strictly less work, and it is the new API's contract |

## R09.2 D8 and D9 — both MERGE, checked rather than assumed

- **D8 (node budget, `#32648`/`#32636`).**  Upstream's `llama_model::graph_nodes_per_tensor()` and its
  `res += 32*n_layer_all` MoE-cache reserve merged cleanly alongside the delivery's KV-rotation leaf accounting
  and the hard "over its node budget" refusal.  Nothing was dropped.  Open question left for R18 (as D8 says):
  upstream's reserve is sized for *their* `--moe-cache`, so **block 09's arena may need its own term** — check it
  with `scripts/gate-node-budget.sh` on a `-ncmoe` arm, not on a dense one.
- **D9 (indexer key rotation, `#32653`).**  `llama_model::lid_k_hdmd()` is in the tree and
  `llama-kv-cache.cpp`'s architecture list is gone, as upstream intends.  At R11 the check is that qwen4exp's
  answer is explicit in its model class (not in the cache's list) and that block 11's QSA/indexer work still sits
  on top; `gate-qwen4exp-quant-coherence.sh` is the gate.

## R09.3 FINDING — the arena-off arm reads different text, and the cause is block 09's own offload relaxation

The block's documented purity property is that the expert cache changes *placement*, not text.  On this tree
the arena arm reproduces r327 exactly, but turning the arena off does not:

| arm (35B-A3B Q8_0 Q8_0, 1 GPU, `-ncmoe 20 -c 8192 -ub 1024`, 5298-token prompt, `MOE_EXPERT_HEATMAP=load`) | text | prefill / decode |
|---|---|---|
| **this rung**, arena default (auto) | `73 chars sha=86b7f9b5aa80` | 1268.8 / 54.4 |
| **this rung**, `MOE_EXPERT_CACHE_MIB=0` (arena off) | `34 chars sha=22584470ac60` | 1338.0 / 29.0 |
| **this rung**, `MOE_EXPERT_CACHE_MIB=0` + `GGML_MOE_HOST_OFFLOAD=0` | **`86b7f9b5aa80`** | 1305.2 / 42.6 |
| r327 (`bin-r327`), arena default | `86b7f9b5aa80` | 2304.7 / 49.7 |
| r327 (`bin-r327`), `MOE_EXPERT_CACHE_MIB=0` | **`86b7f9b5aa80`** | 1128.3 / 21.2 |

So the divergence is **the host-offload relaxation** (`ggml_cuda_device_offload_op`'s MoE relaxation, which lets a
`MUL_MAT_ID` whose expert bank is in the host tier run on the GPU) taking a different kernel path when no arena
is present — the documented class from `GREEDY-PURITY.md` §89 / `AGENTS.md` §5 ("residency selects the MoE kernel
set and the sets are not bit-identical"), here selected by the *arena* rather than the heatmap.  r327 agrees
because the offload decision was reworked in later blocks (block 10's `expert_host_src()`/
`ggml_cuda_device_offload_op` work, block 15's pool) — the same seam `INVARIANTS.md` r305 and the
`gate-host-pool.sh` offload arm cover.

**Not caused by this rung's merge:** the `split_has_nodes` question was tested both ways and both readings
reproduce the same two hashes; `-ub 512` reproduces it as well; pinning the heatmap does not change it.

**Owed:**
1. **R10** — re-run this triple (arena / arena-off / arena-off+`GGML_MOE_HOST_OFFLOAD=0`) and require **all
   three arms to read one text**, as r327 does.  If block 10's offload-decision rework closes it, record it there
   and the R18 obligation becomes a gate arm rather than a question.
2. **R18** — `scripts/gate-host-pool.sh`'s offload arm plus the short-prompt `GGML_SCHED_DEBUG=2` arm (a <= 31
   token prompt must not place `MUL_MAT_ID` on the CPU, `INVARIANTS.md` r305 items 1-2) is the gate that owns this
   seam; do not ship while an arena-off arm reads different text from the arena arm on the same prompt.

## R09.4 Notes for later rungs

- The block's `MOE_EXPERT_CACHE_MIB=0` is the arena kill switch (parsed in `parse_env()`); unset means AUTO
  (size from free VRAM) and an explicit value below 2048 MiB aborts by design.
- On this box `build-rocm` is `GGML_HIP_NO_VMM=OFF`, so the movable-boundary slab and the arena are armed — the
  block's commit message warns that a default HIP build (VMM on UPSTREAM's default) declines the arena and streams
  the experts instead.  Any host-expert measurement at R10+ must stay on the VMM build.
- The host-expert tier's own numbers at this rung (1268.8 / 54.4) are **not** comparable to r327's (2304.7 / 49.7):
  blocks 10-17 (the fusion dispatcher, the streamed KV, the pool) are still missing, and block 10 alone is most of
  the prefill gap.

---

# R10 — block 10, slab, fusion dispatcher and async input ring  ·  **DONE 2026-11-21** — `f5a090a31`, tree `394e0b708`

```
git cherry-pick 7bbf540d5        # 87 files, +9.8k/-… (ggml-cuda-try-fuse.cu 168 KB new, slab, async ring)
```

The rung that owns the most carried obligations, and the first whose resolution is **structural** rather than a
union: upstream's inline fusion dispatcher and block 10's separate-TU dispatcher are two copies of the same
subsystem, so the tree keeps exactly one.

| exit criterion | result |
|---|---|
| conflicts | 7 files / 11 regions (`fwht.cu`, `ggml-cuda.cu`×4, `mmvq.cu`×3, `rope.cu`×2, `set-rows.cu`, `top-k.cu`×2, `topk-moe.cu`×3) |
| no markers left | yes; no `TODO(rebase)` |
| build | green, `all` targets, 0 errors, **0** warnings in the touch-path files |
| coherence 4B | **`100 chars sha=1c5d32ac537d`** — unchanged (and unchanged after the `needs_sync` repair below) |
| `test-backend-ops -o MUL_MAT_ID` | **1079 OK / 0 FAIL** — only after the repair in R10.2; the first run *aborted* on the assert that guards it |
| **R00.6 obligation 1** (drop block 10's `top_k` hunk) | **satisfied by construction**: the two sides' functions are byte-identical, git kept one copy, and only the *comment* conflicted.  Both attributions unioned (the R00.6 pristine-base evidence and block 10's slab/dsv4 evidence); the `CUDA_CHECK(cudaGetLastError())` the R00 repair added is kept |
| **R06.2 obligation (dense half)** | **CLOSED** — 27B UD-IQ3_S with the default MMB: prefill **1139.4 / 38.0 t/s**, `50 chars sha=ebf2a5a9b00a` (**identical to r327's**).  At R06 this arm aborted at `ggml-cuda.cu:808`; block 10's `ctx.pool_stack_*` is the fix, exactly as the R06.2 finding predicted |
| **R06.2 obligation (routed half)** | **STILL OPEN** — the forced routed arm (`GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1`) still aborts on the same pool-LIFO assert (`ggml-cuda.cu:1433`) where r327 reads `73 chars sha=86b7f9b5aa80`.  Now owed at **R15** (the pool/GTT tier) / R18.  Note it is an A/B override, not a shipped default: on RDNA4 the mask excludes Q8_0 and `routed=0` |
| **R08.2 / D4 obligation** | **CLOSED** — see R10.1 |
| **R09.3 obligation** | **CLOSED** — the triple now reads **one** text: arena `86b7f9b5aa80`, arena-off `86b7f9b5aa80` (at R09 the arena-off arm read `22584470ac60`).  Block 10's offload-decision rework is the fix, as predicted; the `GGML_MOE_HOST_OFFLOAD=0` control is no longer needed to reconcile them |
| **one new finding** | R10.3: the arena arm's **decode fell 54.4 -> 20.4 t/s** and no kill switch recovers it |

## R10.1 The dispatcher reconciliation (the rung's real work)

Upstream 919794500 has `ggml_cuda_try_fuse()` + `ggml_cuda_should_fuse_rope_set_rows()` inline in
`ggml-cuda.cu`; block 10 carries them in `ggml-cuda-try-fuse.cu`.  Keeping both would duplicate every symbol
(and, per block 10's own commit message, previously produced a fallback copy that was "upstream's dispatcher
merged with STALE DUPLICATES of delivery arms" — one of which silently diverged on a state-carrying path).
So: **the TU is the single dispatcher** (which is also what r327 ships), and the merge is *reconciling the TU
forward* to this base:

1. **Which arms were missing?**  A mechanical comparison of `ggml_cuda_can_fuse(cgraph, i, {...})` call sites
   between upstream's inline copy and the TU found **exactly two** upstream arms without a counterpart — and
   they are precisely `#32454`'s `{NORM, MUL}` and `{NORM, MUL, ADD}`.  Everything else in the TU was already
   current, which is what made this rung tractable.
2. **`ggml_cuda_should_fuse_rope_set_rows`**: the TU's copy was block 10's, taken before R04 added the
   `GGML_CUDA_DISABLE_ROPE_SET_ROWS` kill switch and before D3's resolution.  HEAD's version (kill switch +
   the "every {F32,F16,BF16} pair has an arm" comment) replaced it.
3. **`ggml_cuda_can_fuse`**: HEAD's version replaced the TU's wholesale — it is upstream@this-base **plus** the
   delivery's blocks 00-08, so it brings in upstream's NORM+MUL predicate (**D4's arm**), the RMS-norm BF16
   looseness, the `RMS_NORM+SCALE` assert→`return false` and the softcap contiguity guard, **and** the
   delivery's `hc_post_gate` arm that block 10's stale copy lacked.  One TU-only line was kept: the
   `set_rows->type != GGML_TYPE_BF16` guard, because `ggml_cuda_op_rms_norm_mul_rope_fused` still writes F32/F16
   rows only.
4. **The two NORM+MUL dispatch arms** were added to the TU **after** `ggml_cuda_norm_rows_try_fuse()` and next to
   the generic `RMS_NORM` arms — i.e. exactly the position r327 ships, which *is* the D4 resolution: the
   delivery's narrow-row matcher wins where it matches, upstream's general epilogue is the fall-through.
5. `ggml-cuda.cu` keeps the delivery's `ggml_cuda_graph_evaluate_and_capture(... ggml_cuda_graph * graph)` — the
   caller (`ggml_backend_cuda_graph_compute`) resolves the entry once from the keyed cache and passes the
   pointer, which is the delivery's API; upstream's key parameter would have meant resolving it twice.

## R10.2 A `needs_sync` / dispatch disagreement that my own union created (caught by the op test)

`ggml_cuda_mul_mat_id_needs_sync()` must return true **exactly** when `ggml_cuda_mul_mat_id()` falls through to
the sort/scatter fallback — the fallback asserts it, and CUDA-graph capture depends on it.  R10's union adopted
upstream's "MMVF reads src0 in pairs, so an even number of columns" guard in the *dispatch* while keeping the
delivery's `else if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE && AMD)` in the *predicate*, so an odd-`k` F16 arm was
answered "no sync" and then fell through: `GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc)) failed`.
The predicate now carries `src0->ne[0] % 2 == 0` too, with a comment saying why the pair must stay in step.
**Class of trap:** when a union changes a dispatch condition, grep for the predicate that predicts that
dispatch — here the two live ~40 lines apart in the same file, and only the op matrix noticed.

## R10.3 FINDING — the arena arm's decode collapsed at this rung (54.4 -> 20.4 t/s), and it is not the fusions

Same arm as R09.3 (35B-A3B Q8_0, 1 GPU, `-ncmoe 20 -c 8192 -ub 1024`, 5298-token prompt, heatmap pinned):

| arm | R09 | R10 |
|---|---|---|
| arena default | 1268.8 / **54.4** | 1378-1382 / **20.4** (two runs, reproducible) |
| `MOE_EXPERT_CACHE_MIB=0` | 1338.0 / 29.0 | 1334.6 / 28.6 |
| `GGML_CUDA_DISABLE_FUSION=1` | — | 1317.5 / 18.5 |
| `GGML_CUDA_SLAB=0` | — | 1640.9 / 13.8 |
| r327 (all 18 blocks) | 2304.7 / 49.7 | — |

So prefill *improved* (block 10's fusions) while decode lost 63 %, and **no kill switch recovers it** — not the
fusions (18.5), not the slab (13.8).  The text is unchanged throughout (`86b7f9b5aa80`).  The remaining
difference against R09 is block 10's cache/allocator tier (`ctx.pool_stack_*`, the MMB slot lifetime changing
from context-lifetime to per-compute via `ggml_cuda_mmb_yield_caches`), which is exactly what blocks **14**
(slab-transient absorption / streamed KV) and **15** (the pool/GTT store) rework — and r327's 49.7 shows the
shipping set does not have it.

**Owed — and this is a RELEASE BLOCKER (maintainer, 2026-11-21: "the decode regression must be fixed before we
release; that can happen later").**  R15 must re-run this arm; if the decode has not recovered by then, the bisect
starts from `pool_stack` / `mmb_yield_caches` / the MMB slot lifetime, and **R18 does not ship until the arena arm
reads ~50 t/s decode again on this config.**  A green ladder is not permission to release past it.

## R10.4 Notes for later rungs

- `ggml-cuda-try-fuse.cu` is now the *only* dispatcher and carries upstream@919794500 wholesale.  Any later
  upstream fusion arm must be added **there**; `ggml-cuda.cu` must not grow a second copy again.
- `fwht.cuh`, `set-rows.cuh`, `mmvq-moe.cu` and `mmvq-ksplit-body.inc` joined the 64-bit-warp-mask repair list
  (4th recurrence of the `#31617` class) — block 10's new TUs were written against the old API.
- The async input ring (`GGML_CUDA_ASYNC_INPUT*`) and the slab (`GGML_CUDA_SLAB*`) are block 10's and stay
  default-on; R18's gates are `gate-node-budget.sh` (block 10 owns the budget's other half, **D8**) and
  `gate-fusion-alias-copy.sh` / `gate-lora-fusion.sh` for the dispatcher.

---

# R11 — block 11, qwen4exp (model, hyper-connections, lazy reader, QSA/indexer)  ·  **DONE 2026-11-21** — `dbe4fdd40`, tree `2b6aec1f0`

```
git cherry-pick 61168c60b        # 52 files, +6.5k/-… (qwen4exp.cpp +2530, ggml.c +450, lazy reader new)
```

| exit criterion | result |
|---|---|
| conflicts | 5 files / 6 regions (`ggml-rpc.h`, `ggml.c`×2, `llama-memory-recurrent.cpp`, `models/qwen4exp.cpp`, `tests/test-backend-ops.cpp`) |
| no markers left | yes; no `TODO(rebase)` |
| build | green, `all` targets, 0 errors, **0** warnings in the touch-path files |
| coherence 4B | **`100 chars sha=1c5d32ac537d`** — unchanged |
| `gate-qwen4exp-quant-coherence.sh` | **PASS** (reduced run: `QUANTS="IQ3_XXS Q4_K_M"`, `QSA_ARM=0`) — Q4_K_M gather on == off == `e58417e70221`, 0 `////`, no GPU faults.  The IQ3_XXS dir is `GSQ-IQ3_XXS` on this box, so that quant SKIPs by name; section E (the sparse QSA arm) was **not** run this session — owed at R18 |
| **`gate-hc-fusions.sh`** | **RED — 3 numeric arms left their anchors** (R11.2).  The HARD arms are clean (no corruption, no GPU fault); the failures are byte-identity anchors, on the class whose header says byte-identity IS expected |
| D13 (upstream's qwen4exp fixes) | **carried, verified**: `llama-model.cpp`'s read-once of `per_layer_tok_embd_scale` and the `llama-model.h` member came in by auto-merge, and the graph's `ggml_scale` application was already in the delivery's builder; this rung's union supplied the missing piece — the **scale tensor creation + F32 validation** in the PLE table setup |
| D9 (indexer key rotation) | **carried**: `llama_model::lid_k_hdmd()` is the cache's answer and qwen4exp's own class provides it; no architecture list came back |

## R11.1 The five resolutions

| path | resolution |
|---|---|
| `ggml-rpc.h` | `RPC_PROTO_PATCH_VERSION 7` (the fork's table diverges, so a peer must refuse a mismatch) and `GGML_OP_COUNT == 106` — upstream's 100 at this base **+ the delivery's 6**, recounted from `enum ggml_op` (107 entries including `GGML_OP_COUNT`) rather than trusting either side's number |
| `ggml.c` (×2) | the same `GGML_OP_COUNT == 106` in both static_asserts |
| `llama-memory-recurrent.cpp` | **both** guards: `has_recr && ubatch.n_pos == 1 && …` — the delivery's `has_recr` (a context whose recurrent filter matches nothing, i.e. qwen4exp's MTP context, must not spam the warning; issue #273) and upstream's `ubatch.n_pos == 1` (the warning is only meaningful on that path) |
| `models/qwen4exp.cpp` | the PLE table setup: **the delivery's lazy-reader managed path** (the `#if LLAMA_LAZY_READER_POSIX` block that keeps only metadata and `pread`s rows on demand) **plus upstream's FP8 scale**: `per_layer_tok_embd_s = create_tensor(…, TENSOR_NOT_REQUIRED)` and its F32 check, created after the `if/else` so both paths get it |
| `tests/test-backend-ops.cpp` | **both**: upstream's wide-table `F8_E4M3` get_rows case and the delivery's `IQ4_NL` sub-block widths (32/128/160/224 — the qwen4exp indexer key row is 128) |

Then one compiler-driven fix: the graph-input classes (`llm_graph_input_qsa`) are not `llm_graph_context`, so
`params.hparams` (upstream's pre-#31733 spelling) and bare `hparams`/`model` are both wrong there — the surviving
spelling at this base is **`params.model->hparams`**, since #31733 made `model` a member of `llm_graph_params`.

## R11.2 FINDING — `gate-hc-fusions.sh` is red: 3 numeric arms left their anchors

`MODEL=GSQ-IQ3_XXS` (the BF16-hc class), `-ngl 99 -fa on`, 1 GPU, the gate's own anchors:

| arm | chars | sha | verdict |
|---|---|---|---|
| `chain_chain` | 444 | `a512ee68c4b3` | reference (true chain) |
| `mix_op` | 447 | `3f049f67781c` | **FAIL** — anchor `6a102dbc62d6` |
| `comm_op` | 444 | `a512ee68c4b3` | ok (exact) |
| `both_ops` | 447 | `3f049f67781c` | ok (== `mix_op`: the combine is exact with the fused mixer on) |
| `cuda_fuse_off` | 444 | `a512ee68c4b3` | ok |
| `dsv4_pre_op` | 443 | `c8db61d5dcc8` | **FAIL** — anchor `a512ee68c4b3` |
| `dsv4_post_op` | 449 | `6a102dbc62d6` | reported (anchor unchanged) |
| `all_ops` | 449 | `6a102dbc62d6` | reported |
| `ops_no_post` | 443 | `c8db61d5dcc8` | **FAIL** — anchor `a512ee68c4b3` |
| `defaults` | 443 | `c8db61d5dcc8` | ok (== `ops_no_post`: `dsv4_hc_post auto=OFF`) |

Reading: the **combine** op is exact (`comm_op == chain`), the **pre** op and the fused-mixer arm are not, and the
chain itself now reads `a512ee68c4b3` where r327's chain read `6a102dbc62d6` — so the *whole* HC chain moved
between r327 and this base, not just the fused arms.

**Half of this was predicted by the block itself.**  Block 11's own commit message (from the r278 campaign) says:
*"Known drift to record at R17: on this tree the two REPORTED dsv4_pre/post arms no longer agree with the chain on
the BF16-hc class … the HC chain itself moved with the re-base while the fused pre/post ops still reproduce the
old one.  The gate reports (does not fail) them, and its header says byte-identity IS expected on this class — the
soft-arm expectation needs a decision."*  What is **not** predicted is `mix_op` failing: that is the fused
**mixer**, and at r327 (all 18 blocks) the equivalent arm agreed.  So either the mixer's fused op is sensitive to
something a later block restores (block 14/15's BF16 producer cache, block 10's MMB down16 knobs — the gate pins
`LLAMA_HC_BLK16=0 LLAMA_HC_RES16=0 GGML_CUDA_MMB_DOWN16=0`), or this rung's tree has a real mixer divergence.

**Owed (blocking, and it needs the maintainer's call on the anchors):**
1. **R13/R14/R17** — re-run this gate once the tree is complete.  If `mix_op` still differs from `chain_chain`, the
   delivery's HC-purity claim is unproven on this base and the fused mixer must not ship default-on; the bisect
   starts at the mixer's own `hc16`/`blk16`/`res16` knobs (the gate pins them off, so a re-anchor to the r327
   values is the first thing to try).
2. **R18** — the anchors are *planted values from r327*; re-anchoring them on the finished tree is a maintainer
   decision (the gate's header says byte-identity is expected on the BF16-hc class, so a re-anchor must be
   justified arm by arm, not wholesale).
3. **R18** — run the full quant-coherence gate (all four quants + `QSA_ARM=1`); this session ran 2 quants and
   skipped section E for time.

## R11.3 Notes for later rungs

- 64-bit warp-mask recurrence **#4**: `fattn-qsa.cu` (×2), `indexer-score.cu`, `hc-mix.cu` — block 11's new TUs
  were written against the pre-`#31617` API, same repair as R01/R05/R08/R10.
- `LLAMA_LAZY_READER_POSIX` + `llama-lazy-reader.{cpp,h}` are new TUs; the PLE table is the only consumer so far
  and its `TENSOR_SKIP_MANAGED` path must stay out of the load-time WILLNEED prefetch.
- Block 11's `qwen4exp.cpp` (2530 lines) is the delivery's implementation, not upstream's; D13's two upstream
  fixes were carried into it rather than replacing it.  Upstream's own qwen4exp (1514 lines, `build_qsa_sel`
  additive mask, plain-op indexer) remains unmerged by design.

---

# R01..R17 — templates (fill as each is climbed)

Each rung's section gets: the cherry-pick command, the conflicted paths, a numbered resolution per conflict
(what upstream did / what we kept / why), the build result, the coherence hash + rates, the census arm, and
the rung's commit id.

---

### R18 — release ladder (do not start before R17 is green)

1. **`TODO` #352 first:** move the container `web` stage from `tools/ui/` to `ui/apps/chat` (+ `ui/lib/ui`,
   `ui/lib/voice`) and the Priority-1 copy to `ui/apps/chat/dist`, else the container loses its embedded UI
   (the HF-asset cache is already retired — `DECISIONS.md` D14, `WORKLOG.md` 261120.006 — so nothing else
   needs seeding).
2. `scripts/gate-per-block-build.sh --all` (a re-base is an `--all` case), plus `scripts/check-gpu-faults.sh`.
3. The block-owned gates for every block whose code the re-base touched (see the *Pick the gate for what you
   changed* table in `AGENTS.md`).
4. `scripts/gate-prefill-logits.sh --record` on the new build, then the gate itself.
4b. **Squash the R00 repair** (`3ea8b0212`) into block 00 — it is a rung repair, not a block of its own
   (`RUNGS.md` R00.6, obligation 2).
5. `scripts/make-patches.sh` from a **canonical fork rebuilt at the new base** via `scripts/apply-all.sh`,
   then `scripts/make-release.sh`, then `scripts/validate-set.sh`.
6. `scripts/fork-refresh.sh --build --gates --push` (personal fork only), tag, then the worker container
   build (`CONTAINERS.md`).
7. `WORKLOG.md` entry (the release narrative), `patches/README.md` per-block notes, `INVARIANTS.md` for any
   fact the re-base changes, `TODO.md` for anything deferred.
