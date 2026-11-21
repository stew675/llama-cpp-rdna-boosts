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
| R07 | 07 | `f5bdbefda` | 1 | D11 | not started |
| R08 | 08 | `fe283f27c` | 9 | **D4**, D3, D11 | not started |
| R09 | 09 | `123d53419` | 4 | D8, D9 | not started |
| R10 | 10 | `7bbf540d5` | 9 | D8, D12 + **drop block 10's `top_k` hunk (R00.6 obligation 1)** | not started |
| R11 | 11 | `61168c60b` | 5 | D9, D13 | not started |
| R12 | 12 | `dad1442b3` | 9 | D5, D6, D7 | not started |
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
