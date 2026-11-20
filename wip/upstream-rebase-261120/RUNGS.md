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
| R00 | 00 | `de99fa2d0` | 4 (7 regions) | D1 (take upstream's), D2 (carry) | **DONE 2026-11-20** — `4cba0ad32`, tree `a3824ea0e`; coherence `1c5d32ac537d`; MUL_MAT_ID 1078/0 (16 dup); one known-red (upstream's qwen4exp indexer top-k, fixed by block 10 at R10) |
| R01 | 01 | `a94e2214c` | 0 | — | **DONE 2026-11-20** — `14a0414bb`, tree `6566ca3ac`; one rung repair (HIP's 64-bit warp mask); 3-device tensor text `1c5d32ac537d` == R00; pipeline census 3 GPUs |
| R02 | 02 | `cef19b032` | 5 | D5 | not started |
| R03 | 03 | `36da15029` | 1 | — | not started |
| R04 | 04 | `cfe40b5f9` | 2 | **D3** | not started || R05 | 05 | `7362b5958` | 7 | — | not started |
| R06 | 06 | `f0564df7d` | 2 | D11 | not started |
| R07 | 07 | `f5bdbefda` | 1 | D11 | not started |
| R08 | 08 | `fe283f27c` | 9 | **D4**, D3, D11 | not started |
| R09 | 09 | `123d53419` | 4 | D8, D9 | not started |
| R10 | 10 | `7bbf540d5` | 9 | D8, D12 | not started |
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

## R00.6 Known-red: upstream's qwen4exp indexer top-k (fixed by block 10 at R10)

**Symptom.** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (2 devices, `-sm tensor`, no `-ncmoe`, `-c 8192 -ub 1024`)
aborts after prefill starts:

```
ROCm error: invalid argument
  current device: 0, in function top_k_argsort_cuda at ggml/src/ggml-cuda/top-k.cu:313
  hipMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows, ...)
```

**It is not this re-base, and it is not the 512 kernel.**  Measured today:

| arm | result |
|---|---|
| pristine base (`build-base-novmm`, VMM **ON**) | prefill **1594.3 t/s**, no crash |
| R00 (VMM **OFF**) | **abort**, any placement (`-sm tensor`, `-sm layer`, `GGML_GLU_INPLACE=0`, `GGML_CUDA_DISABLE_MMID_512=1` all identical) |
| R00 rebuilt with `GGML_HIP_NO_VMM=ON` (`build-r00-novmm`) | prefill **1683.2 t/s**, no crash |
| r327 (`build-rocm/bin-r327`) | prefill **2101.1 t/s**, no crash |
| instrumented R00 | the failing call is the **qwen4exp lightning-indexer** `ggml_top_k`: `ncols=576 nrows=1024 k=512`, `err_after_sort=no error` → the **2-D copy** is refused, not the argsort; the equal-pitch call (`ncols=k=320`) just before it succeeds.  Under VMM=ON the same op runs with `n_pool=832`, so the *pool size* selects the shape that trips it |

**The delivery already owns this fix.**  Block **10** (`7bbf540d5`) replaced exactly this copy with a
`top_k_gather_rows` kernel, and its comment names the same site and the same refusal:

> `cudaMemcpy2DAsync` is NOT usable for it: a padded 2-D copy into a graph tensor view fails with "invalid
> argument" when the destination sits in the movable-boundary slab … Measured here on the dsv4 indexer's top-k

So it is **red from R00 to R09 and must be re-verified at R10** (a Flash-Next prefill past ~500 tokens, the
two-device `-sm tensor` arm above).  Nothing is patched at R00: re-fixing it here would duplicate block 10 and
put the same hunk in two blocks.

**Consequence for the deferred A/B:** `mm_ids_helper_512_10` needs a Flash-Next prefill to measure its
+4.5 %/+5.2 %, and that prefill is exactly what this rung cannot run.  The kernel's *correctness* is
unverified by text too (the model cannot reach the decode step), so the A/B moves to **R10** with the
known-red, not to a later re-base.  `test-backend-ops -o MUL_MAT_ID` does not cover the shape (its largest
`n_mats` is 28), which is why 1078/1078 does not close it.

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
5. `scripts/make-patches.sh` from a **canonical fork rebuilt at the new base** via `scripts/apply-all.sh`,
   then `scripts/make-release.sh`, then `scripts/validate-set.sh`.
6. `scripts/fork-refresh.sh --build --gates --push` (personal fork only), tag, then the worker container
   build (`CONTAINERS.md`).
7. `WORKLOG.md` entry (the release narrative), `patches/README.md` per-block notes, `INVARIANTS.md` for any
   fact the re-base changes, `TODO.md` for anything deferred.
