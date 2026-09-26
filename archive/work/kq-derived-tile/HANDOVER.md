# WIP handover: V3 derived KQ mask on the tile FA kernel

**Status:** implemented + correctness-validated 2026-09-19; **not landed** (a measured decode cost
decides the shape -- see `RESULTS-2026-09-19.md` §6).  **Effort guess:** 1-2 sessions, mostly validation.
**Delivery state at handover:** `v16-ebbb18522-r8` (see `patches/README.md`).
**Develop on** `soar` (3x gfx1201), **verify on** `halo` (gfx1151) and `fingon` (gfx1100).

> **Read `RESULTS-2026-09-19.md` first if you are continuing this.**  This work is DONE and landed as
> `v16-ebbb18522-r9` (a block-15 amendment).  The arm is bit-identical on all three arches (8 KV types
> on gfx1201, 4 each on gfx1151/gfx1100, with tile selected *naturally* on the target arches), it is a
> deep-prefill win on the tile path, and it costs decode nothing: the decode regression it had in its
> first cut came from testing `derived.cell_pos` inside the unrolled KV loop, and hoisting that test to
> once per query row recovered the baseline exactly - so the template split discussed below was never
> needed and the build time is unchanged.  The two invariants that must not be undone are recorded in
> `AGENTS.md`'s block-15 bullet; §2 and §3 of this file remain the map of the code and the three traps.

---

## 1. The gap, in one paragraph

Block 15's **V3** (`LLAMA_KQ_MASK_DERIVED`, default on) drops the materialised `n_kv x n_q` f16
attention mask and derives each cell's visibility in the FA kernel from compact per-cell state.  It is
implemented **only in the MMA FA kernel** (`fattn-mma-f16.cuh`), so the chooser's
`ggml_cuda_flash_attn_ext_supported` (`fattn.cu:846`) **rejects the op outright whenever the tile
kernel would be selected**, and the context-creation probe then disables the knob (since r8 it also
prints exactly why).  The tile kernel is selected for prefill whenever the head is above the per-arch
WMMA cap (RDNA4 576, **RDNA3_5 320**, **RDNA3_0 256**) or when `GGML_CUDA_FA_WMMA_256=0` /
`GGML_CUDA_FA_WMMA_MAX_HEAD` forces it.  That means every **Gemma4** model (head 512) on gfx1100/gfx1151
silently loses the mask-elision memory win.  Goal: implement the derived arm in the tile kernel so V3
works regardless of which FA kernel the chooser picks.

## 2. What is already true (do not re-derive)

* The derived **plumbing already reaches the tile kernel**: `launch_fattn` passes
  `cell_pos`/`tok_lo`/`tok_hi` through `ggml_cuda_kernel_launch` (`fattn-common.cuh:1946`), and the tile
  kernel's signature already has them as three **unused** `const int *` params
  (`fattn-tile.cuh:1133-1136`).  No host-side or dispatch change is needed to feed it.
* The host-side fill already handles **SWA** correctly (`llama_kv_cache::set_input_kq_derived`,
  `llama-kv-cache.cpp:1868`): it computes `swa`/`swa_full` into `tok_lo`/`tok_hi`.  So the derived
  values are correct for a sliding-window model; only the kernel side is missing.
* The derived form is **prefill only**: `kq_mask_derivable` (`llama-kv-cache.cpp:1809`) rejects
  `ubatch.n_tokens <= 8`, so decode/verify keep the packed mask.  Whatever you do must therefore leave
  the decode/verify path *numerically untouched*.
* The reference semantics to copy are the MMA kernel's derived branch,
  `flash_attn_ext_f16_load_mask` (`fattn-mma-f16.cuh`, the `if (derived.cell_pos != nullptr)` block,
  fixed in r7) and the struct `kq_derived_t` (`fattn-mma-f16.cuh:684`).
* The r8 diagnostic already reports the condition in the log, so a misconfiguration is no longer
  silent; do not regress that.

## 3. The implementation (and the three traps)

The mask is read in exactly **one place** in the tile kernel, so the change is small.  The traps are
that two *other* places assume `mask != nullptr` means "a mask exists", which the derived form breaks.

1. **The read + guard** (`fattn-tile.cuh:867`):
   ```cuda
   KQ_acc[...] += (ncols2 > 1 || mask) ? slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;
   ```
   With the derived form `mask` is `nullptr`, so for the `ncols2 == 1` variant the guard evaluates
   **false and the mask contribution is silently dropped**: the attention runs *unmasked*.  The
   condition must become `(ncols2 > 1 || mask || derived)`, and the value must come from
   `cell_pos[k_VKQ_0 + i_KQ]` compared against `tok_lo[j]`/`tok_hi[j]` when derived.
2. **The kernel-config predicate** (`fattn-tile.cuh:1657`):
   ```cuda
   const bool use_gqa_opt = mask && max_bias == 0.0f && ...;
   ```
   The MMA kernel uses `ggml_cuda_flash_attn_ext_has_mask(dst)` (`fattn.cu:173`), which **includes
   `src[5]`**; the tile kernel uses `mask` directly.  Left alone, derived-on-tile selects a *different
   `ncols`* than packed-on-tile, i.e. a different reduction order, and the knob toggle stops being
   bit-identical.  Teach it the same `has_mask` notion.
3. **The safety net is the thing you are removing.**  `ggml_cuda_flash_attn_ext_supported`
   (`fattn.cu:846-857`) currently returns false for derived+tile, which makes the scheduler fall back
   and the probe disable the knob.  Once the kernel implements the arm this check must be relaxed, and
   from then on **a gap in any of the 96 tile instances is silent wrong output, not a safe fallback**.
   That is the reason this work is validation-heavy, not code-heavy.

Reference order for the derived value (must match the packed mask exactly): the packed entry is `0.0f`
when the cell is visible else `-INFINITY`; visible means `cell_pos[k] != INT32_MIN && tok_lo[j] <=
cell_pos[k] && cell_pos[k] <= tok_hi[j]`, and the `oob_check && i >= i_sup` case holds `0.0f` (see the
MMA branch; keep the `slope` multiply and the softcap ordering exactly as the packed path has them).

Optional cleanup: `kq_derived_t` currently lives in `fattn-mma-f16.cuh`; if the tile kernel includes it
too, consider moving it to `fattn-common.cuh` so the two kernels cannot drift.

## 4. Test platforms

| host | arch | GPUs | models present | notes |
|---|---|---|---|---|
| `soar` | gfx1201 | 3x R9700 32 GiB | 9B, 27B, 35B-A3B, Gemma4 12B/31B/E4B, Qwen3.8-Flash-Next | dev box; **ccache is set up** |
| `halo` | gfx1151 | 1 (iGPU) | 9B, 35B-A3B, Gemma4 12B/26B-A4B/31B | head 512 -> tile is the *correct* kernel (cap 320) |
| `fingon` | gfx1100 | 1x 7900 XTX (+ gfx1036 iGPU) | 9B, Gemma4 12B/31B | cap 256 -> head 512 tile is the target case |

**Gemma4 is the natural target and the required SWA gate** (all gemma4: head_dim 512, `swa` 512-1024):

| model | layers | KV/token (f16) | 32k KV |
|---|---|---|---|
| `Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf` | 42 | 168 KiB | 5.3 GiB |
| `Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 30 | 480 KiB | 15 GiB |
| `Gemma4/12B/QAT/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | 48 | 768 KiB | 24 GiB |
| `Gemma4/31B/gemma-4-31B-it-Q8_0.gguf` | 60 | 1920 KiB | 60 GiB |

Start with **E4B** (one card, fast) and validate the full gate set on **26B-A4B** (the r5 reference
cell) and 9B.

**To exercise tile+derived on any model** (needed on `soar`, where head 512 takes MMA under the 576
cap): `GGML_CUDA_FA_WMMA_256=0` (cap becomes 128) or `GGML_CUDA_FA_WMMA_MAX_HEAD=128`.  This is also
how to iterate in seconds on `soar` with the 9B.

## 5. Development loop

```bash
# build (ccache on soar; halo/fingon have no ccache, ~5-6 min for a full FA group)
cd ~/llama.cpp && BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# fast loop: cmake --build build-rocm --target llama-cli llama-bench -j16

# does the derived mask engage?  (the r8 diagnostics)
GGML_CUDA_FA_WMMA_256=0 ./build-rocm/bin/llama-cli -m <model> -ngl 999 -fa 1 -c 8192 \
  -p hi -n 1 --single-turn --no-display-prompt --verbose 2>&1 | grep -i "derived kq mask"
#   "enabled"                          -> the derived op is being used
#   "not supported, set to disabled"   -> the tile kernel rejected it (the bug)
```

Perf A/B harness (writes `tag,depth,derived,pp512,tg128`): `archive/work/kq-mask-derived-ab/data/kq-ab.sh`.
Text-purity: run `llama-cli --single-turn --no-display-prompt` on a `prompts/*.txt` file and hash with
`scripts/extract-generated.py` (do NOT `sed`/`grep` the log; the CLI emits backspace corrections).
Bench protocol for anything decode/MTP related: `benchmarks/mtp-adaptive-methodology.md` (rule 0: pin
reasoning and use `-n 3000` for MTP gates).

## 6. Validation / acceptance criteria

The change is only done when **all** of these hold:

1. **Bit-identity**: for every config, `LLAMA_KQ_MASK_DERIVED=1` and `=0` produce the **same same-seed
   greedy text** (and ideally the same logits).  This is the load-bearing property; the packed path is
   the oracle.  Cover the head-cap cases on gfx1100/gfx1151 (Gemma4 26B-A4B / E4B) and a forced-tile
   head-256 case on gfx1201.
2. **No decode change**: decode/verify keeps the packed mask, so a `tg128` A/B on the tile path must be
   flat within noise.  Measure this explicitly; adding a branch to the tile kernel's hot loop is the
   main regression risk (see §7).
3. **Prefill A/B** (`kq-ab.sh`, `-r 3`): the derived arm should be neutral-to-positive on the tile
   path.  If it is a *net loss* on gfx1100/gfx1151 (likely candidate: the extra `cell_pos` read in the
   inner loop, which the tile kernel does per `(j, i)` with no staging), that is a finding, not a bug;
   decide the default from it.
4. **SWA gate**: gemma4 (sliding window) must be correct, since the derived `tok_lo`/`tok_hi` encode the
   window.  This is a hard gate the delivery already imposes on any kq-mask change.
5. **`test-backend-ops -o FLASH_ATTN_EXT`** stays green (note: it does **not** exercise the derived
   path, since that is a graph-level feature; it only guards the kernel itself).
6. **Full matrix**: 1/2/3-GPU, layer and tensor split, all 8 KV types, and the `W = 1..8` band
   (`GREEDY-PURITY.md` §11/§36).  Also re-run the MTP gates (the delivery's standing requirement for
   anything that touches attention).

## 7. Risk register

| # | risk | why it bites | mitigation |
|---|---|---|---|
| 1 | the `(ncols2 > 1 \|\| mask)` guard drops the mask for `ncols2 == 1` | silent unmasked attention, no crash | change the guard; gate on bit-identity, not on a smoke test |
| 2 | `use_gqa_opt` diverges from the packed path | derived and packed pick different `ncols` -> knob toggle changes output | use the MMA kernel's `has_mask` notion; bit-identity gate covers it |
| 3 | relaxing `flash_attn_ext_supported` removes the CPU fallback | any missed variant becomes silent corruption | treat 96 instances x KV types x splits as the scope; prefer an env-gated opt-in landed first |
| 4 | decode/verify regression from a new inner-loop branch | tile is *the* decode kernel on every arch; r7 and the rejected option-(b) both showed 1-3 % swings from tiny loop changes | hoist the branch out of the hot loop; run the explicit `tg128` A/B (criterion 2) |
| 5 | build-time growth in 96 instances | the r6 work was about exactly this | measure `ggml-hip -j16` before/after (`archive/work/build-time-regression/`) |
| 6 | SWA interaction | gemma4 is the target model and the window lives in `tok_lo`/`tok_hi` | E4B is the smallest SWA gate; run it first |

## 8. The lower-risk alternative (rejected, but worth re-reading)

Instead of touching the tile kernel, let the derived mask **override the WMMA head cap** in the chooser
(derived -> MMA).  Zero risk to the tile/decode path.  Rejected because on gfx1100 head 512 the tile
kernel is **3.5-10 % faster** than MMA at deep prefill (the r5 measurement, `patches/README.md`), so it
would trade prefill for memory and would need its own arch/depth gate; and it silently overrides an
explicit user cap.  Keeping it here as the fallback if the kernel work turns out to be a net prefill
loss (criterion 3).

## 9. Repo conventions you must follow

Read `AGENTS.md` (build/push policy, "Critical facts") before touching anything.  In short:

* `~/llama.cpp` `rdna-boosts` is a **disposable dev tree** (currently r6+r7+r8 changes uncommitted);
  the canonical chains are branches `rdna-boosts-r6` / `-r7` / `-r8`.  **Never push the fork.**
* The delivery is `patches/0000-0015` + `release.json`; regenerate from a canonical chain with
  `scripts/make-patches.sh <worktree> ebbb18522 <tip>`, then `scripts/make-release.sh --tip .. --tree
  .. --release v16-ebbb18522-rN`, then the all-patch (`git diff ebbb18522 <tip> > rdna-boosts-all.patch`)
  and `scripts/validate-set.sh`.  This change belongs to **block 15**.
* Docs that must be updated on landing: `patches/README.md` (header + a dated block-15 amendment),
  `WORKLOG.md` (dated entry at the top), `AGENTS.md` (release/tip refs + the block-15 bullet), and the
  README "VRAM vs prefill" section (which currently documents the MMA-only limitation).
* `llama-cli` **must** be run with `--single-turn` (and `--no-display-prompt` for scripted runs).
  Never run parallel benches on one box.

## 10. Open questions for the next session

1. Is the tile path's derived arm a **prefill win or loss**?  The tile kernel reads the mask per
   `(j, i)` with no shared staging, so the derived form substitutes a `cell_pos` read + compare; the
   MMA experience (r7) says the *loop shape* decides this.  If it is a loss, criterion 3 may push the
   whole effort to a "documented, env-gated opt-in" rather than a default.
2. Is `ncols2 == 1` reachable for a real model on a tile-capable arch (trap 1)?  Find a concrete
   shape, because that is the case that silently loses the mask.
3. Does the tile kernel's `use_sparse`/`oob_check`/`logit_softcap` variant set interact with the
   derived form (trap 3's coverage question)?  Enumerate the instantiated variants
   (`fattn-tile-instance-*.cu`, 96 files) and confirm each path.
4. Should the derived arm be **opt-in via a new env** until validated, or land default-on with the
   bit-identity matrix as the gate?  Given the delivery's purity-first stance, an env gate plus a
   second release that promotes it is probably the safer shape.
