# Task: evaluate and, if it wins, port the RDNA4 GQA-6 decode/verify FA band to gfx1151 (RDNA3_5)

## Context

Repo: `~/llama-cpp-rdna-boosts` (delivery). Current release `v16-84e76d8a2-r4`. The fork checkout is
`~/llama.cpp`, branch `rdna-boosts` (disposable; rebuild from patches if needed). Read `AGENTS.md`
first, especially the Default-on policy, the RDNA3_5 / Strix Halo notes, and the purity rulebook.

GitHub issue #45 (reported by @overdoingism) diagnosed a real inefficiency on gfx1201: at head 256
with GQA 6 (Qwen3.8-27B: 24 Q / 4 KV heads) the tile FA kernel can only fold `ncols2 = 2`, because
its `ncols2` must divide the GQA ratio. So the whole `n_q <= 8` decode/verify band fetches and
dequantizes every K/V element once per head pair, three times per query row.

The r4 fix (folded into block 15) routes the whole band to the existing WMMA kernel with the GQA
group folded into `ncols2 = 8`, and splits the KV round-robin over a fixed `P = nsm` blocks per output
tile so decode and every verify width accumulate identically. It is gated RDNA4-only, so **your arch
is not covered and still runs the tile kernel with `ncols2 = 2`**.

Key code (all paths relative to `~/llama.cpp/ggml/src/ggml-cuda/`):

- `fattn-common.cuh`: `ggml_cuda_fattn_band_wmma_ncols1()`, `ggml_cuda_fattn_band_wmma_split()`,
  `ggml_cuda_fattn_band_wmma_applies()` (the gate:
  `if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) return false;`), and the band branch in
  `launch_fattn`.
- `fattn-mma-f16.cuh`: the `#if defined(AMD_WMMA_AVAILABLE)` band fast path in `flash_attn_ext_f16`
  (`gridDim.y > 1`) and the `kb0_step` parameter of `flash_attn_ext_f16_process_tile`.
- `fattn.cu`: the band check in `ggml_cuda_get_best_fattn_kernel` and the band route in
  `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`.
- `common.cuh`: `amd_wmma_available()` is true for RDNA4 and RDNA3.
- The `(256, ncols1 2|4, ncols2 8)` WMMA instances exist in `template-instances/` for all targets.
- WMMA head caps: RDNA4 576, RDNA3_0 256, RDNA3_5 320, so head 256 passes.
- RDNA3_5 specific: `launch_fattn`'s `prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)` keeps the native
  KV read at prefill; the band's decode/verify path is unaffected by that.

New eval and perf cases for this exact shape are already in `tests/test-backend-ops.cpp` (the r4
patch adds them): `test_flash_attn_ext(256, 256, 4, {6, 1}, kv, nb, ...)` across 8 KV types.

## Build

Use this host's ROCm build script (per-host copy): `~/bin/build-llama-rocm-714`, with
`BUILD_DIR=build-rocm-beta` and `GPU_TARGETS` / `ROCM_714` set for gfx1151. ccache is enabled by
default. Report `rocminfo | grep gfx` and the script's `ROCM_714` / `GPU_TARGETS` in your write-up.

## Step 1: reproduce and measure, do NOT change the committed gate yet

1. Confirm the baseline runs the tile kernel: run
   `test-backend-ops perf -b ROCm0 -o FLASH_ATTN_EXT -p 'nh=4,nr23=.6,1.,kv=16384'` and note the
   us/run for q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl, f16, bf16 at nb = 1, 3, 5, 8.
2. Temporarily relax the gate to allow your arch, rebuild, and repeat the exact same run: in
   `ggml_cuda_fattn_band_wmma_applies` change
   `if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc))` to `if (!amd_wmma_available(cc))`.
   This is a local experiment only. Because the chooser, the ncols dispatcher and `launch_fattn` all
   ask the same predicate, this single edit enables the whole band consistently.
3. Also A/B `ncols1` (env `GGML_HIP_FA_BAND_WMMA=2` vs `4`) and the split `P`
   (`GGML_HIP_FA_BAND_WMMA_SPLIT`; default is `nsm`, which on this iGPU is about 40, not the 64 the
   reporter tuned on). Record the flat region if any.
4. Run the eval suite with the relaxed gate:
   `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT -p 'nr23=.6,1.'` must be 389/389. If it is not, stop
   and report the failing case.

## Step 2: decision

If the band does not win on gfx1151, write up a negative result (op-level table, prefill, why) and
leave the committed gate RDNA4-only. A negative result is a valid deliverable. If it does win, port
it properly in Step 3.

## Step 3: port (only if it wins)

- Relax the gate to include RDNA3_5 (keep it default-on per the repo policy; opt-out is
  `GGML_HIP_FA_BAND_WMMA=0`, and `GGML_CUDA_FA_WMMA_256=0` / `GGML_CUDA_FA_WMMA_MAX_HEAD<256` must
  still disable the band). Keep the `ncols2 == 8` template guard in `launch_fattn`.
- Choose the `ncols1` default from your data (the reporter's gfx1201 default is 4).
- Confirm the `P = nsm` default is right for this iGPU, or add a per-arch default.
- Re-check the native-KV interaction: this arch keeps native reads at decode and prefill, so the band
  reads the cache natively. Make sure the alloc-size query and the launcher still agree.

## Step 4: contracts (mandatory, default-on policy applies)

- `test-backend-ops -o FLASH_ATTN_EXT` all pass, plus the 389 GQA-6 cases.
- `plain == draft-mtp` byte-identical for all 8 native KV types at about 40k context, on this iGPU
  (1 GPU and any multi-GPU split you can run), with healthy MTP acceptance.
- q8_0/q4_0 text identical across `--spec-draft-n-max 3/5/7` (verify widths 4/6/8).
- Prefill flat: `llama-bench` pp512/2048/4096 with the band on and off.
- No change to the non-FA oracles.

## Deliverable

A session report with the op-level and end-to-end tables, the `ncols1`/`P` tuning, the purity
results, and either a patch (folded into block 15 as a gfx1151 amendment) or a documented negative
result. **Do not push.** Ask before touching anything under `wip/` or `archive/`.
