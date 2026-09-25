# Task: evaluate and, if it wins, port the RDNA4 GQA-6 decode/verify FA band to gfx1100 (RDNA3_0)

## Context

Repo: `~/llama-cpp-rdna-boosts` (delivery). Current release `v16-84e76d8a2-r4`. The fork checkout is
`~/llama.cpp`, branch `rdna-boosts` (disposable; rebuild from patches if needed). Read `AGENTS.md`
first; it refers to gfx1100 as "fingon".

GitHub issue #45 (reported by @overdoingism) diagnosed a real inefficiency on gfx1201: at head 256
with GQA 6 (Qwen3.8-27B: 24 Q / 4 KV heads) the tile FA kernel can only fold `ncols2 = 2`, because
its `ncols2` must divide the GQA ratio. So the whole `n_q <= 8` decode/verify band fetches and
dequantizes every K/V element once per head pair, three times per query row.

The r4 fix (folded into block 15) routes the whole band to the existing WMMA kernel with the GQA
group folded into `ncols2 = 8`, and splits the KV round-robin over a fixed `P = nsm` blocks per output
tile so decode and every verify width accumulate identically. It is gated RDNA4-only, so **your arch
is not covered and still runs the tile kernel with `ncols2 = 2`**.

Why gfx1100 looks promising: the gfx1201 win was instruction-issue bound, and gfx1100 is a discrete
GDDR6 card (96 CU, about 960 GB/s), the same class. Head 256 is exactly at the RDNA3_0 WMMA cap
(256), so it passes `Q->ne[0] <= wmma_max_head`.

Key code (all paths relative to `~/llama.cpp/ggml/src/ggml-cuda/`):

- `fattn-common.cuh`: `ggml_cuda_fattn_band_wmma_applies()` (the gate:
  `if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) return false;`), and the band branch in
  `launch_fattn`.
- `fattn-mma-f16.cuh`: the `#if defined(AMD_WMMA_AVAILABLE)` band fast path in `flash_attn_ext_f16`
  (`gridDim.y > 1`) and the `kb0_step` parameter of `flash_attn_ext_f16_process_tile`.
- `fattn.cu`: the band check in `ggml_cuda_get_best_fattn_kernel` and the band route in
  `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`.
- `common.cuh`: `amd_wmma_available()` is true for RDNA4 and RDNA3.
- The `(256, ncols1 2|4, ncols2 8)` WMMA instances exist in `template-instances/` for all targets.

Two things not to confuse with this task:

- This arch keeps the stock AMD `ncols2` rule for the **normal** (non-band) path under tensor split
  (the 2026-09-18 issue #30 fix). The band overrides that for the band only.
- gfx1100 has a separate documented beta issue: the 16-wide routed `mul_mat_vec_q_moe` band broke
  `MUL_MAT_ID` 23/929 on RDNA3_0, which is why the beta patch floors that band at 8. That is MMB,
  unrelated to flash attention.

The r4 eval and perf cases for this shape (`test_flash_attn_ext(256, 256, 4, {6, 1}, ...)`) are
already in `tests/test-backend-ops.cpp`.

## Build

Use this host's `~/bin/build-llama-rocm-714` (per-host copy) with `BUILD_DIR=build-rocm-beta`,
`GPU_TARGETS=gfx1100`, ccache on. Report `rocminfo | grep gfx` and the script's `ROCM_714` /
`GPU_TARGETS` in your write-up.

## Step 1: reproduce and measure, do NOT change the committed gate yet

1. Baseline (tile): `test-backend-ops perf -b ROCm0 -o FLASH_ATTN_EXT -p 'nh=4,nr23=.6,1.,kv=16384'`
   for q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl, f16, bf16 at nb = 1, 3, 5, 8.
2. Temporarily relax the gate to `if (!amd_wmma_available(cc))` (one-line local experiment), rebuild,
   and repeat the identical run. The chooser, the ncols dispatcher and `launch_fattn` all share the
   predicate, so this one edit enables the band consistently.
3. A/B `ncols1` (`GGML_HIP_FA_BAND_WMMA=2` vs `4`) and `P` (`GGML_HIP_FA_BAND_WMMA_SPLIT`; the default
   is `nsm`, which is 96 here, while the reporter's flat region was 48..96 at 64 CU, so verify it
   transfers).
4. Eval with the relaxed gate:
   `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT -p 'nr23=.6,1.'` must be 389/389. If not, stop and
   report the failing case.

If the 27B (about 17.6 GB Q4_K_XL) is tight with long context on 24 GB, do the op-level work first
(it needs no model) and use the 27B at a modest context for the end-to-end gate. `Qwen3.5-4B` is GQA 4
and does **not** exercise this path, so it cannot substitute for the shape.

## Step 2: decision

If the band does not win, write a negative result and leave the committed gate RDNA4-only. If it
wins, port it properly.

## Step 3: port (only if it wins)

- Include RDNA3_0 in the gate, default-on, opt-out `GGML_HIP_FA_BAND_WMMA=0`; keep
  `GGML_CUDA_FA_WMMA_256=0` / `GGML_CUDA_FA_WMMA_MAX_HEAD<256` disabling the band, and keep the
  `ncols2 == 8` template guard.
- Pick `ncols1` and `P` from your data; check whether the WMMA cap interaction or the tensor-split
  `ncols2` rule needs a comment.
- Keep the change scoped: it must not alter the generic head>128 WMMA path or the normal `ncols2`
  selection.

## Step 4: contracts (mandatory)

- `test-backend-ops -o FLASH_ATTN_EXT` all pass plus the 389 GQA-6 cases.
- `plain == draft-mtp` byte-identical for all 8 native KV types, 1 GPU and 2-GPU tensor/layer.
- q8_0/q4_0 identical across `--spec-draft-n-max 3/5/7`.
- Prefill flat (`llama-bench` pp512/2048/4096, band on/off).
- Sanity: `MUL_MAT_ID` and the other oracles unchanged.

## Deliverable

A session report with the tables, tuning, purity results, and either a block-15 gfx1100 amendment
patch or a documented negative result. **Do not push.**
