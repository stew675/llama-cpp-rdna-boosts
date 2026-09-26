# Route 2 implementation plan — native bf16 MMA prefill

**Goal:** make the MMA (prefill) FA kernel compute in **bf16** with a **bf16 K/V cache read**,
deleting the bf16→f16 in-register conversion that costs ~26 % of the kernel (see
`README.md` "Findings (2026-09-18, second pass)").  Target: prefill <= the F16-staging path,
with the F16 scratch gone.

## Premises (validated)

* `v_wmma_f32_16x16x16_bf16_gfx12` runs at **the same rate** as the f16 variant on gfx1201
  (microbench `tools/wmma-bench`, properly clock-warmed: ratio 0.997).  The first measurement
  showed 2.79x purely because f16 ran first on unboosted clocks.
* The bf16-accumulate variant (`__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12`) exists.
* bf16 and f16 are both 2 bytes/elem, so the smem tiles, the byte-copy loader and
  `load_ldmatrix` are byte-identical — only the *element type* and the `mma()` overload change.
* Ceiling already measured: the "f16 cache, no conversion" arm = 219.3 ms vs staged 232.5 ms
  (213.2 FA + 19.5 launcher conversion) => ~6 % *faster* than staged if bf16 WMMA is f16-rate.

## Design

Thread a `bool use_bf16` (RDNA4 only) through the MMA kernel and swap the K/V/Q element type:

```
using kv_t = std::conditional_t<use_bf16, nv_bfloat162, half2>;
```

* `mma_tile_sizes<DV, ncols, use_bf16>` — element type of `T_A_KQ/T_B_KQ/T_A_VKQ/T_B_VKQ` and of
  `T_C_VKQ` when it is a 2-byte tile.  `T_C_KQ` stays `float`.
* `flash_attn_ext_f16` / `_process_tile` / `_iter` gain `bool use_bf16` (last template arg).
* `extern __shared__ half2 tile_Q[]` -> `kv_t`; `tile_K`/`tile_V` -> `kv_t*`; `tile_mask` stays
  `half*` (the mask is type-independent).
* K/V **loader stays as-is** (it is a byte copy): pass `(half2 *)` casts.  For the bf16 kernel the
  `FATTN_KV_NATIVE_BF16` arm must *not* call the converting loader — treat it as the plain copy.
* Q fill: `tile_Q[..] = scale_h2 * make_half2(tmp.x, tmp.y)` -> kv_t conversion (single rounding:
  `make_(b)float162(scale*tmp.x, scale*tmp.y)`-style; decide the exact form against the TILE
  kernel's Q path so prefill and decode agree).
* V-tile zeroing (`make_half2(0,0)` on the gfx11 signed-zero leak fix) -> kv_t zero.
* P: `get_half2(KQ_C[k])` -> bf16 conversion for the VKQ operand.
* `load_ldmatrix_trans`'s `I == 32` branch hardcodes `half2` low/high pack -> make it generic on T
  with a bf16 low/high pack.

### Accumulator types (mirror f16 exactly, minimal structural change)

* KQ: `wmma_f32_16x16x16_bf16` (f32 accumulate) — already exists in `mma.cuh`.
* VKQ: `wmma_bf16_16x16x16_bf16` (bf16 accumulate), mirroring f16's `wmma_f16_..._f16`.  Needs a
  new overload.  NOTE: this is *worse* precision than an f32 accumulate but mirrors the f16
  structure; if the perplexity oracle objects, the follow-up is `T_C_VKQ = float` + f32 accumulate
  (matching the TILE kernel, which already accumulates VKQ in f32).

## Files

1. `ggml/src/ggml-cuda/mma.cuh`
   * generic `load_ldmatrix_trans` for `T` (bf16 low/high pack in the `I==32` branch).
   * bf16 `mma(tile<16,8,bf16> D, A, B)` (bf16 accumulate, gfx12).
   * bf16 `mma(tile<16,16,bf16,SCRAMBLED> D, tile<32,8,bf16> A, tile<16,8,bf16> B)` splitting into two.
   * (f32-accumulate `mma(tile<16,16,float> D, tile<16,8,bf16> A, tile<16,8,bf16> B)` already exists.)
2. `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — the templating above; the `case` host fn gains the flag
   (or a sibling `..._mma_bf16_case`), and the `DECL_FATTN_MMA_F16_CASE` macros emit both variants
   on RDNA4/RDNA3.
3. `ggml/src/ggml-cuda/fattn.cu` — dispatch to the bf16 case when the bf16-MMA gate says so.
4. `ggml/src/ggml-cuda/fattn-common.cuh` — launcher: for the bf16 MMA kernel force native K/V (no
   staging, no conversion) and pass `kv_native_kernel = FATTN_KV_NATIVE_BF16`.
5. `ggml/src/ggml-cuda/fattn-common.cuh` policy — new env gate `GGML_CUDA_FA_BF16_MMA`
   (unset = today's behaviour; `=1` = bf16 MMA).  Temporary, for the A/B; folded into
   `GGML_CUDA_FA_KV_NATIVE` policy if the arm wins.

## Validation

1. Compile: `cmake --build build-rocm --target llama-bench llama-cli -j 16` (~2-4 min per full
   `-j16`; one instance TU is ~86 s).
2. Correctness: `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` (must stay 5951/5951) — but the bf16
   MMA path changes numerics, so the op test needs the arm enabled and its tolerance checked.
3. Greedy purity within the bf16-MMA config: `--spec-type none == draft-mtp` (the f16-staged
   reference is expected to differ — that is the accepted trade).
4. Perplexity vs the f16 reference (the quality gate for the bf16 Q/P/accumulate).
5. Speed: `llama-bench -p 2048,8192 -ctk bf16 -ctv bf16`, gate off vs on, interleaved, r>=5.
6. `rocprofv3` kernel check: FA kernel total should drop to the ~219 ms (f16-cache) region.
