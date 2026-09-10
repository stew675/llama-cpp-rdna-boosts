# V4 — native q8_0 K/V in the flash-attention kernels (F16 staging scratch removal)

**Status: DONE 2026-09-10, OPT-IN (`GGML_CUDA_FA_KV_NATIVE=1`), not in the delivery patch set yet.**
Patch: `patches/0006-v4-native-q8-kv.patch` (6 files, +357/−47; base = W1 0001 + W2 0002 + the kq-mask
oracle 0002-DIAGNOSTIC + V3 0003/0004/0005).  Win: **−744 MiB/GPU on the 4B and −632 MiB (Meta) on the
27B at ctx 204800 / ub 2048 / q8_0 KV**, exactly ctx-linear, at the cost of **−1.7 % prefill** and
**±0.1 % decode**.

---

## 1. Why the scratch exists

With a quantized KV cache the FA launcher staged an F16 copy of the **whole** cache into a scratch
region appended to the FA node, then ran the kernel against that copy:

1. `ggml_cuda_flash_attn_ext_get_alloc_size` (`ggml/src/ggml-cuda/fattn.cu`) asks the selected kernel
   whether it needs F16 K/V (`need_f16_K/V`; MMA always, TILE for every quantized type, VEC only for
   F32) and returns `ggml_nbytes(dst)` + that scratch, so the scratch lives in the compute buffer.
2. `ggml_cuda_flash_attn_ext_get_f16_extra_data` (`fattn-common.cuh`) lays it out after `dst`:
   128-byte pad, `ggml_nelements(K)*2`, `ggml_nelements(V)*2` (V shares K's region when V is a view of K).
3. `launch_fattn` runs `ggml_get_to_fp16_cuda` / `..._nc_cuda` over the whole cache **every call**, per
   layer per ubatch, then rewrites `nb11..nb23` to the F16 layout (head-major, unlike the q8_0 cache's
   cell-major layout).

Measured (4B, `-ctk/-ctv q8_0`, ctx 32768 / ub 8, where the scratch is the whole peak): the TILE node
needs 128.12 MiB and the reserve is 129.55 MiB; with f16 KV the same reserve is 5.47 MiB.  At
ctx 204800 / ub 2048 it is **800 MiB** (K+V for one layer, exactly ctx-linear): q8_0 reserve 1001.13 MiB
vs 256.86 MiB with f16 KV.

Two consumers therefore exist in the *same* buffer: the packed kq mask (V3's target, per-ubatch
`n_kv × n_tps × 2 B`) and this whole-cache F16 staging (V4's target, `n_kv × n_head_kv × DKQ × 2 B × 2`).

## 2. Design

**Dequantize while staging the shared-memory tiles.**  Both kernels already stage K/V into shared as
`half2` and the natural granularity is a 16-byte chunk = **8 elements = exactly a quarter of a q8_0
block**.  So instead of reading an F16 copy, the loaders read the q8_0 source and multiply each group of
8 quants by the block's F16 scale:

```cuda
half d;  ggml_cuda_memcpy_1<sizeof(half), 2>(&d, bp);            // block scale (F16)
const half2 d2 = __half2half2(d);
int8_t q[GGML_CUDA_FA_Q8_CHUNK];
ggml_cuda_memcpy_1<GGML_CUDA_FA_Q8_CHUNK, 2>(q, bp + 2 + off);   // 8 quants
dst[l] = d2 * make_half2(q[2*l+0], q[2*l+1]);                    // 4 half2 out
```

**Why the values are bit-identical** to the F16 scratch: the contiguous conversion the launcher uses
(`dequantize_block_q8_0_f16`, `convert.cu`) computes `__hmul2(make_half2(q0,q1), __half2half2(d))` with
`d` read as the block's F16 scale - one F16 rounding of the exact `int8 × half` product, which is what
the line above computes.  (The strided variant `dequantize_block_cuda` computes `(float)d * (float)q`
and then `__float2half_rn`: the product is exact in F32 - ≤11-bit × 8-bit mantissas - so it also rounds
exactly once and agrees.)  Empirically: the **adaptive-MTP probe reports identical draft acceptance**
(`0.76744`, 66/86, mean len 3.28) on both gates, and that probe is documented as sensitive to
ulp-level differences; same-seed generated text is byte-identical everywhere tested (§4).

**Chunk/block alignment.**  A chunk never straddles two blocks because every chunk starts at a multiple
of 8 elements: the slice/step offsets are 8-element aligned for every kernel config (all MMA
`nbatch_K2`/`nbatch_V2` are multiples of 4 half2 = 8 elements, all TILE `nbatch_K` are multiples of 8 -
checked over the whole config tables), and the loaders derive `blk = el/32`, `off = el%32` from the
absolute element offset, so a slice that starts mid-block (e.g. `el%32 = 8`) is handled correctly.

**One predicate, four users.**  `ggml_cuda_fattn_kv_native_supported(t)` (q8_0, `ne[0] % 8 == 0`, row
contiguous, `FAST_FP16_AVAILABLE`) plus `ggml_cuda_fattn_tile_kv_native(K,V)` (both operands, because the
tile kernel has a single `type_KV` template parameter) are used by:
* `ggml_cuda_flash_attn_ext_get_alloc_size` - decides whether the scratch is reserved;
* `launch_fattn` - decides whether the conversion runs (and sizes the scratch with the *effective*
  need flags, so the two cannot disagree);
* the kernel flags (`use_q8_K` / `use_q8_V`) that select the loaders;
* the tile dispatch (`type_KV = GGML_TYPE_Q8_0`).
The existing `GGML_ASSERT(f16_extra.K != 0)` in the conversion block is the tripwire: a mismatch would
abort instead of corrupting memory.

**Runtime vs compile-time.**  The TILE kernel is already templated on the KV type, so its path is
compile-time (`type_KV == GGML_TYPE_Q8_0`, `need_f16_K/V = K->type != type_KV` handled by the existing
mechanism) - zero cost for the F16/BF16 instantiations.  The MMA kernel is *not* templated on the KV
type (that is the reason the scratch exists at all), so its path is a runtime branch at the top of the
shared loader (`KV_q != nullptr`) plus a `fattn_kv_q8_t {K, V, stride_K, stride_V}` threaded through
`iter`/`process_tile`/the kernel.  The byte strides are passed explicitly rather than derived from
`stride_K = nb11/sizeof(half2)`: the q8_0 cache is **cell-major** (`nb[1]` = `n_head_kv × row_size` =
1088 B for DKQ 256 / 4 heads, `nb[2]` = 272 B per head) while the F16 scratch is head-major, and
`nb11/4` is not guaranteed to be exact for every head size.

**Opt-in, not default.**  A quantized source cannot use `cp_async` (the data has to be transformed), so
the staged tiles are built with synchronous loads and the multi-stage pipeline loses its latency hiding.
That costs ~1.7 % prefill; decode (VEC, already native) and verify (TILE, ~neutral) are unaffected.  Per
the maintainer's rule of 2026-09-10 (**"less than a 2 % loss, saves memory, cannot be closed → opt-in"**)
the gate defaults to **off**.

## 3. Implementation

| file | what |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `GGML_CUDA_FA_Q8_CHUNK`, `ggml_cuda_fattn_kv_native_enabled/supported`, `ggml_cuda_fattn_tile_kv_native`, `fattn_kv_q8_t`, `ggml_cuda_fattn_dequantize_q8_0_chunk`; `launch_fattn` computes `use_q8_K/V`, sizes the scratch with the effective flags and skips the conversion; the shared `fattn_kernel_t` typedef gains the two flags |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | `flash_attn_tile_load_tile_q8_0`; `type_KV` threaded into `flash_attn_tile_iter_KQ`/`flash_attn_tile_iter`; the K and V load sites branch on `type_KV == GGML_TYPE_Q8_0`; the kernel builds `fattn_kv_q8_t` |
| `ggml/src/ggml-cuda/fattn-tile.cu` | dispatch `type_KV = GGML_TYPE_Q8_0` when `ggml_cuda_fattn_tile_kv_native(K,V)` (guarded by `FAST_FP16_AVAILABLE` so the instantiation does not exist on builds without the half2 tile) |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | `flash_attn_ext_f16_load_tile_q8_0`; the shared `flash_attn_ext_f16_load_tile` takes `KV_q/stride_KV_q/el_off` and dispatches (so the F16 path is untouched); `fattn_kv_q8_t` through `iter` (4 loader sites) / `process_tile` / both kernels (main loop + stream-k fixup tail) |
| `ggml/src/ggml-cuda/fattn-vec.cuh`, `fattn-tile.cuh` (kernel) | accept and ignore the two new flags: they are part of the shared `fattn_kernel_t` ABI |
| `ggml/src/ggml-cuda/fattn.cu` | `get_alloc_size`: MMA and TILE use the shared predicates; the ROCm/TILE `need_f16` logic keeps its BF16 arm |

Not covered on purpose: every other quantized type (q4_0/q4_1/q5_0/q5_1/iq4_nl/mxfp4/…), BF16, and
*any* CUDA-side build (the predicate needs `FAST_FP16_AVAILABLE`, which is only defined in the host pass
of a HIP TU - so the native path is HIP-only for now, which also excludes the sparse-gather path that is
NVIDIA-only).

## 4. Validation record (2026-09-10, 3x R9700 gfx1201, `-sm tensor`, `-ctk/-ctv q8_0`, `-fa auto`)

**Reserve matrix** (load only, one GPU job at a time; "off" is the delivered default):

| model | devs | ctx | ub | off | on | Δ |
|---|---|---|---|---|---|---|
| Qwen3.5-4B-Q8_0 | 0 | 204800 | 2048 | 1001.13 | **257.13** | −744.00 |
| Qwen3.5-4B-Q8_0 | 0 | 204800 | 1024 | 901.09 | **129.09** | −772.00 |
| Qwen3.5-4B-Q8_0 | 0 | 204800 | 512 | 851.07 | **65.07** | −786.00 |
| Qwen3.5-4B-Q8_0 | 0 | 32768 | 8 | 129.55 | **5.73** | −123.82 |
| Qwen3.8-27B-Q8_0 (Meta) | 0,1,2 | 204800 | 2048 | 1121.13 | **489.13** | −632.00 |
| Qwen3.8-27B-Q8_0 (Meta) | 0,1,2 | 204800 | 1024 | 961.09 | **245.09** | −716.00 |
| Qwen3.8-27B-Q8_0 (Meta) | 0,1,2 | 204800 | 512 | 881.07 | **123.07** | −758.00 |
| gemma-4-E4B-it-Q8_0 (ISWA) | 0 | 204800 | 2048 | 1078.17 | **452.17** | −626.00 |
| gemma-4-31B-it-qat-Q4_K_XL (ISWA) | 0,1,2 | 204800 | 2048 | 1942.18 | **718.18** | −1224.00 |
| qwen3.8-flash-next IQ4_XS (QSA control) | 0,1,2 | 204800 | 2048 | 3251.39 | 3251.39 | **0** |

The host buffer is unchanged (41.13 MiB on the 4B, 81.13 MiB on the 27B): the scratch was device-only.
The delta is not always the full `n_kv × K/V` size because at ub 2048 the FA node's extra region is
partly overlapped by the activation peak (hence −744 instead of −800 on the 4B, −772 at ub 1024); at
ub 8 the scratch *is* the peak and the delta is exactly the scratch (128 MiB at ctx 32768).

**Coherence** (one binary, gate flipped; same seed, `--temp 0`; generated text compared byte for byte,
`llama-cli` timing footer excluded):

| model | prompt | ub | result |
|---|---|---|---|
| Qwen3.5-4B-Q8_0 | 944 words | 2048 (MMA) | IDENTICAL |
| Qwen3.5-4B-Q8_0 | 62 words | 8 (TILE) | IDENTICAL |
| Qwen3.5-4B-Q8_0 | 944 words, `-ctk q8_0 -ctv f16` (mixed) | 2048 | IDENTICAL |
| Qwen3.8-27B-Q8_0 | 40000 words, ctx 81920 | 2048 | IDENTICAL |
| gemma-4-E4B-it-Q8_0 (ISWA) | 944 words / 40000 words ctx 81920 | 2048 | IDENTICAL |
| gemma-4-31B-it-qat-Q4_K_XL (ISWA) | 944 words | 2048 | IDENTICAL |
| qwen3.8-flash-next (QSA control) | 944 words | 2048 | IDENTICAL |

**Adaptive-MTP gate** (27B inline MTP, `--spec-type draft-mtp`, ctx 32768, 96 tokens, ub 2048):
`draft acceptance = 0.76744 (66 accepted / 86 generated), mean len 3.28` **identical on both gates**;
prompt 1250.55 → 1248.08 t/s, generation 82.98 → 83.32 t/s.  This is the ulp-sensitive probe - identical
acceptance is the strongest available evidence that the staged K/V values match the F16 conversion's.

**Op suite**: `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` - 2/2 backends passed with both gates
(includes the V3 derived-mask cases and the quantized-K/V cases).

**Throughput** (same binary, gate off vs on, interleaved reps, `llama-bench`):

| test | off | on | Δ |
|---|---|---|---|
| 27B pp20480 ub 2048 (3 reps) | 2012.56 / 2013.89 | 1979.38 / 1978.94 | **−1.69 %** |
| 27B tg256 | 39.060 / 39.044 | 39.053 / 39.071 | ±0.0 % |
| 4B pp20480 ub 2048 (3 reps) | 6083.05 / 6081.57 / 6105.04 | 5988.10 / 5992.79 / 5984.31 | **−1.67 %** |
| 4B tg256 | 98.66 / 98.65 / 98.78 | 98.74 / 98.80 / 98.75 | +0.06 % |
| 4B pp2048 ub 8 (TILE-dominated) | 524.5 / 522.7 | 524.2 / 523.8 | ±0.0 % (noise) |

So the TILE-side dequant is free; the whole ~1.7 % is in the MMA prefill path, where the cp_async
pipeline is lost.  A first load-optimisation pass (2-byte accesses for the scale and the 8 quants
instead of 1-byte) did not move the number - consistent with the loss being pipeline, not ALU.

**Not exercised** (documented, not blockers): other quantized KV types (they keep the scratch by
construction); CUDA-side builds (predicate false in the host pass); `use_sparse` (NVIDIA-only, and
`shall_use_sparse` returns false on HIP); prompt-cache/checkpoint restore with the path active;
`--parallel > 1`; a *firing* M-RoPE 2-D derived-mask clause (unchanged by V4, still V3's gap).

## 5. Decision and follow-ups

* **Shipped opt-in** (`GGML_CUDA_FA_KV_NATIVE=1`; unset = off) - the maintainer's rule of 2026-09-10:
  under 2 %, saves memory, gap not closable with reasonable effort in this session → opt-in.
* **The lever for on-by-default** is keeping the async pipeline: stage the raw q8_0 rows into shared
  with `cp_async` (the head-row base and `nb[1]`/`nb[2]` are 16-byte aligned, but a q8_0 block is 34
  bytes and the row length is not always a multiple of 16, so a tail/split path is needed) and
  dequantize shared→shared afterwards.  That costs extra shared memory (the raw rows plus the F16 tile)
  in a kernel whose configs are occupancy 1-2, so it is not obviously a win - measure before building.
* **Other KV types** are mechanical additions of the same chunk decoder (the block layouts differ;
  q4_0/q5_0 need the nibble/bit unpacking, and `dequantize_*` helpers for most of them already exist in
  `fattn-common.cuh` for the VEC path - the arithmetic must be matched exactly, as here).
* If the memory is what matters, the combination **V3 + V4-opt-in** puts the 4B at ctx 204800 / ub 2048
  at **257 MiB/GPU** (from the pristine 6690.40 on qwen4exp, and from 1800.33 pre-V3 on the 4B).
