# The 2-D memcpy fast path is refused inside the slab (fixed in r183, block 10)

A load abort found while setting up the PR #256 assessment, **not** caused by that PR (it reproduces on stock
r182): Flash-Next GSQ-IQ3_XXS with `-sm tensor -ncmoe 48` dies before the first token with

```
ROCm error: invalid argument
  current device: 1, in function ggml_cuda_cpy at ggml/src/ggml-cuda/cpy.cu:479
  hipMemcpy2DAsync(src1_ddc, mc_dpitch, src0_ddc, mc_spitch, mc_width, mc_height, hipMemcpyDeviceToDevice, main_stream)
```

`-sm layer` loads, and `-sm tensor` with `GGML_CUDA_SLAB=0` loads, so the trigger is the **movable-boundary
slab**, and the copy is the qwen4exp PLE conv's weight column: `src/models/qwen4exp.cpp` materialises each tap
with `ggml_cont(ggml_view_2d(..., 1, hc_dim, nb[1], k*nb[0]))`, i.e. a **view of a mirrored model weight**
(`blk.N.ple_conv1d.weight`, F16, `[kern, hc_dim]`, 80 KiB).  Upstream's `cudaMemcpy2DAsync` fast path
(`ggml_cuda_cpy_as_memcpy_2d`, upstream #25057) takes it as `w=2, h=10240, spitch=8, dpitch=2`.

Measured with an instrumented `ggml_cuda_cpy` on gfx1201:

| call | device 0 (before the slab re-place) | device 1 (after) |
|---|---|---|
| buffers | `ROCm0` / `ROCm0`, src alloc 81914, dst 20480 | `ROCm1` / `ROCm1`, same sizes |
| result | succeeds (4 evaluations) | **`invalid argument`** |

Both are the same copy of the same mirrored tensor, so it is the *placement*, not the copy: on device 1 the
tensor is the last thing in its region, and a 2-D copy is validated over whole pitches (the padding of the
last row is touched), which reaches the slab's reserved-but-unmapped tail.

## Fix

`ggml_cuda_cpy_as_memcpy_2d` keeps the fast path only while the padded span (`height*pitch`) fits the tensor's
own allocation, and otherwise falls back to the scalar kernels (bit-identical, and they read only
`width` bytes per row).  Verified: the aborting command now loads and decodes, and its text matches the
`GGML_CUDA_SLAB=0` arm.

## Why a guard and not something narrower

The exact driver rule could not be pinned down (shrinking the 2-D copy by one row, so that even the padded
span fits, still failed, so it is not simply `pitch*height` against the allocation), and the fast path is
**unused by every other model we hold**: an instrumented decode+prefill run of 35B-A3B, 4B Q8_0, 27B
UD-Q4_K_XL and gpt-oss-20b records **zero** 2-D copies.  A conservative guard therefore costs nothing
measurable, while narrowing it further would need the driver's actual rule.

A future model that does reach the path in a slack buffer loses the fast path, not correctness.

## r189 follow-up (2026-10-31) — the reach this note denied does exist, and the fallback is now vectorised

The last sentence above ("a future model that does reach the path in a slack buffer loses the fast path, not
correctness") turned out to matter sooner than expected, and the loss is not free:

* **A shipped model does reach it.**  qwen4exp (Flash-Next) with a **quantized indexer key cache**
  (`-ctk q8_0`) cannot use the fused `INDEXER_SCORE`/`INDEXER_FILL` path (F32/BF16/F16 only), so it takes the
  per-op indexer chain, which makes one refused 2-D copy per layer per prefill chunk (`w=512`, `sp=2048`,
  `dp=512`, `h` = the QSA index-block count, 48 layers).  r183's "0 2-D copies in an instrumented run of
  35B-A3B, 4B Q8_0, 27B and gpt-oss" was true — Flash-Next was simply not in that set.
* **The scalar fallback is pathological on a copy that size**: `cpy_scalar` does three integer divisions and
  three modulos per element, so the refused copies cost **1010 ms** of a 105k-token prefill on gfx1201
  (1433.75 ms of `cpy_scalar<float,float>` in the whole run) and, on the KV campaign's sparse/pool decode
  shape, **5.20 ms/token**.
* **r189** therefore gives the refused case its own kernel, `cpy_2d_vec16` (one 16-byte unit per thread, row from
  the grid): the same copies cost 637 ms instead of 1010 ms.  **The guard itself is unchanged**, so the load
  fix above still stands (re-verified: Flash-Next GSQ-IQ3_XXS, 3 GPUs, `-sm tensor -ncmoe 48` loads and
  generates).

Record: `archive/work/cpy-2d-vec16/README.md`, `benchmarks/2026-10-31-cpy-2d-vec16.md`, `GREEDY-PURITY.md` §76,
`WORKLOG.md` 261031.007.
