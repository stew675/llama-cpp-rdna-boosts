# RESULTS 2026-09-26 — re-base gate: **structurally clean, but the fp8 4B regressed**

Ran on the re-based tree (`fp8-rebase`, tip `bc01a921e`) after the GDN/ssm/doc drop.

| 4B, pp512, gfx1201 | re-based tree | cllm record |
|---|---:|---:|
| **Q8_0** (int8 MMQ, delivery) | **8146 ± 221** | 6177 |
| **F8_E4M3** (`stewfp8-ow.gguf`) | **5607 ± 15** | **7184-7260** |
| fp8 / Q8_0 | **0.69x** | **1.16x** |

**The machine is not the problem** — Q8_0 on the same tree/box is **+32 %** over the cllm-era figure
(the delivery's MMB/GEMM work).  The fp8 path is the problem: it is now *slower than Q8_0*, where cllm
had it 16 % *faster*.

## Where the fp8 time goes (rocprofv3 kernel-trace, pp512, 2 passes = 190.7 ms)

| kernel | ms | share |
|---|---:|---:|
| `mul_mat_fp8_wmma` | 137.1 | **71.9 %** |
| **`fp8_repack_weights`** | **14.0** | **7.3 %** |
| `unary_gated_op_kernel<silu>` | 6.3 | 3.3 % |
| `quantize_fp8_warp<4>` | 6.3 | 3.3 % |
| `gdn_conv_direct_kernel` / `gdn_bf16_scan_cuda` | 5.2 | 2.7 % |
| `flash_attn_ext_f16` | 2.4 | 1.3 % |

## Findings

1. **The fp8 GEMM is the whole story (72 %)** and it is running ~65 TFLOP/s here vs the ~77 TFLOP/s the
   cllm notes recorded — *on the same kernel source*.  Something in the newer base (codegen from the
   delivery's `common.cuh`/`ggml-cuda.cu`, or a changed launch configuration) is costing the kernel.
2. **`fp8_repack_weights` appears to run per call (7.3 %).**  It is meant to be a lazy one-time layout
   change cached in the context (`ggml_cuda_fp8_repack`).  If it re-runs each prefill, that is a
   straight 7 % regression and a likely bug.  Check `ggml_backend_cuda_context::fp8_repack_buf`
   invalidation on the newer base.
3. **The dropped cllm GDN/ssm commits are NOT the cause** — GDN is 3 % of fp8 prefill here.
4. Attention is 1.3 %; this is a GEMM + repack + staging problem.

## Verdict

The re-base is **structurally correct** (13 commits, builds clean, RDNA4-gated, GDN identical to r9)
but **not yet performance-neutral**: the fp8 path must be brought back above the Q8_0 line before the
27B comparison is meaningful.  Gate 2 of `PLAN.md` therefore **FAILS** and blocks Phase 3.

## Next actions

1. A/B `mul_mat_fp8_wmma` on the re-based tree vs the original `~/cllm` build at the same pp512 (same
   box, interleaved) to confirm the kernel regression and isolate whether it is codegen or launch.
2. Fix/confirm the `fp8_repack_weights` caching; target its 7.3 % to ~0.
3. Re-run the 4B gate; only then convert and measure the 27B.
