# UPSTREAM-PR: ggml-cuda — `GET_ROWS` on an `iq4_nl` row that is not a whole QK_K super-block

**Status:** prepared 2026-09-13; applies clean to master `790cf51aa` (`git apply --check`); validated on
gfx1201/ROCm 7.14 against the delivery (block 08, sixth amendment).  Not filed.

**Where it lives in the delivery:** inside `patches/0008-rdna-boosts-block-08-fused-core-prefill-kernels-and-.patch`
(the 2026-09-13 amendment, TODO item 3).

## The bug

`ggml_backend_cuda_device_supports_op()`'s `GGML_OP_GET_ROWS` case lists the 32-value sub-block types
(`IQ4_NL`, `MXFP4`) under a `ne[0] % QK_K == 0` requirement, with the comment that "the row size does
not guarantee the QK_K super-blocks the get_rows kernel iterates on".  That is true of the *kernel the
case was wired to* (`get_rows_cuda_kq<..., dequantize_iq4_nl>`, which dequantizes whole `QK_K`
super-blocks and computes `nsb = ne00/QK_K`), but `iq4_nl` is a 32-value block type — it has a per-block
dequantizer (`dequantize_q4_nl`, the same one the FlashAttention staging uses) and a matching
`get_rows_cuda_q<QK4_NL, QR4_NL, ...>` path.  Any `iq4_nl` `get_rows` whose row width is not a multiple
of 256 is therefore **rejected by the HIP backend and scheduled on the CPU**.

The reachable case today is a quantized KV-cache type on qwen4exp: its QSA indexer key cache tracks
`type_k`, so `--cache-type-k iq4_nl` gives the indexer key gather an `iq4_nl` source with
`idx_dim = 128`.  The scheduler then runs one `GET_ROWS` node per indexer-bearing layer on the host:
the prefill graph goes from 2 to **26** CPU/GPU splits, each with a `hipStreamSynchronize` and a
D2H/H2D round trip.  Measured on 3× R9700, 3-GPU `-sm tensor`, qwen4exp `IQ4_XS`: qwen4exp prefill
pp8192 **1815-1951 -> 2385-2422 t/s** (f16 2348-2416, `q4_0` 2316-2413) and pp32768 **+36 %**; the GPU
busy fraction 0.62 -> 0.96.  The dense-shortcut arm hides it below the indexer selection width (2051),
which is why shallow benchmarks are flat.

## The fix

`ggml/src/ggml-cuda/getrows.cu`: dispatch `iq4_nl` on `ne00 % QK_K` — whole super-blocks keep
`get_rows_cuda_kq<32, ..., dequantize_iq4_nl>`, any other width takes
`get_rows_cuda_q<QK4_NL, QR4_NL, dequantize_q4_nl>`.

`ggml/src/ggml-cuda/ggml-cuda.cu`: accept `IQ4_NL` whenever `ne00 % QK4_NL == 0` (every legal `iq4_nl`
row).  `MXFP4` keeps the `QK_K` requirement — it has no sub-block dequantizer.

`tests/test-backend-ops.cpp`: four `iq4_nl` `GET_ROWS` cases at 32/128/160/224 columns (the sub-`QK_K`
widths the suite never tested).

## Validation

* `test-backend-ops test -o GET_ROWS`: **219/219** (was 215; the four new cases run on the GPU and match
  the CPU).  With `max_nmse_err()` temporarily forced to 0 for `iq4_nl`, the new path is **bit-exact**
  against the CPU at 32/128/160/224/256/512/1024 columns.
* The delivery's gate set (qwen4exp `plain == draft-mtp n_max 3 == n_max 7`, the `W = 1..8` probes, the
  f16/`q4_0`/4B controls) all hold; the 4B coherence is byte-identical (`1c5d32ac537d`).

## What was NOT validated

* No NVIDIA/CUDA hardware was available — the change is backend-generic CUDA source, but the
  `get_rows_cuda_q<QK4_NL, ...>` instantiation was only compiled and run as HIP/ROCm.  Compile it for
  CUDA before filing.
* Other backends (Vulkan/Metal/SYCL/CPU) are untouched and already handled `iq4_nl` `get_rows`
  correctly (the CPU is the reference).
* The absolute output of a config whose graph layout changes can move, because
  `ggml_cuda_check_fusion_memory_ranges()` is address-overlap-driven and the fused MoE router is not
  bit-identical to the generic chain — see the delivery's `GREEDY-PURITY.md` §30 and TODO item 19.  The
  numeric change is a pre-existing upstream property, not this patch's subject; this patch only removes
  the CPU fallback.
