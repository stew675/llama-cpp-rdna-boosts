# `rdna2-norm-gdn` — optional block `0102`: RDNA2 (`gfx1030`) `RMS_NORM` and `GATED_DELTA_NET` dispatch

**Optional artifact, NOT part of the delivery.**  RDNA2 (`gfx1030`) only: every other arch takes the launches
and kernels it took before.  Nothing here is in `patches/`, `release.json.tree`, `rdna-boosts-all.patch` or the
default `scripts/apply-all.sh` apply.

* `0102-rdna2-norm-gdn.patch` — the artifact.  sha256 `ec5a3797d8a610f569b18f1d925d6e125a6f5928b8c9d8f0028029ea38955312`
* Cut against the r324 default tree `d3babb65a68b83a2669933c902d5d981933d807e` (`923b910eb`).  Applies strictly
  (`git am`) there, producing tree `45255a7622be43b4575b380bf0c3ccf50180defa`, and in numeric order after `0100` + `0101`.

## What it changes

1. **`ggml/src/ggml-cuda/norm.cu`** (`rms_norm_f32`, `rms_norm_f32_dispatch`)
   * On RDNA2, `ncols <= 128` launches `rms_norm_f32<.., 32, 4, false, ..>`: one warp stands in for the four
     warps of a 128-thread launch.  Each lane owns its four strided columns, each stand-in warp is reduced with
     the same butterfly, and the per-warp sums are added in the order `block_reduce`'s cross-warp butterfly adds
     them.  `GGML_CUDA_RMS_NORM_NARROW=0` selects a 128-thread launch instead (the A/B switch).
   * On RDNA2, the plain path (`dst16 == nullptr && store_f32`) takes a kernel specialization without the
     `dst16` branch (`has_dst16 = false`).  Other arches keep the one kernel.
   * **The grouping selector is untouched.**  `GGML_CUDA_RMS_NORM_VEC` is still read only through
     `ggml_cuda_rms_norm_vec()`, so the plain norm and every fused norm kernel stay on one grouping.  An earlier
     cut of this block turned the float4 grouping on by default for RDNA2 inside the dispatch; that put the plain
     norm and the fusions on different groupings and is not in this artifact.

2. **`ggml/src/ggml-cuda/gated_delta_net.cu`**
   * With `GGML_CUDA_GDN_CHUNKED` unset, RDNA2 (no BF16 WMMA, FP32 chunked kernel only) takes the chunked prefill
     only for `H * n_seqs <= 4 && n_tokens > 256`, and the sequential kernel otherwise.
     `GGML_CUDA_GDN_CHUNKED=0|1` still forces either.  Note that on RDNA2 most of `test-backend-ops`' chunked
     shapes therefore run the sequential kernel unless `GGML_CUDA_GDN_CHUNKED=1` is set.

3. **`tests/test-backend-ops.cpp`** — six `GATED_DELTA_NET` perf shapes at the gate edge (8 and 12 heads,
   4 heads x 2 sequences).

## Measured on a Radeon Pro V620 (`gfx1030`, ROCm 7.14, idle card)

Correctness (`test-backend-ops test -b ROCm1`, the r324 cut built with the default HIP flags): `-o RMS_NORM` 121/121 with the default launch, with `GGML_CUDA_RMS_NORM_NARROW=0` and with `GGML_CUDA_RMS_NORM_VEC=1`; `-o GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION` 63/63 with the gate, with `GGML_CUDA_GDN_CHUNKED=1` and with `=0`.

Bytes: the `RMS_NORM` output of the narrow launch and of `NARROW=0` was compared with a default-flags build of the
r321 tree + `0100` + `0101` (the 256-thread launch) on 14 shapes (f32 and bf16, `ncols` 1..132 and 4096, up to
20000 rows): identical on all 14.  The match relies on the compiler contracting `x*x` into the first butterfly add
as it does in the wide form (the comment in the kernel says so): writing the square as `fmaf(x, x, 0.0f)` changed
the bits in a build with `-funsafe-math-optimizations`.  It also held in that build.

`RMS_NORM`, us per call (`test-backend-ops perf`):

| shape | before | 128-thread (`NARROW=0`) | this block | float4 grouping (`VEC=1`) |
|---|---|---|---|---|
| f32 `[128,16384]` | 79.3 | 44.9 | 25.5 | 14.9 |
| bf16 `[128,16384]` | 77.8 | 44.5 | 16.6 | 14.0 |
| f32 `[128,32]` | 5.57 | 5.53 | 5.6 | 5.2 |

`[4096, *]` shapes are not touched (they stay on the 1024-thread scalar launch).

`GATED_DELTA_NET`, `S_v = 128`, us per call, sequential / chunked (`GGML_CUDA_GDN_CHUNKED=0` / `=1`):

| heads | n=64 | n=256 | n=512 | n=1024 | gate picks |
|---|---|---|---|---|---|
| 32 | 60 / 193 | 208 / 588 | 409 / 1126 | 812 / 2160 | sequential |
| 16, `v_repeat=3` (Qwen3.8-27B) | 76 / 237 | 271 / 724 | 560 / 1374 | 1168 / 2707 | sequential |
| 12 | | | 282 / 521 | 551 / 1027 | sequential |
| 8 | | 147 / 172 | 288 / 310 | 569 / 607 | sequential |
| 4 x 2 sequences | | | 285 / 302 | | sequential |
| 4 | 76 / 176 | 126 / 150 | 299 / 240 | 589 / 462 | chunked above 256 tokens |

`H * n_seqs` of 5..7 is unmeasured and takes the sequential kernel.

## Not measured

Any arch other than `gfx1030`; a model-level decode / MTP acceptance run.  The perf numbers above were taken on
the r321 default tree + `0100` + `0101`, in a build whose HIP flags included `-funsafe-math-optimizations`; the
r324 cut was built with the default flags and run through the correctness and byte checks only.
