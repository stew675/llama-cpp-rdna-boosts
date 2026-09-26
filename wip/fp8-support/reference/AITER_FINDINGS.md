# AITER inspection - what to lift for the fp8 prefill performance gap

**Session**: 2026-08-06 (afternoon), continuation of HANDOVER.md
**Repo inspected**: `/home/stew675/aiter` (ROCm/aiter @ 22beb1caa, branch main)
**Goal**: find liftable assets that close the gap to vLLM-class performance
for the Qwen3.5-4B FP8 E4M3 prefill (current: 4930 t/s pp512 vs Q8_0 5847 t/s).

---

## 1. What aiter is (and why it is the right reference)

AITER = AMD's AI Tensor Engine for ROCm. It is the **default kernel backend
for vLLM on ROCm** (MHA/MLA/paged attention, fused MoE, GEMM, RMSNorm, etc.)
and is what the target numbers in this project are effectively based on: the
user's vLLM benchmark runs on this box with these kernels.

gfx1201 (Radeon AI PRO R9700, our GPU) is an officially listed supported arch
(experimental). On RDNA the CK/ASM paths are CDNA-only; the working paths are
**Triton JIT kernels** (JIT-compiled per-arch) plus some HIP kernels.

## 2. The exact asset: fp8 blockscale GEMM == our op

`aiter/ops/gemm_op_a8w8.py` -> `gemm_a8w8_blockscale()` +
`aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py` +
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py`.

Semantics match our llama.cpp fp8 mul_mat **1:1**:
- A = fp8 activations [M, K], pre-quantized, per-token scale per 128-k block
  (`x_scale [M, K/128]`) == what our `quantize_fp8` staging produces
- W = fp8 weights [N, K] row-major, per-128x128 block scale
  (`w_scale [N/128, K/128]`) == our `block_f8_e4m3` scale grid
- y = sum over k-blocks of dot(fp8, fp8) * a_scale * w_scale, acc in f32, out bf16/f32

On gfx1201 the dispatch falls to the **Triton** kernel (CK code objects are
CDNA-only). The Triton kernel is a plain staged wmma GEMM (tl.dot with
`matrix_instr_nonkdim=16` = wmma 16x16x16 fp8), K-tile 128, kpack 2.

AMD tuned it for gfx1201 on real hardware: PR #3228 (2026-05-19)
"Add gfx1201 fp8 gemm configs" = 30 config files under
`aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE*.json`.
The generic default (`gfx1201-GEMM-A8W8_BLOCKSCALE.json`) applies to our
shapes (our N/K have no specialized file). Per-M-bound selection (M_LEQ_x).

## 3. Measured performance on this box (this session)

Environment: py3.12 venv `/tmp/aiter-venv` (ROCm torch 2.12.0+rocm7.1 +
triton 3.7.0), `PYTHONPATH=/home/stew675/aiter`, jax stub for import (see
section 8). Bench scripts: `/tmp/bench_aiter_gemm.py`,
`/tmp/bench_aiter_stable.py`, `/tmp/bench_aiter_preshuffle.py`.
100 iters, min of 3 runs, one sync.

M=512 (pp512), our exact Qwen3.5-4B shapes (hidden 2560, inter 9216, 16x256
heads, kv 4, vocab 248320):

| GEMM      | N      | K    | TFLOP/s |
|-----------|--------|------|---------|
| q_proj    | 4096   | 2560 | 121.2   |
| k_proj    | 1024   | 2560 |  89.4   |
| v_proj    | 1024   | 2560 | ~89     |
| o_proj    | 2560   | 4096 | 123.4   |
| gate_proj | 9216   | 2560 | 136.9   |
| up_proj   | 9216   | 2560 | ~137    |
| down_proj | 2560   | 9216 | 124.5   |
| lm_head   | 248320 | 2560 | 128.7   |

Also: M=256 -> 38-133, M=128 -> 22-91 TFLOP/s. AMD-tuned shapes
(M=512, N=8192 K=8192 / N=16384 K=1536 / N=2112 K=7168): 125.1 / 124.9 /
131.2 TFLOP/s. Raw gfx1201 wmma ceiling is ~370 TFLOP/s, so aiter runs at
~35% of raw - typical for a staged kernel with fp8 smem traffic.

**Negative result (important):** the (16,16)-preshuffled blockscale path
(`gemm_a8w8_blockscale_bpreshuffle`) is **5-8x SLOWER** on gfx1201
(14-17 TFLOP/s vs 64-137). The shuffle layout + fallback small tiles are
gfx1250-oriented; on gfx1201 the **row-major non-preshuffled path wins**,
which is exactly our weight layout. Do NOT adopt the shuffle layout.

**Decode (M=1):** aiter's M_LEQ_8 configs use wmma + split-K (NUM_KSPLIT
4-16, tiny 16x16 tiles, 2-kernel reduce). Measured 45-61 us per GEMM, but
that is dominated by Triton/python launch + torch alloc overhead - not a
kernel-time number and not representative of what we would get with a native
launch. Our native dot4 GEMV remains the right decode path; generation is
memory-bound anyway (see section 6).

## 4. Our kernel vs aiter - the measured gap

rocprofv3 kernel trace of `llama-bench -p 512 -n 8 -r 1 -dev ROCm0`
(`/tmp/rocp/soar/*kernel_trace.csv`), normalized per pp512 run
(~104 ms/run at 4929 t/s):

| kernel            | ms/run | share |
|-------------------|--------|-------|
| mul_mat_fp8_wmma  | ~63.6  | 61%   |
| gated_delta_net_cuda (linear attn) | ~17.9 | 17% |
| mul_mat_vec_f (bf16, non-fp8 tensors) | ~13 | 12.5% |
| flash_attn_tile   | ~3.7   | 3.5%  |
| quantize_fp8      | ~1.3   | 1.2%  |
| norms/silu/rope/concat/copy | ~5 | 5%  |

So: fp8 GEMM = 61% of prefill. Our wmma kernel runs at **~55 TFLOP/s**
effective (63.6 ms for 512 * 6.73 GFLOP/token = 3.45 TFLOP) vs aiter's
**~120 TFLOP/s average** on the same shapes: a **2.2x structural gap**.

Projection if our GEMM reaches aiter's rate (everything else unchanged):
~104 -> ~69 ms -> ~7,400 t/s pp512 = **+26% over Q8_0** (5847). Combined
with the two secondary levers below (delta-net kernel, non-fp8 tensors) the
+50% target is reachable: ~+45% with the mmv fix, ~+77% with all three.

## 5. The liftable recipe (from the gfx1201 tuned configs, M_LEQ_512)

Our kernel (fp8.cuh): CTA 32 rows x 128 tokens, **4 warps**, single-buffered
smem (sync between k-blocks), linear pid order, sA[32][132]+sB[128][136].

aiter M_LEQ_512 config: `BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128,
GROUP_SIZE_M=4, num_warps=8, num_stages=2, waves_per_eu=4,
matrix_instr_nonkdim=16, kpack=2, NUM_KSPLIT=1`.

Concrete port:
1. **CTA 128 weight rows x 64 tokens, 8 warps** (4 m-warps x 2 n-warps, each
   warp keeps its 2x2 grid of 16x16 wmma tiles). Staging bytes per 128-k
   block: sA 128x128 = 16 KB + sB 64x128 = 8 KB = 24 KB, i.e. ~87 FLOP/B
   vs our 51 FLOP/B (1.7x better staging reuse).
2. **Double-buffered smem, 2 stages** (aiter num_stages=2): 2 x 24 KB = 48 KB
   <= 64 KB LDS, still 1 CTA/CU, but now 256 threads/CU = full occupancy.
   THIS is why our earlier 2x-smem experiment failed: we kept 4 warps and
   dropped to half occupancy. 8 warps + 2 stages keeps 100%.
3. **GROUP_SIZE_M=4-8 grouped pid swizzle** (L2 weight reuse across the m
   dimension; cheap: reorder the m-block id).
4. K-tile 128, wmma 16x16x16 fp8, weight scales per 128x128 (all unchanged).
5. `waves_per_eu` = 2-4 launch hint (grid-sized); weights loaded with `.cg`
   (L2-only, they are reused across CTAs).
6. M-bound dispatch: M<=8 (decode) uses split-K wmma; M=16-64 -> 32/64-row
   tiles; M>=128 -> 128x64. Keep our GEMV for M<=16 (better than their
   split-K at M=1 natively).

The aiter Triton kernel source (see section 2 file list) is the reference
implementation; its structure is fully portable to hand-written HIP (the
triton JIT runtime itself cannot run inside llama.cpp).

Secondary levers found in aiter (beyond the GEMM):
- **gated delta rule (linear attention)**: aiter has
  `csrc/kernels/chunk_gated_delta_rule_fwd_h.cu` with explicit
  `__gfx1201__` support + `op_tests/test_gated_delta_rule.py`. Our
  `gated_delta_net_cuda` is 17% of pp512; aiter's version is the vLLM path.
  It is a separate op (different data flow than llama.cpp's), so the lift is
  a port, not a swap; but it is the biggest non-GEMM prefill cost.
- **Non-fp8 tensors**: our BF16 tensors (token_embd 248320x2560, norms,
  in_proj_a/b, conv1d, A_log) stay bf16 in both the safetensors dir and the
  `--outtype fp8_e4m3` GGUF; they cost ~12.5% of pp512 via mul_mat_vec_f and
  dominate decode memory traffic. Quantize them (Q8_0) in the converter =
  HANDOVER section 8 lever 1, unchanged by this inspection.

## 6. Decode (generation)

Generation is memory-bound (fp8 GEMV 337 GB/s vs Q8_0 mmvq 381 GB/s). aiter
does NOT change this: at M=1 the weight bytes read per output row dominate,
and their wmma+split-K choice is not a bandwidth win over our dot4 GEMV
(which is also launch-cheaper). The decode levers remain: (a) quantize the
BF16 tensors to shrink the model, (b) MTP drafting (already works). No aiter
lift needed here.

## 7. Latent bug found in our launcher (fp8.cu)

`n_pad` pads n to a multiple of 64 (`GGML_FP8_NWARPS * GGML_FP8_TILE_N`), but the
wmma CTA covers **128** tokens (CTA_N = 128, grid `(n+127)/128`), and the sB
staging load is unmasked. Whenever the last CTA's 128-token span exceeds
`n_pad` the kernel reads past the `k*n_pad` staging allocation: n in [17,64]
(n_pad=64, one CTA covers 128) and every n mod 128 != 0 above 128 (e.g.
n=150: n_pad=192 but CTA 1 covers tokens 128-255). All llama-bench power-of-
two batch sizes >= 128 are safe, which is why it was never caught; real
llama-server batch prefill sizes mostly are not. Correct output today only by
pool-adjacency luck + store masking. Fix while touching the launcher: pad n
to a multiple of 128 (CTA_N), or mask/limit the staging loop to n_pad.

## 8. Reproducibility notes

- aiter import needs a jax stub (gluon module imports jax) and a py3.12 venv
  (the prebuilt `module_aiter_core.so` in the repo is py3.14-linked and does
  not load under py3.12; rebuilding it in the venv works: needs pybind11,
  psutil, packaging, numpy in the venv).
- The Triton path is the only gfx1201 GEMM path; CK/ASM/FlyDSL-wmma are
  CDNA/gfx1250 only on main. FlyDSL wmma tune files
  (`aiter/ops/flydsl/gemm_tune/*wmma_common.py`) are gfx1250 references.
- The 30 gfx1201 configs in PR #3228 were tuned for other models' shapes
  (DeepSeek/Qwen-MoE sizes); our shapes use the M-bounded defaults. After the
  port, per-shape tuning for OUR N/K is available headroom (expected: gate/up
  class shapes reach ~137 TFLOP/s, q/o ~123).

## 9. Recommendation (ranked, for the next session)

1. Port the aiter recipe into `mul_mat_fp8_wmma`: 128x64 CTA, 8 warps,
   2-stage smem pipeline, grouped-M swizzle, .cg weight loads. Expect GEMM
   55 -> ~110-130 TFLOP/s -> pp512 ~7,400 t/s (+26% vs Q8_0).
   This is a kernel-only change in fp8.cuh/fp8.cu; no format changes.
2. Then tackle the linear-attention cost (17%): either port aiter's gfx1201
   gated delta rule kernel or profile/optimize our `gated_delta_net_cuda`.
   Together with #1 this reaches the +50% prefill target.
3. Fix the n_pad OOB read while touching the launcher.
4. Converter: quantize the BF16 non-fp8 tensors in the fp8 GGUF (decode + pp
   memory win; independent of kernels).
5. Do NOT adopt the (16,16) weight shuffle; do NOT switch decode to wmma.

## 10. RESULTS (2026-08-06 afternoon) — port done, measured

Port implemented and committed as `823f70a3` (see git log). Final kernel:
128x64 CTA, 8 warps (4m x 2n, 2x2 wmma tiles), grouped-M swizzle, and the key
finding below. Measured on the same build/GPU (pp512, -r 3):

| config                          | pp512 t/s | vs baseline |
|---------------------------------|-----------|-------------|
| baseline 821d423ac              | 4929.8    | -           |
| + 128x64/8w/double-buffer       | 5110      | +3.7%       |
| + burst fragment loads          | 5260      | +6.7%       |
| + single-buffer smem (2 CTA/CU) | 5655      | +14.7%      |
| final (pipelined frags, same)   | 5546-5596 | +12.5-13.5% |

GEMM kernel rate: 55 -> ~77 TFLOP/s (rocprof: wmma 113.9 -> 80.6 ms per
bench run). tg64 unchanged (71.0-71.4) - decode untouched, as intended.

### Key finding: occupancy, not the double buffer, was the lever

The aiter recipe's real win on gfx1201 is NOT the 2-stage smem pipeline: it is
that the kernel fits **2 CTAs per CU**. Device props on gfx1201:
`smem/block = 64 KB, regs/mp = 196608 (3 CTAs by regs), maxThreads = 2048`.
The double-buffered 128x64 tile needs 52.7-61 KB LDS -> 1 CTA/CU (8 warps);
the aiter kernel's 24 KB tile runs 2 CTAs/CU (16 warps) and its 121 TFLOP/s is
exactly the occupancy effect (measured: 8 warps/CU = 58 TFLOP/s vs 16 warps =
77 TFLOP/s). Single-buffered smem (26.4 KB) + register-staged pipelining gets
the same 2-CTA occupancy with the loads still overlapped with compute.

Failed experiments (do not retry): 16 warps/CTA with 2x1 tiles per warp
regressed to ~50 TFLOP/s (fragment-reuse loss); GROUP_SIZE_M=8 was noise;
wmma 16x16x16 fp8 raw ceiling on this GPU is ~165 TFLOP/s (microbench), so
77 TFLOP/s = 47% of ceiling.

### Compiler bug found (LLVM, rocm clang 20)

The explicit `fp8x8_t fa[4][4]` burst-fragment array (all 32 fragments loaded
up front, then a register-only wmma chain) MISCOMPILES under the single-buffer
loop: 16 of 32 v_wmma instructions get `v[0:1]` as both operands (garbage),
with zero LDS loads in the chain. Same source compiles correctly in the
double-buffered variant. Reverted to the software-pipelined fragment loads
(one k-step lookahead) which is correct and measures within noise of the
burst. If a future burst-style change is needed, use explicit named fragment
variables instead of the array.

### Other discoveries

- The GGUF has **token_embd.weight stored as BF16** (type 30; F8_E4M3 = 43 in
  this fork's enum), not fp8. The lm_head (m=248320) therefore runs the fused
  BF16 mmv path (output_norm + lm_head fused, 2.05 ms/launch) - this is
  efficient and should stay as-is; converting token_embd to fp8 would put
  lm_head on the wmma path at ~5.4 ms (slower). The "quantize non-fp8
  tensors" lever from section 6 is therefore mostly about the small BF16
  tensors (norms, conv1d, in_proj) and is a minor win.
- The n_pad OOB read (section 7) is fixed by the CTA_N=64 tile (the staging
  pad now matches the CTA span exactly).
- Remaining gap to +50% over Q8_0 (8770 t/s): pp512 now ~5550 (+12.5%). The
  next lever is the linear-attention kernel (`gated_delta_net_cuda` = 17% of
  pp512; aiter's gfx1201 gated delta rule kernel exists in this repo).
  Kernel-side, remaining ideas: 3 CTAs/CU via 64x64 tiles (2-acc warps
  regressed, so uncertain), and weight-layout repack at load time for
  16-B-aligned staging loads.

Standing data: baseline pp512 4929.78 t/s, tg64 71.37 t/s (build 821d423ac);
Q8_0 pp512 5847, tg64 89.5.
