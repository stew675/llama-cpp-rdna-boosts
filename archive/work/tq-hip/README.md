# tq-hip — TQ1_0 / TQ2_0 (ternary) weights on the GPU, bit-identical to the CPU backend

Promoted to **block 17** at `v18-919794500-r356` (contributor PR #358, `@overdoingism`; the campaign was
submitted as `wip/tq-hip/`).  The code is `patches/0017-...`; the block reference is `patches/README.md` block
17; the facts are `INVARIANTS.md` r356; the rule is `GREEDY-PURITY.md` §109; the gate is
`scripts/gate-tq.sh` (`MANIFESTS.md` gate 42); the narrative is `WORKLOG.md` 261124.012.

**What the integration changed in the submission** (all three are in the block, not here):

1. **The accumulation is an unfused mul and add.**  The submitted `tq.cu` accumulated each Q8_K block with
   `fmaf` on the premise that the CPU's `sumf` accumulation is an fma.  A Release clang build of `ggml-cpu`
   emits `vmulps` + `vaddps` for `_mm256_add_ps(_mm256_mul_ps(..), sumf)` and **no `vfmadd`**, so the shipped
   claim ("every NMSE exactly 0") did not reproduce: 25 of 61 `MUL_MAT` cases read ~1e-14, and the split was
   diagnostic — every `k=256` case exactly 0, every `k>256` case not.  Three lines (small / vec / MMA kernels)
   now use the file's own `TQ_FMUL`/`TQ_FADD`; with that, all 305 `MUL_MAT` and all 84 `GET_ROWS` values read
   exactly 0, at no cost (TQ2_0 tg128 181.5 → 180.7 t/s, TQ1_0 198.2 → 197.7, noise).  `GREEDY-PURITY.md` §109.
2. **A strided TQ1_0 weight declines the MMA path instead of asserting.**  The int8-WMMA path converts the
   weight to TQ2_0 flat, which needs a contiguous `src0`; the submission asserted that and would have aborted
   where the honest answer is the small-batch kernel (which takes every batch).  Measured reachable-but-inert
   on every validated split (a tensor-split slab is contiguous), so it is a decline, not a bug fixed in anger.
3. **The claims are restated** to what reproduces here (the commit message and the block reference carry the
   measured numbers, the gfx1201/gfx1100/gfx1151 arms and what is NOT covered).

The submitted patch is kept here as `0001-CUDA-HIP-TQ1_0-TQ2_0-MUL_MAT-and-GET_ROWS-on-the-GPU.patch`
(sha256 `c92485b54f81fba9…`, cut against r353, applied tree `d7e012237`) as the reproduction of the failure in
(1) — `scripts/gate-tq.sh --self-test --build <a pre-fix binary>` is how to re-observe it.

## The submitter's record (unchanged below)


Written by Claude (AI).  One `git am` file made on `v17-919794500-r353` (tip `089f90067`, tree `27db9bfd4`); applied, the tree
is `d7e012237`.

| file | touches | switches |
|---|---|---|
| `0001-CUDA-HIP-TQ1_0-TQ2_0-MUL_MAT-and-GET_ROWS-on-the-GPU.patch` | new `tq.cu`, `tq.cuh`; `ggml-cuda.cu`, `ggml-cuda-try-fuse.cu`, `common.cuh`, `convert.cu`, `dequantize.cuh`, `getrows.cu`, `mmvq.cu` (+966 −2) | `GGML_CUDA_TQ=0` (kill switch: the types are refused again), `GGML_CUDA_TQ_FUSE=0` (kill switch for the decode fusions), `GGML_CUDA_TQ_CENSUS=1` (census) |

## What it does

TQ1_0 and TQ2_0 had no CUDA / HIP path: `supports_op` refused them, the weights stayed in host memory and the CPU backend
ran every matmul over them, so a ternary model (BitNet b1.58 and its conversions) decoded at CPU speed with `-ngl 99`.
This adds `MUL_MAT` (not `MUL_MAT_ID`), `GET_ROWS` and the to_fp16 / fp32 / bf16 conversions.

- **Bit-identical to the CPU backend** (its x86 AVX2 build), for every batch size.  src1 is quantized to Q8_K exactly as
  `quantize_row_q8_K_ref` does (no contraction); each 256-value block's integer dot is split into the CPU's eight 32-bit
  lanes (lane L: the values whose index mod 32 is 4L..4L+3, minus the Q8_K sums of values 32L..32L+31); each lane is
  accumulated over the blocks with one fma by `y.d*x.d`; the lanes are summed in `hsum_float_8`'s order.  HIP's
  `__fmul_rn` / `__fadd_rn` are plain operators that `-ffp-contract=fast` would still fuse, so the kernels use
  `#pragma clang fp contract(off)` with plain operators (nvcc keeps `__fmul_rn` / `__fadd_rn`).
- **Decode** (one src1 column): a vec kernel; every thread block quantizes src1 into shared memory itself (no quantize
  launch), at most two blocks per SM take turns on 64-row groups, eight threads per row (one per CPU lane).  With
  `GGML_CUDA_TQ_FUSE` (on unless 0) one launch also takes: the Q, K and V projections; gate + up + SWIGLU (written as the
  GLU kernel computes it); a matmul + the residual ADD after it.  `graph_optimize` moves the V / K projections directly
  behind the Q projection (Llama and BitNet run Q's RESHAPE + ROPE first), the same nodes on one stream.  Every fused
  result is that of the separate ops on this backend, bit for bit.
- **2..16 columns**: a quantize kernel and a small-batch kernel (dp4a).
- **Larger batches, RDNA4**: int8 WMMA, one K = 32 MMA per block and CPU lane, the accumulator starting at minus the lane's
  Q8_K sum; each warp reads its 16 weight rows and 16 columns straight into the WMMA registers (no LDS); TQ1_0 is first
  converted to TQ2_0 (the same values).  Elsewhere the small-batch kernel takes every batch.
- **Census**: `GGML_CUDA_TQ_CENSUS=1` logs, when the context is freed,
  `TQ_CENSUS dev=<d> mul_mat small=<n> mma=<n> get_rows=<n> vec=<n> vec_fused=<n>` (vec: decode launches; vec_fused: the
  matmuls and ADDs those launches took on top of their first matmul).  Host-side decisions: a replayed graph is not counted.

## Measurements

One R9700 (gfx1201), Linux, ROCm 10.0.0.  Arms: **G** r353 + this commit, **F** the same binary with
`GGML_CUDA_TQ_FUSE=0`, **C** the same binary with `GGML_CUDA_TQ=0` (the CPU backend runs the TQ matmuls: stock behaviour),
**V** the official Vulkan release b12597 (`llama-b12597-bin-ubuntu-vulkan-x64`, the same llama.cpp commit `919794500`,
`-dev Vulkan1` = the R9700).  Models: Llama3-8B-1.58-100B-tokens TQ2_0 and TQ1_0 (brunopio's GGUFs), BitNet b1.58 2B4T
TQ1_0.

**Op level.**  `test-backend-ops -o MUL_MAT -p type_a=tq` 61/61 and `-o GET_ROWS -p type=tq` 16/16 pass; with
`LLAMA_TEST_OP_NMSE=1` every reported error is exactly 0 (305 and 84 values).  A standalone element checker (the CPU
backend, this backend and a scalar emulation of the AVX2 lane arithmetic, element by element, memcmp) found no difference in
several million elements: K 1024..14336, 1..100 columns (all three kernels), random src1 and special blocks (the largest
magnitude twice with either sign, all-zero blocks, subnormal blocks, one nonzero value).

**Model level.**  `G` and `F`: a 1024-token prefix prefilled at `-b/-ub 512`, then 256 teacher-forced positions at widths
1, 16 and 64, each position's full-vocabulary log-probabilities written out: **byte-identical** on all three models x three
widths.  `G` and `C`: the same log-probabilities differ (with `GGML_CUDA_TQ=0` other ops of the graph also run on the CPU),
but `llama-cli -p "The capital of France is" -n 64 --seed 42 --temp 0 --single-turn` gives the same text in G, F and C on
all three models.

**Server** (`llama-server --parallel 2 --jinja`, each slot the model's training context, temperature 0; G, F and V): 21 easy
checks per model and arm, graded by substring or JSON fields: few-shot questions; an easy question after a text that fills a
slot to 90 % (7354 tokens on the Llama models), alone and in both slots at once while the other slot serves short requests;
`response_format` json_schema answers, one at a time and two at once; a chat answer from a tool result.  Every check has the
same outcome in G, F and V (Llama3-8B TQ2_0 and TQ1_0 20/21, BitNet 12/21; the failures are the models', on every arm), no
HTTP error, nothing in the server logs, and G and F give the same text, the concurrent long requests included.

**Census** (the log-prob tool above, Llama3-8B TQ2_0): G `small=448 mma=1338 vec=256 vec_fused=318`; F `small=448 mma=1338
vec=448 vec_fused=0` (per decode graph and layer, 7 matmul launches, the GLU and 2 ADDs become 4 launches).

**Speed** (`llama-bench -fa 1 -p 512 -n 128 -r 3`, order G F C V V C F G per model, mean t/s):

| model | test | G | F | C (stock) | V (Vulkan) | G / V |
|---|---|---|---|---|---|---|
| Llama3-8B TQ2_0 | pp512 | 4567.29 | 4566.11 | 98.97 | 3633.84 | 1.26 |
| | tg128 | 187.22 | 170.99 | 11.04 | 183.64 | 1.02 |
| Llama3-8B TQ1_0 | pp512 | 3981.89 | 3982.16 | 46.67 | 2550.55 | 1.56 |
| | tg128 | 201.03 | 177.22 | 9.46 | 105.02 | 1.91 |
| BitNet 2B4T TQ1_0 | pp512 | 7593.01 | 7579.62 | 73.09 | 4976.40 | 1.53 |
| | tg128 | 40.53 | 39.27 | 8.37 | 32.12 | 1.26 |

BitNet's decode is held by its output layer: `models/bitnet.cpp` multiplies by `tok_embd` directly (the FIXME there), an
F16 tensor in host memory, so the CPU runs the 128256 x 2560 output matmul on every token, on both GPU backends.

## Not tested

NVIDIA (only the HIP build was compiled; nvcc's `-use_fast_math` flushes subnormals, so subnormal values may not stay
bit-identical there); RDNA3 / 3.5 (no MMA path there: the small-batch kernel takes every batch, so prefill is slower);
Windows; several GPUs; `MUL_MAT_ID` (a MoE with TQ experts keeps them on the CPU backend); MTP or other draft decoding; a
CPU reference other than x86 AVX2 (an ARM build accumulates in another order); `GET_ROWS` inside a model (the three models
keep their token embeddings in host memory); decode with K > 41216 (that path quantizes in its own kernel, as the 2..16
column path does).  If the decode fusions should be a separate block, `GGML_CUDA_TQ_FUSE` marks all of them (the
`ggml_cuda_tq_try_fuse` arm and the `graph_optimize` move).
