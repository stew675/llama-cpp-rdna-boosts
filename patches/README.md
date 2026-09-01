# rdna-boosts patch set (delivery)

13 patches against the llama.cpp fork point `0eadefebd`
("qwen4exp: support recurrent state rollback (#28123)"; re-based
2026-09-01 from `a7cc83bba`):

| patch | content |
|---|---|
| `0001` | adaptive MTP draft depth |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA; + MTP long-prefill chunked-prefix + sequential K-tail, PR #9) |
| `0003` | BF16 KV cache + native-BF16 flash-attn |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL |
| `0012` | **hybrid HIP all-reduce (block 12)** - the custom internal AR; hybrid dispatch; RDNA4-only gate |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split (block 13)** - prefill fused expert MMQ (RDNA4, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K) + decode item-split; see block 13 notes below |

## Apply (fresh checkout at the fork point)

```bash
git checkout 0eadefebd          # or: git apply each patch on a matching tree
git am patches/000[1-9]-*.patch patches/001[0-3]-*.patch
```

(`git am` for the whole 13-patch series - plain `git apply` of the
concatenated series was observed to silently drop hunks; use `git am`.)

The set is **whitespace-clean**: applying produces no git whitespace
warnings (verified 2026-08-29 after the whitespace-clean regeneration,
re-verified 2026-09-01 on the `0eadefebd` re-base, re-verified 2026-09-01
with block 13 on the 13-patch series).

## Block 12 notes

- **RDNA4-only gate**: the internal all-reduce refuses to init on any
  architecture other than gfx1200/gfx1201 (the pipeline falls back to the
  default RCCL path with a warning).  Community verification on RDNA3 pairs
  is pending; remove the gate's arch check once verified.
- Env knobs (defaults preserve upstream behavior):
  - `GGML_CUDA_ALLREDUCE=hybrid|nccl|internal|none` (hybrid = default on Linux)
  - `GGML_CUDA_AR_PROFILE=1` — per-call spin/phase profiler at teardown
  - `GGML_CUDA_AR_SLEEP=0|1` — s_sleep poll vs dummy spin (default 1)
  - `GGML_CUDA_AR_BF16_THRESHOLD` — F32->BF16 wire round-trip threshold
  - `GGML_CUDA_AR_COPY_THRESHOLD` / `GGML_CUDA_AR_COPY_CHUNK_BYTES` — CE path
  - `GGML_CUDA_AR_SPIN_TIMEOUT_MS` — bounded in-kernel peer-arrival spin
    budget (default 20 ms, `0` = legacy unbounded); on timeout the kernel
    sets a host-mapped poison flag, skips the reduce and exits, and the host
    re-syncs the devices via a butterfly AllReduce on the next call
  - WIP experiments (archived, env-gated OFF by default): `GGML_CUDA_AR_FUSED`,
    `GGML_CUDA_AR_PACE` — see `../archive/work/fused-stage-pacing/README.md`
- Verified 2026-09-01 re-base (3x R9700, ROCm 7.14, gfx1201): clean apply
  on a fresh checkout at `0eadefebd` + full build + llama-cli same-seed
  coherence IDENTICAL + tg64 38.12 / tg512 41.08 (sim build; numbers
  unchanged — the re-base is code-identical to the 2026-08-30 set).
  Depth-16384 decode 38.71 t/s (3-GPU hybrid, unpinned) with the server
  config `HIP_VISIBLE_DEVICES=0,1,2`.
- **Community-report fix round (2026-08-30, issues #5 + #6, reporter
  tungel):** two block-12 fixes integrated into the fork and regenerated
  into this set:
  - `-DGGML_HIP_RCCL=OFF` builds now compile — `comm_init_hybrid`'s
    `try_allreduce_nccl` reference is guarded by `GGML_USE_NCCL` (was an
    unconditional reference to an `#ifdef`-guarded function: build error).
  - The chunked AR kernel's in-kernel peer-arrival spin is now bounded
    (`GGML_CUDA_AR_SPIN_TIMEOUT_MS`, default 20 ms, `0` = legacy).  An
    unbounded spin on RDNA (non-preemptible compute kernels) could wedge
    the queue -> MES `REMOVE_QUEUE` timeout -> MODE1 reset -> `700/719`
    or a whole-machine freeze; on timeout the kernel poisons a host-mapped
    flag, skips the reduce and exits (queue stays removable), and the host
    re-syncs the devices with a butterfly AllReduce on the next call.
    The budget check is decimated to 1-in-512 polls (2x the measured
    typical spin count — p50 184 / p90 283 / p99 369 polls on 2x gfx1201
    hybrid at depth-16384 — rounded up to a power of two), merged with
    the arrival check into a single per-poll branch (a tick without a
    timeout keeps polling the same peer), so the true fast path never
    executes `clock64()` at all and the timeout overshoot stays <0.1% of
    budget.
  - Re-verified 2026-08-30: clean-apply sim at `17252c769` (apply, full
    build, llama-cli same-seed coherence IDENTICAL); RCCL=OFF `ggml-hip`
    compiles; before/after perf on the default hybrid config (2x R9700,
    depth-16384) shows NO measurable impact — pp512 1622.7 -> 1609.1 t/s
    (-0.8%, within noise), tg128 32.63 -> 32.55 t/s (-0.25%, within
    noise).  Note: regenerated from the current `~/llama.cpp` fork
    (rdna-boosts tip `8a426cf79`); commit hashes in the 01-11 patch
    headers drift from the earlier records (the fork was rebuilt; diff
    content is unchanged).
- **Compiler-warning clean** (2026-08-29 follow-up): ROCm 7.14 marks
  `hipError_t` `[[nodiscard]]`, and the original HIP port left 27
  unchecked HIP calls (all `-Wunused-value` in the ggml-hip build).  All
  27 now go through `CUDA_CHECK` (upstream house style, incl. teardown);
  three dead WIP items removed.  The ggml-hip build emits ZERO warnings
  from this patch.
- **AR_PROFILE devices[] init fix** (2026-09-01, PR #8, integrated):
  `ggml_cuda_ar_pipeline_init` now copies the caller's `devices[]` into
  the pipeline BEFORE the per-device profiler hipMallocs.  With
  `GGML_CUDA_AR_PROFILE=1` the buffers were allocated while `devices[]`
  was still zero-filled, so every prof buffer landed on GPU 0 and MTP's
  second pipeline init (draft context) faulted GPU 1 (gfx1201).  A/B on
  3x R9700 (2-GPU, internal AR, MTP n-max 3, `-c 32768`): pre-fix
  reproduced the fault (`Memory Fault Error ... GPU index: 1, kernel:
  ggml_cuda_ar_kernel<float, __hip_bfloat16>`); post-fix runs clean with
  teardown dumps on dev0 AND dev1 in both pipelines, same-seed coherence
  IDENTICAL to the pre-fix golden.  Default serving (profiler off) is
  unaffected.  Do not ship `AR_PROFILE=1` as a daily env — this only
  makes the debug flag safe.
- **MTP chunked-GDN prefix folded into block 02** (2026-09-01, PR #9,
  integrated): block 02's chunked WMMA GDN used to launch only for
  `K == 1` (no MTP snapshots) — with MTP n-max 3 (`K=4`) every prefill
  ubatch stayed on the sequential kernel.  Long single-sequence MTP
  prefills (`K > 1`, `n_seqs == 1`, `n_tokens > K+64`) now run the
  chunked GDN on the prefix (`n_tokens - K`) and sequential GDN only on
  the last K tokens so slots `0..K-1` stay correct (fused-cache graphs
  included; `n_seqs > 1` stays fully sequential).  The chunked ops take
  an `n_tokens_limit` parameter.  Opt out: `GGML_CUDA_GDN_CHUNKED=0`
  (also `GGML_CUDA_GDN_CHUNKED_BF16=0`).  Verified 2026-09-01 on 3x
  R9700 (2-GPU, internal AR, Qwen3.8-27B Q8, ubatch 1024, MTP n-max 3):
  path fire `n=1024 K=4 prefix=1020`; prefill tok/s +7.5% (~5.5k prompt)
  / +7.7% (~38k) vs sequential; 64-token same-seed output token-identical
  to sequential; non-MTP coherence unchanged.  Not bit-identical vs
  sequential in general (same class as the bf16 chunked: near-lossless).
  Lab numbers: `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.

## Block 13 notes

**Fused MoE gate+up+GLU MMQ (prefill) + mmvq short-K item-split (decode).**

- **Prefill fused expert MMQ** (block 13, the `mul_mat_id_glu_ops` pattern):
  the {MUL_MAT_ID(gate), MUL_MAT_ID(up), GLU} triple runs as ONE MMQ kernel
  reading both weight streams with a GLU epilogue in registers.  Types
  instantiated: Q3_K/Q4_K/Q5_K/Q8_0/Q6_K (M4 quant extension).  Env opt-out:
  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`.
  Validated (1-GPU qwen35moe Q6_K/Q4_K_M, the verified config): prefill
  pp16384 Q6_K +5.1% (3344 vs 3181), Q4_K_M +3.6% (3488 vs 3367); fused
  path fires as `FUSED MUL_MAT_ID ffn_moe_down-*` on all layers.
- **Decode item-split** (mmvq `mul_mat_vec_q`/`mul_mat_vec_q_moe`): the
  K-split loop leaves most thread groups idle on short-K MoE GEMMs (down
  K=512 -> 2 K-blocks); the item-split loop spreads (row, kblock) items
  over the groups and scales rows_per_block (rpb 2/4/8) to fill them.
  Re-based on top of the upstream `has_fusion` mmvq path (41ef91f7c),
  which landed in the 0eadefebd re-base - the launcher now dispatches on
  rpb x has_fusion.  Validated: decode tg128 97.28 vs 92.15 pristine
  (+5.6%, 1-GPU Q6_K).
- **Correctness gates** on the try_fuse arms (the 0eadefebd merge
  admitted cases the kernels could not express):
  1. The `x_scale_channel_dst` fold (MoE down x topk weights) now
     supports multi-token MUL_MAT_ID (2026-09-02): the `mul_mat_vec_q_moe`
     epilogue applies `x_scale[channel_dst + token_idx*nchannels_dst]`, one
     scalar per (expert, token), matching the topk-weights layout
     [1, n_expert_used, n_tokens].  try_fuse gates on the weights shape
     (`weights->ne[2] == mm_node->ne[2]`); the launcher assert allows
     nelements == ne1*ne2.  Spec-dec verify batches n=2..8 (up to
     `get_mmvq_mmid_max_batch`) now fuse instead of the separate MUL.
     Validated: test-backend-ops 16222/16222 (multi-token n=4 exercises
     the moe kernel, bit-exact vs CPU ref; sweep extended with
     Q8_0/Q6_K/Q5_K/Q3_K/IQ2_XS); single-token decode unchanged
     (tg128 82.8-83.2).
  2. The fused MoE MMQ arm is gated to the instantiated type list
     (Q3_K/Q4_K/Q5_K/Q8_0/Q6_K): `ggml_cuda_should_use_mmq` returns true
     for q4_0/q4_1/q5_0/IQ/MXFP4/NVFP4 on RDNA4, which would abort in
     `ggml_cuda_mul_mat_q_switch_type_gate`.  MXFP4/NVFP4 support tracked
     in TODO.md.
- Same-seed coherence IDENTICAL (fusion on vs off), test-backend-ops
  2/2 OK.
- 2026-09-01 (block-13 amendment): fixed the ROCm multi-GPU split-load
  pathology this block's qwen35moe validation exposed.  H2D 2D copies
  with a width not multiple of 4 (Q6_K/Q3_K quant blocks are 210/110
  bytes) take ~1000x longer on ROCm (~2300ms vs ~10ms per tensor), so
  Q5_K/Q6_K 2-GPU tensor-split loads took ~3min and looked like hangs
  (the earlier "hangs at EVERY commit" finding was a misdiagnosis -
  every build was just slow-loading).  `set_tensor_2d` now stages
  through device memory with an aligned width + unaligned D2D gather
  (byte-identical, memcmp 0).  Q6_K/Q3_K 2-GPU load now <15s, pp512
  ~4200-4500 t/s and tg32 ~66-82 t/s matching the pre-regression docs
  numbers; Q8_0 unchanged.  The slow-load is present in pristine
  upstream 0eadefebd too (upstream bug, worth filing); the fix ships
  here because the feature it unblocks (qwen35moe 2-GPU MoE) is
  block-13's.
- **Benchmark configs (2026-09-02):** all 1-GPU numbers in this project's
docs require `HIP_VISIBLE_DEVICES=0`; without it llama.cpp layer-splits
across all 3 R9700s and decode drops ~97 -> ~81 t/s (a harness artifact,
NOT a regression - verified 2026-09-02). Canonical command lines + the
baseline table live in `wip/qwen35moe-prefill/bench-config.md`.

## Server config (the +22% deployment win)

`HIP_VISIBLE_DEVICES=0,1,2` (3-GPU), hybrid default, **unpinned** (the
dpm=high/runtime-PM pin is a regression: tg -5-7%, pp -15-18% on RCCL/hybrid
paths).  Depth-16384: 31.79 (2-GPU) -> 38.71 (3-GPU) t/s.
