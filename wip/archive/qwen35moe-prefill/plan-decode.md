# PLAN (c): decode path investigation (qwen35moe, 92 t/s)

## Status: Phase 1 DONE (2026-08-30) -> decode-phase1-results.md

The cost table is built and the hypotheses are resolved:
- FA / GET_ROWS / GDN-AR are NOT the problem (FA 1%, AR 5%) - the earlier
  anomalies were an op-timing blind-spot artifact (join-waits polluted the
  main-stream event pairs; fused nodes were never timed).
- Decode is LATENCY-BOUND: 923 kernels/token, 68% under 20 us. Weight
  reads are ~2 GB/token (MoE 8/256) and only ~1/3 of the time is
  bandwidth-bound (lm_head at 600 GB/s near peak).
- The expert GEMMs are the top target: gate+up fused kernel 37 us @427
  GB/s, down+shared-gate 53 us @166 GB/s (K=512 latency-bound).
- Fusion machinery already worth 24%; gate+up and down+shared-gate+
  residual fusions exist on the mmvq decode path.

Phase 2 priorities (evidence-ranked): 1) one fused expert kernel per
layer (plan-fused-moe.md work, ~1.2 ms decode + prefill win), 2) short-K
mmvq config tuning for the expert shapes, 3) small-kernel-tail fusion
(residual adds, ssm-state GET_ROWS into GDN, elementwise chains).

## Current facts (report.md section 6)

- tg128 = 92.2-92.7 t/s (~10.8 ms/token), **flat with context depth** (92.4
  at KV=512, 92.2 at KV=16384) - so KV-read growth is NOT the decode
  problem (unlike prefill).
- CUDA-graph replay is worth only ~2.5% (90.1 t/s with
  `GGML_CUDA_DISABLE_GRAPHS=1`).
- FA on vs off: 92.4 vs 90.2 (-2%) - flash attention is NOT the decode
  bottleneck.
- The no-graph op-timing picture (FA 5.6 ms, conv-state GET_ROWS 3.9 ms of
  a 19.5 ms step) is NOT representative of the production graph path
  (19.5 ms no-graph vs 10.8 ms graph); do not trust it as the cost model.
- The 10.8 ms/token is inside the CUDA-graph path, which the current
  `GGML_CUDA_OP_TIMING` cannot see (events cannot be recorded during graph
  capture).

Decode step content per token: 30 recurrent layers (qkv+z proj -> conv ->
l2_norm q/k -> GDN autoregressive scan -> gated norm -> ssm_out), 10
full-attn layers (Q/K/V+gate proj -> RMS -> RoPE -> FA -> wo), 40x routed
experts (router + 3 mmid on 8 experts + weights mul + sum), 40x shared
experts (3 dense GEMMs + sigmoid gate), lm head (large: 2048 x 248320).

## Hypotheses (ranked)

1. **The GDN autoregressive chain (30 layers)**: per token each layer runs
   the sequential AR scan kernel plus the conv/qkv/z/beta/alpha/out
   projections. 30 small kernel chains with a sequential dependency inside
   each - the classic linear-attention decode bottleneck. The AR kernel is
   a single launch with grid over (H_v=32 heads x S_v=128) - small work,
   but 30 sequential chains add up. Plausible 4-6 ms.
2. **Routed experts via mmvq (40 layers x 3 mmid)**: decode goes through
   the per-expert vector path (ne12=1 -> mmvq). 8 experts x 1 token per
   layer; sort + 8 vec-dots of [512 x 2560] Q6_K each. The block-10
   k-quant VDR boosts target mmvq/mmq - verify they engage for the expert
   shapes at 1 token. Plausible 2-3 ms.
3. **Shared experts + dense projections**: 40 x 4 dense GEMMs of
   [512/2048 x 2560] x 1 token + the 2048 x 248320 lm head. The lm head
   alone is 0.9 ms (measured). Plausible 2-3 ms total.
4. **conv-state access**: the no-graph timing showed GET_ROWS
   conv_states at 3.9 ms, but graphs likely overlap/serialize this
   differently; re-measure inside the graph path before trusting it.
5. **FA decode kernel selection** (head_dim=160, n_head=16, kv=2, ncols=1):
   ruled out as dominant by the -2% FA-off delta, but check the kernel
   choice anyway (WMMA vs tile) for the decode shapes.

## Phase 1 - see inside the graph path (0.5-1 d)

The op timing cannot instrument capture; options, in order:

a. **Graph-aware timing**: record `cudaEvent`s around each node in the
   DIRECT path, and additionally time the graph replay as a whole; then
   run decode both ways and attribute the graph-path speedup. Crude but
   bounds the unknown.
b. **Component isolation via existing toggles** (cheap, do first):
   - `-fa off` (measured -2%),
   - `GGML_CUDA_FORCE_MMQ=1` (forces MMQ instead of mmvq for the expert
     path - quantifies the vec path vs batched path),
   - the GDN AR bf16 switch (check whether `GGML_CUDA_GDN_CHUNKED_BF16`
     or a sibling env also governs the AR kernel; otherwise add one for
     the A/B),
   - `-ot 'blk.*.ssm_*=CPU'` (override-tensor) to force the GDN weights
     to CPU and observe the wall-time delta per layer - coarse, but
     shows the recurrent chain's share.
c. **-ngl sweep**: 10/20/30/40 layers on GPU vs CPU - the delta per group
   quantifies GPU per-layer cost; combine with (b).

Deliverable: a cost table for decode (FA, GDN chain, experts, shared,
lm-head) on the real graph path, not the no-graph path.

## Handoff notes (2026-08-30 investigation session)

Environment and tooling facts learned while gathering the data above, so the
next session does not repeat them:

### Profiler status on this box (ROCm 10.0.0-gfx1201)

ALL of these fail; do not retry without a reason:
- `rocprofv3 --kernel-trace` and `--sys-trace`: SIGABRT in the app during
  model load (`ggml_backend_cuda_buffer_set_tensor` H2D copy; the actual
  hipError message is swallowed by rocprofv3's output redirection).
- `rocprof-sys-run --rocm=kernel`: heap corruption ("corrupted size vs.
  prev_size") and abort.
- `rocprof-attach` (needs `ROCP_TOOL_ATTACH=1` on the target): ptrace
  injection kills the target with signal 6.
- Crashed rocprofv3 runs ORPHAN their llama-bench child, which keeps the
  model resident in VRAM (~18-29 GB/GPU) and makes the next run fail with
  "failed to load model". Always check `ps`/`rocm-smi --showmeminfo vram`
  and `pkill -9 -f "[l]lama-bench"` (bracket trick - plain `pkill -f
  llama-bench` self-matches and kills the shell) before retrying.

Working alternative: llama.cpp's own instrumentation (no tracer):
- `GGML_CUDA_OP_TIMING=1` + `-v` (block 08): per-op CUDA-event timings per
  graph eval. Needs `-v` (INFO logging is suppressed otherwise). Caveats:
  it disables CUDA graphs, and fused/streamed nodes (the GDN fused kernel,
  concurrent-event nodes) are skipped by the timing - treat its total as a
  lower bound and its per-op shares as main-stream-only.
- Env A/B switches available: `GGML_CUDA_GDN_CHUNKED`, `GGML_CUDA_GDN_CHUNKED_BF16`,
  `GGML_CUDA_FA_WMMA_256`, `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_FORCE_MMQ`.

### llama-bench gotchas

- `-b` sets `n_batch`, `-ub` sets `n_ubatch` (default 512). The prefill
  ubatch is the physical per-eval batch; `-b` alone never changes it. All
  the baseline numbers in `report.md` are at ub=512 unless stated.
- Reproducible measurement pattern: `-r 2` (default warmup) for curves,
  `-r 1 --no-warmup` for op timing.
- Single-GPU profiling needs `HIP_VISIBLE_DEVICES=0`; the multi-GPU load
  path (layer split) fails under tracing.

### Model / build facts

- Model: `/llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf`
  (27.29 GiB). Dense comparison: `/llm/models/Qwen3.6/27B/Q6_K/Qwen3.6-27B-Q6_K.gguf`.
  Build: `~/llama.cpp/build-rocm/bin` (rdna-boosts `4fa92f0ae`).
- Decode numbers to beat: tg128 = 92.2-92.7 t/s flat from KV=512 to
  KV=16384; graphs worth ~2.5%; FA off -2%. No MTP in this build
  (`n_rs_seq=0`), so the decode graph is the pure 40-layer trunk.
- Code refs: GDN AR kernel + dispatch in
  `ggml/src/ggml-cuda/gated_delta_net.cu` (block 02); conv-state read +
  concat in `src/models/delta-net-base.cpp` `build_conv_state` (~500);
  mmid vec path in `ggml/src/ggml-cuda/mmq.cu` / `mmvq.cuh`; fused-MoE
  dead check `ggml_cuda_can_fuse` ~3175 in `ggml-cuda.cu`; `-ot` tensor
  override in `common/arg.cpp`.

## Phase 2 - attack the winner (1-3 d)

Depending on Phase 1:

- **If GDN chain dominates**: fuse the decode chain per layer
  (conv-state read + conv1d + l2_norm + AR scan + gated norm into 1-2
  kernels). The fork already folds the SSM conv_input concat into the qkv
  mmvq for prefill (commit 555e79ab2) - check what the decode path still
  does explicitly. The AR scan itself is sequential per head; a
  cross-head-batched kernel with a better grid (H_v x S_v in one launch
  already) may be limited by launch/sync overhead - consider one kernel
  per layer for the whole chain.
- **If experts dominate**: the fused MoE work (plan-fused-moe.md) also
  applies to decode mmid (gate+up+GLU) - its sort/quantize/launch savings
  are proportionally larger at 1 token.
- **If shared experts / dense proj dominate**: they are already simple
  mmvq GEMMs; check the VDR/mmap kernel selection at [2048/512 x 2560] x
  1-token shapes (block 10 coverage).
- **conv-state**: fuse the GET_ROWS into the conv kernel (same direction
  as the prefill fold).

## Phase 3 - validation (0.5 d)

- tg at KV=512 and 16384 (flatness must hold after fixes),
- tg128 on the real server workload with `--ubatch-size 2048` (decode is
  unaffected by ubatch but the deployment uses the server),
- determinism check on decode outputs (the block 05 bit-identical
  harness / logitdump-style),
- regression: full arch-test suite (608 ROCm configs) + real-model
  bit-identical decode vs the pre-change build.

## Target

92 t/s -> 110+ t/s (+20%) on gfx1201 single GPU; then re-measure on the
3-GPU config (hybrid all-reduce, block 12) and on Strix Halo if available
(the decode chain is latency-bound, so unified-memory effects should be
small - verify).

## Reference material in-tree

- `src/models/qwen35moe.cpp` `build_layer_attn_linear` (the decode chain
  shape), `src/models/delta-net-base.cpp` (AR + conv state).
- `ggml/src/ggml-cuda/gated_delta_net.cu` (sequential AR kernel + block 02
  dispatch), `ggml/src/ggml-cuda/mmvq.cuh` (expert vec path, block 10).
- `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_op_gated_delta_net_fused_cache`
  and the `-ot` override path (`common/arg.cpp`).
