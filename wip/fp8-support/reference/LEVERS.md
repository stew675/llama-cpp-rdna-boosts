# LEVERS.md - remaining performance work, ranked, for the next session

**Purpose**: self-contained list of the remaining performance levers for the
fp8 GDN model on this box, with the current state, the approach, the expected
win, and the validation steps for each. Step through in order; stop when the
gain no longer justifies the risk. Everything here was measured on 2026-08-06
after the chunked GDN correctness fix (ef41a940a) and phase B rewrite
(56e7cb0c9). Read `PERF_HANDOVER.md` for history and `HANDOVER.md` for the
project as a whole.

Box: 1x AMD Radeon AI PRO R9700 (gfx1201, 64 CUs, 2 SIMDs/CU, wave32,
64 KB LDS/CU, 8 MB L2), 2350 MHz. Model: `/llm/models/Qwen3.5/4B/StewFP8/
stewfp8-ow.gguf` (24 delta-net layers, H=32 heads, K=V=128, neqk1=16).

---

## 0. Current state (verified 2026-08-06, after the L9 single-copy embedding session)

| metric | value | notes |
|---|---|---|
| pp512 (chunked) | 7162-7170 (was 7184-7260) | L9: single-copy fp8 embedding, same speed as the two-copy -ow |
| tg64 | 88.3-88.6 (was 88.3-88.5) | unchanged |
| PPL | 6.2250 (was 6.2528) | within noise; generation matches |
| backend tests | all pass (GET_ROWS now covers fp8) | new F8 get_rows cases, bit-exact vs CPU |
| wmma kernel | ~156-159 us/launch avg | unchanged |
| GDN chunked per pp512 | ~10.9 ms (unchanged) | phase A 4.9 + phase B 5.9 |
| Cijk (CK BF16 GEMM) | GONE (0 launches) | in_proj_a/b now on the fp8 wmma path |
| concat_non_cont | GONE (0 launches) | fused into ssm_conv_2src |
| quantize_fp8 per pp512 | ~2.5 ms | warp rewrite |
| model file | 4.17 GiB, 4.33 B params | single fp8 token_embd (was 5.35 GiB / 4.96 B two-copy) |

IMPORTANT box change: the box actually has 3x R9700 (ROCm0/1/2) + a small GPU3,
and a Qwen3.6-27B llama-server occupies ROCm0/1 most of the time. Bench on
`-dev ROCm2` (free). llama-bench on an occupied GPU fails with cudaMalloc OOM
and llama.cpp silently offloads layers to CPU (looks like a perf regression).
Also: llama-cli in this fork is a SERVER CLIENT (spawns its own llama-server or
connects via --server-base) - generation-parity runs must use the same GPU
and model as the benches.

NOTE on bench noise: since 2026-08-06 late the box is noisy (+/-60-170 on
pp512, occasionally +/-1000). Prefer rocprof kernel times (deterministic to
+/-3%) over llama-bench for judging a change; the GDN delta of this session
(~-2.4 ms/pp512) should be ~+2% on pp512 but is hard to pin in bench.

Bench geometry (IMPORTANT, do not rediscover): `llama-bench -p 512` runs the
512-token prompt as ONE batch (n_seqs=1, n_chunks=8) and llama-bench performs
a warmup pass first. The kernel trace therefore shows 2x the per-pp512
launch counts (24 layers x 2) and 2x the per-pp512 kernel times. Per-pp512
numbers in this file are the trace numbers divided by 2 unless stated.

Profile command: `rocprofv3 -r -d /tmp/rocp_x -f csv -- llama-bench -m <model>
-p 512 -n 8 -r 1 -dev ROCm2`, then parse `soar/*kernel_trace.csv`
(Duration = End_Timestamp - Start_Timestamp; Grid_Size_X = work-items,
divide by Workgroup_Size_X for CTA count; VGPR/LDS/Scratch columns).

## 1. Where pp512 time goes (per-pp512, warmup removed, after the L7 in-proj fp8)

| kernel | ms/pp512 | share | note |
|---|---|---|---|
| mul_mat_fp8_wmma | ~39.4 | ~50% | prefill GEMMs (incl. the 48 alpha/beta in-proj GEMMs, ~1.9 ms) |
| gdn_chunk_prepare + state | ~10.9 | ~14% | A ~4.9 + B ~5.9 |
| quantize_fp8 | ~2.8 | ~4% | warp rewrite (was ~6.3); f32->fp8 staging |
| ssm_conv_long_token_2src | ~0.6 | ~1% | fused conv (was concat 4.7 + conv 0.8) |
| flash_attn_tile | ~3.0 | ~4% | 8 attention layers |
| silu | ~2.9 | ~4% | |
| rms_norm (both sizes) | ~4.4 | ~6% | |
| k_bin_bcast / cpy / get_rows / l2_norm / rope / ssm_conv | ~3.3 | ~4% | small |
| (decode kernels: gemv, sequential GDN, etc.) | - | - | not part of pp512 |

pp512 wall = 512 / ~7260 = ~70.5 ms (noisy). The shares above sum to ~90% of that.

The Cijk (CK BF16 GEMM) row is GONE: the delta-net in/out projections
(in_proj_a/b = ssm_alpha/ssm_beta, 48 launches/pp512 at ~112 us each = 5.37 ms)
are now F8_E4M3 in the GGUF and run on the wmma path (~24 us wmma + ~11 us
quantize per launch = ~1.7 ms/pp512). The remaining in_proj cost is split
across the wmma (~1.9 ms) and quantize (~0.3 ms) rows above.

The GDN share is now split nearly evenly: phase A (gdn_chunk_prepare, ~4.9
ms/pp512, 23 syncs, 1 CTA/CU at 58 KB LDS) and phase B (gdn_chunk_state,
~5.9 ms/pp512, 2 CTAs/CU, 1 wave). Both are ~40-65% load-stall-bound with a
substantial FP32-FMA floor (measured by stage-elimination stubs: phase A v+q
loads ~118 us, gram ~34 us, closure ~32 us; phase B kd/q loads ~112 us,
ac+k loads ~62 us, FMA/sync floor ~206 us).

## 2. Ranked levers

### L1. mul_mat_fp8_wmma - measured 2026-08-06 evening, mostly spent; revisit only with a new idea

- Kernel: mul_mat_fp8_wmma, ~40.6 ms/pp512 (~48%), ~77 TFLOP/s effective (92-98 on the big shapes in a hot-loop rig).
- The 2-CTA/CU premise is DONE (single-buffered 26.4 KB, register-staged). The
  residual gap to the aiter triton kernel (121-137 TFLOP/s at the SAME 2
  CTAs/CU) was investigated with a standalone rig (/tmp/bench_wmma.cu, hot
  loop, min of 3) + stage-elimination stubs:
  * fragment LDS chain: ~4 us/launch of the gate GEMM (250 us) - nearly free,
    the 2addr loads + 2-CTA occupancy hide it.
  * scale pass (16 wd + 2 ad LDS + 32 FMA per warp per k-block, serialized
    between the wmma chain and the barrier): ~33 us (13%). Restructuring the
    per-row scales into a contiguous smem array (2 float4 broadcast loads
    instead of 16 scattered reads) did NOT recover it - the cost is the 32
    FMAs + the serial position, not the LDS. The FMA count is irreducible
    (32 acc elements/thread/k-block) and the wd (weight block scale) varies
    per k-block, so the multiply cannot be deferred to the epilogue.
  * staging loads + barriers + wmma issue: the ~85% floor. The 4-B weight
    staging loads coalesce fine; a 2-k-block staging lookahead (double-buffered
    registers, +26 VGPR) REGRESSED ~4% (scheduling).
  * L2: ~280 MB/launch of weight+activation traffic (8x weight re-reads) is
    NOT the limiter (measured 1.1-1.6 TB/s of a 3 TB/s budget).
- aiter's kpack=2 does not port cleanly: it needs the wmma operand smem layout
  rearranged (the 16-k fragment is split across lane halves, so a single wider
  LDS load cannot cover 2 k-steps without a swizzled store side) and the
  triton num_stages=2 doubles the smem (48 KB = 1 CTA/CU, the measured 58
  TFLOP/s config).
- COMPILER MISCOMPILE (new, do not retry): a 2-k-step fragment lookahead with
  a rotate (named fp8x8_t variables, a0_1/a0_2 rotation) MISCOMPILES under the
  single-buffer loop - LLVM folds the rotation copies, reusing the step-0
  operand registers for step 1 (SASS shows 8 v_wmma with identical operands;
  PPL 183142). This is the same class as the burst-array bug in section 4.
  The 1-k-step lookahead (a0_n) is the deepest pipeline that compiles right.
- Expected: nothing cheap left. The aiter numbers came from triton codegen we
  cannot reproduce by hand on this compiler. L1 is spent unless a new
  structural idea appears (e.g. a swizzled kpacked smem layout for A).
- Measurement rig: /tmp/bench_wmma.cu (kernel-only hot loop) + /tmp/check_wmma.cu
  (CPU-reference correctness). CHECK LESSON: random fp8 payload bytes contain
  NaN encodings (0x7F/0xFF) that make fabs(ref-out) NaN on EVERY element -
  the check false-passes (maxerr stays 0, bad stays 0). The check must use
  realistically-quantized data (host-side fp8 encoder); with proper data the
  miscompiled kernel scores maxerr 34 / 7991-of-8192 wrong.
- Depends on: nothing. Independent of L2.

### L2. wmma weight repack at load time - DONE 2026-08-06 evening, +1.6-1.8% pp512

- DONE (commits below): weights are repacked once per tensor on first wmma use into a
  cached layout (fp8 bytes contiguous [m_pad][k] 16-B aligned + scales [m_pad][n_col_blocks]
  f32), so the wmma kernel stages 16-B uint4 loads instead of 4-B misaligned ones. The
  GEMV and scalar paths keep reading the original block layout (decode untouched).
- Measured (rig, gate shape 9216x512x2560): staging-stub delta 260.5 -> 175 us (the
  weight-side staging loads were ~85 us = 33% of the kernel, far above the old ~15%
  estimate at 55 TFLOP/s); repacked kernel 260.5 -> ~228 us = 93 -> 106 TFLOP/s (+14%).
- In-tree: wmma ~204 -> ~186 us/launch average (-9%). The repack itself is one-time per
  tensor (p50 71 us, max 1.09 ms for output.weight at 389 MB; ~13 ms total for the ~200
  tensors, only on the first graph build - CUDA-graph replays reuse the cache, verified
  stable at ~201 entries over a 2200-call PPL run).
- Net: pp512 6220 -> 6319-6334 (+1.6-1.8%), tg64 unchanged. PPL 6.2677 exact; generation
  matches; 47/47 GDN tests.
- Files: fp8.cuh (fp8_repack_weights kernel + wmma staging addressing), fp8.cu (lazy
  per-context cache keyed on src0->data, built in ggml_cuda_fp8_repack), common.cuh
  (fp8_repack_buf in ggml_backend_cuda_context).
- Validation: /tmp/check_repack_kernel.cu (isolated repack kernel check, all model
  shapes incl. m=70000), /tmp/check_repack.cu (bit-exact kernel-vs-kernel).
- Remaining: L2 as designed is spent. The kpacked smem layout for A (the other L2
  benefit) is still blocked by the aiter layout problem (see L1).

### L3. gdn_chunk_prepare (phase A) - main items DONE 2026-08-06, remaining: 2 CTAs/CU

- Kernel: gdn_chunk_prepare, ~4.9 ms/pp512, ~205-210 us/launch, grid 256 CTAs
  (32 heads x 8 chunks), 256 threads, 58 KB LDS (1 CTA/CU), 128 VGPR, no
  spills. 23 dynamic syncs (was ~165).
- DONE (commits a921f8fb0): closure rewritten to an exact 4x4-blocked forward
  substitution (16 syncs vs 146; D(i,i) via warp-shuffle row substitution, the
  off-diagonal blocks in 15 row passes reading A from a preserved packed smem
  copy s_A); decay cumsum to a 2-sync warp-shuffle scan; k rows prefetched
  into registers during the scan; the k_cumsum v and attn_causal q reads
  vectorized to float4 (512 scalar global loads per thread -> 128 float4,
  with an alignment fallback). Net 276.5 -> ~205-210 us/launch (-24%).
- Remaining (measured by stage-elimination, all +/-3%):
  * global loads (v+q) ~118 us of the original 293 - vectorized already, the
    residual is the L2 latency exposure with 8 warps.
  * gram ~34 us (1024 smem loads + 2048 FMAs/thread; symmetric-tile or
    vectorization ideas are load-imbalance or padding-bound, see section 4).
  * closure ~32 us (16 barriers; the pass work is inherently imbalanced -
    pass i has i blocks; barrier count is at the floor for 4x4 granularity).
  * 2 CTAs/CU still needs the smem under 32 KB: s_k [64][129] alone is 33 KB,
    so it requires fp16 s_k (precision risk on the closure, which is
    sensitivity-prone - see GDN_DEBUG_HANDOVER.md) or a chunk split. Do NOT
    attempt casually.
- Validation: 47/47 tests, PPL 6.2677, generation parity. Numpy check of the
  4x4 closure math: /tmp/validate_closure_4x4.py (exact to 1e-15).

### L4. gdn_chunk_state (phase B): chunk-loop latency (partial, delta loads done)

- Kernel: gdn_chunk_state, ~5.9 ms/pp512, ~246 us/launch, 1 wave, 2 CTAs/CU,
  VGPR 160, no spills.
- DONE (commit c8757d649): the step-3 delta reads vectorized (64 scalar loads
  per thread per chunk -> 16 float4, clamped k-row index keeps the padded
  rows in-bounds and contributing zero). 268.8 -> ~246 us/launch (-8.5%).
- Remaining (measured by stage-elimination): kd/q loads ~112 us/launch and
  ac+k loads ~62 us of the 269; the FMA/sync/expf floor is ~206 us. The
  kernel is ~65% load-stall-bound, but the loads are warp-broadcast and the
  chunk loop is serially dependent on S, so a register pipeline is not
  possible: a 2-stage kd/q pipeline hit 256 VGPR + spills (495 us), and
  #pragma unroll on the r-loop regressed (349 us). 16 warps/CU already.
  Realistic remaining ideas: none cheap - revisit only with a fresh profile.
- Validation: same as L3.

### L5. concat_non_cont: delta-net ssm conv plumbing - DONE 2026-08-06 evening, pp512 +4%

- DONE (commits below): the ssm_conv input assembly is fused into the conv.
  New op form `ggml_ssm_conv_2src(ctx, conv_states, qkv, c)` (same GGML_OP_SSM_CONV
  code, optional src[2] = conv states) reads the two contiguous pieces directly:
  conv_states (position-major) + qkv_mixed (channel-fastest, UNTRANSPOSED) with a
  position-major smem tile. The old transpose+concat+conv_input buffer is gone.
  Measured: concat_non_cont 0 launches (was 24, 4.7 ms/pp512); the fused long-token
  kernel is ~0.6 ms/pp512 (24 launches). pp512 6303 -> 6538-6554 (+4%), tg64
  unchanged. PPL 6.2677 exact; generation matches; 47/47 GDN + 83/83 SSM_CONV
  (incl. 24 new test_ssm_conv_2src cases: d_conv 3/4/9 x d_inner 1024/1536/2048 x
  n_t 1/32/64).
- Graph: build_conv_state returns the fused conv output; the state-save view is
  now the transpose of a qkv-tail view (the cache stores [position][channel] with
  the position dim fastest, and the save cpy scatters by the src flat index). The
  concat is only built when a state window overlaps the conv_states piece
  (n_t < d_conv-1, or rollback slots with small batches) - decode keeps it, at
  trivial cost. The conv is expanded before the state-save cpy to keep the
  read-before-write ordering on conv_states_all.
- Files: ggml.c/h (ggml_ssm_conv_2src), ssm-conv.cu (fused short + long kernels,
  dispatch branch on src[2]), ggml-cpu/ops.cpp (2-src reference), ggml-cuda.cu
  (supports_op uses ne[0] for the 2-src form), delta-net-base.cpp + models.h
  (build_conv_state), qwen35/moe/3next.cpp (call sites), test-backend-ops.cpp
  (test_ssm_conv_2src).
- Pitfalls recorded in section 4: the qkv layout is channel-fastest (c*4 + t*nb1),
  so a fixed channel's token window is NOT contiguous - the smem tile must be
  staged coalesced along channels. ggml_view_3d's dim0 stride is ALWAYS the
  element size (nb reset in ggml_new_tensor_impl) - a strided-token view is
  impossible; use transpose-of-view instead. The cpy_scalar flat-index mapping
  follows the SRC ne00, and the state cache layout is position-fastest (row
  p + c*(d_conv-1)), which the reshape_3d read side matches.

### L6. quantize_fp8 - DONE 2026-08-06 night via a warp rewrite (not the fusion)

- Kernel: quantize_fp8 was ~5.1 ms/pp512 (25.4 us/launch on pp512 grids) for the
  f32->fp8 activation staging. It was NOT bandwidth-bound: ~6.5 MB/launch in
  25 us is ~256 GB/s, i.e. latency/barrier-bound (1 float per thread + 2
  __syncthreads per 128-thread CTA).
- DONE (commits below): quantize_fp8_warp<4> - one warp per (token, 128-col
  block), each lane covers 4 consecutive values (float4), the amax reduce is a
  pure warp butterfly (no smem, no barriers), and the fp8 encode uses the gfx12
  hardware v_cvt_pk_fp8_f32 (RNE) which was verified BIT-EXACT vs the software
  encoder fp8_e4m3_from_f32 on 406K finite inputs (every boundary, tie,
  subnormal, saturation case; only NaN/Inf differ, which activations never
  are). 4 k-blocks per warp (KBW) overlap the DRAM latency of the independent
  loads; KBW=4 measured better than 1/2/8.
- Measured (rig, min of 3, per-launch): quantize 25.4 -> 12.6 us on pp512
  grids (-50%). pp512 6630 -> 6818-6841 (+2.9-3.2%). tg64 unchanged. PPL
  6.2677 exact; generation matches; 47/47 GDN tests. The staging is bit-exact
  vs the old kernel (rig check bad=0 on all model shapes), so the wmma/gemv/
  scalar consumers are untouched.
- Dispatch: RDNA4 + n > GGML_FP8_GEMV_MAX_N + 16-B-aligned src1 -> warp
  kernel; else the old quantize_fp8. The decode grids (n <= 16) REGRESSED with
  the warp kernel (1.31 -> 2.04 us/launch: its grid is ceil(k/128/32) x n, too
  few CTAs for tiny n) - keep the old kernel there.
- THE FUSION (the original L6 approach - read f32 in the wmma kernel and
  quantize in-kernel) was built and measured: mul_mat_fp8_wmma_fused loses on
  the big GEMMs that dominate pp512 (+9.8% on the gate shape 9216x512x2560,
  +5.5% on 8192x512x2560) and wins only on the small under-occupied ones
  (-23% on 4096x512x2560). Net per pp512 ~ +0.2 ms (a wash). The in-kernel
  quantize (~42 us/launch added at saturation) costs more than the standalone
  kernel it replaces (25 us) because it competes with the wmma/scale-pass
  pipes and the f32 staging is 4x the bytes. Do NOT retry the fusion; the warp
  rewrite gets the same win with zero risk to the GEMM path.
- Files: fp8.cuh (quantize_fp8_warp template), fp8.cu (dispatch).
- Validation: /tmp/bench_fused.cu + /tmp/bench_pairs.cu (rigs), /tmp/fp8_cvt_test.cu
  (hw-vs-sw encoder bit-exactness).
- Depends on: nothing.

### L7. Cijk BF16 GEMMs (delta-net in/out projections) - DONE 2026-08-06, pp512 +5-6%

- DONE (commits below): in_proj_a/b (ssm_alpha/ssm_beta, [32, 2560] BF16 in the
  GGUF) are now F8_E4M3, so the 48 Cijk launches/pp512 (~5.37 ms at ~112 us
  each for a 32x512x2560 GEMM) run on the fp8 path instead: ~24 us wmma
  (8 CTAs, m_pad 128) + ~11 us quantize per launch = ~1.7 ms/pp512. pp512
  6818-6841 -> 7184-7260 (+5-6%). tg64 89.2-89.6 -> 88.3-88.5 (-1%: the
  alpha/beta decode GEMMs move from the bf16 mmvf path to quantize_fp8 +
  mul_mat_fp8_gemv). PPL 6.2528 (was 6.2677, within noise); generation
  matches; GDN + SSM_CONV backend tests pass.
- Conversion: new flag `--fp8-delta-net-in-proj` in convert_hf_to_gguf.py,
  wired through conversion/base.py; the packing lives in
  `_LinearAttentionVReorderBase._generate_fp8_in_proj` (conversion/qwen.py):
  same V-head reorder as modify_tensors (head_dim=1), then the standard
  block_f8_e4m3 pack (d = BF16(amax/448) per 128-col block, q = fp8(w/d)).
  The 32 gate rows share ONE 128-row scale block (ceil(32/128) = 1), same
  as the model's own scale convention. The fp8 quantization error is the
  same regime as the model's own fp8 tensors (mean rel ~2.25% vs 2.256% in
  the conversion summary).
- The rest of the old "quantizable set" (norms, conv1d, A_log) are NOT GEMMs
  (elementwise/conv ops, F32, no kernel change would apply) - nothing left
  there. The only other BF16 tensors are token_embd (get_rows, keep BF16)
  and eh_proj (not in the 4B main path).
- Recovery lever for the tg64 cost: fuse the quantize into the decode gemv
  (already listed in PERF_HANDOVER section 10 as a decode item; would also
  recover the other 226 decode quantize launches).

### L8. Small kernels (silu 2.9, rms_norm 4.4, k_bin_bcast 1.9) - skip

- Not worth the risk/effort individually. Revisit only if everything else is
  done and a specific one shows up hot in a fresh profile.

### L9. Single-copy fp8 token_embd - DONE 2026-08-06, same speed, -22% file

- DONE (commits below): token_embd is now F8_E4M3 in the GGUF (per-row scales)
  and the separate fp8 output.weight copy is GONE - one embedding serves both
  the input lookup (get_rows) and the lm_head (the graph duplicates token_embd
  as the output tensor when output.weight is absent). File 5.35 -> 4.17 GiB,
  4.96 -> 4.33 B params, ~0.66 GB less VRAM; pp512 and tg64 UNCHANGED (7162-7170
  / 88.3-88.6 on ROCm2); PPL 6.2250 (was 6.2528); generation identical.
- New fp8 get_rows kernel (getrows.cu k_get_rows_f8: one CTA per (row, 128-col
  block), per-row scale d, fp8_e4m3_to_f32 moved to common.cuh) + dispatch in
  ggml-cuda.cu supports_op. The traits table got to_float =
  dequantize_row_f8_e4m3 (ggml.c), which also fixes llama_model_get_tok_embd.
- Conversion: new `--fp8-token-embd` flag (mutually exclusive with
  --fp8-output-weight); `_generate_fp8_token_embd` in conversion/base.py packs
  token_embd with PER-ROW scales (d[r][cb], finer than the 128-row-block
  convention and indexable by both the get_rows and mul_mat kernels).
- PITFALL (do not rediscover): the input layer is ALWAYS kept on the CPU buffer
  by design ("very little benefit to offloading the input layer"), and with
  mmap the ROCm_Host candidate is converted to a plain CPU buffer. The BF16
  embedding works because get_rows runs on CPU reading the mmap in place. For
  F8, the CPU backend blanket-rejected fp8 ops, so the sched placed get_rows on
  CUDA and copied the WHOLE 625 MB embedding into the GPU compute buffer per
  batch (decode 5x slower, 17 t/s, 625 MiB compute buffer). Fix: carve out
  GET_ROWS in ggml_backend_cpu_device_supports_op (CPU get_rows dequantizes via
  traits->to_float) + add the F8 case to ggml_compute_forward_get_rows. With
  get_rows on CPU the embedding stays mmap'd, the sched compute buffer is ~8 MiB,
  and decode is back to 88.5 t/s. Side effect: test-backend-ops now runs fp8
  GET_ROWS cases (CPU ref vs GPU kernel) - all pass bit-exactly.
- The duplicated output tensor (TENSOR_DUPLICATED path, qwen35.cpp) is a real
  second ggml_tensor sharing the file data - the head mul_mat runs on the fp8
  wmma/gemv path exactly like the old output.weight (verified with an
  intermediate two-copy build: same speed, so the head was never the issue).

## 3. Session protocol (commands)

```bash
# build (after any kernel change)
cmake --build /tmp/llama-hip-full -j16 --target llama-bench llama-perplexity test-backend-ops

# bench (pp512 + tg64, 5 reps)
/tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 64 -r 5 -dev ROCm2

# correctness gate (always)
/tmp/llama-hip-full/bin/test-backend-ops test -b ROCm2 -o GATED_DELTA_NET   # 47/47
/tmp/llama-hip-full/bin/llama-perplexity -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf \
    -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm2 2>&1 | tr -d '\0' | grep "Final estimate"  # 6.24

# generation parity (PPL is not enough for fp8 changes)
/tmp/llama-hip-full/bin/llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -c 512 -n 24 \
    -dev ROCm2 -p "The capital of France is" --single-turn --no-conversation   # needs both flags in this fork

# profile
rocprofv3 -r -d /tmp/rocp_x -f csv -- /tmp/llama-hip-full/bin/llama-bench \
    -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 8 -r 1 -dev ROCm2
# parse soar/*kernel_trace.csv; REMEMBER the warmup doubles every kernel count/time

# sequential-vs-chunked A/B
GGML_CUDA_GDN_CHUNKED=0 <same bench>   # sequential GDN for comparison
```

## 4. Pitfalls and failed experiments (do not rediscover)

- Kernel traces include llama-bench's warmup pass: counts and times are 2x
  the per-pp512 values. Divide by 2 for per-pp512 numbers.
- GDN phase B register-tiled 16x16 grid (4x2 tiles/thread) with smem-staged
  kd/q/ac/k blocks: 192 VGPRs + spills (764-1028 B scratch; ISA showed 274
  scratch instructions vs 308 FMAs) -> 675 us/launch. The tile layout spreads
  the V dimension across threads, amplifying shared-input L2 reads 16x and
  blowing the register budget. Thread-per-column (32 cols per warp) is the
  right layout for these 64x128x32 matmuls: every kd/q/k/ac load becomes a
  warp broadcast and live registers stay ~40. Do not go back to register
  tiling here without a concrete reason.
- `__launch_bounds__(256, 2)` forcing 128 VGPRs on a 192-VGPR kernel made it
  worse (more spills). Fix spills by restructuring, not by capping.
- GGML_HIP_GRAPHS=0 is a CMAKE option, not a runtime env var. The runtime
  disable is GGML_CUDA_DISABLE_GRAPHS=1. CUDA graphs are NOT the cause of
  any correctness issue.
- cudaMemcpy/cudaStreamSynchronize inside the launch during graph capture is
  illegal/unreliable. If re-adding the GDN_DUMP debug, all copies must be
  stream-ordered (cudaMemcpyAsync on the compute stream) with one sync after
  the phase B launch.
- wmma experiments that failed: 16 warps/CTA with 2x1 tiles (fragment reuse
  1.5 LDS/wmma, 4936 t/s), double-buffered smem 61 KB (1 CTA/CU, 5110 t/s),
  burst fragment arrays (LLVM miscompile). GROUP_SIZE_M=8 was noise.
- LLVM wmma fragment-pipeline miscompiles (do NOT retry either shape): (1) the
  burst array fp8x8_t fa[4][4] (16/32 v_wmma get v[0:1] operands); (2) a
  2-k-step lookahead with a named-variable rotate (a0_1/a0_2) - the allocator
  folds the rotation copies and step 1 reuses step 0's operand registers
  (verified in SASS; PPL 183142; the check catches it ONLY with
  realistically-quantized data, see below). The 1-k-step lookahead (a0_n)
  compiles correctly. Rule: never let a fragment value be rotated/reused
  across k-steps via register copies in the single-buffer loop.
- fp8 correctness checks false-pass with random payload bytes: the fp8 byte
  space contains NaN encodings (0x7F/0xFF) and random f32 scales are mostly
  NaN/inf, so fabs(ref - out) is NaN on every element, maxerr/bad stay 0.
  Any fp8 kernel check must feed realistically-quantized values (host-side
  encoder, sane scales). /tmp/check_wmma.cu has the working template.
- The chunked GDN closure bug history is in GDN_DEBUG_HANDOVER.md; the
  current closure (4x4-blocked forward substitution, 2026-08-06) is verified
  exact (numpy diff ~1e-15, /tmp/validate_closure_4x4.py). Do NOT simplify
  it back to I+A+A^2+A^3, and do NOT truncate the 4x4 diag power series.
- Phase B kd/q register pipeline (2026-08-06): a 2-stage software pipeline
  (alternating float4 register buffers for kd/q, #pragma unroll 2) hit 256
  VGPRs + 268 B scratch -> 495 us/launch (was 269). The alternating-buffer
  arrays force too many live registers; the thread-per-column layout that
  made phase B fast keeps live registers ~40 and has no room for staging.
- Phase B #pragma unroll on the step-1 r-loop (2026-08-06): 349 us/launch
  (was 269) - the compiler's default scheduling is better; do not force.
- Phase A syncs were NOT the main cost: cutting ~165 -> 23 dynamic barriers
  (closure rewrite + scan) moved phase A only 293 -> 276 us. The real costs
  were the 512 scalar global loads/thread (v in k_cumsum, q in attn_causal)
  - vectorizing those to float4 was the big win (276 -> 210 us).
- qkv_mixed is channel-fastest: qkv[c][t] at c*4 + t*nb1, so a fixed channel's
  token window is strided by d_inner*4. Any kernel reading the qkv window (the
  fused ssm_conv) must stage coalesced along CHANNELS (position-major smem
  tile, smem[p*128 + c]); staging along tokens (channel-major tile) reads
  garbage and faults.
- ggml_view_3d's dim0 stride is ALWAYS the element size: ggml_new_tensor_impl
  resets nb (nb[0] = type_size, nb[1] = nb[0]*ne[0], nb[2] = nb[1]*ne[1]), and
  view_3d only overrides nb[1]/nb[2]. A token-strided window over a
  channel-fastest tensor CANNOT be expressed as a view; use transpose-of-view
  (ggml_view_tensor copies the parent nb, so transpose of a view keeps the
  correct strides). The first state-view attempt read a silent (p+c) channel
  mix (PPL 6.42, decode GPU fault) - caught only by the full PPL gate.
- The conv-state cache is position-fastest: row index p + c*(d_conv-1). The
  state-save cpy_scalar scatters by the src flat index (ne00 = n_state), and
  the next batch's build_rs (get_rows + reshape_3d) reads the same layout -
  the old concat-based save and the new qkv-tail transpose view are bit-exact
  (/tmp/dbg_state2.cpp: old-vs-new maxerr=0).
- Graph write-before-read hazard: a state-save cpy whose source has no data
  dependency on the state buffer (e.g. a qkv view) can run before the op that
  reads the state. The old concat-based save was implicitly ordered by the
  concat's dependency on conv_states. Fix: expand the reading op (the fused
  conv) before the cpy - the graph executes nodes in insertion order. Expanding
  it AFTER the cpy (even in the same function) does not reorder.
- The old quantize_fp8 was NOT bandwidth-bound: ~6.5 MB/launch in 25 us is
  ~256 GB/s (latency/barrier-bound, 1 float/thread + 2 barriers per CTA). The
  warp rewrite (quantize_fp8_warp<4>) gets ~256 GB/s -> ~500+ GB/s on pp512
  grids. Its grid is ceil(k/128/32) x n - on tiny decode grids (n <= 16) that
  is too few CTAs: 1.31 -> 2.04 us/launch REGRESSION. The launcher must keep
  the old kernel for n <= GGML_FP8_GEMV_MAX_N.
- quantize_fp8_warp uses the gfx12 v_cvt_pk_fp8_f32 (RDNA4-only builtin) and
  must be #if-guarded like the wmma kernels; the launcher falls back to the
  portable quantize_fp8 on other archs and when src1 is not 16-B aligned.
- The hardware fp8 encode (v_cvt_pk_fp8_f32 + fmed3f clamp, HIP's OCP E4M3
  recipe) is bit-exact vs the software encoder for all FINITE inputs; only
  NaN/Inf inputs differ (hw returns +/-448, sw returns the NaN encoding).
  Verified on 406K values incl. every rounding boundary/tie/subnormal/
  saturation case (/tmp/fp8_cvt_test.cu). The packed result puts fp8(a) in
  byte 0, fp8(b) in byte 1, and the idst operand is preserved in the other
  word - chaining idst across the two op_sel words packs 4 fp8 per register
  with no shifts/or.
- The fused-quantize-in-wmma experiment (mul_mat_fp8_wmma_fused, reads f32 in
  the staging): at saturation it adds ~42 us/launch to the gate GEMM (more
  than the 25 us standalone quantize it replaces) - the in-kernel quantize
  competes with the wmma/scale-pass pipes and the f32 staging is 4x the bytes.
  Wins only on small under-occupied shapes. Do not retry.

## 5. Suggested order for the next session

1. Re-bench on -dev ROCm2 (the 27B server holds ROCm0/1) to confirm the working
   tree matches section 0 (expect ~7162-7170 pp512; the box is noisy - prefer
   rocprof kernel times for A/B).
2. L1 (wmma), L3/L4 (GDN phase A/B), L7 (in-proj fp8), L9 (single-copy embedding)
   are done; L5 (conv fusion), L6 (quantize rewrite) too. L8 (small kernels) - skip
   unless a specific one shows up hot in a fresh profile.
3. If pp512 headroom is needed again, the remaining structural ideas are L1's kpacked
   smem layout for A (blocked on the aiter layout problem) and the decode-side fused
   quantize-in-gemv (the remaining decode quantize launches, ~+4% tg64 per the
   decode session notes).
