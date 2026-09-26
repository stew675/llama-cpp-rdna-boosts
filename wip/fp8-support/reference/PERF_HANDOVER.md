# PERF HANDOVER - FP8 prefill: state, history, and operational reference

**Purpose**: the operational reference for performance work on this box: what
was done, why, and the standing rules (kernel facts, failed experiments,
tooling, git state, session logs). The FORWARD plan - ranked levers, expected
wins, validation steps, and the suggested order - lives in `LEVERS.md`;
PERF_HANDOVER.md intentionally does not duplicate it. Start a session with
`LEVERS.md`, come here for history and specifics.

Everything was measured on this box (1x R9700 gfx1201, 64 CUs, -dev ROCm0)
with the fp8 GGUFs in `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B StewFP8).
Updated 2026-08-06 after the GDN correctness fix + phase B rewrite: the
chunked GDN is now faster than sequential and pp512 is at 6068.

Read `HANDOVER.md` for the project as a whole and `AITER_FINDINGS.md` for the
aiter inspection details. This file tracks ONLY the performance picture.

---

## 1. Scoreboard (measured, all pp512 unless noted)

| config | pp512 t/s | tg64 t/s | vs Q8_0 pp | vs Q8_0 tg |
|---|---|---|---|---|
| Q8_0 GGUF (target base) | 6177 | 90.9 | - | - |
| fp8 baseline (821d423ac) | 4930 | 71.0 | -20% | -22% |
| fp8 + aiter GEMM port (26b7be281) | 5546-5596 | 71.4 | -5.6% | -21% |
| fp8 now (decode session, commit below) | **5914** | **89.1** | **-4.3%** | **-2.0%** |
| fp8 now (chunked GDN, 2026-08-06) | **6068-6072** | **89.8-90.0** | **-1.7%** | **-1.2%** |
| fp8 now (phase A+B session, 2026-08-06) | **~6100-6200** (noisy box) | **89.8-90.1** | ~-1% | ~-1% |
| fp8 now (L2 weight repack, 2026-08-06) | **6319-6334** | **89.4-90.0** | ~+1.5% | ~-1% |
| fp8 now (L5 conv fusion, 2026-08-06) | **6538-6554** | **89.4-90.0** | ~+5% | ~-1% |
| fp8 now (L6 quantize rewrite, 2026-08-06) | **6818-6841** | **89.2-89.6** | ~+7% | ~-1% |
| fp8 now (L7 in-proj fp8, 2026-08-06) | **7184-7260** | **88.3-88.5** | ~+13% | ~-1.5% |
| fp8 now (L9 single-copy embd, 2026-08-06) | **7162-7170** | **88.3-88.6** | ~+13% | ~-1.5% |
| fp8 now file size (L9) | 4.17 GiB / 4.33 B | vs 5.35 GiB / 4.96 B (L7) | -22% | |
| target prefill (+50% over Q8_0) | 8770 | - | +50% | - |
| target generation (+10%) | - | ~100 | - | +10% |

pp256: 5014, pp1024: 5446, pp2048: 5292, pp128: 3661 (high variance, use -r 5).

Current authoritative numbers (2026-08-06, chunked GDN default, L9 session):
see LEVERS.md section 0. The box is noisy (pp512 +/-60-170, occasionally much
worse): trust rocprof kernel times (deterministic +/-3%) for A/B. GDN per
pp512: phase A ~4.9 ms + phase B ~5.9 ms. quantize_fp8 per pp512: ~2.5 ms.
BENCH ON -dev ROCm2: the box has 3x R9700 and a Qwen3.6-27B llama-server holds
ROCm0/1 most of the time (llama-bench OOMs on an occupied GPU and llama.cpp
silently CPU-offloads layers, which looks like a perf regression).
The canonical GGUF is now the L9 single-copy build (fp8 token_embd, fp8 gates,
no output.weight); the two-copy L7 reference is stewfp8-ow-2copy.gguf.

PPL 6.2250 (L9) vs 6.2528 (L7) vs 6.2464 (Q8_0) on the small
/tmp/corpus_pride.txt corpus (all within error bars); generation matches the
pre-change fp8 output on the same prompt/seed (llama-cli spawns its own local
server - point it at the same GPU as the benches).

## 2. Where pp512 time goes (historical, decode session; current: LEVERS.md section 1)

| kernel | share of pp512 | note |
|---|---|---|
| mul_mat_fp8_wmma | ~48% | ~77 TFLOP/s (aiter port) |
| gated_delta_net_cuda | ~17% | sequential per-token loop; now replaced by the chunked path |
| Cijk (rocBLAS/CK BF16 GEMM) | ~6% | the delta-net in/out projections (BF16 in the GGUF) |
| quantize_fp8 | ~6% | activation staging pass |
| concat_non_cont | ~5% | delta-net plumbing |
| flash_attn_tile | ~4% | 8 attention layers |
| silu / norms / conv / copy | ~9% | ssm_conv_long_token_f32 ~1% |

The per-kernel shares are still representative; the current measured breakdown
(per-pp512, warmup removed) is in LEVERS.md section 1.

Decode (tg64) breakdown, per ~11.1 ms step: mul_mat_fp8_gemv 8.95 ms (79%, 226
GEMMs incl. lm_head at ~496 GB/s), quantize_fp8 0.32 ms, gated_delta_net 0.41
ms, BF16 mmv 0.15 ms, norms/ops ~1.3 ms.

## 3. Ranked levers - SEE LEVERS.md

The ranked levers, expected wins, and validation steps moved to `LEVERS.md`
(updated 2026-08-06 after the GDN work). Summary of what changed there vs the
old L1-L5 below:
- the delta-net port from aiter (old L1) is dead: aiter's chunk kernel does
  not run on gfx1201 (see section 11), and the chunked GDN now beats the
  sequential kernel anyway (section 12).
- the GDN levers are now internal: phase A (gdn_chunk_prepare) and phase B
  (gdn_chunk_state), both documented in LEVERS.md (L3/L4).
- the GEMM levers (occupancy via 24 KB smem tiles, weight repack) survive as
  LEVERS.md L1/L2 (both spent); the quantize lever is L6 (DONE - the fusion
  approach in the original L6 was measured and abandoned in favor of a warp
  rewrite, see section 17).
- the non-fp8-tensor quantization lever (old L5) is folded into LEVERS.md L7
  (Cijk BF16 projections); the quantizable-set note below is still valid.

RESOLVED (L9): the old warning against quantizing token_embd is obsolete. An
fp8 get_rows kernel now exists (getrows.cu k_get_rows_f8 + the CPU get_rows
path via traits->to_float = dequantize_row_f8_e4m3), so token_embd can be
stored once in fp8 and serve both the lookup and the lm_head (no output.weight
copy). The old quantizable set (norms, conv1d, in_proj_a/b, A_log) is closed:
in_proj_a/b are F8_E4M3 (L7); norms/conv1d/A_log are not GEMMs.

## 4. Kernel facts (established this session, do not rediscover)

- **Occupancy is the dominant GEMM lever on gfx1201**: 8 warps/CU = 58 TFLOP/s,
  16 warps/CU = 77 TFLOP/s. The aiter triton kernel's 121 TFLOP/s comes from
  its 24 KB smem tile running 2 CTAs/CU (16 warps) - NOT from its 2-stage
  pipeline. Double-buffered 128x64 (61 KB LDS) = 1 CTA/CU = the old 55-58.
- Raw wmma 16x16x16 fp8 ceiling: ~165 TFLOP/s (conservative microbench; codegen
  dependent up to ~430). Current kernel = 47% of the conservative ceiling.
- gfx1201 device props: smem/block = 65536, regs/mp = 196608, maxThreads = 2048.
- Current kernel shape: CTA 128 rows x 64 tokens, 8 warps (4m x 2n, 2x2 tiles),
  single-buffered smem 26.4 KB (sA 128x136 + sB 64x136 + sS 64), register-staged
  k-block pipelining (loads cb+1 into regs before cb's wmma chain, commit after
  the first of 2 barriers per k-block), grouped-M swizzle GROUP_SIZE_M=4.
- VGPR ~200, LDS 26,624 -> 2 CTAs/CU.

## 5. Failed experiments (do not retry without a reason)

| experiment | result |
|---|---|
| 16 warps/CTA, 2x1 tiles/warp (2 accs) | 4936 t/s (regression; fragment reuse 1.5 LDS/wmma vs 1.0) |
| Burst fragment array `fp8x8_t fa[4][4]` | **LLVM MISCOMPILE**: 16/32 wmma get v[0:1] operands under the single-buffer loop (correct in double-buffer variant). Reverted to software-pipelined loads. |
| GROUP_SIZE_M=8 | noise (5570 vs 5546) |
| Double-buffered smem (61 KB) | 1 CTA/CU -> 5110 t/s |
| (earlier session) 8-warp 64x128 CTA, register prefetch, aligned-A split, MT=4 | all < baseline; see HANDOVER.md section 8 |

The GDN phase B failed designs (2026-08-06) are in section 12. The
consolidated do-not-rediscover list (both GEMM and GDN) is LEVERS.md section 4.

## 6. Repro / tooling

The full session protocol (build + bench + correctness gates + profile) is in
LEVERS.md section 3. Below: the fp8-specific extras that LEVERS.md does not
repeat.

```bash
# Q8_0 reference on the same box/day
/tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -p 512 -n 64 -r 5 -dev ROCm0

# GEMM correctness (fast gate)
/tmp/llama-hip-full/bin/test-backend-ops test -b ROCm0 -o MUL_MAT   # 1186 pass (no F8 cases! it only covers other types)
# note: test-backend-ops has NO F8_E4M3 mul_mat cases (CPU backend can't do fp8) - generation diff is the gate

# generation parity (fp8 changes)
/tmp/llama-hip-full/bin/llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -f /tmp/prompt_pp.txt -n 24 --single-turn -s 42
# expected: prompt echo + "[Start thinking] Thinking Process: 1. Analyze the Request:..." (matches bf16 ref)

# profile (decode-heavy runs crash rocprofv3 at teardown; the trace is still written)
rocprofv3 -r -d /tmp/rocp -f csv -- /tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 8 -r 1 -dev ROCm0
# per-kernel CSV in /tmp/rocp/soar/*kernel_trace.csv (Grid_Size_X = work-items = grid*256; VGPR/LDS columns)
# decode-only view: filter kernels after the last mul_mat_fp8_wmma start
# REMEMBER: llama-bench's warmup pass doubles every kernel count/time - divide by 2 for per-pp512
```

## 7. PPL / parity (do not skip after any kernel or loader change)

```bash
/tmp/llama-hip-full/bin/llama-perplexity -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0 2>&1 | tr -d '\0' | grep "Final estimate"  # ~6.26 on this corpus
```

## 8. Git state

- Branch `cllm`, origin git@github.com:stew675/llama.cpp.git.
- Commit style: concise subject, `Assisted-by:` line, no Co-authored-by.
- Recent commits (2026-08-06):
  - 56e7cb0c9 gated_delta_net : rewrite chunked phase B, 33.8ms -> 12.8ms
  - ef41a940a gated_delta_net : fix chunked phase A closure, enable by default
- Earlier (decode session, 2026-08-05): fp8.cu + fp8.cuh (gemv rewrite + memset
  skip), convert_hf_to_gguf.py + conversion/base.py (--fp8-output-weight),
  docs (PERF_HANDOVER.md, vllm-vs-llamacpp-performance.md).
- Phase A/B session (2026-08-06): a921f8fb0 (phase A rewrite, 293 -> 210 us),
  c8757d649 (phase B delta loads, 269 -> 246 us). See section 13.

## 9. Next-session suggested order - SEE LEVERS.md

The suggested order, and the current state to confirm first, moved to
LEVERS.md section 5 (re-bench ~6818-6841 -> L7 -> re-evaluate).

## 10. Decode session log (2026-08-05 evening)

Goal: close the fp8 generation gap to Q8_0 (tg64 was 71.0-71.4 vs Q8_0 ~90).
Result: tg64 72.0 -> 89.1 (98% of Q8_0's 90.9), pp512 unchanged ~5914.

Three changes, all gated by generation-parity + PPL:

1. fp8.cu: skip the n_pad zero-fill (cudaMemset x2 per GEMM, 450/step) when
   n <= GGML_FP8_GEMV_MAX_N; the GEMV path never reads the padded tokens.
   Removed ~0.5 ms/step.
2. fp8.cuh: rewrote mul_mat_fp8_gemv - fold the per-block scales into the lane
   partial (one deferred warp-reduce instead of 160 shuffles/row), 4 k-blocks
   per iteration for load ILP. Bandwidth 445 -> 496 GB/s (mmvf reference: 520).
3. convert_hf_to_gguf.py + conversion/base.py: --fp8-output-weight writes an
   fp8 output.weight copy of token_embd (same 128x128 convention as the model:
   d = BF16(amax/448), q = fp8(w/(amax/448))). lm_head moves from the fused
   BF16 mmv (2.44 ms/step) to the fp8 GEMV (~1.4 ms/step); token_embd stays
   BF16 for get_rows. +0.66 GiB storage, +0.63 B params.

New GGUF: /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf (5.36 GiB, 442
  tensors: token_embd BF16 + output.weight F8_E4M3). Regenerate with:
  python3 convert_hf_to_gguf.py /llm/models/Qwen3.5/4B/StewFP8 --outtype fp8_e4m3 --fp8-output-weight --outfile ...

Left on the table (decode): gemv 496 -> ~520 GB/s (vectorized x loads, 8-block
ILP), fuse quantize_fp8 into the gemv (226 launches/step), delta-net 0.41
ms/step. Roughly +4% if all taken.

## 11. Delta-net chunked session log (2026-08-05 night)

STATUS 2026-08-06: superseded - the chunked path is now default-on, correct
(closure fix, ef41a940a) and faster than sequential (phase B rewrite,
56e7cb0c9). See section 12 and LEVERS.md. The log below is historical.

Goal: speed up gated_delta_net_cuda (17% of pp512) by replacing the sequential
per-token loop with a chunked formulation.

KEY FINDING: aiter's chunk_gated_delta_rule_fwd_h.cu does NOT run on gfx1201 -
op_tests/test_gated_delta_rule.py skips it on gfx12
("kernel does not support gfx12!"). The handover's "aiter has a gfx1201 port"
premise was wrong. The __gfx1201__ guard there is only for a buffer-resource
constant. So no port; we implemented our own chunked kernel.

Status: CORRECT but SLOWER. ggml-cuda/gated_delta_net.cu has gdn_chunk_prepare
(phase A: per head x seq x chunk of 64 - decay cumsum, closure (I-A)^-1 via
forward substitution, k_cumsum = attn @ (beta v), k_cumdecay = attn @ (beta k P),
attn_causal = L .* (q^T k)) and gdn_chunk_state (phase B: per head x seq x
V-slice of 8 - v_new = k_cumsum - k_cumdecay @ S, o = P .* (q @ S) + attn_causal
@ v_new, S = P_last S + (k .* delta) @ v_new). Opt-in via GGML_CUDA_GDN_CHUNKED=1;
off by default because pp512 = 4200 vs 5925 sequential.

Math: validated in python (/tmp/validate_chunked.py) against the op recurrence
(aiter's chunk_gated_delta_rule_ref structure, decay kept inside via
L[i][j] = exp(decay_i - decay_j)). All 45 GATED_DELTA_NET backend-ops tests pass
on BOTH paths (add 4 S_v=128 prefill cases to the eval set). PPL 6.30 vs 6.26
baseline; generation matches.

Bugs found and fixed (do not rediscover):
- P = exp(cumsum(gate)) underflows to 0 for strong decay (cumsum < -88) ->
  P_i/P_j = 0/0 = NaN. Keep the cumsum and always take exp(diff).
- Phase-A scratch was indexed per chunk only, not per (head, seq): heads raced
  on the same scratch (outputs looked like adjacent heads' data).
- The decay_scratch store used tid = 0..127 over a 64-float per-head region:
  threads 64-127 wrote garbage into the next head's region (nondeterministic
  per-head corruption; single-head tests passed, multi-head flaky).
- cudaStreamSynchronize / cudaEventCreate between the two kernels is ILLEGAL
  during CUDA graph capture ("operation not permitted when stream is
  capturing"). The kernels are on the same stream; no sync needed.

Why it is slow: ~2x the FLOPs of the sequential kernel (closure + materialized
k_cumsum/k_cumdecay/attn_causal) with untiled matmuls reading global memory and
126 barriers in the closure. Next session: tile the matmuls with smem staging
(phase A smem budget: s_attn 16.6 KB + s_k 32 KB fits one CTA/CU), reduce the
closure barrier count, then re-enable by default.

## 12. Delta-net chunked: correctness fixed + phase B rewrite (2026-08-06)

The chunked GDN is now BOTH correct and faster than the sequential kernel.
Committed ef41a940a fixed the phase A closure (block forward substitution,
see GDN_DEBUG_HANDOVER.md). This session rewrote phase B.

Phase B before: gdn_chunk_state, 128 threads, V-slices of 8 (16 slices),
scalar serial reductions, stride-8KB global k loads in the S update.
33.82 ms per pp512 bench (48 launches x 704us, grid 32x16=512 CTAs).

Phase B now: 256 threads, V-slices of 32 (4 slices), one state column per
thread (32 cols per warp -> kd/q/k/ac loads are warp broadcasts, no L2
amplification), 8 token rows per thread (steps 1/2) or 16 state rows
(step 3). s_S (16 KB) + s_vnew (8 KB) smem -> 2 CTAs/CU, 1 wave (grid 128).
6.4 ms per pp512 (24 launches x 267us; the trace shows 48 x 267us = 12.8 ms
because llama-bench runs a warmup pass). VGPR 160, scratch 0, LDS 24.6 KB.

Scoreboard (pp512, stewfp8-ow, -p 512 -n 64 -r 5):
- chunked before this session: 5436 (broken-ish phase B)
- chunked now:               6068-6072  (sequential was 5878)
- tg64 unchanged:            89.8
PPL 6.2426 (sequential 6.2572), 47/47 GATED_DELTA_NET tests.

Failed experiments this session (do not retry without a reason; consolidated
with the GEMM list in LEVERS.md section 4):
- 16x16 thread grid with 4x2 register tiles + smem-staged kd/q/ac/k blocks:
  192 VGPRs + 764-1028 B scratch spills -> 274 scratch instructions vs 308
  FMAs in the ISA; 675-704us/launch. The register-tiled layout spreads the
  V dimension across threads, which both amplifies shared-input L2 reads 16x
  and blows the register budget.
- 16x16 grid, no staging (direct L2 float4): 591-644us/launch (L2-bound).
- __launch_bounds__(256,2) forcing 128 VGPRs: worse (more spills, 644us).
Lesson: for this kernel shape (64x128x32 matmuls, S resident), thread-per-
column with 32 cols/warp is the right layout - it makes every shared-input
load a warp broadcast and keeps live registers near 40.

Remaining GDN perf: the forward plan is LEVERS.md (L3 = phase A, L4 = phase B).
Summary: phase A (gdn_chunk_prepare) is now the bigger kernel (~7.1 ms/pp512,
~8x its FMA floor, ~154 dynamic closure barriers, 1 CTA/CU); phase B is ~10x
its FMA floor (chunk-loop latency).

## 13. Delta-net phase A rewrite + phase B delta loads (2026-08-06, session)

Goal: L3 (phase A closure barriers) per LEVERS.md, then re-evaluate. Result:
phase A 293 -> ~205-210 us/launch (-24%), phase B 268.8 -> ~246 us/launch
(-8.5%). GDN per pp512 13.5 -> ~10.9 ms. Commits a921f8fb0 + c8757d649.

Phase A (gdn_chunk_prepare) changes, all validated 47/47 + PPL + generation:
1. closure: the 16x16-block forward substitution (146 syncs) replaced by an
   exact 4x4-blocked substitution over the whole 64x64 (16 syncs). The 4x4
   diagonal blocks have nilpotency index 4, so D(i,i) = I + Aii + Aii^2 +
   Aii^3 (computed in registers via a warp-shuffle row substitution, 1 sync);
   the off-diagonal blocks D(i,j) = D(i,i) @ (A(i,j) D(j,j) + sum_{j<k<i}
   A(i,k) D(k,j)) follow in 15 row passes. The row passes read A(i,k) from a
   preserved packed strictly-lower smem copy s_A (8 KB; the in-place D writes
   would clobber A otherwise - the row-pass ordering trick does NOT work
   because blocks of the same row run concurrently in different warps). Numpy
   check /tmp/validate_closure_4x4.py: exact to 1e-15 in both random and
   model-like strong-decay regimes. The first implementation had a bug
   (s_A[e - j] instead of s_A[e], NaN) - the packed index IS e.
2. decay cumsum: Hillis-Steele (13 syncs) -> warp-shuffle scan (2 syncs);
   the chunk's k rows are prefetched into registers during the scan (overlaps
   the L2 latency of the staging loads, LEVERS.md approach b).
3. k_cumsum/k_cumdecay and attn_causal: v[j][v0..v0+7] and the q rows were
   512 scalar global loads per thread; now float4 (2 per v row / 4-wide d
   unroll for q) with a runtime alignment fallback ((stride & 3) == 0 &&
   pointer 16-B aligned). This was the REAL win - the sync reduction alone
   (165 -> 23 barriers) moved phase A only 293 -> 276 us.

Phase A stage breakdown (stage-elimination stubs, +/-3%): v loads ~81 us,
q loads ~37 us, gram ~34 us, closure ~32 us of the original 293.

Phase B (gdn_chunk_state): step-3 delta reads vectorized (64 scalar loads per
thread per chunk -> 16 float4 with a clamped k-row index so padded rows stay
in-bounds and contribute zero - a plain guard branch nullified the win).
Stage-elimination: kd/q loads ~112 us, ac+k loads ~62 us, FMA/sync/expf floor
~206 us of 269. The kernel is ~65% load-stall-bound but warp-broadcast loads
+ the serial S dependency kill the obvious fixes: a 2-stage kd/q register
pipeline hit 256 VGPR + spills (495 us), #pragma unroll on the r-loop
regressed (349 us). No cheap remaining lever; revisit with a fresh profile.

Also learned: the box got noisy late 2026-08-06 (pp512 +/-60-170, one run
+/-1000). rocprof kernel times are the reliable A/B signal. fp8 gemv traffic
in the trace stays at ~40 us avg regardless (not affected by these changes).

## 14. wmma GEMM investigation (2026-08-06 evening) - L1 measured as spent

Goal: LEVERS.md L1 (mul_mat_fp8_wmma 2 CTAs/CU, the last big pp512 lever).
Outcome: the lever's premise was already satisfied (2 CTAs/CU since 823f70a3);
the residual gap to the aiter triton kernel (121-137 vs our 92-98 TFLOP/s at
the same occupancy) was investigated and measured as NOT reachable by the
obvious ports. No perf change committed - the tree is unchanged.

Method: standalone rig /tmp/bench_wmma.cu (kernel-only hot loop on the exact
model shapes, min of 3, +/-1%) + /tmp/check_wmma.cu (CPU-reference
correctness). Stage-elimination stubs on the gate GEMM (9216x512x2560, 250 us):

| stub | gate us | TFLOP/s | meaning |
|---|---|---|---|
| none (baseline) | 262.6 | 92.0 | |
| fragment LDS removed | ~258 | ~94 | fragment chain nearly free (~4 us) |
| scale pass removed | 216.8 | 111.5 | scale pass ~33 us (13%) |
| strided scale staging removed | 250.7 | 96.4 | rD/sS loads free (L1 handles) |
| scale pass, sA_d restructure | 245.8* | 98.3 | float4 wd loads: no real gain |

* = with the (later reverted) fragment pipeline; the sA_d-only restructure
measured 261.6 - neutral.

Findings:
1. The scale pass (~13%) is FMA/serialization-bound, not LDS-bound: moving the
   per-row scales to a contiguous smem array (18 -> 6 LDS per warp per
   k-block) did not help. The 32 FMAs/thread/k-block are irreducible and the
   per-k-block weight scale cannot be deferred to the epilogue (it varies per
   k-block).
2. The staging/barrier/wmma floor is ~85%. A 2-k-block staging lookahead
   (double-buffered registers) regressed ~4% (scheduling). The 4-B weight
   staging loads coalesce fine; L2 traffic (~280 MB/launch) is not the
   limiter (1.1-1.6 TB/s used of ~3).
3. aiter's kpack=2 does not port: the 16-k wmma fragment is split across lane
   halves, so no single wider LDS load covers 2 k-steps without a swizzled
   smem store side; triton's num_stages=2 doubles smem (48 KB = 1 CTA/CU =
   the measured 58 TFLOP/s config).

NEW COMPILER MISCOMPILE (do not retry): a 2-k-step fragment lookahead with a
named-variable rotation (a0_1/a0_2) miscompiles under the single-buffer loop -
LLVM folds the rotation copies and step 1 reuses step 0's operand registers
(verified in SASS: 8 v_wmma with identical v[30:31]/v[26:27]/v[34:35]/v[38:39]
operands; PPL 183142). The 1-k-step lookahead (a0_n) compiles correctly. Same
class as the burst-array bug; rule: no register rotation of fragments across
k-steps in the single-buffer loop.

CHECK LESSON (important for the project): the pre-existing assumption that
"standalone fp8 correctness passes" was FALSE for random data - random fp8
payload bytes contain NaN encodings (0x7F/0xFF) and random f32 scales are
mostly NaN/inf, so fabs(ref-out) is NaN everywhere and maxerr/bad stay 0.
A false pass nearly shipped a miscompiled kernel. /tmp/check_wmma.cu now uses
realistically-quantized data (host fp8 encoder, sane scales) and correctly
flags the miscompile (maxerr 34, 7991/8192 bad). Any future fp8 kernel change
must run this check (or the PPL gate) with proper data.

Conclusion for LEVERS.md: L1 is spent unless a new structural idea appears
(e.g. a swizzled kpacked smem layout for A, which the L2 repack would enable).
The remaining GEMM lever with real headroom is L2 (weight repack at load
time): 16-B staging loads + it unblocks the kpacked layout.

## 15. L2 wmma weight repack (2026-08-06 evening, session)

Goal: LEVERS.md L2 (wmma weight repack at load time). Result: wmma kernel
~204 -> ~186 us/launch average (-9%), pp512 6220 -> 6319-6334 (+1.6-1.8%),
tg64 unchanged. PPL 6.2677 exact; generation matches; 47/47 GDN tests.
Commits: (see git log).

## 16. L5: ssm conv concat fusion (2026-08-06 evening, session)

Goal: LEVERS.md L5 (concat_non_cont, 4.7 ms/pp512 - the delta-net conv input
assembly). Result: concat eliminated, pp512 6303 -> 6538-6554 (+4%), tg64
unchanged, PPL 6.2677 exact, 47/47 GDN + 83/83 SSM_CONV.

Design: new op form ggml_ssm_conv_2src(ctx, conv_states, qkv, c) - the same
GGML_OP_SSM_CONV op code with optional src[2] = conv states (nullptr keeps the
old single-input behavior for mamba/kimi/etc.). The fused CUDA kernels
(ssm_conv_2src_f32 for n_t <= 32, ssm_conv_long_token_2src_f32 for n_t > 32)
read conv_states (position-major) and qkv_mixed (channel-fastest, untransposed)
in their natural layouts into a position-major smem tile, coalesced along
channels. The model graph (build_conv_state in delta-net-base.cpp) drops the
transpose + concat + conv_input buffer; the state-save view is now the
transpose of a qkv-tail view, and the concat is only built when a state window
overlaps the conv_states piece (n_t < d_conv-1 or rollback slots with small
batches) - decode keeps it at trivial cost.

Bugs found and fixed (do not rediscover, consolidated in LEVERS.md section 4):
1. The qkv layout is channel-fastest: qkv[c][t] at c*4 + t*nb1. A fixed
   channel's token window is strided by d_inner*4, so staging must be coalesced
   along CHANNELS (position-major tile, smem[p*128 + c]); the first version
   staged along tokens (channel-major tile) and read garbage (GPU faults in
   test-backend-ops, caught by the new test_ssm_conv_2src cases).
2. ggml_view_3d CANNOT express a strided dim0: ggml_new_tensor_impl resets nb
   (nb[0] = element size, nb[1] = nb[0]*ne[0], ...), so a view's dim0 stride is
   always 4 B for f32. A "token-major window over qkv" view is impossible; the
   state save must use transpose-of-view (tail = view_3d(qkv, channels, n_state,
   seqs, token_stride, seq_stride, offset); T = transpose(tail)), which keeps
   the parent nb via ggml_view_tensor. The first attempt (view_3d over a
   transposed qkv) silently read qkv[c'][509] with a (p+c) channel mix -> PPL
   6.42, exactly reproducible, and decode crashed with a GPU fault.
3. The state cache layout is position-fastest: row index p + c*(d_conv-1)
   (the save cpy_scalar scatters by the src flat index with ne00 = n_state, and
   the next batch's reshape_3d read matches). The qkv-tail transpose view
   reproduces the old concat-based save bit-exactly (standalone check
   /tmp/dbg_state2.cpp: old-vs-new state save maxerr=0).
4. Graph write-before-read hazard: the old state-save cpy depended on the concat
   (which depends on conv_states), so it ran after the conv read the state; the
   fused conv + qkv-view cpy have no dependency, so the cpy can overwrite
   conv_states_all before the conv reads it. Fix: expand the fused conv before
   the state-save cpy in build_conv_state (graph executes nodes in insertion
   order). A naive "expand conv at the end" fix did NOT reorder (the cpy was
   already inserted).

Method notes: the CPU reference (ggml_compute_forward_ssm_conv_f32) needed the
same channel-fastest fix (q[i1 + t*stride_q], not q[i1*stride_q + t]) plus the
ir0 channel offset on q/cs in the multi-thread row partition (the first version
read channel 0's data for every thread - caught by gpu-vs-cpu comparison in
/tmp/dbg_2src2.cpp). test-backend-ops compares against the CPU backend with
use_ref=true, which only disables fusion (the compute is still ops.cpp).

The remaining concat-ish plumbing: none. ssm_conv 0.6 ms/pp512 (24 launches).
Next lever per LEVERS.md: L6 (quantize_fp8 fusion).

Goal: LEVERS.md L2 (wmma weight repack at load time). Result: wmma kernel
~204 -> ~186 us/launch average (-9%), pp512 6220 -> 6319-6334 (+1.6-1.8%),
tg64 unchanged. PPL 6.2677 exact; generation matches; 47/47 GDN tests.
Commits: (see git log).

Method:
1. Re-measured the staging share at the current 93 TFLOP/s with the rig
   (/tmp/stub_rig/fp8.cuh: replaced the rA/rD weight-side staging loads with
   constants, kept the smem stores/barriers/wmma chain): 260.5 -> 175 us, so
   the weight staging loads were ~85 us = 33% of the kernel - far above the
   old ~15% estimate from the 55 TFLOP/s era. L2 premise confirmed.
2. Simulated the repacked layout in a rig (/tmp/bench_repack.cu + the shared
   /tmp/repack_kernel.cuh): contiguous fp8 [m_pad][k] rows + separate scales
   [m_pad][n_col_blocks], uint4 staging loads, no bounds predicates (padded
   rows pre-zeroed). 260.5 -> ~228 us on the gate shape = 93 -> 106 TFLOP/s
   (+14%). Bit-exact vs the block-layout kernel (0 diffs of 4.7M outputs,
   /tmp/check_repack.cu).
3. Implemented in-tree: fp8_repack_weights kernel (one warp per row, k-block;
   grid.y chunks rows, loop r += gridDim.y) + lazy per-context cache in
   ggml_cuda_fp8_repack (keyed on src0->data, shape-checked; allocated via
   cudaMalloc - safe because the first call always happens during the
   direct-execution warmup before CUDA-graph capture). The GEMV and scalar
   paths keep reading the original block layout (decode untouched).

BUG FOUND (do not rediscover): the repack kernel looped `r < gridDim.y`
instead of `r < m`. For m > 65535 (output.weight, vocab 152064) grid.y is
clamped to 65535, so rows beyond 65535 were never repacked -> garbage logits
-> PPL 9.56 (was 6.24). The bit-exact kernel-vs-kernel rig did NOT catch it
(its repacked buffer was host-built); the isolated repack-kernel check
(/tmp/check_repack_kernel.cu, compares device repack vs host repack on the
real model shapes incl. m=70000) caught it. Rule: any device-side repack/
transform kernel needs its own isolated content check, including the m >
grid-y-limit case.

Also learned: the fp8 weight tensors get fresh data pointers (and fresh names)
on every graph build in this codebase, but CUDA-graph replay means the lazy
pointer-keyed cache is populated once during the first direct execution and
reused afterwards - verified stable at ~201 entries over a 2200-call PPL run
(no unbounded growth, no per-chunk repack cost). The one-time repack cost is
~13 ms total (~200 tensors, p50 71 us, max 1.09 ms for output.weight).

## 17. L6: quantize_fp8 warp rewrite (2026-08-06 night, session)

Goal: LEVERS.md L6 (quantize_fp8, 5.1 ms/pp512). The L6 plan was to fuse the
quantization into the wmma kernel's staging; the session found the fusion is a
wash and the real win is a barrier-free warp rewrite of the quantize kernel.
Result: quantize 25.4 -> 12.6 us/launch on pp512 grids (-50%), pp512 6630 ->
6818-6841 (+2.9-3.2%), tg64 unchanged, PPL 6.2677 exact, generation matches,
47/47 GDN. Commits: (see git log).

Why the old kernel was slow: it moves ~6.5 MB/launch in 25 us (~256 GB/s),
far below the DRAM budget - it is latency/barrier-bound (1 float per thread,
2 __syncthreads per 128-thread CTA, block reduce via smem).

New kernel quantize_fp8_warp<4> (fp8.cuh): one warp per (token, 128-col
block); each lane covers 4 consecutive values (float4); the amax reduce is a
pure warp butterfly (5 shfl, no smem, no barriers); the fp8 encode uses the
gfx12 hardware v_cvt_pk_fp8_f32 (RNE). Bit-exactness of the hardware encode
vs the software encoder was verified on 406K finite values covering every
rounding boundary, tie, subnormal and saturation case (/tmp/fp8_cvt_test.cu):
only NaN/Inf inputs differ (never present in activations). 4 k-blocks per
warp overlap the DRAM latency of the independent loads (KBW=1/2/4/8 measured;
4 best). The staging output is bit-exact vs quantize_fp8 (rig check bad=0 on
all model shapes), so the wmma/gemv/scalar consumers are untouched.

Dispatch (fp8.cu): RDNA4 + n > GGML_FP8_GEMV_MAX_N + 16-B-aligned src1 -> warp
kernel; else the old quantize_fp8. The warp kernel's grid (ceil(k/128/32) x n)
is too small for the decode grids (n <= 16): it REGRESSED decode quantize 1.31
-> 2.04 us/launch, so the old kernel stays there (decode untouched).

The fusion experiment (mul_mat_fp8_wmma_fused, reads f32 in the wmma staging,
computes scales in-kernel via an aligned-8-lane butterfly): measured on the
real shape mix - it WINS on small shapes (m=1024: -51%, m=4096: -16%) but
LOSES on the big GEMMs that dominate pp512 (m=9216: +10%, m=8192: +5.5%);
net ~ +0.2 ms/pp512 (a wash). The in-kernel quantize adds ~42 us/launch at
saturation - more than the 25 us standalone kernel it replaces - because it
competes with the wmma/scale-pass pipes and the f32 staging is 4x the bytes.
The warp rewrite gets the same win with zero risk to the GEMM path. Do not
retry the fusion. (The fused kernel was removed from the tree; the rigs
/tmp/bench_fused.cu and /tmp/bench_pairs.cu keep the A/B.)

Next lever per LEVERS.md: L7 (Cijk BF16 -> fp8 projections, ~5.3 ms/pp512).

## 18. L7: delta-net in/out projections to fp8 (2026-08-06, session)

Goal: LEVERS.md L7 (the Cijk BF16 GEMMs, 5.3 ms/pp512). Result: the two
remaining BF16 GEMM weights in the delta-net path (in_proj_a/b =
ssm_alpha/ssm_beta, [32, 2560] each) are now F8_E4M3 in the GGUF, so the 48
Cijk launches/pp512 run on the fp8 path instead. pp512 6818-6841 ->
7184-7260 (+5-6%), tg64 89.2-89.6 -> 88.3-88.5 (-1%), PPL 6.2528 (was
6.2677, within noise), generation matches, GDN + SSM_CONV backend tests
pass. Commit: (see git log).

Why the Cijk kernels were 5.37 ms/pp512: 48 launches (24 layers x alpha +
beta) of a 32x512x2560 BF16 GEMM at ~112 us each - a thin m=32 shape that
CK handles terribly (latency/grid bound, ~0.75 TFLOP/s effective). The fp8
path per launch is ~11 us quantize_fp8_warp + ~24 us wmma (8 CTAs, m_pad
128 so 4x the useful FLOPs on the row side) = ~35 us, ~3.2x faster. The
alpha/beta GEMMs read the SAME cur activation, so each is quantized
separately (2x the quantize work on that tensor).

Conversion: new `--fp8-delta-net-in-proj` flag (convert_hf_to_gguf.py),
wired via conversion/base.py; packing in
`_LinearAttentionVReorderBase._generate_fp8_in_proj` (conversion/qwen.py).
The BF16 weight is V-head-reordered exactly as modify_tensors does
(head_dim=1), then packed block_f8_e4m3: d = BF16(amax/448) per 128-col
block, q = fp8(w/d). The 32 gate rows share ONE 128-row scale block
(ceil(32/128) = 1) - same as the model's own scale convention, and the
measured quantization error is the same regime as the model's fp8 tensors
(mean rel ~2.25% vs 2.256% in the conversion summary). Regenerate the
canonical GGUF with the same invocation plus the new flag; the pre-change
file is kept as stewfp8-ow-bf16gate.gguf.

tg64 cost (-1%) and its recovery: in decode (n <= 16) the alpha/beta GEMMs
moved from the bf16 mmvf path to quantize_fp8 (old kernel, ~1.28 us) +
mul_mat_fp8_gemv, adding ~48-54 quantize launches per decode step (~60-70
us/step, ~0.5-0.6% of the 11.1 ms step; the rest of the -1% is the extra
launch/scheduling overhead). The recovery lever is the fused
quantize-in-gemv already listed in section 10 as a decode item ("fuse
quantize_fp8 into the gemv, 226 launches/step") - it would recover this
plus the other 226 decode quantize launches (~+4% tg64 per that note). Not
done here: decode is not the L7 target and the fused gemv is a separate
kernel project.

Also closed out: the rest of the old quantizable set (norms, conv1d, A_log)
are not GEMMs - no fp8 kernel change applies. token_embd stays BF16 (get_rows
has no fp8 kernel); eh_proj is not in the 4B main path. The only fp8-path
quirks for the new tensors: m=32 < GGML_FP8_CTA_M (128), so the wmma kernel
runs with m_pad 128 and its existing mi >= m bounds check handles the
padding; the fp8_repack_weights launcher handles m < 65535 with grid.y = m
(no row loop needed); the block_f8_e4m3 scale grid is [1, 20] (one row
block), which np.repeat(d, 128, axis=0)[:M] replicates per row correctly.

## 19. L9: single-copy fp8 token_embd (2026-08-06, session)

Goal: eliminate the duplicate embedding (token_embd BF16 + output.weight fp8
were two copies of the same 0.64 B-param matrix) so the file shrinks without
losing the fp8 lm_head speed. Result: token_embd is now F8_E4M3 in the GGUF
(per-row scales) and output.weight is gone - the graph's tied-embedding
fallback (qwen35.cpp: output == NULL -> TENSOR_DUPLICATED token_embd) makes
the head read the same fp8 tensor. File 5.35 -> 4.17 GiB (-22%), 4.96 ->
4.33 B params, pp512 7171 -> 7162-7170 and tg64 88.5 -> 88.3-88.6 (SAME
speed), PPL 6.2250 (was 6.2528), generation identical, all backend tests
pass (GET_ROWS now covers fp8, bit-exact vs CPU). Commit: (see git log).

New kernels/paths:
1. CUDA get_rows for F8: k_get_rows_f8 (one CTA per (row, 128-col block),
   256 threads, per-row scale d, fp8_e4m3_to_f32 decode moved to common.cuh so
   getrows.cu and fp8.cuh share it) + dispatch case in
   ggml_cuda_get_rows_switch_src0_type + ggml_cuda_device_supports_op.
2. CPU get_rows for F8: the type_traits table got .to_float =
   dequantize_row_f8_e4m3 (ggml.c; also fixes llama_model_get_tok_embd), the
   CPU supports_op blanket F8 rejection got a GET_ROWS carve-out, and
   ggml_compute_forward_get_rows got the F8 case (it dequantizes via
   traits->to_float).
3. Conversion: --fp8-token-embd (mutually exclusive with --fp8-output-weight);
   _generate_fp8_token_embd packs token_embd with PER-ROW scales
   (d[r][cb], 20 scales per vocab row instead of one shared per 128-row
   block - finer, and both the get_rows and the mul_mat kernels index d per
   row, so no format change).

PITFALL (do not rediscover, this cost most of the session): with F8 token_embd
and NO CPU get_rows support, the sched placed get_rows on CUDA and copied the
whole 625 MB embedding into the GPU compute buffer PER BATCH - decode dropped
to 17 t/s (5x), pp512 to 4504, the compute buffer ballooned to 625 MiB, and
the run was full of serialized host->device copies. The BF16 embedding never
hit this because get_rows runs on CPU reading the mmap in place. Root cause:
the input layer is always kept on the CPU buffer by design
(llama-model.cpp: "very little benefit to offloading the input layer"), and
with mmap the ROCm_Host candidate is converted to a plain CPU buffer. The
CPU-buffer F8 tensor can only be consumed by a CPU get_rows (or a per-batch
device copy). Lesson: any future F8 tensor that lives in a CPU/host buffer
needs a CPU-side consumer; the CUDA get_rows alone is not enough.

Also learned this session: the box has 3x R9700 + a small GPU3; a Qwen3.6-27B
llama-server (started 2026-08-05) holds ROCm0/1, so benches must use
-dev ROCm2. llama-cli in this fork is an HTTP client that spawns its own
llama-server (or --server-base); the earlier "generation parity" runs in this
file may have been against the spawned server on whatever GPU was free - use
the same -dev as the benches and the same model path for a meaningful diff.

Next lever per LEVERS.md: none live - L1 (wmma), L3/L4 (GDN) and L7 (in-proj
fp8) are spent; L8 (small kernels) skip. Remaining structural ideas: L1's
kpacked smem layout for A, and the decode-side fused quantize-in-gemv
(~+4% tg64 per section 10).

## 20. 27B fp8 -ow conversion + tensor-split fixes (2026-08-06, session)

Converted /llm/models/Qwen3.6/27B/FP8/ (Qwen3.5 arch, 64 layers, 27.3B params)
to the fp8 -ow GGUF: /llm/models/Qwen3.6/27B/FP8/Qwen3.6-27B-StewFP8-ow.gguf
(29.5 GB). PPL 4.6439 vs 4.6030 (BF16 gguf) = +0.9%, in the expected fp8 band.
Bench on 2 GPUs (-dev ROCm0,ROCm2 -sm tensor): pp512 1292, tg16 29.5, vs the
Q8_K_XL's 857/25.1 - the fp8 prefill win (~1.5x) holds at 27B scale.

Two code fixes were required:
1. --fp8-output-weight assumed tied embeddings (synthesized a copy of
   token_embd). The 27B is untied (tie_word_embeddings=false, real
   lm_head.weight) -> the flag crashed with "Duplicated tensor name
   output.weight". Fixed: untied models now quantize lm_head.weight itself
   (and pop it from the dequant path); the tied path is byte-identical
   (verified against stewfp8-ow-2copy.gguf).
2. Tensor split (-sm tensor) was broken for the whole qwen35 arch in this
   build: ggml-backend-meta.cpp handle_ssm_conv required src0/src1 on the SAME
   split axis, but the fused 2-src conv (L5 ssm_conv_2src) has qkv split on
   axis 0 and the conv weight/states on axis 1 (same channel partition).
   The assert also hit the Q8_K_XL model. Fixed: the handler now accepts the
   channel-split-everywhere case and returns src0's state. The Q8_K_XL (33 GB)
   now benches on 2 GPUs too, and the 27B fp8 runs with the fused GDN enabled.
   NOTE: llama-bench uses "-dev A/B" (slash), llama-perplexity/llama-cli use
   "-dev A,B" (comma) - the two tools disagree on the separator.

Also noted: the 27B fp8 (27.5 GiB weights + ~27 GiB fp8_repack cache) does NOT
fit one 32 GB R9700 (OOM in ggml_cuda_mul_mat_fp8 at the first repack), and
partial offload (ngl<all) is impossible for fp8 (CPU has no fp8 mul_mat, so
CPU-buffer fp8 weights crash GPU kernels - the L9 lesson). Two GPUs with
tensor split is the only viable config on this box.
