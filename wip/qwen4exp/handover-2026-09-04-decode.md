# HANDOVER 2026-09-04: qwen4exp decode tuning (phase 2)

Status: prefill parked (all levers probed - see handover-2026-09-03.md, the
prefill record, superseded only for the DECODE portion by this doc). New
target: boost DECODE (tg). Context-safe to compact after reading this.

## 0. THE GOAL: decode above ~37-40 t/s (tg128)

Current qwen4exp decode: **tg128 = 37.8-40.1 t/s** (~25-26 ms/token), flat
across every prefill lever tried (top-k GPU radix, -ub, -t, WMMA, QSA
toggles all left tg unchanged). Bench: 3x R9700 tensor-split, -ngl 99,
`-p 512 -n 128 -ub 2048 -r 2`, env `GGML_CUDA_FA_WMMA_256=0
LLAMA_QSA_SPARSE_FA=1`, warm. The reference (docs-era ~39 t/s) is MATCHED -
no decode win has ever landed on this model.

The sibling model (qwen35moe 35B-A3B Q6_K, 1 GPU) decode = 92.2-92.8 t/s
(10.8 ms/token) and its phase-1 cost model is the best evidence we have for
WHERE the qwen4exp decode time goes (same architecture family: QSA +
GATED_DELTA_NET layers, MoE routed experts, big vocab). It is documented in
`wip/qwen35moe-prefill/decode-phase1-results.md` + `plan-decode.md`.

## 1. The qwen35 phase-1 decode cost model (10.8 ms/token = 923 kernels)

| Component | ms | % | Notes |
|---|---|---|---|
| Routed experts (2 fused kernels/layer) | 3.6 | 33% | gate+up 37us @427 GB/s; down+shared-gate 53us @166 GB/s (K short) |
| Shared experts (up/gate/down) | 1.2 | 11% | ~200-380 GB/s |
| Recurrent qkv+z+alpha+beta+out | 1.0 | 9% | 534 GB/s - good |
| lm_head [2048x248320] | 0.86 | 8% | 600 GB/s - near peak |
| RMS_NORM q8_1 folds | 0.76 | 7% | 41 kernels ~18us each |
| Attention (qkv + FA + rope) | 0.45 | 4% | FA ~1% of decode |
| GDN AR scan + state cpy | 0.5 | 5% | 30 sequential ~11us each |
| topk-moe | 0.29 | 3% | 40 x ~7us |
| Elementwise chains, GET_ROWS, CPY | ~1.3 | 12% | visible + fused |
| Launch/graph overhead | ~1 | 9% | 923 kernels/token |

KEY FINDINGS (trust these; they resolved the earlier FA/GET_ROWS red
herrings - the old no-graph op timing was a blind-spot artifact):
1. Decode is **LATENCY-bound, not bandwidth-bound**: 68% of the 923 kernels
   are under 20us; the sum of kernel critical paths dominates. Only lm_head
   and a few large GEMMs are bandwidth-bound.
2. Weight reads ~2 GB/token (MoE 8/256 routing) - small. KV depth does NOT
   grow decode time (flat 92 t/s from KV=512 to 16K).
3. The fusion machinery is worth 24% of decode (70.5 -> 92.8 t/s with
   GGML_CUDA_DISABLE_FUSION=1). Decode fusions exist for: gate+up mmvq,
   down+shared-gate+residual, conv-input into qkv, RMS_NORM+q8_1, GDN cache
   cpy, topk-moe, elementwise chains.
4. CUDA graphs worth only 2.5% on decode; PDL neutral.
5. mmid decode mmvq maps 8 experts onto 8 warps with K as grid.x; the short-K
   down (K=512) gets ~2 K-blocks/warp -> 166 GB/s only.

Phase-2 levers ranked by qwen35 evidence:
1. **One fused expert kernel per layer (gate+up+down)**: 37+53=90us/layer ->
   a single ~60-70us dispatch. Save ~1.2-1.3ms/token (~12% decode) + the
   prefill win. Same work as plan-fused-moe.md (which was measured/assessed
   on qwen35 - see fused-moe-mmq-result.md + item3-router-fusion-failed.md).
2. **Short-K mmvq config tuning** for the decode expert shapes (down K=512
   @166 GB/s): a config-only change in the RDNA4 mmvq param table.
   Potential ~0.5-1ms of the down kernel. MUST keep bit-identical per-row
   accumulation.
3. **Kill the small-kernel tail** (~2ms in <20us kernels): fold ssm-state
   GET_ROWS into the GDN AR epilogue, residual ADD into expert epilogues,
   elementwise chains. Each fusion removes ~10-15us of kernel latency.
4. SKIP: FA decode tuning, GDN AR fusion, PDL, graphs (measured <5%).

## 2. qwen4exp-specific decode facts (the model is NOT qwen35)

- Model: Qwen3.8-Flash-Next "UD-Q4_K_XL" (unsloth mixed quant), 103.68 GiB
  over 4 shards, 3x R9700 (32624 MiB each), -ngl 99. 48 layers = 12 QSA +
  36 GATED_DELTA_NET. 512 experts x 640 hidden, top-10 used. vocab 82774.
  hc = 4 streams. Weights: MoE gate/up Q4_K (+rare Q5_K), down Q5_1 (+rare
  Q8_0) - the Q5_1 down is documented as worst-case for RDNA4 dequant.
  attn/qkv/output Q8_0; hc LoRA Q8_0; hc_inject/router F32.
- Decode 37-40 t/s = 25-26 ms/token. If the qwen35 per-layer cost structure
  scales by active-expert work + layer count, expect the expert GEMMs + the
  elementwise/launch tails to dominate similarly, amplified by: 10 vs 8
  experts used, 640 vs 512 hidden, 48 vs 40 layers, the Q5_1 down dequant.
- result_output (lm_head over 82774 vocab) was measured ~0.6ms alone (vs
  qwen35's 0.86ms over 248K vocab at 600 GB/s near-peak).
- Decode runs per-token ubatches; CUDA graphs ARE active on decode in this
  tree (the prefill graph-disable only affects multi-token graphs).
- Per-token host overhead was flagged early ("25ms/token wall vs 0.9ms GPU
  in the last split") but that split predates the top-k fix and was never
  re-verified; decode flatness under every prefill change says the decode
  time is GPU-side, not host/CPU.

## 3. Open questions to resolve FIRST (cheap discriminators)

1. **1-GPU vs 3-GPU decode on qwen4exp.** The 37-40 t/s was measured on the
   3-GPU tensor-split bench. A single token's tiny per-op batches over a
   tensor split add inter-GPU syncs per kernel; decode may be FASTER on 1
   GPU (all 103 GiB won't fit, but -ngl partial might). Measure tg128 with
   HIP_VISIBLE_DEVICES=0,1,2 vs =0 (and =0,1) at -ngl 99 / -ngl 33. If
   1-GPU > 3-GPU per-GPU throughput, the split is the decode tax and a
   layer-split or replica config is the lever. This is THE first experiment.
2. **Fresh decode op-profile on qwen4exp** (what the 25ms/token is made of).
   CAUTION: the branch's custom GGML_CUDA_OP_TIMING epilogue HANGS after the
   result on the prefill graph (see 09-03 UPDATE 09-04c). The decode graph
   may or may not hit the same hang - try it; if it hangs, fix the
   instrumentation first (suspect: event pairs around the fused-dispatch /
   concurrent-stream boundary; the sync+elapsed-time loop never returns).
   Decode = 1-eval-per-token graphs, so a -n 8 run gives 8 evals.
3. **Per-token weight bytes on qwen4exp**: count the active-expert + dense
   bytes per token (10/512 experts x [gate Q4_K + up Q4_K + down Q5_1] x
   48 layers + all dense layers). This tells us the bandwidth floor vs the
   latency story. qwen35 read ~2GB/token; qwen4exp's 512x640x10/48-layer
   structure is different.
4. **R9V decode reference**: nothing recorded (only prefill 1512 pp8192).
   If the colleague has a vLLM decode number for this model family, it is
   the target to beat - ask.

## 4. Candidate decode levers (qwen4exp, transfer + model-specific)

A. **Fused per-layer expert kernel (gate+up+down) on the decode mmvq path**
   (qwen35 lever 1): the biggest single measured decode item in the family
   (33% routed experts). The decode mmid mmvq already fuses gate+up and
   down+shared-gate+residual; completing the gate+up+down single dispatch
   removes one kernel + its launch per layer. Prefill relevance: the same
   work was explored for MMQ prefill (parked); decode uses the MMVQ path
   (different code - mmvq.cu), so the prefill conclusions do NOT transfer.
B. **Short-K mmvq decode config** (qwen35 lever 2): tune the RDNA4 mmvq
   param table (rpb/nwarps) for the decode expert shapes (down K=640 Q5_1,
   gate/up K=1280 Q4_K at M=1). Config-only, bit-identical requirement.
C. **Small-kernel-tail fusion** (qwen35 lever 3): the QSA layers' decode
   indexer/score ops, the GDN state GET_ROWS/cpy chains, residual folds.
D. **The Q5_1 down dequant**: worst quant for RDNA4 (dense Q5_1 thin-K =
   12.5 TFLOPS vs Q4_K 31.6 measured on the dense bench). If the down
   experts were Q4_K or better, decode (and prefill MoE down) would speed
   up. Requires re-quantizing the down experts - needs the source F16
   (NOT available per 09-03) OR a mixed quant already present (5 layers
   are Q8_0 - the load-time tensor table could bias the rest at LOAD time
   via a converter, no source needed if the loader can re-quantize on load
   - check llama-model-loader capabilities).
E. **Batch/decode parallelism at the server level**: -ub 2048 on parallel
   slots (multiple sequences per decode step) - llama-bench tg is
   single-sequence; the real win for a serving workload may be batching.

## 5. Bench config (decode, MUST follow)

- Model: /models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
- tg/decode: `-p 512 -n 128 -ub 2048 -r 2` (warm; record temps 60C+).
- Env: `HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LLAMA_QSA_SPARSE_FA=1 -t 16 --split-mode tensor`.
- For 1-GPU A/B: HIP_VISIBLE_DEVICES=0 + `--split-mode none` (or layer).
- Current: tg128 = 37.8-40.1 t/s. Reference docs-era ~39. No decode win landed.
- Sibling reference: qwen35moe Q6_K 1-GPU tg128 = 92.2-92.8 (10.8 ms/token).

## 6. Tree / state facts (verified 2026-09-04)

- `~/llama.cpp` `qwen4exp` @ `9dff5cc4f`, CLEAN. Prefill probes all reverted.
- Boosts repo main @ `513a356` (handovers incl. 09-03 prefill record).
- Correctness gate: logitdump A/B via /tmp/build_logitdump.sh (baseline
  /home/ld1024_base.bin at 9dff5cc4f; memory: model load ~3 min; kill stale
  llama-bench first; NEVER write dumps to /tmp - tmpfs).
- Decode-specific fusions already in the tree (qwen35-era): gate+up mmvq,
  down+shared-gate+residual (`ggml_cuda_op_shexp_down_gate`), conv-into-qkv,
  RMS+q8_1, GDN cpy, topk-moe, elementwise chains, fused indexer top-k
  (radix, bpr=8), QSA latency pass. All verified on qwen35; qwen4exp decode
  uses the same machinery.
- Machine: 3x R9700 gfx1201, ROCm /opt/rocm-7.14-gfx1201, manual DPM
  (60-61C, -70mV/250W via tune_r9700.sh), 9950X3D2 CPU (full rebuild ~96s).
- rocprofv3 HANGS on this workload; the op-timing instrumentation hangs on
  the prefill graph (09-04c). Micro-bench harness = /tmp/mmq_bench (rebuilt
  against the current .so when the .so changes).

## 7. First experiments for the next session (in order)

1. tg128 1-GPU vs 3-GPU vs 2-GPU A/B (config-only; ~10 min) - resolves the
   split-tax question, may immediately change the direction.
2. If the split is not the tax: fix-or-try the decode op-profile (single
   -n 8 decode run with GGML_CUDA_OP_TIMING=1) for the qwen4exp per-token
   cost table.
3. Byte-count the per-token weight reads (routing + dense) to bound the
   bandwidth floor.
4. Then pick between levers A (fused gate+up+down mmvq), B (mmvq decode
   config), C (small-kernel tail), D (Q5_1->better down quant at load).

## UPDATE 2026-09-04 (decode session 1): first measurements

Bench: 3x R9700 tensor, ngl 99, ub 2048, t 16, -p 512 -n 128 -r 2, env
GGML_CUDA_FA_WMMA_256=0 LLAMA_QSA_SPARSE_FA=1 (graphs ON unless noted).

NEW NUMBERS (all consistent):
- tg128 = 39.29 +/- 2.34 t/s (25.45 ms/token), graphs ON.
- tg64 graphs OFF (GGML_CUDA_DISABLE_GRAPHS=1) = 38.11 -> graphs worth ~3%.
- tg1500 = 38.80-39.17 (n-insensitive).
- pp64 = 245-247 (for reference only).

DECISIVE GPU-UTILIZATION MEASUREMENT (rocm-smi during tg1500):
- All 3 GPUs at 97-98% busy THROUGHOUT decode. REFUTES the cross-device
  sync-tax / fragment-boundary-wait hypothesis (the per-layer x per-device
  fragment structure of the Meta-TP wrapper costs little; the fragments
  overlap and the GPUs stay saturated).
- Decode time = kernel execution time. Levers must reduce kernel WORK or
  kernel COUNT, not syncs.

CONFIG BLOCKS:
- 2-GPU ngl 99: GGML_ASSERT(meta_buf_ctx->bufs[i]) in the tensor-split
  allocator (103.7 GiB file > 63.7 GiB usable) - the 3-GPU fit is marginal
  (Meta()/CPU_Mapped buffers ~26.9 + 27.4 GiB, lazy mmap; the rest of the
  file is page-cache resident only).
- 1-GPU ngl 99: model load fails (generic "failed to load model"; no detail
  even with --verbose). So no clean 1-vs-3-GPU same-weights decode A/B on
  this model. The split-tax question is MOOT anyway: GPUs are 97% busy, so
  a 1-GPU config cannot beat the current per-GPU efficiency by removing
  waits (it would only remove ~3% graph value and lose 3x parallelism).

OP-PROFILE FINDINGS (GGML_CUDA_OP_TIMING=1 + --verbose; CRITICAL: llama-
bench silently swallows all logs unless --verbose - llama_log_set(null) -
this is why the earlier op-timing runs printed nothing):
- Decode = ~3400+ graph nodes/token evaluated as ~85-144 fragment evals
  (per-layer x per-device Meta-TP splits), ~2301 decode-only eval blocks.
- The main-stream op-event spans sum to ~41.5 ms/token over 3 GPUs (~55%
  of the real 3x25.45=76 ms kernel time) -> the instrumentation only sees
  the MAIN stream; the hc 4-stream + concurrent-event + AR-stage work
  (side streams) is invisible (their main-stream event pairs fire at
  dispatch). ~45% of decode kernel time is unmeasured by this tooling.
- Composition of the VISIBLE (main-stream) ~55%: expert GEMMs ~50%
  (MUL_MAT 31.7 + FUSED MUL_MAT 7.0 + MUL_MAT_ID 6.2 + FUSED MUL_MAT_ID
  5.4), elementwise ~43% (ADD 9.6 SCALE 7.8 MUL 6.2 UNARY 5.4 FUSED UNARY
  3.5 CONT 2.9 RMS_NORM 2.3 GET_ROWS 2.1 REPEAT 2.0 CPY 0.8), lm_head
  result_output [2560x82774x1] = 0.596 ms real single biggest kernel
  (2.3%), attention/QSA ~4% (SOFT_MAX 2.1 + QSA ~1 + ROPE 0.4).
- Per-kernel avg ~7-8 us (3400+ kernels in 25.45 ms): latency-bound kernel
  soup at ~3.7x the sibling model's 923 kernels/token. Same qualitative
  conclusion as qwen35 phase-1 (latency-bound, fusion is the lever),
  amplified by the delta-net/hc/QSA machinery's many small ops.

DIRECTION (evidence-ranked for qwen4exp decode):
1. FEWER KERNELS via fusion of the small-op chains (the node count is the
   problem: 3400 x ~7 us kernels). qwen35 phase-1 measured the fusion
   machinery worth 24% of decode there. Candidates: the hc per-layer
   chains (hc_inject/hc_combine/hc_gate + norms + scales), the GDN state
   GET_ROWS/cpy, residual folds into the expert epilogues, RMS_NORM+q8_1.
   The hc side-stream work (~45% invisible) is the least-understood chunk.
2. mmvq decode expert config tuning (short-K: down K=640 Q5_1 @ low GB/s,
   gate/up K=1280 Q4_K at M=1) - config-only, bit-identical requirement.
3. Fused per-layer expert kernel (gate+up+down) on the mmvq decode path.
4. Batch decode at the server (multiple slots) - converts the tiny-M=1
   kernels into M>1 kernels (the only config lever that needs no kernel
   surgery); llama-bench tg is single-sequence.
NEXT SESSION FIRST STEPS:
- Count the decode graph's per-class node totals STATICALLY (llama-graph
  dump or the op-timing "nodes over N" per class) to size the fusion prize
  precisely before touching code.
- Inspect the hc decode path (the ~45% invisible side-stream work): which
  ops run on the 4 hc streams per token and how many.
- Then pick lever 1 vs 2 vs 3 for the first patch.

## UPDATE 2026-09-04 (decode session 1, continued): the hc machinery is the decode tax

STATIC PER-TOKEN NODE INVENTORY (decode-only op-profile counts /27 evals,
counts exact - the timing distortion does not affect them):
- ~2800 graph nodes/token across 3 GPUs; avg ~9-26 us/node, GPUs 97% busy.
- Elementwise-class 1712/tok (61%): ADD 429, SCALE 356, MUL 264, UNARY 246,
  CONT 131, REPEAT 86, RMS_NORM 88, GET_ROWS 75, CPY 37.
- Dense MUL_MAT 486/tok (17%) - the hc LoRA/inject GEMMs are ~4-6 per layer
  (w_down [10240x320], w_up [320x10240], w_inject [10240x4] at M=1, each
  reading ~1-2 MB Q8_0) + shared-expert dense + qkv + conv...
- FUSED (already-merged chains) 553/tok (20%).
- Routed-expert MUL_MAT_ID only 96/tok (3%) incl. the fused gate.
- Per delta-net layer the graph has ~50-70 nodes.

SOURCE-LEVEL FINDING (src/models/qwen4exp.cpp): EVERY layer is wrapped in
2x build_hc_mix + 2x build_hc_combine (the 4-stream hybrid residual, hc=4):
- build_hc_mix (2x/layer, ~14 nodes each): RMS_NORM -> reshape -> MUL
  (hc_norm) -> LoRA-MM down [10240x320] -> SCALE 1/hc -> SILU -> LoRA-MM up
  [320x10240] -> SIGMOID -> MUL -> reshape -> cont+3xADD+SCALE (hc_mixed
  stream-collapse) + (inject) LoRA-MM [10240x4].
- build_hc_combine (2x/layer): SCALE+SIGMOID+SCALE -> reshape x2 -> REPEAT_4D
  -> MUL -> ADD residual (partly pre-fused: FUSED ADD hc_combine in profile).
- The mix/combine chains are broken by the LoRA GEMMs, so the existing
  elementwise fusion pass can only partially merge them (that is why 61% of
  nodes remain unfused elementwise).

WHY qwen4exp decode (25.45 ms/tok, ~2800 nodes) is 2.4x the sibling qwen35
(10.8 ms/tok, ~923 nodes): the hc 4-stream machinery adds ~40+ tiny
4-stream-wide elementwise nodes + 4-6 tiny M=1 LoRA/inject GEMMs per layer
(~48 layers x 2 wraps). qwen35 (no hc) was expert-GEMM-dominated (33%);
qwen4exp decode is hc-elementwise-dominated (61% of nodes).

CONVERGED TARGET LIST (next patches, evidence-ranked):
1. ONE fused hc_mix kernel per call (like the FUSED GATED_DELTA_NET op that
   already exists for the recurrent block): RMS + 2 tiny LoRA MV (M=1) +
   silu/sigmoid + gate-mul + 4-stream collapse in a single dispatch.
   Saves ~10-12 nodes x ~96 calls/token (~1000 nodes, ~36% of the graph).
   The M=1 LoRA GEMMs become full-BW weight streams inside the kernel.
2. Fuse the second hc_mix (ffn side) the same way (identical shape).
3. Check whether hc_mix(attn) + attention + hc_mix(ffn) could share the
   collapse (the reference arch's stream reuse) - architecture-level, risky.
4. mmvq short-K decode config for the routed experts (config-only, low risk)
   - smaller prize here (3% of nodes) than the qwen35 model (33%): the
   decode's expert GEMMs are NOT the qwen4exp bottleneck.
5. hc_combine residual-chain fusion (already partially fused).

NOTE for the next session: build_hc_mix is called with hc_attn_* and
hc_ffn_* tensors - a fused op would need a new GGML op + HIP kernel + the
fused-dispatch machinery, or a graph-level rewrite of build_hc_mix into a
single custom op (see how FUSED GATED_DELTA_NET was done in
ggml-cuda/gated_delta_net.cu for the pattern). The M=1 MV dot (10240-long
Q8_0 x F32 vector) at full BW ~1.8MB/2us is the easy part; the 4-stream
collapse math must match bit-for-bit (the logitdump gate).

## UPDATE 2026-09-04 (decode session 1, final): fusion A/B + hc_mix implementation plan

FUSION VALUE A/B (measured): GGML_CUDA_DISABLE_FUSION=1 tg128 = 34.19 +/-
1.63 vs 39.29 fused -> the existing fusion machinery is worth ~13% of decode.
The remaining unfused soup (~1712 elementwise nodes) is ~3x the already-
fused amount -> more fusion is the validated decode lever.

HC_STRUCTURE REFINEMENT (source): the 4-stream (hc=4) width affects ONLY the
residual res_hc [2560,4,nt] + the mix/combine wrappers. The attention/ffn/
GDN bodies run on the COLLAPSED 1x-wide mixed [2560,nt] (build_hc_mix
returns the stream-mean). Per layer: 2x build_hc_mix + 2x build_hc_combine
= 96 mixes + 96 combines per token. Per mix: RMS[2560x4] -> MUL(w_norm) ->
LoRA-down MUL_MAT [10240x320 Q8_0, M=1, ~3.4MB read] -> SCALE 1/hc -> SILU
-> LoRA-up MUL_MAT [320x10240] -> SIGMOID -> MUL -> collapse (cont + 3 ADD)
-> SCALE 1/hc + inject MUL_MAT [10240x4]. Weight reads: 192 LoRA + ~96
inject GEMMs/token = ~650MB -> ~0.34ms at 3-GPU aggregate BW (1.3% of
token; NOT the issue). The issue is the ~8-10 tiny kernels per mix x 96 =
~700-900 kernels/token with GEMM-broken chains the elementwise fusion pass
cannot merge (RMS...MUL chains are broken by the LoRA MUL_MATs; the
collapse cont+3xADD+SCALE sits after the SIGMOID/MUL).

IMPLEMENTATION PLAN (fused HC_MIX op, GDN-style - for the next session):
1. New GGML op GGML_OP_HC_MIX (ggml.c compute_forward: reference = the
   exact build_hc_mix op sequence, kept only for the CPU path), 5 inputs:
   x [n_embd,hc,nt] F32, w_norm [hc_dim] F32, w_down [hc_dim,hc_lr] Q8_0,
   w_up [hc_lr,hc_dim] Q8_0, w_inject [hc_dim,hc] Q8_0 (nullable), scale
   = 1/hc as op param. ONE output problem: the model needs BOTH mixed
   [n_embd,nt] (body input) AND inject [hc,nt] (combine input). Options:
   (a) HC_MIX writes mixed; keep inject as a separate MUL_MAT node (it is
   one tiny kernel; the big prize is the 8-node mix chain -> 1); (b) two
   ops. RECOMMEND (a): fuses RMS+MUL+down+SCALE+SILU+up+SIGMOID+MUL+
   collapse+SCALE (10 nodes -> 1), leaves the inject MUL_MAT as-is.
2. HIP kernel (new file ggml/src/ggml-cuda/hc-mix.cu, pattern after
   gated_delta_net.cu): grid over nt; per token: read x[2560x4], RMS each
   2560-row stream (bit-exact vs ggml_rms_norm: same eps, same reduce
   order - check ggml_cuda_rms_norm's reduction), x w_norm (converter
   folded gamma = 1+w), then the two Q8_0 matrix-vector products
   (dequant+dot, mirror the mmvq/mul_mat_vec_q8_0 numerics: per-block
   scale, fp32 accumulate in the same order), silu/sigmoid, gate-mul,
   collapse = (sum of the 4 streams)/hc into mixed[2560].
3. The M=1 Q8_0 MV at 3.4MB/lora: launch enough warps to stream the
   weight at ~full BW; the 320-col x 10240-row down product = 320 dots of
   10240 - grid (nwarps x 320), each warp a row-dot.
4. Model-side: replace the build_hc_mix body with the single ggml_hc_mix
   node (keep build_hc_combine unchanged). The bit-exact gate = logitdump
   A/B vs the baseline /home/ld1024_base.bin (current build, fusion on).
5. Second patch (later): extend the elementwise chain-fuser with a
   REPEAT_4D + MUL + ADD broadcast pattern to eat the 2x/layer combine
   residual chains (SCALE+SIGMOID+SCALE over [hc] + REPEAT+MUL+ADD over
   [n_embd,hc] -> 1 kernel), ~96 chains x 4-5 nodes.
6. Third (config-only, low-risk, parked): mmvq short-K decode config for
   the routed experts (small prize: 3% of nodes here, unlike qwen35).

Expected prize: the 10-node mix chains are ~25-35% of the decode graph;
fusing 96 of them saves ~800-900 kernels/token. The 13% fusion baseline
says per-kernel marginal cost ~5-9us -> potential mid-single-digit to
~10% decode. MUST verify with the fusion-on A/B (39.29) and logitdump
before believing any number.

## UPDATE 2026-09-04 (decode session 2): coherence verification + gate reset

OUTPUT COHERENCE VERIFIED (logitdump A/B, current clean-tree build):
- DETERMINISM: two identical GPU runs (57-token prompt, 3-GPU tensor, ngl
  99) -> BIT-IDENTICAL dumps. No races/nondeterminism in the decode kernels.
- CPU REFERENCE: same stream on CPU (-ngl 0) -> GPU logits agree within F32
  accumulation-order noise (mean abs diff 0.17 on ~15-magnitude logits, ~1%);
  every top-1 disagreement on the prompt run is a tight race (top-2 margins
  0.01-0.42, rival logit within the CPU-GPU diff); top-5 overlap 5/5. The
  model produces well-formed coherent logits.
- STALE-BASELINE FINDING: /home/ld1024_base.bin (09-01 23:03, 1024 x 248320)
  does NOT match ANY reproducible current config. Matrix: flag-on vs flag-off
  (pos0 bit-equal, error accumulates later: mean 0.069); QSA-env on vs off
  (pos0 differs 1.6: layer-0 QSA numerics); every combo differs from the base
  at pos 0 (0.8-1.5). The base was built from an unreproducible tree state
  (the 09-01 session's work-in-progress .so). The ~0.1 mean diffs are F32
  order drift between build generations, NOT a coherence break. RETIRED as a
  bit-exact gate.
- BUILD-FLAG FINDING: the CMAKE_HIP_FLAGS=-mllvm
  --amdgpu-unroll-threshold-local=600 flag (in the cache since 09-02 04:50)
  IS numerically perturbing (pos0 bit-equal but later positions drift ~0.07
  mean vs the noflag build - the recurrent state amplifies the accumulation
  order change) despite being speed-neutral. REMOVED from the cache + build
  (restored to the 09-01-era flags). Current libggml-hip.so = 09-04 noflag
  clean-tree build.
- NEW GATE: /home/ld1024_noflag.bin = the current noflag clean build's dump
  (-n 1024, seed 1234, GGML_CUDA_FA_WMMA_256=0 LLAMA_QSA_SPARSE_FA=1
  GGML_CUDA_ALLREDUCE=none, 3-GPU tensor, ngl 99). CANONICAL REFERENCE for
  the hc_mix work: any change must reproduce it bit-exactly.
- logitdump caveat: GGML_CUDA_ALLREDUCE=none is REQUIRED (the NCCL comm-init
  thread segfaults in hipFuncGetAttributes with the logitdump link line on
  3-GPU tensor; llama-bench is unaffected - do not chase this, it is a
  tool-link artifact).

## UPDATE 2026-09-04 (decode session 3): FUSED HC_MIX OP LANDED (bit-exact, +5% decode)

IMPLEMENTED (llama.cpp qwen4exp branch, all files on top of the coherence-session state):
- New GGML op GGML_OP_HC_MIX: fuses the qwen4exp hyper-connection mix tail
  (down LoRA mm, scale 1/hc, silu, up LoRA mm, sigmoid, gate mul, stream
  collapse) into ONE op dispatch. Inputs: xn [hc_dim, 1] F32 (rms+gamma kept
  as graph nodes - the inject MUL_MAT consumes the same xn), w_down
  [10240,320] Q8_0, w_up [320,10240] Q8_0; output mixed [2560, 1]. hc in
  op_params.
- Activation: DECODE ONLY (nt == 1 && cparams.fused_hc_mix). Prefill (nt>1)
  keeps the unfused chain, so all prefill-path gates stay valid BY
  CONSTRUCTION. CPU reference impl included (ops.cpp) for -ngl 0 decode;
  toggled by LLAMA_FUSED_HC_MIX=0 env (default on, read in llama-context).
- Meta-TP wrapper: GGML_OP_HC_MIX = MIRRORED (all hc tensors verified
  mirrored on the decode graph); each device runs the full op.
- Touch points: ggml.h (enum+builder), ggml.c (names/symbols + builder),
  ggml-cpu (ops.cpp ref + dispatch + wdata), ggml-cuda (hc-mix.cu/.cuh new +
  dispatch + supports_op), ggml-backend-meta.cpp (split case), llama-cparams.h
  + llama-context.cpp (flag + env), qwen4exp.cpp (build_hc_mix branch).
  GGML_OP_COUNT 103->104 (+ggml-rpc.h patch bump). Also FIXED a pre-existing
  off-by-one in GGML_OP_SYMBOL (INDEXER_TOPK had no symbol entry).

BIT-EXACTNESS (the interesting part - THREE bugs found by micro A/B):
1. vec_dot_q8_0_q8_1 takes the y (Q8_1) block POINTER as-is; kbx only offsets
   the weight. Passing the y base instead of &y[kbx] gave mean diff ~2.4.
2. The mmvq dispatch uses a ROWS-PER-BLOCK OVERRIDE on RDNA2+ for short-K
   GEMMs (K=320 -> rpb=16) that changes the per-row accumulation tree. The
   down (K=10240, rpb=1) matched a plain per-row replica; the up (K=320)
   needed the exact rpb=16 clone of mul_mat_vec_q (item loop + per-warp
   butterfly + serial warp adds).
3. The collapse must round each xn*gate product BEFORE adding (the reference
   runs separate MUL and ADD ops); sum += x*g inline lets the compiler
   contract to an FMA -> ~30% of elements off by 1 ulp. Store products to an
   array first.
Debug method: env-gated (HC_MIX_DUMP) dump of the fused op's inputs/stages +
a micro-harness (/tmp/hcchk.cpp) that runs the REAL unfused chain ops on the
GPU with the captured data and compares stage by stage. Proven: lo, gate_raw
and mixed are byte-identical vs the unfused decode chain; the FULL decode
logitdump A/B (16 nt=1 positions, all 97 fused calls/token) is BIT-IDENTICAL
vs the unfused reference (/home/ld_decode_ref.bin; run with -ub 1).

SPEED (same build, llama-bench tg128, 3-GPU tensor, ngl 99, ub 2048):
- fused (default):   41.11 +/- 2.39 t/s  (24.3 ms/token)
- unfused (=0):      39.15 +/- 2.39 t/s  (25.5 ms/token)
- GAIN +5.0%. First decode win on qwen4exp (historical ceiling ~40.1).

ANSWER TO "is bit-exactness costing speed?": no measurable cost. The
reference ordering constraints replicated are (a) the mmvq rpb=16 short-K
config = the tree's own tuned-fast path, and (b) rounding order in the
collapse, which is trivial work dominated by the 4 identical sigmoid expf's.
The +5% comes from removing ~97 fused nodes x ~9-11 dispatches of scheduler +
meta-wrapper + launch overhead; the fused op's 5 sub-kernels (q8_1 quantize,
down dots, silu+quantize, up dots, collapse) are dictated by the sequential
data dependency, not the ordering. Bit-exactness bought: the fused path can
be DEFAULT-ON with zero semantic risk and the logitdump decode A/B stays a
clean regression test (the same bar the tree's ssm_gate_beta fusion met).
Further speed (if wanted) = fewer sub-kernel launches via a grid-synced
mega-kernel, a separate ~1-3% opportunity.

STATE: llama.cpp qwen4exp branch COMMITTED at bbd976ceb (fused hc_mix op),
NOT pushed (user's fork policy: commit ok, no push). Working tree clean.
Boost repo main @ 94a3474: patch preserved as
wip/qwen4exp/patches/0009-hc-mix-fused-decode-op.patch (+ 0009-*.md), apply-
verified clean on parent 9dff5cc4f. Debug scaffolding removed from hc-mix.cu (env-gated dump code
deleted; /tmp/hcchk.cpp + /tmp/hcmix_1_*.bin + /home/ld_decode_*.bin kept for
reference). Prefill gate /home/ld1024_noflag.bin unaffected by construction.
NEXT: commit + (optionally) extend the fuser to hc_combine chains; the
handover's original item 1 (fuse the norm into the mix too, dropping the
RMS+MUL nodes when inject is absent) would save 2 more dispatches/mix.

## UPDATE 2026-09-04 (decode session 3, final): commit + patch block done

- llama.cpp qwen4exp @ bbd976ceb, clean. Fused hc_mix op is DEFAULT ON;
  toggle LLAMA_FUSED_HC_MIX=0 (env, read at ctx init). tg128 fused 41.11
  vs unfused 39.15 (+5.0%). Decode logitdump A/B bit-identical.
- Boost repo main @ 94a3474: wip/qwen4exp/patches/0009-*.patch + .md.
- DECODE-GATE PROCEDURE (rerun after ANY decode-affecting change):
  1. /tmp/logitdump_d -m <model> -o /home/ld_decode_check.bin -p <prompt>
     -ub 1 -ngl 99 --split-mode 3   (env: HIP_VISIBLE_DEVICES=0,1,2
     GGML_CUDA_FA_WMMA_256=0 LLAMA_QSA_SPARSE_FA=1 GGML_CUDA_ALLREDUCE=none;
     add LLAMA_FUSED_HC_MIX=1)
  2. compare byte-for-byte vs /home/ld_decode_ref.bin (unfused reference,
     still valid - the unfused path is unchanged).
  PREFILL gate (unchanged by construction): /home/ld1024_noflag.bin with
  the plain logitdump (n_batch 512).
- NEXT STEPS (ranked for the next session):
  1. Fuse the hc_combine residual chains (2x/layer; SCALE+SIGMOID+SCALE+
     REPEAT+MUL+ADD) - original plan item 5, another ~96 chains x ~4-5
     nodes. The mix op itself is done; combine is the remaining hc
     elementwise soup.
  2. mmvq short-K decode config for the routed experts (config-only; small
     prize here: experts are ~3% of decode nodes).
  3. Optionally drop the RMS+MUL nodes when inject is absent (head call
     only, il=-1 - negligible; the norm must stay for layer mixes because
     inject consumes xn).
  4. Older open threads (pre-decode): v_shifted probe, splitter-to-F32,
     shuffle-to-smem, ggml-backend-meta import gates, multi-device sync in
     meta_tp_test; ML-Kernel patch + gpudh review vs TP-V1; thread 1
     prefill (pp8192 ~2000 t/s target).
- TOOLS on /tmp (survive reboot only): logitdump_d (has -ub flag for nt=1
  decode dumps), hcchk.cpp micro harness (runs the real unfused chain on
  captured fused inputs; needs /tmp/hcmix_<call>_*.bin dumps which are no
  longer produced by the committed kernel - re-add the env dump if needed).
  Reference dumps on /home: ld_decode_ref.bin (decode), ld1024_noflag.bin
  (prefill), ld_decode_final.bin (fused, == ref).

## UPDATE 2026-09-04 (decode session 4): FUSED HC_COMBINE OP LANDED (bit-exact, +4.5% on top of hc_mix)

IMPLEMENTED (llama.cpp qwen4exp branch, on top of bbd976ceb):
- New GGML op GGML_OP_HC_COMBINE (count 104 -> 105, rpc patch 2 -> 3):
  fuses the build_hc_combine tail (SCALE inject 1/hc, SIGMOID, SCALE 2,
  REPEAT of block_out over the hc streams, MUL, ADD residual) into ONE
  dispatch at nt == 1. 96 calls/token (2 per layer x 48). Decode-only;
  prefill keeps the unfused chain. Toggle LLAMA_FUSED_HC_COMBINE=0
  (default on). Touch points mirror the hc_mix op exactly (ggml.h/.c,
  ggml-cpu ops+dispatch, hc-mix.cu kernel, ggml-cuda.cu dispatch +
  supports_op, ggml-backend-meta MIRRORED, cparams + context env,
  qwen4exp.cpp build_hc_combine branch).
- Kernel: one thread per row, w[c] (hc <= 8) computed once per block into
  smem. The 1/hc + 2.0 scales are EXACT in f32, so the only value rounding
  is the sigmoid (inline formula == unary op). Products stored to an array
  before the residual add -> no FMA contraction (same trap as the mix
  collapse). BIT-EXACT ON THE FIRST SHOT: decode logitdump A/B identical
  (16 nt=1 positions, all 96 combines/token) vs /home/ld_decode_ref.bin.
- SPEED (same build, tg128, 3-GPU tensor): both fused 43.05 +/- 2.57 vs
  combine-off (mix on) 41.19 +/- 2.40 -> COMBINE GAIN +4.5%. Progression:
  39.15 (unfused) -> 41.19 (hc_mix) -> 43.05 (mix+combine) = +10% total
  (23.2 ms/token). pp512 unchanged (1492-1500 both runs; decode-only
  construction). NOTE: pp512 now reads ~1500, NOT the 1248.54 recorded in
  session 1 (flag-on build) - discrepancy is pre-fusion/pre-flag-era, NOT
  this change (both runs of this session agree); re-verify pp on the
  noflag build if the prefill thread resumes.
- llama.cpp @ 6a4e2c766 (committed, not pushed). Working tree clean.
  Boost repo: patch 0010-hc-combine-fused-decode-op.patch + .md, apply-
  verified clean on parent bbd976ceb.
- Reference dumps: /home/ld_decode_combine.bin (fused mix+combine, == ref).

NEXT STEPS (ranked):
1. The remaining per-layer hc soup after mix+combine fusion (from the
   session-4 op-profile tail): per layer still ~ RMS_NORM + MUL (hc_norm,
   xn for the fused mix/inject) + the inject MUL_MAT [10240x4] + the
   ffn/attn body ops. Dropping the RMS+MUL nodes requires folding the norm
   into the mix op when inject is absent (head only) OR merging the norm
   into the inject MUL_MAT path - small prize (2 nodes x 97 calls).
   The bigger remaining items per the visible tail: the ffn/attn body
   elementwise nodes + the QSA/GDN machinery (see the old small-kernel-
   tail lever list).
2. mmvq short-K decode config for the routed experts (config-only).
3. Server-level batch decode (M>1 kernels) - the biggest untried lever
   for a serving workload.
4. Older threads (pre-decode items + ML-Kernel/gpudh review + thread-1
   prefill pp8192) unchanged.

## UPDATE 2026-09-04 (decode session 5): mix internals + rms fold - decode 44.81 (+14.5%)

Work on top of 6a4e2c766 (3 commits: b7635f6df, b5f7fd598, 50ae9d134;
patch blocks 0011-0013, apply-chain verified on 6a4e2c766):
1. Collapse kernel threading (1-thread blocks -> (256)-thread coalesced
   blocks) + silu_quant per-block grid: 43.05 -> 43.91.
2. silu+Q8_1 quantize merged into the up-dot kernel prologue (5 -> 4
   kernels per mix). The xn-quantize-into-down merge was tried and
   REVERTED: 320 blocks re-streaming xn (40KB) from L2 = 12.8MB/call,
   down stage 43 -> 80us. Internal per-call timing (env HC_MIX_TIMING,
   removed after use): quant ~16, down ~27, up ~35, coll ~10 = ~90us
   under decode contention - every kernel pays ~8-16us of queueing, so
   kernel COUNT is the lever.
3. RMSNorm+gamma folded INTO the fused op (new contract: x + w_norm
   inputs; dst [n_embd + hc_dim] with mixed at the head and xn persisted
   at the tail; model views both; the F32 inject MUL_MAT consumes the xn
   view UNTOUCHED). rms replication = 1024-thread block_reduce clone of
   rms_norm_f32 + separate gamma rounding - BIT-EXACT (gate passed first
   shot). 43.91 -> 44.81 @ -r 3.
- Definitive same-session A/B @ -r 3: FUSED 44.81 vs UNFUSED 39.79 =
  +14.5% (22.3 vs 25.1 ms/token). pp512 unchanged ~1500.
- Correctness: decode logitdump byte-identical vs /home/ld_decode_ref.bin
  after EVERY change (ld_decode_mixopt/upmerge/final2/xntail dumps all ==).
- Cycle-time lesson: run benches/gates in the FOREGROUND with a generous
  timeout (returns on early completion AND early failure); backgrounding +
  sleep-polling wastes minutes per cycle. -r 3 tightens the bench mean.
- BENCH NOISE: tg128 std +/-2.2-2.7 t/s = +/-5%: single -r 1/-r 2 runs
  cannot resolve sub-2% changes; use -r 3 and/or A/B pairs.

REMAINING LEVERS (evidence-ranked, all with real cycle costs):
1. Fold the F32 inject MUL_MAT into the mix op: the inject weights are F32
   in the GGUF, so the reference = the mmvf kernel (float2-pair FMA
   accumulation, block reduce) - replicable (~+4%) but a fresh exactness
   chase. The Q8_0 assumption in the original design was wrong (hc_inject
   is F32, NOT Q8_0).
2. Decode expert mmvq config (down [K=640 Q5_1-ish], gate/up [K=2560] at
   M=1): ~24% of decode in the expert cluster; the RDNA4 tables were swept
   on qwen35 shapes, not qwen4exp's. Config = compile-time constexpr ->
   rebuild per variant + re-gate (numeric perturbation) + bench.
3. Cooperative single-kernel mix: IMPOSSIBLE bit-exactly - the up phase
   needs 640 blocks (rpb16) > cooperative residency.
4. GDN state GET_ROWS/cpy + body elementwise soup: invasive.
5. Server-level batch decode (M>1 kernels): untried config lever.

## UPDATE 2026-09-04 (decode session 5b): F32 inject fold + THE KERNEL-COUNT WALL

- The inject weights are F32 in the GGUF (NOT Q8_0 - the original design
  assumption was wrong). The op now computes inject internally with an mmvf
  replica (float2 pairs, acc += v*u, 256-thread blocks, zero-padded warp
  butterfly) - BIT-EXACT first try; the dst tail carries inject and the
  combine consumes the view directly (no MUL_MAT node). Commit bd25e63eb,
  patch 0014 (chain 0013+0014 verified on 50ae9d134). Boost @ 9b4d0f2.
- PERF: tg128 44.70 vs 44.81 - FLAT. The inject was already ONE kernel;
  folding it = 5+1 graph kernels -> 6 internal = zero count change.
  LESSON (the decode kernel-count wall): every kernel pays ~8-16us of
  queueing on the saturated 3-GPU decode, so ONLY kernel-count-REDUCING
  changes win (rms+mul fold -2, silu-quant merge -1: the +14.5%) and
  kernel-count-preserving moves are noise (+/-2.2 t/s std). Remaining
  reductions require touching the BODY ops (attn/ffn chains, experts) or
  batching M > 1.
- CURRENT STATE: llama.cpp @ bd25e63eb (clean); decode tg128 ~44.7-44.8
  (+14.5% vs the 39.79 unfused @ -r 3); all changes bit-exact vs
  /home/ld_decode_ref.bin. Patch blocks 0009-0014 preserved.
- NEXT (kernel-count-reducing only): (1) batch decode at the server
  (M > 1 kernels - the config lever with no kernel surgery); (2) the
  decode expert mmvq config (down [K=640], gate/up [K=2560] at M=1 - the
  expert cluster is ~24% of decode time but only ~3% of nodes; config is
  compile-time constexpr -> rebuild + re-gate per variant); (3) the
  attn/ffn body elementwise chains (generic fusion pass work).
- Cycle-time: run benches/gates FOREGROUND with timeout (returns on early
  exit AND early failure); -r 3 for decisive bench means.

## FINAL STATE (context-compaction checkpoint, end of decode session 5)

- llama.cpp qwen4exp branch @ bd25e63eb, WORKING TREE CLEAN, NOT pushed
  (user's fork policy: commit ok, no push).
- Boosts repo main @ 295b6c7, clean, PUSHED. Handover file = this doc.
- Decode: tg128 ~44.7-44.8 t/s = +14.5% vs unfused 39.79 (22.3 ms/token).
  pp512 ~1500 unchanged (prefill untouched by construction - all decode
  changes are gated to nt == 1 via LLAMA_FUSED_HC_MIX / _HC_COMBINE envs,
  default on).
- Correctness: every change verified byte-identical vs
  /home/ld_decode_ref.bin (16 nt=1 positions). The prefill gate
  /home/ld1024_noflag.bin is unaffected by construction.
- Patch blocks 0009-0014 in wip/qwen4exp/patches/ (0009 + 0010 = the two
  ops; 0011-0014 = kernel geometry / silu-quant merge / rms-gamma fold /
  F32 inject fold). Apply chain 0009..0014 verified sequentially on
  6a4e2c766.
- The kernel-count wall (per-kernel ~8-16us queueing on the saturated
  3-GPU decode) means only kernel-count-reducing changes win; next levers
  are batch decode (M > 1) or the decode-expert mmvq config or body-op
  fusion - all listed in the session-5b section above.

## UPDATE (decode session 6): 4-kernel mix + M=1 GEMM floor model + router scoping

- COMMIT af3640f67 (on bd25e63eb), patch 0015 preserved + apply-verified.
  hc_mix internals 6 -> 4 kernels/call: (1) the xn Q8_1 quantize became the
  phase-3 of the rms+gamma kernel (one 1024-thread block per stream
  quantizes its own 80 q8_1 groups with the quantize_q8_1 warp pattern -
  amax/127 roundf, sum - no redundant work, byte-identical y_xn); (2) the
  F32 inject mmvf-replica rows merged into the collapse dispatch (grid =
  n_embd/256 collapse blocks + hc inject blocks, blockIdx branch, both
  paths unchanged). Decode logitdump byte-identical; tg128 45.57 then
  45.56 (two -r 3 runs) vs 44.70-44.81 pre-merge = +2% (reproduced, above
  noise). Campaign total vs the 39.79 -r 3 unfused ref = +14.5% (39.15 ->
  45.56 = +16.4%).
- COST MODEL (micro-probes, settles the queueing-vs-work question):
  chained-graph probe (64 sequential M=1 GEMMs in ONE graph, decode-like):
  17.8 us/GEMM back-to-back vs 26.5 us with per-call graph dispatch.
  Isolated per-op loop times are HOST-DISPATCH-INFLATED (~+9-15us/call);
  the decode replays the whole token graph, so per-kernel = true GPU time.
  ALL M=1 mmvq GEMMs pay a ~15-20us kernel floor in-chain regardless of
  weight size (0.74MB router ~= 16.6MB dense); decode = ~440 GEMM-class
  evals + ~1400 small evals per device per token; wall = sum of small
  kernel times. The hc mix dots (down [10240x320] + up [320x10240], Q8_0)
  are at this floor (~17/19us) and are BIT-EXACT-LOCKED (rpb=1/rpb=16
  replication = the reference's own dispatch); the expert mmid kernels run
  ~250-460 GB/s (near-tuned); the mix internals are now minimal (4
  kernels: rms_quant, down, up_silu, collapse_inject - the 2 GEMMs are
  irreducible, the auxiliaries merged).
- ROUTER FUSION SCOPING (the next big lever, NOT yet done): the ffn
  routing = per layer logits mmvq [2560x512] M=1 (~12-15us, 25 GB/s =
  floor-dominated) + FUSED SOFT_MAX (~4us) + CPU argsort (0 GPU ARGSORT
  instances in the decode profile - the topk selection runs host-side,
  feeding the MUL_MAT_ID ids) + weights via the mmid fusion args. ~48x3
  mirrored chains/token ~25-30us each. The tree HAS a topk_moe fusion
  (topk-moe.cu + ggml_cuda_topk_moe_fusion in ggml-cuda.cu) that fuses
  logits-mm + softmax + argsort + get_rows into ONE kernel, but it matches
  SOFTMAX->RESHAPE->ARGSORT->VIEW->GET_ROWS (deepseek/llama4 order) and
  build_moe_ffn's groupless softmax path builds softmax->argsort->reshape
  (reshape AFTER, on probs) -> the qwen4exp graph does NOT match. Making
  it match = a shared-builder reorder (affects all archs using
  build_moe_ffn) or a qwen4exp-local path; the fused kernel = new numerics
  -> decode-only gate + NEW decode reference + coherence re-verify. Value
  ~+4-6% (kills ~2-3 GPU evals + the per-layer CPU round-trip x 48x3).
- HEAD MIX (il=-1, no inject, once/token) still runs the UNFUSED chain
  (~9 evals, ~90us, 0.4%) - the fused branch requires w_inject != nullptr.
  Fusing it = nullable-w_inject op contract (dst [n_embd] when null) -
  clean + small, but only ~0.3% value.
- OTHER REMAINING: GDN state GET_ROWS/CPY folding (~5% est, invasive);
  decode expert mmvq config (small - experts near-tuned); batch decode
  (server, doesn't move tg128).
- TOOLS: /tmp/m1_probe (M=1 mmvq shape floor probe, 200 iters),
  /tmp/chain_probe (chained vs per-call dispatch discriminator) - both
  compile with --offload-arch=gfx1201 against build-rocm libs.

## UPDATE (decode session 6b): head mix fused + scoping corrections + honesty check

- COMMIT 1f05646fd (on af3640f67), patch 0016 preserved + apply-verified.
  The head hc_mix call (il=-1, no inject) is now fused too: the op accepts
  w_inject == NULL (dst [n_embd], no inject blocks, meta splitter allows
  the null src's UNKNOWN axis). ALL 97 mix calls/token (96 layer + head)
  use the fused op; decode logitdump byte-identical; tg128 ~45.6 (flat -
  the head runs once/token; the change completes coverage).
- SCOPING CORRECTION: the ffn-moe ROUTER IS ALREADY FUSED by the tree's
  topk_moe pass (SOFTMAX->RESHAPE->ARGSORT->VIEW->GET_ROWS -> one
  dispatch labeled "FUSED SOFT_MAX"; the logits MUL_MAT stays separate
  because ggml_cuda_op_topk_moe takes the logits VALUES, not x+W). 0
  standalone ARGSORT instances in the decode profile = the selection runs
  inside the fused kernel, NOT on the CPU (the session-6 handover note's
  CPU-argsort theory was wrong). The session-6 "router fusion" lever is
  DEAD: the routing = logits mm (M=1 floor) + fused topk_moe = minimal.
- HONESTY CHECK on the 4-kernel merge (+2%): the 45.56/45.57 post-merge
  runs vs the 44.70/44.81 pre-merge runs are DIFFERENT SESSIONS; with the
  +/-2.2 t/s run noise the delta (0.81 t/s) is ~0.45 sigma combined - NOT
  decisive. The mechanism (-2 kernel launches x 95 calls/device/token) is
  sound and the change is bit-exact + simpler, so it stays, but treat the
  +2% as provisional. A decisive same-session A/B needs a rebuild of the
  parent commit (~20 min) if the number matters.
- REMAINING (corrected ranking): (1) the M=1 mmvq kernel floor ~15-20us
  per GEMM x ~440 GEMM evals/device/token ~ 55% of the wall - the deep
  lever (occupancy analysis: the hc down runs 320 blocks only on 96 CUs =
  ~1/3 occupancy; the up 640 = ~2/3; the rpb=2 dense shapes leave ~2/3 of
  threads at ~0.6 kblocks/thread - the M=1 kernels are geometry-bound, not
  BW-bound, but the geometry is the maintainers' tuned mmvq tables and any
  change = new reference); (2) GDN state-gather fold (GET_ROWS ~87/device
  ~10us each + CPY ~36/device - foldable half ~1.5-3%, invasive);
  (3) batch decode at the server (M>1 - converts the M=1 floor into
  amortized kernels; config-only, does not move tg128).
- STATE: llama.cpp qwen4exp @ 1f05646fd clean (commits bd25e63eb ->
  af3640f67 -> 1f05646fd this session, none pushed); decode tg128 ~45.6
  (+16.5% vs the 39.15 original, +14.6% vs the 39.79 -r 3 unfused ref);
  all bit-exact vs /home/ld_decode_ref.bin. Boost main @ 24ced14 (patches
  0015 + 0016 + this note). Tools: /tmp/m1_probe, /tmp/chain_probe.

## UPDATE (decode session 7): Item 1 (M=1 mmvq floor) - MEASURED TO A DEAD END

The M=1 GEMM floor investigation, concluded with measurements:
1. FLOOR DECOMPOSITION (in-chain probes, /tmp/floor_probe): the per-GEMM
   cost is NOT a universal launch floor - ADD [2560] = 1.15us, GET_ROWS =
   0.37us, RMS_NORM [2560] = 6.6us (its reduce structure), M=1 mmvq GEMMs
   = 8-19us for 0.18-3.7MB weights (the fixed ~7-8us + block-execution).
   The big GEMMs ([2560,6144] 17.7MB = 36.7us = 455 GB/s) are BW-bound and
   fine. The SMALL GEMMs (0.2-4MB: hc dots, K/V, router, shexp) are
   floor-dominated at ~8-19us.
2. rpb SWEEP (env override GGML_CUDA_MMVQ_RPB over {1,2,4,8,16}, rebuilt
   once): the DEFAULT per-shape formula (RDNA2+ override in mmvq.cu) wins
   or ties nearly every shape; only ~6-11% single-shape wins for Q4_K
   router rpb=4 and Q5_1 shexp rpb=2 (~0.3-1% decode total). The geometry
   IS the qwen35-era tuned optimum.
3. vdr TEST: VDR_Q8_0_Q8_1_MMVQ 4 -> 8 (rebuild): NO change (18.2 vs 18.8
   hc-down; the register/ILP structure is saturated). The RDNA4 comment
   already records 4 as the measured choice.
4. CONCLUSION: the M=1 mmvq floor ~8-19us per small GEMM is the kernel's
   structural latency at M=1 (block lifetime x waves: the K=10240 hc-down
   is rpb=1-forced - its 320 kblocks/row fill the 128 groups 2.5x and the
   row CANNOT split across blocks bit-exactly; rpb 2/4/8/16 all measured
   WORSE). Only M>1 batching (amortizes the floor across tokens - server
   lever) or a from-scratch persistent/pipelined M=1 kernel (major
   project) can beat it. ALL experiment hacks reverted; tree clean at
   1f05646fd (the rebuild = the gated source).

## UPDATE (decode session 8): BATCH DECODE - measured, unblocked, then found broken

KERNEL-LEVEL (batch_probe): M-token mmvq/mmq per-token cost collapses at
M >= 16 (the mmq path): hc-down [10240x320] 23.9us/token at M=1 ->
2.69 at M=16 -> 0.89 at M=64; dense 28.3 -> 2.49 -> 0.82. The M=1 GEMM
floor amortizes 10-25x. (The mmvq ncols 2-8 path at M=2-8 barely helps.)

MODEL-LEVEL BLOCKER #1 (FIXED, commit 131935dce): the QSA indexer_top_k
builder asserted additive->ne[2] == score->ne[2], which compares the kq
mask's size-1 dim [n_kv, n_tps, 1, n_stream] against score->ne[2] =
n_stream - it only passes at n_stream == 1, so ANY n_seq_max > 1 context
crashed at graph build on the first QSA layer (llama-parallel and
llama-server --parallel hit the same assert). The kernel already reads the
additive as 3D (the size-1 dim is a no-op stride), so the fix = a builder
assert relaxation (accept ne[2] == 1 && ne[3] == n_stream). Single-seq
decode logitdump byte-identical. MULTI-SEQ DECODE NOW BUILDS + RUNS.

MODEL-LEVEL MEASUREMENT (bseq harness, /tmp/bseq): K-seq lockstep decode
(one token per seq per step), 3 GPUs, same prompt: K=1 21.9ms/step,
K=2 32.3, K=4 42.4, K=8 67.0, K=16 122.7, K=32 158.0, K=64 35.2 ms/step.
K=64 = 1819 tok/s AGGREGATE (0.55ms/token-equiv, ~40x the single-seq
45.6). The K=16-32 plateau = the mmvq ncols path; K=64 = where the mmq
tiles fill. K=128 OOMs (the per-seq recurrent state; K <= 64 on 3x32GB).

MODEL-LEVEL BLOCKER #2 (FOUND, NOT FIXED): multi-seq decode CORRUPTS
seq >= 1. bseq_val coherence (K=2, per-seq prompts vs the K=1 runs):
seq 0 = IDENTICAL to its K=1 run, seq 1 = degenerate garbage (repeating
tokens - the classic broken-recurrent-state signature). Localized: the
K=2 PREFILL final logits = correct for BOTH seqs (561 561, matching K=1);
the corruption starts at the FIRST DECODE STEP of seq 1 -> the decode
step's per-seq state/cache READ is wrong for seq >= 1 (a pre-existing bug
in the hybrid-memory decode path, unreachable until the assert fix).
Suspected: the GDN recurrent-state gather (build_rs) or the QSA cell read
for seq > 0 at the decode position. The batch-decode throughput finding
stands but the path is NOT usable until this is debugged.
