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

STATE: llama.cpp tree has the fused hc_mix change UNCOMMITTED (ask before
committing). Debug scaffolding removed from hc-mix.cu (env-gated dump code
deleted; /tmp/hcchk.cpp + /tmp/hcmix_1_*.bin + /home/ld_decode_*.bin kept for
reference). Prefill gate /home/ld1024_noflag.bin unaffected by construction.
NEXT: commit + (optionally) extend the fuser to hc_combine chains; the
handover's original item 1 (fuse the norm into the mix too, dropping the
RMS+MUL nodes when inject is absent) would save 2 more dispatches/mix.
