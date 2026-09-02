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
