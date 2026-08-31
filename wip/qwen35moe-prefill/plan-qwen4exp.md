# PLAN: qwen4exp (Qwen3.8-Flash-Next) performance investigation

Status: ACTIVE (2026-08-31) - PREFILL TARGET >1000 t/s ACHIEVED, see qwen4exp-handoff-2026-08-31.md

## 0.7 CPU-burn root cause + prefill fix (2026-08-31, PM session)

- The host-CPU burn (16 OpenMP cores at ~95%, GPU ~60%) is NOT the PLE
  get_prev_tokens scan (that is now O(log n) via PR #27992 index, applied
  + unit-tested, 0 mismatches) and NOT mmap page faults (-lm none equal).
- Root cause: per-ubatch host overhead. Every ubatch re-runs the graph
  scheduler + gallocr reserve + CPU-side GET_ROWS split (token_embd
  644 MB + per_layer_token_embd 27.4 GB ngram table in CPU RAM,
  CPU_Mapped lazy). 16 OpenMP threads spin in kmp barriers waiting for
  the slowest gather row. Cost is fixed per ubatch, so it scales with
  the NUMBER of ubatches: at 16K/ub512 that is 32 x, at ub2048 only 8 x.
- Evidence: -t 1 vs -t 16 = 255 vs 769 t/s at pp16384 (CPU-thread
  sensitive) but pp512 single-ubatch is thread-insensitive (1399/1412).
  ub512 -> ub1024 -> ub2048 at pp16384: 799 -> 969 -> 1061 t/s (+33%).
- FIX (config, no code): -ub 2048 (or 4096; plateau). New best full
  curve at -ub 2048 -t 16 (tensor, r=3): pp512 1469, pp2048 1569,
  pp4096 1447, pp8192 1288, pp16384 1083, tg128 40.0. PREFILL TARGET
  >1000 t/s at 16K ACHIEVED (1083). tg still ~40 (GPU/host per-token
  bound; op timing shows result_output [2560x82774x1] = 0.6 ms alone).
- Note: llama-bench -ub affects prefill batching only; decode is
  single-token ubatches regardless, so tg unaffected by -ub.

## 0.6 Environment root cause (2026-08-31, mid-session)

- The 604->364 swing on the same binary was GPU runtime power-management
  cycling (527 SMU resume cycles/day; all 3 R9700s suspend/resuming every
  1-5 min). FIXED: echo on > .../power/control for 0000:{03,06,09}:00.0.
- /tmp is tmpfs and held 74 GB of stale logitdump .bin files; freed to
  0.3 GB -> free RAM 8 -> 81 GB. Both fixes together: pp16384 460->655
  t/s (beats the 570-620 reference), pp512 975, tg128 39.5.
- NEW bottleneck: host CPU (90-95% vs ~60% GPU). Per-ubatch PLE
  predecessor lookup llama_kv_cache::get_prev_tokens() scans all KV cells
  - O(n_kv) per ubatch. PR #27992 (kv-cache (seq,pos) cell index, O(log n))
  is the fix; open upstream. Related #27941, #27977.
- The 570-620 reference and the >600 claim are now reproduced/beaten with
  the clean env. The "regression" was environmental + block-11 + missing
  ngram patch, not a code regression in the MoE fusion work.

## 0.5 Historical context (user-provided, KEY for the regression hunt)

- GOOD: >600 t/s prefill, ~39 t/s gen observed on **tip of master +
  managed-ngrams patch** (the managed-ngrams work measured 570-620 tok/s
  at 16K on that same combination; mmap-lazy and managed paths were
  bit-identical and equal-speed THERE).
- BAD: ~350 t/s prefill, ~21 t/s gen observed on **rdna-boosts branch
  WITHOUT the managed-ngrams patch**.
- Two variables differ between GOOD and BAD: (1) the base (master vs
  rdna-boosts with blocks 08-12 + MoE fusion work) and (2) the ngram
  patch. This refines H1: the regression is in the rdna-boosts base and/
  or its interaction with the ngram path, NOT a config artifact.
- On master the ngram path (mmap-lazy vs managed) was performance-
  invisible; on rdna-boosts this must be RE-VERIFIED before assuming the
  ngram patch is neutral. Block 11 (skip graphs for multi-token prefill)
  and block 08 (fused-core prefill) are the prime suspects for a base-
  level regression on THIS model; the fused gate+up MMQ (Q4_K cap 96,
  512 experts, n_ff_exp maybe != 512) is untested on qwen4exp.
- Target: rdna-boosts + all MoE work + managed-ngrams -> >1000 t/s
  prefill, ~50 t/s gen at ~16K context.

## 0. Model facts (verified, from provenance + runtime)

- Model: `/models/Qwen3.8/Flash-Next/Q4_K_XL/` 4 shards (~105 GiB total:
  11M header + 47G + 46G + 12G), UD-Q4_K_XL, needs ALL 3 GPUs
  (~78-80 GiB VRAM recorded earlier).
- Arch qwen4exp: 48 layers, n_embd 2560, n_head 24 (head_dim 256),
  n_head_kv 2, top_k 2048, expert_count 512 (10 used), shared ffn 640.
  Full attention only at layers 3,7,11,...,47 (every 4th layer);
  the other 36 layers use recurrent/GDN-style attention (ssm_conv1d,
  gated_delta_net ops seen in the qwen35moe timing dump).
- Quant mix (per-tensor): ffn_gate_exps/ffn_up_exps = Q4_K (layer 2 ->
  Q5_K), ffn_down_exps = Q5_1 (layers 2,4,30,46,47 -> Q8_0),
  indexer.k/q_proj BF16, norms/gate_inp/ssm*/hc_* F32, rest Q8_0.
- PLE n-gram table (per_layer_token_embd.weight): **IQ4_NL, 28.8 GB**,
  ngram_size 3, heads_per_ngram 8 -> ple_n_heads 16 rows/token,
  ple_head_dim 160 (IQ4_NL row = 90 B), table 320,001,536 x 160,
  PLE on layer 1 (recurrent), layer_multipliers
  [23703573157769, 20109073645365, 8052911324071].
  Host-side per-ubatch n-gram hash; gather = ggml_get_rows (CPU dequant).
- Fusion applicability: qwen4exp uses the same `build_moe_ffn` hook as
  qwen35moe (verified in qwen4exp.cpp build_layer_ffn). gate/up are Q4_K
  -> the fused gate+up MMQ kernel (patch 0004) SHOULD fire on this model
  (Q4_K cap 96, RDNA4-gated). This is a primary validation target.

## 1. Goal

1. Recover/beat the first-test numbers (>600 t/s prefill, ~39 t/s gen)
   that regressed to ~350 t/s prefill / ~21 t/s gen.
2. Determine why >600 t/s is achievable-but-not-exceeded for a 6B-active
   MoE (user hopes >1000 t/s prefill; gen ~50 t/s).
3. Use the fused gate+up kernel (validated bit-exact on qwen35moe) on
   qwen4exp; validate bit-exactness there too.
4. Leave a clean, documented state (patches + results) for the Strix Halo
   follow-up (RDNA3.5, gfx1151 - fusion currently RDNA4-gated).

## 2. Working assumptions / hypotheses (to test, not to trust)

- H1 (REVISED per user history): the 600->350 regression is in the
  rdna-boosts base (blocks 08-12) and/or the missing ngram patch, NOT a
  config artifact. GOOD = master+ngram-patch, BAD = rdna-boosts w/o
  patch. Prime suspects: block 11 (skip graphs on multi-token prefill),
  block 08 (fused-core prefill), fused gate+up MMQ on Q4_K/512 experts.
  Decompose: (a) A/B fusion env toggle on rdna-boosts, (b) compare ngram
  path behavior with/without the patch on rdna-boosts.
- H2: the ngram gather (host-side per-ubatch hash + CPU ggml_get_rows on
  IQ4_NL) is a serialization point in the graph. The managed-ngrams
  numbers (570-620 tok/s FLAT across budgets) suggest it was NOT the
  bottleneck at 16K tokens... but those were measured on a different base
  (a7cc83bba) and possibly different graph. Re-measure with op timing.
- H3: decode (~21-39 t/s) is dominated by the recurrent layer 1 PLE
  path + the 36 non-attention layers' recurrent ops + expert GEMMs.
  qwen35moe decode work (mmvq item-split) may or may not transfer;
  qwen4exp gate/up are Q4_K/Q5_K so mmvq's k-quant VDR boosts apply.
- H4: blocks 08-12 (fused-core prefill, k-quant mmvq VDR, skip graphs on
  prefill, hybrid all-reduce) plus our fusion patches are a DIFFERENT
  base than the first test; any of them could regress qwen4exp
  specifically. Bisect by env toggle where possible.
- H5: multi-GPU tensor-split of the expert GEMMs (512 experts / 10 used)
  is inefficient; the per-layer all-reduce cost scales with expert
  count. Compare split modes (layer vs tensor) and GPU counts (2 vs 3).

## 3. Environment / baseline commands

- Build: `~/llama.cpp/build-rocm` (GGML_HIP=ON, gfx1201 native; NEVER set
  HSA_OVERRIDE_GFX_VERSION).
- GPUs: `HIP_VISIBLE_DEVICES=0,1,2` for 3-GPU runs.
- Model: `M=/models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
- Prefill bench: `llama-bench -m $M -p 512,2048 -n 0 -ub 512 -r 2 [--split-mode tensor]`
  (-n 0 = pure prefill; -ub sets the physical prefill batch).
- Decode bench: `llama-bench -m $M -p 512 -n 128 -ub 2048 -r 2 [--split-mode tensor]`
- Op timing: `GGML_CUDA_OP_TIMING=1 llama-bench -p N -n 0 -ub N -r 1 -v`
  (disables CUDA graphs; 2-GPU+ totals inflated by instrumentation, use
  llama-bench numbers as authoritative).
- A/B fusion: `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`.
- Bit-exact: `/tmp/logitdump2 -m $M -p "$(cat /tmp/prompt.txt)" -ngl 99 -c 4096 -o out.bin --split-mode 3`
  (--split-mode is INT: 0=none 1=layer 2=row 3=tensor; row fails on HIP).
- Managed ngram stats: `LLAMA_LAZY_READER_STATS=1`.
- IMPORTANT: `llama-cli` needs `--single-turn` or it hangs.

## 4. Execution order

### Phase A - Repro the regression (no code changes)
- [ ] A1. Current tree, 3 GPUs, tensor split: pp512/pp2048/pp16384 +
      tg128. Record.
- [ ] A2. Same with GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1 (isolate fusion
      effect on qwen4exp).
- [ ] A3. Op-timing dump for pp2048 (fused and unfused). Classify:
      fused gate+up, down, shared-exp, dense gemms, attn (full + GDN),
      conv, ngram gather (PLE), other. Identify top cost.
- [ ] A4. Compare against recorded managed-ngrams baseline (570-620 tok/s
      at 16K) and the user's first-test (>600). Note: different base
      commit, so exact delta is directional, not a bisect.
- [ ] A5. Test with --lazy-buffer-size / LLAMA_LAZY_READER_STATS if the
      ngram shows up in op timing; measure ngram cache behavior.

### Phase B - Apply managed-ngrams patches (combine the work)
- [ ] B1. Apply 0001-0007 in order (verified clean on 4fa92f0ae; no file
      overlap with the fusion patches). Rebuild.
- [ ] B2. Sanity: model loads, logits bit-identical vs pre-patch (logitdump
      A/B on a small prompt, or rely on managed-ngrams validation).
- [ ] B3. Re-run Phase A core (pp512/pp2048 + tg128, fused A/B) to
      re-establish the baseline with both work streams active.

### Phase C - Decode investigation (21-39 t/s)
- [ ] C1. tg128 at depth 512 and 16K (KV growth curve); op timing for
      decode (need -v; single GPU to avoid instrumentation inflation).
- [ ] C2. Identify decode cost split: expert mmvq (routed), shared ffn,
      recurrent qkv (GDN), full-attn layers (FA), lm_head, ngram/PLE.
- [ ] C3. Check whether the mmvq k-quant VDR / item-split boosts (patches
      0002) fire for Q4_K/Q5_K experts (they should: same types as
      qwen35moe). If not, why.
- [ ] C4. Check the recurrent layer-1 PLE path: is the host ngram hash on
      the critical path per token? Is the gather on CPU?
- [ ] C5. If decode is expert-bound: consider the fused down+weighted
      scale path (plan-fused-moe option for down) - but only if the data
      supports it.

### Phase D - Prefill optimization
- [ ] D1. ubatch sweep (256/512/1024/2048) at pp2048: qwen35moe showed
      ub=2048 +65% over 512; verify for qwen4exp (expert count 512,
      10 used -> different balance).
- [ ] D2. Fusion validation on qwen4exp: bit-exact (logitdump fused vs
      unfused), win/loss per ubatch. Q4_K cap 96 may need re-tuning for
      this model's shapes (n_embd 2560 vs 2048).
- [ ] D3. Full-attention layers (12 of 48): FA cost at depth; WMMA FA
      (GGML_CUDA_FA_WMMA_256) A/B like qwen35moe block 04.
- [ ] D4. GDN/recurrent layers (36 of 48): the qwen35moe GDN chunked work
      may apply; verify op timing share first.
- [ ] D5. ngram gather: if it shows up, consider fusing the hash+gather or
      moving the gather to GPU (the table is IQ4_NL; a GPU GET_ROWS with
      on-device table would need the table in VRAM - 28.8 GB - which
      likely does NOT fit with weights; so host path stays, optimize
      around it).
- [ ] D6. Tensor-split scaling: pp512/pp2048 on 2 vs 3 GPUs, layer vs
      tensor split. Expert GEMMs under split (512 experts, 10 used) -
      the MoE-scaling-under-tensor-split question from the handoff.

### Phase E - Root-cause the >600 ceiling
- [ ] E1. From A3/D data: what is the theoretical floor? Sum the top
      categories; compute a compute-bound estimate from the model's
      active FLOPs (6B active) vs GPU FLOPS.
- [ ] E2. If expert GEMMs dominate: are they compute- or bandwidth-bound
      (Q4_K vs Q8_0 load)? The qwen35moe finding was "MMQ is compute/
      bandwidth-bound, not launch-bound" - re-check for this model.
- [ ] E3. If dense gemms (Q8_0 everywhere) dominate: Q8_0 mmvq/mmq tuning
      (block 10 k-quant boosts covered Q8_0? verify) or VDR changes.
- [ ] E4. 1000 t/s prefill feasibility check: 1000 t/s at pp2048 = 2 s/eval
      on 3 GPUs = 666 ms/eval/GPU. Compare to qwen35moe's 380 ms/eval at
      1 GPU for a SMALLER model - sets expectations.

### Phase F - Finalize
- [ ] F1. Save patches (fused-moe already 0003/0004; managed-ngrams
      combined diff exists; document any new changes).
- [ ] F2. Update this plan with measured numbers, mark done items.
- [ ] F3. Write the qwen4exp results doc: numbers, findings, what applies
      to Strix Halo (gating notes).
- [ ] F4. Commit to boosts repo.

## 5. Known risks / watch-outs

- 3-GPU tensor split + logitdump2: the meta-backend host-tensor crash
  (fixed by ngram patch 0007) - keep that patch applied BEFORE running
  tensor-split logitdump.
- Op timing on 2-3 GPUs is inflated 4-5x (per-op instrumentation doubles
  node count): use llama-bench for final numbers, op timing for
  classification only.
- The model is 105 GiB; loading takes minutes. Keep prompts short
  (logitdump 27-word prompt = 2159 tokens via repeats) to bound runs.
- Fusion dispatch requires the {op,op,GLU} pattern anchored at
  ffn_moe_gate; qwen4exp uses LLM_FFN_SILU (not SWIGLU) - VERIFY the GLU
  op and the try_fuse match before assuming fusion fires.
- Do NOT set HSA_OVERRIDE_GFX_VERSION.
- llama-cli always needs --single-turn.

## 6. Dependencies / context for later

- qwen35moe fused kernel: patches 0003 (Q6_K), 0004 (Q3_K/Q4_K/Q5_K/
  Q8_0), RDNA4-gated, J caps Q6_K=64, Q4_K/Q5_K=96, Q8_0/Q3_K=64.
- managed-ngrams: wip/managed-ngrams/patches/0001-0007 + combined diff;
  applies cleanly, no overlap with fusion files.
- Decode machinery: mmvq item-split (0002), k-quant VDR boosts (block 10).
- GDN: GGML_CUDA_GDN_CHUNKED(_BF16) env; full-attn: GGML_CUDA_FA_WMMA_256.
- Baselines (qwen35moe, 1 GPU): pp512 3417-3470, pp2048 ub2048 5191-5247,
  tg128 97.5-98.5. qwen4exp first-test: >600 pp, 39 tg; recent: ~350 pp,
  ~21 tg.
