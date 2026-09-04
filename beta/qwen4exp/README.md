# qwen4exp - BETA (promoted from WIP)

Stable baseline for the next stage of **qwen4exp** (Qwen3.8-Flash-Next)
support work on top of the rdna-boosts core. Status: between WIP and
Release - the content below is the verified, gated baseline; new work in
this area starts from these three patches.

## Contents

Three squashed patch files. They apply IN ORDER on the rdna-boosts core
= upstream master `8b4b3558f` + blocks 01-13 (re-based/regenerated
2026-09-04 from the previous `9cffdcc80`-based `8f2838d1c` set).
Applied together they reproduce the `qwen4exp` branch tip `9ce0d1da1`
tree-identically (checked with `git diff --exit-code`).

The 2026-09-04 re-base re-applied the first two patches onto the current
master (39 commits of upstream drift past the old base) and resolved the
one conflict it surfaced: upstream's qwen4exp attention used the dense
masked-FA path (`build_attn_mha`) where the beta patch installs the QSA
sparse-FA default (`LLAMA_QSA_SPARSE_FA=0` opt-out).  Resolution kept
the beta side — the dense fallback call in the patch is identical to
upstream's current call, so nothing was lost.  Full build clean (ROCm
7.14 gfx1201) after the merge; patch 3 (MTP) added on top 2026-09-04.

### 1. `managed-ngrams.patch`
The managed lazy-reader work (the 0001-0007 set, squashed to one patch):
- new lazy reader (`llama-lazy-reader.cpp/.h`, `--lazy-buffer-size N`,
  pread row reads, meta/CPU_Mapped placement) with managed n-gram (PLE)
  row loading,
- `n_lazy_buf_size` plumbing through `llama_model_params`/loader,
- test-lazy-reader + arch-test roundtrip coverage,
- the PLE n-gram load block + models.h entries.

### 2. `qwen4exp-support.patch`
Everything else qwen4exp on the branch (the 15 cherry-picked commits +
the 2 rebase fixes + the 3 crash/coherence fixes, squashed):
- QSA layers: fused indexer top-k (`GGML_OP_INDEXER_TOPK`, radix),
  sparse flash attention (`GGML_OP_FLASH_ATTN_QSA`) - now the DEFAULT FA
  path; `LLAMA_QSA_SPARSE_FA=0` keeps the dense path; `-fa off` uses the
  manual path; CPU reference for the sparse op,
- fused decode ops `GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE` (+ kernel
  geometry, rms/gamma fold, F32 inject fold, head-call fusion) - the
  +14.5% decode campaign,
- multi-seq decode: QSA 4D-mask assert fix + unary-q8_1 layout fix
  (unlocks K-seq decode),
- fixes found during the server bring-up: meta tensor-split cgraph
  arena-reset dangling fix; QSA sparse FA BF16 mask staging fix (bf16 KV
  garbage);
- QSA DECODE FIX (2026-09-03): stage V in smem for the BF16 sparse FA
  path (the VKQ pass re-read every cell's V from L2 per head-warp);
  slice the decode top-k walk across gridDim.y (was one block for the
  whole list = one CU) with per-slice online softmax + the dense FA
  combine kernel, 64-cell slices. Numerics bit-identical to the
  single-block path; sparse decode now at dense parity short-context and
  faster past ~8K real KV (see the validation section);
- MoE weighted-reduction fusion unblocked on the Meta-TP path: the expert
  aggregation (weighted MUL + per-expert views + the add chain) now fuses
  into one kernel on ALL layers (was 2/48). The Meta backend gained a
  graph_optimize that registers the alloc deps against the scheduler's
  allocator (shared structural matcher in
  ggml/src/ggml-moe-weighted-reduction.h), so the gallocr keeps the
  experts input alive and the CUDA-side memory check passes. Decode
  -0.27 ms/step at real KV 2048; fused == unfused logits bit-identical;
- QSA LAYER-SPLIT CRASH FIX (2026-09-04): heads chunked across launches.
  The kernel is launch-bounded to 16 warps (one per head) but the dispatch
  put ALL Q-heads of a layer in blockDim.y, so a 24-head QSA layer in
  layer-split mode (each GPU runs whole layers) launched a 768-thread
  block and HIP rejected it ("unspecified launch failure", GGML_ABORT).
  Tensor split hid the bug: 24 heads / 3 GPUs = 8 per device.  Fix: one
  launch per QSA_MAX_HEADS (16) head group, `head_base` kernel arg,
  `KQ_w` indexed by the local warp id; blockDim.y capped at 16.  Layer
  split now runs: pp10240 1000 t/s / tg128 34.6 t/s (vs tensor 1345/47.5),
  at parity with the dense masked-FA fallback in layer mode;
- re-allows `LLAMA_SPLIT_MODE_TENSOR` for qwen4exp; CPU INDEXER_TOPK
  reference, hc_mix type gate, lazy-reader F32 path.

### 3. `mtp-draft-support.patch` (2026-09-04)

The NextN/MTP draft head for Qwen3.8-Flash-Next (`--spec-type

draft-mtp` + the unsloth `mtp-*.gguf`), following the upstream draft
PRs ggml-org/llama.cpp#27836/#28243 but implemented on the beta tree:

- head-only MTP exports load (trunk tensors optional via `mtp_only`,
  trailing block under `ml.load_mtp`); trunk exports the wide residual
  as h_nextn for the speculative driver;
- the head folds the next token's embedding into the wide
  hyper-connection residual (eh_proj = fc_embedding + fc_hidden fused),
  runs one trunk-shaped block (dense attention + MoE wrapped in
  hyper-connections) and collapses with its own hc_head_* mixer before
  reusing the trunk's LM head.  Dense is the model-faithful choice: the
  GGUF gives blk.48 compress_ratio 0 (its indexer tensors are dead
  weight), so sparse-in-the-head would degrade drafts; the head still
  rides the fused decode ops (GGML_OP_HC_MIX/HC_COMBINE at nt==1);
- verified 2026-09-04 (3x R9700, IQ4_XS target + Q4_K_M head): loads,
  generates; draft acceptance 0.469 (69/147), mean accepted length 2.38;
  decode +34% vs plain at 8K ctx (57.4 vs 42.8 t/s).

When upstream merges #27836/#28243, drop this patch (and the `-md`/spec
flags stay as-is).

## Apply

```
git checkout <master>            # fresh llama.cpp master pull
bash <delivery>/scripts/apply-all.sh .   # master + rdna-boosts blocks 01-13 (git am)
git apply managed-ngrams.patch
# (optionally commit the lazy-reader work here; the later patches'
#  pre-images are the earlier patches' state either way)
git apply qwen4exp-support.patch
git apply mtp-draft-support.patch
```

Requires ROCm gfx1201 (RDNA4) for the CUDA/HIP kernels. All three
patches apply clean with plain `git apply` on that base (re-verified
2026-09-04 on master `8b4b3558f` + blocks 01-13: applied tree
byte-identical to the branch tip). If master drifts further,
`git apply --3way` (or a manual resolve on the qwen4exp.cpp attention
path) is the fallback — the patch pre-images now match the current
master-based files, so drift has to overlap the patched regions again
before conflicts return.

## Validation status (the gates this baseline holds)

- llama-bench (3x R9700 tensor, ngl 99, ub 2048, warm page cache):
  tg128 45.7 t/s, pp512 1538, pp8192 2024 - matches the pre-rebase refs.
- Single-seq decode byte-identical streams vs the pre-rebase reference
  (`198 42750 367 ...`); K=2 multi-seq seq0 == seq1 == K=1.
- bseq_val CPU-only (-ngl 0) == GPU streams (sparse op CPU reference).
- test-llama-archs: qwen4exp MoE GPU 9.35e-14, CPU 0.00, full matrix
  607 OK / 0 fail.
- llama-server: user full config (ctx 102400, mlock, ctx-checkpoints,
  reasoning, tensor split) loads, reasons coherently, 39.4 t/s decode.
- QSA DECODE FIX numbers (bseq_pp, bf16 KV, tensor split, steady state
  ms/step; sparse vs dense): real KV 512: 20.2 vs 20.2; 2048: 20.6 vs
  20.5; 8192: 21.1 vs 21.3; 32768: 23.3 vs 24.3 (sparse leads). Server
  (user config, HTML prompt): was 38.4 t/s decaying to 29.2 by 1500 gen
  (tg_3s); now flat ~46.5-48 through 4000+ gen (avg 46.7), no decay.
- BF16 KV + sparse FA coherent (mask-staging fix); bf16 is not silently
  downgraded to f16 on this branch.
- Cold page-cache lesson: warm the 104 GiB model file before benching
  (the first evals otherwise pay disk page-ins - a deterministic but
  fake ~4x prefill slowdown).

## Open items (carried forward from WIP)

- "The answer" 3-token K=2 multi-seq decode drift at step 3 (single-seq
  untouched; numeric divergence of the short-odd-prompt batch path).
- Mixed K/V cache types (k=bf16/v=f16) crash at model init
  (meta split-state assert, ggml-backend-meta.cpp:537).
- FA-off + tensor-split: unsupported (upstream Meta-backend constraint:
  SPLIT_MODE_TENSOR requires FA; manual-attention path aborts in
  handle_set_rows on the row-split kq_mask). Would need meta-splitter
  work; no current consumer.
- Decode levers (measured, not landed): server-level batch decode
  (M > 1 kernels), decode-expert mmvq config sweep, GDN state fold
  (~1.5-3%), body-op elementwise fusion.
- Prefill thread (pp8192 at ~2024; further prefill tuning if resumed).
- ML-Kernel/gpudh review vs TP-V1; v_shifted probe; splitter->F32;
  shuffle-to-smem; ggml-backend-meta import gates; multi-device sync in
  meta_tp_test.
- Model specifics: UD-Q4_K_XL 4-shard GGUF (103.68 GiB), 3x R9700
  gfx1201, ROCm /opt/rocm-7.14-gfx1201. Env for the gates:
  `GGML_CUDA_FA_WMMA_256=0` (sparse FA is default; `-fa off` manual).

## Archive

The full WIP history (per-commit patch files 0005-0020 with notes, the
handoff/handover docs, plans, and micro-bench tools) was moved to
`../../wip/qwen4exp/archive/` when this directory was created - the
items there were promoted to this beta directory as the pair of
squashed patches above.
