# qwen4exp - BETA (promoted from WIP)

Stable baseline for the next stage of **qwen4exp** (Qwen3.8-Flash-Next)
support work on top of the rdna-boosts core. Status: between WIP and
Release - the content below is the verified, gated baseline; new work in
this area starts from these two patches.

## **NOTE**

Qwen Sparse Attention is currently broken for scaling at the moment.
If you are trying this beta work out please use LLAMA_QSA_SPARSE_FA=0
to disable the QSA path until I get this issue sorted

## Contents

Two squashed patch files. They apply IN ORDER on the rdna-boosts core
= upstream master `9cffdcc80` + blocks 01-13 (tip commit `8f2838d1c`,
verified 2026-09-03). Applied together they reproduce the `qwen4exp`
branch tip `5cbef4c4d` tree-identically (checked with `git diff --exit-code`).

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
- re-allows `LLAMA_SPLIT_MODE_TENSOR` for qwen4exp; CPU INDEXER_TOPK
  reference, hc_mix type gate, lazy-reader F32 path.

## Apply

```
git checkout 8f2838d1c          # master 9cffdcc80 + rdna-boosts blocks 01-13
git apply managed-ngrams.patch
git apply qwen4exp-support.patch
```

Requires ROCm gfx1201 (RDNA4) for the CUDA/HIP kernels. Both patches
apply clean with `git apply --check` on the stated base.

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
