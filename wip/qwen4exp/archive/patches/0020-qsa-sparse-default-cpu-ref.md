# Patch 0020: QSA sparse flash-attn is now the default; CPU reference added

llama.cpp commit 5cbef4c4d, on top of 1f5d431f7.
Apply: `git apply patches/0020-qsa-sparse-default-cpu-ref.patch`

## The contract

- The fused QSA sparse flash-attention kernel (attend only the indexer-selected
  top-k cells) is the DEFAULT path for the QSA layers when flash attention is
  enabled.
- `LLAMA_QSA_SPARSE_FA=0` keeps the dense masked flash-attention path.
- `-fa off` disables flash attention entirely: the manual (non-FA) attention
  path is used - the sparse branch is now gated on `cparams.flash_attn`.
- Env semantics flipped: unset or 1 = sparse, 0 = dense (was: unset = dense).

## Why a CPU reference

With sparse as the default, every qwen4exp graph (including CPU builds at
`-ngl 0` and the test-llama-archs CPU/Meta runs) contains `GGML_OP_FLASH_ATTN_QSA`
when flash attention is on. The op had only a CUDA/ROCm kernel, so CPU-only
graphs would abort. Added `ggml_compute_forward_flash_attn_qsa` (ops.cpp) with
the same contract as the kernel: q [D, n_tps, n_head, n_stream] F32, k/v
[D, n_kv, n_head_kv, n_stream] F16/BF16/Q8_0 (F32 also handled), idx
[n_top_k, n_tps, 1, n_stream] I32, mask [n_kv, n_tps, 1, n_stream] F16; scores
over the top-k cells with the mask gathered at the idx positions; softmax over
the top-k cells only. Two passes over the cells (max, then exp weights + V sum)
so no per-query scratch is needed. Dispatched from ggml-cpu.c (n_tasks =
n_threads; no wdata).

## Verification

- llama-bench with NO env (default now sparse): pp512 1538.3 / tg128 45.75 -
  matches the previous env=1 numbers (45.7).
- bseq_val with no env: K=1/K=2 streams identical to the known-good refs
  (198 42750 367 367..., 369 9542 11 694...). env=0 (dense) also coherent.
- CPU-only decode (-ngl 0, real model, small prompt): same stream as GPU
  (198 42750 367 367 318) - the CPU reference matches the sparse kernel.
- test-llama-archs qwen4exp: GPU MoE OK (9.35e-14), CPU OK (0.00).
- User's full config at ctx 102400 (no env needed): 39.4 t/s, coherent
  reasoning output.
- FA-off + tensor split is a pre-existing documented constraint
  ("SPLIT_MODE_TENSOR requires flash_attn"); FA-off + layer split works
  (manual attention path, 32.4 t/s decode) - unchanged by this patch.
