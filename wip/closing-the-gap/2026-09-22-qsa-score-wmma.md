# Phase-1 item 15 (follow-up) — QSA prefill scorer fusion (`QSA_SCORE_WMMA`), ported 2026-09-22

**Status:** code complete + op oracle green; **end-to-end qwen4exp gate PENDING** (the box's live
`llama-server` holds ~88 GB of the 124 GB unified VRAM, so the 93 GiB model cannot load).  Default
**OFF** (`LLAMA_QSA_SCORE_WMMA=1` enables) until that gate runs.  Fork `~/llama.cpp` branch
`gap-closing-r13` (r13 + `beta/mmb-general` + gap-closing `0001..0014`), commit `fed70bb36`.
Patch: [`patches/0016-…`](patches/).

## What it is

Our prefill indexer score is the per-op chain
`mul_mat([idx_dim,n_blocks] x [idx_dim, n_idx_h*n_query]) -> reshape -> relu -> head-sum`
(`build_qsa_top_k`), plus the `QSA_SCORE_BOUNDS` strip/trim.  Qwen3.8-Flash-Next (`qwen4exp`)
scores with **4 heads** and `idx_dim = 128`.

`ggml_lightning_indexer` computes exactly `sum_h w[h] * relu(q_h . k) + mask`.  With **all-ones
weights** and a **zero F16 mask** it is a drop-in for the prefill chain in one op.

**On RDNA3_5 the op did not exist for this shape.**  Our `lightning-indexer.cu` was CUDA-only for the
WMMA arm and, on HIP, only instantiated `n_head` 32/64 through the generic vec kernel; `n_head == 4`
was rejected by `ggml_cuda_lightning_indexer_supported`.  So the work is two parts:

1. **The kernel** — port the reference's (`~/pwilkin-llama-cpp` `mma.cuh`-based)
   `qsa_indexer_wmma16_keyreg` + `supports_indexer4` (RDNA3_5, 4-head/128-dim, `k` F32/F16).  The
   generic entry points are renamed `*_generic` and the dispatcher prefers the 4-head WMMA kernel.
   A **generic vec fallback for `n_head == 4`** is added too, so the op stays legal on every
   backend/arch (the cross-backend-consistency rule).
2. **The graph** — in `build_qsa_top_k`, when `n_tps >= 128 && idx_dim == 128 && n_idx_h == 4`, build
   the score with `ggml_lightning_indexer`: `query`/`key` views from `q`/`pooled`, all-ones weights
   and a zero F16 mask created via `ggml_fill` and shared across layers **by name** (the graph
   context outlives a layer build, so `ggml_get_tensor(name)` reuses one tensor per graph).

## Composing with `QSA_SCORE_BOUNDS`

The reference's fused arm *opts out* of the causal trim ("the fused WMMA scorer sizes its shared
zero mask from `n_blocks`").  We do not: the mask is all zeros and **leading-rows views** of the
shared `[n_blocks, strip]` mask/key are valid and satisfy the op's `mask->ne[0] == k->ne[2]` assert,
so the fused arm runs on the **trimmed** width, `score_blocks = qsa_score_key_limits[strip]`.  The
key is a `ggml_view_4d` (not `reshape_4d`) precisely because the trimmed view's stream stride is the
full `n_blocks` stride — the kernel reads `k->nb[2]`/`k->nb[3]`, so the non-contiguous view is fine
and needs no materialisation.

## Gates

| gate | result |
|---|---|
| **Op oracle** `test-backend-ops -o LIGHTNING_INDEXER` | **225/225**, including **81 new `nh=4` cases** (`kv` 64/256/1024, `nb` 1/16/512, `ns` 1/4, `nm` 4/1, `type_K` F32/F16/BF16).  The `nh=4` F32/F16 cases take the WMMA kernel; the BF16 cases take the new generic vec fallback. |
| **Cross-backend consistency** | `n_head == 4` now resolves for every `type_K` in the generic predicate, with vec cases for all 8 types. |
| Build | `llama-cli` / `llama-bench` / `test-backend-ops` green, no warnings. |
| **End-to-end qwen4exp** (`LLAMA_QSA_SCORE_WMMA=1` vs `=0`) | **PENDING** — width probe, same-seed greedy text, coherence, and a `-b/-ub 8192` A/B.  Blocked by the live `llama-server` (PID 3792130, ~88 GB VRAM). |
| decode/verify width purity | untouched by construction: the fused path is gated `n_tps >= 128`, so the `W=1..8` band keeps the per-op reduction order. |

**Why the end-to-end gate matters:** the WMMA kernel loads q/k as F16, so the prefill logits shift
by an F16-rounding ULP versus the F32 matmul — an **approved prefill re-baseline**, not bit-identical.
The op oracle proves the kernel is *correct* against the CPU reference; the pending gate proves the
graph wiring is right and prices the win.  Until then it is default OFF.

## Kill switches

* `LLAMA_QSA_SCORE_WMMA=1` — enables the fused prefill scorer (default OFF pending the gate).
* Flip the default to ON in `build_qsa_top_k` (`return env == nullptr || …`) once the gate is green,
  per the default-on policy.

## Files

`ggml/src/ggml-cuda/lightning-indexer.cu` (the WMMA kernel, `supports_indexer4`, the `n_head == 4`
generic fallback), `src/models/qwen4exp.cpp` (`idx_score_wmma`, `use_wmma`, the shared
weights/mask and the fused branch in `compute_score`), `tests/test-backend-ops.cpp` (the `nh=4`
oracle cases).  Patch: [`patches/0016-…`](patches/).
