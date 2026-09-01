# qwen4exp WIP (Qwen3.8-Flash-Next, sparse attention / ITEM B)

Exploratory work on the **qwen4exp** (Qwen3.8-Flash-Next) path: sparse
flash-attention via the new `GGML_OP_FLASH_ATTN_QSA` op, validated on
3x Radeon AI PRO R9700 (gfx1201).

**Source of truth (the code):** the `qwen4exp` branch of
`~/prs/llama.cpp` (the protected pre-re-base fork). ITEM B is committed
there as `554691a72` (op + tiled all-heads-per-block kernel). This
directory holds the planning docs, handoffs and WIP patches that
survive context compaction; the full history is in the handoffs.

**Not part of the delivery patch set:** qwen4exp is a *model-architecture*
work stream, separate from the MoE work shipped as block 13. It stays WIP
here until ITEM B's latency work + packaging are done.

## Contents

| File | What |
|---|---|
| `plan-qwen4exp.md` | The plan: sparse attention design, depth-constant prefill goal, kernel structure |
| `qwen4exp-handoff-2026-08-31.md` | Session 1 handoff: ITEM A, op design, validation methodology |
| `qwen4exp-handoff-2026-08-31-session2.md` | Session 2 handoff: kernel optimization passes, L2 fix, head-sum fusion (reverted), measured curve |
| `patches/0005-kv-prev-tokens-index.*` | WIP: KV prev-tokens index (earlier qwen4exp infra) |
| `patches/0006-qsa-topk-radix-gpu.patch` | WIP: GPU radix top-k for the indexer (validated, separate from upstream PR #27466's implementation) |
| `patches/0007-ple-batch-cache-archive.patch` | WIP: PLE batch cache (archived neutral) |

## ITEM B status (from handoff UPDATE 12)

- **Committed:** `554691a72` on the qwen4exp branch — sparse QSA op +
  tiled kernel + all-heads-per-block L2 fix.
- **Depth-constant attention:** FLASH_ATTN_QSA 11.9 ms @64K vs dense
  158 ms. Sparse vs dense: +6% @16K, +24% @32K, +56% @64K.
- **Correctness:** inside llama.cpp's own kernel-variance envelope
  (max diff 6.50 / top-1 84% vs the tile-vs-wmma control 7.28 / 84.8%).
- **Remaining:** latency optimization (~35% GPU, per-warp serial chain,
  ~3x headroom); packaging; handoff UPDATE 12+; fold back into
  `rdna-boosts` afterwards.

## Where things live

- Code: `~/prs/llama.cpp` branch `qwen4exp` (checkpoints `d2548f9af`
  base, `e2d2a1f1c` op, `554691a72` kernel-opt).
- Env toggles: `LLAMA_QSA_SPARSE_FA=1` (enable sparse),
  `GGML_CUDA_QSA_IDENTITY=1` (validation), `GGML_CUDA_OP_TIMING=1`
  (op timing, needs `-v`, disables CUDA graphs).
- The fork's `rdna-boosts` branch carries blocks 01-13 (delivery set)
  and does NOT include ITEM B; the merge-back happens after ITEM B
  latency work lands.
