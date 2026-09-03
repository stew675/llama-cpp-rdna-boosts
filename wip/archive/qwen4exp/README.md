# qwen4exp WIP (Qwen3.8-Flash-Next, sparse attention / ITEM B)

Exploratory work on the **qwen4exp** (Qwen3.8-Flash-Next) path: sparse
flash-attention via the new `GGML_OP_FLASH_ATTN_QSA` op, validated on
3x Radeon AI PRO R9700 (gfx1201).

**Source of truth (the code):** the `qwen4exp` branch of
`~/llama.cpp` (re-applied 2026-09-02 onto the 0eadefebd re-base +
blocks 01-13, tip `0ff151bda`). ITEM B is committed there as the
re-apply commit `0ff151bda` (cherry-pick of the protected
`~/prs/llama.cpp` qwen4exp branch commits `e2d2a1f1c` op + `554691a72`
kernel-opt; 1 conflict resolved: ggml-backend.cpp kept the upstream
MUL_MAT alloc-expand case alongside the new FLASH_ATTN_QSA case). The
old pre-re-base branch still lives at `~/prs/llama.cpp` `qwen4exp` @
`554691a72` (protected, for reference/backup only). This directory
holds the planning docs, handoffs and WIP patches that survive context
compaction; the full history is in the handoffs.

**Not part of the delivery patch set:** qwen4exp is a *model-architecture*
work stream, separate from the MoE work shipped as block 13. It stays WIP
here until ITEM B's latency work + packaging are done.

## Contents

| File | What |
|---|---|
| `plan-qwen4exp.md` | The plan: sparse attention design, depth-constant prefill goal, kernel structure |
| `handover-2026-09-02.md` | **CURRENT handover: repo/branch layout, ITEM B status, bench config, captured diffs, next work. READ THIS FIRST.** |
| `handoff-2026-09-02-rebase-reapply.md` | ITEM B re-apply onto the 0eadefebd re-base (conflict + verification) |
| `qwen4exp-handoff-2026-08-31.md` | Session 1 handoff: ITEM A, op design, validation methodology |
| `qwen4exp-handoff-2026-08-31-session2.md` | Session 2 handoff: kernel optimization passes, L2 fix, head-sum fusion (reverted), measured curve |
| `patches/0008-item-b-qsa-sparse-flash-attn.patch` | **ITEM B (the active op/kernel) - full diff, applies clean to rdna-boosts** |
| `patches/0005-kv-prev-tokens-index.patch` | WIP: KV prev-tokens index query side (seq_pos map is already upstream via #27991) |
| `patches/test-kv-prev-tokens.cpp` | Unit test for 0005 (compile standalone) |
| `patches/0006-qsa-topk-radix-gpu.patch` | HISTORICAL ONLY: radix top-k, superseded by upstream #27466 |
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

- Code (ACTIVE): `~/llama.cpp` branch `qwen4exp` @ `0ff151bda` (re-applied
  onto 0eadefebd + blocks 01-13, 2026-09-02). The `rdna-boosts` branch
  carries blocks 01-13 and does NOT include ITEM B; `qwen4exp` is
  rdna-boosts + ITEM B. **Work ONLY on `qwen4exp`; never pollute
  `rdna-boosts`.**
- Code (protected reference): `~/prs/llama.cpp` branch `qwen4exp`
  (checkpoints `d2548f9af` base, `e2d2a1f1c` op, `554691a72` kernel-opt).
  Same binary is op-for-op identical to the current fork on the dense path.
- Env toggles: `LLAMA_QSA_SPARSE_FA=1` (enable sparse),
  `GGML_CUDA_QSA_IDENTITY=1` (validation), `GGML_CUDA_OP_TIMING=1`
  (op timing, needs `-v`, disables CUDA graphs).
