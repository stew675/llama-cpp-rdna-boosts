# Session handoff 2026-09-02: ITEM B re-applied onto the 0eadefebd re-base

## What happened

qwen4exp ITEM B (sparse QSA flash attention, `GGML_OP_FLASH_ATTN_QSA`) was
re-applied from the protected pre-re-base branch onto the CURRENT fork
(0eadefebd + blocks 01-13). The active code is now:

- Branch: `~/llama.cpp` `qwen4exp` @ `0ff151bda`
  = `rdna-boosts` (482837e5a) + one commit `0ff151bda` (the re-apply).
- Protected reference (do NOT edit): `~/prs/llama.cpp` `qwen4exp` @
  `554691a72` (old base, kept for backup/reference).

## Why re-apply, why now

The current fork was re-based onto upstream `0eadefebd` (22 commits) which
touched qwen4exp-adjacent areas (radix TOP_K #27466, MOE fusion #27621,
kv-cache #27991, vulkan top_k #28032, flash-attn XOR swizzle #25635). The
qwen4exp branch on `~/prs/llama.cpp` predates that re-base, so resuming
ITEM B there would mean working against the old base. Cherry-picking onto
the current fork keeps all block 01-13 work + ITEM B in one tree.

## The cherry-pick (what was done)

1. `git fetch ~/prs/llama.cpp qwen4exp` (objects for e2d2a1f1c, 554691a72).
2. Branch `qwen4exp` off `482837e5a`.
3. `git cherry-pick -n e2d2a1f1c` (ITEM B op, 9 files) - ONE conflict:
   - `ggml/src/ggml-backend.cpp` `ggml_backend_op_alloc_size_may_expand`:
     upstream #28071 added `GGML_OP_MUL_MAT` there during the re-base;
     ITEM B added `GGML_OP_FLASH_ATTN_QSA`. Resolution: keep BOTH.
   - Everything else auto-merged (ggml.h op enum, ggml.c op+name, rpc.h
     GGML_OP_COUNT 101->102, backend-meta.cpp split handler, ggml-cuda.cu
     dispatch x2, fattn-qsa.cu/.cuh new files, qwen4exp.cpp build_attn_qsa).
4. `git cherry-pick -n 554691a72` (kernel-opt, fattn-qsa.cu only) - clean.
5. Committed as `0ff151bda` (message records the conflict resolution).

## Verification (2026-09-02)

- Build: `cmake --build build-rocm --target llama-bench llama-cli` OK.
- Smoke: qwen4exp Flash-Next loads on 3 GPUs with tensor split.
- Model: `/models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
  (4 shards, ~103.68 GiB, 48 layers, top_k 2048, 512 experts/10 used).
- Correct invocation: `HIP_VISIBLE_DEVICES=0,1,2 ... --split-mode tensor`.
  WITHOUT `--split-mode tensor` llama-cli OOMs (default layer-split can't
  fit ~80 GiB VRAM across the GPUs).
- Sparse (LLAMA_QSA_SPARSE_FA=1) pp16384 = 454.03 t/s; dense = 488.58 t/s
  (at 16K sparse trails dense ~7%; the +56% sparse win is at 64K depth,
  matching the docs-era curve).
- LLAMA_QSA_SPARSE_FA=1 is REQUIRED to enable the sparse op; the dense
  fallback (indexer + mask) is the default.

## State / next steps

- Repos: `~/llama.cpp` (qwen4exp branch active, working tree clean),
  boosts repo TODO/README updated, both pushed.
- Next work (ITEM B latency, ~35% GPU util): shared-memory K/V tile
  staging, software-pipelined score pass, or 2 columns/block in
  fattn-qsa.cu; re-verify the logitdump envelope + depth curve after each
  change. See `wip/qwen4exp/plan-qwen4exp.md` + session-2 handoff.
- TODO.md: ITEM B remains [IN-PROGRESS] under Active.
