# Session handoff 2026-09-01 (evening): block 13 released; qwen35moe 2-GPU hang flagged

## What happened this session

1. **Block 13 released** - the MoE WIP (mmvq short-K item-split + fused
   gate+up+GLU MMQ + M4 quant extension) is now an official patch:
   - Fork `~/llama.cpp` rdna-boosts: block 13 committed `a14257996`.
   - Boosts `patches/`: 13-patch set, block 12 renamed to the 000N
     convention (`0012-...`), block 13 = `0013-...`.
   - `scripts/make-patches.sh` + `scripts/apply-all.sh` simplified: all 13
     blocks apply with `git am` (block 12's old `git apply` special case
     is gone; branch name configurable via `RDNA_BRANCH`).
   - Verified: fresh checkout at `0eadefebd` -> `apply-all.sh` -> applied
     tree byte-identical to fork tip, zero whitespace warnings, ggml-hip
     builds. `rdna-boosts-all.patch` regenerated (41 files).
   - Validation (1-GPU qwen35moe, the verified config): prefill Q6_K
     +5.1%, Q4_K_M +3.6%; decode tg128 +5.6% (97.28 vs 92.15 pristine);
     same-seed coherence IDENTICAL; test-backend-ops 2/2 OK.
   - Two correctness gates added to the try_fuse arms (multi-token
     x_scale_channel_dst -> single-token only; fused MMQ arm gated to the
     instantiated type list). See TODO.md and patches/README.md.

2. **qwen4exp split out** to `wip/qwen4exp/` (plan + handoffs + patches
   0005-0007). MoE-only work stays in `wip/qwen35moe-prefill/`.

3. **TODO.md created** - cross-project tracker (block 13, qwen4exp ITEM B,
   the multi-token fusion follow-up, MXFP4, 7900XTX ports incl. community
   help, decode push, and the qwen35moe 2-GPU hang).

## NEW: qwen35moe 2-GPU decode hang (highest priority next)

During block-13 validation we discovered that qwen35moe **Q5_K_M/Q6_K**
2-GPU (and 3-GPU) tensor-split decode **hangs** (GPUs busy-wait at 100%
one at a time, no output). Q3_K_M/Q4_K_M 2-GPU work (62 t/s); 1-GPU works
(75 t/s); dense 27B Q8_0 2-GPU works.

Bisected across 5 build points (old a7cc83bba base, pre-41ef91f7c,
pre-f8dbcd618, pure upstream 0eadefebd, fork tip): **hangs everywhere** -
it is NOT a regression from the 22 upstream commits, NOT PR #27466, NOT
block 13. qwen35moe was only ever validated single-GPU; the historical
2-GPU decode numbers in patches/README were the dense Qwen3.8-27B.

Full details, suspects (MUL_MAT_ID multi-GPU ids-gather, RCCL on
Q5_K/Q6_K shapes, mmvq table VDR differences, Q8_0 test), and tooling
pointers are in TODO.md (Priority section). This is the next work item.

## Repo state (all clean, committed)

- `~/llama.cpp` rdna-boosts @ `a14257996` (13 blocks)
- `~/prs/llama.cpp` qwen4exp @ `554691a72` (protected pre-re-base work)
- boosts repo @ `c3c7d0c` (release + move commits on top)
