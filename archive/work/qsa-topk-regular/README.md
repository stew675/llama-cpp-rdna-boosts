# QSA indexer top-k: the regular-layout fast path (contributor PR #320, promoted in r302)

**Status: PROMOTED into official block 11 at r302 (`v17-4f9dfa0e3-r302`).**  Nothing here is applied by anything;
`patches/0011-*.patch` carries the code.  This record keeps the contributor's material as submitted plus this
repo's own re-measurement, because the two disagree on magnitude and the disagreement is the useful part.

## Where it came from

`briansp2020` (Brian) opened **PR #320** against this repo as a `wip/` campaign (branch `wip/qsa-topk-regular-r299`,
one patch cut against r298 + a README + five logs).  The delivery was at r301 when it was assessed.  Their
`indexer-topk.cu` was byte-identical from r298 (`99c4717c6`) to r301, so the patch applied unchanged.

- `README.md` (their submission, verbatim, below in `CONTRIBUTOR-README.md`)
- `0001-QSA-indexer-top-k-regular-layout-fast-path.patch` (their patch as submitted)
- `logs/` (their A/B runs and the two `rocprofv3` kernel-stat CSVs)

## What it does

With one stream and derived visibility the qwen4exp indexer cache is usually laid out in position order: every
visible cell `j` holds position `j` (`cell_pos[j] == j`), sits in block `j/r` (`cell_blk[j] == j/r`), and the
visible cells are a prefix of the cell range.  A query at position `q` then sees exactly `[0, min(q, n_used-1)]`,
so block `b`'s visible cells are its leading `w(b) = clamp(min(q, n_used-1) - b*r + 1, 0, n_in)` cells and every
per-cell quantity of the radix passes follows from the block.  Pass 1 and the block passes compute `w(b)` instead
of reading `cell_blk`/`cell_pos` per cell; `wvis` is not written; `count_cells_grouped` returns early (there is no
cell/block partition mismatch left to recount); the gather walks whole blocks in ascending cell order (the block
key on the visible prefix, `-inf` after it), so every cell lands where the cell-level gather puts it.

`indexer_topk_regular_check()` validates the layout on the device on every call (one workgroup, ~30 us) and any
cache that fails the check (a second stream, no derived visibility, a pool/window/foreign cell, an offset layout)
runs exactly the pre-existing code.

## What this repo changed before folding it in (r302)

1. **A census**, `GGML_QSA_TOPK_REGULAR_LOG=1` (`ENVIRONMENT.md` §8).  The fallback is silent by construction, so
   a re-derivation that breaks the layout would lose the whole win while every text arm stayed green -- the same
   blind spot `GGML_CUDA_QSA_SCORE_LOG` exists for.  The verdict is mirrored by the check kernel into mapped
   pinned host memory (no host sync and no D2H copy: the kernel is part of a captured graph, where neither is
   legal), so the line reports the previous launch's verdict.
2. **The op oracle's regular-layout coverage**, in `tests/test-backend-ops.cpp`: the shipped `INDEXER_TOPK` suite
   had exactly one regular case ("everything visible, exact tail"), so the fast path's `w(b)` edges were untested.
   Eight cases were added: partial visibility, a single visible block, a partial tail block, visibility across that
   tail, a QSA-trimmed cell range (`n_kv > n_blocks*r`), `k == n_kv`, `k` above the visible count, and
   `blk_idx == -1` blocks -- plus the three shapes that must DECLINE.
3. **A gate**, `scripts/gate-qsa-topk-regular.sh`, which asserts the oracle on both arms, the census, a measured
   prefill delta against the kill switch, text identity, and faults.

## This repo's measurement (gfx1201, ROCm 10.1, the delivery's own shapes)

Interleaved on/off with the same binary, `ROCBLAS_USE_HIPBLASLT=0` and `MOE_EXPERT_HEATMAP=load` on every arm:

| shape | prompt | off (t/s) | on (t/s) | delta |
|---|---|---|---|---|
| 3 GPU `-sm tensor`, all-VRAM (`-ncmoe 0`, PLE on host, `-ub 4096`) | ~24k | 2509.5 | 2541.1 | **+1.26 %** |
| 3 GPU `-sm tensor`, all-VRAM | ~96k | 2432.8 | 2517.3 | **+3.47 %** |
| 3 GPU `-sm tensor`, all-VRAM (the gate's arm) | ~84k | 1933.0 | 1975.9 | **+2.22 %** |
| 1 GPU `-sm layer -ncmoe 48` (streamed via the host pool) | ~24k | 494.10 | 494.87 | **+0.16 %** |

Their claims: +1.7-1.9 % at 31.5k and +3.8-3.9 % at 101k (2 GPU, all-VRAM), -2.5 % wall time on the host-pool
setup, +9.3 % on their own tree at 259.6k.  Direction and context-scaling confirmed; magnitude ~10 % lower at
depth; the streamed shape (their `-ncmoe 47` 2-GPU tensor arm) is essentially neutral here, which is expected
because that prefill is host-expert bound.  Decode is unchanged.  Their own kernel trace (the top-k bucket 8.5 %
-> 4.3 % of kernel time at 101k) is the mechanism and it is consistent with a 2-3 % end-to-end prefill.

Correctness / purity (this repo, r302 build):
- `test-backend-ops -o INDEXER_TOPK` **12/12** on both arms (the 8 new cases included); `FLASH_ATTN_QSA` green.
- greedy text byte-identical to r301 in every arm: `1c5d32ac537d` (dense 4B 3-GPU), `b3de97c70c88` (qwen4exp
  1-GPU `-sm layer`, 24k), `5120581a3300` (qwen4exp 3-GPU `-sm tensor`, 84k), `2cf39d00172c` (qwen4exp +
  `draft-mtp n4`, `-n 600`).
- prefill-logit KLD **0.000000 / 100.000 %** at the canonical point, **0.000000 / 99.990 %** on the 3-device
  `-sm tensor` point, and **0.000000 / 100.000 %** on a qwen4exp (Flash-Next) point recorded from the r301 build.
  The qwen4exp point is the one that reaches the changed code; note that it needs the hipBLASLt pin on BOTH arms
  (`INVARIANTS.md` r302 fact 1).
- census: 708/708 indexer-top-k launches took the regular path on the tested shape (a temporary device printf);
  in the shipped census form, 3 `prev=engaged` lines on the fast arm and 0 on the kill-switch arm.
- `llama-batched-bench -npl 1,4,8` flat; the quant-coherence gate's sparse QSA section byte-identical to its
  chain; per-block build 2/2; no GPU faults in any arm.

## Limits (as submitted, still true)

- The fast path engages only for ONE stream with derived visibility; everything else takes the existing code.
- Not tested by the contributor on NVIDIA/CUDA, RDNA3, gfx1151, Windows, or other indexer models.  The check
  kernel's reductions assume 32-wide warps -- which matches this file's existing convention (`tid & 31`,
  `__shfl_down(..., 16)`, `BLOCK_SIZE/32` in `count_cells_grouped`), so it is not a new constraint here, but it
  is a real one for a wave64 backend.
- The delivery's validated scope is `gfx1201` + `gfx1151`/`gfx1100`; the arm was measured on `gfx1201` only.
