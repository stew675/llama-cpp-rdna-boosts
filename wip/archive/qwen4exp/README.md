# qwen4exp delivery archive — 2026-09-13 restructuring

Post-campaign archive after the patch-hygiene pass. The active delivery is:

- `patches/` 0001-0013 (13 top-level blocks on upstream master; **0002/0004/0008/0013
  amended 2026-09-13** to absorb the model-neutral Strix kernel work)
- `beta/qwen4exp/qwen4exp-support.patch` (**ONE patch** — all qwen4exp-specific content)

**Verification (2026-09-13):** fresh worktree at `8b4b3558f` (the fork's master/block base)
+ blocks 0001..0013 (amended set) + `qwen4exp-support.patch`, plain `git apply` in order,
reproduces the qwen4exp fork tip `f5ac11903` **byte-identically (0 diff lines)**. The fork
(`~/llama.cpp` qwen4exp branch) keeps the full 22-commit history as the authoritative record.

## Patch-history trail

### Fork chain (ground truth, `~/llama.cpp` qwen4exp branch)

Base `8b4b3558f` (master) → 13 rdna-boosts block commits (b01 `ed5231b09` … b13
`da67bcb88`) → 22 qwen4exp commits (e1 `9a9ef9fd1` … e22 `f5ac11903`).

### Old delivery (superseded, archived in `patches-21-series-2026-09-13/`)

21 numbered patches applying on `da67bcb88` = one per fork commit (squashed e1+e10's
lazy-reader split). Original files archived here.

### New delivery (2026-09-13 folds)

Each piece folds into the *last* block owning its files (a fold's hunks need the file's
final block state — hence `mmvq.cu`/`ggml-cuda.cu` folds land in 0013, not 0008):

| fold piece (fork commit) | file deltas | absorbed by |
|---|---|---|
| gdn NW16 scan retune (376f02aa0) | gated_delta_net_chunked_bf16_gfx11.cu | **0002** |
| fattn RDNA WMMA row (e7eecb369) | fattn-mma-f16.cuh | **0004** |
| scale-unary fused kernel (f5ac11903) | unary.cu, unary.cuh | **0008** |
| scale-unary try_fuse window (f5ac11903) | ggml-cuda.cu | **0013** |
| routed-compact MoE mmq (1da01fa67) | mmq.cuh | **0013** |
| swiglu-input quantize (7a6a2e97b) | mmq.cu/cuh, quantize.cu/cuh, ggml-cuda.cu | **0013** |
| mwr float4 (f33ffaca7) | moe-weighted-reduction.cu | **0013** |
| split_j + Q8_0 rows (6d457634e) | mmq-config-rdna3-5.cuh, mmq-vec-dot.cuh, mmq.cuh | **0013** |
| quantize chunk (0a3a2b498) | mmq.cu, quantize.cu/cuh | **0013** |
| mul_mat_q_pair kernel (6a80b695c) | mmq.cu/cuh | **0013** |
| weighted-down mmvq kernel (b31940a5e) | mmvq.cu/cuh | **0013** |

**Stayed in beta** (`qwen4exp-support.patch`): everything qwen4exp/QSA-specific —
managed-reader base, qwen4exp support, mtp-draft, WS4 hc fusions, sched-fallback-sync
(core ggml; QSA-adjacent, upstream-PR candidate), QSA shortcut, **weighted-down + pair
try_fuse windows** (their ggml-cuda.cu context depends on beta-support windows), PLE,
QSA_OFF gate, concat-transposed, mmid-512x10, repeat-absorb, mmvq glue.

### Why 0013 and not 0008 for mmvq/ggml-cuda pieces

Block ownership of a file determines where its later deltas can fold: `mmvq.cu` is owned by
0008 *and* 0010 *and* 0013; `ggml-cuda.cu` by 0003/0006/0008/0011/0013. A folded delta's
pre-image must equal the file's state at the fold point, which is only guaranteed at the
**last** owning block. The scale-unary kernel itself (unary.cu/cuh, 0008-owned only) folded
into 0008 while its try_fuse window (ggml-cuda.cu) went to 0013.

## Directory map

- `patches-21-series-2026-09-13/` — the pre-fold 21-patch numbered series (one per fork
  commit; the intermediate stage before folding), archived verbatim.
- `discovery/` — the dated Strix Halo investigation records + older session briefs moved
  out of `benchmarks/` and `wip/strix-halo/` (kept: methodology/gate docs in `benchmarks/`,
  the current `wip/strix-halo/SESSION-BRIEF-2026-09-12.md`, `TODO.md`).
- Active follow-ups live in `TODO.md` (topk quality gate, ssm-pair, launch-ledger remainder,
  mmq upstream report, gfx1100/1200 validation).
