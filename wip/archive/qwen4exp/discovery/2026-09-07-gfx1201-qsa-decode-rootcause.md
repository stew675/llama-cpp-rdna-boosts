# 2026-09-07 — gfx1201 (RDNA4): WHY dense decode surpasses QSA at depth — root cause

Phase 2.1 follow-up (maintainer direction 2026-09-07: QSA was developed on gfx1201 and
showed strong gains there; the dense path beating it at depth is unexpected and points at
gfx1201-side inefficiencies in the sparse decode path, not at QSA's premise).  soar 3x
R9700, Qwen3.8-Flash-Next UD-IQ4_XS, tensor-split bf16, UNPINNED, interleaved r1.

## 1. Fall-off curves (tg128; sparse = QSA default ON, dense = LLAMA_QSA_OFF=1)

| depth | sparse | dense | dense adv |
|---|---|---|---|
| d0 | 50.6 | 51.0 | ~flat |
| d2048 | 44.6 | 49.6 | +11.3% |
| d4096 | 44.4 | 49.6 | +11.7% |
| d8192 | 43.8 | 49.4 | +12.8% |
| d12288 | 43.0 | (49.0 int.) | ~+14% |
| d16384 | 42.2 | 48.7 | +15.5% |
| d24576 | 40.8 | 48.0 | +17.7% |
| d32768 | 39.4 | 47.5 | +19.6% |
| d65536 | 35.1 | 45.3 | +29% |

Sparse falls -31% d0->64K; dense only -11%.  Dense wins decode from ~2K context (NOT 32K).
Prefill inverts: sparse +14.5% @32K / +48% @64K (batched path amortizes the indexer ops);
crossover between 16K-32K ctx.  Selection width n_top_k caps at 2051 (indexer_top_k + r - 1);
at 32K sparse FA reads only 2051 of ~32K cells, yet still loses.

## 2. Isolation experiments @d32768 (tg128) — where the cost is

| config | t/s | read |
|---|---|---|
| QSA_OFF (dense FA over all cells) | 47.5 | baseline |
| QSA sparse-FA (default) | 39.4 | build + sparse kernel |
| QSA top-k build + MASKED-DENSE FA (`LLAMA_QSA_SPARSE_FA=0`) | 38.1 | build + dense kernel |
| QSA sparse-FA, forced 1 slice (`GGML_CUDA_QSA_SLICES=1`) | 32.4 | slicing hurts less |

The attention kernel is NOT the problem: sparse-FA == masked-dense-FA with the same build
(39.4 vs 38.1).  The ~9 t/s vs dense (47.5) is the **top-k build + mask assembly**
(~4.3 ms/token over ~60 layers ~= 70 us/layer).  Slicing the sparse decode list already
helps ~22% (32.4 -> 39.4).  The gap is ~fixed per token (d2K is already -11%, at the width
boundary with ~everything selected), with a secondary n_kv-scaled component 8K->32K
(~1.5-2 t/s while n_top_k stays capped).

## 3. Op census (decode, per layer above the selection width, n_tps=1)

The sparse path runs ~15 ggml ops/layer where dense FA is ~1-2:
build_qsa_top_k: index_k_proj mm -> cpy_k (indexer-key store) -> get_rows(ALL cached raw
keys, [idx_dim x n_kv]) -> r-slice pooling (r x cont+add) -> scale -> rms_norm(n_blocks) ->
rope_multi(n_blocks) -> index_q_proj mm -> q norm -> q rope -> score mul_mat (O(n_blocks))
-> relu -> n_idx_h adds -> bias add -> ggml_indexer_top_k -> then the mask: fill(-INF over
n_kv) -> view/zeros -> set_rows(n_top_k) -> add -> sparse FA (or masked dense FA).

Every decode token RE-POOLS the whole indexer cache from RAW keys (the cache stores raw
keys only - pooling precedes norm/rotation by design for B parity), i.e. the pooled+normed+
rotated per-block representation is recomputed every step over ~12 kernels, vs the dense
path's single fused read.  Prefill amortizes this across the ubatch; n_tps=1 decode does
not.  The fixed per-layer launch stack (not the n_kv-scaled work) dominates the gap -
consistent with d2K already losing (width boundary) and the 8K->32K scaling being small.

## 4. Why dense pulls ahead (history)

QSA's per-token indexer build predates and never got the fusion treatment.  Dense decode
kept improving (upstream flash-attn + the campaign's gfx1151 FA config work + the
ws-fusions), widening the gap.  The "QSA decode fix" (pre-re-base era) restored
correctness/coherence on gfx1201 - it did not restructure the per-token indexer path.
`lightning-indexer` (the fused score kernel in ggml-cuda) is NVIDIA-WMMA-only
("TODO add support for AMD cards via rocWMMA"); AMD takes the vec fallback, and the
qwen4exp decode graph still runs the per-op stack.

## 5. Fix directions (prioritized - for maintainer go/no-go)

A. **Fuse the per-token decode indexer path** (the real fix): one HIP kernel per layer
   that reads the raw cached indexer keys + pools + normalizes + rotates + projects the
   query + scores + selects in a single pass (~12 ops/layer -> ~1-2).  Restores QSA's
   flat-fall-off premise at decode.  Consideration: B-parity/coherence of the scoring
   math (identical arithmetic, fused - measure same-seed vs per-op).
B. **Fold the mask into the sparse kernel**: drop the per-layer materialized [n_kv] mask
   (fill -INF + set_rows + add = 3 O(n_kv)/O(n_top_k) kernels/layer) - pass top_k + a
   per-slice visibility so the sparse FA skips unselected cells natively.
C. **Sparse-FA decode geometry for RDNA4**: inherit the dense decode geometry lessons
   (heads/block, slice size, smem, VDR) - the (1-warp-per-head, 64-cell-slice) decode
   config is crude vs the tuned dense decode kernel.
D. Store-side fusion at shortcut depths (index_k_proj + cpy into the KV write path) -
   minor (shortcut depths are ~flat today at d0).

Expected: A+B recover most of the ~9 t/s at d32K (sparse decode -> ~dense parity at
depth), keeping the prefill wins + the FA-read savings that matter at 128K+ contexts.
First implementation target: A as an env-gated probe (adopt-only-if-wins pattern), then
coherence + Protocol A gates.

## 6. Probe result (GGML_CUDA_QSA_DECODE_SKIP, landed fork e6b7ae6f0, env off)

Skipping every (N+1)-th SPARSE layer (dense attend + indexer store instead of the full
selection build) at d32768 tg128, tensor bf16, interleaved:

| sparse layers kept | tg128 @d32K |
|---|---|
| all (skip=0) | 39.7 |
| half (skip=1) | 42.6 |
| quarter (skip=3) | 44.1 |
| none (dense; QSA_OFF) | 47.5 |

→ the indexer build cost is ~per-layer additive: ~7.8 t/s total @32K spread uniformly
over the sparse layers (NOT one dominating op, NOT purely context-scaled).  Fusion
ceiling: sparse decode ≈ dense parity at depth.  Earlier probe runs that 'skipped odd
layers' and measured ~47 immediately were a mistake: the odd layers are the ratio=0
DENSE layers, so skip=1/3 had removed all sparse layers (pure-dense = the 47.0 control).
The counter-based probe (sparse layers only) gives the true gradient above.

## 7. Fusion design (next implementation - env-gated probe pattern)

New fused op replacing the per-layer decode chain pool->score (~11 ggml ops/layer:
get_rows + r-slice pooling adds + scale + rms_norm + rope_multi + index_q mm + q norm +
q rope + score mm + relu + head-sum adds + bias) with ONE kernel per layer:
per-block thread-groups read the r raw cache cells -> mean pool -> rms-norm (layer
index_k_norm) -> rope-rotate (mrope with the block position; replicate the ggml rope
kernel math exactly for B-parity) -> dot vs the projected/normed/rotated q -> relu +
head-sum -> + bias -> per-block score vector.  Keep ggml_indexer_top_k (already fused)
as the next op.  Decode layers/layer count then ~5-6 ops (score + topk + the q-side
projections + FA) vs ~17.  Mask-side (build_attn_qsa fill/set_rows/add) may be droppable
at decode entirely IF decode's causal mask is uniformly-visible for selected cells
(selected cells are all <= the current position) - verify before relying on it.
Expected @32K: recover most of the ~7.8 t/s -> sparse decode ~45-47 at depth, with the
sparse FA read-savings intact for 128K+ contexts where the dense FA becomes read-bound.
Cross-arch: same fused kernel on gfx1151 (halo) - measure both; the op is arch-neutral
so it likely needs no gating (B-parity = byte-identical scores vs the per-op chain on
both arches).

Raw logs: `wip/qwen4exp/gfx1201/runs/` (2.1 toggles, d2K-d64K fall-off interleaves,
qsa-isolation runs).
