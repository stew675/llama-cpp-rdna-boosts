# Arch-independent memory findings (from the qwen4exp QSA campaign)

Scope: what generalizes **beyond** Qwen3.8-Flash-Next, with the measured evidence for each item.
Nothing here is in the delivery patch set yet; the qwen4exp-specific work lives in
`../qwen4exp/qsa-memory/` (`L1-step1-derived-block-bias-findings.md`, `L2-score-chain-findings.md`).
Everything below was measured on 3x R9700 (gfx1201), ctx 204800, `-ctk/-ctv q8_0`, `-sm tensor -mg 0`,
one GPU job at a time.

## 1. The kq mask is the biggest generic lever - and it costs twice

`llama_context` reserves the mask in the **compute buffer** (per GPU) *and* in the **ROCm_Host
buffer** (once per box), because `set_input_kq_mask` fills it from the CPU and the scheduler then
mirrors it device-side. Both drop by the same amount when the mask goes away - measured on
qwen4exp: -800.00 MiB compute *and* -800.00 MiB host at ub2048 (ub1024: -399.22/-399.22,
ub512: -299.02/-299.02).

Dense control, Qwen3.5-4B-Q8_0 (`tools/model-sweep.sh`, load only, no QSA anywhere in the model):

| ub | compute buffer | host buffer | mask in the peak live set |
|---|---|---|---|
| 2048 | 1800.33 MiB | 840.34 MiB | **800.00 MiB (44% of compute)** |
| 1024 | 1300.30 MiB | 420.30 MiB | ~400 MiB |
| 512 | 1050.28 MiB | 210.28 MiB | ~200 MiB |

Peak live set at ub2048 (from the instrumented-allocator ledger): `attn_inp_kq_mask` 800.0 MB +
`Qcur_full` 64 MB + `FLASH_ATTN` 32 MB + `MUL_MAT` 32 MB + `ffn_out` 20 MB = 948 MB live against a
1800 MB buffer. **The mask alone is ~1.6 GiB of the 2.64 GiB (compute + host) = ~61%** of what a
dense model reserves at this shape.

Scaling law: `mask_bytes = n_kv * n_tps * 2 (F16) * 2 (mirror)`, with `n_tps = ubatch / n_stream`.
So the mask is what makes the ub2048-vs-1024 memory gap so brutal on *every* arch (500 MiB per
1024 ubatch of compute-buffer delta on the 4B, of which 400 is the mask).

## 2. Three ways to reclaim it, in increasing difficulty (all arch-independent)

1. **Device-side fill** (keep the packed mask, drop the host half): fill the mask with a GPU kernel
   instead of the per-ubatch host loop. ~800 MiB/box at ub2048, no numerics change if the fill is
   bit-identical, no graph change. Smallest job, smallest win.
2. **Derived mask** (drop the tensor entirely - what the qwen4exp L1 work did): the mask is not a
   free-form tensor, it is a compact function of *(cell -> sequence set, position)* x *(token ->
   sequence, position)*. The qwen4exp form publishes two I32 arrays (`cell_vis`, `q_vis`, 800 KB
   total) and the top-k + FA kernels derive visibility in-kernel. To generalize, a facility must
   cover: causal, per-sequence membership (parallel), SWA as a position bound, and alibi from
   positions; it must keep the **packed mask as the fallback** for the CPU backend, non-FA
   attention paths, and any mask variant it cannot derive. That means touching the shared graph
   input classes plus each FA backend - a design project, not a patch, but the payoff is ~1.6 GiB
   per GPU at ub2048 on any large-context model.
3. **Derive the consumers instead of the mask** (the L2/L1 pattern): replace an uploaded dense
   tensor with compact index arrays and derive in-kernel (per-block bias: 400 MiB), chunk a giant
   N-dim intermediate and `concat` (L2m: 1040 MiB), or order a unary op before a reshape (L2a:
   1200 MiB). Each is bit-identical and zero-risk, but each needs a model-local implementation.

## 3. ggml-alloc: two issues worth eliminating (shared code, all archs)

### 3a. A reshaped-view parent is never reused in place (a real, silent memory loss)

Evidence (L2a): `ggml_relu` applied *after* a `ggml_reshape_4d` in the qwen4exp top-k cost the
**whole parent twice**: reserve 6690.40 -> 5490.40 MiB (**-1200 MB = 2 x the 600 MB parent**) just by
moving the relu before the reshape, output bit-identical. The allocator's own log said the reason
out loud: "not reusing parent (reshaped) for node_7848 as (nil) is external".

Code: `ggml_gallocr_is_own` / `ggml_gallocr_is_allocated` plus the reuse test in the "count number
of children and views" pass (`ggml/src/ggml-alloc.c` ~L657: `p_hn->n_children == 1 && p_hn->n_views
== 0`, ~L661: `view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data ==
parent->data`). A view whose `data` is still NULL at planning time cannot satisfy that, so the
parent stays separate from the child.

- **Model-author rule (mechanical, zero-risk):** apply elementwise/unary ops *before* the reshape,
  never after. Audit target for other archs: any `reshape*` followed by relu/gelu/silu/add(scale) on
  a large tensor.
- **General fix:** let the allocator reuse through a reshaped parent when the layouts allow.

### 3b. `n_views` accounting for a `ggml_cpy` into a view (suspected leak)

Evidence: 12 x 400 MB `indexer_score-*` leaves were never freed; the qwen4exp reserve crept
4450 -> **6441 MiB** until the graph was changed to assemble with `ggml_concat` instead of copying
into a view (`ggml_cpy`-into-view is the trap). The workaround is in `L2-score-chain-findings.md`.

Code: the two passes count different things - **increment by view-node existence**
(`ggml-alloc.c` ~L737-741: for every node `ggml_impl_is_view(node) && node->op != GGML_OP_NONE` ->
`view_src->n_views += 1`), **decrement by use as a src** (~L805-812, in the free pass, only for a
parent that is itself a view and has lost its last child). `n_views` is a plain `int`, so a
mismatch makes it negative, and both the free check (`n_views == 0`) and the reuse check
(`n_views == 1`) then fail forever.

**Status: symptom measured and reproducible, mechanism as yet unconfirmed.** The next session should
prove it with the allocator's own tracing - `AT_PRINTF` prints exactly `parent %s: %d children,
%d views, allocated: %d` (enabled via `GGML_ALLOCATOR_DEBUG` in `ggml-alloc.c`) - on a *minimal*
repro (`ggml_cpy` into a view of a big allocator-owned tensor, repeated), then fix the accounting.

**Acceptance criteria for any fix here:** repeated graph builds must not grow the reserve; coherence
byte-identical on the 4B / 27B / qwen4exp matrix; `test-backend-ops` + `test-alloc`/`test-batch-alloc`
clean; the qwen4exp L1 numbers (3251.39 MiB/GPU at ub2048) unchanged or better.

## 4. Measurement tooling (all model-agnostic)

| tool | what it gives |
|---|---|
| `../qwen4exp/qsa-memory/tools/model-sweep.sh <bin> <model> [ub...]` | compute + host reserve per ub (the probe used for §1) |
| `../qwen4exp/qsa-memory/tools/l0a-scheddump.sh` (`BIN=... MODEL=...`) + `tools/peak-ledger.py` | peak live-tensor ledger and attribution (needs a `GGML_ALLOCATOR_DEBUG` build; `/tmp/bin-l0base` is one) |
| `tools/ub-sweep.sh <bin> [ub...]` | pp20480/tg256 performance per ub (llama-bench, r=3) |
| `tools/ab-coherence.sh`, `tools/mtp-ab.sh <tag>` | the correctness gates (same-seed text A/B; adaptive-MTP acceptance) |

Rule of thumb for reading the numbers: report **compute buffer**, **host buffer**, and the **mask's
share** separately - the mask is the only term that is both huge and arch-independent.

## 5. Open items / next session, in order

1. **ggml-alloc (3a + 3b).** Confirm 3b with `AT_PRINTF` on a minimal repro, then fix: through-view
   parent reuse (3a) and/or the `n_views` accounting (3b). Validate against the §3b acceptance
   criteria. This is shared code: a win for every arch.
2. **A dense model ledger.** The Qwen3.8-27B Q8 used in earlier sessions is **no longer on disk**
   (`/models` holds only the Flash-Next set); re-run `model-sweep.sh` + the ledger once available -
   and use it to decide whether the derived-mask facility (§2.2) is worth building.
3. **qwen4exp, still open:** the score chain's ~700 MB concat peak and the host-side top-k build
   (`../qwen4exp/qsa-memory/L1-step1-derived-block-bias-findings.md` §2d).
4. **Latent, ~30 min:** `llm_graph_input_attn_k::set_input` (`src/llama-graph.cpp` ~L509) has the
   same unguarded `set_input_kq_mask` shape that blocked the prune; today it is unreachable, but it
   is the same trap (§2d of the L1 findings).
5. **Packaging decision (maintainer):** the L1 mask prune (10 files, +552/-79) and the keys-only
   indexer patch are validated but still WIP - decide whether either becomes a delivery block
   (regeneration via `scripts/make-patches.sh`, dated validation record, clean-apply simulation).
