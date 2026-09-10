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
1800 MB buffer (The buffer high-water is cumulative across allocation records, so it exceeds any single record's live sum - read the mask's share from the two reserve numbers above, not from this sum.). **The mask alone is ~1.6 GiB of the 2.64 GiB (compute + host) = ~61%** of what a
dense model reserves at this shape.

Dense control #2, **Qwen3.8-27B-Q8_0** (`/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf`,
`n_swa = 0` so a single mask, same flags, weights 8678 MiB/GPU). Note: this GGUF carries its **MTP
heads inline**, so `--spec-type draft-mtp` runs on the model alone - no separate `-md` draft file
(unlike qwen4exp, whose draft head is `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`).

| ub | compute buffer | host buffer | mask share |
|---|---|---|---|
| 2048 | 1920.33 MiB | 880.34 MiB | **800 MB = 42% of compute, ~57% of compute+host** |
| 1024 | 1360.30 MiB | 440.30 MiB | ~400 MB |
| 512 | 1080.28 MiB | 220.28 MiB | ~200 MB |

Per-1024-ubatch compute delta: 560 MiB, of which 400 is the mask. Peak live set at ub2048: mask
800 MB + `Qcur_full` 96 + `FLASH_ATTN` 48 + `MUL_MAT` 48 + `ffn_out` 40.

Scaling law: `mask_bytes = n_kv * n_tps * 2 (F16) * 2 (mirror)`, with `n_tps = ubatch / n_stream`.
So the mask is what makes the ub2048-vs-1024 memory gap so brutal on *every* arch (500 MiB per
1024 ubatch of compute-buffer delta on the 4B, of which 400 is the mask). Verified exactly at three
contexts with the new `tools/mask-scaling.sh` (`+57344` ctx -> `+2 x 224` MiB compute, `+224` MiB host
= 4.096 KiB per ctx token per copy, where 4.096 KiB = `n_ubatch * 2 B`).

**There is a second, undocumented ctx-linear consumer of the same size.** With a *quantized* KV cache
the CUDA/HIP FA path materializes an F16 conversion of the whole K and V inside the compute buffer
(`ggml_cuda_flash_attn_ext_get_alloc_size()` / `..._get_f16_extra_data()`,
`ggml/src/ggml-cuda/fattn.cu:694`, `fattn-common.cuh`) - one transient per attention layer, measured
**832 MiB** at ctx 204800 / ub 2048 / q8_0 (L0d instrumentation: `x16 832.00 MB node_* [Meta()]`),
and **absent** with `-ctk/-ctv f16` (compute reserve 1800.33 -> 1056.06 MiB). The F16 KV cache that
would avoid it costs +1031 MiB of cache (1185 -> 2216), so q8_0 + scratch still wins by 287 MiB - but
the scratch is real. Net: at this shape the compute buffer's ctx-linear part is mask (4.096 KiB/ctx
token) + FA scratch (4.06 KiB/token) = **8.0 MiB per 1k context tokens**, against the 27B's 11.7 MiB/1k
KV - i.e. **the real VRAM cost of context is ~1.7x what the KV cache suggests** (and ~1.4x for the
4B). Also note the arena high-water legitimately exceeds the ledger's live-tensor sum: the arena is
sized by `ggml_backend_buft_get_alloc_size()` (the *request*), while the ledger prints
`ggml_nbytes()`. Full design brief for reclaiming both: `DERIVED-MASK-DESIGN.md`.

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
   per GPU at ub2048 on any large-context model. **Design brief written: `DERIVED-MASK-DESIGN.md`**
   (derivability taxonomy, per-backend inventory, the V1/V2/V3/V4 ladder with measured cost/benefit,
   the L1 worked example, acceptance criteria).
3. **Derive the consumers instead of the mask** (the L2/L1 pattern): replace an uploaded dense
   tensor with compact index arrays and derive in-kernel (per-block bias: 400 MiB), chunk a giant
   N-dim intermediate and `concat` (L2m: 1040 MiB), or order a unary op before a reshape (L2a:
   1200 MiB). Each is bit-identical and zero-risk, but each needs a model-local implementation.

## 3. ggml-alloc: two issues worth eliminating (shared code, all archs)

### 3a. A reshaped-view parent is never reused in place (transient - measured NOT to move the peak)

Evidence (L2a): `ggml_relu` applied *after* a `ggml_reshape_4d` in the qwen4exp top-k cost the
**whole parent twice**: reserve 6690.40 -> 5490.40 MiB (**-1200 MB = 2 x the 600 MB parent**) just by
moving the relu before the reshape, output bit-identical. The allocator's own log said the reason
out loud: "not reusing parent (reshaped) for node_7848 as (nil) is external".

Code: `ggml_gallocr_is_own` / `ggml_gallocr_is_allocated` plus the reuse test in the "count number
of children and views" pass (`ggml/src/ggml-alloc.c` ~L657: `p_hn->n_children == 1 && p_hn->n_views
== 0`, ~L661: `view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data ==
parent->data`). A view whose `data` is still NULL at planning time cannot satisfy that, so the
parent stays separate from the child.

- **How systemic is it, and does it cost reserve? (measured with the L0c counter)** The
  "not reusing parent ... is external" branch was extended to count the *true* missed through-view
  reuses (view source allocator-owned, last use, same layout, aliasing view):
  - **27B** prefill graph: 1296 "external" events (336 with "(reshaped)" in the tensor name), of which
    **816 are true missed reuses**, *potential* total 14.9 GiB - and **none of them is live at the peak
    record**, i.e. the reserve would not shrink by a single MiB (the losses are per-layer transients,
    <= 48 MiB each; the peak is the 800 MB mask);
  - **4B**: 648 events -> 408 true misses, 4.7 GiB potential, also **0 live at the peak**.
  => **3a is a correctness/generality fix, not a reserve win for dense models.** It *did* win 1200 MB on
  qwen4exp (L2a) precisely because there the reshaped-parent chain was itself the peak setter.
  Re-check both numbers after any allocator change with `../qwen4exp/qsa-memory/tools/view-reuse-ledger.py` (L0c build:
  `/tmp/bin-l0c`; counter source `/tmp/ggml-alloc.instrumented.c`).
- **Model-author rule (mechanical, zero-risk):** apply elementwise/unary ops *before* the reshape,
  never after. Audit target for other archs: any `reshape*` followed by relu/gelu/silu/add(scale) on
  a large tensor.
- **General fix:** let the allocator reuse through a reshaped parent when the layouts allow.

### 3b. `n_views` accounting for a `ggml_cpy` into a view - **mechanism proven, fix written (WIP)**

**Root cause (proven 2026-09-10).** The counting pass increments `view_src->n_views` for *every* view
**node** in the graph (`ggml/src/ggml-alloc.c` ~L737-741), but the free pass only decrements it inside
the block that runs when the view node *itself is released* (~L805-812) - and a view node with **no
consumers** is never released. A `ggml_cpy` expanded into the graph purely for its side effect (the
"copy into a view of preallocated memory" idiom) is exactly such a node, so its view source's count
stays inflated forever, which blocks **both** the release and the in-place reuse
(`p_hn->n_views == 0`) of the view source. The space is never reused, so the buffer grows by the size
of every such view source - which is the qwen4exp L2 observation (12 x 400 MB `indexer_score-*`
never freed, reserve creeping 4450 -> 6441 MiB).

**Repro** (`repro/ggml-alloc-unused-view.c`, 12 layers x 4 MiB, one graph): the idiom gives a
**56.00 MiB** arena vs **16.00 MiB** for the identical graph with the copies consumed (control), and
with `GGML_ALLOCATOR_DEBUG` the free pass shows 1 free vs 13 and `view_src ...: 4 views` never
returning to 0. With the fix both variants are 16.00 MiB.

**Fix** (`patches/0001-ggml-alloc-release-unused-view-sources.patch`, +40 lines): after the counting
pass, fold the view contribution of every view node that has **no consumers** and is not a graph
output (outputs are never freed and may alias the view source, which must stay allocated for the
application to read it):

```c
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (!ggml_impl_is_view(node) || node->op == GGML_OP_NONE) continue;
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT)              continue;
        if (ggml_gallocr_hash_get(galloc, node)->n_children != 0) continue;
        ... view_src_hn->n_views -= 1; free the view source if it reaches 0/0 ...
    }
```

It is sound because such a node is a dependency of nothing; the *views used as a source* (e.g. the
cpy's destination view) keep their own contributions, so the source still cannot be reused before the
last operation that touches it has run. Any node with `n_children == 0` has had all its consumers
processed by the time the loop reaches it (topological order), so the fold cannot race the free pass.

**Acceptance (all passed, build `/tmp/bin-l3b` vs the pre-fix `/tmp/bin-l1`):**

| check | result |
|---|---|
| repro: idiom vs control | 56.00 -> **16.00 MiB** = control |
| same-seed coherence (4B / 27B / qwen4exp) | **byte-identical** (only llama-cli's timing footer differs) |
| reserve 4B / 27B @ ub2048 | 1800.33 / 840.34 and 1920.33 / 880.34 - **unchanged** |
| qwen4exp reserve ub2048 / ub1024 | 3251.39 / 63.69 and 1675.33 / 33.64 - **unchanged** |
| adaptive MTP acceptance | **0.61616** in both builds (gate >= 0.45) |
| `test-alloc`, `test-batch-alloc` | PASSED / 0 failures |
| `test-backend-ops -o {VIEW,CONT,CPY,DUP,CONCAT}` (CPU, ROCm0) | all OK |

**Value:** the current graph builders do not use the idiom (hence the unchanged numbers), so this is
a **latent trap** fix - it is upstream-shared code, it removes a real leak for any graph that does use
it, and it un-blocks the `ggml_cpy`-into-view assembly that the L2 work had to abandon in favour of
`ggml_concat` (bit-exactness of the concat path keeps it the recommended choice anyway).

**Not in the delivery**: kept as a WIP patch (like everything under `wip/`), to be packaged only on
request. The fork tree is left in the 10-file L1 state.

## 4. Measurement tooling (all model-agnostic)

| tool | what it gives |
|---|---|
| `../qwen4exp/qsa-memory/tools/model-sweep.sh <bin> <model> [ub...]` | compute + host reserve per ub (the probe used for §1) |
| `../qwen4exp/qsa-memory/tools/l0a-scheddump.sh` (`BIN=... MODEL=...`) + `tools/peak-ledger.py` | peak live-tensor ledger and attribution (needs a `GGML_ALLOCATOR_DEBUG` build; `/tmp/bin-l0base` is one) |
| `tools/ub-sweep.sh <bin> [ub...]` | pp20480/tg256 performance per ub (llama-bench, r=3) |
| `tools/ab-coherence.sh`, `tools/mtp-ab.sh <tag>` | the correctness gates (same-seed text A/B; adaptive-MTP acceptance) |
| `../qwen4exp/qsa-memory/tools/mask-scaling.sh <bin> <model> <ub> <ctx...>` | the kq mask's cost vs context (isolates the only term linear in n_ctx) |
| `../qwen4exp/qsa-memory/tools/view-reuse-ledger.py <sched.log...>` | the missed through-view reuse counter (3a) per model |

Rule of thumb for reading the numbers: report **compute buffer**, **host buffer**, and the **mask's
share** separately - the mask is the only term that is both huge and arch-independent.

## 5. Open items / next session, in order

1. ~~**The mask - design brief**~~ **DONE 2026-09-10 -> `DERIVED-MASK-DESIGN.md`** (which corrects
   the accounting: the compute buffer's other 832 MiB is the FA F16 K/V conversion scratch, not the
   mask - see §1). Next: implement **V3 (derived mask), HIP/CUDA-only, gated**, starting from the
   single-sequence prefill case (`cell_pos <= token_pos`), using the L1 patch as the template; **V2**
   (1-bit packed mask) is the semantics-free alternative; **V4** (native quantized K/V in the MMA FA
   path) is an independent, equal-sized win that also removes a per-ubatch conversion.
2. ~~**ggml-alloc 3b (the leak)**~~ **DONE 2026-09-10 - mechanism proven, fix written + validated**
   (`patches/0001-ggml-alloc-release-unused-view-sources.patch`, +40 lines, all acceptance criteria
   passed; see §3b). Not folded into the delivery (WIP rule); the fork tree stays the 10-file L1
   state. Remaining option: propose it upstream - it is shared code and fixes a real leak.
3. **ggml-alloc 3a (through-view reuse)** - correctness/generality only: measured **zero** reserve win
   on the 27B/4B (§3a). Do it if it is cheap and provably safe (no numerics change); do not spend a
   session on it for dense models.
4. **qwen4exp, still open:** the score chain's ~700 MB concat peak and the host-side top-k build
   (`../qwen4exp/qsa-memory/L1-step1-derived-block-bias-findings.md` §2d).
5. **Latent, ~30 min:** `llm_graph_input_attn_k::set_input` (`src/llama-graph.cpp` ~L509) has the same
   unguarded `set_input_kq_mask` shape that blocked the prune.
6. **Packaging decision (maintainer):** whether the L1 mask prune (10 files, +552/-79) and the
   keys-only indexer patch become a delivery block, or stay WIP.
