# Block 0015 handover — campaign wins → beta delivery (+ upstream candidates)

**Rewritten 2026-09-10 (evening) after the maintainer's decisions.**  Read in this order:

1. this file (decisions, the critical path, the plan, the state),
2. `README.md` in this directory (inventory, gate audit, validation protocol, reference numbers),
3. `../../AGENTS.md` (repo rules — they override everything here),
4. `../../upstream/README.md` (the maintainer's PR backlog; entries must be self-sufficient),
5. `../../wip/arch-independent-memory/DERIVED-MASK-DESIGN.md` (**the spec for V3/V4 — the critical path**).

> **Bottom line for the next session: implement V3 (derived kq mask), then V4 (native quantized K/V in
> the MMA FA path).  Block 15's beta window does not open until they land.**  See §3.  Everything else
> (Stage A upstream candidates, the W1–W4 merge/gate/validate) is either independent or waits behind
> them.

## 0. Decisions taken by the maintainer

| # | decision |
|---|---|
| D1 | `upstream/README.md` is the maintainer's own PR backlog.  Entries only need to be **self-sufficient enough for a fresh session to regenerate/pick them up later** — ggml-org permits one PR at a time and one has been pending for four weeks, so a backlog is expected.  No status machinery. |
| D2 | **W4 gate: OPTION C — decided.**  No env knob in the code (it is a bug fix, not a policy); instead a ready-made revert patch for A/B: `ab/w4-revert.patch` (verified round trip).  See §2. |
| D3 | **Kill-switches for W1 (`GGML_QSA_SCORE_MEM`) and W3 (`LLAMA_QSA_KEYS_ONLY`)** — approved; W2 keeps its existing gates, minus the `GGML_QSA_DERIVED_BIAS=2\|3` diagnostics. |
| D4 | Merging and validating the wins **as a combined set** is a full session of work and the campaign's centrepiece. |
| D5 | **Exactly one block: Block 0015.  No Block 16.**  Later campaign wins are amended into Block 15 as dated amendments (the block-13/14 practice).  **REVISED:** Block 15 waits until **V3 and V4 land** — they are included in the block from the start, not amended in later. |
| D6 | Prepare the two extra upstream candidates (keys-only dead-V removal, `attn_k` null-mask guard) in the `upstream/` style (§4). |
| D7 | **V3 includes phase 3.2 (SWA coverage)** — wider coverage is required precisely so that *other* models do not regress; the mask work must not leave SWA models on a different (or unvalidated) path. |
| D8 | **Beta tester material: yes** — `BETA-TESTING.md` in this directory (one-page A/B checklist: gate table, the three measurements, the report template, what not to report). |
| D9 | **V4 ship rule (three-way)**: no prefill-throughput regression → on by default; regression → ship it **opt-in (default off)** for people who need the last 832 MiB, with the trade-off documented; if even that is impractical → future work. |

## 1. Where the campaign stands

**Validated wins** (under `wip/`, nothing in `patches/` yet):

| win | what | source | measured effect (qwen4exp, ctx 204800, ub 2048, q8_0 KV, 3× R9700) |
|---|---|---|---|
| **W1** | L2 score-chain memory: relu before the 4-D reshape + `n_blocks`-chunked assembly with `ggml_concat` | `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch` (1 file) | compute **6690.40 → 4450.40 MiB/GPU** (ub1024 3346.50 → 2274.35, ub512 1724.56 → 1188.56); bit-identical |
| **W2** | L1 derived QSA block bias + derived visibility, the **mask prune**, and the input-fill null guards (incl. the `attn_k` one) | `wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch` (10 files, +557/−80, base = W1) | compute **4450.40 → 3251.39**, host **1262.70 → 63.69** MiB (ub1024 1675.33/33.64); coherence byte-identical; MTP 0.61616 |
| **W3** | keys-only QSA indexer cache (the indexer V buffer is never read) | `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch` (3 files, 62 lines) | indexer KV 956.25 → **318.75 MiB**; box 88.58 → **86.70 GiB**; perf parity; broad validation matrix |
| **W4** | ggml-alloc: release view sources whose views are never consumed (the 3b leak) | `wip/arch-independent-memory/patches/0001-ggml-alloc-release-unused-view-sources.patch` (+35, `ggml-alloc.c`); PR copy in `upstream/` | repro 56.00 → **16.00 MiB**; no change on current models (latent trap); **upstream-applicable, clean on master `9cf3bf256`** |

**V3 is done** (2026-09-10, phases 1+2a+2b+2c) and is the fifth validated win - on by default in the
fork tree, `LLAMA_KQ_MASK_DERIVED=0` forces the packed mask.  Measured: compute **−799.20 MiB/GPU** and
host **−799.21 MiB** on the 4B (1800.33 → 1001.13 / 840.34 → 41.13) and the 27B (1920.33 → 1121.13 /
880.34 → 81.13), **−809.18/−809.18** on gemma-4-E4B (ISWA, both masks) and **−811.17/−811.18** on
gemma-4-31B, scaling exactly as `n_kv × n_tps × 2 B`; generated text **byte-identical** on the 4B/27B/
gemma-4-E4B (3k and 40k prompts), 27B MTP acceptance identical (0.76744), qwen4exp unchanged; cost
prefill −1.1 %, decode −0.7 %.  Record: `../../wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md`
§4.3-§4.5.

**Still to do before Block 15 can be cut** (the critical path, §3):

| id | what | expected effect (dense models, ctx 204800, ub 2048, q8_0 KV) |
|---|---|---|
| **V3** | derived kq mask for the plain (non-QSA) attention path - **DONE 2026-09-10**, see above | **−799 MiB/GPU VRAM − 799 MiB host** measured |
| **V4** | native quantized K/V in the FA path: dequantize into the shared K/V tiles instead of staging an F16 copy of the whole cache in a global scratch.  **Both the MMA loader and the TILE loaders must be fixed** (see §3.2: TILE, not MMA, may be what the reservation is sized for) | **−832 MiB/GPU at ub2048, exactly ctx-linear** (the `ggml_cuda_flash_attn_ext_get_f16_extra_data` scratch) |

**Not pursued**: 3a (through-view reuse — measured **zero** reserve win on the 27B/4B; the brief records
it as a correctness/generality item only), V2 (1-bit packed mask — the *fallback plan* for V3, see
§3.1), and everything in `archive/work/`.

## 2. W4 — DECIDED (option C): no knob, a revert patch for A/B

Rationale (agreed): W4 is a bug fix in shared ggml code, not a policy — "off" means running the leak —
and it is bound for upstream, where an env var that re-enables a leak is exactly what a reviewer would
reject.  It is also a measured no-op on every model we have, so it only matters if some other graph
hits the idiom.

A/B procedure for a beta tester (no code knob, one command):

```bash
cd <the block-15 tree>
git apply -R beta/block-15-campaign-wins/ab/w4-revert.patch    # or: git apply <the file> (it is a
                                                               # forward patch that removes the fix)
# or, equivalently, with the upstream copy:
git apply -R upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch
```

Verify: applying it must leave `ggml/src/ggml-alloc.c` byte-identical to upstream, and the repro
(`wip/arch-independent-memory/repro/ggml-alloc-unused-view.c`) must go back to 56.00 MiB.
`ab/w4-revert.patch` was generated with `git diff -R` and verified: apply W4 → 1 file/+35 → apply the
revert → 0 modified files.

## 3. The critical path — V3 first, then V4

Both are specced in `wip/arch-independent-memory/DERIVED-MASK-DESIGN.md` (§2 options, §3 consumers, §5
the facility shape, §6 cost/benefit, §7 the qwen4exp L1 worked example).  Both must be **on by default
with an env kill-switch**, **bit-identical** to the current behaviour, and **not perf-regressing** —
that last criterion is a ship gate: a memory win that costs throughput does not go into Block 15, it
gets documented as future work.

Honest sizing: V3 is roughly 1–3 sessions (phase 3.1 is the win; 3.2/3.3 add coverage), V4 1–2 sessions
plus perf tuning.

### 3.1 V3 — derived kq mask (the mask's 800 MiB + 800 MiB host)

> **PHASE 1 IS DONE (2026-09-10): the predicate is proven bit-exact on the host.**  A diagnostic
> (env `LLAMA_KQ_MASK_DERIVED_VERIFY=1`) recomputes the mask from the derived form and compares it
> cell-by-cell with the packed fill; every record is `mismatches: core=0 ext=0` on: the 27B dense
> prefill (21 ubatches, `n_tps=2048`, `n_kv` 2304 -> 39680), **gemma-4-E4B ISWA (both the base and
> the SWA cache, `n_swa=512`)**, the **non-causal** SWA path (`--attention non-causal` via
> llama-embedding), the **F32** mask (`-fa 0`), and small verify batches.  See
> `../../wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md` (the spec) +
> `wip/arch-independent-memory/patches/0002-DIAGNOSTIC-verify-derived-kq-mask-predicate.patch`
> (the oracle) + `wip/arch-independent-memory/logs/`.  Two findings refine the plan below:
> **`qwen35` is IMROPE, so `is_pos_2d()` is true for text** (the M-RoPE clause is live but provably
> a no-op for degenerate positions -> a degeneracy guard is the fallback), and **every SWA variant
> reduces to a per-token visibility floor `lo`** (`STANDARD`: `p1-n_swa+1`; `CHUNKED`: the chunk
> start; `SWA_FULL`: `min(lo, seq_pos_min)`), so 3.2 costs nothing beyond that host-side floor.

> **PHASE 2C IS DONE (2026-09-10) — V3 IS COMPLETE AND ON BY DEFAULT.**  The graph plumbing (a new
> `llm_graph_input_kq_derived`, `build_attn_inp_kq_mask` as a context method with an `allow_derived`
> flag, the substitution in `build_attn_mha`), the backend probe (`LLM_FUSED_OP_FLASH_ATTN_DERIVED`,
> requiring a CUDA/HIP-or-Meta GPU device and that the probe graph really contains the derived node)
> and the `LLAMA_KQ_MASK_DERIVED` gate landed as `patches/0005-*` (6 files, +487/-45).  The key design
> decision: **the packed mask tensor is still created in every graph** - it just ends up with no
> consumer on the derived path, so the gallocr leaves it unallocated and the two existing
> `buffer != nullptr` fill guards skip the per-ubatch fill.  Any other consumer (deepseek4's bias
> concat, minimax-m3's msa_kqm, qwen4exp's indexer) therefore keeps the mask materialized
> automatically - there is no model-level allowlist to keep in sync and no way to mis-serve a
> consumer.  Measured (ctx 204800 / ub 2048 / q8_0, one binary flipping the gate): **-799.20 MiB/GPU
> and -799.21 MiB host** on the 4B and the 27B (3-GPU Meta), **-809/-809** on gemma-4-E4B (ISWA, both
> masks), **-811/-811** on gemma-4-31B, scaling exactly as `n_kv x n_tps x 2 B` (ub 1024 -399/-399,
> ub 512 -199/-199); coherence **byte-identical** on the 4B/27B/gemma-4-E4B (3k and 40k prompts); the
> 27B MTP acceptance **identical at 0.76744**; qwen4exp unchanged (its probe correctly reports the
> derived path is not used there).  Cost: prefill **-1.1 %** (pp20480/ub 2048, interleaved 5x) and
> decode -0.7 % - the reserve is 799 MiB smaller, so against the delivered ub 1024 baseline the
> derived ub 2048 is equal speed for 300 MiB less per GPU.  One crash was found and fixed
> (`params.mctx` is the *memory* context - hybrid for most models in this fork - so the reuse check
> must `dynamic_cast`, not `static_cast`).  Full record + the not-exercised list:
> `V3-DERIVED-KQ-MASK-PLAN.md` §4.3-§4.5.

**Scope it in phases; phase 3.1 alone captures the whole memory win for the dense text models.**

* **3.1 (the win)** — derive in the **prefill / MMA** path for the plain cache
  (`llm_graph_input_attn_kv`, `llm_graph_input_attn_k`, and the four standard ISWA/hybrid-iswa
  variants): predicates *cell occupied*, *cell belongs to the token's sequence*, *causal
  (`cell_pos <= tok_hi`)*, *the SWA floor (`cell_pos >= tok_lo`)*.  Keep the **packed mask for
  decode** (`n_tps == 1`, where it costs ~n_kv × 2 B = 400 KiB, i.e. nothing), for **`n_tps <= 8`**
  (small verify batches, where the launcher may pick the vec/tile kernel and the mask is tiny
  anyway), and for every unsupported case — so the kernel change is confined to the MMA path
  (`fattn-mma-f16.cuh`) and the vec/tile kernels are untouched.  Prefill is what sets the reserve,
  so the win is complete without touching the decode path.
* **3.2 (SWA coverage — IN SCOPE for Block 15, D7)** — add **SWA as a position bound**
  (`llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)`, a pure function of the two positions plus
  `n_swa`/`swa_type`; remember the `LLAMA_NON_CAUSAL_TYPE_SWA_FULL` in-span exception, which needs the
  per-sequence minimum batch position) so sliding-window models get the win too.  Rationale: most modern
  models are SWA, and leaving them on a different path is exactly how a regression slips through.
  **Validation models are available locally** (checked 2026-09-10 by the GGUF metadata key):
  `sliding_window` present in `/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf` (small, and there is
  an `…-MTP.gguf` variant for the MTP gate) and
  `/llm/models/Gemma4/31B-QAT/Q4_K_XL/gemma-4-31B-it-qat-Q4_K_XL.gguf`; **absent** in Qwen3.5-4B,
  Qwen3.8-27B and Qwen3.6-35B-A3B (the non-SWA controls).  Note that an SWA model builds **two** masks
  (`self_kq_mask` + `self_kq_mask_swa`, i.e. two derivable sets), which 3.2 must handle.
* **3.3 (optional)** — M-RoPE causality (`p0 == p1 && p0_ext.is_2d_gt(p1_x, p1_y)`) needs the per-cell
  2-D ext positions published as well.  **Alibi stays on the packed mask** (it is a value, not a
  predicate).
* **V2 is the fallback plan**: if the per-backend derivation (3.1) turns out to be more invasive than
  expected, ship the 1-bit packed mask instead — same tensor, 16× smaller, bit-exact by construction,
  no per-cell state — and keep 3.1 as the follow-up.  Decide this early; do not build both.

**Data model** (from the brief §2/§5): per cache cell `kq_cell_state` (I32: bit0 occupied, bits 8–15
the `LLAMA_MAX_SEQ = 8` sequence bitset) + `kq_cell_pos` (I32) — 1.2 MiB at ctx 204800, i.e. 0.25 % of
the mask; per token the existing `attn_inp_pos` I32 input plus a sequence-id array (or reuse
`ubatch->seq_id`).

**Where the code goes** (the ground truth is `llama_kv_cache::set_input_kq_mask` →
`set_input_kq_mask_impl<>`, `src/llama-kv-cache.cpp:1555`–`:1785`; the tensor is built in
`build_attn_inp_kq_mask`, `src/llama-graph.cpp:29`):

1. publish the compact arrays the way the existing `set_input_k_idx/v_idx` inputs do (one set per cache
   instance; the SWA/hybrid caches get their own, and everything not covered keeps the packed mask);
2. extend `ggml_flash_attn_ext` with **optional** srcs (the qwen4exp `ggml_flash_attn_qsa` precedent:
   `ggml.h` + `ggml.c` ctor accepts a **null mask** when the derived arrays are present; the meta
   backend already tolerates null src slots as `SPLIT_AXIS_UNKNOWN`);
3. derive in the kernel at the mask staging sites: `flash_attn_ext_f16_load_mask` in
   `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (two call sites) writes `0` / `-INFINITY` from
   `state`/`pos` instead of loading the global mask — including the "dead column" detector
   (`tile_mask[...] <= -1e30f`) further down, which must see the same values;
4. **fallbacks, decided at graph-build time**: the CPU backend (the reference in
   `ggml/src/ggml-cpu/ops.cpp` aborts with a clear message, as the QSA op does), any non-CUDA/HIP
   backend, `n_swa > 0` before 3.2, alibi, and the MLA/DSA/MSA/ISWA/virtgpu caches.  One shared
   predicate must gate *both* the graph builder and every prune/`can_reuse` site — the L1 lesson: when
   two places could disagree, `LLAMA_QSA_SPARSE_FA=0` read a mask that no longer existed;
5. **guard every input-fill call site** with `if (tensor && tensor->buffer)` — an input the graph does
   not consume is simply not allocated by the gallocr (the L1 lesson again);
6. gate: `LLAMA_KQ_MASK_DERIVED` (default 1 once validated; 0 = always packed) plus the capability
   check; extend `test-backend-ops` with a derived-form FLASH_ATTN_EXT case;
7. **gate the feature on the *backend*, not on the graph**: the mask-vs-derived decision cannot know
   which backend will run the FA node, so use the repo's fused-op probe (`resolve_fused_ops`) and
   make `ggml_cuda_flash_attn_ext_supported()` reject a derived op unless the selected kernel is
   `BEST_FATTN_KERNEL_MMA_F16`.  Keep the mask's *policy* meaning in
   `ggml_cuda_get_best_fattn_kernel` (`gqa_opt_applies` requires a mask and the AMD-WMMA arm depends
   on it) or kernel selection changes silently — use `has_mask = mask || dst->src[5]` for the policy
   tests and `mask` only for the reads.  Details in the plan file section 2.2-2.3.

**Validation** (the L1 protocol, verbatim): one binary that flips the gate, same-seed generated text
**byte-identical** on 4B + 27B at a short and a long prompt, decode + prefill + MTP; the reserve matrix;
`tools/mtp-ab.sh` ≥ 0.45; bench parity (`tools/ub-sweep.sh`, pp20480/tg256 at ub 2048 and 1024);
`test-backend-ops` FLASH_ATTN_EXT on CPU + ROCm0.

### 3.2 V4 — remove the FA F16 K/V staging scratch (the second 832 MiB)

**Mechanism (measured in the source, 2026-09-10):**

1. `ggml_cuda_flash_attn_ext_get_alloc_size` (`ggml/src/ggml-cuda/fattn.cu:694`) computes
   `need_f16_K/V` **per selected kernel** and returns `ggml_nbytes(dst)` plus that scratch — the scratch
   is appended to the FA node's *own* allocation, so it lives in the compute buffer and is sized by the
   reserve graph.
2. `ggml_cuda_flash_attn_ext_get_f16_extra_data` (`ggml/src/ggml-cuda/fattn-common.cuh:56`) lays out
   `{K, V, end}`: pad to 128 B, `+= ggml_nelements(K)*2`, `+= ggml_nelements(V)*2` (V shares K's region
   when V is a view of K).
3. `launch_fattn` (`fattn-common.cuh:976`) is shared by the MMA, TILE and VEC kernels.  When the flags
   are set it runs `ggml_get_to_fp16_cuda(type)` (contiguous) or `ggml_get_to_fp16_nc_cuda` (strided)
   **over the whole cache, every call**, then rewrites `K_data`/`V_data` and `nb11..nb23` to the F16
   layout.  So the price is memory *and* a full-cache conversion pass per ubatch.
4. The MMA staging is `flash_attn_ext_f16_load_tile` (`fattn-mma-f16.cuh:368`): a `half2` loader with
   `cp_async` where available, taking `const half2 * KV` and a `stride_KV` in half2 units.  K is staged
   at 692 / 1029 / 1385 and V at 672 / 1045, plus the fixup and `use_sparse` arms.

**Scoping correction (this section as it stood was too optimistic): V4 has to cover TILE as well as MMA.**

* `need_f16_K/V` is true for **TILE** too (`use_bf16 ? false : K->type != GGML_TYPE_F16`), for every
  quantized KV type.  TILE is what the 2..8-token *verify* batches select (the RDNA4 WMMA arm requires
  `Q->ne[1] > 8`, and VEC is picked only for `ne[1] <= 2` with quantized KV), and the reserve takes the
  max over several reserved shapes (`sched_reserve` reserves the prefill shape → MMA, the n_seqs shape,
  the MTP/verify shapes).  So removing the scratch from MMA alone would *not* shrink the reservation.
* **VEC is already scratch-free** for quantized KV (`need = type == GGML_TYPE_F32` only) — that is why
  q8_0 decode (`n_tps == 1`) never paid for it.  No VEC change is needed for the delivered config.

**First experiment (cheap and decisive — do this before writing any kernel code):** measure the reserve
with q8_0 KV at `-ub 8` (all attention shapes then select VEC/TILE) against `-ub 2048` (MMA).  If the
~832 MiB is present in both, both kernels need the fix and TILE is the *blocking* one (it dominates the
reserve); the deltas also give the exact size to reconcile against
`ggml_nelements(K)*2 + ggml_nelements(V)*2` per layer.  `tools/bufsize.sh` prints exactly this
(remember: it sets `GGML_CUDA_FA_WMMA_256=0`, drop it if that changes the selected kernel).

**Implementation sketch:**

1. The kernels are templated on `DKQ`/`DV`/`ncols`, *not* on the KV type — the scratch exists precisely
   to keep the type out of the kernel.  Adding dequant-on-stage means a runtime type switch in the
   loader (preferred: one new branch, no template explosion; the per-type block layout is
   `ggml_blck_size(type)` / `ggml_type_size(type)`, e.g. q8_0 = 32 elements + F16 scale = 34 B/block, so
   a `DKQ`-wide row is `DKQ/32` blocks).
2. **Consequence to accept up front: `cp_async` is impossible for a quantized K/V source** (the data has
   to be transformed), the 16-byte granularity assumptions of the current loader no longer hold, and the
   staged `tile_KV` (half2) has to be written by hand.  This is where the throughput risk lives.
3. Scope the first increment to **q8_0** (the delivered configuration) and leave `need_f16_K/V` true for
   every other type — then nothing changes for them.  Only then consider the other types
   (`ggml_cuda_fattn_kv_type_supported` / `GGML_CUDA_FA_ALL_QUANTS` gate which exist: q4_0, q4_1, q5_0,
   q5_1, iq4_nl, mxfp4, bf16, ...).
4. Exclude `use_sparse` (NVIDIA-only, its own loader arm) from the first increment: keep the scratch
   there.
5. Preserve the V3 semantics exactly: the staged tile must contain the same values as the packed path
   for `i >= nbatch_fa` / out-of-bounds cells (see `flash_attn_ext_f16_load_mask`'s convention and the
   dead-column detector), and the loaders keep the same `i`/`k` thread mapping so the shared-memory
   layout stays identical.
6. Bonus win: the per-ubatch global conversion pass disappears as well (`n_kv x n_head_kv x DKQ`
   elements per layer per ubatch).

**Validation (D9 three-way ship rule):** same-seed text **byte-identical** with the gate flipped inside
one binary (4B, 27B, gemma-4-E4B as the SWA/ISWA case, qwen4exp as the control); the reserve matrix
(expect **-832 MiB/GPU** at ctx 204800 / ub 2048 with q8_0, and exactly 0 with `-ctk/-ctv f16`); the
adaptive-MTP gate `>= 0.45` and unchanged; bench parity **interleaved** (pp20480 at ub 2048 and 1024,
tg256, 5 passes — the run-to-run noise is ~0.5 %, so a single pass proves nothing); `test-backend-ops
FLASH_ATTN_EXT` on CPU + ROCm0 (the CPU reference is the oracle).  Then: no regression → **on by
default**; regression → **opt-in, default off**, documented in the beta gate table; impractical even as
opt-in → future work.

**Why this is the riskiest item of the campaign:** it touches the hottest kernel in the fork (the WMMA
FA staging), it gives up `cp_async` for quantized KV, and the win is *only* memory.  If the first
increment (MMA or TILE, q8_0) shows more than a low-single-digit prefill cost, take the D9 opt-in route
instead of grinding out per-type variants — Block 15 does not depend on V4 shipping on-by-default.

### 3.3 Then, and only then, Block 15

Block 15 = **W1 + W2 + W3 + W4 + V2-or-V3 + V4** as one patch (D5), merged, gated, combined-validated,
cut, and staged in `beta/` to open the ~4–5 day beta window.  See §5.

## 4. Stage A — the two extra upstream candidates (D6, independent of everything above)

Do these in any session; they do not block Block 15 and Block 15 does not depend on them.  Both follow
the existing `upstream/` convention: `UPSTREAM-PR-<slug>.md` (self-sufficient notes: what, why, root
cause, evidence, verified base, apply-check result, what validation does *not* cover) + `.patch` + one
row in `upstream/README.md`.

### A1. `UPSTREAM-PR-kv-cache-keys-only` — dead indexer V buffer

Adds `bool v_enabled = true` to `llama_kv_cache`'s constructor (no V tensor, no V-side op) and passes
`false` for the qwen4exp QSA indexer store (`src/llama-memory-hybrid-idx.cpp` ~L60).  Upstream evidence
(2026-09-10): the fork patch applies to current master with **0 failed hunks, 1 fuzz**, so upstream has
the same waste.  Work: rebuild the patch against `origin/master` until `git apply --check` is clean,
record the fork's measured numbers (indexer KV 956.25 → 318.75 MiB, −1.9 GiB box) and state plainly
that upstream runtime validation was not run (the fork's CUDA QSA ops are fork-only, so upstream's QSA
path may differ — verify before claiming coherence parity).

### A2. `UPSTREAM-PR-attn-k-null-mask-guard` — latent crash in upstream code

`llm_graph_input_attn_k::set_input` (`src/llama-graph.cpp`) calls `set_input_kq_mask` unguarded while its
own `can_reuse_impl()` accepts a null mask (`self_kq_mask == nullptr || can_reuse_kq_mask(...)`); the
sibling classes all use `if (self_kq_mask && self_kq_mask->buffer)`.  Extract just that hunk from W2
(`patches/0002-derived-qsa-block-bias.patch`), verify `git apply --check` on `origin/master`, write the
notes (inconsistency + crash path + how the fork found it while pruning masks).

## 5. Stage B — Block 15 (after V3/V4 land)

1. **Merge** W1–W4 + V3 (+V4) into one tree.  W1+W2+V3 are already on the fork tree (21 files,
   uncommitted); W3 overlaps W2 in `src/llama-memory-hybrid-idx.cpp`; W4 is a separate 2-file patch
   (`wip/arch-independent-memory/patches/0001-*`) that also applies to upstream master.  Use
   `git add -A` + `git apply -3` (never `patch -F3`) and resolve conflicts by hand; the fork tree is the
   authority for W1+W2+V3.
2. **Gate everything** (D3 + the V3/V4 gates): `GGML_QSA_SCORE_MEM` (W1), `GGML_QSA_DERIVED_BIAS` /
   `GGML_QSA_DERIVED_VIS` / `LLAMA_QSA_SPARSE_FA` (W2, with the `2|3` diagnostics **stripped**),
   `LLAMA_QSA_KEYS_ONLY` (W3), `LLAMA_KQ_MASK_DERIVED` (V3, default 1, validated 2026-09-10), the V4
   switch.  W4 has no knob (§2).  The V3 diagnostic (`LLAMA_KQ_MASK_DERIVED_VERIFY`) and
   `GGML_QSA_DERIVED_BIAS=2|3` must be **stripped** from the patches (both live only in the
   `0002-DIAGNOSTIC`-style snapshots, never in the delivery).  Document every gate + default in
   `patches/README.md` and the beta README.
3. **Combined validation**: `README.md` §4 protocol + the interaction cases listed in the old handover
   notes (W3+W2 on the same cache; W1×W2 in all four gate combinations; `LLAMA_QSA_SPARSE_FA=0` with
   W3; V3 with V4 on/off; the dense models 4B/27B for V3/V4, qwen4exp for W1–W3).
4. **Cut the block**: fork commit (15th block commit; never pushed), `scripts/make-patches.sh` with the
   tip default updated (verify blocks 01–14 come out byte-identical), `scripts/apply-all.sh` 14 → 15,
   `MANIFESTS.md` / `README.md` / `WORKLOG.md` / `BASELINE.md`, `rdna-boosts-all.patch`, then a
   **clean-apply simulation** in a fresh worktree at the fork point + build + coherence.
5. **Stage it here**: `beta/block-15-campaign-wins/block-15-campaign-wins.patch` + the promotion record
   in `README.md` (beta start date, gate table, validation results).  Beta window ~4–5 days.
6. **Upstream-drop check** before shipping: if upstream merged W4 (or another `upstream/` entry), drop
   that hunk from Block 15 and note it in WORKLOG.
7. **Promotion** (after the window): `patches/0015-…`, `apply-all.sh`, `make-patches.sh` tip,
   `MANIFESTS.md`/`README.md`, WORKLOG, beta README marked PROMOTED.  Campaign closed.

## 6. State inventory

**Fork** `~/llama.cpp`, branch `rdna-boosts` at `e2380eb67` (= fork point `9113cc188` + blocks 01–14),
working tree = **21 modified files = W1 + W2 (incl. the `attn_k` guard) + block-15 phase 1 + V3 phases
2a/2b/2c**, deliberately uncommitted.  V3 is *complete and on by default* here; W3 and V4 are not
applied.  Never commit those files except as the Block-15 commit; never push from that checkout.
`build-rocm/{llama-cli,llama-bench,test-backend-ops}` are current with this tree and were used for the
whole V3 validation (2026-09-10).

**Campaign patches** (all `git apply --check` clean, in this order over a clean `9113cc188`):
W1 `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch`,
W2 `…/0002-derived-qsa-block-bias.patch` (10 files, +557/−80), W3
`wip/qwen4exp/keys-only-indexer/0001-…`, W4 `wip/arch-independent-memory/patches/0001-…` + its PR copy
`upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch` (the A/B revert is `ab/w4-revert.patch`), V3
`wip/arch-independent-memory/patches/0002-DIAGNOSTIC-…` (the phase-1 oracle, **diagnostic only - never
in the delivery**), `0003-…` (engine: op + CPU reference), `0004-…` (CUDA MMA kernel), `0005-…` (graph
plumbing + probe + enable).  W3 and W4 are the only campaign pieces not yet on the fork tree.

**Volatile helpers** (rebuild; `/tmp` may be wiped): `/tmp/bin-pristine` (14 blocks), `/tmp/bin-l2` (W1),
`/tmp/bin-l1` (W1+W2 pre-guard), `/tmp/bin-l1guarded` (= the current tree), `/tmp/bin-l3b` (tree + W4),
`/tmp/bin-keysonly` (W3), `/tmp/bin-l0base|l0c|l0d` (instrumented allocators; source
`/tmp/ggml-alloc.instrumented.c`), `/tmp/master-pr` (a master worktree with W4 applied — disposable),
`/tmp/alloc-leak-repro{,2}`.

**Tools** (in-repo): `wip/qwen4exp/qsa-memory/tools/{model-sweep.sh,mask-scaling.sh,bufsize.sh,ub-sweep.sh,ab-coherence.sh,mtp-ab.sh,l0a-scheddump.sh,peak-ledger.py,view-reuse-ledger.py}`; repro
`wip/arch-independent-memory/repro/ggml-alloc-unused-view.c`.

**Rebuild**: `export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH && cmake --build build-rocm --target
llama-cli llama-bench test-backend-ops -j 16`; run with `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
HIP_VISIBLE_DEVICES=0,1,2`.  **Do not add `GGML_CUDA_FA_WMMA_256=0` when V3/V4 behaviour matters** - it
caps WMMA at head 128 and therefore turns V3 off for the 256-wide-head models (the tools under
`wip/qwen4exp/qsa-memory/tools/` set it; drop it there).  Model paths, the V3 numbers and the V3 launch
recipe are in `wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md` §4.4.

**Patch regeneration recipe** (worktree trick): worktree at `HEAD` → apply the base patch → commit →
copy the modified files from `~/llama.cpp` into the worktree → `git diff` → remove the worktree →
verify `git apply --check` on a fresh base.

## 7. Rules that must not be broken

* Nothing under `wip/` enters `patches/` except through promotion (AGENTS.md WIP rule); Block 15 is that
  path and needs the maintainer's go-ahead per stage.
* Never push from `~/llama.cpp`; the fork branch is disposable.
* One GPU job at a time; `llama-cli` always gets `--single-turn`; compare **generated text only**
  (the timing footer always differs).
* Memory numbers come from `bufsize.sh` / `model-sweep.sh` / `mask-scaling.sh`, never llama-bench.
* Anything that can move buffer layout must pass the adaptive-MTP gate (`tools/mtp-ab.sh`, ≥ ~0.45).
* **Anything that touches the kq mask must be validated on an SWA model as well as a non-SWA one** (D7) —
  `gemma-4-E4B-it-Q8_0.gguf` is the cheap SWA target, Qwen3.5-4B/27B are the controls.
* Check for stray `llama-cli`/`llama-bench` processes before measuring; `llama-slot-prox` owns ports
  8037–8039.
* Keep the delivery-repo tree clean between sessions; commit with clear messages.

## 8. Next-session prompt (copy-paste)

```
Continue the RDNA memory campaign in /home/stew675/llama-cpp-rdna-boosts (read AGENTS.md first - its rules
override everything here).  This session is V4: remove the flash-attention F16 K/V staging scratch
(~832 MiB/GPU at ctx 204800 / ub 2048 with -ctk/-ctv q8_0).  It is the LAST campaign item before Block 15.

READ FIRST: beta/block-15-campaign-wins/HANDOVER.md section 3.2 - it is a full V4 map written 2026-09-10
(the mechanism with exact call sites, the scoping correction, the implementation sketch, the validation
and the ship rule).  Then wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md section 4.4 for the V3
numbers/launch recipe and beta/block-15-campaign-wins/BETA-TESTING.md for the gate table.

STATE: V3 is DONE and on by default (patches .../arch-independent-memory/patches/0003, 0004, 0005; the
fork tree has 21 modified files = W1+W2+V3 phases 1/2a/2b/2c, uncommitted by design, builds clean).
Measured -799 MiB/GPU + -799 MiB host on the 4B/27B, -809/-811 on the gemmas, byte-identical output,
MTP acceptance unchanged.  Only V4 is left.  Rebuild: export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH &&
cmake --build build-rocm --target llama-cli llama-bench test-backend-ops -j 16; run with
LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib.  DO NOT set GGML_CUDA_FA_WMMA_256=0 when V3/V4 matters.

DO, in this order:
1. THE FIRST MEASUREMENT, before any code (section 3.2 says why it decides the scope): reserve sizes with
   q8_0 KV at -ub 8 vs -ub 2048, and with -ctk/-ctv f16 as the control, at ctx 204800 on the 4B (and the
   27B if it is quick).  Reconcile the scratch against ggml_nelements(K)*2 + ggml_nelements(V)*2.  If the
   ~832 MiB also shows up at -ub 8, TILE is the blocking kernel for the reservation and must be fixed
   together with MMA (the verify batches select TILE; VEC is already scratch-free for quantized KV).
   Tools: wip/qwen4exp/qsa-memory/tools/bufsize.sh (drop its GGML_CUDA_FA_WMMA_256=0).
2. Implement dequant-on-stage for q8_0 ONLY, keep need_f16_K/V true for every other type, keep use_sparse
   out.  Touch points: ggml_cuda_flash_attn_ext_get_alloc_size (fattn.cu:694), launch_fattn
   (fattn-common.cuh:976 - the to_fp16 conversions), the MMA loader flash_attn_ext_f16_load_tile
   (fattn-mma-f16.cuh:368, K at 692/1029/1385, V at 672/1045) and the TILE loaders if step 1 says they are
   in the reservation.  cp_async is impossible for a quantized source - measure what that costs instead of
   assuming.
3. Validate with the D9 three-way rule: same-seed text byte-identical with the gate flipped in ONE binary
   (4B, 27B, gemma-4-E4B as the SWA case, qwen4exp as the control); the reserve matrix (expect -832 MiB
   with q8_0, exactly 0 with f16); MTP acceptance >= 0.45 and unchanged; INTERLEAVED bench parity
   (pp20480 ub 2048/1024, tg256, 5 passes - single-pass noise is ~0.5 %); test-backend-ops FLASH_ATTN_EXT
   on CPU + ROCm0.  No prefill regression -> on by default; some regression -> opt-in default-off; bad
   enough to be useless as opt-in -> document and stop (3.2 says so explicitly).
4. Snapshot each finished slice as its own patch (.../arch-independent-memory/patches/0006-*.patch) with
   the documented worktree recipe (base = W1+W2+diag+2a+2b+2c+... - see section 6), never commit on the
   fork branch, never push.  If the budget runs out, STOP at a buildable checkpoint, confirm coherence is
   unchanged, snapshot, and update HANDOVER section 3.2 + section 9.

DO NOT: cut Block 15 before V4 is decided (it is the next session's job or the one after), touch
archive/work/, or present wip results as delivery claims.

REPORT: what landed (with the reserve numbers before/after and the coherence evidence), the gate name +
default and its measured justification, what was not exercised and why, and the updated fork-tree /
snapshot state.
```

## 9. Open questions

V4 is the only remaining item and it carries two questions that the first measurement answers (do not
decide them by reasoning):

1. **Which kernel drives the reserved scratch?** If the ~832 MiB is present at `-ub 8` too, TILE is the
   blocking kernel and V4 cannot shrink the reserve without fixing TILE as well as MMA (section 3.2).
2. **What is V4's default?** On unless it costs prefill throughput; opt-in default-off otherwise; and
   *which* KV types it covers (q8_0 first, the rest only if the first increment is cheap) - the
   three-way rule (D9) decides.

One V3 question is left open deliberately: the source of the measured **-1.1 % prefill** (kernel
registers from the derived branch in `flash_attn_ext_f16_load_mask` vs the smaller reserved buffer
changing the graph's buffer layout - the mask bytes themselves account for ~1 ms of ~3 s).  It is
documented, not a blocker, and V4 rewrites that same loader anyway.

Everything else is answered: W4 = option C (a revert patch for A/B), V3 includes 3.2 (D7), the beta
checklist exists (`BETA-TESTING.md`, D8), V4 follows the three-way ship rule (D9), and V3 phase 3.3
(M-RoPE) is **not** included - the degeneracy guard is the delivered behaviour, and the phase-1 oracle
only has to be re-run if the predicate is ever extended.
