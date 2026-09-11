# Block 0015 handover — campaign wins → beta delivery (+ upstream candidates)

> **REVALIDATED 2026-09-11 — see §10.6 for the outcome.**  The beta patch was cut on
> `b425aa8f7` (the 14-block chain) and has been **re-cut against canonical tip `389c5341f`
> (tree `928852cdc`) and re-validated end to end**: new beta tip **`fe4f55278`**
> (tree `ffe197e2f`) — and **re-cut again 2026-09-11** after block 14's hyper-connection band
> amendment moved the canonical tip: base `1d8f53594` -> **beta tip `54859fdda`, tree
> `543ccc015`**, metadata/offset-only (0 changed body lines) — see the beta `README.md`.
> The dependent delta was exactly one file
> (`fattn-common.cuh`), the textual apply was clean, every 2026-09-10 number
> reproduced to the last decimal, and three **pre-existing** follow-ups were
> found (§10.7).  §10.1's original `[PATCH 16/16]` renumbering claim was **wrong** —
> see the correction there.

> **STATUS 2026-09-10 (end of the cut session): DONE — Block 15 is CUT and
> in the delivery.**  `block-15-campaign-wins.patch`
> (canonical tip `09a137566` on a fork rebuilt at `9113cc188`), 15 patches
> total, `scripts/apply-all.sh` applies the 14-block delivery with strict 14/14 `git am` (the beta block-15 patch is applied on top),
> `rdna-boosts-all.patch` refreshed.  The combination validation is complete
> (reserve matrix, byte-identical coherence on all five models, the MTP gate,
> the op suites) on both the merged tree and the tree built from the delivered
> patches — see `README.md` in this directory and the 2026-09-10 entry in
> `../../WORKLOG.md`.  The campaign's critical path is empty.  What remains is
> the **beta window** (~4–5 days of tester feedback) and, after it, either a
> dated amendment (if feedback needs one) or nothing.  The one open follow-up
> is the **bf16 lever** (§3.4: measured, not implemented — it would fold into
> block 15 as an amendment, per D10); the two unfixed items are the V3 −1.1 %
> prefill cost (§9.1, accepted) and the **pre-existing** gemma-4-E4B 3-GPU
> tensor-split abort found during the combination pass (documented in
> `../../patches/README.md` and `README.md`; not a block-15 regression).
> This document is now the historical record of how the block was built.

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
| D9 | **V4 ship rule (three-way)**: no prefill-throughput regression → on by default; regression → ship it **opt-in (default off)** for people who need the last 832 MiB, with the trade-off documented; if even that is impractical → future work.  **Applied 2026-09-10**: the q8_0 arm measured −1.7 % prefill, so it ships **opt-in** (`GGML_CUDA_FA_KV_NATIVE=1`); and the maintainer's refinement of D9 is that **a sub-2 % loss with a memory win and no cheap way to close the gap ships opt-in anyway** (do not grind for the last percent). |
| D11 | **gemma-4-E4B 3-GPU tensor-split abort (2026-09-10): document only, do NOT fix.**  The pre-existing meta-splitter abort found during the Block-15 combination pass (`n_head_kv = 2` is fewer than the 3 devices, so one device gets a zero-extent KV share) stays a documented limitation.  Rationale: it is a small model and running it in 3-GPU tensor-split mode is an unlikely configuration; it runs on 1 GPU, on 2 GPUs and on 3 GPUs with `-sm layer`.  See `README.md` (this directory). |
| D12 | **bf16-native MMA K/V is the one essential follow-up (2026-09-10) — CLOSED the same day: shipped as V5**, a dated amendment to Block 15 (D5/D10), not a new block.  The maintainer's instruction when the measurement said it costs ~1 % prefill: *"treat it similarly to V4, and include it in the Patch 15 block, and have it gated by the same environment variable that V4 does"* — so V5 lives behind `GGML_CUDA_FA_KV_NATIVE` (default 0).  Plan + outcome: `../../wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` (the plan, then **§9 the outcome**); the delivered record is the V5 amendment section in `../../patches/README.md`. |
| D10 | **bf16 (2026-09-10): no pure-bf16 rework.**  llama.cpp is predicated on F16 as the always-available default, so the fork keeps the F16 compute path; bf16 K/V work must remove the *staging* (convert in place, keeping the F16 fragments and the cp_async pipeline), **not** re-instantiate the kernels natively.  A pure-bf16 fork is explicitly out of scope for now.  See §3.4. |

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

**V4 is done** (2026-09-10) and is the sixth validated win - a **q8_0 KV cache is dequantized while
staging the FA tiles** (`ggml_cuda_fattn_dequantize_q8_0_chunk`), so the whole-cache F16 staging scratch
and its per-ubatch conversion pass are gone.  Measured: **−744 MiB/GPU on the 4B** (1001.13 → 257.13) and
**−632 MiB on the 27B** (Meta 1121.13 → 489.13) at ctx 204800 / ub 2048, more at smaller ub (4B ub 1024
−772, ub 512 −786; gemma-4-31B −1224), qwen4exp control unchanged; coherence byte-identical on 4B/27B/
gemma-4-E4B/gemma-4-31B/qwen4exp incl. the SWA models, 27B MTP acceptance identical (0.76744 - the
ulp-sensitive probe), FA op suite green.  Cost: prefill −1.7 % (both models), decode ±0.1 %.  Per the
maintainer's rule of 2026-09-10 (a sub-2 % loss with a memory win and no cheap way to close the gap)
**V4 ships OPT-IN: `GGML_CUDA_FA_KV_NATIVE=1`, default off.**  Record:
`../../wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md`.

**The critical path is empty** - the only remaining campaign step is the Block 15 merge + cut (§5):

| id | what | measured effect (dense models, ctx 204800, ub 2048, q8_0 KV) |
|---|---|---|
| **V3** | derived kq mask for the plain (non-QSA) attention path - **DONE 2026-09-10**, see above | **−799 MiB/GPU VRAM − 799 MiB host** measured |
| **V4** | native q8_0 K/V in the FA path (both the MMA and the TILE loaders) - **DONE 2026-09-10, opt-in**, see above | **−744 MiB/GPU** (4B) / **−632 MiB** (27B) measured |
| — | **merge + cut Block 15** (§5) - the only remaining campaign step | — |

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

## 3. The critical path — empty (V3 and V4 are DONE)

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

### 3.2 V4 — native q8_0 K/V in the FA kernels — **DONE 2026-09-10 (OPT-IN)**

**Result**: the F16 staging scratch and its per-ubatch conversion pass are gone for q8_0 K/V (both the
MMA and the TILE loader dequantize while staging the tiles).  **−744 MiB/GPU on the 4B** (1001.13 →
257.13) and **−632 MiB (Meta) on the 27B** (1121.13 → 489.13) at ctx 204800 / ub 2048, more at smaller
ub, gemma-4-31B −1224 MiB, qwen4exp control unchanged; coherence byte-identical on every model tested
(incl. both SWA gemmas and the 40k prompts), 27B MTP acceptance identical (`0.76744`), FA op suite green.
Cost: prefill **−1.7 %** (27B and 4B, interleaved 3 reps), decode **±0.1 %**, TILE-path prefill neutral.

**Decision (maintainer, 2026-09-10)**: with a sub-2 % loss, a large memory win and no cheap way to close
the gap (the loss is the lost `cp_async` pipeline - a quantized source cannot be copied asynchronously -
not the dequant ALU; a 2-byte-access optimisation pass changed nothing) → **ship opt-in, default off**:
`GGML_CUDA_FA_KV_NATIVE=1` enables it.  Full record (mechanism, design, implementation table, reserve
matrix, validation, the on-by-default follow-up): `../../wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md`.
Patch: `wip/arch-independent-memory/patches/0006-v4-native-q8-kv.patch` (6 files, +357/−47, base
W1+W2+diag+2a+2b+2c).

**What is still worth doing later** (not blockers, docs only): the cp_async-preserving design (stage raw
q8_0 rows into shared and dequantize shared→shared) if the pipeline loss ever needs to be recovered, and
the other quantized KV types (each is the same chunk decoder with a different block layout; the vec-path
`dequantize_*` helpers already exist in `fattn-common.cuh` and their arithmetic must be matched exactly).

### 3.3 Then, and only then, Block 15

Block 15 = **W1 + W2 + W3 + W4 + V2-or-V3 + V4** as one patch (D5), merged, gated, combined-validated,
cut, and staged in `beta/` to open the ~4–5 day beta window.  See §5.

### 3.4 The next memory lever: bf16-native MMA K/V — **DONE: shipped as V5** (`../../wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` §9)

> **Status: implemented, validated and folded into Block 15 on 2026-09-10** (the
> amendment touched `block-15-campaign-wins.patch` only; canonical tip `f5ab5350b`).  A bf16 KV
> cache with `GGML_CUDA_FA_KV_NATIVE=1` now costs exactly what an f16 cache costs
> (4B 968.86 → **256.86** MiB/GPU at ub 2048, 27B 1072.86 → **488.86**,
> gemma-4-E4B 1062.89 → **404.89**, gemma-4-31B 2068.89 → **716.89**), with
> byte-identical output and MTP, for 0.2-2.4 % prefill (growing with the prompt)
> and no decode cost — hence opt-in through V4's switch, as instructed.  The
> exploratory text below is kept as the pre-implementation record.

> **Status: the executable plan is written** (D12).  It supersedes the exploratory text below with a
> measured before-state, the exact code map, the design, the validation protocol and the ship rule.
> Numbers below were measured in the *pre-block-15* tree; the plan file carries the **delivered
> block-15** matrix (4B ub 2048 256.86 → 968.86 MiB = +712; ub 1024 +756; ub 512 +778; 27B ub 2048
> +584; ub 512 +746; ub 8 unchanged; V4 does not change any bf16 row).  Two extra facts the plan
> records: the bf16 arm should ship **on by default** (it keeps the cp_async pipeline), and **mixed
> K/V types are broken independently of bf16** (`-ctk bf16 -ctv q8_0` falls off the GPU attention
> path: graph splits 18 vs 2, pp2048 7924 → 640 t/s) — so the practical choices are same-type K/V.


The maintainer's preferred KV type is **bf16**, and it is the one case V4 does *not* cover: the TILE and
VEC paths already read bf16 natively, but the MMA prefill path still stages an F16 copy of the whole
cache (`need_f16_K/V = true` unconditionally for `BEST_FATTN_KERNEL_MMA_F16`).  Measured on the 4B
(`-ctk/-ctv bf16`, ub 2048, 1 GPU):

| ctx | f16 (no scratch) | bf16 | Δ | K+V F16 scratch for one layer = ctx x 4 KiB |
|---|---|---|---|---|
| 32768 | 256.20 | 296.20 | **+40.00** | 128 |
| 131072 | 256.58 | 680.58 | **+424.00** | 512 |
| 163840 | 256.70 | 808.70 | **+552.00** | 640 |
| 204800 | ~256.5 | ~968 (extrapolated) | **~+712** | 800 |

The relation is exact and linear: **bf16 costs `scratch - 88 MiB` in the peak** (the allocator overlaps
88 MiB of the scratch with other live tensors), so at ctx 204800 / ub 2048 a bf16 user pays ~712 MiB/GPU
that V4 already removes for q8_0.  At ub 8 (TILE) bf16 and f16 are both 5.47 MiB - the verify/decode side
needs nothing.  V3's mask win (-799 compute / -799 host) already applies to bf16 too (the mask is F16
whatever the KV type is).

**Why this is the *easier* case than q8_0** (and the recommended follow-up - **scope fixed by D10**: keep
the F16 fragment path, remove only the staging):

1. bf16 -> f16 is **size- and layout-preserving**: a 16-byte staged chunk is 8 bf16 elements -> 8 f16
   elements, same 16 bytes.  So the **cp_async pipeline can be kept**: copy the raw bf16 row into the
   shared tile exactly like the F16 path (same chunks, same swizzle, same `tile_K`/`tile_V` layout), then
   convert the tile **in place** element-wise (`__float2half(__bfloat162float(x))`) with one extra
   `__syncthreads()` before the wmma fragment loads.  No byte unpacking, no block indices, no lost
   latency hiding - so unlike the q8_0 arm this should measure ~free and could ship **on by default**.
2. It also removes the per-ubatch global conversion pass (which for bf16 reads the whole cache and writes
   the same amount again - 800 MiB + 800 MiB per layer per ubatch at ctx 204800), so it may be a *win*.
3. Bit-exactness is straightforward: bf16 -> f32 is exact and f32 -> f16 rounds once, which is what the
   launcher's `ggml_get_to_fp16_cuda(GGML_TYPE_BF16)` does.
4. Touches only what V4 already touches: the shared predicate (`add BF16`), `launch_fattn`'s conversion
   skip, the MMA loader branch (the `fattn_kv_q8_t` struct generalises to a `{type, ptr, stride}` tag),
   plus a device-side in-place conversion loop.  Validate with the same protocol (reserve matrix,
   same-seed coherence, the ulp-sensitive MTP gate, interleaved pp20480/tg256).

**Honest caveat**: this removes the *staging*, not the cache.  A bf16 KV cache is 2 B/element, ~1.9x the
q8_0 cache (27B at ctx 204800: ~54 GB bf16 vs ~28.5 GB q8_0 across 3 GPUs), and no amount of V4 work
changes that - the compute-buffer scratch is ~712 MiB of it, the rest is the format choice.

**Where this gap comes from (so it does not look like a BF16 regression)**: block 03 ("BF16 KV cache and
native-BF16 flash-attn") implemented native BF16 for the **tile** kernel (`v_dot2_f32_bf16` packed dot,
bf16 PV pairing, bf16 tiles/registers), the **vec** kernel and bf16 RoPE/set_rows - `fattn-mma-f16.cuh`
has **zero** bf16 references to this day, and block 03's own `fattn.cu` hunk added the `use_bf16` arm only
to `BEST_FATTN_KERNEL_TILE`, deliberately leaving `BEST_FATTN_KERNEL_MMA_F16: need_f16_K = true`.  Before
block 04 the AMD WMMA arm was capped at head <= 128, so the 256-wide-head models (4B, 27B, gemma-4)
prefilled on the tile kernel and consumed bf16 natively - no staging.  **Block 04 (RDNA4 WMMA) extended
WMMA to head 576 (RDNA4/RDNA3_0) / 320 (RDNA3_5), which is what moved those models' prefill onto the
F16-operand WMMA kernel** and made the whole-cache F16 staging appear for bf16.  So the trade introduced
was "+WMMA prefill speed" for "+one ctx-linear F16 scratch", and V4-bf16 is the way to keep both.

Coupling worth knowing: V3's derived mask needs the MMA kernel, and (for bf16) the MMA kernel needs the
staging - so today a bf16 user can have **either** the mask win (WMMA on: 808.70 MiB at ctx 163840 /
ub 2048, no mask, staging) **or** native bf16 prefill (WMMA off via `GGML_CUDA_FA_WMMA_256=0`: 896.06,
mask back, no staging), but not both.  The bf16 arm (**V5**, shipped 2026-09-10) gives both (256.86 MiB on the 4B at ub 2048,
i.e. activations only).

**Block numbering**: it landed in Block 15 as planned (same `GGML_CUDA_FA_KV_NATIVE` gate, one more arm)
— **V5, amended 2026-09-10**, see §3.4.  The "exactly one new block" rule was never strained.

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

## 5. Stage B — Block 15 (V3 and V4 have landed)  [executed 2026-09-10; V5 added the same day, see §3.4]

1. **Merge** W1–W4 + V3 + V4 into one tree.  W1+W2+V3+V4 are already on the fork tree (22 files,
   uncommitted); W3 overlaps W2 in `src/llama-memory-hybrid-idx.cpp`; W4 is a separate 2-file patch
   (`wip/arch-independent-memory/patches/0001-*`) that also applies to upstream master.  Use
   `git add -A` + `git apply -3` (never `patch -F3`) and resolve conflicts by hand; the fork tree is the
   authority for W1+W2+V3+V4.
2. **Gate everything** (D3 + the V3/V4 gates): `GGML_QSA_SCORE_MEM` (W1), `GGML_QSA_DERIVED_BIAS` /
   `GGML_QSA_DERIVED_VIS` / `LLAMA_QSA_SPARSE_FA` (W2, with the `2|3` diagnostics **stripped**),
   `LLAMA_QSA_KEYS_ONLY` (W3), `LLAMA_KQ_MASK_DERIVED` (V3, default 1, validated 2026-09-10),
   `GGML_CUDA_FA_KV_NATIVE` (V4, **default 0 = opt-in**, validated 2026-09-10).  W4 has no knob (§2).
   The V3 diagnostic (`LLAMA_KQ_MASK_DERIVED_VERIFY`) and `GGML_QSA_DERIVED_BIAS=2|3` must be
   **stripped** from the patches (both live only in the `0002-DIAGNOSTIC`-style snapshots, never in the
   delivery).  Document every gate + default in `patches/README.md` and the beta README.
3. **Combined validation**: `README.md` §4 protocol + the interaction cases listed in the old handover
   notes (W3+W2 on the same cache; W1×W2 in all four gate combinations; `LLAMA_QSA_SPARSE_FA=0` with W3;
   V3 with V4 on/off; the dense models 4B/27B for V3/V4, qwen4exp for W1–W3).  **Because V4 is opt-in,
   validate the default build first** (`GGML_CUDA_FA_KV_NATIVE` unset - this is what beta testers get)
   **and then a second pass with `=1`** to cover the opt-in path, including the SWA gemmas and the MTP
   gate for both.
4. **Cut the block**: fork commit (15th block commit; never pushed), `scripts/make-patches.sh` with the
   tip default updated (verify blocks 01–14 come out byte-identical), `scripts/apply-all.sh` 14 → 15,
   `MANIFESTS.md` / `README.md` / `WORKLOG.md` / `BASELINE.md`, `rdna-boosts-all.patch`, then a
   **clean-apply simulation** in a fresh worktree at the fork point + build + coherence.
5. **Stage it here**: `beta/block-15-campaign-wins/block-15-campaign-wins.patch` + the promotion record
   in `README.md` (beta start date, gate table, validation results).  Beta window ~4–5 days.
6. **Upstream-drop check** before shipping: if upstream merged W4 (or another `upstream/` entry), drop
   that hunk from Block 15 and note it in WORKLOG.
7. **Promotion** (after the window): promote `block-15-campaign-wins.patch` to `patches/0015-…`, bump `apply-all.sh`/`make-patches.sh` to 15 blocks,
   `MANIFESTS.md`/`README.md`, WORKLOG, beta README marked PROMOTED.  Campaign closed.

## 6. State inventory

**Fork** `~/llama.cpp`: `rdna-boosts` is **pristine at `e2380eb67`** (= fork point `9113cc188` + blocks
01–14, the canonical base for the block-15 cut).  The 22-file campaign tree (= W1 + W2 (incl. the
`attn_k` guard) + block-15 phase 1 + V3 phases 2a/2b/2c + V4, all validated 2026-09-10) is **committed on
the work branch `wip/block15-campaign-wins` = `b26ae06f0`** (working tree clean, branch checked out) and,
as a second copy, as a single delta patch
`wip/arch-independent-memory/snapshots/fork-tree-W1-W2-V3-V4-2026-09-10.patch` (3453 lines, applies
cleanly to `e2380eb67`, 22 files +1871/−210 - `git apply` it in a fresh worktree if the branch is ever
lost).  V3 is *on by default* and V4 is *opt-in* in that tree; W3 is not applied.  **Resume with**
`git checkout rdna-boosts && git merge --squash wip/block15-campaign-wins && git apply -3 <W3> <W4>` then
build/validate and commit the single block-15 commit - or simply keep working on the work branch and cut
the block commit from it.  Never push from that checkout.  `build-rocm/` was current with the work-branch
tree and was used for all V3 and V4 validation (2026-09-10).

**Snapshots**: the work branch above; the whole-tree delta patch
`wip/arch-independent-memory/snapshots/fork-tree-W1-W2-V3-V4-2026-09-10.patch`; per-increment patches
below.

**Campaign patches** (all `git apply --check` clean, in this order over a clean `9113cc188`):
W1 `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch`,
W2 `…/0002-derived-qsa-block-bias.patch` (10 files, +557/−80), W3
`wip/qwen4exp/keys-only-indexer/0001-…`, W4 `wip/arch-independent-memory/patches/0001-…` + its PR copy
`upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch` (the A/B revert is `ab/w4-revert.patch`), V3
`wip/arch-independent-memory/patches/0002-DIAGNOSTIC-…` (the phase-1 oracle, **diagnostic only - never
in the delivery**), `0003-…` (engine: op + CPU reference), `0004-…` (CUDA MMA kernel), `0005-…` (graph
plumbing + probe + enable; the three V3 patches are already in the fork tree), `0006-v4-native-q8-kv.patch`
(V4, 6 files, +357/−47, opt-in, also already in the fork tree).  W3 and W4 are the only campaign pieces
not yet on the fork tree.

> **Revalidation artifacts (2026-09-11, `/tmp` — rebuild if wiped):** `/tmp/bin-15blk` = the delivered
> 15-patch build (baseline), `/tmp/bin-blk15` = the re-cut block-15 build (`fe4f55278`); `/tmp/blk15` =
> the worktree holding the re-cut block-15 commit (branch `blk15-recut`); `/tmp/p16b/` = the regenerated
> 16-patch set; `/tmp/rv-sim2/` = the clean-apply simulation result (16 commits, tree `ffe197e2f`);
> the reusable driver + KV-capable probe are committed in `../../wip/kv-quant-purity-followups/tools/`
> (`rv.sh`, `logits-dump-kv.cpp` — the latter is the width probe with `CTK`/`CTV`, which is what the F1
> table needs).  `/tmp/lw-kv-blk15` and `/tmp/lw-kv-15blk` were the two probe builds.

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

## 8. Next-session prompt (copy-paste — the bf16 follow-up, D12) — **CONSUMED 2026-09-10**

> This prompt was executed on 2026-09-10 and produced V5; it is kept as the record of
> what was asked.  The measurement outcome and the ship decision are in §3.4 above and
> in the V5 amendment section of `../../patches/README.md`.  One difference from the
> prompt: it proposed a *separate* `GGML_CUDA_FA_KV_BF16` gate defaulting to 1; the
> maintainer's follow-up instruction was to reuse **V4's `GGML_CUDA_FA_KV_NATIVE`**
> switch instead (and keep it opt-in, default 0), which is what shipped.

```
Implement bf16-native K/V in the MMA flash-attention path (the last campaign follow-up) in
/home/stew675/llama-cpp-rdna-boosts (read AGENTS.md first - its rules override everything here).

READ FIRST: wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md (the full plan: the measured before-state,
the mechanism with exact call sites, the design, the code map, the validation protocol, the ship rule and
the risks), then beta/block-15-campaign-wins/HANDOVER.md section 3.4 (the scope decision D10) and
wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md (V4 is the template this follows - its implementation
table is the shape to copy).

GOAL: a bf16 KV cache must pay no F16 staging scratch and no per-ubatch conversion pass in prefill,
with the cp_async pipeline kept.  Measured today (delivered block-15 tree, ctx 204800, f16 reference):
4B ub 2048 256.86 -> 968.86 MiB (+712), ub 1024 +756, ub 512 +778; 27B ub 2048 +584, ub 512 +746;
ub 8 (TILE) identical.  Target: bf16 == f16 at every row.

PLAN (detail and code map in the plan file):
1. add a native-bf16 predicate + env gate GGML_CUDA_FA_KV_BF16 (default 1) next to V4's
   ggml_cuda_fattn_kv_native_supported in fattn-common.cuh, and generalise V4's kv tag;
2. skip the whole-cache F16 conversion for a native-bf16 operand in launch_fattn and size f16_extra with
   the effective need flags; make fattn.cu's get_alloc_size ask the same predicates (the existing
   GGML_ASSERT(f16_extra.K != 0) is the tripwire if they disagree);
3. in fattn-mma-f16.cuh: element-wise path converts bf16->f16 while loading; cp_async path copies the raw
   bf16 bytes to the same shared offsets (a 16-byte chunk is 8 elements either way) and converts the
   tile in place after cp_async_wait_all() + __syncthreads() (a linear pass is enough: the swizzles
   permute whole 16-byte units);
4. TILE/VEC need nothing (block 03 already reads bf16 natively).

VALIDATE (all five, on the delivered tree; keep the tree buildable and snapshot the diff as its own
patch before moving on): the reserve matrix (4B 1-GPU, 27B 3-GPU, gemma-4-E4B 1-GPU ISWA,
gemma-4-31B, qwen4exp control; ub 2048/1024/512; gate off/on; f16 reference) - expect bf16 == f16;
byte-identical same-seed coherence (bf16 vs f16 vs gate off, short + 40k prompts); the MTP gate
(27B 0.76744 and qwen4exp 0.44262 unchanged); test-backend-ops FLASH_ATTN_EXT on ROCm0 + CPU; and an
interleaved same-binary prefill/decode A/B (pp20480 ub 2048 + tg256, 4B and 27B) to decide the default
per the plan's three-way ship rule (expectation: on by default, since the cp_async pipeline is kept).

Then: fold the result into the delivery as a dated block-15 amendment (the beta patch regenerated from a
CANONICAL fork rebuilt at 9113cc188 via scripts/apply-all.sh - never from the working checkout's
rdna-boosts tip, which sits two upstream commits past the fork point), update patches/README.md +
WORKLOG.md + the beta record, re-run the clean-apply simulation, and stage the beta patch copy.  If the
measurement says opt-in instead, say so up front - do not ship a regression on by default.

ALSO RECORD (do not fix): mixed K/V types (bf16+q8_0, f16+q8_0, ...) fall off the GPU attention path
today - graph splits 18 vs 2, ~1.5 GiB host buffer, pp2048 7924 -> 640-1049 t/s.  It is pre-existing,
out of scope for this work, and already documented in the plan file section 6.

DO NOT: rebuild the WMMA kernels with bf16 fragments (D10); push anything from ~/llama.cpp; fold wip/
content into patches/ beyond this agreed work; or touch archive/work/.  Keep 3 GPUs sequential, one job
at a time, and check for stray llama processes before measuring.
```

## 9. Open questions

None blocking.  Everything V3/V4 raised is either answered or explicitly deferred:

1. **V3's −1.1 % prefill**: documented, not chased (kernel registers vs the smaller reserved buffer).
2. **V4's −1.7 % prefill**: identified as the lost `cp_async` pipeline (a quantized source cannot be
   copied asynchronously); a 2-byte-access optimisation pass changed nothing.  Per the maintainer's rule
   of 2026-09-10 the feature ships **opt-in**.  The on-by-default route (stage raw q8_0 rows into shared
   and dequantize shared→shared) is written up in
   `../../wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md` §5 - build it only if the pipeline loss
   ever needs to be recovered, and measure first: it costs shared memory in occupancy-1/2 kernels.
3. **V4 for other KV types**: q8_0 only for now; each additional type is the same chunk decoder with a
   different block layout (the vec-path `dequantize_*` helpers are the arithmetic to match).
4. **V3 phase 3.3 (M-RoPE)**: still not included; the degeneracy guard is the delivered behaviour and the
   phase-1 oracle only has to be re-run if the predicate is extended.
5. **Block 15 packaging detail**: V4 is opt-in, so the beta gate table must present it as "default off,
   `GGML_CUDA_FA_KV_NATIVE=1` to enable" and the combined validation must cover both positions (§5.3).
6. **bf16 (the maintainer's preferred KV type)**: measured, not implemented - the MMA staging scratch
   costs a bf16 user ~712 MiB/GPU at ctx 204800 / ub 2048 and is the recommended next lever (§3.4).  It is
   the *easier* case than q8_0 (in-place conversion keeps the cp_async pipeline, so it can likely ship on
   by default) and it belongs in Block 15 if it lands before the cut.


---

## 10. Revalidation against the current delivery (2026-09-11) — **THE CURRENT TASK**

**Why this exists.**  The beta patch was cut on **`b425aa8f7`** — block 14 of the
**14-block** chain (block 13 `e61676292`), i.e. *before* block 00 existed and
before the 2026-09-11 block-02/12/13 amendments.  The delivery is now a
**15-patch set** (block 00 + blocks 01-14) at canonical tip **`389c5341f`**
(net tree **`928852cdc`**).  Block 15 must be re-cut against that tree and
re-validated; §10.3/§10.4 are the gates.

### 10.1 The dependency delta — measured 2026-09-11, and it is small

`git apply --check block-15-campaign-wins.patch` on a worktree at `389c5341f`:
**clean, no rejects**, a single hunk offset (`fattn-common.cuh` hunk #7 lands at
line 1439 = +8 lines).  Per-file base blob hashes (`b425aa8f7` → `389c5341f`) for
all **23** files the patch touches: **22 are identical**, exactly one changed.

| file | old base blob | new base blob | what changed |
|---|---|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | `7442bc22a` | `22eec7d57` | **block 00** (FA small-batch KV-split width invariance, issue #25): inside `launch_fattn` the `parallel_blocks`/`ntiles_dst` heuristic is evaluated as if `n_q == 1` when `Q->ne[1] <= 8`, so every decode/verify width selects the *same* KV split. +8 lines. |

Unchanged (blob-identical to the cut base — their `index` lines stay valid):
`ggml/include/ggml.h`, `ggml/src/ggml-alloc.c`, `ggml/src/ggml-backend-meta.cpp`,
`ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh`,
`fattn-qsa.cu`, `fattn-tile.cu`, `fattn-tile.cuh`, `fattn-vec.cuh`, `fattn.cu`,
`indexer-topk.cu`, `ggml/src/ggml.c`, `src/llama-context.cpp`,
`src/llama-cparams.h`, `src/llama-graph.cpp`, `src/llama-graph.h`,
`src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`,
`src/llama-memory-hybrid-idx.cpp`, `src/llama-memory-hybrid-idx.h`,
`src/models/qwen4exp.cpp`, `tests/test-backend-ops.cpp`.

**Consequences for the re-cut (verified 2026-09-11):** only `fattn-common.cuh`'s
*pre-image* hash changes (along with its post-image and one `@@` hunk header), and the
`From <sha>` identity changes (the re-cut commit).  **CORRECTION to the first draft of this
section: no renumbering is needed, and `[PATCH 16/16]` was wrong.**  The repo convention is
`git format-patch --start-number 0 $BASELINE..$TIP`, which makes the denominator the **last
block index**, not the count: the delivered 15-patch set reads `[PATCH 00/14]`…`[PATCH 14/14]`,
and the beta patch's long-standing **`[PATCH 15/15]` was already correct** (verified: regenerating
the 16-commit range with the same convention reproduces all 15 delivery patch bodies
byte-identically and emits block 15 as `[PATCH 15/15]`).  On promotion the whole set's
denominator simply moves `/14` → `/15`, which `scripts/make-patches.sh` does by construction.

### 10.2 The one real semantic risk — block 00 and V4/V5 in the same function

Block 00's fix and Block 15's V4/V5 operand-staging changes live in **the same
function** (`launch_fattn`) and the same decision region.  Block 00 makes the
KV-split heuristic query-width-independent; V4/V5 change how K/V are staged
(native q8_0/bf16 instead of an F16 scratch) and therefore the smem/alloc-size
queries.  Both feed the split choice.  **A clean textual apply proves nothing
about that interaction**, so the revalidation must:

1. show the selected split is **invariant to the KV staging type** (V4/V5 off vs
   on) across the whole decode/verify band;
2. re-establish the **dense `--spec-draft-n-max <= 7` (W <= 8) guarantee** for
   every gate combination — with the **raw-logit probe**
   (`wip/sm-tensor-plain-vs-spec/logits-dump-singlewidth.cpp`), **never** the
   text gate (a 300-token text match hid already-divergent logits at W=9);
3. confirm `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}` (the single shared type code used
   by the launcher, the alloc-size query and the kernels) still yields the same
   `ntiles`/`ntiles_KV` values as the unamended launcher.

### 10.3 Correctness validations (re-run all; the 2026-09-10 records are history)

| # | check | detail / new-invariant notes |
|---|---|---|
| C1 | reserve matrix | 5 models × ub 2048/1024/512 × V4/V5 off/on; each number must reproduce the per-win records and compose additively (qwen4exp pristine 6690.40 → W1 4450.40 → W1+W2 **3251.39**; W2 alone 5491.39; gates off = 6690.40/1262.70 exactly) |
| C2 | same-seed **byte-identical** output, gates flipped | 4B, 27B, **gemma-4-E4B (SWA)**, **gemma-4-31B (SWA)**, qwen4exp — short + 40k prompt, every gate combo. The SWA models are mandatory: V3 changes the kq mask |
| C3 | **dense `n_max <= 7` probe matrix** *(new invariant)* | 1 GPU / 2-GPU `-sm layer` / 2-GPU `-sm tensor` / 3-GPU `-sm tensor`: W=1..8 bit-identical, W=9 divergent (cause B, accepted). Run per gate combo (V3/V4/V5). **The boundary must not move** |
| C4 | **MoE asterisk** *(new, accepted)* | qwen35moe: default W=1 != W=3 (accepted, rule 3); with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` W=1 == W=3 (`bd138ad2`). Validate the **delivery default first**, then the switch — never flip a switch to make a gate pass |
| C5 | adaptive-MTP gate | 27B, qwen4exp, MoE — per gate combo; acceptance must equal the **in-session** baseline (the absolute number is prompt/config dependent: the MoE read 0.675 in the old record and **0.58378** with the current command — always measure baseline and arm in the same session) |
| C6 | op suites | `FLASH_ATTN_EXT` on ROCm0 (both V4/V5 gates) + CPU incl. **the six derived-mask cases**; `GATED_DELTA_NET`; VIEW/CONT/CPY/DUP/CONCAT; `test-alloc`; `test-batch-alloc` |
| C7 | W4 | repro 56.00 → 16.00 MiB and `ab/w4-revert.patch` restores `ggml-alloc.c` byte-identically |
| C8 | clean-apply simulation | fresh `9113cc188` + `scripts/apply-all.sh` (strict **15/15**, tree == `928852cdc`) + the re-cut block-15 patch (16 commits) + fresh build; record the resulting tree |
| C9 | serving | `--parallel 4` works (the RDNA3_5 V3 fix), gfx1151 pass if the hardware is available (`wip/strix-halo/GATE-2026-09-10-block15-rdna35.md` is the template) |
| C10 | reference discipline | **`GGML_CUDA_ALLREDUCE=nccl` is NOT a bit-identical reference under `-sm tensor`** (2026-09-11 HEADER correction in `AGENTS.md`: the internal AR always BF16-round-trips while NCCL reduces small tensors in FP32). Use a known-good build or the probe |

### 10.4 Performance validations (re-baseline in-session)

| # | check | reference |
|---|---|---|
| P1 | prefill A/B, interleaved same-binary, pp20480/ub2048 | V3 −1.28 % (4B) / +0.28 % (27B); V4 a further −1.85 % (4B) / −1.72 % (27B); V5 0.2–2.4 % by prompt length. Re-measure, do not quote |
| P2 | decode | V3/V4/V5 "within noise" — against **current** baselines: MoE 1 GPU tg128 **101.52** / pp512 **4858.6** (post-MMID-fix), dense 27B 2-GPU tensor tg128 **32.00** / pp512 **2033.5** |
| P3 | MTP | 27B (0.76744 historically), qwen4exp (0.44262 historically), MoE; re-measure the baseline in the same session (see C5) |
| P4 | reserves | the MiB numbers per model/gate (the block's whole point) |
| P5 | switches | if touched: `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` costs −3.0 % MoE decode (tg 101.5 → 98.5); V4/V5 opt-in cost as documented |

### 10.5 Landing procedure (when the validations pass)

1. Work in a canonical fork rebuilt at **`9113cc188`** (or reuse `/tmp/canon-llama`
   at `389c5341f`, tree `928852cdc`, ROCm `/opt/rocm-7.14-gfx1201`, build dir
   `build-base`).  Block SHAs 00-14: `1c7ab0e89`, `aa4108b9d`, `6e81ed5ed`,
   `4dc962aa9`, `03d004517`, `70f330aed`, `d2fc2cb34`, `110b5391d`, `5ea46d1b2`,
   `29880b1e4`, `33a1e5f27`, `f4e75a30a`, `cac14423e`, **`855515420`** (block 13,
   amended twice 2026-09-11), **`389c5341f`** (block 14).
2. Apply `block-15-campaign-wins.patch` on top, squash to one block-15 commit,
   and re-cut the patch file: `From <new sha>`, `[PATCH 16/16]`, the
   `fattn-common.cuh` pre-image hash, body otherwise byte-identical.
3. Record in this directory: the new base (`389c5341f`/`928852cdc`), the new
   block-15 SHA, the resulting tree, and the sim tree; refresh the README/HANDOVER
   numbers and the "14-block"/"14/14" references (they now read 15/15 + block 15).
4. **Promotion (only on the maintainer's go-ahead, beta window closed):** move the
   patch to `patches/0015-…`, make `apply-all.sh` a 16-block flow, and sweep the
   tip/tree + block counts in `AGENTS.md`, `MANIFESTS.md`, `README.md`,
   `BASELINE.md`, `TODO.md`, `WORKLOG.md`, `scripts/make-patches.sh`, plus a new
   WORKLOG entry (never edit the dated 2026-09-10 records in place).
5. Commit and push to **this repo's `origin` only**.

### 10.6 Outcome (2026-09-11) — DONE

**Re-cut:** block 15 = **`fe4f55278`** (tree **`ffe197e2f`**, parent `389c5341f`); the re-cut patch
replaced `block-15-campaign-wins.patch` in this directory.  **Clean-apply:** fresh `9113cc188` +
`apply-all.sh` → strict **15/15**, **0 whitespace warnings**, tree `928852cdc`; + the re-cut patch →
16 commits, tree `ffe197e2f`.

**Every 2026-09-10 claim reproduced.**  The full tables are in `README.md` (status block + §6/§7);
the headlines:

* **Reserves — every number to the last decimal** (27B 1920.3284/880.3360 → 1121.1252/81.1329; 4B
  1800.3284/840.3360 → 1001.1252/41.1329 → 257.1252/41.1329; E4B 1887.3517→1078.1740→452.1740; 31B
  2753.3517→1942.1759→718.1759; qwen4exp 6690.3987/1262.6954 → W1 4450.3987 → W2-only
  5491.3909/63.6876 → **3251.3909/63.6876** with indexer KV 956.26 → **318.76**; bf16/V5 4B
  968.8596→256.8596 = the f16 cost, 27B 1072.8596→488.8596, E4B 1062.8927→404.8927, 31B
  2068.8947→716.8947).
* **The §10.2 risk is cleared**: the 27B width probe reproduces the delivered reference hashes exactly
  (1 GPU `4089b4d4`, 2-GPU tensor `a4817ee6`, 3-GPU tensor `91434ea9`; `W=9` `72af52db`/`b059daa6`/
  `bc3faabd`), so `n_max <= 7` holds and block 15 changes no FA numerics; **V4/V5 on == off
  bit-identically**, so the operand staging does not perturb the split.
* **Coherence byte-identical** across gates (4B, both SWA gemmas, 27B short **+ 40k**, qwen4exp).
* **MoE asterisk intact**: `ac8825358d9adfda`/`bd138ad2326fbbf2`, and
  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` collapses both to `bd138ad2326fbbf2` — identical on both builds.
* **MTP**: 27B 0.90789 and MoE 0.58378 identical on both builds; qwen4exp 0.47826 vs 0.50000 is the
  documented layout sensitivity (its raw logits are bit-identical across builds).
* **Op suites**: `FLASH_ATTN_EXT` 7859/7859 ROCm0 + 7859/7859 CPU (6 derived cases), `GATED_DELTA_NET`
  OK, `test-alloc`/`test-batch-alloc` clean, W4 round trip 16.00 → (revert) 56.00 → 16.00 MiB.
* **Cost**: 4B prefill V3 −1.6 % / V4 −1.75 %, decode flat; 27B V3 −0.3 %, decode flat; headline vs
  the delivery build 27B pp512 −1.7 % / tg128 flat, MoE pp512 −0.3 % / tg128 −1.25 %.
* **gfx1151: not re-run** (no such hardware on this host — 3× gfx1201 + a gfx1036 iGPU); the re-cut
  touches no gfx1151-relevant code.

### 10.7 Follow-ups found by the revalidation (PRE-EXISTING, not block-15 regressions)

Brief + evidence + repro commands: **`../../wip/kv-quant-purity-followups/README.md`**; summary in
`README.md` §7.

* **F1 — quantized-KV width purity.**  `q8_0/q8_0` and `q4_0/q4_0` break the dense `n_max <= 7`
  guarantee (`W=1 == W=2`, then `W=3..8`); text level on the 27B: plain `8ed58aa9` (1330 chars) vs
  `n_max 3 == n_max 7` `da56855b` (1406 chars).  f16, bf16, q4_1, q5_0, q5_1 and iq4_nl are pure.
  The impure set is exactly the two types with a *fast native* both-quantized FA path (>7700 t/s);
  everything else stages through F16 and is ~3.4x slower.  Mixed K/V types are not a usable control
  (2–3.6x slower, different path).
* **F2 — qwen4exp width purity.**  The fused sparse QSA path is not width-invariant
  (`W=1` `dcf1ae66…` != `W=3` `1c801d63…`), identically on both builds; its acceptance gate still
  passes.  Either fix it the block-00 way or document the exemption.
* **F3 — sub-q8_0 quant parity.**  q4_1/q5_0/q5_1/iq4_nl are pure and 1800–2400 MiB but run at
  2197–2293 pp512 / 56–64 tg32 vs 7713–7838 / 95–99 for the native ones.  They have no native FA
  path; extending block 15's own `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}` type-code design to them is the
  obvious route.  **iq4_nl is the same size as q4_0 (1800 MiB) and is pure** — a native iq4_nl would
  obsolete q4_0.  Any new native path must be built **width-invariant**.
* **Decision (maintainer, 2026-09-11):** differing K and V cache *types* are **rejected as an accepted
  limitation** — mixed pairs are always 1.7–3.6x slower than the same-type equivalent and never
  smaller.

### 10.8 What not to do

- **Do not fold block 15 into `patches/`** before the maintainer's go-ahead — it
  is a beta, and `patches/` is the 15-block delivery.
- Never apply anything to `~/llama.cpp`'s remotes; the fork checkout is
  disposable and nothing is ever pushed from it (`AGENTS.md` pushing policy).
- Do not flip `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` / the V4/V5 defaults to make a
  gate pass: validate the delivery's **defaults**, then the switches.
- Do not treat a text match as evidence *against* divergence, and do not use
  `GGML_CUDA_ALLREDUCE=nccl` as a bit-identical reference.
- Do not edit the dated 2026-09-10 validation records in place — add a new dated
  section/entry (this section is that practice).
