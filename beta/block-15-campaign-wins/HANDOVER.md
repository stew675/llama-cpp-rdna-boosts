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

**Still to do before Block 15 can be cut** (the critical path, §3):

| id | what | expected effect (dense models, ctx 204800, ub 2048, q8_0 KV) |
|---|---|---|
| **V3** | derived kq mask for the plain (non-QSA) attention path: stop materialising the `n_kv × n_tps` F16 mask and its host mirror, derive visibility in the FA kernel from compact per-cell state; **phases 3.1 + 3.2 are both in scope — 3.2 adds SWA as a position bound so sliding-window models are covered too (D7)** | **−800 MiB/GPU VRAM − 800 MiB host** (27B: 1920.33 → ~1120 MiB compute; 4B: 1800.33 → ~1000) |
| **V4** | native quantized K/V in the MMA flash-attn path: dequantize into the shared K/V tiles instead of staging an F16 copy of the whole cache in a global scratch | **−832 MiB/GPU at ub2048, exactly ctx-linear** (the `ggml_cuda_flash_attn_ext_get_f16_extra_data` scratch) |

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

**Where it comes from**: `ggml_cuda_flash_attn_ext_get_f16_extra_data` (`ggml/src/ggml-cuda/fattn-common.cuh`)
appends an F16 copy of K and V after the FA op's own dst whenever the selected kernel needs F16 K/V
(`need_f16_K/V` in `ggml_cuda_flash_attn_ext_get_alloc_size`, `ggml/src/ggml-cuda/fattn.cu:694`); with a
quantized KV cache the MMA path (which prefill selects) always needs them.  Measured: `x16 832.00 MB`
for the 4B at ctx 204800 / ub 2048, **0** with `-ctk/-ctv f16`, exactly ctx-linear.

**Approach**: dequantize **into the shared K/V tiles during staging** (the kernel already stages
`tile_K`/`tile_V` as `half`) instead of converting the whole cache into a global scratch — the type
handlers already exist in the vec path (`ggml/src/ggml-cuda/fattn-vec.cuh` handles quantized K/V
natively), and the tile loaders to adapt are `flash_attn_ext_f16_load_K`/`load_V` in
`fattn-mma-f16.cuh`.  Then no scratch is allocated at all *and* the per-ubatch global conversion pass
disappears.

**Risks / ship rule (D9)**: the global conversion is amortized across all heads and query blocks, while
in-tile dequantization repeats work per tile — so the memory win may cost throughput.  Measure pp20480
at ub 2048/1024 against the pre-V4 build and apply the three-way rule: no regression → **on by default**;
regression → **ship opt-in, default off** (for people who need the last 832 MiB), documented in
`patches/README.md` and the beta gate table; impractical even as opt-in → future work.  Bit-identity is
expected (same dequantization, same F16 operand values) but must be proven by A/B, and the decode path is
unaffected (it uses the vec kernel, which needs no scratch).

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

1. **Merge** W1–W4 + V3 (+V4) into one tree.  W1+W2 are already on the fork tree (10 files,
   uncommitted); W3 overlaps W2 in `src/llama-memory-hybrid-idx.cpp`; V3/V4 bring their own files.  Use
   `git add -A` + `git apply -3` (never `patch -F3`) and resolve conflicts by hand; the fork tree is the
   authority for W1+W2.
2. **Gate everything** (D3 + the V3/V4 gates): `GGML_QSA_SCORE_MEM` (W1), `GGML_QSA_DERIVED_BIAS` /
   `GGML_QSA_DERIVED_VIS` / `LLAMA_QSA_SPARSE_FA` (W2, with the `2|3` diagnostics **stripped**),
   `LLAMA_QSA_KEYS_ONLY` (W3), `LLAMA_KQ_MASK_DERIVED` (V3), the V4 switch.  W4 has no knob (§2).
   Document every gate + default in `patches/README.md` and the beta README.
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
working tree = **10 modified files = W1 + W2 (incl. the `attn_k` guard)**, deliberately uncommitted.
W3/V3/V4 are not applied.  Never commit those 10 files except as the Block-15 commit; never push from
that checkout.

**Campaign patches**: W1 `wip/qwen4exp/qsa-memory/patches/0001-…`, W2 `…/0002-derived-qsa-block-bias.patch`
(regenerated 2026-09-10, 10 files, +557/−80, `git apply --check` clean on a fresh W1 base), W3
`wip/qwen4exp/keys-only-indexer/0001-…`, W4 `wip/arch-independent-memory/patches/0001-…` + its PR copy
`upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch`; the A/B revert `ab/w4-revert.patch`.

**Volatile helpers** (rebuild; `/tmp` may be wiped): `/tmp/bin-pristine` (14 blocks), `/tmp/bin-l2` (W1),
`/tmp/bin-l1` (W1+W2 pre-guard), `/tmp/bin-l1guarded` (= the current tree), `/tmp/bin-l3b` (tree + W4),
`/tmp/bin-keysonly` (W3), `/tmp/bin-l0base|l0c|l0d` (instrumented allocators; source
`/tmp/ggml-alloc.instrumented.c`), `/tmp/master-pr` (a master worktree with W4 applied — disposable),
`/tmp/alloc-leak-repro{,2}`.

**Tools** (in-repo): `wip/qwen4exp/qsa-memory/tools/{model-sweep.sh,mask-scaling.sh,bufsize.sh,ub-sweep.sh,ab-coherence.sh,mtp-ab.sh,l0a-scheddump.sh,peak-ledger.py,view-reuse-ledger.py}`; repro
`wip/arch-independent-memory/repro/ggml-alloc-unused-view.c`.

**Rebuild**: `export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH && cmake --build build-rocm --target
llama-cli llama-bench -j 16`; run with `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0`.

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
Continue in /home/stew675/llama-cpp-rdna-boosts (read AGENTS.md first; its rules override this).
Start with beta/block-15-campaign-wins/HANDOVER.md - it holds the maintainer's decisions (W4 = option C,
a revert patch for A/B; Block 15 includes V3 and V4 and waits for them; kill-switches approved for W1
and W3; exactly one block, no Block 16), the critical path, the state inventory and the open questions.

The work now is V3, per HANDOVER section 3.1, using wip/arch-independent-memory/DERIVED-MASK-DESIGN.md
as the spec and wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch as the worked example
of the derived form.  Phase 3.1 = derive the mask in the prefill/MMA flash-attn path for the plain
cache (cell occupied + sequence membership + causal), keep the packed mask for decode and every
unsupported case, gate it with LLAMA_KQ_MASK_DERIVED (default on), and prove byte-identical same-seed
output on Qwen3.5-4B-Q8_0 and Qwen3.8-27B-Q8_0 at a short and a long prompt, plus the reserve matrix
(expect -800 MiB/GPU compute and -800 MiB host at ctx 204800 / ub 2048), the MTP gate, bench parity and
test-backend-ops FLASH_ATTN_EXT.  Phase 3.2 (SWA as a position bound) is IN SCOPE for Block 15 too, and
must be validated on an SWA model - use /llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf (and its
-MTP variant for the MTP gate) alongside the non-SWA controls.  Then V4 (HANDOVER section 3.2), which
ships on-by-default only if it does not cost prefill throughput, otherwise opt-in default-off (D9).

When every win is in place, merge W1-W4 + V3 + V4, gate them, re-validate as a combined set
(beta/block-15-campaign-wins/README.md sections 4 and 6), finalise BETA-TESTING.md's gate table, and cut
Block 15 - exactly one block, no Block 16.

If V3 3.1 turns out to be more invasive than expected, say so early and fall back to V2 (1-bit packed
mask, DERIVED-MASK-DESIGN.md section 4) - do not build both.

Report at the end: what landed, the gate table with defaults, the measured before/after reserves and
throughput, what was not validated, and the updated state in HANDOVER.md.
```

## 9. Open questions

None outstanding — the maintainer answered all three on 2026-09-10: V3 includes 3.2 (D7), the beta
checklist exists (`BETA-TESTING.md`, D8), and V4 follows the three-way ship rule (D9).  Two things are
decided *at implementation time* rather than now: the exact V4 gate name and its default (measurement
decides on/opt-in), and whether V3 phase 3.3 (M-RoPE) is cheap enough to include — alibi stays on the
packed mask either way.
