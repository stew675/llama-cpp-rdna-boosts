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

### 3.4 The next memory lever: bf16-native MMA K/V (measured 2026-09-10, NOT implemented)

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

**Why this is the *easier* case than q8_0** (and the recommended follow-up):

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

**Block numbering**: if it lands before the Block-15 cut it belongs in Block 15 (same `GGML_CUDA_FA_KV_NATIVE`
gate, one more arm); afterwards the "exactly one new block" rule needs a maintainer decision.

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

## 5. Stage B — Block 15 (V3 and V4 have landed)

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
7. **Promotion** (after the window): `patches/0015-…`, `apply-all.sh`, `make-patches.sh` tip,
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
override everything here).  ALL CAMPAIGN WINS ARE DONE (W1, W2, W3, W4, V3, V4) - this session is the
BLOCK 15 MERGE + CUT, per beta/block-15-campaign-wins/HANDOVER.md section 5.

READ FIRST: beta/block-15-campaign-wins/HANDOVER.md (sections 1, 5, 6, 7 and 9), then
beta/block-15-campaign-wins/README.md (inventory, gate audit, validation protocol) and
beta/block-15-campaign-wins/BETA-TESTING.md (the beta gate checklist).  For V3/V4 detail:
wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md section 4 and
wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md.

STATE: ~/llama.cpp has `rdna-boosts` pristine at e2380eb67 (blocks 01-14) and the 22-file campaign tree
(W1 + W2 + V3 phases 1/2a/2b/2c + V4, validated 2026-09-10) committed on the work branch
`wip/block15-campaign-wins` (= b26ae06f0, checked out, clean); a whole-tree delta patch also exists at
wip/arch-independent-memory/snapshots/fork-tree-W1-W2-V3-V4-2026-09-10.patch.  Resume with
`git checkout rdna-boosts && git merge --squash wip/block15-campaign-wins` (then apply W3 + W4).  W3 and
W4 are separate patches (wip/) that still have to be merged in.  V3 is on by default
(LLAMA_KQ_MASK_DERIVED=0 disables it); V4 is OPT-IN (GGML_CUDA_FA_KV_NATIVE=1 enables it, default off).
Patch snapshots live in wip/arch-independent-memory/patches/ (0001 W4 alloc, 0002 the V3 phase-1
diagnostic - NEVER ship it, 0003/0004/0005 V3, 0006 V4), wip/qwen4exp/qsa-memory/patches/ (W1, W2) and
wip/qwen4exp/keys-only-indexer/ (W3).

DO, in this order (HANDOVER section 5 has the detail):
1. Merge W3 and W4 into the fork tree (git add -A + git apply -3 on conflicts, never patch -F3), strip the
   V3 oracle (LLAMA_KQ_MASK_DERIVED_VERIFY / the 0002-DIAGNOSTIC patch) and any GGML_QSA_DERIVED_BIAS=2|3
   diagnostic, then build.
2. Re-validate the COMBINED tree: the reserve matrix (ctx 204800, ub 2048/1024/512, q8_0) with the
   defaults (V4 off) and with GGML_CUDA_FA_KV_NATIVE=1, on 4B + 27B + gemma-4-E4B/31B + qwen4exp;
   same-seed coherence byte-identical on every model with the gates flipped (4B, 27B, both gemmas,
   qwen4exp) at a short and a long prompt; the MTP gate (acceptance >= ~0.45 and unchanged) with V4 off
   and on; the W1xW2xW3 gate combinations; test-backend-ops FLASH_ATTN_EXT on ROCm0/CPU.
3. Cut the block: a single 15th block commit on the fork (never pushed), scripts/make-patches.sh (verify
   blocks 01-14 come out byte-identical), scripts/apply-all.sh 14 -> 15, MANIFESTS.md / README.md /
   WORKLOG.md / BASELINE.md, rdna-boosts-all.patch, then a clean-apply simulation in a fresh worktree at
   9113cc188 + build + coherence.
4. Stage beta/block-15-campaign-wins/block-15-campaign-wins.patch + the promotion record (beta start date,
   gate table with defaults, validation results) and run the upstream-drop check (HANDOVER section 5.6).
5. OPTIONAL, if the maintainer wants it before the cut: extend the V4 gate to bf16 (HANDOVER section 3.4
   - the MMA staging scratch costs a bf16 user ~712 MiB/GPU at ctx 204800 and the in-place conversion
   should keep the cp_async pipeline, so it can likely ship on by default).
6. Report: the gate table with defaults, the measured before/after reserves and throughput, what was not
   validated, and the updated state.

DO NOT: push anything from ~/llama.cpp, fold wip/ content into patches/ beyond this agreed Block 15, or
touch archive/work/.
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
