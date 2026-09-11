# rdna-boosts patch set (delivery)

15 patches (block 00 structural fixes + blocks 01-14) against llama.cpp master `9113cc188`
("ggml : fix msvc+clang ggml_vld1q_u32 (#28284)"; re-based 2026-09-08 from
`050dde50c` ("hexagon: add RELU and LEAKY_RELU ops (#28585)"), itself
re-based 2026-09-07 from `465e49b9c`, re-based 2026-09-06 from `9cffdcc80`,
re-based 2026-09-02 from `0eadefebd`; on the 2026-09-08 re-base block 06's
functional delta was dropped — upstream itself reverted #24233 in #28604 the
same day, matching its end state — and the block now carries only the
host-buffer rationale marker comment (see the block-06 note below); block 14's
quantized-KV tensor-split gate merged additively with upstream #28390's
single-device `SPLIT_MODE_TENSOR` warn; block 08 amended 2026-09-11 with the decode/verify FlashAttention kernel-family fix
(F1: a quantized K/V cache used VEC at `n_q <= 2` and TILE from `n_q = 3`, so plain decode disagreed
with spec-draft-mtp verify — see the block-08 notes below), and again 2026-09-11 with the **quantized
KV-type enablement** (`q4_1`/`q5_0`/`q5_1` become first-class FlashAttention cache types — the
`GGML_CUDA_FA_ALL_QUANTS`-only types are enabled unconditionally, with their three diagonal vec
instances — so they stop disabling flash attention for the whole context; see the block-08 notes
below and `../GREEDY-PURITY.md` §20); block 12 amended 2026-09-04 with the runtime
NCCL-failure fallback (issue #13, see the block-12 notes
below); block 13 amended 2026-09-02 with two MTP regression fixes and
2026-09-05 with the RDNA3.5 (Strix Halo, gfx1151) + RDNA3.0 (gfx1100)
fused-MoE-MMQ gate relaxations, and 2026-09-11 with the F2 cause-2
**decode/verify band-uniformity** fix (the per-type mmvq caps are floored at
`MMVQ_MAX_BATCH_SIZE` and `mul_mat_vec_q_moe` is sized at the band, so
`W = 1..8` is bit-identical — **+14-26 %** at the verify widths) and
2026-09-11 with the **fused shared-expert epilogue band** (the decode-only
`ne[1] == 1` gate now serves the whole `n_tokens <= MMVQ_MAX_BATCH_SIZE` band
— `W = 1..8` bit-identical, and MoE `draft-mtp` acceptance 0.51 -> 0.82) — see the
block-13 notes below; block 14 amended 2026-09-11 with the **QSA decode-arm
band** (the dense arch-policy arm was gated `n_tokens == 1`, so a W=1 decode and
an n-token verify took different attention regimes above the indexer selection
width; the arm now serves the whole decode/verify band, making
`plain == draft-mtp` byte-identical for `n_max <= 7`), and again 2026-09-11 with the
**QSA-vs-KV-type arm gate** (the fused sparse QSA op reads the cache natively for f16/bf16/q8_0 only;
with any other quantized cache the graph now takes the dense masked path instead of building an op the
backend cannot split — which is what aborted the meta splitter on qwen4exp + `-sm tensor`, for `q4_0`
as well) and the **tensor-split gate narrowing** to the types that really have a native FA read path
— see the 2026-09-11 block-14 amendment section below, the MTP
baseline gate in
`../benchmarks/mtp-adaptive-methodology.md`, the Strix record in
`../wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`, and
the gfx1100 record in
`../wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`; block 14
amended 2026-09-07 with the QSA quantized-KV decode gate + the
derived-cache pool gate — see the block-14 notes; block 13
amended 2026-09-08 with the moe_weighted_reduction float4 remainder fix (issue #19); block 14
amended 2026-09-08 with the MUL_MAT_ID pair-fusion layout gate (issue #18) — see the
2026-09-08 fixes section and the block-13/14 notes below; block 14
amended 2026-09-08 with the compiler-warning cleanup (Vulkan/clang-16 + ROCm
host builds), 2026-09-08 with the qwen4exp tensor-split backend gate
(HIP-only) and 2026-09-08 with the quantized-KV tensor-split gate
(`q4_1`-family KV cache types aborting under multi-GPU `SPLIT_MODE_TENSOR`;
an upstream bug — vanilla `050dde50c` reproduced it too) — see the block-14
notes below); block 01 refreshed 2026-09-09 to the llama.cpp PR #27210
review head `d236d41a2` (still one squashed block; blocks 02-14
content-identical on the regeneration — see the 2026-09-09 block-01
refresh section below); block 00 was added 2026-09-10 (see the
2026-09-10 block-00 section below) and the kernel-side masked-V fixes for
freed flash-attention cells were re-homed the same day: the host-side
`zero_freed` row zeroing (added 2026-09-09, gfx1151-only) stays REMOVED
(`llama-kv-cache.{cpp,h}` are back to the upstream state), the Vulkan
`flash_attn_cm1.comp`/`flash_attn.comp` (dead columns never read V) fixes
now live in block 00, and the HIP `fattn-tile.cuh` (packed-bf16 PV) +
`fattn-mma-f16.cuh` (masked-V rows in staged shared tiles) fixes now live
in block 03 (they sit on the native-BF16 FA path block 03 introduces);
block 14 carries none of them; **block 15 (the attention-memory campaign) is NOT a delivery patch** -- it is
staged in `../beta/block-15-campaign-wins/` and applied manually on top of
the 15-block tree):

| patch | content |
|---|---|
| `0000` | **structural and architecture fixes** — FA small-batch KV-split width invariance (issue #25: decode and every speculative verify width now reduce identically, so greedy output no longer changes with the MTP draft length) + Vulkan masked-V/freed-cell fixes (dead columns never read V). Added 2026-09-10; this is the base every other block applies on top of. |
| `0001` | adaptive MTP draft depth | **refreshed 2026-09-09 to the upstream PR #27210 review head** (`d236d41a2`; review-round feedback-handling, option validation + docs) — see the 2026-09-09 block-01 refresh section below.
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA; + MTP long-prefill chunked-prefix + sequential K-tail, PR #9) | **amended 2026-09-06 with the gfx11 NW16 scan retune** (gated_delta_net_chunked_bf16_gfx11.cu, fork 376f02aa0); **amended 2026-09-11 with the K-independent whole-batch chunked prefill** (gated_delta_net.cu; no sequential tail, `GGML_CUDA_GDN_ALIGN_BOUNDARY` gate + its two K-dependent branches **removed**; + the `llama_memory_recurrent` rollback-boundary guard).
| `0003` | BF16 KV cache + native-BF16 flash-attn | **amended 2026-09-10 with the HIP masked-V/freed-cell fixes** (moved here from block 14 on 2026-09-10 — they sit on the native-BF16 PV staging this block introduces): `fattn-tile.cuh` (packed-bf16 PV) + `fattn-mma-f16.cuh` (masked-V rows in staged shared tiles). |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf | **amended 2026-09-06 with the RDNA WMMA (256,256,64) config row** (fattn-mma-f16.cuh, fork e7eecb369).
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results | **amended 2026-09-06 with the scale+unary fused kernel** (unary.cu/cuh, fork f5ac11903). | **amended 2026-09-07 with the mul_mat+add through-view shape guard (PR #15, DanoPTT)** — see the 2026-09-07 re-base section.
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL |
| `0012` | **hybrid HIP all-reduce (block 12)** - the custom internal AR; hybrid dispatch; RDNA4-only gate; runtime NCCL-failure fallback (amended 2026-09-04, issue #13); **amended 2026-09-11 - the small/large crossover is now width-safe** (2-device `32768` -> `131072` elements; see the block-12 notes) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split (block 13)** - prefill fused expert MMQ (RDNA4 + RDNA3.5 + RDNA3.0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K) + decode item-split; **amended 2026-09-02 with the two MTP regression fixes** (mmvq ksplit dispatch for verify batches; rms_norm-fold gate for multi-token MoE); **amended 2026-09-11 with the dense ncols==1 ksplit alignment** (dense `MUL_MAT` rows always ksplit for every K so single-token decode is row-identical to the 2..8-token verify batch; `MUL_MAT_ID`/MoE kept the item-split at that point — superseded by the second 2026-09-11 amendment below) — see the block-13 notes below; **amended again 2026-09-11 with the MoE `MUL_MAT_ID` dispatch fix** (all `MUL_MAT_ID` now use the dedicated MoE kernel, completing what the dense fix left open — `ncols_dst == 1` previously took the dense ksplit kernel with an ids gather; **+6.2% MoE decode**) **and the `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` kill-switch** (the decode-only fused shared-expert epilogue is not bit-exact with the unfused chain — the accepted MoE residual; see the block-13 notes); **amended 2026-09-05 with the RDNA3_5 gate relaxation** (gfx1151 validated; see the block-13 notes) and **with the RDNA3_0 gate relaxation** (gfx1100 validated; see the block-13 notes); see block 13 notes below | **amended 2026-09-06 with the model-neutral Strix MoE mmq folds** (fork 1da01fa67 routed-compact, 7a6a2e97b swiglu-input quantize, f33ffaca7 mwr float4, 6d457634e split_j+Q8_0 rows, 0a3a2b498 quantize chunk, 6a80b695c mul_mat_q_pair kernel, b31940a5e weighted-down mmvq kernel, f5ac11903 scale-unary window). Fold trail: wip/archive/qwen4exp/README.md. | **amended 2026-09-08 with the moe_weighted_reduction float4 remainder fix (issue #19)**; **amended 2026-09-11 with the F2 cause-2 decode/verify band-uniformity fix** (upstream's per-type mmvq caps are floored at `MMVQ_MAX_BATCH_SIZE` and `mul_mat_vec_q_moe`'s launch bound is sized at the band, completing block 13's own "decode == verify" invariant for the whole band — `W=1..8` bit-identical, **+14-26 %** at the verify widths) — see the block-13 notes below; **amended 2026-09-11 with the fused shared-expert epilogue band** (the decode-only `ne[1] == 1` gate now serves the whole `n_tokens <= MMVQ_MAX_BATCH_SIZE` band, with the kernels made token-generic and `nwarps` pinned to the single-token reduction order — `W=1..8` bit-identical, MoE `draft-mtp` acceptance 0.51 -> 0.82 with 167.3 t/s vs plain 96.9 on Qwen3.6-35B-A3B; the asterisk is gone) — see the block-13 notes below.
| `0014` | **qwen4exp support (block 14)** - Qwen3.8-Flash-Next model support promoted from `beta/qwen4exp` (fork delta `c261553a1..dd4301fb4`, squashed + re-based to `050dde50c` 2026-09-07): QSA sparse FA (DEFAULT) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head support, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy; see block 14 notes below | **amended 2026-09-07 with the QSA quantized-KV decode gate** (the fused indexer ops read the raw cache natively in F32/BF16/F16 only; a quantized indexer-key cache, e.g. `--cache-type-k q8_0`, previously aborted `ggml_indexer_fill` at context init — those caches now fall back to the per-op chain) | **amended 2026-09-07 with the derived-cache pool gate** (the F32 block-vector pool is now allocated only when the derived cache is enabled *and* the indexer keys are unquantized — no more dead ~100 MiB buffer + no-op fill launches otherwise) | **amended 2026-09-08 with the MUL_MAT_ID pair-fusion layout gate (issue #18)** — see the block-14 notes below. | **amended 2026-09-08 with the compiler-warning cleanup** — see the block-14 notes below. | **amended 2026-09-08 with the tensor-split backend gate (HIP-only)** — see the block-14 notes below. | **amended 2026-09-08 with the quantized-KV tensor-split gate** — `q4_1`-family KV cache types (`q4_1`/`q5_0`/`q5_1`/`iq4_nl`) abort at graph reserve under multi-GPU `SPLIT_MODE_TENSOR` (upstream bug, also on vanilla `050dde50c`); now rejected at context creation with a clear error when the Meta device is in use — see the block-14 notes below. | **amended 2026-09-09 with the gfx1151-only freed-cell KV-zeroing gate** — the seq_rm/seq_keep/clear row zeroing (strix-port aad5adb08f masked-column guard for the gfx1151 WMMA f16 `x+(-0.0)` inexactness) now enables only when a KV buffer device is gfx1151 (env `LLAMA_KV_ZERO_FREED` overrides); everywhere else pre-block-14 behavior (no per-free GPU memsets) is restored — see the 2026-09-09 block-14 amendment section below. | **amended 2026-09-10: the freed-cell host zeroing is removed and the kernel-side masked-V fixes were re-homed** — `llama-kv-cache.{cpp,h}` are the upstream state (no `zero_freed`/env/GPU memsets); the Vulkan `flash_attn_cm1.comp`/`flash_attn.comp` fixes live in block 00 and the HIP `fattn-tile.cuh`/`fattn-mma-f16.cuh` fixes live in block 03, so block 14 carries none of them — see the 2026-09-10 block-00 section below. | **amended 2026-09-11 with the hyper-connection decode/verify band fix** — `ggml/src/ggml-cuda/hc-mix.cu` + the `src/models/qwen4exp.cpp` gates served `nt == 1` only, so a 1-token decode used the fused `HC_MIX`/`HC_COMBINE` chain while an n-token verify batch used the unfused chain (the "F2" divergence: plain decode != `draft-mtp`).  Both ops now serve the whole band `1 <= nt <= 8` (`HC_FUSED_MAX_TOKENS`), taking the token from `blockIdx.y` and reading `inject` with its own view stride, so every token in the band runs the per-token kernel sequence a single-token decode runs and W=1 is byte-identical to the pre-fix build.  qwen4exp is thereby width-pure for `--spec-draft-n-max <= 3` (f16/bf16 KV; plain == `draft-mtp` text, f16 acceptance 0.500 -> 0.76744, MTP generation 63.3 -> 79.9 t/s); the `W >= 5` grouping is **cause 2**, shared with the `q8_0`/`q4_0` KV impurity, and remains open — see `GREEDY-PURITY.md` §13 and `wip/kv-quant-purity-followups/README.md` (F2).  A `<= 8`-token *prefill* chunk also takes the fused path (indistinguishable from a verify batch). | **amended 2026-09-11 with the QSA decode-arm band** — the arch policy's dense decode arm was gated `n_tokens == 1`, so above the indexer selection width (`indexer_top_k + r - 1` = 2051 on qwen4exp, reached at `n_kv = 2304`) a W=1 decode ran dense while the n-token verify batch fell through to the sparse top-k selection, and `plain != draft-mtp` in text.  The arm now serves the whole decode/verify band (`QSA_DECODE_BAND = 8`, the `n_max <= 7` purity band); prefill keeps the sparse selection.  `plain == n_max 3 == n_max 7` = `804de0576868` (f16 KV) and `plain == n_max 3` = `75d8530c5bb1` (q8_0 KV); MTP `n_max 3` pos-1 acceptance 0.615 with 63.9 t/s vs plain 50.1.  Two width-dependences remain in the *sparse* regime (gfx1151 above 64K) — see `../GREEDY-PURITY.md` §16-18 and the 2026-09-11 block-14 amendment section below. |

## Apply (fresh checkout at the fork point)

```bash
git checkout 9113cc188         # or: git apply each patch on a matching tree
git am patches/0000-*.patch patches/000[1-9]-*.patch patches/001[0-4]-*.patch
```

(`git am` for the whole 15-patch series - plain `git apply` of the
concatenated series was observed to silently drop hunks; use `git am`.
`scripts/apply-all.sh` runs a strict `git am` first and, if that fails
on a drifted base, aborts and retries the series with `git am -3`,
warning that merged hunks may differ from the canonical tree.)

The set is **whitespace-clean**: applying produces no git whitespace
warnings (verified 2026-08-29 after the whitespace-clean regeneration,
re-verified 2026-09-01 on the `0eadefebd` re-base, re-verified 2026-09-01
with block 13 on the 13-patch series, re-verified 2026-09-02 on the
`9cffdcc80` re-base, re-verified 2026-09-02 after the block-13 amendment,
re-verified 2026-09-05 after the block-13 RDNA3_5 gate relaxation,
re-verified 2026-09-05 after the RDNA3_0/gfx1100 fold, re-verified
2026-09-06 on the `465e49b9c` re-base, re-verified 2026-09-07 on the
`050dde50c` re-base with the 14-patch set, re-verified 2026-09-08 after
the block-14 warning-cleanup amendment), re-verified 2026-09-09 after the
block-01 refresh (strict 14/14 `git am`, zero whitespace warnings, applied
tree == fork tip `0f2b7a4e1`), re-verified 2026-09-09 after the block-14
gfx1151-zeroing-gate amendment (strict 14/14 `git am`, zero whitespace
warnings, applied tree == fork tip `27485f1ca`), re-verified 2026-09-10
after the block-14 kernel-side masked-V amendment (strict 14/14 `git am`,
zero whitespace warnings, applied tree == fork tip `ff2b35f49`; blocks
01-13 patch bodies byte-identical to the previous regeneration), and
re-verified 2026-09-10 on the 15-block (block 00 + 01-14) regeneration
(strict **15/15** `git am`, zero whitespace warnings, applied tree == fork
tip `505637d6e`; the net tree is unchanged from the 14-block tip, only the
home of the masked-V fixes moved).
**Block 15 (the attention-memory campaign) is staged in
`../beta/block-15-campaign-wins/`, not delivered** (the 2026-09-10 Strix
Halo/gfx1151 pass validated it and fixed two V3 issues there; see the beta
README and the WORKLOG entry).

## 2026-09-10 block-00: structural and architecture fixes

`patches/0000` is the first block, applied directly on `9113cc188` before
everything else.  It holds baseline-level fixes that later blocks build on:

1. **FA small-batch KV-split width invariance (issue #25).**
   `launch_fattn`'s non-stream-K `parallel_blocks` heuristic maximises wave
   efficiency over `ntiles_dst = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3]`
   with `ntiles_x = ceil(Q->ne[1]/ncols1)`, so a speculative verify batch
   (`n_q = n_draft+1`) picked a different KV split than single-token decode
   (`n_q = 1`).  Different splits group the fp32 online-softmax/PV partials
   differently, the logits drift in the last bits and greedy near-ties flip —
   `--spec-draft-n-max 2` and `4` then produced different text (reported by
   1337hero, issue #25; deterministic within an arm, and reproduced on 1-, 2-
   and 3-GPU gfx1201).  The fix evaluates the heuristic as if `n_q == 1` for
   every `n_q <= 8`; `n_q > 8` (prefill) is unchanged.  Plain decode is
   byte-identical (the fix moves only `n_q >= 2`), and the MTP acceptance gate
   is unchanged.
2. **Vulkan masked-V / freed-cell fixes** — `flash_attn_cm1.comp` and
   `flash_attn.comp` never read V for dead columns.  These are baseline
   shaders, hence the structural block.

The **HIP** masked-V fixes are **not** here: the `fattn-tile.cuh` half uses
the native-bf16 PV staging (`V_k0`/`KQ_k`/`nv_bfloat162`) introduced by
block 03, so both HIP halves were moved into **block 03** on 2026-09-10 (the
earliest block that exercises the leaking code), and block 14 no longer
carries them.  See the WORKLOG entry for the validation record.

`git format-patch --start-number 0` numbers this block `0000` so the file
prefix matches the block number (subjects read `[PATCH 00/14]`…`[PATCH
14/14]`).

## Block 15 (attention-memory campaign) — STAGED in `beta/`, NOT delivered

Block 15 is **not part of the delivery**.  It is staged as
`../beta/block-15-campaign-wins/block-15-campaign-wins.patch`
(V3 derived kq mask, V4/V5 native q8_0/bf16 K/V, W1-W3 QSA
memory, W4 ggml-alloc unused-view release, each with an env A/B
gate) and is applied manually on top of the 15-block tree, pending
the maintainer's promotion go-ahead.  Its gate table, validation
record and the 2026-09-10 Strix Halo (gfx1151) pass live in
`../beta/block-15-campaign-wins/README.md` and
`../wip/strix-halo/GATE-2026-09-10-block15-rdna35.md`; the dated
WORKLOG entries carry the history.

## 2026-09-11 block-14 amendment (third): the QSA arm respects the KV type + the tensor-split gate

Two changes, both needed before `q4_1`/`q5_0`/`q5_1` could be offered as KV cache types under
multi-GPU `-sm tensor`.

**1. The QSA-vs-dense arm now depends on the cache type.**  `build_attn_qsa`
(`src/models/qwen4exp.cpp`) chooses between the fused sparse op (`ggml_flash_attn_qsa`, the default)
and the dense masked path (`LLAMA_QSA_SPARSE_FA=0`), and that choice ignored the KV type.  The fused
kernel reads the cache rows natively for **f16/bf16/q8_0 only**
(`ggml_cuda_flash_attn_qsa_supported()`), so with `q4_0`/`q4_1`/`q5_0`/`q5_1` the graph still built a
`GGML_OP_FLASH_ATTN_QSA` the backend could not run — and under `-sm tensor` that op was never split
across the tensor-parallel devices while the attention gate still was, so the meta splitter hit
`GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` (`ggml-backend-meta.cpp:538`) on
`MUL name=attn_gated-<il>`.  This was **pre-existing**, not a consequence of the enablement: `q4_0`,
which the previous gate allowed, aborted the same way.  `qsa_sparse` now also requires a QSA-native
cache type, so those types take the dense masked path (exactly `LLAMA_QSA_SPARSE_FA=0`, which remains
the A/B knob, and which is a no-op numerically for the gfx1201 decode band — the arch policy was
already dense there, and the `q4_1` probe hash is identical with and without it).  `LLAMA_QSA_OFF=1`
(a plain-dense reference) and `-sm layer` both avoided the abort too, which is how the mechanism was
localised.  Consequence to keep in mind: with a quantized cache on qwen4exp the **prefill** attention
of the indexer layers runs masked-dense instead of the fused sparse op (the decode band was already
dense there by arch policy, so its logits are unchanged — the `q4_1` probe hash is identical with and
without `LLAMA_QSA_SPARSE_FA=0`).  Restoring the fused sparse prefill for those types means teaching
`fattn-qsa` to read them (the same work item as F3 step 2's `iq4_nl`), not reverting this gate.

**2. The tensor-split gate is narrowed.**  `llama_init_from_model` (`src/llama-context.cpp`) rejects,
for `SPLIT_MODE_TENSOR` with a Meta device, any quantized KV type outside `{q4_0, q8_0}` — it cannot
ask the backend (it runs before any backend probe), so it carried a hardcoded list.  The list is now
the helper `llama_kv_type_has_native_fa()` (f32/f16/bf16/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`, mirroring
the backend predicate), the error message lists the allowed set, and `iq4_nl` (and any future
unlisted type) keeps the clean error instead of an abort.  Verified per type on 3-GPU `-sm tensor`
(27B, qwen4exp): all six types create a context and run, `iq4_nl` is rejected with the message.

## 2026-09-11 block-14 amendment: the QSA decode arm is band-uniform

qwen4exp was still not `plain == draft-mtp` in *text* after the hyper-connection
band fix (earlier the same day, `HC_FUSED_MAX_TOKENS`) and after the two block-13
band fixes: `--spec-type none` and `draft-mtp --spec-draft-n-max 3` / `7` shared
only ~100 of ~700 generated characters (f16 KV, 3-GPU `-sm tensor`).  The
sparse-FA kernel was already exonerated; localisation (`LLAMA_QSA_OFF=1` is
byte-identical, `LLAMA_QSA_SPARSE_FA=0` is not) put it in the QSA **indexer**
machinery.

Mechanism (`build_layer_attn`, `src/models/qwen4exp.cpp`): the indexer picks one
of three arms, and the middle one — the arch policy's dense decode arm — was
gated `n_tokens == 1`:

    if (shortcut && n_kv <= width)                                      // dense, store keys
    else if (qsa_dense_decode_until > 0 && n_tokens == 1 && n_kv < ...)  // dense policy arm  <-- width-dependent
    else  top_k = build_qsa_top_k(...)                                   // sparse selection

`width = indexer_top_k + r - 1` = 2051 on qwen4exp (2048 + 4 - 1).  At the first
decode graph the indexer cache held `n_kv = 2304 > 2051`, so arm 1 no longer
applied and the `n_tokens == 1` gate split the two runs: `--spec-type none`
(`n_tokens=1`) took the **dense** arm, `draft-mtp` (`n_tokens=4`) fell through to
the **sparse top-k selection**.  An arm trace proved it (one line per indexer
layer per graph *build*, so it shows the CUDA-graph rebuilds; both runs are
identical for the first 11 builds and split at the first decode graph).  The
trace is kept at `wip/kv-quant-purity-followups/tools/qsa-arm-trace.patch`, and
that is also why the single-step width probe never saw the bug: at
`P <= 2048` the cache stays below `width`.

Fix: `QSA_DECODE_BAND = 8` (the `n_max <= 7` purity band, the same constant class
as `HC_FUSED_MAX_TOKENS`) and the arm takes `n_tokens <= QSA_DECODE_BAND`.
Prefill is untouched (`n_tokens` is far above the band, so it keeps the sparse
selection — the arch policy "prefill is untouched: QSA always"), and on gfx1201
(`qsa_dense_decode_until = 1 << 62`) decode is now dense at every width, which is
what the arm's own comment describes.  Post-fix the arm trace shows
`n_tokens=4 n_kv=2304 -> arm 2` in *both* runs, and decode never builds a top-k
selection.

Measured: `plain == n_max 3 == n_max 7` = `804de0576868` (704 chars, f16 KV) and
`plain == n_max 3` = `75d8530c5bb1` (660 chars, q8_0 KV); MTP gate (f16, `n=96`):
`n_max 3` pos-1 acceptance **0.615** with 63.9 t/s vs plain 50.1 (**+28 %**),
`n_max 7` pos-1 **0.618** (its lower aggregate is the fixed-depth-7 over-drafting
decay, which the gate's rules explicitly exclude).  The plain stream moves with
the fix (658 -> 704 chars): the shared 4-token non-decode shape at `n_kv = 2304`
also moves onto the dense arm — the same "the band must take one path" trade as
the hyper-connection fix (the chosen value is the policy-consistent dense one).

Two width-dependences remain in the **sparse** regime (`../GREEDY-PURITY.md`
§18): the fused indexer score's "byte-identical" claim is measurably false and is
itself `n_tokens == 1`-gated (reachable on gfx1201 only with
`LLAMA_QSA_DENSE_DECODE_UNTIL=0`, but the **default** path on gfx1151 above its
64K crossover), and a residual split survives even with one arm.  RDNA4/gfx1201's
default regime is complete.

## 2026-09-10 block-14 amendment: kernel-side masked-V fixes replace the host zeroing

Freed/stale flash-attention cells are now handled **in the kernels**, and
block 14 no longer touches `llama-kv-cache.{cpp,h}` at all (both files
are byte-identical to the upstream state at the fork point).  The
host-side `zero_freed` row zeroing added 2026-09-09 is **removed** — no
member, no env `LLAMA_KV_ZERO_FREED`, no per-free GPU memsets — and the
three kernel fixes below are folded into block 14 instead.  All three
are **unconditional in their kernel paths** (no arch/env gating): they
are generic correctness fixes for any masked column whose cell is
stale/freed (batch serving, KV eviction), active by default on every
device:

- HIP `fattn-tile.cuh` (packed-bf16 PV path): zero the per-warp V
  register copies of rows whose P is +0.0 across the warp's columns
  (fully-masked rows) before the bf16 dot.
- HIP `fattn-mma-f16.cuh`: after each V-tile slice is staged in shared
  memory, zero the rows the mask tile marks blocked (-inf) for every
  query column of the block; one extra uniform barrier, masked path
  (`ncols2 > 1 || mask_h`) only; compile-time excluded for the
  `V_is_K_view` and NVIDIA-swizzled (`swz_V`) paths.
- Vulkan `flash_attn_cm1.comp` (per-column liveness: dead columns keep V
  at +0.0) + the `flash_attn.comp` scalar path (skip the V load for dead
  columns).

Background: the root cause was a gfx1151/Strix-Halo WMMA f16
`x + (-0.0)` inexactness — a fully masked column still accumulated the
sign of whatever V its cell last held, so request outputs could depend
on what the previous request left in the cache.  The 2026-09-09
amendment guarded it host-side with per-free memsets gated to gfx1151;
this amendment eliminates the leak at the source (masked V is never fed
to the WMMA multiply) and the host workaround is gone entirely.

Validated on the Strix Halo box (single gfx1151, ROCm
7.14-gfx1151 + Vulkan RADV) with the host zeroing disabled (during
development the env `LLAMA_KV_ZERO_FREED=0` gate was used; the env is
gone in block 14):
- 16/16 identical-request determinism gates (per-position top-8 logprobs
  float64-compared) PASS on every KV type each backend's FA supports:
  ROCm f16/bf16/q8_0/q4_0 (zeroing ON==OFF bit-identical over 2064
  cells/run where both were run), Vulkan also q4_1/q5_0/q5_1/iq4_nl.
- `test-backend-ops` FLASH_ATTN_EXT vs CPU: 4591/4591 (ROCm0),
  7822/7822 (Vulkan0).
- CPU same-seed greedy: 51/64 tokens identical, divergence only at a
  near-tie (CPU non-FA vs GPU FA numerics; no coherence concern).
- Depth-16384 llama-bench decode: tg128 within 0.05% of the pre-fix
  build (f16 and bf16 KV); pp16384 within single-run drift.
- Clean-apply sim at `9113cc188`: strict 14/14 `git am`, zero
  whitespace warnings, applied tree == fork tip `ff2b35f49`.

Full record (protocols, leak matrix, per-kernel mechanism notes):
`../wip/strix-halo/kvzero/RECORD-2026-09-09.md` and the handover
`../wip/kv-sign-leak/HANDOVER-2026-09-09-mma-f16.md`.

## 2026-09-09 block-14 amendment: freed-cell KV-row zeroing gated to gfx1151 (superseded 2026-09-10)

**This section describes the 2026-09-09 amendment only; the host zeroing
it documents was REMOVED by the 2026-09-10 kernel-side fix above —
`llama-kv-cache.{cpp,h}` are back to the upstream state and block 14 no
longer contains any of it.  Kept as the historical record.**

Block 14's `seq_rm`/`seq_keep`/`clear` row zeroing (freed KV cells kept at
+0.0 so masked WMMA flash-attention columns never accumulate stale V) was
now gated to the **gfx1151 device family only**.

Background: the zeroing was ported from the strix lineage (commit
aad5adb08f, "kv-cache: zero freed cells so masked-out rows never carry
stale K/V") as a correctness/determinism guard: on gfx11 (RDNA3) WMMA,
f16 `x + (-0.0)` is not exact, so a fully masked flash-attention column
still leaks the sign of whatever V the cell last held — request outputs
can depend on what the previous request left in the cache.  It was
implemented host-side (per-free memsets) rather than in the shader to
avoid a measured 8-18% dense-prefill cost on gfx1151 from the shader
fix's mere presence in `flash_attn_cm1.comp`.

Problem found 2026-09-09: the zeroing lives in the model-agnostic
`llama_kv_cache::seq_rm` path, and on **multi-GPU** setups the per-layer
zeroing memsets decompose through ggml's meta/multi-buffer memset into
~48xN per-cell 512-byte memsets, each a synced `cudaMemsetAsync`
(~30-60 µs) — replacing a ~13k-token KV sequence stalled ~18-24 s before
the next prefill began.  Reproduced on qwen4exp AND a plain dense 4B
(3x R9700 gfx1201, tensor split): ~634k memsets / ~18 s for a 13k-token
eviction.  Single-GPU and the Vulkan-UMA path never hit it (coalesced
host memsets), which is why it went unnoticed on Strix Halo.

Fix: `zero_rows`/`zero_idxs` consult a new `llama_kv_cache::zero_freed`
member, set in the constructor: env `LLAMA_KV_ZERO_FREED=0/1` overrides;
otherwise enabled iff any KV buffer device description carries `gfx1151`
(the same host-side gfx-id mechanism the qwen4exp dense-vs-QSA decode
policy keys off; the HIP device description exposes `(gfx%x)`).  Everywhere
else the caches behave as before block 14 (no freed-cell GPU work).

Verified:
- gfx1201 (3x R9700, ROCm 7.14): A/B stall gone — identical workload
  24.5 s -> ~6 s; zeroing-off determinism gate passes (16 + 8 identical
greedy requests, per-position top-8 logprobs float64-compared, 0
differing) — the same gate that exposed the leak on gfx11.
- gfx1151 (Strix Halo box): boot log "freed-cell KV row zeroing enabled
(gfx1151)"; 16-run control unchanged.
- Clean-apply sim at `9113cc188`: strict 14/14 `git am`, zero whitespace
warnings, applied tree == fork tip `27485f1ca`.

Open follow-up: the host-side mechanism itself remains clumsy; develop a
performant gfx1151 flash-attn kernel-side exactness fix so the zeroing
can be removed entirely (the 8-18% shader-cost figure from aad5adb08f
should be re-measured on the Halo box first).

## 2026-09-09 block-01 refresh: adaptive MTP updated to the PR #27210 review head (current)

Block 01 (adaptive MTP draft depth) was cut from llama.cpp PR #27210
(author: stew675) at its `0994374fd` state; the PR then advanced through a
maintainer review round.  Block 01 is now refreshed to the PR head
`d236d41a2` (github.com/ggml-org/llama.cpp/pull/27210,
issuecomment-5582088497), delivered as one squashed block as before
(`git diff 9113cc188..d236d41a2`, 15 files 519+/35-).  Review-round content:

- `common_params_speculative::has_mtp()` helper; the MTP-type checks in
  arg.cpp (download plan), common.cpp (`load_mtp`), server-context.cpp and
  the init result are refactored through it.
- `accept_partial()` virtual + `common_speculative_accept_partial()`: a
  partial acceptance the context could not apply (checkpoint-restore path
  in tools/server and examples/speculative-simple) is reported once with
  the true accept count; the checkpoint-replay round that follows has
  `n_last` reset and no longer feeds stale draft counts to the adaptive
  controller.  The non-adaptive accept path is unchanged.
- The adaptive depth reset in `begin()` moves ahead of the empty-prompt
  early return, so the controller restarts from the floor on every new
  generation (even empty prompts).
- `--spec-draft-n-min-adaptive` rejects values < 1; registration order /
  example coverage normalized; `--spec-draft-n-min` in adaptive mode
  warns that it is unused.  Docs: docs/speculative.md, tools/cli/README.md,
  tools/server/README.md (type list + option).
- Invalid adaptive range: `GGML_ABORT` -> `std::runtime_error`.
- draft-mtp + draft-mtp-adaptive together are rejected (they would share
  one ctx_dft and both run process() on every batch).
- src/models/delta-net-base.cpp: conv-state snapshot-bound rationale
  comment (speculative verify batches start with the seq's last committed
  token; the fused GDN op relies on the same bound).
- tests/test-arg-parser.cpp: stale "defaults to 2" comment fixed (the
  default is 3) + a value-0 rejection case.
- common/speculative-adaptive.h header comment rewritten (per-depth
  constants referenced instead of enumerated).

Regeneration mechanics: canonical fork rebuilt at `9113cc188` from the
previous set, block 01 replaced in place by the squashed PR-head changeset,
blocks 02-14 re-based on top (`git rebase --onto`, clean — blocks 02-13
touch no block-01 file, block 14's common/arg.cpp/common.cpp/common.h
hunks are disjoint).  Verified: old-tip..new-tip delta is exactly the
review changeset (13 files 129+/70-, == `0994374fd..d236d41a2`), all
other files byte-identical; regenerated 0002-0013 patch bodies
byte-identical to the previous delivery (0014 refreshed only in index
lines / hunk offsets for the 3 common files); 0001's diff body
byte-identical to the PR head changeset.  Clean-apply sim at `9113cc188`:
strict 14/14 `git am`, zero whitespace warnings, applied tree == fork tip
`0f2b7a4e1`.  Rebuilt unit tests `test-arg-parser` + `test-speculative-
adaptive` pass; plain-decode same-seed coherence (3x R9700 gfx1201,
ROCm 7.14) token-IDENTICAL to the known-good `050ec89ce` build.  The
refresh touches no GPU kernels and no non-speculative host decode path.

## 2026-09-07 re-base to 050dde50c + block 14

Upstream master moved **22 commits** past `465e49b9c` (the 2026-09-07
master tip `050dde50c`).  The `~/llama.cpp` fork was rebuilt from
`patches/` with `scripts/apply-all.sh` on the fresh master tip, then
**block 14** (qwen4exp support, promoted from `beta/qwen4exp`) was added.
See the block-14 notes below.  Re-base detail:

- Blocks 01-13: `git am -3` — 12/13 applied with auto-merge; **one manual
  conflict** in `tests/test-backend-ops.cpp` (block 04's perf cases vs
  upstream's new LEAKY_RELU perf cases inserted at the same spot — both
  kept).  The upstream ggml-cuda-touching commits in the drift were
  `b74f590ea` (divergent-barrier fix in f16 flash attention, #27870),
  `73ab7599b` (branchless Q4_K/Q5_K unpack + L2 prefetch mmvq, #26705)
  and `473599738` (gfx90c HIP support, #26454); all merged in disjoint
  regions (upstream's branchless-unpack wrappers and prefetch helpers
  verified byte-identical in the merged tree next to the block k-quant
  VDR/item-split additions).
- Block 14 (qwen4exp): applied from the beta patch with `git apply
  --3way`; **one manual conflict** in `ggml-cuda/common.cuh` — upstream's
  gfx90c GCN-APU arch macros vs the block's exact-SKU
  `GGML_CUDA_CC_IS_GFX1151` predicate; resolved keeping both.
- Canonical am-commits on the new base: `90a816a68..3bebffd6b` (14
  blocks).  Set regenerated with `scripts/make-patches.sh` (base
  `050dde50c`, blocks tip `3bebffd6b`) and `rdna-boosts-all.patch`
  refreshed (87 files).

## 2026-09-08 fixes: MUL_MAT_ID pair-fusion layout gate + MWR remainder

Two genuine bugs in the amended blocks 13/14 were reported by
`briansp2020` (production single-R9700 deployment of the 14-block set,
ROCm 10): a hard `ggml_abort` in the block-13/14 MUL_MAT_ID pair
fusion (issue #18) and a silent wrong-output path in the block-13
`moe_weighted_reduction` float4 rewrite (issue #19).  Both are folded
into the blocks as amendments and the set regenerated (fork am-commits
now `861fb47b6..3529b3497`):

- **Issue #18 — MUL_MAT_ID pair-fusion gate (block 14).**  Block 13's
  `ggml_cuda_mul_mat_q_pair` MUL_MAT_ID arm implements the standard
  sparse-MoE activation layout (`src1 = [n_embd, 1, n_tokens]`, >1
  routed expert) and asserts exactly that (`ne11 == 1 && n_expert_used > 1`,
  `mmq.cu`).  Block 14's try_fuse dispatcher checked only that the two
  nodes share `src1`/`ids` and are mmq-eligible — any MUL_MAT_ID pair
  whose `src1->ne[1] > 1` (e.g. the non-broadcast per-expert-gathered
  activation layout in `test-backend-ops` MUL_MAT_VEC_FUSION) or whose
  routing is top-1 (`ids->ne[0] == 1`) satisfied the gate and then
  aborted the process at the callee assert.  The gate now requires the
  callee's layout preconditions (`node->src[1]->ne[1] == 1 &&
  node->src[2]->ne[0] > 1`), so such pairs fall back to the per-node
  path.  qwen4exp sparse-MoE pairs (standard layout, 10 routed experts)
  are unaffected and still fuse.
- **Issue #19 — `moe_weighted_reduction` float4 remainder (block 13).**
  The 2026-09-06 mwr-float4 fold (f33ffaca7) indexed the kernel and the
  launcher in quads with floor division (`n_embd / 4`) and no remainder
  handling: for `n_embd % 4 != 0` the last 1-3 columns of every output
  row were never written (silent wrong values — `MOE_WEIGHTED_REDUCTION`
  with `n_embd = 63` failed both cases).  The vectorized kernel is also
  only alignment-safe when every expert row starts 16B-aligned, i.e.
  `n_embd % 4 == 0`.  The kernel is now split into the float4 quad
  variant (launched when `n_embd % 4 == 0`; byte-unchanged aligned path)
  and the upstream scalar bounds-checked kernel (any `n_embd`).
- Validated on the local 3x R9700 (gfx1201, ROCm 7.14):
  `test-backend-ops -b ROCm0` full suite **16590/16590** with the fusion
  active (the reporter's exact single-GPU gate; CPU reference for every
  op), MUL_MAT_VEC_FUSION group 1265/1265 and MOE_WEIGHTED_REDUCTION
  6/6; same-seed llama-cli streams on Qwen3.8-Flash-Next (IQ4_XS,
  3-GPU tensor) and on the dense Qwen3.8-27B (single GPU) are
  byte-identical default vs `GGML_PAIR_OFF=1` / `GGML_PAIR_DENSE_OFF=1`,
  and the prefill A/B shows the pair fusion still active (Flash-Next
  pp2048 2780.8 vs 2756.7, pp8192 2675.7 vs 2647.1, `GGML_PAIR_OFF=1`
  controls).  Clean-apply sim re-verified: 14/14 `git am`, zero
  whitespace warnings, applied tree == fork tip.
- Re-verified 2026-09-07: clean-apply sim on a fresh checkout at
  `050dde50c` (`scripts/apply-all.sh`: **zero conflicts, zero whitespace
  warnings**, applied tree byte-identical to the fork tip `3bebffd6b`);
  full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native);
  test-backend-ops 6759/6759 (MUL_MAT / MUL_MAT_ID / FLASH_ATTN_EXT);
  test-llama-archs 617 OK / 0 fail incl. qwen4exp (GPU 9.21e-14, CPU
  0.00); dense 27B Q8_0 same-seed coherence byte-identical to the
  13-block build; qwen4exp IQ4_XS coherence on 3x R9700 — see the
  block-14 notes.
- **Block-08 amendment (2026-09-07, PR #15, reporter/author DanoPTT):**
  the mul_mat+bias fusion through a view node could hand the mmvq/mmvf
  kernels a destination whose shape the guards never checked (a reshape
  moves tokens between dimensions: matmul `ne=[n,1,2]` feeding an add
  `ne=[n,2,1]` on a two-sequence batch) — `GGML_ASSERT(ids ||
  dst->ne[1] == 1)` abort (on Windows surfacing as `0xc0000409`).  The
  guard now requires the through-view destination to satisfy the
  kernels' own shape constraint (`bias_node->ne[1]==1` plain,
  `ne[2]==1` MUL_MAT_ID) before fusing; the single-sequence case is
  unaffected.  Folded into the block-08 commit (delivery convention);
  set regenerated (base `050dde50c`, blocks tip `3bebffd6b`).  Author
  validation: single R9700 (gfx1201), 18 interleaved A/B runs,
  production since 2026-09-07; the multi-GPU coherence gate + the
  2-sequence parallel smoke were run here (3x R9700 gfx1201) — dense
  27B Q8_0 same-seed byte-identical pre vs post fix, 3-GPU hybrid ==
  RCCL IDENTICAL, test-backend-ops 6759/6759, parallel 2-slot
  llama-server decode clean on both the dense 27B and qwen4exp IQ4_XS
  (no asserts).

## Block 14 notes

**Qwen3.8-Flash-Next (qwen4exp) support** — promoted from
`beta/qwen4exp/qwen4exp-support.patch` (the squashed fork delta
`c261553a1..dd4301fb4`) and re-based onto the `050dde50c` core.  The
patch is qwen4exp-specific (the model-neutral kernel work lives in the
amended blocks 02/04/08/13):

- QSA layers: fused indexer top-k (`GGML_OP_INDEXER_TOPK`, radix),
  sparse flash attention (`GGML_OP_FLASH_ATTN_QSA`) — the default FA
  path (`LLAMA_QSA_SPARSE_FA=0` keeps dense; `-fa off` manual); CPU
  reference for the sparse op.
- Fused decode ops `GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE` (+ kernel
  geometry, rms/gamma fold, F32/Q8_0 inject fold, head-call fusion) and
  the fused `INDEXER_POOL`/`INDEXER_SCORE` decode ops with the
  incremental derived block-vector cache (`GGML_CUDA_QSA_INDEXER_CACHE`
  default ON, `=0` disables).
- Managed lazy reader (`llama-lazy-reader.cpp/.h`, `--lazy-buffer-size
  N`, `LLAMA_LAZY_IO_THREADS`) with PLE n-gram row loading + batched
  cold-page fetch.
- MTP draft-head support for the Flash-Next GGUFs (`--spec-type
  draft-mtp`), WS4 hyperconn prefill fusions (`GGML_CUDA_DISABLE_HC_FUSION=1`
  opt-out), the ggml sched alloc-fallback sync fix, the QSA dense
  shortcut (DEFAULT ON; `LLAMA_QSA_DENSE_SHORTCUT=0` opt-out) and the
  per-arch dense/QSA decode policy (`LLAMA_QSA_DENSE_DECODE_UNTIL`;
  gfx1151 default 65536).
- Env gate: `LLAMA_QSA_OFF=1` disables the QSA decode path.
- **QSA quantized-KV decode gate (2026-09-07, folded into block 14):**
  the fused `INDEXER_SCORE`/`INDEXER_FILL` ops (and their CUDA kernels' load
  dispatch) support raw indexer keys in F32/BF16/F16 only — the op
  constructors assert exactly that.  But the indexer sub-cache is created
  with the *same* `--cache-type-k` as the main KV cache, so `q8_0` (and any
  other quantized K type) handed the fused decode path a quantized key
  tensor and aborted with `GGML_ASSERT(k->type == F32/BF16/F16)` at context
  init (`ggml_indexer_fill`, graph-build probe in `sched_reserve`).
  `build_qsa_top_k` now gates the fused decode path on an unquantized
  indexer key type and falls back to the per-op chain (whose `get_rows`
  dequantizes any cache type on gather) — the BF16/f32 fused path is
  byte-identical (same code when the gate passes).  Validated on Strix
  Halo (gfx1151): the reported q8_0 server config (incl. MTP draft
  q8_0) loads and generates (acceptance 0.80), the full KV-type matrix
  f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 all start + generate with
  zero errors, forced-sparse QSA decode (the formerly-crashing deep path)
  runs clean at q8_0, and the BF16 fused fill/score path is unregressed
  (forced-sparse acceptance 0.82).  Record:
  `beta/qwen4exp/README.md`.
- **Derived-cache pool gate (2026-09-07, same amendment):** the F32
  block-vector pool (12 layers x 128 dims x 1 stream ≈ 100+ MiB at a
  70k ctx, 103 MiB at the reported 70144) was allocated for EVERY qwen4exp
  context, but the pool is only ever written/read by the fused
  `INDEXER_FILL` -> `INDEXER_SCORE` path: it needs unquantized indexer
  keys (the gate above) AND the memory-layer derived cache engaged
  (`GGML_CUDA_QSA_INDEXER_CACHE` set; otherwise `qsa_derived_limits`
  emits an empty fill range every step and the pool is dead weight
  ridden by a no-op fill launch per decode step).  `pool_create` now
  skips the allocation unless both hold (logged as `derived indexer
  cache pool skipped (...)`); `get_pool()` returns nullptr and the fused
  score falls back to pooling the raw cache — the same F32 arithmetic,
  byte-identical output, minus the dead buffer.  Validated on Strix Halo
  (gfx1151): pool probe shows allocated+ENABLED only for float keys +
  env set, skipped for bf16/q8_0 defaults and q8_0 + env; BF16
  forced-sparse same-seed decode byte-identical with the pool absent
  (default) vs present + derived engaged (`GGML_CUDA_QSA_INDEXER_CACHE=1`);
  q8_0 runme config + the full KV-type matrix re-run clean (zero
  errors, acceptance unchanged); clean-apply sim tree-identical.
- **MUL_MAT_ID pair-fusion layout gate (2026-09-08, folded into block
  14, issue #18):** the gate+up MUL_MAT_ID pair dispatch routes into
  block 13's `ggml_cuda_mul_mat_q_pair`, whose MUL_MAT_ID arm
  implements the standard sparse-MoE activation layout (`src1 =
  [n_embd, 1, n_tokens]`, >1 routed expert) and asserts exactly that
  (`ne11 == 1 && n_expert_used > 1`).  The dispatcher checked only that
  the two nodes share `src1`/`ids` and are mmq-eligible, so any
  MUL_MAT_ID pair in another layout (e.g. the non-broadcast
  per-expert-gathered activation `src1 = [k, n_used, m]` from
  `test-backend-ops` MUL_MAT_VEC_FUSION, or top-1 routing with
  `ids->ne[0] == 1`) satisfied the gate and aborted the whole process at
  the callee assert.  The gate now requires the callee's layout
  preconditions (`node->src[1]->ne[1] == 1 && node->src[2]->ne[0] > 1`)
  so those pairs fall back to the per-node path.  The qwen4exp
  sparse-MoE pair (standard layout, 10 routed experts) is unaffected
  and still fuses.  Verified (3x R9700 gfx1201): MUL_MAT_VEC_FUSION
  1265/1265 (no abort), full test-backend-ops 16590/16590 with the
  fusion active; Flash-Next same-seed text byte-identical default vs
  `GGML_PAIR_OFF=1` and the prefill A/B still shows the pair active
  (pp2048 2780.8 vs 2756.7, pp8192 2675.7 vs 2647.1).
- **Compiler-warning cleanup (2026-09-08, folded into block 14):** the
  block-14 sources warned under the Vulkan host build (system clang
  16.2.1) and the ROCm 7.14 build.  (1) `ggml.c`: unused `n_blocks`
  local in the `ggml_indexer_fill` builder.  (2) `ggml-cpu.c`
  `-Wswitch`: the CPU compute-forward switch is exhaustive over
  `GGML_OP_*` and had no labels for the new `GGML_OP_INDEXER_SCORE` /
  `GGML_OP_INDEXER_FILL` (GPU-only fused ops with no CPU forward; the
  CPU plan phase already aborts on them as "op not implemented" before
  compute, so the labels are an unreachable `GGML_ABORT`, mirroring
  `GGML_OP_COUNT`).  (3) `ggml-cpu/ops.cpp`: unreachable `break` after
  the noreturn `GGML_ABORT("fatal error")` in the `HC_MIX`/`HC_COMBINE`
  CPU type dispatchers' default cases (dropped, matching the upstream
  convention).  (4) `qwen4exp.cpp`: `idx_cache` had been narrowed to
  `bool`, which made the documented `GGML_CUDA_QSA_INDEXER_CACHE=2`
  debug probe (`idx_cache != 2`) tautological (`-Wtautological-
  constant-out-of-range-compare`); restored to an `int` 0/1/2
  tri-state so probe-2 (score reads the pool WITHOUT the fill) is
  reachable again.  (5) `qwen4exp.cpp`: `-Wsign-compare` in the
  gfx-id sniff loop (now `size_t`).  No generated-code or runtime-
  behavior change in default configs; verified warning-free with the
  exact build flags (4 TUs) and by full Vulkan + ROCm gfx1201 builds
  of the re-applied sim tree.
- **qwen4exp tensor-split backend gate (2026-09-08, folded into block
  14):** block 14 removed upstream's `case LLM_ARCH_QWEN4EXP: // TODO:
  fix test-llama-archs` from `llm_arch_supports_sm_tensor`, enabling
  qwen4exp tensor split on every backend.  It is validated on ROCm/HIP
  only (3x R9700, NMSE 9.87e-14 vs CPU); on backends that cannot run
  the fused QSA/HC/WS4 ops on-device (Vulkan, Metal, SYCL; NVIDIA CUDA
  untested) the CPU-fallback subgraphs leave the meta splitter unable
  to reconcile mirrored-vs-split operand states and it aborts at graph
  reserve (`ggml-backend-meta.cpp` `handle_generic`, e.g. the qwen4exp
  gated-attention `MUL` on Vulkan — `test-llama-archs` died at the
  qwen4exp Meta row).  The enablement is now `#ifdef GGML_USE_HIP`,
  restoring upstream's clean "not implemented" error / arch-test SKIP
  on all other builds.  Verified: Vulkan — full test-llama-archs sweep
  completes RC=0 (457 rows, statuses identical to upstream
  `050dde50c`; qwen4exp Meta SKIP like upstream; single-device still
  OK 9.01e-08, roundtrip OK), llama-cli qwen4exp `-sm tensor` fails
  with the upstream message; HIP — qwen4exp Meta still OK 9.87e-14.

- **Quantized-KV tensor-split gate (2026-09-08, folded into block 14):**
  `q4_1`-family KV cache types (`q4_1`, `q5_0`, `q5_1`, `iq4_nl`) hard-
  aborted during the first graph reserve under multi-GPU
  `SPLIT_MODE_TENSOR` — `ggml-backend-meta.cpp` `handle_generic`
  `GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` — on both
  dense qwen35 (Qwen3.6-27B) and qwen4exp (Flash-Next) on gfx1201
  (3x R9700).  **Upstream bug, not fork-specific**: reproduced on
  pristine vanilla llama.cpp at `050dde50c` (same assert, non-qwen4exp
  model; also at 1 GPU because upstream wraps even a single device in
  the Meta backend) and unfixed on current upstream master.  Mechanism:
  tensor split forces flash attention, whose CUDA/HIP kernels read the
  quantized K/V cache natively only for `q4_0`/`q8_0` (plus the float
  types).  For the q4_1 family the graph cannot express a splittable
  attention, the sched's graph-copy machinery turns the attention I/O
  into op-NONE graph-external leaves (split state MIRRORED), and the
  resulting MIRRORED `attn_pregate` collides with the AXIS-0 elementwise
  gate branch of the qwen35/qwen4exp gated attention (`attn_gated =
  attn_pregate * sigmoid(gate)`).  Fix: a context-creation gate in
  `llama_init_from_model` rejects K/V types outside FA's native set
  (quantized and not `q4_0`/`q8_0`) with an actionable error when the
  Meta device is actually in use (tensor split over >= 2 GPUs).  The
  fork's single-GPU "tensor" mode skips the Meta wrapper (block 07) and
  keeps working; layer split and `f32/f16/bf16/q8_0/q4_0` KV are
  unaffected.  Validated 2026-09-08 on gfx1201 (3x R9700, ROCm 7.14):
  KV-type matrix on dense 27B Q8_0 + Flash-Next IQ4_XS (3-GPU tensor) —
  `f32/f16/bf16/q8_0/q4_0` generate, the four failing types + mixed
  `k=q4_1 v=bf16` / `k=bf16 v=q4_1` fail cleanly (zero asserts); layer
  split + q4_1 Flash-Next 25.9 t/s (unchanged); qwen4exp derived-cache
  pool-gate byte identity holds; dense-27B same-seed coherence A/B
  (gate stripped vs applied) byte-identical; test-llama-archs qwen4exp
  all OK (NMSE 1.01e-13).

Validation is recorded in `beta/qwen4exp/README.md` (the halo/soar
campaigns on the old base) plus the 2026-09-07 delivery checks above;
re-base conflict resolution detail in the 2026-09-07 re-base section.

## 2026-09-06 re-base to 465e49b9c

Upstream master moved **18 commits** past the fold-verified base
`8b4b3558f` (57 past the old delivery fork point `9cffdcc80`).  The
`~/llama.cpp` fork was rebuilt from `patches/` with
`scripts/apply-all.sh` on the fresh master tip: 13/13 `git am` clean,
**zero conflicts, zero whitespace warnings** (the ggml-cuda-touching
upstream commits were `73a43d1f6` (**mmid/mmf race fixes**, #28475) and
`5fdfa6282` (**GDN l2-norm fix**, #28068 — model-layer only); both
landed in disjoint hunks and needed no manual merges).  Applied-tree
check: on all 112 files upstream touched between the bases, the per-file
deltas equal old-fork + upstream-drift exactly; the 14 remaining
differing files are precisely the 2026-09-06 Strix fold delta the old
pre-fold fork lacks.

Set regenerated with `scripts/make-patches.sh` (base `465e49b9c`,
blocks tip `c261553a1`; canonical am-commits `45bf4d291..c261553a1`)
and `rdna-boosts-all.patch` refreshed (45 files; the previous copy was
stale at 41, pre-fold).  Two prerequisites fixed along the way: (1) the
fold (0044cfe) had stripped the format-patch mail headers from
0002/0004/0008/0013 — restored from the pre-fold originals (canonical
subjects/dates, 0013 body) so the set is `git am`-able again; (2) the
re-base record of the 0013 block-13 message trailer re-dated to the
fold's true date.

Re-verified 2026-09-06: clean-apply sim on a fresh checkout at
`465e49b9c` (`scripts/apply-all.sh`: **zero conflicts, zero whitespace
warnings**; applied tree byte-identical to the fork tip `c261553a1`).
Content is unchanged from the 2026-09-02/09-05 records — the re-base
folds upstream's additions into the patch context only.

## 2026-09-02 re-base to 9cffdcc80

Upstream master moved **42 commits** past the fork point `0eadefebd`; the
ggml-cuda-touching ones were `3d3d7c818` (unused-var removals in
`mmq.cuh`/`mmq-vec-dot.cuh`, #28235), `8e93a9773` (**sparse-fa for
DSV4/GLM**, #27970 — fattn-tile/fattn-common territory) and `3466812d1`
(**fused MoE weighted-expert reduction**, #25952 — `ggml_cuda_try_fuse`
territory), plus common/server arg churn (`e750b887a`).  The fork was
rebuilt on the new base (`~/llama.cpp` rdna-boosts = `9cffdcc80` +
blocks `04122bfb5..92f09e80a`) and the set regenerated with
`scripts/make-patches.sh` (base `9cffdcc80`, blocks tip `92f09e80a`);
regenerating from the new base folds upstream's changes into the patch
context, so `scripts/apply-all.sh` is clean again on fresh master.

Three blocks needed manual re-base hunks during the rebuild:

1. **Block 03 vs #27970 (sparse-fa):** upstream added a 4th bool
   (`use_sparse`) to `launch_fattn`'s arg list and updated the fattn-tile
   call sites; block 03 rewrites the same sites (type_KV template
   threading + runtime `need_f16_K`/`need_f16_V`).  Merged: each site
   passes `need_f16_K, need_f16_V, false, false, warp_size` (upstream's
   `stream_k`/`use_sparse` slots stay `false`); the
   `launch_fattn_tile_switch_ncols2` template gained `type_KV`.
2. **Block 08 vs #25952 (MoE expert reduction):** upstream inserted its
   `GGML_OP_MUL` weighted-reduction arm into `ggml_cuda_try_fuse` right
   after `node = cgraph->nodes[i]`; block 08's rms_norm->mmvq
   quantize-fold arm now sits after it (arms are mutually exclusive on
   `node->op`, order-independent).  Also folded into the block-08 commit:
   the block-08-added spec-verify `launch_fattn` call site in
   fattn-tile.cuh still used the pre-#27970 3-bool arg list, which binds
   the `warp_size` int into the new `use_sparse` bool slot (compiles;
   `use_sparse=true`) and aborts at runtime
   (`GGML_ASSERT(n_kv_max > 0)` in fattn-common.cuh).  Fixed to the
   4-bool form.
3. **Block 13 vs #25952:** the block's `disable_moe_mmq` opt-out static +
   `const int cc` decls at the top of `ggml_cuda_try_fuse` were rejected
   (context shifted by upstream's inserted arm); restored after the MoE
   arm.

Re-verified 2026-09-02: clean-apply sim on a fresh checkout at
`9cffdcc80` (`scripts/apply-all.sh`: **zero conflicts, zero whitespace
warnings**; applied tree byte-identical to the fork tip `92f09e80a`),
full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native, zero
errors), llama-cli same-seed coherence **IDENTICAL between hybrid and
RCCL** (3-GPU tensor split, Qwen3.5-4B Q8_0).  Numbers are unchanged
from the 2026-09-01 records — the re-base is content-identical plus
upstream's additions.

> **Build-environment note:** `~/bin/build-llama-rocm-714` hardcodes
> `-DCMAKE_HIP_FLAGS="-mllvm"` (a leftover of the commented
> `-mllvm --amdgpu-unroll-threshold-local=600`).  With CMake >= 4.3 the
> HIP compiler test appends `--cuda-host-only` right after it, and the
> bare `-mllvm` swallows it into LLVM option parsing (configure fails).
> Build with `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` to override.

## Block 08 notes

- **Quantized KV-type enablement (2026-09-11, F3 step 1).**  `ggml_cuda_fattn_kv_type_supported()`
  (`ggml/src/ggml-cuda/fattn.cu`) returned false for `Q4_1`/`Q5_0`/`Q5_1` unless
  `GGML_CUDA_FA_ALL_QUANTS` was defined, and `llama_context::resolve_fused_ops()`' FlashAttention probe
  (which asks the backend whether `GGML_OP_FLASH_ATTN_EXT` is supported) then turned flash attention
  **off for the whole context**: those cache types ran the non-FA attention path, 3.4x slower prefill /
  1.7x decode (4B pp512 2119.6 / tg32 55.94 vs 7366.3 / 94.16 after).  Nothing was missing on the
  kernel side — the tile and mma-f16 families stage K/V through `ggml_get_to_fp16_cuda`, which already
  covers the whole `q4_0/q4_1/q5_0/q5_1/q8_0` set (the trace shows `f16K=1 f16V=1` for `q4_0`/`q8_0`
  too), and the vec family is instantiated per (K,V) pair.  The three types lose the `#ifndef` guard,
  the default (non-`FA_ALL_QUANTS`) vec dispatch gains the three diagonal cases and
  `ggml-{cuda,hip,musa}/CMakeLists.txt` gain the three diagonal instances (3 TUs).
  `GGML_CUDA_FA_ALL_QUANTS` remains the knob for the 42 *mixed* `K != V` pairs; with it off the chooser
  still rejects `K != V` before the family decision, so the reachable pair set is exactly the
  diagonals and the predicate, the dispatch and the instance lists cannot disagree.  The types are
  band-uniform on gfx1201 by construction (with a quantized cache the whole band takes TILE, block
  08's F1 fix below); every newly reachable diagonal was swept `W = 1..8` on both splits and both
  `RS` modes and passed the `plain == draft-mtp` and MTP gates — see the 2026-09-11 (8) WORKLOG entry
  and `../GREEDY-PURITY.md` §20.
- **Decode/verify kernel-family fix (2026-09-11, F1).**  `ggml_cuda_get_best_fattn_kernel()`
  (`ggml/src/ggml-cuda/fattn.cu`) used to return the generic **VEC** kernel for small batches — for
  `n_q == 1` when GQA optimizations do not apply, and for `n_q <= 2` whenever K or V is quantized
  (upstream heuristic, "for small batch sizes the vector kernel may be preferable").  Those two
  conditions are always inside the decode/verify band (`n_q = n_draft + 1 <= 8`; prefill fell through
  to TILE regardless), so with a `q8_0`/`q4_0` KV cache `n_q = 1,2` ran one kernel family and
  `n_q >= 3` another.  The families order the online-softmax/PV reduction differently, so a 1-token
  decode was not bit-identical to a verify batch and plain greedy decode disagreed with
  `--spec-type draft-mtp`.  The branch is deleted: the whole band uses TILE, matching the WMMA guard
  this block added on 2026-08-29 (`Q->ne[1] > 8`) and block 00's `ntiles_dst_eff` in `launch_fattn`.
  Measured: 4B `q8_0`/`q4_0` `W=1..8` bit-identical in **all four split configs** (1 GPU / 2-GPU layer /
  2-GPU tensor / 3-GPU tensor), 27B the same; 27B text plain == `n_max 3` == `n_max 7`; MTP
  acceptance bit-identical (0.90789); tg128 -0.5..-0.9%, pp512 ~-0.2%, reserves byte-identical;
  FLASH_ATTN_EXT 4591/4591.  Only `W=1,2` change, and the new value equals the previous *verify*
  value, so the spec path is untouched.  Debug tool:
  `../wip/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch` (`GGML_CUDA_FA_TRACE=1`);
  details in `../GREEDY-PURITY.md` §14 and the 2026-09-11 (3) WORKLOG entry.

## Block 12 notes

- **Amended 2026-09-11: the hybrid dispatch's small/large crossover is now
  width-safe.**  `ggml_backend_cuda_comm_is_small()` picks the internal
  host-staged pipeline below a per-device-count element count and NCCL above
  it.  The two paths are **not bit-identical** (different summation order; the
  internal path always does the FP32->BF16 round-trip), so any tensor whose
  size straddles the crossover got a different result depending on its *shape*.
  Under `-sm tensor` the reduced tensors scale with the batch width
  (`ne = ne0 * n_tokens`; `ne0 = 5120` on Qwen3.8-27B), so with the old
  2-device value of 32768 a **7-token** speculative verify batch (35840
  elements) was reduced by NCCL while 1..6-token decode stayed on the internal
  pipeline - and `--spec-type none` stopped matching `draft-mtp` from
  `n_max = 6` on 2 GPUs (`GREEDY-PURITY.md` §11, cause A).  The 2-device
  crossover is now **131072**, i.e. the 3-device value: the largest verify
  batch (`--spec-draft-n-max 16` -> 17 tokens = 87040 elements) stays well
  under it, and still far below the internal pipeline's own 1 MB (262144
  element) cap, so nothing is pushed off the fast path.  Only the decode/verify
  band (7..25 tokens) changes path; one-token decode and prefill (>25 tokens)
  are untouched.  Measured (27B Q8_0, 2-GPU tensor, `-ts 1/1`): probe
  `W = 1/6/7/8` all `a4817ee6` (with `W <= 6` bit-identical to the previous
  build, so plain decode is unchanged); text `none == n4 == n6 == n7`
  (`6e8ccd25`; previously pure only to `n_max 4`/5); MTP `n_max 6` acceptance
  0.509 -> 0.533 and 63.6 -> 71.3 t/s (+12%), `n_max 12` 51.7 -> 58.0 t/s
  (+12%), pp512/pp4096/tg128 unchanged within noise.  `W >= 9` still diverges -
  that is the *separate, deliberate* FA tile-vs-WMMA switch (`Q->ne[1] > 8`),
  which caps the guarantee at the designed `n_max <= 7`.
- **RDNA4-only gate**: the internal all-reduce refuses to init on any
  architecture other than gfx1200/gfx1201 (the pipeline falls back to the
  default RCCL path with a warning).  Community verification on RDNA3 pairs
  is pending; remove the gate's arch check once verified.
- Env knobs (defaults preserve upstream behavior):
  - `GGML_CUDA_ALLREDUCE=hybrid|nccl|internal|none` (hybrid = default on Linux)
  - `GGML_CUDA_AR_PROFILE=1` — per-call spin/phase profiler at teardown
  - `GGML_CUDA_AR_SLEEP=0|1` — s_sleep poll vs dummy spin (default 1)
  - `GGML_CUDA_AR_BF16_THRESHOLD` — F32->BF16 wire round-trip threshold
  - `GGML_CUDA_AR_COPY_THRESHOLD` / `GGML_CUDA_AR_COPY_CHUNK_BYTES` — CE path
  - `GGML_CUDA_AR_SPIN_TIMEOUT_MS` — bounded in-kernel peer-arrival spin
    budget (default 20 ms, `0` = legacy unbounded); on timeout the kernel
    sets a host-mapped poison flag, skips the reduce and exits, and the host
    re-syncs the devices via a butterfly AllReduce on the next call
  - WIP experiments (archived, env-gated OFF by default): `GGML_CUDA_AR_FUSED`,
    `GGML_CUDA_AR_PACE` — see `../archive/work/fused-stage-pacing/README.md`
- Verified 2026-09-01 re-base (3x R9700, ROCm 7.14, gfx1201): clean apply
  on a fresh checkout at `0eadefebd` + full build + llama-cli same-seed
  coherence IDENTICAL + tg64 38.12 / tg512 41.08 (sim build; numbers
  unchanged — the re-base is code-identical to the 2026-08-30 set).
  Depth-16384 decode 38.71 t/s (3-GPU hybrid, unpinned) with the server
  config `HIP_VISIBLE_DEVICES=0,1,2`.
- **Community-report fix round (2026-08-30, issues #5 + #6, reporter
  tungel):** two block-12 fixes integrated into the fork and regenerated
  into this set:
  - `-DGGML_HIP_RCCL=OFF` builds now compile — `comm_init_hybrid`'s
    `try_allreduce_nccl` reference is guarded by `GGML_USE_NCCL` (was an
    unconditional reference to an `#ifdef`-guarded function: build error).
  - The chunked AR kernel's in-kernel peer-arrival spin is now bounded
    (`GGML_CUDA_AR_SPIN_TIMEOUT_MS`, default 20 ms, `0` = legacy).  An
    unbounded spin on RDNA (non-preemptible compute kernels) could wedge
    the queue -> MES `REMOVE_QUEUE` timeout -> MODE1 reset -> `700/719`
    or a whole-machine freeze; on timeout the kernel poisons a host-mapped
    flag, skips the reduce and exits (queue stays removable), and the host
    re-syncs the devices with a butterfly AllReduce on the next call.
    The budget check is decimated to 1-in-512 polls (2x the measured
    typical spin count — p50 184 / p90 283 / p99 369 polls on 2x gfx1201
    hybrid at depth-16384 — rounded up to a power of two), merged with
    the arrival check into a single per-poll branch (a tick without a
    timeout keeps polling the same peer), so the true fast path never
    executes `clock64()` at all and the timeout overshoot stays <0.1% of
    budget.
  - Re-verified 2026-08-30: clean-apply sim at `17252c769` (apply, full
    build, llama-cli same-seed coherence IDENTICAL); RCCL=OFF `ggml-hip`
    compiles; before/after perf on the default hybrid config (2x R9700,
    depth-16384) shows NO measurable impact — pp512 1622.7 -> 1609.1 t/s
    (-0.8%, within noise), tg128 32.63 -> 32.55 t/s (-0.25%, within
    noise).  Note: regenerated from the current `~/llama.cpp` fork
    (rdna-boosts tip `8a426cf79`); commit hashes in the 01-11 patch
    headers drift from the earlier records (the fork was rebuilt; diff
    content is unchanged).
- **Community-report fix round (2026-09-04, issue #13, reporter
  tungel):** runtime NCCL/RCCL failures are no longer fatal.  RCCL >=
  2.30.4 can refuse kernel dispatch on the first collective
  (`hipErrorIllegalState`: "the operation cannot be performed in the
  present state") when a GPU sits behind a PCIe root port without
  32/64-bit AtomicOp completer support (e.g. PCH/Z390), even though
  `ncclCommInitAll` succeeds (see ROCm/ROCm#6520) — the process used to
  abort at the first prefill AllReduce (`NCCL_CHECK` -> `GGML_ABORT`)
  although the internal host-staged pipeline was up and stable.  Fix
  (folded into the block-12 commit): on the first NCCL runtime failure
  the comm layer clears the sticky HIP errors the failed dispatch left
  on each AR device (else the fallback aborts on the next CUDA_CHECK),
  warns once with a pointer at the known cause + the
  `dmesg | grep -i atomic` check, permanently stops using NCCL for the
  rest of the run (comm state is unknown), and re-routes AllReduce to
  the internal pipeline when available, otherwise to the meta backend's
  butterfly; the failing call itself returns false so the butterfly
  handles it.  `ncclCommDestroy` at teardown is also no longer fatal.
  No behavior change on healthy setups — the fallback only triggers
  when NCCL itself fails.
  - Re-verified 2026-09-04: set regenerated from the fork (rdna-boosts
    tip `b830050bf`), clean-apply sim at `9cffdcc80` (git am clean,
    zero whitespace warnings, applied tree byte-identical to the fork
    tip), full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native),
    llama-cli same-seed coherence IDENTICAL pre vs post fix (27B Q8_0,
    3-GPU tensor split), perf unregressed at depth-16384 hybrid:
    2-GPU (1,2) tg128 32.48 -> 32.40, 3-GPU (0,1,2) tg128 39.33 ->
    39.31 (both within noise; pp512 within run-to-run spread).  The
    failing-call correctness relies on dispatch-time refusal leaving
    the buffers pristine (nothing executed); the Protocol-A gate on the
    reporter's rig closes the residual partial-execution caveat.
- **Compiler-warning clean** (2026-08-29 follow-up): ROCm 7.14 marks
  `hipError_t` `[[nodiscard]]`, and the original HIP port left 27
  unchecked HIP calls (all `-Wunused-value` in the ggml-hip build).  All
  27 now go through `CUDA_CHECK` (upstream house style, incl. teardown);
  three dead WIP items removed.  The ggml-hip build emits ZERO warnings
  from this patch.
- **AR_PROFILE devices[] init fix** (2026-09-01, PR #8, integrated):
  `ggml_cuda_ar_pipeline_init` now copies the caller's `devices[]` into
  the pipeline BEFORE the per-device profiler hipMallocs.  With
  `GGML_CUDA_AR_PROFILE=1` the buffers were allocated while `devices[]`
  was still zero-filled, so every prof buffer landed on GPU 0 and MTP's
  second pipeline init (draft context) faulted GPU 1 (gfx1201).  A/B on
  3x R9700 (2-GPU, internal AR, MTP n-max 3, `-c 32768`): pre-fix
  reproduced the fault (`Memory Fault Error ... GPU index: 1, kernel:
  ggml_cuda_ar_kernel<float, __hip_bfloat16>`); post-fix runs clean with
  teardown dumps on dev0 AND dev1 in both pipelines, same-seed coherence
  IDENTICAL to the pre-fix golden.  Default serving (profiler off) is
  unaffected.  Do not ship `AR_PROFILE=1` as a daily env — this only
  makes the debug flag safe.
- **MTP chunked-GDN prefix folded into block 02** (2026-09-01, PR #9,
  integrated): block 02's chunked WMMA GDN used to launch only for
  `K == 1` (no MTP snapshots) — with MTP n-max 3 (`K=4`) every prefill
  ubatch stayed on the sequential kernel.  Long single-sequence MTP
  prefills (`K > 1`, `n_seqs == 1`, `n_tokens > K+64`) now run the
  chunked GDN on the prefix (`n_tokens - K`) and sequential GDN only on
  the last K tokens so slots `0..K-1` stay correct (fused-cache graphs
  included; `n_seqs > 1` stays fully sequential).  The chunked ops take
  an `n_tokens_limit` parameter.  Opt out: `GGML_CUDA_GDN_CHUNKED=0`
  (also `GGML_CUDA_GDN_CHUNKED_BF16=0`).  Verified 2026-09-01 on 3x
  R9700 (2-GPU, internal AR, Qwen3.8-27B Q8, ubatch 1024, MTP n-max 3):
  path fire `n=1024 K=4 prefix=1020`; prefill tok/s +7.5% (~5.5k prompt)
  / +7.7% (~38k) vs sequential; 64-token same-seed output token-identical
  to sequential; non-MTP coherence unchanged.  Not bit-identical vs
  sequential in general (same class as the bf16 chunked: near-lossless).
  Lab numbers: `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
- **K-independent whole-batch chunked prefill — free, no tail, no gate**
  (2026-09-11).  Fixes the fork-only plain-vs-spec divergence from the gfx1151
  issue-#25 validation.  The chunked kernel is not bit-exact with the
  sequential one, so a K-dependent boundary makes the post-prefill SSM state
  depend on `n_rs_seq`: plain decode (`K == 1`) chunks the whole prompt while
  the MTP path (`K == n_max + 1`) chunks `n_tokens - K` plus a K-token tail,
  and `--spec-type none` then disagrees with `draft-mtp`.  The alignment is
  done by giving both paths the **same call**: a batch with more than
  `max(K, 16)` tokens is chunked **whole** — exactly what `K == 1` does — and
  anything smaller falls through to the sequential kernel.  No sequential tail,
  no `KTAIL`.
  A batch larger than `max(K, 16)` cannot be a speculative verify batch (a
  verify batch decodes at most `K = n_rs_seq + 1` tokens) and is never rolled
  back into, which is why its K rollback snapshots can be skipped; every batch
  at or below the threshold — in particular every verify batch — stays on the
  sequential kernel and writes the snapshots the spec rollback reads.  The
  threshold must be a constant for `K <= 16` (or the two paths diverge again on
  short prompts); the floor at `K` keeps deeper drafts correct (sequential)
  instead of reading an unwritten slot.  `n_seqs > 1` keeps the whole-ubatch
  path for `K == 1`.
  **Cost: none.**  27B Q8_0 1 GPU pp512/2048/4096 = 1385.3/1356.4/1328.2 vs
  1384.7/1355.0/1327.8 for the old K-dependent boundary (parity); this replaces
  the previous KTAIL=16 tail cost (-0.3..-0.8 %) with zero.  The old
  `GGML_CUDA_GDN_ALIGN_BOUNDARY` gate and its two K-dependent branches were
  **removed** (~118 lines): both were unreachable with the gate on, and the
  opt-out no longer bought anything now that the default is free.
  `GGML_CUDA_GDN_CHUNKED=0` remains the only switch — it forces the sequential
  kernel everywhere (correct, bit-identical, slow) — and is the fallback if the
  snapshot assumption below is ever violated.
  **Guard:** the invariant above (only verify batches are rolled back into) is
  empirical, so `llama_memory_recurrent::seq_rm` now tracks the last batch's
  per-seq token count and logs a **once-only warning** if a rollback ever
  crosses that boundary, instead of silently restoring an unwritten slot.
  Measured against it: llama-cli `draft-mtp` n_max 1/4/8/16 (449 rollbacks) and
  llama-server `--cache-reuse` (20 rollbacks) — every rollback was preceded by a
  batch of <= K tokens, 0 warnings; gfx1201 probe `RS=6 W=6 == RS=0 W=1`
  confirms the prefill is K-independent.
  Verified: gfx1201 probe (`RS=from_w`, P=256) `W = 1/3/5/6` all `a4817ee6`
  (4B 1-GPU `671d6096`); 27B 2-GPU tensor text `none == n1 == n4 == n5`
  (`6e8ccd25`); `test-backend-ops -o GATED_DELTA_NET` OK.
  **The pure `none == draft-mtp` range is `n_max <= 7`, not 15** (an 8-token
  verify batch is the designed limit; beyond it the FA tile-vs-WMMA switch at
  `Q->ne[1] > 8` changes the reduction).  On 2-GPU `-sm tensor` it was
  `n_max <= 5` until the block-12 dispatch fix of 2026-09-11.  See
  `../GREEDY-PURITY.md` §11 and
  `../wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.
  Record: `../wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md`.

## Block 13 notes

**2026-09-11 (fourth amendment) — the MoE shared-expert epilogue is band-uniform.**

The decode-only fused shared-expert down epilogue (`dst = down(swiglu) * sigmoid(gate(x)) + moe_out +
ffn_residual`, a 6-node fusion in `ggml-cuda.cu`) was gated `down_mm->src[1]->ne[1] == 1 &&
gate_mm->src[1]->ne[1] == 1` — with an in-code note that its fused gate reduction does not reproduce the
standalone mmvq/MUL_MAT order, so a 1-token decode ran the fused epilogue and an n-token verify batch ran
the unfused chain.  That was the last width-impurity in the MoE class: `W=1 ac8825358d9adfda` vs
`W>=2 bd138ad2326fbbf2` (Qwen3.6-35B-A3B Q4_K_M, 1 GPU, f16 KV, probe `P=256`).

The band now takes the **fused** path (keeping the +3.1 % decode win rather than disabling it):

- `mmvq.cu`: `shexp_gate_sigmoid` is one block (one warp) per token (`grid: (ncols)`); the token only
  selects the input column, addressed with `x_gate`'s own stride.  `shexp_down_gated_q8_0` is one block
  per `(output row, token)` (`grid: (nrows, ncols)`), addressing `y_swiglu` at the padded row stride and
  `moe_out` / `ffn_residual` / `dst` at the token offset.
- **`nwarps` is pinned to the single-token value** (`calc_nwarps(Q8_0, 1, ...)`): `calc_nwarps` returns
  4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets `blocks_per_iter`, i.e. the reduction order
  of the down projection — pinning it is what makes every width bit-identical (the same class of trap
  as the third amendment's per-type caps).
- `ggml-cuda.cu`: the fusion arm accepts `1 <= ne[1] <= MMVQ_MAX_BATCH_SIZE` with both matmuls the same
  width and the three epilogue operands contiguous; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` still selects
  the unfused reference.

Measured: probe `W = 1, 2, 3, 4, 8` all `ac8825358d9adfda` (the pre-fix `W=1`/fused value); with the
kill-switch all `bd138ad2326fbbf2` (a uniform unfused reference, = the pre-fix `W>=2` value).  The MTP
gate improves substantially, because the verify batch now uses the same epilogue arithmetic as the
draft's single-token decode steps — Qwen3.6-35B-A3B, 1 GPU, f16 KV, `n_max 3`, `n=96`: acceptance
**0.81707** (was 0.51) with 167.3 t/s vs plain 96.9 (**+73 %**).  Dense models and qwen4exp are
unaffected (qwen4exp probe: `plain == n_max 3` still `804de0576868`).  The asterisk is therefore gone —
see `../GREEDY-PURITY.md` §17.

**2026-09-11 (third amendment) — the decode/verify band is band-uniform (F2 cause 2).**

qwen4exp's logits were not bit-identical across decode/verify batch widths (`{1,2,3,4} {5} {6,7} {8}`).
Task 1 (fusion vs graph-builder) was settled by `[GD]` full-graph dumps: the graphs are **identical** at
every stage (2647/2404/2271/1863/1668/1565 nodes at both `W=4` and `W=5`), always containing
`MUL_MAT_ID(ffn_moe_gate)` / `MUL_MAT_ID(ffn_moe_up)` / `GLU(ffn_moe_swiglu)` at `k=76/77/78` — only the
*fusion coverage* differed.  The mechanism, however, is upstream's **per-type mmvq cap**
(`get_mmvq_mmid_max_batch_*`), used in two places:

* `mul_mat_vec_q_moe`'s `__launch_bounds__` was `get_mmvq_mmid_max_batch_for_device<type>()*warp_size`
  while the block is `(warp_size, ncols_dst)` — so the cap is a *capability* limit: launching `IQ3_S`
  (cap 4) with `ncols_dst = 5` is 160 threads > the bound and aborts the run (`ROCm error: unspecified
  launch failure`);
* the same cap drives the mmvq-vs-MMQ choice (`ggml_cuda_mul_mat_id`: `ne2 <= cap → mmvq`, else
  `should_use_mmq → MMQ`), and `use_mmvq` (`ggml-cuda.cu:3730`) gates the `mul_mat_q_pair` fusion —
  which is what actually ran at `W = 5..7`.  mmvq and MMQ reduce in different orders.

The UD-IQ4_XS quant mixes expert types per layer (47 layers `IQ3_S` gate/up → cap 4; layer 2 `IQ4_XS`
→ cap 5; down `IQ4_NL`/`Q8_0` → cap 7), which **predicts the measured census exactly**: fused layers
48/48/48/48/1/0/0/0 for `W = 1..8` (`ffn_moe_up` `MUL_MAT_ID` counts 0/0/0/0/47/48/48) — the 4→5 and
5→6 boundaries; the down's cap 7 is the 7→8 boundary.

**Fix** (completes block 13's own `has_ids` "decode == verify invariant"):
`mmvq_mmid_max_batch_band(cap)` floors the per-type cap at `MMVQ_MAX_BATCH_SIZE` for every AMD arch
lookup, host and device, and `mul_mat_vec_q_moe`'s launch bound becomes `MMVQ_MAX_BATCH_SIZE*warp_size`.
All four cap call sites are `MUL_MAT_ID`-only, so **dense models are untouched** (verified: 4B
bit-identical and perf-identical).

**Validation** (3× gfx1201, f16 KV, P=256): `W = 1..8` all `3adeb313042a871b` (`-sm layer`) and
`dcf1ae667f730879` (`-sm tensor`) — every width equals that split's **pre-fix `W = 1` value**, so plain
decode is bit-unchanged and only `W = 5..8` moved (also pure with `RS=from_w`).  At the MTP gate config
(`n_max 3` = `W=4`) pre/post-fix runs are byte-identical (acceptance 0.76744, 80.0 vs 80.1 t/s); at
`n_max 7` the fix gives **41.8-42.5 vs 36.1 t/s (+16-18 %)** and acceptance 0.59375 vs 0.55556, and
`n_max 3` == `n_max 7` text (`8a50ea24e8d5`) where they previously disagreed.  Perf
(`llama-batched-bench`, interleaved, fixed vs baseline): qwen4exp tg128 b5 **149.5/118.4 (+26 %)**,
b6 **162.5/130.6 (+24 %)**, b7 **171.4/147.0 (+17 %)**, b8 **178.0/155.4 (+14.5 %)**; 35B-A3B MoE b8
**341.3/289.9 (+17.8 %)**; 4B dense unchanged.  `GATED_DELTA_NET` and `FLASH_ATTN_EXT` 4/4 backends OK;
MoE asterisk intact (`ac8825358d9adfda`/`bd138ad2326fbbf2`); clean-apply strict 15/15, 0 whitespace
warnings, tree `4e5f2952f016f1ac160c53261f7b01d346322534`.

**Note (open, not this amendment; superseded 2026-09-11 — see the 2026-09-11 block-14 amendment
section above and `../GREEDY-PURITY.md` §16):** qwen4exp `plain` text still differs from `draft-mtp` — that is a
*pre-existing, independent* multi-step/roll-back effect (the fix is a verified no-op at `n_max 3`/`W=4`),
**localised 2026-09-11 (further measurement): it is in the QSA *machinery*, and the site class is the same as cause 1's.**  `LLAMA_QSA_OFF=1` makes `plain` == `draft-mtp --spec-draft-n-max 3` **byte-identical** (`d4499ac8db72` both, 711 chars) — and the knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`) — while `LLAMA_QSA_SPARSE_FA=0` (dense attention, indexer still on) leaves two different texts (`25f300a81b9e` vs `0d466b2dcf09`), so the defect is **not** the sparse-FA kernel but the **indexer/score machinery** (`indexer-topk.cu` + the `qwen4exp.cpp` gates).  Both QSA-side `n_tokens == 1` gates are the prime suspects — `src/models/qwen4exp.cpp:1094` (`idx_score_fused`, the fused indexer score) and `:1419` (`qsa_dense_decode_until`, the early-decode dense shortcut) — i.e. exactly the cause-1 pattern, and the single-step width probe cannot see them because it never reaches the sparse/indexer decode regime.  The divergence appears only after ~100 chars (~20 tokens) of a 3.3k-prompt greedy run (the first steps agree), so it is not a prefill-state difference; `GGML_CUDA_GDN_CHUNKED=0` moves both sides without making them agree (the known Issue #25 chunked-prefill item is a separate contributor, not this).  **Kill-switch for users meanwhile: `LLAMA_QSA_OFF=1`.**  See `GREEDY-PURITY.md` §15.

**2026-09-11 (second amendment) — MoE `MUL_MAT_ID` decode/verify dispatch + the shared-expert fusion kill-switch.**

- **`MUL_MAT_ID` at `ncols_dst == 1` now uses the dedicated MoE kernel.**
  `mul_mat_vec_q_switch_ncols_dst` used to return early only for `has_ids &&
  ncols_dst > 1`, so a single-token `MUL_MAT_ID` fell through to the *dense*
  ksplit kernel (with an ids gather) while the 2..8-token verify batch ran
  `mul_mat_vec_q_moe`.  Two kernels with different accumulation orders ⇒ the same
  MoE matmul was not bit-identical between a 1-token decode and an n-token verify
  batch.  The dense half of this was fixed earlier the same day (dense rows always
  ksplit); this closes the MMID half.  Cost/benefit: **+6.2% MoE decode** (tg128
  95.62 → 101.52), +1.4% pp512, dense unchanged (tg 31.95 → 32.00).
- **The fused shared-expert window is decode-only and NOT bit-exact with the
  unfused chain.**  `ggml_cuda_op_shexp_down_gate` computes `dst = down(swiglu) *
  sigmoid(gate(x)) + moe_out + ffn_residual` in one kernel, gated on
  `ne[1] == 1`.  Its gate dot (`shexp_gate_sigmoid`) does not reproduce the
  standalone mmvq/MUL_MAT reduction order, and its epilogue multiply was
  contracted into an FMA.  The FMA is removed (`__fmul_rn`); the gate order is
  not, so this window remains the **accepted MoE decode≠verify residual** (MoE is
  exempt from byte-identity by `benchmarks/mtp-adaptive-methodology.md` rule 3,
  and its MTP gate passes: acceptance 0.58378, unchanged from canonical).
  **Kill-switch: `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`** — with it (and the
  dispatch fix above) qwen35moe decode is bit-identical to verify (`bd138ad2`).
  The fusion is worth +3.1% MoE decode (101.5 vs 98.5 t/s), hence ON by default.

**Fused MoE gate+up+GLU MMQ (prefill) + mmvq short-K item-split (decode).**

- **Prefill fused expert MMQ** (block 13, the `mul_mat_id_glu_ops` pattern):
  the {MUL_MAT_ID(gate), MUL_MAT_ID(up), GLU} triple runs as ONE MMQ kernel
  reading both weight streams with a GLU epilogue in registers.  Types
  instantiated: Q3_K/Q4_K/Q5_K/Q8_0/Q6_K (M4 quant extension).  Env opt-out:
  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`.  Decode-side shared-expert fusion opt-out:
  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`.
  Validated (1-GPU qwen35moe Q6_K/Q4_K_M, the verified config): prefill
  pp16384 Q6_K +5.1% (3344 vs 3181), Q4_K_M +3.6% (3488 vs 3367); fused
  path fires as `FUSED MUL_MAT_ID ffn_moe_down-*` on all layers.
- **Decode item-split** (mmvq `mul_mat_vec_q`/`mul_mat_vec_q_moe`): the
  K-split loop leaves most thread groups idle on short-K MoE GEMMs (down
  K=512 -> 2 K-blocks); the item-split loop spreads (row, kblock) items
  over the groups and scales rows_per_block (rpb 2/4/8) to fill them.
  Re-based on top of the upstream `has_fusion` mmvq path (41ef91f7c),
  which landed in the 0eadefebd re-base - the launcher now dispatches on
  rpb x has_fusion.  Validated: decode tg128 97.28 vs 92.15 pristine
  (+5.6%, 1-GPU Q6_K).
- **Correctness gates** on the try_fuse arms (the 0eadefebd merge
  admitted cases the kernels could not express):
  1. The `x_scale_channel_dst` fold (MoE down x topk weights) now
     supports multi-token MUL_MAT_ID (2026-09-02): the `mul_mat_vec_q_moe`
     epilogue applies `x_scale[channel_dst + token_idx*nchannels_dst]`, one
     scalar per (expert, token), matching the topk-weights layout
     [1, n_expert_used, n_tokens].  try_fuse gates on the weights shape
     (`weights->ne[2] == mm_node->ne[2]`); the launcher assert allows
     nelements == ne1*ne2.  Spec-dec verify batches n=2..8 (up to
     `get_mmvq_mmid_max_batch`) now fuse instead of the separate MUL.
     Validated: test-backend-ops 16222/16222 (multi-token n=4 exercises
     the moe kernel, bit-exact vs CPU ref; sweep extended with
     Q8_0/Q6_K/Q5_K/Q3_K/IQ2_XS); single-token decode unchanged
     (tg128 82.8-83.2).
  2. The fused MoE MMQ arm is gated to the instantiated type list
     (Q3_K/Q4_K/Q5_K/Q8_0/Q6_K): `ggml_cuda_should_use_mmq` returns true
     for q4_0/q4_1/q5_0/IQ/MXFP4/NVFP4 on RDNA4, which would abort in
     `ggml_cuda_mul_mat_q_switch_type_gate`.  MXFP4/NVFP4 support tracked
     in TODO.md.
- Same-seed coherence IDENTICAL (fusion on vs off), test-backend-ops
  2/2 OK.
- 2026-09-01 (block-13 amendment): fixed the ROCm multi-GPU split-load
  pathology this block's qwen35moe validation exposed.  H2D 2D copies
  with a width not multiple of 4 (Q6_K/Q3_K quant blocks are 210/110
  bytes) take ~1000x longer on ROCm (~2300ms vs ~10ms per tensor), so
  Q5_K/Q6_K 2-GPU tensor-split loads took ~3min and looked like hangs
  (the earlier "hangs at EVERY commit" finding was a misdiagnosis -
  every build was just slow-loading).  `set_tensor_2d` now stages
  through device memory with an aligned width + unaligned D2D gather
  (byte-identical, memcmp 0).  Q6_K/Q3_K 2-GPU load now <15s, pp512
  ~4200-4500 t/s and tg32 ~66-82 t/s matching the pre-regression docs
  numbers; Q8_0 unchanged.  The slow-load is present in pristine
  upstream 0eadefebd too (upstream bug, worth filing); the fix ships
  here because the feature it unblocks (qwen35moe 2-GPU MoE) is
  block-13's.
- **Benchmark configs (2026-09-02):** all 1-GPU numbers in this project's
docs require `HIP_VISIBLE_DEVICES=0`; without it llama.cpp layer-splits
across all 3 R9700s and decode drops ~97 -> ~81 t/s (a harness artifact,
NOT a regression - verified 2026-09-02). Canonical command lines + the
baseline table live in `wip/qwen35moe-prefill/bench-config.md`.
- **MTP/verify decode regression fix (2026-09-02):** the decode item-split
  kernel + RDNA rows_per_block override collapsed multi-token decode
  batches (ncols 2..8 = the speculative/MTP verify step) on DENSE models,
  and cost ~4% on long-K (K >= 4096) single-token decode.  The per-thread
  accumulator fan-out `tmp[ncols_dst][rpb]` (e.g. a 4-token verify x
  rpb<=16 = up to 64 registers/thread) is register-bound; dense models hit
  it because their verify batch goes through the plain `mul_mat_vec_q`
  (MoE batches use `mul_mat_vec_q_moe`, which was unaffected).  Fix:
  re-added the pre-block-13 K-split kernel as `mul_mat_vec_q_ksplit` and
  dispatch decode batches ncols 2..8 to it; at ncols==1, rows with K >=
  4096 (dense qkv/FFN projections, any quant type) also use ksplit while
  short-K MoE rows (K < 4096) keep the item-split/rpb path.  Verified
  (1x R9700 gfx1201, seed-42 protocol): dense qwen35 27B Q4_K_XL
  adaptive-MTP 18.3 -> 27.5 t/s with output bit-identical to the 12-block
  build, plain decode 29.0 -> 30.1 (+3.1-3.7% at d0/d16384/d65536);
  qwen35moe A3B adaptive-MTP 36.3 -> 55.8 t/s with single-token decode
  unchanged (Q6_K tg128 98.3, recorded baseline 97.59).  The 12-block-era
  build (no block 13) shows the same collapse (16.6 t/s), i.e. this was
  inherent to block 13, not a re-base artifact.
- **MoE MTP verify-numerics regression fix (2026-09-02, second fix):** with
  the first fix in, MoE MTP was still far below plain decode (draft-mtp 53
  vs none 90 t/s on qwen35moe-A3B Q4_K_M-UD) while upstream accelerates
  (+51%).  Root cause: the block-08 rms_norm->mmvq Q8_1 quantize-cache
  fold (try_fuse arm) corrupts multi-token MUL_MAT_ID - the moe-kernel
  path consumes the cached Q8_1 y incorrectly, so verify-batch logits
  diverge from single-token decode and MTP draft acceptance collapses to
  0/1527.  MoE MTP was never baseline-tested (no MTP data existed for
  qwen35moe), so nothing caught it.  Fix: gate the fold to single-token
  MMID (ne[2]==1) and plain MUL_MAT consumers; multi-token MMID decodes
  unfused (same numerics as the unfused path).  Verified: MoE A3B
  draft-mtp acceptance restored to 0.51 (== fully-unfused 0.49 ==
  upstream 0.49; the residual fusion-ordering drift does not depress
  acceptance), rate 119-129 t/s vs upstream ~110-113; plain decode and
  single-token fusion gains unchanged (none 89-95, Q6_K tg128 98.8);
  dense unaffected (mtp 27.2-27.5 / none 30.1).  The MTP gate protocol +
  baselines now live in `benchmarks/mtp-adaptive-methodology.md`.

- **RDNA3_5 (Strix Halo, gfx1151) validation (2026-09-05, folded into
  block 13):** the fused gate+up+GLU MMQ arm (`ggml-cuda.cu` try_fuse)
  and the `J_max_gate` tile-width caps (`mmq.cuh`) were RDNA4-only
  ("disabled until validated on other arches").  Validated on a Ryzen AI
  MAX+ 395 / Radeon 8060S (ROCm 7.14, gfx1151) with Qwen3.6-35B-A3B
  True-Q3_K_M (Q3_K is in the fused type list), ub 2048: same-seed
  coherence IDENTICAL fused-on vs off; the gate is now RDNA4 + RDNA3_5
  and the RDNA4-tuned caps apply on both.  Gains match RDNA4:
  pp2048 1590 -> 1674 (+5.3%), pp16384 1360 -> 1423 (+4.6%), pp512
  ~+14% (noisy, single ubatch); decode unchanged (tg128 71.5).  The
  caps transfer: uncapping J (128) on gfx1151 regressed pp2048 1674 ->
  1111 and pp16384 1423 -> 1334 (register pressure).  Full record:
  `../wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
- **RDNA3_0 (gfx1100, RX 7900 XTX) validation (2026-09-05, folded into
  block 13):** the remaining excluded arch is now ungated — the same
  try_fuse arm + `J_max_gate` caps apply on RDNA3_0 (gfx1100) too.
  Validated on a single RX 7900 XTX (ROCm 7.14, gfx1100, 1-GPU pinned
  with `HIP_VISIBLE_DEVICES=0` to exclude the box's HIP-visible
  gfx1036 iGPU) with Qwen3.6-35B-A3B True-Q3_K_M, ub 2048: fusion
  fires (one-time session log), same-seed coherence IDENTICAL fused-on
  vs off (and the ungated 3-op fallback output is byte-identical to
  the pre-ungate build), gains pp2048 4939 -> 5405 (+9.4%), pp16384
  4162 -> 4487 (+7.8%), pp512 ~+20% (noisy), decode unchanged (tg128
  130.3 vs 130.4).  The RDNA4-tuned J caps transfer: uncapping (J=128)
  on gfx1100 regressed pp2048 5405 -> 4819 and pp16384 4487 -> 4070
  (below the 3-op fallback), and a Q3_K@96 probe (5094/4251) also lost
  to the cap 64 — no per-arch port tuning needed.  Block 12 stays N/A
  here (single GPU); the dual-7900XTX block-12 leg remains a separate
  parallel task.  Full record:
  `../wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
- **moe_weighted_reduction float4 remainder fix (2026-09-08, folded into
  block 13, issue #19):** the 2026-09-06 mwr-float4 fold (f33ffaca7)
  rewrote the kernel and launcher to index in quads with floor division
  (`n_embd / 4`) and no remainder handling, silently leaving the last
  `n_embd % 4` columns of every output row unwritten for `n_embd % 4 != 0`
  (wrong results, not a crash — `MOE_WEIGHTED_REDUCTION` with `n_embd = 63`
  failed both cases, ERR ~0.09-0.13).  A vectorized kernel is only valid
  when every expert row starts 16B-aligned, i.e. `n_embd % 4 == 0`; the
  kernel is therefore split into the float4 quad variant (launched when
  `n_embd % 4 == 0`, byte-unchanged aligned path — real models have
  `n_embd % 4 == 0`) and the upstream scalar bounds-checked kernel for
  the remainder.  Verified (3x R9700 gfx1201): MOE_WEIGHTED_REDUCTION
  6/6, full test-backend-ops 16590/16590.
- **Dense decode/verify MMVQ kernel alignment (2026-09-11, folded into
  block 13):** the block-13 MTP fix above left one asymmetry: at
  `ncols_dst == 1`, dense (non-`MUL_MAT_ID`) rows with `K < 4096` stayed
  on the block-13 item-split kernel while ncols 2..8 unconditionally use
  the ksplit kernel (and `K >= 4096` ncols==1 already used ksplit).  The
  two kernels accumulate K in different orders, so a single-token dense
  `MUL_MAT` was **not** row-identical to the same row inside a 2..8-token
  verify batch — a ~1e-6 logit difference at the first such projection,
  amplified by the recurrent GDN into greedy flips.  Effect:
  `--spec-type none` and `draft-mtp` produced different text (a) on small
  dense models with `K = n_embd < 4096` (e.g. Qwen3.5-4B) and (b) under
  `--split-mode tensor` on any model whose per-GPU K shard drops below
  4096 (Qwen3.8-27B: 5120 -> 2560).  Fix: dense ncols==1 rows use ksplit
  for **every** K (condition `!has_ids || ncols_x >= 4096`); the MoE
  (`MUL_MAT_ID`) rows keep the block-13 item-split + rpb path — their
  multi-token path is the dedicated `mul_mat_vec_q_moe` kernel (one warp
  per token), so the row-bit-identity invariant holds there without
  touching the short-K MoE decode win.  Verified (1x R9700 gfx1201,
  per-process token-0 logit hash, chunked GDN off): Qwen3.5-4B Q8_0
  1-GPU W=1/3/5 bit-identical; Qwen3.8-27B Q8_0 2-GPU tensor W=1/3/5
  bit-identical (was W1-W3 = 0.133).  Perf neutral (llama-bench, 1 GPU):
  4B pp512 7714 -> 7680 / tg128 100.26 -> 100.65; 27B pp512 1394 -> 1390
  / tg128 20.42 -> 20.42; MoE-A3B Q4_K_M pp512 4804 -> 4802 / tg128
  95.66 -> 96.02.  MTP gates unchanged/healthier (dense 27B acceptance
  0.487 / 36.5 t/s, MoE 0.675 / 153.1 t/s); `GATED_DELTA_NET` 46/46 and
  the hybrid-vs-NCCL comparison is text-level only and does not hold under
  `-sm tensor` (the internal path always BF16-round-trips while NCCL reduces
  small tensors in FP32 — see the 2026-09-11 WORKLOG entry on the AR backends);
  `GATED_DELTA_NET` was 46/46.  **Companion:** the
  default-config `-sm tensor` text equality **also** needs the block-02
  K-independent whole-batch chunked GDN prefill (2026-09-11) — both paths
  chunk the whole prompt, so the post-prefill state no longer depends on
  `n_rs_seq` — and it is **free** (no sequential tail; pp parity).
  With both, 27B 2-GPU and 3-GPU tensor and 1-GPU are
  `none == n1 == n4 == n6 == n7` for `n_max <= 7` (the designed pure range;
  the 2-GPU tensor case reached it only after the block-12 dispatch fix of
  2026-09-11 — see `GREEDY-PURITY.md` §11).  See
  `wip/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.

## Server config (the +22% deployment win)

`HIP_VISIBLE_DEVICES=0,1,2` (3-GPU), hybrid default, **unpinned** (the
dpm=high/runtime-PM pin is a regression: tg -5-7%, pp -15-18% on RCCL/hybrid
paths).  Depth-16384: 31.79 (2-GPU) -> 38.71 (3-GPU) t/s.
