# `wip/closing-the-gap/` — the prefill/decode gap to the other solution's `strix-halo`

**Status: WIP planning record. Not part of the delivery.**  Started 2026-09-20 (analysis on gfx1151),
moved here and updated 2026-09-21.

## What is here

| file | what |
|---|---|
| [`closing-the-gap.md`](closing-the-gap.md) | the living analysis. §0–11 are the **2026-09-20 snapshot** (dated measurements); the **Update 2026-09-21** block, **§12** (MTP qualification) and **§13** (phased plan) are current. |
| [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) | the MTP qualification record: plain-vs-MTP on qwen4exp IQ4_NL, ours vs the other solution's, and the `nextn_shared_target_tensors` finding. |
| [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md) | Phase-1 item 1: the `hc_combine_norm` matcher root cause (three bugs) + the `hc_gate_mix` wire-up; +1.5 % / +1.2–1.5 % prefill on fork branch `gap-closing`. |
| [`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md) | Phase-1 item 2: the depthwise conv1d (`gdn-conv.cu` + `ple-conv.cu`) port, default-on, bit-identical, +3.0/+3.2 % qwen4exp IQ4_NL and +6.5/+7.1 % 35B-A3B at `-ub 8192`. |
| [`2026-09-21-hc-cn-b256-rejected.md`](2026-09-21-hc-cn-b256-rejected.md) | Phase-1 item 1's `_b256` follow-up: ported, gated, **closed negative** (not bit-identical, slower); the reference's 554 ms is its BF16 HC traffic, not the thread count.  Also notes item 5's `concat_transposed` is already gone at `-ub 8192`. |
| [`2026-09-22-qsa-block-window-fix.md`](2026-09-22-qsa-block-window-fix.md) | Phase-1 item 3.5 (first fix): the QSA block window is now sized by the highest stored position (`b0f31f587`), for the M-RoPE-image + MTP crash. |
| [`2026-09-22-qsa-item-3.5-audit.md`](2026-09-22-qsa-item-3.5-audit.md) | Phase-1 item 3.5 (**closed**): the other two correctness fixes (`40c0b9c38` maskless, `14fff4f97` −1 sentinels) are **N/A** in our tree — we have no maskless path and our top-k output never carries sentinels; the invariants they protect are already held. |
| [`2026-09-22-norm-rows-fusion.md`](2026-09-22-norm-rows-fusion.md) | Phase-1 item 4: the narrow-row RMS norm (`norm-gated.cu::rms_rows_f32`, 8 rows/block) ported default-on, **bit-identical** (width probe PASS, same-seed text `f61199ba5644`), ~+0.3 % at `-ub 4096`. |
| [`2026-09-22-ubatch-8192-memory-confound.md`](2026-09-22-ubatch-8192-memory-confound.md) | **Methodology finding:** `-ub 8192` runs at 2–3 GB free with ~40 % more reclaim, which can bias an A/B whose arms differ in graph-shape memory; use **`-b/-ub 4096`** for A/B.  The prior rejections audited (the shipped wins are unaffected). |
| [`2026-09-22-qsa3-visibility-fold.md`](2026-09-22-qsa3-visibility-fold.md) | Phase-1 item 6: fold the per-cell QSA visibility into `umask` at merge time (drops the hot-loop check), **bit-identical**, **+2.4 % pp8192 / +1.7 % pp32768**; the QSA pipeline is now 50 ms ahead of the reference's. |
| [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md) | Phase-1 item 7: keep the `M=4` HC inject out of the 384-row tall MMB tile (the tall kernel ran 2× the dispatches), **bit-identical**, **+0.8 % pp8192 / +1.1 % pp32768**. |
| [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md) | Phase-1 item 8 (**closed**): 7/9 QSA graph flags are present/superseded in our block-14/15 QSA; `QSA_SCORE_BOUNDS`+`QSA_QUERY_STRIP` is now ported (session 7), leaving `QSA_SCORE_WMMA` — the last follow-up. |
| [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md) | Phase-1 item 16: the HC BF16 streams (`blk16`/`res16`) ported **default OFF**, default byte-identical, **+4.9 % pp8192 / +4.8 % pp32768** at `-b/-ub 4096`; `res16` is the dominant half.  The MoE-merge `ffn_out` ADD stays F32 because it is not adjacent to the reduction chain — and the reference's merge path is equally dormant on qwen4exp (same builder), so it is not a gap against it. |
| [`2026-09-22-mmb-cvt-out-xn.md`](2026-09-22-mmb-cvt-out-xn.md) | The `mmb_cvt_f32_bf16` gap: the fused combine now emits the BF16 `out_xn` copy the graph already marks (81 % of the traffic; the allocator reuses one `hc_norm` buffer so the cache cannot dedupe), **bit-identical**, **+3.3 % pp8192 / +3.2 % pp32768** at `-b/-ub 4096`. |
| [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md) | The prefill indexer relu+head-sum fusion (`idx-relu-sum`): one kernel replaces the relu + CONT + ADD chain (our L2a relu-before-reshape form), **bit-identical**, **+1.8 % pp32768** at `-b/-ub 4096`. |
| [`2026-09-22-qsa-score-bounds.md`](2026-09-22-qsa-score-bounds.md) | Phase-1 item 15: the QSA prefill scorer trim (`QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`) ported to the fused top-k (cell range clamped to `n_blocks*ratio`), default strip 1024, **bit-identical**, **+0.5 % pp8192 / neutral pp32768** at `-b/-ub 8192`; the `-inf`-padded first cut was a wash. |
| [`2026-09-22-qsa-score-wmma.md`](2026-09-22-qsa-score-wmma.md) | Phase-1 item 15 (follow-up): `QSA_SCORE_WMMA` — the AMD RDNA3_5 4-head/128-dim lightning-indexer WMMA kernel ported + the qwen4exp prefill score fused into one `ggml_lightning_indexer` (all-ones weights, zero F16 mask) composed with the causal trim.  **Op oracle 225/225 (81 new `nh=4` cases)**; **width probe PASS, per-W hashes byte-identical to off**; coherent re-baseline text; **pp32768 +1.0 % / +0.9 %**, pp8192 flat.  **default ON** (`LLAMA_QSA_SCORE_WMMA=0` disables), `patches/0016`. |
| [`2026-09-22-mmb-quant-coverage.md`](2026-09-22-mmb-quant-coverage.md) | Phase-2 item 10: five more MMB weight types — **Q4_0 / Q4_1 / Q5_0 / MXFP4 / NVFP4** (WTYPE 11–15), dequant-vs-CPU oracles green (`MUL_MAT` 48/47/14/46/45, `MUL_MAT_ID` 74/75/3/74/73), PPL parity, pp8192 **+19.5/+22.4/+25.4 %**, 35B-A3B Q4_1 **+63 %**, gpt-oss MXFP4 **+5.2 %**.  **default ON on non-RDNA4**, `patches/0017`. |
| [`2026-09-22-mmb-iq2-coverage.md`](2026-09-22-mmb-iq2-coverage.md) | Model-tree scan (`tools/gguf-types.py`, 132 files) + the **IQ2 family** — **IQ2_S / IQ2_XS / IQ2_XXS** (WTYPE 16–18): oracles 14/14/46 and 4/15/75, real MiniMax IQ2_S **pp4096 +4.9 %** (PPL +1.07 %), dense IQ2_XS **pp8192 +16.5 %** (PPL +0.22 %).  **default ON on non-RDNA4**, `patches/0018`. |
| [`PLAN-mtp-sparse-draft.md`](PLAN-mtp-sparse-draft.md) | **The plan for the sparse MTP draft — IMPLEMENTED 2026-09-22; see the record below.**  Make the qwen4exp MTP draft attend **sparse** (QSA) like the trunk: the memory change (MTP context hybrid-idx), the nextn compress-ratio fallback, and the `graph_mtp` QSA routing, with gates + traps.  Justified by the measured draft dense-attention share: **2.1× all twelve sparse trunk layers at ~150K prefill, 5.6× decode**, growing with depth. |
| [`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md) | **Sparse MTP-draft attention, implemented 2026-09-22, `patches/0020` (OPT-IN, `LLAMA_MTP_SPARSE=1`) + `patches/0021` (derived-indexer default ON) + `patches/0022` (gfx1151 decode crossover 64K → 32K).**  The three plan edits + two memory fixes the plan missed (the empty MTP recurrent child aborts a partial `seq_rm` and spams the non-consecutive warning).  **Prefill pp150K +6.9 %** with a 32K depth gate (927.0 → 990.8 t/s).  **Decode root cause: the incremental QSA indexer (`GGML_CUDA_QSA_INDEXER_CACHE`) was OFF by default** though the graph expects it on; enabling it is **byte-identical** and gives **+9.1 % @80K / +14.6 % @150K** plain decode (f16; +8.7 % bf16), after which the sparse draft decode is **parity / a slight win** (80K: 30.7-31.5 vs 30.7-30.9 t/s, acceptance 0.812 vs 0.792) — the selected-cell `flash_attn_qsa` is 0.05 ms/call vs the dense `flash_attn_tile` 1.03 ms, and it was the per-step indexer that ate the saving.  5K purity PASS (`3553e76d3a9e`), width probe PASS, MTP acceptance unchanged (0.85035), oracles green (QSA 26/26, GDN 46/46).  **Draft default OFF only because the campaign's `GGML_CUDA_MMB_HC16` F32-elision breaks purity under MTP/speculation at depth** (root-caused 2026-09-22: 40K MMB-on plain=draft=`8285d12d40ca` but sparse=`c0a3bda5dff4`; 128K MTP is 5-runs-5-hashes; with `GGML_CUDA_MMB_HC16=0` **all of it is byte-identical to plain** and 128K MTP is deterministic).  Same mechanism as the session-9 eval-callback fix (`patches/0019`).  The draft itself is pure. |
| [`2026-09-22-phase2-sparse-qsa-audit.md`](2026-09-22-phase2-sparse-qsa-audit.md) | **Phase-2 item 9 audit, session 9 — no port.**  `d67d58836`'s selected-cell decode is already our `fattn-qsa.cu::flash_attn_qsa` (`LLAMA_QSA_SPARSE_FA`, default ON, more general) and its incremental indexer is already our derived block-vector cache (`GGML_CUDA_QSA_INDEXER_CACHE`, `pool_layers`/`pool_wm`, default ON); the reference's `qsa_keys[il]=[idx_dim,n_blocks]` is the same block-key cache.  The measured +11–20 % is sparse-recompute vs sparse-incremental, which we hold.  The **one real gap** is the MTP-draft sparse attention (our `graph_mtp` is dense); schedule it as a targeted A/B, not a port. |
| [`2026-09-22-mmb-eval-callback-f32.md`](2026-09-22-mmb-eval-callback-f32.md) | **Correctness fix, session 9** (NEW-SESSION item 2), `patches/0019`: the MMB **HC16** F32-elision is invalid under an **eval callback** (llama-imatrix, `common/debug`) — the scheduler splits the graph at callback nodes and `mmb_begin_graph()` clears the BF16 cache between producer and GEMM, so the GEMM re-converted the never-written F32 (`non-finite values detected in blk.21.attn_output.weight`, 19M PPL).  Fixed by plumbing `has_eval_callback` through `ggml_backend_graph_optimize_params` and standing the elision down in that mode.  imatrix `in_sum2` **byte-identical** to `HC16=0`; width probe PASS on NanBeige BF16 + qwen4exp; serving perplexity unchanged. |
| [`2026-09-22-mtp-shared-nextn-fix.md`](2026-09-22-mtp-shared-nextn-fix.md) | **Correctness fix, delivered in block 00 (r13)**: a shared-NextN MTP head (`nextn_shared_target_tensors`, the IQ4_NL shared Q8_0 sidecar) died every round on the M-RoPE `X < Y` check because `is_mem_shared` was inferred from `ctx_other` alone; gated on the `gemma4-assistant` arch.  0 errors, acceptance 0.287.  Upstream bug (#23398) folded into the block-00 base; the WIP `patches/0015` is superseded. |
| [`patches/`](patches/) | the fork `gap-closing` commits exported as patches, so the code work survives a fork reset.  On the **r13 rebuild** the campaign is `0001..0014`; **`0015` (the shared-NextN MTP fix) is superseded by delivery r13 block 00** — skip it.  `0016` = `QSA_SCORE_WMMA` (default ON), `0017` = MMB quant coverage Q4_0/Q4_1/Q5_0/MXFP4/NVFP4, `0018` = MMB IQ2_S/IQ2_XS/IQ2_XXS (both default ON on non-RDNA4), `0019` = the HC16 F32-elision correctness fix under an eval callback, `0020` = the sparse MTP draft (**OPT-IN**, `LLAMA_MTP_SPARSE=1`), `0021` = the QSA derived-indexer default ON (byte-identical, +9-15 % deep decode), `0022` = the gfx1151 QSA decode crossover 64K → 32K (a 32K-64K decode re-baseline). |
| [`tools/`](tools/) | `gguf-types.py` — header-only GGUF tensor-type scanner (no tensor data read); used for the Q2_*/IQ2_* model-tree sweep. |

## The two moving references this file tracks

* **Ours:** `beta/mmb-general/` — **12 patches**, applied tree
  `bca69f23dd29acef2d8898c6fd492104e078eef1`, `git am` 12/12 on top of the r12 delivery
  (`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).  The body first analysed the pre-beta
  5-patch WIP (`90bf12997`, `~/llama-wip-mmb`).
* **The other solution:** `~/pwilkin-llama-cpp`, branch `strix-halo`.  The body pinned `f5daaa3cf`
  (2026-09-12); the tip is **`b0f31f587`** (2026-09-16), 10 commits ahead.

## Current "our side" build state

* **Current (end of session 10):** `~/llama.cpp` branch **`gap-closing-r13`** @ **`1bb1d794e`**
  (tree `ad7fb9bc…`) = delivery r13 + the 12 `beta/mmb-general/patches/*.patch` + gap-closing
  `0001..0014`/`0016`/`0017`/`0018`/`0019`/`0020`: session 8's `QSA_SCORE_WMMA` + MMB quant coverage,
  session 9's HC16 eval-callback fix, and session 10's **sparse MTP draft (OPT-IN)**.  The historical
  summary below (branch `gap-closing` @ `00d8bbbc9`) is kept for the session 1-7 record:
* `~/llama.cpp` branch **`gap-closing`** @ **`00d8bbbc9`** = `mmb-beta` (r12 `72176ae8a` + the 12
  `beta/mmb-general/patches/*.patch`, tree `bca69f23dd…`) + the 2026-09-21/22 changes: **default-on
  policy** (MMB/HC16/matcher), the `hc_combine_norm` matcher revival, the **`hc_gate_mix` fusion**
  (session 2), the **depthwise conv1d fusions** (session 3), the **QSA block-window fix** (session 3,
  `b0f31f587`), the **narrow-row RMS norm fusion** (session 4), the **QSA visibility fold**
  (session 4, item 6), the **tall-tile min-M** fix (session 4, item 7), the **`-lzm auto` semantics**
  + managed PLE reader gated OFF (session 5), the **MoE BF16 epilogue** gated OFF (session 5), the
  **HC BF16 streams** gated OFF (session 6, item 16), the **`mmb_cvt`/`out_xn` fix** default ON
  (session 6, `patches/0012`), the **prefill indexer relu-sum** default ON (session 6,
  `patches/0013`), and the **QSA prefill scorer trim** default ON (session 7, `patches/0014`), plus
  env-gated debug traces.  **Current product:** qwen4exp IQ4_NL, gfx1151,
  `-b 8192 -ub 8192` with `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1` = **~1379 / 1320 t/s** (pp8192 /
  pp32768, plus the trim's ~+0.5 % pp8192); the default build at `-b/-ub 4096` = ~1308 / 1270.
* Built on this box (gfx1151) with `~/bin/build-llama-rocm-714`.  **All beneficial features are on by
  default** (see the `AGENTS.md` default-on policy); env vars only disable.
* The campaign's patches are exported to [`patches/`](patches/) in case the local fork branch is
  lost: `0001..0014` + `0016..0018` + `0019` (HC16 eval-callback fix) + **`0020` (sparse MTP draft,
  opt-in)** + **`0021` (QSA derived-indexer default ON)** + **`0022` (gfx1151 QSA decode crossover
  64K → 32K)**; `0015` is superseded by r13 block 00.

To reproduce (the **r13 rebuild**):

```sh
cd ~/llama.cpp
git checkout rdna-boosts-r13 && git branch -D mmb-beta gap-closing-r13 2>/dev/null
git checkout -b gap-closing-r13
git am /home/stew675/llama-cpp-rdna-boosts/beta/mmb-general/patches/*.patch
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/00{01,02,03,04,05,06,07,08,09,10,11,12,13,14}-*.patch   # 0015 is in r13 block 00
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0016-*.patch   # QSA_SCORE_WMMA, default ON
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0017-*.patch   # MMB Q4_0/Q4_1/Q5_0/MXFP4/NVFP4
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0018-*.patch   # MMB IQ2_S/IQ2_XS/IQ2_XXS
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0019-*.patch   # HC16 F32-elision under an eval callback
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0020-*.patch   # sparse MTP draft (OPT-IN: LLAMA_MTP_SPARSE=1)
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0021-*.patch   # QSA derived-indexer default ON
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/0022-*.patch   # gfx1151 QSA decode crossover 64K -> 32K
~/bin/build-llama-rocm-714
```

Fork tip after session 10: **`d8334f929`** (`gap-closing-r13`, tree
`fa185bbb453d6de627427ae4f8d868127fa72535`).  The scratch build above is the tree the rebuild was
verified on (`git am` 12/12 + 14/14 + 1/1 + 1/1 + 1/1 + 1/1 + 1/1 + 1/1 + 1/1, no conflicts; applied tree == fork tree).

## Do first (fresh session, in order)

> **Updated end of session 10.**  Items 1–3 below are historical; the campaign now starts at
> “Next” — see the **NEXT SESSION** block of [`closing-the-gap.md`](closing-the-gap.md).

**Session 9:** (a) the **sparse QSA decode + incremental indexer** (Phase-2 item 9, reference
`d67d58836`) — **AUDIT DONE 2026-09-22**: no port, both halves are already in our tree
(`flash_attn_qsa` + the derived block-vector cache); only the MTP-draft sparse attention is a targeted
follow-up ([`2026-09-22-phase2-sparse-qsa-audit.md`](2026-09-22-phase2-sparse-qsa-audit.md)); (b) the
**pre-existing BF16-MMB non-finite** bug — **DONE**, the MMB **HC16** F32-elision under an eval
callback, fixed by `patches/0019` ([`2026-09-22-mmb-eval-callback-f32.md`](2026-09-22-mmb-eval-callback-f32.md)).

**Session 10 — the sparse MTP draft is IMPLEMENTED, `patches/0020`, OPT-IN.**
[`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md): the three plan edits plus two
memory fixes the plan missed (the empty MTP recurrent child aborts a partial `seq_rm` and spams the
non-consecutive warning).  **Prefill pp150K +6.9 %** with a 32K depth gate (927.0 → 990.8 t/s),
**decode a loss** at every measured depth, so the decode arm is off.  **Default OFF** because at 40K
the sparse prefill changes the greedy text while the dense draft matches plain; the target is
logit-width-pure there and even dense MTP diverges at 150K, so the blocker is the iterative
target verify/rollback (a pre-existing MTP-at-depth purity gap), not the memory change.  Remaining
next items: the **gfx1100/gfx1201 validation** of the session-8 additions, and root-causing the
depth verify/rollback divergence (then the sparse draft can be defaulted on).  Detail in
[`closing-the-gap.md`](closing-the-gap.md)'s NEXT SESSION block.

1. **Run the full BETA-TESTING gate suite** — **DONE (session 8)** on gfx1151: Gate 4 MTP
   qwen4exp acceptance **0.85541** (56.5 vs plain 31.7 t/s), `LIGHTNING_INDEXER` 225/225,
   `GATED_DELTA_NET` 46/46, `FLASH_ATTN_QSA` 26/26, width probe PASS.  See
   [`../../beta/mmb-general/BETA-TESTING.md`](../../beta/mmb-general/BETA-TESTING.md) §5.
2. **Target `-b 8192 -ub 8192`** — decision 2026-09-21.  For **long-context (pp65536+)** use
   **`-b/-ub 4096`**: at `-ub 8192` that point memory-thrashes (GPU oscillating, ~844 t/s), while
   `-ub 4096` stays pegged at 100 % (~1093 t/s) — maintainer, 2026-09-22.  The `-ub 16384` failure is
   root-caused and **deferred** (see the “Session-2 record” section): the full-vocab `result_output`
   reserve (15.5 GiB, shared with the other solution) plus qwen4exp's HC `block_out` pin (~18 GiB)
   against the resident PLE table (~27 GiB host).  ubatch 8192 runs clean and is the reproducible
   head-to-head baseline (ours 1212.6 vs its 1346.5 at pp8192, ~10 % behind before items 1+2).
3. **Item 16 (HC BF16 streams `blk16`/`res16`) is DONE 2026-09-22 (session 6, `patches/0011`),
   default OFF** — +4.9 % pp8192 / +4.8 % pp32768 at `-b/-ub 4096`, default build byte-identical —
   [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md).  The MoE-merge `ffn_out`
   `block_out` stays F32: it is not adjacent to the reduction chain, and the reference's merge path is
   equally dormant on qwen4exp (same builder), so it is **not** a gap against the reference.
   **The `mmb_cvt_f32_bf16` item is also DONE (session 6, `patches/0012`)** — the fused combine now
   emits the BF16 `out_xn` copy the graph already marks, bit-identical, **+3.3 %/+3.2 %** —
   [`2026-09-22-mmb-cvt-out-xn.md`](2026-09-22-mmb-cvt-out-xn.md).
   **The prefill indexer relu-sum is DONE too (session 6, `patches/0013`)** — bit-identical, pp32768
   **+1.8 %** — [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md).
   **The `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP` prefill-score trim is DONE (session 7,
   `patches/0014`, default ON)** — the reference's causal scorer bound ported to the fused top-k
   design (cell range clamped to `n_blocks*ratio`), default strip 1024, bit-identical, **+0.5 %
   pp8192 / neutral pp32768** at `-b/-ub 8192` —
   [`2026-09-22-qsa-score-bounds.md`](2026-09-22-qsa-score-bounds.md).
   **The `QSA_SCORE_WMMA` prefill-score fusion is DONE (session 8, `patches/0016`), default ON** — the
   AMD RDNA3_5 4-head/128-dim lightning-indexer WMMA kernel ported, the qwen4exp prefill score fused
   into one `ggml_lightning_indexer` (all-ones weights, zero F16 mask) composed with the causal trim,
   op oracle **225/225 (81 new `nh=4` cases)**, width probe PASS with the per-W hashes byte-identical
   to `=0`, pp32768 **+1.0 % / +0.9 %** —
   [`2026-09-22-qsa-score-wmma.md`](2026-09-22-qsa-score-wmma.md).  **MMB quant coverage is DONE
too (session 8, `patches/0017`)**: Q4_0/Q4_1/Q5_0/MXFP4/NVFP4, oracles green, PPL parity, pp8192
   +19.5/+22.4/+25.4 %, 35B-A3B Q4_1 +63 %, gpt-oss MXFP4 +5.2 % —
   [`2026-09-22-mmb-quant-coverage.md`](2026-09-22-mmb-quant-coverage.md).  The campaign was rebuilt
   on delivery **r13** (fork `gap-closing-r13`, tip `abf3bff76`), dropping the superseded WIP
   `patches/0015`.
   The full session-5 finding (throughput A/B, memory accounting, family diff) is in
   [`closing-the-gap.md`](closing-the-gap.md#session-5-finding-2026-09-22--fresh-target-ubatch-profile-memory-accounting-refined-tasks).

## The current open list (see §13 of the doc)

**Priority sequence (maintainer, 2026-09-21): recall speed + correctness → decode speed + correctness →
MTP tuning + correctness.**

**Phase 1 — recall / long-context prefill + correctness**

1. Wire the existing `hc_gate_mix_kernel` + make `hc_combine_norm` fire (matcher) — the `HC_*`
   ablation is −19.5 % on the other solution's model. **Both halves done 2026-09-21**: the combine+norm
   matcher was revived (+1.5 % prefill) and `hc_gate_mix` is wired and default-on on gfx1151
   (+1.2–1.5 % at pp8192/32768, width-pure, text-identical) — see the record and `patches/0003`.
   Follow-up: the gate-mix kernel is IQ4_NL-only, so the mixed UD-IQ4_XS model is unchanged.
2. Port `gdn-conv.cu` + `ple-conv.cu` (now F32-aware for Flash-Next PLE) — **DONE 2026-09-21
   (session 3)**: default-on, bit-identical, +3.0/+3.2 % qwen4exp IQ4_NL, +6.5/+7.1 % 35B-A3B —
   `2026-09-21-gdn-ple-conv-fusions.md`, `patches/0004`.
3. **`-ub 16384` is deferred** (target is `-ub 8192`).  Root cause in the “Session-2 record”
   section: result_output reserve + HC pin + resident PLE.  **Update 2026-09-22 (session 5):** two terms
   now measured — ~28 GB PLE residency (`-lzm auto` → AUTO → OFF on the gfx1151 IGPU) and ~9 GB HC
   `block_out` pins.  The chosen fix is the **new item 13** (`-lzm auto` = managed PLE loader); the
   old `--lazy-buffer-size` idea is dropped in favour of that env-tunable default.
3.5. Port the other solution's three correctness fixes — **CLOSED 2026-09-22**: `b0f31f587` (size the
   QSA block window by the highest stored position) **ported** —
   [`2026-09-22-qsa-block-window-fix.md`](2026-09-22-qsa-block-window-fix.md), `patches/0005`; the other
   two (`40c0b9c38` maskless-only-where-qsa3-consumes, `14fff4f97` −1 sentinels) audited **N/A** against
   our derived-visibility QSA — [`2026-09-22-qsa-item-3.5-audit.md`](2026-09-22-qsa-item-3.5-audit.md).
4. `norm-gated.cu` (`rms_rows`) — **DONE 2026-09-22 (session 4)**: the narrow-row RMS norm ported,
   default-on, **bit-identical**, ~+0.3 % at the clean `-b/-ub 4096` protocol (it read +0.5–1.1 % at
   `-ub 8192`, which is the memory-pressure confound — see the methodology record) —
   [`2026-09-22-norm-rows-fusion.md`](2026-09-22-norm-rows-fusion.md), `patches/0006`.  **`idx-relu-sum`
   is NOT banked — corrected 2026-09-22 (session 5):** our fused indexer score is `n_tokens == 1` only,
   so prefill still runs a separate `unary_op<relu>` (559 ms) + head-sum adds (see the new item 14).  **Item 6 (`qsa3_attn` body) is DONE 2026-09-22
   (session 4)** — bit-identical, +2.4 %/+1.7 % — [`2026-09-22-qsa3-visibility-fold.md`](2026-09-22-qsa3-visibility-fold.md),
   `patches/0007`; **item 7 (tall-tile min-M) is DONE 2026-09-22 (session 4)** — bit-identical,
   +0.8 %/+1.1 % — [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md), `patches/0008`;
   **item 8 (QSA graph flags) is DONE 2026-09-22 (session 4)** — audit only, 7/9 present/superseded, two
   follow-ups — [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md).
   **Item 1's `hc_combine_norm_f32` `_b256` swap stays CLOSED NEGATIVE** — not bit-identical (it changes
   the greedy text, deterministically), so the rejection does not rest on timing; see
   [`2026-09-21-hc-cn-b256-rejected.md`](2026-09-21-hc-cn-b256-rejected.md).
5. MoE bf16 epilogue + drop `concat_transposed` — **MoE bf16 epilogue DONE 2026-09-22 (session 5,
   `patches/0010`), default OFF** via `GGML_CUDA_MMB_DOWN16=1` (lossy): the IQ4_NL routed-down GEMM
   output is marked bf16-only, the producer stores BF16 in place and `moe_weighted_reduction_bf16_v4`
   reads it — kernel 1479 -> 846 ms at pp32768, `plain == draft-mtp` and width probe PASS.  The
   `concat_transposed` materialisation is already gone at `-ub 8192`.
13. **`-lzm auto` semantics + managed PLE reader perf** — **semantics DONE, reader gated OFF**
    2026-09-22: `on` = mmap-lazy, `off` = preload, `auto` = upstream auto, `--lazy-buffer-size` dropped,
    managed LRU **opt-in via `LLAMA_LAZY_BUF_MB`** and off by default (slowest arm).  **Discriminator:**
    the cost is both an intrinsic streaming overhead (still −4.0 % vs mmap with the table fully cached)
    and page-cache pressure (−9.7 % at the target); the fix is a no-cache parallel-pread fast path like
    the reference's `on-direct`.  It already enables the parked `-b/-ub 16384` (item 3, 1125.5 t/s).
14. Port the prefill indexer **relu+head-sum** fusion (`idx-relu-sum`) — **DONE 2026-09-22 (session 6,
    `patches/0013`), default ON, bit-identical, pp32768 +1.8 %** —
    [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md).  The matcher anchors at the relu and
    accepts our L2a relu-before-the-4-D-reshape form (the head views' `view_src` is the relu while
    their strides come from the reshape).  RDNA3_5-gated like the reference.
15. `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP` — **DONE 2026-09-22 (session 7, `patches/0014`)**, default
    ON, bit-identical: the trim is coupled to the reference's complete-block selection, which our
    fused cell top-k lacks, so it is delivered by clamping the fused top-k's cell range to
    `n_blocks*ratio` instead of trimming the block map.  Default strip 1024, +0.5 % pp8192 / neutral
    pp32768 at `-b/-ub 8192`.  The `-inf`-padded first cut (no kernel change) was a wash —
    [`2026-09-22-qsa-score-bounds.md`](2026-09-22-qsa-score-bounds.md).  **`QSA_SCORE_WMMA` is DONE
    (session 8, `patches/0016`), default ON**: the AMD RDNA3_5 4-head/128-dim lightning-indexer WMMA
    kernel + the fused prefill score (all-ones weights, zero F16 mask) composed with the trim; op
    oracle 225/225 (81 new `nh=4` cases), width probe PASS, pp32768 +1.0 %/+0.9 % —
    [`2026-09-22-qsa-score-wmma.md`](2026-09-22-qsa-score-wmma.md).
16. **BF16 HC streams** (`blk16`/`res16`) — **DONE 2026-09-22 (session 6, `patches/0011`),
    default OFF**: +4.9 % pp8192 / +4.8 % pp32768 at `-b/-ub 4096`, default build byte-identical —
    [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md).  `res16` is the dominant half;
    the MoE-merge `ffn_out` `block_out` stays F32 — not adjacent to the reduction chain, and the
    reference's merge path is equally dormant on qwen4exp (same builder), so not a gap against it.

**Phase 2 — decode speed + correctness**

9. Port sparse QSA decode + incremental indexer state (`d67d58836`) — its +11–20 %; our plain decode is
   already ahead, so this is a hold/repay item.
10. MMB quant coverage — **DONE 2026-09-22 (session 8, `patches/0017` + `0018`), default ON on non-RDNA4**:
    **Q4_0, Q4_1, Q5_0, MXFP4, NVFP4** (WTYPE 11–15) and **IQ2_S, IQ2_XS, IQ2_XXS** (WTYPE 16–18)
    ported to `mmb.cu`.  `MUL_MAT`/`MUL_MAT_ID` dequant-vs-CPU oracles green for all eight, PPL parity,
    pp8192 +19.5/+22.4/+25.4 % (Q4/Q5), 35B-A3B Q4_1 +63 %, gpt-oss MXFP4 +5.2 %, real MiniMax IQ2_S
    +4.9 % pp, dense IQ2_XS +16.5 % pp —
    [`2026-09-22-mmb-quant-coverage.md`](2026-09-22-mmb-quant-coverage.md),
    [`2026-09-22-mmb-iq2-coverage.md`](2026-09-22-mmb-iq2-coverage.md).  NVFP4 and IQ2_XXS are
    oracle-only (no local model).  `Q2_K`/`Q1_0` stay out of scope for quality; the model tree has
    none of the former and one `Q1_0` model (`Bonsai-8B`).

**Phase 3 — MTP tuning + correctness** (parked)

12. Add `nextn_shared_target_tensors` support — **DONE 2026-09-22 (session 7, `patches/0015`), a
    correctness fix**: the shared-NextN MTP sidecar (the other solution's IQ4_NL model ships
    `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`) failed every draft round on an M-RoPE `X < Y` check
    because the MTP driver inferred KV sharing from `ctx_other` alone; the guard is now gated on the
    `gemma4-assistant` arch (tensor sharing and memory sharing are different).  0 draft errors,
    acceptance 0.287, adaptive depth transitions restored —
    [`2026-09-22-mtp-shared-nextn-fix.md`](2026-09-22-mtp-shared-nextn-fix.md).  **The bug is upstream
    (#23398) and is now delivered in the delivery set as block 00 (release `v16-ebbb18522-r13`)** — the
    WIP `patches/0015` is superseded, so a campaign rebuilt on r13 must not apply it.
11. qwen4exp adaptive ceiling sweep (3/5/7/9/12) — fixed-depth MTP is at parity with its, adaptive wins
    recall but over-drafts code/prose at `n_max 12`.

## Cautions

* The body's §6/§9 kernel-level deltas are against `f5daaa3cf`; re-profile `b0f31f587` before trusting
  them (its tree gained the `mmb_quant` dispatcher and dropped env gating).
* the other solution's `ac1ebb4e0` **compiled in** its tuned defaults and deleted the `LLAMA_*` experiment
  switches — the body's Appendix D ablation commands no longer work against the other solution's current HEAD.
* the other solution has **no adaptive MTP controller** (fixed `n_max` + upstream `p_min`/`n_min` early stop);
  our `draft-mtp-adaptive` is a different axis from its per-step decode kernels.  The 2026-09-21
  measurement (see the qualification record) shows our plain decode ahead, fixed-depth MTP speedup at
  parity, and the only real MTP gap is `nextn_shared_target_tensors` — not velocity.
