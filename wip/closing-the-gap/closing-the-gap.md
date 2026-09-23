# Closing the gap — `beta/mmb-general` vs the other solution's `strix-halo` prefill

**Date:** 2026-09-20 (snapshot) · **updated:** 2026-09-22 (session 9 — NEXT-SESSION item 2 fixed)
**Box:** `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5), ROCm 7.14 (`/opt/rocm-7.14-gfx1151`), 123 GiB RAM / 124 GB unified VRAM
**Scope:** a 1:1 prefill comparison on **the other solution's uniform-IQ4_NL model** (not just our mixed UD-IQ4_XS), a kernel-level profile diff on the uniform model, and a gate-ablation of the other solution's stack on this box to price the still-missing families. This is an investigation record, not a delivery change.

> **Fresh session: read this whole file; the "START HERE" block is the handoff.**  The body below
> (§0–13 + the dated records) is the dated investigation record, kept for its measurements.

---

## START HERE — fresh-session handover (end of session 10, 2026-09-22)

> **Session 10 (latest):** the **sparse MTP draft** is implemented (`patches/0020`, fork tip
> `d8334f929`, tree `fa185bbb…`), **OPT-IN** (`LLAMA_MTP_SPARSE=1`).  **Prefill pp150K +6.9 %** with a
> 32K depth gate.  **Decode root-caused:** the incremental QSA indexer
> (`GGML_CUDA_QSA_INDEXER_CACHE`) was **OFF by default** even though the graph expects it on;
> `patches/0021` flips it ON — **byte-identical** and **+9.1 % @80K / +14.6 % @150K** plain decode
> (f16, +8.7 % bf16), after which the sparse draft decode is **parity / a slight win** (the
> selected-cell `flash_attn_qsa` is 0.05 ms/call vs the dense `flash_attn_tile` 1.03 ms — it was the
> per-step indexer, not the attention kernel, that ate the saving).  With the cache ON the gfx1151
> **decode crossover moves 64K → 32K** (`patches/0022`; 16K dense +3.4 %, 32K parity, 48K sparse
> +1.8 %, 64K sparse +4.6 % — a 32K-64K decode re-baseline).  At 5K all arms byte-identical
> (`3553e76d3a9e`), width probe PASS (P=32768 sparse decode included), MTP acceptance unchanged
> (0.85035), oracles green.  **Draft kept OFF** because at 40K the sparse prefill changes the greedy
> text while the dense draft matches plain
> — a pre-existing iterative target verify/rollback divergence at depth (the target is logit-width-pure
> there, and even dense MTP diverges from plain at 150K), not a memory-change bug.  Full record:
> [`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md).  **Use `--ctx-checkpoints 0`
> for any depth purity test** (the checkpoint save/restore makes depth MTP runs nondeterministic).

### Where we are

* **Target is `-b 8192 -ub 8192`.**  For **long-context** (pp65536+) use **`-b/-ub 4096`**: at
  `-ub 8192` that point is right at the memory limit and the GPU oscillates (memory shortfall) at
  ~844 t/s, while `-ub 4096` stays pegged at 100 % and runs **1093 t/s** (maintainer, 2026-09-22).
  `-ub 16384` is root-caused and parked (below); do not spend time on it unless the PLE-lazy fix is
  picked up.
* **Throughput (session-6 snapshot; session 8's `QSA_SCORE_WMMA` adds ~+1 % at pp32768 on this model)**
  (qwen4exp IQ4_NL, gfx1151, `-ctk/-ctv f16`, `pp… -n 0`):

  | config | pp8192 | pp32768 |
  |---|---:|---:|
  | `-b 8192 -ub 8192` (target), `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1` | **1379.2** | **1320.0** |
  | `-b/-ub 4096` (A/B protocol), BF16 rounding path ON | 1324.8 | 1287.2 |
  | `-b/-ub 4096`, BF16 path OFF (the default build) | 1308.2 | 1269.6 |

  The other solution's session-5 cells were 1323.7 / 1374.4 at `-b 8192 -ub 8192` and 1200.1 / 1224.5 at
  `-b/-ub 4096`, i.e. we are now **ahead at pp8192 on both protocols** and ~4 % behind at pp32768 (was
  −10.6 %).  The old “BF16 intermediate-traffic families plus two non-lossy kernel items” gap is
  **closed** (items 16 / `mmb_cvt` / 14, session 6).  The BF16 rounding path is still the maintainer's
  default-OFF exception (lossy), exactly like `MMB_DOWN16`.
* **Phase-1 item 1 (HC fusions) DONE** (session 2, **`patches/0003`**): `hc_combine_norm` matcher
  revived (+1.5 %) and `hc_gate_mix` wired default-on (+1.2–1.5 %) — width-pure, same-seed-text
  identical.
* **Phase-1 item 2 (depthwise conv1d) DONE** (session 3, **`patches/0004`**): `gdn-conv.{cu,cuh}` +
  `ple-conv.{cu,cuh}` ported, default-on (`GGML_CUDA_DISABLE_CONV_FUSION=1` disables), **bit-identical**
  (fused == unfused row-0 hash + width probe PASS), **+3.0/+3.2 %** qwen4exp IQ4_NL and **+6.5/+7.1 %**
  35B-A3B (pp8192/32768, `-ub 8192`) —
  [`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md).
* **Phase-1 item 3.5 first fix DONE** (session 3, **`patches/0005`**): the QSA block window is sized by
  the highest stored position (`b0f31f587`), fixing the M-RoPE-image + MTP assert —
  [`2026-09-22-qsa-block-window-fix.md`](2026-09-22-qsa-block-window-fix.md).
* **CLOSED NEGATIVE — do not redo:** item 1's `hc_combine_norm_f32_b256` swap (not bit-identical — it
  changes the greedy text `1b59d651f2c3` → `fc7c8a10ea45` — and 0.7–0.8 % slower; the reference's
  554 ms is its **BF16** HC traffic (`hc16`/`blk16`/`res16` compiled in), not the thread count) and
  item 5's `concat_transposed` drop (already gone at `-ub 8192`; the remaining BF16 MoE epilogue is a
  lossy/memory candidate, not a 3–4 % win) —
  [`2026-09-21-hc-cn-b256-rejected.md`](2026-09-21-hc-cn-b256-rejected.md).
* **Phase-1 item 3.5 CLOSED** (session 4): the first fix is ported (`patches/0005`); the other two
  (`40c0b9c38` maskless-only-where-qsa3-consumes, `14fff4f97` −1 sentinels) audited **N/A** — our tree
  has no maskless path and its top-k output carries no sentinels; the invariants each protects are
  already held (see the audit section below) —
  [`2026-09-22-qsa-item-3.5-audit.md`](2026-09-22-qsa-item-3.5-audit.md).
* **Phase-1 item 4 (narrow-row RMS norm) DONE** (session 4, `patches/0006`): `norm-gated.cu::rms_rows_f32`
  ported, default-on (`GGML_CUDA_DISABLE_NORM_ROWS=1` disables), **bit-identical** (width probe PASS,
  same-seed text `f61199ba5644`), ~+0.3 % at the clean `-b/-ub 4096` protocol —
  [`2026-09-22-norm-rows-fusion.md`](2026-09-22-norm-rows-fusion.md).
* **METHODOLOGY (2026-09-22): use `-b/-ub 4096` for perf A/Bs.**  At `-ub 8192` the box bottoms at
  2–3 GB free with ~40 % more kswapd reclaim, and a fusion whose arms differ in graph-shape memory runs
  its arms in different pressure regimes — item 4 read +1.05 % at `-ub 8192` but +0.27 % at `-ub 4096`.
  Correctness gates are unaffected; only the timing.  The prior rejections were audited and the shipped
  wins stand — [`2026-09-22-ubatch-8192-memory-confound.md`](2026-09-22-ubatch-8192-memory-confound.md).
* **Phase-1 item 6 (QSA visibility fold) DONE** (session 4, `patches/0007`): the per-cell `cell_vis` check
  is folded into `umask` at merge time and dropped from the `qsa3_attn` hot loop, **bit-identical**
  (width probe PASS, text `f61199ba5644`), `qsa3_attn_kernel` 809.6 -> 672.9 ms, **+2.4 % pp8192 /
  +1.7 % pp32768** at `-ub 4096`; the QSA pipeline is now 50 ms ahead of the reference's —
  [`2026-09-22-qsa3-visibility-fold.md`](2026-09-22-qsa3-visibility-fold.md).
* **Phase-1 item 7 (tall-tile min-M) DONE** (session 4, `patches/0008`): the M=4 HC inject no longer
  takes the 384-row tall tile (it ran 2× the dispatches of the reference); **bit-identical** (text
  `f61199ba5644`), tall `384,64` 1030/380 -> 540/190 ms, **+0.8 % pp8192 / +1.1 % pp32768** at `-ub 4096`
  — [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md).
* **Phase-1 item 8 (QSA graph flags) DONE** (session 4, audit): 7/9 are present or superseded in our
  block-14/15 QSA; **2 are un-ported prefill-score optimizations** — `QSA_SCORE_BOUNDS`+
  `QSA_QUERY_STRIP` (score the prefill in 512-token strips, each trimmed to its visible blocks; ~half
  the indexer score work) and `QSA_SCORE_WMMA` (a fused WMMA prefill score; our fused op is gated to
  `n_tokens == 1`) — [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md).
* **Session-5 item: lazy-mode semantics DONE + managed reader gated OFF** (2026-09-22, `patches/0009`):
  `-lzm on` = mmap, `-lzm off` = resident, `-lzm auto` = upstream auto (managed LRU opt-in via
  `LLAMA_LAZY_BUF_MB`), `--lazy-buffer-size` dropped.  The managed reader measured **slowest**
  (1090/1184 vs mmap 1219/1217 vs resident 1285/1232 t/s) so it is **off by default**; it does unlock
  `-b/-ub 16384` (1125.5 t/s).  All modes are text-identical (`7e4a6a4e66fb`).  See the session-5
  finding below.
* **Session-5 item done: MoE BF16 epilogue** (2026-09-22, **`patches/0010`**): the reference's
  `moe_weighted_reduction_bf16_v4` + the `down16` graph marking are ported, **default OFF** via
  `GGML_CUDA_MMB_DOWN16=1` (the maintainer's explicit exception to the default-on policy for this
  lossy path).  Kernel **1479 -> 846 ms** at pp32768 (matches the reference's 846.5), `plain == n3`
  text and width probe PASS with it on at `-b 2048 -ub 2048`.  This is the **template** for the HC
  port below.
* **Phase-1 item 16 (HC BF16 streams) DONE** (session 6, **`patches/0011`**): the `blk16`/`res16`
  BF16 hyper-connection streams are ported **default OFF** (`LLAMA_HC_BLK16=1` / `LLAMA_HC_RES16=1`),
  the default build is byte-identical, and the arm is **+4.9 % pp8192 / +4.8 % pp32768** at
  `-b/-ub 4096` — [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md).  `res16` is the
  dominant half (+4.7 %/+4.6 % alone); `blk16` adds +3.2 %/+2.5 % alone.  Only the attention-path
  `block_out` takes `blk16`: the FFN `ADD(moe_reduce, shexp)` is not adjacent to the reduction chain
  (the shared-expert branch is expanded between them) and the merge fusion requires adjacency — the
  reference's `build_layer_ffn`/`build_moe_ffn` are the same, so **its merge path is equally dormant
  on qwen4exp**.  Left on the table deliberately (the full note is below the NEXT SESSION block).
* **`mmb_cvt_f32_bf16` gap CLOSED** (session 6, **`patches/0012`**, default ON, **bit-identical**): the
  fused `hc_combine_norm` now emits the BF16 `out_xn` copy the graph already asked for, so every
  consumer stops reconverting the F32 (81 % of the `mmb_cvt` traffic); `mmb_cvt` 520 -> 148 calls /
  1.93e10 -> 3.67e9 elements and **+3.3 % pp8192 / +3.2 % pp32768** at `-b/-ub 4096` —
  [`2026-09-22-mmb-cvt-out-xn.md`](2026-09-22-mmb-cvt-out-xn.md).  The width probe row0, same-seed
  text and the **unfused** combine (`GGML_CUDA_DISABLE_HC_COMB=1`) all agree, so it is a
  consistency fix, not a re-baseline.
* **Prefill indexer relu-sum (item 14) DONE** (session 6, **`patches/0013`**, default ON,
  **bit-identical**): a fused kernel recomputes the relu from the pre-relu scores and sums the heads
  in graph order (the matcher accepts our L2a relu-before-reshape form), pp32768 **+1.8 %** at
  `-b/-ub 4096` — [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md).  RDNA3_5-gated like the
  reference; the kernel is arch-neutral.
* **Session 8 status: BOTH focus items DONE + the owed gate suite closed.**
  `QSA_SCORE_WMMA` (prefill, `patches/0016`) and MMB quant coverage
  (Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 = `patches/0017`; IQ2_S/IQ2_XS/IQ2_XXS = `patches/0018`) are ported,
  default ON, validated; the r13 rebuild and the `beta/mmb-general` BETA-TESTING gate suite (Gate 4
  MTP + oracles) are GREEN.  **Session 9 fixed the Phase-2 correctness bug (`patches/0019`): the MMB
  HC16 F32-elision is invalid under an eval callback (llama-imatrix/`common/debug`), because the
  scheduler splits the graph at callback nodes and `mmb_begin_graph()` clears the BF16 cache between
  the producer and the GEMM → imatrix read the never-written F32, `non-finite values detected in
  blk.21.attn_output.weight`.**  The remaining work is Phase-2 decode (sparse QSA + incremental
  indexer) plus the gfx1100/gfx1201 validation — see the NEW "NEXT SESSION" block immediately below.
  The `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP` trim landed session 7.  Read the session-5
  profile/memory findings and the corrected gate semantics further down too.

### NEXT SESSION (end of session 10) — focus: the depth verify/rollback divergence

Phase-1 (recall/prefill) is closed; session 9 closed the BF16-MMB eval-callback bug and session 10
implemented the sparse MTP draft (`patches/0020`, **opt-in**).  The campaign is `gap-closing-r13`
(r13 + 12 `beta/mmb-general` + gap-closing `0001..0014`/`0016`/`0017`/`0018`/`0019`/`0020`).

**1. The sparse MTP draft is IMPLEMENTED, `patches/0020`, OPT-IN — the remaining item is the depth
verify/rollback divergence it exposed.**  [`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md).
The audit ([`2026-09-22-phase2-sparse-qsa-audit.md`](2026-09-22-phase2-sparse-qsa-audit.md)) found the
selected-cell decode and the incremental indexer already in our tree; only the MTP-draft attention was
missing, and the plan ([`PLAN-mtp-sparse-draft.md`](PLAN-mtp-sparse-draft.md)) was implemented as
the three edits + two memory fixes the plan missed.  Results:

* **Prefill pp150K 927.0 → 990.8 t/s (+6.9 %)** with the 32K depth gate (`LLAMA_MTP_SPARSE_MIN_KV`);
  enabling it from `n_kv > 2051` is a 25 % *loss* (697.3) — the indexer cost is fixed per query while
  the dense attention it replaces is `O(n_kv)`.
* **Decode root cause found (`patches/0021`):** the "decode is slower" reading was the incremental
  QSA indexer being **OFF by default** (`llama-memory-hybrid-idx.cpp` `derived_enabled`, while the
  graph's `idx_cache` defaults ON — the AGENTS/audit say default ON).  With `GGML_CUDA_QSA_INDEXER_CACHE=1`
  plain decode is byte-identical and **+9.1 % @80K / +14.6 % @150K** (f16; +8.7 % bf16); the sparse
  MTP draft decode then reaches **parity / a slight win** at 80K (dense 30.7-30.9 vs sparse
  30.7-31.5 t/s, acceptance 0.792 vs 0.812).  Kernel profile: `flash_attn_qsa` 0.05 ms/call vs
  `flash_attn_tile` 1.03 ms/call; the per-step indexer score/top-k is the overhead the pool removes.
* Gates: 5K purity **PASS** (`3553e76d3a9e` on plain/off/sparse), width probe **PASS**, MTP acceptance
  **0.85035** unchanged, `FLASH_ATTN_QSA` 26/26 / `GATED_DELTA_NET` 46/46.
* **Blocker / default OFF:** at 40K the sparse prefill changes the greedy text (`c0a3bda5dff4`) while
  the dense draft and the hybrid-memory-dense arm both match plain (`687cec808661`).  The hybrid
  memory is innocent; the target is **logit-width-pure** at 40K (the width probe extended to P=32768,
  `RS=0` and `RS=from_w`, both PASS), so the divergence is in the **iterative target
  verify/rollback** path — and even **dense** MTP diverges from plain at 150K.  **Root-cause the
  depth verify/rollback divergence, then the sparse draft can be defaulted on.**  Related confound:
  without `--ctx-checkpoints 0` the depth MTP runs are nondeterministic; use that flag for depth
  purity tests.

**2. Fix the pre-existing BF16-MMB non-finite bug (found session 8) — DONE 2026-09-22 (session 9,
`patches/0019`).**  `llama-imatrix` on `Nanbeige4.2-3B-BF16` emitted *"non-finite values detected in
blk.21.attn_output.weight"* with the default MMB build.  The misattribution in the find (`bf16w`)
is because `GGML_CUDA_MMB_BF16W=0` happens to disable the **HC16** marking as a side effect; the real
switch is `GGML_CUDA_MMB_HC16=0`.  The scheduler splits the graph at every eval-callback node
(imatrix asks for each `MUL_MAT`), so `mmb_begin_graph()` clears the per-graph BF16 activation cache
between the producer's sub-graph and the GEMM's — which then re-converted the F32 activation that the
`bf16_only` mark had told the producer to skip.  The fix plumbs `has_eval_callback` through
`ggml_backend_graph_optimize_params` and stands the F32 elision down in that mode (inert for serving).
`imatrix` now reports PPL 18.4558 and its `in_sum2` tensors are **byte-identical** to `HC16=0`; width
probe PASS on NanBeige BF16 + qwen4exp; serving perplexity unchanged (19.5126) —
[`2026-09-22-mmb-eval-callback-f32.md`](2026-09-22-mmb-eval-callback-f32.md).

**3. Housekeeping: gfx1100 / gfx1201 validation of the session-8 additions.**  The new MMB quant types
(Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 + the IQ2 family) and `QSA_SCORE_WMMA` are **gfx1151-validated only**; the
beta requires the gfx1201 port/validation (`beta/mmb-general/gfx1201-s14-gates.md`, 3-GPU `-sm tensor`,
q8_0 KV, `-b/-ub 2048`).  The dequant code is arch-neutral and gfx1201 keeps its per-type dense policy
(inert), so this is an apply-and-gate job, not a port.

**Parked (do not start):** `-ub 16384` needs the managed PLE reader's no-cache parallel-pread fast path
(item 13); the qwen4exp adaptive-MTP ceiling sweep (3/5/7/9/12) is a tuning item.  Both stay parked
until decode lands.

---

### Session-7/8 record (2026-09-22): QSA scorer trim, `QSA_SCORE_WMMA`, MMB quant coverage

**Session 7 landed the QSA prefill scorer trim and the shared-NextN MTP fix** (fork `gap-closing`
`00d8bbbc9`, exported to [`patches/0014`](patches/) and [`patches/0015`](patches/)): the reference's `QSA_SCORE_BOUNDS` +
`QSA_QUERY_STRIP` ported to the fused-top-k design.  `qsa_score_key_limits` in
`llama-memory-hybrid-idx.*` bounds each query strip to the complete blocks fully inside its causal
prefix, and `build_qsa_top_k` scores each strip at that width; the fused `ggml_indexer_top_k`'s cell
range is clamped to `n_blocks*ratio` in the CUDA and CPU paths so the trimmed cells are never read.
Default strip **1024** (`LLAMA_QSA_SCORE_STRIP`), `LLAMA_QSA_SCORE_BOUNDS=0` isolates the trim.
**+0.5 % pp8192 / neutral pp32768** at `-b/-ub 8192`, bit-identical width probe on f16/bf16/q8_0 and
same-seed text `61cebc1d31a9` — [`2026-09-22-qsa-score-bounds.md`](2026-09-22-qsa-score-bounds.md).
The `-inf`-padded first cut (no kernel change) was a measured **wash**; the clamp + trim is what pays.
The same session fixed the **shared-NextN MTP head** (`nextn_shared_target_tensors`): the MTP driver
inferred KV sharing from `ctx_other` alone, so the other solution's IQ4_NL shared sidecar died every
round on the M-RoPE `X < Y` check.  Gated on the `gemma4-assistant` arch it now runs with 0 draft
errors and acceptance 0.287 — [`2026-09-22-mtp-shared-nextn-fix.md`](2026-09-22-mtp-shared-nextn-fix.md).
This is an **upstream bug (#23398)** now **delivered in the delivery set as block 00 (release
`v16-ebbb18522-r13`)** — an `upstream/` PR candidate for when upstream fixes it, and the WIP
`patches/0015` is superseded (do not apply it on a campaign rebuilt on r13).

**Session 8 (2026-09-22) did the rebuild + BOTH focus items.**  The campaign is rebuilt on **r13**
(fork branch `gap-closing-r13`, tip `abf3bff76` = r13 + 12 `beta/mmb-general` patches + gap-closing
`0001..0014` + `0016`/`0017`, dropping the superseded `0015`), built clean.

* **`QSA_SCORE_WMMA` DONE, default ON** (`patches/0016`): the reference's AMD RDNA3_5 4-head/128-dim
  `qsa_indexer_wmma16_keyreg` WMMA kernel is ported into `lightning-indexer.cu` (with a generic
  `n_head == 4` vec fallback so the op is legal on every backend), `build_qsa_top_k` builds the
  prefill score as one `ggml_lightning_indexer` (all-ones weights + zero F16 mask, shared per graph by
  name) composed with the `QSA_SCORE_BOUNDS` causal trim (leading-rows views).  Op oracle `-o
  LIGHTNING_INDEXER` **225/225 including 81 new `nh=4` cases**; gated on qwen4exp IQ4_NL
  (`width_purity=PASS` with per-W hashes **byte-identical** to `=0`, coherent same-seed text) with
  pp32768 **1311.5 → 1324.9 t/s (+1.0 %, `-b/-ub 8192`)** / **1265.9 → 1277.2 (+0.9 %, `-b/-ub 4096`)**,
  pp8192 flat — [`2026-09-22-qsa-score-wmma.md`](2026-09-22-qsa-score-wmma.md).
* **MMB quant coverage DONE, default ON on non-RDNA4** (`patches/0017`): five more weight types —
  **Q4_0 / Q4_1 / Q5_0 / MXFP4 / NVFP4** (WTYPE 11–15) — with dequant-vs-CPU oracles
  (`MUL_MAT` 48/47/14/46/45, `MUL_MAT_ID` 74/75/3/74/73 with MMB forced on), PPL parity, and
  **+19.5/+22.4/+25.4 %** pp8192 (3B), **35B-A3B Q4_1 pp4096 1500 → 2449 (+63 %)** (routed+GLU),
  **gpt-oss-20b MXFP4 pp4096 +5.2 %** — [`2026-09-22-mmb-quant-coverage.md`](2026-09-22-mmb-quant-coverage.md).
  NVFP4 has no local model (oracles gate it).
* **IQ2 family added too** (`patches/0018`, after a header-only scan of all 132 `*.gguf` in
  `/llm/models` — `tools/gguf-types.py`): only **IQ2_S** was present (MiniMax-M2.7-IQ3_S 124 tensors,
  DeepSeek-V4-Flash-UD-IQ3_XXS 84); the whole family **IQ2_S / IQ2_XS / IQ2_XXS** (WTYPE 16–18) is
  ported and default ON on non-RDNA4.  Oracles 14/14/46 (`MUL_MAT`) and 4/15/75 (`MUL_MAT_ID`); real
  MiniMax iq2_s **pp4096 +4.9 %** (PPL +1.07 %), dense IQ2_XS **pp8192 +16.5 %** (PPL +0.22 %) —
  [`2026-09-22-mmb-iq2-coverage.md`](2026-09-22-mmb-iq2-coverage.md).

**Remaining from the handover:** the Phase-2 sparse QSA decode (`d67d58836`), the gfx1100/gfx1201
validation of the session-8 additions, and the parked items (Phase 3 adaptive ceiling sweep,
`-ub 16384` PLE reader) are unchanged.  The Phase-2 correctness bug (BF16-MMB eval-callback F32
elision) is **DONE** — [`2026-09-22-mmb-eval-callback-f32.md`](2026-09-22-mmb-eval-callback-f32.md),
`patches/0019`.  **The owed `beta/mmb-general`
BETA-TESTING gate suite is now GREEN on gfx1151** (session 8, on this r13+beta+gap-closing campaign):
Gate 4 MTP on qwen4exp prose `-n 3000` — **draft acceptance 0.85541** (pos 0.938/0.853/0.776),
**56.5 t/s vs plain 31.7 t/s** (>= plain); op oracles **LIGHTNING_INDEXER 225/225**,
**GATED_DELTA_NET 46/46**, **FLASH_ATTN_QSA 26/26**, the new-type `MUL_MAT`/`MUL_MAT_ID` oracles (see
[`2026-09-22-mmb-quant-coverage.md`](2026-09-22-mmb-quant-coverage.md)); width probe PASS on qwen4exp.
(`INDEXER_TOPK` has 0 cases in this tree; the MMB-on-vs-r12 byte-identity check is retracted — see the
2026-09-22 correction record.)

**Session 6 landed three items** (fork `gap-closing`, exported to [`patches/`](patches/)):

| item | patch | default | result | record |
|---|---|---|---|---|
| HC BF16 streams (`blk16`/`res16`) | `0011` | **OFF** (lossy) | +4.9 %/+4.8 % at `-b/-ub 4096` | [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md) |
| `mmb_cvt` / fused `out_xn` BF16 copy | `0012` | **ON** (bit-identical) | +3.3 %/+3.2 % | [`2026-09-22-mmb-cvt-out-xn.md`](2026-09-22-mmb-cvt-out-xn.md) |
| prefill indexer relu-sum | `0013` | **ON** (bit-identical) | +1.8 % pp32768 | [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md) |

The measured product is the table at the top of this handover (1379 / 1320 t/s at the `-b 8192 -ub 8192`
target with the BF16 rounding path on; 1308 / 1270 on the default build).  The session-5 BF16-traffic and
conversion gaps are closed.

#### Historical: the session-8 prerequisite — rebuild the campaign on delivery r13 (DONE)

The shared-NextN MTP fix now lives in **delivery block 00, release `v16-ebbb18522-r13`** (`main`
`ae076ab`; canonical tip `8491bf2bff8eb3a56e5120c3c9c17533a94ea6bf`, tree
`bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12`).  Rebuild this campaign against r13 and **drop
`wip/closing-the-gap/patches/0015`** (block 00 already carries that fix; applying 0015 on r13 would
conflict).  `patches/0001..0014` should apply unchanged — none touch `common/speculative.cpp`.  Then
run the full `beta/mmb-general` BETA-TESTING gate suite once — still owed since session 5: Gate 4 (MTP
acceptance) + the op oracles.  Purity is an **intra-build** contract (`GREEDY-PURITY.md`):
`test-logits-width-probe` PASS (worst maxdiff 0) on f16/bf16/q8_0, `plain == draft-mtp` greedy text,
acceptance > ~0.45 at pos 1, coherence.  The MTP gate can now use the shared-NextN sidecar (fixed).
This was housekeeping; the two focus items below were the session's real work — both are now DONE.

#### Historical: Focus 1 — `QSA_SCORE_WMMA` (prefill indexer score → fused WMMA) — DONE in session 8

Our prefill indexer score is the per-op chain (`mul_mat` → L2a relu → head-sum); the fused decode
score op (`ggml_indexer_score`) is `n_tokens == 1` only.  **`ggml_lightning_indexer`** — the DSA WMMA
op we already ship for deepseek32/deepseek4/glm-dsa/hy-v4
(`ggml/src/ggml-cuda/lightning-indexer.cu`) — computes exactly `sum_h w[h] * relu(q_h · k) + mask`, so
with **all-ones weights** and a **zero F16 mask** it is a drop-in for the prefill chain, for the band
`n_tps >= 128 && idx_dim == 128 && n_idx_h == 4` (qwen4exp is exactly 128/4).

Reference recipe (`~/pwilkin-llama-cpp` `ddaf5214b`, `qwen4exp.cpp` ~line 1150; the env flag was
`LLAMA_QSA_SCORE_WMMA` and later compiled in):
* per graph, build shared tensors (named, so layers sharing a ratio reuse them):
  `weights = ggml_fill(F32, [n_idx_h, strip, 1, n_stream], 1.0f)` and
  `zero_mask = ggml_fill(F16, [n_blocks, strip, 1, n_stream], 0.0f)`;
* per strip: `query = view(q, [idx_dim, n_idx_h, n_query, n_stream])`,
  `key = reshape(pooled, [idx_dim, 1, n_blocks, n_stream])`,
  `weights_v = view(weights, [n_idx_h, n_query, 1, n_stream])`,
  `mask_v = view(zero_mask, [n_blocks, n_query, 1, n_stream])`,
  `score = reshape(ggml_lightning_indexer(query, key, weights_v, mask_v), [n_blocks, n_query, n_stream])`.

**Compose with the score-bounds trim (our improvement over the reference).**  The reference's fused arm
*opts out* of `QSA_SCORE_BOUNDS` ("the fused WMMA scorer sizes its shared zero mask from `n_blocks`,
so it opts out"), which would regress the `+0.5 %` trim we just landed in the `n_tps >= 128` band.  The
mask is all zeros — the kernel reads its `ne0` rows contiguously and uses the tensor's `nb[1]` for the
query stride — so a leading-rows view `[score_blocks, n_query]` of the shared `[n_blocks, strip]` mask
is valid and satisfies the op's `mask->ne[0] == k->ne[2]` assert.  Trim `key`/`mask` together per strip.

**Gate (numerics change).**  WMMA half q/k vs the F32 matmul is not bit-identical, so this is an
approved *prefill* re-baseline: width probe PASS on f16/bf16/q8_0, `plain == draft-mtp` greedy text
(the MTP gate can now use the shared head), plus an A/B at `-b 8192 -ub 8192`.  Env
`LLAMA_QSA_SCORE_WMMA`, default ON once green.  The audit prices it as the other half of the indexer
score work ("low-single-digit % prefill"); full flag table in
[`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md).

#### Historical: Focus 2 — MMB quant coverage — DONE in session 8

`ggml_cuda_mmb_supported_mm/_mmid/_glu` (`mmb.cu`, `beta/mmb-general/patches/0001`) currently accept
**IQ4_NL, Q8_0, Q4_K, Q5_1, IQ3_S, Q5_K, Q6_K, IQ4_XS, Q3_K, IQ3_XXS** (IQ3_XXS routed-only and
default-off).  The completeness gap is the five types above.

| type | shape | closest existing template | notes |
|---|---|---|---|
| Q4_0 | 32 vals, half scale | `Q8_0` (2 blocks/64) | simplest; no min |
| Q4_1 | 32 vals, half2 d/m | `Q5_1` | same dm pair layout |
| Q5_0 | 32 vals, half d + qh | `Q5_1` minus the min | 5th bit in `qh` |
| MXFP4 | 32 vals, E2M1 + E8M0 shared exp | new | 4-bit float; ggml has `dequantize.cuh` helpers |
| NVFP4 | E2M1 + E4M3 per-4-block scale | new | ggml has `dequantize.cuh` helpers |

Per-type checklist: (1) a bf16-to-LDS dequantizer in `mmb.cu`/`mmb.cuh` (the Q8_0/Q5_1 pairs are the
templates for Q4_0/Q4_1/Q5_0; MXFP4/NVFP4 need new exponent handling), (2) the
`mmb_supported_mm/_mmid/_glu` predicate, (3) the graph-optimizer `mmb_dense_will_take` /
`_routed_will_take` gate so the fusions stand down only when MMB will take the type, (4) **PPL parity**
with the MMQ path (the beta's rule: "PPL parity everywhere says the dequants are correct") plus a
throughput A/B, (5) the default-on policy (a win → ON; the env var only disables).  **`Q2_K`, `IQ1_*`,
`IQ2_*` stay deliberately out of scope** (quality — documented in the beta README).

Models: Q4_0/Q4_1/Q5_0 are common local quants; MXFP4 = gpt-oss; NVFP4 = recent NVIDIA-quantized
models.  Record PPL + pp2048/pp8192 (`-ub 2048` bf16 KV) per type in the beta README table.

#### After session 8 — deferred housekeeping + tuning

* Full `beta/mmb-general` BETA-TESTING suite (Gate 4 MTP acceptance + op oracles) if not already done
  with the r13 rebuild.
* **Phase 2:** sparse QSA decode + incremental indexer state (`d67d58836`, +11–20 %; a hold/repay item
  since our plain decode is already ahead).
* **Phase 3:** qwen4exp adaptive ceiling sweep (3/5/7/9/12) — tuning; adaptive wins recall but
  over-drafts code/prose at `n_max 12`.
* **Parked:** `-ub 16384` (item 3) — needs item 13's managed PLE reader fast path (a no-cache
  parallel-pread reader like the reference's `on-direct`; the LRU arena measured intrinsically slower).

#### What is deliberately NOT being done

* **The FFN `block_out` in `blk16`.**  `ffn_out = ADD(ffn_moe_out, ffn_shexp_gated)` is not adjacent to
  the MoE reduction chain: `build_moe_ffn` expands the reduction into the graph and the shared-expert
  branch is built after it, so the first node after the reduction is `ffn_gate`.  The reference's merge
  detection requires `nodes[i + node_count] == add`, and its `build_layer_ffn`/`build_moe_ffn` are the
  same, so **its merge path does not fire on qwen4exp either** — the `moe_weighted_reduction_*_out`
  variants are dormant.  Capturing it would need a builder reorder (graph-topology, allocator,
  meta-split and CUDA-graph re-validation) or a separate dataflow bf16-merge op, for ~1/5 of the HC
  stream traffic (`ffn_out` is `[n_embd,T]`; the residual the `res16` arm already covers is
  `[n_embd,hc,T]` = 4×).  Not worth it — do not re-litigate.
* **The parked items stay parked:** `-ub 16384` (needs the PLE-lazy reader), the managed PLE reader
  perf (item 13), and the qwen4exp adaptive-MTP ceiling sweep (Phase 3 tuning).  **Not parked anymore:**
  the sparse QSA decode + incremental indexer (Phase 2, `d67d58836`) is the *next* work — see the new
  NEXT SESSION block at the top — and `nextn_shared_target_tensors` is **DONE** (delivery r13 block 00,
  session 7).

#### Reproduce (copy-paste)

```sh
cd ~/llama.cpp && ~/bin/build-llama-rocm-714                       # full build (ccache; ~4 min warm)
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
for f in ${MU%/*}/*-0000*.gguf; do cat "$f" >/dev/null; done   # warm page cache first
# default build at the A/B protocol:
~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 -b 4096 -ub 4096 -p 8192,32768 -n 0 -r 3
# the BF16 rounding path on, at the target ubatch:
LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1 ~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 \
  -ctk f16 -ctv f16 -b 8192 -ub 8192 -p 8192,32768 -n 0 -r 2
# purity gate (expect width_purity=PASS, worst maxdiff 0):
~/llama.cpp/build-rocm/bin/test-logits-width-probe "$MU" prompts/prose-rdna-boosts.txt 1024 512
```

### Session-6 record (2026-09-22): items 16, `mmb_cvt`, 14 — the BF16/conversion gap closed

Three items, all gated and documented:

| item | patch | default | delta (`-b/-ub 4096`) | record |
|---|---|---|---|---|
| HC BF16 streams (`blk16`/`res16`) | `0011` | OFF (lossy) | +4.9 % pp8192 / +4.8 % pp32768 | [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md) |
| `mmb_cvt` / fused `out_xn` BF16 copy | `0012` | ON (bit-identical) | +3.3 % / +3.2 % | [`2026-09-22-mmb-cvt-out-xn.md`](2026-09-22-mmb-cvt-out-xn.md) |
| prefill indexer relu-sum | `0013` | ON (bit-identical) | flat pp8192 / +1.8 % pp32768 | [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md) |

**Final product:** qwen4exp IQ4_NL, gfx1151, `-b 8192 -ub 8192`, BF16 rounding path on:
**1379.2 / 1320.0 t/s** (pp8192 / pp32768); default build at `-b/-ub 4096`: 1308.2 / 1269.6.  The other
solution's session-5 cells were 1323.7 / 1374.4 at the target — we are ahead at pp8192 and ~4 % behind
at pp32768 (was −10.6 %).

**Two facts for the next session:**
* The `mmb_cvt` win is a **consistency fix**: the graph already marked `out_xn` BF16-only and the fused
  combine was the one producer that did not emit the copy.  The unfused `norm.cu` path already did, and
  fused-before / fused-after / unfused all produce the same text — bit-identical, not a re-baseline.
* `ffn_out`/`blk16` is **not** a gap against the reference: its `build_layer_ffn`/`build_moe_ffn` are the
  same, so its merge path is equally dormant on qwen4exp.  See the handover note above; do not
  re-litigate.

### Session-5 finding (2026-09-22) — fresh target-ubatch profile, memory accounting, refined tasks

**Throughput** (qwen4exp IQ4_NL, gfx1151, `-ctk/-ctv f16`, `-n 0`, our full default set vs the
reference's launcher env):

| config | pp | ours | reference | delta |
|---|---:|---:|---:|---:|
| `-b/-ub 4096` (A/B protocol) | 8192 | 1255.5 | 1200.1 | **+4.6 %** |
| `-b/-ub 4096` | 32768 | 1206.4 | 1224.5 | −1.5 % |
| `-b 8192 -ub 8192` (target) | 8192 | 1280.8 | 1323.7 | −3.2 % |
| `-b 8192 -ub 8192` | 32768 | 1228.4 | 1374.4 | **−10.6 %** |

So at the clean `-b/-ub 4096` protocol we are **ahead at pp8192** and ~1.5 % behind at depth; the
`-ub 8192` depth point is where the gap reopens.  The reference gains +12 % from `-ub 4096 → 8192` at
pp32768, we gain +1.8 %.

**Memory accounting** (`-ub 8192`).  Peak system used: ours `-lzm auto` **102 GB**, ours `-lzm on`
**74 GB**, the reference `-lzm off` **112 GB**, the reference `on-direct` **86 GB**.  Two terms explain
it:

* **~28 GB = the PLE residency.**  Our default is `-lzm auto`, but on gfx1151 the ROCm device reports
  `mmap_support = false` (`ggml-cuda.cu`: `props->type != IGPU`), so `llama-model.cpp` resolves
  `AUTO → OFF` and `per_layer_token_embd.weight` stays resident in `ROCm_Host`.  The reference's
  `on-direct` is its own managed-lazy mode, so it never has this.
* **~9 GB = the qwen4exp HC `block_out`/`inject` graph-output pins** that make the `hc_combine_norm`
  matcher possible (~1.1 MiB/token; ≈9 GB at 8192, ≈18 GB at 16384 — the session-2 record).  The
  reference has no pin.

**Neither is the throughput cause**: the reference scores **1378.7 t/s** with `-lzm on-direct` and
**1381.8 t/s** with `-lzm off` at `-b8192 -ub8192 -p32768`.  They are a *product-quality* regression
(wasted memory) and a `-ub 16384` enabler, not a speed one.

**Kernel family diff** (ours − reference, ms; `-b 8192 -ub 8192 -p 32768 -n 0 -r 1`; total
**52994 vs 46737 = +13.4 %**, matching the ~11 % t/s gap):

| family | ours | ref | Δ | what it is |
|---|---:|---:|---:|---|
| **hc_combine_norm** | 3888.0 | 2076.9 | **+1811** | ours F32 1024t; its `_b256` reads **BF16** `blk16`/`res16` |
| **mmb_cvt_f32_bf16** | 2088.7 | 610.3 | **+1478** | identical kernel + launch; ours 1040 calls vs its 1544 (per-call 5× bigger) |
| mmb_dense | 14533.8 | 13898.6 | +635 | 4304 vs 3920 calls (`<128,128,32,64>` +384) |
| **moe_reduce** | 1479.5 | 846.5 | **+633** | ours `f32_vec4` vs its `bf16_v4` (`MMB_DOWN16`) |
| **indexer** | 1254.6 | 664.5 | **+590** | ours histogram passes; its `idx_relu_sum` + `qsa_expand_complete_blocks_512` |
| elementwise | 3179.4 | 2756.9 | +423 | |
| mmq/mmvf | 595.2 | 211.4 | +384 | |
| mmb_f32split | 1653.6 | 1329.6 | +324 | |
| mmb_routed_glu | 7326.2 | 7827.4 | **−501** | we win |
| hc_gate_mix / rms | 1577.7 / 1819.6 | 1665.3 / 1831.5 | −88 / −12 | we win |

**Read:** at depth the gap is now dominated by **BF16 intermediate traffic** (HC combine + MoE epilogue
≈ 2.4 s, ~4.6 %) — the same "larger decision for the maintainer" the `_b256` rejection record flagged
(the class matches the MMB bf16 WMMA we already ship default-on).  **The MoE half is now DONE**
(`patches/0010`, default OFF); the **HC half is the next session's target** (see NEXT SESSION above).
The two non-lossy targets are `mmb_cvt` (+1.5 s, identical kernel, so our activation cache / `mmb_root`
keying must be converting redundant/large tensors) and the prefill **indexer relu-sum** (+0.6 s; the
audit's "already banked" line is wrong for prefill — our fused score op is `n_tokens == 1`, so prefill
runs a separate `unary_op<relu>` (559 ms) + head-sum adds, while the reference fuses them).

**`-lzm auto` semantics (maintainer, 2026-09-22).**  `AUTO` resolving to `OFF` on the iGPU is what
hides the PLE memory win from every default run.  The agreed semantics for our branch:

| mode | meaning |
|---|---|
| `-lzm off` | full preload (PLE resident) |
| `-lzm on`  | classic mmap-lazy |
| `-lzm auto` | upstream `auto`; the managed LRU PLE reader is **opt-in** via `LLAMA_LAZY_BUF_MB=<MiB>` |

`--lazy-buffer-size` is dropped as a CLI argument; the managed reader's buffer is the env var.

**Measured the same day** (qwen4exp IQ4_NL, `-b 8192 -ub 8192`, pp8192/32768, t/s):

| arm | pp8192 | pp32768 | peak used |
|---|---:|---:|---:|
| `-lzm off` (resident) | **1284.7** | **1232.3** | ~102 GB |
| `-lzm on` (mmap) | 1219.3 | 1216.8 | ~74 GB |
| `-lzm auto` + `LLAMA_LAZY_BUF_MB=4096` (managed) | 1090.5 | 1183.5 | ~76 GB |
| `-lzm auto` + `LLAMA_LAZY_BUF_MB=16384` (managed) | 1098.0 | 1187.0 | — |

The managed LRU is the **slowest** of the three — its arena adds a per-row copy + clock-eviction on a
table whose prefill access pattern is streaming, and a 16 GB budget does not help — so it is **gated
OFF by default** (env opt-in, source kept for later work; it measured slower than both alternatives).
`-lzm off` stays the throughput default.  `-lzm on` is the memory-freeing option at ~1.3 % depth cost
(and ~5 % at pp8192).  The managed reader **does** enable the parked `-b/-ub 16384` context
(1118.7 t/s at pp16384, where the resident build fails to create the context), so it is a candidate
for item 3 once its cost is reduced.  All three lazy modes produce **identical greedy text**
(`7e4a6a4e66fb`, 322 chars, prose prompt) — the mode is a memory/latency choice, not a numerics one.

**Cache discriminator (2026-09-22).**  To separate the managed reader's intrinsic cost from page-cache
shortfall, the three arms were compared at two levels of our own memory pressure (page warmed before
each arm; the managed arm's own footprint leaves free = 123 − used for page cache):

| config | resident (`off`) | mmap (`on`) | managed (`auto`+4 GB) | managed vs mmap |
|---|---:|---:|---:|---:|
| `-ub 2048 -p 8192` (used 106/79/81 GB → ~42 GB cache free) | 1236.7 | 1204.7 | 1157.2 | **−4.0 %** |
| `-ub 8192 -p 8192` (used 120/93/95 GB → ~28 GB cache free) | 1284.5 | 1216.9 | 1098.4 | **−9.7 %** |

So the shortfall is **not** the whole story: even with the whole 27 GB PLE table comfortably in page
cache (`-ub 2048`), the managed reader is still ~4 % slower than mmap vs ~10 % under pressure.  **Two
independent causes:** (1) the LRU arena has an intrinsic overhead on a streaming prefill access pattern
(the reference's `on-direct` is a *no-cache* parallel-pread reader, not an LRU arena — that is why its`on-direct` ≈ `off`);  and (2) page-cache pressure adds a further ~6 % to the managed arm (and ~2.7 % to
mmap).  The PLE task's fix is therefore the **reader design** (streaming/no-cache fast path), not only a
smaller footprint.  Source kept, gated OFF; item 13.

### Do these in order

> **Superseded (end of session 8): this is the session-5-era list, kept for its gate rationale.  The
> live handoff is the START HERE / NEXT SESSION block at the top of this file.**

1. **Run the full `beta/mmb-general` BETA-TESTING gate suite on the current default build**
   ([`../../beta/mmb-general/BETA-TESTING.md`](../../beta/mmb-general/BETA-TESTING.md)).  **Purity is an
   intra-build contract** (`GREEDY-PURITY.md`): the decode/verify band `W=1..8` must take one reduction
   path (`plain == draft-mtp` greedy text, byte-identical), `test-logits-width-probe` must print
   `width_purity=PASS (worst maxdiff 0)`, and the output must be coherent.  **Cross-build bit-identity
   (vs r12, or MMB on vs off) is NOT a gate** - a prefill kernel swap (MMB's dequant->bf16 WMMA) is the
   "approved prefill re-baseline", so its greedy text legitimately differs; see the 2026-09-22
   correction record below.  Green before any promotion, and do not trust performance numbers as "the
   product" until then.  **Still owed: the MTP gate (Gate 4).**  Already green: the op oracles
   (`GATED_DELTA_NET`, `INDEXER_TOPK`, `FLASH_ATTN_QSA` 26/26, `FLASH_ATTN_EXT` 5955 OK / 0 FAIL,
   re-run 2026-09-22), width probe PASS on 27B + qwen4exp + 35B-A3B, conv fused==unfused row-0 hashes.
2. **Next code items, in this order** (session-5 profile; the item-8 follow-ups are now deferred behind
   the bigger families):
   1. **HC BF16 streams (`blk16`/`res16`), default OFF** — the next session's target; full scoping in
      the NEXT SESSION block above (item 16).  ~1.8 s / ~3.8 % at depth, lossy.
   2. **BF16 MoE epilogue (`down16`)** — **DONE** (`patches/0010`, default OFF; kernel 1479 -> 846 ms).
      Kept as the worked template for the HC port.
   3. **`-lzm auto` semantics + managed PLE reader** — semantics DONE; the managed LRU is measured
      slower than resident/mmap, so it ships **gated OFF** (env `LLAMA_LAZY_BUF_MB` opt-in) with the
      source kept.  It does unlock the parked `-b/-ub 16384` context (1125.5 t/s); the perf work before
      it can default-on is in item 13.
   4. **`mmb_cvt_f32_bf16`** — **DONE (session 6, `patches/0012`)**: the fused combine now emits the
      BF16 `out_xn` copy the graph already marks (the cache could not dedupe), bit-identical, +3.3 %/+3.2 %.
   5. **Prefill indexer relu-sum** — **DONE (session 6, `patches/0013`)**: fused relu+head-sum, bit-identical,
      +1.8 % at pp32768.  (The audit wrongly marked it as banked; our fused score op is `n_tokens == 1` only.)
   6. `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`, then `QSA_SCORE_WMMA` — **the next item** (item 15).
3. **Use `-b/-ub 4096` for perf A/Bs** (session-4 methodology finding).  The `-ub 8192` absolute target
   is fine for a single number, but an A/B whose arms change the graph's memory footprint compares two
   pressure regimes there.  Record the min free memory with any `-ub 8192` result.
4. Keep the **default-on policy**: every beneficial feature is ON; its env var only *disables* it.  Never
   run a benchmark with a feature left off.

### Correction 2026-09-22 (session 5) — the `MMB=0 == r12` gate is retracted

The "Gate 1" copied into this handover (and its `README.md`) was the beta's **opt-in-era** regression
check: while `GGML_CUDA_MMB` was `getenv ? atoi : 0`, "MMB unset" *literally was* "r12 + the beta's
arch-neutral always-on groups", so diffing greedy text against r12 caught a neutral group with a numeric
side effect.  It was a bisection aid, not the purity doctrine.

**The contract is intra-build** (`GREEDY-PURITY.md` §5: *"Bit-identical to stock is a reproducibility
requirement, not a correctness requirement"*; §6: the verify batch and the one-at-a-time decode must
take the **same** association order **within a build**).  Different builds are expected to produce
different greedy text: `beta/mmb-general/README.md` calls the MMB on/off difference the **"approved
prefill re-baseline"** and its own width-probe table shows row-0 hashes differing above
`MMB_MIN_T=512` while `width_purity` stays PASS.

Under the default-on policy the gate's premise is gone anyway: MMB is default-on and four other
gap-closing features are default-on, so `MMB=0` is not "the default minus MMB".

Measured 2026-09-22 (dense 27B Q8, 128-token greedy, `prompts/prose-rdna-boosts.txt`, seed 42 / temp 0):

| build / env | text |
|---|---|
| r12 base | `2eb597253646` |
| `gap-closing`, `GGML_CUDA_MMB=0` | `2eb597253646` (== r12) |
| `gap-closing`, default (MMB on) | `efad2aa9a83e` (the approved prefill re-baseline) |

So the legacy check happens to pass, but it says nothing about the product.  **The gate set that matches
the contract is:** `test-logits-width-probe` PASS (worst maxdiff 0) on f16/bf16/q8_0, `plain ==
draft-mtp` greedy text within the build, the MTP acceptance gate, coherence, and the op oracles.  A
*cross-build* comparison is legitimate only as a **targeted** assertion (e.g. "MMB is prefill-only, so
W=1 decode logits are unchanged from r12"), never as a blanket equality gate.

References corrected in the same change: this file's START HERE and session-3 record, `README.md`'s
"Do first" item 1, and `beta/mmb-general/BETA-TESTING.md` §0/Gate 1.  See the 2026-09-22 `WORKLOG.md`
entry.

### Rebuild / run (copy-paste)

```sh
cd ~/llama.cpp && ~/bin/build-llama-rocm-714                      # full build (ccache; ~4 min warm)
# fast loop:  cmake --build build-rocm --target llama-bench llama-cli -j 16
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MM=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
for f in ${MU%/*}/*-0000*.gguf; do cat "$f" >/dev/null; done   # warm page cache first
# ubatch-8192 baseline (ours, full default set, no env) -- the campaign's absolute target:
~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 8192 -ub 8192 -p 8192,32768 -n 0 -r 2
# A/B protocol: use -b/-ub 4096 (memory-safe; see the 2026-09-22 methodology record).
# At -ub 8192 the box bottoms at 2-3 GB free and the two arms of a graph-shape change run in
# different pressure regimes, so a <2 % delta there is not trustworthy.
~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 4096 -ub 4096 -p 8192,32768 -n 0 -r 3
# long context: use -ub 4096 (at -ub 8192 the pp65536 point memory-thrashes at ~844 t/s;
# -ub 4096 is clean and pegged at 100 %, ~1093 t/s):
~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 4096 -ub 4096 -p 65536 -n 0 -r 1
# gates used repeatedly (qwen4exp IQ4_NL):
#   width probe:   ~/llama.cpp/build-rocm/bin/test-logits-width-probe "$MU" \
#                    prompts/prose-rdna-boosts.txt 1024 512     (expect width_purity=PASS, worst maxdiff 0)
#   op oracles:    test-backend-ops -o GATED_DELTA_NET / INDEXER_TOPK / FLASH_ATTN_QSA / FLASH_ATTN_EXT
# the other solution's same-config reference (its launcher env):
( set -a; . archive/work/wip-archive/iq4nl-prefill/launcher-env.txt; set +a; \
  ~/pwilkin-llama-cpp/build-rocm/bin/llama-bench -m "$MU" -dev ROCm0 -ngl 999 -fa on \
  -lm none -lzm on-direct -ctk f16 -ctv f16 -b 8192 -ub 8192 -p 2048,8192,16384 -n 0 -r 2 )
```

### Reference scoping — the item-3.5 QSA correctness fixes (CLOSED 2026-09-22)

**Result: `b0f31f587` ported (`patches/0005`); `40c0b9c38` and `14fff4f97` audited N/A.**  See
[`2026-09-22-qsa-item-3.5-audit.md`](2026-09-22-qsa-item-3.5-audit.md) for the full mapping.  The
short version, kept here because it is the mapping a future QSA change will need:

* `40c0b9c38` (maskless only where qsa3 consumes it): **N/A** — we have no `maskless`/
  `NO_DENSE_MASK` path.  The derived path passes `mask=nullptr` **only** to `ggml_flash_attn_qsa`, and
  **both** QSA kernels honour `cell_vis`/`q_vis` (`fattn-qsa.cu:122-131`, `fattn-qsa3.cu:371-462`);
  qsa3 refuses a maskless op without derived vis (`fattn-qsa3.cu:620-621`), so the decode band falls
  to the VEC kernel that also honours them.  The V3 derived kq mask on `FLASH_ATTN_EXT` is guarded by
  `ggml_cuda_flash_attn_ext_supported` (`fattn.cu:853`: false unless MMA/TILE).
* `14fff4f97` (−1 sentinels into the masked path): **N/A** — our selection list is a radix top-k of
  real cell indices (`indexer-topk.cu` writes `col`/`c` in `[0, n_kv)`; the `-INF` bin is filled with
  a real index).  The `-1`/`INT32_MAX` sentinels live only in the kernel **inputs**
  `blk_idx`/`blk_tail`, and `cell_blk` is mapped to a valid `dead_bid` (`llama-memory-hybrid-idx.cpp:750`).
  So `ggml_set_rows(kq_mask_all, zeros, top_k_3d)` (`qwen4exp.cpp:1714`) indexes rows in `[0, n_kv)`.

`git -C ~/pwilkin-llama-cpp show 40c0b9c38` / `14fff4f97` remain the reference diffs if this needs
re-checking.  One defensive note for a future change: the VEC QSA kernel dereferences `maskh` when
`cell_vis` is null, so a change that could make both null would fault — add the reference's assert (or
enforce `mask || cell_vis` in `ggml_cuda_flash_attn_qsa`) at that point.

### Next item — items 4, 6, 7 DONE; item 8 DONE (audit); the QSA prefill-score follow-up is next

**Item 4 is DONE (session 4, `patches/0006`).**  The narrow-row RMS norm (`norm-gated.cu::rms_rows_f32`,
8 rows/block) is ported, default-on and bit-identical; ~+0.3 % at the clean `-b/-ub 4096` protocol —
[`2026-09-22-norm-rows-fusion.md`](2026-09-22-norm-rows-fusion.md).  `idx-relu-sum` was already banked.

**Item 6 is DONE (session 4, `patches/0007`).**  The re-profile showed the gap is not geometry (same
grid/regs/LDS) but the per-cell `cell_vis` check our design applies and the reference's maskless qsa3
does not.  Folding it into `umask` at merge time (where the bit already exists and already means -inf)
is bit-identical and drops the check from the hot loop: `qsa3_attn_kernel` 809.6 -> 672.9 ms, +2.4 %/+1.7 %
end-to-end — [`2026-09-22-qsa3-visibility-fold.md`](2026-09-22-qsa3-visibility-fold.md).

**Item 7 is DONE (session 4, `patches/0008`).**  The tall `384x64` tile's 2× launch count was the `M=4`
HC inject sharing the tile with the `M=320` down; the gate is now `M >= 16`, so the inject takes the dense
tile — bit-identical, +0.8 %/+1.1 % — [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md).

**Item 8 is DONE (session 4, audit).**  The reference deleted its 9 QSA graph flags in `ac1ebb4e0`
("compile in the tuned defaults"); 7 are present/superseded on our tree (whole-attn / block-selection /
compact-metadata / direct-indices / maskless→derived-vis / token-embd, and the fused score we have for
decode).  **Two are genuine un-ported prefill-score optimizations** and are the next work:

1. **`QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`** (do this first).  The reference scores the prefill in
   `min(n_tokens, 512)`-token strips and, per strip, trims the scorer to the first
   `(max_query_pos+1)/ratio` blocks — safe because the trimmed blocks are `-inf` in the visibility
   metadata, so the selection cannot change.  Our chain scores the full `n_blocks` every time; the
   indexer score/top-k chain is ~185 ms + the score matmul at pp8192, and the bound roughly halves it.
   Port needs `qsa_position_prefix(ubatch)` + `qsa_prefix_limits(...)` in `llama-memory-hybrid-idx.*`,
   a `qwen4exp_query_strip()` in `qwen4exp.cpp`, and the strip/limits threaded through the QSA graph
   input + `can_reuse` (the reserve-time synthetic ubatch must stay unbounded).  Gate: width probe PASS
   + same-seed text + A/B at `-b/-ub 4096`.
2. **`QSA_SCORE_WMMA`** — extend a fused score to the prefill band (`n_tps >= 128`, `idx_dim==128`,
   `n_idx_h==4`); ours (`GGML_CUDA_QSA_INDEXER_SCORE`) is `n_tokens == 1` only.  This is a numerics
   change if the fused reduction order differs, so it needs the width probe + same-seed text and a
   careful read of the reference's op before porting.

Both are recall-speed (Phase-1) items; the audit record has the full flag table and the port scoping.

### Current state (exact)

| what | where / value |
|---|---|
| fork `~/llama.cpp` | branch **`gap-closing-r13`** @ **`575c4c091`** (`git rev-parse HEAD^{tree}` = `dadc99000db4472056be920d3f73e3308eef3f3a`) = r13 + the 12 `beta/mmb-general` patches + gap-closing `0001..0014`/`0016`/`0017`/`0018`/`0019` (**`0015` dropped** — it is in delivery r13 block 00) |
| fork build | `~/llama.cpp/build-rocm` (gfx1151, ROCm 7.14), full feature set **default** |
| this repo | branch `gap-closing`, `wip/closing-the-gap/patches/0001..0014` + `0016..0019` |
| the other solution | `~/pwilkin-llama-cpp` @ `b0f31f587`, `build-rocm` |
| model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93 GiB, qwen4exp) |
| MoE test model | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| MTP sidecar | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (the shared-NextN head — **loads now** that r13 block 00 carries the fix; acceptance 0.855) |

Rebuild the campaign from scratch (the verified flow): `git checkout rdna-boosts-r13 && git checkout -b
gap-closing-r13 && git am <repo>/beta/mmb-general/patches/*.patch && git am
<repo>/wip/closing-the-gap/patches/00{01..14}-*.patch <repo>/wip/closing-the-gap/patches/0016-*.patch
<repo>/wip/closing-the-gap/patches/0017-*.patch <repo>/wip/closing-the-gap/patches/0018-*.patch
<repo>/wip/closing-the-gap/patches/0019-*.patch` (skip `0015`).  Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm-714`.  Runtime:
`export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0`.
The exports are the fork commits so the code survives a fork reset; the applied tree is verified to
match (`scripts`-free check in the session-8 records).

### What NOT to redo (sessions 3–4)

* **`hc_combine_norm_f32_b256`** — not bit-identical, slower; the reference's speed is its BF16 HC
  traffic.  If HC-combine speed is revisited, the change is the **BF16** `blk16`/`res16` path (lossy,
  needs the maintainer's call), not the thread count.
* **`concat_transposed` drop (item 5)** — already gone at `-ub 8192`.
* **`-ub 16384`** — parked; the `-ub 8192` long-context point memory-thrashes, so use `-ub 4096` there.
* **MMB-off byte-identity and the MTP gate** — the MTP gate is **DONE** (session 8: acceptance
  0.85541, 56.5 vs plain 31.7 t/s) and the MMB-off **cross-build** byte check is **retracted** (it is
  not the purity contract; see the 2026-09-22 correction record).
* **Don't trust an ad-hoc GGUF type parser** (2026-09-22): a session's parser used the wrong ggml enum
  and reported the IQ4_XS model's types as IQ2_XS when they are **IQ4_NL**.  The IQ4_XS model has **no
  IQ2_XS/IQ1_S at all** (IQ4_NL 45.7 GiB, IQ3_S 31.6, Q8_0 8.3, IQ4_XS 0.8, Q6_K 0.5, small F32/BF16) —
  all MMB-covered.  Its **PLE is IQ4_NL, 26.8 GiB** (not IQ2_XS), so a "PLE→Q4_0" swap cannot shrink
  it (both are 4.5 bpw).  Use `llama-gguf`/`llama-bench`'s own type output, never a hand-rolled enum.

---

## Session-4 record (2026-09-22): items 3.5 (audit), 4, 6, 7, 8 — and the ubatch confound

Five items closed, four records + one methodology finding.  All performance changes were gated
**bit-identical** (width probe PASS, same-seed greedy text unchanged) and **default ON**.

| item | result | record |
|---|---|---|
| 3.5 (audit) | `40c0b9c38` + `14fff4f97` are **N/A** — no maskless path, no sentinels in the top-k output | [`2026-09-22-qsa-item-3.5-audit.md`](2026-09-22-qsa-item-3.5-audit.md) |
| 4 narrow-row RMS norm | `patches/0006`, ~+0.3 % at `-ub 4096` | [`2026-09-22-norm-rows-fusion.md`](2026-09-22-norm-rows-fusion.md) |
| 6 QSA visibility fold | `patches/0007`, **+2.4 %/+1.7 %**, qsa3 809.6→672.9 ms | [`2026-09-22-qsa3-visibility-fold.md`](2026-09-22-qsa3-visibility-fold.md) |
| 7 tall-tile min-M | `patches/0008`, **+0.8 %/+1.1 %**, tall 1030/380→540/190 ms | [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md) |
| 8 QSA graph flags | audit: 7/9 present/superseded; **2 un-ported prefill-score items** hand to the next session | [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md) |

**Methodology finding (important for every future A/B):** `-ub 8192` on this box bottoms at 2–3 GB free
with ~40 % more kswapd reclaim, and a fusion whose arms change the graph's memory footprint run their
arms in **different pressure regimes** — item 4 read +1.05 % at `-ub 8192` but +0.27 % at `-ub 4096`.
Use **`-b/-ub 4096`** for perf A/Bs — [`2026-09-22-ubatch-8192-memory-confound.md`](2026-09-22-ubatch-8192-memory-confound.md).
The prior rejections were re-audited; the shipped wins stand (their sign was positive under the harsher
protocol), and `hc_combine_norm_f32_b256` still fails on **correctness** (deterministic text change), so
its rejection never rested on timing.

**Reference status after session 4** (qwen4exp): we are now **ahead** of `b0f31f587` on the QSA pipeline
(761 vs 812 ms) and on the IQ4_XS model (1244 vs 967 t/s); the residual IQ4_NL prefill gap is the
still-missing families priced in §13 (the dense/HC/conversion kernels), not the QSA path.

---

## Session-3 record (2026-09-21): Phase-1 item 2 — the depthwise conv1d is DONE

### Result

The other solution's `gdn-conv.{cu,cuh}` + `ple-conv.{cu,cuh}` are ported into the delivery, default
**ON** (`GGML_CUDA_DISABLE_CONV_FUSION=1` disables).  Fork tip **`1004c65db`**, exported as
[`patches/0004`](patches/0004-gap-closing-WIP-port-the-depthwise-conv1d-fusions-GD.patch).  Full record:
[`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md).

| model | pp | off | default (fused) | delta |
|---|---:|---:|---:|---:|
| qwen4exp IQ4_NL | 8192 | 1223.1 | **1259.9** | +3.0 % |
| qwen4exp IQ4_NL | 32768 | 1158.4 | **1195.6** | +3.2 % |
| qwen35moe 35B-A3B Q4_K_M | 8192 | 2128.1 | **2265.9** | +6.5 % |
| qwen35moe 35B-A3B Q4_K_M | 32768 | 1561.9 | **1672.5** | +7.1 % |

**Bit-identical**: the width-probe row-0 logits hash is identical fused vs disabled
(qwen4exp `268e0673300b7a33`, 35B-A3B `e97c9e304ce1ca8f`), `width_purity=PASS (worst maxdiff 0)` on
both, and same-seed greedy text is identical (`bb820cccf620`, 637 chars).

### Two adaptations our tree needed

1. `ple_conv_check`: our post-re-base `grouped_norm` emits a **3-D** `[n_embd, hc, T]` MUL, so the
transpose's view root has `ne[2] = hc`; the reference's `x->ne[2] == 1` test rejected every PLE layer.
The check now derives `C`/`T` from the transpose and only requires a contiguous F32 root of `C*T`
elements (the flat layout is the same `[hc_dim, T]` buffer either way).
2. `gdn_conv_check`: the shared `build_conv_state` (qwen35moe/qwen35/qwen3next) writes the snapshot
`cpy(view(concat), dst)` with a raw view; the sweep rejected that CPY.  It now accepts a CPY whose
source is a 3-column view of the concat (covered by `tail_from`) and rejects any other concat reader.

### Gates still owed (unchanged from session 2, plus the conv fusion)

* The **MTP** gate (Gate 4).  (The **MMB-off byte-identity** check was retracted 2026-09-22 - cross-build
  identity is not the purity contract; see the 2026-09-22 correction record above.)
* The **27B dense** width-probe run (session 2/3 ran qwen4exp + 35B-A3B).
* Re-check the delivery's **MoE/general GDN prefill records** now that the GDN fusion also fires on
qwen35moe/qwen35/qwen3next (the snapshot-cpy adaptation); the 35B-A3B numbers above are the first
signal.

---

## Session-2 record (2026-09-21): ubatch 8192 target, the 16k diagnosis, and the gatemix win

### Decision — ubatch 8192 is the target

Target **`-b 8192 -ub 8192`** for the head-to-head and all further prefill work.  The `-b 16384 -ub 16384`
regime is parked for a later session.

Uniform IQ4_NL (the other solution's checkpoint), gfx1151, `-ctk f16 -ctv f16`, `-n 0 -r 2`, ours with
**no env** (full default set), the other solution with its full launcher env:

| pp | ours, `-ub 8192` (pre-gatemix) | other, `-ub 8192` | ours, `-ub 2048` (old) | other, `-ub 16384` |
|---:|---:|---:|---:|---:|
| 2048  | 1159.1 | 1233.6 | 1181.5 | 1233.0 |
| 8192  | 1212.6 | 1346.5 | 1149.3 | 1338.8 |
| 16384 | 1179.0 | 1387.5 | 1129.2 | 1399.3 |

So ubatch 8192 is itself a real gain over our ubatch 2048 (+5.5 % at pp8192) and gives a stable,
reproducible baseline: at a **matched** ubatch we are **~10 % behind at pp8192 / ~15 % behind at
pp16384**, which is the number the §13 phase plan exists to close.  It also confirms the old "~4 %
behind" was partly the ubatch mismatch (ours ub2048 vs its ub16384).  Session 2 then added gatemix
(+1.2–1.5 % at pp8192/32768); re-measure the baseline with the current default before comparing.

### Gatemix — Phase-1 item 1's second half, DONE

The `hc_gate_mix_kernel` + `ggml_cuda_hc_gate_mix` existed in `mmb.cu` with no call site, so the HC gate
GEMM (`w_up @ lo`, `[320 -> 10240]`) dispatched standalone and the mix ran in the `dsv4_hc_pre` op.
Session 2 wired the matcher/call site in `ggml_cuda_try_fuse` (`94694a38e`, `patches/0003`):
`ggml_cuda_hc_mix_closed()` plus a branch at the gate `MUL_MAT` that handles **both** the unfused chain
and our delivery's explicit `ggml_dsv4_hc_pre` op (recovering the `[hc*n_embd, T]` activation the bf16
cache is keyed on from the gate GEMM's own activation input).  Default **on** on gfx1151
(`LLAMA_HC_GATEMIX=0` disables); RDNA4/RDNA3_0 stay off.

| pp | `LLAMA_HC_GATEMIX=0` | default | delta |
|---:|---:|---:|---:|
| 8192  | 1198.8 | **1212.6** | +1.2 % |
| 32768 | 1138.6 | **1155.6** | +1.5 % |

Width probe **PASS** (worst maxdiff 0) and same-seed greedy text **identical** (`471d102e7b7d`).  The
fusion shifts prefill logits by a bf16-epilogue ULP (three width-pure variants; the text is stable) and
is prefill-only, so the decode/verify band is untouched.  Full detail:
[`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md).  Caveat: the kernel is **IQ4_NL-only**,
so the mixed UD-IQ4_XS model is unchanged (Q8_0 gate) — a follow-up.

### Why `-ub 16384` fails (root cause, deferred not fixed)

`llama-bench -b 16384 -ub 16384 -p 16384` fails at `llama_init_from_model` because the single pp-graph
reserve needs a **33794 MiB** compute buffer and `cudaMalloc` returns OOM.  Measured breakdown at
`n_tokens = 16384` (allocator trace):

| term | size | notes |
|---|---:|---|
| `result_output` | 15520 MiB | `[n_vocab=248320, n_tokens]` f32 — the reserve uses `n_outputs = n_tokens` |
| qwen4exp HC pin | ~18114 MiB | every layer's `block_out` pinned as a graph output (~1.1 MiB/token) |
| graph working set | ~160 MiB | reused heavily |

* **The other solution allocates the same 15520 MiB `result_output`** (verified by enabling
  `GGML_ALLOCATOR_DEBUG` in its tree and running the same config) — its pp reserve is
  `n_outputs = 16384` too.  So this term is **not** something they solved and we did not; it is
  upstream's worst-case logits reserve.  Its total is 16980 MiB because it has **no HC pin**; our
  unpinned total is 15680 MiB — i.e. our graph is already ~1.3 GiB *smaller* than theirs (block-15
  W4/V3 work), and the entire 18 GiB excess is the pins.
* The pin lives in `src/models/qwen4exp.cpp::build_hc_combine` (`ggml_set_output(block_out)` /
  `ggml_set_output(inject)`) and exists so the fused `hc_combine_norm` matcher can read the **narrow**
  `block_out` base without the allocator reusing its buffer for the norm output.  It is what lets the
  matcher run at all; without the pin the matcher declines and the unfused (bit-identical but ~8 %
  slower) chain runs.  At `n_tokens <= 8192` the pin is affordable (~9 GiB); at 16384 it is not.
* A `nt <= 8192` guard on the pin **does** make 16384 create a context and run (verified: mixed IQ4_XS
  `-b/-ub 16384 -p 16384` -> 1069 t/s, compute reserve 15.68 GiB), and is bit-identical to the pinned
  path (`test-logits-width-probe` W=1..8 worst maxdiff 0).  It was **reverted** with the rest of the
  experiments because it is a workaround that costs the fusion above 8192; not needed for the 8192 target.
* **Do not ship the "matcher without the pin" path unverified.**  With the flag requirement removed and
  the pin off, the matcher fires but the logits differ from both the pinned fusion and the unfused chain
  (width probe W1 hash `5b7861bc` vs `02ece229`) — the alias check missed a real overlap.  The pin/flag
  gate is load-bearing.

### The PLE residency (the other half of the memory picture)

The 27.45 GiB `ROCm_Host` model buffer is **entirely `per_layer_token_embd.weight`** (27466 MiB; the
remaining `token_embd.weight` 644 MiB).  Findings:

* With **`-lzm on`** the PLE is mmap-lazy: it moves to a 26.8 GiB **CPU** mapping and `ROCm_Host` drops to
  0.63 GiB.  This is exactly what the other solution's launcher gets from `-lzm on-direct` (its summary:
  `ROCm0 67591 / ROCm_Host 341 / CPU_Mapped 27465` MiB).
* With the **default `-lzm auto`** our `llama-bench` does **not** lazy-load it: the loader sees
  `lazy_read::mode == 0 (OFF)`.  `params.lazy_mode` is parsed as AUTO (1) in `llama-bench`
  (`LAZYPARSE v0=1`) but `lazy_read::add` sees 0, so the value is lost between `to_llama_mparams()` and
  `llama_model_load()`.  `llama-bench` also has no `--lazy-buffer-size`, so the managed ~5 GiB capped
  reader is unreachable from it.
* **Candidate fixes for the later 16k session** (in order of preference): (a) fix the lazy-mode
  propagation so the PLE is mmap-lazy by default (frees ~27 GiB of pinned host memory and likely makes
  the pinned 16384 reserve fit without touching the HC matcher); (b) expose `--lazy-buffer-size` in
  `llama-bench` and use the managed ~5 GiB reader; (c) keep the `nt <= 8192` pin guard as a fallback.
  (a)/(b) preserve the HC design, which is the point.

### Current state (exact)

| what | where / value |
|---|---|
| this repo, `main` | `== origin/main == 830770a` (clean) |
| this repo, `gap-closing` | **the WIP branch; all session-2 work is committed here** (`b8aeaa1`, `c1c0b33`) |
| fork `~/llama.cpp` | branch **`gap-closing`** @ **`94694a38e`** (local; based on `mmb-beta` = r12 + the 12 `beta/mmb-general` patches + the three gap-closing WIP commits) |
| fork build | `~/llama.cpp/build-rocm` (gfx1151, ROCm 7.14), built 2026-09-21; full feature set **default** (incl. `hc_gate_mix`) |
| pre-port WIP (reference) | `~/llama-wip-mmb` @ `90bf12997` (`wip-mmb-general`), build at `build-rocm` |
| the other solution | `~/pwilkin-llama-cpp` @ `b0f31f587`, **rebuilt** (`build-rocm`) |
| model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93 GiB, qwen4exp) |
| MTP sidecar | `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (**not** the IQ4_NL dir's `shared-Q8_0` — see §12) |

Rebuild: `cd ~/llama.cpp && ~/bin/build-llama-rocm-714`.  Runtime:
`export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0`.
The **three** `gap-closing` fork commits are also exported to [`patches/`](patches/) (`0001..0003`, tip
`94694a38e`) so the code survives a fork reset.

### The policy (also in `AGENTS.md`)

**A beneficial feature that has passed the gates is ON by default; its env var only DISABLES it.**  Applied
in `0c860fe77`: `GGML_CUDA_MMB` default **on** (`=0` disables; RDNA3_0 included, `GGML_CUDA_MMB_RDNA3=0`
disables that arm); MMB `hc16` default **on** (`GGML_CUDA_MMB_HC16=0` disables); the HC `hc_combine_norm`
matcher default **on** (`LLAMA_FUSED_DSV4_HC_POST=1` forces the slower `DSV4_HC_POST` op); the HC
**`hc_gate_mix`** default **on** on gfx1151 (`LLAMA_HC_GATEMIX=0` disables; session 2).  **Never** run a
benchmark with a feature left off — if you type `FOO=1 <bench>`, ask why the default is not already `1`.

### What changed this session (vs the 2026-09-20 snapshot)

* **MMB/HC16 were opt-in and I benchmarked with them off** (835 t/s instead of 1136).  Fixed by the policy
  above; the real default is now the full set (body §G).
* **Found and fixed a latent bug:** the two graph-optimizer HC16 marking sites in `ggml-cuda.cu` read a
  hardcoded `getenv(...) : 0` and **ignored the arch config**, so HC16 never engaged even when the config
  said on.  Now default from the config.
* **Revived the `hc_combine_norm` prefill matcher** (fork `121ad7935`): three bugs — see
  [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md).  The matcher default now beats
  `DSV4_HC_POST` by ~+1.5 % prefill.
* **Session 2: wired the `hc_gate_mix` fusion and made it default-on** (fork `94694a38e`, `patches/0003`)
  — +1.2–1.5 % at pp8192/32768, width-pure, text-identical.  See `2026-09-21-hc-combine-norm.md`
  and the session-2 record above.  Phase-1 item 1 is therefore done.  Caveat: the kernel is IQ4_NL-only,
  so the mixed UD-IQ4_XS model is unchanged.
* **Session 2: ubatch 8192 adopted as the target** and the `-ub 16384` failure root-caused and deferred
  (see the session-2 record above).
* **MTP qualified** (parked): see [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md).
  Our plain decode is ahead, fixed-depth MTP speedup is at parity, and the only real MTP gap is
  `nextn_shared_target_tensors` support.

### Numbers to reproduce (gfx1151, qwen4exp IQ4_NL, prefill, `-b/-ub 2048` — session-1 record, superseded
by the ubatch-8192 table above)

| pp | **default (full set)** | `GGML_CUDA_MMB=0` | pre-port WIP (full set) |
|---:|---:|---:|---:|
| 2048 | 1158.6 | — | 1167.1 |
| 8192 | 1136.2 | 835.1 | 1136.4 |
| 32768 | 1070.9 | 811.1 | — |

### Strategic framing (maintainer, 2026-09-21)

We **should** be significantly faster: our chunked GDN graphs are **~5× faster than the other solution's tiled GDN**,
and our MMB optimizations are more refined.  The recurring pattern is that we optimize very well once we
find what is missing — the other solution simply **has more things**.  So the work is to keep finding the missing
pieces (the HC `gate_mix`/combine+norm fusions, the depthwise conv1d, the gated RMS-norm, the indexer
relu-sum, the MoE bf16 epilogue, the sparse QSA decode + incremental indexer) and land them default-on.

### Known caveats before touching the fork

* The fork branch `gap-closing` carries **env-gated debug traces** (`LLAMA_HC_CN_DEBUG` in `ggml-cuda.cu`
  and `ggml.c`); they are inert by default but should be removed before any patch is cut.
* The revived `hc_combine_norm` matcher and the `hc_gate_mix` fusion have passed the **width probe and
  same-seed text** gates (uniform IQ4_NL) but not the full BETA-TESTING suite; the MMB-off byte check and
  the MTP gate are still owed.
* `LLAMA_FUSED_DSV4_HC_PRE`/`_POST` env toggles are WIP A/B knobs.
* **Follow-up:** the delivery's `hc_combine_norm_f32` is the 1024-thread/3-column variant; the reference's
  `hc_combine_norm_f32_b256` (256 threads, two packed elements/thread) is the obvious kernel swap behind
  the same matcher.  Also `hc_gate_mix_kernel` is IQ4_NL-only (Q8_0 for the mixed models is a follow-up).
* The `-ub 16384` bug is **out of scope for MMB**; it is a delivery graph/allocator issue (see the
  session-2 record).

---

## Update 2026-09-21 (session 1) — current state of play

> Session-1 record; the session-2 handoff is the START HERE block at the top.  This section still has
> the reference inventory (what each side has, commit deltas, priorities) that the phase plan builds on.

This section supersedes the stale references in the body. The body's measurements remain valid as
**dated, gated-tree** evidence, but two references have moved and the plan needs five additions.

### A. Our side: `wip/mmb-general` was promoted to `beta/mmb-general`, 5 → 12 patches

The body compared a **5-patch, gfx1151-only WIP** at tip `90bf12997` (`~/llama-wip-mmb`). The current
reference is **`beta/mmb-general`** — **12 patches**, applied tree
**`bca69f23dd29acef2d8898c6fd492104e078eef1`**, verified `git am` **12/12** on top of the r12 delivery
(`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).

* The new patches are the **gfx1201 (RDNA4) port** (0006–0010) and the **gfx1100 (RDNA3_0) deltas**
  (0011–0012). They are arch-scoped; on **gfx1151** the code is the core the body measured, so the body's
  gfx1151 numbers carry over except where §C says otherwise.
* **What the 12-patch beta still does NOT add** (grep of the applied tree, not inference):
  `gdn-conv.cu`, `ple-conv.cu`, `norm-gated.cu`, `idx-relu-sum.cu` and `hc-cn.cu` are still **absent**
  (items 2–6 below).  **Item 1 is done in session 2**: `hc_gate_mix_kernel` was wired (the matcher/call
  site, not the kernel) and is default-on on gfx1151 — see the session-2 record at the top.  So body
  §8.2 **items 2–6 remain open**, item 1 is closed.

### B. The other solution's side: `f5daaa3cf` → `b0f31f587`, 10 new commits

The body pinned `f5daaa3cf` (2026-09-12). The branch tip is **`b0f31f587`** (2026-09-16). The delta:

**Prefill-relevant**

| commit | what | why it matters here |
|---|---|---|
| `40a9f4d01` | *hip: extend MMB quants and fuse Flash-Next F32 PLE* | MMB quant coverage goes from a handful of types to **23** (adds Q4_0/Q4_1/Q5_0, Q2_K, the whole IQ1/IQ2 family, MXFP4, NVFP4) through a new `mmb-quant.cuh` generic dispatcher; the direct **PLE conv** now also takes **F32** weights (Flash-Next's PLE weights). Real gap: our beta covers **10** types and has **no** `ple-conv.cu` at all. |
| `40c0b9c38` | *qsa: drop the dense mask only where the qsa3 kernel will consume the op* | Correctness **and** a prefill win on its tree: the mask-forced workaround cost `pp16384` 945.68 → 1067.80 t/s; the fix decides maskless at the use site and `GGML_ASSERT`s the invariant in the dispatcher. The failure mode it fixed was decode non-determinism (10/10 → 1/10) that collapsed long sessions into repetitions. Our derived-visibility/maskless path should be audited against this. |

**Decode / MTP-relevant**

| commit | what | why it matters here |
|---|---|---|
| `d67d58836` | *hip: enable sparse QSA decode and incremental indexer state* | New **sparse selected-cell decode** kernels (`qsa-decode.cuh` SIMT + `qsa-decode-wmma.cuh` WMMA) that read selected F16 K/V cells directly, plus an **incremental indexer-key cache** (`src/qsa-prefix-state.h`, `llama-memory-hybrid-idx.*`). Measured on its tree: serial depth-40000 **25.85 → 28.82 t/s**, MTP 40680-token **31.17 → 35.57** (first) / **32.69 → 39.10** (repeat), for 104 MiB @65k / ~416 MiB @256k of cache. This is a **new axis** the body only gestured at (its item 11). Our delivery has a *different* QSA-sparse-FA decode path plus an incremental **derived-block-vector** cache (`GGML_CUDA_QSA_INDEXER_CACHE`, default on) — overlapping, not equivalent; needs a 1:1 audit. |
| `0f2950198` | *qwen4exp: skip unused HIP decode indexer work* | A temporary dense-decode bypass, **superseded** by `d67d58836`. Listed only so it is not mistaken for the current state. |

**Correctness / housekeeping**

| commit | what |
|---|---|
| `b0f31f587` | QSA block window sized by the highest stored position, not the occupied-cell count (fixed an M-RoPE image + MTP crash ~300 tokens after an image). |
| `14fff4f97` | Keep the `-1` selection sentinels out of the masked attention path. |
| `ac1ebb4e0` | **Compile in the tuned defaults and drop the env gating** — `mmb_enabled()`, `gdn_conv_enabled()`, `norm_gated_enabled()`, `norm_rows_enabled()`, `ple_conv_enabled()` now return `true`, and the `LLAMA_*` experiment switches are gone. |
| `0cfb81512` | Drop stale comment references to the removed gates. |
| `be905cf7d` | Recurrent cache: no warning for positions in a stateless cache. |
| `31b38632c` | server: a zero draft length means speculation off. |

### C. Impact on the plan

1. **The §6 ablation price-list can no longer be reproduced against the other solution's current HEAD.**
   `ac1ebb4e0` deleted the env switches Appendix D zeroed. Re-measure against `b0f31f587` as a *default*
   build, or bisect by reverting the compiled-in defaults; do not re-run the old env ablations.
2. **The prefill gap inventory (§8.2 items 1–6) is unchanged** — the beta set did not close any of them.
   Item 3 is now *larger*, because the other solution also fuses the **F32 PLE** conv.
3. **Item 11 is promoted from a footnote to a first-class decode item** (sparse QSA decode + incremental
   indexer). It is the one newer feature the other solution has that is a measurable, self-contained optimisation
   rather than a refinement, and it is orthogonal to the prefill campaign — workable in parallel.
4. **MMB quant coverage:** our beta's 10 types vs its 23. Unlikely to move the uniform-IQ4_NL gap
   (both fire there), but a completeness/robustness gap for arbitrary GGUFs (Q4_0/Q4_1/Q5_0 and
   MXFP4/NVFP4 are common). Low-to-medium priority.
5. **Three cheap correctness items to port/audit** independent of perf: `40c0b9c38` (maskless only
   where qsa3 consumes it), `b0f31f587` (position-vs-cell block window), `14fff4f97` (sentinel
   handling). They prevent long-session corruption and are far cheaper than the perf items.
6. **MTP is not a missing optimisation in the other solution's favour — it is a different axis.** It has upstream
   `draft-mtp` with a **fixed** `n_max` and only upstream's per-step `p_min`/`n_min` early stop; there
   is **no** cross-round adaptive controller in its tree. Our `draft-mtp-adaptive` controller is a
   depth-policy advantage that composes with its per-step decode gains. **Measured 2026-09-21** (see
   [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) and §12): our plain decode is
   ahead (+2–6 %), the fixed-depth MTP **speedup is at parity** (ours `n3` 1.90/1.78/2.05 vs its
   1.91/1.79/2.02 on code/prose/recall), and our adaptive wins recall (2.40x) but over-drafts code and
   prose on qwen4exp — a tuning item, not a structural one.  The one real MTP gap was
   **`nextn_shared_target_tensors` support** — **FIXED 2026-09-22 (session 7, `patches/0015`)**: the
   MTP driver inferred KV sharing from `ctx_other` alone, but a shared-NextN head only *borrows* the
   target's `token_embd`/`output`; gated on the `gemma4-assistant` arch, the shared sidecar now runs
   with 0 draft errors and acceptance 0.287 (previously every draft round past the first failed an
   M-RoPE `X < Y` check) —
   [`2026-09-22-mtp-shared-nextn-fix.md`](2026-09-22-mtp-shared-nextn-fix.md).  It is an **upstream
   bug (#23398)** now delivered in block 00 (release `v16-ebbb18522-r13`); the WIP `patches/0015` is
   superseded.

### D. Body §6/§9 caveat

The kernel-level comparisons (`mmb_dense` +809 ms, HC, `rms`, `qsa3_attn` +195 ms) were made against
`f5daaa3cf`. Before trusting them again, re-profile `b0f31f587`: its tree gained the `mmb_quant`
dispatcher and dropped env gating, and the QSA decode/indexer changes add kernels to the trace.

### E. Where the current beta was built and validated

Applied and built on **gfx1151** on 2026-09-21 for the beta re-validation window
(`beta/mmb-general/BETA-TESTING.md`). Build: `~/bin/build-llama-rocm-714` from the `mmb-beta` branch of
`~/llama.cpp` (r12 + 12 patches, tree `bca69f23dd…`). The gfx1151 numbers in the body were measured on
the pre-beta WIP; the beta re-run is what confirms they still hold.

### F. Priority sequence (maintainer, 2026-09-21)

**Recall speed + correctness → decode speed + correctness → MTP tuning + correctness.**  The MTP
qualification is therefore **done to "is our MTP behind its?" depth only** and parked; its result and
the one real MTP gap are in [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) and
§12 below.  The headline: our plain decode is ahead, the fixed-depth MTP **speedup** is at parity, and
the only MTP gap is **`nextn_shared_target_tensors` support** (we cannot load the shared MTP head
the other solution's IQ4_NL model ships).  The body's prefill items 1–8 are the "recall" phase.

### G. Default-on policy + the recovered full-set numbers (2026-09-21)

**Policy (now in `AGENTS.md`):** a beneficial feature that has passed the gates is **ON by default**; the
env var only **disables** it.  Applied on fork branch `gap-closing`:

* `GGML_CUDA_MMB` default **ON** (`=0` disables), RDNA3_0 included (`GGML_CUDA_MMB_RDNA3=0` disables that arm);
* MMB `hc16` default **ON** (`GGML_CUDA_MMB_HC16=0` disables) — the two graph-optimizer HC16 marking
  sites in `ggml-cuda.cu` had a hardcoded env default of 0 and ignored the arch config, so HC16 never
  engaged without an explicit env (this is why the earlier full-set A/B looked flat);
* the HC `hc_combine_norm` matcher default **ON** (`LLAMA_FUSED_DSV4_HC_POST=1` forces the slower op).

gfx1151, qwen4exp IQ4_NL, `-b/-ub 2048`, prefill, **no env at all**:

| pp | default (full set) | `GGML_CUDA_MMB=0` | delta |
|---:|---:|---:|---:|
| 2048 | 1158.6 | — | |
| 8192 | 1136.2 | 835.1 | **+36 %** |
| 32768 | 1070.9 | 811.1 | +32 % |

Parity with the pre-port WIP at pp8192 (1136.4), i.e. the gfx1201/gfx1100 port did **not** strand the
win.  The earlier search for “1180–1220” was the `-ub 16384` regime (blocked by the context bug, item 3),
run/thermal variance, and ~1 % port cost — not a lost kernel.  Same-seed default run reproducible
(`extract-generated 01509ffcc688` twice); full `BETA-TESTING` gates still pending.

---

## 0. TL;DR (2026-09-20 snapshot)

1. **On the other solution's model the WIP is no longer 2x behind.** It is **within ~4% at matched `-ub 2048`** and **~9% behind at the other solution's best config (`-ub 16384`)**. The WIP took the uniform model from the delivery base's **734 t/s → 1149 t/s at pp8192/ub2048 (+57%)**; the other solution gets 1194. Our earlier "2x behind" figure was the mixed model measured against *its fast path not firing there*.

2. **The remaining gap is NOT MMB and NOT the QSA attention kernel.** The WIP already **beats** the other solution on `mmb_routed_glu` (−111 ms), the GDN recurrence (−250 ms), the F32 path (−239 ms vs its rocBLAS), the qsa3 sort (−86 ms) and the `mmb_cvt` bucket (−121 ms). The gap is concentrated in **three families the other solution fuses and we do not**:
   * the **hyper-connection (HC) prefill fusions** — `hc_combine_norm` + `hc_gate_mix` — worth **−19.5%** on its stack when disabled;
   * the **depthwise conv1d** (`gdn_conv_direct`/`ple_conv`) — worth **−10.5%**;
   * the **gated RMS-norm** (`norm-gated`/`rms_rows`) and the **indexer relu-sum** — worth −2.9% / −1.3%.

3. **Our delivery's `hc_combine_norm` (in `hyperconn.cu`) is present but never fires** — verified 0 calls with and without the WIP's HC16 gate. This is the single highest-value fix on the table: the other solution's equivalent fires 190× (554 ms) and it has an additional 408 ms `hc_gate_mix_kernel` we do not have at all. The missing gate-mix fusion also moves ~190 gate GEMMs *into* our `mmb_dense` (1266 launches vs its 956), which is most of the `mmb_dense` +809 ms delta.

4. **A separate, pre-existing delivery bug:** our tree (base r12 *and* the WIP) **cannot create a context at `n_batch == n_ubatch == n_ctx == 16384`** (`-b 16384 -ub 16384 -p 16384`), independently of MMB/HC16/QSA and of offload. the other solution's tree runs the same config at **1399 t/s**. This caps the useful ubatch and is why we have no pp16384/ub16384 point.

5. **Priority:** (a) make/fix the HC `combine_norm` + gate-mix fusion; (b) port the depthwise conv1d; (c) the `-ub 16384` context bug; (d) `norm-gated` + `idx-relu-sum`; (e) tune `qsa3_attn` and the `mmb_dense` tall tile. (a)+(b) are ~30% of end-to-end prefill on the other solution's numbers, which is exactly the "1300+" delta.

---

## 1. Why this investigation

`wip/mmb-general/` generalized the other solution's `mmb` weight GEMM, ported its `qsa3` attention and the bf16-producer machinery, and measured **+43–48%** over the delivery base on **our mixed UD-IQ4_XS** model. But the other solution's headline `1300+` numbers were on **its uniform-IQ4_NL** checkpoint. The open question was: *what still stands between us and those numbers?*

The previous gap analysis (README "Attribution", `archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-…`) said the gap was the QSA kernel and the weight GEMM; both were since ported. This session re-measured everything 1:1 on the actual the other solution GGUF, profiled both stacks, and priced the residual with the other solution's kill-switches.

---

## 2. Environment, builds, model

| | |
|---|---|
| WIP build | `~/llama-wip-mmb/build-rocm/bin/llama-bench`, tip **`90bf12997`** (38 commits / 5 thematic patches), base r12 applied tree `8a80535e…`, `LLAMA_QSA3_ENABLE=1` (compile-time) |
| delivery base | fresh worktree `/tmp/llama-r12-base` @ **`8568aaddb`** (block 15, r12 tree), built for this session (6 min with ccache) |
| the other solution's build | `~/pwilkin-llama-cpp/build-rocm/bin/llama-bench`, branch `strix-halo` @ **`f5daaa3cf`** |
| uniform model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93.16 GiB, 176.94 B params) |
| mixed model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (87.24 GiB) |
| the other solution env | `archive/work/wip-archive/iq4nl-prefill/launcher-env.txt` (its `install.sh` "optimized" set, verbatim) |

Rules observed: **page cache warmed** (`cat` all shards to `/dev/null`) before every run; **no parallel benches**; `-p … -n 0 -r 2`; `rocprofv3 --output-format csv` (the ROCm 7.14 rocpd/SQLite writer aborts without it); `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib`; `HIP_VISIBLE_DEVICES=0`.

The other solution runs are `-dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct`; WIP runs are `-ngl 99 -fa 1`. Both `-ctk f16 -ctv f16`. WIP all-on = `GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1`.

> **Run-to-run variance is real: ~±2–3%** on this box (the other solution pp8192/ub16384 measured 1320.9 and 1338.8 in the same session; WIP all-on measured 1191.7 and 1220.5). Treat single-digit differences as noise; the profile and the ablations are the reliable signals.

---

## 3. Throughput — the 1:1 comparison

### 3.1 Uniform IQ4_NL (the other solution's checkpoint)

| build | ubatch | pp2048 | pp8192 | pp16384 |
|---|---:|---:|---:|---:|
| **base r12** | 2048 | — | 733.6 | 737.1 |
| **base r12** | 16384 | — | 755.6 | **ctx failed** |
| **WIP all-on** | 2048 | 1181.5 | 1149.3 | 1129.2 |
| **WIP all-on** | 16384 | 1176.3 | 1220.5 | **ctx failed** |
| **the other solution full env** | 2048 | 1233.2 | 1194.2 | 1187.0 |
| **the other solution full env** | 16384 | 1233.0 | **1338.8** | **1399.3** |

* Base → WIP: **+56.7%** (pp8192/ub2048), **+61.5%** (pp8192/ub16384), **+53.2%** (pp16384/ub2048).
* WIP as % of the other solution: **96.2%** (ub2048 pp8192), **91.2%** (ub16384 pp8192), **95.1%** (ub2048 pp16384). The other solution's best config (ub16384) is the one we cannot fully run.

### 3.2 Mixed UD-IQ4_XS (our checkpoint)

| build | ubatch | pp2048 | pp8192 | pp16384 |
|---|---:|---:|---:|---:|
| **base r12** | 2048 | — | ~707* | 710.5 |
| **base r12** | 16384 | — | 747.5 | — |
| **WIP all-on** | 2048 | 1137.9 | 1110.3 | 1082.2 |
| **WIP all-on** | 16384 | 1139.0 | 1192.3 | **ctx failed** |
| **the other solution full env** | 2048 | 901.4 | 899.3 | 1046.9 |
| **the other solution full env** | 16384 | 904.0 | 1070.1 | 1130.8 |

\* from `benchmarks/2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md` (same box).

On the mixed model the WIP is **+23–26% over the other solution** at pp2048–8192/ub2048, because the other solution's `mmb_supported_mmid`/`_glu` predicates reject anything but **IQ4_NL**, so the model's **IQ3_S gate/up experts (36% of bytes, ~2/3 of the MoE FLOPs)** and its Q8_0 dense tensors fall back to its MMQ path. Ours accelerates them. At pp16384/ub16384 the other solution catches up (its 1130.8 vs our ub2048 1082.2).

**Conclusion:** the mixed-model comparison is *not* apples-to-apples in the direction the original "1.79x behind" implied. Each stack wins on the model its fast path was built for. The honest 1:1 is the uniform model, where the residual gap is ~4–9%.

---

## 4. The `-ub 16384` context-creation failure (delivery bug)

### 4.1 Symptom

```
llama_bench: error: failed to create context with model '…/Qwen3.8-Flash-Next-UD-IQ4_XS-…gguf'
```
`llama-bench` calls `llama_init_from_model` and gets `nullptr`. No underlying error is printed. It reproduces on **both** checkpoints, on the **base r12 build** as well as the WIP, and the other solution's tree runs the same config fine (its 1399.3 on uniform).

### 4.2 Bisect matrix (uniform model, `-p 16384`)

| `n_batch` | `n_ubatch` | `n_ctx` | result |
|---:|---:|---:|---|
| 16384 | 16384 | 16384 | **ctx failed** |
| 16384 | 8192 | 16384 | 1192 t/s ✓ |
| 8192 | 16384 | 16384 | 1166 t/s ✓ (ubatch clamped to batch) |
| 16384 | 4096 | 16384 | 1162 t/s ✓ |
| 16384 | 2048 | 16384 | 1119 t/s ✓ |
| 16384 | 16384 | 8192 | 1221 t/s ✓ (`-p 8192`) |

**Trigger:** the triple **`n_batch == n_ubatch == n_ctx` = 16384** — i.e. a *full-batch, full-context prefill graph*.

### 4.3 Ruled out

* **Not OOM / not model VRAM:** still fails with `-ngl 50` (half the model unoffloaded). `rocm-smi` shows ~0.6 GiB VRAM used during the failing init (the model is mmap'd/GTT).
* **Not the WIP:** the **base r12** build fails identically → pre-existing delivery bug.
* **Not MMB/HC16:** fails with `GGML_CUDA_MMB=0 GGML_CUDA_MMB_HC16=0`.
* **Not QSA:** fails with `LLAMA_QSA_OFF=1`, `LLAMA_QSA_DENSE_SHORTCUT=1`, and `LLAMA_QSA_SPARSE_FA=0`.
* **`llama-cli` with the identical cparams works** (`n_ctx=16384 n_batch=16384 n_ubatch=16384`), because its init `graph_reserve` builds for `n_tokens = 64`; `llama-bench`'s init reserves the full-ubatch graph and dies there.

### 4.4 Hypothesis

The compute-graph reserve for a 16384-token × 16384-KV graph allocates one or more very large tensors (an `[n_kv, n_tokens]` mask/bias class tensor is 1 GiB as F32, 512 MiB as F16; the packed kq mask is `n_kv*n_tokens*2`), or hits an allocator/shape limit. The failure is silent, so the next step is a debug build that prints the reserve failure (or `gdb` on `llama_init_from_model`) — **not yet done**. Whatever it is, it is independent of the WIP and it costs us the `-ub 16384` regime where the other solution is 9% ahead.

---

## 5. Kernel profile diff — uniform IQ4_NL, `-b/-ub 16384`, pp8192, r=1

Both runs profiled with `rocprofv3 --kernel-trace`. WIP grand kernel sum **13591 ms**; the other solution **11393 ms** (ratio 1.19). (Kernel-sum ratio > t/s ratio because the profiler captures the whole process; use the *family deltas*, not the absolute ratio.) Both traces confirmed the relevant fast paths were live: WIP `mmb_dense`/`mmb_routed_glu`/`mmb_routed` present, `qsa3_attn` present, `flash_attn_qsa` absent; the other solution `mmb_*`, `qsa3_attn`, `hc_combine_norm_f32_b256`, `hc_gate_mix_kernel`, `gdn_conv_direct_kernel` all present.

### 5.1 Family table (Δ = WIP − the other solution, ms)

| family | WIP ms | WIP % | the other solution ms | other % | **Δ(WIP−other)** |
|---|---:|---:|---:|---:|---:|
| `mmb_dense` | 4008.6 | 29.5 | 3199.4 | 28.1 | **+809.2** |
| `rms_norm` (all) | 1059.9 | 7.8 | 505.3 | 4.4 | **+554.5** |
| MoE concat+reduction | 744.8 | 5.5 | 278.6 | 2.5 | **+466.1** |
| `ssm_conv_long_token` (conv) | 303.0 | 2.2 | 7.0 | 0.1 | **+296.0** |
| `qsa3_attn` | 813.8 | 6.0 | 618.4 | 5.4 | **+195.4** |
| elementwise | 706.9 | 5.2 | 555.2 | 4.9 | +151.8 |
| copy | 243.4 | 1.8 | 129.0 | 1.1 | +114.4 |
| indexer | 156.0 | 1.2 | 63.6 | 0.6 | +92.4 |
| mmq/mmvq | 120.5 | 0.9 | 55.1 | 0.5 | +65.4 |
| `mmb_tiny_m` (F32) | 54.7 | 0.4 | 0.0 | 0.0 | +54.7 |
| `mmb_routed` | 979.0 | 7.2 | 944.3 | 8.3 | +34.7 |
| `mmb_f32split` | 295.9 | 2.2 | 273.0 | 2.4 | +23.0 |
| HC (dsv4/hc_*) | 1103.8 | 8.1 | 962.0 | 8.4 | +141.8 |
| `mmb_routed_glu` | 1863.6 | 13.7 | 1974.7 | 17.3 | **−111.0** |
| `mmb_other`/`mmb_cvt` | 1.1 | 0.0 | 122.0 | 1.1 | **−120.8** |
| rocBLAS | 0.0 | 0.0 | 239.4 | 2.1 | **−239.4** |
| GDN | 938.3 | 6.9 | 1188.2 | 10.4 | **−250.0** |

(Watch the bucketing: the other solution's `gdn_conv_direct_kernel` 250 ms landed in the GDN row, so the true conv comparison is our `ssm_conv_long_token_f32` 303 vs its `gdn_conv` 250 + `ple_conv` 7. And its GDN "1188" = `gated_delta_net_tiled` 936 + `gdn_conv_direct` 250; our pure recurrence is 938 — **parity**.)

### 5.2 The `mmb_dense` detail (raw instantiations)

| tile `WTYPE` | WIP ms / calls | the other solution ms / calls | Δ |
|---|---:|---:|---:|
| `<128,256,64,64,0>` | 1627.3 / 168 | 1735.7 / 168 | −108 (we win) |
| `<128,128,32,64,0>` | 1060.0 / 594 | 785.6 / 498 | +274 / **+96 calls** |
| `<384,64,96,32,0>` (tall) | 1004.6 / **380** | 589.6 / **190** | +415 / **+190 calls** |
| misc | 316.7 | 228.5 | +88 |
| **total** | **4008.6 / 1266** | **3199.4 / 956** | **+809 / +310 calls** |

Our dense MMB launches **1266** GEMMs vs its **956** (+310), and the tall `384x64` tile runs **twice** as often (380 vs 190). A large part of this is structural, not tile tuning: the other solution's **`hc_gate_mix_kernel`** (408 ms, 190 calls) fuses the HC gate GEMM + sigmoid + mix and *removes* ~190 dense GEMMs from its `mmb_dense`; we run those in `mmb_dense` and then do the mix separately in `dsv4_hc_pre/post`.

### 5.3 The RMS/HC detail

| | WIP | the other solution |
|---|---|---|
| `rms_norm_f32<1024,true>` | **622.2 ms / 196 calls** | — |
| `rms_norm_f32<256,true>` | 288.8 / 168 | 0.3 / 24 |
| `rms_norm_f32<256,false>` | 148.8 / 144 | 166.4 / 144 |
| `rms_norm_f32<1024,false>` | — | 35.9 / 10 |
| **`rms_rows_f32<true>`** | — | **220.0 / 72** |
| **`rms_rows_f32<false>`** | — | **82.7 / 72** |
| `dsv4_hc_post_f32<false>` | **739.0 / 188** | — |
| `dsv4_hc_pre_f32<true,true,true>` | 364.7 / 190 | — |
| **`hc_combine_norm_f32_b256`** | **0** | **554.3 / 190** |
| **`hc_gate_mix_kernel<4>`** | **0** | **407.7 / 190** |

`rms_rows_f32` is the other solution's fused **gated** RMS-norm (`LLAMA_NORM_GATED`/`LLAMA_NORM_ROWS`); the 622 ms `rms_norm_f32<1024,true>` is our HC normalized stream. It folds the HC combine + norm into `hc_combine_norm_f32_b256`, and it has a whole `hc_gate_mix` kernel we have no analogue of.

### 5.4 The MoE detail

| | WIP | the other solution |
|---|---|---|
| `concat_transposed_src1_dim0` | **375.3 / 74** | 0 |
| `moe_weighted_reduction_f32_vec4` | **369.5 / 96** | — |
| `moe_weighted_reduction_bf16_v4` | — | **213.3 / 94** |

The other solution's MoE epilogue reads **bf16** expert outputs (its `LLAMA_MMB_DOWN16` / `store_f32=0` routed-down) and avoids the `concat_transposed` materialisation entirely. We still materialise the concat and reduce in F32. (The WIP added non-temporal hints to these two kernels in session 14, but did not remove the concat or move to bf16 inputs.)

### 5.5 QSA

Same kernel name, same 24 calls, **813.8 vs 618.4 ms** — our `qsa3_attn_kernel` is 32% slower at identical work. Our `qsa3_rows`/`merge` are **faster** (64.6 vs 151.0). So the port's *sort/merge* is a win and the *attention body* is a regression, or its `qsa.cu` has an arch/tile difference the port did not carry.

---

## 6. What the missing pieces are worth — the other solution's ablations on this box

Run on the uniform model, `-b/-ub 16384`, `-p 8192`, r=2, source its `launcher-env.txt` and zero one family at a time. This is the cleanest "what is missing" price list, because it is the *same tree, same model, same box*.

| arm | pp8192 t/s | Δ vs full | % |
|---|---:|---:|---:|
| **FULL (baseline)** | 1320.9 | — | — |
| **NO `HC_*` (all 6)** | **1063.6** | **−257.3** | **−19.5%** |
| **NO `GDN_CONV`+`PLE_CONV`** | **1182.6** | **−138.3** | **−10.5%** |
| NO `NORM_GATED`+`NORM_ROWS` | 1282.1 | −38.8 | −2.9% |
| NO `IDX_RELU_SUM` | 1303.2 | −17.7 | −1.3% |
| NO `MMB_DOWN16` | 1321.8 | +0.9 | +0.1% (nil) |

The `HC_*` set zeroed is `HC_CN_SHAPE`, `HC_GATEMIX`, `HC_MIX_FUSE`, `HC_BLK16`, `HC_RES16`, `HC_PACK_DI`. The `GDN_CONV`/`PLE_CONV` ablation removes the *whole* direct-conv path (kernel + the concat/tail/reorder chain it replaces), so its 10.5% is more than the 257 ms of the two kernels themselves.

For cross-reference, the archived `iq4nl-prefill` Phase-1 ranking (pp16384, older build) measured: NO HC −289 (−21%), NO MMB −470 (−33%), NO QSA −696, NO CONV −62, NO NORM −30, NO IDX −10. The HC/NORM/IDX magnitudes reproduce; the fresh CONV number is larger because the `-ub 16384` single-shot prefill exposes the concat chain more.

---

## 7. WIP's own gate contributions on this model (for contrast)

Uniform IQ4_NL, `-b/-ub 16384`, pp8192, r=2:

| arm | pp8192 | Δ |
|---|---:|---:|
| WIP all-on | 1191.7 | — |
| WIP `HC16=0` | 1097.5 | HC16 bf16 producers **+8.6%** |
| WIP `MMB=0 HC16=0` | 856.2 | MMB **+28.2%** |
| WIP all-on `LLAMA_QSA_DENSE_SHORTCUT=1` | 1179.2 | always-QSA **+1.1%** |
| WIP all-on `LLAMA_QSA_OFF=1` | 1087.7 | QSA **+9.6%** |

So on the uniform model the WIP's MMB and bf16-producer work are doing exactly what they should. The gap is elsewhere.

---

## 8. Gap inventory (file + gate level, vs the other solution @ `f5daaa3cf`)

### 8.1 Ported / integrated (not the gap)

| the other solution work | status |
|---|---|
| `mmb.cu` dequant→bf16 WMMA weight GEMM | ported **and generalized** to 9 weight types (`wip/mmb-general/patches/0001`) |
| `qsa.cu` qsa3 rows/merge/attn | ported as `fattn-qsa3.cu` (`patches/0002`) |
| bf16-producer marking (`mark_bf16_only`, `out_xn_bf16`) | ported (`patches/0004`) |
| F32 split / tiny-M | ported/ours (`patches/0003`) |
| non-temporal hints | **ours** (its tree has zero) |
| fused indexer top-k | **ours** (`patches/0005`; its tree uses `top_k_nary_search_cuda`) |
| `dsv4_hc_pre`/`hc_mix_reduce` | in delivery block 14 / WIP |

### 8.2 Missing or inactive

| # | the other solution feature | its file / gate | our status | measured worth here |
|---|---|---|---|---|
| 1 | **HC gate-mix fusion** | `mmb.cu::hc_gate_mix_kernel`, `LLAMA_HC_GATEMIX` | **absent** | inside the −19.5% HC ablation |
| 2 | **HC combine+norm fusion (b256)** | `hc-cn.cu::hc_combine_norm_f32_b256` | delivery has `hyperconn.cu::hc_combine_norm_f32` (1024-thread) but it **never fires** (0 calls) | inside the −19.5% HC ablation |
| 3 | **depthwise conv1d, GDN + PLE** | `gdn-conv.cu`, `ple-conv.cu`; `LLAMA_GDN_CONV`/`LLAMA_PLE_CONV` | **absent** (we still build `concat`+transpose + `ssm_conv_long_token_f32`) | **−10.5%** |
| 4 | **gated RMS-norm** | `norm-gated.cu::rms_rows_f32`; `LLAMA_NORM_GATED`/`LLAMA_NORM_ROWS` | **absent** | −2.9% |
| 5 | **indexer relu-sum** | `idx-relu-sum.cu`; `LLAMA_IDX_RELU_SUM` | **absent** | −1.3% |
| 6 | **MoE bf16 epilogue / concat elimination** | `moe_weighted_reduction_bf16_v4` + `LLAMA_MMB_DOWN16` | F32 epilogue + concat still materialised | ~+466 ms kernel time |
| 7 | **`hc_combine_norm` b256 variant** | `hc-cn.cu` | only the 1024-block form exists | — |
| 8 | QSA graph-side options | `qwen4exp.cpp`: `QSA_WHOLE_ATTN`, `_BLOCK_SELECTION`, `_COMPACT_METADATA`, `_DIRECT_INDICES`, `_NO_DENSE_MASK`, `_QUERY_STRIP`, `_SCORE_BOUNDS`, `_SCORE_WMMA`, `_TOKEN_EMBD` | **audited 2026-09-22**: 7/9 present/superseded on our block-14/15 QSA; **2 un-ported** (`QSA_SCORE_BOUNDS`+`_QUERY_STRIP`, `QSA_SCORE_WMMA`) — [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md) | low-single-digit % (prefill score) |
| 9 | HC knobs `HC_CN_SHAPE`/`HC_MIX_FUSE`/`HC_BLK16`/`HC_RES16`/`HC_PACK_DI` | HC variants | partial (we have the bf16 `xn` stream but not the variants) | inside −19.5% |
| 10 | depthwise conv2d | `conv2d-dw.cu` | absent | 0 on these models |
| 11 | MTP-side QSA | `LLAMA_MTP_QSA`, `_MTP_QSA_MIN_T`, `LLAMA_MTP_EH_FLATTEN` | absent | **decode/MTP, not prefill** |
| 12 | host/loader | `LLAMA_PLE_PREFETCH`, `LLAMA_LOAD_LOCALS`, `--lazy-mode on-direct` | our analogue | load-time only |

---

## 9. Root-cause notes and hypotheses

### 9.1 `hc_combine_norm` does not fire — highest-value item

* The delivery's `ggml_cuda_op_hc_combine_norm` lives in `ggml/src/ggml-cuda/hyperconn.cu`; the graph-optimizer match is at `ggml-cuda.cu:5514` and `:5677` (two sites) and is a long `ok_a…ok_f` shape/type/alias predicate.
* `ggml_cuda_hc_combine_norm_supported` would accept this model (`n_embd=2560 ≤ HC_CN_MAX_EMB=3072`, `warp_size=32`, `hc≤16`), so the **supported** gate is not the blocker.
* Empirically it is **0 calls** on the uniform model with **both** `GGML_CUDA_MMB_HC16=1` and `=0`, and the fallback is `dsv4_hc_pre_f32<true,true,true>` + `rms_norm_f32<1024,true>` + `dsv4_hc_post_f32<false>`.
* Therefore the failure is in the *pattern match* (`ok_*`, `ggml_can_fuse_subgraph_ext`, alias `overlap`), i.e. our `qwen4exp` graph no longer presents the shape the delivery's matcher expects, **or** the matcher was only ever validated in the beta and has been dormant since. the other solution's equivalent fires 190× on the same model.
* **Action:** instrument the matcher (log which `ok_*` fails per layer), fix the pattern, or port the other solution's `hc-cn.cu` + `hc_gate_mix_kernel` directly. Then add `LLAMA_HC_GATEMIX`-equivalent: fused gate GEMM + sigmoid + mix, which also removes ~190 `mmb_dense` launches and the `rms_norm_f32<1024>` pass.

### 9.2 Depthwise conv1d

Our path: `build_conv_state`/concat + `ggml_ssm_conv` → `ssm_conv_long_token_f32` (303 ms), plus the surrounding `concat_cont`/`cpy_scalar`/transpose traffic. The other solution's `gdn_conv_direct_kernel` reads `state`+`x` directly and writes the conv output (+ optional silu), 250 ms, and `ple_conv_kernel` 7 ms, with **no concat tensor**. Porting `gdn-conv.cu`/`ple-conv.cu` (and their graph-optimizer match hooks, `*_match_at_concat`/`*_match_at_conv`/`*_match_at_tap` + `*_write_tail`/`*_direct`) is self-contained and worth ~10.5% end-to-end on the other solution's measure.

### 9.3 `mmb_dense` +809 ms / +310 launches — mostly structural, not tile tuning

The WIP already closed the tile-tuning question (session 6: every tile/BN/VDR knob is a wash or worse; the kernel is at 54% of bf16 peak). The delta is that the other solution **does fewer GEMMs**: `hc_gate_mix` absorbs ~190 gate GEMMs, and its tall tile runs 190× not 380×. Fixing item 1 should collapse most of this; the tall-tile 2x is worth a separate look (is the same A-panel dequantized/routed twice, or is our `mmb_tall` predicate applied to a tensor it handles with `<128,128>`?).

### 9.4 `qsa3_attn` +195 ms at identical launch counts

Our port is 32% slower on the body while our sort is faster. This is a kernel-shape/arch issue, not a graph issue. A/B the WIP `fattn-qsa3.cu` against the other solution's `qsa.cu` on this exact model (the WIP's own qsa3 measurements were on the mixed model / gfx1201 for some arms). Possible causes: the pack layout (`qsa_pack_keys/values` graph vs its `src[6]/src[7]`), the `G=4`/`umask` handling, or the `ncols2`/`Q->ne[1]` selection.

### 9.5 The `-ub 16384` context bug

Pre-existing delivery (base r12 fails, WIP fails, the other solution works), all WIP gates ruled out, not model VRAM. Blocks the pp16384/ub16384 point where the other solution is strongest. Needs a debug print/gdb on the reserve, then a fix in the base graph/allocator. It also means our ub16384 numbers above are only valid up to pp8192.

---

## 10. Recommended next steps (prioritized)

| # | action | expected | effort |
|---|---|---|---|
| 1 | Make `hc_combine_norm` fire (debug the matcher) **or** port `hc-cn.cu`; add the `hc_gate_mix` fusion | large — the `HC_*` ablation is **−19.5%** | 2–4 days; pattern debug may be hours |
| 2 | Port `gdn-conv.cu` + `ple-conv.cu` + their graph-optimizer matches | **−10.5%** (+ fewer concat/copy kernels) | 2–3 days |
| 3 | Fix the `n_batch==n_ubatch==n_ctx` context creation (unlocks `-ub 16384` and pp16384/ub16384) | access to the other solution's best regime | 0.5–2 days |
| 4 | Port `norm-gated.cu` (`rms_rows`) + `idx-relu-sum.cu` | −2.9% / −1.3% | 1–2 days |
| 5 | MoE: bf16 epilogue + drop `concat_transposed` (its `moe_weighted_reduction_bf16_v4`, `MMB_DOWN16`) | ~+466 ms kernel (~3–4%) | 1–2 days |
| 6 | Tune/port-align `qsa3_attn` body against `qsa.cu` | ~+195 ms (~1.5%) | 1–2 days |
| 7 | Investigate the tall `384x64` 2× launch count | unknown (part of +809) | 0.5–1 day |
| 8 | Audit the 9 QSA graph-side flags vs block-14/15 equivalents | low-single-digit % (2 un-ported) | 0.5 day | **DONE 2026-09-22 (session 4)**: 7/9 present/superseded; 2 un-ported prefill-score items — [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md) |

Items 1+2 alone are ~30% of end-to-end prefill on the other solution's ablations — comfortably the difference between our 1221 and 1300+.

---

## 11. Caveats and data provenance

* **Variance.** Single runs on this box swing ±2–3%; the the other solution baseline measured 1320.9 and 1338.8 in the same session. All family ablations share one session so their *relative* deltas are meaningful, but one or two points are within noise.
* **Kernel-sum ratio ≠ t/s ratio.** The profiles capture the whole process (including warm-up), so use the family deltas, not `13591/11393 = 1.19`.
* **Profiler caveat (`rocprofiler-register`, ROCm issue #10196).** Under `rocprofv3`, an env-gated path can read as *unset* (measured to flip `GGML_CUDA_QSA3` before it was made compile-time). I verified the fast paths were live from the kernel names in each trace (`mmb_*`, `qsa3_attn`, `hc_combine_norm_f32_b256`, `gdn_conv_direct_kernel` all present). The WIP's MMB/HC16 are still env-gated and could in principle flip; the family table is consistent with the un-profiled throughput, so it did not.
* **The `mmb_dense`/`rms`/HC kernels are *not* the same code in the two trees**, so their per-kernel times are not a pure A/B; the ablation (§6) is the authoritative price of the missing behaviour.
* **`-ub 16384` is required to reproduce the other solution's 1339/1399**; our ub16384 numbers only exist up to pp8192 because of the context bug.
* **Not done:** gdb/debug of the context-creation failure; a 1:1 audit of the QSA graph-side flags; a from-scratch attempt to make `hc_combine_norm` fire; and any actual port work.

---

## Appendix A — raw throughput (t/s, `llama-bench -n 0 -r 2`)

```
Uniform IQ4_NL, base r12:
  ub2048  pp8192 733.62 ± ?      pp16384 737.06 ± 5.73
  ub16384 pp8192 755.59 ± 3.98   pp16384 FAIL
Uniform IQ4_NL, WIP all-on:
  ub2048  pp2048 1181.46 ± 3.31  pp8192 1149.32 ± 4.25  pp16384 1129.16 ± 1.19
  ub16384 pp2048 1176.28 ± 4.01  pp8192 1220.52 ± 1.53  pp16384 FAIL
Uniform IQ4_NL, the other solution full env:
  ub2048  pp2048 1233.16 ± 39.79 pp8192 1194.24 ± 6.02  pp16384 1187.04 ± 0.49
  ub16384 pp2048 1233.03 ± 32.96 pp8192 1338.77 ± 37.03 pp16384 1399.34 ± 0.00

Mixed UD-IQ4_XS, base r12:
  ub16384 pp8192 747.48 ± 9.26   ; ub2048 pp16384 710.48 ± 7.75
Mixed UD-IQ4_XS, WIP all-on:
  ub2048  pp2048 1137.89 ± 1.59  pp8192 1110.32 ± 1.99  pp16384 1082.21 ± 0.35
  ub16384 pp2048 1139.03 ± 5.17  pp8192 1192.31 ± 1.11  pp16384 FAIL
Mixed UD-IQ4_XS, the other solution full env:
  ub2048  pp2048 901.36 ± 96.74  pp8192 899.30 ± 61.23  pp16384 1046.86 ± 1.18
  ub16384 pp2048 904.04 ± 91.36  pp8192 1070.10 ± 29.01  pp16384 1130.80 ± 5.31
```

## Appendix B — the other solution family ablations (uniform, `-b/-ub 16384`, pp8192, r=2)

```
FULL                     1320.88 ± 37.97
NO NORM_GATED+ROWS       1282.10 ± 31.20   -38.78  -2.94%
NO GDN_CONV+PLE_CONV     1182.58 ± 31.49   -138.30 -10.47%
NO IDX_RELU_SUM          1303.17 ± 41.83   -17.71  -1.34%
NO MMB_DOWN16            1321.76 ± 33.37   +0.88   +0.07%
NO HC_* (all 6)          1063.62 ± 30.46   -257.26 -19.48%
```

## Appendix C — WIP gate contributions (uniform, `-b/-ub 16384`, pp8192, r=2)

```
WIP all-on                1191.66 ± 4.29
WIP HC16=0                1097.46 ± 3.47
WIP MMB=0 HC16=0           856.21 ± 11.06
WIP all-on DENSE_SHORTCUT=1 1179.19 ± 6.23
WIP all-on QSA_OFF=1       1087.71 ± 25.75
```

## Appendix D — exact commands

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MM=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
W=/home/stew675/llama-wip-mmb/build-rocm/bin/llama-bench
BASE=/tmp/llama-r12-base/build-rocm/bin/llama-bench
OTHER=/home/stew675/pwilkin-llama-cpp/build-rocm/bin/llama-bench
ENV=/home/stew675/llama-cpp-rdna-boosts/archive/work/wip-archive/iq4nl-prefill/launcher-env.txt

# warm the page cache
for f in /llm/models/Qwen3.8/Flash-Next/IQ4_NL/*-0000*.gguf; do dd if=$f of=/dev/null bs=4M; done

# WIP all-on
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 $W -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 0 -r 2

# the other solution full env
( set -a; . $ENV; set +a; \
  $PW -m "$MU" -dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 0 -r 2 )

# the other solution ablation (e.g. HC)
( set -a; . $ENV; set +a; \
  LLAMA_HC_CN_SHAPE=0 LLAMA_HC_GATEMIX=0 LLAMA_HC_MIX_FUSE=0 LLAMA_HC_BLK16=0 LLAMA_HC_RES16=0 LLAMA_HC_PACK_DI=0 \
  $PW -m "$MU" -dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 8192 -n 0 -r 2 )

# profile (csv; the rocpd writer aborts on this ROCm)
rm -rf /tmp/prof && mkdir -p /tmp/prof
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 /opt/rocm-7.14-gfx1151/bin/rocprofv3 \
  --kernel-trace -f csv --output-format csv -d /tmp/prof -o k -- \
  $W -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 -b 16384 -ub 16384 -p 8192 -n 0 -r 1

# delivery base (built this session)
cd ~/llama.cpp && git worktree add --detach /tmp/llama-r12-base 8568aaddb
# then configure/build as in wip/mmb-general/HANDOVER.md §3
```

---

## 12. MTP qualification (2026-09-21): adaptive (ours) vs fixed (the other solution's)

This is the MTP half of the gap analysis, added because the other solution's newer commits are decode/MTP-heavy
and it is easy to read its MTP t/s as a gap. It is **not** the same axis as our advantage, and the
qualification below is what the 2026-09-21 plan asks for before either side is claimed.

### 12.0 Result (measured 2026-09-21 — see [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md))

Two findings, and one correction to the premise:

* **Our plain decode is ahead** of its on qwen4exp IQ4_NL (code 32.4 vs 31.1, prose 31.8 vs 29.9,
  recall 32.4 vs 31.9 t/s).  Absolute MTP t/s therefore flatters its stack; the fair metric is the
  **speedup over each tree's own plain decode**.
* **At fixed depth the MTP speedup is at parity** — ours `n3` **1.90x / 1.78x / 2.05x** vs its fixed
  **1.91x / 1.79x / 2.02x** (code / prose / recall).  It did **not** adopt our controller
  (`common/speculative-adaptive.h` is absent from its tree) and it is not ahead.
* **Our adaptive controller is mixed on qwen4exp** — the opposite of the 27B dense record.  It wins
  **recall** (2.34–2.40x) but over-drafts code and prose at `n_max 9..12` (code `adaptive 12`
  per-position acceptance falls 0.94 → 0.45 → 0.22 → 0.07); `adaptive 7` already beats `n3` on code
  (63.1 vs 61.4 t/s).  So the qwen4exp adaptive **ceiling is a tuning item**, not a structural gap.
* **The one real MTP gap was correctness/compat, not speed: `nextn_shared_target_tensors`.**
  **FIXED 2026-09-22 (session 7, `patches/0015`)**: the sidecar the other solution's IQ4_NL model ships
  is a *shared* MTP head; the MTP driver inferred KV sharing from `ctx_other` alone, took the gemma4
  arm, re-used one position for every draft token and died on the M-RoPE `X < Y` check on the second
  step.  Gating `is_mem_shared` on the `gemma4-assistant` arch fixes it (0 draft errors, acceptance
  0.287) — [`2026-09-22-mtp-shared-nextn-fix.md`](2026-09-22-mtp-shared-nextn-fix.md).  The comparison
  above used the non-shared `Q4_K_M` sidecar, which both trees run clean.

### 12.1 Structural standing

| | the other solution (`b0f31f587`) | ours (r12 + `beta/mmb-general`) |
|---|---|---|
| spec type | upstream **`draft-mtp` only** | `draft-mtp` **and** `draft-mtp-adaptive` |
| depth | **fixed** `--spec-draft-n-max` (default 3), capped at `n_mtp_layers` when chaining heads | adaptive controller picks the depth each round; `--spec-draft-n-start`, `n_min_adaptive`, clamp at 15 |
| cross-round feedback | none — only upstream's **within-round** `p_min`/`n_min` early stop | credit-bucket `common_speculative_adaptive` (delta = `n_accepted - depth`; full accept credits `max(1, n_accepted-1)`; surplus/deficit carried; `drop_pressure = max(60, 10*depth)`, `climb_budget = 20 + 6*(depth-1)`, cold start `cap-3`) |
| per-step cost | **new** sparse selected-cell decode (`qsa-decode.cuh` SIMT + `qsa-decode-wmma.cuh`) + incremental indexer key state (`d67d58836`): serial d40000 **25.85 → 28.82 t/s**, MTP 40680 **31.17 → 35.57** / **32.69 → 39.10** | our own QSA-sparse-FA decode + derived-block-vector cache; no dedicated selected-cell decode kernel for this model |

**The two optimise different things and compose.** Its `d67d58836` lowers the cost of each verify/draft
step; our block-01 controller decides *how deep* to draft. Median accepted length is the quantity the
controller moves and its kernels do not.

Delivery evidence for the controller (all at `-n 3000`, `benchmarks/mtp-adaptive-methodology.md` rule 0):
+13 % prose, +28 % code, +61 % recall vs fixed `n3` (`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`),
and +72 % recall for `draft-mtp-adaptive` + `ngram-mod` (`benchmarks/2026-09-17-mtp-ngram-combo.md`).
At `-n 256` the same controller *lost* to fixed `n3` (-5 % code) — the length rule matters here as much
as anywhere.

### 12.2 Hypothesis and falsification

**Hypothesis:** on the same model and workload our adaptive depth beats our fixed `n3` (and its fixed
`n3`) by a margin larger than its per-step decode gains, because the depth policy is the term the
per-step kernels do not touch.

**Falsifiers:**
- if `draft-mtp-adaptive` ≤ `draft-mtp --spec-draft-n-max 3` on the four axes at `-n 3000` on
  qwen4exp, the controller does **not** transfer to this model (a real finding — it would need a
  model-specific investigation);
- if its *absolute* MTP t/s exceeds ours by more than its per-step kernel advantage explains
  (measured as our fixed-`n3` vs its reported fixed-`draft-mtp`), our depth policy is not the whole
  story and the decode kernels are the gap after all.

### 12.3 Protocol — single-build A/B (the portable claim)

On our beta build (`~/llama.cpp/build-rocm`, r12 + 12 patches), **the other solution's uniform IQ4_NL model**,
gfx1151, `-ctk f16 -ctv f16`, seed 42 / temp 0, `-n 3000` (reasoning pinned: `on` for R, `off` for
P/C/K), per `benchmarks/mtp-adaptive-methodology.md`. Four arms per axis:

```sh
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
BIN=~/llama.cpp/build-rocm/bin/llama-cli
for axis in P C R K; do for arm in none fixed3 adaptive adaptive12; do
  case $axis in R) REA=on;; *) REA=off;; esac
  case $arm in none) SPEC="--spec-type none";;
                fixed3) SPEC="--spec-type draft-mtp --spec-draft-n-max 3";;
                adaptive) SPEC="--spec-type draft-mtp-adaptive";;
                adaptive12) SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 12";; esac
  # <prompt for $axis>, --log-verbosity 4 -> capture Generation t/s + acceptance + acc per pos
  $BIN -m "$MU" -ngl 99 -fa 1 --reasoning $REA $SPEC \
    --seed 42 --temp 0 --predict 3000 --single-turn --no-display-prompt \
    -p "$(cat prompts/<axis-prompt>.txt)" 2>&1 | tee /tmp/mtpq-${axis}-${arm}.log
done; done
```

Also run the **40680-token prompt** (its long case) for A1/A2/A3 only; record Generation t/s and mean
accepted length. `-n` is recorded with every number (rule 0).

### 12.4 Cross-build comparison — separate the axes

The other solution's 35.83 / 39.01 t/s include its per-step kernels, so our absolute numbers are expected to be
lower. Decompose, do not compare totals:

* **depth-policy delta** = `adaptive` − `fixed3` on *our* build (its kernels absent from both arms);
* **per-step-cost delta** = our `fixed3` vs its reported fixed-`draft-mtp` (same model, same depth) —
  this prices the sparse-decode + incremental-indexer gap in milliseconds per step;
* only the residual neither term explains is a genuine MTP gap.

### 12.5 Conclusion and the plan it implies

Measured, not predicted: our fixed-depth MTP is at parity with its, our plain decode is ahead, and our
adaptive controller is a clear win on recall and a tuning problem on code/prose for this model.  The
plan is therefore:

* **do not** treat its MTP as a speed gap;
* fold the other solution's per-step decode path in as **item 9** (sparse selected-cell decode + incremental
  indexer) — that is the term its absolute numbers get for free;
* add **`nextn_shared_target_tensors` support** as a correctness/compat item (it gates its own model's
  MTP head);
* park the qwen4exp adaptive **ceiling sweep** (3/5/7/9/12) until the MTP phase, per the maintainer's
  priority sequence.

### 12.6 What NOT to conclude

* **Do not** read its 39.10 t/s as "our adaptive MTP is 39 t/s behind" — it is measuring a fixed-depth
  stack plus its decode kernels on a different tree.
* **Do not** compare absolute MTP t/s without each build's own plain decode next to it.
* **Do not** compare at `-n 256`: our controller's warm-up transient inverts the ranking there.
* **Do not** use `none == draft-mtp` byte purity above `n_max 7` as the MTP gate; use acceptance and
  MTP-vs-plain throughput (rule 4).

---

## 13. Revised action plan — phased (2026-09-21)

**Maintainer's priority sequence (2026-09-21): recall speed + correctness → decode speed + correctness
→ MTP tuning + correctness.**  The 2026-09-20 §10 order was: (1) HC combine_norm/gate-mix, (2) depthwise
conv1d, (3) `-ub 16384` context bug, (4) norm-gated + idx-relu-sum, (5) MoE bf16 epilogue, (6) qsa3_attn
body, (7) tall tile, (8) QSA graph flags; items 1–9 survive, regrouped below.

### Phase 1 — recall (long-context prefill/attention) speed + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 1 | Make `hc_combine_norm` fire (debug the matcher) and **wire the existing `hc_gate_mix_kernel`** | large — `HC_*` ablation **−19.5 %** | 2–4 d | **DONE 2026-09-21**: matcher revived (+1.5 % prefill) and `hc_gate_mix` wired + default-on on gfx1151 (+1.2–1.5 % at pp8192/32768, width-pure, text-identical) — [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md), `patches/0003`. Follow-up: IQ4_NL-only kernel (mixed UD model unchanged) |
| 2 | Port `gdn-conv.cu` + `ple-conv.cu` + matches (now incl. **F32 PLE**) | **−10.5 %** | 2–3 d | **DONE 2026-09-21 (session 3)**: ported default-on, bit-identical, +3.0/+3.2 % qwen4exp IQ4_NL and +6.5/+7.1 % 35B-A3B (**measured at `-ub 8192` — the sign is robust, the magnitude carries the memory confound; re-measure at `-ub 4096` if a precise number is needed**) — [`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md), `patches/0004`.  Two adaptations (3-D `grouped_norm` root + the shared-builder snapshot cpy) |
| 3 | Fix the `n_batch==n_ubatch==n_ctx` context creation | unlocks `-ub 16384` | 0.5–2 d | pre-existing delivery bug |
| 3.5 | **Port the three correctness fixes** (`40c0b9c38`, `b0f31f587`, `14fff4f97`) | prevents long-session corruption | 0.5–1 d | **CLOSED 2026-09-22**: `b0f31f587` (QSA block window by highest stored position) **ported**, `patches/0005` — [`2026-09-22-qsa-block-window-fix.md`](2026-09-22-qsa-block-window-fix.md); the other two audited **N/A** (no maskless path; top-k output carries no sentinels) — [`2026-09-22-qsa-item-3.5-audit.md`](2026-09-22-qsa-item-3.5-audit.md) |
| 4 | Port `norm-gated.cu` (`rms_rows`) + `idx-relu-sum.cu` | −2.9 % / −1.3 % | 1–2 d | **`rms_rows` DONE 2026-09-22 (session 4)**: ported default-on, bit-identical, ~+0.3 % at `-ub 4096` — [`2026-09-22-norm-rows-fusion.md`](2026-09-22-norm-rows-fusion.md), `patches/0006`.  **`idx-relu-sum` is NOT banked (corrected 2026-09-22 session 5):** our fused indexer score is `n_tokens == 1` only, so prefill still runs `unary_op<relu>` 559 ms + head-sum adds — see the new item 14 |
| 5 | MoE: bf16 epilogue + drop `concat_transposed` | **+633 ms kernel** | 1–2 d | **MoE bf16 epilogue DONE 2026-09-22 (session 5, `patches/0010`), default OFF** via `GGML_CUDA_MMB_DOWN16=1` (lossy): the IQ4_NL routed-down GEMM output is marked bf16-only, the producer stores BF16 in place and `moe_weighted_reduction_bf16_v4` reads it — kernel 1479 -> 846 ms at pp32768 (matches the reference's 846.5), `plain == draft-mtp` and width probe PASS with it on.  The `concat_transposed` materialisation is already gone at `-ub 8192` |
| 6 | Tune/port-align `qsa3_attn` body vs `qsa.cu` | ~+195 ms (~1.5 %) | 1–2 d | **DONE 2026-09-22 (session 4)**: the gap was the per-cell `cell_vis` check, not geometry; folded into `umask` at merge time, bit-identical, `qsa3_attn` 809.6 -> 672.9 ms, +2.4 %/+1.7 % — [`2026-09-22-qsa3-visibility-fold.md`](2026-09-22-qsa3-visibility-fold.md), `patches/0007` |
| 7 | Investigate the tall `384x64` 2× launch count | unknown (part of +809) | 0.5–1 d | **DONE 2026-09-22 (session 4)**: the 2× was the M=4 HC inject admitted by the tall gate; a min-M bound keeps it on the dense tile, bit-identical, +0.8 %/+1.1 % — [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md), `patches/0008` |
| 8 | Audit the 9 QSA graph-side flags vs block-14/15 | low-single-digit % (2 un-ported) | 0.5 d | **DONE 2026-09-22 (session 4)**: 7/9 present/superseded; the 2 un-ported (`QSA_SCORE_BOUNDS`+`_QUERY_STRIP`, `QSA_SCORE_WMMA`) are now **item 15**, deferred behind the bigger families |
| 13 | **`-lzm auto` semantics + managed PLE reader perf** | memory: ~28 GB; `-ub 16384` unlock | 0.5–1 d (semantics **DONE**, reader **gated OFF**) | session-5: `on`=mmap, `off`=resident, `auto`=upstream auto, managed LRU **opt-in** via `LLAMA_LAZY_BUF_MB` and **off by default** because it is the slowest arm (1090/1184 vs mmap 1219/1217 vs resident 1285/1232 at pp8192/32768).  `--lazy-buffer-size` dropped.  **Discriminator (2026-09-22):** the cost is *not* only page-cache pressure — with the table fully cached (`-ub 2048`) the reader is still **−4.0 % vs mmap** (vs −9.7 % under pressure), so the arena has an intrinsic streaming overhead; the fix is a no-cache parallel-pread fast path like the reference's `on-direct`, then reconsider defaulting it on.  It already enables `-b/-ub 16384` (1125.5 t/s) |
| 14 | Port the prefill indexer **relu+head-sum** fusion (`idx-relu-sum`) | **DONE 2026-09-22 (session 6): +1.8 % pp32768 at `-b/-ub 4096`, bit-identical, default ON** | done | **DONE** — [`2026-09-22-idx-relu-sum.md`](2026-09-22-idx-relu-sum.md), `patches/0013`.  `ggml_cuda_match_idx_relu_sum` anchors at the RELU and accepts our L2a relu-before-the-4-D-reshape form (`idx_relu_sum_f32`), reading the block scores once instead of H times.  RDNA3_5-gated like the reference.  Kill switch `GGML_CUDA_DISABLE_IDX_RELU_SUM=1` |
| 15 | `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`, then `QSA_SCORE_WMMA` | low-single-digit % | 2–3 d | item-8 follow-ups; the bounds trim is coupled to the reference's complete-block selection (`compact`/`maskless`), which our fused cell top-k does not have, so scope carefully |
| 16 | **BF16 HC streams** (`blk16`/`res16`) | **DONE 2026-09-22 (session 6): +4.9 % pp8192 / +4.8 % pp32768 at `-b/-ub 4096`, default OFF** | done | **DONE** — [`2026-09-22-hc-bf16-streams.md`](2026-09-22-hc-bf16-streams.md), `patches/0011`.  BF16 arms in both combine+norm kernels, the structure-only combine identifier + blk16/res16 marking in `graph_optimize`, the MoE bf16-out reduction variants.  Default byte-identical.  `res16` is the dominant half (+4.7 %/+4.6 %); `blk16` alone +3.2 %/+2.5 %.  Limit: the qwen4exp `ffn_out` MoE-merge ADD is not adjacent to the reduction chain, so only the attention-path `block_out` takes `blk16` — **the reference's merge path is equally dormant on qwen4exp** (same builder), so this is not a gap against it; see the handover's "What is deliberately NOT being done" |

**Session-5 re-rank (2026-09-22, `-b 8192 -ub 8192 -p 32768` profile):** the remaining gap is the BF16
intermediate traffic — **HC combine (`blk16`/`res16`) + MoE epilogue ≈ 2.4 s, ~4.6 %** — plus the
`mmb_cvt` (closed session 6, `patches/0012`) and indexer relu-sum (item 14, the next item).  The
`HC_*` ablation's −19.5 % is therefore **not** closed by the item-1 fusions alone; the residual is the
BF16 streams.

Items 1+2 remain ~30 % of end-to-end prefill on the other solution's ablations.

### Phase 2 — decode speed + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 9 | **Port sparse QSA decode + incremental indexer state (`d67d58836`)** | **+11–20 % MTP/decode** | 2–4 d | this is the per-step term its absolute numbers get for free; audit vs our `GGML_CUDA_QSA_INDEXER_CACHE` (default on) first |
| 10 | MMB quant coverage: Q4_0/Q4_1/Q5_0/Q2_K/IQ1/IQ2/MXFP4/NVFP4 | completeness | 1–2 d | low priority for the delivery's models |

Our **plain decode is already ahead** of its (+2–6 % on qwen4exp, §12), so item 9 is a *hold/repay*
item, not a catch-up.

### Phase 3 — MTP tuning + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 12 | **`nextn_shared_target_tensors` support** | gates the other solution's IQ4_NL MTP head | 1–2 d | our build fails every draft position past the first (M-RoPE `X < Y`); see §12.0 |
| 11 | qwen4exp adaptive **ceiling sweep** (3/5/7/9/12) + a long-prompt run | recovers the recall win without over-drafting code/prose | 0.5–1 d | `adaptive 7` already beats `n3` on code; the 27B result does not transfer at `n_max 12` |

Item 11/12 are parked until Phase 1–2 land, per the maintainer's sequence.

