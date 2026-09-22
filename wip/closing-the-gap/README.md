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
| [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md) | Phase-1 item 8 (**closed**): 7/9 QSA graph flags are present/superseded in our block-14/15 QSA; **2 are un-ported prefill-score optimizations** (`QSA_SCORE_BOUNDS`+`QSA_QUERY_STRIP`, `QSA_SCORE_WMMA`) — the next follow-ups. |
| [`patches/`](patches/) | the fork `gap-closing` commits (`90f081550..1004c65db`) exported as patches, so the code work survives a fork reset. |

## The two moving references this file tracks

* **Ours:** `beta/mmb-general/` — **12 patches**, applied tree
  `bca69f23dd29acef2d8898c6fd492104e078eef1`, `git am` 12/12 on top of the r12 delivery
  (`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).  The body first analysed the pre-beta
  5-patch WIP (`90bf12997`, `~/llama-wip-mmb`).
* **The other solution:** `~/pwilkin-llama-cpp`, branch `strix-halo`.  The body pinned `f5daaa3cf`
  (2026-09-12); the tip is **`b0f31f587`** (2026-09-16), 10 commits ahead.

## Current "our side" build state

* `~/llama.cpp` branch **`gap-closing`** @ **`4a75744fa`** = `mmb-beta` (r12 `72176ae8a` + the 12
  `beta/mmb-general/patches/*.patch`, tree `bca69f23dd…`) + the 2026-09-21/22 changes: **default-on
  policy** (MMB/HC16/matcher), the `hc_combine_norm` matcher revival, the **`hc_gate_mix` fusion**
  (session 2), the **depthwise conv1d fusions** (session 3), the **QSA block-window fix** (session 3,
  `b0f31f587`), the **narrow-row RMS norm fusion** (session 4), the **QSA visibility fold**
  (session 4, item 6), and the **tall-tile min-M** fix (session 4, item 7), plus env-gated debug traces.
* Built on this box (gfx1151) with `~/bin/build-llama-rocm-714`.  **All beneficial features are on by
  default** (see the `AGENTS.md` default-on policy); env vars only disable.
* The ten `gap-closing` commits (`0001..0010`) are exported to [`patches/`](patches/) in case the local
  fork branch is lost.

To reproduce:

```sh
cd ~/llama.cpp
git checkout rdna-boosts && git branch -D mmb-beta gap-closing 2>/dev/null
git checkout -b gap-closing
git am /home/stew675/llama-cpp-rdna-boosts/beta/mmb-general/patches/*.patch
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/*.patch   # tip 4a75744fa
~/bin/build-llama-rocm-714
```

## Do first (fresh session, in order)

1. **Run the full BETA-TESTING gate suite** on the current default build —
   [`beta/mmb-general/BETA-TESTING.md`](../../beta/mmb-general/BETA-TESTING.md).  **Purity is an
   intra-build contract**, not cross-build: the decode/verify band `W=1..8` must agree with itself
   (`plain == draft-mtp` greedy text) and `test-logits-width-probe` must print
   `width_purity=PASS (worst maxdiff 0)`.  **Do NOT gate on `MMB=0 == r12`** (or on any MMB on/off
   equality) — that was the beta's opt-in-era bisection aid, and a prefill re-baseline legitimately
   changes the greedy text; see the 2026-09-22 correction record in
   [`closing-the-gap.md`](closing-the-gap.md#correction-2026-09-22-session-5--the-mmb0--r12-gate-is-retracted).
   Plus the MTP gate (Gate 4).  Green before any promotion.
2. **Target `-b 8192 -ub 8192`** — decision 2026-09-21.  For **long-context (pp65536+)** use
   **`-b/-ub 4096`**: at `-ub 8192` that point memory-thrashes (GPU oscillating, ~844 t/s), while
   `-ub 4096` stays pegged at 100 % (~1093 t/s) — maintainer, 2026-09-22.  The `-ub 16384` failure is
   root-caused and **deferred** (see the “Session-2 record” section): the full-vocab `result_output`
   reserve (15.5 GiB, shared with the other solution) plus qwen4exp's HC `block_out` pin (~18 GiB)
   against the resident PLE table (~27 GiB host).  ubatch 8192 runs clean and is the reproducible
   head-to-head baseline (ours 1212.6 vs its 1346.5 at pp8192, ~10 % behind before items 1+2).
3. **Next code items, in this order** (session-5 profile, 2026-09-22):
   1. **`-lzm auto` semantics + managed PLE reader** — **DONE 2026-09-22 (partial)**: `on`=mmap,
      `off`=resident, `auto`=upstream auto, `--lazy-buffer-size` dropped, managed LRU reader
      **opt-in via `LLAMA_LAZY_BUF_MB`** and **OFF by default** (it measured slowest: 1090/1184 vs mmap
      1219/1217 vs resident 1285/1232).  It does unlock `-b/-ub 16384` (1118.7 t/s).  Making it beat
      mmap is the remaining work (item 13).
   2. **BF16 HC + MoE streams** (`blk16`/`res16`, `MMB_DOWN16`) — ~2.4 s, ~+4.5 % at depth, but lossy
      (greedy text changes) → **maintainer's call**.
   3. **`mmb_cvt_f32_bf16` (+1478 ms)** — non-lossy; our calls convert far larger tensors than the
      reference's (activation cache / `mmb_root` keying).
   4. **Prefill indexer relu-sum (+590 ms)** — non-lossy; the audit wrongly marked `idx-relu-sum` as
      banked (our fused score op is `n_tokens == 1` only).
   5. `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`, then `QSA_SCORE_WMMA` — the item-8 follow-ups, now below
      the bigger families.
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
14. Port the prefill indexer **relu+head-sum** fusion (`idx-relu-sum`, ~+590 ms, non-lossy).  Our graph
    applies relu *before* the 4-D reshape (the L2a win), so the reference matcher cannot port verbatim.
15. `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`, then `QSA_SCORE_WMMA` — the item-8 follow-ups; the trim is
    coupled to the reference's complete-block selection, which our fused cell top-k lacks.
16. **BF16 HC streams** (`blk16`/`res16`) — **~1.8 s, ~3.8 % at depth, the biggest remaining item;
    NOT STARTED (scoped 2026-09-22 session 5)**, default OFF like item 5.  Consumer arms +
    `ggml_cuda_hc_combine_norm_args` fields + the ~120-line HC stream marking block.

**Phase 2 — decode speed + correctness**

9. Port sparse QSA decode + incremental indexer state (`d67d58836`) — its +11–20 %; our plain decode is
   already ahead, so this is a hold/repay item.
10. MMB quant coverage (Q4_0/Q4_1/Q5_0/Q2_K/IQ1/IQ2/MXFP4/NVFP4) — completeness.

**Phase 3 — MTP tuning + correctness** (parked)

12. Add `nextn_shared_target_tensors` support — we currently cannot load the shared MTP sidecar
    the other solution's IQ4_NL model ships (draft decode fails on an M-RoPE `X < Y` check).
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
