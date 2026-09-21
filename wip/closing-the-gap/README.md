# `wip/closing-the-gap/` — the prefill/decode gap to pwilkin's `strix-halo`

**Status: WIP planning record. Not part of the delivery.**  Started 2026-09-20 (analysis on gfx1151),
moved here and updated 2026-09-21.

## What is here

| file | what |
|---|---|
| [`closing-the-gap.md`](closing-the-gap.md) | the living analysis. §0–11 are the **2026-09-20 snapshot** (dated measurements); the **Update 2026-09-21** block, **§12** (MTP qualification) and **§13** (phased plan) are current. |
| [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) | the MTP qualification record: plain-vs-MTP on qwen4exp IQ4_NL, ours vs pwilkin's, and the `nextn_shared_target_tensors` finding. |

## The two moving references this file tracks

* **Ours:** `beta/mmb-general/` — **12 patches**, applied tree
  `bca69f23dd29acef2d8898c6fd492104e078eef1`, `git am` 12/12 on top of the r12 delivery
  (`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).  The body first analysed the pre-beta
  5-patch WIP (`90bf12997`, `~/llama-wip-mmb`).
* **pwilkin:** `~/pwilkin-llama-cpp`, branch `strix-halo`.  The body pinned `f5daaa3cf`
  (2026-09-12); the tip is **`b0f31f587`** (2026-09-16), 10 commits ahead.

## Current "our side" build state

* `~/llama.cpp` branch **`mmb-beta`** = `rdna-boosts` (r12, `72176ae8a`) + the 12
  `beta/mmb-general/patches/*.patch`.  Tree `bca69f23dd…`.
* Built on this box (gfx1151) with `~/bin/build-llama-rocm-714` for the beta window —
  see `beta/mmb-general/BETA-TESTING.md` for the four gates and the reference numbers.

To reproduce:

```sh
cd ~/llama.cpp
git checkout rdna-boosts && git branch -D mmb-beta 2>/dev/null
git checkout -b mmb-beta
git am /home/stew675/llama-cpp-rdna-boosts/beta/mmb-general/patches/*.patch
git rev-parse HEAD^{tree}        # bca69f23dd29acef2d8898c6fd492104e078eef1
~/bin/build-llama-rocm-714
```

## The current open list (see §13 of the doc)

**Priority sequence (maintainer, 2026-09-21): recall speed + correctness → decode speed + correctness →
MTP tuning + correctness.**

**Phase 1 — recall / long-context prefill + correctness**

1. Wire the existing `hc_gate_mix_kernel` + make `hc_combine_norm` fire (matcher) — the `HC_*`
   ablation is −19.5 % on pwilkin's own model.
2. Port `gdn-conv.cu` + `ple-conv.cu` (now F32-aware for Flash-Next PLE) — −10.5 %.
3. Fix the `n_batch == n_ubatch == n_ctx` context-creation bug (unlocks `-ub 16384`).
3.5. Port pwilkin's three correctness fixes (`40c0b9c38`, `b0f31f587`, `14fff4f97`).
4. `norm-gated.cu` + `idx-relu-sum.cu` — −2.9 % / −1.3 %.
5. MoE bf16 epilogue + drop `concat_transposed` (beta's `MMB_DOWN16` is gated off).

**Phase 2 — decode speed + correctness**

9. Port sparse QSA decode + incremental indexer state (`d67d58836`) — his +11–20 %; our plain decode is
   already ahead, so this is a hold/repay item.
10. MMB quant coverage (Q4_0/Q4_1/Q5_0/Q2_K/IQ1/IQ2/MXFP4/NVFP4) — completeness.

**Phase 3 — MTP tuning + correctness** (parked)

12. Add `nextn_shared_target_tensors` support — we currently cannot load the shared MTP sidecar
    pwilkin's own IQ4_NL model ships (draft decode fails on an M-RoPE `X < Y` check).
11. qwen4exp adaptive ceiling sweep (3/5/7/9/12) — fixed-depth MTP is at parity with his, adaptive wins
    recall but over-drafts code/prose at `n_max 12`.

## Cautions

* The body's §6/§9 kernel-level deltas are against `f5daaa3cf`; re-profile `b0f31f587` before trusting
  them (his tree gained the `mmb_quant` dispatcher and dropped env gating).
* pwilkin's `ac1ebb4e0` **compiled in** his tuned defaults and deleted the `LLAMA_*` experiment
  switches — the body's Appendix D ablation commands no longer work against current pwilkin HEAD.
* pwilkin has **no adaptive MTP controller** (fixed `n_max` + upstream `p_min`/`n_min` early stop);
  our `draft-mtp-adaptive` is a different axis from his per-step decode kernels.  The 2026-09-21
  measurement (see the qualification record) shows our plain decode ahead, fixed-depth MTP speedup at
  parity, and the only real MTP gap is `nextn_shared_target_tensors` — not velocity.
