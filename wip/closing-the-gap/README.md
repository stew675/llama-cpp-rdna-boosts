# `wip/closing-the-gap/` — the prefill/decode gap to the other solution's `strix-halo`

**Status: WIP planning record. Not part of the delivery.**  Started 2026-09-20 (analysis on gfx1151),
moved here and updated 2026-09-21.

## What is here

| file | what |
|---|---|
| [`closing-the-gap.md`](closing-the-gap.md) | the living analysis. §0–11 are the **2026-09-20 snapshot** (dated measurements); the **Update 2026-09-21** block, **§12** (MTP qualification) and **§13** (phased plan) are current. |
| [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) | the MTP qualification record: plain-vs-MTP on qwen4exp IQ4_NL, ours vs the other solution's, and the `nextn_shared_target_tensors` finding. |
| [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md) | Phase-1 item 1 start: the `hc_combine_norm` matcher root cause (three bugs) and the +1.5 % prefill prototype on fork branch `gap-closing`. |
| [`patches/`](patches/) | the fork `gap-closing` commits (`90f081550..94694a38e`) exported as patches, so the code work survives a fork reset. |

## The two moving references this file tracks

* **Ours:** `beta/mmb-general/` — **12 patches**, applied tree
  `bca69f23dd29acef2d8898c6fd492104e078eef1`, `git am` 12/12 on top of the r12 delivery
  (`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).  The body first analysed the pre-beta
  5-patch WIP (`90bf12997`, `~/llama-wip-mmb`).
* **The other solution:** `~/pwilkin-llama-cpp`, branch `strix-halo`.  The body pinned `f5daaa3cf`
  (2026-09-12); the tip is **`b0f31f587`** (2026-09-16), 10 commits ahead.

## Current "our side" build state

* `~/llama.cpp` branch **`gap-closing`** @ **`94694a38e`** = `mmb-beta` (r12 `72176ae8a` + the 12
  `beta/mmb-general/patches/*.patch`, tree `bca69f23dd…`) + the 2026-09-21 changes: **default-on policy**
  (MMB/HC16/matcher), the `hc_combine_norm` matcher revival, and env-gated debug traces.
* Built on this box (gfx1151) with `~/bin/build-llama-rocm-714`.  **All beneficial features are on by
  default** (see the `AGENTS.md` default-on policy); env vars only disable.
* The two `gap-closing` commits are exported to [`patches/`](patches/) in case the local fork branch is
  lost.

To reproduce:

```sh
cd ~/llama.cpp
git checkout rdna-boosts && git branch -D mmb-beta gap-closing 2>/dev/null
git checkout -b mmb-beta
git am /home/stew675/llama-cpp-rdna-boosts/beta/mmb-general/patches/*.patch
git am /home/stew675/llama-cpp-rdna-boosts/wip/closing-the-gap/patches/*.patch   # tip 94694a38e
~/bin/build-llama-rocm-714
```

## Do first (fresh session, in order)

1. **Run the full BETA-TESTING gate suite** on the current default build —
   [`beta/mmb-general/BETA-TESTING.md`](../../beta/mmb-general/BETA-TESTING.md).  Gate semantics changed:
   **Gate 1 = `GGML_CUDA_MMB=0`** (byte-identical to r12), **Gate 2 = the default** (no env).  Plus the
   width probe and the MTP gate.  Green before any promotion.
2. **Target `-b 8192 -ub 8192`** — decision 2026-09-21.  The `-ub 16384` failure is root-caused and
   **deferred** (see the “Update 2026-09-21 (later)” section of the doc): it is the full-vocab
   `result_output` reserve (15.5 GiB, shared with the other solution) plus qwen4exp's HC `block_out`
   pin (~18 GiB) against the resident PLE table (~27 GiB host).  ubatch 8192 runs clean and is the
   reproducible head-to-head baseline (ours 1212.6 vs its 1346.5 at pp8192, ~10 % behind).
3. Then resume the phased investigation below against that baseline.

## The current open list (see §13 of the doc)

**Priority sequence (maintainer, 2026-09-21): recall speed + correctness → decode speed + correctness →
MTP tuning + correctness.**

**Phase 1 — recall / long-context prefill + correctness**

1. Wire the existing `hc_gate_mix_kernel` + make `hc_combine_norm` fire (matcher) — the `HC_*`
   ablation is −19.5 % on the other solution's model. **Both halves done 2026-09-21**: the combine+norm
   matcher was revived (+1.5 % prefill) and `hc_gate_mix` is wired and default-on on gfx1151
   (+1.2–1.5 % at pp8192/32768, width-pure, text-identical) — see the record and `patches/0003`.
   Follow-up: the gate-mix kernel is IQ4_NL-only, so the mixed UD-IQ4_XS model is unchanged.
2. Port `gdn-conv.cu` + `ple-conv.cu` (now F32-aware for Flash-Next PLE) — −10.5 %.
3. **`-ub 16384` is deferred** (target is `-ub 8192`).  Root cause in the “Update 2026-09-21 (later)”
   section: result_output reserve + HC pin + resident PLE.  Candidate fixes: default the PLE to
   mmap-lazy (fix the `-lzm auto` propagation), or expose `--lazy-buffer-size` in `llama-bench`.
3.5. Port the other solution's three correctness fixes (`40c0b9c38`, `b0f31f587`, `14fff4f97`).
4. `norm-gated.cu` + `idx-relu-sum.cu` — −2.9 % / −1.3 %.
5. MoE bf16 epilogue + drop `concat_transposed` (beta's `MMB_DOWN16` is gated off).

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
