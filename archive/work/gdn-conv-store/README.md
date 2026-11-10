# archive/work/gdn-conv-store (CLOSED, promoted in r244) — contributor PR #292

**Status: CLOSED 2026-11-09 — promoted into the delivery as a block-10 + block-12 amendment in
`v16-a691b2345-r244`.**  Nothing here is delivery work and nothing is open.  The code is in `patches/0010-*`
(the conv restore and the one-kernel conv step) and `patches/0012-*` (the layer-input copy order); the record
of *shipping* it is `WORKLOG.md` 261109.009, `INVARIANTS.md` r244, the `ENVIRONMENT.md` rows for
`GGML_CUDA_FUSE_GDN_CONV_STORE` and `LLAMA_LAYER_INP_EARLY`, and `MANIFESTS.md` blocks 10 and 12.  This file is
the campaign record: what the contributor offered, what we measured, what we changed, and what we deliberately
left out.

Contributor: **overdoingism** — <https://github.com/stew675/llama-cpp-rdna-boosts/pull/292>, three `git am`
files cut against **r241** (`wip/gdn-conv-31660-store/`), plus a README.  PR head at acceptance:
`01fa3867962080184e1956aaaf78ba2c42e9833c` (the second commit is a comments/README correction only — it renamed
the q/k norm reference to `rms_norm_scale_pair_f32<256>` and refreshed 0003's sha256).  The three patches apply
cleanly to the r243 tree as they stand.

Artifacts kept here:

| file | what | sha256 |
|---|---|---|
| `0001-gdn-conv-take-upstream-31660-s-rollback-store-the-4-.patch` | as submitted (`ggml/src/ggml-cuda/gdn-conv.cu`, +14) | `6706687edc311cba02932ba05e0ad106bf6f6a1ab14a2258cb6c0f7e67cf094f` |
| `0002-llama-graph-the-DFlash-layer-input-copies-right-afte.patch` | as submitted (`src/llama-graph.cpp`, +25) | `1bda5bc0f8337dd475a1384545c56051872ef46797861e7e64f7bfd688ad6380` |
| `0003-gdn-conv-store-a-decode-verify-GDN-conv-step-as-one-.patch` | as submitted (new `gdn-conv-store.cu`/`.cuh`, `ggml-cuda-try-fuse.cu` +6, `ggml-cuda.cu` +11) | `163d8929ecc793be20e26095c34ab4dab822dcc181a54ee81c5a587eabbaca86` |
| `0001-0003-final-cut.diff` | the **delivery's** form (`git diff 72f7753fd 443104a13 -- <the six files>`): the three commits plus the two adjustments below | `218cf588b815792612c13064576fc80d084215c89c261afecdba96557c7dcf06` |

## What it does

1. **`gdn_conv_check()` takes upstream #31660's rollback store again.**  `build_conv_state_store()` writes the
   snapshots with one 4-D window view of the concat (`ne = [n_cols, Kw, C, n_seqs]`, dim-1 stride one element)
   and one `SET_ROWS` that reads it; the matcher knew only the old one-CPY shape and rejected an unknown reader,
   so from the r236 re-base the direct conv declined on **every** GDN layer of every `n_rs_seq > 0` graph.  The
   window view's columns now join the materialized tail and the `SET_ROWS` is accepted as a reader of it (the
   snapshot fold stays off).  Kill switch: the existing `GGML_CUDA_DISABLE_CONV_FUSION`.
2. **The DFlash layer-input copies move next to each layer.**  Upstream #31506 appends each `SET_ROWS` after the
   whole graph; the copy's nodes are rotated to just after the next layer's input node when that is a node of its
   own and comes after the copy's source.  `LLAMA_LAYER_INP_EARLY=0` restores upstream's order.
3. **The decode / verify GDN conv step as one kernel** on that store (1..16 tokens, one sequence, RDNA4): one
   launch per layer writes the conv+silu row, the normalized q / k and the `Kw` snapshot windows.
   `GGML_CUDA_FUSE_GDN_CONV_STORE=0`.

## What we measured (1 × R9700 gfx1201, ROCm 10.1, `ROCBLAS_USE_HIPBLASLT=0`)

- **The r241 record was wrong about the scope of the decline**, and that is the biggest single finding: the
  direct conv declined on every `#31660`-store graph, single sequence included, not "on a multi-sequence
  ubatch".  At r243 the census is empty for a `draft-mtp n3` run and the run dispatches `CONCAT` 456 times; with
  1 it prints `GDN_CONV_DIRECT ... snap=0` and dispatches `CONCAT` 24 times.  `INVARIANTS.md` r241 is corrected.
- Prefill **+0.87 %** (dense 27B + DFlash2, 5246 tokens, 1260.4 → 1271.4 t/s) and **+1.0 %** (MTP); server arm
  +0.94 %.  The one-kernel step is **−1.22 %** of the per-verify-forward time (the submitter's own metric,
  `predicted_ms / (predicted_n − draft_n_accepted)`, six interleaved server arms) and +1.3 % end-to-end.
  Launch census per 32-token MTP run: `SET_ROWS` 432 → 48, deferred state `GET_ROWS` 568 → 184.
- The layer-input order: compute buffer **756.99 → 580.99 MiB** (server `-np 2 --kv-unified`) and
  **756.97 → 624.99 MiB** (`llama-cli -np 1`), steady-state VRAM **−185 MB**.  The PR's README says "~582" and
  "−176 MiB"; both reproduce for the server config it measured, and the CLI graph settles at 625 MiB.
- Bit-identity: prefill-logit KLD **−0.000000 / 99.990 %** in the store path (`GGML_FORCE_N_RS_SEQ=3`, base
  recorded from r243); width probe `P=2048/3584/4096` on 27B and 35B-A3B, 1 GPU and 3-GPU `-sm tensor` /
  `-sm layer`, identical to r243; text and draft acceptance identical in every arm; Protocol A acceptance
  **0.50588** (dense) and **0.73899** (MoE) identical to r243.

## The two adjustments made on acceptance

1. **`GGML_CUDA_FUSE_GDN_CONV_STORE` now stands down with `GGML_CUDA_DISABLE_CONV_FUSION`.**  The new kernel
   replaces the same conv step as the direct conv but honoured only its own switch, so the documented
   "chain" A/B arm (and the reference arm of gate 10's width probe) still ran it.  That half-off state was
   **numerically wrong**: `test-logits-width-probe P=3584` read `W=1 maxdiff_vs_W8 5.88239`, `width_purity=FAIL`,
   where the previous release, the default arm and the both-switches-off arm all read maxdiff 0 and the same
   W8/row-0 hashes.  With `gdn_store_enabled()` reading both switches the arm is the chain again.
2. **A census.**  `GGML_CUDA_CONV_DEBUG=1` now prints `GDN_CONV_STORE C=.. T=.. Kw=.. col0=.. K=.. private_chain=..`
   once per process, the way the direct conv prints `GDN_CONV_DIRECT`.  Without it a run of this path cannot be
   told from the chain, i.e. the gate would be vacuous.

## Not taken / not validated

- **Multi-sequence ubatches still decline** (`st->ne[2] == 1`), and `ggml_cuda_ssm_conv_in_try_fuse` still does
  not take the store graph: an `-np 2` server whose ubatch really carries two sequences runs the chain.
- **The one-kernel step is RDNA4-only** (`GGML_CUDA_CC_IS_RDNA4`): gfx1100/gfx1151 get the conv restore and the
  layer-input copies, not the step.  Per the scope policy that is a documented limitation, not a blocker.
- **The step's `#pragma clang fp contract(off)` is clang-only.**  An NVRTC/nvcc build would ignore it, which is
  one more reason the kernel's reachability is gated on the AMD cc; no NVIDIA arm was built or run (non-AMD
  behaviour is out of scope).
- Windows, `GGML_CUDA_DISABLE_GRAPHS=1`, RDNA3 and the other GDN arches (qwen3next, qwen4exp) were not exercised
  beyond the width-probe/arch gating above; the plan's `-ncmoe` / host-expert arms were not re-run (the change
  does not touch that path, and the dense 1-GPU and 3-GPU splits were gated).
