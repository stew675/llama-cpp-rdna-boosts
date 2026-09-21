# gfx1100 port — S9 addendum: deferred MMB tuning + the `nwarps` MoE candidate (2026-09-21)

The deferred optional work the maintainer asked for after S9: the MMB routed/GLU threshold + DBUF
sweep, and the `mmvq` `nwarps` MoE candidate.  Raw evidence; the plan is the source of truth.

**Verdict:** **no code changes.**  The gfx1151-tuned MMB thresholds transfer (32 is optimal), DBUF is
refuted/not-viable (gfx1100's 64 KB LDS cannot hold the big tile's double-buffered A panel), the
IQ3_XXS GLU arm is a wash, and the `nwarps` MoE candidate is **real but cannot be expressed with the
available dispatch axes** — it needs a shape/model discriminator threaded into the *decode* path,
which is a focused refactor plus a verify-width/purity sweep.  All data below.

## Part A — MMB routed/GLU/DBUF/IQ3_XXS

### A.1 Routed + GLU threshold sweep (`GGML_CUDA_MMB_{ROUTED,GLU}_THRESH`, 35B-A3B, `r=5`)

The threshold picks the small (BN=32) expert tile for low-row experts; the default is 32
(gfx1151-tuned).

| thresh | pp8192 | pp32768 |
|---|---:|---:|
| 8 | 3847.18 | 3097.23 |
| **32 (default)** | **3870.02** | **3145.90** |
| 128 | 3718.98 | 3030.76 |

**Default 32 is optimal** (128 is −3.9 %/−3.7 %; 8 is −0.6 %/−1.5 %).  No change.

### A.2 DBUF

* **Big dense tile (`mmb_dense_kernel<128,256,64,64,…,true>`) does not compile**: HIP reports
  `local memory (73728) exceeds limit (65536)` — the double-buffered A panel needs 72 KB but gfx1100
  has 64 KB LDS per workgroup.  This is why **no dispatch passes `DBUF=true`** on any gfx11/gfx12
  arch.  (Small `128,128` tiles would fit at 54 KB, but they are the minority path.)
* **Routed big tile (`mmb_routed_kernel<128,128,32,64,…,true>`, 54 KB) fits.**  A build with DBUF on
  all 10 big routed tiles was A/B'd on the 35B (`r=5`, interleaved):

  | | pp8192 | pp32768 |
  |---|---:|---:|
  | DBUF off (current) | 3909.77 / 3865.74 | 3154.18 / 3139.87 |
  | DBUF on | 3860.64 / 3860.06 | 3132.57 / 3123.29 |

  → **neutral-to-slightly-negative** (−1.3 %/−0.1 % at pp8192; −0.7 %/−0.5 % at pp32768).  **Refuted**
  on gfx1100, consistent with the in-source "DBUF2 MEASURED REFUTED" and the GLU DBUF "+5.8 % loss"
  notes.  No change.

### A.3 IQ3_XXS fused-GLU arm (`GGML_CUDA_MMB_IQ3XXS`, default 0; 35B, `r=5`)

| | pp8192 | pp32768 |
|---|---:|---:|
| off (default) | 3884.25 / 3880.40 | 3141.09 / 3138.08 |
| on | 3878.89 / 3879.79 | 3145.76 / 3142.93 |

A wash (the 35B "True-Q3_K_M" has no meaningful IQ3_XXS GLU work).  **Keep default off.**

## Part B — the `nwarps` MoE candidate: investigated, not landed

**The win is real** (S9 `gfx1100-s9-mmvq-results.md`): on RDNA3_0, `nwarps=1` for all wide types
versus the current per-type table gives **35B-A3B +2 % decode / +4.5 % draft-mtp**, gemma-26B +4.7 %,
while costing gemma-12B dense Q8_0 −1.9 % (and tying the 27B dense).  The obstacle is the
discriminator.

### B.1 The shapes do not separate by `(type, K)` — or even `M` alone

Weight shapes from `GGML_CUDA_MMB_LOG=1` (the same weight dims the decode GEMMs use):

| model | Q8_0 dense shapes (M×K) |
|---|---|
| gemma-12B dense | 512×3840, 2048×3840, 4096×3840, 8192×3840, 3840×4096, 3840×8192, 3840×15360, 15360×3840 |
| 35B-A3B MoE | 512×2048, 2048×512, 512×2048, 2048×4096, 512×2048 |
| 27B dense | (large-M per ssm_out/attn; a tie either way) |

* `K` cannot separate them: gemma's smallest Q8_0 `K` is 3840 and the 35B has `K` = 512/2048/4096 —
  but the 35B's `ssm_out` is `K`=4096, *above* gemma's smallest, so any `K` cut puts one on the wrong
  side.
* `M` alone cannot separate them either: both have `M`=512 and `M`=2048 shapes (gemma `attn_k`
  M=512/2048, 35B `attn_k/v` M=512, `ssm_out`/`down_shexp` M=2048).
* The *dominant* decode work differs (gemma is dominated by the large-M ffn shapes, the MoE by its
  expert/routed kernels), so an **`M >= 4096` rule is the plausible hypothesis** — it would keep
  gemma's large-M Q8_0 at 8 and give the MoE's M≤2048 shapes 1 — but it is untested and would still
  have to be proven across the 27B/gemma-26B, not assumed.

### B.2 Why it is not a small change: the decode path does not use the `(type,K)` rule

The `mmvq` dispatch has **two** dense kernels and they differ in how `nwarps` is chosen:

* **decode (item-split `mul_mat_vec_q`, `ncols_dst == 1`):** the kernel body uses
  `calc_nwarps(type, ncols_dst, table_id)` — the plain per-type table.  This is the path that
  improved on the MoE, and it has **no shape axis** today.
* **verify (ksplit `mul_mat_vec_q_ksplit`, `ncols_dst 2..8`):** the body uses
  `calc_nwarps_weight(type, ncols_dst, table_id, long_k)`, the `(type,K)` rule that RDNA4's block-13
  amendment added — but for RDNA3_0 it falls straight through to the table, and it only reaches the
  verify body, not the decode body.

So the RDNA4 `Q8_0 && !long_k → 8` rule is **not reusable** for gfx1100 (its direction is wrong for
the MoE anyway), and adding an M axis means editing the item-split dispatch (host `calc_launch_params`
construction + the kernel's `__launch_bounds__` + the body's `calc_nwarps`), i.e. a new template
dimension in the hottest decode kernel, with a **band-uniformity/purity sweep** (invariant 2:
`W = 1..8` must agree) and a re-run of the verify-width gate.

**Decision:** **do not land it now.**  It is a genuine +4.5 % MTP / +2 % decode MoE prize, but it is a
focused kernel-dispatch change with a purity contract, not a safe end-of-session edit.  It is recorded
here with the shape data and the `M >= 4096` hypothesis for a dedicated session (and it should be
implemented for the *decode* path, keeping the verify path band-uniform).

## Reproduce

* Threshold/IQ3_XXS: env vars, no rebuild (`GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1
  GGML_CUDA_MMB_ROUTED_THRESH=<n> GGML_CUDA_MMB_GLU_THRESH=<n>`).
* DBUF: `sed` `, true>` into the `mmb_routed_kernel<128, 128, 32, 64, N>` dispatch templates in
  `mmb.cu` (the big dense `128,256` arm will fail to compile — expected).  Source reverted.
* `nwarps`: the variant table edits from `gfx1100-s9-mmvq-results.md`.

The worktree is clean (`git status` empty), tree `064a2ad65ebb970e7302d74ea5a285e6a2dd5352`, current
build is the unmodified tree.
