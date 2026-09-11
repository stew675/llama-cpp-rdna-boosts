# Block 00 validation on gfx1151 (Strix Halo) — issue #25 and the masked-V fixes

Date: 2026-09-11.  Machine: Strix Halo APU, Radeon 8060S (gfx1151), 1 device,
ROCm 7.14 (`/opt/rocm-7.14-gfx1151`).  Vulkan via RADV.

Validates the GFX1201 agent's `structural-fixes` branch (delivery-preview
`0000`+`0001-0014`) on RDNA3_5.  Root-cause record:
`../issue-25-mtp-batch-width/README.md`; delivery block:
`patches/0000-rdna-boosts-block-00-structural-and-architecture-fix.patch`.

## Setup

- Fresh worktree `/home/stew675/ll25/sf` at `9113cc188` +
  `scripts/apply-all.sh` -> **strict 15/15 `git am`**, applied tree
  `26690e4d9` — identical to the canonical tree recorded by the GFX1201 agent.
- HIP build `sf/build` (gfx1151, f16 KV, `-fa auto`) and Vulkan build
  `sf/build-vulkan` (RADV STRIX_HALO).
- Model: `Qwen3.8-27B-Q8_0.gguf` (exact reporter model; `qwen35` hybrid);
  `Qwen3.5-4B-Q8_0.gguf` for the gates; `gemma-4-12b-it-Q8_0.gguf` as a
  pure-attention control.

## 1. Issue #25 — FIXED on gfx1151

`--spec-draft-n-max 2` vs `4`, greedy seed 42, f16 KV, `-fa auto`; sha1 of
`reasoning_content+content`.  Baseline (b14, pre-block-00) vs `sf`:

| prompt | b14 n2 / n4 | sf (block 00) n2 / n4 |
|---|---|---|
| p0 hash map | `a39e9459` / `4b4b99b0` | `bba7741d` / `bba7741d` |
| p1 bash | `77306a04` / `6d6d8c11` | `77da9bb7` / `77da9bb7` |
| p2 TCP/QUIC | `2c476e43` / `3b1014ce` | `6bf38127` / `6bf38127` |
| p3 LRU | `e3d05db2` / `c4e194e7` | `85de49b7` / `85de49b7` |
| p4 sheep | `c7c88fa6` / `0697b28a` | `d2d943c9` / `d2d943c9` |

**5/5 identical** (baseline 5/5 different).  This is the reporter's exact
symptom, fixed on RDNA3_5.

### Logit-level confirmation (FA isolated)

The reporter's mechanism is real: the FA `parallel_blocks` KV split keyed off
`Q->ne[1]`.  It needs a long enough KV to show (`ntiles_KV`), so the probe must
use `P=1024`; `GGML_CUDA_GDN_CHUNKED=0` removes the separate GDN-chunked
effect.  MTP-faithful `n_rs_seq = W-1`, row 0, max |logit| diff:

| build | P=1024, GDN off: W1-W3 | W3-W5 |
|---|---|---|
| b14 (pre-block-00) | 0.327178 | 0.333723 |
| sf (block 00) | **0.000000** | **0.000000** |

So block 00 makes decode, and every verify width <= 8, bit-identical on
gfx1151 — the fork's `decode == verify` property restored for the FA path.

## 2. Residual (separate) issue — GDN chunked prefill

`--spec-type none` still differs from MTP, and this is **not** what issue #25
is about.  On `sf`, p0: `none = 9216c6d1`, `n2 = n4 = bba7741d`.  With
`GGML_CUDA_GDN_CHUNKED=0`: `none = n2 = n4 = bba7741d` — all three identical.

Mechanism (matches the GFX1201 isolation matrix): the plain prefill runs the
chunked GDN (`K=1`, `n_tokens>1`), while the spec prefill runs sequential GDN
(`K=n_max+1>1`, prefix branch needs `n_tokens > K+64`).  A probe artifact of the
same class: the MTP chunked-prefix boundary is `n_tokens-K`, so for prompts
`> K+64` the post-prefill state depends on `n_max` (`P=256, RS=from_w`:
W3-W5 = 0.210405 on b14 and unchanged on sf; `GDN_CHUNKED=0` -> 0).  A 2.8k-token
real-MTP n2/n4 test did **not** flip in 200 generated tokens (both arms
`2b0b6d6d`), so this is a latent, not observed, divergence.  Not addressed by
block 00; needs the maintainer's call (fix vs gate).

## 3. Vulkan masked-V / freed-cell fixes (Block 00) — VALIDATED

- Kernel probe matrix (`run-probe-matrix.sh vulkan`, `BUILD_DIR=sf/build-vulkan`):
  **f16 36/36 PASS, bf16 36/36 PASS** (was LEAK pre-fix).
- `test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT` vs CPU:
  **7845/7845 passed**.
- Server determinism gate (`run-gate.sh`), 4B Q8_0:
  **f16 PASS 16/16, bf16 PASS 16/16**.
- `LLAMA_KV_ZERO_FREED` / `zero_freed` are gone from the tree, so these are
  kernel-fix-only results (no host zeroing).

## 4. HIP masked-V fix (now in Block 03) — VALIDATED

- Probe matrix (`BUILD_DIR=sf/build`): **f16 36/36 PASS; bf16 34/36**, the 2
  failures being the previously-documented live-cell `LEAK-ROW` diagnostic
  (hsk256 2.8e-14, hsk128 1.1e-13, `det=0`, arm-independent, bf16-only) — same
  as the 2026-09-10 records.
- Server gate: **f16 PASS 16/16, bf16 PASS 16/16**.

## 5. Corrections to the earlier gfx1151 record

`RECORD-2026-09-11-issue25.md` (this directory) was written before the root
cause was known and is **superseded on two points**:

1. It claimed the reporter's "ncols 3 vs ncols 5 verify logits differ" does not
   hold.  That was an artifact of probing with `n_rs_seq=0` (K=1) and a KV too
   short to trigger `ntiles_KV`.  With the MTP-faithful `n_rs_seq=W-1` and a
   long KV, the verify widths **do** differ pre-fix — the reporter was right;
   the cause is the FA `parallel_blocks` split.
2. It attributed the residual text split to a "spec-side" defect.  The residual
   is the GDN chunked-prefill path (section 2); the n-max 2 vs 4 part is FA.

The earlier GDN/mmvq leads were real but are **different** issues: the GDN
chunked prefill (none-vs-spec) and the block-13 n_q=1 short-K mmvq variance
(pure-attention W1-W3).  Neither drives issue #25.
