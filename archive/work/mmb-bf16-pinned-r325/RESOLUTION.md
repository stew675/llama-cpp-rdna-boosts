# r336 resolution — PR #340 (`mmb-bf16-pinned-r325`): taken, folded into block 06

**Status: PROMOTED at r336** (`v17-919794500-r336`).  This directory holds the contributor's six patches,
his `README.md` and `logs/ab-r325.log` **verbatim** (`0001`-`0006`, `README.md`, `logs/`), plus this
resolution.  Nothing here is applied by `scripts/apply-all.sh`; the delivered form is block 06's patch.

## What was taken

All six patches, squashed into **block 06** (which owns `mmb.cu`):

| contributor patch | what it does | reachable when |
|---|---|---|
| `0001` | BF16 dense weights take the MMB bf16-WMMA tile from 33 tokens (`mmb_bf16_dense`, `mmb_hipblaslt_off()`, `GGML_CUDA_MMB_BF16_DENSE` / `_BF16_MIN_T`) | `ROCBLAS_USE_HIPBLASLT=0` only |
| `0002` | RDNA4 stops writing the hc gate's BF16 side copy (`hc16` off on RDNA4) | **always** (default-path win) |
| `0003` | tiny-M warp-per-token kernel for BF16 weights with `M <= 8`, dispatch factored into `mmb_tiny_m_launch` | pin |
| `0004` | that kernel reads the activation's bf16 copy (`GGML_CUDA_MMB_TINY_BF16_XH`) | pin |
| `0005` | takes the producer-written RNE bf16 activation from `bf16_src1_cache` (`GGML_CUDA_MMB_PRODUCER_BF16`) | pin |
| `0006` | 128x64 / 64x32 dense tiles when the 256x128 grid is thin (`GGML_CUDA_MMB_SMALL_TILE` / `_SMALL_GRID`) | pin |

## What was verified here (not taken on trust)

gfx1201 / ROCm 10.1, 2-GPU `-sm tensor`, Flash-Next GSQ IQ3_S with host experts — full numbers in
`benchmarks/2026-11-21-mmb-bf16-pinned.md`:

* census `0 / 80 (cap) / 0` for default / pinned / `BF16_DENSE=0` — the gate and the kill switch are real;
* pinned prefill **1691 -> 2196 t/s (+29.9 %)**, above the default arm's 2027; decode flat;
* default path **bit-identical** (`-0.000000 / 100.000 %`, identical greedy text on two models);
* `SMALL_TILE` / `PRODUCER_BF16` / `HC16` A/Bs **0.000000 / 100.000 %** (the contributor's bit-identical claims
  hold); `TINY_BF16_XH` moves numerics (0.014143 / 96.402 %) — which he did not claim, and which is why that
  one is documented as a numerics change;
* `test-logits-width-probe` on the affected model, pinned: `width_purity=PASS`, every W=1..8 hash identical to
  the base — the change is **prefill-only, demonstrated**;
* `test-backend-ops -o MUL_MAT` pinned 2691/2692 (the pre-existing `f16 n=2105344` failure);
* the pinned family switch (0.013781 / 96.427 % against the r334 pinned base) is **smaller** than r334's own
  default-vs-pinned gap (0.015033 / 96.24 %), and r336's is 0.014772 / 96.41 % — the pinned arm moves marginally
  *towards* the default.

## What was corrected or re-cut, and why

* **Rebased from r325 to r334/r336.**  The series applied cleanly at the r334 tip but its delta had to live
  *inside* block 06, and blocks 10 and 14 also touch `mmb.cu`.  Six conflicts were resolved so that the
  **applied tree is byte-identical to the tip the contributor's series produces** (`213764a779`), with one
  side effect worth knowing: the tiny-M `XBF16` activation plumbing that block 10 used to add now lives in
  block 06 with the shared launcher, so block 10's patch is smaller by those hunks.  Block 06 is a layer, not
  a standalone build.
* **No ROCm-version gate was added, deliberately.**  The feature's gate is the user's own
  `ROCBLAS_USE_HIPBLASLT=0`, which *is* the version gate and a more accurate one: the pin is only needed below
  hipBLASLt 1.5 (the `c50ea60130f` fix, ROCm 10.2), a 10.2 user who pins for another reason (a hipBLASLt OOM at
  a tight slab headroom) keeps the speed, and a version probe would mis-model a fix that lands in
  rocm-libraries nightlies before a ROCm release.  Reasoning and the one gap (hipBLASLt absent entirely →
  `GGML_CUDA_MMB_BF16_DENSE=1`): `ENVIRONMENT.md` §10.2.
* **Documentation the series did not carry**: nine `ENVIRONMENT.md` rows (§10.2), the block-06 note in
  `patches/README.md`, a `GREEDY-PURITY.md` §44 addendum, `INVARIANTS.md` r336, `MANIFESTS.md` block 06's gate
  row, and `TODO` #362 for the ROCm 10.2 question the series raises but cannot answer here.

## What the contributor could not reach, and is still owed nowhere

RDNA3_5 / RDNA3_0 and NVIDIA arms, and 1-GPU runs.  The change is unreachable on non-RDNA4 by construction
(every new default is a field of the RDNA4 row of `mmb_arch_defaults`; the launcher refactor is a pure move),
which is the delivery's documented, accepted state for a non-validated arch — no work is owed.
