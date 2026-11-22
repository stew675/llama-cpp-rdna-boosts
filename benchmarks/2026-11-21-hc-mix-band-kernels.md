# The Q8_0 hc mixer's decode band, restructured (r339, contributor PR #344)

**Claim.** The fused Q8_0 `GGML_OP_HC_MIX` (block 11, r313) still ran one block per `(row, token)` on the down
dots and emulated eight reference blocks per fused block on the up+collapse. Serving every token of the band
from one block per weight row, and computing each up row's gate from the reference's two chunk partials
directly, is **bit-identical** and takes the op from 33.1 to 21.6 us/run at the 4-token verify width.

Both mechanisms are verified here on the delivery's own Q8_0-hc class (unsloth `UD` files: `hc_attn_down` Q8_0,
`hc_attn_inject` F32 or Q8_0). BF16 hc weights (the GSQ quants) take `hc_mix_bf16_*` and are untouched.

| | |
|---|---|
| GPU | 3 x R9700 (gfx1201), 9950X3D |
| toolchain | **ROCm 10.1.0-gfx120X** for the A/B (scratch tree, one path, ccache), ROCm 10.2.0-gfx120X for the release arms |
| arms | `bin-base` = the r338 tree (`2f0d05010`), `bin-pr344` = the same tree + the two patches (fixed, see §5) |
| models | `Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL` (Q8_0 hc + **F32** inject), `Qwen3.8/Flash-Next/Q4_K_M` (Q8_0 hc + **Q8_0** inject), `Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S` (BF16 hc) |

## 1. Per op — `test-backend-ops perf -o HC_MIX`, one R9700, us/run

`n_embd=2560, hc=4, hc_lr=320, inject=1, wtype=q8_0` (the op's Flash-Next shape):

| n_tokens | base | patched | the submission's own table (stock -> patched) |
|---|---|---|---|
| 1 | 20.20 | **16.65** | 20.64 -> 16.03 |
| 2 | 22.37 | **18.80** | 23.68 -> 18.25 |
| 3 | 27.04 | **19.59** | 27.45 -> 19.50 |
| 4 | 33.13 | **21.62** | 31.19 -> 22.01 |
| 5 | 37.67 | **24.34** | 35.40 -> 24.57 |
| 8 | 51.33 | **32.67** | 49.27 -> 31.24 |

`wtype=bf16` is unchanged (16.61 / 18.42 / 21.04 / 25.69 / 31.58 / 44.72 -> 16.58 / 18.52 / 21.26 / 25.93 /
31.71 / 44.98).  The kill-switch arm (`GGML_CUDA_HC_DOWN_MT=0 GGML_CUDA_HC_UP_ROWS=0`) reproduces the base
column to within 0.1-0.5 us, i.e. the win is the new code and not the environment.
`test-backend-ops -o HC_MIX`: **30/30** on both builds.

## 2. The delivery's reference texts do not move

Same-seed greedy, `--seed 42 --temp 0`, text hashed with `scripts/extract-generated.py`:

| arm | base | + PR #344 | + PR #344, both kill switches `0` |
|---|---|---|---|
| canonical 4B Q8_0 coherence, 3-dev `-sm tensor`, `-n 20` | `100 chars 1c5d32ac537d` | `100 chars 1c5d32ac537d` | — |
| 2-dev Flash-Next GSQ IQ3_S, prose, `-n 128` (the r338 anchor) | `653 chars c420dfe04c42` | `653 chars c420dfe04c42` | — |
| 3-dev UD-Q4_K_XL (`-ncmoe 48`), `-p` Eiffel, `-n 96` | `431 9a72cc49804d` | `431 9a72cc49804d` | `431 9a72cc49804d` |
| the same + `draft-mtp n3` (verify widths 2-4) | `444 455791ba340b` | `444 455791ba340b` | `444 455791ba340b` |
| the same, 5246-token prose prompt | `489 35d0945cd4e3` | `489 35d0945cd4e3` | — |
| **Q4_K_M** (Q8_0 *inject* rows; `test-backend-ops` never covers them) | `436 d32a66bfd8c2` | `436 d32a66bfd8c2` | `436 d32a66bfd8c2` |
| `GGML_CUDA_HC_MIX_PREQ=0` (no pre-quantize; the MT block writes no `y_v`) | `431 9a72cc49804d` | `431 9a72cc49804d` | — |
| `GGML_CUDA_HC_UP_FUSE=3` (the emulation's 1-sub-block shape) | `431 9a72cc49804d` | `431 9a72cc49804d` | — |

`scripts/gate-hc-fusions.sh` on the Q8_0-hc class, 3-dev `-sm tensor`, both builds (10 arms each): **every arm
reads the same sha** --

| arm | both builds (ROCm 10.1) |
|---|---|
| `chain_chain` (true chain) | `450 e91d0e8c106e` |
| `mix_op` / `both_ops` | `433 8cd639da2fbf` |
| `comb_op` / `cuda_fuse_off` | `450 e91d0e8c106e` |
| `dsv4_pre_op` | `452 5e4258c26236` |
| `dsv4_post_op` | `448 ecbc57182a59` |
| `all_ops` / `ops_no_post` / `defaults` | `460 2f9c4b5da28b` |

`mix_op`'s `8cd639da2fbf` is the value r313 recorded for this class, i.e. the op's numerics are unchanged since
the fold.  On the **release toolchain** (ROCm 10.2) the same gate was run against an r338 build of the same tree
(`build-rocm-r338`) and the **r339 release build**, and all ten arms are byte-identical again -- `chain_chain`
`442 23d3d237af9d`, `mix_op`/`both_ops` `439 e50a19238893`, `comb_op`/`cuda_fuse_off` `442 23d3d237af9d`,
`dsv4_pre_op` `436 d32a66bfd8c2`, `dsv4_post_op` `457 f7439634f50d`, `all_ops`/`ops_no_post`/`defaults`
`431 df42686290aa`.  Those hashes are *different from the 10.1 row* for the same pairs of builds: a generated
text is per-toolchain and per-split, so the two rows are two references, each compared base-to-change.

## 3. Verify-band acceptance is identical to the digit

Protocol A of `mtp-adaptive-methodology.md` pointed at the class that actually builds `GGML_OP_HC_MIX` (the
canonical dense 27B arm never does), `--spec-draft-n-max 7` (so the verify width sweeps 2-8), `-n 900`,
`--seed 42 --temp 0`, acceptance from `-lv 4`:

| axis | acceptance (base) | acceptance (patched) | gen t/s base -> patched |
|---|---|---|---|
| K recall | **0.96652** (433/448), mean len 7.77 | **0.96652** (433/448), mean len 7.77 | 91.6 -> 95.0 (+3.7 %) |
| R reasoning | **0.36678** (647/1764), mean len 3.57 | **0.36678** (647/1764), mean len 3.57 | 35.3 -> 37.5 (+6.2 %) |
| P prose | **0.63463** (733/1155), mean len 5.44 | **0.63463** (733/1155), mean len 5.44 | 70.8 -> 74.1 (+4.7 %) |
| C code | **0.78675** (760/966), mean len 6.51 | **0.78675** (760/966), mean len 6.51 | 77.9 -> 82.8 (+6.3 %) |

The accepted/drafted totals and the mean verify length are equal too, so this is the same band, not a
same-average-different-run read.

## 4. Rate on a real shape

3-device UD-Q4_K_XL, `-sm tensor -ncmoe 48`, `llama-bench`, interleaved pairs:

| rep | base tg128 | + PR #344 tg128 |
|---|---|---|
| 1 | 49.73 ± 2.46 | **50.89 ± 2.59** |
| 2 | 49.75 ± 2.38 | **50.96 ± 2.83** |

**+2.3 %**.  The shape is host-expert dominated (~64 GiB of experts in the pinned pool, ~45 t/s), so this is a
lower bound: on this class the op is 0.6 % of a token (2 hc sites/layer is 20-40 us of ~22 ms), and the
per-op table plus the four-axis row is the instrument that resolves it.  A denser decode shows more: the
`draft-mtp` verify widths above read +3.7 to +6.3 %.

`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` on the dense 27B UD-Q4_K_XL (a graph that never builds the op,
so this is the "no collateral damage" leg): B=1 51.5/50.8 vs 51.5/51.9, B=4 157.1/156.2, B=8 248.8/247.2 t/s,
all inside the run-to-run spread of one build.  The same instrument on the **host-expert** UD-Q4_K_XL shape is
not usable: three consecutive loads of the same binary scored 175, 162 and 106 t/s at B=8 (the pinned pool's
coverage moves with page-cache pressure), so it measures residency, not kernels.

## 5. The submission did not build (fixed in the fold)

Its five shuffle calls passed a 32-bit mask literal (`__shfl_sync(0xFFFFFFFF, …)`,
`__shfl_xor_sync(0xFFFFFFFF, …)`).  HIP's `hip/amd_detail/amd_warp_sync_functions.h` rejects that with
`static_assert(sizeof(MaskT) == 8, "The mask must be a 64-bit integer…")` -- a hard error on **ROCm 10.1.0,
10.2.0 and 7.14.1** alike (reproduced by compiling the patched `hc-mix.cu` alone against each line).  The fold
uses the tree's own `ggml_cuda_shfl()` / `ggml_cuda_shfl_xor()` wrappers, which pass the 64-bit full mask and
move the same lanes as the DPP-XOR path `warp_reduce_sum()` takes -- so the values are the same, and the
per-op/text/acceptance identity above is measured with that form.

## 6. Gates

* Builds green on both ROCm lines (whole `all` target).
* `scripts/gate-hc-fusions.sh` PASS on the release build (Q8_0-hc class, 3-dev `-sm tensor`), every hard arm
  byte-identical to the true chain.
* `scripts/gate-prefill-logits.sh` **-0.000000 / 100.000 %** against the r338-recorded base (the op is not in a
  dense prefill graph; the gate is owed by the release, not by the change).
* Coherence anchors unchanged (§2), and no GPU fault in the kernel journal or any arm's log
  (`check-gpu-faults.sh` via `gate-hc-fusions.sh`).

## 7. Not measured / not claimed

* NVIDIA/CUDA, Windows, gfx1151 and gfx1100 are untested (as submitted).  The code is backend-neutral after the
  mask fix (`hc-mix` uses only `vec_dot_q8_0_q8_1_impl`, `warp_reduce_*` and `ggml_cuda_shfl*`), but a build is
  not a measurement.
* The merged `hc_*_down_inject` layout (post-#31028) with Q8_0 hc weights: no such file here (same accepted
  limitation as r313).
* The BF16 path is untouched, so nothing about it is claimed beyond "the per-op number did not move".
