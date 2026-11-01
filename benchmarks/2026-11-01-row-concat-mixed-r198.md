# The mixed-type dense row concat (r198, PR #260) — the re-cut, and the gfx1201 MTP cost named

Dated record for **r198** (`WORKLOG.md` 261101.004).  One contributor PR folded into **block 10** (it owns the
matcher and `mmvq.cu`), default-on with a `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` kill switch, bit-identical.
Supersedes the held r192 measurement of the first cut
([`2026-10-31-r192-row-concat-mixed.md`](2026-10-31-r192-row-concat-mixed.md)); that record stays as the history
of why r192 was not tagged.

## The change (re-cut)

The r177 RDNA4 row concat runs the 1-token Q/K/V projections in one ksplit launch, but only when all three
weights share a type.  In the K-quant "M" mixes `attn_v` is `Q6_K` in about half the layers while Q/K are
`Q4_K`, `Q5_K` or `IQ4_XS` (Qwen3-Coder-30B-A3B Q4_K_M: 12 of 48; Qwen3.6-35B-A3B: 6 of 10), so V ran as its
own launch.  `mul_mat_vec_q_ksplit_cat2<type, Q6_K>` runs the `Q6_K` body (with that segment's own row stride)
for V's row blocks and the launch type's body for the rest.  Both types take the same nwarps (`static_assert`),
every segment's row count is even (a 2-row block cannot straddle a boundary), and the launch tensor is the
largest segment (its row count sizes the row guards).  The matcher takes either graph order and, in the mixed
case, requires every segment that runs the launch type's body to share the launch tensor's `nb[1]` (one row
stride is passed for those blocks) — a hardening added on integration.  `=0` restores the same-type-only
matcher.

**The codegen fix.**  The first cut turned `mul_mat_vec_q_ksplit` into an inlined `__device__` body.  That
recompiled the **multi-column verify kernels** (gfx1201 Q4_K ncols 5: 118 -> 97 VGPRs, 864 -> 1029
instructions) and cost **6-7 % of MTP decode** with an identical kernel/graph census, compute buffers, text and
acceptance.  The re-cut moves the body verbatim into `mmvq-ksplit-body.inc`, `#include`d into the unchanged
kernel and into the cat2 launch.  A per-kernel VGPR census of the MTP profile confirms the pre-existing ksplit
set is identical (the only new entries are the two cat2 instantiations).  This closed `TODO` #128;
`GREEDY-PURITY.md` §80.

## Environment

* 3x Radeon AI PRO R9700 (gfx1201), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), one job at a time.
* Baseline = **r197** (`ba7c2b7b8`, tree `eaf673210a`) and the patched arm = the same build directory at the
  PR re-cut (`40c915262`), plus the integration hardening.  Final gates ran on the integrated r198 tree
  (`28fbe75ca`, tree `52337cb38a`).
* `ROCBLAS_USE_HIPBLASLT=0` for every fresh-process comparison (issue #67).  `GGML_CUDA_FA_QUANTS` is the
  default set in both build dirs.

## MTP (the arm that rejected the first cut)

Dense Qwen3.8-27B UD-Q4_K_XL, `--spec-type draft-mtp --spec-draft-n-max 3`, bf16 KV, `ROCBLAS_USE_HIPBLASLT=0`:

| arm | r197 | + the re-cut | drafts |
|---|---:|---:|---|
| 1 GPU, `-c 16384 -n 500`, 5 interleaved rounds | 76.4 / 76.4 / 76.3 / 76.5 / 76.3 | 76.3 / 76.2 / 76.3 / 76.4 / 76.3 | 417 / 360 both |
| 3 GPU `-sm tensor`, `-c 102400 -n 800`, 3 rounds | 133.0 / 134.1 / 133.9 | 133.8 / 134.2 / 133.8 | 670 / 575 both |

The first cut read 70.7 (1 GPU) and 126.4 (3 GPU) on this box.  Four-axis adaptive gate (27B, 1 GPU,
`-n 3000`, reasoning pinned per axis, `-c 32768`, bf16 KV) is identical on every axis, including acceptance:

| axis | r197 plain/draft t/s | r198 plain/draft t/s | drafts (both) |
|---|---|---|---:|
| R | 58.5 | 58.4 | 3303 / 1897 |
| P | 72.0 | 72.0 | 2160 / 1730 |
| C | 80.0 | 80.1 | 2408 / 2196 |
| K | 85.5 | 85.8 | 375 / 372 |

## Decode throughput (Qwen3-Coder-30B-A3B Q4_K_M, f16 KV, 3 interleaved rounds, mean)

| arm | tg128 | @ d4096 |
|---|---:|---:|
| r197 | 161.54 (164.02 / 159.96 / 160.63) | 137.69 (137.89 / 137.83 / 137.34) |
| + the mixed concat | **162.57** (162.71 / 162.00 / 162.99) | **139.14** (139.27 / 139.05 / 139.10) |
| delta | **+0.6 %** | **+1.05 %** |

3 GPU `-sm tensor` tg128: **171.62 -> 173.53 t/s (+1.1 %)** (the submitter marked multi-GPU untested).
The submitter's own re-cut measured **+0.6 / +1.2 %** on their R9700, matching.

`rocprofv3` census (30B-A3B, 8 decode steps): switch off **1234** ksplit launches; on **898** ksplit + **168**
`mul_mat_vec_q_ksplit_cat2` = 1066, i.e. 168 `cat2` launches replace 168 standalone `Q6_K` launches.

## Purity

| arm | instrument | result |
|---|---|---|
| the concat | `gate-prefill-logits.sh --ub 1 --ab "GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0"` (30B-A3B, the firing model) | **0.000000 KLD / 100.000 %** |
| the tree | canonical 27B Q8_0 prefill-logit gate | **0.000000 / 100.000 %** |
| the concat | greedy text, `ROCBLAS_USE_HIPBLASLT=0`, base == pat | 30B f16 `1065 chars sha=000d77b327e9`; 30B q8_0 KV `1131 chars sha=9fb5b1832d34`; 30B 3 GPU tensor `1019 chars sha=bbdb61ea2adb`; 27B MTP n3 `1270 chars sha=5dbf95984c65` |
| the tree | rule-5 `llama-batched-bench -npl 1,4,8` (27B, q8_0 KV, 3 GPU tensor) | B1 **48.96 -> 49.91**, B4 **159.88 -> 162.90**, B8 **267.24 -> 270.66** (no worse) |
| the tree | `test-backend-ops -o MUL_MAT` | **2445/2445** |

## Cross-arch (the cat2 body is compiled out off RDNA4)

| box | build | cat2 launches | greedy text | tg128 |
|---|---|---:|---|---|
| RX 7900 XTX (gfx1100) | clean | **0** | 30B-A3B `1263 chars sha=8392d4044a0e` base == pat | 208.99/211.03 -> 211.30/211.30 |
| Strix Halo (gfx1151) | clean | **0** | 35B-A3B `1177 chars sha=21eee00556dd` base == pat | 63.31/63.13 -> 63.03/63.03 |

## Release-flow gates

* `scripts/gate-per-block-build.sh --fast --changed-from ba7c2b7b8` — **2/2** (block 10 8 s, tip 10 s).
* `scripts/validate-set.sh` — green: 16/16 strict `git am` on a fresh `6050b467c` tarball, tree
  `52337cb38a1e4e4ebdaa941f6b8f801ab6aeac32`, Windows portability lint.
* Coherence and MTP arms above; `check-gpu-faults.sh` clean (journal + logs).
* The re-cut applies cleanly to the r197 tree and to block 10 (the only block that touches `mmvq.cu` /
  `common.cuh` after block 09).

## Reproduce

```sh
scripts/scratch-tree.sh keep r197                         # ~/llama.cpp/build-rocm/bin-base
scripts/scratch-tree.sh --fast arm /tmp/pr260/0001.patch  # bin-pat
scripts/gate-prefill-logits.sh --ppl <build>/bin/llama-perplexity \
  --model /llm/models/Qwen3/30B-A3B/Q4_K_M/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --ub 1 --chunks 40 --base-dir /tmp/gates --ab "GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0"
```
