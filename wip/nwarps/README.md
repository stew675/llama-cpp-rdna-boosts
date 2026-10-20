# `wip/nwarps` — RDNA3_0 per-shape `nwarps` for the dense mmvq weight kernel: **RESOLVED**

**Status: RESOLVED 2026-10-19 (opened 2026-09-21) — the width impurity is fixed, the knob is bit-pure
at every threshold, the scope is retuned from measurement, and a default-on promotion patch is cut
(§6).  Still WIP in this directory (not part of the delivery until the main development host folds it
into a block; not in `beta/`).**

* **Deliverables** (both apply to the current `release.json` tree, `git am`/`git apply` verified):
  * [`patches/per-shape-nwarps-rdna3-0-promoted.patch`](patches/per-shape-nwarps-rdna3-0-promoted.patch)
    (sha256 `55a7cd76…`) — **the one to fold into a block**: single commit, **default-on**
    (`GGML_CUDA_MMVQ_RDNA3_SMALL_M=0` is the kill-switch).
  * [`patches/per-shape-nwarps-rdna3-0.patch`](patches/per-shape-nwarps-rdna3-0.patch)
    (sha256 `eb32d36c…`) — the same change **default-off**, for an A/B, a beta test or a bisect.
  * The two differ only in the `getenv` default.  Fold/promotion instructions, the doc edits that go
    with it and the per-box gates: [`PROMOTION.md`](PROMOTION.md).
* The originating record is
  [`archive/work/mmb-general/gfx1100-s9-nwarps-results.md`](../../archive/work/mmb-general/gfx1100-s9-nwarps-results.md)
  (the gfx1100 session that found the candidate and stopped on the impurity).  Its conclusion —
  "re-derive the width-invariant reduction mapping for `nwarps=1`" — was **wrong about the cause**;
  see §2.  All its performance numbers were taken with the impure patch in place and are superseded
  by §5.

## 1. What the candidate is

The dense `mmvq` **weight** kernel (`mul_mat_vec_q_ksplit`) on RDNA3_0 takes `nwarps` from the
per-type table (`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL` → **8**, everything else → 1).  gfx1100 S9
found that this is the wrong choice for the small-M dense layers of a MoE (35B-A3B, gemma-26B),
which want `nwarps=1`.  The candidate adds a **shape-scoped** choice: the weight's row count `M`
decides, so the choice is fixed per tensor.

## 2. Root cause of the impurity (the actual finding)

The original patch computed

```c
small_m = table_id == RDNA3_0 && !has_fusion && nrows_x <= threshold;
```

i.e. it **excluded the fused single-token launches**.  A fusion launch (`ncols_dst == 1` only: the
rms_norm→Q8_1 fold, the SSM gate/beta fusions, the shared-expert down fold) then kept the table's 8
warps while the plain dense ksplit launch of *the same tensor* at `W = 2..8` dropped to 1.  Decode and
the verify batch therefore took different K-split orders — exactly the `W = 1..8` impurity the
campaign was chasing — and the bigger the tensor's contribution, the bigger the divergence (gemma-26B
3.35 at `M <= 4096`, 35B 0.150 at `M <= 2048`, 27B 0.166 blanket).

**No reduction-mapping re-derivation was needed.**  The ksplit kernel's K loop is

```c
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter)
```

with `blocks_per_iter = vdr*nwarps*warp_size/qi`: each thread owns a fixed **residue class of K
blocks**, independent of `ncols_dst`, `rows_per_block` and the token index, and the `warp_reduce_sum`
+ serial cross-warp combine trees are fixed for a given `nwarps`.  So *one* `nwarps` value per shape
**is** the width-invariance requirement.  The impurity was purely the exclusion.

Fix: apply the row-scoped value to **every** launch of the tensor, and to the three fusion kernels
that reproduce a dense weight launch (`ggml_cuda_op_ssm_gate_beta`,
`ggml_cuda_op_ssm_conv_l2_gatebeta`, `ggml_cuda_op_shexp_down_gate`) — the same "the fusion must
reproduce the launch it replaces" contract those kernels already document for `long_k`.

## 3. The patch

| file | change |
|---|---|
| `ggml/src/ggml-cuda/mmvq-rdna.cuh` | `mmvq_rdna_nwarps()` takes `bool small_m`; the RDNA3_0 wide types return `small_m ? 1 : 8`.  RDNA4 / RDNA3_5 and every other table untouched |
| `ggml/src/ggml-cuda/mmvq-rdna-params.cuh` | `calc_nwarps_weight()` takes and forwards `small_m`; `mmvq_rdna3_0_small_m()` (the env knob), `mmvq_rdna3_0_small_m_type()` (the instantiation gate), `mmvq_rdna3_0_small_m_weight(type, M)` (the rule) |
| `ggml/src/ggml-cuda/mmvq.cu` | `calc_nwarps()`, `calc_launch_params()`, `mul_mat_vec_q_ksplit` + `..._switch_fusion_ksplit` take the `small_m` template axis (bounds *and* body from the same value, so host and kernel can never disagree); the host computes it once from `nrows_x` and passes it to every launch; the three fusion ops above take the row-scoped value |

`GGML_CUDA_MMVQ_RDNA3_SMALL_M=<n>` (**default 0 = off**, byte-identical to the delivery — verified:
the width-probe hashes at threshold 0 are unchanged).  When non-zero:

| type | narrow (1-warp) block when |
|---|---|
| `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` | always (narrow wins at every row count measured) |
| `Q8_0` | `MMVQ_RDNA3_SMALL_M_MIN` (2048) `<= M <= n` |
| `Q6_K`, `IQ4_NL` | never (the wide block wins) |

**Instantiation cost** (the superseded patch added an ungated axis for every type: +41 % object /
 +100 % ksplit symbols, which is why the campaign was not shipped as it stood).  Here the axis is
`if constexpr`-gated to the types and columns that can fire, i.e. the 7 RDNA3_0 wide types at
`ncols_dst <= 8`:

| `mmvq.cu.o` (gfx1100, ROCm 10.1.0) | delivery | this patch |
|---|---:|---:|
| object size | 13.64 MiB | 15.36 MiB (**+12.6 %**) |
| `mul_mat_vec_q_ksplit` symbols | 2188 | 2640 (**+20.7 %**) |
| all `mul_mat_vec_q*` symbols | 2960 | 3413 (+15.3 %) |

## 4. Purity (the blocker) — now PASS everywhere

`test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512`, `KV=f16` unless noted,
`HIP_VISIBLE_DEVICES=0` (single RX 7900 XTX).  Every cell below was re-run **with the shipped patch**;
`0` is the unmodified delivery path (the 35B row also verified hash-identical to the pre-patch build):

| threshold | 27B Q4_K_M | 35B-A3B Q4_K_M | gemma-26B Q4_K_XL | gemma-12B Q8_0 |
|---|---|---|---|---|
| 0 (delivery) | PASS (delivery table) | PASS (hashes identical to the delivery) | PASS (delivery table) | PASS (delivery table) |
| 2048 | PASS | PASS (was **FAIL 0.150**) | PASS | — |
| 4096 | — | PASS | PASS (was **FAIL 3.35**) | PASS |
| 6144 | PASS (was **FAIL 0.165**) | — | — | — |
| 8192 (**recommended**) | PASS | PASS | PASS | PASS |
| 1e9 (blanket) | PASS (was **FAIL 0.166**) | PASS | PASS (was **FAIL**) | PASS (was **FAIL**) |

27B × the eight native KV types (`f16 bf16 q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl`) at threshold 8192: all
PASS, worst maxdiff 0.  (`—` = not measured; the intermediate thresholds only select *which* shapes
flip, and the rule is per-shape, so the recommended `8192` row plus each previously-failing cell is the
meaningful coverage.)

Quality: 35B-A3B PPL (wikitext-2, 64 × 2048, `-fa on`, f16 KV) **5.7861 both** thresholds, per-chunk
losses identical; greedy text coherent and near-identical (one word differs in a 180-token answer —
a near-tie flip, expected when a reduction order changes).

## 5. Performance (gfx1100, 1 GPU; `llama-bench -p 0 -n 128 -r 3`, or `llama-cli -n 400` for MTP)

Model shapes the rule acts on (`M` = weight rows, `K` = input width, dense types only):

| model | M values (K) |
|---|---|
| 35B-A3B Q4_K_M | 512 (2048), 2048 (512 / 4096), 4096 (2048), 8192 (2048), 248320 (2048) — all Q8_0 + Q6_K lm_head |
| gemma-26B Q4_K_XL | 1024, 2048, 2112, 2816 (n_embd / n_ff both ways), 4096, 8192, 262144 — all Q4_0 |
| gemma-12B Q8_0 | 512, 2048, 3840, 4096, 8192, 15360, 262144 — all Q8_0 |
| 27B Q4_K_M | 48 (×96), 1024 (Q8_0/Q6_K), 5120 (Q6_K/IQ4_NL), 10240, 12288, 17408, 248320 |

The flat-M rule's threshold sensitivity (this is the evidence for the per-type refinement; tg128,
`-d 16384`, except gemma-12B `-d 0`):

| `GGML_CUDA_MMVQ_RDNA3_SMALL_M` | 35B-A3B | gemma-26B | gemma-12B | 27B |
|---|---:|---:|---:|---:|
| 0 | 124.95 | 125.13 | 54.08 | 40.13 |
| 2048 | 124.77 | 126.20 | 54.34 | — |
| 4096 | 124.47 | 128.50 | 54.34 | — |
| 8192 | **127.49** | 128.77 | 54.29 | 39.90 |
| 1e9 | 127.35 | **131.12** | 53.15 | 39.94 |

Read off it: the 35B's win is its **M = 8192** Q8_0 shapes (nothing below); gemma-26B gains from
Q4_0 at *every* row count (its lm_head is worth ~1.8 %); gemma-12B loses 2 % when **M = 15360**
(ffn_gate/up) is flipped and the 27B loses 0.6 % when **M = 48 / 1024** (Q8_0, K = 5120) or its
Q6_K/IQ4_NL shapes are.  Those three counter-examples are why the shipped rule is per-type with a
Q8_0 band instead of one flat threshold.

The shipped rule at the recommended `=8192`:

| model / bench | thr 0 | thr 8192 | delta |
|---|---:|---:|---:|
| 35B-A3B Q4_K_M tg128 d16384 | 125.09 / 125.59 | 127.23 / 127.38 | **+1.7 … 2.0 %** |
| gemma-26B Q4_K_XL tg128 d16384 | 125.14 | **131.45** | **+5.0 %** |
| gemma-12B Q8_0 tg128 d0 | 54.05 | 54.28 | +0.4 % |
| 27B Q4_K_M tg128 d16384 | 40.14 | 40.16 | 0 (neutral) |

35B-A3B MTP (`--spec-type draft-mtp --spec-draft-n-max 3 -n 400 --temp 0`, repeatable to ±0.2 %):

| axis | thr 0 | thr 8192 | delta | acceptance / mean len (0 → 8192) |
|---|---:|---:|---:|---|
| prose (P) | 200.6 t/s | **222.0 t/s** | **+10.7 %** | 0.653/2.96 → 0.750/3.24 |
| code (C) | 235.5 t/s | **250.2 t/s** | **+6.3 %** | 0.801/3.40 → 0.851/3.55 |

Part of the MTP gain is a higher acceptance rate (a changed reduction order moves near-ties in the
draft/verify agreement); the plain-decode gains above are kernel-time only.  Plain 35B decode for
reference: 129.1 t/s (thr 0).  The old record's "+2.1 % MTP / +9 % blanket" numbers were measured with
the impure patch — do not quote them.

## 6. Promotion

The rule is non-negative on all four measured models, so the recommendation is to promote it into a
block with the knob **default-on**.  That state is already cut and handed over as
[`patches/per-shape-nwarps-rdna3-0-promoted.patch`](patches/per-shape-nwarps-rdna3-0-promoted.patch)
(verified: unset environment == the measured `=8192` build, `=0` == the r91 table).

The fold itself — which block owns which file (the two policy headers are **block 08**'s; `mmvq.cu` is
shared with 07/09/10), the ready-to-paste `ENVIRONMENT.md` row and `GREEDY-PURITY.md` invariant, the
per-box gates (including "gfx1201 must be byte-identical" — the scope is RDNA3_0-only), the expected
RDNA3_0 numbers and the rollback — is [`PROMOTION.md`](PROMOTION.md).

Nothing outside `mmvq-rdna*.cuh` + `mmvq.cu` changes, and the knob is
`table_id == MMVQ_PARAMETERS_RDNA3_0` only, so no other backend's dispatch is touched (scope policy
satisfied).  gfx1151/RDNA4 want their own sweeps before any analogous change — they have their own
band-uniform tables.

## 7. Reproduce

```sh
cd ~/llama.cpp                       # at the release.json tree
git apply <repo>/wip/nwarps/patches/per-shape-nwarps-rdna3-0.patch
~/bin/build-llama-rocm               # or: cmake --build build-rocm --target llama-cli test-logits-width-probe

# purity (the gate this campaign existed for)
for t in 0 2048 4096 8192 1000000000; do
  GGML_CUDA_MMVQ_RDNA3_SMALL_M=$t HIP_VISIBLE_DEVICES=0 \
    ./build-rocm/bin/test-logits-width-probe <model> <repo>/prompts/prose-rdna-boosts.txt 1024 512 | tail -1
done
# KV matrix: prefix KV=bf16|q8_0|q4_0|... (8 native types)

# performance (never run two benches at once; RDNA3_0 box, one GPU)
GGML_CUDA_MMVQ_RDNA3_SMALL_M=8192 HIP_VISIBLE_DEVICES=0 \
  ./build-rocm/bin/llama-bench -m <model> -ngl 99 -p 0 -n 128 -d 16384 -r 3
GGML_CUDA_MMVQ_RDNA3_SMALL_M=8192 HIP_VISIBLE_DEVICES=0 \
  ./build-rocm/bin/llama-cli -m <35B> -ngl 99 --spec-type draft-mtp --spec-draft-n-max 3 \
    -f <repo>/prompts/prose-rdna-boosts.txt -n 400 --single-turn --no-display-prompt --temp 0 -lv 4
```

## 8. Do not reintroduce

* **Excluding fusion launches from a shape scope.**  A fused single-token launch and the plain
  multi-token launch of the same tensor must take the same `nwarps`; that is the whole bug.  Any new
  per-shape `nwarps`/`rpb`/VDR rule has to be forwarded to every kernel that reproduces one of those
  launches (`GREEDY-PURITY.md` §19/§25, invariant 2).
* **A host/kernel dispatch mismatch.**  Compute the value once on the host and pass it to *both* the
  launch dims and the kernel template (the first cut of this candidate launched a 1-warp block for an
  8-warp kernel — the 35B emitted `///////`; `GREEDY-PURITY.md` §20).
* **An ungated template axis.**  `small_m` must stay `if constexpr`-gated to the affected types and
  to `ncols_dst <= MMVQ_MAX_BATCH_SIZE`, or the object grows ~2× on every HIP target for a rule that
  can only fire on one.
