# Prefill "arrangement" analogues: packed-QSA and the chunked-GDN case

**Status:** exploration / scoping (2026-09-13).  Not delivery work; no patches applied.  Companion to
`wip/tiled-gdn/05-where-the-speed-comes-from.md` and the archived
`archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md`.

## 0. The concept

pwilkin's prefill wins are all the same move: **transform the data once into a tensor-core-friendly
form, then consume it with a dense WMMA kernel.**  Four instances:

| arrangement | what is transformed | pwilkin artifact | our status |
|---|---|---|---|
| weights -> bf16 shadow | IQ4_NL weight rows -> cached bf16 | `mmb.cu` | parked port (`GGML_CUDA_MMB`, default off) |
| selected KV -> packed blocks | top-k cells -> sorted, block-aligned f16 blocks | `qsa_pack_keys`/`values` + `qsa3_*` | **missing** (we gather per cell, VEC compute) |
| activations -> bf16 streams | producer output kept bf16 | `mark_bf16_only` | missing |
| conv state/input -> depthwise | concat+transpose chain -> one kernel | `gdn-conv.cu`/`ple-conv.cu` | missing (generic `ggml_ssm_conv`) |

The question here: which of these has a **chunked-GDN / QSA** analogue in our tree?  Answer:
**QSA has a large one; chunked GDN is already on the arrangement path.**

## 1. QSA: the missing packed-block WMMA attention (the real money)

### What we do now

`ggml/src/ggml-cuda/fattn-qsa.cu` (738 lines): one block per (query column, `<= QSA_MAX_HEADS=16`
heads).  For each 32-cell top-k tile it **gathers the selected cells individually by index** into
smem and scores them with **scalar FMA / `half2`/`bfloat162` `ggml_cuda_mad`** — a VEC flash-attn,
not a WMMA one.  The cells are indirect (`idx[]`), so each tile is a scattered read.

### What pwilkin does

`ggml/src/ggml-cuda/qsa.cu` (443 lines) + the qwen4exp graph:

- `qsa_pack_keys` / `qsa_pack_values` build **contiguous f16 block buffers** once per graph
  (consumed as `cur->src[6]`/`src[7]`), gated `n_query >= 128`.
- `qsa3_rows_kernel` + `qsa3_merge_kernel` **merge `G = 4` consecutive queries' top-k lists** into a
  sorted, deduplicated, block-aligned descriptor: block ids (`ublk`) + a 16-bit per-query membership
  mask (`umask`) + a count.
- `qsa3_attn_kernel` runs **WMMA f16** (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) over the packed
  blocks, four queries at a time as three 16x16 tiles, with the membership mask applied to the scores.

### Why it is much faster

- contiguous f16 block reads instead of per-cell gathers (coalescing + L2 reuse);
- tensor cores for both the KQ and PV products instead of scalar FMA;
- the **union** of 4 queries' selections is smaller than 4x top-k *and* block-aligned, so fewer
  attention cells and a mask that folds into the score pass.

### Evidence (archived handover, gfx1151, pp8192)

| kernel | ours | pwilkin | gap |
|---|---:|---:|---:|
| QSA attention | `flash_attn_qsa` 2.78 s | `qsa3_attn` 0.73 s | **-2.05 s** |
| IQ4_NL weight GEMM | 8.53 s | 6.40 s | -2.13 s |

The handover's verdict is blunt: after the `mmb` port the remaining gap is dominated by the QSA
kernel, and its own next step is *"port his QSA v3 sparse-attention kernel (`qsa3_attn_kernel` +
`qsa3_rows_kernel`, `LLAMA_QSA_SCORE_BOUNDS`/`PACK_KEYS`/`FA_V3`) -> expected ~+117 t/s; plus the
bf16-producer marking (~+32) and the HC fusions (~+42) -> ~1130."*

### Within our framework ("our own trail")

1. Add the packing as a graph-side step: `ggml_qsa_pack_keys`/`values` (or build the packed layout
   inside the existing `GGML_OP_FLASH_ATTN_QSA`), producing f16/bf16 block buffers plus the merged
   descriptor.  Reuse our existing QSA input/`idx` plumbing (`build_qsa_top_k`, `llm_graph_input_qsa_k`).
2. New WMMA attention kernel; **keep `fattn-qsa.cu` unchanged as the reference and the
   decode/verify-band path.**
3. **Gate it prefill-only** (`n_query >= 128`, his threshold).  This keeps `W = 1..8` on the exact VEC
   kernel, so the packed kernel's different reduction order cannot split decode from verify.  The
   *prefill* output changes (a re-baseline, not an impurity) — document it and check PPL against the
   dense masked oracle and same-seed coherence.
4. Alternative: make the packed kernel serve the whole band (`W = 1..8`) so the whole path is
   uniform; that needs its own width matrix but preserves the delivery's plain == draft-mtp guarantee.

### Purity / risk

- prefill-only packed + VEC in the band -> the band is untouched; `plain == draft-mtp` holds by
  construction, at the cost of a prefill re-baseline.
- band-uniform packed -> still pure iff the kernel is identical across `W = 1..8`; verify with the
  `W=1..8` logits matrix and `mstep`.
- The packed kernel is a new arithmetic; it must not be defaulted on until the PPL/coherence and MTP
  gates pass.

### Effort / expected

| item | effort |
|---|---|
| pack ops + merged descriptor (graph) | 1-2 days |
| `qsa3`-style WMMA attention + mask | 3-5 days |
| prefill gate + wiring + A/B | 1-2 days |
| validation (PPL, coherence, W=1..8, MTP) | 1-2 days |
| **total** | **~6-10 days** |

Expected ~+100-120 t/s on qwen4exp pp16384 (the handover's +117), i.e. ~934 -> ~1050 with `mmb`,
or ~787 -> ~900 on the current delivery.

## 2. Chunked GDN: already arranged

- `gated_delta_net_chunked_bf16.cu` **is already the "weights -> bf16 -> WMMA" pattern applied to the
  recurrence**: fp32 tensor I/O converted to bf16 while staging, every GEMM on the RDNA4 WMMA units,
  the recurrent state held in WMMA accumulators across the chunk loop, the KKT inverse in a compact
  bf16 scratch.  There is no per-cell gather to pack.
- On gfx1201 the delivery's chunked bf16 GDN is **~5x faster than pwilkin's tiled GDN** at the op
  level (see `wip/tiled-gdn/03-validation-gfx1201.md`), so there is no GDN-side arrangement gap of
  the `qsa3` scale.
- Two smaller analogues do remain:
  a. **bf16 producer marking** — keep the q/k/v streams bf16 so the chunked kernel skips the
     on-the-fly fp32 -> bf16 conversion and the fp32 write.  pwilkin's "keep streams in bf16 end to
     end" is 1.08x; the handover measured the missing bf16-producer cache at `mmb_cvt` 645 ms.  Our
     tree has no `mark_bf16_only` equivalent.
  b. **depthwise conv1d** for the GDN (and PLE) conv — `src/models/qwen35.cpp` still builds
     `build_conv_state` + generic `ggml_ssm_conv` (concat + transpose + cont chain); pwilkin's
     `gdn-conv.cu`/`ple-conv.cu` fuse it (the 1.08x "depthwise conv1d" family).

## 3. Priority

1. **Packed-QSA WMMA** — the largest single lever (~+100 t/s, 2.8x on its own kernel).  Confirm the
   archived 2842/730 ms attribution on the current tip first, then port.
2. **bf16 producer marking** — benefits the GDN and the whole elementwise tail; cheap-ish.
3. **Depthwise conv1d** (GDN + PLE) — 1.08x family.
4. `mmb` weight GEMM — already parked; resume per its §12 checklist.

## 4. Our-trail constraints (unchanged)

- Keep the VEC QSA as the band/reference until a packed kernel is proven band-uniform.
- Anything that changes the arithmetic is **prefill-only** (`n_query >= 128`, as `mmb` is `T >= 512`)
  or **opt-in, default-off**, with an accepted-risk note.
- Every such win carries the full delivery gate: PPL/coherence, `W = 1..8`, MTP acceptance,
  `test-recurrent-state-rollback`.
- The delivery's prefill/band split is the whole point of this repo; the arrangements above are only
  admissible if they respect it.

## 5. Two clarifications (added 2026-09-13)

### 5.1 The weight arrangement is a load-time shadow, not an on-disk format

pwilkin's `mmb` does not consume a specially arranged GGUF.  It builds a **bf16 shadow of the weight
rows at load/graph time** (dequantize once, cache, run WMMA).  The only "special" thing about his
checkpoint is that it is **uniformly IQ4_NL**, which makes his `mmb_supported_mmid`/`_glu` predicates
accept *every* expert weight.  Nothing on disk is pre-arranged.

Consequence for us: the same trick works from any normal GGUF for any weight type.  What is needed is
a **per-type dequant-to-bf16 staging** kernel.  pwilkin wrote IQ4_NL first because that is his model;
our checkpoint's experts are IQ3_S, which his predicates reject.  So our generalization is a
*generic dequant-to-bf16 shadow* (a small family of kernels, one per quant type), not a new file
format — that is the parked `mmb-port.patch`'s next step.

### 5.2 The GDN lever is already spent

On the target box (27B Q6_K, gfx1201) the chunked bf16 GDN is **~1.1 % of a pp2048 pass**:

| | per op (n=2048, est.) | x 48 layers | share of the 1991 ms pass |
|---|---:|---:|---:|
| chunked bf16 (ours) | ~0.46 ms | ~22 ms | **~1.1 %** |
| sequential | ~3.9 ms | ~187 ms | ~9.4 % |

The whole sequential -> chunked transition bought **~9 %** end-to-end (177 ms).  There is almost
nothing left to win in the GDN op itself.  The 1.79x qwen4exp gap sits in the **weight GEMMs**
(`mmb`), the **QSA attention**, the **HC fusions** and the **bf16 stream tail** — the archived
handover's kernel attribution puts `flash_attn_qsa` (2.78 s vs 0.73 s), the IQ4_NL weight GEMM
(8.53 s vs 6.40 s), `quantize_mmq_q8_1` (1.33 s vs ~0.2 s) and HC (1.62 s vs 0.93 s) at the top, with
the GDN **not among them**.

This is why "a weight/layout-optimized chunked GDN" cannot close the model-level gap: the GDN op
**has no weights** (the weights are in the projections that feed it) and it **already runs the
bf16/WMMA layout**.  The weight arrangement belongs to the projections (`mmb`); the data arrangement
belongs to QSA (packed blocks).  Optimizing the GDN further — bf16 producer marking, depthwise
conv1d — is worth the smaller 1.08x-class deltas, not the model-level gap.

### 5.3 On "a lot of sequential processing" in the chunked GDN

The chunked GDN is **two passes**: `kkt_solve` (fully parallel over `(chunk, head)`: the gram + the
KT inverse) and `chunk_scan` (the state is sequential **across chunks** but parallel across `S_v/16`
column slices x H heads x n_seqs).  So exactly one dimension is sequential; everything else is not.
That is why it beats pwilkin's tiled kernel, which is a pure per-token scan (tiled LDS staging, fixed
block parallelism).

## 6. Bottom line

The "chunked GDN + QSA equivalent" of pwilkin's weight arrangements exists, and it is mostly on the
**QSA** side: pack the selected KV into block-aligned f16 tiles and run WMMA (his `qsa3`), which the
archived handover measured as the single biggest remaining prefill item.  The **chunked GDN is
already arranged** (bf16 staged, WMMA, state in accumulators) and already beats his tiled kernel on
gfx1201; its remaining analogues are the bf16 stream marking and the depthwise conv1d fusion.
Follow our own trail: VEC stays the band/reference, packed-QSA lands prefill-only and opt-in, and
nothing defaults on until the purity gates pass.

**Implementation plan:** [`../packed-qsa/PORT-PLAN.md`](../packed-qsa/PORT-PLAN.md).
