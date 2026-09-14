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

## 5. Bottom line

The "chunked GDN + QSA equivalent" of pwilkin's weight arrangements exists, and it is mostly on the
**QSA** side: pack the selected KV into block-aligned f16 tiles and run WMMA (his `qsa3`), which the
archived handover measured as the single biggest remaining prefill item.  The **chunked GDN is
already arranged** (bf16 staged, WMMA, state in accumulators) and already beats his tiled kernel on
gfx1201; its remaining analogues are the bf16 stream marking and the depthwise conv1d fusion.
Follow our own trail: VEC stays the band/reference, packed-QSA lands prefill-only and opt-in, and
nothing defaults on until the purity gates pass.
