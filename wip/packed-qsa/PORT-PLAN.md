# Packed-QSA port plan

**Status:** P3 done (2026-09-13) — the packed WMMA kernel is correct (rel 3e-5 vs VEC) and ~1.35x
faster at the op level, but **~2 % slower end-to-end** at pp8192 on gfx1201 (pack+merge -1.1 %,
kernel -0.9 %); see `P3-NOTES.md`.  Fork `packed-qsa` `4f464941a`.  P3.5 (now required) / P4 / P5.
P3 is next.
Branch: `packed-qsa` (delivery repo) / `packed-qsa` (fork).  Companion concept doc:
`../prefill-arrangements/README.md`.  Archived measurement:
`archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md`.

## 1. Objective

Replace the per-cell gather + VEC dot of our sparse QSA attention with a **packed-block WMMA**
kernel, following pwilkin's `qsa3`: merge a group of queries' top-k selections into a sorted,
block-aligned union, pack the F16 KV cache into contiguous 4-cell blocks once per graph, and run the
attention on the WMMA units.

Why this is the pick: it is the **single largest remaining qwen4exp prefill item** (archived
attribution: `flash_attn_qsa` 2842 ms vs his `qsa3` 730 ms at pp8192; ~75% of the ~2.5 s needed to
reach 1100 t/s, ~+117 t/s), and it is **quant-agnostic** — it needs an F16 KV cache, not a uniform
weight set, so it works on our mixed-expert UD-IQ3_XXS / UD-IQ4_XS / UD-Q4_K_XL checkpoints with no
PPL-per-bit trade.

## 2. Current implementation (what we are replacing)

| item | fact |
|---|---|
| op | `GGML_OP_FLASH_ATTN_QSA`; builder `ggml_flash_attn_qsa(q,k,v,idx,mask,scale,softcap,cell_vis,q_vis)` (`ggml/src/ggml.c:5629`) |
| sources | `src[0..4]` = q,k,v,idx,mask; `src[5],src[6]` = `cell_vis`,`q_vis` (derived visibility, block 15 V3) |
| shapes | q `[D,n_head,n_tps,n_stream]` (after `permute(0,2,1,3)`); k `[D,n_kv,n_kv_heads,n_stream]`; `idx [n_top_k,n_tps,1,n_stream]` I32; out `[D,n_head,n_tps,n_stream]` |
| kernel | `flash_attn_qsa<D,type_KV,softcap>` in `ggml/src/ggml-cuda/fattn-qsa.cu` |
| geometry | one block per (token column, top-k slice, stream, head chunk); `QSA_MAX_HEADS=16`, one warp per head, head chunk = `min(16, gqa)`; **32-cell tile staged into smem via a per-cell `idx` gather**; scalar `ggml_cuda_mad` (half2/bfloat162) dots; online softmax |
| slicing | decode (`base_blocks < nsm`) slices the top-k list across `gridDim.y`, each slice writes a partial + `(max,sum)`, a combine kernel merges (`GGML_CUDA_QSA_SLICES` override) |
| KV types | F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL (quantized dequantized while staging) |
| support | `ggml_cuda_flash_attn_qsa_supported(int device, const ggml_tensor *)` (`fattn-qsa.cu:724`); dispatch `ggml-cuda.cu:2568` / `:6893` |

Inefficiencies the pack removes: indirect per-cell gathers (re-done per head-chunk block), no reuse
across query columns, and scalar FMA instead of WMMA.

## 3. Reference design (pwilkin's `qsa3`)

Files: `strix-halo:ggml/src/ggml-cuda/qsa.cu` (443 lines), `strix-halo:src/models/pack.inc`,
`strix-halo:src/models/qwen4exp.cpp` (the `layout_prefill` gate + `src[6]/src[7]` wiring).

1. **Graph pack** (`src/models/pack.inc`, standard ggml ops, no CUDA):
   - `qsa_pack_keys`: F16 `[256,n_kv,n_kvh,1]` -> `reshape(16,16,4,n_kv/4*n_kvh)` ->
     `permute(0,2,1,3)` -> `cont` -> **`[16,4,16,n_kv/4*n_kvh]`**.
   - `qsa_pack_values`: F16 -> `reshape(256,4,nblocks)` -> `permute(1,0,2,3)` -> **`[4,256,nblocks]`**.
   - Built once per graph; enables reading a 4-cell block as one contiguous unit.
2. **`qsa3_rows_kernel`** (256 thr/query): validate/clamp each `idx` row to `[0,nk)`, rank-sort it if
   not non-decreasing, record the flag.
3. **`qsa3_merge_kernel`** (128 lanes / group of 4 queries): merge the 4 sorted rows into a
   block-aligned union; emit block ids (`ublk`), a 16-bit per-query membership mask (`umask`), and a
   count (`ucount`).
4. **`qsa3_attn_kernel`** (256 thr; `dim3(ngroups, k->ne[2])`): WMMA f16 (`wmma_f32_16x16x16_f16_w32`)
   over the packed blocks, **4 queries as three 16x16 tiles**, membership mask folded into the score,
   online softmax, F32 output.
5. **Gate** (his): `layout_prefill = n_tps>=128 && n_stream==1 && flash_attn && offload_kqv &&
   !softcap && maskless`; pack only when the KV type is **F16** and `ne[0]==256`, `ne[1]%4==0`,
   `ne[3]==1`; `qsa3 = packed_keys && packed_values && n_query>=128`.
   **No weight-type condition anywhere** — the requirement is F16 *cache*.

## 4. Our-trail port design

### 4.1 Op/interface
Extend `GGML_OP_FLASH_ATTN_QSA` with two optional sources — `src[7]=packed_keys`, `src[8]=packed_values`
— and let the backend pick packed vs VEC from their presence plus the shape/type gate.  (Alternative:
a new `GGML_OP_FLASH_ATTN_QSA3`; extending keeps one op and one graph builder.)  `GGML_MAX_SRC` has
room (we use 7 today).

### 4.2 Graph packing
Port `qsa_pack_keys`/`qsa_pack_values` into `src/models/qwen4exp.cpp` (or a `pack.inc`) as plain ggml
ops.  Build them only for the qwen4exp shape (D=256, F16 KV, `n_stream==1`, `n_query>=128`) and only
under the packed gate; otherwise leave them null and the op falls back to the VEC path.

### 4.3 Descriptor + attention kernels
Port `qsa3_rows_kernel`, `qsa3_merge_kernel`, `qsa3_attn_kernel` into our tree (a new
`ggml/src/ggml-cuda/qsa-packed.cu`, or inside `fattn-qsa.cu`).  Adapt:
- **arch fragments** — the reference uses the gfx11 `wmma_f32_16x16x16_f16_w32` with 16-half
  fragments; RDNA4 (gfx1201) uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` with **8-half**
  fragments (see `mma.cuh:1232` vs `:1239`).  The fragment load/permute code must be re-derived for
  gfx12.  This is the main portability risk.
- **our visibility** — the reference is `maskless` (derives from top-k).  We have both a base `mask`
  and the derived `cell_vis`/`q_vis` path; decide which the packed kernel consumes and keep the
  membership mask consistent with it.
- **output layout** — ours is `[D,n_head,n_tps,n_stream]` (un-permuted q); map his store accordingly.

### 4.4 KV-type / shape gate
Packed path only when: F16 KV, D=256, gqa 12, `k->ne[1]%4==0`, `n_query>=128`, `n_stream==1`.  Every
other case keeps the existing VEC kernel.  (Our VEC kernel's quantized-cache support is untouched.)

### 4.5 Purity
Default **prefill-only**: the packed kernel serves `n_query >= 128`; the decode/verify band
(`W=1..8`) stays on the VEC kernel, so `plain == draft-mtp` holds by construction.  The packed
reduction order differs from VEC, so this is a **prefill re-baseline** (document it; validate PPL vs
the dense/VEC reference and same-seed coherence).  A later band-uniform variant is possible but needs
its own `W=1..8` matrix.

## 5. Phased implementation

| phase | work |
|---|---|
| P1 ✅ | graph pack ops + the two new op sources + the gate (no kernel change yet) — `P1-NOTES.md` |
| P2 ✅ | port `qsa3_rows` + `qsa3_merge` (descriptor builder); unit-check the union/mask against the `idx` rows — `P2-NOTES.md` |
| P3 (part 1 ✅) | gfx12 f16 WMMA primitive + fragment-layout self-test validated; kernel body designed — `P3-DESIGN.md` |
| P3 (part 2) ✅ | implement the kernel body (grid/block, 8-way dim split, masking, online softmax, LDS P-transpose) — `P3-NOTES.md` |
| P3.5 (optional) | op optimisation: softmax shuffles, P transpose, K/V prefetch, wider key chunks |
| P4 | support predicate + dispatch + fallback; RDNA4 (gfx1201) and RDNA3.5 (gfx1151) |
| P5 | validate: correctness vs VEC, PPL, `W=1..8`, MTP acceptance, perf A/B |

## 6. Validation gates

1. Op-level correctness vs the VEC kernel on the production shape (same selected cells; NMSE within
   the `FLASH_ATTN_QSA` test tolerance) — the backend test already has 22 cases; add the packed path.
2. `test-backend-ops -o FLASH_ATTN_QSA` green on gfx1201 and gfx1151.
3. PPL vs the current VEC build (and the dense masked oracle) within noise.
4. Same-seed coherence; `W=1..8` logits matrix unchanged (packed is prefill-only).
5. MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`) on the packed-on build.
6. Perf: `flash_attn_qsa` op time at the production shape (target >= 2x), qwen4exp pp8192/pp16384
   (target ~+117 t/s).

## 7. Effort

| item | effort |
|---|---|
| graph pack + op sources + gate | 0.5-1 day |
| rows + merge kernels | 1 day |
| WMMA attention + gfx12 fragments | 2-4 days |
| dispatch/support + fallback | 0.5-1 day |
| validation (PPL/coherence/W/MTP/perf) | 1-2 days |
| **total** | **~6-9 days** |

## 8. Risks / open questions

1. **gfx12 f16 fragments** — the reference is gfx11-only; the 8-half vs 16-half WMMA fragment layouts
   must be re-derived/probed.  Highest risk; de-risk in P3 with a synthetic comparison.
2. **Cache layout** — confirm our `get_k()`/`permute(0,2,1,3)` cache is packable like his
   `original_keys` (contiguity, `ne[1]%4`, `ne[3]==1`).
3. **`idx` ordering** — `qsa3_rows` rank-sorts rows that are not non-decreasing; confirm our
   `ggml_indexer_top_k` output is block-aligned/sorted (his assumption).
4. **Visibility semantics** — maskless/derived `cell_vis` vs the membership mask; make sure the
   packed path reproduces the VEC path's visibility exactly (else it is a quality bug, cf. §21).
5. **Pack cost** — a per-graph contiguous copy of the used cache; measure that it is amortized.
6. **RDNA3.5 vs RDNA4** — pwilkin only built gfx11; gfx1201 needs its own validation.
7. **Exactness** — the packed path is a prefill re-baseline, not bit-identical; keep it opt-in until
   the PPL gate passes.

## 9. References

- concept: `wip/prefill-arrangements/README.md` (§1 QSA, §5 clarifications)
- archived measurement + next steps:
  `archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md` (§11)
- our kernel: `ggml/src/ggml-cuda/fattn-qsa.cu`, `ggml/src/ggml.c:5629`, `ggml/src/ggml-cuda/ggml-cuda.cu:2568`
- our graph: `src/models/qwen4exp.cpp` (`build_qsa_store_k` ~1038, `build_qsa_top_k` ~1244, the
  `qsa_sparse` call ~1652)
- reference: `pwilkin/llama.cpp` `strix-halo` — `ggml/src/ggml-cuda/qsa.cu`, `src/models/pack.inc`,
  `src/models/qwen4exp.cpp`
- WMMA fragments: `ggml/src/ggml-cuda/mma.cuh:1232` (gfx12), `:1239` (gfx11)
