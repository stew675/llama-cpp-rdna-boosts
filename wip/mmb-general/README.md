# WIP: generalising `mmb` (bf16-WMMA dequant weight GEMM) beyond IQ4_NL

**Status: ACTIVE (opened 2026-09-19).  Not part of the delivery.**  Code lives in the
`~/llama-wip-mmb` worktree (branch `wip-mmb-general`, based on the `~/llama.cpp`
delivery tree at `8a2567e1e`); nothing here is in `patches/`.

## Why

On Strix Halo (gfx1151) our prefill is ~2x behind the tuned RDNA3_5 stacks
(pwilkin `strix-halo` full env 1349/1403 t/s on uniform IQ4_NL; halogen-flash
1246/1424 on the same GGUF files) because our weight GEMMs run the integer/vector
MMQ path while both use bf16 WMMA tensor cores.  The parked port
(`archive/work/wip-archive/iq4nl-prefill/mmb-port.patch`, +18.4 %) was IQ4_NL-only,
so it did little for the delivery's own models.  This picks it back up in a
**general-purpose** form.

**GFX1201 note (why it was shelved):** the `mmb` kernels use the **first-gen gfx11**
WMMA builtin `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`.  gfx12 needs
`..._bf16_w32_gfx12` (see `gated_delta_net_chunked_bf16_gfx11.cu` vs
`gated_delta_net_chunked_bf16.cu` in the delivery).  So this is RDNA3-gated by
construction; gfx1201/gfx1100 keep the existing MMQ/QSA path.

## Done (2026-09-19)

- `mmb_dq_row_q4k` / `mmb_dq_row_q5_1` — on-the-fly bf16 LDS dequant, no bf16 shadow
  (a 120 GiB model cannot afford one for its experts).
- `WTYPE 3` (Q4_K) and `WTYPE 4` (Q5_1) in `mmb_tile_gemm`, `mmb_tile_gemm_glu`,
  `mmb_dense_kernel`, `mmb_routed_kernel` and `mmb_routed_glu_kernel`.
- `ggml_cuda_mmb_supported_mm` / `_mmid` / `_glu` accept Q4_K/Q5_1 (K%256 guard for Q4_K).
- **Gate: RDNA3_5 only by default** (`GGML_CUDA_CC_IS_RDNA3_5`).  RDNA3_0 shares the gfx11 WMMA
  builtin but is untested, so it takes `GGML_CUDA_MMB_RDNA3=1` to open.  gfx12 is excluded (needs the
  `_gfx12` builtin).
- **Multi-arch build safe**: the WMMA calls go through `mmb_wmma_bf16`/`mmb_wmma_f16` wrappers;
  the RDNA4 branch is a deliberate no-op (runtime-gated off there), so `mmb.cu` compiles for gfx1201
  (verified on the recorded compile command) and gfx1151 is unchanged.
- Graph optimizer fusions (MoE pair, SWIGLU->mmq) now stand down only when MMB will
  actually take **that weight type** (`ggml_cuda_mmb_dense_will_take` /
  `_routed_will_take`) — resume-checklist item #5.

### Supported weight types (2026-09-19)

| type | dense | routed (expert) + GLU | note |
|---|:--:|:--:|---|
| IQ4_NL | ✓ | ✓ | the original port |
| Q8_0 | ✓ | ✓ | routed added; dense is the PLE table |
| Q4_K | ✓ | ✓ | Q4_K_M experts |
| Q5_1 | ✓ | ✓ | |
| Q5_K | ✓ | ✓ | UD-Q5_K_M experts |
| Q6_K | ✓ | ✓ | on-the-fly now (no 6 GiB shadow needed) |
| IQ4_XS | ✓ | ✓ | MiniMax-M3 UD-IQ4_XS 35 %, UD-Q3_K_M down experts |
| IQ3_S | ✓ | ✓ | IQ4_XS experts |
| Q3_K | ✓ | ✓ | Q3_K_S/M/L all map to this tensor type |
| IQ3_XXS | ✓ | routed only | fused GLU is a net loss, so it is default-off (`GGML_CUDA_MMB_IQ3XXS=1`) |
| F16 / BF16 | — | — | already tensor-core via `mmf` WMMA (no MMB needed) |
| F32 | ✓ (f32split) | — | f16 hi/lo WMMA |

Types with **no representation in current ggml** (named in older llama.cpp READMEs, checked
2026-09-19): `IQ3_M`, `IQ3_XS`, `IQ4_S`, `IQ4_M`.  `Q2_K`/`IQ2_*`/`IQ1_*` are deliberately out of
scope (quality).

### Measured (gfx1151, ROCm 7.14, `-ub 2048` bf16 KV; PPL on `prompts/prose-rdna-boosts.txt`, `-c 2048`)

| model / test | MMB off | MMB on |
|---|---:|---:|
| **Q4_K_M pp2048** | 604.6 | **1015.2 (+68 %)** |
| **Q4_K_M pp8192** | 568.9 | **888.1 (+56 %)** |
| Q4_K_M PPL | 10.3328 | 10.2716 |
| IQ4_XS pp2048 | 723.3 | 838.0 (+15.9 %) |
| IQ4_XS pp8192 | 685.5 | 779.5 (+13.7 %) |
| IQ4_XS PPL | 10.6938 | 10.6440 |
| Gemma4-26B-A4B Q8_0 pp2048 | 2058.5 | 2340.1 (+13.7 %) |
| Gemma4-26B-A4B Q8_0 pp8192 | 1723.5 | 1928.5 (+11.9 %) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp2048 | 2045.3 | 2303.0 (+12.6 %) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp8192 | 1917.8 | 2141.3 (+11.7 %) |
| Qwen3.6-35B-A3B UD-Q5_K_M PPL | 14.4349 | 14.3907 |
| Qwen3.6-35B-A3B Q6_K pp2048 | 2128.8 | 2143.2 (+0.7 %) |
| Qwen3.6-35B-A3B Q6_K pp8192 | 1974.6 | 2012.3 (+1.9 %) |
| Qwen3.6-35B-A3B Q6_K PPL | 14.3682 | 14.3296 |
| Qwen3.6-35B-A3B UD-Q3_K_M pp2048 | 2091.9 | 2295.2 (+9.7 %) |
| Qwen3.6-35B-A3B UD-Q3_K_M pp8192 | 1941.6 | 2120.2 (+9.2 %) |
| Qwen3.6-35B-A3B UD-Q3_K_M PPL | 14.6517 | 14.6248 |

The big Q4_K_M jump is the MoE expert GLU + routed down on WTYPE 3; IQ4_XS is IQ3_S gate/up +
IQ4_NL down; the Q5_K_M and Gemma4 Q8_0 gains are their expert types.  Q6_K moves little because its
MMQ path is already efficient on this workload.  UD-Q3_K_M moves little because only its **down**
experts are IQ4_XS (the gate/up are **IQ3_XXS**, still unsupported).  PPL parity everywhere says the
dequants are correct.

**Model composition note:** the file names mislead.  *Qwen3.8-Flash-Next UD-IQ4_XS* is by bytes
IQ4_NL 52 % + IQ3_S 36 % + Q8_0 9.5 %, with the IQ4_XS *type* only 1 %.  *MiniMax-M3 UD-IQ4_XS* is
IQ3_S 56 % + IQ4_XS 35 % + Q8_0 + Q6_K — now fully covered.

### Post-MMB profile (Q4_K_M pp8192, total 17.75 s, was ~28 s)

`flash_attn_qsa` **2.94 s (16.6 %)** is now the largest kernel; then `mmb_dense_kernel` 4.06 s
(Q8_0 PLE + Q5_1), `mmb_routed_glu` 2.46 s, `mmb_routed` 1.39 s, HC pre+post 1.44 s,
`mmb_cvt` 0.65 s, `mmb_f32split` 0.65 s.  Our VEC QSA already uses `v_dot2_f32_f16`, so its gap is
algorithmic (per-token gather + VEC vs packed-block WMMA), not instruction selection.

## Next

1. **QSA v3 packed-WMMA attention** — now the #1 kernel.  Plan: graph-side `qsa_pack_keys`/`_values`,
   the `qsa3_rows`/`qsa3_merge` block descriptor, then the `qsa3_attn_kernel` WMMA; prefill-only
   (`n_query >= 128`), VEC kept for the W=1..8 band.  Estimated ~1017 -> ~1160 t/s on Q4_K_M.
2. **Q8_0 IU8-WMMA** (int8 weights/activations straight to the int8 tensor cores, no dequant) — the
   biggest single weight cost is now the Q8_0 dense path (23 %), and Q8_0 is already int8.  Profile
   whether it is compute- or memory-bound first (the PLE table is lazily read).
3. bf16-producer marking (kills `mmb_cvt`) and the HC prefill fusion.
4. Optional, only after gfx1151 is exhausted: a single 7900 XTX (gfx1100) small-model test with
   `GGML_CUDA_MMB_RDNA3=1` — the tiling likely needs an RDNA3_0 pass.

## QSA / Q8_0 investigation (2026-09-19)

Both post-MMB leaders were probed with cheap experiments before committing to a rewrite:

* **Q8_0 dense (23 %)** — shapes logged: `attn_qkv M=10240 K=2560`, `attn_gate M=6144 K=2560`,
  `ssm_out M=2560 K=6144`, `hc_* M=320/10240`, `ffn_*_shexp M=640 K=2560`.  Forcing the narrow or wide
  MMB tile changes IQ4_XS pp8192 by +0.6 % / -2.4 % (802 / 798 / 778 t/s), so the `M>=6144` heuristic
  is already optimal — the kernel is **occupancy/LDS-bound, not tiling-bound**.  The next move is an
  **int8 IU8-WMMA** variant (int8 weights+activations straight to the tensor cores; less LDS than the
  dequant-to-bf16 path), not tile tuning.
* **QSA (16.6 %)** — the f16/bf16 gather widened 8B -> 16B is **neutral** (796 vs 798 t/s), so it is
  not load-issue-bound.  It is compute/reduction/occupancy-bound: the VEC kernel does 1 query column
  x 16 heads per block and reduces with `v_dot2`, while a 16x16x16 WMMA tile does 16 heads x 16 cells
  per instruction — that is the `qsa3` gap.

Neither is reachable by tuning; both need a kernel restructure (QSA v3 below, and an IU8 Q8_0 path).
Diagnostics are left gated in the tree: `GGML_CUDA_MMB_LOG=1` (shape log),
`GGML_CUDA_MMB_TILE=0/1` (tile override).

## qsa3 — packed-block WMMA prefill for the QSA sparse attention (2026-09-19, session 2)

**DONE and validated** (was `NEXT WORK #1`).  `GGML_CUDA_QSA3=1` opts in; default OFF.

### What it is

The VEC kernel (`fattn-qsa.cu`) walks the top-k list cell by cell with `v_dot2` and one query
column per block.  `fattn-qsa3.cu` (new, ported from the Strix Halo branch's `qsa-attn`) shares a
block of work across **G = 4 queries x 12 q-heads** (48 output rows) and runs the score and PV
passes on the **F16 WMMA tensor cores** over a package of 4 key blocks (16 keys) at a time.

Three kernels: `qsa3_rows_kernel` (per-row sortedness check + rank-sort), `qsa3_merge_kernel`
(merge 4 queries' rows into a sorted, deduplicated, block-aligned union + a 16-bit per-query
membership mask), `qsa3_attn_kernel` (16x16x16 F16 WMMA, mask folded into the score pass).

### Plumbing

* The pack is a **pure graph composition** (reshape + permute + cont) - **no new ggml op**.
  Helpers `qsa_pack_{keys,values}_graph` in `src/models/qwen4exp.cpp`.
* The op gained two optional srcs: `ggml_flash_attn_qsa_set_packed(a, packed_keys, packed_values)`
  (`src[7]`/`src[8]`).  NULL/NULL = the VEC path, unchanged.
* **Every KV type is supported**, because the kernels read only the F16 packs.  The cast must happen
  on the cache's *natural contiguous* view before any permute (`qsa3_f16_cast`), and the quantized
  types route through F32 - **the backend `dup` only dequantizes quantized->F32 and cannot permute a
  quantized tensor at all** (getting this wrong aborts in `ggml/src/ggml-cpu/ops.cpp:578`, once per
  QSA layer).
* **Prefill-only by construction**: the support check requires `q->ne[1] >= 128` and RDNA3_5, so the
  whole W = 1..8 decode/verify band keeps the VEC kernel and width purity is untouched.
* Portable WMMA wrapper (`qsa3_wmma_f16`, no-op on `RDNA4`) so a multi-arch build still compiles -
  verified for **gfx1201 and gfx1100** as well as gfx1151.

### Results (gfx1151, ROCm 7.14, IQ4_XS, `-b/-ub 2048`)

| KV type | pp4096 off -> on | pp8192 off -> on |
|---|---:|---:|
| f16 | 855.5 -> 870.5 (+1.8 %) | 819.4 -> 861.7 (+5.2 %) |
| bf16 | 836.9 -> **874.8 (+4.5 %)** | 791.2 -> **854.5 (+8.0 %)** |
| q8_0 | 817.5 -> **872.9 (+6.8 %)** | 780.5 -> **856.6 (+9.8 %)** |

All three converge at depth - the kernel reads the same F16 packs, so the KV type no longer matters
for the attention arithmetic.

**PPL parity** (wikitext):

| c | VEC | qsa3 |
|---|---:|---:|
| 16384, bf16 | 3.3932 | 3.3900 |
| 16384, q8_0 | 3.3861 | 3.3879 |
| 16384, f16 | 3.3883 | 3.3869 |
| 32768, bf16 | 4.3378 | 4.3353 |

All within +/-0.002 (the run's own error bar is +/-0.027).  Greedy text is coherent and agrees for
~40 tokens before the approved **prefill re-baseline** near-tie flip.

### Kernel profile (rocprofv3, pp8192)

| kernel | VEC | qsa3 |
|---|---:|---:|
| attention | 2944.4 ms (`flash_attn_qsa`) | **682.3 ms** (`qsa3_attn_kernel`) |
| rows / sortedness | - | 441.0 ms (`qsa3_rows_kernel`) |
| merge / union | - | 28.6 ms (`qsa3_merge_kernel`) |
| **total** | **2944.4 ms** | **1151.9 ms (2.56x)** |

### Two findings worth not re-deriving

* **The dense startup portion must stay.**  `LLAMA_QSA_DENSE_SHORTCUT` (default ON) sends
  `n_kv <= indexer_top_k + r - 1` (= 2051) to the dense arm, because there the selection covers every
  cell so sparse saves nothing while paying the indexer.  Re-measured with qsa3: forcing QSA in the
  startup region is a **pessimization** (pp2048 912.5 -> 903.2; pp8192 862.5 -> 856.9), so the
  shortcut stays.  **Consequence: qsa3 only engages at n_kv > 2051**, so a `-c 2048` PPL test proves
  nothing about it (it takes the dense arm silently).
* **The top-k rows are UNSORTED**, so `qsa3_rows_kernel`'s rank-sort is on the critical path and is
  the single largest remaining qsa3 cost (441 ms of 1152 ms).  Proven by disabling the sort: the
  rows kernel drops 441 -> **8 ms** but the attn kernel explodes 682 -> **19593 ms** and throughput
  collapses 866 -> 474 t/s (unsorted rows break the merge kernel's binary searches).  So the sort is
  necessary and **the PPL parity above did exercise and validate it**.

### Next qsa3 optimization (deferred)

Make the sortedness path cheaper - either emit sorted rows from the indexer (`ggml_indexer_top_k`;
but the `idx` order is shared with the VEC decode path, so a re-order there is a decode numerics
change and must not be done casually), or replace the O(ns^2) rank-sort with a block-aware sort
(the selection is by whole 4-key blocks, so sorting the ~ns/4 block ids and expanding is ~16x less
work).  Target: ~441 ms -> ~30 ms, worth ~4 % of prefill.

### Where the time goes now

With qsa3 on, the pp8192 profile is led by **`mmb_*` kernels** (`mmb_f32split_kernel` 2164 ms,
`mmb_routed_glu_kernel` 2085 + 1626 ms, `mmb_dense_kernel` 1904 + 1606 ms, `mmb_cvt_f32_bf16`
648 ms) - QSA is no longer the #1 kernel.  The MMB follow-ups (SS 7-9) are now the bigger lever.

## Gates before this could be opt-in, let alone defaulted on (from the parked handover)

- W = 1..8 logits matrix with `GGML_CUDA_MMB=1` == off (prefill-only, `T >= 512`).
- MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
- `test-recurrent-state-rollback`; `test-backend-ops` suites.
- Same-seed prefill re-baseline documented; gfx1100/gfx1201 compile + consistency.
