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
| f16 | 855.8 -> 893.8 (+4.4 %) | 816.0 -> 882.2 (+8.1 %) |
| bf16 | 834.2 -> **899.8 (+7.9 %)** | 791.9 -> **884.3 (+11.7 %)** |
| q8_0 | 827.1 -> **896.4 (+8.4 %)** | 784.6 -> **873.5 (+11.3 %)** |

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
| attention | 2944.4 ms (`flash_attn_qsa`) | **674.9 ms** (`qsa3_attn_kernel`) |
| rows / sortedness | - | 25.5 ms (`qsa3_rows_kernel`) |
| merge / union | - | 28.2 ms (`qsa3_merge_kernel`) |
| **total** | **2944.4 ms** | **728.6 ms (4.04x)** |

The rows figure is after the 2026-09-19 bitmap-sort rewrite below (it was 441.0 ms and the qsa3
total 1151.9 ms / 2.56x before it).

### Two findings worth not re-deriving

* ~~**The dense startup portion must stay.**~~ **SUPERSEDED 2026-09-19 - see the always-QSA section
  below; the shortcut is now default OFF.**  The original measurement (forcing QSA in the startup
  region was a pessimization, pp2048 912.5 -> 903.2) was taken *before* the bitmap sort, when the
  startup regime also paid the 441 ms rank-sort.  After the sort fix the same regime measures:
  **qsa3 137.8 ms vs dense `flash_attn_ext_f16` 149.9 ms (qsa3 8 % faster on the attention even when
  every cell is selected)**, with indexer+top-k 20.9 ms and qsa3 rows+merge 6.6 ms on the QSA side.
  So the path only still lost because the indexer cost more than the kernel saved.
* **The top-k rows are UNSORTED**, so `qsa3_rows_kernel` must sort them.  Proven by disabling the
  sort: the rows kernel drops 441 -> **8 ms** but the attn kernel explodes 682 -> **19593 ms** and
  throughput collapses 866 -> 474 t/s (unsorted rows break the merge kernel's binary searches).  So
  the sort is required and **the PPL parity above did exercise and validate it**.

### `qsa3_rows_kernel` sort rewrite - DONE (2026-09-19)

The rank sort was O(ns^2) and dominated qsa3 (441 ms of 1152 ms).  It is now a **bitmap counting
sort**: the row is a *set* of cell ids, so a `nk`-bit presence bitmap + a popcount scan enumerates
it in ascending order - **exactly the order the rank sort produced** - for O(ns + nk/32) per row.

Get the data first: an early assumption that the row is a set of whole 4-key blocks was **wrong**.
The real rows (dumped from a live run) are a handful of **long contiguous runs** - row0 is one run of
2051 keys, row2 is runs of 436/1611/4, i.e. 1-6 runs per row - so the bitmap is dense and the
popcount scan is cheap.  (This also explains why the row looks like "whole 4-key blocks" to an
aligned-group scan: a long run of consecutive keys has `ent[i] == ent[i-1]+1` everywhere.)

Implementation notes: the `nk`-bit bitmap plus 256 per-lane scan offsets share the rows kernel's
dynamic smem after the key array (48 KiB budget, i.e. `nk` up to ~1.5M); the kernel takes
`bitmap_words` and **falls back to the original rank sort when it is 0** (too large a cache).
Sentinels are appended after every valid key, matching the rank sort's placement.

**Result: rows 441.0 -> 25.5 ms (17x); qsa3 total 1151.9 -> 728.6 ms (4.04x vs VEC).**  PPL is
**bit-identical** to the pre-rewrite build (c16384 3.3900, c32768 4.3353, q8_0 3.3879) - the sort is
order-exact, not merely equivalent.

### Where the time goes now

With qsa3 on, the pp8192 profile is led by **`mmb_*` kernels** (`mmb_f32split_kernel` 2164 ms,
`mmb_routed_glu_kernel` 2085 + 1626 ms, `mmb_dense_kernel` 1904 + 1606 ms, `mmb_cvt_f32_bf16`
648 ms) - QSA is now **728.6 ms (4th-ish)** and no longer the #1 kernel.  The MMB follow-ups
(SS 7-9) are the bigger lever.

## Always-QSA prefill + F32 dense weights off (2026-09-19, session 4)

Two default flips, both measured; plus the revert of a failed experiment.

### 1. `LLAMA_QSA_DENSE_SHORTCUT` default ON -> OFF = **always QSA** (maintainer decision)

The shortcut sent `n_kv <= indexer_top_k + r - 1` (= 2051) to the dense masked FA arm on the
reasoning that there the top-k selects *every* cell, so sparse attention saves no work while still
paying the indexer.  **qsa3 changed that.**  Measured in the fully-dense startup regime (pp2048,
single ubatch, every cell selected, gfx1151, `rocprofv3`):

| | attention kernel | indexer + top-k | qsa3 rows/merge | total |
|---|---:|---:|---:|---:|
| dense (`shortcut=1`) | 149.9 ms (`flash_attn_ext_f16`) | - | - | **149.9 ms** |
| QSA (`shortcut=0`) | **137.8 ms** (`qsa3_attn_kernel`) | 20.9 ms | 6.6 ms | **165.3 ms** |

So qsa3's kernel is already **8 % faster than the dense FA kernel even when nothing is skipped** -
the path only still lost because the indexer + top-k cost 20.9 ms against the 12.1 ms the kernel
saved.  End to end the flip is ~neutral:

| pp | always-QSA | dense-shortcut | delta |
|---|---:|---:|---:|
| 512 | 703.8 | 700.3 | +0.5 % |
| 1024 | 839.0 | 841.5 | -0.3 % |
| 2048 | 909.3 | 919.4 | -1.1 % |
| 4096 | 904.0 | 913.9 | -1.1 % |
| 8192 | 900.1 | 902.9 | -0.3 % |

(within ~1 % run variance for most points).  What it buys: **no numerics seam at `n_kv == width`**,
and qsa3 is now exercised at *every* context length - a `-c 2048` PPL exercise used to be silently
dense, which is why the early "qsa3 is neutral" readings were vacuous.  **Decode is unaffected**: it
stays dense via the existing arch policy (`qsa_dense_decode_until` = 64K on gfx1151, always on
gfx1201) - verified `tg64` shallow 25.80 -> 25.83.

PPL moves the right way: c16384 bf16 **3.3900 -> 3.3821**, q8_0 3.3879 -> 3.3861; c32768 bf16
4.3353 -> 4.3397 (noise).  Greedy text coherent.  `LLAMA_QSA_DENSE_SHORTCUT=1` restores the dense
arm (still the `LLAMA_QSA_SPARSE_FA=0` cross-check).

**Remaining QSA gap = the indexer**, not attention: `indexer_topk_radix_histogram` 10.0 ms,
`indexer_topk_deterministic_write` 4.7, `indexer_topk_count` 3.3, `indexer_topk_radix_select` 2.9
(pp2048, n=96 launches each).  Halving that makes always-QSA a win even at pp2048.

### 2. `GGML_CUDA_MMB_F32SPLIT` default 2 -> 0 (MMB F32 dense weights off)

The F32 dense weights are all **tiny-M**: MoE router `ffn_gate_inp` M=512, `ssm_alpha`/`ssm_beta`
M=48, `hc_*_inject` M=4, `ffn_gate_inp_shexp` M=1 (per-layer inventory via `GGML_CUDA_MMB_LOG=1`).
Both paths cost ~1.0 s at pp8192 (12 % of prefill):

* `mmb_f32split_kernel<128,128,32,64>` computes a padded **128-row A tile**, so M=4 wastes 32x of its
  WMMA work (for `hc_attn_inject` M=4/K=10240 the launch computes ~53 GFLOP of WMMA for a 0.17 GFLOP
  problem) - 2164 ms in the profile, `n=2160`;
* `F32SPLIT=0` (rocBLAS, `Cijk_Alik_Bljk_SB_MT32x32x8_...`) lands on the same shape bound - 2076 ms,
  `n=1784`.

Both are ~10x off the memory-bound floor (A traffic = `(T/BN)*M*K*4`, B traffic = `(M/BM)*T*K*4`;
for M=512/T=2048 that is 168 MB against a 26 MB floor).  Since rocBLAS measured *faster*,
the MMB default is now **off**: IQ4_XS pp4096 896.0 -> **915.9**, pp8192 896.8 -> **902.9**.
`GGML_CUDA_MMB_F32SPLIT=1` opts it back in.

**Tried and rejected:** a 16x256 small-M tile for `M <= 64` (aimed at the 32x padding).  It is
**worse** - 870/847 t/s vs 895/885 for the 128 tile - because `BN=256` halves the block count in a
kernel that is already parallelism-starved, and `BM=16` does not help M=512.  Reverted; the real fix
is a dedicated tiny-M (or split-K) kernel.

## UPDATE — session 5 (2026-09-19): the fresh profile + the shape-aware F32 split

### Fresh post-session-4 profile — reprioritises the list

`rocprofv3` kernel trace, MMB + QSA3 on, IQ4_XS `-b/-ub 2048`, in % of total kernel time:

| family | pp8192 (17.77 s) | pp32768 (77.02 s) |
|---|---:|---:|
| `mmb_*` | 52.0 % | 48.2 % |
| **F32 rocBLAS** (`Cijk_...`) | **11.7 %** | **12.0 %** |
| `qsa3` attn/rows/merge | 4.9 % | 6.4 % |
| `rms_norm_f32` | 5.9 % | 5.5 % |
| `dsv4_hc_pre` + `_post` | 8.0 % | 7.4 % |
| PACK/copy (qsa3 pack) | 3.0 % | 3.5 % |
| **indexer top-k family** | **0.99 %** | **3.2 %** |
| indexer score | 0.98 % | 3.2 % |

**This contradicts the "indexer is next-work #1" ordering carried in from session 4.**  The indexer
is ~1 % at 8K (it was 0.5 % at the pp2048 startup regime that ordering was based on) and only reaches
3.2 % at 32K.  It does scale with `n_kv x n_tps` while its output is capped at 2051 cells, so it
matters at depth - but the **F32 path is 12 % at *both* depths** and was the larger target.  The
indexer is deferred, not dropped.

### The fix: F32 dense weights, split on shape (+2.1 / +2.6 / +3.1 %)

The F32 dense weights were all-or-nothing between rocBLAS and the MMB f32split tile, and the two
paths disagree about *which shape* each wins - so the previous default was a wash (mode 2 was even the
worst).  Per-shape, from the trace (the launcher grid is `ceil(M/128) x ceil(T/128)` workgroups, so
the launch identity is exact):

| shape | rocBLAS | MMB f32split | winner |
|---|---:|---:|---|
| `ffn_gate_inp` M=512 K=2560 | 2.044 ms | **0.846 ms** | **MMB 2.4x** |
| `hc_attn/ffn_inject` M=4 K=10240 | **1.332 ms** | 1.825 ms | rocBLAS 1.37x |
| `ssm_alpha/beta` M=48 K=2560 | **0.359 ms** | slower | rocBLAS |

**The discriminator is M alone.**  Mode 1 (new default) takes MMB only when `M >= 128`:

| pp | mode 0 (all rocBLAS) | **mode 1 (M>=128)** | mode 2 (all MMB) |
|---|---:|---:|---:|
| 2048 | 914.87 | **933.92 (+2.1 %)** | 901.73 |
| 4096 | 908.93 | **933.03 (+2.6 %)** | 901.01 |
| 8192 | 901.99 | **929.75 (+3.1 %)** | 902.97 |

F32 GEMM total 1947 -> ~1537 ms.  Why rocBLAS wins the small-M/short-K shapes: its `MT32x32x8`
kernel split-Ks hard (M=512/K=2560 launches 262144 blocks for ~1.0M outputs, ~64 threads per output,
so it pays partial-sum traffic) and loses the 2.4x there; on M=4/K=10240 the 128-row MMB tile wastes
WMMA rows and loses.

**Numerics parity**: PPL c16384 bf16 3.3821 (mode 0) vs 3.3875 (mode 1), error bars ±0.027.  Decode
untouched (tg64 25.95 vs 26.00) - the `T >= 512` gate is unchanged, so `W=1..8` purity still holds by
construction.  Greedy text coherent.

**Lesson (do not relearn): a first attempt added "or `K >= 4096`" to the rule, expecting the WMMA
path to help the K=10240 hc inject pair.  It is wrong and costs 375 ms.**  The trap that hid it:
bucketing launches by grid alone put `hc_inject` (760) + `ssm` (576) + others into one 1.065 ms
average that looked like an MMB win.  **Split the bucket before believing a per-shape number.**

## UPDATE — session 5b (2026-09-19): tiny-M F32 kernel for the hc `*_inject` GEMMs (+2.3-3.3 %)

Next-work #1 from the session-5 list (the remaining F32 tiny-M) is done.

**The shape:** `hc_attn_inject` / `hc_ffn_inject` are M=4 (hc = the hyperconnection stream count),
K=10240, T=2048 — the largest remaining F32 cost at **1012 ms, 1.330 ms/launch, 5.7 % of prefill**.

**Why neither existing tile could serve it:** M=4 gives no M parallelism, so rocBLAS launches
`ceil(T/32)=64` blocks and the 128-row MMB f32split tile pads the A panel 32x.  Both read the 84 MB
activation exactly once (memory floor ~0.33 ms) yet sit at **~63 GB/s**, while `rms_norm_f32` on the
same part sustains **~330 GB/s** (1044 ms for 84 MB read + 84 MB write per launch).  So it is a
**parallelism wall**, not a bandwidth or arithmetic one — 16-64 blocks cannot keep enough loads in
flight.  (The two injects also cannot be fused: they consume different `xn`, pre-attn vs post-attn.)

**The fix:** `mmb_tiny_m_f32_kernel` — one warp per token, every lane accumulating a k-strided
partial *for all M rows*, so X is read once, coalesced (consecutive lanes read consecutive float4)
and reused across M in registers.  256 blocks at T=2048 instead of 16-64.

| | per launch | GB/s |
|---|---:|---:|
| rocBLAS | 1.330 ms | 63 |
| MMB f32split | 1.825 ms | 46 |
| **tiny-M warp-per-token** | **0.500 ms** | **168** |

End to end: **pp2048 933.0 → 964.1 (+3.3 %), pp4096 934.4 → 962.0 (+2.9 %), pp8192 934.6 → 955.9
(+2.3 %)**.  hc_inject total 1012 → 568 ms.  PPL c16384 bf16 3.3875 → **3.3851** (noise); decode
identical (tg64 25.96 both — M=4 here is a weight row count, not a token count, so the `T >= 512`
gate still keeps the whole decode/verify band off this path by construction).

**Tuning, both negative results recorded:**

* **Tokens-per-warp (TT): TT=1 is best.**  TT=2 (954.8) and TT=4 (952.3) measured *worse* than TT=1
  (957.4) at pp8192.  The motivation was W traffic — each lane walks a k-strided slice so a warp
  collectively reads all 164 KB of W (336 MB from L2 at 2048 warps) against X's 84 MB from DRAM.
  L2 serves the panels well enough that the extra live registers and reduced block count cost more.
* **MMAX specialisation: +0.5 %.**  The real M is 4, so `<4,TT>` (not `<8,TT>`) is the right
  instantiation; `<8,1>` is kept as the general path for M in 5..8.

**The trap worth carrying forward:** the first version changed only the *launcher* and measured
neutral (+/-0.3 %) — because the `M >= 128` rule I had just added rejected the shape in the *gate*
before the launcher was ever reached, so the kernel never ran.  A bench delta alone would have read
as "the idea failed".  **The kernel trace is what caught it: the symbol was simply absent.**  When a
new kernel measures flat, verify it actually executed.

## Gates before this could be opt-in, let alone defaulted on (from the parked handover)

- W = 1..8 logits matrix with `GGML_CUDA_MMB=1` == off (prefill-only, `T >= 512`).
- MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
- `test-recurrent-state-rollback`; `test-backend-ops` suites.
- Same-seed prefill re-baseline documented; gfx1100/gfx1201 compile + consistency.
