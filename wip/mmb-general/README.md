# WIP: generalising `mmb` (bf16-WMMA dequant weight GEMM) beyond IQ4_NL

**Status: ACTIVE (opened 2026-09-19).  Not part of the delivery.**  Code lives in the
`~/llama-wip-mmb` worktree (branch `wip-mmb-general`, based on the `~/llama.cpp`
delivery tree at `8a2567e1e`); nothing here is in `patches/`.

> **New session?  Read [`HANDOVER.md`](HANDOVER.md) first** — its "FOR THE NEXT SESSION" brief at the
top is the self-contained handoff (environment, build/run, gates, the prioritized remaining work).
> This file is the running (dated) record.

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

## UPDATE — session 13 (2026-09-20): the `dsv4_hc_pre`/`_post` residual was L2/MALL pollution —
## non-temporal accesses close it (**−18.6 % pre, −31 ms post**), bit-identically

Session 5e left the `dsv4_hc_pre` residual as "~18 % kernel-local headroom" (a probe said the
pattern can do 232.6 GB/s, the kernel did 197) with "address arithmetic / strided dst write" as the
remaining suspects.  This session re-measured it in the bf16 era and found the real cause: **the
kernel is not slow, its buffers are polluting the cache the surrounding GEMM weight streams need**.

**The measurement that cracked it.**  A faithful standalone copy of the exact kernel
traffic/shape (n_embd=2560, hc=4, nt=2048; bf16 x + bf16 gate read, F32 dst + bf16 dst16 written)
runs at **0.434-0.535 ms** (215 GB/s), but the kernel in the model takes **0.605 ms** (190 GB/s).
The same trace shows `dsv4_hc_post` at 206 GB/s on the same machine, so the environment is not
bandwidth-limited.  Tiling is not it either: `vec4`/`vec8` (2D grid, `uint2`/`uint4` loads, proved
bit-identical) are both **worse in situ** (625 / 642 us vs 605).  What fixed it was making the bulk
accesses **non-temporal** (`__builtin_nontemporal_load`/`_store`, value-preserving hints):
`dsv4_hc_pre_f32` **605 -> 493 us/call (-18.6 %, 190 -> 234 GB/s)** and `dsv4_hc_post_f32`
**918 -> 880 us/call (691 -> 660 ms over 752 calls)**.  All-kernel total 15596 -> 15501 ms.

The hint is a no-op in the standalone microbench (there is no competing traffic there) — the same
reason session 5e/6 could not find it.  **Methodology rule: a microbenchmark that isolates a kernel
can miss a real in-situ win, because the win is about coexisting with the rest of the model.**

It is **bit-identical** (cache hints only): PPL c2048 stays **10.6015**, greedy
**`9c281c415082`**, `FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK, width probe PASS,
`plain == draft-mtp` byte-identical.  e2e is within run noise (pp2048 1041.6 -> 1047.2, pp8192
1015.6 -> 1017.9, same-session interleaved) — judge it on the kernel trace, per the standing rule.

**Two things ruled out on the way:** (1) `vec4`/`vec8` beat the scalar kernel standalone (~5 %) but
lose in situ, so the load-width/ILP line is closed for good; (2) `mixed` (the `dsv4_hc_pre` output)
is written as F32 **and** bf16 because `all_bf16_consumers` rejects it: its consumers include the GDN
`ssm_alpha/beta` `[2560x48]` F32 GEMM and the MoE router F32 GEMM (plus the expert GEMMs, which are
bf16).  Making it BF16-only would save ~21 MB/call but requires changing the GDN recurrence and MoE
routing input numerics — out of scope, documented as a follow-up.

**Broader follow-up (not done):** the same non-temporal treatment is untested on the other large
pure-streaming kernels (`concat_transposed_src1_dim0` 358 ms, `moe_weighted_reduction` 384 ms,
`ssm_conv_long_token_f32` 292 ms, the qsa3 pack).  It must **not** be applied blindly — kernels that
reuse data (the K/V cache in `qsa3_attn`, the weight panels in `mmb_*`) may lose.  Judge each on
`rocprofv3` kernel time.

## UPDATE — session 12 (2026-09-20): the HC normalized stream `xn` is now BF16-only — **+4.4 % pp2048 /
## +3.8 % pp8192** over session 11, a numerics change (PPL re-baselined)

Session 11's dead-F32-store skip stopped at `xn`: its two delayed consumers, `dsv4_hc_pre` src[0] and
the tiny-M `hc_*_inject` F32 GEMMs, still read F32, so the graph only set `bf16_copy` and the fused
`rms_norm+mul` kept writing the F32 output.  This session gives both consumers a BF16 arm and lets
`xn` be marked **BF16-only**:

* `dsv4-hc.cu`: `dsv4_hc_pre_f32` gains an `xbf16` template arm; when `xn` is BF16-only the launcher
  reads the bf16 slot instead of the (never written) F32 tensor.
* `mmb.cu`: `mmb_tiny_m_f32_kernel` gains an `XBF16` arm (the tiny-M launcher looks up the bf16 slot);
  `ggml_cuda_mmb_reads_bf16_act()` is the new predicate that says a dense GEMM reads the activation
  through the bf16 cache (all weight types, plus the tiny-M F32 kernel).
* `ggml-cuda.cu`: `all_bf16_consumers` now accepts a tiny-M F32 `MUL_MAT` and a `DSV4_HC_PRE`
  `src[0]` as bf16-aware consumers, so `xn` classifies BF16-only.

**A slot lifetime bug had to be fixed on the way (it cost a NaN PPL).**  Slot 0 is shared by every
generic activation copy, and `dsv4_hc_pre` *reads* `xn` from a slot while *writing* its own output to
slot 0; between the `xn` producer and its delayed consumers the intervening producers (e.g. `lo =
silu(scale(down))`) overwrite slot 0.  So `xn` now gets a **dedicated producer slot (4)**: the graph
optimizer assigns it when it sees the `DSV4_HC_PRE` consumer (`ggml_cuda_mmb_mark_bf16_slot`), and the
producers reserve through `ggml_cuda_mmb_reserve_auto` (dedicated slot if assigned, else 0).  The
first attempt (no dedicated slot) produced a coherent-looking but corrupt 492474 PPL — the huge
apparent speedup was degenerate MoE routing, exactly the session-5e trap: **always PPL before
believing a prefill gain**.

Gates: PPL c2048 **10.6015** (`HC16=0` 10.5771; session 11 was 10.6428 — all within the ±0.68 bar);
greedy `-f prompts/prose-rdna-boosts.txt -n 128 --seed 42 --temp 0 -c 8192` **`9c281c415082`** (624 ch,
reproducible; `HC16=0` gives `c3f24ac9c114`); `FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT`
OK; width probe PASS (maxdiff 0); `plain == draft-mtp` byte-identical (`c0f8fb2b6fc7`).

Same-session A/B (gfx1151 IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`, `-r 2`, two interleaved reps):

| tip | pp2048 | pp8192 |
|---|---:|---:|
| `af2f70580` (session 11) | 999 | 977 |
| this session (`xn` BF16-only) | **1043** | **1014** |

The kernel trace attributes it (pp8192, `rocprofv3 --kernel-trace`, all-kernel total 16744 ->
15594 ms = **-6.9 %** for the whole `HC16` port): `mmb_cvt_f32_bf16` 648.8 -> 0.8 ms (this session's
last conversion is `ple_embd`), `dsv4_hc_pre_f32` 743.8 -> 460.9 ms (bf16 `x` + `gate`),
`mmb_tiny_m_f32_kernel` 576 -> 431 ms (bf16 `X`), `rms_norm_f32<1024,true,false>` 649.5 -> 569.5 ms
(the skipped F32 store), `unary_gated` 255.6 -> 227.6 ms.

## UPDATE — session 11 (2026-09-20): skip the dead F32 store in the producer port — **+2.3 % pp8192 /
## +3.3 % pp2048**, still bit-identical

Session 9's producers wrote F32 *and* the BF16 copy.  A profile showed that of the 650 ms of `mmb_cvt`
removed, only ~489 ms was net (the producers paid ~161 ms in extra BF16 stores).  But for activations
whose every consumer reads the BF16 cache, the F32 output is dead.  The graph optimizer now classifies
consumers: all-`bf16` (quantized `MUL_MAT`/`MUL_MAT_ID` through views) -> mark BF16-only and the
producer skips its F32 store; otherwise keep F32 + copy.  The five producer kernels gained a
`store_f32` flag.  Bit-identical (PPL 10.6428, greedy `9930c674a6ca`, width probe pure, FA/GDN pass).

| config | pp2048 | pp8192 |
|---|---:|---:|
| `HC16=0` | 975.3 | 952.4 |
| `HC16=1` (session 9, always-emit) | ~994 | ~966 |
| `HC16=1` (session 11, dead-store skip) | **1007.4** | **974.0** |

`unary_gated` 302 -> 229 ms.  `xn` still keeps its F32 (its `dsv4_hc_pre`-src0 and tiny-M consumers
read F32); adding BF16 arms there is the next ~1 % but is a numerics change on the hidden stream.

## UPDATE — session 10 (2026-09-20): `ssm_alpha/beta` (M=48) profiled — **rocBLAS stays**

The `ssm_alpha/beta` GEMMs are `[M=48, K=2560]` F32 and run on rocBLAS at **0.360 ms/call
(207.3 ms, 1.26 % of kernels)**.  Two faster-looking replacements lose: the BM=64 WMMA f32 tile is
0.312 ms but its f16-hi/lo split costs **+0.04 PPL** (alpha/beta gate the GDN recurrence), and an
exact-f32 SIMT tile is 0.437 ms (too low an FMA:LDS ratio).  `MMB_F32SPLIT_MIN_M=128` is unchanged;
both experiments reverted.  Full table in `HANDOVER.md` §session 10.

## UPDATE — session 9 (2026-09-20): the bf16-producer port is done — the whole `mmb_cvt` bucket is
## gone, bit-identically (+1.3 % pp8192 / +2.0 % pp2048); plus a **delivery** op-name bug

Session 8 did the HC gate and normalized stream.  This session generalised the mechanism: the graph
now marks the **activation of every MMB dense prefill GEMM** `bf16_copy`, and the fused
`rms_norm+mul`, fused `sigmoid/silu+mul`, fused `scale+unary`, generic unary and `dsv4_hc_pre` all
emit the copy into slot 0 alongside their F32 output.  The GEMM finds it in the slot and skips its
`mmb_cvt` pass.  All copies are RNE-rounded exactly as `mmb_cvt_f32_bf16` and the F32 outputs stay
valid, so everything is **bit-identical**: PPL c2048 stays 10.6428, greedy sha stays `9930c674a6ca`,
FLASH_ATTN_QSA/GATED_DELTA_NET/FLASH_ATTN_EXT pass, W=1..8 width probe pure.

| config | pp2048 | pp8192 |
|---|---:|---:|
| `HC16=0` | 977.1 | 951.6 |
| `HC16=1` | **996.7 (+2.0 %)** | **963.7 (+1.3 %)** |

`LLAMA_MMB_CVT_LOG=1` now shows only the `ple_embd` model-tensor conversion — the `hc_norm`,
`hc_mixed`, `final_output`, `attn_gated` and unary families are gone.

**Delivery bug found and fixed:** `GGML_OP_NAME` in `ggml/src/ggml.c` is missing `"INDEXER_FILL"`
(the enum has `GGML_OP_INDEXER_FILL` from delivery block 14; the base `8a2567e1e` confirms it).
`ggml_op_name()` is therefore shifted by one from there on (`UNARY` prints as `MAP_CUSTOM1`).  It is
cosmetic in the delivery but it mislabeled every `MMB_CVT` log; fixed by WIP commit `d1463bff3` and
flagged to move into the delivery.

## UPDATE — session 8 (2026-09-20): the bf16-producer port begins — HC gate + normalized stream (+1.2 %
## pp8192 / +2.2 % pp2048), behind `GGML_CUDA_MMB_HC16=1`

Session 7 scoped the bf16-producer port (`BF16-PRODUCER-PORT.md`); this session landed its first two
producers.  Tip `ccf28bc65`, two commits on top of the session-7 pack.

* **Mark lifetime**: `ggml_cuda_mmb_marks_clear()` is now called on the first optimize after a
  compute.
* **Gate**: the graph marks a gated `DSV4_HC_PRE`'s gate (dense `MUL_MAT [320 x 10240]`, MMB-taken,
  single consumer) BF16-only; MMB dense writes it into the pinned slot 1 (which already existed and
  was unused) and `dsv4_hc_pre` reads the BF16 copy through a new `wbf16` arm.  A real numerics
  change (PPL c2048 10.5771 -> 10.6428).
* **HC normalized stream (`xn`)**: the fused `rms_norm+mul` now also emits a BF16 copy into slot 0,
  RNE-rounded exactly as `mmb_cvt_f32_bf16`, so the MMB dense down projection skips its conversion
  pass.  Bit-identical (all 57 `hc_norm` `MMB_CVT`s disappear, PPL unchanged from the gate build).

| config | pp2048 | pp8192 | PPL c2048 |
|---|---:|---:|---:|
| `HC16=0` | 973.1 | 950.8 | 10.5771 |
| `HC16=1` gate+xn | **994.2** | **961.9** | 10.6428 |

**What remains:** the `hc_mixed` producer (`dsv4_hc_pre`'s own output — self-contained, but its
consumers must all read BF16) and the `final_output` / `MAP_CUSTOM1` families.  Same-seed greedy text
unchanged (`9930c674a6ca`).

## UPDATE — session 7 (2026-09-20): the qsa3 pack is 0.34 %, not 3 % — a misattribution corrected

Session-5b's next-work proposed fusing the qsa3 pack for ~3 % of prefill.  Measured by diffing QSA3
**on vs off** (`rocprofv3`, `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/`, pp8192, bf16 KV,
`GGML_CUDA_MMB=1`), the **whole pack is 57.8 ms of 16782 ms = 0.34 %** (bf16 KV); with a q8_0 KV
cache, 86.3 ms = 0.48 %.  The session-5 "PACK/copy (qsa3 pack) 3.0 %" bucket was really
`concat_transposed_src1_dim0` (357.5 ms, the **MoE output concat**, present with QSA3 off too) +
`cpy_scalar<float,float>` (110.7 ms, base-graph copies) + the actual pack (57.8 ms).

**Implemented (same session):** the graph no longer builds `pk`/`pv`; it materialises only the natural
contiguous F16 view, and two new launcher kernels (`qsa3_pack_keys_kernel` / `qsa3_pack_values_kernel`)
do the whole re-layout in one pass each.  **Bit-identical** (PPL c2048 10.5771 both; greedy
`sha=04ddb94b1529` both), and the pack kernels go **50.5 -> 6.5 ms** (bf16, save **0.26 %** pp8192),
**86.4 -> 49.1 ms** (q8_0, save 0.22 %).  Full tables in `HANDOVER.md` §session 7.

**Then investigated the next targets (`dsv4_hc_pre`+`_post` bf16 intermediates 8.5 %, `mmb_cvt`
producer marking 3.8 %): they are the *same multi-session port*, not a cast.**  A plain `ggml_cast` is
a net loss — the cast *is* the existing `mmb_cvt` (12 B/elem vs 4 B/elem for a native-bf16 producer).
The reference gets bf16 for free from **fused producers** (`rms_norm`+`mul` with an `out_xn_bf16`
output; `ggml_cuda_mmb_mark_bf16_only` on MMB chains); our tree lacks them, `ggml_cuda_mmb_marks_clear`
is never called, and `LLAMA_MMB_CVT_LOG=1` shows the conversions are for `hc_norm` (5.24 M×2/layer),
`final_output` (3.15 M), `hc_mixed` (1.31 M×2) — all non-MMB producers.  `dsv4_hc_post` is already at
the bandwidth ceiling.  Cheap checks rejected: `GGML_CUDA_MMB_CACHE` 32/128 is a wash (960/957/957).
**Recommendation: scoped follow-up port, ~2.3 % ceiling.**

## UPDATE — session 6 (2026-09-20): the `mmb_*` kernels are at their gfx1151 ceiling

Session 5 left "`mmb_dense` (21 %) + `mmb_routed_glu` (16 %) need a split-K / int8-IU8 restructure".  This session closed both ideas, plus the bf16-shadow alternative.  **No code change** — the
worktree stays clean at `7e431fc82`.  Full detail, tables and traps are in `HANDOVER.md` §session 6;
the short version:

* **int8 WMMA is not faster than bf16 WMMA on gfx1151.**  `tools/wmma-peak-gfx1151.cpp` (new) measures
  **27.5 vs 27.6 T-MAC/s**; the Q8_0 per-block-scale epilogue then drops int8 to **14.1** vs bf16's
  **19.7**.  The 174 T-MAC/s figure that motivated §9 is **gfx1201**.  **The IU8 restructure is a
  net loss on the target arch.**
* **A bf16 weight shadow is 2.44x slower.**  The same `attn_qkv` shape is 2.08 ms as Q8_0 (WTYPE 1,
  dequant) and 5.07 ms as native BF16 (WTYPE 2), measured in situ on the BF16 twin of the model.  The
  kernel is weight-cache/bandwidth bound, not dequant-ALU bound; on-the-fly dequant is correct.
* **Every tile knob is a wash or worse:** dense `BM=64` -6.6 %, GLU big `BM=128` ~-1 %, GLU
  `BN_SMALL=64` wash, force-wide -2.8 %, activation cache 16/64 wash.  The geometry *was* tuned.
* **Efficiency:** `mmb_dense` Q8_0 = 14.8 T-MAC/s = **54 %** of the 27.6 bf16 peak; GLU ~**36 %**
  (it pays the dequant twice).  The residue is dequant-issue contention and is inherent.
* **Fast iteration model for the next session:** `Qwen3.6-35B-A3B-Q4_K_M` (21 GiB) exercises the same
  kernels with the same shares as the 94 GiB Flash-Next, but loads in seconds.  `rocprofv3`'s
  `grid_size_x` is `blocks.x x 256` — divide before matching a shape.

**Revised next step:** the `mmb_*` kernels' remaining gains need arithmetic that is already bf16
(none), so the prefill lever is **outside `mmb_*`** (FA 11.3 %, GDN 5.5 %, MoE concat+reduction 6.8 %,
`rms_norm` ~5 %), or **promotion** — all §11 gates are green (sessions 5c/5d), which is now the
highest-value step.

## Next

1. **QSA v3 packed-WMMA attention** — now the #1 kernel.  Plan: graph-side `qsa_pack_keys`/`_values`,
   the `qsa3_rows`/`qsa3_merge` block descriptor, then the `qsa3_attn_kernel` WMMA; prefill-only
   (`n_query >= 128`), VEC kept for the W=1..8 band.  Estimated ~1017 -> ~1160 t/s on Q4_K_M.
2. ~~**Q8_0 IU8-WMMA**~~ — **REFUTED on gfx1151 (session 6): int8 == bf16 WMMA, and the epilogue makes
   it slower.**  Do not pursue.  See the session-6 UPDATE above.
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

## UPDATE — session 5e (2026-09-19): dsv4_hc investigated — ~18 % kernel-local headroom, and the real
lever is bytes (bf16 intermediates)

`dsv4_hc_pre_f32` + `dsv4_hc_post_f32` = 1432.5 ms (8.5 % of pp8192); pwilkin's reference is ~0.93 s
vs our 1.44 s.  **Nothing landed** — the tree is clean, baseline pp8192 restored to 953.5 t/s.

**Measure the ceiling, don't infer it — and sweep the grid.**  `tools/dram-bw-probe.cpp`
(`hipcc --offload-arch=gfx1151 -O3`), 192 MB buffers, grid swept, 34 C, no other GPU work:

| pattern | GB/s | % of 256 GB/s spec |
|---|---:|---:|
| pure sequential read (best grid) | **241.5** | 94.3 % |
| copy (read + write) | ~208-216 | 81-84 % |
| write-only | ~217 | 85 % |
| **the exact `dsv4_hc_pre` shape** (x + gate, 4 streams each, + dst), best grid | **232.6** | 90.8 % |
| the real `dsv4_hc_pre_f32` | **197** | 77.0 % |

The part sustains **~240 GB/s** as measured here (a separate report puts it at ~255 achievable, which
would widen the gap slightly) — and **the dsv4_hc_pre access pattern is not inherently slow**: a clean
kernel of the identical shape reaches 232.6 GB/s.  So our kernel sits **~18 % below what its own
pattern allows** (and ~23 % below the best pure read), not at a wall.

Two corrections this took, both my own:

1. **The ceiling was inferred from our own kernels** in the first pass, giving "already at the memory
   wall, nothing to gain".  It has to be measured.
2. **The grid must be swept before quoting a ceiling.**  grid=4096 everywhere gave 231.6 GB/s (read) /
   227.2 (pattern); the *same kernels* at grid=16384 give **241.5 / 232.6** — ~5 %.  Instruction
   sequence matters too: a 4-accumulator x4-unrolled read variant measured **worse** (221-230) than a
   plain single-accumulator grid-stride loop, so "more ILP" is not automatically more bandwidth on
   this part (consistent with this machine's number depending on the exact sequence used).

The split:

* **~18 % kernel-local** (197 -> 232.6) = ~133 ms = **~0.8 % of prefill**.  Unexplained so far; what
  differs from the probe kernel is the sigmoid (measured free), the runtime-stride address arithmetic,
  and the strided `dst` write.
* **the dominant lever is bytes** — bf16 intermediates cut unique traffic 189 -> ~105 MB (1.8x); at
  232.6 GB/s that is ~0.45 ms/launch vs 0.98 = **~2.3 % of prefill**, matching the reference's
  1.44 -> 0.93 s.  It needs the `hc_norm`/`hc_gate` producers to write bf16, so it is a **graph-level
  change**.

Kernel-level A/B (`rocprofv3`):

| variant | `dsv4_hc_pre_f32` |
|---|---:|
| production (`expf`) | 745.0 ms |
| `__expf` | 744.8 ms |
| identity (no sigmoid) | 745.3 ms |
| `float4` over the contiguous `i0` axis | neutral (end-to-end, value-preserving) |
| hc loop unrolled + `__restrict__` | 729.9 ms (**-2.0 %** = 0.09 % of prefill, not kept) |

No register spilling anywhere (VGPR 24->32, `Scratch_Size` 0).

**Two probe traps, both mine:** size buffers by *bytes* and index by *float4 count* (mixing them is a
4x out-of-bounds read that presents as `Memory access fault ... Page not present`, looking like a
HIP/driver problem); and count *write* bytes as the iterations actually performed, not one whole
buffer (counting 3x192 MB when 50 MB was written reported **302 GB/s on a 256 GB/s part** — exceeding
spec is the tell that the accounting, not the kernel, is wrong).

### METHODOLOGY — prefill time here is DATA-DEPENDENT; judge kernel variants on kernel time

The identity-sigmoid variant measured **-13 % end-to-end** (951 -> 824 t/s) while **its own kernel was
unchanged** (745.3 vs 745.0 ms).  The whole swing was downstream on *identical launch counts*:
`mmb_routed_glu` 3803.6 -> 5906.6 ms (+55 %), `mmb_routed` +392 ms, `qsa3_attn` +167 ms.  Changing the
activations changes the MoE routing, and the GLU launch is sized worst-case with early-return slots,
so different routing = different work.

**Rule: a value-changing kernel variant must be judged on `rocprofv3` kernel time, never end-to-end
t/s.**  Corollary for the rest of this WIP: every end-to-end A/B here also changed numerics (MMB,
QSA3), so those wins were confirmed in the kernel profile rather than taken from t/s alone.

Second trap from the same episode: the first version passed the flag as a **runtime kernel argument**,
so the "no sigmoid" variant still compiled the `expf` *and* a select — it did strictly *more* work.
A diagnostic whose whole point is to remove work must be a **template** parameter.

## UPDATE — session 5d (2026-09-19): W=1..8 width-purity probe — the last gate item closes

The `W = 1..8` logits matrix was the one §11 gate never run because no harness existed.  It does now:
**`tests/test-logits-width-probe.cpp`** (`cmake --build build-rocm --target test-logits-width-probe`),
adapted from `archive/work/strix-halo/issue25/logits-width.cpp`.  Gate **PASSES**:

| prefill P | `MMB=0` row0 | `MMB=1` row0 | width purity |
|---|---|---|---|
| 256 (below `MMB_MIN_T = 512`) | `6228d03bd2b501b4` | `6228d03bd2b501b4` — **identical** | PASS, maxdiff 0 |
| 1024 | `ac4d5de3d40a2b1d` | `3703c13f03c4b25d` | PASS, maxdiff 0 |
| 2048 | `1996b44e491de5c9` | `e3e4220fe83831da` | PASS, maxdiff 0 |

Below the threshold MMB is unreachable in both the prefill and the decode batch, so the configs are
bit-identical across every row of every width — the "identical by construction" claim demonstrated.
Above it the row-0 hashes differ by the **approved prefill re-baseline** (MMB replaces the MMQ
reduction with a dequant-to-bf16 WMMA one), while **`width_purity` stays PASS with MMB on** — MMB
introduces no width dependence, and `T >= 512` is what keeps it out of the `W <= 8` band.

Two probe bugs, same class as the `-md` trap: `llama_batch_init(ubatch)` sizes for the *micro*-batch
so a `P`-token prefill overruns it (silent SIGSEGV, plus a `GGML_ASSERT` for the `n_batch` half); and
`n_batch` (per `llama_decode`) vs `n_ubatch` (per micro-batch) are different knobs, both scaling
with `P`.  Also: `rows[W-1].data()` hashes the `std::vector` objects, not the floats.

## UPDATE — session 5c (2026-09-19): the promotion gates, run

With the remaining big kernels out of reach for a safe change (`mmb_routed_glu` 22.7 %,
`mmb_dense` 21.1 % — both need a split-K / IU8 restructure), this session ran the §11 gates that the
handover listed as never run.

Test models: `build-rocm/bin/test-llama-archs -o /tmp/test-models` (the `test-generate-models` fixture;
~200 tiny GGUFs).  The test binaries are not in `build-rocm/bin` by default — build them by target.

| gate | result |
|---|---|
| `test-recurrent-state-rollback` qwen35-dense / nemotron_h-dense / deepseek4-moe | **PASS** (max diff 0) |
| `test-recurrent-state-depth` (n_rs_seq 1..15) | **PASS** (`total failures = 0`) |
| `test-backend-ops -o FLASH_ATTN_QSA` | **22/22** (older notes say 18/18 — cases were added) |
| `test-backend-ops -o GATED_DELTA_NET` | **46/46** |
| `test-backend-ops -o FLASH_ATTN_EXT` | **OK** |
| `plain == draft-mtp` greedy text (qwen4exp, MMB+QSA3 on **and** off) | **PURE, byte-identical** |

All with `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`.

### `plain == draft-mtp` on qwen4exp — PURE, gate PASSES

| config | plain | draft-mtp | verdict |
|---|---|---|---|
| `MMB=1 QSA3=1` | `bbd4bcb519e4` | `bbd4bcb519e4` | **identical** (1700 chars) |
| `MMB=0 QSA3=0` (features OFF) | `5120b28f2879` | `5120b28f2879` | **identical** (1720 chars) |

Hashes differ *between* configs — that is the approved prefill re-baseline — but *within* each config
the two arms are byte-identical.  These are also two of the previously-missing re-baseline hashes.

### METHODOLOGY TRAP — reported WRONG first

The first attempt concluded `plain != draft-mtp` ("320 chars vs 1942") and attributed it to pre-existing
cause 3.  **Fabricated by the harness:**

* the plain arm must **not** get `-md`; passing the draft model makes llama-cli initialise an MTP
  context even with `--spec-type none`, which fails (`this model is an MTP draft head without a
  trunk`, `llama_server exited with code 1`) and the run **exits 1 without generating**;
* its stdout still had the `Loading model... |\b-\b\\...` **spinner**, and `grep -v '^\[' | tr -d
  '\b'` turns that spinner into exactly 320/318 "chars" — so the comparison was **a spinner vs real
  text**;
* the 1942-char side was genuine; the 320-char side never generated a token.

Rules: **never pass `-md` to the plain arm**; **assert the arm generated output** (`$?`, non-empty, not
the spinner) before comparing.  A one-sided load failure always "diverges" and looks like a real
near-tie flip.

`LLAMA_QSA_OFF=1` is separately unusable with `-md` (`llama_server exited with code 1`).  And note
`FLASH_ATTN_QSA` is 22/22 now, not 18/18.

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
