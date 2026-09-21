# gfx1201 port — S10: the dense tile geometry (2026-09-21)

Session S10 of `gfx1201-porting.md` §13.  This is the raw evidence; the plan is the source of truth.
**Headline: the RDNA4 dense path lost for every weight type because of the gfx1151-tuned geometry,
not because of the architecture.  A 256x128 tile makes IQ3_S's dense GEMM genuinely beat the
delivery's MMQ by 4.5 % (kernel-time backed), so RDNA4 now enables the dense path per weight TYPE
(IQ3_S only) and the 27B UD-IQ3_S model gains +0.46..+0.52 % prefill.  A by-product is a correction
to the S7 record: the 35B UD-Q3_K_M "+6.7 % MoE win" does not reproduce (§6).**

## 0. Setup

* 3x Radeon AI PRO R9700 (gfx1201), ROCm `/opt/rocm-7.14.1-gfx102X`.  All numbers here are **1 GPU**
  unless stated (the 27B UD-IQ3_S fits one card, so no all-reduce noise).
* Models: `/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf` (11.2 GiB, dense "qwen35"),
  `.../27B/Q8_0/` (29 GB), `.../Flash-Next/IQ4_XS/` (87 GiB, 3 shards, qwen4exp, 3-GPU tensor),
  `/llm/models/Qwen3.6/35B-A3B/Q3_K_M/` (17 GB, `qwen35moe`).
* Build: `~/llama.cpp` branch `mmb-port-qsa3`.  A real (non-comment) `mmb.cu` edit rebuilds in
  **14 s** with ccache, and the A/B reference (`GGML_CUDA_MMB=0`) is a *runtime* switch in the same
  binary -- so the sweep was done by editing the geometry constants and rebuilding, **not** with a
  runtime `MMB_GEOM` selector (which would have needed a second instantiation set for the sake of
  hiding a 14 s rebuild).  A comment-only edit hits ccache and does **not** invalidate
  (ccache hashes the preprocessed source, so a trailing comment is stripped and it reports a hit;
  it replays the stored warnings, which is confusing).
* Instrument: `rocprofv3 --kernel-trace` (`-d <dir> --output-format csv`; the output lands in
  `<dir>/<host>/<pid>_kernel_trace.csv`).  It carries `VGPR_Count`, `LDS_Block_Size`,
  `Scratch_Size`, `Workgroup_Size_*`, `Grid_Size_*` per launch -- that is what made this session
  possible.  Note `Grid_Size_X` is in **work-items**, not workgroups (an MMB launch with BM=256 and
  M=10240 reports 10240 = 40 blocks x 256 threads).
* Prefill protocol as §13.0: `-p ... -n 0 -b 2048 -ub 2048`, `r=5` for decisions and interleaved
  back-to-back rounds; the first prefill of an invocation is cold-start-limited.

## 1. The mechanism: LDS-bound occupancy and a half-idle weight dequant

`rocprofv3` on 27B UD-IQ3_S, `-p 8192`, 1 GPU.  OFF = delivery, ON = `MMB=1 MMB_DENSE=1` with the
**shipped (S7) geometry**:

| kernel | OFF | ON | VGPR | LDS | blocks/CU (LDS) |
|---|---|---|---|---|---|
| `mul_mat_q<Q2_K, 80>` | 4.097 s | 4.101 s | 256 | 0 | 3 |
| `mul_mat_q<IQ3_S, 128>` | 3.887 s | 0.079 s | 216 | **0** | 3 |
| `mmb_dense_kernel<128,256,64,64,5>` | -- | 2.580 s | 248 | **55296** | **1** |
| `mmb_dense_kernel<128,128,32,64,5>` | -- | 1.405 s | 224 | **36864** | **1** |
| `mul_mat_q<IQ4_XS, 128>` | 1.653 s | 0.030 s | 224 | 0 | 3 |
| `mmb_dense_kernel<128,256,64,64,8>` | -- | 0.807 s | 248 | 55296 | 1 |
| `mmb_dense_kernel<128,128,32,64,8>` | -- | 1.086 s | 224 | 36864 | 1 |
| `mmb_cvt_f32_bf16` | -- | 0.195 s | 24 | 0 | -- |
| **total kernel time** | **17.364 s** | **17.901 s** | | | |

`hipDeviceProp_t.sharedMemPerBlock` on this part is **65536 B**, so:

* the dense **big** tile (As 128x72 + Bs 256x72 halfs = 55296 B) leaves room for exactly **one**
  workgroup per CU -> 8 warps -> **2 warps per SIMD**;
* the delivery's MMQ uses **no LDS at all** (it dequantizes into registers) and gets **3** blocks;
* so MMB runs the WMMA at **1/3 the occupancy** of the kernel it replaces.  It is fast enough to
  nearly compensate (IQ3_S: 3.985 s of mmb vs 3.887 s of MMQ = +2.5 %) -- but that is still a loss,
  and the `mmb_cvt_f32_bf16` activation conversion (+0.195 s) is on top.

The second defect is in the dequant loop: `A_ITEMS = ceil(BM / MMB_NT)`, so with **BM=128 and 256
threads only threads 0..127 dequantise** the A (weight) panel and half the block idles; the A
dequant is serialised with the WMMA work anyway (`DBUF` is unwired -- §4).

## 2. The geometry sweep

Cost model that guided it: the total A-dequant volume is `M*K*T/BN` (so **BN**, not BM, sets the
dequant *total*), while **BM** sets the dequant *thread utilisation* (`A_ITEMS >= 1`, and
`BM >= 256` uses every thread), and `(BM+BN)*72*2` sets the LDS -> occupancy class.

**Hard validity rule (this cost an hour):** `BN` must equal `(8 / (BM/WTM)) * WTN`.  The kernel's
warp grid (`wm = wave % (BM/WTM)`, `wn = wave / (BM/WTM)`) only ever covers
`(8/(BM/WTM)) * WTN` output columns, so a larger *declared* `BN` allocates and loads more B while
computing/storing fewer columns -- **half the output is silently never computed and the kernel looks
dramatically faster.**  A `256x192/WTN=48` arm measured a fake **-39 %** on IQ3_S this way.  Every
candidate in the sweep therefore carries a **same-seed text-hash gate**; the geometry is otherwise
numerics-neutral (§3).

`pp4096`, 1 GPU, `MMB_TILE=1` (one geometry for every shape).  OFF reference: IQ3_S (MMQ) 1.944 s,
IQ4_XS (MMQ) 0.829 s, total 8.575 s.

| arm | BMxBN | WTMxWTN | TMxTN | acc | LDS | IQ3_S | IQ4_XS | total | hash |
|---|---|---|---|---|---|---|---|---|---|
| **A** | **256x128** | **64x64** | **4x4** | 128 | 55296 | **1.856 (-4.5 %)** | 0.919 (+10.9 %) | 8.672 | `42cdf36d0633` |
| B | 256x128 | 128x32 | 8x2 | 128 | 55296 | 1.939 (-0.3 %) | 0.974 (+17.5 %) | 8.824 | `42cdf36d0633` |
| I | 192x192 | 96x48 | 6x3 | 144 | 55296 | 1.874 (-3.6 %) | 0.959 (+15.7 %) | 8.741 | `42cdf36d0633` |
| J | 192x128 | 96x32 | 6x2 | 96 | 46080 | 2.180 (+12 %) | 1.083 (+30.6 %) | 9.167 | `42cdf36d0633` |
| G | 128x192 | 32x96 | 2x6 | 96 | 46080 | 2.334 (+20 %) | 1.000 (+20.6 %) | 9.218 | `42cdf36d0633` |
| (C) | 128x256 | 64x64 | 4x4 | 128 | 55296 | 3.985 (pp8192) | 1.893 (pp8192) | -- | -- |
| (C2) | 128x128 | 32x64 | 2x4 | 64 | 36864 | (the shipped small tile) | | | |
| -- | 128x64 | 32x32 | 2x2 | 32 | 27648 | 744 vs 929 t/s = **-20 %** (pp8192, gate 2 blk) | | | -- |
| -- | 384x64 | 96x32 | 6x2 | 96 | 64512 | 2.768 (+42 %) | 1.264 (+52 %) | 9.947 | `42cdf36d0633` |
| (D) | 256x192 | 64x48 | 4x3 | 96 | 64512 | *invalid* -- fake -39 % | | | -- |
| DBUF | 128x128 | 64x32 | 4x2 | 64 | 55296 | 839 vs 929 t/s = **-9.8 %** (pp8192) | | | -- |

Readings:

* **The whole sweep confirms the mechanism.**  `BN=64` (2 blocks/CU) is *far* worse (-20 %): the
  dequant total scales as 1/BN and dominates.  `BN=192/384` (fewer blocks, less dequant, but 1
  block/CU) are also worse.  `BM=256` is the one axis that helps, and only because it removes the
  half-idle-thread dequant -- **not** because it changes the dequant total (it does not).
* **No geometry makes IQ4_XS win** (its best is A at +10.9 %); the delivery's `mul_mat_q<IQ4_XS,128>`
  is simply good.  IQ3_S is the type where MMB's tile genuinely wins.
* `DBUF` (double-buffered A, so the next k-step's dequant overlaps this step's WMMA) **is not a
  win**: -9.8 % at BN=128 (the dequant-volume penalty of the smaller BN swamps the overlap).
  Its only LDS-compatible shapes are small-BN ones.  The code comment claiming -6.7 % for the
  IQ3_S GLU remains unwired and unverified on RDNA4 (S12).

## 3. Correctness: geometry is numerics-neutral

Every **valid** geometry above produced the identical same-seed hash, and so did the shipped
(S7) geometry and the delivery (`MMB=0`):

```
27B UD-IQ3_S, --seed 42 --temp 0 -n 24, "The capital of France is"
  delivery            -> 119 chars sha=42cdf36d0633
  MMB=1 (S7 geometry) -> 119 chars sha=42cdf36d0633
  MMB=1 (geometry A)  -> 119 chars sha=42cdf36d0633
  MMB=1 (geometry B/I/J/G/384x64) -> sha=42cdf36d0633 each
```

This is expected: for a given output element the K accumulation is a fixed sequential walk
(`ks` ascending, `kk` = 0,16,32,48, fixed gfx12 fragment layout), and the geometry only changes
*which warp* computes it and how many columns a block covers.  **So a geometry change needs no
purity re-validation** -- but the hash gate is still mandatory, because it is what catches an
*invalid* geometry (§2).

## 4. The landing

`mmb.cu`:

* A per-TYPE dense policy, `mmb_dense_tmask()` / `mmb_dense_type_ok(t)`, replacing the boolean
  `mmb_dense_flag()` at the two dense-path predicates (`ggml_cuda_mmb_supported_mm`'s `quant` and
  `ggml_cuda_mmb_dense_will_take`).  RDNA4 default = **IQ3_S only**; everywhere else `~0ull`, so the
  pre-S10 behaviour is preserved exactly.  `GGML_CUDA_MMB_DENSE_TYPES=<csv>` overrides it and
  `GGML_CUDA_MMB_DENSE=0|1` still forces the whole path off/on.
* A `mmb_dense_launch_t<BM,BN,WTM,WTN>` helper (the type switch once), so the dispatch is
  `mmb_dense_launch_t<256,128,64,64>` on RDNA4 and the untouched gfx1151 big/small chain otherwise.
  The RDNA4 branch is runtime-gated on `GGML_CUDA_CC_IS_RDNA4(devices[0].cc)` (the `RDNA4` macro is
  device-pass-only, so a compile-time guard cannot select a *host* launch).
* The graph's MMQ-fusion stand-down flows through the same predicate, so every excluded (type, path)
  keeps the delivery's MMQ path and costs nothing.

Results (27B UD-IQ3_S, 1 GPU, interleaved back-to-back, `-p 8192,32768 -n 0 -b 2048 -ub 2048`):

| round | | pp8192 | pp32768 |
|---|---|---|---|
| r=5 r1 | OFF / ON | 928.33 / 933.21 | 853.18 / 857.03 |
| r=5 r2 | OFF / ON | 928.19 / 933.01 | 853.15 / 857.04 |
| | **delta** | **+0.52 %** | **+0.46 %** |

Two rounds agree to 0.02 %, so this is well outside noise.  Per-config breakdown (r=3, two rounds,
pp8192 / pp32768): IQ3_S-only **+1.03 % / +0.94 %**, `+IQ3_XXS` +1.03 / +0.96, `+IQ4_NL` +1.06 /
+0.97, **whole IQ family (adds IQ4_XS) +0.04 / +0.04** (the IQ4_XS regression cancels the win
exactly as the kernel times predict).

Regression checks:

| model | result |
|---|---|
| 27B Q8_0 (dense, no MMB-eligible type) | 1373.58 / 1374.30 (OFF) vs 1373.67 / 1373.82 (ON) = **exactly neutral** |
| Flash-Next IQ4_XS (qwen4exp, 3-GPU tensor) | OFF 2726.08 / 2722.40 -> ON 2814.04 / 2783.31 = **+3.2 % / +2.2 %** (r2; r1 +4.5/+2.7) -- **the S7 qwen4exp win reproduces** |

**gfx1151 is untouched, verified at the instruction level.**  Compile `mmb.cu` for gfx1151
(`--cuda-device-only -S`, offload arch swapped, command from `build-rocm/compile_commands.json`)
before and after, split the asm on the `.type <sym>,@function` / `.size <sym>` delimiters, and
compare instruction streams with comments, `__hip_cuid_*`, `.LBB<n>_`, `.Lfunc_end<n>` and `%bb.<n>`
normalised:

* **79 of 79 existing kernels are instruction-identical** (combined md5
  `fc698705809822f4c821adb115367dc8`);
* **11 new kernels** appear (`mmb_dense_kernel<256,128,64,64,0..10>`), unreachable on gfx1151
  because the branch is runtime-gated on the device cc.

(The naive whole-file diff shows 64 "differing" kernels -- that is only `.LBB<n>_` numbering, plus
one artifact where a kernel's captured asm buffer absorbs the *next* kernel's YAML metadata.  Split
on `.size` and normalise the labels, or the check is worthless.)

## 5. What the four sessions of `mmb` on RDNA4 now say

| path | RDNA4 verdict | evidence |
|---|---|---|
| routed MoE (`MUL_MAT_ID`) | **break-even**, see §6 | kernel times |
| qwen4exp HC / tall-M / tiny-M | **wins** (+3.2 / +2.2 %) | this session + S7 |
| generic quantized dense GEMM | **wins for IQ3_S**, loses for IQ4_XS, break-even for IQ3_XXS/IQ4_NL | §2, §4 |
| F32 split (MoE router) | loses (S7) | S7 |
| activation conversion (`mmb_cvt_f32_bf16`) | a ~1 % tax on every MMB dense GEMM; the 4-entry cache never hits (680 conversions for ~340 GEMMs at pp4096) | §1; `GGML_CUDA_MMB_CACHE` is the knob, untested |

## 6. Correction to the S7 record: the 35B UD-Q3_K_M MoE win does not reproduce

S7 recorded **+6.7 % / +5.6 %** for `35B-A3B UD-Q3_K_M` under the split policy.  Re-measuring the
**unmodified S7 binary** (tree `580db5174`, `mmb.cu` at `50b813085`, rebuilt) on this box:

| build | OFF pp8192 / pp32768 | ON | delta |
|---|---|---|---|
| S7 source (pre-S10) | 5920.59 / 4860.59 | 5838.98 / 4798.23 | **-1.4 % / -1.3 %** |
| S10 landed | 5938.25 / 4864.17 | 5853.15 / 4799.80 | -1.4 % / -1.3 % |

The model has **no IQ3_S** (types: F32, Q8_0, IQ3_XXS, IQ4_XS, Q6_K, Q3_K, BF16, Q4_K), so the S10
change cannot affect it -- confirmed by the identical deltas.  The kernel breakdown (`-p 8192`, 1
GPU, OFF -> ON) explains it:

| | OFF | ON |
|---|---|---|
| `mul_mat_q<IQ3_XXS, 64>` | 0.650 s | 0.000 |
| `mul_mat_q_routed_compact<IQ4_XS, 64>` (the delivery's fused MoE kernel) | 0.230 s | 0.000 |
| `mmb_routed_kernel<128,128,32,64,10>` + `<128,32,32,16,10>` | -- | 0.478 + 0.171 |
| `mmb_routed_kernel<128,128,32,64,8>` + `<128,32,32,16,8>` | -- | 0.163 + 0.063 |
| `mm_ids_helper<8>` | 0.127 s | **0.192 s** |
| **total** | **2.704 s** | **2.734 s (+1.1 %)** |

i.e. the routed MMB path (0.875 s) only *matches* the delivery's `mul_mat_q_routed_compact` +
`mul_mat_q<IQ3_XXS,64>` (0.880 s), and standing the fused kernel down costs an **extra
`mm_ids_helper`** launch (+0.065 s).  Net -1.4 %.

**Consequence, and the open question for the next session:** the RDNA4 default (`GGML_CUDA_MMB=1`
-> IQ-family weight types + routed) is a **~1.4 % loss on this MoE model** and a ~0 % win on the
models the type mask excludes.  The MoE routing default should be re-decided (S12/TODO) -- either
the S7 number was contaminated, or the delivery's block-13 `mul_mat_q_routed_compact` got faster
after S7 was written.  Until then, **`GGML_CUDA_MMB=1` on RDNA4 is only a clear win for qwen4exp
(HC/QSA models) and IQ3_S-heavy dense models.**

## 7. Open items handed on

1. **The routed MoE default** (§6) -- measure `mul_mat_q_routed_compact` vs `mmb_routed_kernel`
   per type on at least two MoE models, and fix the `mm_ids_helper` overhead (it may be avoidable if
   MMB consumed the same compact routing the fused kernel does).
2. `GGML_CUDA_MMB_CACHE` -- the activation-conversion cache never hits; a larger cache may cut the
   ~1 % `mmb_cvt_f32_bf16` tax (measure the memory cost).
3. `DBUF` remains unwired; the LDS-compatible shapes are the small-BN ones where it does not pay.
   A DBUF design that keeps BN large would need the A panel out of LDS entirely (a register-side
   A dequant in the gfx12 fragment layout) -- that is the only remaining route to 2 blocks/CU at
   BN>=128, and it is a rewrite, not a knob.
4. `MMB_LDS_STRIDE` (72) bank behaviour was not swept; at BM=256 it is a second-order term.
5. S11: the new geometry is selected by a runtime `GGML_CUDA_CC_IS_RDNA4` check inside the dispatch;
   it belongs in the `mmb_arch_defaults(cc)` table with the rest of the tunables.
