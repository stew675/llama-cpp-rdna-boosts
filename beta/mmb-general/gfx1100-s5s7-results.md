# gfx1100 port — S5/S6/S7 record: G1 `mmb` (2026-09-21)

Sessions S5 (open + correctness), S6 (per-type/per-path matrix + re-tune) and S7 (routed path /
policy) of `gfx1100-porting.md` §6.5.  Raw evidence; the plan is the source of truth.

**Headline: on gfx1100 `mmb` is a *large win*, the opposite of gfx1201's dense regression.**  The
only loser is the F32 MoE-router split, which is now defaulted off on RDNA3_0 (patch `0008`).  With
that one policy change, MMB on a single RX 7900 XTX measures:

| model | prefill win (MMB on vs off) |
|---|---|
| **27B UD-Q4_K_M** (dense, mixed K-quant) | **+14.4 % pp8192, +13.9 % pp16384** |
| **gemma-12B Q8_0** (dense) | **+11.3 % pp8192, +9.9 % pp16384** |
| **35B-A3B Q3_K_M** (`qwen35moe`) | **+5.6 % pp8192, +4.7 % pp32768** |
| gemma-26B-A4B (gemma4 MoE) | **neutral** (its quantized weights are Q4_0, not an MMB type) |

Decode is untouched (`mmb_min_t = 512`), width purity passes, and MTP is unaffected/slightly better.

## Setup

* Hardware: 1× RX 7900 XTX (gfx1100, 24 GB), ROCm `/opt/rocm-7.14-gfx1100`, `HIP_VISIBLE_DEVICES=0`.
* Build: `~/llama-wip-gfx1100`, branch `mmb-gfx1100` = the 6 canonical WIP patches + the gfx1100
  overlay (`0007` qsa3 predicate).  MMB is env-gated, so no rebuild is needed to toggle it.
* `llama-bench … -n 0 -r 5`, interleaved off/on rounds; the loops that used a `$ENV` variable were
  buggy (`$ENV` is not reparsed as leading assignments — it became the command name), so all numbers
  below come from explicit-env runs / the `ab.sh` helper with literal assignments.

## S5 — open + correctness

`GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1` reaches MMB on gfx1100.  `GGML_CUDA_MMB_LOG=1` shows the
kernels firing:

| model | logged categories | logged types |
|---|---|---|
| 35B-A3B `qwen35moe` | `MMB_DENSE` ×80, `MMB_GLU` ×2 (routed runs silently) | `q3_K`, `q4_*`, `q5_K`, `q6_K`, `q8_0`, `f32` |
| gemma-26B-A4B | `MMB_DENSE` ×80, **all `type=f32`** | `f32` only → the quantized weights are Q4_0 (unsupported); only the F32 router is taken |

**PPL parity** (prose, `-c 2048`):

| model | MMB off | MMB on | Δ |
|---|---:|---:|---:|
| 27B UD-Q4_K_M | 10.0174 ± 0.62345 | **9.9258 ± 0.61417** | −0.9 % (slightly better) |
| 35B-A3B Q3_K_M | 14.8302 ± 1.00741 | 14.8887 ± 1.01162 | +0.4 % (noise) |
| gemma-26B-A4B | 353.6138 ± 42.66 | 359.2559 ± 43.45 | +1.6 % (noise; the absolute PPL is high for this model on the prose prompt, both arms) |

A fragment-layout error would be an exact-permutation error (orders of magnitude), so parity means
the dequant/WMMA path is correct.

## S6 — the per-type / per-path matrix

### 6.1 MMB with the gfx1151 default (F32 split ON)

`r=5` interleaved, two rounds each:

| model | point | MMB off | MMB on | Δ |
|---|---|---:|---:|---:|
| 27B UD-Q4_K_M (dense) | pp8192 | 1022.21 / 1022.13 | 1169.67 / 1170.77 | **+14.4 %** |
| 27B UD-Q4_K_M | pp16384 | 981.20 / 983.16 | 1116.55 / 1116.30 | **+13.8 %** |
| gemma-12B Q8_0 (dense) | pp8192 | 2146.42 / 2142.67 | 2385.93 / 2387.53 | **+11.3 %** |
| gemma-12B Q8_0 | pp16384 | 1904.32 / 1901.96 | 2090.05 / 2093.81 | **+9.9 %** |
| 35B-A3B (MoE) | pp8192 | 3674.86 / 3666.97 | 3762.01 / 3761.93 | +2.4 % |
| 35B-A3B | pp32768 | 3005.33 / 3001.06 | 3068.24 / 3069.01 | +2.1 % |
| gemma-26B-A4B (MoE) | pp8192 | 3302.55 / 3291.95 | 3194.29 / 3198.67 | **−3.1 %** |
| gemma-26B-A4B | pp32768 | 2418.69 / 2410.20 | 2360.95 / 2367.19 | **−2.1 %** |

So the **quantized dense GEMM wins big on gfx1100** (unlike gfx1201, where it lost for every type),
but the MoE models show a small regression.

### 6.2 Isolating the regression: it is the F32 router, not the dense GEMM

| config | gemma-26B pp8192 / pp32768 | 35B-A3B pp8192 / pp32768 |
|---|---|---|
| MMB off | 3302.55 / 2418.69 | 3674.86 / 3005.33 |
| MMB on, default (F32split 1) | 3194.29 / 2360.95 | 3762.01 / 3068.24 |
| MMB on, `GGML_CUDA_MMB_F32SPLIT=0` | **3299.82 / 2410.00** | **3878.87 / 3142.61** |
| MMB on, `GGML_CUDA_MMB_DENSE=0` (routed only) | 3320.37 / 2413.29 | 3843.33 / 3117.07 |

* gemma-26B's regression **disappears entirely** with the F32 split off (it equals MMB-off).  Its
  only MMB work is the F32 router, so the regression *was* the F32 split.
* The 35B gains **further** with the F32 split off: pp8192 3879 vs 3762 (+3.1 %) and pp32768 3143 vs
  3068 (+2.4 %).
* `DENSE=0` (which also removes the quantized dense GEMM) is slightly *below* `F32SPLIT=0` on the
  35B (3843 vs 3879), i.e. the quantized dense contribution is ~+1 % on top of the routed path; the
  routed path itself is the bulk of the MoE win (~+4 % vs MMB-off).

**Conclusion:** on gfx1100 the dense GEMM and the routed path win; the **F32 router split loses**
(exactly as on gfx1201; gfx1151 is the arch where it wins).  The fix is arch-scoped: default the F32
split off on RDNA3_0/RDNA4 and keep it on for RDNA3_5.

### 6.3 The shipped gfx1100 default (patch `0008`)

`mmb_f32split_mode()` now returns `GGML_CUDA_CC_IS_RDNA3_5(cc) ? 1 : 0` unless
`GGML_CUDA_MMB_F32SPLIT` overrides.  Re-measured with the plain env
(`GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1`), `r=5`, two rounds:

| model | point | MMB off | MMB on (new default) | Δ |
|---|---|---:|---:|---:|
| 27B UD-Q4_K_M | pp8192 | 1022.54 / 1022.52 | 1170.85 / 1168.08 | **+14.3 %** |
| 27B UD-Q4_K_M | pp16384 | 980.84 / 982.22 | 1118.00 / 1115.86 | **+13.8 %** |
| 35B-A3B | pp8192 | 3668.74 / 3666.18 | 3873.08 / 3881.40 | **+5.6 %** |
| 35B-A3B | pp32768 | 3003.20 / 3006.19 | 3145.92 / 3144.12 | **+4.7 %** |
| gemma-26B-A4B | pp8192 | 3322.88 / 3289.27 | 3294.78 / 3289.65 | neutral |
| gemma-26B-A4B | pp32768 | 2410.90 / 2410.06 | 2413.88 / 2412.10 | neutral |

## S7 — routed / GLU and the remaining tuning

The `MMB_GLU` fused gate/up+swiglu path is part of the MoE win (the `DENSE=0` isolate still gains
~+4 %).  The finer routed knobs (`GGML_CUDA_MMB_ROUTED_THRESH`, `_GLU_THRESH`, `DBUF`, the IQ3_XXS
GLU arm) were **not** swept — the win is already large and the shapes are gfx1151-tuned.  Follow-up
for a later session if more is wanted.

## Correctness gates for G1

| gate | result |
|---|---|
| PPL parity (27B/35B/g26) | within −0.9 % / +0.4 % / +1.6 % (noise) |
| width purity `test-logits-width-probe` 27B, MMB on, f16 | **`width_purity=PASS (worst maxdiff 0)`** |
| decode `tg128 @ d16384` 27B | off 37.97, on 38.03 — **unchanged** (`mmb_min_t = 512` keeps MMB out of decode/verify) |
| same-seed greedy, long-context (prose, 48 tok) | 27B: off `019ffd12ba95` (239 ch), on `140fe1b2d244` (225 ch) → **differs** (the approved kind of re-baseline: a different GEMM contraction; PPL is on parity/slightly better).  35B: **identical** (`5a3bb565f0ad`). |
| MTP 27B (MMB on) | acceptance **0.80000** (off 0.78070), gen 72.3 t/s (off 70.6), prefill 1153.9 t/s (off ~1000) |

The 27B long-context text re-baseline is expected for a bf16-WMMA weight GEMM (it is not the MMQ
contraction); it is the same class as the approved qsa3 re-baseline.  Width purity (the delivery's
actual contract) holds, and MMB is opt-in (`GGML_CUDA_MMB=1`), so the delivery's default text is
untouched.

## Packaging

* Code: `mmb-gfx1100` commits `df30be334` (patch `0007`, qsa3) and `1359b1c09` (patch `0008`, MMB
  gfx1100 policy).
* Exported to `wip/mmb-general/gfx1100/patches/0008-WIP-mmb-default-the-F32-split-OFF-on-RDNA3_0-gfx1100.patch`
  (sha256 `c60cb88c…`), applied after `0007` on top of the canonical 6.

## Carry-forward

* **S6/S7 are effectively done for the headline win.**  Optional: sweep the routed/GLU thresholds
  and `DBUF` for a further ~1 % on the MoE models.
* **S8 (G3b/c + HC16)** is next: the F32/tiny-M knobs (note the F32 split is now off, so re-check
  whether any F32 shape still wants MMB) and the HC16 bf16 producers (`GGML_CUDA_MMB_HC16=1`).
* **S9** must still re-check the delivery re-examination items (§2.3-§2.7) — the `mmvq` RDNA3_0
  `nwarps` table, the `VDR_Q8_0` MoE choice, the FA head cap 256 (gemma-4 is the probe), and the
  native-KV auto policy — plus the full B1-B9 matrix on the MMB-on tree.
* **The big open question:** MMB is now such a large gfx1100 win that the *delivery's own* gfx1100
  baseline is exposed as weak.  Whether MMB should become a delivery default (rather than opt-in) is
  a maintainer decision; per the WIP rules it stays opt-in here.
