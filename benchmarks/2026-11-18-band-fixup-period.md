# The RDNA4 band's fixup period `P`: one constant, `nsm/2`, for every K/V type (r316, block 05, `TODO` #346)

> **Amended by r322 (issue #333):** the width compromise measured below stands, but "one constant" was one
> constant *per shape* and this record's shape was a 1-GPU run (4 output tiles per device, where `nsm/2` gives
> 64 blocks = the saturation point).  A tensor split cuts the per-device tile count, so the constant has to
> scale with it; see `benchmarks/2026-11-19-band-split-blocks-per-sm.md`.  Every `P` table below was read on
> 4 tiles, which is why 16 measured best here.

Everything here was measured on the primary box: 3× RX 9070 XT / R9700 (gfx1201), ROCm 10.1.0,
`~/llama-scratch` (one long-lived source path, `build-rocm`), the r315 block-05 chain plus this change.
`ROCBLAS_USE_HIPBLASLT=0` on every fresh-process comparison (INVARIANTS r44).  `nsm` as ggml counts it on this
part is **32** (64 CUs).

**What changed.** `ggml_cuda_fattn_band_wmma_split()` returned `nsm` for the native-quantized K/V types and
`nsm/2` for the 2-byte ones (f16/bf16).  It now returns `max(2, nsm/2)` for **every** type -- `P` 16 on this
part -- so `ggml_cuda_fattn_band_wmma_two_byte()` and the launcher's `P` fallback are gone and
`GGML_HIP_FA_BAND_WMMA_SPLIT` remains the tuning override.  `P` is still a per-device constant that cannot
depend on `n_q` or on the KV extent, so decode and verify keep one summation order (`GREEDY-PURITY.md` §98).

**Why one constant has to be a compromise.** `P` is the period of the fixup layout (`b = tile*P + i`), and the
optimum differs by *width*: the decode width wants the larger value, the verify widths want half of it.  The
shipping number is therefore chosen on end-to-end arms (§3), not on the op table alone.

## 1. Per op: the sweep that the item's 36-row table came from, extended to the comments' own range

`test-backend-ops perf -o FLASH_ATTN_EXT`, band forced on (`GGML_HIP_FA_BAND_WMMA_H256=1` /
`GGML_HIP_FA_BAND_WMMA_H128=1`) with `GGML_HIP_FA_BAND_WMMA_SPLIT` = 8/16/24/32/48/64/96, `us/run`, one R9700.
The harness is a temporary block of perf cases in `tests/test-backend-ops.cpp` (stripped before the patch was
cut).  **q8_0 rows are the median of 3 independent processes**; the f16 rows are a single pass and are quoted
only where the f16 family has to be shown.

### 1.1 The 256-dim class (head 256, 4 K/V heads, GQA 6 -- Qwen3.8-27B), q8_0 K/V

| kv | n_q | P8 | **P16** | P24 | P32 (old default) | P48 | P64 | P96 | best |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 1 | 49.9 | **44.2** | 57.1 | 64.1 | 46.6 | 54.0 | 68.7 | P16 |
| 512 | 8 | 54.7 | 71.3 | **48.3** | 53.7 | 67.1 | 79.7 | 103.6 | P24 |
| 1024 | 1 | 52.6 | 55.1 | 44.0 | **42.6** | 51.8 | 59.3 | 73.8 | P32 |
| 1024 | 8 | 57.5 | 52.5 | **50.5** | 57.0 | 70.5 | 83.2 | 107.3 | P24 |
| 2048 | 1 | 57.7 | **43.9** | 48.3 | 50.7 | 55.5 | 63.7 | 78.4 | P16 |
| 2048 | 8 | **48.3** | 52.7 | 59.8 | 67.4 | 79.0 | 91.9 | 117.2 | P8 |
| 4096 | 1 | 51.8 | **45.3** | 61.7 | 54.4 | 65.3 | 74.1 | 86.5 | P16 |
| 4096 | 8 | **62.1** | 74.0 | 81.8 | 90.0 | 107.5 | 124.0 | 152.3 | P8 |
| 8192 | 1 | 68.8 | **66.3** | 74.2 | 72.0 | 76.8 | 83.7 | 100.7 | P16 |
| 8192 | 8 | **95.1** | 105.4 | 111.2 | 117.3 | 134.9 | 151.8 | 185.5 | P8 |
| 16384 | 1 | 104.0 | **92.9** | 98.5 | 99.5 | 103.9 | 117.2 | 126.6 | P16 |
| 16384 | 8 | **150.1** | 170.2 | 171.0 | 179.7 | 195.4 | 211.6 | 259.7 | P8 |
| 20480 | 1 | 122.8 | **106.1** | 122.7 | 112.4 | 119.8 | 124.0 | 140.8 | P16 |
| 20480 | 8 | **179.6** | 198.0 | 199.0 | 209.9 | 225.6 | 240.5 | 288.9 | P8 |
| 32768 | 1 | 180.9 | **151.2** | 162.0 | 163.9 | 175.0 | 190.9 | 223.2 | P16 |
| 32768 | 8 | **269.5** | 279.7 | 304.8 | 318.4 | 344.9 | 368.6 | 401.1 | P8 |
| 102400 | 1 | 558.2 | **425.0** | 467.2 | 447.5 | 457.2 | 467.4 | 491.5 | P16 |
| 102400 | 8 | **797.1** | 817.3 | 830.6 | 840.2 | 862.6 | 881.2 | 924.4 | P8 |

Reading it: **the decode width's optimum is 16 and the verify widths' is 8**, monotonically in `P` at the
verify widths and with a clear minimum at 16 at the decode width above 2048 cells.  P16 against the old
quantized default of 32 is worth **+5 % to +14 %** everywhere the band actually runs (kv ≥ 16384: 92.9 vs 99.5,
106.1 vs 112.4, 151.2 vs 163.9, 425.0 vs 447.5 at `n_q` 1; 170.2 vs 179.7, 198.0 vs 209.9, 279.7 vs 318.4 at
`n_q` 8), and it loses only at the short caches every band gate keeps off -- including the item's anomalous row,
which is **real and reproducible** (1024 cells, `n_q` 1: P16 55.0 ± 0.1 against P32 42.6 ± 0.1, three runs
each; see §5 for why it is unreachable).

### 1.2 The 128-dim class (head 128, 8 K/V heads, GQA 10 -- Ripple-65B), q8_0 K/V

| kv | n_q | P8 | P16 | P24 | P32 (old default) | P48 | P64 | P96 | best |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | **33.4** | 40.1 | 39.2 | 43.5 | 52.9 | 59.4 | 73.8 | P8 |
| 1024 | 8 | **44.2** | 55.3 | 60.2 | 69.5 | 87.3 | 105.4 | 138.8 | P8 |
| 2048 | 1 | **38.6** | 43.9 | 47.8 | 54.6 | 58.7 | 65.9 | 80.2 | P8 |
| 2048 | 8 | **57.7** | 67.1 | 76.2 | 87.4 | 102.6 | 120.7 | 154.6 | P8 |
| 4096 | 1 | **55.4** | 56.6 | 61.1 | 65.4 | 73.3 | 83.2 | 95.0 | P8 |
| 4096 | 8 | **87.7** | 93.7 | 102.7 | 112.3 | 132.6 | 152.6 | 186.7 | P8 |
| 8192 | 1 | 87.7 | **86.2** | 88.1 | 91.7 | 99.4 | 108.1 | 122.2 | P16 |
| 8192 | 8 | **148.6** | 149.7 | 157.2 | 166.0 | 184.7 | 204.5 | 260.9 | P8 |
| 16384 | 1 | 155.1 | 146.4 | **144.8** | 145.9 | 152.5 | 160.8 | 178.0 | P24 |
| 16384 | 8 | 271.2 | **261.0** | 266.5 | 274.4 | 295.8 | 325.8 | 374.1 | P16 |
| 32768 | 1 | 296.0 | 281.3 | 277.5 | **276.1** | 281.0 | 290.3 | 307.3 | P32 |
| 32768 | 8 | 527.8 | 507.0 | **505.8** | 512.8 | 528.9 | 548.7 | 590.5 | P24 |

The class prefers a *smaller* `P` than the 256-dim one (its blocks are 128-dim with `ncols1` 4/2), and at its
deepest decode row P32 is 1.9 % better than P16.  P16 against P32 is a win at every cell ≤ 16384 and a ≤ 2 %
loss at 32768/`n_q` 1, so the change is a win or neutral for this class too -- and §3 shows the end-to-end
Ripple arm is flat.

### 1.3 The 2-byte family, and reconciling the old comment

The in-code note next to the helper claimed, from an earlier measurement, that "the verify cost is flat for
`P` = 48..96 at kv 20K..200K and degrades below 32 or above 128".  **It does not reproduce on this kernel.**
f16, single pass (`us/run`):

| shape | n_q | P8 | P16 | P32 | P64 |
|---|---|---|---|---|---|
| 256-dim, kv 16384 | 1 | 121.3 | **97.7** | 130.5 | 183.7 |
| 256-dim, kv 16384 | 8 | 144.9 | 172.5 | 201.2 | 254.7 |
| 256-dim, kv 20480 | 8 | 194.3 | 215.7 | 241.0 | 288.9 |
| 256-dim, kv 32768 | 8 | 289.7 | 312.3 | 337.3 | 385.2 |
| 128-dim/10, kv 16384 | 1 | 118.8 | 132.6 | 160.1 | 209.0 |
| 128-dim/10, kv 32768 | 1 | 254.7 | 275.2 | 308.6 | 336.1 |

At the **verify** widths the curve is monotone in `P` (each doubling costs 10-25 %); nothing is flat between 48
and 96.  What *is* flat-ish above 24 is the **decode** width at a long KV (256-dim q8_0, kv 102400: P24 467.2,
P32 447.5, P48 457.2, P64 467.4, P96 491.5), and the "degrades below 32" half reproduces there too (P8 558.2).
So the old note is a *decode-width, long-KV* measurement whose scope was mis-stated, which is why it and the
item's verify-width table disagreed.  Both comments now say what the numbers say.

## 2. Below the gates: the tile kernel, and the crossover the two gates read

Band ÷ tile (`test-backend-ops perf`, q8_0, one pass per arm), P16 band against `GGML_HIP_FA_BAND_WMMA=0`:

| shape | cells | n_q | band P16 | tile | ratio |
|---|---|---|---|---|---|
| 256-dim GQA 6 | 8192 | 1 | 66.3 | 62.2 | 1.07 (loss) |
| 256-dim GQA 6 | 16384 | 1 | 92.9 | 109.1 | **0.85** |
| 256-dim GQA 6 | 20480 | 1 | 106.1 | 138.2 | **0.77** |
| 256-dim GQA 6 | 32768 | 1 | 151.2 | 222.0 | **0.68** |
| 256-dim GQA 6 | 16384 | 8 | 170.2 | 733.5 | **0.23** |
| 128-dim GQA 10 | 4096 | 1 | 56.6 | 53.4 | 1.06 (loss) |
| 128-dim GQA 10 | 8192 | 1 | 86.2 | 97.7 | **0.88** |
| 128-dim GQA 10 | 16384 | 1 | 146.4 | 184.4 | **0.79** |
| 128-dim GQA 10 | 32768 | 1 | 281.3 | 368.0 | **0.76** |
| 128-dim GQA 5 | 8192 | 1 | 60.2 | 97.5 | **0.62** |
| 128-dim GQA 7 | 4096 | 1 | 42.3 | 70.5 | **0.60** |
| 128-dim GQA 12 | 8192 | 1 | 87.0 | 82.2 | 1.06 (tile correctly chosen) |
| 128-dim GQA 12 | 32768 | 1 | 281.8 | 309.7 | **0.91** |

**Both gates stay correctly calibrated with P16** and their decisions are unchanged in cell terms
(`K->ne[1] * passes_saved >= 20000`: band from 16384 cells for the 256-dim GQA 6 class, from 8192 for the
128-dim ratio 5 / 10, 4096 for ratio 7, 32768 for ratio 12).  P16 only improves the margins on the band side
(the 128-dim ratio-10 row at 4096 cells moves 1.22 → 1.06 and the 256-dim 8192-cell row 1.16 → 1.07); no
`TILE` decision that the gate makes is a measured loss, and no `BAND` decision is a loss either.

## 3. End to end

3 GPUs `-sm tensor`, `-t 16`, **q8_0 K/V** (the arm the change acts on), `ROCBLAS_USE_HIPBLASLT=0`, both arms of
the same binary (`GGML_HIP_FA_BAND_WMMA_SPLIT` is exactly equivalent to the shipped constant when only one K/V
type is in play), interleaved `A B B A`.  Two runs per arm.

| arm | P16 (shipped) | P32 (old default) | delta |
|---|---|---|---|
| 27B Q8_0 tg64 d0 (band off -- control) | 45.45 | 45.53 | -0.2 % (noise) |
| 27B tg64 d16384 | 43.63 / 43.69 | 43.52 / 43.53 | +0.3 % |
| 27B tg64 d32768 | 42.41 / 42.48 | 42.61 / 42.63 | -0.4 % |
| Ripple-65B tg64 d16384 | 26.07 / 26.07 | 25.97 / 25.95 | +0.4 % |
| Ripple-65B tg64 d32768 | 25.42 / 25.39 | 25.44 / 25.43 | -0.1 % |
| 27B `draft-mtp n7`, 20.5K prose prompt | **88.74 / 89.07** | 75.83 / 75.82 | **+17.1 %** |
| 27B `draft-mtp n7`, ~10.5K prose prompt | **79.21 / 78.78** | 74.87 / 74.95 | **+5.8 %** |

Plain decode is at the DRAM roof (a 27.3 B Q8_0 model reads ~1.2 TB/s per token) and the band is a small part
of a step, so ±0.4 % there is noise -- this matches r315's finding that the band is a verify-width kernel.
**The attention-bound arm is where P pays.**  Decomposing the 20.5K row: mean accepted length 3.82 against 3.26
gives 26.6 against 30.3 steps for the same 128 tokens, i.e. **~4.5 % per step is the kernel** and the rest is
the acceptance re-roll that any numeric-order change carries (a different summation order re-rolls greedy
near-ties; it can move either way, and it is a property of the *run*, not of the kernel).  The 10.5K row,
where acceptance barely moved (0.33333 vs 0.32222), shows the kernel's own share as **+4.5-5 % per step**.

`P` 8 was measured on the same arms and **not** taken, even though it wins the op-level verify rows: 40.86 t/s
at d32768 (against P16's 42.45), 78.33 on the 10.5K arm and 75.68 on the 20.5K arm.  The verify widths of a
real MTP step are not the `n_q` 8 op case; the constant has to be chosen end to end.

`llama-batched-bench -c 32768 -b 2048 -ub 512 -npp 2048 -ntg 64 -npl 1,4,8` (27B, q8_0 K/V): tg
44.35 / 151.21 / 257.49 against 44.31 / 151.28 / 257.55 t/s, aggregate 878.90 / 1583.44 / 1811.30 against
878.96 / 1584.09 / 1812.73 S t/s -- unchanged.

## 4. Correctness and purity

* `test-backend-ops -o FLASH_ATTN_EXT`: **7381/7381** on each of the 3 GPU backends in all four arms --
  default (`auto`, both work gates), `GGML_HIP_FA_BAND_WMMA_H256=1 GGML_HIP_FA_BAND_WMMA_H128=1` (the band
  itself), `GGML_HIP_FA_BAND_WMMA=0` (tile) and `GGML_HIP_FA_BAND_WMMA=2` (the `ncols1 = 2` variant).
* `test-logits-width-probe` -- **37 arms, PASS, worst maxdiff 0**: 27B (head 256) at P 3584 / 4096 / 10240 ×
  `KV=q8_0/f16/bf16` × {1 GPU, `SPLIT=tensor`, `SPLIT=layer`} = 27 (P 10240 is where the 256-dim gate itself
  selects the band, 10496 cells); Ripple-65B (head 128) at P 3584 / 4096 / 10240 × `KV=q8_0` ×
  {`SPLIT=tensor`, `SPLIT=layer`} = 6, plus `KV=f16` at P 10240 × 2.  A single-GPU Ripple arm is impossible (a
  45 GiB model on a 32 GiB card: "model load failed"), which is why the 128-dim class is quoted on the
  multi-device arms only.  A constant `P` is width-pure by construction; this is the proof that it is.
* `gate-prefill-logits.sh`: canonical point (27B, f16 K/V, `--device 0`) **mean KLD 0.000000 / same-top-p
  100.000 %** against the recorded base.  Family-switch point, `--kv q8_0 --ub 4 --ctx 16384 --chunks 2
  --ab "GGML_HIP_FA_BAND_WMMA_SPLIT=32"` (base = the old behaviour), on both head classes:

  | class | candidate (P16) | same-kernel control (`-ub` 256) |
  |---|---|---|
  | 27B (head 256) | **0.002733 / 99.243 %** | 0.004182 / 98.822 % |
  | Ripple-65B (head 128) | **0.000823 / 98.981 %** | 0.004133 / 97.772 % |

  Both inside the limits and both **better than a benign ubatch reordering of the same kernel** -- the point
  has to be banded for P to be visible at all (a prefill batch at the default ubatch is not), which is why the
  arm is `-ub 4`.
* The four MTP axes (27B, `draft-mtp n3`, `-n 512`, `-c 102400`, `--reasoning on` for R and off for P/C/K):
  **byte-identical to r315** on bf16 K/V -- R plain `7dca501eeb0e` / draft `759c1a4515c6`, P `fcc0bdeb636b`,
  C `eb89102cb7d8`, K `a87c4318b649`, acceptance 348/486, 357/460, 368/427, 372/375 reproduced exactly.  The
  same axes on q8_0 K/V (R `000b437b135b`, P plain `120c74935967` / draft `c1f9d0b3f5b6`, C `eb89102cb7d8`,
  K `a87c4318b649`, acceptance R 346/489, P 355/467, C 369/424, K 372/375) reproduce **r315's own q8_0 arms
  under all three P values** -- the axes sit below the band gate (~5-6K cells), so they are a regression check
  for this change, not a test of it.  The P axis's plain/draft divergence on q8_0 (`120c74935967` vs
  `c1f9d0b3f5b6`) is **inherited from r315 and P-independent** (identical at P 16 and P 32 on the r315 build).
  *Corrected after this record was published:* this file and `WORKLOG.md` 261118.011 first attributed that
  divergence to the quantized-cache fold family.  **That is refuted** -- disabling all six fold switches
  (`GGML_CUDA_KV_FOLD_Q8_0`, `GGML_CUDA_QK_ROTATE_HADAMARD`, `GGML_CUDA_FUSE_V_FWHT`,
  `GGML_CUDA_FA_COMBINE_FWHT`, `GGML_CUDA_QK_PAIR_FWHT`, `GGML_CUDA_FUSE_ROPE_FWHT`) leaves both hashes
  unchanged -- and it is not cache-type specific either: the **bf16** reasoning axis diverges the same way on
  the same split (`7dca501eeb0e` vs `759c1a4515c6`, which is r315's own published pair).  Both need
  **3 devices on `-sm tensor`** and are equal on 1 device, on 2 devices and on 3 devices `-sm layer`.  It is
  tracked as **issue #330**; it is a pre-existing greedy-purity question about the tensor split, not this
  change and not a fold.
* Coherence (3 GPU `-sm tensor`, `-c 4096/8192/2048`, `-n 96`, `--seed 42 --temp 0`, no `-lv`): with the default
  f16 K/V **nothing moved** -- 4B `8d89080db673`, 27B `da2e2d192e21`, 35B-A3B `265c381dbc11`, Flash-Next
  `de61b9b1bed7`, Ripple-65B (`-c 2048`) `c1e00039b55c`, all identical to r315.  With q8_0 K/V and a deep
  prompt (where the band runs): 27B moves `f7d157b1e07e` (P32) -> **`774d1c325068`**, which is **exactly the
  r315 build's `GGML_HIP_FA_BAND_WMMA_SPLIT=16` arm**, and `SPLIT=32` on the new build reproduces
  `f7d157b1e07e` again (the override still pins).  35B-A3B (`282ceadd4028`) and Ripple-65B (`47b936867d64`) do
  not move at all -- P16 and P32 agree on those two prompts.
* `gate-per-block-build.sh --fast --repo ~/llama-scratch --tip bc4e8a5c9 --changed-from 2a81262f6`: **2/2**
  (block 05 and the tip; every other block's delta is unchanged).  `validate-set.sh` GREEN, canonical tree
  `4119a6f86`.  GPU kernel journal and every run log clean (`check-gpu-faults.sh`).

## 5. Why the short-cache losses do not matter, and what is left open

The one place P16 loses to P32 is the short cache (1024 cells, `n_q` 1: 55.0 against 42.6 us).  Every band gate
keeps the band off there -- 10000 cells for the 256-dim class, 5-9k (ratio-dependent) for the 128-dim one -- and
with the *tile* kernel on those shapes the question does not arise.  It is recorded rather than hidden because
a future gate relaxation would walk straight into it.

**Open, and deliberately not acted on:** the verify widths' own optimum is 8 for *both* type families, and the
2-byte family still carries the r92-era compromise 16 (`P` 8 is 12 % faster at 256-dim kv 16384/`n_q` 8 and
16 % at kv 512).  A single constant cannot be optimal for both widths, `P` 8 loses the decode width and every
end-to-end arm measured here, and keying `P` on `n_q` is forbidden (`GREEDY-PURITY.md` §98).  If a future
change ever makes the verify width dominant end to end (a deeper default draft depth, say), this table is the
starting point.

## 6. Not validated

gfx1100 / gfx1151 (`fingon` / `halo`) and Windows were not exercised; `P` is expressed as a fraction of `nsm`
so a smaller RDNA4 part scales down with it, and the kernels are the ones both classes already ran.  A 128-dim
GQA-5/7/12 *model* was not available -- those ratios were measured at the op level only (§2).
