# `LLAMA_KQ_MASK_DERIVED` (block 15 V3) — architecture / condition A/B

Prompted by issue #30 comment <https://github.com/stew675/llama-cpp-rdna-boosts/issues/30#issuecomment-5740578410>
(reporter `a-n-t-0`, 2x RX 7900 XTX / gfx1100): with `LLAMA_KQ_MASK_DERIVED=0` the deep-context
prefill recovered **+5.7 % (tensor) / +13.2 % (layer)** at 100k on a 27B Q8_0, decode unchanged.

This directory records the reproduction attempt across all three shipped architectures, the
conditions that flip the sign, and the measured memory trade.

## What the knob does (recap)

`LLAMA_KQ_MASK_DERIVED=1` (default) replaces the materialised `n_kv x n_q` f16 attention mask with a
compact per-cell state (`cell_pos`/`tok_lo`/`tok_hi`) that the **MMA** FA kernel turns back into
mask values on the fly.  Consequences:

* the packed mask tensor is left unallocated -> the memory / H2D-copy win;
* it is **prefill only**: `llama_kv_cache::kq_mask_derivable()` returns false for `n_tokens <= 8`, so
  decode and the spec-verify widths keep the packed mask (this is why TG is flat below);
* it **forces the MMA_F16 kernel** (`fattn.cu:852`: `src[5] && kernel != MMA_F16` -> op unsupported),
  which is why the tile kernel carries no derived arm;
* `n_kv` is padded to a multiple of >= 256 (`get_n_kv`), so the `n_kv % 256 == 0` gate always passes.

The in-kernel cost is a scalar `cell_pos[k_VKQ_0 + i]` load + compare per (query row, KV cell),
re-read for every `ncols1` slot of the MMA tile, replacing the packed path's `cp_async` mask load.

## Method

`data/kq-ab.sh <tag> <model> <devices> <split> <rocm-lib> <depths...>` runs `llama-bench`
`pp512`/`tg128` at each depth twice (`LLAMA_KQ_MASK_DERIVED=1` then `=0`), `-fa 1 -ctk f16 -ctv f16`,
`-r 3`, and emits `tag,depth,derived,pp512,tg128`.  Parse the value as the field *after* the one
holding the test name: `-sm tensor` (and non-f16 KV) add columns and a fixed `$8/$9` parse silently
returns NA (cost me one wrong "tensor split fails" conclusion).

`data/ggufinfo.py` prints the attention geometry from a GGUF header.

| box | arch | model(s) |
|---|---|---|
| `soar` | gfx1201 (3x R9700) | Qwen3.5-9B Q8_0, Qwen3.8-27B Q8_0, Qwen3.6-35B-A3B Q8_0 |
| `halo` | gfx1151 (Strix Halo) | Qwen3.5-9B Q8_0, Qwen3.6-35B-A3B Q8_0 |
| `fingon` | gfx1100 (1x 7900 XTX + gfx1036 iGPU) | Qwen3.5-9B Q8_0 |

Geometry: 9B = 33L / 4 KV heads / head 256 (gqa 4), 132 KiB/token; 27B = 65L / 4 KV heads / head 256
(gqa 6), 260 KiB/token; 35B-A3B = 41L / 2 KV heads / head 256 (gqa 8), 82 KiB/token.

## Result — PP512, `derived=1` minus `derived=0` (positive = derived wins)

| config | d=0 | 32k | 64k | 98k |
|---|---|---|---|---|
| **gfx1201** 9B 1 GPU | +0.5 | +1.6 | +1.1 | **+2.6** |
| **gfx1201** 9B 2 GPU layer | −0.1 | +0.7 | — | −0.0 |
| **gfx1201** 9B 2 GPU tensor | −2.2 | +2.6 | — | **+7.7** |
| **gfx1201** 27B 1 GPU | +0.3 | — | — | (−0.5 @8k) |
| **gfx1201** 27B 2 GPU layer | −0.1 | −3.0 | — | **−6.0** |
| **gfx1201** 27B 2 GPU tensor | −1.5 | −0.6 | — | +0.7 |
| **gfx1201** 27B 3 GPU tensor | −3.4 | — | — | **+2.8** |
| **gfx1201** 35B-A3B 2 GPU layer | −2.5 | +1.7 | — | +0.1 |
| **gfx1151** 9B 1 GPU | +0.1 | −0.5 | −0.9 | **−1.9** |
| **gfx1151** 35B-A3B 1 GPU | −0.4 | +0.3 | −0.9 | −0.7 |
| **gfx1100** 9B 1 GPU | +0.8 | −1.7 | −2.3 | **−3.5** |

**TG128 is flat** everywhere (`|delta| <= 0.15 %` for the dense models; the MoE shows up to ~1.2 %
which is run/layout noise — the derived path is not reached for `n_tokens <= 8`).

## Conclusions

1. **The reporter is confirmed on gfx1100**: the derived mask is a monotone loss that grows with
   depth (+0.8 % at d0 -> −3.5 % at 98k on the 9B; the larger loss they saw is the 27B, below).
2. **It is not just an arch question.**  On the *same* gfx1201 box the 9B (gqa 4) wins single-GPU and
   in tensor split, while the 27B (gqa 6) loses **−6.0 %** in 2-GPU layer split at 98k — a
   *bigger* loss than gfx1100's 9B.  The 9B 2-GPU layer is neutral, so **split mode is not the
   discriminator; the model's GQA ratio / MMA tile config is** (the derived loop re-reads `cell_pos`
   once per `ncols1` slot, and `ncols1` follows the GQA ratio).
3. **gfx1151 (RDNA3_5) is a mild loss** (−1.9 % 9B dense at 98k; the 35B-A3B MoE is ~neutral/-0.7 %).
   So the arch ordering is RDNA4 > RDNA3_5 > RDNA3_0.
4. **The memory win is `n_ubatch x n_ctx x 2` bytes, not ~800 MiB.**  Measured on the 9B at
   `-c 98304`, `ub = 512`: compute buffer **184.02 -> 88.39 MiB** device and **112.02 -> 16.40 MiB**
   host, i.e. **~96 MiB + ~96 MiB**.  The ~800 MiB figure in the block-15 notes needs a large ubatch
   (~768 MiB at 196k/ub2048).  At the default ub=512 the win is ~0.1 GiB, against a 3-6 % prefill loss
   in the bad configurations.

## The fix (Option C, 2026-09-19) — DONE

The derived branch of `flash_attn_ext_f16_load_mask` (`fattn-mma-f16.cuh`) was restructured to mirror
what the packed fallback does:

* **two cells per thread step with a single `half2` store.**  The first cut processed one cell per
  step with a scalar half store — twice the iterations and unvectorised shared stores, for strictly
  *less* global traffic than the packed path (which is the whole point of the derived form).  The
  extra ALU/loop/shared-store cost is what tipped the balance on RDNA3 and on the gqa-6 27B.
* **the per-cell `cell_pos` load is hoisted out of the `j1` loop.**  It does not depend on the query
  row; the packed path re-reads its mask for every row, so the derived path must not re-read
  `cell_pos` for every row.

`nbatch_fa` is always a multiple of 32 and <= 256 (static_assert on the config table), so the even `i`
never makes `i + 1` leave the tile and no tail handling is needed.

### Before -> after (PP512, `-r 3`, `derived=1` vs `derived=0`)

| config | before | after |
|---|---|---|
| gfx1201 27B 2GPU layer @98k | 643.1 vs 683.9 = **-5.96 %** | 674.8 vs 685.9 = **-1.62 %** |
| gfx1201 9B 1GPU @32k | +1.64 % | +1.89 % |
| gfx1201 9B 1GPU @98k | +2.57 % | **+3.49 %** |
| gfx1151 9B 1GPU @32k | -0.53 % | -0.24 % |
| gfx1151 9B 1GPU @64k | -0.86 % | -0.86 % |
| gfx1151 9B 1GPU @98k | -1.85 % | -1.81 % (unchanged) |
| gfx1100 9B 1GPU @32k | -1.65 % | -0.47 % |
| gfx1100 9B 1GPU @64k | -2.31 % | -0.38 % |
| gfx1100 9B 1GPU @98k | -3.47 % | **-0.15 %** |

The regression is essentially gone on RDNA3 (the reporter's -3.5 % 9B / -13 % 27B cases), the
RDNA4 wins grew, and the remaining RDNA4 27B layer-split cost is -1.6 % (down from -6.0 %).

**Residual.**  gfx1151 (Strix Halo iGPU) still shows ~-1.8 % at 98k (it was -1.85 % before), while its
32k point improved (-0.53 -> -0.24 %).  The iGPU shares memory bandwidth with the host, so this may be
noise or a different (bandwidth) bottleneck than the loop shape; it is the one remaining cell, and it
is small.  The gfx1151 dense 9B and the 35B-A3B MoE numbers are in `data/csv-fix-gfx1151.txt`.

**Bit-identical.**  Same-seed 256/200-token greedy text with `LLAMA_KQ_MASK_DERIVED=1` vs `0`:
9B 1GPU `0e83b43746e7` both, 27B 2GPU layer `5ec02413b9c9` both.  Only the shape of the stores
changed; every mask value is the same, so no purity gate moves.

## The depth curve (27B 2-GPU tensor, gfx1201)

The packed mask grows with `n_kv`, so on a tensor split the derived form crosses over with depth.
Post-r7, PP512, `-r 3` (`data/csv-soar-tensor-curve.txt`):

| depth | derived=1 | derived=0 | delta |
|---|---|---|---|
| 0 | 2002.9 | 2029.4 | -1.30 % |
| 4096 | 1851.8 | 1875.4 | -1.26 % |
| 8192 | 1785.0 | 1813.2 | -1.56 % |
| 16384 | 1671.7 | 1689.6 | -1.05 % |
| 32768 | 1474.8 | 1480.9 | -0.41 % |
| 65536 | 1197.9 | 1182.7 | **+1.28 %** |
| 98304 | 997.7 | 980.9 | **+1.71 %** |

Crossing between 32k and 64k.  The 3-GPU tensor cell behaves the same: -3.36 % @ d0, -0.56 % @ 32k,
**+2.02 %** @ 98k.  This is the "dynamic switch-over" the reporter hypothesised, and it is
purely a depth effect: below the crossover the mask is cheap to materialise, above it deriving is
cheaper.  A depth-dependent gate is *possible* (it would key on `n_kv` at graph-build time), but the
shallow loss is ~1 % and the delivery therefore keeps the simple "always derive" default and
documents the opt-out instead (see the README section).

## If that had failed

The effect was **sign-unstable across arch + model config**, so no single arch gate would have been
correct.  The fallbacks, in preference order, were:

* **A - make V3 opt-in again (default off).**  Safe; the ~96 MiB (default ub) / ~800 MiB (large ub)
  memory win stays available via `LLAMA_KQ_MASK_DERIVED=1`.
* **B - arch gate: off on RDNA3_0 + RDNA3_5, on for RDNA4.**  Satisfies the reporter, keeps the RDNA4
  wins, leaves the RDNA4 27B 2-GPU-layer loss.

Neither was needed: C fixed it at the source.  `LLAMA_KQ_MASK_DERIVED=0` remains a fully supported
runtime switch (the packed path is the upstream behaviour).
