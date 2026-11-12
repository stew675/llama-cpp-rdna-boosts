# The derived kq mask must take the same FA kernel variants as the packed mask (2026-11-12, r268) — one predicate, and +1 % prefill where the tile launcher is the chosen one

Why this file exists: contributor **PR #303** (`overdoingism`) reported that two launcher-side `gqa_opt`
checks still ask `dst->src[5]` after upstream #29510 took that slot for `kv_rows` and moved this block's
derived kq mask to `src[6..8]`.  The finding was right, but the *effect* is not where the PR claimed it:
the report's own arm (dense 27B, head 256, gfx1201) never executes either line, because a derived op of more
than 8 tokens takes the MMA kernel there and the MMA `ncols2` dispatch already asks
`ggml_cuda_flash_attn_ext_has_mask()`.  This file records what the divergence actually costs, where it is
reachable, and the numbers the r268 amendment shipped with.

**Do not re-derive these numbers, and do not "fix" this by lowering the band's `Q->ne[1] <= 8` gate.**  The
band predicate cannot see a derived op at all (see below); the fix is the *predicate*, not a threshold.

## What was wrong (r236..r267)

| site | asked | should ask |
|---|---|---|
| `ggml_cuda_get_best_fattn_kernel` (fattn.cu) | `has_mask(dst)` = `src[3] \|\| src[5] \|\| src[6]` | — |
| `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2` (fattn.cu) | `has_mask(dst)` | — |
| `ggml_cuda_fattn_band_wmma_applies` (fattn-rdna.cuh) | `mask != nullptr \|\| src[5] != nullptr` | `has_mask(dst)` |
| `launch_fattn_tile_switch_ncols2` (fattn-tile.cuh) | `mask != nullptr \|\| src[5] != nullptr` | `has_mask(dst)` |

`has_mask()` lived as a `static` in `fattn.cu`, so the two `.cuh` launchers kept a hand-copy of an *older*
form of it.  The copies still work for the packed mask (`src[3]`) and the `kv_rows` form (`src[5]`, which
always carries a mask tensor — `ggml.c` asserts it), but a **derived** op has no mask tensor and no
`kv_rows`, so both launchers took the `ncols2 = 1` variants while the same op with a packed mask took
`ncols2 = 4`/`8`.  r268 moves `has_mask()` into `fattn-common.cuh` (ahead of the `fattn-rdna.cuh` include)
and both launchers ask it.

**The band hunk is inert by construction.**  A derived batch is always `n_tokens > 8` with one KV stream
(`llama_kv_cache::kq_mask_derivable()`), i.e. `Q->ne[1] > 8`, while `ggml_cuda_fattn_band_wmma_applies()`
ends with `Q->ne[1] <= 8`.  The two are disjoint, so the change cannot alter a band launch — it is carried so
the four sites cannot drift apart again.

## Where it is reachable at all

A derived op with `K->ne[1] % FATTN_KQ_STRIDE == 0` (the derived form requires `n_kv % 256 == 0`) and
`n_q > 8` takes the WMMA route on gfx1201 for every head inside the arch cap (`GGML_CUDA_FA_WMMA_MAX_HEAD`,
576 there) — so the tile launcher is entered only on:

* RDNA3_5 (cap 320) and RDNA3_0 (cap 256, minus its head-512 row): heads above the cap — e.g. gemma-4-26B-A4B
  (head 512) on Strix Halo, which is the delivery's own validated gfx1151 arm;
* RDNA2 (`amd_wmma_available()` false → the generic tile fallback);
* any arch with `GGML_CUDA_FA_WMMA_256=0` or a lowered `GGML_CUDA_FA_WMMA_MAX_HEAD` (the documented escape
  hatches), which is how the arms below force the code path on gfx1201.

## Measured (gfx1201, 1× R9700, ROCm 10.1.0, `ROCBLAS_USE_HIPBLASLT=0`, fresh process per arm)

**Numerics: byte-identical, so the variant choice is not arithmetic.**  `llama-perplexity
--save-all-logits` on both arms, `-c 512`, f16 KV, tile launcher forced with
`GGML_CUDA_FA_WMMA_MAX_HEAD`:

| model (head / gqa) | tile `ncols2`: old → new | all-logits md5 | r267 vs r268 |
|---|---|---|---|
| Qwen3.5-4B Q8_0 (256 / 4) | 1 → 4 | `ae3b6835dd0d8077bb14eb7cbeaae931` (8 chunks) | identical |
| Qwen3-Coder-30B-A3B Q4_K_M (128 / 8) | 1 → 8 | `dc4ef6213c8fb6a5f7bf709be7db4b56` (6 chunks) | identical |

Both dumps are 1.01 GB and match the *packed-mask* arm as well, so `LLAMA_KQ_MASK_DERIVED=1`, `=0` and the
pre-change build produce the same logits on this path; `gate-prefill-logits.sh --ab
"LLAMA_KQ_MASK_DERIVED=0"` with the launcher forced reads **0.000000 / 100.000 %**
(`prefill-logit-gfx1201-Qwen3.8-27B-Q8_0-tile-gqa.kld`, base PPL 5.9326).

**Speed: prefill, `llama-bench pp512 -n 0`, `-r 3`, interleaved.**

| arm | old (`src[5]` copy) | new (`has_mask`) | delta |
|---|---|---|---|
| Qwen3.5-4B Q8_0, `FA_WMMA_MAX_HEAD=128` | 7861.8 / 7840.5 / 7856.1 | 7955.3 / 7907.5 / 7923.1 (r268 build: 7896.2 / 7892.0 / 7908.8) | **+0.6 … +1.0 %** |
| Qwen3-Coder-30B-A3B Q4_K_M, `FA_WMMA_MAX_HEAD=64` | 3828.1 / 3820.5 | 3881.0 / 3867.1 (r268 build: 3862.3 / 3876.4 / 3881.3) | **+1.3 %** |

Both are inside one standard deviation of the run-to-run spread (±130-180 t/s on the 4B, ±17-60 on the
30B), so the honest reading is "~1 %, in the expected direction, three rounds each" — this is a
consistency fix that happens to be slightly faster, not a tuning win.

**On the default gfx1201 configuration both changed lines are unreachable**, so the canonical gates see
nothing: `gate-prefill-logits.sh` 27B Q8_0 f16 **0.000000 / 100.000 %** and `--kv q8_0` **0.000000 /
100.000 %**, coherence 27B `d45e2221805a`, 4B `1c5d32ac537d`, phi-4 f16 `a1b7eb2ee7a6` — all identical to
r267.  That is expected and is why the forced-launcher arms above exist.

## The oracle, and why the op suite needed a new case

`test-backend-ops -o FLASH_ATTN_EXT` had exactly **one** derived case that reached the tile launcher
(`hsk` 128, gqa 4, `n_kv` 4090, `n_tps` 33 — found with a temporary `REACHED` probe on the pre-r268 tree),
and its `n_kv` is *not* `FATTN_KQ_STRIDE`-aligned, so `use_gqa_opt` stayed false even with the fix: the
suite could not see the change at all (`6978/6978` either way).  r268 adds an aligned head-512 derived case
(`n_kv` 4096, gqa 4), which is the MMA arm on gfx1201 and the **tile GQA** arm on any cap ≤ 320 layout, plus
the raw variant coverage the per-op oracle gives.  With the suite run twice — default, and
`GGML_CUDA_FA_WMMA_MAX_HEAD=128` so the head-256/512 derived cases take the tile launcher too — the result
is **6979/6979 on both arms**.

## What this does NOT establish

* No arm ran on RDNA3_5 or RDNA2 hardware; the tile path was reached through
  `GGML_CUDA_FA_WMMA_MAX_HEAD` on gfx1201, which selects the same launcher and the same `ncols2` rows.
* The MTP four-axis protocol is not part of this: the changed dispatch is unreachable for a derived op on
  gfx1201 (MMA), and the verify-width (`n_q` 9..16) derived cases are covered by the op suite's forced-tile
  arm instead.
* A derived op reaching the DKQ 320 / DKQ 576 tile instances would have **aborted** before this change
  (`GGML_ABORT`, they have no non-GQA arm).  Not observed — it needs a non-MLA head-320/576 model on a
  layout whose chooser picks the tile kernel (MLA models pass `allow_derived = false`, so they never form a
  derived op) — but the one predicate closes the class.
