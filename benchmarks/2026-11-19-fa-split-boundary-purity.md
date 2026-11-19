# The FA tile kernel's KV split was keyed on the padded used KV extent: issue #330

Date: 2026-11-19 (r319). Box: 3 x R9700 (gfx1201), ROCm 10.1.0, `ROCBLAS_USE_HIPBLASLT=0` on every
arm, `HIP_VISIBLE_DEVICES` pinned. Model `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` unless
stated. Prompts: `prompts/prose-rdna-boosts.txt` (`fabdec65…`, 5258 tokens) and `prompts/reasoning.txt`
(385 tokens).

## 1. The report, reproduced

[GitHub issue #330](https://github.com/stew675/llama-cpp-rdna-boosts/issues/330): a greedy
`--spec-type draft-mtp` run must reproduce `--spec-type none`, and on a 3-device `-sm tensor` split it
did not.  Both manifests reproduce on the r318 build, to the character:

| manifest (`-n 512`, seed 42, temp 0) | plain | `draft-mtp n3` | equal |
|---|---|---|---|
| A: q8_0 K/V, prose, reasoning off | 1570 chars `120c74935967` | 1559 chars `c1f9d0b3f5b6` | **NO** |
| B: bf16 K/V, reasoning, reasoning on | 2746 chars `7dca501eeb0e` | 2745 chars `759c1a4515c6` | **NO** |

Pre-existing (the B pair is the r315 record's own published pair) and independent of the band's fixup
period `P` (r316), as the issue established.

## 2. Where it lands: a 256-cell KV-padding boundary

`llama_kv_cache::get_n_kv()` pads the graph's `K->ne[1]` to `GGML_PAD(used, 256)`, and the decode
batch is applied **before** it is computed, so a W-row verify batch at position `p` makes the extent
`pad(p + W)` while the plain decode of the same row makes it `pad(p + 1)`.  The two agree except in the
W-1 cell window at each 256-cell multiple, where the verify steps one padding block ahead.

At the FA op the tile launcher's KV split `parallel_blocks` is chosen by an efficiency search over
`ntiles_KV = ceil(K->ne[1] / nbatch_fa)` (`gridDim.y = parallel_blocks`; block b takes KV iterations
b, b+P, b+2P, ..., and `flash_attn_combine_results` adds the P partials in block order).  **P is the
summation order**, so when the two arms' extents land on different padding blocks the same row can be
computed with a different P.

A temporary sampler dump (`common/sampler.cpp`, not shipped) comparing the target's top-2 logits at the
same position in both arms pins it exactly:

* positions 5258..5629: **bit-identical** top-1 and top-2 logits, 373 positions;
* position 5630 onward: they differ, first by 0.025 (plain 27.7634354 against draft 27.7384148), and a
  near tie flips at 5636 (`draft top0=78379 : 20.366, top1=4275 : 20.291` against
  `plain top0=4275 : 20.355, top1=78379 : 20.316`, a 0.04 gap and a swapped pair), after which the
  texts diverge for the rest of the generation.

That is the boundary: the verify at `pmax=5629` has used 5630 -> extent 5632; the next verify has used
5634 -> extent 5888, while the plain decode at that same position still has extent 5632.  A temporary
`FA_SPLIT_TRACE` in the launcher (also not shipped) reads, on the plain arm:

```
nq=1 nkv=5632 ntiles_KV=176 ndst_eff=3 bpw=576 mbps=18 old=175 new=176
nq=1 nkv=5888 ntiles_KV=184 ndst_eff=3 bpw=576 mbps=18 old=183 new=184
```

The old search stops one block below `ntiles_KV` when two candidate split counts print the same integer
efficiency (525/576 and 528/576 both read 91 %), so P = `ntiles_KV - 1`: block 0 owns iterations 0
**and 175** and the combine adds iteration 175 before blocks 1..174.  That is not the ascending order the
`P = ntiles_KV` arms get, so the last bits move.

## 3. The fix (block 05, `ggml/src/ggml-cuda/fattn-common.cuh`)

For the decode/verify band (`Q->ne[1] <= 8`) the split is pinned to a **run constant**: the value the
same efficiency search returns at one full wave (`blocks_per_wave = nsm * max_blocks_per_sm`), then
`parallel_blocks = min(P_const, ntiles_KV)`.  Consequences, all by construction:

* deep KV: P = `P_const` on every step, so a longer extent only *appends* KV iterations (fully masked
  for earlier rows, exact no-ops) and the P partials combine in one fixed order;
* shallow KV (`ntiles_KV < P_const`): P = `ntiles_KV`, i.e. one KV iteration per block -- the combine's
  order is ascending KV, and the trailing blocks above the row's own tile count are neutral (their
  `M = -inf`, `expf(-inf - kqmax) = 0`), which the combine adds as exactly 0;
* crossing `ntiles_KV = P_const` is exact for the same reason: at that point the round-robin split is
  one iteration per block too.

`GGML_CUDA_FA_SPLIT_CONST=0` restores the old `ntiles_KV`-driven search (the A/B arm and the negative
control).  The RDNA4 band kernel already pins its own P (`nsm/2`, r316) for exactly this reason; this is
the tile path, which the band declines below its work gate (10 000 cells for GQA 6 at 256-dim).

## 4. Result: plain == draft-mtp, and only the affected arm moves

| arm (27B Q8_0, 3 GPU `-sm tensor`, `-n 512`) | post-fix plain | post-fix draft | pre-fix plain (`SPLIT_CONST=0`) |
|---|---|---|---|
| A: q8_0, prose | 1559 `c1f9d0b3f5b6` | 1559 `c1f9d0b3f5b6` | 1570 `120c74935967` |
| B: bf16, reasoning | 2689 `2fbc183348a8` | 2689 `2fbc183348a8` | 2746 `7dca501eeb0e` |

The control arm (same binary, `GGML_CUDA_FA_SPLIT_CONST=0`) reproduces the pre-fix text exactly, so the
whole difference is the split choice.  The **3-device `-sm tensor` layout** is what moves: in A the plain
arm moves onto the text the draft already read (`c1f9d0b3f5b6`, the old draft), and in B both arms move (the
old draft `759c1a4515c6` -> `2fbc183348a8`) because the old search keyed *every* band batch on its own padded
extent.  The other layouts are untouched:

| arm | post-fix | pre-fix |
|---|---|---|
| 4B Q8_0, 1 GPU `-sm layer` | `be6407e421c8` | `be6407e421c8` |
| 27B, 2 GPU (0,2) `-sm tensor` | `b1c84ba23a2c` | `b1c84ba23a2c` (issue's published hash) |
| 27B, 3 GPU `-sm layer` | `f59aa0be6536` | `f59aa0be6536` (issue's published hash) |
| 27B, 3 GPU `-sm tensor` | `c1f9d0b3f5b6` | `120c74935967` |

## 5. The instrument: the width probe's boundary arm

`tests/test-logits-width-probe.cpp` compared widths 1..8 at ONE prefix length, and a prefix on a
256-cell multiple (256/4096/10240) puts every width in the same padding block -- which is why the probe
read PASS on this defect.  It now runs a second arm at
`P_b = 256*(P/256 + 1) - NW/2`, the prefix where W=1 lands below the multiple and W=8 above it, and
requires both arms to pass:

| probe arm (27B, 3 GPU `-sm tensor`, `P=4096`) | pre-fix (`SPLIT_CONST=0`) | post-fix |
|---|---|---|
| q8_0, flat `P=4096` | PASS (maxdiff 0) | PASS (maxdiff 0) |
| q8_0, boundary `P=4348` | **FAIL maxdiff 0.232868** | **PASS maxdiff 0** |
| bf16, boundary `P=4348` | **FAIL maxdiff 0.196008** | **PASS maxdiff 0** |
| q8_0, 3 GPU `-sm layer`, boundary | PASS (this layout's `ndst_eff` makes the old search pick the same split) | PASS |

The last row is the reminder that the *mechanism* is a padding-boundary straddle plus a split count,
not a device count: the 3-device `-sm tensor` layout is the shape whose `ndst_eff = 3` (against
`bpw = 576`) made the old search pick the off-by-one split at those extents, while the layer layout's
`ndst_eff` makes it pick `ntiles_KV` and stay ascending.  The fix is layout-independent because it
removes the dependence on the extent entirely.

## 6. Throughput: flat at every depth to 131k

`llama-bench -p 512 -n 128 -d … -r 2/3`, 3 GPU `-sm tensor`, interleaved `1,0,0,1`:

| depth | tg128 `SPLIT_CONST=1` | tg128 `=0` |
|---|---|---|
| d0 | 46.59 / 46.62 | 46.59 / 46.62 |
| d8192 | 45.59 / 45.55 | 45.52 / 45.56 |
| d32768 | 42.27 / 42.27 | 42.26 / 42.29 |
| d65536 | 39.33 / 39.29 | 39.30 / 39.31 |
| d98304 | 36.81 / 36.78 | 36.77 / 36.78 |
| d131072 | 34.59 / 34.57 | 34.57 / 34.57 |

`pp512` is identical at every depth (the change is decode/verify-only).  The split was never the
parallelism: at deep KV `P_const` is the first split count that fills one wave, which is the value the
old search already returned whenever it did not trip the off-by-one.

## 7. The gates this run

* `scripts/gate-spec-purity.sh` (new): manifest A and B `plain == draft-mtp`, byte for byte, with a
  non-vacuity check (the draft arm's Generation t/s >= 1.2x plain's) and `--control` running the pre-fix
  arm and requiring it to DIVERGE.  PASS, control diverges.
* four-axis MTP Protocol A (1 GPU, 27B Q4_K_XL, bf16 KV, `-n 3000`): plain == draft on R / P / C / K,
  acceptance **0.53656 / 0.80437 / 0.86819 / 0.99200**.
* the same four axes on **3 GPU `-sm tensor`, `-n 1000`, q8_0 AND bf16** (the arm the issue asked the
  gate to gain): plain == draft on all eight, acceptance **0.57755 / 0.81776 / 0.86555 / 0.99200**.
* `llama-batched-bench -npl 1,4,8 -npp 512 -ntg 128` (35B-A3B Q8_0, 3 GPU tensor): TG
  **138.04 / 395.29 / 596.86** t/s against r317's **137.50 / 395.24 / 597.08** (noise).
* `test-backend-ops -o FLASH_ATTN_EXT`: **7381/7381**.
* `gate-prefill-logits.sh`: **0.000000 / 100.000 %** at the canonical point and **0.000000 / 99.990 %**
  at `--device 0,1,2 --sm tensor` (a decode-only change, as expected).
* `validate-set.sh` GREEN (tree `9f53a85322`, optional cumulative `bccfacc44`), per-block `--fast`
  ladder 18/18, `check-gpu-faults.sh` clean.
