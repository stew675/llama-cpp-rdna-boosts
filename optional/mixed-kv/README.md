# `mixed-kv` — a q8_0 K cache with a smaller V cache, read natively

**Optional artifact, not part of the delivery.  Cut for `r280`** (against `release.json` tree
`b83424ef5932ca6d23f8397b3dc51480c3c8c16f`).  **Built and measured on `r279`** (tree `d6372d47ae…`): `r280`
differs from `r279` in `tests/test-llama-archs.cpp` only, which this patch does not touch and none of the
checks below build or run.

`--cache-type-k q8_0` with `--cache-type-v q5_1`, `q5_0` or `q4_0`, on models whose attention heads are all
256 wide (the Qwen3.5 / 3.6 / 3.8 hybrids, dense and MoE).  The delivery refuses every mixed pair today.

* `mixed-kv.patch` — the artifact.  sha256
  `ce136de5612f712eb2f83f0f6477ba79b59190dcc5bd170ac3364be56f85f113`
* applies at `release.json` tree `b83424ef59…`, produces tree `86fcba68af9dfd0afdeb704913ac492d18e8891c`
  (verified by strict `git am` on a checkout of `.tip` `f25fd3ff2`).  The same file applies at `r279`'s
  `f61ac4bd8` and produces `ea6e3d7d5ac63e191521a018b0f8f37b3ad4c00a`, the tree everything below ran on
* 11 files, 162 insertions / 42 deletions: `ggml/src/ggml-cuda/fattn-tile.cuh`, `fattn-rdna.cuh`,
  `fattn-tile.cu`, `fattn.cu`, `template-instances/generate_cu_files.py` and three generated tile instances,
  `src/llama-context.cpp`, `tests/test-backend-ops.cpp`, `tests/test-logits-width-probe.cpp`
* the two existing artifacts (`host-pool-fill-src`, `two-lane-prefill`) did not apply at `f61ac4bd8` on their own
  when this was cut (they fail in files this patch does not touch), so the combined apply is not verified

## Why the delivery refuses mixed pairs, and what this changes

`llama_init_from_model` rejects `type_k != type_v` with the note that every mixed pair measured 1.7-3.6x slower
than its same-type equivalent and never smaller.  The cause is in the kernel selection, not in the pair:

* the tile kernel has ONE K/V type parameter, so `ggml_cuda_fattn_tile_kv_native_type()` returns NONE for a
  mixed pair and the launcher stages BOTH operands through the F16 scratch, on every decode step;
* the RDNA4 decode/verify band asks for `kv_native(K) == kv_native(V)`, so a mixed pair leaves the band and
  takes that staged tile kernel too;
* the MMA kernel already takes a native type per operand (`kv_native.type_K` / `.type_V`), so prefill was
  never affected.

The patch gives the tile kernel a second type parameter for V (`type_V`, defaulting to `type_KV`).  K is loaded
with K's dequant arm and V with V's; the arms are the existing ones.  Three instances are generated, for head
256 only: K `q8_0` with V `q4_0`, `q5_0`, `q5_1`.  `ggml_cuda_fattn_kv_mixed_native()` names those pairs, the
band accepts them, and `ggml_cuda_fattn_tile_type()` / `_type_V()` keep the allocator and the dispatch on one
predicate.  The context gate accepts the three pairs when every attention layer has `n_embd_head_k == 256` and
`n_embd_head_v == 256`, and keeps refusing everything else, with the pair named in the error text.

`GGML_CUDA_FA_KV_MIXED=0` switches the native read of a mixed pair off (the staged path, as before).  It is a
kill switch: unset means on.  A same-type cache does not reach any of the new code.

## Verification (r279, the delivery's default settings)

RX 9070 (gfx1201) + RX 7900 GRE (gfx1100).  "Stock" is `f61ac4bd8` built from the same tree with the same
options; "patched" is stock plus this patch.

| check | result |
|---|---|
| `test-backend-ops test -o FLASH_ATTN_EXT` | 7021/7021 on gfx1100 and on gfx1201, 3/3 backends.  The patch adds 24 cases: K `q8_0` with V `q4_0` / `q5_0` / `q5_1`, head 256, GQA 4 and 6, `nb` 1 / 3 / 8 / 32 |
| default path, patched against stock, K = V = `q8_0` (`llama-perplexity --kl-divergence`, ctx 512, 40 chunks) | mean KLD 0.000000, same top p 100.000 % on the tensor split, on gfx1201 alone and on gfx1100 alone; maximum KLD 0.000066 / 0.000059 / 0.000057 |
| the same tool, stock against stock, tensor split | mean KLD 0.000000, same top p 100.000 %, maximum KLD 0.000066: the maximum above is the tool's floor |
| stock with K `q8_0`, V `q5_1` | refused at load, as before |
| `test-logits-width-probe … 4096 1024`, `W = 1..8` | PASS, worst maxdiff 0, in all 16 cells below |
| kernel journal, GPU faults | none over any run |

Width probe (the patch adds `KVV=<type>` to the tool for the V cache; `KV` stays the K cache):

| K / V | tensor split, `RS=0` | tensor split, `RS=7` | gfx1201 alone | gfx1100 alone |
|---|---|---|---|---|
| `q8_0` / `q8_0` | `3a41a784e5b4ddfe` | `3a41a784e5b4ddfe` | `09a1bc34606b5a18` | `b253559ae2a370e6` |
| `q8_0` / `q5_1` | `a4378447daad2245` | `a4378447daad2245` | `af7adb1c0d46acba` | `d55f8656cf3f08c7` |
| `q8_0` / `q5_0` | `221e36202da68a4e` | `221e36202da68a4e` | `944372ffc6f1c15b` | `ae262c5d90e9e17b` |
| `q8_0` / `q4_0` | `49a85df9f1db0946` | `49a85df9f1db0946` | `9305f211d2100d44` | `8915b234a3eb92a6` |

Tensor split: Qwen3.8-27B Q4_K_M.  One card: Qwen3.8-27B UD-IQ4_XS.  Text: WikiText-2 test.

## What it measured

### Setup

* RX 9070 16 GB (gfx1201) in the CPU slot, PCIe 4.0 x16; RX 7900 GRE 16 GB (gfx1100) in the **chipset** slot of
  a B550 board, PCIe 3.0 x4.  Ryzen 7 5700X3D, 64 GB.  Linux 7.3.3 (CachyOS), `linux-firmware-amdgpu` 20261019.
* Built with ROCm 7.2.4 (AMD clang 22), `-DGGML_HIP=ON -DGPU_TARGETS="gfx1100;gfx1201"`, Release; run against
  the ROCm 10.1 runtime libraries (HIP 7.16).
* No `GGML_CUDA_*` / `LLAMA_*` variables set.  `-fa on -ngl 99 -ub 1024`.  Tensor split: `-sm tensor -dev
  ROCm1/ROCm0` (the 9070 first), even split.
* `llama-bench -p 2048 -n 64 -d 0,32768 -r 3` unless a table says otherwise; tok/s, the standard deviation of
  the three repeats in brackets where it matters.  Arms are compared inside one session only: the same arm
  moves by up to 0.7 % between sessions.

### VRAM at load, tensor split, `-c 204800`, Qwen3.8-27B Q4_K_M (plain `llama-server`, no draft, no mmproj)

| build, K / V | GRE + 9070, MiB | against `q8_0` / `q8_0` |
|---|---:|---:|
| stock, `q8_0` / `q8_0` | 12161 + 12168 | |
| patched, `q8_0` / `q8_0` | 12161 + 12168 | 0 |
| patched, `q8_0` / `q5_1` | 11661 + 11665 | −1003 |
| patched, `q8_0` / `q5_0` | 11561 + 11565 | −1203 |
| patched, `q8_0` / `q4_0` | 11361 + 11365 | −1603 |
| patched, `q8_0` / `q4_0`, `GGML_CUDA_FA_KV_MIXED=0` | 11929 + 11897 | −503 (at load; the staging arena is taken in use) |

### Quality against an f16 KV cache (tensor split, WikiText-2 test, 8 chunks of 4096)

| K / V | mean KLD | median | 99 % | same top p | PPL / PPL(f16) |
|---|---:|---:|---:|---:|---:|
| `q8_0` / `q8_0` | 0.0030 ± 0.0008 | 0.00041 | 0.015 | 98.44 ± 0.10 % | 1.0045 |
| `q8_0` / `q5_1` | 0.0036 ± 0.0008 | 0.00053 | 0.018 | 98.24 ± 0.10 % | 1.0061 |
| `q8_0` / `q5_0` | 0.0045 ± 0.0012 | 0.00057 | 0.018 | 98.21 ± 0.10 % | 1.0031 |
| `q8_0` / `q4_0` | 0.0057 ± 0.0013 | 0.00100 | 0.029 | 97.78 ± 0.12 % | 1.0087 |

### Speed, tensor split, Qwen3.8-27B Q4_K_M

| build, K / V | prefill d0 | decode d0 | prefill d32768 | decode d32768 |
|---|---:|---:|---:|---:|
| stock, `q8_0` / `q8_0` | 470.4 | 38.19 (0.38) | 413.4 | 36.44 (0.59) |
| patched, `q8_0` / `q8_0` | 470.3 | 38.08 (0.35) | 413.3 | 36.35 (0.56) |
| patched, `q8_0` / `q5_1` | 470.3 | 37.90 (0.34) | 413.3 | 35.91 (0.56) |
| patched, `q8_0` / `q5_0` | 470.3 | 37.81 (0.33) | 413.1 | 35.62 (0.59) |
| patched, `q8_0` / `q4_0` | 470.4 | 37.99 (0.32) | 413.2 | 36.10 (0.58) |
| stock, `q8_0` / `q8_0`, repeated last | 470.2 | 38.19 (0.39) | 413.1 | 36.44 (0.57) |

Prefill does not move.  Decode on an empty cache does not move.  At 32768 cells the mixed pairs read 0.9-2.2 %
under `q8_0` / `q8_0`, inside the spread of three repeats.

### The staged path against the native one (tensor split, `GGML_CUDA_FA_KV_MIXED=0`)

| K / V, path | decode d0 | decode d32768 | decode d100000 |
|---|---:|---:|---:|
| `q8_0` / `q8_0`, stock | 38.19 | 36.44 | 32.12 (0.49) |
| `q8_0` / `q5_1`, native | 37.90 | 35.91 | 31.26 (0.43) |
| `q8_0` / `q5_1`, staged | 37.84 | 31.74 | **23.29** (0.28) |
| `q8_0` / `q4_0`, native | 37.99 | 36.10 | |
| `q8_0` / `q4_0`, staged | 37.95 | 32.05 | |

(`-d 100000` is `llama-bench -p 0 -n 64 -d 100000 -r 3`.)  The staged path is level on an empty cache, 11-12 %
behind at 32768 cells and 25 % behind at 100000: its cost is the whole-cache conversion, so it grows with
depth, which is the direction of the 1.7-3.6x the gate's comment records.  The native mixed pair is 2.7 %
under `q8_0` / `q8_0` at 100000 cells.

### Speed, one card, Qwen3.8-27B GSQ-RCO IQ3_S

All arms of one card in one session, stock repeated after the patched build:

| build, K / V | gfx1201: prefill d0 | decode d0 | prefill d32768 | decode d32768 | gfx1100: prefill d0 | decode d0 | prefill d32768 | decode d32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| patched, `q8_0` / `q8_0` | 857.7 | 35.59 | 649.1 | 32.19 | 387.1 | 33.24 | 315.3 | 30.12 |
| stock, `q8_0` / `q8_0` | 854.0 | 35.49 | 649.3 | 32.16 | 384.3 | 33.16 | 314.4 | 30.03 |
| stock, `q5_1` / `q5_1` | 854.0 | 35.33 | 648.1 | 30.69 | 384.0 | 32.96 | 314.6 | 28.66 |
| stock, `q4_0` / `q4_0` | 854.6 | 35.41 | 651.6 | 31.34 | 384.1 | 33.01 | 314.8 | 29.03 |
| patched, `q8_0` / `q5_1` | 855.2 | 35.38 | 649.5 | 31.02 | 383.8 | 32.93 | 314.3 | 29.51 |

Decode repeats are within 0.03-0.11 tok/s here.  The first arm of a session reads up to 0.8 % higher in
prefill whichever build it is (gfx1100: 387.2 for stock when stock ran first, 387.1 for the patched build
when it ran first, 384.3 for the other one both times), so the builds do not differ.

An earlier session with all three mixed pairs, against the patched `q8_0` / `q8_0` of that session
(31.95 on gfx1201, 30.01 on gfx1100) at 32768 cells: V `q4_0` 31.29 / 29.77 (−2.1 % / −0.8 %), V `q5_1`
30.96 / 29.55 (−3.1 % / −1.5 %), V `q5_0` 30.70 / 29.11 (−3.9 % / −3.0 %).

### Decode at depth is slower with the smaller V types — and that is not the mixing

At 32768 cells on one card a `q5_1` V decodes 1.7-3.6 % under `q8_0` / `q8_0`.  The delivery's own same-type
caches are slower still: stock `q5_1` / `q5_1` is 4.6 % under on both cards and stock `q4_0` / `q4_0` 2.6-3.3 %.
So the cost comes with the cache type, a mixed pair pays less of it than the matching small pair, and it keeps
the keys at 8 bits.  Nothing is lost on an empty cache or in prefill, and on the tensor split, where each card
reads two of the four KV heads, it sits inside the noise.

Part of it on gfx1201 is the fetch-ahead band, which only `q8_0` / `q8_0` takes.  One session, gfx1201 alone,
`llama-bench -p 0 -n 64 -d 0,32768 -r 3`:

| arm | decode d0 | decode d32768 |
|---|---:|---:|
| `q8_0` / `q8_0` | 35.32 | 31.90 |
| `q8_0` / `q8_0`, `GGML_HIP_FA_BAND_FETCH_AHEAD=0` | 35.34 | 31.42 (−1.5 %) |
| `q8_0` / `q4_0` | 35.35 | 31.27 (−2.0 %) |
| `q8_0` / `q5_1` | 35.34 | 31.00 (−2.8 %) |

So about 1.5 points of the gap are the missing fetch-ahead and the rest is the dequant arm (the nibble types
go byte by byte through FP32 where `q8_0` multiplies packed halves).  That reading is ours and is not tested
further.  Letting a mixed pair take the fetch-ahead variant for its `q8_0` K is an obvious follow-up; it is not
in this patch.

### A second model: Index-Translate 35B-A3B (qwen35moe) Q5_K_M, tensor split

| build, K / V | prefill d0 | decode d0 | prefill d32768 | decode d32768 |
|---|---:|---:|---:|---:|
| patched, `q8_0` / `q8_0` | 3319 | 112.9 (3.0) | 2406 | 110.9 (4.5) |
| stock, `q8_0` / `q8_0` | 3326 | 113.2 (2.4) | 2407 | 110.7 (4.4) |
| stock, `q8_0` / `q8_0`, again | 3323 | 113.3 (2.5) | 2405 | 110.9 (4.5) |
| patched, `q8_0` / `q5_1` (earlier session) | 3298 | 111.6 (2.5) | 2397 | 106.2 (4.4) |
| patched, `q8_0` / `q4_0` (earlier session) | 3298 | 111.8 (2.5) | 2403 | 106.8 (4.3) |

It loads and runs with a mixed pair; stock and patched agree; the decode spread is too wide to rank the pairs.
(The very first arm on this file, before it was in the page cache, read 2374 / 2094 in prefill and is left out.)

## What it does not cover

* Head sizes other than 256.  The mixed instances exist for `(256, 256)` only; the gate refuses a mixed pair on
  any other model, so nothing takes the staged path by accident.  Adding a size is the generator list, the
  extern declarations and the three `256` checks; the kernel body does not change.
* K types other than `q8_0`, and V types other than `q4_0` / `q5_0` / `q5_1`.
* gfx1151 and CDNA.  Three GPUs.  `-sm layer`.
* A K-quant model on one card (the two 27B files that fit one 16 GB card here are IQ quants).
* Quality at depth (the KLD run is 4096-token chunks) and MTP acceptance per pair.
* The vec kernel.  `GGML_CUDA_FA_QUANTS` lists same-type pairs only, and the HIP chooser does not reach it for
  a quantized cache, so nothing here depends on it.

## Questions

* **Always compiled.**  The three tile instances are globbed like every other `fattn-tile*.cu`, whatever
  `GGML_CUDA_FA_QUANTS` says.  Compiling them on demand would need more than the CMake list: the gate lives in
  `llama_init_from_model`, which cannot see how the backend was built, so a default build would accept the
  pair and stage it.  It needs a backend query for "this pair is native".  I left it always-on; it is three
  translation units.
* **On by default.**  A mixed pair is only reached with two different `--cache-type-*` values, so no extra
  opt-in variable is asked for; `GGML_CUDA_FA_KV_MIXED=0` is the way out.
* **A block, not an artifact?**  Unlike `ce-q8-wire` this needs no special hardware and the default path is
  bit-identical, so it may belong in block 05.  It is cut as an artifact because the gates above are mine.
