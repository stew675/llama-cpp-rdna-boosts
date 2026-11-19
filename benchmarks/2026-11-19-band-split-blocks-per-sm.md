# The RDNA4 band's fixup period `P` is NOT one constant: it is "enough blocks to fill the machine", and the tile count the device's slice of a split gets decides it (r322, block 05, issue #333)

Everything here was measured on the primary box: 3x R9700 (gfx1201, `nsm` 32), ROCm 10.1.0, one warm source
path.  `nsm` as ggml counts it on this part is **32**, and the band's 32-column kernel is an **8-warp (256
thread) block whose runtime `max_blocks_per_sm` is 3**; the 128-dim row's block is 4 warps (128 threads) with a
higher limit.  `ROCBLAS_USE_HIPBLASLT=0` where a fresh-process comparison needed it (`INVARIANTS.md` §44).

**What changed.**  `ggml_cuda_fattn_band_wmma_split()` returned `max(2, nsm/2)` for every shape (r316).  It now
returns

```
P = max(nsm/2, ceil(nsm * blocks_per_sm / tiles))        blocks_per_sm = max(1, 16 / nwarps)
```

where `tiles = ntiles_z_gqa * K->ne[2] * Q->ne[3]` is the number of **output tiles this device's slice of the
split actually gets** and `nwarps` is the launcher's block width.  The grid is `tiles x P`, so `P` alone does not
decide how many blocks a launch has: with 4 tiles per device `nsm/2` already gives 64 blocks (2 per SM), which is
why r316's constant was right for the shape it was measured on -- and wrong by up to 2x everywhere else.  The
`nsm/2` floor keeps r316's value for every shape with 4 or more tiles per device, i.e. the change only ever
*raises* `P`.

## 1. The mechanism: the band underfills the machine when a split cuts the tile count

`P` is the period of the band's KV-interleave (`block (tile, i)` owns iterations `i, i+P, i+2P, ...`) and the
launch is `gridDim = (tiles, P)`.  Total blocks = `tiles x P`.  Two things follow:

* at **small `tiles`** the launch cannot fill the machine no matter how long the KV is: 2 tiles x `nsm/2` = 32
  blocks on 32 SMs = **one block per SM** out of a 3-block limit, and 1 tile is 16 blocks = **half the SMs idle**;
* at **large `tiles`** `nsm/2` already overshoots and a smaller `P` is enough (the block cap is what matters, not
  the per-block KV length).

A **tensor split cuts `K->ne[2]`** (each device owns a contiguous range of K/V heads), so `tiles` is a *per
device* number that falls as devices are added: a 2-K/V-head model (qwen4exp/Qwen3.8-Flash-Next, GQA 12) is 4
tiles on one GPU, **2 tiles on two devices**, and a 4-K/V-head model (Qwen3.8-27B, GQA 6) is 2/1/1 on three.  That
is the reporter's setup (`sm tensor` on 2x R9700) and it is why r316's constant silently cost them 4-9 %: it was
chosen on the **1-GPU** qwen4exp/27B shape (4 tiles), the only one it fits.

## 2. Per op: the optimum is a tile-count law, not a value

`test-backend-ops perf -o FLASH_ATTN_EXT`, band forced on (`GGML_HIP_FA_BAND_WMMA_H256=1` /
`..._H128=1`), `GGML_HIP_FA_BAND_WMMA_SPLIT` = the column, `us/run`, one R9700, q8_0 K/V.  `tiles` is
`ceil(gqa/8) x K/V heads`; each row is the same K/V iteration count, so the columns differ only in how it is
cut up.  (The harness is a temporary block of perf cases in `tests/test-backend-ops.cpp`; the block was stripped
before the patch was cut, as in the r316 record.)

### 2.1 The 256-dim class, `n_q` = 1 / 4 (decode and the MTP verify width)

| shape (head 256, GQA, K/V heads) | tiles | kv | P8 | P16 (r316) | P24 | P32 | P48 | P64 | P96 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| GQA 6, 1 head | 1 | 16384 | -- | 69.2 | 69.5 | **58.4** | 64.0 | 61.5 | 82.6 | P32 |
| GQA 6, 1 head | 1 | 32768 | -- | 120.8 | -- | **75.8** | 79.3 | 80.1 | 101.9 | P32 |
| GQA 6, 1 head | 1 | 102400 | -- | 278.2 | -- | 194.8 | 153.1 | **138.0** | 147.8 | P64 |
| GQA 12, 1 head | 2 | 16384 | 115.6 | 69.0 | 69.5 | **65.8** | 79.3 | 78.6 | -- | P32 |
| GQA 12, 1 head | 2 | 102400 | 502.4 | 291.7 | 258.2 | **215.0** | 234.1 | 230.2 | -- | P32 |
| GQA 12, 2 heads | 4 | 16384 | 100.6 | **88.2** | 95.6 | 95.2 | 100.9 | -- | -- | P16 |
| GQA 12, 2 heads | 4 | 102400 | 519.9 | **401.4** | 422.9 | 406.4 | 412.8 | -- | -- | P16 |
| GQA 6, 4 heads | 4 | 102400 | 555.4 | **424.4** | 466.6 | 446.2 | 455.5 | -- | -- | P16 |
| GQA 6, 8 heads | 8 | 16384 | **150.0** | 161.1 | -- | 187.3 | -- | -- | -- | P8 |
| GQA 6, 8 heads | 8 | 102400 | **812.9** | 822.3 | -- | 839.8 | -- | -- | -- | P8 |
| GQA 6, 16 heads | 16 | 16384 | 297.2 | 312.5 | -- | 353.0 | -- | -- | -- | P4 (280.3) |
| GQA 6, 16 heads | 16 | 102400 | 1590.3 | 1589.8 | -- | 1617.6 | -- | -- | -- | P4 (1586.2) |

`P4` (the 16-tile row's optimum, not shown in every column) is `tiles x P = 64` too.  Read down the `tiles`
column: **1 -> 64, 2 -> 32, 4 -> 16, 8 -> 8, 16 -> 4**, i.e. `tiles x P = 64 = 2 blocks per SM` in every row, and
the measured best column agrees within 5 % (the 1-tile rows' 5 % residual is the depth trade: P32 at 16-32k, P64
at 102k, and one constant may not chase it -- §4).

### 2.2 The 128-dim class (Ripple-65B, GQA 10, 8 K/V heads), q8_0

| shape | tiles | kv | P4 | P8 | P16 (r316) | P32 | P64 | best |
|---|---|---|---|---|---|---|---|---|
| GQA 10, 1 head | 2 | 16384 | 191.9 | 117.0 | 81.4 | 55.6 | **53.2** | P64 |
| GQA 10, 1 head | 2 | 102400 | 1025.1 | 535.9 | 293.2 | 191.0 | **144.9** | P64 |
| GQA 10, 8 heads | 16 | 16384 | 251.2 | 157.4 | 148.7 | **147.0** | 161.5 | P32 |
| GQA 10, 8 heads | 16 | 102400 | 1349.3 | 890.3 | 825.9 | 785.3 | **773.2** | P64 |

This class's block is **4 warps**, so "2 blocks per SM" would leave it at half the warps: the `16 / nwarps`
term gives it 4 blocks per SM, i.e. `P` 64 at 2 tiles (measured best) and `P` 16 at 16 tiles (measured 148.7 vs
the 147.0 best at 16384; 825.9 vs 785.3 at 102400, and the 16-tile row is where r316's value *already* was, so the
floor keeps it).

## 3. The default now lands on the optimum, for every tile count

Same harness, no `GGML_HIP_FA_BAND_WMMA_SPLIT` at all (the shipped default), kv 102400:

| shape | tiles | default P | measured | that P's own column in §2 |
|---|---|---|---|---|
| GQA 12, 2 heads (256) | 4 | 16 | 403.6 | 401.4 |
| GQA 12, 1 head (256) | 2 | 32 | 211.5 | 215.0 |
| GQA 6, 1 head (256) | 1 | 64 | 137.8 | 138.0 |
| GQA 6, 8 heads (256) | 8 | 8 | 817.1 | 812.9 |
| GQA 6, 16 heads (256) | 16 | 4 | 1585.4 | 1586.2 |
| GQA 10, 1 head (128) | 2 | 64 | 144.1 | 144.9 |
| GQA 10, 8 heads (128) | 16 | 16 | 811.9 | 825.9 |

## 4. End to end

### 4.1 The reporter's shape: qwen4exp on 2 GPUs (the 4-9 % they reported)

Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS + the shared MTP head, **2x R9700, `-sm tensor`**, q8_0 K/V, `-ncmoe`-free
`--load-mode dio --lazy-mode mlockr` (this box's equivalent of their `--host-experts mmap`), `-t 7`.  Their arms
are reproduced with `GGML_HIP_FA_BAND_WMMA_SPLIT=16` (r321's constant) against the new default, which for this
shape is 32.

Kernel level (`rocprofv3 --kernel-trace`, a 39485-token prompt, one `-n 16` greedy decode): the band's own
kernel `flash_attn_ext_f16<256,256,4,8,...>` is **103.2 us -> 82.1 us per launch per device (-20 %)**.

Server decode (`llama-server` + `/completion`, greedy, 128 tokens after a fresh prompt, `draft-mtp n3`; one run
per cell, so the acceptance re-roll is in the number -- see the note below):

| prompt tokens | r321 (P 16) t/s | new default (P 32) t/s | delta |
|---|---|---|---|
| 39485 | 134.3 | 128.8 | -4.1 % |
| 65715 | 107.9 | 116.0 | **+7.5 %** |
| 104931 | 112.8 | 124.4 | **+10.2 %** |
| 196865 | 106.3 | 103.9 | -2.3 % |
| 242721 | 83.7 | 87.1 | **+4.1 %** |

The new arm reproduces a `GGML_HIP_FA_BAND_WMMA_SPLIT=32` arm on the same build cell for cell (116.66 vs 115.95,
124.80 vs 124.37, 103.63 vs 103.85, 87.04 vs 87.11), i.e. the default now *is* the arm the reporter runs.  The
two rows that go the other way are the acceptance re-roll: a greedy speculative arm's token rate is
`mean_len x steps/s`, and the numeric-order change re-rolls the draft acceptance (their own write-up says the
same of their ~65k row).

**Plain decode** (no speculator at all, so no step-mix confound), same config, `-n 96`:

| prompt tokens | r321 (P 16) | new default | delta |
|---|---|---|---|
| 65715 | 70.3 t/s | 72.3 t/s | **+2.8 %** |
| 104931 | 66.9 t/s | 70.5 t/s | **+5.4 %** |

### 4.2 This delivery's own 3-device arm: 27B Q8_0, 4 K/V heads -> 2/1/1 tiles, P 32/64/64

`-sm tensor`, q8_0 K/V, 32k prompt.  Plain decode (`-n 128`, 3 interleaved repeats): 42.4/42.4/42.5/42.4 t/s at
P 16 against **42.6/42.6/42.6/42.6** at the new default (+0.4 %).  A weight-bound 27B step is at the DRAM roof, so
this arm can only move a little.

`draft-mtp n3`, `-n 512`: **31.5 ms/step both** (86.5 against 85.2 t/s; the token rate differs only because the
acceptance re-roll moved the step count 188 -> 191 -- derived from `tokens = accepted + steps`).  Kernel trace of
the same arm: per layer, attention + fixup 164.1 us -> 139.0 us on the 2-tile device and 159.2 -> 106.5 on the
1-tile devices, i.e. the per-layer attention is 15-33 % faster and the step is unchanged at this depth.

An `-n 128` reading of the same pair had suggested -3.4 %/step; that arm only has ~45 steps, so a single step of
arithmetic resolution is 2 %, and at `-n 512` the difference is zero.  Quote the long arm for a step-rate claim.

### 4.3 What `P` costs at the other end

The fixup combine is proportional to `P` (it reads `P` partials per tile): 8.8 us -> 31.3 us at P 16 -> 64 on the
27B trace.  That is why `P` is not simply "as many blocks as possible": the op table in §2 is the trade already
priced in (it times the whole op, combine included), and the 1-tile rows' optimum moves from P 32 (16-32k) to
P 64 (102k) as the fixup's share falls.  A constant cannot follow that; the shipped rule takes the satu**r**ation
point (2 blocks/SM) at every depth.

## 5. Correctness and purity

* `test-backend-ops -o FLASH_ATTN_EXT`: **7381/7381** on every backend (ROCm0/1/2 + CPU), as r321.
* `scripts/gate-spec-purity.sh`: **PASS**.  `plain == draft-mtp` byte-identical on both caches on the 3-device
  `-sm tensor` split where `P` is now 32/64/64 (q8_0 `c1f9d0b3f5b6`, bf16 `2fbc183348a8`), all four Protocol A
  axes pass on both caches, and the **width probe** (flat + 256-cell-boundary prefix) is `PASS`, **worst maxdiff
  0**, at `KV=q8_0` and `KV=bf16`.  A constant-per-shape `P` is width-pure by construction: `tiles` is a function
  of the KV head count, the GQA ratio and `Q->ne[3]`, never of `n_q` (`ntiles_x` is 1 for every width 1..8, in
  both the 32-column and the band_wide 64-column row).
* Deep-context purity where `P` actually differs from r321:
  * 27B Q8_0, 3 GPUs, q8_0 KV, a 39185-token prompt, `-n 32`: plain == `draft-mtp n3`, `58439631ccc9`.
  * qwen4exp GSQ-IQ3_XXS, 2 GPUs, q8_0 KV, a 39485-token prompt, `-n 32`: `--spec-type none` 74.9 t/s against
    `draft-mtp n3` 100.3 t/s (1.34x, so the arm is not vacuous) and the text is byte-identical
    (`35699210a30e`, 182 chars).
* Coherence, f16 K/V, 3 GPUs `-sm tensor`, below the band's work gate: 4B `8d89080db673`, 27B `da2e2d192e21`,
  35B-A3B `265c381dbc11`, and the 4B at `-ctk/-ctv q8_0` `8d89080db673` -- **all identical to r316/r321**.  The
  band does not run below its gate, so the change is inert there.
* `scripts/gate-prefill-logits.sh`: canonical point (27B, f16, `--device 0`) **mean KLD 0.000000 / same-top-p
  100.000 %** -- the band is decode/verify only.
* `scripts/gate-per-block-build.sh --fast --changed-from 4262662b8`: **2/2** (block 05 and the tip).
* `scripts/validate-set.sh` GREEN, canonical tree `de717e7a5`.  Kernel journal and every run log clean
  (`check-gpu-faults.sh`).

## 6. Not validated

`gfx1100` / `gfx1151` and Windows were not exercised; `P` is expressed in `nsm` and in the launcher's own
`nwarps`, so a smaller RDNA4 part and the 128-dim row scale with the same rule, and the kernels are the ones both
classes already ran.  The band_wide (query widths 5..8, 16-warp) row is **not** measured: `16 / 16 == 1` block per
SM is what its `nsm/2` floor gives, so it keeps r316's value for every shape with 2 or more tiles, which is
deliberate -- only the two measured rows (8 warps -> 2 blocks/SM, 4 warps -> 4) may move a shape.
