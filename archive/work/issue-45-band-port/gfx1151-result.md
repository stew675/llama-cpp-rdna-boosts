# gfx1151 (RDNA3_5, Strix Halo) — issue #45 GQA-6 FA band evaluation

**Result: NEGATIVE — do not port. Leave `ggml_cuda_fattn_band_wmma_applies` RDNA4-only.**

Session date 2026-09-25, host `halo`, AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (gfx1151, RDNA3_5,
40 CU, 124 GiB unified LPDDR5X). This is a `wip/` session record, **not** part of the delivery. No
committed gate was changed; no delivery patch was produced. The verdict matches the sibling
`gfx1100-result.md`: the band is *functionally* correct on RDNA3 but **not width-pure**, and it
trades a plain-decode regression for a verify-width win.

## 0. Environment and build

- `rocminfo | grep gfx`: `gfx1151`, `amdgcn-amd-amdhsa--gfx1151`, `amdgcn-amd-amdhsa--gfx11-generic`.
  `nsm = 40` (the band's `P = nsm` default is therefore 40, not the reporter's gfx1201 64).
- ROCm: `/opt/rocm-7.14-gfx1151` (AMD clang 23.0.0git), `GPU_TARGETS=gfx1151`, ccache on.
- The host `~/bin/build-llama-rocm-714` on `halo` hardcodes `build-rocm` / `gfx1151` and ignores
  `BUILD_DIR`, so an equivalent wrapper (`/tmp/build-r4.sh`) that honours `BUILD_DIR` was used.
- The existing `~/llama.cpp` `rdna-boosts` branch was **stale** (on `mmb-beta`, pre-r4). Two fresh
  worktrees were created from the canonical r4 tree `5938da09d294a01e0862c2d561b0c7ca154de90a`,
  both applied strictly (16/16 `git am`):
  - `~/llama-r4-gfx1151-base` — committed gate (RDNA4-only), build `build-rocm-base2`
    (clean build **6 m 35 s**);
  - `~/llama-r4-gfx1151` — band worktree with the **uncommitted** one-line relaxation, build
    `build-rocm-band` (clean build **5 m 28 s**).
- Experiment gate change (never committed), exactly as the handoff asks:
  ```c
  // ggml/src/ggml-cuda/fattn-common.cuh, ggml_cuda_fattn_band_wmma_applies()
  - if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) {
  + if (!amd_wmma_available(cc)) {
  ```
  Runtime band control: `GGML_HIP_FA_BAND_WMMA=0` (tile), `=4` (ncols1 4, default), `=2`
  (ncols1 2); `GGML_HIP_FA_BAND_WMMA_SPLIT` overrides `P` (default `nsm` = 40).
- Model: `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (qwen35 27B, head 256, 24 Q / 4 KV =
  GQA 6; the only 27B head-256/GQA-6 model on this host — **no Q4_K_M and no separate MTP file**, but
  the Qwen3.8-27B GGUF carries its own NextN/MTP head, so `--spec-type draft-mtp` works in-tree).
- Op-level A/B harness: the r4 perf loop in `tests/test-backend-ops.cpp` only had `nb = {1,3,5,8}`;
  it was locally widened to `nb = 1..8` in **both** worktrees (a test-harness-only change) so the
  whole decode/verify band is covered in one run. No library or delivery source was altered.
- All runs pinned `HIP_VISIBLE_DEVICES=0`, no parallel work on the GPU.

## 1. Op-level: `test-backend-ops perf`, kv 16384

```
test-backend-ops perf -b ROCm0 -o FLASH_ATTN_EXT -p 'nh=4,nr23=.6,1.,kv=16384'
```

µs/run; lower is faster. `nc4`/`nc2` = relaxed gate with `ncols1 = 4`/`2` (band on); `tile` = band
off. `f16`/`bf16` are a **control**: the gate does not apply (f16 has no native read; bf16 native is
opt-in), so they are unchanged and prove the relaxation only moved the six native quantized types.

| type | nb | tile | nc4 | nc4/tile | nc2 | nc2/tile |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 1 | 311.6 | 311.0 | 1.00 | 310.6 | 1.00 |
| f16 | 4 | 606.0 | 611.1 | 1.01 | 608.3 | 1.00 |
| f16 | 8 | 1011.9 | 1015.1 | 1.00 | 1019.5 | 1.01 |
| bf16 | 1 | 310.7 | 310.5 | 1.00 | 310.7 | 1.00 |
| bf16 | 4 | 598.1 | 597.7 | 1.00 | 602.7 | 1.01 |
| bf16 | 8 | 1066.3 | 1065.3 | 1.00 | 1076.5 | 1.01 |
| q8_0 | 1 | 186.0 | 411.4 | **2.21** | 349.7 | **1.88** |
| q8_0 | 2 | 381.6 | 412.3 | 1.08 | 350.7 | 0.92 |
| q8_0 | 3 | 497.9 | 408.9 | 0.82 | 486.5 | 0.98 |
| q8_0 | 4 | 664.4 | 410.6 | **0.62** | 488.6 | 0.74 |
| q8_0 | 5 | 774.2 | 584.2 | 0.75 | 611.0 | 0.79 |
| q8_0 | 6 | 963.3 | 584.1 | 0.61 | 614.8 | 0.64 |
| q8_0 | 7 | 1096.6 | 587.6 | 0.54 | 769.6 | 0.70 |
| q8_0 | 8 | 1248.2 | 591.2 | **0.47** | 771.7 | 0.62 |
| q4_0 | 1 | 202.8 | 328.2 | **1.62** | 286.6 | **1.41** |
| q4_0 | 2 | 399.3 | 328.9 | 0.82 | 287.8 | 0.72 |
| q4_0 | 3 | 566.6 | 329.2 | 0.58 | 371.8 | 0.66 |
| q4_0 | 4 | 717.9 | 332.3 | **0.46** | 371.4 | 0.52 |
| q4_0 | 5 | 916.1 | 530.8 | 0.58 | 511.3 | 0.56 |
| q4_0 | 6 | 1059.2 | 532.6 | 0.50 | 514.1 | 0.49 |
| q4_0 | 7 | 1265.2 | 537.1 | 0.42 | 667.7 | 0.53 |
| q4_0 | 8 | 1414.3 | 539.4 | **0.38** | 667.4 | 0.47 |
| q4_1 | 1 | 222.0 | 324.3 | 1.46 | 307.0 | 1.38 |
| q4_1 | 4 | 750.3 | 327.9 | 0.44 | 405.3 | 0.54 |
| q4_1 | 8 | 1422.7 | 530.4 | 0.37 | 748.5 | 0.53 |
| q5_0 | 1 | 275.9 | 415.9 | 1.51 | 395.2 | 1.43 |
| q5_0 | 4 | 1009.6 | 421.8 | 0.42 | 639.1 | 0.63 |
| q5_0 | 8 | 1877.0 | 681.4 | 0.36 | 1026.8 | 0.55 |
| q5_1 | 1 | 229.5 | 433.8 | **1.89** | 374.6 | 1.63 |
| q5_1 | 4 | 861.6 | 439.9 | 0.51 | 579.8 | 0.67 |
| q5_1 | 8 | 1618.6 | 722.9 | 0.45 | 940.9 | 0.58 |
| iq4_nl | 1 | 292.8 | 406.0 | 1.39 | 393.2 | 1.34 |
| iq4_nl | 4 | 956.2 | 409.1 | 0.43 | 603.2 | 0.63 |
| iq4_nl | 8 | 1730.5 | 669.8 | 0.39 | 963.9 | 0.56 |

Reading:

- **The band wins big at the verify widths.** On the six native quantized types at `nb = 4` (the
  default `draft-mtp n3` verify width) the band is **1.6–2.4× faster**; at `nb = 8` (the widest pure
  width, `n_max 7`) it is **2.1–2.8× faster**. This is a *much larger* verify win than gfx1100 saw.
- **The band loses badly at `nb = 1`** (plain decode): **1.4–2.2× slower**. No `ncols1` (2 vs 4) and
  no `P` (see §2) fixes this — see below.
- `nc4` is the better verify configuration at every `nb >= 3`; `nc2` is slightly better at `nb 1–2`
  but still 1.4–1.9× slower than tile. A single `ncols1` must serve the whole band, and `nc4` is the
  right one for the verify widths.
- `nb 5` costs a step because `ntiles_x = ceil(nb/ncols1)` moves from 1 to 2 output-tile columns;
  the band's per-launch work roughly doubles there while the tile kernel scales smoothly.

### 1.1 `ncols1` and `P` tuning

`ncols1`: `4` beats `2` for every `nb >= 3` (e.g. q4_0 `nb 8`: 539 vs 667 µs) and `2` only wins
marginally at `nb 1` (q4_0 287 vs 328 µs, still **1.41×** the tile kernel). Neither closes the
decode gap.

`P` sweep (`GGML_HIP_FA_BAND_WMMA_SPLIT`, `ncols1 = 4`), µs/run:

| type | nb | tile | P=4 | P=8 | P=16 | P=40 (=nsm) |
|---|---:|---:|---:|---:|---:|---:|
| q8_0 | 1 | 186.0 | 539.8 | 480.1 | 430.2 | **394.2** |
| q8_0 | 4 | 664.4 | 558.0 | 481.6 | 434.6 | **397.4** |
| q8_0 | 8 | 1248.2 | 931.2 | 641.8 | 619.9 | **666.3** |
| q4_0 | 1 | 202.8 | 514.1 | 459.6 | 336.5 | **310.0** |
| q4_0 | 4 | 717.9 | 516.8 | 459.9 | 339.3 | **315.5** |
| q4_0 | 8 | 1414.3 | 892.4 | 610.3 | 549.6 | 627.6 |

`P = nsm = 40` is the best (or within noise of the best) for the whole band; smaller `P` is worse at
both ends (too few blocks at `nb <= 4`, not enough work to amortise the fixup). **No `P` value brings
`nb = 1` anywhere near the tile kernel** — the best q8_0 point is 394 µs vs tile 186 µs, the best
q4_0 point 310 µs vs tile 203 µs. The `nb = 1` cost is intrinsic to the RDNA3 WMMA band path, not the
split.

## 2. Op-level correctness

- Relaxed gate, **`test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT` 6340/6340**.
- The r4 GQA-6 subset `-p 'nr23=.6,1.'` alone is **389/389** (all 8 KV types, both layouts, both
  `permute` variants).
- `MUL_MAT_ID` **929/929** with the band on (unchanged, as expected — the change is FA-only).

So the band is **functionally correct** on gfx1151; it is a numerical (width-purity) failure, not an
op-level correctness failure.

## 3. End-to-end

### 3.1 Plain decode vs depth (`llama-bench`, B=1, q8_0 KV)

| config | band off (tile) | band on (nc4) | delta |
|---|---:|---:|---:|
| `tg128 @ d16384` | 7.65 | 7.43 | **−2.9 %** |
| `tg128 @ d40960` | 7.38 | 6.99 | **−5.3 %** |

The regression is real and depth-stable, but smaller than gfx1100's ~10–12 % (the 27B is strongly
memory-bandwidth bound here, so the FA kernel is a smaller share of the token). It is still a
regression on the CLI-default path.

### 3.2 MTP throughput and purity (`llama-cli`, greedy, seed 42, `--reasoning off`)

Deep ≈ 42 k context (the 5 246-token `prose-rdna-boosts.txt` concatenated 8×), `-n 256`:

**q8_0 KV — band OFF (tile): pure**

| run | t/s | hash |
|---|---:|---|
| `--spec-type none` | 7.4 | `fc8702ea9800` |
| `draft-mtp n3` | 17.1 | `fc8702ea9800` |
| `draft-mtp n7` | 18.1 | `fc8702ea9800` |

**q8_0 KV — band ON (nc4): pure, and +10…+18 %**

| run | t/s | hash |
|---|---:|---|
| `--spec-type none` | 6.9 | `59284d1d7473` |
| `draft-mtp n3` | 18.8 | `59284d1d7473` |
| `draft-mtp n7` | 21.4 | `59284d1d7473` |

**q4_0 KV — band OFF (tile): pure**

| run | t/s | hash |
|---|---:|---|
| `--spec-type none` | 7.3 | `1fcf172a6d73` |
| `draft-mtp n3` | 18.0 | `1fcf172a6d73` |
| `draft-mtp n7` | 18.7 | `1fcf172a6d73` |

**q4_0 KV — band ON (nc4): NOT pure**

| run | t/s | hash |
|---|---:|---|
| `--spec-type none` | 7.0 | `55024c332213` |
| `draft-mtp n3` | 18.9 | `8e54f97184a6` |
| `draft-mtp n7` | 20.9 | `78c235eb8f3f` |

So at deep context the band is pure for q8_0 (by luck — a predictable continuation with no flipped
near-tie) but **impure for q4_0**: the mandatory contract *q8_0/q4_0 text identical across
`--spec-draft-n-max 3/5/7`* fails. The `n3`/`n7` gain on q4_0 (+5 % / +12 %) therefore buys a
different token stream, which is exactly the trade the repo's purity policy forbids.

### 3.3 Shallow context ≈ 5.2 k (`prose-rdna-boosts.txt` once), `-n 256`

**q8_0 — band OFF (pure) / ON (NOT pure)**

| run | off t/s | off hash | on t/s | on hash |
|---|---:|---|---:|---|
| `--spec-type none` | 7.8 | `6ba54479a92a` | 7.6 | `86dbc247629a` |
| `draft-mtp n3` | 19.0 | `6ba54479a92a` | 19.0 | `b455710193b3` |
| `draft-mtp n7` | 20.4 | `6ba54479a92a` | 20.5 | `6ba54479a92a` |

**q4_0 — band OFF (pure) / ON (NOT pure)**

| run | off t/s | off hash | on t/s | on hash |
|---|---:|---|---:|---|
| `--spec-type none` | 7.7 | `a087865bf095` | 7.6 | `f807e7c5815f` |
| `draft-mtp n3` | 21.4 | `a087865bf095` | 20.5 | `e880b1f55bf0` |
| `draft-mtp n7` | 26.3 | `a087865bf095` | 20.8 | `dcc078a8ff51` |

Both q8_0 and q4_0 are reproducibly impure at shallow context (every band-on run re-run gave the same
hash: `86dbc247629a` / `b455710193b3` are deterministic). At shallow q4_0 the band even *loses* at
`n7` (26.3 → 20.8 t/s, though the divergent stream confounds that number).

### 3.4 Shallow band-on purity matrix, all 8 native KV types

`--spec-type none` vs `draft-mtp n3`, `-n 256`:

| KV type | plain hash | n3 hash | pure? |
|---|---|---|---|
| f16 ¹ | `b455710193b3` | `b455710193b3` | yes (band off) |
| bf16 ¹ | `890f76a2c4d4` | `890f76a2c4d4` | yes (band off) |
| q8_0 | `86dbc247629a` | `b455710193b3` | **NO** |
| q4_0 | `f807e7c5815f` | `e880b1f55bf0` | **NO** |
| q4_1 | `4331d66c2254` | `4331d66c2254` | yes (luck) |
| q5_0 | `ea53ea24fab4` | `db317b810b7e` | **NO** |
| q5_1 | `890f76a2c4d4` | `890f76a2c4d4` | yes (luck) |
| iq4_nl | `8146bd9fc687` | `8146bd9fc687` | yes (luck) |

¹ f16/bf16 do not take the band at all, so they are a control and are pure by construction. The
quantized types that happen to be pure in this 256-token sample are pure only because no greedy
near-tie flipped; the op-level timing (§1) proves the band *is* selected for q4_1/q5_1/iq4_nl, so a
longer/different continuation can flip them too.

### 3.5 Prefill flat (contract)

`llama-bench -p 512,2048,4096`, q8_0 KV, 27B:

| pp | band off | band on | on/off |
|---|---:|---:|---:|
| 512 | 472.44 | 469.30 | 0.993 |
| 2048 | 463.09 | 460.46 | 0.994 |
| 4096 | 456.97 | 455.38 | 0.996 |

Flat within noise, as expected: the band is gated to `n_q <= 8`.

## 4. Decision

**Negative. Keep `ggml_cuda_fattn_band_wmma_applies` RDNA4-only.** Reasons, in order:

1. **The `plain == draft-mtp` (and `n3 == n7`) purity invariant fails on gfx1151**, exactly as on
   gfx1100. It fails at shallow context for q8_0, q4_0 and q5_0, and at deep (~42 k) context for
   q4_0. That violates the band's core design guarantee (decode and every verify width reduce
   identically) and the handoff's mandatory Step-4 contract (`q8_0/q4_0` identical across
   `--spec-draft-n-max 3/5/7`). A verify-width win that changes the token stream is not a win under
   the repo's purity policy.
2. **Plain decode regresses** at every measured depth (−2.9 % at d16384, −5.3 % at d40960), and the
   op-level `nb = 1` cost is 1.4–2.2×. Plain is the CLI default, so a default-on band would ship a
   decode regression.
3. The `nb = 1` cost is **intrinsic to the RDNA3 WMMA band path**: no `ncols1` (2/4) and no `P`
   (4…40) closes the gap, so the band only repays the decode loss in the MTP-at-depth corner
   (+10…+18 % `n3`/`n7` at ~42 k) — a fundamentally different trade from gfx1201, where the band is
   a 2.2–3.5× verify win with only a ~5 % `n_q = 1` loss and full width purity.

This reproduces the gfx1100 conclusion on a second RDNA3 part (RDNA3_0 and RDNA3_5), which points at
the **RDNA3 WMMA implementation of the band fast path** (`fattn-mma-f16.cuh`, the
`#if defined(AMD_WMMA_AVAILABLE)` `gridDim.y > 1` branch and its `kb0_step` round-robin) as the shared
cause, not at anything gfx1151-specific.

### If it is ever revisited

- The blocker is **numerical**, in the RDNA3 WMMA band fast path's width handling — the same place
  gfx1100 localised. The decode/verify groups for q8_0/q4_0/q5_0 differ; the pure types are pure by
  luck. A root cause is needed before *any* width, partial or whole-band, is enabled on RDNA3.
- Do not attempt a partial gate (`n_q >= 2` band, `n_q = 1` tile): it breaks purity by construction
  and was already ruled out on gfx1100.
- The `nb = 1` cost is not the split (`P = nsm = 40` is optimal) and not `ncols1` (2 and 4 both
  lose); it is the RDNA3 WMMA per-launch/per-tile overhead at one active query row.
- Judge any future RDNA3 enablement with the three instruments used here: the
  `nh=4,nr23=.6,1.,kv=16384` op-level sweep at `nb = 1..8`, `llama-bench tg` at depth for plain
  decode, and the `plain == n3 == n7` greedy-hash gate at both shallow and deep context for
  q8_0 **and** q4_0.

## 5. Artifacts

- Worktrees: `~/llama-r4-gfx1151` (canonical r4 + the **uncommitted** one-line gate relaxation in
  `ggml/src/ggml-cuda/fattn-common.cuh`), `~/llama-r4-gfx1151-base` (pristine RDNA4 gate). Builds
  `build-rocm-band` / `build-rocm-base2`. The local test-harness widening of the perf `nb` loop is
  uncommitted in both.
- Wrapper: `/tmp/build-r4.sh`; logs `/tmp/build-{base,base2,band}.log`.
- Raw outputs: `/tmp/op-base8b.txt`, `/tmp/op-band8-nc4.txt`, `/tmp/op-band8-nc2.txt`,
  `/tmp/op-band8-P{4,8,16,40}.txt`, `/tmp/eval-gqa6-band.txt`, `/tmp/eval-fa-full-band.txt`,
  `/tmp/eval-mmid-band.txt`, `/tmp/cli-*.log`, `/tmp/long-prompt.txt`.
- The gfx1100 handoff result is `archive/work/issue-45-band-port/gfx1100-result.md`; this is the gfx1151
  counterpart. **Neither arch was ported.**
