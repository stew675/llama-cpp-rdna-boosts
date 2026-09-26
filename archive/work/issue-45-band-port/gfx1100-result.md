# gfx1100 (RDNA3_0) — issue #45 GQA-6 FA band evaluation

**Result: NEGATIVE — do not port. Leave `ggml_cuda_fattn_band_wmma_applies` RDNA4-only.**

Session date 2026-09-25, host `fingon`, AMD Radeon RX 7900 XTX (gfx1100, Navi 31, 96 CU, 24 GiB).
This is a `wip/` session record, **not** part of the delivery. No committed gate was changed; no
delivery patch was produced.

## 0. Environment and build

- `rocminfo | grep gfx`: `gfx1100`, `amdgcn-amd-amdhsa--gfx1100`, `amdgcn-amd-amdhsa--gfx11-generic`.
- ROCm: `/opt/rocm-7.14-gfx1100` (AMD clang 23.0.0git), `GPU_TARGETS=gfx1100`, ccache on.
- Fork checkout: the existing `~/llama.cpp` `rdna-boosts` branch was **stale** (tree `08fe2b77…`,
  pre-r4, no band code). The canonical r4 tree was rebuilt from the delivery:
  ```
  git worktree add --detach ~/llama-r4-gfx1100 84e76d8a2
  cd ~/llama-r4-gfx1100
  RDNA_BRANCH=rdna-boosts-r4-gfx1100 \
    bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .
  # -> strict 16/16 git am, applied tree 5938da09d294a01e0862c2d561b0c7ca154de90a == release.json
  ```
- Build: `BUILD_DIR=build-rocm-beta` (clean build **5m28s**, ccache). Note the host's
  `~/bin/build-llama-rocm-714` on `fingon` is an older copy that **ignores** `BUILD_DIR`/`GPU_TARGETS`
  and hardcodes `build-rocm`/`gfx1100`; I used an equivalent local wrapper that honours them. The
  binary reports `build f51a60069 (11189)`.
- Experiment binary: the one-line relaxation of the gate (never committed), exactly as the handoff
  asks:
  ```c
  // ggml/src/ggml-cuda/fattn-common.cuh
  - if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) {
  + if (!amd_wmma_available(cc)) {
  ```
  Runtime band control: `GGML_HIP_FA_BAND_WMMA=0` (tile, the baseline), `=4` (ncols1 4, default),
  `=2` (ncols1 2); `GGML_HIP_FA_BAND_WMMA_SPLIT` overrides `P` (default `nsm` = 96).
- Models: `/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` (qwen35 27B, head 256, 24 Q /
  4 KV = GQA 6, 16.4 GB). `Qwen3.5-4B` is GQA 4 and cannot reach the band, so it was not used.
- All runs pinned `HIP_VISIBLE_DEVICES=0`, no parallel work on the GPU.

## 1. Op-level: `test-backend-ops perf`, kv 16384

Command (per type/nb, filtered):
```
test-backend-ops perf -b ROCm0 -o FLASH_ATTN_EXT \
  -p 'nh=4,nr23=.*kv=16384.*type_K=<T>,type_V=<T>'
```
µs/run; lower is faster. `band nc4/nc2` = relaxed gate, `nc4/tile` = ratio (>1 = band slower).

| type | nb | tile | band nc4 | band nc2 | nc4/tile | nc2/tile |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 1 | 69.6 | 71.4 | 71.4 | 1.03 | 1.03 |
| f16 | 3 | 185.9 | 186.4 | 185.6 | 1.00 | 1.00 |
| f16 | 5 | 299.1 | 299.5 | 298.9 | 1.00 | 1.00 |
| f16 | 8 | 453.4 | 458.3 | 455.3 | 1.01 | 1.00 |
| bf16 | 1 | 75.2 | 76.3 | 74.8 | 1.01 | 0.99 |
| bf16 | 3 | 200.4 | 203.2 | 201.1 | 1.01 | 1.00 |
| bf16 | 5 | 312.1 | 313.9 | 313.2 | 1.01 | 1.00 |
| bf16 | 8 | 474.8 | 479.0 | 477.6 | 1.01 | 1.01 |
| q8_0 | 1 | **96.5** | 313.4 | 272.4 | **3.25** | 2.82 |
| q8_0 | 3 | **226.5** | 315.1 | 316.8 | **1.39** | 1.40 |
| q8_0 | 5 | 354.6 | 382.5 | 357.3 | 1.08 | 1.01 |
| q8_0 | 8 | 543.7 | **387.4** | 440.6 | 0.71 | 0.81 |
| q4_0 | 1 | **112.8** | 321.9 | 295.3 | **2.85** | 2.62 |
| q4_0 | 3 | **258.5** | 323.0 | 340.2 | **1.25** | 1.32 |
| q4_0 | 5 | 402.1 | 435.9 | 388.4 | 1.08 | 0.97 |
| q4_0 | 8 | 625.5 | **438.9** | 453.0 | 0.70 | 0.72 |
| q4_1 | 1 | **120.0** | 325.2 | 309.6 | **2.71** | 2.58 |
| q4_1 | 3 | **262.6** | 325.9 | 355.1 | **1.24** | 1.35 |
| q4_1 | 5 | 406.2 | 441.1 | 395.6 | 1.09 | 0.97 |
| q4_1 | 8 | 629.2 | **441.0** | 469.2 | 0.70 | 0.75 |
| q5_0 | 1 | **147.9** | 367.6 | 351.9 | **2.49** | 2.38 |
| q5_0 | 3 | **337.3** | 371.2 | 438.5 | **1.10** | 1.30 |
| q5_0 | 5 | 530.7 | 506.7 | **472.2** | 0.95 | 0.89 |
| q5_0 | 8 | 817.5 | **505.8** | 552.9 | 0.62 | 0.68 |
| q5_1 | 1 | **124.7** | 369.7 | 326.4 | **2.96** | 2.62 |
| q5_1 | 3 | **288.8** | 360.6 | 399.5 | **1.25** | 1.38 |
| q5_1 | 5 | 452.9 | 493.4 | **443.0** | 1.09 | 0.98 |
| q5_1 | 8 | 700.6 | **491.3** | 511.1 | 0.70 | 0.73 |
| iq4_nl | 1 | **157.6** | 389.9 | 382.6 | **2.47** | 2.43 |
| iq4_nl | 3 | **336.2** | 393.7 | 451.2 | **1.17** | 1.34 |
| iq4_nl | 5 | 515.6 | **505.1** | 494.9 | 0.98 | 0.96 |
| iq4_nl | 8 | 791.2 | **509.4** | 592.7 | 0.64 | 0.75 |

Reading:

- **The band does not apply to f16/bf16** (f16 has no native read; bf16's native arm is opt-in), so
  those rows are a control and are unchanged — confirming the relaxation only moved the six native
  quantized types.
- **The band loses badly at the widths that matter most.** `n_q` 1 is 2.5–3.3× slower; `n_q` 3 is
  1.1–1.4× slower; `n_q` 5 is roughly a tie; only `n_q` 8 wins, and only by ~1.4×.
- `ncols1 = 2` helps `n_q` 1 a little (272 vs 313 for q8_0) but is still 2.8× the tile kernel, and it
  is worse than `ncols1 = 4` at `n_q` 8. No single `ncols1` rescues the band.
- The `P` sweep (q8_0, `ncols1 = 4`) found the shallow best at `P` 32–64 (`n_q` 1 ≈ 302–308 µs) but
  the default `P = nsm = 96` is fine at `n_q` 8 (~387 µs); **no `P` value brings `n_q` 1 anywhere near
  the tile kernel's 96.5 µs.**

### Contrast with the r4 gfx1201 numbers (from `WORKLOG.md` r4, same shape/kv, q8_0)

| n_q | gfx1201 tile | gfx1201 band | gfx1201 band/tile | gfx1100 tile | gfx1100 band nc4 | gfx1100 band/tile |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 141 | 148 | 1.05 | 96.5 | 313 | **3.25** |
| 3 | 326 | 150 | 0.46 | 226.5 | 315 | **1.39** |
| 5 | 524 | 225 | 0.43 | 354.6 | 382 | 1.08 |
| 8 | 814 | 230 | 0.28 | 543.7 | 387 | 0.71 |

The difference is architectural and consistent: gfx1100's **tile** kernel is faster than gfx1201's at
every width (96 CU vs 64 CU, and the native q8_0 tile path is strong), while gfx1100's **RDNA3 WMMA**
band is ~1.6–2.1× slower than gfx1201's RDNA4 WMMA band. On gfx1201 the band is a 2.2–3.5× win at
verify widths and a ~5 % `n_q` 1 loss; on gfx1100 the win shrinks to ~1.4× and the `n_q` 1 loss becomes
~3.3× (op-level).

## 2. Op-level correctness

- Relaxed gate, **`test-backend-ops -o FLASH_ATTN_EXT` 6340/6340 on ROCm0** (includes the 389 new
  qwen35 GQA-6 cases, all eight KV types, both layouts). `-p 'nr23=.6,1.'` alone is **389/389**.
- `MUL_MAT_ID` **929/929** with the band on and off (unchanged, as expected — the change is FA-only).
- So the band is *functionally correct* on gfx1100; it is just slow.

## 3. End-to-end

### 3.1 Plain decode / verify widths (`llama-batched-bench`, shallow context)

`-c 16384 -b 2048 -ub 512 -ctk/ctv q8_0 -npp 512 -ntg 64 -npl 1,4,8`:

| B | band off (tile) | band nc4 | band nc2 |
|---|---:|---:|---:|
| 1 | 39.23 | 35.57 | 36.10 |
| 4 | 102.71 | 102.45 | 102.31 |
| 8 | 130.42 | 130.11 | 130.16 |

### 3.2 Plain decode vs depth (`llama-bench`, B=1)

| config | band off (tile) | band nc4 | band nc2 |
|---|---:|---:|---:|
| `tg128 @ d16384`, q8_0 KV | **37.79** | 33.13 (−12.3 %) | 34.37 (−9.1 %) |
| `tg128 @ d40960`, q4_0 KV | **33.96** | 29.98 (−11.7 %) | 30.70 (−9.6 %) |

The plain-decode regression is real, large and depth-stable — the most common decode path.

### 3.3 MTP (`llama-cli`, greedy, seed 42, `--reasoning off`, prose prompt)

Shallow ≈ 5.2 k ctx, q8_0 KV, `-n 256`:

| config | band off | band nc4 | band nc2 |
|---|---:|---:|---:|
| `--spec-type none` | 39.0 | 34.5 | 35.5 |
| `draft-mtp n3` | 68.5 | 65.1 (−5 %) | 63.5 |
| `draft-mtp n7` | 54.9 | 56.8 (+3.5 %) | 54.2 |

Deep ≈ 42 k ctx (the prose prompt concatenated 8×, 41 992 tok), q4_0 KV, `-n 256`:

| config | band off | band nc4 |
|---|---:|---:|
| `--spec-type none` | 33.6 | 29.6 (−12 %) |
| `draft-mtp n3` | 56.7 | **61.6 (+8.6 %)** |
| `draft-mtp n7` | 49.5 | **55.0 (+11.1 %)** |
| prefill (pp) | 840–848 | 838–847 |

So the reporter's regime does reproduce *for MTP at depth* (+9–11 % at ~42 k), but only there; plain
decode loses ~12 % everywhere, and at shallow context even `n3` loses. This is a fundamentally
different trade from gfx1201 (where the band is a large verify win and only a ~5 % decode loss).

### 3.4 Prefill flat (contract)

`llama-bench -p 512,2048,4096`, q8_0 KV, 27B:

| pp | band off | band nc4 |
|---|---:|---:|
| 512 | 1136.2 | 1127.5 |
| 2048 | 1064.1 | 1055.8 |
| 4096 | 1047.3 | 1043.5 |

Flat within noise, as expected: the band is gated to `n_q <= 8`.

## 4. Purity — the band is **not width-pure on gfx1100**

The `plain == draft-mtp` contract is measured at `-n 256`, greedy, prose prompt, q8_0 KV (shallow,
the non-repetitive workload) and q4_0 KV (deep, the repeated prose workload).

**Band OFF (tile) — pure:**

| run | t/s | hash |
|---|---:|---|
| shallow plain | 39.0 | `aec8db1f0c44` |
| shallow n3 | 68.5 | `aec8db1f0c44` |
| shallow n7 | 54.9 | `aec8db1f0c44` |
| deep plain / n3 / n7 | 33.6 / 56.7 / 49.5 | `17e944808d91` (all three) |

**Band ON (nc4) — shallow q8_0 is NOT pure (deterministic, reproduced):**

| run | t/s | hash |
|---|---:|---|
| shallow plain | 34.5 | `aec8db1f0c44` |
| shallow n3 | 65.1 | `1bc204b7d24d` |
| shallow n7 | 56.8 | `59b85cb75ab2` |
| `--spec-draft-n-max 1` (W=2) | 52.4 | `e9411edb4dad` |
| `--spec-draft-n-max 2` (W=3) | 62.7 | `59b85cb75ab2` |
| `--spec-draft-n-max 4` (W=5) | 62.2 | `1bc204b7d24d` |
| `--spec-draft-n-max 5` (W=6) | 60.9 | `1e60c385126c` |
| `--spec-draft-n-max 6` (W=7) | 56.5 | `d8a7b9a96155` |

`ncols1 = 2` behaves the same (`plain aec8db1f0c44`, `n3 4c697c13ca22`, `n7 aec8db1f0c44`). The first
divergence from plain is early (char ~157 of the generation), and acceptance is unaffected
(n3 band off 0.72199 vs band on 0.71193) — i.e. the verify logits are close enough to keep accepting,
but the band's decode (`n_q = 1`) and verify (`n_q ≥ 2`) do **not** reduce identically on gfx1100, so
a greedy near-tie flips. A separate `-n 1000` plain run confirms the band also changes `n_q = 1`
numerics vs tile (`0ea2be9a5613` tile vs `5b410dd40af4` band).

The deep q4_0 run happened to stay pure (`2db17391f04b` for all three) because the repeated-prose
workload is highly predictable and produced no near-tie — but that is luck, not a guarantee. The
shallow deterministic divergence is sufficient to say the r4 "decode and every verify width reduce
identically" invariant does **not** hold on gfx1100 as implemented. (This also means a hypothetical
partial gate — band only for `n_q ≥ 5`, tile for `n_q = 1` — would break purity by construction, and
is not a viable fallback.)

## 5. Decision

**Negative. Keep `ggml_cuda_fattn_band_wmma_applies` RDNA4-only.** Reasons, in order:

1. **Plain decode regresses ~10–12 %** at every measured depth (shallow, d16384, d40960; op-level
   `n_q = 1` is 2.5–3.3× slower). Plain is the CLI default, so a default-on band would ship a decode
   regression.
2. **The verify-width win is much smaller than on gfx1201** (1.4× at `n_q` 8 vs 3.5×; already a tie at
   `n_q` 5), so it only repays the `n_q = 1` loss in the MTP-at-depth corner (+9–11 % `n3`/`n7` at
   ~42 k, but `n3` still loses at ~5 k).
3. **The `plain == draft-mtp` purity invariant fails deterministically** at shallow context, which
   contradicts the band's core design guarantee and would have to be root-caused before any port.

The band is functionally correct on gfx1100 (`FLASH_ATTN_EXT` 6340/6340 vs CPU, `MUL_MAT_ID` 929/929)
and prefill is untouched, but correctness-without-purity plus a decode regression is not a win.

### If it is ever revisited

- The blocker is the RDNA3 WMMA `n_q = 1` cost, not the split: no `ncols1` (2/4) or `P`
  (8…96) value closes the 3× gap.
- The purity divergence at `n_q ≥ 2` needs a root cause (the band's fast path in
  `fattn-mma-f16.cuh`, `gridDim.y > 1`) before any width could be enabled on RDNA3.
- Any future arch enablement should be judged with the same three instruments used here: the
  `nh=4,nr23=.6,1.,kv=16384` op-level perf sweep, `llama-bench tg` at depth for plain decode, and the
  `plain == draft-mtp` hash gate at both shallow and deep context.

## 6. Artifacts

- Worktree: `~/llama-r4-gfx1100` (canonical r4 tree, branch `rdna-boosts-r4-gfx1100`, plus the
  **uncommitted** one-line gate relaxation in `ggml/src/ggml-cuda/fattn-common.cuh`).
- Raw outputs: `/tmp/band-baseline.raw`, `/tmp/band-nc2.raw`, `/tmp/band-nc4.raw`,
  `/tmp/band-eval-full.txt`, `/tmp/mtp-*.log`, `/tmp/deep-*.log`, `/tmp/long-prompt.txt`.
- The gfx1151 handoff (`gfx1151-prompt.md`) is a separate evaluation and was **not** performed here.
