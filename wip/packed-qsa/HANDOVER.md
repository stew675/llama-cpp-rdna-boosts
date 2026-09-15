# Packed-QSA — session handover (2026-09-13)

> **CONTINUATION (2026-09-15): the gfx1201 side is CLOSED.**  The port now wins **+2.8/3.5/3.9 %**
at ub2048/4096/6144 (under the 5 % bar; fast RDNA4 VEC baseline + a ~5 % fixed merge share), and
**gfx1201/gfx1100 are closed as not applicable**.  The gfx1151 work continues — **read
`HALO-HANDOVER.md`, which is the self-contained entry point for the Strix Halo (`halo`) agent.**
> 
> Note for the maintainer's plan to find another ~5 % in the *VEC* implementation: any VEC speedup
> raises the bar for the packed path (its relative win shrinks) and is the more promising direction
> for gfx1201, so closing the packed side here does not close the QSA story on gfx1201.

**Read this file top to bottom; it is self-contained.**  It tells you what the work is, where both
repos stand, the exact commands to rebuild and re-validate, what P1–P3 already proved, and where the
work must go next.  The detailed per-phase records are linked at the bottom; you do not need them to
start, only to go deep.

---

## 0. TL;DR of the current state

We are porting pwilkin's packed-block WMMA QSA attention (`qsa3`) into the delivery's QSA path to
close part of the qwen4exp prefill gap.

* **P1, P2, P3 are implemented and committed** on the fork branch `packed-qsa`.
* **P3.5 verdict (2026-09-15):** the approach is **validated on gfx1151** (`halo`, pwilkin's own
gfx11 kernel): packed QSA is **+9.1 % at ub2048**, **+15.2 % at ub8192**, **+15.9 % at ub16384**
(pp8192, IQ4_XS).  On gfx1201 the **gfx12 softmax transpose** (2026-09-15) moved the port from
−4.5 % to **+2.8 / +3.5 / +3.9 %** at ub2048/4096/6144 — a real win, but **below the 5 % bar**, so
the maintainer's call is to **close gfx1201/gfx1100 as not applicable** (fast RDNA4 VEC baseline +
a ~5 % fixed merge share on a 3.6 s pass vs ~1.3 % on a 14 s gfx1151 pass).  The gfx1151 side is
the one to finish (`halo`).  Details: `P3.5-NOTES.md` §§9-9.4.
* **P4 (tensor-split support) is implemented and validated** (packed shadows split with the kv-head
  axis, `rel ≈ 3.9e-5`).

Everything is **opt-in and default-off**, so the delivery is unaffected whatever you do.

---

## 1. First actions for a new session (do these before anything else)

### 1.1 Rebuild everything in `~/llama.cpp/build-rocm/`

The build directory may predate the current source.  **Assume a full rebuild is required**:

```bash
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
cmake --build build-rocm -j 16          # rebuilds all 98 executables + libs
```

If that cache is missing or stale (e.g. a different machine), reconfigure with the same settings:

```bash
cd ~/llama.cpp
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
```

The build config is (from `build-rocm/CMakeCache.txt`): `AMDGPU_TARGETS=gfx1201`,
`CMAKE_BUILD_TYPE=Release`, `GGML_HIP=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=OFF`,
`GGML_HIP_RCCL=1`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`.
`EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is required with CMake ≥ 4.3 (the build script hardcodes a
bare `-DCMAKE_HIP_FLAGS="-mllvm"`).

The fork's `packed-qsa` branch contains `b214621da` (a tiled-GDN spike, env-gated, default off) as
P1's parent.  Ignore it for this work; it does not affect the packed path.

### 1.2 Smoke test (fast, no model)

```bash
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
./build-rocm/bin/test-backend-ops test -o FLASH_ATTN_QSA -b ROCm0    # expect 22/22
```

### 1.3 Headline A/B check on a real model (the gate for any kernel change)

```bash
M=/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
python3 -c "print('The history of computing spans centuries of innovation and discovery. '*150)" > /tmp/qsa-long.txt

HIP_VISIBLE_DEVICES=0,1,2 \
  LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 GGML_CUDA_QSA_ATTN_CHECK=1 \
  timeout 1500 ./build-rocm/bin/llama-cli -m "$M" -ngl 99 -sm layer -mg 0 -fa on \
  -ctk f16 -ctv f16 -c 4096 -b 2048 -ub 2048 -p "$(cat /tmp/qsa-long.txt)" -n 0 \
  --seed 42 --temp 0 --no-display-prompt --single-turn 2>&1 | grep -E "PACKED-QSA"
```

Expected today (gfx1201): a `PACKED-QSA attn check` line with `rel` in the `1e-5` range and a
`PACKED-QSA op timing` line showing packed ≈ 1.35x faster than vec.  If `rel` is `~1.0`, the kernel
is not actually running or is wrong — see §6 "two bring-up traps".

### 1.4 The end-to-end gate (this is what must improve)

```bash
M=/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
B="-m $M -ngl 99 -sm tensor -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -p 8192 -n 0 -r 3"

# pure VEC (no pack at all)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 \
  ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
# P1 pack only (no merge)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 \
  GGML_CUDA_QSA_PACKED_MERGE=0 ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
# pack+merge, VEC attention (isolates the P1+P2 overhead)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 \
  GGML_CUDA_QSA_PACKED_ATTN=0 ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
# pack+merge+WMMA (the full packed path)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 \
  ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
```

These four numbers are the P3.5 scoreboard.  Measured 2026-09-15:
`2289.18 / 2286.27 / 2177.92 / 2185.97` — the merge is the cost, the pack is free, the WMMA kernel
is at parity.  The old `-sm layer` numbers (`893.75 / 883.89 / 876.42`) are superseded; see
`P3.5-NOTES.md`.

---

## 2. Repository / branch state

*Synced 2026-09-15 against delivery r5 (`main` @ `a0d1de7`, release `v16-790cf51aa-r5`, tree
 `d735d6c11258ae939cfd392511e3f29ac22a7686`).*

| repo | branch | state |
|---|---|---|
| `~/llama-cpp-rdna-boosts` (delivery, **docs only**) | **`packed-qsa`** | rebased onto `main` @ `a0d1de7`; `wip/packed-qsa/` (this tree, + `P3.5-NOTES.md`), `wip/tiled-gdn/`, `wip/prefill-arrangements/` |
| `~/llama-cpp-rdna-boosts` | `main` @ `a0d1de7` | the r5 delivery docs |
| `~/llama.cpp` (fork, **the code**) | **`packed-qsa`** | rebased onto the r5 delivery tip `65001ac96` (tree `d735d6c1`): P3 → P2 → P1 → tiled-GDN spike, plus the 2026-09-15 P3.5 levers (tensor-split assertion relaxation, `GGML_CUDA_QSA_PACKED_MERGE`, `GGML_CUDA_QSA_MERGE_NOROWS`) |
| `~/llama.cpp` | `rdna-boosts` @ `65001ac96` | clean, the r5 16-block delivery tip |

**Rules (do not deviate):**
* All code changes go on `packed-qsa` in **`~/llama.cpp`** (the fork).
* All docs go on `packed-qsa` in **`~/llama-cpp-rdna-boosts`** (the delivery repo).
* **Never** put work on `main`/`master`.  **Never push the fork** (`~/llama.cpp`) anywhere — its
  `packed-qsa` is disposable.
* The delivery repo may be pushed to **its own** `origin` (`github.com:stew675/llama-cpp-rdna-boosts`),
  `packed-qsa` only.
* `git am`/apply of the delivery patch set is irrelevant here — this is scratch work on top of the
  delivery tip, not a delivery change.

Reference (read-only): `~/pwilkin-llama-cpp`, branch `strix-halo`, file
`ggml/src/ggml-cuda/qsa.cu` (443 lines, `qsa3_rows/merge/attn`).  Regenerate a local copy with:
`cd ~/pwilkin-llama-cpp && git show strix-halo:ggml/src/ggml-cuda/qsa.cu > /tmp/pwilkin-qsa.cu`.

---

## 3. The goal and the crystallised conclusion

**Goal:** port pwilkin's packed-block WMMA QSA attention (his `qsa3`) so qwen4exp prefill gets
faster, without changing the K/V format (it needs only an F16 cache, not a uniform weight set, so it
applies to our mixed-expert checkpoints with no PPL-per-bit trade).

The qwen4exp prefill gap on the Strix Halo box (pwilkin ~1.8x) decomposes into **three independent
arrangement classes**:

| class | artifact | needs uniform IQ4_NL? | gfx1151 gap share |
|---|---|---|---|
| weights → bf16 WMMA | `mmb` | **yes** (IQ4_NL-only predicates) | −33 % family |
| **selected KV → packed f16 blocks + WMMA** | **`qsa3`** | **no** (F16 KV only) | ~2.1 s / ~+117 t/s |
| streams/layout → bf16 marking, HC, conv1d | `mark_bf16_only`, `hc-cn`, `gdn-conv` | no | ~+74 t/s |

Facts not to re-derive:
* The tiled GDN is exact (bit-neutral) and is **not** the lever; the delivery's chunked bf16 GDN
  already beats it on gfx1201.
* The uniform-weight requirement belongs to `mmb`, **not** to QSA.  All qwen4exp models on this box
  are mixed-expert, so `mmb` would not fire without a per-type dequant kernel.
* **The QSA lever does not transfer to gfx1201** — and the original "−2 %" was itself measured
  in the wrong split mode.  In the canonical `-sm tensor -b/-ub 2048` config the packed path is
  **−4.5 %**, the merge is the whole cost, and the kernel is at parity.  See §4/§5 and
  `P3.5-NOTES.md`.

---

## 4. Progress: P1–P3 (what is already done and proven)

### P1 — graph pack + op sources (fork `2b84c7c62`) — record `P1-NOTES.md`
* `ggml_flash_attn_qsa_set_packed(a, packed_keys, packed_values)` sets optional **`src[7]`/`src[8]`**
  on `GGML_OP_FLASH_ATTN_QSA`: `packed_keys [16,4,16,n_blocks]` f16, `packed_values [4,256,n_blocks]`
  f16, `n_blocks = k->ne[1]/4 * k->ne[2]`.
* `src/models/qwen4exp.cpp`: `qsa_pack_keys`/`qsa_pack_values` (plain `reshape`/`permute`/`cont`,
  no new backend op) built behind **`LLAMA_QSA_PACKED=1`** (default off) and the `layout_prefill`
  gate: F16 KV, D=256, gqa 12, `n_stream==1`, `n_tokens >= 128`.
* `ggml-backend-meta.cpp`: `handle_flash_attn_qsa` tolerates `src[7]/src[8]` (mirrored-or-unknown).
* Proven: 22/22 `FLASH_ATTN_QSA`; gate-off == gate-on same-seed text `b72fb4d76af5`.

### P2 — merge descriptor (fork `1697ad10e`) — record `P2-NOTES.md`
* New `ggml/src/ggml-cuda/qsa-packed.{cu,cuh}`: `qsa3_rows_kernel` + `qsa3_merge_kernel` (ported)
  and `ggml_cuda_qsa_merge_build` (allocates `ublk`/`umask`/`ucount`/`srow`/`sflag`, launches).
  **`ggml_cuda_qsa3_attn` lives here too.**
* Descriptor contract (consumed by the kernel): `ublk[g][0..ucount)` = ascending block ids padded to
  a multiple of 4 with `0xFFFF`; `umask[g]` = bit `4*qi + (key&3)`; `cap = (4*ns+3)&~3`.
* **Found and fixed a latent defect in the reference merge kernel**: a duplicate index in a row made
  it emit the same block twice (the packed attention would double-count).  Both passes now coalesce
  all keys of a block into one entry.
* Validation: `GGML_CUDA_QSA_MERGE_CHECK=1` → host cross-check + a 202-case synthetic self-test, both
  green; `rows_sorted=0` at small ub, but **`rows_sorted=1` at `-ub 2048`** — the rows are
  *not* guaranteed ascending, so the rows kernel is required (see `P3.5-NOTES.md` §3).

### P3 — the WMMA kernel (fork `91f5e41a0` primitive, `4f464941a` kernel) — records `P3-DESIGN.md`, `P3-NOTES.md`
* gfx12 f16 WMMA primitive + layout self-test (the "two runs of four" 8-half A/B fragment,
  `D[m][n] = Σ_k A[m][k]·B[n][k]`).
* `qsa3_attn_kernel`: `grid=(ngroups, n_kv_heads)`, 256 threads = 8 waves; wave `w` owns head-dim
  tiles `2w,2w+1` (KQ) and V-dim tiles `2w,2w+1` (output); waves 0..2 own the softmax for row-tiles
  0..2 and publish `alpha`/`l`/`P` via LDS; 16 keys (4 union blocks) per iteration; P staged through
  an LDS transpose.
* Dispatch: when `dst->src[7] != nullptr`, build the descriptor and run the packed path;
  **`GGML_CUDA_QSA_PACKED_ATTN=0` forces the VEC kernel with the pack still built** (the A/B lever).
* **Correct**: `GGML_CUDA_QSA_ATTN_CHECK=1` A/B vs the VEC kernel over identical inputs gives
  `rel ≈ 3e-5`, `max_abs ≈ 6e-4` (f16 rounding only); `mean_packed == mean_vec`.
* **Perf**: op-level 1.34–1.36× (16.5 vs 22.0 ms @ n_q=2756, ns=2051); **end-to-end −2 %** (§1.4).

### The two facts that must not be lost
1. `#if defined(RDNA4)` is **false in the host compilation pass** (it comes from `vendors/hip.h` and
   is set only for the device pass).  Guard only the **intrinsic**; leave kernels and host launchers
   unguarded, or the launch is silently compiled out.  (This bit us once: the first "A/B" compared
   uninitialised memory and the first "WMMA self-test OK" was vacuous.)
2. The gfx12 **D layout holds 8 rows per lane** (`m = 8*(lane>>4)+e`, `n = lane%16`), so the softmax
   statistics are **per-row** and reduce over the 16 key-lanes `r` (8 separate 4-step shuffles).  A
   whole-array reduction sums 8 rows and silently shrinks the output.

---

## 5. P3.5 — measured (2026-09-15) — CLOSED on gfx1201

**Read `P3.5-NOTES.md` for the full record.**  Three results:

1. **gfx1201 (`soar`, `-sm tensor`) — the gfx12 softmax transpose** (fork `6f7a38b3f`) took the op
   from **1.30x to 4.4x** and the end-to-end from **−4.5 % to +2.8 / +3.5 / +3.9 %** at
   ub2048/4096/6144.  The P2 merge still costs ~5 %; with it removed the ceiling is +9.4 %.  Below
   the 5 % bar → **gfx1201/gfx1100 closed as not applicable.**
2. **gfx1151 (`halo`, pwilkin's gfx11 kernel):** **+9.1 %** at ub2048 and **+15.2/15.9 %** at
   ub8192/16384 — the approach works where the VEC baseline is weak; finish it there.
3. Large ubatch does *not* help on gfx1201 (the op speedup peaks at n_q≈2 K and declines; ub8192 is
   0.97x), and tensor split cannot fit beyond ub6144 for this model.

The gfx12 softmax transpose is architecture-general and is the main keep-worthy result for the
gfx1151 port.  Track A/Track B below are the historical plan.

---

## 6. Then: P4 and P5

**P4 — dispatch / split coverage.  PARTLY DONE (2026-09-15).**
* `-sm tensor` is now **supported and validated**: the packed shadows split with the K/V kv-head
  axis (keys axis 3, values axis 2), the P1 "must be mirrored" assertion is relaxed accordingly and
  the A/B is still `rel ≈ 3.9e-5` (fork `packed-qsa`).
* Still open: the real dispatch predicate (device cc, `n_tokens`, `n_stream`, gqa 12, D=256) and the
  VEC fallback; `ggml_cuda_flash_attn_qsa_supported()` currently accepts on shape/type alone.
* **RDNA3.5 (gfx1151)** still needs a 16-half fragment instantiation (`wmma_f32_16x16x16_f16_w32`,
  `#elif defined(RDNA3)`) — gfx1201 was done first.  This is the box where the QSA share is largest
  (14 %), so it is the only place left where a win could materialise.

**P5 — validation before any promotion.**
* PPL vs the VEC build (the packed path is a prefill re-baseline).
* `W = 1..8` logits matrix — must be **unchanged**, because the packed path is gated `n_tokens >= 128`
  and the decode/verify band stays on the VEC kernel (`GREEDY-PURITY.md`).
* MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
* End-to-end A/B at pp8192/pp16384 on gfx1201 and (when available) gfx1151.

Reminder from the repo rules: nothing here is a delivery change until it goes through the promotion
path (`beta/` staging, env-gated A/B, maintainer go-ahead).  This is `wip/` work.

---

## 7. Cookbook (env vars, models, commands)

**Environment variable reference** (all default-off/off-path unless stated):

| env | effect |
|---|---|
| `LLAMA_QSA_PACKED=1` | P1: build the packed K/V shadows and attach `src[7]/src[8]` (the master switch) |
| `LLAMA_QSA_DENSE_SHORTCUT=0` | force the sparse selection even below `indexer_top_k + r - 1` (use in every packed test) |
| `GGML_CUDA_QSA_PACKED_ATTN` | `0` = keep packed sources but run the **VEC** kernel (isolates the pack/merge overhead) |
| `GGML_CUDA_QSA_PACKED_MERGE` | `0` = skip the P2 merge build entirely (isolates the P1 pack cost) |
| `GGML_CUDA_QSA_MERGE_NOROWS` | `1` = skip merge kernel A (diagnostic only — **breaks correctness**) |
| `GGML_CUDA_QSA_MERGE_CHECK=1` | P2: host cross-check + 202-case merge self-test + the WMMA layout self-test |
| `GGML_CUDA_QSA_ATTN_CHECK=1` | P3: run packed **and** VEC on the same inputs, print `rel`/`max_abs` + op timing |
| `GGML_CUDA_QSA_ATTN_DBG=1` | P3: dump the first group/tile/chunk score tile, P tile and `l` |
| `GGML_CUDA_QSA_IDENTITY` | (pre-existing) force `idx = 0..n_top_k-1`, dense-equivalent validation |

**Model** (3× R9700, ~82 GB; **use `-sm tensor -b/-ub 2048`** — the canonical serving config;
`/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf`
(others: `IQ4_XS`, `Q4_K_M`, `Q4_K_XL` in the same tree).  Geometry: 48 layers, D=256,
24 q-heads / 2 kv-heads (gqa 12), `indexer.top_k=2048`, `compress_ratios` ratio 4 on every 4th
layer, so the QSA op fires on 12 layers.

**Regenerate the long prompts** (they live in `/tmp`, which is not durable):
```bash
python3 -c "print('The history of computing spans centuries of innovation and discovery. '*150)" > /tmp/qsa-long.txt   # ~2.4k tokens
python3 -c "print('The history of computing spans centuries of innovation and discovery. '*250)" > /tmp/qsa-long2.txt  # ~2.7k tokens
```

**Text hashes** (same-seed coherence) use the delivery helper:
`python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py <log> [--text]`.

**Always** pass `--single-turn` (and `--no-display-prompt` for scripted runs) to `llama-cli`, and wrap
anything blocking in `timeout` — otherwise it enters the interactive loop and hangs.

**Do not run two GPU benchmarks in the same tool block** — they contend and the numbers are junk.

---

## 8. Key files

| path | what |
|---|---|
| `~/llama.cpp/ggml/src/ggml-cuda/qsa-packed.cu` / `.cuh` | the whole ported P2+P3 body: WMMA primitive, merge kernels, WMMA attention kernel, all the check/self-test/dump helpers |
| `~/llama.cpp/ggml/src/ggml-cuda/fattn-qsa.cu` | the VEC kernel (the reference); the dispatch that chooses packed vs VEC + the A/B harness |
| `~/llama.cpp/ggml/src/ggml.c` | `ggml_flash_attn_qsa*` builders, including `set_packed` |
| `~/llama.cpp/src/models/qwen4exp.cpp` | `qsa_pack_keys`/`qsa_pack_values`, the `LLAMA_QSA_PACKED` gate, the op call site |
| `~/llama.cpp/ggml/src/ggml-backend-meta.cpp` | `handle_flash_attn_qsa` (the `-sm tensor` split rule) |
| `~/pwilkin-llama-cpp` @ `strix-halo`, `ggml/src/ggml-cuda/qsa.cu` | the reference kernels (`qsa3_rows/merge/attn`) |
| this tree's `P3-DESIGN.md` | the fragment address mappings and the kernel design (derived, ready to reuse) |

---

## 9. Decisions closed / still open

**Closed:**
* Extend `GGML_OP_FLASH_ATTN_QSA` with `src[7]/src[8]` (rather than a new op).
* Purity gate: prefill-only (`n_tokens >= 128`), the VEC kernel keeps the `W = 1..8` band.
* The pack is a graph-side op composition (no new ggml op, no on-disk format).
* Visibility for P3: the kernel consumes the **same** source the VEC op gets — base `mask` when
  present, else the derived `cell_vis`/`q_vis` — and the A/B proves they agree.
* P4 tensor-split support: the packed shadows follow the K/V kv-head split; validated `rel ≈ 3.9e-5`.
* **P3.5: gfx1201 cannot win.**  The pack is free, the merge A kernel is the whole cost, and the P3
  kernel is at parity under `-sm tensor`.  The ceiling with a free merge is +0.8 %.  (2026-09-15)

**Open:**
* Whether the campaign moves to **gfx1151** (QSA ~14 % of the pass, the slow VEC this design targets)
  — and if so, how to make the merge A kernel cheap there (`P3.5-NOTES.md` §3).  Nothing on gfx1201.
* P4: the real dispatch predicate / VEC fallback; gfx1151 16-half fragment instantiation.
* P5 (only if the campaign continues): PPL, `W = 1..8` purity, MTP, pp8192/16384 A/B.

---

## 10. Reading order (for depth)

1. `PORT-PLAN.md` — the original P1–P5 plan.
2. `P1-NOTES.md`, `P2-NOTES.md` — the P1/P2 records and their call-outs.
3. `P3-DESIGN.md` — the fragment address mappings + kernel design.
4. `P3-NOTES.md` — the P3 result, the two bring-up bugs, the end-to-end decomposition, the P3.5 list.
5. **`P3.5-NOTES.md`** — the canonical-config scoreboard, the `qsa3_rows_kernel` root cause, and the
   park-on-gfx1201 decision (2026-09-15).
6. `../prefill-arrangements/README.md` — the arrangement landscape + the GDN-is-spent argument.
7. `../tiled-gdn/05-where-the-speed-comes-from.md` — where the journey's ~2.2x really is.
8. `../../archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md` —
   the archived `mmb` port + the kernel attribution (and its §12 purity rationale).

## 11. Environment notes

* Dev box `soar`: 3× R9700 (gfx1201, RDNA4), ROCm 7.14 at `/opt/rocm-7.14-gfx1201`; set
  `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib` and `HIP_VISIBLE_DEVICES=0,1,2`.
* The GDN/other measurements in the sibling docs come from a **gfx1151 (Strix Halo)** box — different
  hardware, different fractions; do not mix numbers across boxes without saying so.
* The build directory `~/llama.cpp/build-rocm` is not durable across machines: **rebuild before
  trusting any number** (§1.1).
