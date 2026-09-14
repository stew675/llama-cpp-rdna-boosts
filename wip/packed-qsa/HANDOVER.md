# Packed-QSA — session handover (2026-09-13)

**Read this file top to bottom; it is self-contained.**  It tells you what the work is, where both
repos stand, the exact commands to rebuild and re-validate, what P1–P3 already proved, and where the
work must go next.  The detailed per-phase records are linked at the bottom; you do not need them to
start, only to go deep.

---

## 0. TL;DR of the current state

We are porting pwilkin's packed-block WMMA QSA attention (`qsa3`) into the delivery's QSA path to
close part of the qwen4exp prefill gap.

* **P1, P2, P3 are implemented and committed** on the fork branch `packed-qsa`.
* **The kernel is correct**: it matches the VEC reference to `rel ≈ 3e-5` (f16 rounding only).
* **But the end-to-end result on gfx1201 is negative**: at `pp8192`, pure VEC = 893.75 t/s,
  pack+merge+VEC = 883.89, pack+merge+WMMA = 876.42 — i.e. the packed path is **−2 %**.
  So the op does not pay for its pack/merge overhead on this box.
* **The immediate job (P3.5) is to make the end-to-end go positive.**  Then P4 (dispatch/split) and
  P5 (PPL / purity / MTP).
* Everything is **opt-in and default-off**, so the delivery is unaffected whatever you do.

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
B="-m $M -ngl 99 -sm layer -fa on -ctk f16 -ctv f16 -p 8192 -n 0 -r 3"

# pure VEC (no pack at all)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 \
  ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
# pack+merge, VEC attention (isolates the P1+P2 overhead)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 \
  GGML_CUDA_QSA_PACKED_ATTN=0 ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
# pack+merge+WMMA (the full packed path)
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 \
  ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
```

The three numbers are the P3.5 scoreboard.  Today: `893.75 / 883.89 / 876.42`.

---

## 2. Repository / branch state

| repo | branch | state |
|---|---|---|
| `~/llama-cpp-rdna-boosts` (delivery, **docs only**) | **`packed-qsa`** @ `656cb73` | `wip/packed-qsa/` (this tree) + `wip/tiled-gdn/`, `wip/prefill-arrangements/`; pushed to `origin/packed-qsa` |
| `~/llama-cpp-rdna-boosts` | `main` @ `2ebf725` | untouched; has `wip/tiled-gdn` + `wip/prefill-arrangements` but **not** `wip/packed-qsa` |
| `~/llama.cpp` (fork, **the code**) | **`packed-qsa`** @ `4f464941a` | `4f464941a` P3 kernel → `91f5e41a0` P3 primitive → `1697ad10e` P2 → `2b84c7c62` P1 → `b214621da` tiled-GDN spike → `b36517087` delivery tip |
| `~/llama.cpp` | `rdna-boosts` @ `b36517087` | clean, the 16-block delivery tip |

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
* **The QSA lever does not transfer to gfx1201** (measured: −2 % end-to-end).  This is the central
  new fact; see §4.

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
  green; `rows_sorted=0` shows the indexer rows are already distinct/ascending.

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

## 5. The immediate job: **P3.5** (now required, not optional)

The packed path must beat the VEC path end-to-end at `pp8192`.  Two independent tracks, and the
scoreboard in §1.4 is the gate:

**Track A — remove the pack/merge overhead (worth ~1.1 %).**
* The P1 pack is rebuilt **per graph** (`build_attn_qsa`) as `cont(permute(reshape(...)))` — two full
  copies of the K/V cache per layer per ubatch.  Ideas: skip the pack when the cache/selection did
  not change; keep one packed buffer and update it incrementally; check whether the extra `cont`
  intermediates can be avoided.
* The P2 merge is rebuilt **per op call** (`ggml_cuda_qsa_merge_build`), writing ~50 MB for a
  2048-token ubatch.  Ideas: fold the merge into the pack build; cache the descriptor when the `idx`
  tensor is unchanged; shrink `cap` (it is the loose upper bound `4*ns`).

**Track B — make the kernel genuinely faster (worth ~0.9 % today; the target is ≥2× op).**
* The per-row softmax costs 8 separate shuffle reductions — restructure so the 8-row vector is
  shuffled once per step (e.g. transpose the 8 values across the wave before reducing).
* Replace the LDS P-transpose with a shuffle network (pwilkin's gfx11 approach, adapted).
* Prefetch the next chunk's K/V fragments (pwilkin does).
* Widen the key chunk beyond 16 keys, and/or reconsider the 8-wave head-dim split.
* **Investigate the op-vs-end-to-end contradiction first**: the isolated `x20` benchmark reuses a
  warm packed K/V while the full run reads it cold.  A cold-cache op benchmark (or `rocprofv3`) will
  tell you whether the 1.35× is real.

Keep the correctness gate green after every change:
`GGML_CUDA_QSA_ATTN_CHECK=1` must stay `rel ≈ 1e-5`.

---

## 6. Then: P4 and P5

**P4 — dispatch / split coverage.**
* `ggml_cuda_flash_attn_qsa_supported()` currently accepts the op on shape/type alone; the packed
  path is chosen purely by `src[7]`.  Decide the real predicate (device cc, `n_tokens`, `n_stream`,
  gqa 12, D=256) and the VEC fallback.
* **`-sm tensor`**: the meta splitter asserts the pack is *mirrored* (the pack folds `n_kv` and
  `n_kv_heads` into one block dim, so a kv-head split is not representable).  Either confirm the
  mirrored full-pack-per-device path or gate the packed path off under tensor split.
* **RDNA3.5 (gfx1151)** needs a 16-half fragment instantiation (`wmma_f32_16x16x16_f16_w32`,
  `#elif defined(RDNA3)`) — gfx1201 was done first.  This is the box where the QSA share is largest
  (14 %), so it is the most likely place for the win to materialise.

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
| `GGML_CUDA_QSA_MERGE_CHECK=1` | P2: host cross-check + 202-case merge self-test + the WMMA layout self-test |
| `GGML_CUDA_QSA_ATTN_CHECK=1` | P3: run packed **and** VEC on the same inputs, print `rel`/`max_abs` + op timing |
| `GGML_CUDA_QSA_ATTN_DBG=1` | P3: dump the first group/tile/chunk score tile, P tile and `l` |
| `GGML_CUDA_QSA_IDENTITY` | (pre-existing) force `idx = 0..n_top_k-1`, dense-equivalent validation |

**Model** (3× R9700, ~82 GB, use `-sm layer -mg 0`):
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

**Open:**
* P3.5: how to make the pack/merge free and the kernel faster (§5).
* P4: `-sm tensor` pack layout (mirrored vs gated off); gfx1151 fragment instantiation.
* Whether the campaign is worth continuing at all on gfx1201, or should move to gfx1151 where the op
  is a larger share.  The `pp8192` decomposition (§1.4) is the deciding evidence.

---

## 10. Reading order (for depth)

1. `PORT-PLAN.md` — the original P1–P5 plan.
2. `P1-NOTES.md`, `P2-NOTES.md` — the P1/P2 records and their call-outs.
3. `P3-DESIGN.md` — the fragment address mappings + kernel design.
4. `P3-NOTES.md` — the P3 result, the two bring-up bugs, the end-to-end decomposition, the P3.5 list.
5. `../prefill-arrangements/README.md` — the arrangement landscape + the GDN-is-spent argument.
6. `../tiled-gdn/05-where-the-speed-comes-from.md` — where the journey's ~2.2x really is.
7. `../../archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md` —
   the archived `mmb` port + the kernel attribution (and its §12 purity rationale).

## 11. Environment notes

* Dev box `soar`: 3× R9700 (gfx1201, RDNA4), ROCm 7.14 at `/opt/rocm-7.14-gfx1201`; set
  `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib` and `HIP_VISIBLE_DEVICES=0,1,2`.
* The GDN/other measurements in the sibling docs come from a **gfx1151 (Strix Halo)** box — different
  hardware, different fractions; do not mix numbers across boxes without saying so.
* The build directory `~/llama.cpp/build-rocm` is not durable across machines: **rebuild before
  trusting any number** (§1.1).
