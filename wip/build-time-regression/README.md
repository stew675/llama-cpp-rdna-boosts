# Build-time regression: the delivery's own FA instantiations (2026-09-15, r5)

**Status: the `fattn-tile` half is FIXED and VERIFIED in `v16-790cf51aa-r5`; the unroll-warning flood
is FIXED in `v16-ebbb18522-r2` (suppressed, see the unroll section).  The `fattn-mma-f16` compile-time
half is DIAGNOSED, NOT FIXED — it needs a code-path change with its own A/B (see "Remaining").**

## The report

A fresh ROCm build had become slow, and one TU (`fattn-tile.cu`) took **over 4 minutes** to compile.
Separately, the Strix Halo build printed "tens of thousands of can't unroll loop" diagnostics.

## What was measured (gfx1201, 16 cores, `-j16`, `cmake --build --target ggml-hip` from clean)

| | before r5 | after r5 |
|---|---|---|
| `ggml-hip` backend, clean, `-j16` | **538 s** | **330 s** |
| **full fresh build** (`~/bin/build-llama-rocm-714`: `rm -rf build-rocm` + configure + **all targets**, `-j16`) | ~570 s | **362 s** |
| worst single TU | **`fattn-tile.cu` 509 s** (95 % of the wall) | `fattn-mma-f16-instance-ncols1_8-ncols2_4.cu` 250 s |
| `fattn-tile.cu` itself | 509 s | **< 10 s** |
| tile instance TUs (12) | 9-26 s (2 cases each) | 9-136 s (8 cases each) |
| MMA instance TUs (21) | 200-250 s (unchanged) | 200-250 s |

The whole backend build was gated by **one translation unit**, so no amount of `-j` helped.

## Root cause 1 (fixed): the tile type axis was instantiated in the dispatch TU

Block 03 made `type_KV` a template parameter of `ggml_cuda_flash_attn_ext_tile_case` (so the tile
kernel could read BF16 natively); later blocks added the quantized arms.  But
`DECL_FATTN_TILE_CASE` / `EXTERN_DECL_FATTN_TILE_CASES` kept covering only **F16 and BF16** — the
upstream mechanism splits the *head-size* axis across 12 generated
`template-instances/fattn-tile-instance-*.cu` files and `extern`s it in the dispatch, and the type axis
was never added to it.  Because the dispatch has an unconditional `case` per native type, the other six
types were **implicitly instantiated in the dispatch TU**:

* `fattn-tile.cu.o` defined **96** `tile_case` symbols: 12 head-size combos x 8 types, of which 24
  (F16/BF16) were `extern` and **72 were compiled there** — each pulling in both softcap variants and
  the whole `ncols2` chain;
* each generated instance file defined just **2** (F16, BF16).

`nm -C .../fattn-tile.cu.o` before the fix: 72 `T`/`W` + 24 `U`; after: **96 `U`** and the instances
carry 8 each.

**Fix** (block 15 amendment, `ggml/src/ggml-cuda/fattn-tile.cuh` only, +31/-8): `DECL_FATTN_TILE_CASE` /
`EXTERN_DECL_FATTN_TILE_CASES` now expand per type (`DECL_FATTN_TILE_CASE_TYPE(DKQ, DV, T)` for F16,
BF16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl), so the 12 generated files emit 8 cases each and the
dispatch TU holds only externs.  Pure code placement: same template arguments, same flags, same device
code.

## Root cause 2 (diagnosed, NOT fixed): the MMA native arms duplicate the WMMA kernel 8x per TU

The new critical path is ours too: the native-KV arm chain in `ggml_cuda_flash_attn_ext_mma_f16_case`
(`GGML_CUDA_FATTN_MMA_NATIVE_ARM` x 6 + the BF16 branch + the F16 fallback) instantiates the **whole
WMMA kernel once per KV type inside every instance TU**.  Measured on the same TU
(`template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu`, 8 head-size cases in both cases):

| | base `790cf51aa` | delivery |
|---|---|---|
| object size | 0.90 MB | **7.26 MB** (8.1x) |
| compile time | **6.7 s** | **229 s** (34x) |

So of the remaining ~330 s, the MMA group is ~1950 s of CPU across 21 TUs.  Two ways out, both needing
an A/B: (a) finer generated-file granularity (one file per head-size -> ~150 TUs, better packing, same
total work), or (b) make the WMMA loader's KV type a runtime dispatch (one kernel copy, ~8x less code,
at the cost of a uniform branch in the tile load).  See `TODO.md`.

## The unroll warnings (upstream noise, amplified by us) — FIXED in r2

The full build used to emit **10,362** `warning: loop not unrolled ...
[-Wpass-failed=transform-warning]` lines,
and **every one** comes from `fattn-mma-f16.cuh` (attributed to the kernel's declaration line,
`2049:24`); nothing else in the backend emits any.  `-Rpass-missed=loop-unroll` on one instance file
localises them to the bare `#pragma unroll` loops whose bounds are runtime values — dominantly
`fattn-mma-f16.cuh:421` (`for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k)`, 4148 remarks in one
file), with `1979:17`/`1997:21` (the `use_sparse` combine path) next.  Those pragmas are **upstream's**
and exist identically at the base; there are no forced `#pragma unroll N` in the file.  The *count* is
ours: it scales with the instantiation count that root cause 2 multiplied.  They are warnings, not
errors, and the loops still compile — the flood is log noise (10k lines per build).

**Fix (`v16-ebbb18522-r2`, block 15):** `ggml/src/ggml-hip/CMakeLists.txt` appends
`-Wno-pass-failed` to `CMAKE_HIP_FLAGS`.  The hints are advisory and the unroll pass already failed,
so this is a diagnostic-only change (no codegen).  Verified on the worst TU
(`fattn-mma-f16-instance-ncols1_8-ncols2_4.cu`): **1692 -> 0** `-Wpass-failed` warnings, and the clean
build now warns only on upstream's pre-existing `llama-kv-cache.h` unused field.  HIP-only: the
diagnostic is a Clang/AMDGPU one and the CUDA toolchains do not emit it.  The proper upstream fix (drop
the bare `#pragma unroll` from the runtime-bounded loops, or the runtime KV-type dispatch of root
cause 2) is still worth an upstream PR — see `upstream/`.

## Zero-runtime-change verification (the fix is build-time only)

| gate | result |
|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | **5951/5951, 0 failures** (identical to the r4 reference) |
| 27B text gate, 128 greedy tokens, prose prompt | q8_0 `472b282950b5`, q4_0 `118eb7f5fe85`, f16 `70960317a203` — **bit-identical** to the pre-amendment build |
| 27B `tg64@32768` / `pp8192`, 5 KV types | within **0.12 %** of the recorded r4 numbers (see `results/`) |
| control: q4_1 staged 23.18 (ref 23.14), default `-r 3` 25.58±0.13 (ref 25.44) | the sub-0.1 % offsets seen in the first pass were single-sample noise |

## Reproducing

```bash
# per-TU durations (no .ninja_log with Unix Makefiles)
wip/build-time-regression/tools/tu-timer.sh /tmp/tu.txt &
cd ~/llama.cpp && rm -rf build-rocm/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir
cmake -S . -B build-rocm && cmake --build build-rocm --target ggml-hip -j16 | tee /tmp/build.log
# NOTE: deleting CMakeFiles/<t>.dir removes build.make -- re-run cmake -S/-B to restore it
awk '{if ($2>m[$1]) m[$1]=$2; if(!(($1) in f)) f[$1]=$3} END {for(p in m) print m[p], f[p]}' /tmp/tu.txt | sort -rn | head

# the runtime verification (text gate + per-type perf vs the r4 reference)
wip/build-time-regression/tools/verify-r5.sh
```
