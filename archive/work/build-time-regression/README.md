# Build-time regression: the delivery's own FA instantiations (2026-09-15, r5)

**Status (2026-09-18, `v16-ebbb18522-r6`): FIXED.**  The `fattn-tile` half landed in
`v16-790cf51aa-r5`; the unroll-warning flood in `v16-ebbb18522-r2`; and the `fattn-mma-f16` half in
r6 — the MMA instances are generated per `(ncols1, ncols2, head size)` and the head-512 ones are
listed **first** in the backend source order (clean `ggml-hip -j16` **323.4 -> 236.0 s**), the tile
instances are split per `(head size, KV type)`, and the fused-gate MMQ instances moved out of
`mmq.cu`.  The **reorder, not the split, is what delivers the win** (the six `dkq512` MMA TUs move
from a t=80-150 s start to t=0).  The per-TU native-arm duplication is **deliberate** (the loaders are
force-inlined because that is what makes the native staging fast) and **option (b) was tried and
rejected** on 2026-09-18 — see "Root cause 2" and the ccache section below.  The 2026-09-15 sections
below are the diagnosis that led to r6.

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

## Root cause 2 (understood; the duplication is deliberate): the MMA native arms

The critical path was ours: the native-KV arm chain in `ggml_cuda_flash_attn_ext_mma_f16_case`
(`GGML_CUDA_FATTN_MMA_NATIVE_ARM` x 6 + the BF16 branch + the F16 fallback) instantiates the **whole
WMMA kernel once per KV type inside every instance TU**.  Measured on the same TU
(`template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu`, 8 head-size cases in both cases):

| | base `790cf51aa` | delivery |
|---|---|---|
| object size | 0.90 MB | **7.26 MB** (8.1x) |
| compile time | **6.7 s** | **229 s** (34x) |

**r6 (2026-09-18)** fixed the *scheduling* half: the instances are generated per `(ncols1, ncols2,
head size)` and the head-512 ones are listed first (clean `ggml-hip -j16` **323.4 -> 236.0 s**).  The
per-TU duplication remains and is **deliberate**: the native loaders are `__forceinline__`, and the
optimiser's cross-inlining of them into the kernel is what makes native staging fast.  **Option (b)
was tried and rejected (2026-09-18):**

- **runtime KV-type dispatch** (one loader, runtime switch): object 2.80 -> 2.46 MB but compile
  **236 -> 304 s** — *worse*; one giant 36-copy CFG optimises more slowly than six specialised
  functions.
- **`__noinline__` on the native loader**: clean build **236 -> 136 s** (-43 %), but a universal
  **-1.5..-2.5 % prefill** (f16 KV too — the outlined call sites degrade the kernel's register
  allocation) and -0.3..-0.6 % decode.
- **hybrid** (inline q8_0/q4_0, outline the rest): **no perf recovery** (q8_0 measured == full
  outline) at 168 s — pointless.

The answer is **ccache**, below: keep the optimised codegen, make rebuilds free.

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
archive/work/build-time-regression/tools/tu-timer.sh /tmp/tu.txt &
cd ~/llama.cpp && rm -rf build-rocm/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir
cmake -S . -B build-rocm && cmake --build build-rocm --target ggml-hip -j16 | tee /tmp/build.log
# NOTE: deleting CMakeFiles/<t>.dir removes build.make -- re-run cmake -S/-B to restore it
awk '{if ($2>m[$1]) m[$1]=$2; if(!(($1) in f)) f[$1]=$3} END {for(p in m) print m[p], f[p]}' /tmp/tu.txt | sort -rn | head

# the runtime verification (text gate + per-type perf vs the r4 reference)
archive/work/build-time-regression/tools/verify-r5.sh
```

## The answer: ccache (2026-09-18)

The build is compiler/CPU-bound, not I/O-bound: a full out-of-source build in `/tmp` (tmpfs) took
237.9 s vs 236.0 s on the USB SSD, so moving the build dir does not help.  The optimisation time is
the price of the force-inlined FA codegen, and that is the right default.  The right way to make the
develop/edit loop cheap is a compiler cache.

**`ccache` works with ROCm clang HIP device compilation** (4.12.3 tested).  Wire it in with the CMake
launcher form; `~/bin/build-llama-rocm-714` does this automatically when `ccache` is on `PATH`
(`CCACHE=0` opts out):

```bash
-DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
-DCMAKE_C_COMPILER_LAUNCHER=ccache \
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Measured on gfx1201 (16 cores), the script's `rm -rf "$BUILD_DIR"` + full build:

| | wall |
|---|---|
| first build (populates the cache) | 282.3 s |
| wiped rebuild of the *same* sources | **4.2 s** (657/657 compile steps hit) |

The cache key is the preprocessed source + the exact command line, so it survives the `rm -rf`, and
ccache replays the compiler's own objects — **the cached build is the same code**: `test-backend-ops
-o FLASH_ATTN_EXT` 4/4 and `llama-bench` equal to the uncached build within noise (pp2048 d0 7548 vs
7489, tg128 d16384 89.51 vs 89.47).  Costs: the cache grows ~0.1 GB per full build (limit raised to
20 GB), and any `fattn-*.cuh` edit invalidates the whole FA group (the header is in every instance
TU).
