# HANDOVER — bf16 prefill parity (step 2 CLOSED as a negative result)

**Status: step 1 (route 2, native bf16 MMA) and step 2 (cheapen the bf16→f16 conversion) are both
CLOSED as negative results.**  The V5 bf16-native arm cannot reach prefill parity on gfx1201: the
in-loader conversion is already the RN minimum, and no cheaper sequence exists.  Nothing pushed
except this repo's own `origin/main`.
Repo root: `~/llama-cpp-rdna-boosts` — read `AGENTS.md` first; its rules override anything here.
Project record: `archive/work/bf16-native-prefill/README.md` — **the load-bearing part is the new section
"Step 2 findings (2026-09-18, fifth pass) — CLOSED as a negative result"**; §2a/§2b/§2c below are
superseded by it where they disagree.

---

## 0. Outcome (step 2)

Three conversion sequences were implemented in `flash_attn_ext_f16_load_tile_bf16` and measured in
the real kernel (35B-A3B pp8192, 2×R9700, `-sm layer`):

| loader | ops / 2 elems | native FA | staged FA |
|---|---:|---:|---:|
| raw bit copy (floor, wrong values) | 0 | **198 ms** | 206 ms |
| RTZ packed `v_cvt_pkrtz_f16_f32` | 3 | 270.9 ms | 221.3 ms |
| PACK `2× v_cvt_f16_f32` + `v_pack_b32_f16` (**bit-exact RN**, exhaustive 2^16×2) | 4 | 276.1 ms | 220.7 ms |
| RN baseline (compiler `.l`/`.h`) | 4 | 276.9 ms | 214.1 ms |

**No sequence reaches parity.**  4→3 ops saves ~5-6 ms of ~63; the RN-exact PACK form is no faster
than the baseline.  The cost is **per converted word**, not per op — and the native *read* is already
~4 % faster than the staged dense read, so fixes A/B/C in the README are moot.  **V5 stays opt-in;
no delivery change.**

The structural reason (new): `cp_async_available()` is NVIDIA-only, so `nstages = 0` on RDNA4 — the
AMD MMA loader is synchronous and the conversion ALU/latency is exposed next to the MMA.  The kernel
is at the 256-VGPR ceiling in every variant, so there is no register budget for prefetch. Hiding the
conversion would need a loader/kernel pipelining change (its own A/B), not a loader-only tweak.

Also fixed: `tools/profile-fa.sh` only forced the native read for the exact label `native`, so
`pack_native`-style labels silently measured STAGED (a false parity result that cost a round).  It
now matches `*native*`.

---

## 0b. Historical mandate (superseded — kept for context)

**Make the delivery's V5 bf16-native flash-attention arm reach prefill *parity* with the
F16-staging path by making the in-register bf16→f16 conversion cheap — keeping f16 compute.**

Do **not** try to remove the F16 compute. That was step 1 and it is measured-dead on gfx1201
(§2). The target is the conversion, which is the one large, clean, addressable cost:
measured **~57 ms of the FA kernel** on the 35B-A3B, against a total V5 deficit of ~44 ms
(it also saves the 19.7 ms launcher conversion), i.e. ~15 ms of conversion would reach parity.

**Done means:** on the validation models, the V5 arm (`GGML_CUDA_FA_KV_NATIVE=1` +
`GGML_CUDA_FA_STAGE_MAX_MB=1`, native bf16 read) is **within ~0.2 %** of the staged path
(`GGML_CUDA_FA_KV_NATIVE=1`, staging allowed) at prefill — with the F16 staging scratch still
gone, decode unchanged, and **output byte-identical** (see §2, "why bit-identity holds").

Then V5 can be reconsidered for default-on, and the block-15 arena blind spot (issue #38
follow-up) disappears for bf16.

**Non-goals:** the MMA element type (step 1, closed), correctness work (bf16→f16 is exact so there
is none), mixed K/V types, the staging-buffer *type* question, the tensor-split `--fit` beta.

---

## 1. The subject, precisely

The MMA (prefill, `n_q > 8`) flash-attention kernel is an **F16 WMMA** kernel. For a bf16 K/V
cache, block 15's V5 arm makes the launcher skip the whole-cache F16 staging pass and lets the
kernel read the **raw bf16 cache** while converting each 16-byte chunk bf16→f16 in registers:

* `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
  * `flash_attn_ext_f16_load_tile_bf16(...)` — **the function to fix** (~line 471). The conversion
    is the inner loop (the current body):
    ```cpp
    ggml_cuda_memcpy_1<16>(tmp_bf, KV + (size_t) i_KV*stride_KV + (el_off + k*h2_per_chunk*2)*2);
    #pragma unroll
    for (int l = 0; l < h2_per_chunk; ++l) {          // h2_per_chunk == 4 half2 == 8 elements
        tmp[l] = __float22half2_rn(ggml_cuda_cast<float2>(tmp_bf[l]));
    }
    ```
    `tmp_bf`/`tmp` are `__align__(16) nv_bfloat162 tmp_bf[4]` / `half2 tmp[4]`.
    `ggml_cuda_cast<float2>(nv_bfloat162)` for HIP is
    `make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)))`
    (see `ggml/src/ggml-cuda/convert.cuh`).
  * the dispatch that reaches it: `flash_attn_ext_f16_load_tile(...)`, the
    `if (kv_native_type == FATTN_KV_NATIVE_BF16) { ...load_tile_bf16...; return; }` arm.
* `ggml/src/ggml-cuda/fattn-common.cuh` — the launcher/policy (not expected to change):
  `GGML_CUDA_FA_KV_NATIVE` (unset = auto: q8_0/q4_0 native on, bf16 native **off**; `=1` force on;
  `=0` force off), `GGML_CUDA_FA_STAGE_MAX_MB` (MiB/operand staging cap; **`=1` forces the native
  read = V5**, i.e. it makes the bf16 converting loader run at prefill).

The kernel is register-saturated (**256 VGPR** in every variant), so *anything that raises
pressure loses* — that is why step 1's f32 accumulator regressed.

---

## 2. Established facts — do NOT re-derive

### 2a. The measurement that defines the problem

`rocprofv3` per-kernel, **Qwen3.6-35B-A3B-Q8_0, 2×R9700 gfx1201, `-sm layer`, `-fa 1`, pp8192**,
MMA FA-kernel total (`flash_attn_ext_f16<256,256,8,8,...>`, 320 dispatches):

| arm | FA kernel | launcher `convert_unary` | total |
|---|---|---|---|
| f16 cache, raw interleaved (the **floor** — no conversion is possible) | 219.3 ms | 0 | 219.3 |
| **bf16 staged (today's default = the reference)** | **213.2 ms** | **19.7 ms** | **232.9** |
| **bf16 + V5 (native read, converting loader — the target)** | **276.6 ms** | 0.5 ms | 277.1 |
| bf16 + route 2 (bf16 MMA, no F16 at all) | 321.7 ms | 0 | — |
| bf16 + route 2, f32-accumulate VKQ | 389.4 ms | 0 | — |

**Decomposition:** native read = free (219.3); **in-loader conversion = ~57 ms** (→276.6);
bf16 *compute* = a further ~45 ms (→321.7). Step 1 spent ~102 ms of bf16-compute penalty to
remove a 19.7 ms launcher conversion. **Step 2 attacks the 57 ms and keeps f16 compute.**

### 2b. The corrected root cause (the old story is wrong)

The recorded V5 rationale — "the F16 staging pass is really a de-interleave, and the interleaved
native read is the cost" — is **wrong**. Measured: dense-vs-interleaved layout is worth **~3 %**
(the f16-cache arm reads the *same* interleaved stride through the *same* F16 loader at 219.3).
The real cost is the **conversion ALU**, re-paid on every K/V tile re-read (the launcher converts
each element once). Anyone re-litigating "reorder the loader / head-major cache" is chasing 3 %.

### 2c. The ISA limit that killed step 1 (verified with the assembler — `-S` proves nothing!)

`hipcc --offload-arch=gfx1201 --cuda-device-only -c` (the assembler is what rejects):

| instruction | gfx1201 | |
|---|---|---|
| `v_pk_mul_bf16`, `v_pk_add_bf16` | **not supported on this GPU** | packed bf16 arith is gfx950/CDNA4-only |
| `v_cvt_pk_bf16_f32` | **not supported on this GPU** | f32→bf16 packing is gfx950-only |
| `v_cvt_pk_f16_bf16` | **invalid instruction** | no bf16→f16 pack exists in the ISA |
| `v_cvt_f32_bf16` | not supported (that name) | bf16→f32 is the integer shift below |
| `v_pk_mul_f16` | OK | f16 keeps the full packed set |
| `v_lshlrev_b32`, `v_and_b32` | OK | bf16→ **f32** is `<<16` / `&0xffff0000` |
| `v_dot2_f32_bf16` | OK | the only packed-ish bf16 op RDNA has (f32 accumulate) |

Also verified: **bf16 WMMA is equal-rate to f16** (`wmma_f32_16x16x16_bf16` 0.99×,
`wmma_bf16_16x16x16_bf16` 1.00×, clock-warmed) and occupancy is identical (256 VGPR / 128 SGPR).
And `__float22half2_rn(float2)` on gfx12 lowers to **SALU**: `s_cvt_f16_f32` ×2 +
`s_pack_ll_b32_b16` (not the old `v_cvt_pk_f16_f32`, which does not exist under that name here).

### 2d. Why bit-identity holds (and must be preserved)

bf16 has 7 mantissa bits, f16 has 10, so **bf16→f16 is exact** for every in-range value (the same
fact the delivery already relies on). So a *cheaper* bf16→f16 conversion is a pure codegen change:
same values in the tile, same arithmetic, **byte-identical output**. That is why this route is
strictly better than step 1 — every existing validation gate stays valid, including the
`--spec-type none == draft-mtp` purity invariant, and `test-backend-ops -o FLASH_ATTN_EXT` must
stay **5952/5952**.

### 2e. Tooling lessons

* **`rocprofv3` on this box aborts in its rocpd/SQLite writer**
  (`ROCPD_STATUS_ERROR_SQL_SCHEMA_INVALID_VERSION`) and then hangs in its signal handler.
  **Always pass `--output-format csv`** — `tools/profile-fa.sh` / `tools/profile-bf16mma.sh` do.
* **`-S` output is not proof an instruction exists** — only `-c` (the assembler) is. Cost me a
  full wrong-turn this session.
* The GPU boosts after ~2 benchmark rounds; microbenches must warm up and interleave arms or they
  lie (an early f16-vs-bf16 WMMA run showed 2.8× purely from clock ramp).

---

## 3. The task — **DONE (negative result)**

> **Result:** the emitted sequence is already the RN minimum, and no candidate (RTZ packed, RN-exact
> PACK) reaches parity.  The SALU suggestion in task 2 below is impossible (the tile data is
> per-lane; scalar converts cannot be used) and the earlier "~15 ms of conversion would reach
> parity" premise is wrong (the cost is ~63 ms and per-word, not per-op).  Tasks 1-3 below are kept
> as the method; see §0 for the outcome.

1. **See what the conversion currently costs in ISA** (done).  The compiler emits, per 32-bit word
   (2 bf16 elements): `v_lshlrev_b32` + `v_and_b32` + `v_cvt_f16_f32 v.l` + `v_cvt_f16_f32 v.h` =
   **4 ops**.  Assemble-probe the candidates with `llvm-mc -triple=amdgcn -mcpu=gfx1201` (never trust
   `-S`): `v_cvt_pk_f16_f32` (packed RN) is **absent**; `v_cvt_pkrtz_f16_f32` (packed RTZ) and
   `v_pack_b32_f16` exist.
2. **The minimal sequence.**  The only cheaper packed form is RTZ (3 ops), which is **not bit-exact**
   for f16 subnormals; an RN-exact form (`2× v_cvt_f16_f32` full-reg + `v_pack_b32_f16`, 4 ops) is
   no faster.  Neither reaches parity.

---

## 4. Validation protocol

* **A/B (same binary, no rebuild):** `GGML_CUDA_FA_KV_NATIVE=1` with `GGML_CUDA_FA_STAGE_MAX_MB`
  unset (staged, reference) vs `=1` (V5 native). Interleave arms, `-r >= 5`; the effect is ~1 % so
  r=3 noise will fool you.
* **Floor reference:** `-ctk f16 -ctv f16` (raw f16 cache, no conversion possible) — the best the
  FA kernel can be; V5 must approach it, not the staged number indefinitely.
* **Kernel-level check (the decisive instrument):**
  `archive/work/bf16-native-prefill/tools/profile-fa.sh <label> <model> 8192` then
  `python3 archive/work/bf16-native-prefill/tools/fa-stats.py profiles/<label>/prof_kernel_trace.csv 'flash_attn_ext_f16<'`.
  Success = the V5 FA kernel total moves from ~276.6 ms toward ~234 ms (parity with
  213.2 + 19.7) / ~219 ms (the floor).
* **Correctness:** `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT` must stay **5952/5952** (it will —
  bf16→f16 is exact — but prove it); greedy text `--spec-type none == draft-mtp` byte-identical
  within the `n_max <= 7` band; same-seed output byte-identical across the A/B (that is the whole
  point of this route).
* **Memory win retained:** the F16 scratch stays gone (`llama_get_memory_breakdown` / the reserve
  matrix in `archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` §4).
* Never run parallel benches. Decode is not the subject here (verify it is unchanged, do not tune).

---

## 5. Environment

* 3× Radeon AI PRO R9700 (gfx1201, 32 GiB), ROCm 7.14 (`/opt/rocm-7.14-gfx1201`), 16 threads.
  GPUs 0/1/2 only (GPU 3 is a 4 GiB iGPU).
* `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`
* **Dev tree `~/llama.cpp`, branch `rdna-boosts`, HEAD `cc586b391`, build `build-rocm/`.**
  **Restored to a clean delivery tree at the start of step 2**: the 4 `ggml/src/ggml-cuda/*` files
  that carried the step-1 route-2 arm were reverted and the tree rebuilt, so the binaries match the
  delivery. The only remaining modifications are the **pre-existing, unrelated** ones:
  `common/common.cpp` + `common/fit.cpp` + `docs/multi-gpu.md` + `ggml/include/ggml-backend.h` +
  `ggml/src/ggml-backend-meta.cpp` (the issue-#38 `--fit` fix and the tensor-split `--fit` beta).
  Do not revert those, and do not commit them into this repo's WIP.
* Build: `cd ~/llama.cpp && cmake --build build-rocm --target llama-bench llama-cli test-backend-ops -- -j16`
  (~5 min for a full FA rebuild at `-j16`; one MMA instance TU is ~86 s). Configure is pinned —
  do not re-run cmake; a fresh configure needs `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` (see AGENTS.md).
* Step-1's route-2 code is preserved as
  `archive/work/bf16-native-prefill/patches/route2-native-bf16-mma.patch` (731 lines, `ggml/src/ggml-cuda/`
  only). **Do not apply it for step 2** (it changes the element type; step 2 is loader-only). It is
  kept so the arm can be re-measured on packed-bf16 hardware (gfx950/CDNA4) without re-deriving it.
* Models: 35B-A3B `/llm/models/Qwen3.6/35B-A3B/Q8_0/` (memory-bound, exposes the effect best),
  4B `/llm/models/Qwen3.5/4B/Q8_0/`, 27B `/llm/models/Qwen3.8/27B/Q8_0/`, gemma-4-E4B / -31B
  (SWA), EfficientThink 27B (MTP).

---

## 6. Repo rules that bite

* `llama-cli` **always** gets `--single-turn` (+ `--no-display-prompt` for scripted runs); wrap in
  `timeout`.
* **WIP is not the delivery.** Nothing under `wip/` (including the patch above) is ever applied to
  the fork or folded into `patches/` unless the maintainer explicitly asks — and never as a
  permanent drift fix.
* Never push anything out of `~/llama.cpp`. Push only `~/llama-cpp-rdna-boosts`'s `origin/main`.
* A change that keeps output byte-identical still needs the coherence + purity gates before it is
  claimed.

---

## 7. Artifacts

```
archive/work/bf16-native-prefill/
  HANDOVER.md                          # this file (step-2 brief)
  README.md                            # the project record: both attempts, measurements, ISA table, conclusions
  IMPLEMENTATION-PLAN.md               # step-1 design (route 2) — historical
  patches/route2-native-bf16-mma.patch # step-1 code, for re-testing on new hardware only
  tools/profile-fa.sh                  # rocprofv3 staged-vs-native FA profile (csv!)
  tools/profile-bf16mma.sh             # same, with the step-1 gate on
  tools/agg-kernels.py, fa-stats.py    # aggregation (total / per-kernel / per-dispatch stats)
  profiles/{staged,native,bf16mma,bf16mma2}/  # the raw traces behind §2a
```

External context: `archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` (the V5 plan),
`patches/README.md` (the block-15 V4/V5 notes), `TODO.md` item 23, `benchmarks/mtp-adaptive-methodology.md`.

---

## 8. Git state

* `~/llama-cpp-rdna-boosts`, branch `main`. Step-1 commits (all local unless pushed):
  `b9cc078` (route-2 implementation), `ea9adce` (f32-accumulator negative result), `a913e02`
  (ISA-verified limit + revised next step), plus this handover rewrite. Earlier:
  `5b6b5b6` (corrected root cause), `1e599bb`/`0cb5cf9` (project open).
* `git status --short` in this repo must be clean before pushing; push `origin/main` only.

---

## 9. Copy-paste prompt for the fresh session

```
Work in /home/stew675/llama-cpp-rdna-boosts.  Read AGENTS.md first (its rules override
everything else), then archive/work/bf16-native-prefill/HANDOVER.md and archive/work/bf16-native-prefill/README.md.

NOTE: this project's step 2 (cheapen the bf16->f16 conversion in the MMA tile loader) is
CLOSED as a negative result -- see HANDOVER section 0 and README "Step 2 findings".  The
conversion is already the RN minimum, and neither the packed RTZ form nor an RN-exact
v_pack_b32_f16 form reaches parity; V5 stays opt-in.  Only reopen with a NEW idea (e.g. AMD
loader pipelining, or gfx950/CDNA4 packed-bf16 hardware), not by re-trying the sequences.

The target function was flash_attn_ext_f16_load_tile_bf16 in
ggml/src/ggml-cuda/fattn-mma-f16.cuh; the delivery tree is restored to the unmodified state.

Measure with the kernel-level instrument, not end-to-end guesses:
archive/work/bf16-native-prefill/tools/profile-fa.sh / fa-stats.py.  Baseline: V5 FA kernel 276.6 ms,
staged 213.2 + 19.7 ms, f16-cache floor 219.3 ms on Qwen3.6-35B-A3B pp8192 (2xR9700,
-sim layer).  A/B is GGML_CUDA_FA_KV_NATIVE=1 with GGML_CUDA_FA_STAGE_MAX_MB unset (staged) vs
=1 (V5), interleaved, r>=5.  Gates: test-backend-ops -o FLASH_ATTN_EXT 5952/5952 and
--spec-type none == draft-mtp byte-identical.

Environment: ~/llama.cpp (rdna-boosts, build-rocm) restored to a clean delivery tree -- the
step-1 arm is NOT applied (it is preserved in archive/work/bf16-native-prefill/patches/ for re-testing on
gfx950/CDNA4 hardware only).  Keep everything under archive/work/bf16-native-prefill/; WIP is never the
delivery.  Push only this repo's origin/main.  If the conversion cannot be made cheap, a
well-evidenced negative result is a valid outcome -- do not force a change.
```
