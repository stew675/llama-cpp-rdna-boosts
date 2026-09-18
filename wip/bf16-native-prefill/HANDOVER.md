# HANDOVER — bf16 prefill parity, STEP 2: cheapen the bf16→f16 conversion (`wip/bf16-native-prefill`)

**Status: step 1 (route 2, native bf16 MMA) is CLOSED as a negative result. Step 2 is the live task.
Nothing pushed except this repo's own `origin/main`.**
Repo root: `~/llama-cpp-rdna-boosts` — read `AGENTS.md` first; its rules override anything here.
Project record: `wip/bf16-native-prefill/README.md` (the full narrative; §"Route 2 implemented",
§"Attempt 2", §"What it would actually take" are the load-bearing parts).

---

## 0. Mandate (step 2)

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

## 3. The task

1. **First, see what the conversion currently costs in ISA.** Build a one-file probe (or `-S` the
   real TU) for gfx1201 and dump the emitted sequence for
   `__float22half2_rn(ggml_cuda_cast<float2>(x))` on an `nv_bfloat162`, i.e. the per-chunk inner
   loop. Count the ops per 2 elements. Baseline that before changing anything — the whole task is
   "get this sequence shorter / onto the idle pipe".
   ```bash
   cd ~/llama.cpp
   cat > /tmp/cvt.hip <<'EOF'
   #include <hip/hip_runtime.h>
   #include <hip/hip_fp16.h>
   #include <hip/hip_bfloat16.h>
   typedef __hip_bfloat162 nv_bfloat162;
   __global__ void k(const nv_bfloat162 *i, half2 *o, int n) {
       int j = blockIdx.x*blockDim.x + threadIdx.x; if (j >= n) return;
       nv_bfloat162 x = i[j];
       o[j] = __float22half2_rn(make_float2(__bfloat162float(__low2bfloat16(x)),
                                            __bfloat162float(__high2bfloat16(x))));
   }
   EOF
   /opt/rocm-7.14-gfx1201/bin/hipcc -O3 --offload-arch=gfx1201 --cuda-device-only -S -o /tmp/cvt.s /tmp/cvt.hip
   awk '/_Z1k/,/s_endpgm/' /tmp/cvt.s | grep -E "^\s+v_|^\s+s_" | head -30
   ```
   (If the `__hip_bfloat162` type is not found, `-I/opt/rocm-7.14-gfx1201/include`; this exact
   include dance burned time before — see README.) Then, for any candidate mnemonic you want to
   use, **prove it with `-c`**, not `-S`.
2. **Write the minimal bf16→f16 sequence.** The mathematical minimum per 2 elements is:
   bf16→f32 as an integer op (`v_lshlrev_b32 x,16` / `v_and_b32 x,0xffff0000`) for the low/high
   halves, then one packed f32→f16 (`s_cvt_f16_f32` + `s_pack_ll_b32_b16`, or whatever gfx12
   actually offers — find the real packing op with an ISA probe). Targets, in order:
   * avoid HIP's `__low2bfloat16`/`__high2bfloat16`/`__bfloat162float` round-trip if it emits more
     than the shift/and pair;
   * keep the whole sequence on the **scalar (SALU)** pipe if possible — the loader's pressure is
     on the vector/memory pipe and the kernel is VGPR-ceiling-bound, so SALU work can overlap;
   * consider processing 32 bytes per thread iteration if the smem swizzle granularity permits it
     (it is 16 B — check before assuming);
   * **do not** add live registers/locals to the hot loop (that is how step 1's fix lost).
3. **A/B and measure** (§4). Iterate on the loader only; keep the diff minimal and local.

**Fallback / honest outcome:** if the conversion cannot be made cheap (e.g. the minimum sequence is
already what the compiler emits and the 57 ms is inherent), then the answer is "V5 stays opt-in"
plus the measured reason, and step 2 is documented as closed. A negative result with a clean
decomposition is a valid deliverable here — step 1 is the precedent. **Do not force a change.**

**Also document** (small, cheap, valuable): the corrected root cause (§2b) belongs in `TODO.md`
item 23 and the block-15 notes so nobody re-derives the de-interleave story.

---

## 4. Validation protocol

* **A/B (same binary, no rebuild):** `GGML_CUDA_FA_KV_NATIVE=1` with `GGML_CUDA_FA_STAGE_MAX_MB`
  unset (staged, reference) vs `=1` (V5 native). Interleave arms, `-r >= 5`; the effect is ~1 % so
  r=3 noise will fool you.
* **Floor reference:** `-ctk f16 -ctv f16` (raw f16 cache, no conversion possible) — the best the
  FA kernel can be; V5 must approach it, not the staged number indefinitely.
* **Kernel-level check (the decisive instrument):**
  `wip/bf16-native-prefill/tools/profile-fa.sh <label> <model> 8192` then
  `python3 wip/bf16-native-prefill/tools/fa-stats.py profiles/<label>/prof_kernel_trace.csv 'flash_attn_ext_f16<'`.
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
  `wip/bf16-native-prefill/patches/route2-native-bf16-mma.patch` (731 lines, `ggml/src/ggml-cuda/`
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
wip/bf16-native-prefill/
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
everything else), then wip/bf16-native-prefill/HANDOVER.md and wip/bf16-native-prefill/README.md.

This is STEP 2 of the bf16 prefill-parity project: make the delivery's V5 bf16-native
flash-attention arm reach prefill parity with the F16-staging path by making the in-register
bf16->f16 conversion in the MMA tile loader CHEAP.  Do NOT try to remove the F16 compute -- that
was step 1 (full native bf16 MMA) and it is measured-dead on gfx1201: bf16 WMMA is equal-rate to
f16 but gfx1201 has no packed bf16 arithmetic and no bf16->f16 pack, so the element-wise glue
costs more than the conversion it removes.  See HANDOVER sections 2a-2c.

The target function is flash_attn_ext_f16_load_tile_bf16 in
ggml/src/ggml-cuda/fattn-mma-f16.cuh (~line 471); the current inner loop is
tmp[l] = __float22half2_rn(ggml_cuda_cast<float2>(tmp_bf[l])).  Start by dumping the emitted
ISA for that sequence (HANDOVER section 3 has the exact probe), then hand-write the minimal
bf16->f16 sequence using only ops that ASSEMBLE on gfx1201 (prove with `-c`, never trust -S).
bf16->f16 is exact, so the output must stay byte-identical.

Measure with the kernel-level instrument, not end-to-end guesses:
wip/bf16-native-prefill/tools/profile-fa.sh / fa-stats.py.  Baseline: V5 FA kernel 276.6 ms,
staged 213.2 + 19.7 ms, f16-cache floor 219.3 ms on Qwen3.6-35B-A3B pp8192 (2xR9700,
-sim layer).  A/B is GGML_CUDA_FA_KV_NATIVE=1 with GGML_CUDA_FA_STAGE_MAX_MB unset (staged) vs
=1 (V5), interleaved, r>=5.  Gates: test-backend-ops -o FLASH_ATTN_EXT 5952/5952 and
--spec-type none == draft-mtp byte-identical.

Environment: ~/llama.cpp (rdna-boosts, build-rocm) restored to a clean delivery tree -- the
step-1 arm is NOT applied (it is preserved in wip/bf16-native-prefill/patches/ for re-testing on
gfx950/CDNA4 hardware only).  Keep everything under wip/bf16-native-prefill/; WIP is never the
delivery.  Push only this repo's origin/main.  If the conversion cannot be made cheap, a
well-evidenced negative result is a valid outcome -- do not force a change.
```
