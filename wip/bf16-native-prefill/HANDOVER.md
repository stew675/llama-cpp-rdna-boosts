# HANDOVER — native bf16 prefill parity (`wip/bf16-native-prefill`)

**Status: ACTIVE investigation, first measurement pass done, no fix yet.  Nothing pushed.**
Read §0–§3 to pick up; §4–§6 are the evidence and the plan; §7–§10 are environment, rules,
artifacts and the copy-paste session prompt.

Repo root: `~/llama-cpp-rdna-boosts` (read `AGENTS.md` first — its rules override anything here).
Project record: `wip/bf16-native-prefill/README.md`.  This file is the turnkey brief.

---

## 0. Mandate / definition of done

**Make native bf16 K/V flash-attention reach prefill *parity* with the F16-staging path — or beat
it.**  The maintainer's framing: the repo's "native BF16 support" arc is not finished while bf16 is
measurably slower, so this is a real problem, not a rounding error.  Staging *into* f16 vs bf16 is a
separate question — the core issue is the native-read penalty.

Done means: on the validation models, the V5 arm (`GGML_CUDA_FA_KV_NATIVE=1`, native bf16) is
**≤ ~0.2 %** off the staged path at prefill (or faster), with decode and greedy purity unchanged, and
the memory win (no F16 staging scratch) retained.  Then V5 can be reconsidered for default-on and the
block-15 arena blind spot (issue #38 follow-up) disappears for bf16.

Non-goals: correctness work (V5 is already bit-identical), the staging-buffer type question, mixed
K/V types (separate pre-existing bug), and the tensor-split `--fit` beta (separate, already staged).

---

## 1. The subject in one paragraph

Block 15's **V5** arm (`GGML_CUDA_FA_KV_NATIVE=1`, opt-in) makes the MMA flash-attention kernel read
a **bf16** K/V cache natively, converting each staged 16-byte chunk in registers
(`flash_attn_ext_f16_load_tile_bf16`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh`).  This removes V5's F16
**node staging scratch**, so a bf16 cache costs what an f16 cache costs in memory, output is
bit-identical, and decode is unaffected.  **But prefill is ~0.2–1.4 % slower** than
bf16-with-staging, so V5 ships opt-in.  The penalty is the whole investigation.

Why it *looks* paradoxical: the native path does *less* work (no whole-cache conversion pass), yet is
slower.  The V5 campaign's recorded explanation is that the F16 staging pass is really a
**de-interleave**: the raw WMMA K/V cache is `[token][head][dim]`, so for one head consecutive tokens
are `n_head_kv` rows apart, while the staged F16 copy is **dense** (`nb[1]` normalised to one row).
The tile loader re-reads the strided native view on every K/V staging pass.  The conversion itself is
free (native bf16 is within **0.17 %** of an f16 cache), and `cp_async` is NVIDIA-only, so it should be
purely the access pattern — **but §4 shows the simple `n_head_kv` story does not fit the data, so
that root cause is not yet confirmed.**

Reference: `archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` (the V5 plan: mechanism with
call sites, design, ship rule, the 5-part validation protocol).  V4 is the template V5 mirrors:
`archive/work/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md`.

---

## 2. What is already established (do not re-derive)

* **Gate/policy** (`ggml/src/ggml-cuda/fattn-common.cuh:110-160`):
  `GGML_CUDA_FA_KV_NATIVE` unset = auto (q8_0/q4_0 native **on**, bf16 native **off**), `=1` force
  all on, `=0` force the F16-staging path.  `=1` is what this project tests.
* **A/B knob for "native prefill"**: `GGML_CUDA_FA_STAGE_MAX_MB=1` makes the per-operand staging cap
  fail, so `launch_fattn` reads the raw cache at prefill.  It only affects the prefill width
  (`native_width`), never decode/verify.  **This is the switch for the experiments — no rebuild per
  arm.**
* **Cache layout** (`src/llama-kv-cache.cpp:234`):
  `ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)` with
  `n_embd_k_gqa = head_dim * n_head_kv`.  So the cache is **`[token][head][dim]`**, and for a fixed
  head consecutive tokens are **`n_head_kv` rows** apart.  The dense staged F16 copy has token stride
  = 1 row.
* **Which kernel**: on gfx1201 (RDNA4), head 128/256/512 > 128 uses the **WMMA/MMA** path at prefill
  (`Q->ne[1] > 8`); decode/verify (`n_q <= 8`) uses **TILE**, where bf16 is already native (block 03)
  and there is no staging.  So the penalty is a **prefill/MMA** phenomenon.
* **`cp_async` is NVIDIA-only** (`common.cuh:373`), so on RDNA4 both loaders are the synchronous
  `ggml_cuda_memcpy_1<16>` path; the difference is dense-vs-interleaved addressing, not async.
* **The F16 staging pass is also a de-interleave**: `launch_fattn` runs `to_fp16_nc` over the whole cache
  view and rewrites the strides to dense (`nb11 = ne[0]*sizeof(half)`, `nb12 = ne[1]*nb11`, …)
  before the kernel reads it.  *(Confirmed in code: `ggml_is_contiguously_allocated(K)` is false for the
  FA view, so the `to_fp16_nc` branch runs and the copy really is dense.  **But §4's correction shows
  this dense layout is worth only ~3 % of the kernel time — it is not why staging wins.**)*
* **An f16 cache can never be staged** (f16 *is* the staging type), so f16 is stuck on the native
  side too — the dense staged copy is currently the fastest path for any type.

---

## 3. What this session established (2026-09-18, first pass)

`llama-bench`, `-sm layer`, 2×R9700 gfx1201, `-fa 1`, `GGML_CUDA_FA_KV_NATIVE=1`; **staged** =
default (512 MiB cap), **native** = `GGML_CUDA_FA_STAGE_MAX_MB=1`; r=3.

**Qwen3.5-4B-Q8_0 (n_head_kv=4, head 256, gqa 4):**

| test | bf16 staged | bf16 native | f16 native | native vs staged | bf16nat vs f16nat |
|---|---|---|---|---|---|
| pp2048 | 11707.6 | 11687.4 | 11638.5 | −0.17 % | +0.42 % |
| pp8192 | 12978.4 | 12916.9 | 12932.0 | −0.47 % | −0.12 % |
| pp20480 | 11859.3 | 11892.6 | 11971.8 | +0.28 % | −0.66 % |
| tg128 | 86.60 | 86.52 | 86.33 | −0.09 % | +0.22 % |

**Qwen3.6-35B-A3B-Q8_0 MoE (n_head_kv=2, head 256, gqa 8):**

| test | bf16 staged | bf16 native | native vs staged |
|---|---|---|---|
| pp2048 | 6861.7 | 6813.5 | −0.70 % |
| pp8192 | 7644.3 | 7542.1 | −1.34 % |
| tg128 | 81.62 | 81.39 | −0.28 % |

**Qwen3.8-27B-Q8_0 (n_head_kv=4, head 256, gqa 6), earlier r=2:** pp512 −1.6 %, pp2048 −1.0 %,
pp8192 −1.1 %, tg128 +0.1 %.

**Conclusions:**

1. **The `n_head_kv` scaling hypothesis FAILED.**  The 2-head 35B-A3B shows a *larger* penalty than
   the 4-head 4B.  The interleave factor alone does not predict the cost.
2. `bf16-native ≈ f16-native` (within ~0.66 % on the 4B), reproducing the campaign's "conversion is
   free" claim.  So the gap is **staged-dense vs native-interleaved**, not bf16 vs f16.
3. The effect is **small and noisy at r=3**; the 4B is compute-bound (~12k t/s pp) and likely masks
   it, the 35B is memory-bound (~7.6k t/s) and exposes it.  A clean scaling test needs two models of
   **similar size/boundness with different `n_head_kv`**, which the local set does not have.
4. The 4B pp20480 sign differs from the V5 plan (+0.28 % now vs −1.06 % then) — re-baseline against
   the current r3 tree before trusting historical rows.

---

## 4. Hypotheses (ranked) and what each experiment decides

> ### CORRECTED 2026-09-18 (second pass) — read this first
>
> **H4 was right, H1 is ~3 %.**  The profile (35B-A3B pp8192, MMA FA kernel total) is:
> bf16 staged 213.2 ms / **f16 cache (interleaved, no conversion) 219.3 ms (+2.9 %)** /
> bf16 native (converting loader) 276.6 ms (+29.8 %).  So the interleaved layout costs
> ~3 % and **the in-register bf16→f16 conversion costs ~26 %** — re-paid on every K/V
> tile re-read, unlike the launcher's once-per-element staging pass.  §2's
> "the F16 staging pass is really a de-interleave" is *not* the reason staging wins.
>
> **Do NOT pursue candidates B (reorder the loader) or C (head-major cache) for this
> penalty** — they target the 3 %.  A (dense-layout gate) is likewise only the 3 %.
> The real structural gap is that **the MMA kernel is F16-only** (decode's TILE kernel
> already computes in native bf16): V5 is a native *read* with an f16 *compute*.
> §6's routes now reduce to (1) cheapen the conversion, keeping f16 WMMA and
> bit-identity (ceiling ~+6 % vs staged if the conversion were free), or (2) give the
> MMA kernel a **bf16 WMMA** path and delete the conversion (the actual "no F16" fix;
> `mma.cuh:1267` already has the RDNA4 bf16 builtin — but Q must become bf16 too, so
> the logits change and V5-on/off is no longer bit-identical).  Details in README.md
> "Findings (2026-09-18, second pass)".  Profiling recipe: `tools/profile-fa.sh`
> (uses `--output-format csv`; the default rocpd/SQLite writer aborts and hangs here).

| # | hypothesis | test that decides it |
|---|---|---|
| H1 | the interleaved native layout is the cost, and it scales with the stride | **MEASURED: ~3 %.**  f16 cache (interleaved, F16 loader, no convert) vs bf16 staged (dense) |
| H2 | the cost is the **number of K/V re-reads** (layers × query tiles), not the stride | subsumed: the re-read *amplifies* the conversion (H4), it is not itself the cost |
| H3 | the 4B/35B inversion is a **boundness** artefact (compute-bound masks it) | confirmed in effect: the FA share of total work sets the end-to-end size (kernel +30 % -> run +1.2 % on a MoE) |
| H4 | it is something else in the loader (register pressure, address math, the `el_off` path, conversion ALU) | **MEASURED: ~26 % — the bf16->f16 conversion in the staging loop.**  f16 cache 219.3 vs bf16 native 276.6 |

**H4 is the answer; design the fix around removing/cheapening the conversion, not the layout.**

---

## 5. Investigation plan (next session — start here)

1. **Profile, don't guess.**  `rocprofv3` the FA kernel, staged vs native, on the memory-bound
   35B-A3B at pp8192:
   ```bash
   cd ~/llama.cpp
   B=./build-rocm/bin/llama-bench
   M35=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
   HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_FA_KV_NATIVE=1 \
     rocprofv3 --kernel-trace --stats -o /tmp/prof_staged -- \
     $B -m $M35 -ngl 99 -sm layer -fa 1 -ctk bf16 -ctv bf16 -p 8192 -n 0 -r 1
   HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_FA_KV_NATIVE=1 GGML_CUDA_FA_STAGE_MAX_MB=1 \
     rocprofv3 --kernel-trace --stats -o /tmp/prof_native -- \
     $B -m $M35 -ngl 99 -sm layer -fa 1 -ctk bf16 -ctv bf16 -p 8192 -n 0 -r 1
   ```
   Compare per-kernel time for the FA kernels (`flash_attn_ext_f16` / the MMA instantiation) and the
   launcher conversion kernels; then `--pmc` (memory-pipe busy, L2 hit, DRAM bytes) if the trace
   points at the FA kernel.  Note `rocprofv3` may need `--pmc` enabled and the usual ROCm
   permissions; capture the raw counters to a CSV in `wip/bf16-native-prefill/profiles/`.
2. **Controlled microbenchmark.**  One model, one variable.  Cheapest options in order:
   * Use `GGML_CUDA_FA_STAGE_MAX_MB` to switch staging on/off on the *same* binary (already the A/B).
   * To vary the *layout* independently of the model, add a temporary debug knob (e.g. an env that
     makes `launch_fattn` stage-and-convert but into a **padded** dense copy with an artificial
     stride) — a scratch-only patch, never committed.  Then sweep the stride and measure.
   * If a synthetic op test is easier, drive `FLASH_ATTN_EXT` directly via `test-backend-ops` with
     bf16 K/V and a pinned layout.
3. **High-rep interleaved A/B** (`-r 10`, alternate arms within one build) so the error bars are
   under ~0.2 %; the current r=3 spreads overlap.
4. **Re-baseline the V5 curve** on the current r3 tree (4B pp2048/8192/20480, 27B pp20480, f16
   reference) before quoting the historical plan numbers.
5. **Then choose** (see §6).  If parity is not reachable cheaply, the fallback deliverable is a
   well-measured recommendation: "V5 default-on despite X %" vs "keep opt-in" — with the profile as
   evidence.

---

## 6. Fix candidates (REVISED 2026-09-18 — the profile re-aimed these)

The original A-D table targeted the *layout* (measured ~3 %).  The penalty is the *conversion*
(~26 %), so the routes are now:

| # | idea | effort | notes |
|---|---|---|---|
| **1** | **cheapen the in-register bf16→f16 conversion** (fewer ISA ops / better scheduling / wider chunks) | small-medium | keeps the f16 WMMA and therefore remains **bit-identical** to the staged path (bf16→f16 is exact in range, so only the loader changes).  Ceiling is real: if the conversion were free, native ≈ 219 ms vs staged ≈ 232.5 ms (213 + the 19.5 ms conversion pass) — **up to ~6 % *faster* than staged**.  But it will not be free; expect partial recovery.  This is the only route that preserves the current ship guarantee. |
| **2** | **give the MMA kernel a bf16 WMMA path** and delete the conversion — the actual "no F16 anywhere" fix and the only route to *true* parity/win | medium-large | `mma.cuh:1267` already has `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` (used by the GDN work) and `bf16_mma_hardware_available()` exists; the TILE kernel already does exactly this for decode.  **Catch:** WMMA needs both operands the same type, so **Q (f32 today) must round to bf16** (7 vs 10 mantissa bits) — K/V stay lossless.  So prefill logits change and **V5-on is no longer bit-identical to staged-off**; it would instead make prefill match decode's existing bf16 precision (arguably *more* consistent).  Needs a re-baseline and maintainer sign-off. |
| 3 | keep the head-major cache (old **C**) or reorder the tile loader (old **B**) | large / medium | **deprioritized**: each buys only the ~3 % layout share.  Do not start here. |
| 4 | old **A** — restrict the native arm to already-dense layouts (`n_head_kv == 1`) | small | still a valid correctness-of-claim tidy-up (there the compact read is free), but it does not touch the GQA case. |

Ship rule to respect (maintainer's D9 refinement): a sub-2 % loss with a large memory win and **no
cheap fix** ships *opt-in*.  Route 1 failing means the honest outcome is "V5 stays opt-in" plus the
measured reason; route 2 is a numerical change and is a maintainer decision, not a tuning result.

---

## 7. Environment

* 3× Radeon AI PRO R9700 (gfx1201, 32 GiB), ROCm 7.14 (`/opt/rocm-7.14-gfx1201`), 16 threads.
  `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`.  GPUs 0/1/2 are the R9700s; GPU 3 is a 4 GiB
  iGPU (never use it).
* **Dev tree**: `~/llama.cpp`, branch `rdna-boosts`, HEAD `cc586b391` (the delivery tip), built in
  `build-rocm/`.  It currently has **uncommitted** changes that are *unrelated* to this project:
  * `common/common.cpp` — the issue-#38 `--fit` fix,
  * `common/fit.cpp` + `ggml/include/ggml-backend.h` + `ggml/src/ggml-backend-meta.cpp` +
    `docs/multi-gpu.md` — the tensor-split `--fit` beta.
  The FA kernels (`fattn-*.cuh/.cu`) are **clean**, so bf16 experiments are unaffected.  Build with:
  ```bash
  cd ~/llama.cpp
  cmake --build build-rocm --target llama-bench llama-cli llama-fit-params -j 16
  ```
  (Configure is pinned; do not re-run cmake unless needed.  `-DCMAKE_HIP_FLAGS="-mllvm"` quirk:
  `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is required for a fresh configure with CMake ≥ 4.3.)
  Runtime libs are in `build-rocm/bin/`; `LD_LIBRARY_PATH` is usually not needed for the binaries.
* **The `-sm layer` choice**: used for the 2-GPU benches above to avoid Meta/tensor-split
  complications.  `-sm tensor` is also valid; keep it consistent within a comparison.

### Local models (and their KV geometry)

| model | path | n_head / kv | head | notes |
|---|---|---|---|---|
| Qwen3.5-4B-Q8_0 | `/llm/models/Qwen3.5/4B/Q8_0/` | 16 / 4 | 256 | fast, compute-bound |
| Qwen3.5-9B-Q8_0 | `/llm/models/Qwen3.5/9B/Q8_0/` | 16 / 4 | 256 | |
| Qwen3.6-27B-Q8_0 | `/llm/models/Qwen3.6/27B/Q8_0/` | 24 / 4 | 256 | same shape as 3.8-27B |
| Qwen3.8-27B-Q8_0 | `/llm/models/Qwen3.8/27B/Q8_0/` | 24 / 4 | 256 | the measured ~1 % case |
| Qwen3.6-35B-A3B-Q8_0 | `/llm/models/Qwen3.6/35B-A3B/Q8_0/` | 16 / 2 | 256 | MoE, memory-bound |
| gemma-4-E4B-it-Q8_0 | `/llm/models/Gemma4/E4B-IT/` | 8 / 2 | 512 | ISWA |
| gemma-4-31B-it-Q8_0 | `/llm/models/Gemma4/31B/` | 32 / per-layer 16,4 | 512 | ISWA, per-layer kv |
| Qwen3.8-27B EfficientThink | `/llm/models/Qwen3.8/27B/EfficientThink/` | 24 / 4 | 256 | + MTP head, for the MTP gate |

`n_head_kv` per model read with `gguf-dump --no-tensors <file> | grep head_count_kv`.

---

## 8. Repo rules that bite

* **WIP is not the delivery.**  Never fold `wip/` into `patches/`, never apply it to the fork unless
  the maintainer explicitly asks.  Experimental patches stay in `wip/bf16-native-prefill/`.
* **`llama-cli` must always get `--single-turn`** (add `--no-display-prompt` for scripted runs);
  wrap blocking commands in `timeout`.
* **Never run parallel/background benches.**
* **Decode perf is validated at depth-16384**, but this project is a **prefill** subject: use
  `llama-bench -p` (and `-n` only to confirm no regression).
* **Greedy purity is an invariant**: any kernel arithmetic change needs
  `--spec-type none == draft-mtp` byte-identical within the `n_max <= 7` band, plus the MTP gate.
  V5 is expected to be bit-identical (the conversion is the launcher's own rounding) — verify, do not
  assume.
* **Pushing**: only this repo's own `origin` (`git@github.com:stew675/llama-cpp-rdna-boosts`), never
  anything out of `~/llama.cpp`.  SSH now works; plain `git push origin main` is fine.  Confirm scope
  with the maintainer for anything beyond the repo's `main`.

---

## 9. Artifacts and paths

```
wip/bf16-native-prefill/
  README.md        # the project record: goal, root cause, measured tables, hypotheses, options, protocol
  HANDOVER.md      # this file
  profiles/        # (create) rocprofv3 CSV/txt output
  tools/           # (create) any sweep scripts; keep them self-contained
```

External context:
* `archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` — the V5 plan (mechanism, design,
  5-part validation protocol, risks).
* `archive/work/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md` — V4, the template.
* `archive/work/block-15-campaign-wins/README.md` — the V5 promotion record (the "Cost — and why the
  ship rule landed on opt-in" section, line ~274).
* `patches/README.md` — the V4/V5 block-15 notes.
* `TODO.md` item 23 — this project.
* `benchmarks/mtp-adaptive-methodology.md` — the MTP gate protocol.

---

## 10. Git state and push

* Repo `~/llama-cpp-rdna-boosts`, branch `main`.  Last local commits:
  `1e599bb wip(bf16-native-prefill): open the native-bf16 prefill-parity project`, plus `TODO.md`
  item 23 in the same commit, **not yet pushed**.
* The tensor-fit beta (`0cd4856`, `a028966`, `89b7311`) is **already pushed**.
* Before pushing: `git status --short` must be clean, then `git push origin main` (SSH works).
* `~/llama.cpp` is never pushed (disposable fork checkout — see AGENTS.md "Pushing policy").

---

## 11. Copy-paste prompt for the fresh session

```
Work in /home/stew675/llama-cpp-rdna-boosts.  Read AGENTS.md first (its rules override
everything else), then wip/bf16-native-prefill/HANDOVER.md and wip/bf16-native-prefill/README.md.

Mandate: make native bf16 K/V flash-attention reach prefill PARITY with the F16-staging path
(block 15 V5, GGML_CUDA_FA_KV_NATIVE=1).  Today native prefill is ~0.2-1.4% slower on gfx1201.

Do what the handover says, starting at section 5: PROFILE FIRST (rocprofv3, staged vs native,
Qwen3.6-35B-A3B pp8192), then a controlled microbenchmark and a high-rep interleaved A/B.  The
recorded "GQA interleave scales with n_head_kv" explanation does NOT fit the data (35B-A3B
n_head_kv=2 is worse than 4B n_head_kv=4), so do not design a loader fix before the profile
distinguishes H1 (interleave/stride) from H2 (number of K/V re-reads) / H3 (boundness).

Environment: ~/llama.cpp (branch rdna-boosts, build-rocm), 2x R9700 gfx1201, ROCm 7.14;
rocprofv3 is at /usr/bin/rocprofv3.  The A/B switch for native prefill is
GGML_CUDA_FA_STAGE_MAX_MB=1 (no rebuild needed); GGML_CUDA_FA_KV_NATIVE=1 enables the arm.

Keep everything under wip/bf16-native-prefill/ (WIP is never the delivery).  Record findings and
raw counters there as you go.  When the investigation converges, the options are A (gate to dense
layouts, small), B (dense native read, the target), C (head-major cache, large/upstream) - see
handover section 6.  Nothing goes into patches/ without the maintainer's go-ahead.  Push only this
repo's origin/main.
```
