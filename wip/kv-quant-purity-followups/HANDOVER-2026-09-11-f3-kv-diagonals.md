# HANDOVER — F3 step 1: a *real* native FA path for the flag-gated KV types (`q4_1`, `q5_0`, `q5_1`)

**Point the next session at this file.**  It is self-contained for F3 step 1.  The shared
infrastructure (environment, models, probe, harnesses, landing procedure, trap list) is described in
**`HANDOVER-2026-09-11-remaining-work.md`** in this same directory — read §2-§5 and §12-§13 of that
file first, then this one.  **`iq4_nl` is deliberately out of scope here** (it is F3 step 2: see §1.3).

## 1. The mission

**Goal**: make `--cache-type-k`/`--cache-type-v` in {`q4_1`, `q5_0`, `q5_1`} (K type == V type) a
*properly supported* KV cache on RDNA4/gfx1201: a real native FA path (not a per-step f16 conversion),
bit-identical across the decode/verify band (`W = 1..8`), and usable under multi-GPU
`SPLIT_MODE_TENSOR`.

### 1.1 Why it matters (per-type inventory)

| type | size/KV (per 1k cells × 8 heads × 128 dims, f16 ≈ 2 MiB) | source of the cost today |
|---|---|---|
| `q4_0` | ~0.6 MiB | works (shipped diagonal instance) |
| `q4_1`, `q5_0`, `q5_1` | 0.6-0.9 MiB | `ggml_cuda_fattn_kv_type_supported()` returns **false** ⇒ no native path |
| `iq4_nl` | ~0.6 MiB (same as `q4_0`) | same, **plus** it has no instance and no V-side dequant at all (step 2) |
| `q8_0` | ~1.1 MiB | works (shipped diagonal instance) |
| `f16` / `bf16` | 2 MiB | works (native) |

The three step-1 types are **pure** (measured 2026-09-11: the whole `{q4_1, q5_0, q5_1, iq4_nl}` set is
width-pure — they are *not* impure, only *slow*), and they currently cost roughly **3.4× prefill /
~1.7× decode** versus the f16-staged reference, because every FA call materialises f16 copies of K and
V through the launcher's `f16_extra` scratch.  Under `GREEDY-PURITY.md` §19 this is a **memory play**:
it does not improve MTP accuracy (these are lower-precision caches), it makes 0.6-0.9 MiB KV usable at
speed.

### 1.2 The K==V policy is load-bearing (read this before designing)

Mixed K/V cache types are a **rejected configuration** (maintainer decision 2026-09-11), and — usefully
— the FA chooser already encodes it:

```cpp
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;   // fattn.cu ~578
    }
#endif
```

Consequences for the design:

1. **Only the 7 diagonal (K,V) pairs are reachable** out of the 49 cross-product instances.  Each pair
   is its own kernel family and — per the F1 lesson — *each newly enabled pair needs its own `W=1..8`
   purity sweep*, so keeping the diagonal-only invariant is what makes the validation tractable.  Do
   **not** unlock mixed pairs (i.e. do not define `GGML_CUDA_FA_ALL_QUANTS` globally without thinking:
   it both unlocks the three types *and* removes the K==V guarantee).
2. **`-DGGML_CUDA_FA_ALL_QUANTS=ON` is the wrong tool**: it compiles all 45 extra cross-product VEC
   TUs (1-3 h build, memory-hungry) of which at most 3 are reachable.  See §4 — the whole question is
   which *family* those types would actually use.
3. **`iq4_nl` cannot be half-done**: with K==V mandatory there is no "iq4_nl K with an f16 V" shortcut,
   so step 2 must ship `dequantize_V_iq4_nl` and an instance.  (Out of scope here.)

### 1.3 Explicitly NOT in this session

* `iq4_nl` (step 2 — the V-side `dequantize_V_iq4_nl` + instance + the same sweep).
* The gfx1151 bundle (the crossover re-evaluation + `GREEDY-PURITY.md` §18's two sparse-regime items) —
  deliberately deprioritised; do not start it.
* Any MTP-accuracy/memory work (block 15 promotion is a separate, window-gated track).

## 2. State you are starting from

* **Canonical fork**: `/tmp/canon-llama`, branch `rdna-boosts`, tip **`6f07fe67a`**, net tree
  **`0c9dece6b0798e41360b8a8366187f38f37e1566`**, 15 blocks (00-14), clean, `build-base` built.
  (Step 1 landed on 2026-09-11: **block 08 was amended** with the enablement — the predicate, the vec
  dispatch and the three diagonal instances in `ggml-{cuda,hip,musa}/CMakeLists.txt` — and **block 14
  twice more**: the QSA-vs-KV-type arm gate and the `llama_kv_type_has_native_fa()` tensor-split gate.)
  *(If `/tmp` was wiped, rebuild it: clone at `9113cc188`, `scripts/apply-all.sh`, then
  `BUILD_DIR=build-base EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` — the
  `EXTRA_CMAKE_FLAGS` override is required on CMake ≥ 4.3.)*
* **Delivery repo**: `~/llama-cpp-rdna-boosts`, `main` == `origin/main` == **the commit that contains
  this file** (pushed), 15 patches `0000`-`0014`, `make-patches.sh` default tip `6f07fe67a`.
* **Block 15 beta**: base `6f07fe67a` → beta commit `8c377b958`, tree `34527a292` (sixth re-cut,
  2026-09-11; the merge threads the KV type through block 15's `qwen4exp_qsa_sparse()` via new
  `llama_cparams::type_k/type_v` fields); **any block
  amendment invalidates it and it must be re-cut** (`beta/block-15-campaign-wins/`, `HANDOVER.md`
  §10.5; note it now needs `git am -3`).
* **Reference state**: the fork's FA story is *already* fork-specific — block 08's F1 amendment made
  the decode/verify band use one kernel family (`BEST_FATTN_KERNEL_TILE` throughout), and block 15
  (beta) contains the fork's per-operand native-KV machinery (`FATTN_KV_NATIVE_{NONE,Q8_0,BF16}` +
  `GGML_CUDA_FA_KV_NATIVE`, opt-in default 0).  **Read those two before designing anything.**

## 3. What is VERIFIED about the mechanism (checked in the source 2026-09-11)

1. `ggml_cuda_fattn_kv_type_supported()` — `ggml/src/ggml-cuda/fattn.cu:471` — returns `false` for
   `q4_1`/`q5_0`/`q5_1` unless `GGML_CUDA_FA_ALL_QUANTS` is defined, and `false` for `iq4_nl`
   unconditionally (`default:`).
2. `ggml/src/ggml-hip/CMakeLists.txt`: `fattn-tile*.cu` (12 head-size instances) and `fattn-mma*.cu`
   (21) are globbed **unconditionally**; **only** `fattn-vec*.cu` is gated — all 49 cross-product TUs
   under the flag, or exactly four diagonals without it (`f16-f16`, `q4_0-q4_0`, `q8_0-q8_0`,
   `bf16-bf16`).
3. The TILE launcher **converts everything except f16 (and bf16 on capable hardware) to f16**:
   `fattn.cu:706-717` sets `need_f16_K/need_f16_V` from `K->type != GGML_TYPE_F16`, and the comment
   says so explicitly.  `fattn-tile.cuh` contains no dequant calls for K/V, and there is no
   `dequantize_K_*` family.
4. The V-side dequant set (`fattn-common.cuh`) is `f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0` — i.e.
   **step 1's three types already have a V-side dequant**, and `iq4_nl` does not (that is step 2's
   work).
5. The VEC family *is* per-(K,V)-type with native quantized support
   (`FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)` etc., `fattn.cu:402-435`), and its
   `need_f16_*` only triggers for F32 — but **every VEC return in the chooser sits inside
   `turing_mma_available(cc)` / `volta_mma_available(cc)` branches**, i.e. those returns are
   NVIDIA-only, and the AMD path (`amd_mfma_available`, `fattn.cu:638+`) falls through to
   `MMA_F16`/`TILE`.
6. `src/llama-context.cpp:~3716` rejects, for multi-GPU `SPLIT_MODE_TENSOR`, every quantized KV type
   except `q4_0`/`q8_0`, with the comment that the FA kernels "read the quantized K/V cache natively
   only for a subset" and the rest cannot be expressed by the meta splitter.  This is the *same* root
   cause as the slowness, so F3 step 1 also **narrows this gate** (§8).

## 4. FIRST TASK (before any code): establish which path these types need

> **ANSWERED 2026-09-11 — see §10 for the result and what step 1 landed.**  Short version: the AMD band
> takes `BEST_FATTN_KERNEL_TILE` with the launcher's f16 staging (`need_f16_K/V = 1`) for *every*
> quantized cache type the predicate accepts, and the flag-gated types produced **no FA call at all**
> (the failed probe disabled FA for the whole context).  That is **case A/B crossed**: no new kernel is
> needed (the staging path covers these types), the enablement is the predicate + the vec dispatch +
> three diagonal instances, and the interesting work turned out to be a *pre-existing* qwen4exp
> tensor-split abort that the enablement exposed.  The instrument recipe below is kept for step 2
> (`iq4_nl`), where the family question is still open.

The design is *not* settled by the source alone — points 3-5 above leave a real ambiguity: if the
AMD band always lands on TILE (which converts to f16), then flipping the predicate buys **nothing**, and
the fix has to be a *native consumer* for these types.  Resolve it empirically, cheaply:

1. Apply the existing instrument `wip/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch`
   (it is the trace written during the F1 work; `GGML_CUDA_FA_TRACE=1` prints the chosen family and the
   launch plan per call), rebuild `ggml-hip`, and run the **same** short prompt at depth with:
   `-ctk q8_0 -ctv q8_0` (supported, fast), `-ctk q4_0 -ctv q4_0` (supported, fast),
   `-ctk q4_1 -ctv q4_1` (unsupported, slow).
   Read off, per call: the chosen family, `need_f16_K/V`, and (from the log) whether a conversion
   launch happens.
2. Also record the **`llama-batched-bench`** pp/tg for those three configs (with `RV_DEV`/`RV_SM` for
   the model you use) as the baseline you must beat, plus the KV reserve figures.
3. Write the outcome (family + need_f16 + numbers) into this file or a sibling note **before** writing
   code.  The next step depends on it:
   * **Case A — the band already uses a native-capable family for quantized KV, and only the predicate
     blocks it** (e.g. the fork's block-15 `FATTN_KV_NATIVE` staging code is what serves `q8_0`
     natively): then step 1 is *enable the types in that machinery* — likely adding the three types to
     the staging type-code enum plus the predicate — and the block-15 dependency must be resolved
     (ride the promotion, or reimplement in a delivery block; block 15 is beta-only).
   * **Case B — the band uses TILE and TILE converts these types**: then a predicate flip is useless.
     Either extend the TILE path to read them natively (a kernel change with a strict band-uniformity
     requirement, F1-style), or route those types to the VEC family *for the whole band on AMD*
     (`can_use_vector_kernel` + a uniform `Q->ne[1] <= 8` predicate — never a band split), which
     requires the three diagonal VEC instances to be compiled (3 TUs — cheap — but remember point 1.2:
     the define must not silently unlock mixed pairs).
   * **Case C — neither is cheap**: stop and write the finding up; do not start a kernel port in the
     same session.

## 5. Purity gate for a newly enabled type (mandatory, whatever the mechanism)

A newly enabled (K,V) pair is a *new kernel family* on the decode/verify path, which is exactly how F1
split the band (`GREEDY-PURITY.md` §14).  So, per type:

```sh
# width purity, both splits (see HANDOVER-2026-09-11-remaining-work.md §4.1 for the probe build)
for w in 1 2 3 4 5 6 7 8; do
  HIP_VISIBLE_DEVICES=0,1,2 W=$w NGL=99 SPLIT=layer  RS=0     CTK=q4_1 CTV=q4_1 CB=0 /tmp/lw-f2 <model> <p0long.txt> 256
  HIP_VISIBLE_DEVICES=0,1,2 W=$w NGL=99 SPLIT=layer  RS=from_w CTK=q4_1 CTV=q4_1 CB=0 /tmp/lw-f2 <model> <p0long.txt> 256
  HIP_VISIBLE_DEVICES=0,1,2 W=$w NGL=99 SPLIT=tensor RS=0     CTK=q4_1 CTV=q4_1 CB=0 /tmp/lw-f2 <model> <p0long.txt> 256
done
```

All widths must give **one** hash per (split, RS) — and it must be stable versus the pre-change build
for the widths that already agreed.  Then the user-visible gate: `plain` vs
`draft-mtp --spec-draft-n-max 3` **and** `7` byte-identical text (f16-QA'd harness in the sibling
handover §4.2 — **default verbosity**, not `--log-verbosity 4`), and the adaptive-MTP gate
(acceptance **at pos 1** ≥ ~0.45, MTP ≥ plain at `n_max 3`).  Also run
`test-backend-ops -o FLASH_ATTN_EXT` (4/4 backends) and a **SWA sanity check** on
gemma-4-E4B (1 GPU) — the AGENTS rule for anything touching the FA path.

## 6. Perf + memory

* **The swappable-`.so` A/B works here** (the change is in `ggml-hip`), unlike the recent libllama-side
  fixes: keep a pre-change `libggml-hip.so.0.23.0` copy and use
  `wip/kv-quant-purity-followups/tools/sobench.sh` for an interleaved fixed/base comparison
  (`pl=1..8`).  Report pp and tg at `pl=1` and the verify widths.
* Target: **at least** the f16-staged reference for the same type (the point is to remove the
  conversion), and no regression for the four existing diagonals.  Under §19 a few percent is
  acceptable if the conversion is genuinely gone — but "+0 %" combined with no change in the trace
  means the enablement did nothing (case B in §4).
* Record the KV reserve deltas (the reason to use these types at all).

## 7. Tensor-split gate narrowing (part of the deliverable)

`src/llama-context.cpp:~3716` currently rejects the types at context creation when a Meta device is in
use.  Once the types have a real native path, narrow the gate to the types that are actually
unsupported (and keep the clear error for those).  Verify per type on **3-GPU `-sm tensor`**: context
creation succeeds, coherence is sane, and the width probe is pure there too.  Note this file is
**block 14's** (the gate came from the 2026-09-08 block-14 amendment), so this is a block-14
amendment.

## 8. Landing

Follow the sibling handover §12 exactly (owner-based block amendment via
`git rebase -i <prev-block-sha>` + `edit` → `git commit --amend --no-edit` → `--continue`;
`scripts/make-patches.sh /tmp/canon-llama 9113cc188 <new tip>`; update the default tip; refresh
`rdna-boosts-all.patch` by hand; clean-apply sim with the tree check; **re-cut block 15**; docs sweep;
commit + push **only** to the delivery repo's `origin`).  Owner determination by blame, not by guess:
the FA chooser/instances/HIP CMake are block 08's or block 13's area (check `git log` on the specific
files — `ggml/src/ggml-hip/CMakeLists.txt` is upstream-owned today, so a change there needs a
conscious owner choice) and the tensor-split gate is block 14's.

Deliverables for step 1: the enabling change, the purity matrix (per type × split × RS), the perf
matrix (fixed vs base, interleaved), the three-GPU tensor-split verification, the block-14 gate
narrowing, and a dated `WORKLOG.md` entry.  **Then** hand back with a short note on what step 2
(`iq4_nl`) needs, which by then should be precise: the V dequant + instance + the same sweeps.

## 9. Traps specific to this work

* **Do not use `-DGGML_CUDA_FA_ALL_QUANTS=ON` as the fix**: 1-3 h build, 45 TUs, at most 3 reachable,
  and it silently permits mixed K/V (§1.2).  It is fine as a *throwaway diagnostic* build if you want
  to see what the VEC family does for these types — in a separate build dir.
* **`--log-verbosity 4` interleaves log lines into the generated text** — text-purity runs use the
  default verbosity; the acceptance line needs 4; run those two measurements separately.
* **The MTP gate metric is acceptance at pos 1**, not the aggregate (see `GREEDY-PURITY.md` §19).
* The width probe cannot see this class of bug if the types ever become *impure* at a width the probe
  cannot reach (`P <= 2048` + one step) — the text runs are the backstop; and **a fresh type needs a
  positive control** that the new path is actually taken (that is §4's trace, or a
  `GGML_CUDA_DISABLE_...=1`-style A/B if the change adds a switch).
* Never rebuild while a bench is running; never run benches in parallel.
* `git checkout -- <path>` restores from the **index** — check `git status --porcelain` (first column =
  staged) and `git reset` first if a leftover was staged.
* The block-15 beta patch needs `git am -3` since the block-14 amendment.

## 10. OUTCOME — step 1 landed 2026-09-11 (addendum by the session that did it)

**What the trace said** (`GGML_CUDA_FA_TRACE=1`, 4B, `W=8`, `P=256`):

| KV type | `BEST_FATTN_KERNEL_*` | `need_f16_K/V` | note |
|---|---|---|---|
| f16 | TILE (200) | 0/0 | native |
| `q8_0` | TILE (200) | 1/1 | **staged**, not native |
| `q4_0` | TILE (200) | 1/1 | **staged**, not native |
| `q4_1` | *no FA call* | — | `ggml_cuda_fattn_kv_type_supported()` false ⇒ `resolve_fused_ops()`'s FA probe disabled FA for the context |

So the "supported" types were never native either: the tile/mma families read what
`ggml_get_to_fp16_cuda` covers through the launcher's staging copy, and the vec family (the only
per-(K,V)-pair family) is not reachable on AMD at all — every VEC return in the chooser sits in a
`turing_mma_available`/`volta_mma_available` branch.  The three types therefore needed **no new
kernel**: the predicate, the default vec dispatch list and the three diagonal instance lists had to be
made consistent, and that is the whole block-08 amendment.

**What landed**

* **Block 08 (second 2026-09-11 amendment)**: `Q4_1`/`Q5_0`/`Q5_1` lose the
  `#ifndef GGML_CUDA_FA_ALL_QUANTS` guard; the non-`FA_ALL_QUANTS` branch of
  `ggml_cuda_flash_attn_ext_vec` gains the three diagonal cases; `ggml-{cuda,hip,musa}/CMakeLists.txt`
  gain the three diagonal instances.  `FA_ALL_QUANTS` keeps its meaning for the mixed `K != V` pairs,
  and K==V stays enforced without it.
* **Block 14 (third 2026-09-11 amendment)**: (a) `qsa_sparse` in `build_attn_qsa` now also requires a
  QSA-native cache type (f16/bf16/`q8_0`), because with any other type the graph built a
  `GGML_OP_FLASH_ATTN_QSA` the backend cannot run — under `-sm tensor` that op was never split, so its
  output stayed mirrored while the attention gate stayed hidden-split and the meta splitter aborted on
  `MUL name=attn_gated-<il>` (`ggml-backend-meta.cpp:538`).  `q4_0` hit this too: **the previous gate's
  allowance was wrong, not just incomplete**.  (b) `llama_init_from_model`'s tensor-split gate now uses
  `llama_kv_type_has_native_fa()` (f32/f16/bf16/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`), so the three new
  types are allowed and `iq4_nl` keeps a clean error.

**Numbers** (details in the 2026-09-11 (8) `WORKLOG.md` entry): 4B pp512/tg32 `q4_1`
2119.6/55.94 -> **7366.3/94.16**; qwen4exp 3-GPU `-sm tensor` `q4_1` within 1 % of f16 everywhere;
`W=1..8` pure on 4B/27B(both splits)/MoE/gemma(SWA)/qwen4exp for all six types; plain == `n_max 3`
== `7` for the three new types on 27B and qwen4exp; MTP pos-1 acceptance `q4_1` 0.628 (qwen4exp) /
0.893 (27B); `FLASH_ATTN_EXT` 5599/5599.

**What is left: step 2 = `iq4_nl`.**  It is *not* the same shape as step 1: `iq4_nl` is rejected by
`ggml_cuda_fattn_kv_type_supported()`'s `default:` clause (so the predicate change above does not reach
it), it has no vec instance, and — the real work — **no V-side dequant**: the seven
`dequantize_V_*` in `fattn-common.cuh` stop at `q8_0`.  So step 2 = `dequantize_V_iq4_nl` + a
`fattn-vec-instance-iq4_nl-iq4_nl.cu` (+ the CMake entry, in all three backends) + the predicate case +
the same sweeps *and* the `-sm tensor` type verification.  The [F3 handover's §3 facts] still hold, and
the block-15 beta already carries a per-operand native-KV staging type code
(`FATTN_KV_NATIVE_{NONE,Q8_0,BF16}`) if a *native* (non-staged) path is wanted instead.
