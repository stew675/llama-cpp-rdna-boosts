# BF16-native MMA K/V — implementation plan (block-15 amendment candidate)

**Status: PLANNED, not implemented.**  Measured 2026-09-10 in the *delivered block-15 tree* (the numbers
below are the "before" the next session will see).  This file is the primary input for the bf16 session;
`beta/block-15-campaign-wins/HANDOVER.md` §3.4 is the short pointer and §8 carries the copy-paste
prompt.

**Goal:** make the MMA (prefill) flash-attention path read a **bf16** K/V cache natively, so that a bf16
KV user pays no F16 staging scratch and no per-ubatch conversion pass — the same win V4 gave q8_0, but
for the maintainer's preferred KV type and with the `cp_async` pipeline intact (expected ~free, so it
should ship **on by default**).

**Scope fixed by D10:** keep the F16 fragment path (`T_A_KQ`/`T_B_KQ` stay F16/WMMA f16 builtins) and
convert the staged tiles in place.  Do **not** re-instantiate the kernels with bf16 fragments
(`__builtin_amdgcn_wmma_f32_16x16x16_bf16`): it is a large change, and it would actually *change* the
numerics, because every bf16 value is exactly representable in f16 (8 mantissa bits < 10), so today's
bf16→f16 staging is lossless for all in-range values.

---

## 1. Why (and what it is worth)

* The maintainer's preferred KV type is **bf16**; today it is the one type that still pays the whole
  staging cost, because `need_f16_K/V = true` is unconditional for `BEST_FATTN_KERNEL_MMA_F16`
  (`ggml/src/ggml-cuda/fattn.cu:735-736` only subtracts the q8_0 arm).
* The TILE (decode/verify) and VEC paths already read bf16 natively (block 03's `v_dot2_f32_bf16`
  work), so **only prefill pays**.

### Measured before-state (delivered block-15 tree, V3 on, ctx 204800, `-ctk/-ctv bf16`, f16 = reference)

Qwen3.5-4B-Q8_0, 1 GPU:

| ub | f16 | bf16 | **Δ (the scratch)** |
|---|---|---|---|
| 2048 | 256.86 | 968.86 | **+712.00** |
| 1024 | 128.82 | 884.82 | **+756.00** |
| 512 | 64.80 | 842.80 | **+778.00** |
| 8 (TILE) | 8.09 | 8.09 | **0** |

Qwen3.8-27B-Q8_0, 3-GPU Meta:

| ub | f16 | bf16 | **Δ** |
|---|---|---|---|
| 2048 | 488.86 | 1072.86 | **+584.00** |
| 512 | 122.80 | 868.80 | **+746.00** |

Notes:
* the delta is the K+V F16 scratch (`ctx × 4 KiB` = 800 MiB at ctx 204800) minus whatever the allocator
  overlaps with other live tensors — hence 584…778 MiB depending on model/ub.  It is **not** a single
  number; always re-measure per model/ub.
* `GGML_CUDA_FA_KV_NATIVE=1` (V4) does **not** change any bf16 number (V4's predicate is q8_0-only) —
  verified: 4B ub 2048 bf16 = 968.86 with V4 on and off.
* the ub-8 (verify/TILE) equality confirms prefill-only scope, as block 03 intended.

## 2. Mechanism (exactly where the cost lives)

1. `ggml/src/ggml-cuda/fattn.cu:702` `ggml_cuda_flash_attn_ext_get_alloc_size()` — the MMA arm sets
   `need_f16_K/V` (lines 735-736); this sizes the node's scratch via
   `ggml_cuda_flash_attn_ext_get_f16_extra_data()` (`fattn-common.cuh:59`), which appends
   `{K, V, end}` after `dst` with a 128-byte pad.
2. `fattn-common.cuh:1082+` `launch_fattn()` — when `need_f16_*` is set and the type is not F16, it runs
   `ggml_get_to_fp16_cuda()` / `_nc_cuda()` over the **whole cache** every call (lines 1144-1185) and
   then rewrites `K_data/V_data` + `nb11..nb23` to point at the scratch.
3. `fattn-mma-f16.cuh:440` `flash_attn_ext_f16_load_tile()` — stages tiles from that scratch into shared
   memory, either with `cp_async_cg_16` (the `use_cp_async` path, lines 455-500) or element-wise
   (`half2` loads, the `else` path).  Call sites: K at 772, K preload at 1468, V at 751, V at 1127
   (plus 1110 for K in the stream-k/fixup tail).
4. Bf16 therefore pays *both* the scratch (1-2) and the conversion pass (2) — while TILE users pay
   neither (block 03) and q8_0 users now pay neither (V4).

## 3. Design

### 3.1 The enabler

A 16-byte staged chunk is `4 × half2` = **8 f16 = 8 bf16** (both `half2` and `nv_bfloat162` are 4 B).
The shared tile layout, the chunking and the swizzles are therefore **byte-identical** for bf16 and f16
sources: the loader can `cp_async` the raw bf16 row into exactly the positions the F16 tile would
occupy, and then convert the tile **in place** — `__float2half(__bfloat162float(x))`, one rounding,
bit-identical to what `ggml_get_to_fp16_cuda(GGML_TYPE_BF16)` produces.

Swizzle note: `swz_K`/`swz_V` permute **16-byte units** and preserve element order inside a unit, so a
linear in-place pass over the shared tile converts exactly the staged elements.  Elements that
`cp_async` zero-filled for OOB rows/columns convert to +0.0 (no-ops); any tile padding that was never
written holds garbage and is never read (same as today).

### 3.2 Changes (5 sites, mirroring V4's shape)

| # | file | change |
|---|---|---|
| 1 | `fattn-common.cuh` (predicate block, ~L110-160) | add `ggml_cuda_fattn_kv_bf16_enabled()` (env `GGML_CUDA_FA_KV_BF16`, **default 1**) and `ggml_cuda_fattn_kv_bf16_supported(t)` = `type == GGML_TYPE_BF16` && `FAST_FP16_AVAILABLE` && `ne[0] % 8 == 0` && `nb[0] == ggml_type_size(BF16)` (the `% 8` is the 16-byte cp_async row constraint).  Generalise the V4 tag `fattn_kv_q8_t` to `fattn_kv_native_t { const char * K; const char * V; int stride_K; int stride_V; int type; }` (type: `NONE`/`Q8_0`/`BF16`) **or** add a parallel `fattn_kv_bf16_t` — one ABI, chosen by the same launcher logic |
| 2 | `fattn-common.cuh:1120-1130` (`launch_fattn`) | compute `use_bf16_K/V` from the predicate; size `f16_extra` with the *effective* need flags (as V4 does); pass the bf16 tag/flag down |
| 3 | `fattn-common.cuh:1144-1185` | skip the whole-cache conversion for a native-bf16 operand (the existing `!use_q8_*` conditions gain the bf16 arm) |
| 4 | `fattn.cu:735-736` | `need_f16_K = !use_q8_K && !use_bf16_K`, `need_f16_V = !use_q8_V && !use_bf16_V` (same `V_is_K_view` rule); the two places must ask the *same* predicates (the existing `GGML_ASSERT(f16_extra.K != 0)` is the tripwire) |
| 5 | `fattn-mma-f16.cuh` loader | element-wise path: convert bf16→f16 while loading (like V4's q8 path).  cp_async path: copy raw bf16 bytes to the same shared offsets, then **after `cp_async_wait_all()`** (call sites 749, 775, 1103, 1130) run a linear in-place conversion over the tile (`nbatch*stride_tile/2` `half2`-sized units) and `__syncthreads()` before the fragment loads.  The mask's cp_async (L627) is unaffected (the mask is already F16) |
| 6 | TILE / VEC | **nothing** — `fattn.cu:725-726` already sets `need_f16 = false` for bf16 in the TILE arm (`use_bf16`), and VEC needs F16 only for F32 sources |

### 3.3 Gate and ship rule

`GGML_CUDA_FA_KV_BF16` (kill-switch, **default 1**): unlike V4 (opt-in because it measurably loses the
`cp_async` pipeline), the bf16 arm keeps the pipeline and only adds a per-tile conversion pass, so the
expectation is ~neutral.  Three-way rule (same spirit as D9):

* prefill within noise (say ≤ 1 %) → **on by default**;
* a small measurable loss (≈ 1-2 %) with the full memory win → on by default is still defensible, but
  ask: the maintainer's D9 refinement says a sub-2 % loss with a large memory win and no cheap fix ships
  *opt-in* — so if the loss is real and not closable, default it to 0 and document;
* a large loss or any correctness doubt → leave the code in place with the gate off and report.

## 4. Validation protocol (the same five parts as V4 — do all of them)

1. **Reserve matrix** (ctx 204800): 4B (1 GPU), 27B (3-GPU Meta), gemma-4-E4B (1 GPU, ISWA),
   gemma-4-31B (3-GPU, ISWA), qwen4exp (3-GPU control) × ub 2048/1024/512 × `GGML_CUDA_FA_KV_BF16`
   off/on, **with f16 as the reference** (the target: bf16 == f16 at every row).
2. **Coherence, byte-identical** (same seed, `--temp 0`, generated text only): bf16 vs f16 and vs
   `GGML_CUDA_FA_KV_BF16=0`, on every model above, at a short and a 40k-token prompt.  Expect identical
   (the conversion is exactly the launcher's current conversion).
3. **Adaptive-MTP gate** (`benchmarks/mtp-adaptive-methodology.md`, protocol A): the 27B inline
   `--spec-type draft-mtp` probe and the qwen4exp draft probe, both gates — acceptance must be
   unchanged (today: 27B **0.76744**, qwen4exp **0.44262**).
4. **Op suites**: `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` (and CPU) with both gates; keep the
   whole-suite run (5.1k cases) as the backstop.
5. **Throughput, interleaved same-binary A/B** (llama-bench, `-fa on`, the `--output csv` parser in
   `wip/qwen4exp/qsa-memory/tools/` is not csv-quote-safe — use a real CSV reader):
   `-ctk/-ctv bf16` with the gate off vs on, pp20480/ub 2048 and tg256, on the 4B and the 27B; and the
   f16 reference for context.

Plus a **negative control**: `GGML_CUDA_FA_KV_BF16=0` must reproduce today's numbers exactly.

## 5. Risks / open questions

* **cp_async alignment**: the 16-byte chunk needs 16-byte-aligned globals.  The bf16 cache rows are
  `ne[0] × 2` bytes (256 → 512 B) and `nb[1]`/`nb[2]` are row-aligned, so this should hold — gate on
  `ne[0] % 8 == 0` and assert `nb[0] == 2`; verify on a *small* head dim too (the mixed/verify cases).
* **Element order inside a 16-byte unit**: verified by construction for the swizzles used here
  (`swizzle_b128`-style permutes whole units) — confirm by diffing a bf16 fragment against the
  converted-one reference (that is what `test-backend-ops` will do).
* **Sparse (`use_sparse`)**: `cp_async` is incompatible with the sparse gather, and on HIP
  `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse()` is false, so the sparse path does not arise.
  Keep the conversion guarded by the same `if constexpr (use_cp_async)` split.
* **`V_is_K_view`**: K and V may be the same tensor (both bf16) — one tag/flag, like V4.
* **Non-HIP**: `FAST_FP16_AVAILABLE` is defined only in the device pass on CUDA, so the host-side
  predicate cannot say "native" there → the arm is HIP-only (as V4 is).  Say so in the notes.
* **Interaction with V4**: independent arms (bf16 vs q8_0).  Note that a *mixed* bf16/q8_0 pair is
  broken for an unrelated reason (§6) — do not chase it here.
* If the conversion turns out to cost more than expected, the fallback is to convert during the
  fragment load (skip shared entirely) or to leave the arm opt-in; measure before redesigning.

## 6. Separate, pre-existing finding to record (not part of this work)

**Mixed K/V types are catastrophic today**: any pair with different types (`bf16`+`q8_0`, `f16`+`q8_0`,
both directions) drops the attention off the GPU kernel path — the reserve shows
`graph splits = 18` (vs 2 for a same-type pair), a ~1.5 GiB host compute buffer and no FA scratch, and
the throughput collapses:

| KV types (4B, pp2048/tg128, 1 GPU) | pp2048 | tg128 |
|---|---|---|
| `q8_0` / `q8_0` | 7924.47 | 98.94 |
| `bf16` / `q8_0` | 640.25 | 61.57 |
| `q8_0` / `bf16` | 1048.66 | 68.54 |
| `f16` / `q8_0` | 852.57 | 54.39 |

So `-ctk bf16 -ctv q8_0` (a natural "precision where it matters" choice) is **not** usable today; the
practical options are same-type K/V.  This is pre-existing (block 14 as well) and out of scope here —
document it and, if someone wants it, fix it as its own item (the FA kernels need to accept a mixed
`(type_K, type_V)` pair, which is a bigger change than either V4 or this plan).

## 7. Next-session prompt (copy-paste)

```
Implement bf16-native K/V in the MMA flash-attention path (the last campaign follow-up) in
/home/stew675/llama-cpp-rdna-boosts (read AGENTS.md first - its rules override everything here).

READ FIRST: wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md (the full plan: the measured before-state,
the mechanism with exact call sites, the design, the code map, the validation protocol, the ship rule and
the risks), then beta/block-15-campaign-wins/HANDOVER.md section 3.4 (the scope decision D10) and
wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md (V4 is the template this follows - its implementation
table is the shape to copy).

GOAL: a bf16 KV cache must pay no F16 staging scratch and no per-ubatch conversion pass in prefill,
with the cp_async pipeline kept.  Measured today (delivered block-15 tree, ctx 204800, f16 reference):
4B ub 2048 256.86 -> 968.86 MiB (+712), ub 1024 +756, ub 512 +778; 27B ub 2048 +584, ub 512 +746;
ub 8 (TILE) identical.  Target: bf16 == f16 at every row.

PLAN (detail and code map in the plan file):
1. add a native-bf16 predicate + env gate GGML_CUDA_FA_KV_BF16 (default 1) next to V4's
   ggml_cuda_fattn_kv_native_supported in fattn-common.cuh, and generalise V4's kv tag;
2. skip the whole-cache F16 conversion for a native-bf16 operand in launch_fattn and size f16_extra with
   the effective need flags; make fattn.cu's get_alloc_size ask the same predicates (the existing
   GGML_ASSERT(f16_extra.K != 0) is the tripwire if they disagree);
3. in fattn-mma-f16.cuh: element-wise path converts bf16->f16 while loading; cp_async path copies the raw
   bf16 bytes to the same shared offsets (a 16-byte chunk is 8 elements either way) and converts the
   tile in place after cp_async_wait_all() + __syncthreads() (a linear pass is enough: the swizzles
   permute whole 16-byte units);
4. TILE/VEC need nothing (block 03 already reads bf16 natively).

VALIDATE (all five, on the delivered tree; keep the tree buildable and snapshot the diff as its own
patch before moving on): the reserve matrix (4B 1-GPU, 27B 3-GPU, gemma-4-E4B 1-GPU ISWA,
gemma-4-31B, qwen4exp control; ub 2048/1024/512; gate off/on; f16 reference) - expect bf16 == f16;
byte-identical same-seed coherence (bf16 vs f16 vs gate off, short + 40k prompts); the MTP gate
(27B 0.76744 and qwen4exp 0.44262 unchanged); test-backend-ops FLASH_ATTN_EXT on ROCm0 + CPU; and an
interleaved same-binary prefill/decode A/B (pp20480 ub 2048 + tg256, 4B and 27B) to decide the default
per the plan's three-way ship rule (expectation: on by default, since the cp_async pipeline is kept).

Then: fold the result into the delivery as a dated block-15 amendment (patches/0015 regenerated from a
CANONICAL fork rebuilt at 9113cc188 via scripts/apply-all.sh - never from the working checkout's
rdna-boosts tip, which sits two upstream commits past the fork point), update patches/README.md +
WORKLOG.md + the beta record, re-run the clean-apply simulation, and stage the beta patch copy.  If the
measurement says opt-in instead, say so up front - do not ship a regression on by default.

ALSO RECORD (do not fix): mixed K/V types (bf16+q8_0, f16+q8_0, ...) fall off the GPU attention path
today - graph splits 18 vs 2, ~1.5 GiB host buffer, pp2048 7924 -> 640-1049 t/s.  It is pre-existing,
out of scope for this work, and already documented in the plan file section 6.

DO NOT: rebuild the WMMA kernels with bf16 fragments (D10); push anything from ~/llama.cpp; fold wip/
content into patches/ beyond this agreed work; or touch archive/work/.  Keep 3 GPUs sequential, one job
at a time, and check for stray llama processes before measuring.
```

## 8. Deliverable state the next session starts from

* fork `~/llama.cpp`: `rdna-boosts` = `8ee104f33` (block 15 on a master 2 commits past the fork point —
  **never regenerate from it**); the canonical chain is the local branch **`block15-canonical`** =
  `09a137566` (rebuilt at `9113cc188`); clean tree.
* delivery repo: `patches/` = 15 patches (block 15 = the campaign wins), `upstream/` = 4 candidates,
  `beta/block-15-campaign-wins/` = the beta record; working tree clean.
* build: `export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH && cmake --build build-rocm --target
  llama-cli llama-bench test-backend-ops -j 16`; run with
  `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib HIP_VISIBLE_DEVICES=0[,1,2]`.
* probes used for the tables above: `/tmp/b15kv.sh <tag> <4b|27b> <ub> <ctk> <ctv>` (reserve with
  arbitrary KV types, no `GGML_CUDA_FA_WMMA_256` override) and `/tmp/b15bench.sh` (interleaved
  llama-bench A/B, with `/tmp/b15parse.py` as the CSV parser).
