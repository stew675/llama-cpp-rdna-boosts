# HANDOVER — general-purpose `mmb` (bf16/i8-WMMA dequant weight GEMM) + QSA/Q8_0 next steps

**Date:** 2026-09-19.  **Status:** ACTIVE WIP, not part of the delivery, **not** pushed to any fork.
This document is the self-contained entry point for the next session.  Read it top to bottom first,
then `README.md` (the running record) beside it.

> **One-line summary.** A general prefill weight-GEMM on the tensor cores (dequant-to-bf16 → WMMA)
> is now implemented for **every** weight type the delivery's models use, validated PPL-parity, and
> measured at up to **+68 % pp2048 / +56 % pp8192** on the Flash-Next Q4_K_M.  **QSA v3 (packed-block
> WMMA sparse attention) was then delivered in session 2**: the QSA attention kernel went **2944 ms ->
> 1152 ms (2.56x)** and prefill +5-10 % on every KV type, PPL-parity.  The next levers are the
> `qsa3_rows_kernel` sort (441 ms) and the now-dominant `mmb_*` kernels.

---

## UPDATE — session 2 (2026-09-19): QSA v3 is DONE

**Read §8 as history; the work is landed.**  Everything below is still accurate for the MMB part.

Session 2 delivered the packed-block WMMA QSA prefill (§8) and extended it to every KV cache type:

* New file `ggml/src/ggml-cuda/fattn-qsa3.cu` (rows / merge / attn kernels, ported from the Strix
  Halo branch's `qsa-attn`).
* New optional op srcs `src[7]`/`src[8]` via `ggml_flash_attn_qsa_set_packed`; the pack itself is a
  pure graph composition (no new ggml op), built in `src/models/qwen4exp.cpp`.
* Gate: **`GGML_CUDA_QSA3=1`**, default OFF.  Prefill-only (`q->ne[1] >= 128`) and RDNA3_5, so the
  W = 1..8 decode/verify band still takes the VEC kernel and width purity holds by construction.
* gfx1201 + gfx1100 TU compiles verified (portable WMMA wrapper, no-op on RDNA4).

Measured (gfx1151, IQ4_XS, `-b/-ub 2048`): pp8192 **f16 819.4->861.7, bf16 791.2->854.5, q8_0
780.5->856.6**; PPL parity at c16384/c32768 on all three (e.g. bf16 3.3932 vs 3.3900); greedy text
coherent.  Kernel profile pp8192: **VEC 2944.4 ms -> qsa3 1151.9 ms** (attn 682.3 + rows 441.0 +
merge 28.6).

Three things session 3 must not re-derive - full detail in `README.md`:

1. **Do not remove the dense startup arm.**  `n_kv <= 2051` takes the dense path because the
   selection covers every cell there; forcing QSA there is measurably worse (pp2048 912.5 -> 903.2).
   Consequence: **qsa3 only engages above 2051 tokens**, so `-c 2048` tests silently prove nothing.
2. **The top-k rows are unsorted**, so the rank-sort in `qsa3_rows_kernel` is required.  Disabling it
   drops that kernel 441 -> 8 ms but blows the attn kernel up 682 -> 19593 ms and halves throughput.
3. **Cast on the natural contiguous cache view, never on a permuted one**, and route quantized types
   through F32 - the backend `dup` cannot permute a quantized tensor and only dequantizes to F32
   (otherwise: `ggml/src/ggml-cpu/ops.cpp:578` abort, once per QSA layer).

**Revised next-work order** (after session 2):

1. `qsa3_rows_kernel` sort: 441 ms of the 1152 ms qsa3 total (~4 % of prefill).  A block-aware sort
   (sort the ~ns/4 block ids, then expand) is ~16x less work.  **Do not** reorder `idx` at the
   indexer: that tensor is shared with the VEC decode path, so it would be a decode numerics change.
2. The `mmb_*` kernels now lead the profile (`mmb_f32split` 2164 ms, `mmb_routed_glu` 2085+1626,
   `mmb_dense` 1904+1606, `mmb_cvt_f32_bf16` 648) - see §9/§10 for the Q8_0 IU8 and bf16-producer ideas.
3. QSA v3 for **gfx1200/gfx1100** (needs an RDNA4/RDNA3_0 WMMA variant; the wrapper is currently a
   deliberate no-op there, so only the VEC path runs).

---

## 0. TL;DR for the next session

1. Everything lives in the worktree `~/llama-wip-mmb`, branch **`wip-mmb-general`**, based on the
   maintainer's applied delivery tree at **`8a2567e1e`**.
2. The complete change is backed up here as
   **`wip/mmb-general/mmb-general.patch`** (combined, 3 files, +1451/−2) and
   **`wip/mmb-general/patches/0001..0009-*.patch`** (per-commit), plus `commits.txt`.
   Verified: `git apply --check` passes on a clean checkout of `8a2567e1e`.
3. Rebuild with the commands in §3; reproduce the numbers in §5.
4. **QSA v3 is DONE** (see the UPDATE section) — the remaining QSA work is the 441 ms sort in
   `qsa3_rows_kernel`.  The optional second lever is now **Q8_0 IU8-WMMA** (§9).

---

## 1. Why (context in one paragraph)

On Strix Halo (gfx1151) the delivery's prefill was ~2x behind the tuned RDNA3_5 stacks
(pwilkin `strix-halo` full env: 1349/1403 t/s on uniform IQ4_NL; halogen-flash 1246/1424 t/s on the
same llama.cpp GGUFs).  Root cause: our weight GEMMs run the **integer/vector MMQ** path while both
use **bf16 WMMA tensor cores** (roof ≈ 59 vs 29.7 TFLOPS on this part; prefill is compute-bound).
The parked port (`archive/work/wip-archive/iq4nl-prefill/`) proved the idea but was IQ4_NL-only and
was parked on the purity clash.  This session generalised it to the types the delivery's own models
actually use and made it purity-safe.  **GFX1201 note:** the kernels use the first-gen gfx11 WMMA
builtin, which gfx12 does not have — hence the RDNA3 gate and the portable wrapper (§6).

---

## 2. Where the code is / what changed

Only **3 files** change:

| file | change |
|---|---|
| `ggml/src/ggml-cuda/mmb.cu` | new (+1381): the dequant row helpers, tile GEMM kernels, routed/GLU kernels, dispatch, predicates, shadow helpers |
| `ggml/src/ggml-cuda/mmb.cuh` | new (+33): the exported API + `ggml_cuda_mmb_{dense,routed}_will_take` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | ±39: MMB dispatch hooks in `ggml_cuda_mul_mat` / `_mul_mat_id` / `_glu`, and the **per-weight-type fusion stand-down** in the graph optimizer |

`mmb.cu` is picked up automatically by the existing `file(GLOB … "*.cu")` in
`ggml/src/ggml-cuda/CMakeLists.txt` (no CMake edit).

### WTYPE map (used throughout the file)

| WTYPE | type | bytes per k-step (64 values) | row block |
|---|---|---|---|
| 0 | IQ4_NL | 36 | 18 B / 32 |
| 1 | Q8_0 | 68 | 34 B / 32 |
| 2 | BF16 (direct, no dequant) | 128 | 2 B / value |
| 3 | Q4_K | 144 (16 hdr + 128) | 144 B / 256 |
| 4 | Q5_1 | 48 | 24 B / 32 |
| 5 | IQ3_S | 110 | 110 B / 256 |
| 6 | Q5_K | 176 | 176 B / 256 |
| 7 | Q6_K | 210 | 210 B / 256 |
| 8 | IQ4_XS | 136 | 136 B / 256 |
| 9 | Q3_K | 110 | 110 B / 256 |
| 10 | IQ3_XXS | 98 | 98 B / 256 |

### Key design decisions (do not undo)

* **Prefill-only.**  `ggml_cuda_mmb_supported_*` require `T >= GGML_CUDA_MMB_MIN_T` (default 512).
  The decode/spec-verify band is `W = 1..8`, so MMB is unreachable there — `W=1..8` bit-identity and
  `plain == draft-mtp` hold **by construction**, not by measurement.
* **Per-weight-type fusion stand-down.**  The graph optimizer's MoE pair and SWIGLU→mmq fusions
  stand down only when MMB will actually take *that weight type*
  (`ggml_cuda_mmb_dense_will_take` / `_routed_will_take`), never on the global gate.  This is the
  parked handover's resume-checklist item #5 and it is what stopped MMB-on from regressing models
  whose expert type it could not accelerate.
* **RDNA3_5 only by default.**  `mmb_enabled()` gates to `GGML_CUDA_CC_IS_RDNA3_5`; RDNA3_0
  (gfx1100/1101/1102) shares the gfx11 WMMA builtin but is untested, so it needs
  `GGML_CUDA_MMB_RDNA3=1`.  gfx12 is excluded (different builtin).
* **Portable WMMA wrappers.**  `mmb_wmma_bf16` / `mmb_wmma_f16` select the gfx11 builtin on
  non-`RDNA4` and are a **deliberate no-op** on `RDNA4`, so `mmb.cu` *compiles* for gfx1201 while
  never being launched there.  (Verified: `mmb.cu` and `fattn-qsa.cu` both compile for gfx1201 — §7.)
* **On-the-fly dequant, no shadow.**  A persistent bf16 shadow is impossible for a 120 GiB model's
  experts, so every non-trivial type is dequantized per k-step.  The legacy Q6_K shadow path still
  exists but is no longer needed for Q6_K (WTYPE 7 is on-the-fly).
* **IQ3_XXS fused GLU is default-off.**  Its routed path is a win, its fused GLU is a measured net
  loss (§5); `GGML_CUDA_MMB_IQ3XXS=1` re-enables the GLU arm for tuning.

---

## 3. Build / run

The build dir in the worktree is already configured; to rebuild:

```sh
cd ~/llama-wip-mmb
export ROCM_PATH=/opt/rocm-7.14-gfx1151
HIPCXX=$ROCM_PATH/lib/llvm/bin/clang HIP_PATH=$ROCM_PATH cmake -S . -B build-rocm \
  -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 -DHIP_PLATFORM=amd \
  -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_PATH/lib" \
  -DCMAKE_C_COMPILER=$ROCM_PATH/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS= \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-rocm -j 16 --target llama-bench llama-perplexity llama-cli llama-server
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
```

To recreate from scratch (the backup is authoritative):

```sh
git -C ~/llama.cpp worktree add --detach /tmp/mmb-rebuild 8a2567e1e
cd /tmp/mmb-rebuild
git apply /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/mmb-general.patch
# then the cmake configure+build above with -B build-rocm
```

**Caveat:** the WIP base is `8a2567e1e` (the maintainer's applied tree), **not** the current canonical
release tip (r9 = `76b10f1fb8391562e30364d6c307e6606100bc57`).  Before landing, the change must be
rebased onto a canonical fork rebuilt at the release base `ebbb18522` + `scripts/apply-all.sh`, then
the delivery `patches/` regenerated (see `AGENTS.md`).  None of the touched hunks are in code that
r9 changed (the r9 work was the block-15 tile kq-mask), so the rebase should be mechanical.

---

## 4. A/B and PPL commands used

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
M=<model>.gguf

# prefill A/B (MMB off vs on)
./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 \
    -p 2048,8192 -n 0 -d 0 -r 2
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 \
    -b 2048 -ub 2048 -p 2048,8192 -n 0 -d 0 -r 2

# PPL parity (the correctness gate for each new dequant)
F=~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt
./build-rocm/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -f $F -c 2048 -b 2048 -ub 2048
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -f $F -c 2048 -b 2048 -ub 2048
```

Always **warm the model into page cache first** (`dd if=$M of=/dev/null bs=4M`); the box's page-cache
state swings prefill noticeably, and never run benches in parallel.

`GGML_CUDA_MMB=1` prints a few one-shot `MMB_GLU …` / `MMB_SHADOW …` lines — that is normal.

---

## 5. Measured results (gfx1151, ROCm 7.14, `-b/-ub 2048`, bf16 KV)

| model / test | MMB off | MMB on | PPL off → on |
|---|---:|---:|---|
| **Q4_K_M** pp2048 | 604.6 | **1015.2 (+68 %)** | — |
| **Q4_K_M** pp8192 | 568.9 | **888.1 (+56 %)** | 10.3328 → 10.2716 |
| IQ4_XS (Flash-Next) pp2048 | 723.3 | 838.0 (+15.9 %) | — |
| IQ4_XS pp8192 | 685.5 | 779.5 (+13.7 %) | 10.6938 → 10.6440 |
| Gemma4-26B-A4B Q8_0 pp2048 | 2058.5 | 2340.1 (+13.7 %) | — |
| Gemma4-26B-A4B Q8_0 pp8192 | 1723.5 | 1928.5 (+11.9 %) | parity (prose prompt is tokenizer-mismatched; both arms same ballpark) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp2048 | 2045.3 | 2303.0 (+12.6 %) | — |
| Qwen3.6-35B-A3B UD-Q5_K_M pp8192 | 1917.8 | 2141.3 (+11.7 %) | 14.4349 → 14.3907 |
| Qwen3.6-35B-A3B Q6_K pp2048 | 2128.8 | 2143.2 (+0.7 %) | — |
| Qwen3.6-35B-A3B Q6_K pp8192 | 1974.6 | 2012.3 (+1.9 %) | 14.3682 → 14.3296 |
| Qwen3.6-35B-A3B UD-Q3_K_M pp2048 | 2091.9 | 2295.2 (+9.7 %) | — |
| Qwen3.6-35B-A3B UD-Q3_K_M pp8192 | 1941.6 | 2120.2 (+9.2 %) | 14.6517 → 14.6248 |

**Interpretation**

* The Q4_K_M jump is its **Q4_K MoE expert GLU + routed down** (WTYPE 3); IQ4_XS is IQ3_S gate/up +
  IQ4_NL down; Q5_K_M/Gemma4 are their expert types.
* **Q6_K is correct but barely moves** (+2 %): its MMQ path is already efficient on this workload.
  Keep it for coverage, not as a headline win.
* **IQ3_XXS finding:** routed is a win, fused GLU is a loss.  `UD-Q3_K_M` full-on 1921 vs routed-only
  2120 vs off 1942 t/s pp8192 → the fused GLU is default-off (`GGML_CUDA_MMB_IQ3XXS=1` re-enables).
* **Model composition trap:** file names mislead.  *Flash-Next UD-IQ4_XS* is IQ4_NL 52 % + IQ3_S 36 %
  + Q8_0 9.5 % (IQ4_XS type only 1 %).  *MiniMax-M3 UD-IQ4_XS* is IQ3_S 56 % + IQ4_XS 35 % → now
  fully covered.  *UD-Q3_K_M* is IQ3_XXS 47 % + IQ4_XS 33 %.

### Coverage matrix (final)

| type | dense | routed + GLU | note |
|---|:--:|:--:|---|
| IQ4_NL, Q8_0, Q4_K, Q5_1, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K | ✓ | ✓ | |
| IQ3_XXS | ✓ | routed only | fused GLU default-off (net loss) |
| F16 / BF16 | — | — | already tensor-core via `mmf` WMMA |
| F32 | ✓ (f32split) | — | f16 hi/lo WMMA |
| Q2_K, IQ2_*, IQ1_* | — | — | deliberately out of scope |
| IQ3_M, IQ3_XS, IQ4_S, IQ4_M | — | — | **do not exist in current ggml** |

Not yet covered (optional): the legacy **Q4_0 / Q4_1 / Q5_0** (32-value blocks; Q5_1 is already done,
so these are small additions).

---

## 6. Environment knobs (all in `mmb.cu`)

| env | default | meaning |
|---|---|---|
| `GGML_CUDA_MMB` | 0 | master gate (default **off**) |
| `GGML_CUDA_MMB_MIN_T` | 512 | prefill-only threshold |
| `GGML_CUDA_MMB_RDNA3` | 0 | allow RDNA3_0 (untested) |
| `GGML_CUDA_MMB_GLU` | 1 | fused gate/up+swiglu arm |
| `GGML_CUDA_MMB_IQ3XXS` | 0 | enable the (net-loss) fused GLU arm for IQ3_XXS |
| `GGML_CUDA_MMB_BF16W` | 1 | BF16 dense weights via MMB |
| `GGML_CUDA_MMB_F32SPLIT` | 2 | F32 dense via f16-hi/lo WMMA |
| `GGML_CUDA_MMB_TALL` | 2 | the tall-M tile class |
| `GGML_CUDA_MMB_SHADOW` / `_SHADOW_MB` | 0 / 6144 | legacy bf16 shadow (Q6_K/IQ4_NL); not needed now |
| `GGML_CUDA_MMB_TILE` | -1 | **diagnostic**: force narrow(0)/wide(1) tile |
| `GGML_CUDA_MMB_LOG` | 0 | **diagnostic**: one-shot `MMB_DENSE` shape log |

---

## 7. Post-MMB profile + where the time now goes

Q4_K_M pp8192, `GGML_CUDA_MMB=1`, rocprofv3 kernel trace (**total 17.75 s**, was ~28 s off):

| kernel | time | share |
|---|---:|---:|
| `flash_attn_qsa` | 2.94 s | **16.6 %** |
| `mmb_dense_kernel` (Q8_0 PLE + Q5_1) | 4.06 s | 23 % |
| `mmb_routed_glu` (Q4_K gate/up) | 2.46 s | 14 % |
| `mmb_routed` (Q4_K down) | 1.39 s | 7.8 % |
| `dsv4_hc_pre` + `_post` | 1.44 s | 8.1 % |
| `rms_norm_f32<1024>` | 0.65 s | 3.6 % |
| `gdn_bf16_scan` | 0.62 s | 3.5 % |
| `mmb_f32split` | 0.65 s | 3.7 % |
| `mmb_cvt_f32_bf16` | 0.65 s | 3.7 % |

**Both leading items were probed and are not tuning-reachable** (see `README.md` §investigation):

* **Q8_0 tiling is optimal** — forcing narrow/wide changes pp8192 by +0.6 % / −2.4 %
  (802 / 798 / 778 t/s) → the kernel is occupancy/LDS-bound.
* **QSA gather 8B→16B is neutral** (796 vs 798) → not load-issue-bound; it is
  compute/reduction/occupancy-bound.

---

## 8. QSA v3 — **DONE in session 2** (kept as the design record)

**Status: landed behind `GGML_CUDA_QSA3=1`; results and the remaining optimization are in the
"UPDATE" section above and in `README.md`.  The plan below is what was actually built - keep it for
the gfx1200/gfx1100 follow-up.**

**Goal (achieved):** replace, for prefill only, the VEC `flash_attn_qsa` score/PV with a packed-block
WMMA implementation (pwilkin's `qsa3`).  Result: 2944 ms -> 1152 ms on the kernel, +5-10 % prefill.

**Approved:** the maintainer accepts a prefill re-baseline (greedy prefill output changes) as long as
it is **consistent, deterministic and coherent**.  Keep the **VEC `flash_attn_qsa` for the W=1..8
band** (gate the new path `n_query >= 128`), so width purity and `plain == draft-mtp` are untouched
by construction.  Document the re-baseline with a new same-seed hash.

**Reference:** `~/pwilkin-llama-cpp` (branch `strix-halo`), `ggml/src/ggml-cuda/qsa.cu` (445 lines):
`qsa3_rows_kernel`, `qsa3_merge_kernel`, `qsa3_attn_kernel`, plus the graph's `qsa_pack_keys` /
`qsa_pack_values` tensors (his `src[6]`/`src[7]`).

**Concrete steps**

1. **Pack ops (graph side).**  Build contiguous f16 key/value block buffers once per graph from the
   cache + the selected indices.  In our tree that is `src/models/qwen4exp.cpp` +
   `GGML_OP_FLASH_ATTN_QSA` (see `build_qsa_top_k`, `build_qsa_store_k`) and
   `ggml/src/ggml-cuda/fattn-qsa.cu`.  Decide whether to add the buffers as extra op `src[]` or to
   build them inside the op's launcher (the latter avoids graph/allocator changes but re-packs per
   call).  The handover for the parked port (`archive/work/wip-archive/iq4nl-prefill/`) and
   `wip/prefill-arrangements/README.md` both scope this.
2. **Descriptor.**  Port `qsa3_rows_kernel` + `qsa3_merge_kernel`: merge `G = 4` consecutive queries'
   top-k lists into a sorted, deduplicated, block-aligned array of block ids + a 16-bit per-query
   membership mask + count (his `ublk`/`umask`/`ucount`).
3. **Attention.**  Port `qsa3_attn_kernel` (16×16×16 f16 WMMA over the packed blocks, membership
   mask folded into the score pass, online softmax as now).  Reuse the portable-WMMA pattern from
   `mmb.cu` so gfx1201 still compiles (or keep it gfx11-only with stubs, matching
   `gated_delta_net_chunked_bf16_gfx11.cu`).
4. **Gate + validate.**
   * prefill-only (`n_query >= 128`); VEC kernel unchanged in the band;
   * `test-backend-ops -o FLASH_ATTN_QSA` must stay green (add cases as needed);
   * the `W = 1..8` logits matrix on `LLAMA_QSA_SPARSE_FA` on/off/new must keep the band identical;
   * same-seed greedy text: new build vs old, once, to record the prefill re-baseline hash;
   * PPL vs the dense masked oracle (`LLAMA_QSA_SPARSE_FA=0`) within noise — **not** MTP acceptance,
     which is blind to a defect present in both draft and target (`GREEDY-PURITY.md` §21).

**Size estimate:** the scoping doc priced this at 6-10 days.  It is the single largest remaining item.

---

## 9. NEXT WORK #2 — Q8_0 IU8-WMMA dense path

**Goal:** replace the dequant-to-bf16 MMB Q8_0 dense path (4.06 s, 23 %) with an **int8 → int8 tensor
core** GEMM (`v_wmma_i32_16x16x16_iu8`), avoiding the LDS dequant staging that makes the current
kernel occupancy-bound.

**Facts already established**

* The big Q8_0 shapes: `attn_qkv M=10240 K=2560`, `attn_gate M=6144 K=2560`, `ssm_out M=2560 K=6144`,
  `hc_* M=320/10240`, `ffn_*_shexp M=640 K=2560`.
* The tile heuristic is optimal; the kernel is **occupancy/LDS-bound**, not tiling-bound.
* `v_wmma_i32_16x16x16_iu8` measured ~171-175 T-MAC/s on gfx1201 (see
  `wip/q8-prefill-tuning/README.md`); the same doc notes the mainline MMQ Q8_0 kernel reaches only
  31-34 % of the int8-WMMA ceiling and its k-loop has no double-buffering.

**Steps:** quantize the activations to int8 (Q8_1-style scales) — note this is a numerics change too,
so the same prefill-only gate applies — then a WMMA-iu8 tile kernel with int32 accumulators; A/B vs
the current bf16 Q8_0 MMB path on IQ4_XS/Q4_K_M.

---

## 10. Smaller remaining items

* **bf16-producer marking** — kill `mmb_cvt_f32_bf16` (0.65 s, 3.7 %): the parked port has the
  plumbing (`ggml_cuda_mmb_mark_bf16_only`, `ggml_cuda_mmb_cache_reserve`,
  `ggml_cuda_mmb_cache_lookup`) but nothing ever calls it.  Needs a graph-optimizer pass that marks a
  producer's output bf16-only when every consumer is MMB.  Prefill-only.
* **HC prefill fusion** — `dsv4_hc_pre`+`_post` are 8.1 %; pwilkin's `hc_combine_norm`/`hc_mix_reduce`
  are ~0.93 s vs our 1.44 s.  Block 14 already has the decode-band fused hc ops; the prefill arm is
  the gap.
* **Legacy Q4_0 / Q4_1 / Q5_0** — trivial next to what is done (32-value blocks).
* **Optional 7900 XTX test** (gfx1100, single card, small model) with `GGML_CUDA_MMB_RDNA3=1` — the
  tiling almost certainly needs an RDNA3_0 pass.  The maintainer wants gfx1151 exhausted first.

---

## 11. Purity / landing gates (unchanged from the parked handover)

Before this can be opt-in, let alone defaulted on:

1. `W = 1..8` logits matrix with `GGML_CUDA_MMB=1` == off (should be identical **by construction** —
   `T >= 512` — but must be run).
2. MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
3. `test-recurrent-state-rollback`.
4. `test-backend-ops` suites (FLASH_ATTN_QSA / GATED_DELTA_NET / FLASH_ATTN_EXT).
5. Same-seed **prefill re-baseline** documented with a fresh hash.
6. gfx1100 / gfx1201 compile + consistency (gfx1201 TU compile already verified — §12).
7. Default stays **off**; the env kill-switch (`GGML_CUDA_MMB=0`) is retained.

The repo rulebook is `AGENTS.md` + `GREEDY-PURITY.md`; landing must regenerate `patches/` from a
canonical fork rebuilt at the release base (`scripts/make-patches.sh` / `make-release.sh`), never from
the drifted working tree.

---

## 12. Verification already done this session

* Combined patch `git apply --check` clean on a fresh `8a2567e1e` worktree.
* gfx1151 build clean after every commit; `llama-bench` / `llama-perplexity` / `llama-cli` /
  `llama-server` all build.
* **gfx1201 TU compiles verified** for `mmb.cu` **and** `fattn-qsa.cu` (extract the command from
  `build-rocm/compile_commands.json`, swap `--offload-arch=gfx1151` → `gfx1201`).
* PPL parity recorded for: Q4_K_M, IQ4_XS, UD-Q5_K_M, Q6_K, UD-Q3_K_M (table in §5).

---

## 13. Gotchas the next session should not re-learn

* `llama-cli` **must** be run with `--single-turn` (and usually `--no-display-progress`) or it blocks.
* Warm the page cache before every bench; never run benches in parallel.
* `llama-bench` `-b/-ub 16384` on the 94 GiB IQ4_XS can OOM at pp16384 (context creation fails); use
  `-ub 2048` (the server's config) for these models.
* The `-lm none -lzm on` flags are needed for the Q4_K_M/lazy models in `llama-bench`.
* `mmb_dq_row_*` for the K-/i-quants are loaded from the row base inside `store_lds` (the fields are
  non-contiguous), so their `load_regs` branch is intentionally a no-op — do not "optimise" it back
  into the register prefetch.
* A `case`/`if` added to the predicates must be added to **all** of:
  `supported_mm`, `supported_mmid`, `supported_glu`, `dense_will_take`, `routed_will_take`, **and**
  the three dispatch helpers — a missing one is either a silent fallback or an uninstantiated launch.
* `mmb.cu` uses `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (gfx11); gfx12 needs
  `…_gfx12` with different fragment types.  The wrapper is a no-op on RDNA4 on purpose.

---

## 14. Repo conventions

* This WIP lives under `wip/` in `llama-cpp-rdna-boosts` (the only push target:
  `github.com:stew675/llama-cpp-rdna-boosts`).  **Never** push anything out of `~/llama.cpp`.
* Commit records/doc updates to the delivery repo; keep the experimental code in the worktree and
  regenerate `mmb-general.patch` from it whenever the code changes.
* When the work is promoted, it becomes a block amendment (the parked handover suggests block 08 for
  the MMB module) and follows the promotion rule in `AGENTS.md`.
