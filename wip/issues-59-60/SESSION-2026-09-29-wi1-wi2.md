# Session 2026-09-29 (WI-1 + WI-2) — r21 re-integration and the geometry-aware prefill score

Host `gfx1100` (one RX 7900 XTX, ROCm 7.14, `~/bin/build-llama-rocm-714`).  Base: `main` = r21
(`feefecfb…`, tree `9975a333…`).  No qwen4exp model on this box, so the reporter's 196K
`llama-bench` gate is still outstanding (see §4).

---

## 1. WI-1 — re-integrate on r21

Rebuilt the canonical chain from a fresh local clone at `84e76d8a2` + the r21 `patches/`
(`scripts/apply-all.sh`, strict 16/16, tree `9975a333…`), applied `fix-r20-full.patch`, and folded
it into **block 15** (the tip) with `git commit --amend`:

* The 8-file patch applied with **no conflicts** on r21 — the r21 GQA-6 band touches
  `fattn-common.cuh`/`fattn-mma-f16.cuh` but not the `launch_fattn` staging block the fix edits, so
  the "expect conflicts" note in `FINDINGS-2026-09-29.md` did not materialise.  `git apply --check`
  was clean and the amended tip compiles on gfx1100.
* Fork tip **`0a31421cf`**, tree **`addc735ee541ac9410f19f0fa47f374972fd1775`**.
* Regenerated with `scripts/make-patches.sh ~/llama-r21 84e76d8a2 0a31421cf` and
  `scripts/make-release.sh --tip … --tree … --release v16-84e76d8a2-wip-issue59-60`.
* `scripts/validate-set.sh`: checksums OK, strict **16/16** `git am`, applied tree ==
  `addc735e…`.  No tag, nothing pushed.
* The block-15 commit message now carries the `Issue #59/#60 amendment` note (qsa3 gate, the
  geometry-aware score policy, and the staging-arena accounting).

## 2. WI-2 — root cause and the chosen fix

**Root cause (as the reporter measured it).**  The failing `wip default` is the *unfused*
`mul_mat + relu + head_sum` prefill score (the #59(b) gate made the fused 4-head
`LIGHTNING_INDEXER` unsupported on RDNA3_0).  At deep context the unfused score is
`score_blocks × n_idx_h × n_query × n_stream × 4` F32 bytes, assembled in `chunk`-block slices and
`ggml_concat`-ed (`build_qsa_top_k`, the `score_mem && score_bytes > 128 MiB` branch).  The chunking
bounds the per-slice intermediate, but the accumulated `acc` concat must coexist with the next
`acc`, so the unfused score still reserves more than the fused op's single
`[score_blocks, n_query, n_stream]` F32 output.  At 196K (`score_bytes ≈ 200 MiB`) that tips the
graph reserve from 425 to 430.45 MiB and crosses the ~480 MiB headroom; the fused path fits
(the reporter's `…_INDEXER4_GFX1100=1` PASS at 869 t/s).

**Why not "just shrink it".**  The preferred shrink would have to remove the `acc`-doubling in the
chunked concat (the chunk itself is already ≤ 4096 blocks).  Replacing the concat with an in-place
assembly was rejected: the existing comment records that `ggml_cpy` into a view of an
allocator-owned tensor leaks, which is exactly why `ggml_concat` is used.  A smaller chunk does not
remove the doubling.  (A cheaper, bit-identical shrink still worth measuring on real qwen4exp is
`ggml_relu_inplace` for the `mul_mat` result in the chain — it removes one `[chunk, n_idx_h×n_query]`
F32 buffer, ~16.8 MiB at 196K, and the reporter's deficit is only ~5.45 MiB — but it cannot be
validated here, and the geometry policy below already bounds the reserve.)  The deterministic and
low-risk fix is therefore the second option the handover named, and it also restores a dead knob.

**Implemented (issue #60):**
* `ggml/src/ggml-cuda/lightning-indexer.cu`: removed the #59(b) gate — `ggml_cuda_lightning_indexer_supported`
  is a pure capability query again (`supports_indexer4 || generic`).  The generic 4-head vec kernel
  was already there and works (it was the default on gfx1100 before r20's gate).
* `src/models/qwen4exp.cpp`:
  * `LLAMA_QSA_SCORE_WMMA` is now tri-state: unset = **auto**, `=0` force chain, `=1` force fused.
  * `use_wmma` computes `score_bytes` first, reads the arch once (`qsa_arch_gfx()`, the existing
    helper; `0x1100..0x114f` = RDNA3_0) and auto-selects:
    `!rdna3_0 || score_bytes > 128 MiB`.  So on RDNA3_0 the chain keeps shallow/medium prefill
    (its measured ~5.4 % win) and the fused op takes over exactly where the chain already switches
    to its chunked-memory assembly; RDNA3_5/RDNA4 stay fused everywhere.  Because the reserve is
    the max over the run and the chain is only used up to the 128 MiB score size, the reserve is
    bounded by the (smaller) chain peak below the threshold.
  * `lightning_indexer_op_supported()` stays as the backend capability probe; its doc no longer
    claims an RDNA3_0 arch policy.

The `128 MiB` threshold is deliberately the *same* constant the unfused chain already uses to enter
its chunked-memory branch, so no new graph-structure boundary is introduced.

## 3. gfx1100 verification of this build

| check | result |
|---|---|
| build (`~/bin/build-llama-rocm-714`, gfx1100) | clean, 8.7 s incremental (ccache) |
| `test-backend-ops -o FLASH_ATTN_QSA` | **23/23** |
| `test-backend-ops -o LIGHTNING_INDEXER` (default) | **225/225** (was 144/144 with the 81 `nh=4` cases `not supported`; the generic 4-head path now runs) |
| `test-backend-ops -o LIGHTNING_INDEXER` + `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1` | **225/225** |
| `test-backend-ops -o TOPK_QSA` | **4/4** |
| `test-backend-ops -o FLASH_ATTN_EXT` | **6354/6354** |
| dense 27B `Swift-Qwen3.8-27B-Q4_K_S`, seed 42, temp 0, 64 tok | `113 chars sha=1acb04bd9104` — **byte-identical** to the r20-based build |

The `LIGHTNING_INDEXER` default count change is the intended #60 behavior; the op is now
*supported* on gfx1100 and the model (not the backend) decides when to use it.

## 4. Still open — the reporter's 196K gate

There is no qwen4exp model on this host, so the single `llama-bench -p 196608 -n 0`
(q5_1 KV, `-b/-ub 256`) cannot be run here.  The expectation for @samuelmchan's re-run:

* the default now uses the **chain** while `score_blocks × 4 × n_query × 4 <= 128 MiB` and the
  **fused generic-vec op** above it, so the 196K prefill should fit (the fused arm is the one that
  already PASSed at 869 t/s) and the reserve should not grow past the chain peak at the threshold;
* `LLAMA_QSA_SCORE_WMMA=1` forces the fused op everywhere (A/B), `=0` forces the chain everywhere;
* `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1` still upgrades the fused arm to the WMMA4 kernel (not
  bit-exact) — unchanged.

Ask for: the single-196K pass/fail, the `compute` reserve row, and the interleaved
default / `QSA3=1` / `INDEXER4=1` prefill table from `FINDINGS-2026-09-29.md` §1.
