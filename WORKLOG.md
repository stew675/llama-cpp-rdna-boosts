# WORKLOG — dated delivery records

Reverse-chronological log of every delivery-affecting change to the
**rdna-boosts 14-patch set** (block amendments, community-fix
integrations, re-baselines, regeneration + clean-apply re-verifications).
Newest entry first.  The README's
[Current state](README.md) section is a lean summary and points here
for the full record; per-block technical notes live in
`patches/README.md`, the verification contract in `MANIFESTS.md`.

---

- **Two-lineage reconciliation (2026-09-08):** the 2026-09-07 local
  delivery (`9850143`: block-14 **derived-cache pool gate**, regen at
  fork tip `bfcc4be99`) had never been pushed; the 2026-09-08 lineage on
  `origin/main` (issue #18 MUL_MAT_ID pair-fusion layout gate + issue
  #19 moe_weighted_reduction float4 remainder, both folded into blocks
  13/14; the block-14 compiler-warning cleanup; the qwen4exp
  tensor-split HIP gate — regen tip `2f1dc384b`) had been authored from
  a clone without it.  The two block-13/14 regens touched disjoint
  source hunks, so blocks 13/14 now carry all of it: QSA quantized-KV
  decode gate, derived-cache pool gate, the issue-18/19 fixes, the
  warning cleanup and the tensor-split backend gate.  Canonical fork
  rebuilt at `050dde50c` (am-commits `7df708e66..72f0ee944`, block-14
  tip `72f0ee944`); set regenerated with `scripts/make-patches.sh`;
  clean-apply sim re-verified 2026-09-08 (14/14 strict `git am`, zero
  whitespace warnings, applied tree == fork tip `72f0ee944`).
- **Block-14 amendment (2nd) — qwen4exp tensor-split backend gate
  (2026-09-08):** follow-up to the Vulkan validation sweep: block 14
  had removed upstream's `case LLM_ARCH_QWEN4EXP: // TODO: fix
  test-llama-archs` from `llm_arch_supports_sm_tensor`, enabling
  qwen4exp tensor split for every backend.  That is validated on
  ROCm/HIP only (3x R9700, NMSE 9.87e-14 vs CPU); on backends that
  cannot run the fused QSA/HC/WS4 ops on-device (Vulkan, Metal, SYCL;
  NVIDIA CUDA untested) the CPU-fallback subgraphs leave the meta
  splitter unable to reconcile mirrored-vs-split operand states and it
  aborts at graph reserve (`ggml-backend-meta.cpp` `handle_generic`,
  e.g. the qwen4exp gated-attention `MUL` on Vulkan — `test-llama-archs`
  died at the qwen4exp Meta row).  The enablement is now `#ifdef
  GGML_USE_HIP`, restoring upstream's clean "not implemented" error /
  arch-test SKIP on all other builds.  Verified: Vulkan — full
  test-llama-archs sweep completes RC=0 (457 rows, statuses identical
  to upstream 050dde50c, qwen4exp Meta SKIP like upstream), qwen4exp
  single-device still OK (9.01e-08, roundtrip OK), llama-cli
  qwen4exp `-sm tensor` fails with the upstream message; HIP —
  qwen4exp Meta still OK 9.87e-14 (validated path unchanged).  Canonical
  fork rebuilt at `050dde50c`; block-14 tip `13719e3ca` →
  `2f1dc384b`; set regenerated; clean-apply sim re-verified (14/14
  `git am`, zero whitespace warnings, applied tree == fork tip).
- **Block-14 amendment — compiler-warning cleanup (2026-09-08):** the
  block-14 sources warned under the `build-llama-vulkan` (system clang
  16.2.1, `-Wall -Wextra`) and `build-llama-rocm-714` (ROCm clang)
  host builds.  Five warnings, all from block-14 code, fixed and
  folded into the block-14 commit:
  - `ggml.c` — unused `n_blocks` local in the `ggml_indexer_fill`
    builder (removed).
  - `ggml-cpu.c` — `-Wswitch`: the exhaustive CPU compute-forward
    switch had no case labels for the new `GGML_OP_INDEXER_SCORE` /
    `GGML_OP_INDEXER_FILL` ops (GPU-only fused ops; the CPU plan
    phase already aborts on them as "op not implemented" before
    compute, so the case is an unreachable `GGML_ABORT`, mirroring
    `GGML_OP_COUNT`).
  - `ggml-cpu/ops.cpp` — two `-Wunreachable-code-break` warnings: the
    `break` after the noreturn `GGML_ABORT("fatal error")` in the
    `HC_MIX`/`HC_COMBINE` CPU type dispatchers' default cases
    (dropped, matching upstream convention).
  - `qwen4exp.cpp` — `idx_cache` was narrowed to `bool`, making the
    documented `GGML_CUDA_QSA_INDEXER_CACHE=2` debug probe
    (`idx_cache != 2`) tautologically true (`-Wtautological-constant-
    out-of-range-compare`); restored to an `int` with the 0/1/2
    tri-state so probe-2 (pool read without the fill) is reachable
    again.  `-Wsign-compare` in the gfx-id sniff loop (`size_t`
    counter vs `ggml_backend_dev_count()`).
  No generated-code or runtime-behavior change in default configs.
  Verified: the four TUs compile warning-free with the exact
  build-vulkan flags; full Vulkan + ROCm 7.14 (gfx1201) builds clean
  on the re-applied sim tree.  Canonical fork rebuilt at `050dde50c`;
  block-14 tip moved `3529b3497` → `13719e3ca`; set regenerated
  (14/14 `git am`, zero whitespace warnings, applied tree
  byte-identical to the fork tip); `rdna-boosts-all.patch` refreshed.
- **Block-14 amendment — MUL_MAT_ID pair-fusion layout gate (2026-09-08,
  issue #18):** community report + detailed root-cause analysis by
  `briansp2020` (production single-R9700 deployment of the 14-block
  set, ROCm 10): the block-13/14 MUL_MAT_ID gate+up pair fusion
  aborted the process with `GGML_ASSERT(ne11 == 1 && n_expert_used > 1)`
  in `ggml_cuda_mul_mat_q_pair` whenever two MUL_MAT_ID nodes shared
  src1/ids in a layout the fused kernel does not express (src1->ne[1] > 1
  or top-1 routing) — `test-backend-ops -b ROCm0` died in the
  MUL_MAT_VEC_FUSION group.  The dispatcher gate now requires the
  callee's layout preconditions; such pairs fall back to the per-node
  path, and the qwen4exp sparse-MoE pair (standard layout) still fuses.
- **Block-13 amendment — moe_weighted_reduction float4 remainder fix
  (2026-09-08, issue #19):** community report by `briansp2020`: the
  2026-09-06 mwr-float4 fold dropped the last `n_embd % 4` columns of
  every output row for `n_embd % 4 != 0` (silent wrong output;
  `MOE_WEIGHTED_REDUCTION(n_embd=63, ...)` failed).  The float4 quad
  kernel is now gated to `n_embd % 4 == 0` (where it is also
  alignment-safe) and the upstream scalar bounds-checked kernel covers
  the rest; the aligned path is byte-unchanged.
  Both fixes validated here (3x R9700 gfx1201, ROCm 7.14):
  `test-backend-ops -b ROCm0` **16590/16590** with the fusion active,
  MUL_MAT_VEC_FUSION 1265/1265, MOE_WEIGHTED_REDUCTION 6/6, same-seed
  llama-cli streams byte-identical (Flash-Next IQ4_XS 3-GPU and dense
  27B single-GPU; default vs `GGML_PAIR_OFF=1`/`GGML_PAIR_DENSE_OFF=1`),
  prefill A/B confirms the pair fusion still fires (pp2048/pp8192
  default > pair-off beyond noise), Flash-Next full model runs clean on
  CPU (`-ngl 0`).  Fork tip moved `3529b3497`; set regenerated;
  clean-apply sim re-verified (14/14 `git am`, zero whitespace
  warnings).  Full record:
  [`patches/README.md`](patches/README.md).
- **Block-14 amendment — QSA quantized-KV decode gate (2026-09-07):** a
  quantized KV cache type (e.g. `--cache-type-k q8_0`) aborted qwen4exp
  context init (`GGML_ASSERT` in `ggml_indexer_fill`: the fused decode
  indexer ops read raw cache rows in F32/BF16/F16 only, but the indexer
  sub-cache shares the main `--cache-type-k`).  `build_qsa_top_k` now
  falls back to the per-op chain for quantized indexer keys.  Validated
  on Strix Halo across the full KV-type matrix f32/f16/bf16/q8_0/
  q4_0/q4_1/iq4_nl/q5_0/q5_1 (start + generate, zero errors; BF16 fused
  path unregressed).  Fork tip moved `60aa4173d`; set regenerated.
- **Block-08 amendment — PR #15 integrated (2026-09-07):** community
  report + fix by DanoPTT (single R9700, production since 2026-09-07):
  block 08's mul_mat+bias fusion through a view node handed the
  mmvq/mmvf kernels a destination whose shape the guards never checked
  (a reshape moves tokens between dimensions on multi-sequence
  batches) → `GGML_ASSERT(ids || dst->ne[1] == 1)` abort.  Fix folded
  into the block-08 commit (delivery convention): require the
  through-view destination to satisfy the kernels' shape constraint
  before fusing.  Fork tip moved `3bebffd6b`; set regenerated;
  verified here (3x R9700): clean-apply sim tree-identical, build
  clean, test-backend-ops 6759/6759, dense same-seed byte-identical
  pre vs post fix, 3-GPU hybrid == RCCL, parallel 2-slot decode clean.
- **Re-baseline to upstream master `050dde50c` + block 14 (2026-09-07):**
  fork point moved from `465e49b9c` to the current master tip (22
  upstream commits; the ggml-cuda-touching ones — `b74f590ea` f16 FA
  divergent-barrier fix #27870, `73ab7599b` branchless Q4_K/Q5_K mmvq
  unpack #26705, `473599738` gfx90c HIP support #26454 — merged in
  disjoint hunks).  The `~/llama.cpp` `rdna-boosts` fork was rebuilt
  from `patches/` via `scripts/apply-all.sh` (13/13 `git am` clean at
  `050dde50c` after one manual block-04 conflict in
  `tests/test-backend-ops.cpp` — upstream LEAKY_RELU perf cases kept
  alongside block 04's) and **block 14 (qwen4exp support) was promoted
  from `beta/qwen4exp`** (fork delta `c261553a1..dd4301fb4`, re-based;
  one manual `common.cuh` conflict — upstream gfx90c APU macros kept
  alongside the block's `GGML_CUDA_CC_IS_GFX1151`).  Set regenerated
  with `scripts/make-patches.sh` (base `050dde50c`, canonical
  am-commits `90a816a68..3bebffd6b`, 14 blocks) and
  `rdna-boosts-all.patch` refreshed (87 files).  Clean-apply sim at
  `050dde50c` re-verified 2026-09-07 (applied tree byte-identical to
  the fork tip).  Full record:
  [`patches/README.md`](patches/README.md).
- **Re-baseline to upstream master `465e49b9c` (2026-09-06):** fork point
  moved from `9cffdcc80` to the current master tip (18 upstream commits
  past the fold-verified base `8b4b3558f`, 57 past the old fork point;
  the ggml-cuda-touching ones — `73a43d1f6` mmid/mmf race fixes #28475,
  `5fdfa6282` GDN l2-norm fix #28068 — merged in disjoint hunks, zero
  conflicts).  The `~/llama.cpp` `rdna-boosts` fork was rebuilt from
  `patches/` via `scripts/apply-all.sh` (13/13 `git am` clean, zero
  whitespace warnings; per-file content check on all 112
  upstream-touched files passed) and the set regenerated with
  `scripts/make-patches.sh` (base `465e49b9c`, canonical am-commits
  `45bf4d291..c261553a1`).  Two prerequisites: the 0044cfe fold had
  stripped the format-patch mail headers from 0002/0004/0008/0013 —
  restored from the pre-fold originals (delivery commit 0610b75) — and
  the block-13 message's fold-amendment trailer was re-dated to the
  fold's true date (tip amended `b4b760eb8` -> `c261553a1`).
  `rdna-boosts-all.patch` refreshed (45 files; was stale at 41,
  pre-fold).  Clean-apply sim at `465e49b9c` re-verified 2026-09-06
  (applied tree byte-identical to the fork tip).  The `qwen4exp` fork
  branch was rebuilt on the new base + the consolidated beta support
  patch (see `beta/qwen4exp/README.md`).
- **Campaign date re-stamp (2026-09-06):** the gfx1151/qwen4exp campaign
  docs had run a week ahead of the real calendar; every
  `wip/`/`beta/`/archive date (filenames + text) was collapsed onto the
  real git dates (2026-09-05/06) and the moved records' stale
  `benchmarks/2026-09-*` references were repointed at
  `wip/archive/qwen4exp/discovery/`.
- **Block-13 RDNA3.0 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps are now also on RDNA3_0 (gfx1100), validated on a single RX
  7900 XTX (ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M (ub 2048,
  1-GPU pinned): fusion fires, same-seed coherence IDENTICAL fused-on
  vs off, prefill gains pp2048 +9.4% (5405 vs 4939), pp16384 +7.8%
  (4487 vs 4162), decode unchanged (tg128 130.3 vs 130.4).  The
  RDNA4-tuned J caps transfer (uncapping regressed pp2048 5405 -> 4819
  / pp16384 4487 -> 4070, below the 3-op fallback; a Q3_K@96 probe
  also lost to the cap 64).  Set regenerated from a canonical fork
  rebuilt at `9cffdcc80` (13 am-commits, block-13 tip `8c2ace510`);
  clean-apply sim verified (zero whitespace warnings, applied tree
  byte-identical to the fork tip).  Full record:
  [`wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`](wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md).
- **Block-13 RDNA3.5 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps were RDNA4-only; validated on Strix Halo (Ryzen AI MAX+ 395 /
  Radeon 8060S, gfx1151, ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M
  (ub 2048): same-seed coherence IDENTICAL fused-on vs off, prefill
  gains match RDNA4 (pp2048 +5.3% 1590 -> 1674, pp16384 +4.6% 1360 ->
  1423), decode unchanged (tg128 71.5). The RDNA4-tuned J caps
  transfer (uncapping regressed pp2048 1674 -> 1111 / pp16384 1423 ->
  1334).  Full record:
  [`wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`](wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md).
- **Block-13 MTP regression fixes (2026-09-02, folded into block 13):**
  (1) dense adaptive-MTP collapse (18.3 -> 27.5 t/s) — the mmvq
  item-split/rpb kernel is register-bound at multi-token decode batches
  (ncols 2..8 = the speculative verify step); fixed with a re-added
  pre-block-13 K-split kernel (`mul_mat_vec_q_ksplit`) for those batches
  and long-K single-token rows (plain decode 29.0 -> 30.1, output
  bit-identical to the 12-block build).  (2) MoE adaptive-MTP collapse
  (draft acceptance 0/1527, 53 t/s vs plain 90) — the block-08
  rms_norm->mmvq Q8_1 quantize-cache fold corrupts multi-token MUL_MAT_ID,
  so verify logits diverge from single-token decode; the fold is now gated
  to single-token MMID + plain MUL_MAT consumers (acceptance 0 -> 0.51,
  MTP 126 t/s vs upstream ~113).  MoE MTP had no baseline data, which is
  why it slipped.  Details + verification: `patches/README.md` block-13
  notes.  The adaptive-MTP baseline gate and expectations now live in
  [`benchmarks/mtp-adaptive-methodology.md`](benchmarks/mtp-adaptive-methodology.md)
  — run Protocol A there before shipping decode/fusion changes.
- **Fork tip:** the fork block-12 commit was amended 2026-09-04 with the
  runtime NCCL-failure fallback (issue #13); block 13 was amended
  2026-09-02 with the two MTP regression fixes, 2026-09-05 with the
  RDNA3.5 (Strix Halo) then RDNA3.0 (gfx1100) fused-MoE-MMQ gate
  relaxations and 2026-09-06 with the model-neutral Strix MoE mmq
  folds.  The set was regenerated 2026-09-06 from a canonical fork
  rebuilt at `465e49b9c` (13 am-commits, block-13 tip
  `c261553a1`); the clean-apply sim at `465e49b9c` applies with zero
  conflicts/whitespace warnings and its tree is byte-identical to the
  fork tip.
- **Fork point (baseline):** llama.cpp master at `465e49b9c` (re-based
  2026-09-06 from `9cffdcc80`, itself re-based 2026-09-02 from
  `0eadefebd`; 57 commits of drift from the old fork point — see
  `patches/README.md` for the dated re-base record, incl. the 2026-09-02
  manual merges vs upstream's #27970 (sparse-fa) and #25952 (fused MoE
  expert reduction)).
- **Set:** 14 patches in `patches/` (`0001`-`0014`).
- **Verified:** clean apply + full build + llama-cli same-seed coherence
  IDENTICAL (hybrid vs RCCL, 3-GPU) on the rebuilt fork; the clean-apply
  sim at `465e49b9c` applies with zero conflicts/whitespace warnings and
  its tree is byte-identical to the fork tip (`c261553a1`; 2026-09-06
  regeneration — earlier regenerations were re-verified on the RX 7900
  XTX box with sim build coherence identical + perf reproduced). tg64
  38.12 / tg512 41.08 and the block-13 numbers are unchanged — the
  re-base is content-identical plus upstream's additions.
- **Whitespace-clean apply:** the regenerated set applies with **zero git
  whitespace warnings** (`git am` 01-13; re-verified 2026-09-02 on a
  fresh checkout at `9cffdcc80`, re-verified 2026-09-04 after the
  block-12 amendment, re-verified 2026-09-05 after the block-13 RDNA3.5
  gate relaxation and again after the RDNA3.0/gfx1100 fold,
  re-verified 2026-09-06 on the `465e49b9c` re-base).
- **Deployment:** 3-GPU hybrid (`HIP_VISIBLE_DEVICES=0,1,2`, unpinned) gives
  depth-16384 decode 38.71 t/s (+21.8% vs 2-GPU). See
  [`patches/README.md`](patches/README.md) for block-12 env knobs and the
  server config.
- **RDNA4-only gate:** block 12 refuses to init off gfx1200/gfx1201 and
  falls back to RCCL (community RDNA3 verification pending).
- **Runtime NCCL-failure fallback (2026-09-04, issue #13):** block 12 no
  longer aborts when NCCL/RCCL fails at runtime — on the first failure it
  clears the sticky HIP errors on each AR device, warns once, stops using
  NCCL for the rest of the run and re-routes AllReduce to the internal
  pipeline (or the meta backend's butterfly).  This covers RCCL >= 2.30.4
  refusing kernel dispatch on a PCIe root port without AtomicOp completer
  support (e.g. PCH/Z390; `ncclCommInitAll` succeeds — see
  ROCm/ROCm#6520).  Folded into the block-12 commit; re-verified
  2026-09-04 (clean-apply sim, build, same-seed coherence IDENTICAL pre
  vs post fix on 27B Q8_0, depth-16384 tg unregressed: 2-GPU 32.48 ->
  32.40, 3-GPU 39.33 -> 39.31).
- **Block-12 AR_PROFILE fix (2026-09-01, PR #8):** AR-profile `devices[]`
  init order fixed — `GGML_CUDA_AR_PROFILE=1` no longer faults GPU 1
  under MTP (pre-fix reproduced on 3x R9700; post-fix clean, profiler
  dumps on every device).  Regenerated into the set; coherence unchanged.
- **Block-02 MTP chunked-GDN prefix (2026-09-01, PR #9):** block 02 now
  runs its chunked WMMA GDN on long single-sequence MTP prefills (prefix
  `n_tokens-K` + sequential K-tail) — +7.5% prefill at ~5.5k prompt,
  +7.7% at ~38k on 3x R9700, 64-token same-seed output token-identical
  to sequential.  Opt out: `GGML_CUDA_GDN_CHUNKED=0`.
