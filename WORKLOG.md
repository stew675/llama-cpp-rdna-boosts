# WORKLOG — dated delivery records

Reverse-chronological log of every delivery-affecting change to the
**rdna-boosts 15-patch set** (block amendments, community-fix
integrations, re-baselines, regeneration + clean-apply re-verifications).
Newest entry first.  The README's
[Current state](README.md) section is a lean summary and points here
for the full record; per-block technical notes live in
`patches/README.md`, the verification contract in `MANIFESTS.md`.

---

- **Block 15 cut (2026-09-10) — the attention-memory campaign wins;
the set is now 15 patches (block-15 tip `09a137566` on the canonical fork
rebuilt at `9113cc188`), beta-staged in `beta/block-15-campaign-wins/`.**
  The campaign (`wip/arch-independent-memory/`, `wip/qwen4exp/qsa-memory/`)
  was merged into one block by replaying the validated work-branch tree
  onto block 14, then re-validated **as a combination** (the per-win
  records did not carry over on their own).  Six wins, each with an
  environment A/B gate; **V4 is opt-in** (an *enable* switch) per the
  maintainer's rule of 2026-09-10 (a sub-2 % loss with a large memory win
  and no cheap fix ships opt-in):

  | win | mechanism | gate (default) | measured (ctx 204800, q8_0 KV, ub 2048) |
  |---|---|---|---|
  | W1 | QSA score chain: relu before the 4-D reshape + `n_blocks`-chunked `ggml_concat` assembly | `GGML_QSA_SCORE_MEM` (1) | qwen4exp 6690.40 -> 4450.40 MiB/GPU (ub1024 3346.50 -> 2274.35) |
  | W2 | derived QSA per-block bias + derived visibility; bias/mask no longer materialised; input-fill null guards (incl. the `llm_graph_input_attn_k` one) | `GGML_QSA_DERIVED_BIAS` (1), `GGML_QSA_DERIVED_VIS` (1), `LLAMA_QSA_SPARSE_FA` (sparse) | qwen4exp 4450.40 -> **3251.39** MiB/GPU, host 1262.70 -> **63.69** MiB |
  | W3 | keys-only QSA indexer cache (`v_enabled` in `llama_kv_cache`; no V tensor, no V-side op) | `LLAMA_QSA_KEYS_ONLY` (1) | indexer KV 956.26 -> **318.76** MiB/GPU |
  | W4 | ggml-alloc releases view sources whose views are never consumed (the uncounted-view leak) | none — a bug fix; `beta/block-15-campaign-wins/ab/w4-revert.patch` | repro 56.00 -> 16.00 MiB; no reserve change on any model |
  | V3 | derived kq mask: `GGML_OP_FLASH_ATTN_EXT` src[5..7] carry compact per-cell state and the MMA FA kernel derives visibility in-kernel; the packed mask tensor is still built in every graph and simply loses its consumer (so no model allowlist and no mis-served consumer) | `LLAMA_KQ_MASK_DERIVED` (1; `0` = packed) | 4B 1800.33 -> **1001.13**, 27B 1920.33 -> **1121.13** MiB/GPU; host -799.21; gemma-4-E4B/-31B (ISWA) -809.18/-811.17; scales as `n_kv x n_tps x 2 B` |
  | V4 | native q8_0 K/V in the FA kernels: dequantise during the shared-tile staging (16-byte chunk = 8 elements = a quarter q8_0 block) instead of staging a whole-cache F16 copy | `GGML_CUDA_FA_KV_NATIVE` (**default 0 = opt-in**) | 4B -> **257.13**, 27B -> **489.13**, gemma-4-31B -1224 MiB/GPU; qwen4exp unchanged |

  **The wins compose additively** — qwen4exp ub 2048: pristine 6690.40 ->
  W1 only 4450.40 -> W2 only 5491.39 -> W1+W2 3251.39 (W1 -2240, W2
  -1199, W3 -637.5/GPU, V3 -799, V4 -744/-632); both W gates off
  reproduces the pristine 6690.40/1262.70 exactly.  **Cost**: V3 -1.28 %
  prefill (4B pp20480/ub 2048, interleaved same-binary A/B) / +0.28 %
  (27B), decode -0.32 %/-0.15 %; V4 a further -1.85 % (4B) / -1.72 %
  (27B) prefill — the loss is the lost `cp_async` pipeline (a quantized
  source cannot be copied asynchronously; a 2-byte-access pass changed
  nothing), decode within noise, hence opt-in.

  **Combination validation (all on the merged tree, and then re-run from
  the delivered patches — see below):** reserve matrix on 4B (1 GPU), 27B
  (3-GPU Meta), gemma-4-E4B (1 GPU), gemma-4-31B (3-GPU) and qwen4exp
  (3-GPU) at ub 2048/1024/512 x V4 off/on — every number matches the
  per-win records; same-seed generated text **byte-identical** on all
  five models across every gate combination (V3 x V4 on the dense
  models; W1/W2/W3/V3/V4 — 7 configurations — on qwen4exp) at a short and
  a 40k-token prompt; adaptive-MTP gate **unchanged** (27B inline draft
  0.76744 (66/86, mean 3.28) in all four gate combinations; qwen4exp
  draft 0.44262 (54/122) in all six, **equal to the block-14 baseline**,
  and MTP stays +26 % over plain decode at ctx 32768); `test-backend-ops`
  FLASH_ATTN_EXT on ROCm0 (both V4 gates) and CPU, the six derived FA
  cases, VIEW/CONT/CPY/DUP/CONCAT, `test-alloc`, `test-batch-alloc`; the
  W4 revert restores `ggml-alloc.c` byte-identically to block 14.

  **Two things worth recording.**  (1) Re-validation caught a real wiring
  bug before the cut: the W3 gate was passed to `v_enabled` with the
  wrong polarity, so the indexer cache stayed keys-only-disabled (956.26
  MiB) while `LLAMA_QSA_KEYS_ONLY=0` enabled it — fixed and re-verified
  (`956.26 -> 318.76` on the default, `956.26` with the gate off).  This
  is exactly what the combination pass is for.  (2) A **pre-existing**
  bug was found (it reproduces on block 14=HEAD, so it is not a block-15
  regression): `gemma-4-E4B-it` on **3 GPUs with `-sm tensor`** aborts in
  the meta splitter (`ggml-backend-meta.cpp:1177`) on a FLASH_ATTN_EXT
  node whose K source has zero extent on one buffer, because `n_head_kv =
  2` is fewer than the device count (2 heads / 3 devices leaves one
  device with nothing).  It runs on 1 GPU, on 2 GPUs and on 3 GPUs with
  `-sm layer`; the 27B (4 KV heads) and gemma-4-31B (4/16) are
  unaffected.  Diagnosed by instrumenting the failing assert to print the
  op/tensor/split geometry (temporary change, reverted).  Left unfixed —
  out of scope for this block — and documented in `patches/README.md`.

  **Regeneration and delivery mechanics.**  The reference fork checkout
  (`~/llama.cpp`, branch `rdna-boosts`) had been rebased onto a master
  that is **two commits newer than the recorded fork point** (`f3f1a8f27`
  iGPU lazy-load default + `304665fe7` SYCL IQ-type-for-MoE, both
  2026-09-08/09, i.e. after `9113cc188`), so `format-patch
  9113cc188..tip` there would have exported those two upstream commits as
  patches 0001/0002 — a latent trap for any future regeneration.  The
  patches were therefore regenerated from a **canonical fork rebuilt at
  `9113cc188`** via `scripts/apply-all.sh` (strict 15/15 `git am`, zero
  whitespace warnings), and the resulting tree was verified identical to
  the validated tree except for the 3 files of those two upstream commits
  (`ggml-sycl` x2, `src/llama-model.cpp` — outside the validated paths).
  The delivered `0001`-`0014` files were kept byte-for-byte (the
  regenerated ones differ only in the `From <sha>` line and the
  `[PATCH NN/15]` series count, verified content-identical hunk by hunk);
  `0015-rdna-boosts-block-15-campaign-memory-wins.patch` is new.
  `make-patches.sh`'s default tip is now `09a137566`, the canonical
  block-15 commit (the local branch `block15-canonical` in the fork
  checkout keeps that chain alive).  `rdna-boosts-all.patch` = `git diff
  9113cc188..09a137566` (98 files).

  **Clean-apply simulation (the delivered artifact, end to end):** fresh
  worktree at `9113cc188` -> `scripts/apply-all.sh` (15/15 strict
  `git am`) -> fresh `gfx1201` Release build -> reserves (4B 1001.13 /
  257.13, 27B 1121.13 / 489.13, qwen4exp 3251.39 with the indexer KV at
  318.76), byte-identical coherence on 4B/27B/gemma-4-E4B/qwen4exp with
  every gate flipped, MTP 0.76744 / 0.44262, and the op suites — all
  green.

  **Upstream-drop check (2026-09-10):** GitHub was unreachable from this
  host (SSH key denied), so the check ran against the recorded upstream
  base `9cf3bf256`: the `ggml-alloc` unused-view release (W4), the
  keys-only indexer cache (W3) and the `llm_graph_input_attn_k`
  null-mask guard are all **still absent upstream** (the first two apply
  cleanly, the guard's call site is still unguarded while its own
  `can_reuse_impl` accepts a null mask), so Block 15 keeps every hunk.
  Re-check after the next `git fetch` before filing the `upstream/`
  candidates.

- **Block-14 amendment (2026-09-10) — freed-cell KV handling moved from the
  host-side zeroing to kernel-side masked-V elimination; the gfx1151-only
  `zero_freed` host zeroing (2026-09-09 amendment) is REMOVED (block-14 tip
  `ff2b35f49` on `9113cc188`, regenerated 2026-09-10; blocks 01-13 patch
  files byte-identical).**  `src/llama-kv-cache.{cpp,h}` are back to the
  upstream state — no `zero_freed`/`rows_hw`/`sharers` wiring, no env
  `LLAMA_KV_ZERO_FREED`, no per-free GPU memsets; evicting a resident KV
  sequence is pure host cell bookkeeping again on every device.  In its
  place block 14 now carries the three **kernel-side** fixes that make the
  content of fully-masked (freed/stale) flash-attention cells unreadable,
  so the host workaround is unnecessary:
  - HIP `fattn-tile.cuh` (packed-bf16 PV path): zero the per-warp V
    register copies of rows whose P is +0.0 across the warp's columns
    before the bf16 dot.
  - HIP `fattn-mma-f16.cuh`: after each V-tile slice is staged in shared
    memory, zero the rows the mask tile marks blocked (-inf) for every
    query column of the block; one extra uniform barrier, masked path
    (`ncols2 > 1 || mask_h`) only; compile-time excluded for the
    `V_is_K_view` and NVIDIA-swizzled (`swz_V`) paths.
  - Vulkan `flash_attn_cm1.comp` + `flash_attn.comp` scalar path: never
    read V of fully masked columns (dead columns keep V = +0.0).
  All three are unconditional in their kernel paths (no arch/env gating) —
  generic correctness fixes for masked/freed FA cells (batch serving, KV
  eviction) active by default on every device.  Root cause (Strix Halo,
  gfx1151): WMMA f16 `x + (-0.0)` is inexact, so a masked column leaked
  the sign of whatever V its cell last held; the fix guarantees masked
  cells contribute exactly +0.0 at the multiply.  Validation on the
  gfx1151 box (ROCm 7.14-gfx1151 + Vulkan RADV), host zeroing disabled:
  16/16 identical-request determinism gates PASS on every KV cache type
  each backend's FA supports — ROCm f16/bf16/q8_0/q4_0 (plus ON==OFF
  bit-identical over 2064 cells/run), Vulkan also q4_1/q5_0/q5_1/iq4_nl;
  `test-backend-ops` FLASH_ATTN_EXT vs CPU 4591/4591 (ROCm) and
  7822/7822 (Vulkan); CPU same-seed greedy 51/64 tokens identical,
  divergence only at a near-tie (CPU non-FA vs GPU FA numerics);
  depth-16384 llama-bench decode tg128 within 0.05% of pre-fix, pp within
  single-run drift.  Full record:
  `wip/strix-halo/kvzero/RECORD-2026-09-09.md` +
  `wip/kv-sign-leak/HANDOVER-2026-09-09-mma-f16.md`.  Delivery:
  regenerated `patches/0014` only (blocks 01-13 patch bodies
  byte-identical) + `rdna-boosts-all.patch`; clean-apply sim at
  `9113cc188` strict 14/14 `git am`, zero whitespace warnings, applied
  tree == fork tip `ff2b35f49`; final-tree rebuild (delta vs the
  validated kernel-fix tree = the llama-kv-cache revert only) passes the
  16/16 gate and no longer logs the freed-cell zeroing.

- **Block-14 amendment (2026-09-09) — freed-cell KV-zeroing gated to gfx1151
  (fork block-01 commit `7c4d9c4e0`, block-14 tip `27485f1ca`, 14 commits on
  `9113cc188`; previous tip `0f2b7a4e1` superseded).**  Block 14's
  `seq_rm`/`seq_keep`/`clear` row zeroing (freed KV cells kept at +0.0 as a
  masked-column guard for the gfx1151/Strix-Halo WMMA f16 `x+(-0.0)`
  inexactness, ported from the strix lineage commit aad5adb08f) is now
  **enabled only when a KV-cache buffer device description carries `gfx1151`**
  (env `LLAMA_KV_ZERO_FREED=0/1` overrides the auto detection).  Everywhere
  else the pre-block-14 behavior is restored: evicting a resident KV sequence
  is pure host cell bookkeeping again.  Reason: without the gate, freeing an
  N-token sequence issued ~48×N per-cell 512-byte memsets (ggml's
  meta/multi-buffer memset decomposes one per-layer zeroing call into one
  synced `cudaMemsetAsync` per cell across the GPU head-split sub-buffers,
  each ~30-60 µs), so replacing a ~13k-token KV stalled ~18-24 s before the
  new prefill began on multi-GPU RDNA4 (3x R9700 gfx1201; reproduced on a
  plain dense 4B model too — model-agnostic).  Verified: on gfx1201 the
  A/B stall is gone (identical workload 24.5 s -> ~6 s) and the zeroing-off
  determinism gate passes (16 + 8 identical greedy requests, per-position
  top-8 logprobs float64-compared — the same gate that found the leak on
  gfx11); on the gfx1151 Halo box the gate enables
  ("freed-cell KV row zeroing enabled (gfx1151)") and the 16-run control is
  unchanged.  Regenerated `patches/0014` only (blocks 01-13 patch bodies
  byte-identical); clean-apply sim at `9113cc188` strict 14/14 `git am`,
  zero whitespace warnings, applied tree == fork tip `27485f1ca`.
  Follow-up (open): develop a performant gfx1151 flash-attn kernel-side fix
  so the host-side zeroing can be removed entirely.

- **Block-01 refresh (2026-09-09) — adaptive MTP draft depth updated to the
  llama.cpp PR #27210 review head (fork block-01 commit `7c4d9c4e0`,
  block-14 tip `0f2b7a4e1`, 14 commits on `9113cc188`).**  Block 01 was cut
  from PR #27210 (author: stew675) at its `0994374fd` state; the PR then
  advanced through a maintainer review round (`8408cdabf` comment fixes +
  `d236d41a2`, the review-response changeset).  The block is now refreshed
  to the PR head `d236d41a2`, still delivered as **one squashed patch
  block** (`git diff 9113cc188..d236d41a2` = 15 files, 519+/35-, applied
  as the single block-01 commit; blocks 02-14 re-based on top untouched).
  Review-round content now in block 01: `common_params_speculative::
  has_mtp()` helper (arg.cpp/common.cpp/server-context.cpp/init result
  refactored through it); a new `accept_partial()` virtual +
  `common_speculative_accept_partial()` so a partial acceptance the
  context could not apply (checkpoint-restore path in tools/server and
  examples/speculative-simple) is reported once and the following replay
  round cannot feed stale draft counts to the adaptive controller
  (non-adaptive accept path unchanged); the adaptive depth reset moves
  ahead of the empty-prompt early return in `begin()`; `
  --spec-draft-n-min-adaptive` rejects values < 1 and is documented
  (docs/speculative.md, tools CLI/server READMEs); the invalid-range
  check is `GGML_ABORT` -> `std::runtime_error`; draft-mtp +
  draft-mtp-adaptive together are rejected (shared ctx_dft); the delta-
  net conv-state snapshot-bound rationale comment; stale "defaults to 2"
  test comment fixed (default is 3) + value-0 rejection case.
  Regeneration mechanics: canonical fork rebuilt at `9113cc188` from the
  previous set (am-tip `050ec89ce`), block 01 replaced in place by the
  squashed PR-head changeset, blocks 02-14 `git rebase --onto` (clean,
  no conflicts — blocks 02-13 touch no block-01 file, block 14's
  common/arg/common.h hunks are disjoint).  Tree verification: old-tip..
  new-tip delta is exactly the review changeset (13 files, 129+/70-, ==
  `0994374fd..d236d41a2`), every other file byte-identical; regenerated
  0002-0013 patch bodies byte-identical to the previous delivery, 0014
  refreshed only in index lines/hunk offsets for the 3 common files;
  regenerated 0001 diff body byte-identical to the PR head changeset.
  Verification (local 3x R9700, gfx1201, ROCm 7.14): clean-apply sim at
  `9113cc188` strict 14/14 `git am`, zero whitespace warnings, applied
  tree == fork tip; rebuilt `test-arg-parser` + `test-speculative-
  adaptive` pass; plain-decode same-seed coherence (seed 42/temp 0,
  Qwen3.5-4B-Q8_0) token-IDENTICAL to the known-good `050ec89ce` build.
  The refresh touches no GPU kernels and no non-speculative host decode
  path — all changes live in the MTP-typed/adaptive code, the option
  parser and comments/docs.

- **Re-base (2026-09-08) — delivery moved to upstream master `9113cc188`
  (block-14 tip `78e67a3d8`).**  Upstream moved 14 commits past the
  `050dde50c` fork point (server checkpoint eviction, Kimi-K3 recurrent
  rollback, chat-parser split, ggml_prec spec, metal/vulkan/opencl fixes,
  spec single-device meta-wrapper handling #28390, and — decisive for this
  re-base — `d4389a4dd`/PR #28604 which **reverted #24233**, the very
  change block 06 diverged from).  An `apply-all.sh` run against the fresh
  master tip failed at block 06 in a way even `git am -3` cannot fix: the
  upstream revert deleted block 06's pre-image, so the block's change is a
  no-op on the new base (nothing left for the patch to do).  Resolution:
  block 06 was reduced to a host-buffer **rationale marker** commit (6
  comment lines above the now-unconditional `integrated = false` in
  `ggml-cuda.cu`), keeping the 14-block structure and all downstream block
  numbers intact; block 14's quantized-KV tensor-split gate merged
  **additively** with #28390's single-device `SPLIT_MODE_TENSOR` warn in
  `src/llama-context.cpp` (both kept, in sequence; #28390's code comment
  shows the same single-device-no-meta-wrapper intent as block 07, so no
  semantic collision).  Content verification against the previous delivery
  (re-applied at `050dde50c`): blocks 01-05 and 07-13 are byte-identical;
  block 06 differs as designed; block 14 differs only in the
  llama-context.cpp resolution region.  Regenerated at `9113cc188`
  (`f84549d23..78e67a3d8`) and clean-apply re-verified (strict 14/14
  `git am`, zero whitespace warnings, applied tree == fork tip
  `78e67a3d8`).  Coherence verified on the Strix box (gfx1151, ROCm 7.14):
  llama-cli same-seed output IDENTICAL to the canonical `72f0ee944` build
  (tensor + layer split x f16/q8_0/bf16 KV, and a long-prompt run at depth
  16384), clean runtime diagnostics, and the dense adaptive-MTP gate green
  on the new build (draft acceptance 0.833 at acc/pos 0.944/0.833/0.722;
  draft-mtp 20.3 t/s vs plain 7.9 t/s on the same prose prompt; MTP
  same-seed byte-identical old-vs-new).  The previous `050dde50c`-based
  regeneration (`d65a96084..ce641322e`) is superseded; the pre-re-base fork
  chain is preserved at `backup-rdna-boosts-bfcc4be99` and the known-good
  `72f0ee944` binary under `/tmp/rdna-ref-bin/` (session-local).

- **Block-14 amendment (3rd on 2026-09-08) — quantized-KV tensor-split
  gate:** the `q4_1`-family KV cache types (`q4_1`, `q5_0`, `q5_1`,
  `iq4_nl`) aborted during the first graph reserve under multi-GPU
  `SPLIT_MODE_TENSOR` on gfx1201 (3x R9700) —
  `ggml-backend-meta.cpp:538 GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)`
  — on both dense qwen35 (Qwen3.6-27B) and qwen4exp (Flash-Next), with
  `f32/f16/bf16/q8_0/q4_0` KV and layer split passing.  Root cause is
  **upstream**: reproduced on pristine vanilla llama.cpp at the fork
  point `050dde50c` (identical assert, non-qwen4exp Qwen3.5-4B; also at
  1 GPU, since upstream wraps even a single device in the Meta backend)
  and still unfixed on current upstream master.  Tensor split forces
  flash attention, whose CUDA/HIP kernels read the quantized K/V cache
  natively only for `q4_0`/`q8_0` (plus the float types); for the
  q4_1-family types the attention subgraph is externalized into
  op-NONE graph leaves (MIRRORED split state) which collide with the
  AXIS-0 elementwise gate branch of the qwen35/qwen4exp gated attention
  at the `attn_gated` `MUL` — the meta splitter cannot reconcile
  MIRRORED x AXIS-0.  Fix: a context-creation gate in
  `llama_init_from_model` (`llama-context.cpp`, block-14-owned in the
  set) that rejects K/V types outside FA's native set with a clear
  error when the Meta device is actually in use (tensor split over
  >= 2 GPUs; the fork's single-GPU "tensor" mode skips the Meta wrapper
  and is untouched — upstream, whose 1-GPU mode also wraps Meta, gets
  the clean error too).  Validated 2026-09-08 on gfx1201 (3x R9700,
  ROCm 7.14): KV-type matrix on dense 27B Q8_0 + Flash-Next IQ4_XS
  (3-GPU tensor) — `f32/f16/bf16/q8_0/q4_0` generate;
  `q4_1/q5_0/q5_1/iq4_nl` and `k=q4_1 v=bf16` / `k=bf16 v=q4_1` fail
  cleanly (zero asserts, actionable message); layer split + q4_1
  Flash-Next 25.9 t/s (unchanged); qwen4exp derived-cache pool-gate
  byte identity holds (tokens identical with the pool skipped vs
  `GGML_CUDA_QSA_INDEXER_CACHE=1`); dense-27B same-seed coherence A/B
  (gate stripped vs applied on the same tree) byte-identical;
  test-llama-archs qwen4exp all OK (NMSE 1.01e-13).  Canonical fork
  rebuilt at `050dde50c` (am-commits `d65a96084..ce641322e`, block-14
  tip `ce641322e`); set regenerated with `scripts/make-patches.sh`;
  clean-apply sim re-verified 2026-09-08 (14/14 strict `git am`, zero
  whitespace warnings, applied tree == fork tip `ce641322e`, full build
  clean, coherence byte-identical to the validation tree).
 the 2026-09-07 local
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
