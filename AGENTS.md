# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing
anything in `~/llama-cpp-rdna-boosts/` (or acting on its behalf).

## What this repo is

A **delivery repo**: it packages the RDNA/ROCm work of the
[`stew675/llama.cpp`](https://github.com/stew675/llama.cpp) fork
(`rdna-boosts` branch) as a **15-patch set** (block 00 + blocks 01-14) that
applies to a clean llama.cpp checkout at the fork point **`9113cc188`** (re-based 2026-09-08
from `050dde50c`, itself re-based 2026-09-07 from `465e49b9c`, itself
re-based 2026-09-06 from `9cffdcc80`, re-based 2026-09-02 from `0eadefebd`).

- Block **00** (`patches/0000-rdna-boosts-block-00-structural-and-architecture-fix.patch`):
  **structural and architecture fixes** — the base every later block applies on
  top of.  Added 2026-09-10 with (1) FA small-batch KV-split width invariance
  (issue #25: decode and every speculative verify width now reduce identically,
  so greedy MTP output no longer changes with `--spec-draft-n-max`) and (2) the
  Vulkan masked-V/freed-cell fixes (`flash_attn_cm1.comp`/`flash_attn.comp`).
  See the 2026-09-10 block-00 section in `patches/README.md`.
- Blocks **01-11** (`patches/0001-…0011-…`): MTP draft depth, fused chunked
  GDN, BF16 KV (block 03 also carries the **HIP masked-V/freed-cell fixes**
  since 2026-09-10), WMMA flash-attn, CPU bit-identical decode, host-buffer
  revert, meta wrapper skip, fused core, meta headroom, k-quant boosts,
  CUDA prefill-graph skip.  **Block 08 amended 2026-09-11**: the decode/verify
  FlashAttention kernel-family fix (F1) and the **quantized KV-type enablement** —
  `q4_1`/`q5_0`/`q5_1` were behind `GGML_CUDA_FA_ALL_QUANTS`, which made the FA
  probe disable flash attention for the whole context (3.4x slower prefill / 1.7x
  decode); they are enabled unconditionally with their three diagonal vec instances,
  while the flag remains the knob for the *mixed* K!=V pairs (K==V is still enforced
  without it).  See the block-08 notes in `patches/README.md` and `GREEDY-PURITY.md` §20.
- Block **12** (`patches/0012-rdna-boosts-block-12-hybrid-HIP-all-reduce-RDNA4-gat.patch`): the hybrid HIP
  all-reduce (custom internal AR for the small-tensor decode path +
  per-size hybrid dispatch vs RCCL), **RDNA4-only** (gfx1200/gfx1201; falls
  back to RCCL elsewhere). The fused-stage/pacing experiments it spawned are
  archived, env-gated OFF, in `archive/work/fused-stage-pacing/`.
  Amended 2026-09-04 with the runtime NCCL-failure fallback (issue #13):
  on the first NCCL runtime failure the comm layer clears the sticky HIP
  errors, warns once, stops using NCCL for the rest of the run and
  re-routes AllReduce to the internal pipeline (or meta-butterfly) — see
  the block-12 notes in `patches/README.md`.
- Block **13** (`patches/0013-…-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch`): fused MoE gate+up+GLU MMQ (prefill)
  + mmvq short-K item-split (decode); see the block-13 notes in `patches/README.md`.
  Amended 2026-09-02 with two regression fixes folded into the block: (1) the
  mmvq item-split/rpb kernel collapse of multi-token decode batches (ncols 2..8,
  the speculative verify step — dense MTP 18.3 -> 27.5 t/s, ksplit dispatch);
  (2) the rms_norm->mmvq Q8_1-cache fold corrupting multi-token MUL_MAT_ID
  (MoE MTP acceptance 0 -> 0.51, draft-mtp 53 -> 126 t/s, fold gated to
  single-token MMID).  Amended 2026-09-05 with the RDNA3_5 (Strix Halo,
  gfx1151) gate relaxation: the fused gate+up+GLU MMQ arm + its
  `J_max_gate` tile caps were RDNA4-only; validated on a Ryzen AI MAX+ 395
  (Qwen3.6-35B-A3B True-Q3_K_M, ub 2048) — pp2048 1590 -> 1674 (+5.3%),
  pp16384 1360 -> 1423 (+4.6%), coherence IDENTICAL, decode unchanged;
  the RDNA4-tuned J caps transfer (uncapping regresses).  Amended again
  2026-09-05 with the RDNA3_0 (gfx1100) gate relaxation: validated on a
  single RX 7900 XTX (Qwen3.6-35B-A3B True-Q3_K_M, ub 2048, 1-GPU
  pinned) — fusion fires, coherence IDENTICAL fused-on vs off, pp2048
  4939 -> 5405 (+9.4%), pp16384 4162 -> 4487 (+7.8%), decode unchanged
  (tg128 130.3); the RDNA4-tuned J caps transfer there too (uncapping
  regressed below the 3-op fallback; a Q3_K@96 probe also lost to the
  cap 64).  Details + numbers:
  `patches/README.md` block-13 notes and
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md` +
  `wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
  **Also amended 2026-09-11 (fourth amendment) with the fused shared-expert
  epilogue band**: the decode-only `ne[1] == 1` gate on `ggml_cuda_op_shexp_down_gate`
  now serves the whole `n_tokens <= MMVQ_MAX_BATCH_SIZE` band — the two kernels are
  token-generic and `nwarps` is pinned to the single-token reduction order — so the
  MoE decode and verify take one arithmetic (`W = 1..8` bit-identical, the
  "MoE asterisk" is gone) and MoE `draft-mtp` acceptance rises 0.51 -> 0.82
  (167.3 t/s vs plain 96.9 on 35B-A3B).
- Block **14** (`patches/0014-rdna-boosts-block-14-qwen4exp-support.patch`):
  qwen4exp / Qwen3.8-Flash-Next support, promoted from `beta/qwen4exp`
  2026-09-07 — QSA sparse FA (default) + fused indexer top-k/score,
  HC_MIX/HC_COMBINE fused decode ops, managed lazy reader + PLE n-gram
  loading, MTP draft-head, WS4 hyperconn prefill fusions, per-arch
  dense/QSA decode policy; **the 2026-09-09 gfx1151-only freed-cell host
  zeroing stays REMOVED** (`llama-kv-cache.{cpp,h}` are the upstream state; no
  `zero_freed`/env `LLAMA_KV_ZERO_FREED`/per-free GPU memsets) and the
  kernel-side masked-V fixes it was replaced with were re-homed on 2026-09-10:
  the Vulkan `flash_attn_cm1.comp`/`flash_attn.comp` fixes now live in block 00,
  the HIP `fattn-tile.cuh`/`fattn-mma-f16.cuh` fixes now live in block 03 (they
  sit on the native-BF16 FA path block 03 introduces), so block 14 carries none
  of them.  **Amended 2026-09-11**: the fused hyper-connection ops
  (`ggml_cuda_op_hc_mix`/`_hc_combine` in `ggml/src/ggml-cuda/hc-mix.cu`) and the
  `src/models/qwen4exp.cpp` gates now serve the whole decode/verify band `1 <= nt <= 8` (they were
  `nt == 1`), which fixes qwen4exp's decode-vs-verify divergence up to `--spec-draft-n-max 3`; the
  ops map the token onto `blockIdx.y` with explicit per-token strides, and a <= 8-token *prefill*
  chunk also takes the fused path (it cannot be told apart from a verify batch — that is the point).
  **Also amended 2026-09-11 with the QSA decode-arm band**: the arch policy's dense decode arm
  (`build_layer_attn`, `src/models/qwen4exp.cpp`) was gated `n_tokens == 1`, so above the indexer
  selection width (`indexer_top_k + r - 1` = 2051) a W=1 decode ran dense while the n-token verify
  batch fell through to the sparse top-k selection — the cause-3 text divergence.  The arm now serves
  the whole band (`QSA_DECODE_BAND = 8`); prefill keeps the sparse selection.
  **Also amended 2026-09-11 with the QSA-vs-KV-type arm gate + the tensor-split gate
  narrowing**: the fused sparse QSA op reads the cache natively for f16/bf16/q8_0 only,
  so with any other quantized cache type the graph now takes the dense masked path
  (`qsa_sparse` also requires a QSA-native cache type) — otherwise the un-split op left
  the attention output mirrored while the gate stayed hidden-split and the meta splitter
  aborted on `attn_gated`, which hit `q4_0` too (**pre-existing**).  The tensor-split gate
  (`llama_init_from_model`) now asks `llama_kv_type_has_native_fa()` instead of a
  hardcoded `{q4_0, q8_0}`, so `q4_1`/`q5_0`/`q5_1` are allowed under tensor parallelism
  and `iq4_nl` keeps a clean error.  Verified per type on 3-GPU `-sm tensor` (27B, qwen4exp).
  **Also amended 2026-09-11 with the QSA quantized-KV enablement and the K/V-head chunking fix**:
  the fused QSA kernel now dequantizes `q4_0`/`q4_1`/`q5_0`/`q5_1` while staging a tile
  (`get_dequantize_V<type, half, 4>`), so every cache type takes the same attention path on
  qwen4exp (prefill 2076 -> 2384 t/s at 32k on `-sm tensor`, uniform with f16), and its head
  chunking is now `min(QSA_MAX_HEADS, gqa_ratio)` instead of `QSA_MAX_HEADS` — the old split put
  16 heads in one block whose shared smem K/V tile mixed **two** K/V heads (qwen4exp: 24 q-heads /
  2 kv-heads = gqa 12), i.e. a silent quality bug (perplexity 7.33 -> 6.53 = the dense masked
  oracle).  The same amendment adds the missing **CPU reference** for the four new types in
  `ggml/src/ggml-cpu/ops.cpp` plus a `FLASH_ATTN_QSA` backend-op test (18 cases) — the kernel had
  no oracle at all before, which is why a width-pure corruption survived every gate.  See
  `GREEDY-PURITY.md` §21 and the block-14 notes in `patches/README.md`.
  **Amended 2026-09-12 (seventh) with the MTP-export logits-purity fix** (the last layer always gathers
  its output rows; the unmasked `embeddings_nextn` export gets a separate full-row tail for `t_h_nextn`
  — `GREEDY-PURITY.md` §28) **and (eighth) with the QSA indexer-score decode/verify band-uniformity fix**
  (the score flattens the indexer heads into `ne11 = n_idx_h * n_tps = 4 * n_tps`, which crossed
  `MMVF_MAX_BATCH_SIZE` at `n_tps = 3`, so the verify batch fell through to MMF while decode stayed on
  MMVF and a top-k near-tie flipped; the guard now covers the whole flattened band
  `MMVF_MAX_BATCH_SIZE_FLAT = 32` with `mul_mat_vec_f` instantiated for `ncols_dst` 9..32 —
  `GREEDY-PURITY.md` §29).
  See the block-14 notes in `patches/README.md` and the beta
  validation record in `beta/qwen4exp/README.md`.
- Block **15** (STAGED in `beta/block-15-campaign-wins/`, **NOT a delivery patch**): the attention-memory campaign wins --
  **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA
  per-block bias + derived visibility + the input-fill null guards
  (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA
  indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view
  release (no gate), **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by
  default), **V4** native q8_0 K/V and **V5** native bf16 K/V in the FA
  kernels (both behind the same `GGML_CUDA_FA_KV_NATIVE`, **opt-in,
  default 0**).  The beta patch was cut 2026-09-10 and **amended twice on
  2026-09-10: V5, then the RDNA3_5/gfx1151 fix** (beta patch tip `377f8e790`; the gfx1151
  amendment enables V3 on a HIP iGPU -- the probe had rejected
  `GGML_BACKEND_DEVICE_TYPE_IGPU` -- and requires a single KV stream in
  `kq_mask_derivable()` so `n_seq_max > 1` contexts no longer abort in
  `ggml_flash_attn_ext_add_kq_derived`); ~3.4 GiB/GPU
  + ~1.2 GiB host on qwen4exp and ~800 MiB/GPU + 800 MiB host on dense
  models (a bf16 KV cache saves a further 712/584/658/1352 MiB with V5
  enabled), byte-identical output, ~1.3 % prefill / ~0.3 % decode cost
  (V4 ~1.7 %, V5 0.2-2.4 % depending on prompt length, V5 measured
  against the scratch it removes; on **gfx1151** the arms are *cheaper*/win
  -- V4 +2.6 % at pp20480, V5 0.4-0.9 %); beta window open, pending the
  maintainer's promotion go-ahead.  See
  `beta/block-15-campaign-wins/README.md` and the
  `WORKLOG.md` entry.

The repo is NOT the fork: the fork (source of truth for the block commits)
lives at `~/llama.cpp`, branch `rdna-boosts`.  **Fork-state warning (read
before any regeneration):** the working `~/llama.cpp` checkout has at cut
time been rebased onto a master **two commits newer than the recorded fork
point** (`f3f1a8f27` iGPU lazy-load default + `304665fe7` SYCL
IQ-type-for-MoE, both dated after `9113cc188`), so
`git format-patch 9113cc188..<that branch's tip>` there would export those
two upstream commits as patches 0001/0002.  The **canonical** 15-block
chain is a rebuild of the delivery set at `9113cc188` (tip `c6f1e8e78`, net tree
  `e1e42e23c2913cd529b0064eb1cb74525a746098`,
built by applying the delivery patches with `scripts/apply-all.sh` at
`9113cc188`; block 02 amended 2026-09-11 with the whole-batch
K-independent chunked GDN prefill and again 2026-09-12 with the rollback-bounded
chunked threshold (`n_rs_batch`) + the pre-batch snapshot slots; block 08 amended 2026-09-11 with the
decode/verify FA kernel-family fix, again with the quantized-KV-type
enablement (`q4_1`/`q5_0`/`q5_1`), and again with the `iq4_nl` enablement (the predicate, the 15
new `fattn-vec-instance-iq4_nl-*.cu` files, `dequantize_q4_nl` and the three non-contiguous
converters); block 13 amended 2026-09-11 with the MoE
decode/verify mmvq band, again with the fused shared-expert epilogue band, and
again 2026-09-12 with the column-blocked epilogue (its band launch shape made
the kernel read the down-weight row once per token and idle 7 of its 8 warps —
a bit-identical restructure repays the band amendment's `pl=8` cost), and
again 2026-09-12 with the RDNA3_5 single-token-only mmvq fusion skip (the dense
gate+up+GLU fusion and the weighted-down MoE tail are single-token-only and do
not reproduce the standalone mmvq arithmetic, so a 1-token decode and an n-token
verify took different reductions on gfx1151; gated there — `GREEDY-PURITY.md` §25);
block 14 amended 2026-09-11 with the
hyper-connection decode/verify band fix, again with the QSA decode-arm
band, again with the QSA-vs-KV-type arm gate + the tensor-split gate
narrowing, and again with the `iq4_nl` QSA/CPU-oracle/test entries, and again
2026-09-12 (sixth) with the configurable QSA prefill arm + the device-query arm gate —
the prefill axis is now depth-configurable (`qsa_dense_prefill_until`, env
`LLAMA_QSA_DENSE_PREFILL_UNTIL`) with the documented arch policy preserved as its
default: **0 = QSA prefill always, every arch and split** (the 2026-09-07 policy —
Soar QSA wins prefill from ~8K monotonically to +181 % @160K, Halo from ~16K), so
the delivery stays byte-identical to the pre-amendment build and the arm is an
opt-in A/B, and
`qsa_kv_native`'s hand-maintained copy of the kernel's type list is replaced by a
`ggml_backend_dev_supports_op()` query on a shaped probe tensor (under `-sm
tensor` the Meta device's `all_of()` IS the meta-split safety condition) — see the
2026-09-12 block-14 amendment in `patches/README.md` and `GREEDY-PURITY.md` §26),
which is what
`scripts/make-patches.sh`'s default tip refers
to; always regenerate from a canonical fork rebuilt at the fork point.
**Block 15 (the attention-memory campaign) is NOT in the delivery** -- it
is staged in `beta/block-15-campaign-wins/`.

Block provenance on the canonical chain: block 00 added 2026-09-10 (FA
small-batch KV-split width invariance, issue #25, plus the Vulkan
masked-V fixes — see the block-00 section in `patches/README.md`);
blocks 01-14 = the fork's block
commits on master `9113cc188` (2026-09-08 re-base; block 01 refreshed
2026-09-09 to the upstream PR #27210 review head `d236d41a2`, still one
squashed block, and amended 2026-09-11 so `--spec-draft-n-max` is clamped to 7
with a visible notice + `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escape hatch; block 03 amended 2026-09-10 with the HIP masked-V/
freed-cell fixes, re-homed from block 14; block 14's 2026-09-09 gfx1151-only
freed-cell host zeroing is removed and its 2026-09-10 masked-V fixes were
re-homed — Vulkan to block 00, HIP to block 03; on the re-base block 06 was
reduced to a host-buffer
rationale marker — upstream itself reverted #24233 in #28604 on
2026-09-08, matching its end state, so the functional delta is now
upstream (see the WORKLOG re-base entry); block 12 carries the
2026-09-04 runtime NCCL-failure fallback, issue #13, and was amended
2026-09-11 so the hybrid dispatch's small/large crossover no longer changes
the reduction algorithm across the decode/verify band (2-device `32768` ->
`131072` elements); block 13 amended
2026-09-02/09-05/09-06 as above, 2026-09-08 with the
moe_weighted_reduction float4 remainder fix (issue #19, reported by
briansp2020) and 2026-09-11 with the F2 cause-2 decode/verify
**band-uniformity** fix (the per-type mmvq caps are floored at
`MMVQ_MAX_BATCH_SIZE` and `mul_mat_vec_q_moe`'s launch bound is sized at the
band, so `W = 1..8` is bit-identical — **+14-26 %** at the verify widths);
block 14 added 2026-09-07 and amended
2026-09-07 with the QSA quantized-KV decode gate + the derived-cache
pool gate (quantized indexer-key caches no longer abort the fused
decode path, and the F32 derived-cache pool is allocated only when the
fused path can actually use it — see the block-14 notes in
`patches/README.md`) and 2026-09-08 with the MUL_MAT_ID pair-fusion
layout gate (issue #18, reported by briansp2020 — MUL_MAT_ID pairs in
non-standard layouts now fall back to the per-node path instead of
aborting), 2026-09-08 with the compiler-warning cleanup
(Vulkan/clang-16 + ROCm host builds) and 2026-09-08 with the qwen4exp
tensor-split backend gate (`llm_arch_supports_sm_tensor(qwen4exp)`
true on HIP builds only — the ROCm-validated backend; other builds
keep upstream's clean "not implemented" error / arch-test SKIP instead
of the meta-splitter abort found on Vulkan),
2026-09-11 with the mixed-K/V hard reject
(`params.type_k != params.type_v` now fails context creation for every model,
not just MLA/DeepSeek4); block 08
amended 2026-09-07 with the PR #15 mul_mat+add through-view shape
guard and 2026-09-11 with the decode/verify FA kernel-family fix (F1: a
quantized K/V cache used VEC at `n_q <= 2` and TILE from `n_q = 3`, so
plain decode disagreed with spec verify — `GREEDY-PURITY.md` §14)). The
canonical `9113cc188` fork used for `make-patches.sh`
regeneration is disposable and is re-created from `patches/` +
`scripts/apply-all.sh` whenever it needs rebuilding (fresh clone at the
fork point + apply) — the last regeneration (2026-09-10, the 15-block set
with block 00 and the re-homed masked-V fixes) applied strict 15/15 `git am`
and produced tip `505637d6e` (the 2026-09-11 block-02 amendment re-ran the
regeneration: strict 15/15 `git am`, applied tree `fcf3e4bb7` == canonical,
tip `7b79930b2`; the 2026-09-11 block-13 dense-MMVQ-alignment amendment
re-ran it once more: strict 15/15 `git am`, zero whitespace warnings,
applied tree `c0775c33c` == canonical, tip `27bd754b6`; the 2026-09-11
block-02 default-flip (GGML_CUDA_GDN_ALIGN_BOUNDARY now opt-**out**) re-ran
it again: strict 15/15 `git am`, zero whitespace warnings, applied tree
`31e153fe3` == canonical, tip `27bd754b6`; the 2026-09-11 block-02 re-cut to
the whole-batch chunked prefill (free alignment, gate + K-dependent branches
removed, rollback guard added) re-ran it last: strict 15/15 `git am`, zero
whitespace, applied tree `928852cdc` == canonical, tip `389c5341f`).  Apart from the
block-02 and block-13 hunks the blocks' bodies are byte-identical to the
previous regeneration apart from the `From <sha>` line and the
`[PATCH NN/15]` series count (plus the block-00 Vulkan and block-03 HIP
hunks).  (The block-15 attention-memory campaign was
temporarily staged as a 15th patch and then un-promoted; it lives only in
`beta/block-15-campaign-wins/`.)  Older fork states are
preserved on the `stew675/llama.cpp` fork remote (`rdna-boosts` =
previous tip `482837e5a` on `0eadefebd`; `rdna-boosts-orig`, …) and in
older local reference clones — never rely on them for the current
delivery.

## Pushing policy (MANDATORY — read before any `git push`)

**Never push anything out of the `~/llama.cpp` fork checkout — never to
upstream llama.cpp, and never to the personal fork unless the maintainer
explicitly requests it.**

- All deliverable changes live in THIS repo (`llama-cpp-rdna-boosts`) as
  the `patches/` set.  That is the only thing that gets pushed (to this
  repo's own `origin`, `github.com:stew675/llama-cpp-rdna-boosts`).
- The `~/llama.cpp` checkout exists to host the block commits and to
  apply/test the diff set locally.  Its `rdna-boosts` branch is
  **disposable**: the sanctioned flow is to **delete the pre-patched
  branch and re-apply our diff set** (`scripts/apply-all.sh` on a fresh
  checkout at the fork point) — never to push the branch anywhere.
- If the maintainer explicitly asks to push a fork sub-branch, the ONLY
  permitted target is the personal fork
  (`git@github.com:stew675/llama.cpp.git`, the `fork` remote).  NEVER
  push to upstream `ggml-org/llama.cpp` (the `origin` remote in
  `~/llama.cpp`) — a bare `git push` there would target upstream.
- Confirm the exact branch name and intent with the maintainer before any
  such push; if history rewrites are involved use `--force-with-lease`,
  never a bare `--force`.
- Repeated attempts to push directly to llama.cpp can result in an account
  ban.  When in doubt: don't push, ask.

## Layout

| path | what |
|------|------|
| `README.md` | consumer overview + workflow (start here) |
| `MANIFESTS.md` | apply order, per-block verification, validation history |
| `BASELINE.md` | fork point, patch provenance, drift policy |
| `GREEDY-PURITY.md` | the purity rulebook (index + invariants + per-finding claims; read before shipping) — its dated narratives/evidence for the closed cases are in `archive/docs/GREEDY-PURITY-FINDINGS.md` under the same `§` numbers |
| `patches/` | **the delivery set** (0000-0014: block 00 + blocks 01-14) + apply README |
| `scripts/apply-all.sh` | the verified apply flow (`git am` block 00 + blocks 01-14, automatic `git am -3` fallback on a drifted base) |
| `scripts/make-patches.sh` | regenerates the set from the fork |
| `rdna-boosts-all.patch` | the entire 15-patch net as ONE patch (fork point only) |
| `benchmarks/` | dated benchy/v1/v2 records + methodology + graphs; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) |
| `wip/` | exploration docs, tuning tools, session handoffs — **NOT part of the delivery** (see the WIP rule below) |
| `beta/` | **promoted-from-WIP staging** (e.g. `beta/qwen4exp/` = qwen4exp support + its validation record; `qwen4exp-support.patch` promoted into the delivery as block 14).  `beta/block-15-campaign-wins/` is the beta record for **Block 15** (staged 2026-09-10, **NOT a delivery patch**; it lives only in `beta/block-15-campaign-wins/block-15-campaign-wins.patch`): its README is the promotion/gate record and `BETA-TESTING.md` the tester checklist — see the WIP rule below |
| `upstream/` | **upstream-PR candidates** — self-contained changes that could be filed against unadulterated `ggml-org/llama.cpp` master, each with a `UPSTREAM-PR-*.md` note + `.patch` (see its README for the double-apply caution and the status table) |
| `archive/docs/` | moved-out historical records (validation history, baseline history) — reference only |
| `archive/work/` | closed experiments, preserved for future re-evaluation |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |

## Scope policy — RDNA first, other backends uninjured (2026-09-11)

This repo is **RDNA/ROCm-specific**: its validation, tuning and claims cover the AMD devices the
maintainer runs (gfx1201 = RDNA4, plus validated gfx1151/RDNA3_5 and gfx1100/RDNA3_0 work).  The
patch set is generic llama.cpp, so it should not *break* other backends (NVIDIA/CUDA, MUSA, SYCL,
Vulkan, CPU) — that is why the shared CMake lists, the dispatch tables and the predicates are kept
mutually consistent even when a change is unreachable on AMD — but **behaviour and performance on
non-AMD backends are explicitly out of scope**: no tuning, no validation, no waiting on hardware there.
NVIDIA parts have their own developers and maintainers; that is not this repo's job.

Consequences, so it is not re-litigated:

* A fix that is reachable on AMD only may be landed **without** its non-AMD counterpart, as long as
  the non-AMD paths stay *consistent* (no aborts, no uninstantiated pairs) and the difference is
  documented.  Worked example: the F1 decode/verify band fix deleted the VEC arms in the chooser's
  generic fallback (the only one AMD reaches); the NVIDIA (`turing_`/`volta_mma_available`) arms are
  **left alone deliberately** — the staged `upstream/UPSTREAM-PR-fa-decode-verify-kernel-family.*`
  carries them for upstream, and nothing AMD-side depends on it.
* New KV-cache types / instances / predicates **are** kept cross-backend consistent, because an
  inconsistent set is a crash on whichever backend reaches it (see `GREEDY-PURITY.md` §20 and the
  block-08 notes) — that is correctness, not scope creep.
* "Not validated on NVIDIA" is an acceptable, documented state — never a blocker for an RDNA win.

## Critical facts (do not re-derive)

- **Apply method:** all 15 blocks with **`git am`** (each block is a
  committed fork commit, exported with `git format-patch`; block 12 is a
  regular commit like the rest, no special `git apply` step).
  Plain `git apply` of the concatenated series **silently drops
  hunks** (30 files/2483 lines vs the correct 35/6094 — verified
  2026-08-29). `scripts/apply-all.sh` is the tested path.
- **Naming collision:** in OLD docs ("block 12" in BASELINE.md's historical
  records), "block 12" can mean the old *k-quant umbrella* (now block 10).
  In the current delivery, **block 12 = the hybrid all-reduce, period.**
- **tg/throughput is NOT a correctness signal.** Always verify coherence:
  llama-cli same-seed comparison (see below) or `wip/tools/ar_kernel_unit.cpp`.
- **Everything is fast at depth 0** — decode perf work must be validated at
  depth-16384 (benchy protocol), not shallow llama-bench.
- **Never run parallel/background benches** — they contaminate results.
- **Mixed K/V cache types are HARD-REJECTED** (`params.type_k != params.type_v` fails context
  creation with a message naming both types).  Maintainer decision 2026-09-11: every mixed pair
  measured 1.7–3.6× slower than the same-type equivalent and never smaller, and the attention path
  (including the split/FA one) assumes `type_k == type_v`.  Implemented as a block-14 amendment with
  a `f16`/`f16`-style pairing in every gate; test scripts must pass matching `-ctk`/`-ctv`.
- **`--spec-draft-n-max` is capped at 7** (a clamp + one warning, not an error).  A verify batch
  decodes `n_max + 1` rows and the HIP FA chooser switches kernel family above 8 rows, so deeper
  drafts can change greedy output between plain and MTP.  Block-01 amendment; see
  `GREEDY-PURITY.md` §11/§19.
- **The pin regressed** (session 7): `~/bin/high-power` (dpm=high +
  runtime-PM) costs tg -5-7% / pp -15-18% on RCCL/hybrid paths. Server runs
  UNPINNED, 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`), hybrid default.
- **The set applies whitespace-clean**: `apply-all.sh` prints no git
  whitespace warnings (re-verified 2026-09-01 on `0eadefebd`,
  2026-09-02 on the `9cffdcc80` re-base, 2026-09-04 after the
  block-12 amendment, and 2026-09-05 after the block-13 RDNA3_5 gate
  relaxation, and again 2026-09-05 after the RDNA3_0/gfx1100 fold,
  and again 2026-09-06 on the `465e49b9c` re-base, and again 2026-09-07
  on the `050dde50c` re-base + block 14).
- **The QSA op has an oracle now, and it needed one (2026-09-11).**  `test-backend-ops -o FLASH_ATTN_QSA`
  compares the GPU kernel against `ggml_compute_forward_flash_attn_qsa` (CPU) over all seven KV types,
  `gqa` 1 and 8, both decode/verify widths and the sliced walk — **18/18** must pass.  Two hard-won
  facts: (1) the `W=1..8` logits-purity matrix is *blind* to a width-uniform corruption (it can only
  prove widths agree with each other), and the probe cannot even reach this op by default (the indexer
  selection width is 2051 > the probe's max `P`; force it with
  `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`); (2) **MTP acceptance is not a quality
  signal when the defect is in both the draft and the main context** — the corrupted pair is
  self-consistent and accepts *more* (0.65 vs 0.49).  The comparable quality metric is the
  **perplexity ratio against the dense masked path** (`LLAMA_QSA_SPARSE_FA=0`, same attention, FA
  kernels), which must match within noise.  A fused op with several heads sharing one staging buffer
  must keep the block homogeneous in every index the staging reads (QSA: the K/V head) — see
  `GREEDY-PURITY.md` §21.
- **Block 02 (0002) now also carries the MTP chunked-prefix dispatch
  (PR #9, 2026-09-01):** long single-sequence MTP prefills (`K > 1`,
  `n_seqs == 1`, `n_tokens > K+64`) run the chunked WMMA GDN on the
  prefix (`n_tokens - K`) and sequential GDN only on the last K snapshot
  slots.  Fired + verified on 3x R9700 (2-GPU, internal AR, Qwen3.8-27B
  Q8, ubatch 1024, MTP n-max 3): +7.5% prefill at ~5.5k prompt, +7.7% at
  ~38k; 64-token same-seed output token-identical to sequential.  Opt
  out: `GGML_CUDA_GDN_CHUNKED=0` (also `GGML_CUDA_GDN_CHUNKED_BF16=0`).
  Bench record: `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
  Amended 2026-09-11 with the **whole-batch K-independent chunked prefill**:
  a batch with more than `max(K, 16)` tokens is chunked whole — the exact same
  call `K == 1` makes — and anything smaller stays on the sequential kernel, so
  plain decode and the MTP path agree (`--spec-type none == draft-mtp`) with
  **no sequential tail and no cost** (27B pp512/2048/4096 = 1385/1356/1328,
  parity with the old K-dependent boundary).  A batch larger than `max(K, 16)`
  cannot be a verify batch (those decode `<= K` tokens) and is never rolled back
  into, so its snapshots are skipped; a **once-only guard** in
  `llama_memory_recurrent::seq_rm` warns if that assumption is ever violated.
  The `GGML_CUDA_GDN_ALIGN_BOUNDARY` gate and its two K-dependent branches were
  **removed** (~118 lines) — both were unreachable with the gate ON and the
  opt-out no longer bought any performance.  `GGML_CUDA_GDN_CHUNKED=0` is the
  only switch left (forces the sequential kernel: correct, bit-identical,
  slow).  **Amended 2026-09-12 with the rollback-bounded chunked threshold (`n_rs_batch`)**:
  the whole-batch path wrote no rollback snapshots, on the assumption that a
  batch above `max(K, 16)` "cannot be a verify batch" — false for long-draft
  speculators (`n_rs_seq` comes from `speculative.draft.n_max` = 7, while
  `--spec-ngram-mod-n-max` can draft 64), so a 65-token verify batch followed by
  a small tail rollback restored an unwritten plane and the recurrent state
  silently rewound.  The threshold is now
  `max(K > 16 ? K : 16, n_rs_batch)` with `n_rs_batch =
  common_speculative_n_max() + 1` (a new `ggml_gated_delta_net` op param,
  threaded through `llama_context_params`/`llama_cparams` and into the
  `seq_rm` guard), and the pre-batch ssm/conv state is written into slot
  `n_tokens` when `0 < n_tokens < K` so a whole-batch rollback restores the
  state before it.  Default configs are unaffected (`n_rs_batch` 1 / 8 <= 16);
  validated by **FAIL -> PASS** on `test-recurrent-state-rollback`
  (`max diff 6.5366, first at seq 0 pos 16` -> `max diff 0`) and
  `GATED_DELTA_NET` 46/46 on gfx1151 — `GREEDY-PURITY.md` §27,
  `patches/README.md` (the 2026-09-12 block-02 amendment).
  **Note the pure `none == draft-mtp` range is `n_max <= 7`, not 15** —
  an 8-token verify batch is the designed limit (the FA tile-vs-WMMA switch at
  `Q->ne[1] > 8` changes the reduction beyond it); on 2-GPU `-sm tensor` it was
  `n_max <= 5` until the block-12 dispatch fix described below
  (`GREEDY-PURITY.md` §11, follow-ups Part 3).
  Record: `wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md`.
- **Block-12 AR_PROFILE init fix (2026-09-01, PR #8, integrated):**
  `devices[]` is filled from the caller list before the profiler
  hipMallocs — with `GGML_CUDA_AR_PROFILE=1` the buffers were allocated
  while the array was still zero-filled, so every buffer landed on GPU 0
  and MTP's second pipeline (draft context) faulted GPU 1 (gfx1201).
  Pre-fix reproduced (GPU-1 memory fault in `ggml_cuda_ar_kernel`);
  post-fix runs clean with teardown dumps on every device; default
  serving is byte-for-byte unchanged.
- **Block-12 runtime NCCL-failure fallback (2026-09-04, issue #13,
  folded into block 12):** RCCL >= 2.30.4 can refuse kernel dispatch at
  the first collective (`hipErrorIllegalState`) when a GPU sits behind a
  PCIe root port without AtomicOp completer support (e.g. PCH/Z390;
  `ncclCommInitAll` succeeds — see ROCm/ROCm#6520), which used to abort
  the run at the first prefill AllReduce.  On the first NCCL runtime
  failure the comm layer now clears the sticky HIP errors on each AR
  device, warns once (`dmesg | grep -i atomic` check), permanently stops
  using NCCL, and re-routes AllReduce to the internal pipeline (or the
  meta backend's butterfly when no pipeline); the failing call returns
  false so the butterfly handles it; `ncclCommDestroy` at teardown is
  non-fatal.  No behavior change on healthy setups.  Re-verified
  2026-09-04: clean-apply sim + build + same-seed coherence IDENTICAL
  pre vs post fix (27B Q8_0, 3-GPU); depth-16384 tg unregressed (2-GPU
  32.48 -> 32.40, 3-GPU 39.33 -> 39.31).
- **Verified numbers (2026-09-02 re-base, unchanged):** clean-apply build
  tg64 38.12 / tg512 41.08; depth-16384 3-GPU hybrid 38.71 t/s
  (unpinned); 2-GPU (1,2) 31.79.  The re-base is content-identical plus
  upstream's additions (42 commits, 2026-09-02) — numbers carry over.
- **Block-13 MTP regression fixes (2026-09-02, folded into block 13):**
  (1) dense adaptive-MTP collapse — the block-13 mmvq item-split/rpb kernel
  is register-bound at multi-token decode batches (ncols 2..8 = the spec
  verify step); fixed by re-adding the pre-block-13 K-split kernel as
  `mul_mat_vec_q_ksplit` for ncols 2..8 + long-K (K >= 4096) ncols==1 rows
  (dense MTP 18.3 -> 27.5, plain 29.0 -> 30.1, output bit-identical to the
  12-block build).  (2) MoE MTP collapse — the block-08 rms_norm->mmvq Q8_1
  quantize-cache fold corrupts multi-token MUL_MAT_ID (moe kernel consumes
  the cached y wrongly), so MoE verify logits diverge from single-token
  decode and MTP acceptance collapses to 0; the fold is now gated to
  single-token MMID + plain MUL_MAT consumers (MoE acceptance 0 -> 0.51,
  draft-mtp 53 -> 126 t/s vs upstream ~113).  MoE MTP had no baseline data
  — that is why it slipped; the MTP gate now lives in
  `benchmarks/mtp-adaptive-methodology.md`.  Verify decode changes with
  Protocol A there (acceptance must stay > ~0.45 **at pos 1**, MTP >= plain at
  the default depth 3) before relying on llama-bench numbers.  **Purity ranks above raw non-MTP
  throughput**: a fix that makes the verify batch compute what the decode computes may cost a few
  percent at the wide verify widths — land it, record the delta and file the optimisation follow-up
  (measured 2026-09-11: −2.4 % at `pl=8` bought MoE acceptance 0.51 -> 0.81707, +73 % MTP; that
  particular cost was repaid on 2026-09-12 by the column-blocked epilogue below — `pl=8` 461.0 ->
  475.4 t/s, bit-identical).  See
  `GREEDY-PURITY.md` §19.
- **MoE (`qwen35moe`) decode/verify IS byte-identical by default (fixed 2026-09-11).**
  The fused shared-expert window (`ggml_cuda_op_shexp_down_gate`, +3.1% MoE
  decode) does not reproduce the unfused chain's arithmetic: its gate dot uses
  its own reduction order rather than the standalone mmvq order (the epilogue FMA
  was removed 2026-09-11).  Until 2026-09-11 it was therefore gated to `n_tokens == 1`
  and the unfused chain served the verify batch — a width-dependence, not just a
  numerical drift.  The kernels are now token-generic with `nwarps` pinned to the
  single-token reduction order, and the **whole band** `1 <= nt <= 8` takes the
  fused path: probe `W = 1,2,3,4,8` all `ac8825358d9adfda`, and with
  **`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`** (kept for A/B) all `bd138ad2326fbbf2`.
  MoE MTP gained too (acceptance 0.51 -> 0.81707, 167.3 t/s vs plain 96.9 on
  35B-A3B).  The companion block-13 fix of the same day — all
  `MUL_MAT_ID` use the dedicated MoE kernel, not the dense ksplit-with-ids path —
  is **+6.2% MoE decode** (tg128 95.62 -> 101.52); see the 2026-09-11 WORKLOG
  entries.  **2026-09-12:** the epilogue's `grid = (nrows, ncols)` (one block per
  `(output row, token)`) was replaced by a `ncols_dst`-templated kernel with the
  token loop inside the k-block loop and `grid = (nrows)` — one weight read per
  `(row, k-block)` for the whole band, per-token accumulators, `nwarps` still
  pinned and every token's reduction order unchanged, so it is a **no-op at every
  gate** (old-vs-new `.so` A/B: all hashes equal) while `pl=8` gains 3.1 %,
  `pl=4` 2.4 % and `pl=1` is flat; the fused default now beats the unfused
  reference at every width (see `GREEDY-PURITY.md` §24).
- **The QSA decode arm is band-uniform (2026-09-11) — but the QSA *sparse* regime has two open items.**
  qwen4exp's `--spec-type none` vs `draft-mtp` text divergence ("cause 3") was the
  dense arch-policy arm gated `n_tokens == 1` in `src/models/qwen4exp.cpp`: above
  `width = indexer_top_k + r - 1` (= 2051) a W=1 decode stayed dense while the
  verify batch fell through to the sparse top-k selection.  The arm now serves the
  whole band (`QSA_DECODE_BAND = 8`), so `plain == n_max 3 == n_max 7`
  byte-identically (`804de0576868` f16, `75d8530c5bb1` q8_0); an arm trace proved
  it (`wip/kv-quant-purity-followups/tools/qsa-arm-trace.patch`).  **Re-measured on
  gfx1151 2026-09-12 (the two previously-recorded sparse-regime items):** both were
  artifacts of the block-13 RDNA3_5 mmvq-fusion impurity (fixed 2026-09-12) — the fused
  indexer score is byte-identical to the per-op chain (512-token forced-sparse A/B:
  same text with `GGML_CUDA_QSA_INDEXER_SCORE`/`_CACHE` default vs 0), the pre-fix
  divergence reproduces only with `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`,
  and **default gfx1151 configs are pure** (shallow dense on every KV type, deep sparse
  at ~74K on f16 and q8_0).  The 64K crossover stays.  **Item 4 is closed (2026-09-12 (12),
  block-14 amendment (seventh))**: sub-item (a) — the MTP target's unmasked `embeddings_nextn`
  export (`common/speculative.cpp:1431`) defers qwen4exp's last-layer output gather (`gather_now`,
  `src/models/qwen4exp.cpp`) and shifted the prefill's last-position logits by a ULP
  (`ad3acaa7…` vs `b624a79f…`) — is **fixed** (the last layer always gathers its output rows;
  the export gets a separate full-row tail, `mstep NEXTN=1` 0 mismatches, was 1); sub-item (b) was
  **re-opened by the gfx1201 investigation and root-caused + fixed** (2026-09-12 (13), block-14
  amendment (eighth)): the *forced*-sparse q8_0 forward is width-dependent at W >= 3 because the
  indexer score flattens its heads into `ne11 = 4 * n_tps`, which crosses `MMVF_MAX_BATCH_SIZE` at
  `n_tps = 3` (verify -> MMF, decode -> MMVF) and flips a top-k near-tie; the guard now covers the whole
  flattened band (`MMVF_MAX_BATCH_SIZE_FLAT` = 32, `ncols_dst` 9..32 instantiated) so W = 1..8 is
  bit-identical with decode's `Thash` unchanged (the earlier "driver-level, not a width dependence"
  conclusion was drawn from gfx1151's `mstep`, which is pure there).  The gfx1151 `plain != draft-mtp`
  **text** residual (`a57bc13bbf2a` vs `n3 3124adfd2b94`) did not reproduce on gfx1201; a cross-check on
  gfx1151 against branch `block14-band-uniformity` is pending (TODO item 17).  See `GREEDY-PURITY.md`
  §§16-18, §28-§29 and
  `wip/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`.
- The one-sided AR wait (dev0/bus-06 dispatch-gap asymmetry, ~12.7 µs/call)
  is a **platform-level CP/driver property**, not reachable from the AR
  kernel, graph tail, or host-side pacing — fusion/pacing are CLOSED
  (`archive/work/fused-stage-pacing/`).
- **WIP rule (MANDATORY):** everything under `wip/` — including the loose
  patch/diff files in `wip/qwen4exp/patches/`,
  `wip/qwen35moe-prefill/patches/`, `wip/hybrid-allreduce/` and
  `wip/managed-ngrams/patches/` — is **experimental work, NOT part of the
  delivery**. Never apply any `wip/` item to the `~/llama.cpp` fork or any
  llama.cpp checkout, never fold `wip/` content into `patches/`, and never
  present `wip/` results as delivery claims, **unless the user explicitly
  asks you to work with a specific `wip/` item**. They are kept for future
  re-evaluation only.
- **Promotion rule (the sanctioned way out of `wip/`):** a campaign's
  *validated* wins are collected under `beta/` (for the memory campaign:
  `beta/block-15-campaign-wins/`), each win gets an environment kill-switch so
  it can be A/B tested and bisected, the **combination** is re-validated (the
  individual validations do not carry over), and only then is a new delivery
  block cut — for this campaign **Block 0015** — with the maintainer's
  go-ahead after a ~4–5 day beta window.  Anything that is also applicable to
  unadulterated upstream `ggml-org/llama.cpp` gets a copy under `upstream/`
  (as `UPSTREAM-PR-<slug>.md` + `.patch`) so it can be filed as a PR.
- **Block 15 is the memory campaign (staged in `beta/block-15-campaign-wins/`, NOT a delivery patch).**
  Its wins are **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`),
  **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`,
  `GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache
  (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate;
  A/B with `beta/block-15-campaign-wins/ab/w4-revert.patch`), **V3** derived
  kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default — the packed mask is still
  created in every graph and simply loses its consumer, so the allocator
  leaves it unallocated), **V4** native q8_0 K/V and **V5** native bf16
  K/V in the FA kernels (both behind `GGML_CUDA_FA_KV_NATIVE`, **opt-in,
  default 0**: V4 costs ~1.7 % prefill — the lost `cp_async` pipeline —
  for −744/−632 MiB/GPU, V5 0.2–2.4 % for a bf16 cache to cost exactly
  what an f16 one does; the per-operand staging source is one shared type
  code `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}`, so the launcher, the alloc-size
  query and the kernels cannot disagree).  Two
  validation facts to protect: same-seed output is **byte-identical**
  across every gate combination on every model, and the adaptive-MTP gate
  is unchanged (27B 0.76744, qwen4exp 0.44262 = the block-14 baseline).
  RDNA3_5 (gfx1151) validated 2026-09-10: V3 now engages on a HIP iGPU and
  `kq_mask_derivable()` requires a single KV stream so a multi-slot context
  keeps the packed mask instead of aborting; the same-seed and MTP gates
  hold there, and V4 is *faster* at depth (+2.6 % pp20480, decode flat).
  Anything that touches the kq mask must still be validated on an **SWA**
  model (gemma-4-E4B / -31B).  Known pre-existing issue: gemma-4-E4B-it on
  3 GPUs with `-sm tensor` aborts in the meta splitter (2 KV heads < 3
  devices) — use 1/2 GPUs or `-sm layer`.
  **Revalidated 2026-09-11** against the 15-patch delivery (re-cut beta tip
  `fe4f55278`, tree `ffe197e2f`, base `389c5341f`): the dependency delta was
  exactly one file (`fattn-common.cuh`, block 00's `ntiles_dst_eff`), every
  2026-09-10 number reproduced to the last decimal, and the width probe
  reproduces the delivered reference hashes — see the beta `README.md` +
  `HANDOVER.md` §10.
- **The dense greedy-purity guarantee (`--spec-draft-n-max <= 7`) depends on the KV cache type.**
  It holds for **f16, bf16, q4_1, q5_0, q5_1 and iq4_nl**, but **NOT for a
  `q8_0` or `q4_0` K/V cache**: there `W=1 == W=2` and `W=3..8` agree,
  but the two groups differ (`W=2→3`, *not* block 00's `n_q <= 8`), and at
  the text level plain vs `draft-mtp` differ for real (27B, q8_0 KV:
  `8ed58aa9` vs `da56855b`).  This is **pre-existing** (bit-identical on a
  build without any block-15 code; `GGML_CUDA_FA_KV_NATIVE` on/off
  identical; reproduced on 1 GPU, so it is not the all-reduce) and it is a
  *trade*: the impure set was exactly the two types with a fast native
  both-quantized FA path (the rest stage through F16 and are ~3.4x
  slower).  **FIXED 2026-09-11** (block-08 amendment): the cause was the FA
  *kernel-family* chooser — VEC at `n_q <= 2` vs TILE from `n_q = 3` — not
  the KV staging, and the band is TILE throughout now, so `q8_0`/`q4_0` are
  width-pure on every split config (only `W=1,2` moved; MTP bit-identical,
  tg128 -0.5..-0.9 %).  `GREEDY-PURITY.md` §14.
  qwen4exp's two stacked causes (root-caused 2026-09-11) are now **half fixed**: its
  hyperconnection fusions (`hc-mix.cu`, gated `nt == 1`) were the cause-1 defect and the block-14
  2026-09-11 amendment routes the whole **decode/verify band `1 <= nt <= 8`** through them, so
  qwen4exp is now **width-pure for `W <= 4`** (`-sm layer` W=1..4 `3adeb313042a`, `-sm tensor`
  `dcf1ae66`, on f16/bf16; W=1 decode byte-identical to the pre-fix build for every KV type), plain
  == `draft-mtp --spec-draft-n-max 3` greedy text, f16 MTP acceptance 0.500 -> **0.76744** and MTP
  generation 63.3 -> 79.9 t/s.  Cause 2 (a kernel-dispatch band at `W >= 5`) is **still open**, so
  the band stops at `n_max 3` for qwen4exp — **until 2026-09-11**, when cause 2 was **fixed** as a
  block-13 amendment: the boundary was a **fusion-coverage** flip at `n_q = 5` (graphs are identical
  across widths) whose *mechanism* is upstream's **per-type mmvq cap** — `mul_mat_vec_q_moe`'s
  `__launch_bounds__` was `cap × warp_size` (so `ncols_dst > cap` cannot launch) and the same cap
  routes the upper band to MMQ through `mul_mat_q_pair`; the UD-IQ4_XS per-layer expert types
  (IQ3_S cap 4 / IQ4_XS cap 5 / IQ4_NL cap 7) predict the whole `{1..4}{5}{6,7}{8}` grouping.  The
  fix floors the cap at the band and sizes the kernel at it: `W = 1..8` is bit-identical on both
  splits (every width = that split's pre-fix `W = 1` value), **+14-26 %** at the verify widths, MTP
  `n_max 7` +16-18 % t/s, dense untouched.  The `n_max <= 7` guarantee now holds **logit-wise** for
  qwen4exp.  **Cause 3 (open): `plain` still != `draft-mtp` *text*** — pre-existing and independent
  (at `n_max 3`/`W = 4` the cause-2 fix is a verified no-op: byte-identical logits, text and
  acceptance), a **multi-step/roll-back** effect since the single-step probe is pure on both splits
  and with `RS=from_w`; **localised 2026-09-11 (further measurement): it is in the QSA *machinery*, and the site class is the same as cause 1's.**  `LLAMA_QSA_OFF=1` makes `plain` == `draft-mtp --spec-draft-n-max 3` **byte-identical** (`d4499ac8db72` both, 711 chars) — and the knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`) — while `LLAMA_QSA_SPARSE_FA=0` (dense attention, indexer still on) leaves two different texts (`25f300a81b9e` vs `0d466b2dcf09`), so the defect is **not** the sparse-FA kernel but the **indexer/score machinery** (`indexer-topk.cu` + the `qwen4exp.cpp` gates).  Both QSA-side `n_tokens == 1` gates are the prime suspects — `src/models/qwen4exp.cpp:1094` (`idx_score_fused`, the fused indexer score) and `:1419` (`qsa_dense_decode_until`, the early-decode dense shortcut) — i.e. exactly the cause-1 pattern, and the single-step width probe cannot see them because it never reaches the sparse/indexer decode regime.  The divergence appears only after ~100 chars (~20 tokens) of a 3.3k-prompt greedy run (the first steps agree), so it is not a prefill-state difference; `GGML_CUDA_GDN_CHUNKED=0` moves both sides without making them agree (the known Issue #25 chunked-prefill item is a separate contributor, not this).  **Kill-switch for users meanwhile: `LLAMA_QSA_OFF=1`.**
  **Differing K/V cache *types* are rejected** (maintainer
  decision 2026-09-11: mixed pairs are 1.7–3.6x slower than the same-type
  equivalent and never smaller).  Details, repro tooling and the follow-up
  items (F1 purity — **fixed 2026-09-11**; F2 qwen4exp — cause 2 **fixed 2026-09-11** (block-13
  amendment), cause 3 (plain vs `draft-mtp` text; see above) **open**; F3 sub-`q8_0` parity
  — note a native `iq4_nl` would be the same 1800 MiB as q4_0, pure, and
  3.4x faster; F3's first experiment is a `GGML_CUDA_FA_ALL_QUANTS=ON` build
  A/B, since the slow types are rejected by
  `ggml_cuda_fattn_kv_type_supported()` rather than missing a kernel):
  `GREEDY-PURITY.md` §12 and `wip/kv-quant-purity-followups/`.

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 9113cc188
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 14 commits
```

### Verify (the coherence gate — mandatory after any change)

```bash
HIP_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-cli -m ~/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn
```

Diff the output against a known-good build.  Same-seed output must be
IDENTICAL.

**`GGML_CUDA_ALLREDUCE=nccl` is NOT a bit-identical reference under
`-sm tensor`.**  The internal AR always BF16-round-trips
(`GGML_CUDA_AR_BF16_THRESHOLD` defaults to 1) while the NCCL path reduces small
tensors in FP32, so the two backends differ by design: measured 2-GPU tensor,
27B Q8_0, 300-token greedy `--spec-type none` -> text `6e8ccd25` (hybrid) vs
`6129e077` (nccl), and the token-0 logits differ too (W=6: `a4817ee6` vs
`73ff91bf`).  Treat it as a smoke comparison only.  For splits that do no
cross-device reduction (1 GPU, `-sm layer`) the two are identical, because the
AR backend is then never reached.

### Regenerate the patches (after fork changes)

`scripts/make-patches.sh` (defaults: fork `~/llama.cpp`, base `9113cc188`,
blocks tip `c6f1e8e78`): `git format-patch --start-number 0` the block
commits (all 15 blocks are committed fork commits; block 00 keeps the file
prefix `0000`; `git diff <base>..<tip>` yields
`rdna-boosts-all.patch`).  NOTE on the fork topology: **the working
`~/llama.cpp` checkout's `rdna-boosts` branch is NOT the canonical chain**
— it may be rebased onto a master two commits newer
than the fork point (`f3f1a8f27`, `304665fe7`), so a raw
`9113cc188..HEAD` range there exports those two upstream commits as patches
0001/0002.  The canonical 15-block chain is a rebuild of the delivery set at
`9113cc188` (tip `c6f1e8e78`), which is what the default tip names.  Always regenerate from a
canonical fork rebuilt AT `9113cc188`; a rebuilt fork produces its own
commit SHAs, so patch bodies stay identical but the `From <sha>` line and
the `[PATCH NN/15]` series count change.  Then
re-verify the clean-apply simulation (worktree at the fork point,
apply-all, build, coherence) before committing.

### Build the fork

```bash
cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# fast loop: cmake --build build-rocm-hybrid --target llama-cli llama-bench -j 16
# runtime libs: LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
```

The `EXTRA_CMAKE_FLAGS` override is required with CMake >= 4.3: the build
script hardcodes a bare `-DCMAKE_HIP_FLAGS="-mllvm"` (leftover of the
commented `-mllvm --amdgpu-unroll-threshold-local=600`), and CMake's HIP
compiler test now injects `--cuda-host-only` directly after it — the bare
`-mllvm` swallows it into LLVM option parsing and the configure aborts.

## What NOT to do

- Do not `git apply` the concatenated 01-13 series (drops hunks).
- Do not hand-edit the committed patches as a permanent drift fix —
  regenerate from the fork (`scripts/make-patches.sh`) and re-verify.
- Do not mix the historical `baseline/*` branches or `block/*` tags with the
  current `patches/` — they are different patch sets for different baselines.
- Do not push anything from the `~/llama.cpp` checkout — the fork branch
  is disposable and must be re-applied from the diff set, not pushed (see
  the Pushing policy above).  The only permitted push target outside this
  repo is the personal fork, and only on explicit maintainer request.
- Do not present old docs as current: MANIFESTS/BASELINE validation records
  are dated history; the current claims are the header sections + `patches/README.md`.
- Do not add new WIP experiments to the delivery patch set — WIP stays in
  `wip/` (or `archive/work/` once closed), env-gated OFF, excluded from
  `patches/`.
- **Never apply anything from `wip/`** (loose patches/diffs, experiment
trees, tools) to the fork or a llama.cpp checkout, and never fold `wip/`
content into the delivery — **unless the user explicitly asks for that
specific `wip/` item** (see the WIP rule under Critical facts).

## Editing the docs

The docs have a freshness problem by design (fast-moving project): the
historical records are kept, and the CURRENT state is stated in the header
sections (`patches/README.md`, `README.md`, the top of MANIFESTS/BASELINE).
When you change the delivery, update those headers; never edit the dated
validation records in place — add a new dated record instead.
Delivery-affecting changes (block amendments, community-fix integrations,
re-baselines, regenerations) get a dated entry at the top of `WORKLOG.md`
(newest first), and the README `Current state` section stays a lean summary
that points there rather than accumulating the record itself.  Session/dev
handovers belong under `wip/` or `archive/docs/`, not at the repo top level.
