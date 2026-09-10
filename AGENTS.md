# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing
anything in `~/llama-cpp-rdna-boosts/` (or acting on its behalf).

## What this repo is

A **delivery repo**: it packages the RDNA/ROCm work of the
[`stew675/llama.cpp`](https://github.com/stew675/llama.cpp) fork
(`rdna-boosts` branch) as a **14-patch set** that applies to a clean
llama.cpp checkout at the fork point **`9113cc188`** (re-based 2026-09-08
from `050dde50c`, itself re-based 2026-09-07 from `465e49b9c`, itself
re-based 2026-09-06 from `9cffdcc80`, re-based 2026-09-02 from `0eadefebd`).

- Blocks **01-11** (`patches/0001-…0011-…`): MTP draft depth, fused chunked
  GDN, BF16 KV, WMMA flash-attn, CPU bit-identical decode, host-buffer
  revert, meta wrapper skip, fused core, meta headroom, k-quant boosts,
  CUDA prefill-graph skip.
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
- Block **14** (`patches/0014-rdna-boosts-block-14-qwen4exp-support.patch`):
  qwen4exp / Qwen3.8-Flash-Next support, promoted from `beta/qwen4exp`
  2026-09-07 — QSA sparse FA (default) + fused indexer top-k/score,
  HC_MIX/HC_COMBINE fused decode ops, managed lazy reader + PLE n-gram
  loading, MTP draft-head, WS4 hyperconn prefill fusions, per-arch
  dense/QSA decode policy; **amended 2026-09-10 with the kernel-side
  masked-V fixes, and the 2026-09-09 gfx1151-only freed-cell host
  zeroing it replaces is REMOVED** (`llama-kv-cache.{cpp,h}` back to the
  upstream state; no `zero_freed`/env `LLAMA_KV_ZERO_FREED`/per-free GPU
  memsets).  In its place block 14 carries three unconditional kernel
  fixes that keep masked (freed/stale) flash-attention cells at exactly
  +0.0 on every device: HIP `fattn-tile.cuh` (packed-bf16 PV), HIP
  `fattn-mma-f16.cuh` (masked-V rows in staged shared tiles), Vulkan
  `flash_attn_cm1.comp` + `flash_attn.comp` (dead columns never read V) —
  see the 2026-09-10 block-14 amendment section in
  `patches/README.md`); see the block-14 notes in
  `patches/README.md`
  and the beta validation record in `beta/qwen4exp/README.md`.

The repo is NOT the fork: the fork (source of truth for the block commits)
lives at `~/llama.cpp`, branch `rdna-boosts` — currently the 14 block
commits on master `9113cc188` (2026-09-08 re-base; block 01 refreshed
2026-09-09 to the upstream PR #27210 review head `d236d41a2`, still one
squashed block; block 14 amended 2026-09-10 with the kernel-side
masked-V fixes — the 2026-09-09 gfx1151-only freed-cell host zeroing is
removed; block-14 tip `ff2b35f49` on the local regeneration; on the re-base block 06 was
reduced to a host-buffer
rationale marker — upstream itself reverted #24233 in #28604 on
2026-09-08, matching its end state, so the functional delta is now
upstream (see the WORKLOG re-base entry); block 12 carries the
2026-09-04 runtime NCCL-failure fallback, issue #13; block 13 amended
2026-09-02/09-05/09-06 as above and 2026-09-08 with the
moe_weighted_reduction float4 remainder fix (issue #19, reported by
briansp2020); block 14 added 2026-09-07 and amended
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
of the meta-splitter abort found on Vulkan); block 08
amended 2026-09-07 with the PR #15 mul_mat+add through-view shape
guard). The
canonical `9113cc188` fork used for `make-patches.sh`
regeneration is disposable and is re-created from `patches/` +
`scripts/apply-all.sh` whenever it needs rebuilding (fresh clone at the
fork point + apply) — the last regeneration (2026-09-10) updated
block 14 only, tip `ff2b35f49` (amended with the kernel-side masked-V
fixes; the 2026-09-09 gfx1151-only freed-cell KV-zeroing gate is
removed); blocks 01-13 patch files are byte-identical to the previous
full regeneration `7c4d9c4e0..27485f1ca`. Older fork states are
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
| `GREEDY-PURITY.md` | block-10 decode-variance analysis (read before shipping) |
| `patches/` | **the delivery set** (0001-0014) + apply README |
| `scripts/apply-all.sh` | the verified apply flow (`git am` blocks 01-14, automatic `git am -3` fallback on a drifted base) |
| `scripts/make-patches.sh` | regenerates the set from the fork |
| `rdna-boosts-all.patch` | the entire 14-patch net as ONE patch (fork point only) |
| `benchmarks/` | dated benchy/v1/v2 records + methodology + graphs; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) |
| `wip/` | exploration docs, tuning tools, session handoffs — **NOT part of the delivery** (see the WIP rule below) |
| `beta/` | **promoted-from-WIP staging** (e.g. `beta/qwen4exp/` = qwen4exp support + its validation record; `qwen4exp-support.patch` promoted into the delivery as block 14).  `beta/block-15-campaign-wins/` is where the memory campaign's validated wins are collected and gated for the **Block 0015** beta patch — see the WIP rule below |
| `upstream/` | **upstream-PR candidates** — self-contained changes that could be filed against unadulterated `ggml-org/llama.cpp` master, each with a `UPSTREAM-PR-*.md` note + `.patch` (see its README for the double-apply caution and the status table) |
| `archive/docs/` | moved-out historical records (validation history, baseline history) — reference only |
| `archive/work/` | closed experiments, preserved for future re-evaluation |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |

## Critical facts (do not re-derive)

- **Apply method:** all 14 blocks with **`git am`** (each block is a
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
- **Block 02 (0002) now also carries the MTP chunked-prefix dispatch
  (PR #9, 2026-09-01):** long single-sequence MTP prefills (`K > 1`,
  `n_seqs == 1`, `n_tokens > K+64`) run the chunked WMMA GDN on the
  prefix (`n_tokens - K`) and sequential GDN only on the last K snapshot
  slots.  Fired + verified on 3x R9700 (2-GPU, internal AR, Qwen3.8-27B
  Q8, ubatch 1024, MTP n-max 3): +7.5% prefill at ~5.5k prompt, +7.7% at
  ~38k; 64-token same-seed output token-identical to sequential.  Opt
  out: `GGML_CUDA_GDN_CHUNKED=0` (also `GGML_CUDA_GDN_CHUNKED_BF16=0`).
  Bench record: `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
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
  Protocol A there (acceptance must stay > ~0.45, MTP >= plain at depth 3)
  before relying on llama-bench numbers.
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

Diff the output against a known-good build (or against RCCL via
`GGML_CUDA_ALLREDUCE=nccl`). Same-seed output must be IDENTICAL.

### Regenerate the patches (after fork changes)

`scripts/make-patches.sh` (defaults: fork `~/llama.cpp`, base `9113cc188`,
blocks tip `ff2b35f49`): `git format-patch` the block commits (all 14
blocks are committed fork commits; `git diff <base>..<tip>` yields
`rdna-boosts-all.patch`).  NOTE on the current fork topology: `~/llama.cpp`
`rdna-boosts` is synced AT the fork point (upstream master `9113cc188`
+ the 14 blocks re-applied, block-14 tip
`ff2b35f49` after the 2026-09-10 amendment; the previous full
regeneration was `7c4d9c4e0..27485f1ca`), so a raw
`9113cc188..HEAD` range there is exactly the 14 block commits — but the
fork branch is disposable, so the patches
must still be generated from a canonical fork rebuilt AT `9113cc188`
(`scripts/apply-all.sh` of the current delivery; the last regeneration,
2026-09-10, updated block 14 only to tip
`ff2b35f49` — block-01/13 content untouched, block 14 amended
2026-09-10 with the kernel-side masked-V fixes, replacing the
2026-09-09 gfx1151-only freed-cell KV-zeroing gate).  Then
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
