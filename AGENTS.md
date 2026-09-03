# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing
anything in `~/llama-cpp-rdna-boosts/` (or acting on its behalf).

## What this repo is

A **delivery repo**: it packages the RDNA/ROCm work of the
[`stew675/llama.cpp`](https://github.com/stew675/llama.cpp) fork
(`rdna-boosts` branch) as a **13-patch set** that applies to a clean
llama.cpp checkout at the fork point **`9cffdcc80`** (re-based 2026-09-02 from `0eadefebd`).

- Blocks **01-11** (`patches/0001-…0011-…`): MTP draft depth, fused chunked
  GDN, BF16 KV, WMMA flash-attn, CPU bit-identical decode, host-buffer
  revert, meta wrapper skip, fused core, meta headroom, k-quant boosts,
  CUDA prefill-graph skip.
- Block **12** (`patches/0012-rdna-boosts-block-12-hybrid-HIP-all-reduce-RDNA4-gat.patch`): the hybrid HIP
  all-reduce (custom internal AR for the small-tensor decode path +
  per-size hybrid dispatch vs RCCL), **RDNA4-only** (gfx1200/gfx1201; falls
  back to RCCL elsewhere). The fused-stage/pacing experiments it spawned are
  archived, env-gated OFF, in `archive/work/fused-stage-pacing/`.
- Block **13** (`patches/0013-…-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch`): fused MoE gate+up+GLU MMQ (prefill)
  + mmvq short-K item-split (decode); see the block-13 notes in `patches/README.md`.
  Amended 2026-09-04 with two regression fixes folded into the block: (1) the
  mmvq item-split/rpb kernel collapse of multi-token decode batches (ncols 2..8,
  the speculative verify step — dense MTP 18.3 -> 27.5 t/s, ksplit dispatch);
  (2) the rms_norm->mmvq Q8_1-cache fold corrupting multi-token MUL_MAT_ID
  (MoE MTP acceptance 0 -> 0.51, draft-mtp 53 -> 126 t/s, fold gated to
  single-token MMID).  Details + numbers: `patches/README.md` block-13 notes.

The repo is NOT the fork: the fork (source of truth for the block commits)
lives at `~/llama.cpp`, branch `rdna-boosts` — rebuilt 2026-09-02 onto
upstream master `9cffdcc80` from the delivery patches, block 13 amended
2026-09-04 with the two MTP regression fixes (13-commit branch;
blocks `04122bfb5..8f2838d1`). That checkout is disposable and is
re-created from `patches/` + `scripts/apply-all.sh` whenever it needs
rebuilding (fresh clone at the fork point + apply). Older fork states are
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
| `patches/` | **the delivery set** (0001-0013) + apply README |
| `scripts/apply-all.sh` | the verified apply flow (git am for blocks 01-13) |
| `scripts/make-patches.sh` | regenerates the set from the fork |
| `rdna-boosts-all.patch` | the entire 13-patch net as ONE patch (fork point only) |
| `benchmarks/` | dated benchy/v1/v2 records + methodology + graphs; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) |
| `wip/` | exploration docs, tuning tools, session handoffs — **NOT part of the delivery** (see the WIP rule below) |
| `archive/docs/` | moved-out historical records (validation history, baseline history) — reference only |
| `archive/work/` | closed experiments, preserved for future re-evaluation |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |

## Critical facts (do not re-derive)

- **Apply method:** all 13 blocks with **`git am`** (each block is a
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
  whitespace warnings (re-verified 2026-09-01 on `0eadefebd` and
  2026-09-02 on the `9cffdcc80` re-base).
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
- **Verified numbers (2026-09-02 re-base, unchanged):** clean-apply build
  tg64 38.12 / tg512 41.08; depth-16384 3-GPU hybrid 38.71 t/s
  (unpinned); 2-GPU (1,2) 31.79.  The re-base is content-identical plus
  upstream's additions (42 commits, 2026-09-02) — numbers carry over.
- **Block-13 MTP regression fixes (2026-09-04, folded into block 13):**
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

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 9cffdcc80
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 13 commits
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

`scripts/make-patches.sh` (defaults: fork `~/llama.cpp`, base `9cffdcc80`,
blocks tip `8f2838d1`): `git format-patch` the block commits (all 13
blocks are committed fork commits; `git diff <base>..<tip>` yields
`rdna-boosts-all.patch`). Then
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
