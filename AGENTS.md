# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing
anything in `~/llama-cpp-rdna-boosts/` (or acting on its behalf).

## What this repo is

A **delivery repo**: it packages the RDNA/ROCm work of the
[`stew675/llama.cpp`](https://github.com/stew675/llama.cpp) fork
(`rdna-boosts` branch) as a **12-patch set** that applies to a clean
llama.cpp checkout at the fork point **`a7cc83bba`** (re-based 2026-08-30; previously `17252c769`).

- Blocks **01-11** (`patches/0001-…0011-…`): MTP draft depth, fused chunked
  GDN, BF16 KV, WMMA flash-attn, CPU bit-identical decode, host-buffer
  revert, meta wrapper skip, fused core, meta headroom, k-quant boosts,
  CUDA prefill-graph skip.
- Block **12** (`patches/12-hybrid-allreduce-hip.patch`): the hybrid HIP
  all-reduce (custom internal AR for the small-tensor decode path +
  per-size hybrid dispatch vs RCCL), **RDNA4-only** (gfx1200/gfx1201; falls
  back to RCCL elsewhere). The fused-stage/pacing experiments it spawned are
  archived, env-gated OFF, in `archive/work/fused-stage-pacing/`.

The repo is NOT the fork: the fork (source of truth for the block commits)
lives at `~/llama.cpp`, branch `rdna-boosts` — re-based 2026-08-30 onto
upstream master `a7cc83bba` (12-commit branch; blocks tip `8fbf10e5b`,
block 12 committed as `4fa92f0ae`; the 2026-08-29 whitespace-clean
rebuild `12d10267b`/`cc985ba9a` against `17252c769` is superseded). The
historical fork state
(block commits `2b7a135cb..f6f8f6778` + the archived WIP commits, tip
`155debcdc`) is preserved on the `old-rdna-boosts` branch.

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
| `patches/` | **the delivery set** (0001-0011 + 12-hybrid) + apply README |
| `scripts/apply-all.sh` | the verified apply flow (git am 1-11, git apply 12) |
| `scripts/make-patches.sh` | regenerates the set from the fork |
| `rdna-boosts-all.patch` | the entire 12-patch net as ONE patch (fork point only) |
| `benchmarks/` | dated benchy/v1/v2 records + methodology + graphs |
| `wip/` | exploration docs, tuning tools, HANDOFF (session log) |
| `archive/docs/` | moved-out historical records (validation history, baseline history) — reference only |
| `archive/work/` | closed experiments, preserved for future re-evaluation |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |

## Critical facts (do not re-derive)

- **Apply method:** blocks 01-11 with **`git am`**, block 12 with `git apply`.
  Plain `git apply` of the concatenated 01-11 series **silently drops
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
  whitespace warnings (the 8 inert trailing-whitespace lines + 1 EOF blank
  line were removed at the source 2026-08-29 and the set regenerated;
  trees are otherwise byte-identical — verified).
- **Verified numbers (2026-08-29):** clean-apply build tg64 38.12 / tg512
  41.08; depth-16384 3-GPU hybrid 38.71 t/s (unpinned); 2-GPU (1,2) 31.79.
- The one-sided AR wait (dev0/bus-06 dispatch-gap asymmetry, ~12.7 µs/call)
  is a **platform-level CP/driver property**, not reachable from the AR
  kernel, graph tail, or host-side pacing — fusion/pacing are CLOSED
  (`archive/work/fused-stage-pacing/`).

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a7cc83bba
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 12 commits
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

`scripts/make-patches.sh` (defaults: fork `~/llama.cpp`, base `a7cc83bba`,
blocks tip `8fbf10e5b`): `git format-patch` the block commits + the
block-12 delta for the 12th (block 12 is committed as `4fa92f0ae`; the
script's `git diff <blocks-tip>` picks it up from the clean tree). Then
re-verify the clean-apply simulation (worktree at the fork point,
apply-all, build, coherence) before committing.

### Build the fork

```bash
cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid ~/bin/build-llama-rocm-714
# fast loop: cmake --build build-rocm-hybrid --target llama-cli llama-bench -j 16
# runtime libs: LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
```

## What NOT to do

- Do not `git apply` the concatenated 01-11 series (drops hunks).
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

## Editing the docs

The docs have a freshness problem by design (fast-moving project): the
historical records are kept, and the CURRENT state is stated in the header
sections (`patches/README.md`, `README.md`, the top of MANIFESTS/BASELINE).
When you change the delivery, update those headers; never edit the dated
validation records in place — add a new dated record instead.
