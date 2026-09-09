# llama-cpp-rdna-boosts

A delivery repo for a **14-patch set** of **RDNA3 / RDNA3.5 / RDNA4**
(ROCm) feature enhancements and performance fixes for llama.cpp:
**blocks 01-11** (MTP, GDN, BF16 KV,
WMMA flash-attn, fused core, k-quant boosts, CUDA prefill-graph skip),
**block 12** (the hybrid HIP all-reduce; amended 2026-09-04 with a
runtime NCCL-failure fallback — see [Current state](#current-state)),
**block 13** (fused MoE gate+up+GLU MMQ + mmvq short-K item-split;
amended 2026-09-02 with two MTP regression fixes and 2026-09-05 with
the RDNA3.5 (Strix Halo, gfx1151) + RDNA3.0 (gfx1100) fused-MoE-MMQ
gate relaxations, and 2026-09-08 with the moe_weighted_reduction
float4 remainder fix (issue #19) — see
[Current state](#current-state)) and
**block 14** (qwen4exp / Qwen3.8-Flash-Next support, promoted from
`beta/qwen4exp` — QSA sparse FA + indexer, HC fused decode ops, managed
lazy reader, MTP draft-head, per-arch dense/QSA decode policy; amended
2026-09-07 with the QSA quantized-KV decode gate + the derived-cache
pool gate and 2026-09-08 with the MUL_MAT_ID pair-fusion layout gate
(issue #18), the compiler-warning cleanup, the qwen4exp tensor-split
HIP gate and the quantized-KV tensor-split gate (an upstream
multi-GPU `SPLIT_MODE_TENSOR` abort for `q4_1`-family KV cache types —
see
[Current state](#current-state)).
The patches apply to a clean
llama.cpp checkout at the recorded fork point `9113cc188` (re-based 2026-09-08 from `050dde50c`, itself re-based 2026-09-07 from `465e49b9c`, itself re-based 2026-09-06 from `9cffdcc80`, itself re-based 2026-09-02 from `0eadefebd`).

`scripts/apply-all.sh` automates the apply: it creates a fresh `rdna-boosts`
branch and applies blocks 01-14 with `git am`, one commit each.

## Supported architectures

The set targets the **RDNA3 / RDNA3.5 / RDNA4** GPU families:

| family | arches | example parts |
|--------|--------|---------------|
| RDNA 3 | `gfx1100` | RX 7900 XTX/XT, RX 7800 XT, ... |
| RDNA 3.5 | `gfx1150`/`gfx1151` | Strix Point / Strix Halo APUs |
| RDNA 4 | `gfx1200`/`gfx1201` | RX 9060 XT; RX 9070 / 9070 XT |

**RDNA4 (gfx120x) sees the most benefit** — the WMMA flash-attn path, the
chunked-GDN kernel, the k-quant VDR boosts and block 12's internal
all-reduce were all first built and validated there. As much of that work
as possible is back-ported to the RDNA3/3.5 families instead of being
gated off:

- block 02's **chunked gated-delta-net** bf16/WMMA prefill ships as two
  arch-segregated kernels: a dedicated first-gen WMMA port for gfx11
  (`gated_delta_net_chunked_bf16_gfx11.cu`) next to the RDNA4 kernel;
- block 04's **WMMA flash-attn** is *not* RDNA4-only despite the block
  name — RDNA3.0 runs it with the same 576-head limit as RDNA4, RDNA3.5
  with a tuned 320-head limit;
- block 10 adds a **dedicated RDNA3.5 mmvq parameter table** (previously
  folded into the RDNA2 fallback) on top of the RDNA4 k-quant boosts.

Arch selection is **runtime** everywhere in the set (device `cc` /
`gcnArchName`; there is no compile-time arch gating), so a multi-arch
build such as `GPU_TARGETS="gfx1100;gfx1151;gfx1201"` yields one binary
that picks the right path on whichever of these it runs on. The one
genuine exception is **block 12** — its internal all-reduce is RDNA4-only
(gfx1200/gfx1201) and falls back to RCCL elsewhere (see
`patches/README.md` for the gate and env knobs).  Block 13's fused
MoE MMQ gate now covers RDNA4 + RDNA3_5 + RDNA3_0 (gfx1151 validated
2026-09-05, gfx1100 validated 2026-09-05 — see
[Current state](#current-state)).

## Current state

The current delivery is a **14-patch set** for llama.cpp at the fork
point `9113cc188` (blocks 01-14 in `patches/`, applied with `git am` via
`scripts/apply-all.sh`; block-14 tip `27485f1ca`, block 01 refreshed
2026-09-09 to the llama.cpp PR #27210 review head and block 14 amended
2026-09-09 with the gfx1151-only freed-cell KV-zeroing gate — regenerated
2026-09-09).  The set applies **whitespace-clean** and each block is
build- and coherence-verified — see [`MANIFESTS.md`](MANIFESTS.md) (apply
order + verification contract), [`patches/README.md`](patches/README.md)
(per-block notes, env knobs, server config) and
[`BASELINE.md`](BASELINE.md) (fork point + drift policy).

All delivery-affecting changes (block amendments, community-fix
integrations, re-baselines, regenerations) are tracked as dated entries
— newest first — in **[`WORKLOG.md`](WORKLOG.md)**; the current-state
summary below is deliberately short and does not repeat them.

- **Latest entry (2026-09-09): block-14 gfx1151-only freed-cell
  KV-zeroing gate (regeneration `7c4d9c4e0..27485f1ca`).**  Block 14's
  seq_rm row zeroing (the strix-lineage masked-column guard for the
  gfx1151 WMMA f16 `x+(-0.0)` inexactness) now enables only on gfx1151
  devices (env `LLAMA_KV_ZERO_FREED` overrides).  Off gfx1151 the
  pre-block-14 behavior is restored: replacing a resident multi-GPU KV
  sequence no longer issues ~48xN synced per-cell memsets — the ~18-24 s
  pre-prefill stall on 3x R9700 gfx1201 (qwen4exp and plain dense 4B) is
  gone (identical workload 24.5 s -> ~6 s), and the zeroing-off
  determinism gate is clean.  gfx1151 (Halo box) keeps the zeroing
  enabled; 16-run control unchanged.  Full record in
  [`WORKLOG.md`](WORKLOG.md).

- **2026-09-09 (previous): block-01 refresh to the PR #27210 review
  head.**  Block 01 (adaptive MTP draft depth) was cut from llama.cpp PR
  #27210 (author: stew675) at its `0994374fd` state; the PR advanced
  through a maintainer review round and the block is refreshed to the PR
  head `d236d41a2`, still one squashed patch.  The review changeset adds
  the `accept_partial()` feedback path (checkpoint-restore rounds can no
  longer feed stale accept counts to the adaptive controller),
  `has_mtp()` refactor, adaptive-depth reset before the empty-prompt
  early return, `--spec-draft-n-min-adaptive` value validation + docs,
  and dual-MTP-type rejection.  Blocks 02-14 content-identical.
  Verification: clean-apply sim strict 14/14 `git am`, zero whitespace
  warnings, applied tree == fork tip `0f2b7a4e1`; unit tests pass;
  plain-decode same-seed coherence token-IDENTICAL to the known-good
  `050ec89ce` build.  Full record in [`WORKLOG.md`](WORKLOG.md).

- **2026-09-08 (previous): re-base onto master `9113cc188`.**
  The fork was 14 commits behind upstream; upstream had itself reverted
  #24233 in #28604 on 2026-09-08, matching block 06's end state, so the
  re-base reduced block 06 to a host-buffer rationale marker (kept for
  numbering/history) and merged block 14's quantized-KV tensor-split gate
  additively with upstream #28390's single-device `SPLIT_MODE_TENSOR`
  warn.  Blocks 01-05 + 07-13 are content-identical to the previous
  delivery.  Clean-apply sim: strict 14/14 `git am`, zero whitespace
  warnings, applied tree == fork tip `78e67a3d8`.  Verified on the Strix
  box (gfx1151, ROCm 7.14): plain-decode same-seed coherence IDENTICAL to
  the `72f0ee944` build across tensor/layer, f16/q8_0/bf16 KV and depth
  16384; MTP adaptive gate acceptance 0.833 with draft-mtp 20.3 t/s vs
  plain 7.9 t/s.  Full record in [`WORKLOG.md`](WORKLOG.md).

- **2026-09-08 (previous):** block-14 quantized-KV tensor-split gate
  — `q4_1`-family KV cache types (`q4_1/q5_0/q5_1/iq4_nl`) aborted at
graph reserve under multi-GPU `SPLIT_MODE_TENSOR` on dense qwen35 and
qwen4exp.  Root cause is an **upstream bug** (reproduced on pristine
vanilla llama.cpp at the fork point `050dde50c`, unfixed on current
master): tensor split forces flash attention, whose kernels read the
quantized K/V natively only for `q4_0`/`q8_0`; the q4_1-family attention
subgraph becomes MIRRORED graph-external leaves that collide with the
AXIS-0 gate branch of the qwen35 gated attention.  The amendment rejects
those KV types at context creation with a clear error when the Meta
device is in use (layer split, single-GPU and `q8_0/q4_0`/float types
are unaffected).  Canonical fork rebuilt at `050dde50c`
(`d65a96084..ce641322e`), set regenerated, clean-apply sim re-verified
2026-09-08 (14/14 `git am`, zero whitespace warnings, applied tree ==
fork tip `ce641322e`).  Full record in
[`WORKLOG.md`](WORKLOG.md).

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # block 10 decode-variance analysis (read before shipping)
├── WORKLOG.md             # dated delivery records (newest first; README points here)
├── rdna-boosts-all.patch  # convenience: the entire 14-patch net as ONE patch
├── patches/               # the delivery set: 0001-0014
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── wip/                   # exploration docs + tuning tools + HANDOFF (session log)
└── archive/               # the rest: archive/work/ (closed experiments) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 14 blocks

| patch | what |
|-------|------|
| `0001` | adaptive MTP draft depth (`--draft-mtp-adaptive`) |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA, arch-segregated gfx12/gfx11) |
| `0003` | BF16 KV cache + native-BF16 flash-attn |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf (WMMA path also runs on RDNA3.0/3.5, tuned head limits) |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04; amended 2026-09-07 with the mul_mat+add through-view shape guard, PR #15) |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions; adds a dedicated RDNA3.5 mmvq table) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate (bounded in-kernel spin since 2026-08-30 fix round; builds without RCCL) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — prefill fused expert MMQ (RDNA4 + RDNA3_5 + RDNA3_0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K, env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) + decode item-split (rpb 2/4/8) merged with the upstream has_fusion mmvq path |
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA (default) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy (promoted from `beta/qwen4exp`; see `patches/README.md` block-14 notes) |

> **Greedy-purity note (read before shipping):** on the K-split decode
> paths, block 10 (`0010`) is the only patch that changes decode numerics on
> ANY architecture — its VDR kernels reorder the fp32 reduction. Compute
> outputs are not bit-identical to a build without it (max logit diff 0.184
> vs 0.203 for flash-attn on/off; greedy streams are deterministic within a
> build but can flip across configs). This is a different rounding path, not
> a correctness change. If you require 100% greedy purity across builds, do
> not install `0010-…k-quant-boosts…patch` — it is one line to drop from
> `scripts/apply-all.sh`. Full discussion:
> [`GREEDY-PURITY.md`](GREEDY-PURITY.md). **Block-13 caveat (2026-09-02):**
> block 13 rewrites the small-batch mmvq decode kernel and is a second
> decode-numerics source on the rows that run it (short-K K<4096 ncols==1
> rows, MoE projections; ncols 2..8 and long-K rows were restored to the
> pre-block-13 K-split kernel by the 2026-09-02 fix). Excluding block 10 no
> longer reproduces stock bits exactly on those rows — see
> GREEDY-PURITY.md §9.

## Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 9113cc188        # the SHA recorded in patches/README.md

# 2. apply the set (automated; VERIFIED 2026-08-29, re-verified 2026-09-01/02/05/06/07 and 2026-09-08)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0001…0014  (one commit per block on a fresh `rdna-boosts` branch)

# 3. build + verify (trim -DGPU_TARGETS to your GPU arch for a faster build)
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# coherence gate (same-seed output must match a known-good build):
./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 -p "The capital of France is" \
  -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn
```

> **Do not use `git apply` on the concatenated 1-11 series** — it silently
> drops hunks (30 files / 2483 lines vs the correct 35 / 6094, verified
> 2026-08-29). `git am` (or `scripts/apply-all.sh`) is the required flow.

### Manual equivalent

```bash
git am patches/000[1-9]-*.patch patches/001[0-4]-*.patch   # blocks 01-14
git add -A && git commit -m "rdna-boosts: block 14: qwen4exp support"
```

## When upstream master moves

The patches are static against `9113cc188`. When upstream drifts and hunks
no longer apply, regenerate the whole set from the fork with
`scripts/make-patches.sh` (needs the `~/llama.cpp` fork checkout, which
carries the block commits), then update
`patches/README.md` and this README with the new fork point. The old
`baseline/<sha>`-branch-per-upstream-range workflow was retired when the
delivery moved to the flat 13-patch set on `main`.

## Upstreaming

Some blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. Block 12's internal all-reduce is gated to
RDNA4 pending community verification on RDNA3 pairs. See `MANIFESTS.md`
for per-block verification and `BASELINE.md` for provenance.


## Community Acknowledgements

This work is becoming a community effort and I'd like to offer special thanks to the
following users for the assistance in finding issues and offering solutions!

- https://github.com/1337hero
- https://github.com/bakon11
- https://github.com/briansp2020  (block-13 moe_weighted_reduction float4 remainder fix + block-14 MUL_MAT_ID pair-fusion layout gate, issues #19 and #18)
- https://github.com/eoprede
- https://github.com/tungel
- https://github.com/DanoPTT  (block-08 mul_mat+add through-view shape guard, PR #15)

I, and everyone else who benefits from this work, really appreciate you!

## Inspirational Works

While most of the work in this repository are original works of my own, there are
some significant portions, most notably around the prefill tuning, inspired by the
excellent work performed by the community of: https://github.com/halo-box/strix-llama.cpp

Thank you to all the maintainers of the Strix Halo Llama.cpp project

Of course none of this would be possible without the baseline that all of this rests
on, and that is the huge community over at https://github.com/ggml-org/llama.cpp

Many thanks to the llama.cpp team

## License

Same as llama.cpp (MIT).
