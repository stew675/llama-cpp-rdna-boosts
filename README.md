# llama-cpp-rdna-boosts

A delivery repo for a **13-patch set** of **RDNA3 / RDNA3.5 / RDNA4**
(ROCm) feature enhancements and performance fixes for llama.cpp:
**blocks 01-11** (MTP, GDN, BF16 KV,
WMMA flash-attn, fused core, k-quant boosts, CUDA prefill-graph skip),
**block 12** (the hybrid HIP all-reduce; amended 2026-09-04 with a
runtime NCCL-failure fallback — see [Current state](#current-state)) and
**block 13** (fused MoE gate+up+GLU MMQ + mmvq short-K item-split;
amended 2026-09-02 with two MTP regression fixes and 2026-09-05 with
the RDNA3.5 (Strix Halo, gfx1151) fused-MoE-MMQ gate relaxation — see
[Current state](#current-state)).
The patches apply to a clean
llama.cpp checkout at the recorded fork point `9cffdcc80` (re-based 2026-09-02 from `0eadefebd`).

`scripts/apply-all.sh` automates the apply: it creates a fresh `rdna-boosts`
branch and applies blocks 01-13 with `git am`, one commit each.

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
`patches/README.md` for the gate and env knobs). Block 13's fused
MoE MMQ additionally excludes RDNA3.0 (gfx1100) until validated there —
RDNA3.5 (Strix Halo) was validated 2026-09-05 (see
[Current state](#current-state)).

## Current state (2026-09-05)

- **Block-13 RDNA3.5 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps were RDNA4-only; validated on Strix Halo (Ryzen AI MAX+ 395 /
  Radeon 8060S, gfx1151, ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M
  (ub 2048): same-seed coherence IDENTICAL fused-on vs off, prefill
  gains match RDNA4 (pp2048 +5.3% 1590 -> 1674, pp16384 +4.6% 1360 ->
  1423), decode unchanged (tg128 71.5). The RDNA4-tuned J caps
  transfer (uncapping regressed pp2048 1674 -> 1111 / pp16384 1423 ->
  1334). Set regenerated from a canonical fork rebuilt at `9cffdcc80`
  (13 am-commits, block-13 tip `ace0a5d54`); clean-apply sim verified.
  Full record:
  [`benchmarks/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`](benchmarks/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md).
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
  2026-09-02 with the two MTP regression fixes and 2026-09-05 with the
  RDNA3.5/Strix Halo fused-MoE-MMQ gate relaxation (the set was
  regenerated 2026-09-05 from a canonical fork rebuilt at `9cffdcc80`;
  block-13 tip `ace0a5d54`); the clean-apply sim at `9cffdcc80` applies
  with zero conflicts/whitespace warnings and its tree is byte-identical
  to the fork tip.
- **Fork point (baseline):** llama.cpp master at `9cffdcc80` (re-based
  2026-09-02 from `0eadefebd`; 42 commits of drift — see `MANIFESTS.md`
  for the dated re-base record, incl. the block 03/08/13 manual merges
  vs upstream's #27970 (sparse-fa) and #25952 (fused MoE expert
  reduction)).
- **Set:** 13 patches in `patches/` (`0001`-`0013`).
- **Verified:** clean apply + full build + llama-cli same-seed coherence
  IDENTICAL (hybrid vs RCCL, 3-GPU) on the rebuilt fork; the clean-apply
  sim at `9cffdcc80` applies with zero conflicts/whitespace warnings and
  its tree is byte-identical to the fork tip (`ace0a5d54`). tg64 38.12 /
  tg512 41.08 and the block-13 numbers are unchanged — the re-base is
  content-identical plus upstream's additions.
- **Whitespace-clean apply:** the regenerated set applies with **zero git
  whitespace warnings** (`git am` 01-13; re-verified 2026-09-02 on a
  fresh checkout at `9cffdcc80`, re-verified 2026-09-04 after the
  block-12 amendment, re-verified 2026-09-05 after the block-13 RDNA3.5
  gate relaxation).
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

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # block 10 decode-variance analysis (read before shipping)
├── rdna-boosts-all.patch  # convenience: the entire 13-patch net as ONE patch
├── patches/               # the delivery set: 0001-0013
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am for blocks 01-13)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── wip/                   # exploration docs + tuning tools + HANDOFF (session log)
└── archive/               # the rest: archive/work/ (closed experiments) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 13 blocks

| patch | what |
|-------|------|
| `0001` | adaptive MTP draft depth (`--draft-mtp-adaptive`) |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA, arch-segregated gfx12/gfx11) |
| `0003` | BF16 KV cache + native-BF16 flash-attn |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf (WMMA path also runs on RDNA3.0/3.5, tuned head limits) |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04) |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions; adds a dedicated RDNA3.5 mmvq table) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate (bounded in-kernel spin since 2026-08-30 fix round; builds without RCCL) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — prefill fused expert MMQ (RDNA4, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K, env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) + decode item-split (rpb 2/4/8) merged with the upstream has_fusion mmvq path |

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
git checkout 9cffdcc80        # the SHA recorded in patches/README.md

# 2. apply the set (automated; VERIFIED 2026-08-29, re-verified 2026-09-01 and 2026-09-02)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0001…0013  (one commit per block on a fresh `rdna-boosts` branch)

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
git am patches/000[1-9]-*.patch patches/001[0-3]-*.patch   # blocks 01-13
git add -A && git commit -m "rdna-boosts: block 13: fused MoE gate+up+GLU MMQ + mmvq short-K item-split"
```

## When upstream master moves

The patches are static against `9cffdcc80`. When upstream drifts and hunks
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
- https://github.com/briansp2020
- https://github.com/eoprede
- https://github.com/tungel

I, and everyone else who benefits from this work, really appreciate you!

## License

Same as llama.cpp (MIT).
