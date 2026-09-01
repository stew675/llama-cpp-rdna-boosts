# llama-cpp-rdna-boosts

A delivery repo for a **13-patch set** of **RDNA3 / RDNA3.5 / RDNA4**
(ROCm) feature enhancements and performance fixes for llama.cpp:
**blocks 01-11** (MTP, GDN, BF16 KV,
WMMA flash-attn, fused core, k-quant boosts, CUDA prefill-graph skip),
**block 12** (the hybrid HIP all-reduce) and
**block 13** (fused MoE gate+up+GLU MMQ + mmvq short-K item-split).
The patches apply to a clean
llama.cpp checkout at the recorded fork point `0eadefebd` (re-based 2026-09-01; previously `a7cc83bba`).

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
`patches/README.md` for the gate and env knobs).

## Current state (2026-09-01)

- **Fork point (baseline):** llama.cpp master at `0eadefebd` (re-based
  2026-09-01 from `a7cc83bba`; 22 commits of drift — see `MANIFESTS.md`
  for the dated re-base record, incl. the block-08 3-way merge vs
  upstream's #25635/#27466/#27621 CUDA changes).
- **Set:** 13 patches in `patches/` (`0001`-`0013`).
- **Verified:** clean apply + full build + llama-cli same-seed coherence
  IDENTICAL to the fork build; tg64 38.12 / tg512 41.08 on the sim build
  (2026-09-01 re-base; numbers unchanged — the re-base is code-identical
  to the 2026-08-30 set). Block 13 re-verified 2026-09-01: clean apply,
  applied tree byte-identical to the fork tip; 1-GPU qwen35moe prefill
  +5.1% (Q6_K) / +3.6% (Q4_K_M), decode +5.6%, coherence IDENTICAL.
- **Whitespace-clean apply:** the regenerated set applies with **zero git
  whitespace warnings** (`git am` 01-13; re-verified
  2026-09-01 on a fresh checkout at `0eadefebd`).
- **Deployment:** 3-GPU hybrid (`HIP_VISIBLE_DEVICES=0,1,2`, unpinned) gives
  depth-16384 decode 38.71 t/s (+21.8% vs 2-GPU). See
  [`patches/README.md`](patches/README.md) for block-12 env knobs and the
  server config.
- **RDNA4-only gate:** block 12 refuses to init off gfx1200/gfx1201 and
  falls back to RCCL (community RDNA3 verification pending).
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
├── rdna-boosts-all.patch  # convenience: the entire 12-patch net as ONE patch
├── patches/               # the delivery set: 0001-0013
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am 1-11, git apply 12)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── wip/                   # exploration docs + tuning tools + HANDOFF (session log)
└── archive/               # the rest: archive/work/ (closed experiments) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 12 blocks

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

> **Greedy-purity note (read before shipping):** block 10 (`0010`) is the
> only patch that changes decode numerics on ANY architecture — its VDR
> kernels reorder the fp32 reduction. Compute outputs are not bit-identical
> to a build without it (max logit diff 0.184 vs 0.203 for flash-attn
> on/off; greedy streams are deterministic within a build but can flip
> across configs). This is a different rounding path, not a correctness
> change. If you require 100% greedy purity across builds, do not install
> `0010-…k-quant-boosts…patch` — it is one line to drop from
> `scripts/apply-all.sh`. Full discussion: [`GREEDY-PURITY.md`](GREEDY-PURITY.md).

## Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 0eadefebd        # the SHA recorded in patches/README.md

# 2. apply the set (automated; VERIFIED 2026-08-29, re-verified 2026-09-01)
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

The patches are static against `0eadefebd`. When upstream drifts and hunks
no longer apply, regenerate the whole set from the fork with
`scripts/make-patches.sh` (needs the `~/llama.cpp` fork checkout, which
carries the block commits + the block-12 working-tree delta), then update
`patches/README.md` and this README with the new fork point. The old
`baseline/<sha>`-branch-per-upstream-range workflow was retired when the
delivery moved to the flat 12-patch set on `main`.

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
