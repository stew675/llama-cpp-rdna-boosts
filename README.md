# llama-cpp-rdna-boosts

A delivery repo for a **12-patch set** of ROCm-targeted feature enhancements
and performance fixes for llama.cpp: **blocks 01-11** (MTP, GDN, BF16 KV,
WMMA flash-attn, fused core, k-quant boosts, CUDA prefill-graph skip) plus
**block 12** (the hybrid HIP all-reduce). The patches apply to a clean
llama.cpp checkout at the recorded fork point `17252c769`.

`scripts/apply-all.sh` automates the apply: it creates a fresh `rdna-boosts`
branch, applies blocks 01-11 with `git am` and block 12 with `git apply`,
and commits each block.

## Current state (2026-08-29)

- **Fork point (baseline):** llama.cpp master at `17252c769`.
- **Set:** 12 patches in `patches/` (`0001`-`0011` + `12-hybrid-allreduce-hip.patch`).
- **Verified:** clean apply + full build + llama-cli same-seed coherence
  IDENTICAL to the fork build; tg64 38.12 / tg512 41.08 on the sim build.
- **Whitespace-clean apply:** the regenerated set applies with **zero git
  whitespace warnings** (`git am` 01-11, `git apply` 12; verified
  2026-08-29 on a fresh checkout at `17252c769`). The applied tree is
  byte-identical to the previous set except 8 inert whitespace lines + 1
  EOF blank line.
- **Deployment:** 3-GPU hybrid (`HIP_VISIBLE_DEVICES=0,1,2`, unpinned) gives
  depth-16384 decode 38.71 t/s (+21.8% vs 2-GPU). See
  [`patches/README.md`](patches/README.md) for block-12 env knobs and the
  server config.
- **RDNA4-only gate:** block 12 refuses to init off gfx1200/gfx1201 and
  falls back to RCCL (community RDNA3 verification pending).

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # block 10 decode-variance analysis (read before shipping)
├── rdna-boosts-all.patch  # convenience: the entire 12-patch net as ONE patch
├── patches/               # the delivery set: 0001-0011 + 12-hybrid-allreduce-hip.patch
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
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04) |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate |

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
git checkout 17252c769        # the SHA recorded in patches/README.md

# 2. apply the set (automated; VERIFIED 2026-08-29)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0001…0011  +  git apply patches/12-hybrid-allreduce-hip.patch
#    (one commit per block on a fresh `rdna-boosts` branch)

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
git am patches/000[1-9]-*.patch patches/001[01]-*.patch   # blocks 01-11
git apply patches/12-hybrid-allreduce-hip.patch           # block 12 (WIP patch)
git add -A && git commit -m "rdna-boosts: block 12: hybrid HIP all-reduce"
```

## When upstream master moves

The patches are static against `17252c769`. When upstream drifts and hunks
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

## License

Same as llama.cpp (MIT).
