# llama-cpp-rdna-boosts

A patch collection that brings **AMD RDNA-specific performance work** to
[llama.cpp](https://github.com/ggml-org/llama.cpp): MTP decode, chunked
gated-delta-net prefill, BF16 KV and WMMA flash-attention, fused MoE and
k-quant decode paths, a hybrid all-reduce, qwen4exp (Qwen3.8-Flash-Next)
support, and an attention-memory campaign that frees several GiB of VRAM.

It ships as **16 patches** (block 00 + blocks 01-15) for a clean llama.cpp
checkout at the fork point **`790cf51aa`**.  Each block is a self-contained
`git am` commit, so you can apply the whole set or pick the ones you want:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 790cf51aa
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts
```

- One-line summary of each block: [The 16 blocks](#the-16-blocks)
- Apply details, env knobs, server config: [`patches/README.md`](patches/README.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md)
- Current status and validation: [Current state](#current-state)

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

- **16-patch set** (block 00 + blocks 01-15) for llama.cpp at the fork point
  **`790cf51aa`** (re-based 2026-09-13; previously `9113cc188`).
- Patches `patches/0000-…0015-…`, applied with **strict 16/16 `git am`** by
  `scripts/apply-all.sh` (no 3-way fallback, whitespace-clean).
- Canonical 16-block chain: tip `f27dc6d8006188d00ff96dadab6eb0edf79e2b7c`,
  net tree `bbbe005e95381301fdc71e5d636f448bab147a65`.
- Greedy purity: plain decode == `draft-mtp` verify for
  `--spec-draft-n-max <= 7` across the supported KV types (4B and 27B all
  eight; qwen4exp MTP).
- Every block is build- and coherence-verified.  Detail lives in:
  [`WORKLOG.md`](WORKLOG.md) (dated record, newest first),
  [`patches/README.md`](patches/README.md) (per-block notes, env knobs, server
  config), [`MANIFESTS.md`](MANIFESTS.md) (apply order + verification contract)
  and [`BASELINE.md`](BASELINE.md) (fork point + drift policy).

**Latest change (2026-09-13, latest) — block-14 pair-fusion `ncols_opt` fix: dense prefill regression
repaired.**  The 2026-09-13 re-base merged upstream's new `mmq_args::ncols_opt` field, but block-14's
`ggml_cuda_mul_mat_q_pair` builds its `mmq_args` by hand and still left it `0`, so the MMQ tile
heuristic selected the narrowest tile (`J=8`) — up to **2.2x slower dense prefill** (and 14-48% below
the pre-rebase delivery) on every dense model.  Both pair arms now set it like the standalone (dense:
the token count; `MUL_MAT_ID`: the RDNA per-expert average) and the heuristic falls back to
`ncols_max` when unset.  pp4096 restored/beaten: 27B Q8_0 623 -> **1363** (1 GPU) / 1718 -> **2176**
(tensor), 27B UD-Q4_K_XL 905 -> **1264** / 1693 -> **2040**, 4B 5386 -> **7304**; qwen4exp unaffected.
Numerics unchanged (pair on == off, same-seed `d03d0bc727a8`).  Canonical tip `f27dc6d80`, tree
`bbbe005e9538`.  Full record: [`WORKLOG.md`](WORKLOG.md) 2026-09-13 (latest) and the 2026-09-13
block-14 (ninth) section of [`patches/README.md`](patches/README.md).

**Previous change (2026-09-13, later) — the fused MoE router is bit-identical (TODO item 19).**
The block-08 `iq4_nl` `GET_ROWS` fix moved the qwen4exp greedy text, and the reason was a pre-existing
upstream fragility: the fused `topk_moe` MoE router was **not** bit-identical to the generic
`soft_max -> argsort -> get_rows -> norm` chain, and the fusion is selected by a **buffer-address
overlap** guard — so the model output depended on the allocation plan.  The three gaps are fixed:
`topk-moe.cu` reproduces the generic `block_reduce` softmax order (per-warp + cross-warp butterfly)
and the `reduce_rows_f32` sum order and divides by the clamped sum like `ggml_div`; `argsort.cu`'s
bitonic network breaks ties by index (matching the CUB path and the fused router's iterative argmax).
Fused == `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` for **all eight native KV types** on both `-sm tensor`
and `-sm layer`, `plain == n_max 3 == n_max 7` still holds, `test-backend-ops` is 18065/18065 and the
4B coherence is unchanged (`1c5d32ac537d`).  Full
record: [`WORKLOG.md`](WORKLOG.md) 2026-09-13 (even later) and the 2026-09-13 block-08 (seventh) section
of [`patches/README.md`](patches/README.md).

**Previous change (2026-09-13) — block-08 `iq4_nl` `GET_ROWS` CPU-fallback fix (TODO item 3).**
The qwen4exp `iq4_nl` KV-cache prefill delta was the QSA indexer key gather: the CUDA `GET_ROWS`
support predicate required `ne[0] % QK_K == 0` for the 32-value sub-block types, and the indexer key
row is 128, so an `iq4_nl` cache sent the gather to the **CPU** (26 graph splits per prefill graph, a
host round trip per indexer layer).  `getrows.cu` now takes the sub-`QK_K` path
(`dequantize_q4_nl`) and the predicate accepts every `ne00 % QK4_NL == 0`; qwen4exp `iq4_nl` prefill
pp8192 1815-1951 -> **2385-2422 t/s** (= f16/`q4_0`), pp32768 **+36 %**, `GET_ROWS` 215/215 ->
**219/219**.  The `W = 1..8` and control-hash gates hold and the 4B coherence is unchanged
(`1c5d32ac537d`).  Full record: [`WORKLOG.md`](WORKLOG.md) 2026-09-13 (later) and the 2026-09-13
block-08 section of [`patches/README.md`](patches/README.md).

**Previous change (2026-09-13) — re-base onto master `790cf51aa` (70 commits).**
Four upstream clashes resolved (`16378d93f` gfx1201 FA tuning, `5a4d0feca`
`GGML_FA_QUANTS`, `d4abd573f` MoE MMQ `ncols_opt`, `311d4211b` indexer V
cache).  The FA head-to-head kept the block-04 head-256 configs (upstream's
WMMA prefill tuning is worth only ~+0.5–0.9 % at 27B `pp16384` and breaks 4B
`q4_0` width purity); validation was byte-identical to the previous delivery.

All delivery-affecting changes (block amendments, community-fix integrations,
re-baselines, regenerations) are tracked as dated entries — newest first — in
**[`WORKLOG.md`](WORKLOG.md)**; this section holds only the latest one.

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # purity rulebook: index, invariants, per-finding claims (read before shipping)
│                          #   narratives/evidence for the closed cases: archive/docs/GREEDY-PURITY-FINDINGS.md
├── WORKLOG.md             # dated delivery records (newest first; README points here)
├── rdna-boosts-all.patch  # convenience: the entire 16-patch net as ONE patch
├── patches/               # the delivery set: 0000-0015
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── wip/                   # ACTIVE exploration docs / handoffs (currently: iq4nl-prefill/)
├── beta/                  # promotion staging (currently: beta/qwen4exp/); promoted campaigns move on
├── upstream/              # upstream-PR candidates (UPSTREAM-PR-*.md + .patch) + their index
└── archive/               # the rest: archive/work/ (closed experiments + the archived wip/ trees) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 16 blocks

| patch | what |
|-------|------|
| `0000` | **structural and architecture fixes** — FA small-batch KV-split width invariance (issue #25) + Vulkan masked-V/freed-cell fixes (dead columns never read V). The base every later block applies on top of. |
| `0001` | adaptive MTP draft depth (`--draft-mtp-adaptive`) |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA, arch-segregated gfx12/gfx11) |
| `0003` | BF16 KV cache + native-BF16 flash-attn (+ the HIP masked-V/freed-cell fixes since 2026-09-10) |
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
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA (default) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy (promoted from `beta/qwen4exp`; see `patches/README.md` block-14 notes). The masked-V/freed-cell fixes it once carried now live in blocks 00 (Vulkan) and 03 (HIP). |
| `0015` | **attention-memory wins (block 15)** — promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`: **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default), **V4** native q8_0 + **V5** native bf16 K/V in the FA kernels (both behind `GGML_CUDA_FA_KV_NATIVE`, opt-in default 0), **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate; A/B revert in `archive/work/block-15-campaign-wins/ab/`).  ~3.4 GiB/GPU + ~1.2 GiB host saved on qwen4exp, ~800 MiB/GPU + ~800 MiB host on dense models, at ~1.3 % prefill / ~0.3 % decode. |

> **Block 15 (attention-memory wins) is part of the delivery since
> 2026-09-12** (`patches/0015`, promoted from
> `archive/work/block-15-campaign-wins/`; a fresh set is now **16 patches**,
> blocks 00-15).

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
git checkout 790cf51aa        # the SHA recorded in patches/README.md

# 2. apply the set (automated; re-verified 2026-09-13 on the `790cf51aa` base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0015  (one commit per block on a fresh `rdna-boosts` branch)

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
git am patches/000[1-9]-*.patch patches/001[0-5]-*.patch   # blocks 01-15
git add -A && git commit -m "rdna-boosts: block 15: campaign memory wins"
```

## When upstream master moves

The patches are static against `790cf51aa`. When upstream drifts and hunks
no longer apply, regenerate the whole set from the fork with
`scripts/make-patches.sh` (needs the `~/llama.cpp` fork checkout, which
carries the block commits), then update
`patches/README.md` and this README with the new fork point. The old
`baseline/<sha>`-branch-per-upstream-range workflow was retired when the
delivery moved to the flat 16-patch set on `main`.

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
