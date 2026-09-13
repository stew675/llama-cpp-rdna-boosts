# llama-cpp-rdna-boosts

A delivery repo for a **16-patch set** (block 00 + blocks 01-15) of **RDNA3 / RDNA3.5 / RDNA4**
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
[Current state](#current-state)) and
**block 15** (attention-memory wins — a derived kq mask (V3), opt-in
native q8_0/bf16 K/V (V4/V5), QSA score-chain/bias/indexer-cache pruning
(W1-W3) and the ggml-alloc unused-view release (W4)) is **promoted to the
delivery** as `patches/0015` (promoted 2026-09-12 from
`archive/work/block-15-campaign-wins/`).
The patches apply to a clean
llama.cpp checkout at the recorded fork point `9113cc188` (re-based 2026-09-08 from `050dde50c`, itself re-based 2026-09-07 from `465e49b9c`, itself re-based 2026-09-06 from `9cffdcc80`, itself re-based 2026-09-02 from `0eadefebd`).

`scripts/apply-all.sh` automates the apply: it creates a fresh `rdna-boosts`
branch and applies blocks 01-15 with `git am`, one commit each.

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

The current delivery is a **16-patch set** (block 00 + blocks 01-15) for
llama.cpp at the fork
point `9113cc188` (blocks 00-15 in `patches/`, applied with `git am` via
`scripts/apply-all.sh`; canonical 16-block tip `907799de3`, net tree
`c2e284c2acc032238ef85cb35d427c1598ed0949`, rebuilt at the
fork point; block 15 promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`; block 02 amended 2026-09-12 with the rollback-bounded chunked-GDN threshold
(`n_rs_batch`) and the pre-batch snapshot slots, block 13 amended 2026-09-11 with the MoE
decode/verify mmvq band and the fused shared-expert epilogue band, block 14 amended 2026-09-11 with the hyper-connection band, the QSA
decode arm and the iq4_nl QSA enablement and 2026-09-12 with the configurable QSA prefill arm
(default `0` = QSA prefill always) + the device-query arm gate, the MTP-export logits-purity fix,
and the QSA indexer-score decode/verify band-uniformity fix (`ne11 = 4 * n_tps` crossed
`MMVF_MAX_BATCH_SIZE` at `n_tps = 3`, so the verify batch fell to MMF while decode stayed on MMVF and
flipped a top-k near-tie; the band now stays on MMVF), and block 08 amended 2026-09-11 with the `iq4_nl` FA
enablement -- qwen4exp and the MoE are both width-pure for `--spec-draft-n-max <= 7` now, and every
KV cache type the delivery supports takes the f16 path -- see the WORKLOG entries)
fork point); block 13 was amended again 2026-09-11 with the MoE `MUL_MAT_ID`
decode/verify dispatch fix (**+6.2% MoE decode**) and the `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` kill-switch (see the WORKLOG entry -- the
decode-only fused shared-expert window is the accepted MoE decode!=verify
residual).  **Block 15 (the attention-memory campaign) was promoted on
2026-09-12** to `patches/0015` (previously staged in
`archive/work/block-15-campaign-wins/`) -- the notes below are its promotion record.
The set applies
**whitespace-clean** (strict `git am`, no 3-way fallback) and each block
is build- and coherence-verified — see [`MANIFESTS.md`](MANIFESTS.md)
(apply order + verification contract), [`patches/README.md`](patches/README.md)
(per-block notes, env knobs, server config) and
[`BASELINE.md`](BASELINE.md) (fork point + drift policy).

All delivery-affecting changes (block amendments, community-fix
integrations, re-baselines, regenerations) are tracked as dated entries
— newest first — in **[`WORKLOG.md`](WORKLOG.md)**; the current-state
summary below is deliberately short and does not repeat them.

- **Latest entry (2026-09-12): block 15 (attention-memory campaign)
  promoted to the delivery** — the beta patch is now
  `patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch`, so the
  set is **16 patches** (`0000`-`0015`, block 00 + blocks 01-15) and
  `scripts/apply-all.sh` / `make-patches.sh` are 16-block flows (block 15
  is applied with `git am` like every other block; the earlier
  "beta patch applied manually on top" flow is gone).  Canonical 16-block
  tip `907799de3`, net tree `c2e284c2acc032238ef85cb35d427c1598ed0949`;
  strict **16/16** `git am` on a fresh worktree at `9113cc188`, zero
  whitespace warnings, applied tree == the re-validated beta tree.  The
  promoted patch is byte-identical to
  `archive/work/block-15-campaign-wins/block-15-campaign-wins.patch` except its
  `From <sha>` line.  The seven wins keep their env gates: **W1**
  QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA
  per-block bias + visibility + input-fill null guards
  (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only
  QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view
  release (no gate), **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by
  default), **V4** native q8_0 K/V and **V5** native bf16 K/V (both behind
  `GGML_CUDA_FA_KV_NATIVE`, **opt-in, default 0**).  Re-validated 2026-09-11
  against the then-15-patch delivery and re-cut onto the current base
  2026-09-12: every reserve number reproduces to the last decimal
  (qwen4exp ub 2048 compute 6690.40 → 3251.39 MiB/GPU, host 1262.70 →
  63.69, indexer KV 956.26 → 318.76; dense 4B 1800.33 → 1001.13, 27B
  1920.33 → 1121.13; a further −744/−632 MiB/GPU with V4), the width
  probe reproduces the delivered reference hashes
  (1 GPU `4089b4d4`, 2-GPU tensor `a4817ee6`, 3-GPU tensor `91434ea9`;
  `W=9` divergent as accepted), `V4/V5` on == off bit-identically, coherence
  is byte-identical across gates on 4B / both SWA gemmas / 27B (short +
  40k) / qwen4exp, the op suites pass (`FLASH_ATTN_EXT` 7859/7859 ROCm0 +
  CPU, `GATED_DELTA_NET` 46/46, `FLASH_ATTN_QSA` 22/22), the MTP gate is
  unchanged (27B `0.76744`, qwen4exp `0.44262`), and W4 round-trips
  56.00 → 16.00 MiB.  Cost ~1.3 % prefill / ~0.3 % decode (V4 ~1.7 %,
  V5 0.2-2.4 %).  One accepted caveat (do not re-report): W2's derived
  per-block bias is not bit-exact for `iq4_nl` (its greedy text/MTP
  acceptance differ from the delivery's while the sparse-arm PPL is
  identical at `6.5244`; `GGML_QSA_DERIVED_*=0` restores the delivery's
  values).  Full record: `patches/README.md` (block-15 promotion section),
  [`WORKLOG.md`](WORKLOG.md) and
  [`archive/work/block-15-campaign-wins/README.md`](archive/work/block-15-campaign-wins/README.md)
  (marked PROMOTED).


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
git checkout 9113cc188        # the SHA recorded in patches/README.md

# 2. apply the set (automated; VERIFIED 2026-08-29, re-verified 2026-09-01/02/05/06/07 and 2026-09-08)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0014  (one commit per block on a fresh `rdna-boosts` branch)

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
