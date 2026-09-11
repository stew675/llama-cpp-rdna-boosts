# llama-cpp-rdna-boosts

A delivery repo for a **15-patch set** (block 00 + blocks 01-14) of **RDNA3 / RDNA3.5 / RDNA4**
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
(W1-W3) and the ggml-alloc unused-view release (W4)) is **NOT part of the
delivery yet** — it is staged in `beta/block-15-campaign-wins/` and applied
manually on top of the 15-block tree, pending the maintainer's go-ahead.
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

The current delivery is a **15-patch set** (block 00 + blocks 01-14) for
llama.cpp at the fork
point `9113cc188` (blocks 00-14 in `patches/`, applied with `git am` via
`scripts/apply-all.sh`; canonical 15-block tip `27bd754b6` rebuilt at the
fork point).  **Block 15 (the attention-memory campaign) is staged in
`beta/block-15-campaign-wins/` and is not delivered yet** -- the notes
below are the beta record.  The set applies
**whitespace-clean** (strict `git am`, no 3-way fallback) and each block
is build- and coherence-verified — see [`MANIFESTS.md`](MANIFESTS.md)
(apply order + verification contract), [`patches/README.md`](patches/README.md)
(per-block notes, env knobs, server config) and
[`BASELINE.md`](BASELINE.md) (fork point + drift policy).

All delivery-affecting changes (block amendments, community-fix
integrations, re-baselines, regenerations) are tracked as dated entries
— newest first — in **[`WORKLOG.md`](WORKLOG.md)**; the current-state
summary below is deliberately short and does not repeat them.

- **Latest entry (2026-09-10): block 00 (structural and architecture fixes)
  is the new first block.**  `patches/0000` holds the FA small-batch
  KV-split width invariance (issue #25 — `launch_fattn`'s `parallel_blocks`
  heuristic keyed off `Q->ne[1]`, so decode and a speculative verify batch
  grouped the online-softmax/PV partials differently and greedy output
  changed with the MTP draft length; it now evaluates the heuristic as if
  `n_q == 1` for every `n_q <= 8`) plus the Vulkan masked-V/freed-cell fixes
  (`flash_attn_cm1.comp`/`flash_attn.comp`).  The **HIP** masked-V fixes
  moved from block 14 into block 03 (they sit on the native-BF16 FA path
  block 03 introduces).  The set is now 15 patches (`0000`-`0014`);
  validated clean-apply 15/15, issue #25 fixed on 1/2/3-GPU, plain decode
  byte-identical, MTP acceptance gate unchanged.  Full record in
  [`WORKLOG.md`](WORKLOG.md).

- **Previous entry (2026-09-10): block 15 (attention-memory campaign) is
  STAGED in `beta/block-15-campaign-wins/`, NOT part of the 15-patch
  delivery** (beta patch tip `377f8e790`; V5 native bf16 K/V and the
  RDNA3_5/gfx1151 V3 fix amended in).  Seven validated wins
  in one block, each with an environment A/B gate (V4 is opt-in):
  **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived
  QSA per-block bias + visibility + the input-fill null guards
  (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only
  QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc
  unused-view release (no gate; revert patch in
  `beta/block-15-campaign-wins/ab/`), **V3** derived kq mask
  (`LLAMA_KQ_MASK_DERIVED`, on by default), **V4** native q8_0 K/V and
  **V5** native bf16 K/V in the FA kernels (both behind the same
  `GGML_CUDA_FA_KV_NATIVE`, **default off**).
  Measured at ctx 204800 / q8_0 KV: qwen4exp ub 2048 compute
  6690.40 → **3251.39** MiB/GPU and host 1262.70 → **63.69** MiB, plus
  the indexer KV 956.26 → 318.76 MiB/GPU; dense models −799 MiB/GPU +
  −799 MiB host (4B 1800.33 → 1001.13, 27B 1920.33 → 1121.13), and a
  further −744/−632 MiB/GPU with V4 enabled.  With V5 enabled a bf16 KV
  cache costs exactly an f16 one: 4B 968.86 → **256.86** MiB/GPU
  (27B 1072.86 → **488.86**, gemma-4-E4B 1062.89 → **404.89**,
  gemma-4-31B 2068.89 → **716.89**).  Same-seed generated text
  byte-identical on 4B / 27B / gemma-4-E4B (ISWA) / gemma-4-31B (ISWA)
  / qwen4exp across every gate combination; MTP acceptance unchanged
  (27B 0.76744, qwen4exp 0.44262); prefill cost ~1.3 % (V3) and ~1.7-1.9 %
  more (V4), and −0.2 % (pp2048) to −2.4 % (pp40960) for V5, decode
  within noise.  RDNA3_5 (gfx1151) validated 2026-09-10: the block-14
  masked-V fixes and V3/V4/V5 are clean there, V3 is now enabled on the
  HIP iGPU and a multi-slot context keeps the packed mask, the reserves
  and Flash-Next W deltas match RDNA4 exactly, and the opt-in arms are
  cheaper than on RDNA4 (V4 **+2.6 %** at pp20480, V5 −0.4…−0.9 %).
  Full record in
  [`WORKLOG.md`](WORKLOG.md) and
  [`beta/block-15-campaign-wins/README.md`](beta/block-15-campaign-wins/README.md).

- **D12 closed (2026-09-10): V5 native bf16 K/V**, folded into block 15 the same
  day.  A bf16 KV cache no longer pays the whole F16 staging scratch in prefill
  (was **+712 MiB** on the 4B at ctx 204800 / ub 2048, +584 on the 27B, +1352 on
  gemma-4-31B); with `GGML_CUDA_FA_KV_NATIVE=1` it costs exactly what an f16 cache
  costs, with byte-identical output.  It ships **opt-in through the same switch as
  V4** because dropping the scratch costs ~1-2.4 % prefill (growing with the
  prompt: the conversion itself is free, but the launcher's F16 copy is a dense
  normalised copy of the interleaved cache view).  Design + measurements:
  [`wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`](wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md);
  what would make it free is recorded in [`TODO.md`](TODO.md).  Two pre-existing
  issues remain **documented, not fixed**: mixed K/V cache types (`bf16`+`q8_0`,
  `f16`+`q8_0`) fall off the GPU attention path (~⅓ of decode, ~88-92 % of prefill
  lost) — same-type K/V is the practical choice — and `gemma-4-E4B-it` on 3 GPUs
  with `-sm tensor` aborts in the meta splitter (maintainer's call: document only).

- **Previous entry (2026-09-10): block-14 freed-cell KV handling moved to
  kernel-side masked-V elimination (regeneration tip `ff2b35f49`).**  The
  host-side `zero_freed` row zeroing (gfx1151-only, added 2026-09-09) is
  REMOVED — `src/llama-kv-cache.{cpp,h}` are back to the upstream state.
  In its place block 14 carries three unconditional kernel fixes that make
  masked (freed/stale) flash-attention cells contribute exactly +0.0:
  HIP `fattn-tile.cuh` (packed-bf16 PV), HIP `fattn-mma-f16.cuh`
  (masked-V rows in staged shared tiles), Vulkan `flash_attn_cm1.comp` /
  `flash_attn.comp` (dead columns never read V).  Validated on the Strix
  Halo gfx1151 box with the host zeroing disabled: 16/16 determinism
  gates PASS on every KV type each backend's FA supports (ROCm
  f16/bf16/q8_0/q4_0; Vulkan also q4_1/q5_0/q5_1/iq4_nl), zeroing
  ON==OFF bit-identical, test-backend-ops vs CPU 4591/4591 (ROCm) +
  7822/7822 (Vulkan), no measurable decode/prefill regression.  Full
  record in [`WORKLOG.md`](WORKLOG.md).


## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # block 10 decode-variance analysis (read before shipping)
├── WORKLOG.md             # dated delivery records (newest first; README points here)
├── rdna-boosts-all.patch  # convenience: the entire 15-patch net as ONE patch
├── patches/               # the delivery set: 0000-0014
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── wip/                   # exploration docs + tuning tools + HANDOFF (session log)
├── beta/                  # promotion staging: beta/qwen4exp/ + beta/block-15-campaign-wins/
├── upstream/              # upstream-PR candidates (UPSTREAM-PR-*.md + .patch) + their index
└── archive/               # the rest: archive/work/ (closed experiments) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 15 blocks

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

> **Block 15 (attention-memory wins: V3 derived kq mask, V4/V5 native
> q8_0/bf16 K/V, W1-W4) is NOT part of this delivery — it is staged in
> `beta/block-15-campaign-wins/` and applied manually on top of the
> 15-block tree, pending the maintainer's go-ahead.**

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
