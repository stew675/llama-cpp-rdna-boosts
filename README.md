# llama-cpp-rdna-boosts

A patch collection that brings **AMD RDNA-specific performance work** to
[llama.cpp](https://github.com/ggml-org/llama.cpp): MTP decode, chunked
gated-delta-net prefill, BF16 KV and WMMA flash-attention, fused MoE and
k-quant decode paths, a hybrid all-reduce, qwen4exp (Qwen3.8-Flash-Next)
support, and an attention-memory campaign that frees several GiB of VRAM.

It ships as **16 patches** (block 00 + blocks 01-15) for a clean llama.cpp
checkout at the fork point **`84e76d8a2`** (upstream master, 2026-09-24
re-base).  Each block is a self-contained `git am` commit, so you can apply
the whole set or pick the ones you want.  The **`mmb` (bf16-WMMA weight GEMM) / QSA / indexer
campaign**, formerly the 28-patch opt-in `beta/mmb-general/` set, is now **folded into the delivery
blocks** — the `mmb` core into block 08, the catch-all host-buffer/CPU fixes into block 06, and the
qwen4exp/QSA/HC/indexer work into block 15 — so the **16 patches alone reproduce the full campaign
tree `24bb0f5acb…`**.  `beta/mmb-general/` is retained only as the historical verification record;
see [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery).  (This is the
state on the **`beta-integration`** branch; `main` still carries r7 + the separate beta set.)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts
```

- One-line summary of each block: [The 16 blocks](#the-16-blocks)
- The folded `mmb`/QSA campaign: [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery)
- Apply details, env knobs, server config: [`patches/README.md`](patches/README.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md)
- Current status and validation: [Current state](#current-state)

> **The `mmb`/QSA/indexer campaign fold lives on the `beta-integration` branch (2026-09-25).**  That
> branch rewrites the 16 delivery patches to absorb the 28 `beta/mmb-general/` patches; the applied
> tree is `24bb0f5acb…` and the build is clean on gfx1201.  `main` still carries the un-integrated r7
> delivery plus the separate beta set.  The working plan, per-patch mapping and validation record
> are in [`wip/beta-integration/integration.md`](wip/beta-integration/integration.md).

## Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(the tag is the release identity: `v16-<fork-point>-r<N>`, e.g.
**`v16-84e76d8a2-r7`**, where `r1` is the re-base, `r2` the block-10 MoE-VDR arch-scope fix, `r3` the
block-14 `hc_combine` CPU-reference fix (issue #44) + the beta re-base, `r4` the block-15 RDNA4
GQA-6 decode/verify flash-attention band (issue #45), `r5` the block-15 f16/bf16 band coverage
(issue #45 follow-up), `r6` the block-15 bf16 native default flip, `r7` the block-14 Meta-tensor-split
scheduler race fix, `r8-integrated` the **`beta/mmb-general` fold into the 16 blocks** on the
`beta-integration` branch, and each later release on the
same base increments `N`).  `release.json.release` must equal the tag — CI
checks it — and only a tag push cuts a release.  Each release carries
`rdna-boosts-all.patch`, `patches.tar.gz`, `release.json`
and `SHA256SUMS`, so a consumer can pin a tag and verify the artifacts instead
of tracking a moving `main`.

`release.json` is the delivery's single source of truth (fork point, canonical
tip/tree, block count, per-artifact sha256); `scripts/apply-all.sh`,
`scripts/validate-set.sh` and CI all read it.  The container pipeline is
**tag-driven**, so ordinary commits to `main` (docs / benchmarks / `WORKLOG.md`)
only run the cheap patch validation — see [`CONTAINERS.md`](CONTAINERS.md) for
the release process and the prebuilt ROCm images.

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
├── prompts/               # versioned, hash-stable test prompts (sha256-recorded; never edited in place)
├── wiki/                  # source for the GitHub wiki (Home, MTP & Adaptive MTP, Quick Reference); see wiki/README.md
├── wip/                   # ACTIVE exploration docs / handoffs (currently: iq4nl-prefill/)
├── beta/                  # historical verification record (beta/mmb-general/, now folded into patches/)
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
| `0006` | host-buffer revert for discrete GPUs — now the delivery's **catch-all** block for mixed backend/scheduler/CPU fixes (the FA instance build-time work, `--fit` under `-sm tensor`, the host-buffer input layer and the tiny-CPU-split single-thread fix) |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04; amended 2026-09-07 with the mul_mat+add through-view shape guard, PR #15). **Now folds the `mmb` (bf16-WMMA dequant weight GEMM) core, the RDNA4 fragment port / per-arch tuning, and the GDN/PLE conv1d + narrow-row RMS-norm prefill fusions** (absorbed from the former `beta/mmb-general` campaign). |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions; adds a dedicated RDNA3.5 mmvq table) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate (bounded in-kernel spin since 2026-08-30 fix round; builds without RCCL) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — prefill fused expert MMQ (RDNA4 + RDNA3_5 + RDNA3_0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K, env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) + decode item-split (rpb 2/4/8) merged with the upstream has_fusion mmvq path |
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA (default) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy (promoted from `beta/qwen4exp`; see `patches/README.md` block-14 notes). The masked-V/freed-cell fixes it once carried now live in blocks 00 (Vulkan) and 03 (HIP). |
| `0015` | **attention-memory wins (block 15)** — promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`: **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default), **V4** native q8_0 + **V5** native bf16 K/V in the FA kernels (both behind `GGML_CUDA_FA_KV_NATIVE`, opt-in default 0), **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate; A/B revert in `archive/work/block-15-campaign-wins/ab/`).  ~3.4 GiB/GPU + ~1.2 GiB host saved on qwen4exp, ~800 MiB/GPU + ~800 MiB host on dense models, at ~1.3 % prefill / ~0.3 % decode. **Now also folds the qwen4exp/QSA/HC/indexer campaign** (`qsa3` packed-block WMMA attention, fused indexer top-k + prefill score fusions, HC16 native-BF16 producers, `hc_gate_mix`, sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the host-buffer/CPU/meta fixes) — the former `beta/mmb-general` work. |

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
# 1. fresh clone of llama.cpp, at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
FORK=https://github.com/ggml-org/llama.cpp
git clone $FORK && cd llama.cpp
git checkout "$BASE"

# 2. apply the set (automated; strict 16/16 git am on the recorded base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0015  (one commit per block on a fresh `rdna-boosts` branch)

# 3. build + verify (trim -DGPU_TARGETS to your GPU arch for a faster build)
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# coherence gate (same-seed output must match a known-good build):
./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 -p "The capital of France is" \
  -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn
```

> **Speed up rebuilds with ccache.**  The set's flash-attention template instances are the build's
> critical path, and their native-KV loader arms are deliberately force-inlined — the optimiser's
> cross-inlining is what makes them fast at runtime *and* slow to compile.  With `ccache` on PATH,
> a wiped rebuild of *unchanged* sources is a full cache hit: measured **282 s -> 4.2 s** on a
> 16-core gfx1201 box, **321.8 -> 5.2 s** on gfx1151 and **383.9 -> 4.8 s** on gfx1100 (657/657
> compile steps hit on each).  Add
> `-DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
> (the launcher form works with ROCm clang HIP device compilation; ccache 4.12.3 tested, on CMake
> 4.3).  ccache
> replays the compiler's own objects, so the cached build is the same code — verified with
> same-seed greedy text (identical hash on every host before and after enabling it),
> `llama-bench` (within noise) and `test-backend-ops`.  Any header change (e.g. `fattn-mma-f16.cuh`)
> invalidates its dependents, i.e. the whole FA group.
>
> **Do not use `git apply` on the concatenated 1-11 series** — it silently
> drops hunks (30 files / 2483 lines vs the correct 35 / 6094, verified
> 2026-08-29). `git am` (or `scripts/apply-all.sh`) is the required flow.

### Manual equivalent

```bash
git am patches/000[1-9]-*.patch patches/001[0-5]-*.patch   # blocks 01-15
git add -A && git commit -m "rdna-boosts: block 15: campaign memory wins"
```

### The `mmb` campaign is in the delivery

On the **`beta-integration`** branch the `mmb` (bf16-WMMA dequant weight GEMM) / QSA / indexer
campaign — the former 28-patch `beta/mmb-general/` set — is **folded directly into the 16 delivery
blocks**, so the normal workflow above is all there is to apply.  There is **no separate beta layer
any more**:

* **block 08** absorbs the `mmb` core, the RDNA4 fragment port and per-arch tuning, and the
  GDN/PLE conv1d + narrow-row RMS-norm prefill fusions;
* **block 06** (the catch-all) absorbs the host-buffer input layer and the tiny-CPU-split
  single-thread fix;
* **blocks 13/14** absorb the `mmb` fusion stand-downs and the extended MMVQ routed band;
* **block 15** absorbs `qsa3`, the fused indexer top-k + prefill score fusions, HC16, `hc_gate_mix`,
  sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the meta/CPU backend fixes.

Applying the 16 patches to `84e76d8a2` therefore reproduces the **full campaign tree
`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`** in one pass:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2
bash <path-to-this-repo>/scripts/apply-all.sh .   # 16/16 strict, tree 24bb0f5acb…
```

The per-patch fold mapping and validation record are in
[`wip/beta-integration/integration.md`](wip/beta-integration/integration.md).  Much of the MMB
kernel work is heavily adapted from **[pwilkin](https://github.com/pwilkin)**'s
[`strix-halo` fork](https://github.com/pwilkin/llama.cpp/commits/strix-halo/), with thanks.

> **Historical record only.**  [`beta/mmb-general/`](beta/mmb-general/) (the 28 patch files, the
> `mmb-general.patch`, `BETA-TESTING.md`, the gfx1201/gfx1100 records) is kept as the campaign's
> verification record; its patches are **no longer applied separately** and the `apply-beta.sh`
> helper has been **removed** (the delivery itself now contains the campaign).  Its gfx1151
> beta-window re-validation was **GREEN**
> (2026-09-25 — the four gates + the recurrent rollback; see `BETA-TESTING.md` §8), which is what
> the fold relies on.

## Recommended configuration — adaptive MTP + `ngram-mod`

The best general-purpose speculative-decoding configuration measured on this delivery combines the
**adaptive MTP controller** with the draftless **`ngram-mod`** speculator:

```bash
--spec-type draft-mtp-adaptive,ngram-mod \
  --spec-ngram-mod-n-match 45 \
  --spec-draft-n-max 9 --spec-draft-n-start 9
```

`ngram-mod` supplies the long verbatim-recall drafts the MTP head cannot match, while the adaptive
controller keeps the MTP depth right for everything else.  Measured against plain `draft-mtp-adaptive`
on the Q8_0 2-GPU reference cell (`-n 3000`, a 4-prompts-per-axis corpus): **recall +67.5 %,
code +0.8 %, prose +0.5 %, reasoning −1.9 %, overall +13.6 %**; on the dense 1-card cell it is
code/prose-neutral with the same recall win.  The small reasoning cost is the price of the deeper
`n-max 9`; a workload with no verbatim recall is marginally better served by plain
`draft-mtp-adaptive`.

Guidance:

* **Cap.**  `9` is the general-purpose pick and the optimum on a **single card** (a cap of `6` costs
  11-14 % on code there).  On a **multi-card tensor split** `6-7` is ~1 % better.  A lower cap saves
  only a small verify-batch scratch, not the model/KV memory.
* **`n_match`.**  Keep it `>= 40` **and an integer multiple of the cap** — `9/45`, `8/48`, `6/42`.
  Too short (`nm24`) makes ngram fire on incidental code repeats and lose code throughput; a
  non-multiple (e.g. `nm42` at cap 12) degrades acceptance.
* The combo changes the draft strategy, so it is **opt-in** — the controller default stays plain
  `--spec-type draft-mtp-adaptive`.

Full derivation (the 4×4 corpus, all four cells, and the rejected controller alternatives):
**[`wip/mtp-journey-2026-09-17/SUMMARY.md`](wip/mtp-journey-2026-09-17/SUMMARY.md)** (narrative in its
[`README.md`](wip/mtp-journey-2026-09-17/README.md)); the dated controller records are in
[`benchmarks/`](benchmarks/README.md), newest `2026-09-15-adaptive-mtp-tuning.md`.

## VRAM vs prefill — the derived KQ mask (`LLAMA_KQ_MASK_DERIVED`)

**On by default, deliberately.**  The delivery derives the attention mask inside the FA kernel from
compact per-cell state instead of materialising the `n_kv x n_q` f16 mask.  That removes
`n_ubatch x n_ctx x 2` bytes of compute-buffer VRAM **plus the same again on the host** — measured on
a 9B at `-c 98304` / ub 512: **184.0 -> 88.4 MiB** device and **112.0 -> 16.4 MiB** host.  The saving
scales linearly with the ubatch, which is the point: a deep-context **MoE** or **qwen4exp /
Qwen3.8-Flash-Next** workload wants a large ubatch, and that is exactly the configuration where the
mask is biggest (~800 MiB/GPU at ub 2048 / 196k) and where the VRAM the feature frees is the
difference between fitting the context and not.

The cost is **prefill only** — decode is untouched, because the derived path only fires for batches
larger than 8 tokens (speculative verify keeps the packed mask, so `n_max <= 7` stays bit-identical).
Measured PP512, mask on vs off, `-r 3` ("+" = the mask helps):

| config | d0 | 32k | 64k | 98k |
|---|---|---|---|---|
| gfx1201 9B dense 1 GPU | — | +1.9 % | — | **+3.5 %** |
| gfx1201 27B 2 GPU **tensor** | −1.3 % | −0.4 % | **+1.3 %** | **+1.7 %** |
| gfx1201 27B 2 GPU layer | — | — | — | −1.6 % |
| gfx1201 27B 3 GPU tensor | −3.4 % | −0.6 % | — | **+2.0 %** |
| gfx1151 9B dense 1 GPU | +0.4 % | −0.2 % | −0.9 % | −1.8 % |
| gfx1151 35B-A3B MoE 1 GPU | −0.3 % | −0.3 % | −0.8 % | −1.6 % |
| gfx1100 9B dense 1 GPU | −0.4 % | −0.5 % | −0.4 % | **−0.2 %** |

The tensor-split shape is the one to understand: the packed mask grows with `n_kv`, so on a
**tensor split at shallow depth the mask is a small loss (−1.3 % at d0, crossing zero near 48k) and
becomes a win by 64k+**; on the maintainer's 3-GPU tensor serving setup it is a win at depth.  gfx1100
and gfx1151 pay a depth-growing ~1–2 % (they did not recover as much from the r7 kernel fix as
gfx1201 — the iGPU shares host bandwidth and the 7900 XTX has more of its own).  gfx1100 on a
dual-card **`-sm tensor`** split is the one cell we still cannot measure here (only a single 7900 XTX
is available); a community report on 2x RX 7900 XTX is pending.

**Turning it off.**  `LLAMA_KQ_MASK_DERIVED=0` restores the packed mask (upstream's behaviour).
Worth doing if you are on **gfx1100/gfx1151** and want the last ~1–2 % of deep prefill, or on a
**tensor split at shallow depth** and prefill latency matters more than the VRAM.  For a deep-context
MoE / qwen4exp workload the default is the right side of the trade.

**Not a correctness knob:** same-seed output is byte-identical either way (the derived mask produces
the same values; only the memory layout and prefill cost differ).

**It works on both prefill kernels (r9).**  The derived mask is implemented by the **MMA** and the
**tile** flash-attention kernels, so it is no longer tied to the chooser picking MMA: a head above the
per-arch WMMA cap (RDNA4 576, RDNA3_5 320, RDNA3_0 256) used to lose the mask entirely, and that is
every **Gemma4** (head 512) on gfx1100/gfx1151 — the two arches that live on the tile kernel.  There
the mask is a *win*, not a tax (PP512, mask on vs off):

| config (tile kernel, natural selection) | cell | delta |
|---|---|---|
| gfx1100 Gemma4 12B, q8_0 KV | @ 16k / @ 32k | **+1.2 %** / **+0.9 %** |
| gfx1151 Gemma4 12B, q8_0 KV | @ 16k / @ 32k | **+1.6 %** / +0.6 % |
| gfx1201 Gemma4 E4B (tile forced) | @ d0 | **+2.3 %** |

and **decode pays nothing for it.**  Decode and the spec verify batch always take the tile kernel (the
chooser's WMMA branch requires `ne[1] > 8`), so the derived branch there is a cost on every arch; it is
hoisted out of the KV loop so the packed path's code generation is unchanged.  Measured r8 vs r9 at
that kernel: `tg128` deltas of **+0.01 %** (gfx1201 9B @ d16384), **+0.02 %** (gfx1100 9B @ d16384),
and flat on gfx1151 — the earlier per-iteration form cost −0.5..−0.8 % at depth before the hoist.

**The vec kernel has no derived arm**, but it is decode/verify-only (`n_tps <= 2`) while the derived
form only exists for prefill-shaped batches (`kq_mask_derivable()` rejects `n_tokens <= 8`), so it
cannot be selected for one.  If the launch log *does* print `derived kq mask flash attention not
supported, set to disabled` on a CUDA/HIP backend, the FA node did not reach the GPU at all — check
which kernel serves that head (the log adds a note pointing there).

One performance caveat with nothing to do with this knob: forcing the **tile** kernel for a head it
would not normally serve (a stale `GGML_CUDA_FA_WMMA_256=0`, a fixed env in the September qwen4exp
gates, is the usual cause) makes a head-256 model on gfx1201 **~3x slower at deep prefill** (9B,
`-d 98304`: 2104 -> 710 t/s).  That env is worth removing regardless of the derived mask.  One
exception worth knowing: **qwen4exp / Qwen3.8-Flash-Next gets its deep-context mask elision from the
QSA path's own derived visibility** (`GGML_QSA_DERIVED_VIS`, the code's "-800 MiB win"), which is
independent of this knob; `LLAMA_KQ_MASK_DERIVED` only serves that model's dense shortcut
(`n_kv <= 2051`), where the mask is tiny.

Full matrix, raw CSVs and the A/B harness: [`wip/kq-mask-derived-ab/`](wip/kq-mask-derived-ab/); the
2026-09-19 block-15 (r7) amendment in [`patches/README.md`](patches/README.md).

## When upstream master moves

The patches are static against the fork point in `release.json.base`. When upstream
drifts and hunks no longer apply, re-base the block commits (the fork checkout carries
them), regenerate the whole set with `scripts/make-patches.sh`, then refresh
`release.json` (`scripts/make-release.sh --base … --tip … --tree …`) and update the
current-state headers. The old `baseline/<sha>`-branch-per-upstream-range workflow
was retired when the delivery moved to the flat 16-patch set on `main`.

## Upstreaming

Some blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. Block 12's internal all-reduce is gated to
RDNA4 pending community verification on RDNA3 pairs. See `MANIFESTS.md`
for per-block verification and `BASELINE.md` for provenance.

## Current state

- **16-patch set** (block 00 + blocks 01-15) for llama.cpp at the fork point
  **`84e76d8a2`** (upstream master "metal : fix graph capture and handle empty graphs", 2026-09-24 re-base).
- Canonical 16-block chain on the **`beta-integration`** branch: tip
  **`f373450de489dd0fafba5bd285e71844109cd0ec`**, net tree
  **`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`** (the folded campaign); release candidate
  **`v16-84e76d8a2-r8-integrated`**.  The un-integrated `main` release is still
  **`v16-84e76d8a2-r7`** (tip `596a22db…`, tree `7726e514…`).
- **The `mmb`/QSA/indexer campaign is folded into the delivery** (2026-09-25, `beta-integration`):
  the former 28-patch opt-in `beta/mmb-general/` set is now part of the 16 block patches — the
  `mmb` (bf16-WMMA dequant weight GEMM) core, the RDNA4 fragment port / per-arch tuning and the
  GDN/PLE/RMS prefill fusions in **block 08**; the catch-all host-buffer/CPU fixes in **block 06**;
  `qsa3`, the fused indexer, HC16, `hc_gate_mix`, sparse MTP-draft and the MMVQ band in **block 15**
  (with block 14's pair stand-down and block 13's GLU stand-down).  Strict `git am` 16/16
  reproduces the full campaign tree `24bb0f5acb…` and the gfx1201 build is clean.
  `beta/mmb-general/` is kept as the historical verification record and the `apply-beta.sh` helper
  has been removed.  See [`wip/beta-integration/integration.md`](wip/beta-integration/integration.md).
- **The RDNA4 GQA-6 decode/verify FA band covers f16 (and, through its native arm, bf16) too
  (block 15, r5, 2026-09-25, issue #45 follow-up, reported by
  [@DanoPTT](https://github.com/DanoPTT)):** the
  head-256 GQA-6 `n_q <= 8` band no longer pays the tile kernel's 3x K/V re-fetch/dequantization;
  the whole band runs the WMMA kernel with the GQA group folded into one block (`ncols2 = 8`) and the
  KV split round-robin over a fixed `P` (independent of `n_q` and the KV length), so decode and every
  verify width reduce identically.  It started quantized-only (r4, reported by
  [@overdoingism](https://github.com/overdoingism)) and r5 extends it to the 2-byte types with a
  per-element-size config: native-quantized keeps `ncols1 = 4` / `P = nsm`, the 2-byte types take
  `ncols1 = 2` / `P = max(2, 3*nsm/4)`.  f16 kv 102400 verify widths 2.1-3.2x faster; 27B
  `draft-mtp n3` at ~30k +13 % (f16) / +14 % (bf16 native); plain f16 decode -2.5..-4.2 %.  **r6
  (2026-09-25) flips the bf16 native K/V arm to default ON** (`GGML_CUDA_FA_KV_NATIVE` unset now
  enables it, `=0` disables every native arm), since r5 made native bf16 the path to the band and the
  incoming beta prefill boosts outweigh its small prefill cost.  Default on
  (`GGML_HIP_FA_BAND_WMMA=0` opts out); prefill untouched.  See `patches/README.md`
  (2026-09-25 block-15 r4/r5/r6) and `WORKLOG.md`.
- **The qwen4exp CPU `hc_combine` reference is correct** (block 14, r3, 2026-09-25, issue #44):
  `ggml_compute_forward_hc_combine_f32` read `block_out` with a `t*ne[1]` row stride and `inject`
  with `t*hc`, but the model hands both over as multi-token tensors whose own `nb[1]` differs
  (`block_out` is a contiguous `[n_embd, nt]`, `inject` a view into the mix output with row stride
  `n_embd+hc`).  Every fused multi-token ubatch therefore read the wrong rows and a CPU-resident
  qwen4exp decoder layer emitted EOS as its first generated token; nt == 1 was accidentally correct.
  The reference now uses each tensor's own `nb[1]` (0 for a broadcast `ne[1] == 1`), mirroring the
  CUDA kernel, and is bit-identical at nt == 1.  Validated with an op-level CPU-vs-HIP oracle that
  fails 7/8 multi-token cases pre-fix and passes all 8 post-fix (gfx1100).
- **The wide-VDR MoE expert path is RDNA4/RDNA3_0-only** (block 10, r2, 2026-09-25): the
  `VDR_Q4_K/Q5_K/Q6_K_Q8_1_MMVQ_MOE` entry points were unconditional while only Q8_0 was arch-gated,
  so RDNA3_5 (gfx115x) ran the Q4_K/Q6_K experts — the Q4_K_M expert types — with the wide chunk the
  block-10 comment reserved for RDNA4/RDNA3_0.  `get_vec_dot_q_cuda()`/`get_vdr_mmvq()` now ignore
  `moe` on every other target in one place; base-16 MoE `draft-mtp n3` 0.73967 → 0.76484, 87.5 → 89.6 t/s.
- **Shared-NextN MTP heads are usable** (block 00, r13, 2026-09-22): a head with
  `nextn_shared_target_tensors` (no `token_embd`/`output` of its own, e.g. the qwen4exp
  `mtp-…-shared-Q8_0.gguf` sidecar) died every draft round on the M-RoPE `X < Y` check because the
  MTP driver inferred KV sharing from `ctx_other` alone.  `is_mem_shared` is now gated on the
  `gemma4-assistant` arch; it is an upstream bug (`04eb4c446`, #23398) folded into the block-00 base.
- **`--fit` works under `-sm tensor`** (block 6, r12, promoted from `beta/tensor-fit-fix/`): upstream
  threw `not implemented for SPLIT_MODE_TENSOR` and swallowed it, so the default-**on** `--fit` was a
  silent no-op under tensor split.  The Meta device's accessors are now exposed and `common/fit.cpp`
  has a dedicated tensor path (per-device targets from `--fit-target`, a proportional split or an
  honoured `-ts`, then auto-`n_ctx` reduction and an `-ngl` binary search); an explicit `-c` is never
  overridden.  Re-validated on r11 before promotion (fit decisions, 7 end-to-end loads with zero
  out-of-memory and zero compute-buffer growth, byte-identical same-seed gate).
- **The compute reserve accounts for the reachable (packed) kq mask** (issue #42, block 15,
  2026-09-20): V3's derived kq mask is a per-*batch* optimization, so a 2-D M-RoPE image/audio batch or a
  multi-sequence batch allocates the packed mask (`n_kv*n_tokens*2` bytes), which the reserve — measured
  with the derived form on — did not contain.  At depth that mask is hundreds of MiB, so a deep-context
  image batch grew the compute buffer mid-run; under the default `--fit-target 256` that growth failed
  (`cudaMalloc failed: out of memory`, `failed to process mtmd chunk`) and the next request asserted.
  `sched_reserve()` now measures with the packed mask **when such a batch is reachable** (the new
  `kq_mask_packed_reachable()`: M-RoPE or `n_seq_max > 1`), so `--fit` counts it exactly where it can
  happen.  Same-seed output is byte-identical and throughput is unchanged; the reporter's M-RoPE model
  pays 8960 tokens / -4.4 % of fitted context, while a non-M-RoPE single-sequence model keeps V3's
  reserve untouched.  A failed buffer allocation now also invalidates the allocator's layout instead of
  asserting on a later graph.
- **Block 11 replays HIP graphs for split-MoE decode again** (issue #41, 2026-09-20): the pre-fill
  test keyed off `nodes[0]->ne[1]`, which is `n_expert_used` (10) on the expert tensor a one-token
  decode split starts with under `-ncmoe`, so every decode split was skipped as multi-token.  A new
  `ggml_cuda_graph_is_multi_token()` reads the real token count from `MUL_MAT_ID`'s `ne[2]` / a weight
  `MUL_MAT`'s `src1->ne[1]` (0 -> 50 warmups / 0 -> 687 replays, `tg` 10.6 -> 12.8 t/s on
  Qwen3.8-Flash-Next UD-Q4_K_XL, output bit-identical), and on HIP the exec is now
  destroyed/re-instantiated instead of updated, avoiding the ROCm <= 10.0 `hipGraphExecUpdate` leak
  (`GGML_HIP_GRAPH_FORCE_UPDATE=1` opt-out).
- **FA instance build-time fix** (blocks 06/13/15, 2026-09-18): the MMA instances are generated per
  `(ncols1, ncols2, head size)` and the head-512 ones are listed first in the backend source order,
  the tile instances per `(head size, KV type)`, and the fused-gate MMQ instances moved out of
  `mmq.cu`.  Clean `ggml-hip -j16` **323.4 -> 236.0 s (-27 %)**, identical instantiations and symbols,
  no runtime change; the order, not the split, is what delivers it.
- **gfx1100 (RDNA3_0) WMMA FA is capped at head 256** (r5, block 04, issue #30): the 2026-09-14
  RDNA4 #28102 config transfer shipped RDNA4-tuned rows *and* a lifted head cap to gfx1100, so head
  512 took WMMA where stock takes tile and lost up to 23 % of deep prefill (gemma-4-26B-A4B
  `pp2048 @ d98304` q8_0 661 -> 773 t/s, bf16 656 -> 851); head 256 keeps WMMA, a +44-52 %
  deep-prefill win.  RDNA4 (576) / RDNA3_5 (320) are unchanged.
- **gfx1100 (RDNA3_0) tensor split keeps the stock AMD FA `ncols2` rule** (block 04, issue #30):
  the 2026-09-14 split-aware hint (wider generic `ncols2` for tensor-split attention) was RDNA4-tuned
  and cost RDNA3_0 deep prefill (`pp100K` 667.5 -> 779.4 t/s on 2× RX 7900 XTX, stock 805.0; decode
  unchanged).  A single gfx1100 card is unaffected (it already took the AMD rule).
- `--fit` no longer SIGSEGVs with `--spec-type draft-mtp-adaptive` and a minimal per-tier MTP
  head (issue #38; block 01, one line in `common/common.cpp`).
- A clean HIP build no longer prints the ~10k FA "loop not unrolled" warnings
  (`-Wno-pass-failed`, block 15; no codegen change).
- Patches `patches/0000-…0015-…` apply with **strict 16/16 `git am`** (no 3-way
  fallback, whitespace-clean) via `scripts/apply-all.sh`.  `scripts/validate-set.sh`
  re-checks the artifact hashes, the strict apply and the applied tree against `release.json`.
- **`hybrid` is the default all-reduce**; `GGML_CUDA_ALLREDUCE=ce` selects the opt-in
  copy-engine (SDMA) 2-GPU mode and `=nccl` forces RCCL.
- **Greedy purity**: plain decode == `draft-mtp` verify for `--spec-draft-n-max <= 7`
  across the supported KV types.  Depths 8..15 are allowed with a visible notice (a
  verify wider than 8 rows switches kernel family); `> 15` is clamped (the recurrent
  rollback snapshot bound).
- Last full `test-backend-ops` on this cut (gfx1201): **18083/18083**, with
  `FLASH_ATTN_EXT` **5952/5952** and `FLASH_ATTN_QSA` **22/22**.

The **dated record of every change** (re-bases, block amendments, issue fixes,
measurements) is [`WORKLOG.md`](WORKLOG.md), newest first.  Per-block notes, env knobs
and server configuration live in [`patches/README.md`](patches/README.md); apply order
and the verification contract in [`MANIFESTS.md`](MANIFESTS.md); fork point and drift
policy in [`BASELINE.md`](BASELINE.md); the purity rulebook in
[`GREEDY-PURITY.md`](GREEDY-PURITY.md).

## Community Acknowledgements

This work is becoming a community effort and I'd like to offer special thanks to the
following users for the assistance in finding issues and offering solutions!

- https://github.com/1337hero
- https://github.com/bakon11
- https://github.com/briansp2020  (block-13 moe_weighted_reduction float4 remainder fix + block-14 MUL_MAT_ID pair-fusion layout gate, issues #19 and #18)
- https://github.com/eoprede
- https://github.com/overdoingism  (issue #45: the RDNA4 head-256 GQA-6 decode/verify flash-attention band, reported with the diagnosis, op-level data, the round-robin KV split idea and a working opt-in patch; the r4 block-15 band is built on that submission)
- https://github.com/pwilkin  (the `strix-halo` fork at https://github.com/pwilkin/llama.cpp/commits/strix-halo/, heavily adapted for the MMB bf16-WMMA dequant-weight GEMM work, now folded into the delivery — formerly `beta/mmb-general/`)
- https://github.com/tungel
- https://github.com/DanoPTT  (block-08 mul_mat+add through-view shape guard, PR #15; and issue #45 follow-up: the f16 verify-width diagnosis / the f16 + bf16 band coverage folded into block 15 in r5/r6, measured on their R9700)

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
