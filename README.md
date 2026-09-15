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

## Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(e.g. **`v16-790cf51aa`** — the tag names the block count and the fork point).
Each release carries `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json`
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

## Current state

- **16-patch set** (block 00 + blocks 01-15) for llama.cpp at the fork point
  **`d1d3c3396`** (re-based 2026-09-15; previously `790cf51aa`, re-based 2026-09-13 from `9113cc188`).
- Patches `patches/0000-…0015-…`, applied with **strict 16/16 `git am`** by
  `scripts/apply-all.sh` (no 3-way fallback, whitespace-clean).
- Canonical 16-block chain: tip `af9ce375ded5238b59598290ad7366760b7dc6e0`,
  net tree `c6896785a5fefdf9438d26974c0274bf99f43263`.
- Release **`v16-d1d3c3396-r1`**.  `scripts/validate-set.sh` passes strict 16/16
  (applied tree == the recorded tree).
- Greedy purity: plain decode == `draft-mtp` verify for
  `--spec-draft-n-max <= 7` across the supported KV types (4B and 27B all
  eight; qwen4exp MTP).  Depths 8..15 are allowed with a visible notice that
  the two may differ (kernel-family switches above an 8-row verify); `> 15` is
  clamped (the recurrent rollback snapshot bound, 2026-09-13).
- Every block is build- and coherence-verified.  Detail lives in:
  [`WORKLOG.md`](WORKLOG.md) (dated record, newest first),
  [`patches/README.md`](patches/README.md) (per-block notes, env knobs, server
  config), [`MANIFESTS.md`](MANIFESTS.md) (apply order + verification contract)
  and [`BASELINE.md`](BASELINE.md) (fork point + drift policy).

**Latest change (2026-09-15) — re-base onto upstream master `d1d3c3396` (release
`v16-d1d3c3396-r1`).**  51 upstream commits past `790cf51aa`; three files conflicted (block 00's
Vulkan masked-V fix vs upstream's sparse FA, the FA test matrix, and qwen4exp's `{n_embd, hc}` norm
fold — plus the MTP `nextn.hc_head_norm` load-shape crash the merge exposed and validation caught).
Revalidated end-to-end on gfx1201: `FLASH_ATTN_EXT` 5951/5951, all custom ops pass, plain ==
`draft-mtp` byte-identical on 27B and qwen4exp, and the delivery is ahead of a stock build at the
same base on every gate (dense 27B +8-13 %, MoE 35B-A3B +9-17 %, qwen4exp 3.2× prefill / +44 %
decode, batched verify-width B=8 195 vs 120 t/s).  Full record: [`WORKLOG.md`](WORKLOG.md)
2026-09-15 (re-base).

**Previous change (2026-09-15, r5) — build time: a clean backend build was gated by one translation
unit.**  A fresh ROCm build had grown slow and `fattn-tile.cu` alone took **509 s of a 538 s** `-j16`
backend build.  Cause (ours): block 03 made the tile kernel's `type_KV` a template parameter, but
`DECL_FATTN_TILE_CASE`/`EXTERN_DECL_FATTN_TILE_CASES` still covered only F16/BF16 — and since the
dispatch has an unconditional `case` per native type, the other **six** types were instantiated
*implicitly in the dispatch TU* (72 of its 96 `tile_case` symbols; the 12 generated instance files had
2 each).  Those macros now expand per type, so the generated files carry 8 cases each and the dispatch
TU only externs: **538 s -> 330 s**, `fattn-tile.cu` **509 s -> < 10 s**, with the kernels, flags and
device code unchanged — `test-backend-ops -o FLASH_ATTN_EXT` 5951/5951, 27B text hashes bit-identical,
`tg64@32768`/`pp8192` within 0.12 % across five KV types.  The new critical path is the
`fattn-mma-f16` instance set, which our native-KV arms made 8x larger (0.90 -> 7.26 MB, 6.7 -> 229 s per
TU) — diagnosed and left as a follow-up.  Canonical tip `6f76c1cb1`, tree `d735d6c11`; release
**`v16-790cf51aa-r5`**.  Full record: [`WORKLOG.md`](WORKLOG.md) 2026-09-15, `patches/README.md` (the
2026-09-15 build-time block-15 amendment), and `wip/build-time-regression/`.

**Earlier change (2026-09-15, r4) — issue #30 second round: the reporter's q4_0 NaN, the prefill band
split, and the last four native KV arms.**  @briansp2020's r3 re-run found **4 NaNs** in
`test-backend-ops -o FLASH_ATTN_EXT` with a q4_0 K/V; chasing them closed four items.  (1) The tile
kernel is instantiated with ONE `type_KV` for both operands while `launch_fattn` chose its native read **per
tensor**, so a mixed pair fell back to the F16 tile with the native operand's staging skipped and read
raw q4_0 as F16 — `launch_fattn` now takes the kernel's native type explicitly.  (2)
`get_alloc_size`'s TILE case never learned the q4_0 arm, so **the q4_0 memory win had never been
delivered**: `-c 196608` q4_0 **849 -> 123 MiB**.  (3) TODO 21: a prefill now stages K/V while
decode/verify reads natively, with the scratch in a per-context, per-stream arena instead of the
graph reserve (which sized it for `n_ctx` — the adaptive-MTP `-c 196608` load failure); gfx1201 q8_0
`pp150000` **691/1077/1199** (1/2/3 GPU, from 661/996/1111).  Arch-gated: gfx1151 keeps its native
prefill (faster at every measured depth).  (4) TODO 2: native arms for `q4_1`/`q5_0`/`q5_1`/`iq4_nl`
(tg64 @ d32768 +9-13 % on gfx1201, +22-27 % on gfx1151).  Gates: `test-backend-ops` **5951/5951 on both
arches**, greedy text `native == staging` identical for all eight KV types on both, `W=1..8` one hash per
type.  Canonical tip `b19c70b34`, tree `7fab975d9`; release **`v16-790cf51aa-r4`**.  Full record:
[`WORKLOG.md`](WORKLOG.md) 2026-09-15, `patches/README.md` (the 2026-09-15 block-15 amendment),
`GREEDY-PURITY.md` §36, and `wip/issue-30-mtp-decode-regression/`.

**Previous change (2026-09-14 (later), r3) — issue #30 wide-configuration: the RDNA prefill regression is
fixed and made split-aware.**  The delivery's prefill fell off ~51 % faster with depth than stock (1 GPU,
27B f16 pp150k 609.5 vs 686.9) — not q8_0-specific.  Two causes: the head-256 `ncols=64` WMMA config was
a Strix-Halo (gfx1151) half-tile row used for all WMMA calls, and the delivery omitted stock's AMD
`switch_ncols2`.  Block 04 now makes the config `cc`-aware (RDNA3_5 keeps the halo row, RDNA4/RDNA3_0
take upstream's) and `ncols2` **split-aware**: a frontend hint (`ggml_set_fa_tensor_parallel`, set in
`llama_context` from `split_mode() == TENSOR && n_cuda_dev > 1`) selects generic `ncols2=8` for tensor
split and stock's AMD `ncols2=2` for a whole card.  Result (pp150K, f16, vs stock): 1-card **703** (+2.4 %),
2-card **1087** (+6.9 %), 3-card **1219** (+9.6 %); q8_0 KV at parity (−1.2 / −0.2 / +2.4 %); the 4B q4_0
`W=1..8` band stays pure.  Canonical tip `a2c8d06a7`, tree `eb5b7583`; release **`v16-790cf51aa-r3`**.
Full record: [`WORKLOG.md`](WORKLOG.md) 2026-09-14 (later), `patches/README.md` (block-04 amendment),
`GREEDY-PURITY.md` §35, and `wip/issue-30-mtp-decode-regression/`.

**Earlier change (2026-09-14, r2) — issue #30 wide-configuration: block 15's V4 native staging is
now the default for the sub-F16 KV quants, and q4_0 gained a native arm.**  The whole-cache F16 staging
pass is a *decode-depth* cost, so a quantized KV cache fell off with depth vs stock (1 GPU, 27B
UD-Q4_K_XL: q8_0 `tg64` d65536 18.92 = 66.1 % of d0 vs stock 22.43 = 80.1 %).  With
`GGML_CUDA_FA_KV_NATIVE` = *unset = auto* (native q8_0/q4_0 **on**, native bf16 off; `=1` force all on,
`=0` force the old staging path) plus the new q4_0 arm, q8_0 is **23.29** (+23 %) and q4_0 **22.82**
(+16 %) at d65536, ~1.2-1.3 % prefill, bit-identical to the staging conversion and `W=1..8`-pure.  The
same ~744 MiB scratch was the 260 MiB margin the adaptive-MTP draft context needed at `-c 196608` q8_0 /
ceiling 12, so that config **now loads** at the default 4 slots.  The head-256 WMMA question (#28867) was
investigated and needs no change (the `Q->ne[1] > 8` guard covers `W<=8`; `n_q=9..N` is at parity with
TILE).  Canonical tip `9ee71c356`, tree `58317e0d64dd`; release `v16-790cf51aa-r2`.  Full record:
[`WORKLOG.md`](WORKLOG.md) 2026-09-14, `GREEDY-PURITY.md` §34, and
`wip/issue-30-mtp-decode-regression/`.

**Previous change (2026-09-13) — issue #30: the `--spec-draft-n-max` clamp is raised from 7 to
15, and the qwen4exp QSA decode arm is band-matched to the verify width.**  The park reason for depth 15
was a claimed recurrent-rewind corruption on qwen4exp.  A new deterministic reference-context sweep
(`tests/test-recurrent-state-depth`: `n_rs_seq` 1..15, every rollback, plus deep drafts) is green on
qwen35/dsv4/kimi-k3/qwen4exp — **no rewind corruption in the allowed range** — and the qwen4exp
depth-15 divergence past the 2051 indexer selection width was the **QSA dense decode arm** flipping to
the sparse top-k arm for a 9..16-row verify (`QSA_DECODE_BAND = 8`); the arm band is now
`max(QSA_DECODE_BAND, cparams.n_rs_batch)`.  The residual purity loss above depth 7 is the documented
kernel-family switch at 8 rows (FA tile/MMA **and** matmul MMVQ/MMVF -> MMQ), now an accepted trade
with a visible notice instead of a clamp; only `> 15` (the recurrent snapshot bound) is clamped.  The
default `n_max 3` is unaffected.  The adaptive-MTP four-axis table is re-presented at the mode's
recommended ceiling **12**, measured at a realistic length (`-n 3000`) with reasoning pinned per axis
(`--reasoning off` for prose/code/recall): adaptive `n12` vs fixed `n3` is reasoning -1%, prose **+13%**,
code **+28%**, recall **+61%** (109.6 t/s, mean accepted length 8.95), and vs the old ceiling 7 it is
prose +26%, code +35%, recall +44%.  **Two protocol requirements are now part of the gate** (and were
both wrong in the first cut): `-n 3000` (`-n 2000` floor) -- a 256-token run measured the warm-up and
inverted the code ranking -- and the per-axis reasoning flag.  See
`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md` and the gate rule in
`benchmarks/mtp-adaptive-methodology.md`.  Canonical tip `c45244c72`, tree
`a5683e1b008e`.  Full record:
[`WORKLOG.md`](WORKLOG.md) 2026-09-13 (latest) and the issue-#30 section of
[`patches/README.md`](patches/README.md).


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
