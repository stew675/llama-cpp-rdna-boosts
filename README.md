# llama-cpp-rdna-boosts

## Summary Overview

- **What It Is**: Upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) at the pinned commit in
  [`release.json`](release.json) plus 19 RDNA-targeted patches (18 official blocks + the temporary `0300`).
  Not a divergent fork: the series is re-cut onto each new upstream base.  Resyncs to upstream happen weekly.
- **Hardware**: `gfx1100` (RDNA3), `gfx1150`/`gfx1151` (RDNA3.5) and `gfx1200`/`gfx1201` (RDNA4) are
  supported, `gfx1030` (RDNA2) is best effort. See [Supported architectures](#supported-architectures).
- **What it buys you**: RDNA-tuned versions of what upstream already has (flash attention, quantized prefill
  GEMM, k-quant decode, chunked GDN prefill, the MoE host tier, the compute buffer, CUDA graph replay, MTP
  draft depth, server prompt admission, the qwen4exp implementation), plus a few capabilities it does not
  (ternary weights on the GPU, the streamed KV, the APU expert store, the host-expert pool).
- **Get it**:
  - clone `https://github.com/stew675/llama.cpp`, check out the `rdna-boosts` branch, build as usual; or
  - apply the patch block series to a clean checkout with `scripts/apply-all.sh`. See [Consumer workflow](#consumer-workflow); or
  - run a prebuilt ROCm image (7.14.1 and 10.2, `ghcr.io/stew675/llama-cpp-rdna-boosts`): [`CONTAINERS.md`](CONTAINERS.md).
- **What it costs you**: you are on a pinned base rather than upstream `master`, so you re-apply at each new base or follow the branch.
- **Where to read next**: [`patches/README.md`](patches/README.md) (per block, its knobs),
  [`ENVIRONMENT.md`](ENVIRONMENT.md) and [`DIAGNOSTIC.md`](DIAGNOSTIC.md) (every variable),
  [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md) (measured configuration tables), the top of
  [`WORKLOG.md`](WORKLOG.md) (what changed recently).

---

## Detailed Overview

A patch collection that brings **AMD RDNA-specific performance work** to
[llama.cpp](https://github.com/ggml-org/llama.cpp): MTP decode, chunked
gated-delta-net prefill, BF16 KV and WMMA flash-attention, fused MoE and
k-quant decode paths, a hybrid all-reduce, qwen4exp (Qwen3.8-Flash-Next)
support, and an attention-memory campaign that frees several GiB of VRAM.

**Everything this delivery adds is default-on — you should not need to set any environment variable.**
[`ENVIRONMENT.md`](ENVIRONMENT.md) documents every end-user variable the set adds or repoints — the opt-ins,
the tuning values and the options, with their defaults — and [`DIAGNOSTIC.md`](DIAGNOSTIC.md) holds the
kill-switches, A/B arms and diagnostics, which you only reach for when something is wrong or you are bisecting.

It ships as **19 default patches** for a clean llama.cpp checkout at the fork point **`919794500`**
(upstream master, the 2026-11-20 re-base).

- **18 official supported blocks `0000`-`0017`**
- **2 optional blocks** that are NOT in the default set and are applied only with `--include-optional`
  (see [`optional/README.md`](optional/README.md))
  - `0100` (two-lane prefill): [`optional/two-lane-prefill/`](optional/two-lane-prefill/README.md)
  - `0101` (mixed K/V cache types): [`optional/mixed-kv/`](optional/mixed-kv/README.md)
- **temporary block `0300`**, applied last -- a local `-sm relaxed-tensor` relaxation for the MoE amax
  architectures, kept separate so it can be dropped; see
  [block 0300](patches/README.md#block-0300---sm-relaxed-tensor-for-the-moe-amax-arches-temporary-expect-removal)

Each block is one `git am` commit, and the series is linear (a block assumes its predecessors), so the set
is applied whole.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 919794500
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts (19 commits)

# OPT-IN, not part of the delivery: optional artifacts on top (NOT the release tree)
bash <path-to-this-repo>/scripts/apply-all.sh --include-optional .       # asks [y/N] per artifact
bash <path-to-this-repo>/scripts/apply-all.sh --include-optional-all .   # every artifact, no prompt

# OPT-IN, not part of the delivery: a beta candidate on top (NOT the release tree -- beta testing only)
bash <path-to-this-repo>/scripts/apply-all.sh --include-beta .   # 19 default blocks + beta/*/*.patch
```

- One-line summary of each block: [The patch blocks](#the-patch-blocks-official-0000-0017--temporary-0300--optional-01000101)
- What each block contains, its knobs and its purity notes: [`patches/README.md`](patches/README.md)
- The folded `mmb`/QSA campaign: [block 06](patches/README.md#block-06--mmb-and-mmf32-the-quantized-and-the-f32-prefill-gemm)
- Every environment variable, with its default and class: [`ENVIRONMENT.md`](ENVIRONMENT.md) (the knobs)
  and [`DIAGNOSTIC.md`](DIAGNOSTIC.md) (the switches, A/B arms and diagnostics)
- **Running a model bigger than your VRAM on one card** (`-ncmoe` + the MoE expert cache, with the
  measured config tables and the measurement method): [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md) (today's entries at the top, the index of earlier days at
  the end)
- Current status and validation: [`release.json`](release.json) (source of truth) and [`MANIFESTS.md`](MANIFESTS.md)

### Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(the tag is the release identity: `v<blocks>-<fork-point>-r<N>`, e.g. **`v18-919794500-r377`**; the block
count in the tag is the **official** `00xx` count, so `v15-*` tags are the 15-block releases up to r177,
`v16-*` the 16-official-block releases from r178 on, and `v18-*` the current 18.  The temporary `0300` and
the optional `01xx` blocks are never counted (`AGENTS.md`).  `r1` is the
re-base of the set onto that fork point and every later `rN` is a dated change recorded in
[`WORKLOG.md`](WORKLOG.md), newest first, with the earlier days in [`archive/worklog/`](archive/worklog/)).

### Supported architectures

The set targets the **RDNA3 / RDNA3.5 / RDNA4** GPU families, with RDNA2 as best-effort:

| family | arches | example parts | status |
|--------|--------|---------------|--------|
| RDNA 3 | `gfx1100` | RX 7900 XTX/XT, RX 7800 XT, ... | supported |
| RDNA 3.5 | `gfx1150`/`gfx1151` | Strix Point / Strix Halo APUs | supported |
| RDNA 4 | `gfx1200`/`gfx1201` | RX 9060 XT; RX 9070 / 9070 XT | supported |
| RDNA 2 | `gfx1030` | RX 6800/6800 XT/6900 XT, Radeon Pro V620 | best-effort; externally validated (V620), not run in our fleet |

> **Strix Point (`gfx1150`) is in the target set as of 2026-10-05.**  Container images published before then
> carry no `gfx1150` code object and fail on the first kernel launch (`ROCm error: device kernel image is
> invalid`); build from source with `gfx1150` in the target list, or set `HSA_OVERRIDE_GFX_VERSION=11.5.1` to
> take the `gfx1151` code objects.  See [Strix Point (`gfx1150`)](CONTAINERS.md#strix-point-gfx1150).
>
> **RDNA2 (`gfx1030`) is best-effort: not a supported target, but no longer unvalidated.**  It is compiled
> into the image so RDNA2 owners pick up the runtime-dispatched boosts instead of a first-kernel
> `ROCm error: invalid kernel file`, the RDNA3/4-only paths fall back to the upstream non-WMMA kernels, and
> there is still no RDNA2 device in the maintainer's fleet -- so an RDNA2 result is informational, never a
> delivery gate.  Validation and both paths: [RDNA2
> (`gfx1030`)](CONTAINERS.md#rdna2-gfx1030--best-effort-externally-validated).
>

> **RDNA4 (gfx120x) sees the most benefit** — the WMMA flash-attn path, the
chunked-GDN kernel, the k-quant VDR boosts and block 12's internal
all-reduce were all first built and validated there. As much of that work
as possible is back-ported to the RDNA3/3.5 families instead of being
gated off.

### Repository Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # the policy layer: what is mandatory, scope, how a release is cut
├── INVARIANTS.md          # the engineering facts: measured invariants / "do not re-derive" (pre-ship reading)
├── MANIFESTS.md           # apply order + the verification contract (release gates per block)
├── BASELINE.md            # fork point, what the patches are, how they are cut, drift policy
├── TODO.md                # forward-looking tracker: open items, limitations, parked ideas only
├── GREEDY-PURITY.md       # purity rulebook: invariants + one summary entry per finding (read before shipping)
│                          #   narratives/evidence: archive/worklog/ + benchmarks/ (and pre-r100: archive/docs/GREEDY-PURITY-FINDINGS.md)
├── WORKLOG.md             # the dated record: today's entries, then an index of the earlier days
├── archive/worklog/       # the record's earlier days, one file per day (newest first; the only release history)
├── ENVIRONMENT.md         # the end-user env vars (opt-in/tuning/options), with default and class
├── DIAGNOSTIC.md          # the kill-switches, A/B arms, test hooks and diagnostics
├── CONTAINERS.md          # the ROCm container images + the off-host build watcher
├── COMMUNITY-CONFIG.md    # running a model bigger than VRAM (`-ncmoe` + the expert cache)
├── CONTRIBUTORS.md        # the credit register: who owns what (co-authored-by / reported-by per block)
├── NOTICE                 # licence scope of the applied tree (block 03's Apache-2.0 kernels)
├── rdna-boosts-all.patch  # convenience: the entire 19-patch default net as ONE patch
├── patches/               # the delivery set: official 0000-0017 + temporary 0300 (one `git am` block each)
│   ├── README.md          # the per-block reference: what each block is, its knobs, its purity notes
│   └── FILE-SET-RETIREMENTS.tsv # re-base decision rows: a deliberate retirement a block may lose (gate 38)
├── scripts/               # apply / regenerate / validate / publish tooling; index: scripts/README.md
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift; --include-beta)
│   ├── gate-prefill-logits.sh  # prefill-logit KLD release gate (issue #113)
│   ├── gate-per-block-build.sh # bisect only (r332): which block broke a build
│   ├── check-rung-file-sets.py # re-base file-set reconciliation (TODO #358): no block may lose a file,
│   │                      #   define, env var or option the patch it replaces carried
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # dated bench records + the release-gate methodologies (MTP four axes, prefill-logit KLD)
├── prompts/               # versioned, hash-stable test prompts (sha256-recorded; never edited in place)
├── wiki/                  # source for the GitHub wiki (Home, MTP & Adaptive MTP, Quick Reference); see wiki/README.md
├── wip/                   # ACTIVE exploration only, never applied; index: wip/CAMPAIGNS.md
├── beta/                  # promotion-candidate patches pending a beta test (NOT part of the delivery;
│                          #   apply with `apply-all.sh --include-beta`; currently: the two-arm server
│                          #   (v2, replacing the withdrawn block 17) as three patch files -- KNOWN OPEN
│                          #   DEFECT, it corrupts under heavy concurrent load (TODO #378); see beta/README.md
├── optional/              # opt-in blocks 0100/0101 (NOT part of the default delivery; apply with
│                          #   `apply-all.sh --include-optional` and a [y/N] per artifact, numeric order)
├── upstream/              # upstream-PR candidates: one sub-directory each (README.md + patch) + index
├── third_party/radiance/  # Apache-2.0 attribution for block 03's libr4d-derived GDN kernels (LICENSE + NOTICE)
├── utilities/             # measurement/reporting tools, never applied (e.g. storage-oracle/); see utilities/README.md
└── archive/               # the rest: archive/work/ (closed experiments + the archived wip/ trees) + archive/docs/ (history)
```

### The patch blocks: official `0000`-`0017` + temporary `0300` + optional `0100`/`0101`

One `git am` commit per row, applied in filename order: `0000`-`0017` then `0300`.  The optional
`0100`/`0101` are **not** in this table and never in the default apply -- see
[`optional/README.md`](optional/README.md).  A row is one sentence; the block number links to its
section in [`patches/README.md`](patches/README.md), which carries the detail (what it contains, its
knobs, its invariants and the dated amendments).

| patch | subject | concern |
|-------|---------|---------|
| [`0000`](patches/README.md#block-00--foundation-band-uniformity-the-meta-wrapper-the-allocator) | **foundation** | band-uniformity and allocator fixes, the Meta wrapper, and the meta hot-path dispatcher. |
| [`0001`](patches/README.md#block-01--hybrid-hip-all-reduce) | **hybrid HIP all-reduce** (RDNA4-gated) | the internal all-reduce, with its large-tensor arm chosen per model (copy engine for a dense model, RCCL's ring for a host-offloaded one); gate [`scripts/gate-prefill-logits.sh`](scripts/gate-prefill-logits.sh). |
| [`0002`](patches/README.md#block-02--adaptive-mtp-draft-depth) | **adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`) | the draft depth follows acceptance, and the draft KV type follows `-ctk`/`-ctv` unless `-ctkd`/`-ctvd` pins it. |
| [`0003`](patches/README.md#block-03--chunked-gated-delta-net-prefill--the-recurrent-rollback-tracker) | **chunked gated-delta-net prefill** (bf16/WMMA) | the chunked GDN prefill kernel per architecture plus its per-sequence rollback tracker. |
| [`0004`](patches/README.md#block-04--bf16-kv-cache--native-bf16-flash-attention) | **BF16 KV cache + native-BF16 flash attention** | a BF16 KV cache read natively by the flash-attention kernels, no F16 scratch. |
| [`0005`](patches/README.md#block-05--the-fa-band-native-quantized-k-v-the-derived-kq-mask) | **flash-attention band, native quantized K-V, derived kq mask** | the band kernel and the derived mask, with a per-shape work gate that keeps the band only where it pays. |
| [`0006`](patches/README.md#block-06--mmb-and-mmf32-the-quantized-and-the-f32-prefill-gemm) | **MMB quantized prefill GEMM** | the bf16-WMMA dequant-weight prefill GEMM (and the F32 one), dispatched per architecture at runtime. |
| [`0007`](patches/README.md#block-07--k-quant-boosts-mmvq-vdr-q6_k-mmq-q1_0-grid-pad) | **k-quant boosts** | the mmvq VDR dot products, Q6_K MMQ, Q1_0 and the grid-stall pad. |
| [`0008`](patches/README.md#block-08--fused-kernels-including-the-fused-moe-gateupglu) | **fused kernels** | the conv1d/RMS/top-k-MoE/verify fusions and the fused MoE gate+up+GLU. |
| [`0009`](patches/README.md#block-09--host-resident-moe-iface-scheduler-expert-cache-staging-rings) | **host-resident MoE** | the host-expert tier: iface, scheduler, expert cache, movable-boundary slab and prefill staging. |
| [`0010`](patches/README.md#block-10--the-slab-the-fusion-dispatcher-the-async-input-ring-infrastructure) | **infrastructure** | the slab, the fusion dispatcher, the async input ring and the GDN/RoPE decode fusions, split out of `ggml-cuda.cu`. |
| [`0011`](patches/README.md#block-11--qwen4exp-model-hyper-connections-lazy-reader-qsaindexer) | **qwen4exp** (Qwen3.8-Flash-Next) | the model graph, hyper-connections, the managed lazy reader and the QSA/indexer. |
| [`0012`](patches/README.md#block-12--campaign-remainder) | **campaign remainder** | CPU-side wins, DFlash, mmap reclaim, the batched state copy and the host-expert pool. |
| [`0013`](patches/README.md#block-13--mtp-draft-vocabulary-subset---spec-draft-mtp-vocab-n-opt-in) | **MTP draft vocabulary subset** (**opt-in**, default off) | the MTP draft scores only the LM-head rows below `N` and scatters them into an `-inf` row, so the target's verification is untouched. |
| [`0014`](patches/README.md#block-14--the-streamed-kv-the-resident-window---kv-window-the-attendance-pool---kv-attendance-pool-and-the-mtp-draft-window---spec-draft-kv-window-default-0) | **streamed KV** (**all three placements opt-in, default OFF since r247**) | the resident window `--kv-window`, the attendance pool `--kv-attendance-pool` and the MTP draft ring `--spec-draft-kv-window` -- fittability features, so a config whose KV fits pays nothing for them. |
| [`0015`](patches/README.md#block-15--the-uma-gtt-expert-store-an-apus-moe-expert-set-served-from-one-registered-region--lme-sizes-it) | **UMA GTT expert store** | an APU's expert set in one registered, `mlock`ed region that `-lme` sizes (and opts into streaming), inert on a discrete GPU. |
| [`0016`](patches/README.md#block-16--the-servers-prompt-admission-policy-the-fair-share--ib-the-skew-the-duty-cycle) | **the server's prompt admission policy** | the max-min fair share of a step, `-ib`, the shortest-remaining skew and the decode duty cycle, plus (since r365) the KV admission of a whole task; gates [`scripts/gate-prefill-fairness.sh`](scripts/gate-prefill-fairness.sh), [`scripts/gate-prefill-duty.sh`](scripts/gate-prefill-duty.sh), [`scripts/gate-kv-admit.sh`](scripts/gate-kv-admit.sh). |
| [`0017`](patches/README.md#block-17--tq1_0--tq2_0-ternary-weights-on-the-gpu-new-at-r356-contributor-pr-358) | **TQ1_0 / TQ2_0 (ternary) weights on the GPU** (**default-on**) | `MUL_MAT`, `GET_ROWS` and the dequantise rows for both ternary types, bit-identical to the CPU's AVX2 path (contributor PR #358); gate [`scripts/gate-tq.sh`](scripts/gate-tq.sh). |
| [`0300`](patches/README.md#block-0300---sm-relaxed-tensor-for-the-moe-amax-arches-temporary-expect-removal) | **`-sm relaxed-tensor`** (**TEMPORARY**, `flint`/`mellum`/`mistral4` only) | a tensor split that also admits a replicated tensor whose values differ per GPU (the MoE amax row max); not a supported feature, expect removal. |

### Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
FORK=https://github.com/ggml-org/llama.cpp
git clone $FORK && cd llama.cpp
git checkout "$BASE"

# 2. apply the set (automated; strict 19/19 git am on the recorded base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0017 then patches/0300  (one commit per block on a fresh `rdna-boosts` branch)

# 3. build + verify (trim -DGPU_TARGETS to your GPU arch for a faster build)
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# coherence gate (same-seed output must match a known-good build).
# PIN the devices on any multi-device run: without it the iGPU is enumerated and the load aborts
# with `ROCm error: invalid kernel file`.
HIP_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn
```

> **Or skip steps 1-3.**  The release's prebuilt ROCm **7.14.1** and **10.2** images already carry the applied
> tree (`ghcr.io/stew675/llama-cpp-rdna-boosts`: `server-rocm-<ver>`, plus the immutable
> `-<fork-point>` and `-<release>` tags, e.g. `server-rocm-10.2-v18-919794500-r377`).  Tags, provenance labels,
> the RDNA2/`gfx1150` caveats and the off-host build: [`CONTAINERS.md`](CONTAINERS.md).
>
> **Put `ccache` on PATH and pass the launchers** (`-DCMAKE_HIP_COMPILER_LAUNCHER=ccache`, the `C` and
> `CXX` equivalents): the FA template instances are the build's critical path, so an unchanged rebuild drops
> from minutes to seconds (282 s -> 4.2 s measured on gfx1201).  It replays the compiler's own objects, so
> the code is identical; any header change (e.g. `fattn-mma-f16.cuh`) invalidates the whole FA group.
>
> **Do not use `git apply` on the concatenated series** — it silently
> drops hunks (30 files / 2483 lines vs the correct 35 / 6094, verified
> 2026-08-29). `git am` (or `scripts/apply-all.sh`) is the required flow.
>
> **Every HIP configuration builds, including the stock one.**  `GGML_HIP_NO_VMM` defaults to **ON** upstream,
> so a plain `-DGGML_HIP=ON` build is the *no-VMM* configuration, not the one this repo develops on.  r347-r350
> shipped a no-VMM link break (`undefined reference to ggml_cuda_pool_peer_copy_ok` in every executable, which a
> shared library is allowed to hide and lld tolerates --
> [issue #353](https://github.com/stew675/llama-cpp-rdna-boosts/issues/353)); **r351** fixes it, and that
> configuration is now built and run on every release, checked by `scripts/check-shlib-symbols.sh` (`ldd -r` over
> every shipped library, `MANIFESTS.md` gate 40).  The switch selects the per-context allocation **pool** only:
> the movable-boundary slab and the host-expert tiers use the driver's VMM calls directly and are reached in both
> configurations -- measured on 35B-A3B `-ncmoe 48` / 1 GPU, the two builds print an identical slab and
> host-pool census and read the same pinned text (`47 chars sha=49bf8044ec4b`, 2/2 each).
>
> **On Windows, the HIP runtime must be the ROCm/TheRock one, not the display driver's.**  `amdhip64_7.dll`
> normally resolves from `C:\Windows\System32`, i.e. the copy the **Adrenalin** driver ships, and the two are
> not interchangeable for the host-expert path: on the driver copy the pinned tier stops at ~9.3 GiB of adapter
> shared memory and latches off (one reported box then spun in the runtime's own wait loop after the
> post-prefill re-arm -- [issue #278](https://github.com/stew675/llama-cpp-rdna-boosts/issues/278)), where the
> same binary with the TheRock runtime beside it filled a 141/141-table mirror at 370.7 t/s prefill against
> 39.8.  Copy the ROCm runtime DLLs next to `llama-server.exe` and check that the startup line names that
> directory, not `System32`; a box that cannot switch runtimes sets `MOE_HOST_POOL_PIN=0` (correct, slower).
> Both variables and the diagnosis: `DIAGNOSTIC.md` §1.2.1.

### When upstream master moves

The patches are static against the fork point in `release.json.base`. When upstream
drifts and hunks no longer apply, re-base the block commits (the fork checkout carries
them), regenerate the whole set with `scripts/make-patches.sh`, then refresh
`release.json` (`scripts/make-release.sh --base … --tip … --tree …`) and update the
current-state headers. The old `baseline/<sha>`-branch-per-upstream-range workflow
was retired when the delivery moved to the flat patch set on `main`.

**A re-base also owes `scripts/check-rung-file-sets.py`** (the one check no build can replace): a rung that
lands fewer files than the patch it replaces still compiles, because the missing hunks live inside other files
-- that is how block 00's `ggml-alloc.c` half was dropped at r328 and shipped for two releases (issue #345).
Run it **per rung while porting** (`--rung NNNN --new-chain ~/llama.cpp --new-base <base>`) and on the whole
set before the tag; a deliberate disappearance is a decision row in `patches/FILE-SET-RETIREMENTS.tsv` with a
reason. `scripts/validate-set.sh` runs it too, so a re-cut that shrinks a block fails the delivery check.

### Upstreaming

Some blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. The internal all-reduce (**block `0001`**) is gated to
RDNA4 pending community verification on RDNA3 pairs. See `MANIFESTS.md`
for the verification contract and `BASELINE.md` for provenance.

---

## Community Acknowledgements

This work is becoming a community effort and I'd like to offer special thanks to the
following users for the assistance in finding issues and offering solutions!

- https://github.com/1337hero
- https://github.com/bakon11
- https://github.com/burnoutIA  (PR #317: the 2..8-column speculative-verify matvec family in WMMA `iu8` — Q4_K/Q5_K/Q6_K/IQ4_XS read in their GGUF layout on the existing q8_1 activations, each weight dequantized once for all the verified columns, plus the design notes and the min-term pitfall (`d.SUM(xq)`, not `ds.y`).  Folded into block 07 in r290 as an opt-in (`--wmma-verify N|auto`, `GGML_CUDA_WMMA_MMVQ` kept as the A/B alias) with the Q8_0 case and the per-format stride unit added here; the band and end-to-end numbers are in `WORKLOG.md` 261115.002 and `ENVIRONMENT.md` §7.6.  r312 made `auto` a per-type floor from a DRAM-cold sweep (5 for Q4_K/Q5_K, 8 for Q6_K/IQ4_XS/Q8_0) and refuted the "narrow low-band arm" follow-up: at 2..4 columns the K-split kernel is at 92-98 % of the memory roof — `WORKLOG.md` 261118.005)
- https://github.com/briansp2020  (block-13 moe_weighted_reduction float4 remainder fix + block-14 MUL_MAT_ID pair-fusion layout gate, issues #19 and #18; PR #213: the split-state cache hit returns instead of copying the state, folded into block 09 in r132); PR #326: the Q8_0 hyper-connection mixer path for UD quants (persistent `down_dots` counters, planar output, a Q8_1 epilogue, the combine+norm+Q8_1 kernel, a weight-preloading up band, an up+collapse launch and the F32 inject rows on the down grid) -- folded into block 11 in r313, output byte-identical on `UD-Q4_K_XL` across all ten hc-fusion arms, +7.1 % decode, with the redundant `_Q8` sub-switches folded into the switches that already covered the class (`WORKLOG.md` 261118.007); PR #315: sequence parallelism for the qwen4exp hyper-connection work -- the residual streams stay token-split through a 2+-device `-sm tensor` prefill, a new `GGML_OP_TOKEN_SCATTER` and a copy-engine reduce-scatter / token all-gather pair on the all-reduce's own scratch, default ON (`LLAMA_SEQPAR=0` off), +7 % copy-engine and +2.2 % default-chain prefill at 31.5k; reviewed, reproduced to the digit and extended to 3 cards here)
- https://github.com/eoprede
- https://github.com/jstamagal  (PR #338: three RDNA2-`gfx1030` improvements, folded into the delivery as RDNA2-gated code in r334 — a one-warp `RMS_NORM` launch for `ncols <= 128` with a `dst16`-free kernel specialization, a `GATED_DELTA_NET` chunked-prefill gate keyed on `H * n_seqs` (gfx1030 has no BF16 WMMA, so the FP32 chunked kernel only wins on a low-head long prefill), the removal of block 05's stale RDNA2 `cols_per_block`/`ncols2` caps (the RDNA2 occupancy is computed by `launch_fattn` itself now), and a `mul_mat_vec_q_moe` K-block walk with the rows side by side — bit-identical, 1.26-1.32x on quantized `MUL_MAT_ID` at `n = 1..8`; every piece is compile-gated to the RDNA2 device pass and armed at run time from the device's `cc`, so gfx1201/gfx1151/gfx1100 keep their numerics, `ENVIRONMENT.md` §7.4)
- https://github.com/mjohanning99  (PR #312: the mixed K/V cache types — a q8_0 K with a q4_0 / q5_0 / q5_1 V read natively through a second tile type parameter, head 256 only, with the QSA refusal and the one-V-type-function contract; promoted as block 17 in r287, moved to optional block `0101` in the r288 block-number re-org, `scripts/gate-mixed-kv.sh`)
- https://github.com/overdoingism  (PR #360: the server's tool-text protection — a tool result's, a tool call's arguments' and an assistant text's special-token strings kept as TEXT so data from a tool cannot forge a turn; folded into block 12 as the r357 amendment, default-on with a kill switch and a census, `scripts/gate-tool-text.sh`, with three integration fixes named in `archive/work/tool-text/README.md` — the space fallback keyed on the request's media rather than the server's mmproj capability, the mark strip armed with the feature so the kill switch restores stock, and a detokenize-and-compare check on the piecewise re-tokenization.  Issue #45: the RDNA4 head-256 GQA-6 decode/verify flash-attention band, reported with the diagnosis, op-level data, the round-robin KV split idea and a working opt-in patch; the r4 block-15 band is built on that submission.  PR #223: the fully-masked KV-group skip extended to the band itself, so a resident idle conversation no longer taxes the active one's decode — folded into block 05 in r133.  Issue #224: the r126 row change left widths 5..8 on a different per-column KV split, so greedy decode with a draft diverged from plain decode; the band's own 64-column instantiation is back in r134, with a build-time guard and a width probe that reaches it)
- https://github.com/pwilkin  (the `strix-halo` fork at https://github.com/pwilkin/llama.cpp/commits/strix-halo/, heavily adapted for the MMB bf16-WMMA dequant-weight GEMM work, now folded into the delivery — formerly `archive/work/mmb-general/`)
- https://github.com/slavap  (the gfx1030 / Radeon Pro V620 bring-up: the block 00-04 add-on set (PR #165) and the FA follow-ups (PR #181, #198); PR #218: MMB's per-arch dispatch keyed to the running device (a mixed-arch box ran RDNA3 kernels on the RDNA2 card and zeroed its output projection) and the Q8_1 cache's view-offset identity + source-range guard — folded into blocks 06/08/10 in r132)
- https://github.com/RikshaDriver  (issue #221 / PR #222: the three kq-mask fill sites that lacked their sibling's buffer guard — a hard startup abort at `-np 1` on `bailingmoe3`, root-caused and validated on gfx1102; folded into block 05 in r131)
- https://github.com/tungel
- https://github.com/DanoPTT  (block-08 mul_mat+add through-view shape guard, PR #15; and issue #45 follow-up: the f16 verify-width diagnosis / the f16 + bf16 band coverage folded into block 15 in r5/r6, measured on their R9700)

I, and everyone else who benefits from this work, really appreciate you!

---

## Inspirational Works

While most of the work in this repository are original works of my own, there are
some significant portions, most notably around the prefill tuning, inspired by the
excellent work performed by the community of: https://github.com/halo-box/strix-llama.cpp

Thank you to all the maintainers of the Strix Halo Llama.cpp project

Of course none of this would be possible without the baseline that all of this rests
on, and that is the huge community over at https://github.com/ggml-org/llama.cpp

Many thanks to the llama.cpp team

---

## License

Same as llama.cpp (MIT), with one documented exception: block 03's chunked Gated DeltaNet kernels
(`gated_delta_net_chunked_bf16.cu`, `gated_delta_net_chunked_bf16_gfx11.cu`) are ports of libr4d kernels,
and libr4d ships inside [radiance](https://codeberg.org/StillDeadcode/radiance) under **Apache-2.0**.  Those
parts stay Apache-2.0; radiance's `LICENSE` and `NOTICE` are reproduced unchanged in
[`third_party/radiance/`](third_party/radiance/).  See [`NOTICE`](NOTICE) for the exact scope — it matters
if you redistribute the applied tree.
