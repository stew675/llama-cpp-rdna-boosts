# llama-cpp-rdna-boosts

A patch collection that brings **AMD RDNA-specific performance work** to
[llama.cpp](https://github.com/ggml-org/llama.cpp): MTP decode, chunked
gated-delta-net prefill, BF16 KV and WMMA flash-attention, fused MoE and
k-quant decode paths, a hybrid all-reduce, qwen4exp (Qwen3.8-Flash-Next)
support, and an attention-memory campaign that frees several GiB of VRAM.

**Everything this delivery adds is default-on — you should not need to set any environment variable.**
[`ENVIRONMENT.md`](ENVIRONMENT.md) documents every variable the set adds or repoints: its default, and
whether it is a kill-switch (default-on, set to disable), an opt-in, a tuning value, or a diagnostic.

It ships as **16 patches** (block 00 + blocks 01-15) for a clean llama.cpp checkout at the fork point
**`a691b2345`** (upstream master).  Each block is one `git am` commit, and the series is linear (a block
assumes its predecessors), so the set is applied whole.  **Block 14 is temporary** -- a local
`-sm relaxed-tensor` relaxation for the MoE amax architectures, kept separate so it can be dropped; see
[block 14](patches/README.md#block-14--sm-relaxed-tensor-for-the-moe-amax-arches-temporary-expect-removal)
and treat it as a stop-gap, not a feature.  The current release is **`v16-a691b2345-r251`**: the release
identity (fork point, canonical tip/tree, block count and per-artifact sha256) is
[`release.json`](release.json), and the release note for it -- and for every earlier `rN` -- is the top of
[`WORKLOG.md`](WORKLOG.md) (newest first; it is the only dated record).  This README describes the set,
not the changelog.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a691b2345
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts (16 commits)

# OPT-IN, not part of the delivery: optional artifacts on top (NOT the release tree)
bash <path-to-this-repo>/scripts/apply-all.sh --include-optional .       # asks [y/N] per artifact
bash <path-to-this-repo>/scripts/apply-all.sh --include-optional-all .   # every artifact, no prompt

# OPT-IN, not part of the delivery: a beta candidate on top (NOT the release tree -- beta testing only)
bash <path-to-this-repo>/scripts/apply-all.sh --include-beta .   # 16 blocks + beta/*/*.patch
```

- One-line summary of each block: [The 16 blocks](#the-16-blocks-v21)
- What each block contains, its knobs and its purity notes: [`patches/README.md`](patches/README.md)
- The folded `mmb`/QSA campaign: [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery)
- Every environment variable, with its default and class: [`ENVIRONMENT.md`](ENVIRONMENT.md)
- **Running a model bigger than your VRAM on one card** (`-ncmoe` + the MoE expert cache, with the
  measured config tables and the measurement method): [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md)
- Current status and validation: [`release.json`](release.json) (source of truth) and [`MANIFESTS.md`](MANIFESTS.md)

## Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(the tag is the release identity: `v<blocks>-<fork-point>-r<N>`, e.g. **`v16-a691b2345-r243`**; the block
count in the tag follows the set, so `v15-*` tags are the 15-block releases up to r177 and `v16-*` the
16-block releases from r178 on.  `r1` is the
re-base of the set onto that fork point and every later `rN` is a dated change recorded in
[`WORKLOG.md`](WORKLOG.md), newest first).

## Supported architectures

The set targets the **RDNA3 / RDNA3.5 / RDNA4** GPU families, with RDNA2 as best-effort:

| family | arches | example parts | status |
|--------|--------|---------------|--------|
| RDNA 3 | `gfx1100` | RX 7900 XTX/XT, RX 7800 XT, ... | supported |
| RDNA 3.5 | `gfx1150`/`gfx1151` | Strix Point / Strix Halo APUs | supported |
| RDNA 4 | `gfx1200`/`gfx1201` | RX 9060 XT; RX 9070 / 9070 XT | supported |
| RDNA 2 | `gfx1030` | RX 6800/6800 XT/6900 XT, Radeon Pro V620 | best-effort; externally validated (V620), not run in our fleet |

> **Strix Point (`gfx1150`) is in the target set as of 2026-10-05.**  Images
> published before then (for example the `server-rocm-10.0` build at `2dbece4`)
> carry no `gfx1150` code object and fail on the first kernel launch (`ROCm
> error: device kernel image is invalid`).  For those, build from source with
> `gfx1150` in the target list, or set `HSA_OVERRIDE_GFX_VERSION=11.5.1` at
> runtime to select the `gfx1151` code objects.  See [Strix Point
> (gfx1150)](CONTAINERS.md#strix-point-gfx1150) for the validation and both
> paths.

> **RDNA2 (`gfx1030`) is best-effort: not a supported target, but no longer unvalidated.**  It is
> compiled into the image so RDNA2 owners pick up the generic, runtime-dispatched
> boosts instead of a first-kernel `ROCm error: invalid kernel file` (issue
> #158), and the RDNA3/4-only paths fall back to the upstream non-WMMA kernels at
> runtime.  As of **r76** the set also folds `@slavap`'s tuned RDNA2 tile-FA and
> k-quant work (PR #165: per-DKQ tile config, launch caps, WGP-aware occupancy,
> the kv=113 partial-batch load fix and the wide-kernel Q-from-global path;
> Navi21 MMVQ rows/warps + MMQ float scale).  All of it is gated to RDNA2, so
> gfx1100/gfx1151/gfx1201 keep their previous numerics.  **External validation
> (TODO #66, closed):** a Radeon Pro V620 (`gfx1030`) passed `test-backend-ops`
> (`FLASH_ATTN_EXT` 6389/6389, `MUL_MAT` 2444/2444) and generated coherently with
> `-fa 1` in a 3-card mixed `gfx1100`+`gfx1030` `-sm tensor` split down to 32K
> depth.  There is still no RDNA2 device in the maintainer's fleet, so there are
> no performance claims and no purity gates for it — an RDNA2 result is
> informational, never a delivery gate — and `-fa on` can still hit upstream
> RDNA2 tile-kernel defects, where `-fa off` is the workaround.  See
> [RDNA2
> (gfx1030)](CONTAINERS.md#rdna2-gfx1030--best-effort-externally-validated).

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

Arch selection is **runtime** everywhere in the set (device `cc` / `gcnArchName`),
so a multi-arch build such as `GPU_TARGETS="gfx1030;gfx1100;gfx1151;gfx1201"`
yields one binary that picks the right path on whichever of these it runs on.
The WMMA kernel *bodies* are compile-gated by `RDNA3`/`RDNA4`; on RDNA2 the
non-WMMA arms compile and the runtime predicates keep the WMMA kernels from
being launched. The one genuine exception is **block 12** — its internal all-reduce is RDNA4-only
(gfx1200/gfx1201) and falls back to RCCL elsewhere (see
`patches/README.md` for the gate and env knobs).  Block 13's fused
MoE MMQ gate now covers RDNA4 + RDNA3_5 + RDNA3_0 (gfx1151 validated
2026-09-05, gfx1100 validated 2026-09-05 — see
[`WORKLOG.md`](WORKLOG.md)).

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # the policy layer: what is mandatory, scope, how a release is cut
├── INVARIANTS.md          # the engineering facts: measured invariants / "do not re-derive" (pre-ship reading)
├── MANIFESTS.md           # apply order + the verification contract (release gates per block)
├── BASELINE.md            # fork point, what the patches are, how they are cut, drift policy
├── TODO.md                # forward-looking tracker: open items, limitations, parked ideas only
├── GREEDY-PURITY.md       # purity rulebook: invariants + one summary entry per finding (read before shipping)
│                          #   narratives/evidence: WORKLOG.md + benchmarks/ (and pre-r100: archive/docs/GREEDY-PURITY-FINDINGS.md)
├── WORKLOG.md             # dated delivery records (newest first; the only release history)
├── ENVIRONMENT.md         # every env var the set adds or repoints, with its default and class
├── CONTAINERS.md          # the ROCm container images + the off-host build watcher
├── COMMUNITY-CONFIG.md    # running a model bigger than VRAM (`-ncmoe` + the expert cache)
├── rdna-boosts-all.patch  # convenience: the entire 16-block net as ONE patch
├── patches/               # the delivery set: 0000-0015 (one `git am` block each)
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/               # apply / regenerate / validate / publish tooling; index: scripts/README.md
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift; --include-beta)
│   ├── gate-prefill-logits.sh  # prefill-logit KLD release gate (issue #113)
│   ├── gate-per-block-build.sh # a clean `all` build per block commit
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── prompts/               # versioned, hash-stable test prompts (sha256-recorded; never edited in place)
├── wiki/                  # source for the GitHub wiki (Home, MTP & Adaptive MTP, Quick Reference); see wiki/README.md
├── wip/                   # ACTIVE exploration only, never applied; index: wip/CAMPAIGNS.md
├── beta/                  # promotion-candidate patches pending a beta test (NOT part of the delivery;
│                          #   apply with `apply-all.sh --include-beta`; currently empty, see beta/README.md)
├── optional/              # opt-in features with no validated win here (NOT part of the delivery; apply with
│                          #   `apply-all.sh --include-optional` and a [y/N] per artifact; see optional/README.md)
├── upstream/              # upstream-PR candidates: one sub-directory each (README.md + patch) + index
└── archive/               # the rest: archive/work/ (closed experiments + the archived wip/ trees) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 16 blocks (V2.1)

`release.json` (V2.1; the current release).  Each row is one `git am` commit.

| patch | subject | concern |
|-------|---------|---------|
| `0000` | **foundation** — band-uniformity, Meta-wrapper, allocator fixes | old-00 + old-05 + old-07 + old-09a + the multi-ubatch logits readback + the `ggml-alloc` view release; + the meta hot-path dispatcher restoration (r86) |
| `0001` | **hybrid HIP all-reduce** (RDNA4-gated) | the old block 12; + no per-AR `ev.ker` and the opt-in P2P-large path (r52); + the large-tensor arm is chosen per model with **peer access never enabled** — copy engine for a dense model, RCCL's ring for a host-offloaded one (r165, `TODO #109`) |
| `0002` | **adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`) | old-01; + opt-in MTP prompt-tail ingest and the last-row prompt copy (r52); + the embeddings-layer-input preset API the draft setup calls (r94 block-boundary repair); + the draft context's KV cache type follows the main `-ctk` / `-ctv` unless `-ctkd` / `-ctvd` pins it (r125, issue #216) |
| `0003` | **chunked gated-delta-net prefill** (bf16/WMMA GDN) | old-02; + the per-sequence rollback tracker folds in every ubatch and `LLAMA_RS_ROLLBACK_DEBUG` (r91) |
| `0004` | **BF16 KV cache + native-BF16 flash attention** | old-03 |
| `0005` | **flash-attention band / native quantized K-V / derived kq mask** | old-04 FA + the old-15 FA work; + the 2-byte arm joins the 32-column band block and the head-512 MMA row is retuned (r92); + the q8_0 fetch-ahead band variant, default on (`GGML_HIP_FA_BAND_FETCH_AHEAD=0`), r98 PR #195; + one tile K/V type selection for the instantiation and the F16 scratch, and an RDNA2-gated checked tail, r99 PR #198; + the derived mask's input is reusable through a hybrid memory context, so a >8-token verify step stops rebuilding its graph (r164, PR #242) |
| `0006` | **MMB quantized prefill GEMM** | old-08 MMB |
| `0007` | **k-quant boosts** (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) | old-10 + old-04 MMQ + old-15 Q1_0/grid/PR57; + the RDNA3_0 per-shape mmvq `nwarps` (r93) |
| `0008` | **fused kernels** (conv1d/RMS/top-k MoE/verify fusions + fused MoE gate+up+GLU) | old-08 fusions + old-13 fused MoE + old-15 PR63/114; + `topk_moe` sorted-key argmax (r52); + the RDNA3_0 per-shape mmvq `nwarps` (r93) |
| `0009` | **host-resident MoE** (iface, scheduler, Meta, movable-boundary slab, cache engine, prefill cache staging) | old-06 MoE + old-09b + old-11 + old-13 engine + old-15 hooks/arena; + the arena-host staging engine and its residency snapshot (r83); + the slab drop reference (r85); + the slab transient absorption common half and the derived draft reserve (r86); + the complete device-iface fills and the pre-loaded draft-model handoff (r94 block-boundary repair); + the batched f32 CPY run on RDNA3_0 (r95, PR #193); + the restored copy callback, so the upstream `--moe-cache-mib` engine works (r99, TODO #56); + the split-state walk's per-thread scratch arena, so a graph re-allocation stops allocating one heap node per visited tensor (r100, issue #184); + the `--fit` draft reserve charged only to the KV layers a drafter owns (r101), and a loaded-but-unused draft model is not loaded or given a context (r101); + the cache's traffic report divides a decode-band op census by the sizing-time table count (r144), and `GGML_CUDA_GRAPH_TRACE` reports how every `graph_compute` call was served (r144); + the decode band no longer reads the routing back to the host: a device-side slot remap (`GGML_MOE_DEVREMAP`, host policy one call late) and a `--host-experts pool` resident miss read in place (`MOE_HOST_POOL_COLD`) — +8.4 % TG pinned / +4.8 % pool, byte-identical text and MTP acceptance (r203, PR #266); + the remap is **tensor-split and covering-pool aware** (r221, TODO #269): the Meta backend forwards it per device (+16.1 % / +28.4 % decode at 2 / 3 devices) and a host pool that covers its tensor is a cold source (+9.9 % on the default `-lme auto` mirror) |
| `0010` | **infrastructure** — slab, fusion dispatcher, async input ring, GDN/rope fusions | the translation units split out of `ggml-cuda.cu` (`try-fuse`, slab/arena/vmm, MoE cache hooks) + short-K MoE rows-per-block (r52); + the `stage_from_cache` hook (r83); + the slab narrow band (r85); + the slab transient absorption CUDA/slab half (r86); + the async graph-input ring and its r90 split-input snapshot fix (default-on); + the fixed-D sequential GDN snapshot tail (r91, `GGML_CUDA_GDN_TAIL=0`); + the work-boundary shrink invalidates captured graphs (r94, issue #187); + the verify-band Q8_1 fusions and the GDN conv verify path on RDNA3_0 (r95, PR #192 + #193); + gemma4 dense `-sm tensor` (the Meta split state accepts an empty-or-whole slice for per-row ops, the rotated-K `mul_mat` and flash-attention K/V), the expert cache sizing / multi-slice tables / segmented async upload, the iSWA `q, v, k` KV-builder order and the per-origin async-input ring (r101); + the GDN/attention-gate decode fusions (r102, PR #200) with the r108 multi-sequence fix, the bf16 RoPE destinations (r103), and the r183 GDN decode folds with the shared-expert gate fold (PR #256) plus the 2-D copy guard; + the `q4_0` K/V cache write fold in the rope + FWHT kernel (r199, PR #262, `GGML_CUDA_KV_FOLD_Q4_0=0`), +10.8 % tg128 on a `q4_0`-KV 30B-A3B; + the fusion-gate oracle's 33-case `{BF16,1024}`/`{Q8_0,1024}`/`{Q8_0,5120}` matrix (r202, tests only) |
| `0011` | **qwen4exp** (model, hyper-connections, lazy reader, QSA/indexer) | old-14 + old-15 HC16/policy/QSA; the model graph, the QSA/indexer/HC ops and kernels, the managed lazy reader and the keys-only indexer cache, the qwen4exp tensor-split arch gate, and the qwen4exp fusion arms; split out of `0010` in r109 (tree-identical) |
| `0012` | **campaign remainder** (CPU wins, DFlash, FA build grouping, mmap reclaim, batched state copy; **+ r207: host-pool-speed** — a POOL-backed table gathers in the prefill band by default, the `RAM_PCT` ceiling scales every table, `MOE_HOST_POOL_SIZING=1` (opt-in), the pool's diagnostics report the run, `MOE_HOST_POOL_PREFETCH` removed) | old-15 W1-W3 + old-06/15 mmap reclaim + old-16; + the expected `--fit` `ctx_other` probe failure logs as a warning, and the draft-context creation is gated on an enabled type that uses it (r101); + the batched state save/restore, the pipelined copy over it, and the threaded streaming host copy (r152, PR #235); + the DFlash draft's compute reserve split over its injection and noise-block graphs (r162, issue #239); + the DFlash target writes its extracted layer inputs into F1's device store from inside the graph, so they stop being graph outputs and the host rows are reserved only when the host path runs (r163, issue #240) |
| `0013` | **MTP draft vocabulary subset** (`--spec-draft-mtp-vocab N`, **opt-in**, default off) | the MTP draft scores only the LM-head rows for token ids < N (+ control/user ids) and scatters them into a full-vocab row of `-inf` — the target's verification is untouched, so the draft can only lose proposals; the `ggml-backend-meta` row-range read that makes it work under `-sm tensor` (r107) |
| `0015` | **streamed KV** (**all three placements opt-in, default OFF since r247**: `--kv-window` / `--kv-attendance-pool` window and pool, and `--spec-draft-kv-window`, the MTP draft ring — see `ENVIRONMENT.md` §17 for why the draft ring's old default was a decode regression) | the KV's VRAM cost becomes a resident **window** of the newest `W` cells instead of `-c` cells: one VA reservation holds the cache, the window is VRAM and everything older is pinned system RAM, so a context that OOMed now runs (27B Q8_0 `-c 262144` = 128 MiB VRAM + 8576 MiB pinned host) and the VRAM the KV does not need *yet* goes to the expert-cache arena (+2.2…+2.9 GiB at `-c 262144`, `-n 2000`). A spilled KV is numerically transparent (prefill-logit KLD −0.000000 / 100.000 % on a spilling arm); unset = byte-identical to a build without it, including the OOM (r178, promoted from `archive/work/kv-residency`)  **The same block also carries the attendance pool** (r190, Goal B: `--kv-attendance-pool on\|off`, whose size is derived from the model since r247 — the KV's authoritative copy in a pinned host mirror plus a compact device pool of indexer-block pages the sparse decode resolves through, 204 MiB instead of 1.6-2.4 GiB of resident KV at 103k-180k, `--fit`-visible) and the streaming arm policy (a streamed cache takes the sparse decode arm, +34 %/+47 % on a spilled KV).  Both placements are **opt-in and default off**  + r204: the same **window across a `-sm tensor` split** (`TODO` #124): one VMM region per child device of the Meta wrapper, the grow fanned out to every child, `--fit` sizing it against the split's per-device budget (a MIRRORED dsv4/HRM cache still declines it); + r222: the provider binds the device at every driver-reaching entry point and a buffer type answers capability without touching the driver, so a **plain** 2-GPU `-sm tensor` run does no VMM driver call at all (issue #275: the construction-time probe ran cross-context, which crashed the first kernel launch on Windows); + r248: the MTP draft ring is a **true** ring, so a batch the position window cannot place (an M-RoPE image chunk) evicts the drafter's oldest cells instead of failing the request, and a drafter failure can no longer return a 500 (issue #297) |
| `0014` | **`-sm relaxed-tensor`** (**TEMPORARY**, `flint`/`mellum`/`mistral4` only) | tensor split that also admits a replicated tensor whose values differ per GPU (the MoE amax row max, which no device can complete). Exact on RDNA/HIP, zero VRAM, and **not a supported feature**: it relaxes upstream #31081 and is expected to be dropped when upstream fixes the amax under a tensor split. Delete this patch to get back to the r136 tree |

Phase-2 additionally moved each concern into a self-contained file: the `ggml_cuda_try_fuse` arms, the
MMVQ vec-dot/VDR selectors + MoE kernel (`mmvq-vecdot.cuh`, `mmvq-moe.cu`), the FA native-KV/band
predicates (`fattn-rdna.cuh`), and the four `ggml-cuda.cu` TUs (`ggml-cuda-try-fuse.cu`,
`ggml-cuda-slab.cu`, `ggml-cuda-moe-cache-hooks.cu`, `ggml-cuda-arena.cu`; `ggml-cuda.cu` 9819 → 6370
lines).

The **pre-V2.0 16-block notes** (provenance and edit history of the old block numbering) are in
[`archive/docs/pre-v2-block-notes.md`](archive/docs/pre-v2-block-notes.md).

## Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
FORK=https://github.com/ggml-org/llama.cpp
git clone $FORK && cd llama.cpp
git checkout "$BASE"

# 2. apply the set (automated; strict 16/16 git am on the recorded base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0011  (one commit per block on a fresh `rdna-boosts` branch)

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

### Windows: use the ROCm (TheRock) HIP runtime, not the display driver's

On Windows the HIP runtime (`amdhip64_7.dll`) is usually resolved from `C:\Windows\System32`, i.e. the copy
the **Adrenalin display driver** ships, even when a ROCm/TheRock tree sits beside `llama-server.exe`.  The two
runtimes are not interchangeable for the host-expert path.  On the driver copy the pinned host tier stops at
**~9.3 GiB** of adapter shared (aperture) usage even on a 64 GiB machine, the tier latches off, and on one
reported box the process then spins inside the runtime's own wait loop after the post-prefill re-arm
([issue #278](https://github.com/stew675/llama-cpp-rdna-boosts/issues/278)).  The **same binary** with the
official TheRock ROCm 10.1 runtime beside it pinned **16-40 GiB**, filled the whole host expert set
(141/141 tensors, a full mirror) and ran the same arm clean at **370.7 t/s** prefill where the driver runtime
managed 39.8.

Copy the ROCm/TheRock runtime DLLs next to the executable -- `amdhip64_7.dll`, `amd_comgr.dll`,
`hiprtc0716.dll`, `rocm_kpack.dll`, plus the rocBLAS/hipBLAS/hipBLASLt DLLs from the same tree -- and confirm
the startup line names the binary's directory:

```
moe_cache_init: HIP runtime 7.16.26385 from C:\path\to\llama-server\amdhip64_7.dll
```

If it instead names `C:\Windows\System32`, the delivery logs a WARN with this remedy.  A box that cannot
switch runtimes can still run the host-expert path with **`MOE_HOST_POOL_PIN=0`**: the pinned tier stays closed
(no pool, no bounce/span ring) and every fill takes the pageable model mapping -- correct and slower, but it
avoids the tier that the driver runtime cannot grow.  See `ENVIRONMENT.md` for both variables.

### Manual equivalent

```bash
git am patches/00*.patch   # block 00 + blocks 01-15
```

### The `mmb` campaign is in the delivery

On the **`beta-integration`** branch the `mmb` (bf16-WMMA dequant weight GEMM) / QSA / indexer
campaign — the former 28-patch `archive/work/mmb-general/` set — is **folded directly into the 14 delivery
blocks**, so the normal workflow above is all there is to apply.  There is **no separate beta layer
any more**:

* **block 08** absorbs the `mmb` core, the RDNA4 fragment port and per-arch tuning, the
  GDN/PLE conv1d + narrow-row RMS-norm prefill fusions, the `mmb` fusion stand-downs and the extended
  MMVQ routed band;
* **block 09** absorbs the host-buffer input layer and the tiny-CPU-split single-thread fix (the system
  work of the old catch-all block 06);
* **block 10** absorbs the deferred `mmb` pair-fusion stand-down and the infrastructure split out of
  `ggml-cuda.cu` (slab, fusion dispatcher, async input ring, the GDN/rope fusions);
* **block 11** absorbs the qwen4exp group: `qsa3`, the fused indexer top-k + prefill score fusions, HC16,
  `hc_gate_mix`, sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults and the lazy reader
  (the old block 14 + the old block 15 remainder).

Applying the 16 patches to the fork point in [`release.json`](release.json) therefore reproduces the
**full campaign tree** in one pass:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a691b2345          # the base recorded in release.json
bash <path-to-this-repo>/scripts/apply-all.sh .   # 16/16 strict, tree checked against release.json
```

The per-patch fold mapping and validation record are in
[`archive/work/beta-integration/integration.md`](archive/work/beta-integration/integration.md).  Much of the MMB
kernel work is heavily adapted from **[pwilkin](https://github.com/pwilkin)**'s
[`strix-halo` fork](https://github.com/pwilkin/llama.cpp/commits/strix-halo/), with thanks.

> **Historical record only.**  [`archive/work/mmb-general/`](archive/work/mmb-general/) (the 28 patch files, the
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

> **With host-resident MoE experts (`-ncmoe`/`-cmoe`), keep every draft inside the expert cache's verify
> band.**  A verify batch is `n_draft + 1` rows and the expert cache serves it from its VRAM arena only up to
> its decode/verify band (16 tokens on RDNA4, 8 on RDNA3).  Above the band the routed experts are streamed
> from host memory on **every** step, which on a host-expert config costs far more than the extra accepted
> tokens buy — the interaction this README used to warn about.  This is now handled for you:
>
> * `--spec-draft-n-max` is clamped to 15 (a 16-row verify) by the delivery.
> * The **draftless** n-gram speculators (`ngram-mod`, `ngram-simple`, `ngram-map-k*`), which default to a
>   48-64 token draft, are clamped to the band automatically (r123, `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0`
>   opts out).  Measured on 2× R9700 with Flash-Next IQ4_NL `-sm tensor -ncmoe 36`: the combo below goes
>   **73.9 -> 168.3 t/s** and `draft-mtp,ngram-mod` **74.1 -> 160.2 t/s**.
>
> So `--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45 --spec-draft-n-max 9` remains a
> good pick on a host-expert config; no extra flags are needed.

> **On a card that is close to full.**  The MTP draft context is a second context on the same weights, so
> it costs a real, fixed slice of VRAM on top of the model: its own KV cache, its own compute buffer and
> its own output buffer.  The KV half now follows the main model's cache type (r125), so `-ctk`/`-ctv` set
> the draft's size too; `-ctkd`/`-ctvd` pin it independently if you want the deep-lookback f16 draft.  For
> scale, a dense 27B IQ3_S with `-c 131072 -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024` asks for about **14.8
> GiB** of device memory in total (11.0 weights + 2.3 target KV + 0.6 recurrent state + 0.14 draft KV +
> 0.8 compute + the driver's ~1.4 GiB of unaccounted code objects and workspace), and that is the *steady*
> state; the first prefill peaks about 1.5 GiB above it.  On Windows a set that exceeds the card is silent
> for a while (WDDM pages it to system memory) and then surfaces as a device loss, which a lazy rocBLAS
> init reports as `rocBLAS error: Could not initialize Tensile host: No devices found` -- see issue #216.
> The levers, most effective first: `-c` (the target KV dominates), `-ub` (the compute buffer), `-ngl`
> (real layer offload -- note it is a **layer count** and a model with a NextN head has
> `n_layer + 1` layers, so `-ngl 85` on a 65-layer model offloads everything, exactly like `-ngl 99`),
> and the KV types.  Leave `-ngl` and `-c` unset if you want `--fit` to size the context for you: with
> either one set, the fit declines and exits.  Two things it does **not** fix:
>
> * A model with **no active arena** (fully device-resident, or the CPU expert path
>   `MOE_EXPERT_CACHE_MIB=0`) is never clamped — there the wide draft is the win (144.7 -> 378.7 t/s on the
>   same model fully resident), so keep the defaults.  A host-expert config on a card too small to create the
>   movable-boundary slab lands on the same streaming path (the loader prints `the movable-boundary slab
>   does NOT fit ... STREAMING`) and is **also** never clamped (r129): the clamp is decided after the
>   context exists, i.e. after the slab has been created or has declined, so a streaming config keeps its
>   wide draft (2x R9700 control: 67.6 vs 52.2 t/s capped).  The slab needs the estimated work buffer + the
>   `GGML_CUDA_SLAB_MIN_ARENA_MIB` floor (2048) + the `GGML_CUDA_SLAB_RING_MIB` hole, and on a 12 GiB card
>   lowering the floor or setting `GGML_CUDA_SLAB_RING_MIB=0` can be what makes it fit — the decline
>   message now names all three knobs.
> * If you deliberately set `--spec-ngram-mod-n-max` above the band (issue #214's `256`, for instance), it
>   is clamped too; note that a naked `--spec-ngram-mod-n-max 15` **without** lowering
>   `--spec-ngram-mod-n-min` (default 48) silently disables `ngram-mod` rather than shortening it.
>
> Mechanism and measurements: [`GREEDY-PURITY.md`](GREEDY-PURITY.md) §50,
> [`ENVIRONMENT.md`](ENVIRONMENT.md) §3, and
> [`archive/work/ngram-mod-host-experts/EVIDENCE-2026-10-24.md`](archive/work/ngram-mod-host-experts/EVIDENCE-2026-10-24.md).

Guidance:

* **Cap.**  `9` is the general-purpose pick and the optimum on a **single card** (a cap of `6` costs
  11-14 % on code there).  On a **multi-card tensor split** `6-7` is ~1 % better.  A lower cap saves
  only a small verify-batch scratch, not the model/KV memory.
* **`n_match`.**  Keep it `>= 40` **and an integer multiple of the cap** — `9/45`, `8/48`, `6/42`.
  Too short (`nm24`) makes ngram fire on incidental code repeats and lose code throughput; a
  non-multiple (e.g. `nm42` at cap 12) degrades acceptance.
* **Host-resident experts.**  Use the combo above (it is clamped to the verify band automatically).  If
  you run with the expert cache disabled (`MOE_EXPERT_CACHE_MIB=0`, the CPU expert path) or on RDNA3
  (where the band is 8 tokens), a draft deeper than the band is a warning, not a clamp for the MTP family:
  lower `--spec-draft-n-max` yourself if the decode rate disappoints.
* The combo changes the draft strategy, so it is **opt-in** — the controller default stays plain
  `--spec-type draft-mtp-adaptive`.

Full derivation (the 4×4 corpus, all four cells, and the rejected controller alternatives):
**[`archive/work/mtp-journey-2026-09-17/SUMMARY.md`](archive/work/mtp-journey-2026-09-17/SUMMARY.md)** (narrative in its
[`README.md`](archive/work/mtp-journey-2026-09-17/README.md)); the dated controller records are in
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

Full matrix, raw CSVs and the A/B harness: [`archive/work/kq-mask-derived-ab/`](archive/work/kq-mask-derived-ab/); the
2026-09-19 block-15 (r7) amendment in [`patches/README.md`](patches/README.md).

## Cross-start determinism on ROCm (issue #67)

Issue [#67](https://github.com/stew675/llama-cpp-rdna-boosts/issues/67) (from #58 item D, reported by
[@DanoPTT](https://github.com/DanoPTT) on Windows / ROCm 10 / gfx1201) tracked Q6_K greedy output that
intermittently swapped one near-tie across fresh `llama-server` starts while staying deterministic within
a start.  It is **closed as an external ROCm issue — nothing to fix in this patch set.**  Two separate
defects were in play:

* **A real bit-transparency bug in the fused rope path, fixed in r25.**  The address-gated
  `ROPE -> VIEW -> SET_ROWS` fusion did not reproduce the unfused chain bit-for-bit: clang contracted the
  fused `<float,__half>` and unfused `<float,float>` template instantiations differently, and one element
  of the prefill K-cache write landed on opposite sides of an f16 rounding boundary.  r25's
  `#pragma clang fp contract(off)` in `rope.cu` makes every instantiation round identically, verified at
  the logits level on Linux (`W=1` hash `60e77916673db071` with the fusion on and off).  See
  [`WORKLOG.md`](WORKLOG.md) for the r25 record.

* **The residual per-start flip is hipBLASLt solution selection in ROCm, not ggml** — and it reproduces
  on a stock upstream build with none of these patches.  The F32 `mul_mat` for `ssm_alpha`/`ssm_beta`
  (5120 -> 48) at 512 prefill columns misses the mmvf/mmf kernels, goes to `hipblasSgemm`, and rocBLAS
  routes it to hipBLASLt (`rocblaslt_matmul`, T,N, m=48 n=512 k=5120).  `HIPBLASLT_LOG_MASK=160` shows a
  per-process pick between a bit-identical reference solution pair (`140231`/`140232`) and a deviating
  pair (`140216`/`140217`); it is random from process to process, so layer 0's prefill output — and with
  it the whole run — differs at startup.  This is
  [ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126).

**Workaround: `ROCBLAS_USE_HIPBLASLT=0`.**  On the reporter's setup, 150 fresh starts per arm: default
**5/150** deviate, `ROCBLAS_USE_HIPBLASLT=0` **0/150** (bit-identical to the reference class), with no
measurable prefill/decode cost on the current build (the earlier text-hash runs measured higher flip
rates, up to ~14 %).  `ROCBLAS_DEFAULT_ATOMICS_MODE=0` does not help.  The selection is **prefill-only**
(decode at n=1 uses mmvf), so decode throughput is unaffected.

**Linux / ROCm 7.14 does not reproduce it** — the reporter's box (Windows / ROCm 10) is where the
per-process selection actually varies.  **Linux with ROCm 10.x does** (2026-10-21, gfx1201 / Linux /
ROCm 10.1.0, which predates the upstream fix: gemma4 26B-A4B F32 MoE router, `m=128 n=375 k=2816 T,N`,
`solution_index 140231` vs `140232`, 3 logit classes in 18 fresh starts of one MTP config; 10/10 identical
once `ROCBLAS_USE_HIPBLASLT=0`).  The fix is `rocm-libraries` `c50ea60130f` (TheRock nightlies from
2026-10-06).  Keep hipBLASLt **enabled** — disabling it costs up to -15 % decode; set
`ROCBLAS_USE_HIPBLASLT=0` for a run that must be reproducible.  The r24 rope-fusion kill switches
(`GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`, `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`) and r25's fix remain
shipped for bisection and for the real defect above.  Full record: [`GREEDY-PURITY.md`](GREEDY-PURITY.md)
§44.  **Second call site (2026-10-21, closes `TODO #75`):** on **35B-A3B** the same lottery rides the GDN F32
`ssm_alpha`/`ssm_beta` GEMM (`F32 2048x32 x N`): 3 logit classes in 8 fresh starts on ROCm 10.1.0, 1 class in
8/8 with `ROCBLAS_USE_HIPBLASLT=0`, the class being the TensileLite spine hipBLASLt picked — and there the
workaround is **free** (+1-3 % prefill, flat decode, ~70 MiB less peak VRAM).  Removing the hipBLASLt dispatch
rather than pinning it now ships as **block 06's `mmf32`** (`GREEDY-PURITY.md` §47): deterministic
(12/12 fresh starts) and +4.2 % prefill on that model.  The campaign record, including the two refuted
shortcuts and the in-code pin prototype that was *not* shipped, is
[`archive/work/hipblaslt-bypass/`](archive/work/hipblaslt-bypass/README.md).

## Concurrent serving with `--kv-unified` is not deterministic (issue #134)

Mixing `-np > 1` with `--kv-unified` does **not** give run-to-run reproducible greedy output: a
sequence's attention result depends on the KV-tile indices of its cells **modulo the per-launch
attention KV-split period** (`pb x nbatch_fa`), and under unified KV every new **decode** token takes the
next cell from the *shared* head, so that residue follows the co-resident session's allocation/scheduling
history.  Identical concurrent repetitions can differ, and a concurrent run can differ from the same
request run solo, at the rounding level (a greedy near-tie can flip).  **Use `--no-kv-unified` when
strict reproducibility is required.**  This is inherent to upstream's unified-KV design — it reproduces
on unmodified upstream (the same behaviour was measured on `097cf33d9`) — and not a defect introduced by this
patch set; the analysis,
reproducers and the exact rule are preserved (closed 2026-10-19) at
`archive/work/kv-cell-offset-nondeterminism/` (`RESOLUTION.md`).  A fix requires period-granular
per-sequence cell *ownership* in the unified pool — an upstream allocator redesign with no throughput
win, so it is not implemented.  Single-request and non-unified runs are unaffected by the unified-KV
effect, and the delivery's op-level purity invariants (`W = 1..8`, `plain == draft-mtp`,
`native == staging`) are scoped to a fixed batch and are unchanged.

**Separate from the unified-KV case (2026-10-16, issue #151 follow-up).**  The reporter (R9700 /
Windows) also sees a start-to-start greedy flip with HIP graphs on, and reports that `--no-kv-unified`
does not remove it.  We could **not** reproduce it on Linux/ROCm 7.14: fresh-process decode-logit
probes (W=1..8) are bit-identical across starts, graphs on and off, including on the reporter's 27B
model with q8_0 KV (`archive/work/kv-cell-offset-nondeterminism/FINDINGS-2026-10-16-revalidation.md`).
It is most likely the known Windows-only per-start flip (the r25 `rope -> view -> set_rows` bit-transparency
fix, and the external hipBLASLt solution-selection issue, workaround `ROCBLAS_USE_HIPBLASLT=0`), both
prefill-driven and independent of graphs.  **Confirmed and closed (TODO #67, 2026-10-20):** the reporter
ran the bisection on decode logits — with `ROCBLAS_USE_HIPBLASLT=0` all starts are bit-identical, and with
it on the same logit classes appear with graphs **on and off**, so graphs play no part.  See
[`archive/work/kv-cell-offset-nondeterminism/FINDINGS-2026-10-20-issue151-reporter-confirmation.md`](archive/work/kv-cell-offset-nondeterminism/FINDINGS-2026-10-20-issue151-reporter-confirmation.md).

## When upstream master moves

The patches are static against the fork point in `release.json.base`. When upstream
drifts and hunks no longer apply, re-base the block commits (the fork checkout carries
them), regenerate the whole set with `scripts/make-patches.sh`, then refresh
`release.json` (`scripts/make-release.sh --base … --tip … --tree …`) and update the
current-state headers. The old `baseline/<sha>`-branch-per-upstream-range workflow
was retired when the delivery moved to the flat patch set on `main`.


## Upstreaming

Some blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. The internal all-reduce (**block 01** in V2.0 numbering) is gated to
RDNA4 pending community verification on RDNA3 pairs. See `MANIFESTS.md`
for the verification contract and `BASELINE.md` for provenance.


## Community Acknowledgements

This work is becoming a community effort and I'd like to offer special thanks to the
following users for the assistance in finding issues and offering solutions!

- https://github.com/1337hero
- https://github.com/bakon11
- https://github.com/briansp2020  (block-13 moe_weighted_reduction float4 remainder fix + block-14 MUL_MAT_ID pair-fusion layout gate, issues #19 and #18; PR #213: the split-state cache hit returns instead of copying the state, folded into block 09 in r132)
- https://github.com/eoprede
- https://github.com/overdoingism  (issue #45: the RDNA4 head-256 GQA-6 decode/verify flash-attention band, reported with the diagnosis, op-level data, the round-robin KV split idea and a working opt-in patch; the r4 block-15 band is built on that submission.  PR #223: the fully-masked KV-group skip extended to the band itself, so a resident idle conversation no longer taxes the active one's decode — folded into block 05 in r133.  Issue #224: the r126 row change left widths 5..8 on a different per-column KV split, so greedy decode with a draft diverged from plain decode; the band's own 64-column instantiation is back in r134, with a build-time guard and a width probe that reaches it)
- https://github.com/pwilkin  (the `strix-halo` fork at https://github.com/pwilkin/llama.cpp/commits/strix-halo/, heavily adapted for the MMB bf16-WMMA dequant-weight GEMM work, now folded into the delivery — formerly `archive/work/mmb-general/`)
- https://github.com/slavap  (the gfx1030 / Radeon Pro V620 bring-up: the block 00-04 add-on set (PR #165) and the FA follow-ups (PR #181, #198); PR #218: MMB's per-arch dispatch keyed to the running device (a mixed-arch box ran RDNA3 kernels on the RDNA2 card and zeroed its output projection) and the Q8_1 cache's view-offset identity + source-range guard — folded into blocks 06/08/10 in r132)
- https://github.com/RikshaDriver  (issue #221 / PR #222: the three kq-mask fill sites that lacked their sibling's buffer guard — a hard startup abort at `-np 1` on `bailingmoe3`, root-caused and validated on gfx1102; folded into block 05 in r131)
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
