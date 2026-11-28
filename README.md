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
  - apply the poatch block series to a clean checkout with `scripts/apply-all.sh`. See [Consumer workflow](#consumer-workflow).
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
- **2 optional blocks** that are NOT in the default set applied only with `--include-optional` (see [`optional/README.md`](optional/README.md))
  - `0100` (two-lane prefill)
  - `0101` (mixed K/V cache types), 
- **temporary block `0300`**, which is applied last.
  - **Block `0300` - a local`-sm relaxed-tensor` relaxation for the MoE amax architectures, kept separate so it can be dropped
  - see [block 0300](patches/README.md#block-0300--sm-relaxed-tensor-for-the-moe-amax-arches-temporary-expect-removal)


Each block is one `git am` commit, and the series is linear (a block
assumes its predecessors), so the set is applied whole. 

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 919794500
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts (18 commits)

# OPT-IN, not part of the delivery: optional artifacts on top (NOT the release tree)
bash <path-to-this-repo>/scripts/apply-all.sh --include-optional .       # asks [y/N] per artifact
bash <path-to-this-repo>/scripts/apply-all.sh --include-optional-all .   # every artifact, no prompt

# OPT-IN, not part of the delivery: a beta candidate on top (NOT the release tree -- beta testing only)
bash <path-to-this-repo>/scripts/apply-all.sh --include-beta .   # 18 default blocks + beta/*/*.patch
```

- One-line summary of each block: [The blocks](#the-blocks)
- What each block contains, its knobs and its purity notes: [`patches/README.md`](patches/README.md)
- The folded `mmb`/QSA campaign: [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery)
- Every environment variable, with its default and class: [`ENVIRONMENT.md`](ENVIRONMENT.md) (the knobs)
  and [`DIAGNOSTIC.md`](DIAGNOSTIC.md) (the switches, A/B arms and diagnostics)
- **Running a model bigger than your VRAM on one card** (`-ncmoe` + the MoE expert cache, with the
  measured config tables and the measurement method): [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md) (today's entries at the top, the index of earlier days at
  the end)
- Current status and validation: [`release.json`](release.json) (source of truth) and [`MANIFESTS.md`](MANIFESTS.md)

### Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(the tag is the release identity: `v<blocks>-<fork-point>-r<N>`, e.g. **`v16-4f9dfa0e3-r295`**; the block
count in the tag is the **official** `00xx` count, so `v15-*` tags are the 15-block releases up to r177 and
`v16-*` the 16-official-block releases from r178 on.  The temporary `0300` and the optional `01xx` blocks are
never counted (`AGENTS.md`).  `r1` is the
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

> **Strix Point (`gfx1150`) is in the target set as of 2026-10-05.**
> 
> **RDNA2 (`gfx1030`) is best-effort: not a supported target, but no longer unvalidated.**
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
├── rdna-boosts-all.patch  # convenience: the entire 19-patch default net as ONE patch
├── patches/               # the delivery set: official 0000-0016 + temporary 0300 (one `git am` block each)
│   ├── README.md          # apply instructions + block-12 env knobs + server config
│   └── FILE-SET-RETIREMENTS.tsv # re-base decision rows: a deliberate retirement a block may lose (gate 38)
├── scripts/               # apply / regenerate / validate / publish tooling; index: scripts/README.md
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift; --include-beta)
│   ├── gate-prefill-logits.sh  # prefill-logit KLD release gate (issue #113)
│   ├── gate-per-block-build.sh # bisect only (r332): which block broke a build
│   ├── check-rung-file-sets.py # re-base file-set reconciliation (TODO #358): no block may lose a file,
│   │                      #   define, env var or option the patch it replaces carried
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
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
└── archive/               # the rest: archive/work/ (closed experiments + the archived wip/ trees) + archive/docs/ (history)
```

### The patch blocks: official `0000`-`0017` + temporary `0300` + optional `0100`/`0101`

`release.json` (the current release).  Each row is one `git am` commit, applied in filename order:
`0000`-`0017` then `0300`.  The optional `0100`/`0101` are **not** in this table and never in the default
apply -- see [`optional/README.md`](optional/README.md).


| patch | subject | concern |
|-------|---------|---------|
| `0000` | **foundation** — band-uniformity, Meta-wrapper, allocator fixes | old-00 + old-05 + old-07 + old-09a + the multi-ubatch logits readback + the `ggml-alloc` view release; + the meta hot-path dispatcher restoration (r86) |
| `0001` | **hybrid HIP all-reduce** (RDNA4-gated) | the old block 12; + no per-AR `ev.ker` and the opt-in P2P-large path (r52); + the large-tensor arm is chosen per model with **peer access never enabled** — copy engine for a dense model, RCCL's ring for a host-offloaded one (r165, `TODO #109`) |
| `0002` | **adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`) | old-01; + opt-in MTP prompt-tail ingest and the last-row prompt copy (r52); + the embeddings-layer-input preset API the draft setup calls (r94 block-boundary repair); + the draft context's KV cache type follows the main `-ctk` / `-ctv` unless `-ctkd` / `-ctvd` pins it (r125, issue #216) |
| `0003` | **chunked gated-delta-net prefill** (bf16/WMMA GDN) | old-02; + the per-sequence rollback tracker folds in every ubatch and `LLAMA_RS_ROLLBACK_DEBUG` (r91) |
| `0004` | **BF16 KV cache + native-BF16 flash attention** | old-03 |
| `0005` | **flash-attention band / native quantized K-V / derived kq mask** | old-04 FA + the old-15 FA work; + the 2-byte arm joins the 32-column band block and the head-512 MMA row is retuned (r92); + the q8_0 fetch-ahead band variant, default on (`GGML_HIP_FA_BAND_FETCH_AHEAD=0`), r98 PR #195; + one tile K/V type selection for the instantiation and the F16 scratch, and an RDNA2-gated checked tail, r99 PR #198; + the derived mask's input is reusable through a hybrid memory context, so a >8-token verify step stops rebuilding its graph (r164, PR #242); + every fused-op probe that cannot see its node in the probe graph asks the acting device instead and SAYS how it verified (r311) |; **+ r314 (PR #329): the band also takes 128-dim heads whose GQA ratio is not a multiple of 8** (5, 6, 7, 9..12 -- an odd ratio folds nothing outside it), **on where it pays**: a layer takes it only when `K->ne[1] * passes_saved >= 20000`, keyed on the padded **used** extent the op executes (`INVARIANTS.md` r315 corrects r314's "cache size" here; a sliding-window layer holds `n_swa + n_ubatch` cells for the whole run) and never on the query width -- `GGML_HIP_FA_BAND_WMMA_H128` (`auto`/`0`/`1`), `GGML_HIP_FA_BAND_H128_MIN_WORK`; **+ r315: the 256-dim head class gets the same gate** (`K->ne[1] * max(passes_saved, 1) >= 20000` -- band from 10000 cells for GQA 6, 20000 for GQA 8/12; a verify-width kernel, so the decode width decides), worth **+4.4 %** (27B d0), **+7.4 %** (35B-A3B d0), **+3.1 %** (Flash-Next d0) of tg64 and +2.0 % on a MoE MTP arm, at -1.1 % on one dense MTP arm, with the deep `draft-mtp n7` band kept where it is worth +6.4 % -- `GGML_HIP_FA_BAND_WMMA_H256` (`auto`/`0`/`1`), `GGML_HIP_FA_BAND_H256_MIN_WORK` |
| `0006` | **MMB quantized prefill GEMM** | old-08 MMB |
| `0007` | **k-quant boosts** (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) | old-10 + old-04 MMQ + old-15 Q1_0/grid/PR57; + the RDNA3_0 per-shape mmvq `nwarps` (r93) |
| `0008` | **fused kernels** (conv1d/RMS/top-k MoE/verify fusions + fused MoE gate+up+GLU) | old-08 fusions + old-13 fused MoE + old-15 PR63/114; + `topk_moe` sorted-key argmax (r52); + the RDNA3_0 per-shape mmvq `nwarps` (r93) |
| `0009` | **host-resident MoE** (iface, scheduler, Meta, movable-boundary slab, cache engine, prefill cache staging) | old-06 MoE + old-09b + old-11 + old-13 engine + old-15 hooks/arena; + the arena-host staging engine and its residency snapshot (r83); + the slab drop reference (r85); + the slab transient absorption common half and the derived draft reserve (r86); + the complete device-iface fills and the pre-loaded draft-model handoff (r94 block-boundary repair); + the batched f32 CPY run on RDNA3_0 (r95, PR #193); + the restored copy callback, so the upstream `--moe-cache-mib` engine works (r99, TODO #56); + the split-state walk's per-thread scratch arena, so a graph re-allocation stops allocating one heap node per visited tensor (r100, issue #184); + the `--fit` draft reserve charged only to the KV layers a drafter owns (r101), and a loaded-but-unused draft model is not loaded or given a context (r101); + the cache's traffic report divides a decode-band op census by the sizing-time table count (r144), and `GGML_CUDA_GRAPH_TRACE` reports how every `graph_compute` call was served (r144); + the decode band no longer reads the routing back to the host: a device-side slot remap (`GGML_MOE_DEVREMAP`, host policy one call late) and a `--host-experts pool` resident miss read in place (`MOE_HOST_POOL_COLD`) — +8.4 % TG pinned / +4.8 % pool, byte-identical text and MTP acceptance (r203, PR #266); + the remap is **tensor-split and covering-pool aware** (r221, TODO #269): the Meta backend forwards it per device (+16.1 % / +28.4 % decode at 2 / 3 devices) and a host pool that covers its tensor is a cold source (+9.9 % on the default `-lme auto` mirror) |; **+ r338: the opt-in `-of tp-ffn-even` even FFN split** — `-sm tensor` floors a 640-wide expert set's 2-device share to 256 (256/384); with the tag (or `LLAMA_TP_FFN_EVEN=1`) the granularity drops toward `lcm(blck_size,64)` and it splits **320/320** (**+7 % `draft-mtp` decode, +4 % prefill** there).  OFF by default because the granularity IS the reduction's grouping (2-device Flash-Next mean KLD 0.0140, below that point's own `-ub` reorder control of 0.0152 but 2.8x the prefill-logit limit), inert on every 3-GPU shape the delivery validates; **`-of`/`--optional-features`** is the new discoverable spelling for an opt-in switch that would otherwise be environment-only.  `ENVIRONMENT.md` §4.1, `GREEDY-PURITY.md` §104, `benchmarks/2026-11-21-ffn-even-split.md` |
| `0010` | **infrastructure** — slab, fusion dispatcher, async input ring, GDN/rope fusions | the translation units split out of `ggml-cuda.cu` (`try-fuse`, slab/arena/vmm, MoE cache hooks) + short-K MoE rows-per-block (r52); + the `stage_from_cache` hook (r83); + the slab narrow band (r85); + the slab transient absorption CUDA/slab half (r86); + the async graph-input ring and its r90 split-input snapshot fix (default-on); + the fixed-D sequential GDN snapshot tail (r91, `GGML_CUDA_GDN_TAIL=0`); + the work-boundary shrink invalidates captured graphs (r94, issue #187); + the verify-band Q8_1 fusions and the GDN conv verify path on RDNA3_0 (r95, PR #192 + #193); + gemma4 dense `-sm tensor` (the Meta split state accepts an empty-or-whole slice for per-row ops, the rotated-K `mul_mat` and flash-attention K/V), the expert cache sizing / multi-slice tables / segmented async upload, the iSWA `q, v, k` KV-builder order and the per-origin async-input ring (r101); + the GDN/attention-gate decode fusions (r102, PR #200) with the r108 multi-sequence fix, the bf16 RoPE destinations (r103), and the r183 GDN decode folds with the shared-expert gate fold (PR #256) plus the 2-D copy guard; + the `q4_0` K/V cache write fold in the rope + FWHT kernel (r199, PR #262, `GGML_CUDA_KV_FOLD_Q4_0=0`), +10.8 % tg128 on a `q4_0`-KV 30B-A3B; + the fusion-gate oracle's 33-case `{BF16,1024}`/`{Q8_0,1024}`/`{Q8_0,5120}` matrix (r202, tests only) |
| `0011` | **qwen4exp** (model, hyper-connections, lazy reader, QSA/indexer) | old-14 + old-15 HC16/policy/QSA; the model graph, the QSA/indexer/HC ops and kernels, the managed lazy reader and the keys-only indexer cache, the qwen4exp tensor-split arch gate, and the qwen4exp fusion arms; split out of `0010` in r109 (tree-identical)  **+ r303: both GGUF layouts** -- the post-#31028 merged `hc_*_down_inject` file (upstream's converter) now builds at all (it segfaulted at `sched_reserve`, contributor PR #321) and keeps the fused hyper-connection mixer, by feeding the op two row views of the merged tensor; the merged shape has a synthetic fixture (`scripts/make-merged-hc-gguf.py`) and a CI arm in `test-llama-archs`. |
| `0012` | **campaign remainder** (CPU wins, DFlash, FA build grouping, mmap reclaim, batched state copy; **+ r207: host-pool-speed** — a POOL-backed table gathers in the prefill band by default, the `RAM_PCT` ceiling scales every table, `MOE_HOST_POOL_SIZING=1` (opt-in), the pool's diagnostics report the run, `MOE_HOST_POOL_PREFETCH` removed) | old-15 W1-W3 + old-06/15 mmap reclaim + old-16; + the expected `--fit` `ctx_other` probe failure logs as a warning, and the draft-context creation is gated on an enabled type that uses it (r101); + the batched state save/restore, the pipelined copy over it, and the threaded streaming host copy (r152, PR #235); + the DFlash draft's compute reserve split over its injection and noise-block graphs (r162, issue #239); + the DFlash target writes its extracted layer inputs into F1's device store from inside the graph, so they stop being graph outputs and the host rows are reserved only when the host path runs (r163, issue #240) |; **+ r291: qwen4exp sequence parallelism** — the hyper-connection chain runs on each device's own token slice of a 2+-device `-sm tensor` prefill (`LLAMA_SEQPAR`, **default on**, `=0` off; `LLAMA_SEQPAR_MIN` 512, clamped to 128).  +7 % copy-engine / +2.2 % default-chain prefill at 31.5k, MTP prefill +15 %; a batch-width rounding change (`GREEDY-PURITY.md` §92), off-path bit-inert, gate [`scripts/gate-seqpar.sh`](scripts/gate-seqpar.sh); **+ r357: the server's tool-text protection** — a tool result's, a tool call's arguments' and an assistant text's special-token strings stay TEXT (two noncharacter marks before the template, then a tokenize pass that treats a marked string as ordinary text), so data from a tool can no longer forge a turn; default-on (`LLAMA_SERVER_TOOL_TEXT=0` restores the previous behaviour byte for byte, `LLAMA_SERVER_TOOL_TEXT_CENSUS=1` prints the census), server code only, gate [`scripts/gate-tool-text.sh`](scripts/gate-tool-text.sh), contributor PR #360
| `0013` | **MTP draft vocabulary subset** (`--spec-draft-mtp-vocab N`, **opt-in**, default off) | the MTP draft scores only the LM-head rows for token ids < N (+ control/user ids) and scatters them into a full-vocab row of `-inf` — the target's verification is untouched, so the draft can only lose proposals; the `ggml-backend-meta` row-range read that makes it work under `-sm tensor` (r107) |
| `0014` | **streamed KV** (**all three placements opt-in, default OFF since r247**: `--kv-window` / `--kv-attendance-pool` window and pool, and `--spec-draft-kv-window`, the MTP draft ring — see `ENVIRONMENT.md` §17 for why the draft ring's old default was a decode regression) | the KV's VRAM cost becomes a resident **window** of the newest `W` cells instead of `-c` cells: one VA reservation holds the cache, the window is VRAM and everything older is pinned system RAM, so a context that OOMed now runs (27B Q8_0 `-c 262144` = 128 MiB VRAM + 8576 MiB pinned host) and the VRAM the KV does not need *yet* goes to the expert-cache arena (+2.2…+2.9 GiB at `-c 262144`, `-n 2000`). A spilled KV is numerically transparent (prefill-logit KLD −0.000000 / 100.000 % on a spilling arm); unset = byte-identical to a build without it, including the OOM (r178, promoted from `archive/work/kv-residency`)  **The same block also carries the attendance pool** (r190, Goal B: `--kv-attendance-pool on\|off`, whose size is derived from the model since r247 — the KV's authoritative copy in a pinned host mirror plus a compact device pool of indexer-block pages the sparse decode resolves through, 204 MiB instead of 1.6-2.4 GiB of resident KV at 103k-180k, `--fit`-visible) and the streaming arm policy (a streamed cache takes the sparse decode arm, +34 %/+47 % on a spilled KV).  Both placements are **opt-in and default off**  + r204: the same **window across a `-sm tensor` split** (`TODO` #124): one VMM region per child device of the Meta wrapper, the grow fanned out to every child, `--fit` sizing it against the split's per-device budget (a MIRRORED dsv4/HRM cache still declines it); + r222: the provider binds the device at every driver-reaching entry point and a buffer type answers capability without touching the driver, so a **plain** 2-GPU `-sm tensor` run does no VMM driver call at all (issue #275: the construction-time probe ran cross-context, which crashed the first kernel launch on Windows); + r248: the MTP draft ring is a **true** ring, so a batch the position window cannot place (an M-RoPE image chunk) evicts the drafter's oldest cells instead of failing the request, and a drafter failure can no longer return a 500 (issue #297) |
| `0015` | **the UMA GTT expert store** (an APU's MoE expert set in one registered region; `-lme` sizes it and is the streaming opt-in) | new in r263: the arena tier per `-lme`, the MTP head pinned outside that budget, the direct fill, the RDNA3_5 transport start arm, and prefill streaming.  Inert on a discrete GPU (measured) |
| `0017` | **TQ1_0 / TQ2_0 (ternary) weights on the GPU** (**default-on**, `GGML_CUDA_TQ=0` refuses the types again) | a ternary model (`BitNet` b1.58 and its conversions) had no GPU path at all — `supports_op` refused the two types, so with `-ngl 99` the weights stayed in host memory and the CPU backend ran every matmul (~11 t/s decode on three R9700s).  This adds `MUL_MAT` (not `MUL_MAT_ID`), `GET_ROWS` and the fp16/fp32/bf16 dequantise rows, **bit-identical to the CPU backend's x86 AVX2 path** (the eight-lane integer dot, the Q8_K quantisation and `hsum_float_8`'s order, with an UNFUSED mul+add per block because ggml-cpu's is not contracted) — `test-backend-ops` with `LLAMA_TEST_OP_NMSE=1` reads exactly 0 on all 305 `MUL_MAT` and 84 `GET_ROWS` values.  Decode takes a vec kernel that quantises src1 in shared memory and can fuse Q/K/V, gate+up+SWIGLU or matmul+residual-ADD (`GGML_CUDA_TQ_FUSE=0` turns all of it off); prefill takes an int8-WMMA kernel on RDNA4 and a small-batch `dp4a` kernel elsewhere.  Measured (one R9700): TQ2_0 pp512 **4177 vs 258** and tg128 **181.5 vs 21.8** against the CPU-backend arm; TQ1_0 **3693 vs 99.5** / **198.2 vs 16.1**; every non-TQ model untouched (text and rate measured).  Contributor PR #358; gate `scripts/gate-tq.sh`; knobs `DIAGNOSTIC.md` §25 |
| `0300` | **`-sm relaxed-tensor`** (**TEMPORARY**, `flint`/`mellum`/`mistral4` only) | tensor split that also admits a replicated tensor whose values differ per GPU (the MoE amax row max, which no device can complete). Exact on RDNA/HIP, zero VRAM, and **not a supported feature**: it relaxes upstream #31081 and is expected to be dropped when upstream fixes the amax under a tensor split. Delete this patch to get back to the r136 tree |


### Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
FORK=https://github.com/ggml-org/llama.cpp
git clone $FORK && cd llama.cpp
git checkout "$BASE"

# 2. apply the set (automated; strict 18/18 git am on the recorded base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0016 then patches/0300  (one commit per block on a fresh `rdna-boosts` branch)

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
expected to stay fork-local. The internal all-reduce (**block 01** in V2.0 numbering) is gated to
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
