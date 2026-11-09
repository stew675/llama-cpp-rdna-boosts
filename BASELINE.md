# BASELINE — provenance and drift policy

**What this file answers:** what the patches are generated *against*, how they are cut, and what to do when
upstream moves.  It is not a changelog — the current release, tip/tree and per-artifact hashes are in
[`release.json`](release.json) (the single source of truth) and the dated record of every re-base is in
[`WORKLOG.md`](WORKLOG.md).  Dated baseline records from the pre-V2.0 era (the 16-block set, the
`9113cc188`..`ebbb18522` re-bases, the old `baseline/*` checkpoints) are preserved in
[`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).

## Current baseline

| item | value |
|---|---|
| fork point (upstream `ggml-org/llama.cpp` master) | **`release.json .base`** |
| delivery | 16 blocks, `block 00` + `blocks 01-15` (`release.json .n_blocks`); **block 14 is temporary**; block 15 is the promoted windowed KV |
| canonical chain | `release.json .tip` / `.tree` |
| release identity | `release.json .release` (`v<blocks>-<base>-r<N>`) |

**The base is fixed by a re-base.**  The last one is **`v16-a691b2345-r236`** (2026-11-08): it replayed the
set from `6050b467c` (the 2026-10-25 base) onto **`a691b2345`** — 318 upstream commits — with every block
re-derived rather than re-applied: the CPU backend was rewritten around block 12's hunks, upstream grew its
own `qwen4exp`, upstream's `tile_spec` replaced the ids grid block 00 rode on, and blocks 09/10/15 were
re-expressed on the new slab/pool/graph-input surfaces.  The ladder, the per-rung conflict map and the
decisions it forced (D1, D2, D3, D8, D12, D13, D15) are in `archive/work/upstream-rebase-261108/` until the campaign
is archived, and the narrative is `WORKLOG.md` 261108.001-.015.  The previous re-base was
`v15-6050b467c-r138`: it replayed the set from
`3f03b1772` (2026-10-24) onto **`6050b467c`** (2026-10-25) — 21 upstream commits, 98 files.  Small but
semantic: it carries a **fused FFN gate/up+GLU MMQ kernel (#29948, NVIDIA-only)** that duplicates block 08's
RDNA fused gate path (both now ship, arch-disjoint, on one dispatch) and a **merged-tensor qwen4exp HC
layout (#31028)** that subsumes block 11's two-GEMM down/inject pair.  The previous re-base was
`v14-3f03b1772-r127`: `097cf33d9` (2026-10-12) onto `3f03b1772` (2026-10-24), 261 upstream commits,
including a chunked GDN prefill kernel, a MoE cache rewritten to SLRU, chunked CUDA graphs, a 4 GiB
pinned-host cap, a fused QSA mask op and reworked MMQ/MMVQ tuning.  Between re-bases every `rN` is a
**block amendment** at the same base
(the release `v14-3f03b1772-r136` was one: idle-slot parking in the server KV cache, which
changes no kernel), so the block
series and its recorded trees stay comparable.  A re-base is a deliberate event with its own `WORKLOG.md`
entry and a record in [`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).

## What the patches are

Each `patches/000N-*.patch` is **one block commit** exported with `git format-patch`.  The blocks are the
delivery's unit of review, bisection and amendment; they are not a file split, and most blocks touch
several subsystems.  Block 00 is the foundation every later block assumes (band uniformity, the Meta
wrapper, allocator/graph-layout fixes).

| block | subject | concern |
|---|---|---|
| 00 | foundation: band uniformity, Meta wrapper, allocator | the decode/verify band rule, `ggml-backend-meta` split state, `ggml-alloc` view release, the multi-ubatch logits readback |
| 01 | hybrid HIP all-reduce | the internal AR for small tensors + per-size hybrid dispatch vs RCCL |
| 02 | adaptive MTP draft depth | `--spec-type draft-mtp-adaptive` and its controller |
| 03 | chunked gated-delta-net prefill | bf16/WMMA chunked GDN (two arch-segregated kernels) + the recurrent rollback tracker; **the only chunked GDN kernel the delivery dispatches** — upstream's RDNA3.5 MMA kernel stays reachable only where this path stands down (`kda`, `S_v` outside {16,32,64,128}, or `GGML_CUDA_GDN_CHUNKED=0` for A/B) |
| 04 | BF16 KV cache + native-BF16 flash attention | the bf16 cache type and its native FA arm |
| 05 | flash-attention band / native quantized K-V / derived kq mask | the RDNA FA band kernels, native KV staging, the derived mask (and its reuse through a hybrid memory context, r164) |
| 06 | MMB quantized prefill GEMM | bf16-WMMA dequant-weight GEMM and its tuning |
| 07 | k-quant boosts | `mmvq` VDR, Q6_K MMQ, Q1_0, grid pad — **the decode-numerics block** |
| 08 | fused kernels | conv1d/RMS/top-k MoE/verify fusions, fused MoE gate+up+GLU, per-shape `mmvq` `nwarps` |
| 09 | host-resident MoE | iface, scheduler, Meta, movable-boundary slab, expert cache, prefill cache staging |
| 10 | infrastructure: slab, fusion dispatcher, async input ring | the TUs split out of `ggml-cuda.cu` (slab/arena/vmm, `try-fuse`, MoE cache hooks), the async graph-input ring, the GDN/rope/MMB fusions |
| 11 | qwen4exp | model, hyper-connections, lazy reader, QSA/indexer, the keys-only indexer cache (r109 split of the old block 10) |
| 12 | campaign remainder | CPU wins, backend iface fills, DFlash (incl. the draft-reserve split, r162/issue #239, and the in-graph layer-input store, r163/issue #240), FA build grouping, mmap reclaim |
| 13 | MTP draft vocabulary subset | `--spec-draft-mtp-vocab N` (opt-in, default off) |

Per-block *content* and edit history is [`patches/README.md`](patches/README.md); per-block verification is
[`MANIFESTS.md`](MANIFESTS.md).  (`patches/README.md`'s dated notes below its header use the **pre-V2.0
16-block** numbering and are kept as provenance; r109's split kept the infrastructure at block 10 and moved
qwen4exp to block 11.)

## How the patches are cut

The **canonical chain is the `rdna-boosts` branch of the maintainer's personal fork**, rebuilt from
`release.json.base` with [`scripts/apply-all.sh`](scripts/apply-all.sh) and pushed on every release.  The
working `~/llama.cpp` checkout is *not* the canonical chain: it drifts past the base, so a raw
`git format-patch <base>..<tip>` there can export upstream commits as delivery blocks.

To regenerate: apply the set to a clean checkout at `release.json.base`, make the change as a commit on
the owning block (amend, do not append), then run `scripts/make-patches.sh` (defaults from
`release.json`) and `scripts/make-release.sh` (rewrites `release.json` and `rdna-boosts-all.patch`).  Never
hand-edit a committed patch as the fix — regenerate and re-verify.  See AGENTS.md → "Regenerate the
patches".

## Provenance of the two fixes that are not upstream's

Two changes carried since the fork lineage are worth knowing about when a hunk conflicts, because they are
not upstream code and have no upstream counterpart to merge:

1. **Test-harness seeding** (block `0003`) — the fork's chunked-GDN work had replaced
   `test-backend-ops`' `std::random_device` seeding with a fixed seed, which deterministically exposes a
   pre-existing numerical fragility in `rms_norm_back` / `cross_entropy_loss_back` on RDNA4.  The delivery
   keeps `random_device` seeding; the GDN kernels are unaffected either way.
2. **Meta compute-container headroom** (block `0000`/`0009`) — hybrid recurrent models create
   `~2*(n_rs_seq+1)` conv-state snapshot views per recurrent layer during graph allocation, which
   overflowed the original 16x container.  The headroom is now supplied by the compute-margin path
   (`GGML_COMPUTE_BUFFER_MARGIN_PCT`), see AGENTS.md → critical facts.

The full diagnostics for both are in [`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).

## Drift policy

The patches are static against the fork point.  When upstream master moves:

1. `scripts/apply-all.sh` is already the tested flow: strict `git am`, with a `git am -3` fallback on
   drift (3-way merge against the baseline blobs).
2. If the 3-way merge fails — the normal case in a fresh clone, where the pre-image blobs are absent —
   re-base the failing hunks against current master by hand, in the block that owns them.
3. **Do not hand-edit the committed patches as the permanent fix.**  When more than one block needs manual
   hunks, re-base the block commits on a clean checkout at the new base, regenerate the whole set
   (`scripts/make-patches.sh`), refresh `release.json` (`scripts/make-release.sh --base … --tip …
   --tree …`), and update the current-state headers (`README.md`, `MANIFESTS.md`, `patches/README.md`,
   `ENVIRONMENT.md` where a default moved).
4. Re-verify: `scripts/validate-set.sh` (strict apply on a fresh tarball, tree equality), a full build,
   the coherence gate, and the release gates in `MANIFESTS.md`.  Record the re-base in `WORKLOG.md` with
   the new base, tip and tree.

A patch that resolves *naturally* because upstream fixed the same thing locally is a re-base conflict, not
a bug: keep upstream's fix, drop the local hunk, and note it in the `WORKLOG.md` entry.
