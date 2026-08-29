# llama-cpp-rdna-boosts

A set of sectioned patches that may be applied to a fresh llama.cpp master
branch, offering a variety of feature enhancements and ROCm-targeted
performance fixes. When the `scripts/apply-all.sh` script is run, a new
`rdna-boosts` branch is automatically created with all the patches applied in
order.

## Branches (one per verified upstream range)

`main` is a floating pointer: it always points at the CURRENT checkpoint
branch, so a plain clone (default branch) gets the recommended set. The
`baseline/<sha>` branches are pinned checkpoints - one per verified upstream
range. To cut a new checkpoint, create `baseline/<new-sha>` and re-point
`main` at it (`git branch -f main baseline/<new-sha> && git push origin
main --force`).

| branch | upstream range | status |
|--------|----------------|--------|
| `main` | points at the current checkpoint | currently `baseline/fe235f434` |
| `baseline/fe235f434` | `192067b72` .. `fe235f434` | **current, recommended** - block 01 re-based for the `common_speculative_impl` n_max drift (fork `adaptive-mtp` branch); block 06 folded into block 10 (mmvq umbrella, 2026-08-28), numbering compacted to 01-10 with block 11 (CUDA prefill-graph skip) added on top, set is 01-11; zero-fuzz apply to master at `fe235f434` (applied tree byte-identical to the pre-fold validated tree) |
| `baseline/192067b72` | `d222767c7` .. `192067b72` | same patch files as `d222767c7`, zero-fuzz apply to master at `192067b72`, fresh build + real-model sanity validated 2026-08-26 (ROCm 7.14/gfx1201); superseded by the drift-fix cut above |
| `baseline/d222767c7` | `758443071` .. `d222767c7` | 12-block set, full suite validated 14883/14883 on ROCm 7.14/gfx1201, includes the test-harness seeding fix |
| `baseline/758443071` | `758443071` (single point) | original set mirroring the fork; known-good for the old upstream, carries the fork's deterministic test-harness seed (see BASELINE.md) |

Pick the branch whose recorded range matches your upstream tip (`main` is
fine when your master is at or near the current checkpoint); the workflow
below uses the current branch.

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── GREEDY-PURITY.md       # block 10 decode variance: reduction mechanics, correctness, numbers
├── MANIFESTS.md           # THE CONTRACT: apply order, deps, verify, failure handling
├── BASELINE.md            # pinned baseline SHA, per-patch provenance, drift policy
├── rdna-boosts-all.patch      # convenience: the entire net as ONE patch
├── patches/
│   └── 01-adaptive-mtp.patch          … 10-k-quant-boosts.patch, 11-cuda-prefill-graph-skip.patch
└── scripts/
    ├── make-patches.sh    # regenerates all patches from the fork (needs the fork checkout)
    └── apply-all.sh       # applies in manifest order + runs the verification commands
```

## The 11 blocks

| patch | what | size |
|-------|------|------|
| `01-adaptive-mtp.patch` | adaptive speculative draft depth for MTP (`--draft-mtp-adaptive`) | 5 commits |
| `02-chunked-gdn.patch` | fused chunked gated-delta-net prefill kernel (fp32 + bf16/WMMA tensor-core as S_v=128 default on RDNA4 gfx12 AND RDNA3/3.5 gfx11; the two arch kernels are fully SEGREGATED - gfx12 is the fork's validated RDNA4-only file restored verbatim, gfx11 is its own first-gen WMMA port file, no shared code; runtime-cc dispatch - fixes issue #1 build error, the host-pass gate regression, and the gfx12 NaN/inf regression from the in-place dual-arch refactor; launch-rejection fallback + kkt barrier - fixes issue #2) | 13 commits + 4 consumer-side fixes |
| `03-bf16-kv-cache.patch` | native-BF16 flash-attn tile kernel, BF16 KV cache, IMRoPE + set-rows fusion | 8 commits |
| `04-wmma-flash-attn.patch` | RDNA4 + RDNA3_0 WMMA flash-attention path (heads to 576, `GGML_CUDA_FA_WMMA_MAX_HEAD` override); Q6_K mmq prefill tuning | 1 commit |
| `05-bit-identical-decode-cpu.patch` | bit-identical CPU decode / speculative-verify batches | 1 commit |
| `06-host-buffer-revert.patch` | back out integrated-GPU host buffers on HIP (PR #24233) | 1 commit |
| `07-meta-device-wrapper-skip.patch` | skip the Meta device wrapper with a single GPU | 1 commit |
| `08-fused-core.patch` | fused quantized-matmul kernels + graph fusions (SSM/MoE), GPU bit-identical decode, RDNA3_0 mmvq Q6_K nwarps 2->8 | 16 commits, 1 diff |
| `09-meta-headroom.patch` | meta-buffer compute-container headroom | 1 commit |
| `10-k-quant-boosts.patch` | k-quant + mmvq-parameter umbrella: Q6_K VDR=2 (ex-block 09) + Q4_K/Q5_K/Q8_0 VDR=4 mmvq decode (RDNA4 + RDNA3_0), RDNA3_5 gfx1151 param table (ex-block 06), mmq scale-load hoist, RDNA4 MoE mmid whitelist | combined |
| `11-cuda-prefill-graph-skip.patch` | skip CUDA graphs for multi-token PRE-FILL graphs (varying ubatch makes capture pure overhead); decode keeps graph replay | 1 commit |

Blocks are listed in apply order; `08-fused-core` needs blocks 03+04 in the
tree (see MANIFESTS.md), `09` is an independent follow-up fix, `10` is the
k-quant + mmvq-parameter umbrella (all k-quant VDR/decode work and all mmvq
parameter-table tuning, present and future, lives here) and `11` is an
independent CUDA prefill-graph skip applied last. **Blocks 09 and 06
(old numbering) were retired**: block 09's Q6_K VDR=2 decode work and block
06's gfx1151 (RDNA3_5) mmvq parameter table are folded into block 10, so the
set is 11 patches numbered 01-11 (the 01-10 compaction, then block 11 added
on top).

> **Greedy-purity note (read before shipping):** block 10 is the only patch
> that changes decode numerics on ANY architecture - its VDR kernels reorder
> the fp32 reduction, and its RDNA3_5 nwarps=2 table (ex-block 06) changes
> the reduction on gfx1151 too - so compute outputs are not bit-identical
> to a build without it (max logit diff 0.184 vs 0.203 for flash-attn
> on/off; greedy streams are deterministic within a build but can flip
> across configs). This is a different rounding path, not a correctness
> change: block 10 computes the same real-number result as stock through a
> different order of fp32 additions - stock is the reference standard, not
> the mathematically special order (PPL is bit-unchanged: 6.3162/6.3563).
> If you require 100% greedy purity across builds, do not install
> `10-k-quant-boosts.patch` - it is the last patch that changes numerics
> (block 11 is a pure graph-dispatch change), so excluding it is a
> one-line change to `scripts/apply-all.sh` (and it restores purity on
> RDNA3, RDNA3_5 and RDNA4 alike). **Full discussion of the variance, the
> reduction mechanics, and what "correctness" means here:
> [`GREEDY-PURITY.md`](GREEDY-PURITY.md).**

## Consumer workflow

```
# 0. pick the baseline branch matching YOUR upstream version.
#    (branches are named baseline/<sha>; MANIFESTS.md records the upstream
#    range each was generated against - choose the closest one)
git clone https://github.com/stew675/llama-cpp-rdna-boosts ../rdna-boosts
cd ../rdna-boosts && git checkout baseline/fe235f434

# 1. fresh clone of upstream llama.cpp (or an existing one), at the matched range
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout fe235f434   # the SHA recorded in MANIFESTS.md

# 2. fresh branch
git checkout -b rdna-boosts

# 3. apply in manifest order, verifying each (block 08 after blocks 03+04; then 09, 10, 11)
git apply ../rdna-boosts/patches/01-adaptive-mtp.patch
... # blocks 02-05, 06, 07, then 08, then 09
git apply ../rdna-boosts/patches/08-fused-core.patch
git apply ../rdna-boosts/patches/09-meta-headroom.patch
git apply ../rdna-boosts/patches/10-k-quant-boosts.patch   # omit for greedy purity
git apply ../rdna-boosts/patches/11-cuda-prefill-graph-skip.patch

# 4. full verification
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ...
cmake --build build -j
./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET   # GDN block: expect 46/46
... # per-block checks from MANIFESTS.md
```

`scripts/apply-all.sh` automates step 3.

### Git-native alternative: block tags

The repo also carries `block/01-… block/11-…` tags - one squashed
commit per block, stacked in apply order.
Block numbers are the apply order everywhere (patch filenames, tag names,
block labels in the git history): `01` applies first, `11` last. The fused
core needs blocks 03+04 in the tree, so it sits at position 08. Each tag's
diff-vs-parent is exactly that block:

```
git remote add rdna-boosts git@github.com:stew675/llama-cpp-rdna-boosts.git
git fetch rdna-boosts --tags
git cherry-pick block/01-adaptive-mtp   # ... then block/02 … block/07, in order
git cherry-pick block/08-fused-core     # after blocks 03+04
git cherry-pick block/09-meta-headroom  # ... then block/10, block/11, in order
git cherry-pick block/10-k-quant-boosts
git cherry-pick block/11-cuda-prefill-graph-skip
```

Cherry-picking uses 3-way merge, so blocks degrade more gracefully than
`git apply` when upstream drifts. The tag->patch mapping is in
`MANIFESTS.md`; the tags are regenerated deterministically by
`scripts/make-patches.sh`.

## When upstream master moves

```
# is there a newer baseline branch matching your new master?
git fetch origin && git branch -r | grep baseline

# a) yes -> switch the whole patch set
git checkout baseline/<new-sha>
cd ../llama.cpp && git checkout origin/master && git branch -D rdna-boosts
# then redo steps 2-4 above

# b) no -> re-apply the current set with drift fixes
git checkout master && git pull
git branch -D rdna-boosts && git checkout -b rdna-boosts origin/master
git apply patches/01-adaptive-mtp.patch   # re-apply; fix drift with git apply -3 / manual rebase
...
```

When more than one block needs manual re-base hunks, the maintainer regenerates
the set against the new upstream tip (`scripts/make-patches.sh`, run in the
fork), cuts a new `baseline/<new-sha>` branch here, and re-points `main` at
it so the default branch follows:

```
git checkout baseline/<new-sha>
# ... validate (apply-all.sh + build + suite), commit the doc updates ...
git branch -f main baseline/<new-sha>
git push origin main --force   # main is a pointer; force is intended
```

Old branches stay as the known-good sets for older upstream versions.

## Upstreaming

Some of these blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. The branch-per-upstream-range versioning keeps
every patch set reproducible against a pinned upstream tip. See
`MANIFESTS.md` for per-block verification and `BASELINE.md` for provenance.

## License

Same as llama.cpp (MIT).
