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
| `main` | points at the current checkpoint | currently `baseline/192067b72` |
| `baseline/192067b72` | `d222767c7` .. `192067b72` | **current, recommended** - same patch files as `d222767c7`, zero-fuzz apply to master at `192067b72`, fresh build + real-model sanity validated 2026-08-26 (ROCm 7.14/gfx1201) |
| `baseline/d222767c7` | `758443071` .. `d222767c7` | 12-block set, full suite validated 14883/14883 on ROCm 7.14/gfx1201, includes the test-harness seeding fix |
| `baseline/758443071` | `758443071` (single point) | original set mirroring the fork; known-good for the old upstream, carries the fork's deterministic test-harness seed (see BASELINE.md) |

Pick the branch whose recorded range matches your upstream tip (`main` is
fine when your master is at or near the current checkpoint); the workflow
below uses the current branch.

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── MANIFESTS.md           # THE CONTRACT: apply order, deps, verify, failure handling
├── BASELINE.md            # pinned baseline SHA, per-patch provenance, drift policy
├── rdna-boosts-all.patch      # convenience: the entire net as ONE patch
├── patches/
│   └── 01-adaptive-mtp.patch          … 11-meta-headroom.patch, 12-k-quant-boosts.patch
└── scripts/
    ├── make-patches.sh    # regenerates all patches from the fork (needs the fork checkout)
    └── apply-all.sh       # applies in manifest order + runs the verification commands
```

## The 11 blocks

| patch | what | size |
|-------|------|------|
| `01-adaptive-mtp.patch` | adaptive speculative draft depth for MTP (`--draft-mtp-adaptive`) | 5 commits |
| `02-chunked-gdn.patch` | fused chunked gated-delta-net prefill kernel (fp32 + bf16/WMMA tensor-core as S_v=128 default) | 13 commits |
| `03-bf16-kv-cache.patch` | native-BF16 flash-attn tile kernel, BF16 KV cache, IMRoPE + set-rows fusion | 8 commits |
| `04-wmma-flash-attn.patch` | RDNA4 WMMA flash-attention path; Q6_K mmq prefill tuning | 1 commit |
| `05-bit-identical-decode-cpu.patch` | bit-identical CPU decode / speculative-verify batches | 1 commit |
| `06-gfx1151-mmvq-table.patch` | RDNA3_5 mmvq parameter table + nwarps=2 Q8_0 decode | 1 commit |
| `07-host-buffer-revert.patch` | back out integrated-GPU host buffers on HIP (PR #24233) | 1 commit |
| `08-meta-device-wrapper-skip.patch` | skip the Meta device wrapper with a single GPU | 1 commit |
| `10-fused-core.patch` | fused quantized-matmul kernels + graph fusions (SSM/MoE), GPU bit-identical decode | 16 commits, 1 diff |
| `11-meta-headroom.patch` | meta-buffer compute-container headroom | 1 commit |
| `12-k-quant-boosts.patch` | k-quant umbrella: Q6_K VDR=2 (ex-block 09) + Q4_K/Q5_K/Q8_0 VDR=4 mmvq decode, mmq scale-load hoist, RDNA4 MoE mmid whitelist | combined |

Blocks are listed in apply order; `10-fused-core` needs blocks 03+04 in the
tree (see MANIFESTS.md), `11` is a follow-up fix applied last, `12` is the
k-quant umbrella (all k-quant VDR/decode work, present and future, lives
here). **Block 09 was retired**: its Q6_K VDR=2 decode work is folded into
block 12, so the set is 11 patches numbered 01-08, 10, 11, 12.

> **Greedy-purity note (read before shipping):** block 12 is the only patch
> that changes decode numerics - its VDR kernels reorder the fp32 reduction,
> so compute outputs are not bit-identical to a build without it (max logit
> diff 0.184 vs 0.203 for flash-attn on/off; greedy streams are deterministic
> within a build but can flip across configs). PPL is unaffected (prefill
> path untouched). If you require 100% greedy purity across builds, do not
> install `12-k-quant-boosts.patch` - it is the last patch, so excluding it
> is a one-line change to `scripts/apply-all.sh`.

## Consumer workflow

```
# 0. pick the baseline branch matching YOUR upstream version.
#    (branches are named baseline/<sha>; MANIFESTS.md records the upstream
#    range each was generated against - choose the closest one)
git clone https://github.com/stew675/llama-cpp-rdna-boosts ../rdna-boosts
cd ../rdna-boosts && git checkout baseline/d222767c7

# 1. fresh clone of upstream llama.cpp (or an existing one), at the matched range
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout d222767c7   # the SHA recorded in MANIFESTS.md

# 2. fresh branch
git checkout -b rdna-boosts

# 3. apply in manifest order, verifying each (block 10 LAST before blocks 11/12)
git apply ../rdna-boosts/patches/01-adaptive-mtp.patch
... # blocks 02-08, then 10, then 11
git apply ../rdna-boosts/patches/10-fused-core.patch
git apply ../rdna-boosts/patches/11-meta-headroom.patch
git apply ../rdna-boosts/patches/12-k-quant-boosts.patch   # omit for greedy purity

# 4. full verification
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ...
cmake --build build -j
./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET   # GDN block: expect 46/46
... # per-block checks from MANIFESTS.md
```

`scripts/apply-all.sh` automates step 3.

### Git-native alternative: block tags

The repo also carries `block/01-… block/08-…`, `block/10-… block/12-…` tags
(block 09 retired) - one squashed commit per block, stacked in apply order.
Block numbers are the apply order everywhere (patch filenames, tag names,
block labels in the git history): `01` applies first, `12` last. The fused
core needs blocks 03+04 in the tree, so it sits at position 10. Each tag's
diff-vs-parent is exactly that block:

```
git remote add rdna-boosts git@github.com:stew675/llama-cpp-rdna-boosts.git
git fetch rdna-boosts --tags
git cherry-pick block/01-adaptive-mtp   # ... then block/02 … block/10, in order
git cherry-pick block/10-fused-core     # last
... # blocks 11-12 after the fused core
git cherry-pick block/11-meta-headroom
git cherry-pick block/12-k-quant-boosts
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
