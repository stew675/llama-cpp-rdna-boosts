# llama-cpp-rdna-boosts

A set of sectioned patches that may be applied to a fresh llama.cpp master
branch, offering a variety of feature enhancements and ROCm-targeted
performance fixes. When the `scripts/apply-all.sh` script is run, a new
`rdna-boosts` branch is automatically created with all the patches applied in
order.

## Branches (one per verified upstream range)

| branch | upstream range | status |
|--------|----------------|--------|
| `baseline/d222767c7` | `758443071` .. `d222767c7` | **current, recommended** - full suite validated 14883/14883 on ROCm 7.14/gfx1201, includes the test-harness seeding fix |
| `baseline/758443071` | `758443071` (single point) | original set mirroring the fork; known-good for the old upstream, carries the fork's deterministic test-harness seed (see BASELINE.md) |

Pick the branch whose recorded range matches your upstream tip; the
workflow below uses the current branch.

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── MANIFESTS.md           # THE CONTRACT: apply order, deps, verify, failure handling
├── BASELINE.md            # pinned baseline SHA, per-patch provenance, drift policy
├── rdna-boosts-all.patch      # convenience: the entire net as ONE patch
├── patches/
│   └── 01-adaptive-mtp.patch          … 10-fused-core.patch, 11-meta-headroom.patch
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
| `09-q6k-mmvq-vdr2.patch` | Q6_K mmvq VDR=2 decode kernel | 1 commit |
| `10-fused-core.patch` | fused quantized-matmul kernels + graph fusions (SSM/MoE), GPU bit-identical decode | 16 commits, 1 diff |
| `11-meta-headroom.patch` | meta-buffer compute-container headroom | 1 commit |

Blocks are listed in apply order; `10-fused-core` needs blocks 03+04 in the
tree (see MANIFESTS.md), `11` is a follow-up fix applied last.

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

# 3. apply in manifest order, verifying each (block 10 LAST before block 11)
git apply ../rdna-boosts/patches/01-adaptive-mtp.patch
... # blocks 02-05, 07-10 in any order
git apply ../rdna-boosts/patches/10-fused-core.patch

# 4. full verification
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ...
cmake --build build -j
./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET   # GDN block: expect 46/46
... # per-block checks from MANIFESTS.md
```

`scripts/apply-all.sh` automates step 3.

### Git-native alternative: block tags

The repo also carries `block/01-… block/11-…` tags - one squashed commit per
block, stacked in apply order. Block numbers are the apply order everywhere
(patch filenames, tag names, block labels in the git history): `01` applies
first, `11` last. The fused core needs blocks 03+04 in the tree, so it sits
at position 10. Each tag's diff-vs-parent is exactly that block:

```
git remote add rdna-boosts git@github.com:stew675/llama-cpp-rdna-boosts.git
git fetch rdna-boosts --tags
git cherry-pick block/01-adaptive-mtp   # ... then block/02 … block/10, in order
git cherry-pick block/10-fused-core     # last
git cherry-pick block/11-meta-headroom
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
fork) and cuts a new `baseline/<new-sha>` branch here. Old branches stay as the
known-good sets for older upstream versions.

## Upstreaming

Some of these blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. The branch-per-upstream-range versioning keeps
every patch set reproducible against a pinned upstream tip. See
`MANIFESTS.md` for per-block verification and `BASELINE.md` for provenance.

## License

Same as llama.cpp (MIT).
