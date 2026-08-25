# llama-cpp-rdna-boosts

Squashed, standalone diff blocks of RDNA-specific performance and correctness
work from the `chunked-gdn` branch of the
[llama.cpp fork](https://github.com/stew675/llama.cpp), packaged for easy
application to mainline llama.cpp.

The branch is 48 commits of RDNA work on top of upstream master
(`758443071`). Those commits decompose into **10 functional blocks**: 9 clean
standalone diffs and 1 "fused core" (16 mutually-entangled commits extracted as
one combined diff, applied last).

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── MANIFESTS.md           # THE CONTRACT: apply order, deps, verify, failure handling
├── BASELINE.md            # pinned baseline SHA, per-patch provenance, drift policy
├── patches/
│   ├── 01-adaptive-mtp.patch          … 10-q6k-mmvq-vdr2.patch
│   └── rdna-boosts-all.patch          # convenience: the entire 48-commit net, one patch
└── scripts/
    ├── make-patches.sh    # regenerates all patches from the fork (needs the fork checkout)
    └── apply-all.sh       # applies in manifest order + runs the verification commands
```

## The 10 blocks

| patch | what | size |
|-------|------|------|
| `01-adaptive-mtp.patch` | adaptive speculative draft depth for MTP (`--draft-mtp-adaptive`) | 5 commits |
| `02-chunked-gdn.patch` | fused chunked gated-delta-net prefill kernel (fp32 + bf16/WMMA tensor-core as S_v=128 default) | 13 commits |
| `03-bf16-kv-cache.patch` | native-BF16 flash-attn tile kernel, BF16 KV cache, IMRoPE + set-rows fusion | 8 commits |
| `04-wmma-flash-attn.patch` | RDNA4 WMMA flash-attention path; Q6_K mmq prefill tuning | 1 commit |
| `05-bit-identical-decode-cpu.patch` | bit-identical CPU decode / speculative-verify batches | 1 commit |
| `06-fused-core.patch` | fused quantized-matmul kernels + graph fusions (SSM/MoE), GPU bit-identical decode; **apply last** | 16 commits, 1 diff |
| `07-gfx1151-mmvq-table.patch` | RDNA3_5 mmvq parameter table + nwarps=2 Q8_0 decode | 1 commit |
| `08-host-buffer-revert.patch` | back out integrated-GPU host buffers on HIP (PR #24233) | 1 commit |
| `09-meta-device-wrapper-skip.patch` | skip the Meta device wrapper with a single GPU | 1 commit |
| `10-q6k-mmvq-vdr2.patch` | Q6_K mmvq VDR=2 decode kernel | 1 commit |

## Consumer workflow

```
# 0. pick the baseline branch matching YOUR upstream version.
#    (branches are named baseline/<sha>; MANIFESTS.md records the upstream
#    range each was generated against - choose the closest one)
git clone https://github.com/stew675/llama-cpp-rdna-boosts ../rdna-boosts
cd ../rdna-boosts && git checkout baseline/758443071

# 1. fresh clone of upstream llama.cpp (or an existing one), at the matched range
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 758443071   # the SHA recorded in MANIFESTS.md

# 2. fresh branch
git checkout -b rdna-boosts

# 3. apply in manifest order, verifying each (block 06 LAST)
git apply ../rdna-boosts/patches/01-adaptive-mtp.patch
... # blocks 02-05, 07-10 in any order
git apply ../rdna-boosts/patches/06-fused-core.patch

# 4. full verification
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ...
cmake --build build -j
./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET   # GDN block: expect 46/46
... # per-block checks from MANIFESTS.md
```

`scripts/apply-all.sh` automates step 3.

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

These blocks are candidates for upstream PRs to ggml-org/llama.cpp. The
branch-per-upstream-range versioning keeps every patch set reproducible against
a pinned upstream tip. See `MANIFESTS.md` for per-block verification and
`BASELINE.md` for provenance.

## License

Same as llama.cpp (MIT).
