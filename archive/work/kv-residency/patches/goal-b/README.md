# `patches/goal-b` — the Goal B series (`git am` reproduction of the campaign branch)

**STATUS 2026-10-31: PROMOTED.**  Goal B is delivered — the pool and B2 are **block 15** of
`v16-6050b467c-r191` (r190 shipped the pool as its own block 16; r191 folded it into block 15) and A13 is
block 11 (A12 shipped separately as r189).  **The delivery is
now the reference**: `patches/README.md` block 15 holds the block's own notes, gates and numbers, and the
`--fit` accounting this series does not carry (session A14) is in that commit too.  This directory is kept only
so a later session can reproduce the exact tree the campaign's Goal B **numbers** were measured on, without the
fork branch existing; it is **not** delivery work and nothing here is applied by `apply-all.sh`.

## How to reproduce the tree

The series applies to the **r188** release tip `0e42c6bc2` (`v16-6050b467c-r188`).

```bash
cd ~/llama.cpp && git checkout -b wip-kv-residency-b 0e42c6bc2
git am ~/llama-cpp-rdna-boosts/archive/work/kv-residency/patches/goal-b/*.patch     # 0001..0007, in order
~/bin/build-llama-rocm
```

**Verified 2026-10-31 (A14):** `git am` 7/7 clean on `0e42c6bc2`, applied tree
**`4a049e8a9aae23dacc2d9d87ed5b3b271b650202`** (= the campaign branch's tree `266882ee7`), build OK.

`0001` (B1's per-cache window override) and `0003` (B1(c)'s attended-set census) are **campaign diagnostics**
and are deliberately **not** in the delivery; `0006`/`0007` shipped as block 10 (r189) and block 11 (r190), and
`0002`/`0004`/`0005` as block 15.

## The series

<!-- regenerate with: for f in *.patch; do git -C <a checkout at the base> apply --numstat "$f" | …; done -->

```
0001  B1   per-cache window override (DIAGNOSTIC, campaign only)
0002  B2   a STREAMING KV takes the sparse QSA decode arm (+34 / +47 %)
0003  B1c  attended-set census (top-k trace + replayer)
0004  B-A  the attendance pool for the attention KV
0005  B-A  the pool mirror is host memory — accept it in the async state copies
0006  A12  a vectorised fallback for the 2-D copy the r183 guard refuses
0007  A13  the fused indexer scorer accepts QUANTIZED keys
```

`diffstat.txt` and `SHA256SUMS` are generated.  The campaign's design, gates and every number:
[`../../BA-POOL-DESIGN.md`](../../BA-POOL-DESIGN.md) §7-§10; the open work that remains:
[`../../COLD-START-PROMPT.md`](../../COLD-START-PROMPT.md).
