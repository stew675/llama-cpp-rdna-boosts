# MANIFESTS - apply order and verification contract

Squashed, standalone diff blocks of RDNA-specific performance and correctness
work from the [llama.cpp fork](https://github.com/stew675/llama.cpp)
(`rdna-boosts` branch), packaged for easy application to mainline llama.cpp.

The **current delivery** is a **12-patch set** against the fork point
`17252c769`: blocks 01-11 (`patches/0001-…0011-…`, format-patch of the
fork's `rdna-boosts` block commits — the current whitespace-clean
regeneration `3209e83b4..cc985ba9a`; the original fork hashes
`2b7a135cb..f6f8f6778` are preserved on the `old-rdna-boosts` branch) plus
**block 12** (`patches/12-hybrid-allreduce-hip.patch`, the hybrid HIP
all-reduce — delta vs the block-11 tip, RDNA4-gated). Apply flow: `git am`
for 01-11 (plain `git apply` of the concatenated series SILENTLY DROPS
HUNKS — verified 2026-08-29), `git apply` for the 12th;
`scripts/apply-all.sh` automates it. **The set is whitespace-clean** —
applying produces zero git whitespace warnings (verified 2026-08-29).

> **Naming collision warning:** in the OLD pre-delivery docs (the historical
> records below, BASELINE.md, the `baseline/*` branches), "block 12"
> sometimes means the old *k-quant umbrella* (now block 10) and sometimes
> means the *hybrid all-reduce* (the current block 12). In THIS document
> and the current delivery, block 12 = the hybrid all-reduce, period.

History of the block structure: the work originated as 48 commits on the
fork's `chunked-gdn` branch (upstream `758443071`), decomposed into
functional blocks. Block numbering was compacted (2026-08-28): the k-quant
umbrella absorbed retired blocks 09 and 06, the set ran 01-11, and block 12
(hybrid all-reduce) was added as the delivery's final patch (2026-08-29).
Blocks 09 and 06 (old numbering) are retired: their content (Q6_K VDR=2
decode + the gfx1151 RDNA3_5 mmvq table) is folded into the k-quant umbrella
(block 10) so all k-quant VDR/decode work and all mmvq parameter-table
tuning lives in the one patch; excluding block 10 restores 100%
greedy-purity on ALL architectures (see the Greedy-purity note below).

This is the authoritative apply order and the verification contract for the
patch set. It is written for humans AND LLM coding agents. Follow it exactly;
do not skip blocks.

Current state: `main` is the delivery branch (flat history, 12-patch set
against `17252c769`). The `baseline/<sha>` branches and `block/01-…11` tags
are HISTORICAL checkpoints of the old pre-block-12 structure (older
upstream ranges, `git apply` flow); do not use them for the current
delivery — use `patches/` + `scripts/apply-all.sh`.


## Apply order (current delivery)

| # | patch file | content | deps |
|---|-----------|---------|------|
| 01 | `0001-…-block-01-adaptive-MTP-draft-depth.patch` | adaptive MTP draft depth | none |
| 02 | `0002-…-block-02-fused-chunked-gated-delta-net-p.patch` | fused chunked GDN prefill (bf16/WMMA; gfx12+gfx11 arch-segregated files, runtime-cc dispatch) | none |
| 03 | `0003-…-block-03-BF16-KV-cache-and-native-BF16-f.patch` | BF16 KV cache + native-BF16 flash-attn | none |
| 04 | `0004-…-block-04-RDNA4-WMMA-flash-attn-Q6_K-mmq-.patch` | WMMA flash-attn + Q6_K mmq prefill perf | none |
| 05 | `0005-…-block-05-CPU-bit-identical-decode-verify.patch` | CPU bit-identical decode/verify batches | none |
| 06 | `0006-…-block-06-host-buffer-revert-for-discrete.patch` | host-buffer revert for discrete GPUs | none |
| 07 | `0007-…-block-07-meta-device-wrapper-skip.patch` | meta device-wrapper skip | none |
| 08 | `0008-…-block-08-fused-core-prefill-kernels-and-.patch` | fused-core prefill kernels + GPU bit-identical results | **blocks 03 and 04 MUST be applied first** (fattn-tile.cuh / fattn.cu territory) |
| 09 | `0009-…-block-09-meta-buffer-compute-container-h.patch` | meta-buffer compute-container headroom | none |
| 10 | `0010-…-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch` | k-quant + mmvq-parameter umbrella (VDR kernels, RDNA3_5 table, MoE mmid) — the only decode-numerics patch | none (omit for greedy purity) |
| 11 | `0011-…-block-11-skip-CUDA-graphs-for-multi-toke.patch` | skip CUDA graphs for multi-token prefill | none |
| 12 | `12-hybrid-allreduce-hip.patch` | **hybrid HIP all-reduce** (internal AR for the small-tensor decode path + per-size hybrid dispatch vs RCCL; RDNA4-only gate: refuses to init off gfx1200/gfx1201, falls back to RCCL) | none (apply last) |

Block numbers are the apply order: `01` applies first, `12` last. All blocks
are mutually independent except **block 08 (fused core) requires blocks 03
and 04 in the tree**. Apply 01-11 with `git am` (or `scripts/apply-all.sh`),
then `git apply` the 12th — the concatenated-series `git apply` trick
silently drops hunks.


## Verified apply sequence (current delivery, 2026-08-29)

On a fresh checkout of the fork point `17252c769`:

```
git am patches/000[1-9]-*.patch patches/001[01]-*.patch   # blocks 01-11
#      (or: scripts/apply-all.sh — same thing, plus the 12th and a commit each)
git apply patches/12-hybrid-allreduce-hip.patch           # block 12
```

Verified end-to-end 2026-08-29: clean apply, full build, llama-cli
same-seed coherence IDENTICAL to the fork build, tg64 38.12 / tg512 41.08
(matches the fork build). **Do not `git apply` the concatenated 01-11
series directly — it silently drops hunks** (30 files / 2483 lines vs the
correct 35 / 6094). The historical validation records below (14883/14883,
GDN 46/46, etc.) are from the older 01-11 structure and remain the
verification evidence for the block content, which is byte-unchanged.

### Whitespace-clean regeneration (2026-08-29, follow-up)

The previous patch files carried trailing-whitespace lines (8
pure-whitespace blank lines in block 02's two bf16 GDN files + one
blank-at-EOF line in block 12's `allreduce.cuh`), which made `git am` /
`git apply` print whitespace warnings on every apply. Fixed at the source:
the fork's `rdna-boosts` block commits were rebuilt in place (each
commit's diff re-applied with `git apply --whitespace=fix`) and the whole
12-patch set re-generated with `scripts/make-patches.sh` (block-12 tip
`cc985ba9a`, block 12 now committed as `12d10267b`).

Re-verified end-to-end 2026-08-29: `scripts/apply-all.sh` on a fresh
checkout at `17252c769` runs with **ZERO whitespace warnings**; the
applied tree is byte-identical to the previous applied tree except the 8
whitespace lines and 1 EOF blank line (all inert — pure-whitespace blank
lines, no string-literal or continuation content). No behavioral change:
the validation records above still describe this set.


## Verification per block

| block | verify command | expected |
|-------|----------------|----------|
| 01 | `./bin/test-speculative-adaptive && ./bin/test-arg-parser`; llama-server `--draft-mtp-adaptive` smoke | pass |
| 02 | `./bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET` | 46/46 on all four dispatch configs (default bf16, `GGML_CUDA_GDN_CHUNKED_BF16=0`, `GGML_CUDA_GDN_CHUNKED=0`, +/- graphs) |
| 03 | `./bin/test-backend-ops -b ROCm0 -o FA_ATTN_*` (BF16 KV cases) + bf16-KV model run | pass |
| 04 | `./bin/test-backend-ops -b ROCm0` (attention correctness) + decode/prefill perf on gfx1201 | pass / perf |
| 05 | speculative-decoding determinism test with the CPU backend | identical decode vs verify batches |
| 06 | build + Q8_0 decode on gfx1151 | pass / perf |
| 07 | build + decode perf on integrated-GPU HIP target | pass / perf |
| 08 | `./bin/test-backend-ops -b ROCm0` (MUL_MAT Q6_K cases) + Q6_K decode on gfx1201 | 1194/1194 MUL_MAT OK |
| 09 | build + server `--split-mode tensor` + MTP draft smoke | loads/serves, no graph-alloc abort |
| 10 | `./bin/test-backend-ops -b ROCm0` (MUL_MAT + MUL_MAT_ID q4_K/q5_K cases) + Q4_K/Q5_K/Q8_0 decode on gfx1201; RDNA3_5 table: Q8_0 decode on gfx1151 | 54/54 MUL_MAT, 76/76 MUL_MAT_ID OK |
| 11 | build + prefill perf A/B on gfx1201 (pp128/256/512 vs longer) | decode unchanged; prefill +6-18% for single-ubatch (pp <= ~512), neutral (~0.1%) beyond |
| 12 | llama-cli same-seed coherence (2- and 3-GPU) + depth-16384 decode A/B (hybrid vs nccl vs internal) | same-seed output IDENTICAL to RCCL; 3-GPU hybrid 38.71 t/s (unpinned) at depth-16384; tg64 38.12 / tg512 41.08 |

Convenience: `rdna-boosts-all.patch` (repo root) is the entire 12-patch net
as ONE patch (applies cleanly on `17252c769` alone; not a substitute for the
per-block flow in `patches/` when you want reviewable increments).


## Failure handling (agent instruction)

> Apply the patches in `patches/` in the order given in this file to a fresh
> branch from the fork point `17252c769` (see `scripts/apply-all.sh` for the
> automated flow). After each patch, run its verification command. Blocks
> 01-11 apply with `git am`; if a patch fails, `git am -3` / `git apply -3`
> (3-way merge against the baseline blobs), then manually rebase the hunks
> against the current master and continue. Do not skip blocks. The fused
> core (`0008-...`) is applied at position 8; blocks 09-11 go after it, then
> the 12th (`12-hybrid-allreduce-hip.patch`, `git apply`). If more than one
> block needs manual re-base hunks, regenerate the whole set from the fork
> with `scripts/make-patches.sh` instead of hand-editing the committed
> patches.


## Known notes (working)
- **iq1_m MUL_MAT_ID flake — FIXED in block 08** (Q8_1 input cache keyed by `src1->data`); full story: `archive/docs/validation-history.md`
- **MTP draft + `--split-mode tensor` graph-alloc crash — FIXED by block 09** (meta headroom 16x -> 128x); full story: `archive/docs/validation-history.md`
- **Block 02 carries the test-harness seeding fix** (restores `random_device` seeding; the fork's deterministic seed exposed `rms_norm_back`/`cross_entropy_loss_back` fragility on RDNA4). GDN kernels untouched. See `archive/docs/baseline-history.md` for the diagnostic.
- **Block 08 test hunk** was re-based against the original baseline: its
  original context lines (Q6_K perf cases) were added by block 04. On this
  branch the hunk applies with the re-based placement; content is identical
  to the fork, only position differs.
- **Block 10 test hunks** anchor on the Q6_K/Q8_0 decode-shape rows and the
  `qwen3-30b-a3b` loops (folded in from retired block 09). Applied in
  manifest order they sit on the re-based block-09 hunk; do not re-order
  the test hunks when re-basing.
- **`test-backend-ops.cpp` is shared** by blocks 02/03/04/08/10. The hunks
  are in different case regions; if upstream adds cases in those regions,
  re-base the affected hunks (each patch applies independently on the
  baseline, so re-basing is local to the failing file).
- **Block 08 is genuinely inseparable**: its 16 commits co-developed the
  fused mmvq kernel region, the `ggml-cuda.cu` try_fuse machinery, and the
  `fattn.cu` dispatch cluster. It is extracted as ONE combined diff on
  purpose. Do not try to split it.
- **Blocks 06 and 09 (old numbering) were retired**: their content folded into block 10 (the k-quant umbrella); numbering compacted to 01-10, then 01-11, then +12. Full history: `archive/docs/baseline-history.md`.
- When upstream master moves past the fork point and more than one block needs manual re-base hunks, regenerate from the fork with `scripts/make-patches.sh` (see `BASELINE.md` drift policy).

## History
All dated validation records, the retired block-structure churn, and the old block-11 perf profile now live in `archive/docs/validation-history.md` and `archive/docs/baseline-history.md`.
