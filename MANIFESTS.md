# MANIFESTS - apply order and verification contract

Squashed, standalone diff blocks of RDNA-specific performance and correctness
work from the [llama.cpp fork](https://github.com/stew675/llama.cpp)
(`rdna-boosts` branch), packaged for easy application to mainline llama.cpp.

The **current delivery** is a **13-patch set** against the fork point
`9cffdcc80` (re-based 2026-09-02 from `0eadefebd`): blocks 01-13
(`patches/0001-…0013-…`, format-patch of the
fork's `rdna-boosts` block commits — the re-based regeneration
`04122bfb5..b830050bf` against `9cffdcc80`; block 12 was amended
2026-09-04 with the runtime NCCL-failure fallback (issue #13) and block
13 was amended 2026-09-02 with two MTP regression fixes, see the dated
records below; the previous `0eadefebd`-based regeneration
`b25bc8a9c..482837e5a` is superseded and preserved on the fork's
history/remotes). Apply flow: `git am`
for the whole 01-13 series (plain `git apply` of the concatenated series
SILENTLY DROPS HUNKS — verified 2026-08-29);
`scripts/apply-all.sh` automates it. **The set is whitespace-clean** —
applying produces zero git whitespace warnings (verified 2026-08-29,
re-verified 2026-09-01 on the `0eadefebd` re-base, re-verified with block
13 on the 13-patch series 2026-09-01, re-verified on the `9cffdcc80`
re-base 2026-09-02, re-verified after the 2026-09-02 block-13 amendment,
re-verified after the 2026-09-04 block-12 amendment).

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
greedy-purity on ALL architectures on the K-split decode paths (12-block-era
claim; with block 13 installed, its rewritten short-K mmvq rows also deviate
from stock — see `GREEDY-PURITY.md` §9).

This is the authoritative apply order and the verification contract for the
patch set. It is written for humans AND LLM coding agents. Follow it exactly;
do not skip blocks.

Current state: `main` is the delivery branch (flat history, 13-patch set
against `9cffdcc80`). The `baseline/<sha>` branches and `block/01-…11` tags
are HISTORICAL checkpoints of the old pre-block-12 structure (older
upstream ranges, `git apply` flow); do not use them for the current
delivery — use `patches/` + `scripts/apply-all.sh`.


## Apply order (current delivery)

| # | patch file | content | deps |
|---|-----------|---------|------|
| 01 | `0001-…-block-01-adaptive-MTP-draft-depth.patch` | adaptive MTP draft depth | none |
| 02 | `0002-…-block-02-fused-chunked-gated-delta-net-p.patch` | fused chunked GDN prefill (bf16/WMMA; gfx12+gfx11 arch-segregated files, runtime-cc dispatch; MTP long-prefill chunked-prefix + sequential K-tail, PR #9) | none |
| 03 | `0003-…-block-03-BF16-KV-cache-and-native-BF16-f.patch` | BF16 KV cache + native-BF16 flash-attn | none |
| 04 | `0004-…-block-04-RDNA4-WMMA-flash-attn-Q6_K-mmq-.patch` | WMMA flash-attn + Q6_K mmq prefill perf | none |
| 05 | `0005-…-block-05-CPU-bit-identical-decode-verify.patch` | CPU bit-identical decode/verify batches | none |
| 06 | `0006-…-block-06-host-buffer-revert-for-discrete.patch` | host-buffer revert for discrete GPUs | none |
| 07 | `0007-…-block-07-meta-device-wrapper-skip.patch` | meta device-wrapper skip | none |
| 08 | `0008-…-block-08-fused-core-prefill-kernels-and-.patch` | fused-core prefill kernels + GPU bit-identical results | **blocks 03 and 04 MUST be applied first** (fattn-tile.cuh / fattn.cu territory) |
| 09 | `0009-…-block-09-meta-buffer-compute-container-h.patch` | meta-buffer compute-container headroom | none |
| 10 | `0010-…-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch` | k-quant + mmvq-parameter umbrella (VDR kernels, RDNA3_5 table, MoE mmid) — the only decode-numerics patch | none (omit for greedy purity) |
| 11 | `0011-…-block-11-skip-CUDA-graphs-for-multi-toke.patch` | skip CUDA graphs for multi-token prefill | none |
| 12 | `0012-…-block-12-hybrid-HIP-all-reduce-RDNA4-gat.patch` | **hybrid HIP all-reduce** (internal AR for the small-tensor decode path + per-size hybrid dispatch vs RCCL; RDNA4-only gate: refuses to init off gfx1200/gfx1201, falls back to RCCL) | none (apply last) |
| 13 | `0013-…-block-13-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** (prefill fused expert MMQ, RDNA4, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K + decode item-split, re-based on the upstream has_fusion mmvq path; multi-token mmvq x_scale_channel_dst fusion for MoE down x topk-weights, spec-dec verify batches n=2..8; ROCm unaligned-width split-load fix for Q6_K/Q3_K 2-GPU) | none (apply last) |

Block numbers are the apply order: `01` applies first, `13` last. All blocks
are mutually independent except **block 08 (fused core) requires blocks 03
and 04 in the tree**. Apply the whole 01-13 series with `git am` (or
`scripts/apply-all.sh`) — the concatenated-series `git apply` trick
silently drops hunks.


## Verified apply sequence

### Re-baseline to 9cffdcc80 (2026-09-02)

Upstream master moved **42 commits** past the fork point `0eadefebd`; 3
touching ggml-cuda — `3d3d7c818` (unused-var removals, #28235),
`8e93a9773` (sparse-fa for DSV4/GLM, #27970: a 4th `use_sparse` bool on
`launch_fattn`, fattn-tile/fattn-common edits) and `3466812d1` (fused MoE
weighted-expert reduction, #25952: a new arm in `ggml_cuda_try_fuse`) —
plus common/server arg churn (`e750b887a`). The fork's `rdna-boosts`
branch was rebuilt from the delivery patches on the new base
(`~/llama.cpp`, blocks `04122bfb5..92f09e80a`; plain `git am`, with the
failed hunks resolved by hand per the BASELINE drift policy — no
fork-blob 3-way crutches, matching what a fresh puller experiences) and
the set regenerated with `scripts/make-patches.sh` (base `9cffdcc80`,
blocks tip `92f09e80a`). Three blocks needed manual re-base hunks:

1. **Block 03 vs #27970:** 6 fattn-tile.cuh call sites + the
   `launch_fattn_tile_switch_ncols2` template line (type_KV threading);
   merged as `need_f16_K, need_f16_V, false, false, warp_size` on each
   `launch_fattn` call (upstream's `stream_k`/`use_sparse` stay false).
2. **Block 08 vs #25952:** the rms_norm->mmvq quantize-fold arm now sits
   after upstream's `GGML_OP_MUL` MoE-reduction arm in
   `ggml_cuda_try_fuse`.  Also folded into the block-08 commit: the
   block-08 spec-verify `launch_fattn` call site in fattn-tile.cuh still
   passed the pre-#27970 3-bool arg list — the `warp_size` int bound
   into the new `use_sparse` bool slot (compiles; `use_sparse=true`;
   runtime `GGML_ASSERT(n_kv_max > 0)` in fattn-common.cuh).  Fixed to
   the 4-bool form; caught by the coherence gate (crash), not the build.
3. **Block 13 vs #25952:** `disable_moe_mmq` opt-out static + `const int
   cc` decls at the top of `ggml_cuda_try_fuse` restored after
   upstream's inserted arm.

Regenerating from the new base folds upstream's changes into the patch
context, so **`scripts/apply-all.sh` applies all 13 blocks with plain
`git am` — zero conflicts, zero whitespace warnings** on a fresh checkout
at `9cffdcc80`. Re-verified end-to-end 2026-09-02: clean-apply sim
(fresh worktree at `9cffdcc80`, applied tree byte-identical to the fork
tip `92f09e80a`), full build clean (ROCm 7.14 gfx1201, RCCL+graphs+
native, zero errors; build note: `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="`
is required with CMake >= 4.3 — the build script's bare `-mllvm`
swallows the HIP-test-injected `--cuda-host-only`), llama-cli same-seed
coherence **IDENTICAL between hybrid and RCCL** (3-GPU tensor split,
Qwen3.5-4B Q8_0). Numbers unchanged (content-identical plus upstream's
additions).

### Block-13 MTP regression fixes (2026-09-02, current)

Block 13 (the fork's block-13 commit, amended in place) now carries two
regression fixes found by the adaptive-MTP investigation of 2026-09-02:

1. **Dense MTP/verify decode collapse** (`mmvq.cu`): the block-13 mmvq
   item-split kernel + RDNA rows_per_block override is register-bound at
   multi-token decode batches (ncols 2..8 = the speculative verify step;
   `tmp[ncols_dst][rpb]` fan-out).  Fix: re-add the pre-block-13 K-split
   kernel as `mul_mat_vec_q_ksplit` and dispatch ncols 2..8 + long-K
   (K >= 4096) ncols==1 rows to it.  Dense qwen35 27B Q4_K_XL adaptive-MTP
   18.3 -> 27.5 t/s (output bit-identical to the 12-block build); plain
   decode 29.0 -> 30.1 (+3.1-3.7% at d0/d16384/d65536).
2. **MoE MTP verify-numerics collapse** (`ggml-cuda.cu` try_fuse arm): the
   block-08 rms_norm->mmvq Q8_1 quantize-cache fold corrupts multi-token
   MUL_MAT_ID (the moe kernel consumes the cached Q8_1 y wrongly), so MoE
   verify-batch logits diverge from single-token decode and MTP draft
   acceptance collapses to 0/1527 (draft-mtp 53 vs plain 90 t/s on
   qwen35moe-A3B Q4_K_M-UD; upstream accelerates +51%).  Fix: gate the
   fold to single-token MMID (ne[2]==1) and plain MUL_MAT consumers.
   MoE acceptance restored to 0.51 (== fully-unfused 0.49 == upstream
   0.49), draft-mtp 119-129 t/s vs upstream ~110-113; MoE plain decode
   and the single-token fusion gains unchanged; dense unaffected.

Re-verified end-to-end 2026-09-02 after the amendment: set regenerated
from the fork (`scripts/make-patches.sh`, base `9cffdcc80`, blocks tip
`8f2838d1`), clean-apply sim at `9cffdcc80` (git am clean, zero
whitespace warnings, applied tree byte-identical to the fork tip),
build clean, dense same-seed coherence identical, MoE MTP sanity on the
sim build (acceptance 0.54, 128 t/s).  The adaptive-MTP baseline gate +
numbers now live in `benchmarks/mtp-adaptive-methodology.md` (the
decode-only suites cannot see MTP regressions — verify batches ncols 2..13
and the draft context are never exercised there).

### Runtime NCCL-failure fallback (2026-09-04, issue #13, current)

Community issue #13 (reporter tungel — same reporter as the #5/#6 fix
round): on a topology where RCCL >= 2.30.4 cannot dispatch its kernels,
`ncclCommInitAll` succeeds but the first collective aborts
(`hipErrorIllegalState` when a GPU sits behind a PCIe root port without
32/64-bit AtomicOp completer support — e.g. PCH/Z390; see
ROCm/ROCm#6520), and the process died at the first prefill AllReduce
(`NCCL_CHECK` -> `GGML_ABORT`) even though the internal host-staged
pipeline was up and stable.  Folded into the block-12 commit: on the
first NCCL runtime failure the comm layer clears the sticky HIP errors
on each AR device (else the fallback aborts on the next CUDA_CHECK),
warns once with a pointer at the known cause + the
`dmesg | grep -i atomic` check, permanently stops using NCCL (comm
state is unknown), re-routes subsequent AllReduce to the internal
pipeline (or the meta backend's butterfly when no pipeline is
available), and the failing call itself returns false so the butterfly
handles it; `ncclCommDestroy` at teardown is non-fatal too.  No behavior
change on healthy setups — the fallback only triggers when NCCL itself
fails.

Re-verified end-to-end 2026-09-04: set regenerated from the fork
(`scripts/make-patches.sh`, base `9cffdcc80`, blocks tip `b830050bf`),
clean-apply sim at `9cffdcc80` (git am clean, zero whitespace warnings,
applied tree byte-identical to the fork tip), full build clean (ROCm
7.14 gfx1201, RCCL+graphs+native), llama-cli same-seed coherence
IDENTICAL pre vs post fix (27B Q8_0, 3-GPU tensor split), and perf
unregressed at depth-16384 hybrid: 2-GPU (1,2) tg128 32.48 -> 32.40,
3-GPU (0,1,2) tg128 39.33 -> 39.31 (both within noise; pp512 within
run-to-run spread).

### Re-baseline to 0eadefebd (2026-09-01)

Fork point moved from `a7cc83bba` to upstream master `0eadefebd` (22
commits of drift; 3 touching ggml-cuda — XOR-swizzle fattn #25635, radix
TOP_K #27466, MOE-fusion #27621). The fork's `rdna-boosts` branch was
rebuilt from the delivery patches on the new base (worktree at
`0eadefebd`; blocks 01-07 + 09-12 applied cleanly, block 08 via
`git am -3` auto-3way; the tree is byte-identical to the verified
2026-09-01 cross-version apply below) and the set regenerated with
`scripts/make-patches.sh` (base `0eadefebd`, blocks tip `d7bdd0a91`,
block 12 committed as `ce9182473`; the old `a7cc83bba`-based fork state
is preserved on the `rdna-boosts-a7cc83bba` branch). Regenerating from
the new base folds upstream's changes into the patch context, so
**`scripts/apply-all.sh` now applies all 12 blocks with plain `git am` —
zero conflicts, zero whitespace warnings** on a fresh checkout at
`0eadefebd` (the 2026-09-01 apply below needed `git am -3` for block 08
only because the set then still carried the old base's context). Verified
end-to-end 2026-09-01: clean-apply sim on a fresh clone at `0eadefebd`
(sim tree byte-identical to the fork tip `ce9182473`), full build clean,
llama-cli same-seed coherence IDENTICAL to the pre-re-base known-good
build, tg64 38.12 / tg512 41.08 unchanged (code-identical content;
re-measured 2026-09-01: sim 36.87±4.83 / 40.72±1.02, prs 37.92±4.67 /
40.90±0.86 — within noise).

### AR_PROFILE devices[] init fix + fork re-sync (2026-09-01)

- **Fix (PR #8, integrated into block 12 + the fork):** in
  `allreduce-hip.cu`, `p->devices[]` is now filled from the caller list
  BEFORE the per-device profiler hipMallocs.  Under
  `GGML_CUDA_AR_PROFILE=1` the buffers were previously allocated while
  `devices[]` was still zero-filled, so every buffer landed on GPU 0 and
  MTP's second pipeline init (draft context) faulted/hung GPU 1
  (gfx1201).  A/B on 3x R9700 (2-GPU, internal AR, MTP n-max 3,
  `-c 32768`, AR_PROFILE=1): pre-fix reproduced — GPU-1 memory fault in
  `ggml_cuda_ar_kernel` (exit 134); post-fix runs clean with teardown
  profiler dumps on dev0 AND dev1 in both pipelines; llama-cli same-seed
  coherence IDENTICAL to the pre-fix golden (default serving, profiler
  off, is byte-for-byte unchanged).
- **Fork re-sync:** the fork's `rdna-boosts` was rebuilt as a clean
  12-commit branch directly on `0eadefebd` (block 01 `b25bc8a9c` .. block
  11 `43f5ab71d`, block 12 `93e8b09bb`).  The previous fork rebuild had
  picked up upstream master's `kleidiai` docs commit `518b76236` as a
  13th base commit; that upstream commit is NOT part of the block set
  (it remains in upstream `origin/master`) and was dropped from the
  branch.
- **Set regenerated:** `scripts/make-patches.sh` (base `0eadefebd`,
  blocks tip `43f5ab71d`) re-exported blocks 01-11 (content-identical to
  the previous delivery — only the `From <sha>` headers moved) + the
  block-12 delta (with the AR_PROFILE fix); `rdna-boosts-all.patch`
  regenerated as `git diff 0eadefebd..93e8b09bb`.
- **Re-verified 2026-09-01:** clean-apply sim on a fresh clone at
  `0eadefebd` — `scripts/apply-all.sh` applied all 12 blocks with ZERO
  whitespace warnings and the applied tree is byte-identical to the fork
  tip (`d42fc80…`); the fork tree was fully built (ROCm 7.14 gfx1201,
  clean) and coherence-tested as part of the A/B above.

### MTP chunked-GDN prefix folded into block 02 (2026-09-01, PR #9)

- **Change (PR #9, integrated into block 02 — NOT a new block):** block
  02's chunked WMMA GDN only launched for `K == 1` (no MTP snapshots);
  with MTP n-max 3 (`K=4`) every prefill ubatch stayed on the sequential
  kernel (rocprof ~4k wrap: sequential GDN at 10.45%).  The dispatch now
  runs, for long single-sequence prefills (`!kda && K > 1 && n_seqs == 1
  && n_tokens > K+64`), the chunked GDN on the prefix (`n_tokens - K`)
  and sequential GDN only on the last K tokens so slots `0..K-1` stay
  correct (fused-cache graphs included; `n_seqs > 1` stays fully
  sequential).  The chunked ops gained an `n_tokens_limit` parameter
  (`gated_delta_net_chunked{.cu,.cuh,_bf16.cu,_bf16_gfx11.cu}`).
  Opt out: `GGML_CUDA_GDN_CHUNKED=0`.  Because the change is confined to
  block-02 files, it was folded into the block-02 commit (fixup +
  autosquash; blocks 03-12 replayed cleanly, tree unchanged).
- **Verified 2026-09-01 (3x R9700 gfx1201, ROCm 7.14, 2-GPU 0,1,
  internal AR, Qwen3.8-27B Q8, ubatch 1024, MTP n-max 3):** path fire
  confirmed (`MTP chunked GDN prefix n=1024 K=4 prefix=1020`); prefill
  tok/s: ~5.5k prompt 1406.5 -> 1511.4 (+7.5%), ~38k 1303.0 -> 1403.1
  (+7.7%); 64-token same-seed output token-IDENTICAL vs sequential (only
  the timing line differed).  Non-MTP serving unchanged (coherence
  IDENTICAL to the pre-change golden).  Full clean build passes.  PR's
  lab numbers (up to +8.1% at 40k, GSM8K 19/50 vs 18/50):
  `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
- **Set regenerated:** blocks 01, 03-11 content-identical (only `From
  <sha>` headers moved); block 02 = old block 02 + the PR #9 hunks;
  `rdna-boosts-all.patch` regenerated.  Clean-apply sim re-verified on a
  fresh clone at `0eadefebd` (zero whitespace warnings, applied tree
  byte-identical to the fork tip `b90eb525e`).

### Re-baseline to a7cc83bba (2026-08-30, superseded)

Fork point moved from `17252c769` to upstream master `a7cc83bba` (24
commits of drift; 6 touching ggml-cuda). The fork's `rdna-boosts` branch
was rebuilt from the delivery patches on the new base and the set
regenerated with `scripts/make-patches.sh` (base `a7cc83bba`, blocks tip
`8fbf10e5b`, block 12 committed as `4fa92f0ae`). Blocks 01-07 and 09-12
applied cleanly; the ONE conflict was block 08 vs upstream's SWIGLU_CLAMP
(#27930, landed 2026-08-30): its `glu_limit` additions to the mm-fusion
args structs (`common.cuh`) and `mmvq.cu` (decls, fusion-assign, the
GLU-switch/result-write restructure, `fusion_local`) were merged alongside
block 08's `dst_gate`/`conv_*`/`x_scale_channel_dst` work (verified: the
merged files diff vs block-08's post-image blobs = exactly upstream's
additions, nothing else). Verified end-to-end 2026-08-30: clean-apply sim
on a fresh clone at `a7cc83bba` (zero conflicts, zero whitespace
warnings), full build clean, llama-cli same-seed coherence IDENTICAL to
the pre-re-base known-good build.

### Cross-version apply at 0eadefebd (2026-09-01, record)

Upstream master moved 22 commits past the fork point; 3 touched
ggml-cuda (XOR-swizzle fattn #25635, radix TOP_K #27466, MOE-fusion
#27621). The 12-patch set was applied to a fresh clone at `0eadefebd`
(branch `rdna-boosts`): blocks 01-07 + 09-12 clean (`git am` / `git
apply`); the ONE conflict was block 08, resolved by `git am -3` 3-way
merge (auto-resolved, zero manual hunks). Full-tree zero-drift check:
`fork-tip → HEAD` = exactly the 51-file upstream delta, all diffs
content-identical vs the fork delivery tip `4fa92f0ae`. Build clean
(ROCm 7.14 gfx1201) and same-seed coherence hybrid == RCCL (3-GPU).
Fork point unchanged at `a7cc83bba` per the drift policy (regeneration
triggered only when >1 block needs manual re-base hunks; here: one
block, auto-3way). The verified applied state is tagged
`rdna-boosts-0eadefebd` in the ~/prs/llama.cpp clone. Full record:
`BASELINE.md`.

### Baseline 17252c769 (2026-08-29, superseded)

On a fresh checkout of the fork point `17252c769`:

```
git am patches/000[1-9]-*.patch patches/001[0-3]-*.patch   # blocks 01-13
#      (or: scripts/apply-all.sh — same thing, one commit each)
```

Verified end-to-end 2026-08-29: clean apply, full build, llama-cli
same-seed coherence IDENTICAL to the fork build, tg64 38.12 / tg512 41.08
(matches the fork build). Re-verified 2026-09-01 with block 13 on the
13-patch series: clean apply, applied tree byte-identical to the fork tip.
**Do not `git apply` the concatenated 01-13 series directly — it silently
drops hunks** (30 files / 2483 lines vs the correct 35 / 6094 for the old
12-set; the same caveat applies). The historical validation records below
(14883/14883, GDN 46/46, etc.) are from the older 01-11 structure and
remain the
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

### Block-12 compiler-warning cleanup (2026-08-29, follow-up)

The HIP port's `allreduce-hip.cu` was the HIP build's ONLY source of
compiler warnings.  ROCm 7.14 marks the entire `hipError_t` enum
`[[nodiscard]]`, so every unchecked HIP call emitted `-Wunused-value`
(27 sites / 54 warning lines in the ggml-hip build — every other file in
the tree checks or `(void)`-casts each hip call).  Fixed at the source in
the fork: all 27 sites wrapped in `CUDA_CHECK(...)` (upstream house
style, including teardown frees/destroys, which the CUDA original leaves
unchecked but HIP's nodiscard enum flags), plus three dead WIP items
removed (unused `stage_marker` kernel parameter, unused `wire_bf16` local
in the stage hook, uncalled `ggml_cuda_ar_arrival_ptr` helper).  The
block-12 commit was amended in the fork (tip now `43e6ced06`) and the
12-patch set re-generated with `scripts/make-patches.sh`.

Re-verified 2026-08-29: full HIP build (ROCm 7.14, gfx1201) emits ZERO
compiler warnings from the patch (the only remaining build warning is the
build script's `-mllvm` link-time artifact — pre-existing, unrelated);
`scripts/apply-all.sh` on a fresh checkout at `17252c769` applies with
zero whitespace warnings and the applied tree is byte-identical to the
fork tip; llama-cli same-seed coherence still IDENTICAL to RCCL (3-GPU);
the `GGML_CUDA_AR_PROFILE=1` teardown path (where most of the new
CUDA_CHECKs live) runs clean.


### Community-report fix round (2026-08-30, issues #5 + #6)

External report (tungel, 2x gfx1201) surfaced two block-12 bugs, fixed at
source in the fork (`~/llama.cpp` rdna-boosts, tip `8a426cf79`) and the
12-patch set regenerated with `scripts/make-patches.sh`:

- **RCCL-less build failure (#5):** `comm_init_hybrid` referenced
  `ggml_backend_cuda_comm_try_allreduce_nccl` (defined only under
  `GGML_USE_NCCL`) unconditionally — `-DGGML_HIP_RCCL=OFF` builds failed
  to compile.  The reference is now guarded; the no-NCCL flavor keeps the
  internal-pipeline behavior (verified: RCCL=OFF `ggml-hip` builds).
- **Unbounded in-kernel spin (#6):** the chunked AR kernel spun on peer
  arrival with no exit condition; on RDNA (non-preemptible compute
  kernels) a lost arrival wedges the queue -> MES `REMOVE_QUEUE` timeout
  -> MODE1 reset -> `700/719` or whole-machine freeze.  The spin is now
  bounded (`GGML_CUDA_AR_SPIN_TIMEOUT_MS`, default 20 ms, `0` = legacy);
  on timeout the kernel sets a host-mapped poison flag, skips the reduce
  and exits, and the host re-syncs the devices with a butterfly AllReduce
  on the next call.

Re-verified 2026-08-30: clean-apply sim at `17252c769` (apply, full
build, llama-cli same-seed coherence IDENTICAL); RCCL=OFF `ggml-hip`
compiles; before/after perf (default hybrid, 2x R9700, depth-16384):
pp512 1622.7 -> 1609.1 t/s and tg128 32.63 -> 32.55 t/s — both within
run-to-run noise, no measurable impact from the bounded-spin fix.

> **Hash-drift note:** the 01-11 patch `From:` headers now carry the
> current fork's commit hashes (e.g. block 01 = `142ab7846`); the earlier
> records (tip `12d10267b` / `43e6ced06`) refer to the previous fork
> build, which was rebuilt with identical content but new hashes.  The
> diff bodies are unchanged.

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
| 13 | `test-backend-ops` MUL_MAT_ID_FUSION sweep (bs 1/4/512; 16222/16222) + MoE decode perf; **MTP regression gate** — `benchmarks/mtp-adaptive-methodology.md` protocol A on the dense Q4_K_XL-UD and MoE Q4_K_M-UD (seed-42: draft acceptance > ~0.45, draft-mtp >= plain at depth 3) | fused types pass; Q6_K tg128 >= 97.6; dense mtp 27.5 / plain 30.1; MoE acceptance 0.51, draft-mtp 126 t/s |

Convenience: `rdna-boosts-all.patch` (repo root) is the entire 13-patch net
as ONE patch (applies cleanly on `9cffdcc80` alone; not a substitute for the
per-block flow in `patches/` when you want reviewable increments).


## Failure handling (agent instruction)

> Apply the patches in `patches/` in the order given in this file to a fresh
> branch from the fork point `0eadefebd` (see `scripts/apply-all.sh` for the
> automated flow). After each patch, run its verification command. Blocks
> 01-11 apply with `git am`; if a patch fails, `git am -3` / `git apply -3`
> (3-way merge against the baseline blobs), then manually rebase the hunks
> against the current master and continue. Do not skip blocks. The fused
> core (`0008-...`) is applied at position 8; blocks 09-11 go after it, then
> blocks 12-13 (hybrid AR, fused MoE). If more than one
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
- **Block 08 vs upstream SWIGLU_CLAMP (2026-08-30 re-base):** upstream's
  #27930 added `glu_limit` to the same mm-fusion regions block 08 rewrites
  (`common.cuh` args structs; `mmvq.cu` decls / fusion-assign /
  GLU-switch restructure / `fusion_local`). The re-base merged them side
  by side; the SWIGLU_CLAMP case now lives inside block 08's restructured
  switch on `result_val`. If a future re-base hits this again: keep
  upstream's `glu_limit` lines, re-apply block 08's additions around
  them, and verify with the post-image-blob diff (merged file minus
  block-08 blob must equal exactly upstream's additions).
- **Blocks 06 and 09 (old numbering) were retired**: their content folded into block 10 (the k-quant umbrella); numbering compacted to 01-10, then 01-11, then +12. Full history: `archive/docs/baseline-history.md`.
- When upstream master moves past the fork point and more than one block needs manual re-base hunks, regenerate from the fork with `scripts/make-patches.sh` (see `BASELINE.md` drift policy).

## History
All dated validation records, the retired block-structure churn, and the old block-11 perf profile now live in `archive/docs/validation-history.md` and `archive/docs/baseline-history.md`.
