#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline the patches are generated against
#                 (default: 790cf51aa, see MANIFESTS.md)
#   blocks-tip    the fork commit carrying block 00 + all 15 feature
#                 blocks (default: c45244c728dfcbcad86ae95aa97ae76f94ee9f7f,
#                 the block-15 commit of the
#                 CANONICAL fork rebuilt at 790cf51aa, after the 2026-09-11
#                 block-13 amendment -- the MoE decode/verify mmvq band --
#                 the 2026-09-11 amendments to block 14 (the QSA decode
#                 arm band, the QSA quantized-KV enablement + the
#                 K/V-head-aware block chunking, the iq4_nl enablement
#                 whose block-08 half -- the predicate, the vec
#                 instances and the non-contiguous converters -- is the
#                 fifth block-08 amendment, and the mixed-K/V hard reject),
#                 the block-01 amendment that caps --spec-draft-n-max
#                 at 7 with a visible notice, and the block-02
#                 K-independent whole-batch chunked GDN prefill, and the
#                 2026-09-12 block-13 column-blocked shared-expert
#                 epilogue (bit-identical; repays the band amendment's
#                 pl=8 cost) and the 2026-09-12 block-13 RDNA3_5 single-token-only
#                 mmvq fusion skip (the dense gate+up+GLU fusion and the weighted-down
#                 MoE tail are single-token-only and not bit-identical with the
#                 standalone arithmetic on gfx1151; gated there unless the A/B opt-in
#                 GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1 is set), the 2026-09-12 block-02
#                 amendment (the rollback-bounded chunked-GDN threshold `n_rs_batch` + the
#                 pre-batch snapshot slots), and the
#                 2026-09-12 block-14 QSA prefill crossover (qsa_dense_prefill_until,
#                 per-split defaults: gfx1151 8192 / tensor split 16384 / other 0, env
#                 LLAMA_QSA_DENSE_PREFILL_UNTIL) together with the qsa_op_supported()
#                 device query that replaced the hand-maintained qsa_kv_native type list,
#                 and the 2026-09-12 block-14 (seventh) MTP-export logits-purity fix
#                 (the last layer always gathers its output rows; the unmasked
#                 embeddings_nextn export gets a separate full-row tail for t_h_nextn),
#                 and the 2026-09-12 block-14 (eighth) decode/verify band-uniformity fix
#                 for the QSA indexer score (its flattened n_idx_h*n_tps N dimension crossed
#                 MMVF_MAX_BATCH_SIZE at n_tps=3, so the verify batch fell through to MMF
#                 while decode stayed on MMVF; the guard now covers the whole flattened band
#                 MMVF_MAX_BATCH_SIZE_FLAT=32 and mul_mat_vec_f is instantiated for
#                 ncols_dst 9..32),
#                 and the 2026-09-12 block-15 promotion (the attention-memory campaign:
#                 W1 GGML_QSA_SCORE_MEM, W2 GGML_QSA_DERIVED_BIAS/_VIS, W3
#                 LLAMA_QSA_KEYS_ONLY, W4 ggml-alloc unused-view release, V3
#                 LLAMA_KQ_MASK_DERIVED, V4/V5 GGML_CUDA_FA_KV_NATIVE (opt-in);
#                 promoted from archive/work/block-15-campaign-wins/).
#                 And the 2026-09-12 (16) block-08 + block-10 amendment (issue #30):
#                 the RDNA4 calc_nwarps table is band-uniform nwarps=1 and block 10's
#                 VDR=4 mmvq boost is reverted (see patches/README.md).
#                 And the 2026-09-12 (17) block-10 amendment: the VDR is split per kernel --
#                 the dense mmvq selectors (mul_mat_vec_q item-split / _ksplit) stay at the
#                 upstream VDR while the MoE expert kernel (mul_mat_vec_q_moe) takes block
#                 10's VDR=4 through its own selectors, both still band-uniform internally.
#                 And the 2026-09-12 (18) block-13 amendment: the dense mmvq *weight* kernel
#                 (_ksplit) picks nwarps per (type, K) -- a Q8_0 weight with K < 4096 (the
#                 MoE attention qkv/gate and the lm_head) takes the pre-2026-09-11 wide
#                 block (nwarps=8), every other shape stays at 1; the pinned fusion ops
#                 (GDN/SSM, shared-expert, the gate fusions) keep band-uniform calc_nwarps.
#                 And the 2026-09-13 block-08 amendment (TODO item 3): the iq4_nl
#                 `GET_ROWS` sub-QK_K path (a row width that is not a whole number of
#                 QK_K super-blocks - the qwen4exp indexer key row, 128 - took the
#                 QK_K-only kernel and was rejected by the support predicate, so the
#                 HIP backend sent the gather to the CPU; 13 indexer layers per ubatch
#                 became host<->GPU round trips, ~25 % of long-context qwen4exp prefill
#                 for a `--cache-type-k iq4_nl` cache).  getrows.cu now dispatches on
#                 `ne00 % QK_K` (the sub-block path reuses `dequantize_q4_nl`), the
#                 predicate accepts any `ne00 % QK4_NL == 0`, and test-backend-ops
#                 covers iq4_nl get_rows at 32/128/160/224 columns.
#                 And the 2026-09-13 block-08 amendment (TODO item 19): the fused MoE
#                 router is now bit-identical to the generic softmax -> argsort ->
#                 get_rows -> norm chain, so the address-overlap guard that decides
#                 whether the fusion fires can no longer change the model output.
#                 topk-moe.cu reproduces the generic soft_max_f32 block_reduce
#                 reduction order (per-warp butterfly + cross-warp butterfly) and the
#                 reduce_rows_f32 sum_rows order, and divides by the (clamped) sum like
#                 the generic div; argsort.cu's bitonic network breaks ties by index
#                 (the CUB path is a stable radix sort and the fused router's
#                 iterative argmax also picks the smaller expert index), so the
#                 routed top-k no longer depends on the allocation plan.
#                 `2026-09-13 block-14 amendment (ninth)`: the re-base's new `mmq_args::ncols_opt`
#                 field was left unset (0) by block-14's hand-built `ggml_cuda_mul_mat_q_pair`
#                 args, so the MMQ tile heuristic picked the narrowest tile (J=8) - up to 2.2x
#                 slower dense prefill.  Both pair arms now set it like the standalone and the
#                 heuristic falls back to ncols_max when unset.
#                 `2026-09-13 block-01 + block-14 amendment (issue #30 clamp policy)`: the
#                 --spec-draft-n-max clamp is raised from 7 to 15 (the recurrent rollback
#                 snapshot bound: a verify batch decodes n_max + 1 = K <= 16 rows, which is the
#                 constant the K-independent chunked-GDN threshold was built around), with a
#                 visible purity notice above 7 instead (a verify wider than 8 rows switches
#                 kernel family - the FA tile/MMA chooser and the matmul MMVQ/MMVF -> MMQ switch -
#                 so plain and draft-mtp may no longer be bit-identical).  It also adds the
#                 `tests/test-recurrent-state-depth` snapshot sweep (n_rs_seq 1..15, the whole
#                 rollback range, including deep drafts) and block-14's QSA decode-arm band fix:
#                 the arm's band is now max(QSA_DECODE_BAND, cparams.n_rs_batch), so a verify of
#                 the full built-in draft depth takes the same dense arm as the W=1 decode instead
#                 of flipping to the sparse top-k arm above W=8 (the 2051-selection-width
#                 divergence reported for qwen4exp at depth 15).
#                 The block-15 tip of the *working*
#                 fork checkout (~/llama.cpp rdna-boosts) is a different SHA,
#                 because that branch is a local rebuild -- do not use it for
#                 regeneration unless it was rebuilt at the fork point.
#                 See MANIFESTS.md.)
#
# All 16 blocks are the fork commits baseline-sha..blocks-tip, exported with
# `git format-patch --start-number 0` (the canonical, verified form; applies
# with `git am`).  Block 00 is the structural/architecture-fix commit that
# sits directly on the baseline; the feature blocks 01-15 follow.  Every
# block is a committed fork commit - including block 12 (the hybrid HIP
# all-reduce), which was previously a working-tree delta.  Blocks 01-11
# keep their original subjects; 12, 13, 14 and 15 keep theirs too, so the
# 000N file naming is uniform across the set.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-790cf51aa}"
TIP="${3:-c45244c728dfcbcad86ae95aa97ae76f94ee9f7f}"
PATCHES="$REPO_DIR/patches"

if [ ! -e "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-5]-*.patch

# block 00 + blocks 01-15: format-patch (keeps the original subjects; applies with git am).
# --start-number 0 makes the first commit's file 0000-* so the file number matches
# the block number (block 00 -> 0000, block 01 -> 0001, ... block 15 -> 0015).
git format-patch --start-number 0 "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

echo "Regenerated $PATCHES:"
ls "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-5]-*.patch | wc -l
echo "patches (16 blocks: 00 + 01-15).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE."
