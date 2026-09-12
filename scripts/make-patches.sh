#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline the patches are generated against
#                 (default: 9113cc188, see MANIFESTS.md)
#   blocks-tip    the fork commit carrying block 00 + all 14 feature
#                 blocks (default: c6f1e8e78cfb2a70958998cdd81fad363e869f93,
#                 the block-14 commit of the
#                 CANONICAL fork rebuilt at 9113cc188, after the 2026-09-11
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
#                 embeddings_nextn export gets a separate full-row tail for t_h_nextn).
#                 The block-15
#                 (attention-memory campaign) work is NOT part of the
#                 delivery; it is staged in beta/block-15-campaign-wins/
#                 and applied manually.  The block-14 tip of the *working*
#                 fork checkout (~/llama.cpp rdna-boosts) is a different SHA,
#                 because that branch is a local rebuild -- do not use it for
#                 regeneration unless it was rebuilt at the fork point.
#                 See MANIFESTS.md.)
#
# All 15 blocks are the fork commits baseline-sha..blocks-tip, exported with
# `git format-patch --start-number 0` (the canonical, verified form; applies
# with `git am`).  Block 00 is the structural/architecture-fix commit that
# sits directly on the baseline; the feature blocks 01-14 follow.  Every
# block is a committed fork commit - including block 12 (the hybrid HIP
# all-reduce), which was previously a working-tree delta.  Blocks 01-11
# keep their original subjects; 12, 13 and 14 keep theirs too, so the
# 000N file naming is uniform across the set.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-9113cc188}"
TIP="${3:-c6f1e8e78cfb2a70958998cdd81fad363e869f93}"
PATCHES="$REPO_DIR/patches"

if [ ! -e "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-4]-*.patch

# block 00 + blocks 01-14: format-patch (keeps the original subjects; applies with git am).
# --start-number 0 makes the first commit's file 0000-* so the file number matches
# the block number (block 00 -> 0000, block 01 -> 0001, ... block 14 -> 0014).
git format-patch --start-number 0 "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

echo "Regenerated $PATCHES:"
ls "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-4]-*.patch | wc -l
echo "patches (15 blocks: 00 + 01-14).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE."
