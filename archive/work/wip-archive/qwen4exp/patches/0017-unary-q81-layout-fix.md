# Patch 0017: fix multi-seq decode corruption (unary-mul-q8_1 fusion quantize layout)

llama.cpp commit 9ed77c905, on top of 131935dce (the QSA multi-seq assert fix).
Apply: `git apply patches/0017-unary-q81-layout-fix.patch`

## The bug

Multi-seq decode (n_seq_max > 1, any parallel decode) corrupted every seq >= 1
from the FIRST decoded token. All single-seq decode (the entire decode tuning
campaign) was unaffected, so the corruption predated the batch-decode work and
was unreachable until 131935dce unblocked multi-seq graphs.

Root cause: `unary_gated_q8_1_op_kernel` (ggml/src/ggml-cuda/unary.cu) is the
fusion that computes the z-gated norm product of the GDN (sigmoid(z) x
rms_norm_out) and writes the Q8_1 quantized product into the per-graph
quantize arena, which the downstream ssm_out mmvq matmul finds via the q8_1
cache and consumes instead of re-quantizing. The kernel wrote the blocks in
the FLAT element layout (`ib = i/32` over the whole product). The mmvq
launcher reads the PADDED PER-ROW layout produced by quantize_row_q8_1_cuda:
each row (ne00 = 1920) occupies GGML_PAD(ne00, MATRIX_ROW_PADDING)/32 = 64
Q8_1 blocks, so channel (seq) c starts at block c*64, not c*60.

Consequences of the mismatch:
- seq 0 (channel 0) reads blocks 0-59 = correct (the first 60 flat blocks).
- seq >= 1 (channel c) reads blocks c*64..c*64+59 = the previous channel's
  tail plus garbage, shifted by the padding = corrupt from the first token.
- ne12 == 1 (single-seq) has no second channel, so the flat and padded
  layouts coincide: the single-seq decode stayed byte-identical. The (2,2)
  prefill never fused (the fusion requires ncols_dst == 1), so only the
  (n_seq_tokens=1, n_seqs>=2) decode geometry hit the bug - exactly the
  geometry observed as failing across the whole debug campaign.

## The fix

The kernel takes the matmul's ne00 (`row_len = mm->src[1]->ne[0]`, the flat
length of one channel row) and writes `ib = (i/row_len)*blocks_per_row +
(i%row_len)/32` with `blocks_per_row = GGML_PAD(row_len, 512)/32`, matching
the quantize_row_q8_1_cuda layout. The mmvq dot reads only the unpadded
row_len/32 blocks of each row, so the pad blocks may stay unwritten.

## Verification

- Coherence (bseq_val, K=2/K=3, same + different + unequal-length prompts):
  every seq now matches its K=1 single-seq reference token-for-token
  (e.g. K=2 "Quantum": seq0 = seq1 = 198 42750 367; K=2 "Quantum" + "The
  answer": 198.../369... both == their K=1 runs; K=3 likewise).
- Single-seq decode logitdump A/B (clean 131935dce vs the fix, same 16-pos
  seeded stream): BYTE-IDENTICAL (the fix cannot change ne12==1 numerics).
- Throughput (bseq, pp20, 3 GPUs): K=1 ~45 t/s, K=8 62.8 ms/step, K=16 114,
  K=32 138, K=64 32.85-33.19 ms/step = 1928-1948 tok/s AGGREGATE (reproduced
  twice) - the session-8 prize, now correct.
