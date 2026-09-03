# 0011 - hc_mix decode kernels: collapse in coalesced blocks, silu_quant per-block grid

Applies: ggml-cuda/hc-mix.cu only.

What it does: the stream-collapse kernel launched one 1-thread block per
output element (2560 blocks, no coalescing, heavy block-scheduling) and
the silu+Q8_1 quantize serialized all 10 Q8_1 blocks in one 32-thread
block. Both are pure thread-organization changes: collapse now runs
(256)-thread blocks over the output elements (stream reads coalesce),
silu_quant gets one 32-thread block per Q8_1 block. Per-element
arithmetic and rounding order are untouched, so decode stays bit-exact.

Result: tg128 43.91 vs 43.05 (+2%). Commit b7635f6df (parent 6a4e2c766).
