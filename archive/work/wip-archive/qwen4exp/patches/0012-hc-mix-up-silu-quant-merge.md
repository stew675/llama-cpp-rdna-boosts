# 0012 - hc_mix decode: merge the silu+Q8_1 quantize into the up dot kernel

Applies: ggml-cuda/hc-mix.cu only.

What it does: the standalone v = silu(lo/hc) + Q8_1 quantize kernel
(~10.6us + its launch boundary) is folded into the up-dot kernel's
prologue: each of the 640 blocks re-quantizes all 10 v kblocks (values
identical across blocks, ~1us added), so the op runs one fewer kernel.
The per-kblock reduction mirrors quantize_q8_1 (one warp over 32
consecutive values) so y_v is bit-identical.

A companion merge of the xn quantize into the down dots was tried and
REVERTED inside this patch's development: each of the 320 down blocks
re-streamed all of xn from L2 (320x redundancy), slowing the kernel
43 -> 80us.

Result: direct per-call measurement 45 -> ~40us for the up stage; the
llama-bench A/B is noise-limited (+/-2.4 t/s). Commit b5f7fd598 (parent
b7635f6df).
