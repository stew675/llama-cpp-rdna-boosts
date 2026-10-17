# kv-cache-q8-write: fold the q8_0 K and V cache writes into the rope + FWHT kernel

Thank you for the issue #67 work that made the address-selected rope fusions round exactly like the chains they
replace; this patch keeps to that rule.  It does for a q8_0 cache with the KV rotation on what the existing rope ->
set_rows fusion does for f16 / f32 caches, in the rope kernel that already runs the FWHT (#172).  Bit-identical.

**Stacked:** one format-patch on top of `v12-097cf33d9-r71` + #172 (kv-rot-fwht-fusion, 0001..0002) + #171
(q8-decode-small-kernels, 0001..0002), applied in that order (applied tree `01067997`).  It needs #172's fused kernel
and reuses #171's q8_0 group quantize.  5 files (`rope.cu`, `rope.cuh`, `set-rows.cu`, `set-rows.cuh`,
`ggml-cuda-try-fuse.cu`), +188 / -34.  Not arch-specific; `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` turns it off.

**Hardware: RX 7900 XTX (gfx1100, RDNA3, 24 GB)**, Ryzen 7 9800X3D, Windows 11 + WSL2 (Ubuntu 24.04), ROCm 10.2 nightly
`20261013-38325106112` (HIP 7.17.26405) in a container.  All speed numbers below are from this box under WSL2, except the RDNA4 section (labelled there).

## What it does

With a q8_0 cache, each layer's K path ends `..., MUL_MAT(Hadamard), RESHAPE, VIEW, SET_ROWS(K)`, and the V write
`VIEW, SET_ROWS(V)` of the V row (rotated earlier) follows right after.  On that pattern the fused
`rms_norm_mul_rope_f32` (+ FWHT) writes both cache rows itself:

- K: warp 0 quantizes the rotated row straight from its FWHT registers into the K cache row.
- V: each block (one KV head) also quantizes its slice of the token's V row into the V cache row.

Both go through `ggml_cuda_quantize_q8_0_group`, which is `k_set_rows_q8_0_group`'s body moved into `set-rows.cuh`,
so the cache bytes are the same as the two SET_ROWS nodes write.  The two `set_rows` launches per layer go away; the
F32 rotation output of K is no longer written (its only reader was the K SET_ROWS).

Matcher: the #172 rope + FWHT match, then the RESHAPE / VIEW / SET_ROWS chain with single-use, non-output nodes, one
contiguous F32 row of `n_head_kv * head_dim` per token, I64 indices, q8_0 cache views; the V part needs the same row
shape.  The cache writes must not overlap anything the kernel reads.  Inside a concurrent-stream region
(`GGML_CUDA_GRAPH_OPT=1`) the V row comes from another stream, so only the K write is folded there.

## Tests

Qwen3-30B-A3B Q4_K_M (MoE, 48 layers) and Qwen3.8-27B Q4_K_M (dense), your README build (`-DGPU_TARGETS=gfx1100`), no
tuning env, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`.

**Kernel trace** (rocprofv3, one 400-token greedy MoE request): no `set_rows` launches remain; the K-path
`rms_norm_mul_rope_f32<…, 128>` with the cache writes takes 9.4 us per call (7.4 us without them), against two
`k_set_rows_q8_0_group` launches of 3.7 us each removed (about 5.4 us net per layer).

**Decode** (`llama-bench -p 0 -n 256 -r 10`, MoE, four alternated rounds over two builds of the same tree):

| | mean |
|---|---|
| r71 | 130.6 t/s |
| r71 + #172 + #171 + this, `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` | 136.8 t/s |
| r71 + #172 + #171 + this | 140.2 t/s (+2.4 % vs the switch, +7.4 % vs r71) |

Dense: 38.76 / 38.78 -> 38.63 / 38.85 t/s (flat, within noise).

**Correctness.**
- Greedy hashes, 400 tokens, identical to r71: MoE plain decode prose `0116af00d69e` / code `8fd18d2f0b83`; dense with
  MTP (draft-mtp, n-max 3) prose `8345c992d1fa` / code `47114faa2042`.  With `GGML_CUDA_GRAPH_OPT=1` (K-only fold):
  MoE hashes again identical to r71's.
- Your `test-logits-width-probe` (q8_0 KV, FA): PASS with worst maxdiff 0 at prefix 256 and 3584, on both models.
- `test-backend-ops -o SET_ROWS`, `-o ROPE`, `-o MUL_MAT`: pass (they don't build the fused pattern; the hashes and
  the probe do).

## RDNA4

**Hardware: 1 x Radeon AI PRO R9700 (gfx1201)**, native Linux, ROCm 10.0; same models, llama-server
`-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0`, the same stack (applied tree `01067997`).

- Greedy hashes identical to plain r71 on both models (MoE prose `744db56ea1c8` / code `6bb5966d061f`, dense prose
  `d12cdeede326` / code `e8a93c816237`), also with `GGML_CUDA_GRAPH_OPT=1`; width probe PASS (maxdiff 0) at 256 and
  3584; `test-backend-ops` SET_ROWS 159/159, ROPE 450/450, MUL_MAT 2444/2444.
- Trace (MoE decode): no `set_rows` launches remain; the K-path kernel with the cache writes takes 3.25 us per call
  (2.19 us with the switch set), against two `k_set_rows_q8_0_group` launches of 1.34 us each removed per layer
  (about 1.6 us net).
- MoE decode (two alternated rounds, consistent within 0.2 t/s): r71 132.7, the stack with
  `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS=1` 139.2, the stack 143.2 t/s (+2.9 % vs the switch, +7.9 % vs r71).  31.5k
  prefill within 0.5 %.

## Limitations

- Needs #172 and #171 first; happy to rebase it onto whichever release folds them.
- q8_0 caches with the rotation only; other cache types keep the existing rope -> set_rows fusion or the plain path.
- Measured on gfx1100 (WSL2) and gfx1201 (native Linux).

No urgency on our side.  Happy to re-run anything on your config.
