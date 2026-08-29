# rdna-boosts patch set (delivery)

12 patches against the llama.cpp fork point `17252c769`
("metal : add remaining fa-vec tunings for M4 Pro (#27915)"):

| patch | content |
|---|---|
| `0001` | adaptive MTP draft depth |
| `0002` | fused chunked gated-delta-net prefill kernel |
| `0003` | BF16 KV cache + native-BF16 flash-attn |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL |
| `0012` | **hybrid HIP all-reduce (block 12)** — the custom internal AR; hybrid dispatch; RDNA4-only gate |

## Apply (fresh checkout at the fork point)

```bash
git checkout 17252c769          # or: git apply each patch on a matching tree
git am patches/000[1-9]-*.patch patches/001[01]-*.patch
git apply patches/12-hybrid-allreduce-hip.patch
```

(`git am` for the 1-11 series — plain `git apply` of the concatenated series
was observed to silently drop hunks; use `git am`.)

## Block 12 notes

- **RDNA4-only gate**: the internal all-reduce refuses to init on any
  architecture other than gfx1200/gfx1201 (the pipeline falls back to the
  default RCCL path with a warning).  Community verification on RDNA3 pairs
  is pending; remove the gate's arch check once verified.
- Env knobs (defaults preserve upstream behavior):
  - `GGML_CUDA_ALLREDUCE=hybrid|nccl|internal|none` (hybrid = default on Linux)
  - `GGML_CUDA_AR_PROFILE=1` — per-call spin/phase profiler at teardown
  - `GGML_CUDA_AR_SLEEP=0|1` — s_sleep poll vs dummy spin (default 1)
  - `GGML_CUDA_AR_BF16_THRESHOLD` — F32->BF16 wire round-trip threshold
  - `GGML_CUDA_AR_COPY_THRESHOLD` / `GGML_CUDA_AR_COPY_CHUNK_BYTES` — CE path
  - WIP experiments (archived, env-gated OFF by default): `GGML_CUDA_AR_FUSED`,
    `GGML_CUDA_AR_PACE` — see `../archive/work/fused-stage-pacing/README.md`
- Verified 2026-08-29 (3x R9700, ROCm 7.14, gfx1201): clean apply + full
  build + llama-cli same-seed coherence IDENTICAL + tg64 38.12 / tg512 41.08
  (matches the fork build).  Depth-16384 decode 38.71 t/s (3-GPU hybrid,
  unpinned) with the server config `HIP_VISIBLE_DEVICES=0,1,2`.

## Server config (the +22% deployment win)

`HIP_VISIBLE_DEVICES=0,1,2` (3-GPU), hybrid default, **unpinned** (the
dpm=high/runtime-PM pin is a regression: tg -5-7%, pp -15-18% on RCCL/hybrid
paths).  Depth-16384: 31.79 (2-GPU) -> 38.71 (3-GPU) t/s.
