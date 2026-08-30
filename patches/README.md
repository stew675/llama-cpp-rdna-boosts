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

The set is **whitespace-clean**: applying produces no git whitespace
warnings (verified 2026-08-29 after the whitespace-clean regeneration).

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
  - `GGML_CUDA_AR_SPIN_TIMEOUT_MS` — bounded in-kernel peer-arrival spin
    budget (default 20 ms, `0` = legacy unbounded); on timeout the kernel
    sets a host-mapped poison flag, skips the reduce and exits, and the host
    re-syncs the devices via a butterfly AllReduce on the next call
  - WIP experiments (archived, env-gated OFF by default): `GGML_CUDA_AR_FUSED`,
    `GGML_CUDA_AR_PACE` — see `../archive/work/fused-stage-pacing/README.md`
- Verified 2026-08-29 (3x R9700, ROCm 7.14, gfx1201): clean apply + full
  build + llama-cli same-seed coherence IDENTICAL + tg64 38.12 / tg512 41.08
  (matches the fork build).  Depth-16384 decode 38.71 t/s (3-GPU hybrid,
  unpinned) with the server config `HIP_VISIBLE_DEVICES=0,1,2`.
- **Community-report fix round (2026-08-30, issues #5 + #6, reporter
  tungel):** two block-12 fixes integrated into the fork and regenerated
  into this set:
  - `-DGGML_HIP_RCCL=OFF` builds now compile — `comm_init_hybrid`'s
    `try_allreduce_nccl` reference is guarded by `GGML_USE_NCCL` (was an
    unconditional reference to an `#ifdef`-guarded function: build error).
  - The chunked AR kernel's in-kernel peer-arrival spin is now bounded
    (`GGML_CUDA_AR_SPIN_TIMEOUT_MS`, default 20 ms, `0` = legacy).  An
    unbounded spin on RDNA (non-preemptible compute kernels) could wedge
    the queue -> MES `REMOVE_QUEUE` timeout -> MODE1 reset -> `700/719`
    or a whole-machine freeze; on timeout the kernel poisons a host-mapped
    flag, skips the reduce and exits (queue stays removable), and the host
    re-syncs the devices with a butterfly AllReduce on the next call.
    The budget check is decimated to 1-in-512 polls (2x the measured
    typical spin count — p50 184 / p90 283 / p99 369 polls on 2x gfx1201
    hybrid at depth-16384 — rounded up to a power of two), so the true
    fast path never executes `clock64()` at all and the timeout overshoot
    stays <0.1% of budget.
  - Re-verified 2026-08-30: clean-apply sim at `17252c769` (apply, full
    build, llama-cli same-seed coherence IDENTICAL); RCCL=OFF `ggml-hip`
    compiles; before/after perf on the default hybrid config (2x R9700,
    depth-16384) shows NO measurable impact — pp512 1622.7 -> 1609.1 t/s
    (-0.8%, within noise), tg128 32.63 -> 32.55 t/s (-0.25%, within
    noise).  Note: regenerated from the current `~/llama.cpp` fork
    (rdna-boosts tip `8a426cf79`); commit hashes in the 01-11 patch
    headers drift from the earlier records (the fork was rebuilt; diff
    content is unchanged).
- **Compiler-warning clean** (2026-08-29 follow-up): ROCm 7.14 marks
  `hipError_t` `[[nodiscard]]`, and the original HIP port left 27
  unchecked HIP calls (all `-Wunused-value` in the ggml-hip build).  All
  27 now go through `CUDA_CHECK` (upstream house style, incl. teardown);
  three dead WIP items removed.  The ggml-hip build emits ZERO warnings
  from this patch.

## Server config (the +22% deployment win)

`HIP_VISIBLE_DEVICES=0,1,2` (3-GPU), hybrid default, **unpinned** (the
dpm=high/runtime-PM pin is a regression: tg -5-7%, pp -15-18% on RCCL/hybrid
paths).  Depth-16384: 31.79 (2-GPU) -> 38.71 (3-GPU) t/s.
