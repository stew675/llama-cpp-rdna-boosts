# pool-cache-cap: an opt-in cap on the bytes the legacy CUDA/HIP pool keeps cached

One format-patch on top of `v16-a55e952b8-r39`, applied with `git am`.  Opt-in: with `GGML_CUDA_POOL_CACHE_MB`
unset (or 0) the pool behaves exactly as today.

## What it addresses

On a long prefill the pool's temporaries grow with the KV length (flash-attention staging and friends).  Each new,
larger request leaves its smaller predecessors in `buffer_pool`, so a 256K prompt ends with a stack of freed buffers
that nothing will use again.  That VRAM is invisible to everything that allocates outside the pool (compute-buffer
re-reserves, the slab, FA staging), so the cards run within a few MiB of full at the end of the prompt.

With `GGML_CUDA_POOL_CACHE_MB=N`, `ggml_cuda_pool_leg` tracks the bytes of the freed buffers it holds and, before a
new device allocation, frees them once they exceed N MiB.  Nothing changes while the cache stays under the cap, and the buffers that are
reused come straight back on the next ubatch, so decode and normal prefill do not notice it.

## Measurements (2 x R9700 32 GB, Flash-Next GSQ-RCO IQ3_XXS + MTP, `-sm tensor`, all experts in VRAM, 256K context)

VRAM peak per card while a 259,620-token prompt is processed (sampled every 2 ms):

| build | card 1 peak | card 2 peak | of 32,624 MiB |
|---|---:|---:|---|
| stock r39 | 32,551 MiB | 32,609 MiB | 15-73 MiB left |
| r39 + this patch, `GGML_CUDA_POOL_CACHE_MB=128` | 30,945 MiB | 31,414 MiB | 1,210-1,679 MiB left |

Greedy output (sha `45e68f1bdfeb`) is the same with and without the cap.  Speed: in our older builds the cap at
1024 MiB vs off gave the same decode and the same 259.6K prefill within noise, 128 vs 256 MiB measured the same as
each other, and a 12-round soak (mixed decode + 2 concurrent requests) at 256 MiB kept VRAM flat to +-15 MiB.  We
have run it at 128 MiB in our own builds since r26.

## Notes

- Only the legacy pool (`ggml_cuda_pool_leg`) is touched; the VMM pool is unchanged.  HIP builds use the legacy
  pool by default (`GGML_HIP_NO_VMM` is ON), so this applies to ROCm out of the box; CUDA builds use the VMM pool by
  default and are unaffected unless built with `GGML_CUDA_NO_VMM` or run on devices without VMM support.
- The flush is skipped while a CUDA/HIP graph is being captured (the `cudaDeviceSynchronize` would be illegal there,
  and the capture may already reference the cached buffers); the next allocation outside capture flushes instead.  Like the existing out-of-memory flush, a flush invalidates
  the captured graphs on that device; they are recaptured once on the next decode step.
- Unset, 0, negative or non-numeric values all mean "no cap".
- It frees the whole cache rather than evicting largest-first: simpler, and in our runs the buffers that are reused
  come straight back on the next ubatch.  Happy to change that, or the name (`_MIB` like the other knobs), if you
  prefer.
- 128 MiB was our pick for this model: small enough to keep the headroom, large enough that the usual decode /
  verify temporaries stay cached.  A smaller cap only frees more often.
