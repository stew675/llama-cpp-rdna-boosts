# wip/fa-kq-blocks-reuse — one FA KV-group bitmap per graph evaluation (TODO #348)

Written by Claude (AI).  Item 3 of #334, re-sent on its own as asked, with the two lifetime defects fixed and a census.  One
`git am` file made on `v17-4f9dfa0e3-r324` (tip `923b910eb`, tree `d3babb65a`).

| file | touches | switches |
|---|---|---|
| `0001-CUDA-HIP-FA-KV-group-bitmap-built-once-per-graph-eva.patch` | `common.cuh`, `fattn-common.cuh`, `fattn.cu`, `fattn.cuh`, `ggml-cuda.cu` (+147 −15) | `GGML_CUDA_FA_KQ_BLOCKS_REUSE=0` (kill switch), `GGML_CUDA_FA_KQ_BLOCKS_CENSUS=1` (census) |

## What it does

The first attention layer of a graph evaluation builds the packed KV-group bitmap (`flash_attn_mask_to_KV_blocks`) into a
buffer the CUDA context owns; a later layer of the same evaluation with the same mask tensor, mask data pointer, stream and
producer arguments reuses it without the memset and the prepass.  Anything else builds per layer into the pool as before:
another stream within the same evaluation, a bitmap above 16384 words, a first allocation while a graph is being captured, a
call outside a graph evaluation.

Reach, as you measured: a single-sequence batch takes the derived bitmap and never gets here; the packed form needs a
multi-sequence ubatch, or the RDNA4 band at depth (its own work floor, `K->ne[1]*saving >= 20000`).

## Changed from #334 after your review

1. **Lifetime.**  The process-global maps keyed by `const ggml_backend_cuda_context *` are gone.  The state is a member of
   `ggml_backend_cuda_context` (`fattn_kq`), the buffer is allocated on first use through `ggml_cuda_device_malloc(...,
   ctx.device)` like the context's other small buffers (`moe_red_cnt`, `hc_q8_cnt`), and `~ggml_backend_cuda_context` frees it
   (`ggml_cuda_fattn_kq_blocks_free`).  A context created later, at the same address or on another device, starts with no
   bitmap, so a stale buffer can no longer be handed to another device.  It is still outside `ggml_cuda_pool` (a fixed 64 KiB
   per context, like those buffers), so `--fit` does not see it.
2. **Census.**  `GGML_CUDA_FA_KQ_BLOCKS_CENSUS=1` logs, when the context is freed,
   `FA_KQ_BLOCKS_CENSUS dev=<d> reuse=on|off built=<n> reused=<n> fallback=<n>`.  It counts host-side decisions: a replayed graph
   repeats the decisions it was captured with and is not counted.  (`llama-bench` silences ggml's log unless `-v`.)
3. **The key.**  It is the producer's full argument list besides the mask (`n_kv`, `n_tps`, `s31`, `s33`, `n_tiles`,
   `n_streams`, `n_mask_streams`) plus its template parameter `ncols1`; `n_groups` follows from `n_kv` and `FATTN_KQ_STRIDE`.  A
   change to the band's tile or KV split that reaches the producer through these arguments changes the key with it; a change to
   what the producer computes from the same arguments must extend the key.  The comment in `fattn-common.cuh` says so, and it
   should be re-audited whenever the band's split changes (r319 / #330 and r322 / #333 did not change the producer's arguments
   or body).

## Measurements

One R9700 (gfx1201), Linux, ROCm 10.0.0, one binary = stock r324 + this commit + the #347 re-submission (graph replay to 16
tokens, inert here); `ROCBLAS_USE_HIPBLASLT=0`, `GGML_CUDA_FA_KQ_BLOCKS_CENSUS=1`, one fresh process per run.

**Your best-case shape**: Qwen3.8-27B UD-Q4_K_XL, `-np 2 --kv-unified -c 65536`, q8_0 K/V, `-fa on`, no draft; slot 0 parks a
25,043-token prompt (`n_predict 1`), then slot 1 decodes three 300-token greedy requests.  Arms A = `GGML_CUDA_FA_KQ_BLOCKS_REUSE=0`,
B = unset, order A B B A A B.

| arm | decode t/s (three requests per run) | census |
|---|---|---|
| A (off) | 31.64 31.75 31.70 · 31.65 31.67 31.66 · 31.65 31.70 31.69 | — |
| B (on) | 31.70 31.73 31.72 · 31.66 31.70 31.67 · 31.66 31.70 31.71 | built=1 reused=15 fallback=0 (each run) |

Mean 31.680 → 31.693 t/s (+0.04 %): nothing measurable on Linux, as you found.  The three texts hash the same in all six runs.

**Lifetime**: `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` with the reuse on: 7381/7381 passed, census `built=2260 reused=0
fallback=0` at exit (single-node graphs, so every build is a first build).  `llama-bench -v -m <27B> -m <4B> -p 512 -n 16
-d 0,8192 -r 1` (two models in one process, a fresh context for each of the eight tests): exit code 0 and eight census lines,
one per context as it is freed (`built=0`: these single-sequence tests take the derived bitmap, so the buffer is never
allocated there).  Every server run of the shape above allocated the buffer and freed it at exit, the census line coming from
the destructor.

## Where it could go

There is no measured win here.  The remaining argument is the launch count (15 memset + prepass pairs fewer per step in the
shape above), which matters where launches are expensive, i.e. Windows / WDDM; we have no Windows measurement.  So, as you
suggested, this fits `optional/fa-kq-blocks-reuse/` rather than the default set, unless a Windows run shows a win.

## Not tested

Windows; RDNA3 / 3.5; NVIDIA; several GPUs (the cross-device case is fixed by construction, not exercised); `--fit`.
