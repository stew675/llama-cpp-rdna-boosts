# gdn-state-gather-skip: the sequential GDN kernel reads its initial state straight from the recurrent cache

One format-patch on top of `v16-a55e952b8-r37`, applied with `git am` (also applies to r36).  Switch:
`GGML_CUDA_FUSE_GDN_STATE_GATHER=0` restores the gather.  Bit-identical.

## What

`build_rs` gathers each GDN layer's recurrent state with a `GET_ROWS` (393,216 floats = 1.5 MiB per GPU under
`-sm tensor` on Flash-Next) only for `GATED_DELTA_NET` to read it once.  In a decode trace that gather is one
`k_get_rows_float_vec` per GDN call, 4.2 us against the GDN kernel's 8.2 us, about 0.8 % of GPU busy time.

In the decode/verify band (one sequence, <= 16 tokens, so the sequential kernel) the CUDA graph loop now skips that
`GET_ROWS` and records it; the GDN impl sees the recorded gather on its state input and passes the row ids and row
stride, and the kernel loads `s0` from cache row `s_ids[seq]` instead of the gathered copy.  The gather is a plain copy,
so the values are identical.  Every GDN dispatch path (the beta-sigmoid fusion, the cache-cpy fusion, the plain op)
goes through the same impl, so they all honour it.

The skip requires: the GDN in the same graph; every other reader of the gathered rows empty (the zero-row extra-states
copy of a single sequence); no node in between writing the cache tensor; and row ids that outlive the gather.

## Two small changes outside the CUDA backend

The last condition needed them, and they are the part worth a careful look:

- `src/llama-graph.cpp`: `rs_s_copy` is also flagged as an output, so the allocator never reuses it.
- `ggml/src/ggml-backend.cpp`: when the scheduler copies a split input to another backend with `n_copies == 1`, the copy
  keeps the source's `OUTPUT` flag (with `n_copies > 1` it is already set).

Without them, the per-device copy of `rs_s_copy` lived in the compute buffer and its last reader was the last layer's
gather; once that gather was skipped, the allocator reused the ids buffer before that layer's GDN read it, and the
kernel read garbage row ids (aperture violation on both GPUs on the first request).  The deferral check requires the
flag, so on any graph where it does not reach the ids the gather simply runs as before.

## Results on r37 (2 x R9700 / gfx1201, PCIe 5.0 x8 each, ROCm 10.0, Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head)

One build (r37 + this patch), switch 0 vs 1, one run each; `--spec-draft-n-max 3 --spec-draft-p-min 0`, q8_0 KV,
`-ub 2048 -b 2048`.

| config | metric | off | on |
|---|---|---|---|
| `-sm tensor`, all experts in VRAM, 256K | greedy 400-token decode (2 reps) | 125.5 / 124.1 t/s | 127.0 / 125.9 t/s |
| | prefill 1.8k / 37k (2 reps) / 155k | 1996 / 2562, 2531 / 2051 t/s | 1790 / 2548, 2582 / 2082 t/s |
| | greedy sha | `45e68f1bdfeb` | `45e68f1bdfeb` |
| `-sm layer -ts 59,41`, all experts in VRAM, 128K | greedy (2 reps) | 102.5 / 102.8 t/s | 103.9 / 104.0 t/s |
| | greedy sha | `f2471f845142` | `f2471f845142` |
| `-sm tensor --n-cpu-moe 48` (pinned host experts), 256K | greedy, 2nd request | 108.2 t/s | 108.3 t/s |
| | greedy sha | `45e68f1bdfeb` | `45e68f1bdfeb` |

So about +1.3 % decode with all experts in VRAM under either split, and no change with host experts (the expert
traffic dominates there).  Prefill is unchanged: a prefill chunk keeps the gather (the 1.8k first-row difference is the
usual noise of the first prompt after a warm-up).

`test-backend-ops -o GATED_DELTA_NET`: OK on the patched build.  0 GPU faults in every run.  Earlier trace on r20
(same patch): gather kernels 9,576 -> 504 per 400-token greedy request.

## Not measured

RDNA3 / NVIDIA; `n_seqs > 1` (deliberately excluded by the matcher); 3-GPU layouts.
