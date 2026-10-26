# graph-replay-keys

One patch against release v15-6050b467c-r144 (tree b8d08c51; applied tree 46891711):

- `0001-ggml-cuda-graph-cache-keys-that-tell-per-layer-split.patch` (git format-patch, 3 files, +92/-4: ggml-cuda.cu, common.cuh, moe-expert-cache.cu)

## What it does

1. The CUDA graph cache is also keyed on the first device-resident weight the subgraph reads. If weights are host-resident, it uses nodes[0]->src[0]. A cheap shape/name signature is added. Layers, and the two expert-width splits of one layer, get their own entries. The key only picks the entry. Replay still needs every node to compare equal.
2. Node properties are compared without src[]/view_src/buffer object identities. op, data, ne, nb, type, op_params, flags, name and every source's data/ne/nb/type are still compared.
3. ggml_cuda_graph_mem_freed is bumped when the expert cache frees an arena or remap buffer, so a capture that baked one in is never replayed.

Why: our reading (not certain) is that cause 3 in wip/graph-replay-placement was a cache-key collision on our box. All layers start on the same activation buffer, so the old key sent them to one cached graph.

Kill switches:

- `GGML_CUDA_GRAPH_KEY_WEIGHT=0`: old key
- `GGML_CUDA_GRAPH_PROPS_STRICT=1`: old compare

## Setup

- 2x R9700 (gfx1201), native Linux, second slot Gen4-limited.
- Arms: stock r144 vs r144 + patch. Only the patch differs. Stock default env otherwise.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (qwen4exp) + MTP Q4_0 draft head (--spec-draft-n-max 3), -sm tensor --n-cpu-moe 47, default host-expert mode. Not the note's IQ4_NL.
- 1 GPU control: Qwen3.6-35B-A3B Q4_K_M, no draft.
- Decode: 400-token greedy /completion (temp 0, top_k 1), one prose and one code prompt, after a warm-up. One server per arm, pair order alternated. Not the note's 2000-token arms.
- Census: GGML_CUDA_GRAPH_TRACE=1 over the whole server run.

## Census

| | direct | replay | capture | properties_changed |
|---|---|---|---|---|
| stock | 100155 | 25734 | 225 | 61617 |
| patched | 22286 | 103502 | 2192 | 242 |

Patched direct is mostly prefill and warm-up (multi-token graphs skip the graph path). In steady decode on our own build: 0 direct calls.

## Decode, t/s

| Config | Prompt | Stock | Patched | Change |
|---|---|---|---|---|
| -sm tensor, 4 pairs | prose | 93.9 | 98.8 | +5.15 % [95 % CI +1.71, +8.59] |
| -sm tensor, 4 pairs | code | 102.7 | 108.7 | +5.82 % [+3.58, +8.07] |
| -sm layer -ts 59,41, 2 pairs | prose | 83.4 / 83.6 | 86.3 / 85.5 | |
| -sm layer -ts 59,41, 2 pairs | code | 95.1 / 94.6 | 98.1 / 98.3 | |
| 1 GPU, 2 pairs | prose | 124.5 / 125.3 | 125.0 / 125.0 | flat |
| 1 GPU, 2 pairs | code | 125.4 / 125.4 | 125.1 / 125.0 | flat |

## Controls

- Stock + GGML_CUDA_DISABLE_GRAPHS=1: 93.6 / 99.7 t/s (prose / code), same as stock with graphs on.
- Patched + both kill switches: 93.2 / 101.8 t/s, properties_changed 108233. The switches restore stock behaviour.

## Gates

- Greedy text hashes identical, stock vs patched, 0 faults:
  - tensor: 65f85498c191 / 645768148473
  - layer: 84f575149b88 / 63c224728083
  - 1 GPU: 16f78131e888 / c8105294d9c2
- test-backend-ops -o MUL_MAT_ID: OK.
- Not run: scripts/gate-prefill-logits.sh KLD, ROCBLAS_USE_HIPBLASLT=0 deterministic arm, arch sweep, soak.

## Caveat

Fusion checks look at source object identity and use counts, which the relaxed compare no longer sees. We found no case where the Meta rebuild swaps which object a consumer references within a split.

## Notes

One hunk touching our local device-remap code was dropped for this port. The same code has run in our own build since 2026-10-25 (+1.6 % decode there).
