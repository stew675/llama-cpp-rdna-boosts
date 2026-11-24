# wip/gdn-norm-gate-once — the GDN norm-gate fusion matches once per GDN layer, not twice

Written by Claude (AI).  A follow-up to #357 (block 10's SILU widening in r353): host time only, bit-identical.  One `git am`
file made on `v17-919794500-r353` (tip `089f90067`, tree `27db9bfd4`); applied, the tree is `a00e6ef09`.

| file | touches | switches |
|---|---|---|
| `0001-CUDA-HIP-the-GDN-norm-gate-fusion-matches-once-per-G.patch` | `ggml-cuda-try-fuse.cu` (+44 −12) | none of its own (host-side, bit-identical); `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0` still turns the SILU fold off, and `GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1` still counts it |

## What it does

`ggml_cuda_gdn_norm_gate_match()` ends with three whole-graph reader counts: `ggml_cuda_count_uses` for the `RMS_NORM`
and for its `MUL`, and the gate `MUL`'s inside `ggml_cuda_find_mul_q8_1_matmul`.  It ran twice per GDN layer, at the
`RMS_NORM` (the skip) and again at the gate node, and the gate node then called `ggml_cuda_find_mul_q8_1_matmul` once more.
That made seven whole-graph walks per GDN layer where the unfused gate takes one.  Before #357 Qwen3.x stopped at the
SIGMOID check, so none of this ran.  It is paid on every `graph_compute` that runs the node loop: graphs off, the direct runs
and recaptures after a graph change, prefill, and multi-token verify.  A replayed graph pays nothing.

- The gate node takes the skip's decision.  The skip records the gate node (`gdn_norm_gate_at`), and the pending norm has
  exactly one reader, the gate's `MUL`, so no other node can match it.  The norm is the pending `MUL`'s `RMS_NORM` source;
  the matmul is the node after the gate `MUL`, or the one after its no-op reshape (the nodes `find_mul` accepted).
- The three counts are taken in one walk (`ggml_cuda_count_uses3`).  `ggml_cuda_find_mul_q8_1_matmul` takes an already
  counted value through a defaulted parameter; every other caller is unchanged.  The conditions are the same, so the
  decisions are.

## Measurements

Arms: **S** r353, **P** r353 + this commit, **P0** P with `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0` (the matcher stops at the
activation check, the same path as r353 with `=0`).  Greedy, one fresh process per run.

**Windows** (one R9700, Windows 10, Ryzen 7 5700G, TheRock ROCm 10.0, clean r353 and r353 + this commit, Qwen3.8-27B Q5_K_M):

- **Graphs off** (`GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GCDBG=1`, llama-bench tg128, `-ub 512 -b 1024 -fa 1`, order
  S P P0 P0 P S):

  | arm | `try_fuse` host / `graph_compute` | decode t/s |
  |---|---|---|
  | S | 14.4 / 14.3 ms | 26.82 / 27.06 |
  | P | 7.4 / 7.6 ms | 28.32 / 28.08 |
  | P0 | 8.0 / 7.6 ms | 27.96 / 28.14 |

  On this box the host cost shows in the speed: S is 4.0 % below P0, and P is 0.5 % above it.
- **The #359 deep case**.
  - Setup: llama-server, a 100 605-token prompt, 512 tokens with `ignore_eos`, no draft, f16 K/V,
    `-c 110592 -ub 512 -b 1024 -np 1 -t 6 --poll 0`, block 16 off.
  - Runs: a fresh server per visit, order S P P0 P0 P S. Each visit makes one cold request and two with `cache_prompt`.
  - Decode: S 22.140, P 22.155, P0 22.107 t/s; the text is identical in all 18 requests.
  - `GGML_CUDA_GRAPH_TRACE=1` reads direct 207, replay 1521, capture 7 in every arm.  The direct runs are almost all the 197
    prefill ubatches.  Decode replays the graph for nearly every step, so this change does not move the speed there.

**Linux** (one R9700, Ryzen 9 3900X, ROCm 10.0.0, `ROCBLAS_USE_HIPBLASLT=0`).  Both builds also carry PR #360's server change
and upstream's Inkling 2 Mini model support (neither touches ggml; they are there to run the Inkling model below).  S and P
differ only by this commit.

- **Host time**, graphs off as above, Qwen3.8-27B UD-Q5_K_M: `try_fuse` per call S 13.5 / 12.3 ms, P 7.1 / 6.8 ms, P0
  7.0 / 7.1 ms.  Decode with graphs on: S 28.87 / 28.88, P 28.88 / 28.88 t/s.  This box is GPU-bound, so the saving does not
  show in the speed.
- **Coverage** (S and P; text, fold census and graph counts compared):

  | case | setup | fused silu |
  |---|---|---|
  | concurrency | 27B UD-Q5_K_M, 4 slots `--kv-unified`, q8_0 K/V, ub 2048: four prompts in one request; eight requests at once (0 errors) | 1008 |
  | draft | 27B UD-Q4_K_XL + DFlash2 draft 5, 2 slots: two prompts in one request, then one alone (same draft counts) | 864 |
  | deep | the #359 case above (100 605 tokens) | 336 |
  | deep, 2 slots | two 40 000-token prompts in one request | 336 |
  | other models | Mistral-7B Q8_0, gpt-oss-20b MXFP4, phi-4 IQ4_XS, Qwen3-30B-A3B Q4_K_M, Inkling-2-Mini IQ4_NL | 0 (no GDN) |
  | | Qwen3.6-35B-A3B UD-Q4_K_M | 150 |
  | | Qwen3.8-27B MXFP4 | 240 |

  Identical in every case.  The eight concurrent requests' texts depend on their timing and are not compared.

## Not tested

RDNA3, NVIDIA (HIP build only), several GPUs, the #359 reporter's setup (Windows 11, HIP 10.2, UD-Q5_K_XL).

## Not in this change

Even with the fold off, `ggml_cuda_try_fuse` takes ~7 ms per `graph_compute` on these boxes (27B, ~890 calls), because
other matchers count readers over the whole graph too.  Reader counts taken once per graph would cut that further wherever
the node loop runs.  That is a change to the matcher design, so it is left out.
