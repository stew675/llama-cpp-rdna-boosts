# Strix Halo (RDNA3.5 / gfx1151) — prefill ROOT CAUSE FOUND + FIXED: the PLE gather ran as a CPU graph get_rows over a 28.8 GB mmap

Date: 2026-09-11. Follows the investigation record (2026-09-10-prefill-rootcause-investigation).
This closes item 1 of the plan: A's CPU input-embedding cost is root-caused and fixed at the
model level (commit `8b62ac25a`, on `b31940a5e`).

## Root cause (measured, not hypothesized)

- A's per-layer token embedding table `per_layer_token_embd.weight` [160 x 320,001,536] IQ4_NL =
  28.8 GB. Both token_embd and the PLE are classified LLM_TENSOR_LAYER_INPUT, and the generic
  loader pins the input layer to the CPU ("very little benefit to offloading the input layer"),
  so the tables are host-mmap resident even at -ngl 99.
- A's PLE path built `rows` + a graph `get_rows` node over that host table. GET_ROWS on the CPU
  backend is single-threaded (n_tasks = 1). The 2048-token ubatch needs 16 heads x 2048 =
  32768 random 90-B rows; each random 4 KB mmap page faults one at a time (~120-170 us/row from
  NVMe; first row of each decode 3-35 ms) => 2.7 s/decode. Instrumented proof (reverted):
  [cpunode] node_1 GET_ROWS src0=per_layer_token_embd.weight 2697 ms; q8_0 token_embd rows were
  1-2 us (cached region).
- The graph get_rows also cut the graph into CPU+GPU splits (extra launch + sync per ubatch).
- llama-cli paid it too (its earlier fast pp2048 was page-cache luck); B avoids it structurally.

## Fix (port of B's proven third path)

`LLAMA_QSA_PLE_HOSTGATHER` DEFAULT ON (env =0 restores the old graph path): when the table's
buffer is host (and no managed lazy reader), `build_inp_ple` feeds an F32 graph input
[ple_head_dim*n_heads, n_tokens] (no rows tensor, no get_rows node) and `set_input` dequantizes
the rows in-place (same ggml traits to_float as the CPU get_rows, so numerics identical) after
batching every distinct page into one madvise(MADV_WILLNEED) sweep (B's pattern: page faults
~0.2 ms each become parallel disk reads). Byte-identical output verified (cli same seed/prompt,
host-gather vs LLAMA_QSA_PLE_HOSTGATHER=0). B has exactly this behavior for host tables.

## Same-session depth-0 results (r3, A=8b62ac25a vs B c7af5c6c2)

| row | A before | A now | B | A/B now | A/B before |
|---|---|---|---|---|---|
| pp16384 | 574 | 625.85 | 599.73 | **1.04x A wins** | 1.04x (behind) |
| pp8192  | 544 | 637.03 | 679.02 | 0.94x | 1.25x |
| pp4096  | 488 | 645.81 | 734.18 | 0.88x | 1.39x |
| pp2048  | 400 | 654.61 | 775.53 | 0.84x | **1.93x** |
| pp1024  | 416 | 643.15 | 730.71 | 0.88x | 1.76x |
| pp512   | 409 | 590.92 | 645.39 | 0.92x | 1.58x |
| tg128@0 | 25.07 | 25.95 | 26.01 | 0.998x parity | 0.965x |

Every depth-0 data point improved; pp16384 now beats B and tg is at parity. A's rate curve is
now nearly flat 626-655 over pp1024-16384 (both peak at pp2048). Why pp512/1024 still lag more
than deep rows: fixed per-ubatch host+submit+GPU overheads dominate when GPU work per ubatch is
small; and the kernel-level deltas below (~0.5 s/pass GPU at pp2048) remain.

## Remaining pp gap components (next targets, ~0.85-0.92x at pp512-8192)

1. concat: A `concat_non_cont` 0.41 s vs B `concat_transposed_src1_dim0` 0.13 s per pp2048 pass
   (3x) - port B's specialized transposed-src1 concat kernel.
2. swiglu-input quantize: B fuses `quantize_mmq_q8_1_swiglu`; A runs separate swiglu+quantize
   (A quantize 0.50 s/1608 vs B 0.29 s/1326 + fused 0.105 s).
3. Q8_0 gate/up mmq J128: A 1.56 s vs B 1.38 s on the same 1182 calls (+13%) - feeding differs.
4. A-only mm_ids_helper<10> 423 calls / 0.31 s; host per-ubatch submit (re-measure now that the
   CPU split is gone for the PLE).
   Near-parity already: routed-compact expert mmq, rocBLAS GEMMs, hc fusions, dense FA.

## Also verified

- Old graph path still reachable (LLAMA_QSA_PLE_HOSTGATHER=0), byte-identical => A/B gates and
  text checks unaffected. Depth 12k/32k and tg rows should also gain (PLE gather is per ubatch
  everywhere; cli-path tg ~25.6 vs ~24.4 pre-fix) - re-derive in the next session's full matrix.

Raw: /tmp/gateA/ladder-{A-hg,B}.txt, tg-{A,B}.txt, plhg-{1,0}.txt (coherence), prior
bnd/cpunode/rowdbg instrumentation logs in /tmp/gateA.
