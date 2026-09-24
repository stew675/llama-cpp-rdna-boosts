# Phase-1 item 2 — depthwise conv1d (GDN + PLE) direct fusions (2026-09-21)

**Status:** DONE and gated (WIP).  Fork branch `gap-closing` @ **`1004c65db`**, exported as
[`patches/0004`](patches/0004-gap-closing-WIP-port-the-depthwise-conv1d-fusions-GD.patch).  Default
**ON**; `GGML_CUDA_DISABLE_CONV_FUSION=1` is the A/B / bisect kill switch.

## What was ported

The other solution's `ggml/src/ggml-cuda/gdn-conv.{cu,cuh}` (111 lines) and `ple-conv.{cu,cuh}`
(182 lines), verbatim, plus the graph-optimizer call sites in `ggml_cuda_try_fuse` and the
allocation-dependency entries in `ggml_backend_cuda_graph_optimize` (`ggml-cuda.cu`).

* **GDN** (`gdn_conv_direct_kernel`): replaces `CONCAT(state, transpose(x))` + `GGML_OP_SSM_CONV` +
  `silu` with one kernel that reads `state` + `x` and writes the conv+silu output.  The CONCAT is
  still materialized **only for the recurrent snapshot copies** (its tail, via
  `gdn_concat_tail`), which is what the existing `GGML_OP_SSM_CONV`+`silu` fusion already did.
* **PLE** (`ple_conv_kernel<4,3>`): replaces the graph-level 4-tap `mul`/`add`/`silu` chain (the
  dilated depthwise conv built by `build_ple`) with the same direct kernel.  The kernel forces the
  per-tap multiply and add as separate `v_mul_f32`/`v_add_f32` rounds (`ple_mul_rn`/`ple_add_rn`) so
  it reproduces the graph's unfused MUL+ADD rounding **bit for bit** — this is not FMA.

Both matchers are prefill-only (`T >= 256`, `C % 256 == 0`) and require a single sequence
(`cc->ne[2] == 1`); the QSA/HC decode path is untouched.

## The two adaptations our tree needed

1. **`ple_conv_check` C/T derivation.**  The reference's `grouped_norm` emits a 2-D `[hc_dim, T]`
   MUL; after the 2026-09-15 re-base (`41abbfd59`), our `grouped_norm` emits a 3-D
   `[n_embd, hc, T]` MUL.  Both are the same flat `[hc_dim, T]` F32 buffer, but the transpose's
   `view_src` (the view chain collapses to the MUL root) has `ne[2] = hc`, so the reference's
   `x->ne[2] == 1` test rejected every PLE layer.  The check now derives `C`/`T` from the transpose
   and only requires the root to be a contiguous F32 buffer of `C*T` elements.
2. **`gdn_conv_check` snapshot cpy.**  qwen4exp's `build_conv_state_at` writes the snapshot as
   `cpy(cont(view(concat)))`, which the reference accepts.  The shared `build_conv_state`
   (qwen35moe / qwen35 / qwen3next) writes `cpy(view(concat), dst)` with a raw view, and the sweep's
   "unknown consumer of the concat" check rejected it.  The matcher now accepts a `GGML_OP_CPY`
   whose source is a 3-column view of the concat (already covered by `tail_from`), and rejects any
   other cpy that reads the concat.

Both are noted in the patch body.  Everything else (kernel bodies, `gdn_conv_write_tail`,
`ple_conv_check` tap/weight tracing, the `ggml_can_fuse_subgraph_ext` gate) is the reference code.

## Gates

**Bit-identical**: `test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512` row-0
logits hash is **identical** fused vs `GGML_CUDA_DISABLE_CONV_FUSION=1`, and `width_purity=PASS
(worst maxdiff 0)` on both:

| model | fused row0 hash | unfused row0 hash |
|---|---|---|
| qwen4exp IQ4_NL | `268e0673300b7a33` | `268e0673300b7a33` |
| qwen35moe 35B-A3B Q4_K_M | `e97c9e304ce1ca8f` | `e97c9e304ce1ca8f` |

Same-seed greedy text is identical on qwen4exp IQ4_NL (`bb820cccf620`, 637 chars, `-n 24`).

## Performance (gfx1151, `-b/-ub 8192`, `-ctk/-ctv f16`, `-n 0 -r 2`, no other env)

| model | pp | off | default (fused) | delta |
|---|---:|---:|---:|---:|
| qwen4exp IQ4_NL | 8192 | 1223.1 | **1259.9** | +3.0 % |
| qwen4exp IQ4_NL | 32768 | 1158.4 | **1195.6** | +3.2 % |
| qwen35moe 35B-A3B Q4_K_M | 8192 | 2128.1 | **2265.9** | +6.5 % |
| qwen35moe 35B-A3B Q4_K_M | 32768 | 1561.9 | **1672.5** | +7.1 % |

The other solution's ablation prices `GDN_CONV`+`PLE_CONV` at −10.5 % on its stack; our +3.0/+3.2 %
on qwen4exp is the part still on the table for us (we already had some of the surrounding
optimisations, and the model's `ssm_conv_long_token_f32` was the 303 ms item in the 2026-09-20
profile).  The qwen35moe win is larger because every layer carries a GDN conv there.

## Follow-ups

* The GDN fusion now also applies to qwen35moe / qwen35 / qwen3next via the shared
  `build_conv_state` (the snapshot-cpy adaptation above), so re-check the delivery's MoE prefill
  records on those models.
* PLE is qwen4exp-only; `ple_conv_kernel` is instantiated for F16 and F32 weights (the IQ4_NL
  checkpoint's `ple_conv1d` is F16), matching the other solution's `40a9f4d01` F32-PLE extension.
* The one-shot `GGML_CUDA_CONV_DEBUG=1` trace prints a resolved match per process
  (`GDN_CONV_DIRECT …` / `PLE_CONV_DIRECT …`); it is inert by default and kept for the next
  matcher-debug session.

## Where item 4 stands (assessed 2026-09-21, not started)

* **`idx-relu-sum` is already covered on our tree.**  The reference's `LLAMA_IDX_RELU_SUM` fuses a
  graph-level `RELU -> VIEW(0) -> CONT -> (VIEW(h) -> ADD)*` head-reduction chain.  Our delivery's
  `GGML_CUDA_QSA_INDEXER_SCORE` (default on) is a single fused indexer-score kernel that already
  computes `bias + sum_h relu(dot_h)` in h order (`indexer-score.cu` line ~232), so the chain the
  reference fuses does not exist in our graph.  Item 4's `-1.3 %` is therefore already banked; the
  profile's `indexer 156 ms vs its 63.6 ms` is our fused kernel being slower, not a missing fusion
  (a separate tuning item).
* **`norm-gated.cu` (`rms_rows`) is lower-ROI here than the doc's `-2.9 %`.**  Its target is the
  narrow-row (`ncols <= 256`) norms.  In the 2026-09-20 profile ours already run as the block-per-row
  `rms_norm_f32<256,{true,false}>` (288.8 + 148.8 ms), while the reference's `rms_rows_f32` pair is
  220.0 + 82.7 ms — about 135 ms (~1.2 %) on the whole process.  Worth doing, but behind the
  `hc_combine_norm_f32` kernel swap below.
* **Item 1's `hc_combine_norm_f32` `_b256` swap is CLOSED NEGATIVE** (session 3, after this record was
  written): the reference's `_b256` was ported and gated, but it is **not bit-identical** (the
  256-thread RMS reduction changes the greedy text `1b59d651f2c3` → `fc7c8a10ea45`) and **0.7–0.8 %
  slower** on gfx1151/qwen4exp, so it was reverted — [`2026-09-21-hc-cn-b256-rejected.md`](2026-09-21-hc-cn-b256-rejected.md).
  The next kernel item is **item 5** (MoE bf16 epilogue).
* **Item 3.5 (the three QSA correctness fixes) is an audit, not a cherry-pick.**  `40c0b9c38`
  (maskless only where qsa3 consumes it) and `14fff4f97` (-1 sentinel) are about the reference's
  `qwen4exp_select_complete_blocks` / qsa3-cell selection; `b0f31f587` sizes the block window by the
  highest stored position in `llama-memory-hybrid-idx.*`.  Our delivery has its own derived-visibility
  / keys-only / fused-indexer QSA and its own indexer cache, so each fix has to be checked against
  our equivalent before it is ported (the long-session non-determinism the first one fixes is the
  reason this stays a Phase-1 correctness item).
