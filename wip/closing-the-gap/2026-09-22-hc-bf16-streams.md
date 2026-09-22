# HC BF16 streams (`blk16` / `res16`) — Phase-1 item 16, ported default-OFF (2026-09-22)

**Status:** done, gated **OFF** by default (`LLAMA_HC_BLK16=1` and/or `LLAMA_HC_RES16=1`); the default
build is byte-identical to `4a75744fa`.  Fork branch `gap-closing` @ **`37e8b1751`**, exported as
[`patches/0011`](patches/0011-gap-closing-WIP-port-the-HC-BF16-streams-LLAMA_HC_BL.patch).

## What it does

Ports the reference's 16-bit hyper-connection streams so the fused `hc_combine_norm` reads/writes
BF16 instead of F32 for the two large HC tensors:

* **`res16`** — the residual stream (`residual` in / `out_res` out) is BF16 **in place** over each
  tensor's own buffer.  This is the dominant half.
* **`blk16`** — the attention out-proj `block_out` (`[n_embd, 1, T]`) is BF16 in place.  Its producer
  (the MMB dense GEMM) already honoured a BF16 mark.
* **`out_xn_bf16` / `store_xn_f32`** — the BF16 arms for the fused kernel's `out_xn` (the existing
  HC16 activation marking already marks `out_xn` BF16-only, but the fused combine was the one
  producer that did **not** emit the copy).  Kept for completeness; enabled with the two flags.

## Files

| file | change |
|---|---|
| `ggml/src/ggml-cuda/hyperconn.cuh` | `out_xn_bf16`/`store_xn_f32`/`res_in_bf16`/`res_out_bf16`/`blk_in_bf16` args fields |
| `ggml/src/ggml-cuda/hyperconn.cu` | BF16 arms in `hc_combine_norm_f32` **and** `hc_combine_norm_single_f32`; `hc_bf2f32`/`hc_f2bf32` |
| `ggml/src/ggml-cuda/moe-weighted-reduction.{cu,cuh}` | `_bf16_v4_out` + `_f32in_bf16out_v4` (bf16-in/bf16-out with the shared-expert merge), the `merge` argument |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_hc_combine_norm_identify` (structure-only), `ggml_cuda_hc_combine_norm_set_bf16`, the blk16/res16 marking block in `graph_optimize` (+ residual alloc deps), the MoE merge detection |

## Adaptations our tree needed (vs the reference `b0f31f587`)

1. **The combine matcher is ours, not `hc-match.inc`.**  Our fused combine is matched inline in
   `ggml_cuda_try_fuse` (the repeat-anchored variant fires on qwen4exp).  Marking has to run in
   `graph_optimize` (before the producers), so a **structure-only** identifier
   (`ggml_cuda_hc_combine_norm_identify`) reproduces the repeat-anchored walk; if it does not match,
   nothing is marked and the flags stay inert.  Only the repeat-anchored form is recognised.
2. **The MoE shared-expert ADD is not adjacent to the reduction chain.**  In our graph
   `ffn_out = ADD(ffn_moe_out, ffn_shexp_gated)` and `ffn_moe_out` is followed by `ffn_gate`, not by
   the ADD, so the reference's merge fusion (`nadd = i + node_count`) cannot fire and the `ffn_out`
   block_out is left F32.  The reduction/merge arm is kept in the marking for trees where they are
   adjacent; on qwen4exp only the 48 attention-path `MUL_MAT` block_outs take blk16.
3. **`blk_in_bf16` is independent of `res16`.**  The reference nests it inside its `res16` block; we
   test `ggml_cuda_mmb_blk16()` on its own so `blk16` alone is a coherent configuration.
4. **`blk16`/`res16` are gated together at the fusion site.**  `out_xn` is already marked BF16-only
   in the *default* build (`all_bf16_consumers`), but its F32 store is deliberately kept there, so
   `set_bf16` returns early unless one of the two flags is on — that is what keeps the default
   byte-identical.

## Gates

Box: `halo` (gfx1151), qwen4exp IQ4_NL (`Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-…`), `-ctk/-ctv f16`.

* **Default (both env vars unset) is byte-identical** to the pre-change build @ `4a75744fa`:
  `test-logits-width-probe` W=1..8 hashes and `row0_row1_hashes 1:268e0673300b7a33/…`,
  `width_purity=PASS (worst maxdiff 0)`; same-seed 128-token greedy text `b639e324753d` (617 chars)
  on `prompts/prose-rdna-boosts.txt`.  (The 4 changed TUs were rebuilt on both sides via `git stash`.)
* **With `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1`:** coherent output;
  `width_purity=PASS (worst maxdiff 0)` (prefill is lossy, so the hashes move — expected, do **not**
  gate on cross-build equality); `plain == draft-mtp --spec-draft-n-max 3` greedy text
  **byte-identical** (`984263fb8e0f`, 434 chars, Q4_K_M MTP sidecar, `--reasoning off`).
* Both streams fire (verified with an env-gated mark log): 48 attention `block_out` `MUL_MAT`
  outputs marked + the MMB dense producer takes `MMB_BLK16 dense BF16 in place`; ~46 of 48
  `hc_combine-N` residual ADDs marked (only `l_last-0`/`l_last-46` fail the reader check —
  GET_ROWS/ADD readers).

## Performance — `-b/-ub 4096`, `-p 8192,32768 -n 0 -r 3` (the A/B protocol)

| arm | pp8192 | pp32768 | Δ8192 | Δ32768 |
|---|---:|---:|---:|---:|
| default (both off) | 1252.4 | 1206.5 | — | — |
| `blk16` only | 1292.3 | 1236.5 | +3.2 % | +2.5 % |
| `res16` only | 1310.8 | 1262.1 | +4.7 % | +4.6 % |
| `blk16`+`res16` | **1313.6** | **1264.0** | **+4.9 %** | **+4.8 %** |

The residual stream is the dominant half.  This is larger than the session-5 estimate (~1.8 s /
3.8 %) which priced only the `hc_combine_norm` kernel; the residual also removes traffic from the
combine's neighbours.

## Follow-ups / limits

* The `ffn_out` (MoE-merge) block_out stays F32 on our graph (see adaptation 2).  Capturing it
  would need either an adjacent ADD or a separate bf16-aware ADD path.
* `out_xn_bf16` is wired but inert on qwen4exp because the mark already exists and the flags are
  what enables it; a future default-on decision there is a separate numerics change (verify with the
  width probe + same-seed text).
* Lossy — this is why the maintainer asked for default-OFF (the same exception as `MMB_DOWN16`).
