# C2 — the indexer top-k at plain decode (the radix path, `TODO` #125)

**Status: PROMOTED in `v16-6050b467c-r196`** (block 11 amended; campaign branch `wip-kv-residency-125`,
`83513d733`, off r194).  Release record: `WORKLOG.md` 261101.001.

## What the item actually was

The fused indexer top-k (`ggml_indexer_top_k`, `ggml/src/ggml-cuda/indexer-topk.cu`) **already is a
radix select**: four 8-bit passes over the ordered F32 keys, a deterministic ascending-cell gather, and
a block key-shared fast path that does three of the four passes at block granularity.  The handover's
"replace the sort with a radix select" premise was stale.

What was left is that **plain decode ran the slowest form**:

* at `n_tokens == 1` the graph kept the uploaded attention mask as the op's per-cell additive, so the
  op could not use the block key-shared path (a per-cell additive means every cell of a block no
  longer shares one key) and re-read the `[n_kv]` F16 mask on every pass;
* the verify/MTP width already used the block path (derived bias + derived visibility), which is why
  its **0.9 ms/step** was never the same problem and is unchanged by this work.

## What changed

`src/models/qwen4exp.cpp` (`build_qsa_top_k`): the per-cell half of the bias is **derived at any
width**.  At `n_tokens == 1` the per-block bias stays the uploaded tensor (the fused decode score op
applies it), but the top-k now gets `cell_vis`/`q_vis` instead of the mask, so `additive == nullptr`
and the block path runs.  The attention still reads the mask
(`qwen4exp_want_derived_vis` keeps its own `n_tokens > 1` gate), so this changes the top-k only.
`llm_graph_input_qsa` stores the build-time device-support flag (`vis_supported`) and `can_reuse`
compares the same predicate (the per-cell form no longer follows the per-block one).

`ggml/src/ggml-cuda/indexer-topk.cu`: the block path needs only `cell_pos` + the block geometry;
`blk_idx`/`blk_tail` are the opt-in per-block **bias** derivation and may be null (the block key is
then the score itself).  `indexer_topk_histogram_blocks` guards its incomplete-block skip on
`blk_idx != nullptr`.

`tests/test-backend-ops.cpp`: a new `INDEXER_TOPK` case with null `blk_idx`/`blk_tail` (the plain-decide
form) against the CPU reference.

## Measured — 103k prompt, plain decode, pool, 1 GPU

`rocprofv3` short/long subtraction (`kv-kernel-census.py`, `-n 2` → `-n 130`):

| bucket | before (mask / cell radix) | after (derived vis / block radix) |
|---|---|---|
| radix top-k, total | **2.062 ms/token** | **1.143 ms/token** |
| — histogram<__half> | 1.060 | — |
| — write_blocks<__half> | 0.663 | — |
| — histogram_pass1 | — | 0.201 |
| — histogram_blocks | — | 0.194 |
| — write_blocks_grouped | — | 0.268 |
| — count_cells_grouped | — | 0.151 |
| — radix_select / hist_accum / base_scan / init | 0.339 | 0.329 |
| **decode kernel time, total** | **17.520 ms/token** | **16.363 ms/token** |

Wall A/B, interleaved, `-n 512`: **40.48 / 40.48 → 42.11 / 42.11 t/s (+4.0 %,
−0.95 ms/token)**.  The two arms are the same binary with `GGML_QSA_DERIVED_VIS=0` (the old path).

## Gates

* Greedy text byte-identical to the r194 binary at **5.3k** (`332 dd1981213373`) and **103k**
  (`834 263da53898a3`); identical to `LLAMA_INDEXER_NOBLOCK=1` (block path vs grouped cell path, the
  isolation A/B); and **MTP `draft-mtp` depth 4, `-n 512`** is `2259 f3553fb82f82` on both binaries.
* `test-backend-ops -o INDEXER_TOPK` **4/4** including the new null-`blk_idx` case;
  `-o FLASH_ATTN_QSA` 26/26.
* `scripts/gate-prefill-logits.sh` on the 27B Q8_0 point: **0.000000 KLD / 100.000 % top-p** (the
  change is decode-only by construction).
* `check-gpu-faults.sh`: clean.

## Note found on the way (pre-existing, not this change)

`GGML_QSA_DERIVED_VIS=0` is **not text-neutral at prefill** on the 5.3k pool arm: the r194 binary
also reads `305 chars a32ea64e94af` with it, vs `332 dd1981213373` default.  So it is an A/B arm, not
a no-op; it is still the kill switch for this change (it disables the derived per-cell form at every
width).

## Not done

The remaining top-k mass is the block path itself (1.14 ms/token): `write_blocks_grouped` (0.268) and
`count_cells_grouped` (0.151) are full cell walks, and the recount is required whenever `cell_blk`
is not contiguous (issue #89).  Three 11-bit passes would cut one pass but `hist_accum` uses one
thread per bin and 2048 bins exceeds the 1024-thread block limit; fusing select+accum saves only
launch overhead.  Promotion (block 11 amendment, docs, `validate-set.sh`) is the maintainer's call.
