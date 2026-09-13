# GDN recurrent-state rollback bound (`n_rs_batch`) — integrated 2026-09-12 (10)

**Status: landed in the delivery** as the block-02 amendment of 2026-09-12 (canonical tip
`47a9d4d86`, tree `c24871386c479865d41476726cf1f01c43b23ea6`).  This directory is the *record*: the
patch was developed on the gfx1201 box and handed over for integration; the delivery copy of the
investigation is here, and the block-02 note + `GREEDY-PURITY.md` §27 are the delivery-side record.

| file | what |
|---|---|
| `gdn-rs-rollback-bound.patch` | the handed-over patch, verbatim (20 files, +96/-23) — integrated as-is into block 02 |
| `investigation-README.md` | the gfx1201 investigation's own summary (`~/ngram-mod/README.md`) |
| `investigation-fix-proposal.md` | the original prompt + findings + fix options A-E (`~/ngram-mod/fix-ngram-mod.md`) |
| `../strix-halo/qsa-item4/` | the separate QSA sparse-regime work (item 4) — unrelated to this |

## The defect

The whole-batch **chunked** GDN kernel writes no rollback snapshots (slot 0 only).  Block 02's
2026-09-11 "K-independent whole-batch chunked prefill" assumed a batch above `max(K, 16)` "cannot be
a speculative verify batch" and is never rolled back into.  That holds only while every speculator's
maximum draft is bounded by `n_rs_seq` — and `n_rs_seq` is sized from `speculative.draft.n_max` (7),
while `--spec-ngram-mod-n-max` can draft 64.  A 65-token verify batch therefore took the chunked path,
and a small tail rollback then restored a snapshot plane that batch never wrote: the recurrent state
silently rewound (finite but wrong, so decoding "worked").  The `seq_rm` guard added by block 02 on
2026-09-11 detects exactly this — the reported warning is real, not a false positive.

## The fix (as integrated)

1. **`n_rs_batch`** — the longest per-seq batch that can be rolled back into =
   `common_speculative_n_max(&params.speculative) + 1`.  It flows
   `llama_context_params::n_rs_batch` -> `llama_cparams` -> `ggml_gated_delta_net()` (**new op param
   1**, with the Vulkan reference-clone updated) -> the CUDA dispatch, and into
   `llama_memory_recurrent` so `seq_rm`'s guard stays a real invariant check.  The chunked threshold
   becomes `GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16, n_rs_batch)` — a batch that can be rolled
   back into always runs the sequential kernel, which writes its `K` snapshots.  Snapshot memory is
   unchanged; sizing `n_rs_seq = 64` instead would have cost ~+8 GiB.
2. **Pre-batch slot** — for `0 < n_tokens < K` the graph also writes the *pre-batch* ssm and conv
   state into slot `n_tokens` (`src/models/delta-net-base.cpp`), so a rollback of the whole last batch
   restores the state before it.  Graph-level copy, no kernel change, no output effect for
   `n_tokens >= K`.

## Validation on this box (gfx1151, Strix Halo)

| gate | result |
|---|---|
| in-tree `test-recurrent-state-rollback` (`-m Qwen3.8-27B-Q8_0 -c 512 -b 512 -ub 512`), **unpatched** | **FAIL** — `multi-seq split replay logits mismatch (max diff 6.5366, first at seq 0 pos 16)` |
| the same test, **patched** | **PASS** — `multi-seq split replay matched (max diff 0)` + `seq-1-only decode independent of seq 0 (max diff 0)`, cache fills `0x00` and `0x3e` |
| `test-backend-ops -o GATED_DELTA_NET` | **46/46** (ROCm0 + CPU) |
| `FLASH_ATTN_QSA` (beta tree) | 22/22 |
| neutrality, 27B hybrid | `plain == draft-mtp n_max 7` = `e164f09af338` (670 chars), **identical** before/after |
| neutrality, qwen4exp | `plain` = `0fc4910d5824`, **identical** before/after |
| neutrality, prefill | 27B pp2048/pp8192 450.4/428.0 -> 451.0/428.9 (within noise) |
| beta re-cut (15th) | `eb15f3ee1` / tree `ffa3a11c30ba6d42dea2520f402126370df3bbb6`, patch 3 819 lines, round-tripped; all four gate combos + `draft-mtp n_max 3` byte-identical `0fc4910d5824` |

Default configs are unaffected because `n_rs_batch <= 16` there (1 with no speculator, 8 with MTP
`n_max 7`), so the chunked path keeps its whole-batch, K-independent shape.  Only a long-draft
speculator's verify batches move to the sequential kernel — and for those the alternative is
corruption, so correctness decides (`GREEDY-PURITY.md` §19/§27).

## Known limitation (not fixed here)

The upstream `ggml_ssm_scan` (Mamba, `src/models/mamba-base.cpp`) path has the same two
characteristics and needs the same bound + pre-batch slots; it is deliberately untouched because this
work targets the qwen35/delta-net models.
