# `recurrent-moved-cell-snapshots` — a moved recurrent cell keeps its rollback snapshots

**Status: PROMOTED (r195, block 03).**  Contributor: overdoingism (PR #261).  The patch is the delivery's
block 03 fold; this record is the campaign's history plus the delivery-side validation, and the upstream
candidate is [`upstream/recurrent-moved-cell-snapshots/`](../../../upstream/recurrent-moved-cell-snapshots/README.md).
The `wip/` copy is retired (this directory replaced it), per the promotion rules in `AGENTS.md`.

> **Delivery-side resolution (r195, 2026-10-31).**  The mechanism was confirmed on the delivery tree, not
> only on plain llama.cpp: `build_rs_inp_impl` logged `n_rs=3 n_seqs=1` and `n_rs=3 n_seqs=2` at `head=0` on a
> 3-slot MTP server (35B-A3B UD-Q4_K_M, `-sm tensor`, 3x R9700), i.e. the extra-state path really runs and
> is not reserve-only.  A deterministic gate now exists as **Phase C** of
> `tests/test-recurrent-state-depth.cpp` (three sequences; seq 1 verifies a K-token batch, then `[seq 0,
> seq 2]` decodes one token each as ONE ubatch (moves seq 1 -> extra state) or as two single-sequence
> ubatches (no move), then seq 1 rolls back `r` tokens and the replayed logits are compared).  On the r194
> tree it fails **15/15** arms (`n_rs_seq` 1/2/4/8 x `r` = 1..n, largest logit difference **2.94 - 9.64**,
> [`evidence/phasec-r194-1gpu.log`](evidence/phasec-r194-1gpu.log)); with the fold it passes **15/15** on one
> GPU and with `-sm tensor` on three ([`evidence/sweep-4b-3gpu-tensor-r195.log`](evidence/sweep-4b-3gpu-tensor-r195.log)),
> and the pre-existing Phase A/B failure set is unchanged (130 failures on this box, 4B Q8_0, gfx1201:
> [`evidence/sweep-4b-1gpu-r194.log`](evidence/sweep-4b-1gpu-r194.log) vs
> [`evidence/sweep-4b-1gpu-r194plusfix.log`](evidence/sweep-4b-1gpu-r194plusfix.log)).  Cost, one binary with
> the copy toggled off, 35B-A3B UD-Q4_K_M, MTP `n_max 5` (`n_rs_seq` 5), 3 concurrent x 200 tokens, three
> rounds interleaved: 3 slots 115.95/115.53 (off) vs 115.95/114.74 (on) t/s, 4 slots 94.57/95.25 (off) vs
> 94.75/96.26 (on) t/s, i.e. inside the arm's own spread at depth 5 (the worst case, an extra cell copied on
> every step, is `n_rs_seq x (R_row + S_row) x recurrent layers` = ~315 MiB read plus the same written per
> token on this model, so deeper drafts are worth re-measuring).  The merge request's `upstream/` copy was
> verified too: both patches apply (`git apply --check`) on `6050b467c` and on master `0ec4027a2`, and their
> sha256s match the READMEs.

The contributor's original record follows, unchanged.

Written by Claude (AI).  One `git am` file (one commit), made on `v16-6050b467c-r187`; it also applies on r188 (`git apply --check`, not built there).  It touches `src/llama-graph.cpp`, `src/llama-graph.h`, `src/llama-memory-recurrent.cpp` and `src/llama-memory-recurrent.h` (+70 / -1).
`recurrent-moved-cell-snapshots.patch` sha256 `4b61f5a2ed829560211b5a88a3f94fc4a042846bbad4f1ceb0109d8b27167c17`.

**The defect is llama.cpp's own.**  `ggml-org/llama.cpp` master (`0ec4027a2`, checked 2026-11-01) and `6050b467c` have the same `build_rs`, so a copy cut against plain `6050b467c` is in `upstream/recurrent-moved-cell-snapshots/`.  It also passes `git apply --check` on that master.  Whether and when it goes to ggml-org is yours to decide.

**Symptom.**  With 3 or more slots of a unified KV cache on a recurrent / hybrid model with speculative decoding, a reply running next to others can turn into unrelated text and end early, with no error in the log.  Reasoning of `In one or two sentences: what does a bill of materials list?` at temperature 0.7 ended like this:

```
Here's
Here's'all')":
...Catégorie:Verbes irréguliers en anglais
```

and one reply on inventory turnover ran into the Hong Lou Meng text another slot was continuing (`...笑的不行，寶玉便要...`).

**Cause.**  From 3 cells on, `find_slot` can move a cell out of the range of a later ubatch of the same decode after it verified a draft.  `build_rs` copies only plane 0 of such an extra state, so a rollback of that sequence reads the snapshot planes another sequence left at the new row.

**What changes.**  `build_rs` also copies planes 1..n_rs_seq of the extra states, through a new input `s_copy_extra`, gathered before the ubatch writes its snapshots.  With 2 cells or fewer `n_rs` equals `n_seqs`, so there is no extra state and the graph does not change.

**Kill switch.**  None: it corrects wrong results, and the graph is unchanged where it does not apply.

## Results

Linux: Ryzen 9 3900X, one R9700 (gfx1201), ROCm 10.0.0.  `ROCBLAS_USE_HIPBLASLT=0` (GREEDY-PURITY §44) in the text comparisons.

**The mechanism, without a server.**  Three sequences, one verified, moved by a later ubatch of the same decode, then rolled back, against the order that rolls back first (Qwen3.8-2B Q8_0, plain llama.cpp `6050b467c`):

| | sequence A, largest logit difference | A's state |
|---|---|---|
| plain, CPU | 10.52 | differs |
| + fix, CPU | 0 (no logit differs) | same |
| plain, ROCm | 10.31 | differs |
| + fix, ROCm | 0 | same |

On ROCm the other two sequences differ by up to 0.26 with and without the fix: the two orders place the KV cache differently.

**Requests.**  r187 against r187 + this patch, `ngram-mod,draft-dflash` (n-max 15, n-min 1, n-match 30), `--spec-draft-n-max 5 --kv-unified -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0`.  16 requests per round (tool-calling requests, two thinking questions at temperature 0.7, short chats, long-text continuations), at most 8 in flight, a seeded order.  Empty = finished with `stop` or `eos`, no content, no tool call; all were thinking questions.  A broken reply that goes on to write an answer is not counted.

| model, slots | rounds | replies | empty, r187 | empty, + fix |
|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_M + DFlash, 4 | 12 | 192 | **6** | **0** |
| same, 3 | 6 | 96 | **3** | **0** |
| Qwen3.8-27B UD-Q4_K_XL + DFlash2, 4 | 6 | 96 | **2** | **0** |

The same request set on r184 with `--spec-type draft-dflash` alone gave 1 of 192.

**Where it should change nothing.**  UD-Q4_K_XL + DFlash2, `draft-dflash`, r187 / fix / fix / r187, one request at a time (three ERP tool requests at temperature 0, one thinking question at 0.7 with a seed, an 8K continuation): the same text in all four starts with 1 slot and with 2 slots.  Two requests at once on 2 slots: both fix starts match the second r187 start; the first r187 start differs, so this pair also varies without the patch.

## Not validated

- Speed and memory with 3+ slots: the gather costs n_rs_seq extra state rows per extra cell; not measured, and the texts differ, so t/s would not compare.
- `-ncmoe` / host-resident experts, several GPUs and `-sm tensor`, Windows, other recurrent architectures, `draft-mtp`.
- r188 was only checked with `git apply --check`.
- No single broken reply above was traced through the rollback; the link is the mechanism test plus the table.
