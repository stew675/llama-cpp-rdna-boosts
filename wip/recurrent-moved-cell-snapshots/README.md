# wip/recurrent-moved-cell-snapshots — a moved recurrent cell keeps its rollback snapshots

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
