# spec-batch-budget: an optional load-aware MTP draft cap (`LLAMA_SPEC_BATCH_BUDGET`)

One format-patch on top of `v12-097cf33d9-r71` (tree `dc2cc150`; applied tree `02fba607`), for `git am`.  It also applies cleanly to
`v12-097cf33d9-r72` (tree `43115425`; applied tree `946c6048`); the tests below are on r71.  It touches
`tools/server/server-context.cpp` (+21 / -1).

## What we saw

With the MTP draft and several slots generating, the verify batch is `n_slots * (1 + n_draft)` tokens: 16 tokens per
step with 4 slots at `--spec-draft-n-max 3`.  At that width the extra draft tokens cost more step time than the
accepted ones save, so 4 concurrent requests run slower than they would with shorter drafts.

## The patch

- `LLAMA_SPEC_BATCH_BUDGET=N` caps each slot's draft at `max(1, N / n_generating - 1)` while more than one slot is
  generating.  With 8: 2 slots keep 3 draft tokens, 3 or 4 slots get 1.
- Every generating slot keeps at least one draft token, and a single request is never capped.
- Unset (or 0), nothing changes.

## Tests (2 x R9700 32 GB, gfx1201)

Same config as the command below, no other tuning env.  One binary (r71 + patch), a fresh server per run after
`llama-warmup.py`, budget unset then 8, two rounds.  Each concurrent run is N chat requests of 300 tokens.

```
llama-server -m <IQ3_XXS> -c 262144 -ngl 99 -sm tensor --n-cpu-moe 0 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -ot per_layer_token_embd.*=CPU -lm none -lzm off \
  --spec-type draft-mtp -md <mtp Q8_0> -ngld 99 --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0
```

| t/s (round 1 / round 2) | unset | `LLAMA_SPEC_BATCH_BUDGET=8` |
|---|---|---|
| greedy 400 tokens, 2nd request | 142.2 / 142.3 | 142.3 / 142.4 |
| 2 concurrent, aggregate | 160.7 / 159.5 | 160.3 / 161.6 |
| 3 concurrent, aggregate | 171.5 / 168.6 | 167.9 / 160.1 |
| 4 concurrent, aggregate | 165.8 / 167.8 | 180.6 / 181.9 |

4 concurrent requests go from 166.8 to 181.3 t/s (+8.7 %), in both rounds.  2 concurrent is unchanged, as it should be
(the cap is 3 there).  3 concurrent varies by several t/s between rounds in both columns (the stock r71 binary gave 172.6 and 160.9 in
two separate runs), so we read it as noise.

Single-request greedy output is the same (`a7a81f01b3bc`, draft acceptance 274/374 in every run).  The concurrent
outputs vary from run to run with and without the budget, as they do on stock.  No faults or aborts.

8 is the value we tuned on this model and GPU; the best budget probably depends on where a GPU's small-batch decode
kernels stop paying off.

No urgency on our side.  Happy to sweep other budgets or slot counts on your config if that's useful.
