# wip/share-budget-r366: server -- the max-min prompt share divides the pass's budget

**Status: PROMOTED at r373 into block `0016` (the --parallel prompt admission).**  The submitter's patch, analysis
and repro are kept verbatim below as the campaign record; what the delivery carries is the amended block, which
divides `budget_before` (the pass's budget) instead of `budget` -- confirmed independently on this box before the
change was cut, and now guarded by `scripts/gate-prefill-fairness.sh`'s HARD `share split` check.

What reproduced and what did not, because the two are different claims:

* **The defect, exactly as described.**  `-lv 5` on 4B Qwen3.5 Q8_0, one R9700, `-np 2 -b 2048 -ub 2048`, two
  fresh 22797-token prompts: the census alternates `added = 1365` / `682` on every step where both prompts are
  active, and after the fix it reads `1024` / `1024` on every one.  The arithmetic in the submission (2046 -> 1023
  + 511 -> the pass closing at 1365 + 682) is right to the token, and a four-slot window reads 748 / 560 / 421 /
  316 on the pre-fix build.
* **Not reproduced: the throughput table.**  The rotation (`slot_next`) alternates which slot leads, so the 2:1 is
  per PASS; cumulatively the two prompts get the same tokens and finish in the same step.  Measured here: the
  pre-fix build 7.15 s (3278 / 3234 t/s) against the fixed build 7.01 s (3309 / 3309 t/s) for the same two
  22797-token prompts, i.e. no end-to-end difference -- and an unequal pair (22797 + 14141) is proportional in
  both arms.  Every step fills the whole window on both builds, so there is no throughput to recover.  The 25.9 ->
  22.1 s and 45.1 -> 36.8 s in the submission are therefore attributed to something else in that run (the two
  clients' arrival alignment, or state left by the arm that ran first) rather than to this change, and are not
  carried into the record.  The change is a fairness/determinism fix with the documented max-min split as its
  claim, not a speedup.
* **A second consequence, not in the submission**: the per-pass share is also what `LLAMA_SERVER_PREFILL_SKEW`
  weights, so a 4:1:1 window now really is 1365 / 341 / 341 instead of 1777 / 147 / 123 whenever the skewed slot
  happened to be first in the rotated order.

Record: `WORKLOG.md` 261127.026, `INVARIANTS.md` r373, `patches/README.md` block 16, `MANIFESTS.md` gate 16.

Thank you for the fair-share prompt admission (r300). With it, `--parallel N` stops serialising prompts, and we've relied on it ever since. While measuring two long prompts prefilling together, we noticed that the window is split unevenly between them. We're sharing the data and a small patch (0001) in case it's useful. Nothing is urgent on our side, and we run it in our own build in the meantime.

**What we saw.** In each pass of the share loop, a slot's share is `budget * w / wsum`, but the loop subtracts each slot's take from `budget` before the next slot computes its share. With two prompts and a 2048-token window, one token each is taken first, which leaves 2046. The first slot then gets 2046 / 2 = 1023, and the second slot's share comes from what the first left, (2046 - 1023) / 2 = 511. The following passes keep the same 2:1 ratio, so the step ends with 1365 + 682 tokens, not 1024 + 1024. `budget_before`, which the loop already keeps for its stop check, is the pass's budget, and dividing it gives the max-min share the comment above the loop describes.

**0001** (`tools/server/server-context.cpp`, +3/-1): `share_i` divides `budget_before` instead of `budget`. With one prompt prefilling, `budget_before` equals `budget` in every pass, so the allocation is unchanged. A slot that needs less than its share still returns the unused part to the budget for the next pass, as before.

**Repro** (`logs/repro-r366.log`, clients `logs/shrep.py` and `logs/shalone.py`): Flash-Next IQ3_XXS, 2x R9700, `-sm tensor -c 262144 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0` (4 slots, unified KV, default admission), no MTP. `shrep.py` sends two fresh prompts at once (n_predict 16, greedy), first a ~39K + ~34K pair and then a ~59K + ~59K pair. `shalone.py` then sends each prompt by itself on the same server. We ran one server per arm: stock r366 (`e10879b0b3`) and r366 + 0001.

| pair | stock r366 (wall / prompt t/s) | + 0001 (wall / prompt t/s) |
|---|---|---|
| 39K + 34K | 25.9 s / 1597 + 1621 | 22.1 s / 1891 + 1722 |
| 59K + 59K | 45.1 s / 1516 + 1335 | 36.8 s / 1640 + 1654 |

Each prompt sent alone gives the same reply hash in both arms (`094677ad`, `660aea63`). In the concurrent pairs, some replies differ from the alone run, in both arms. We think that comes from the batch shape, because a different request differs in each arm, but we haven't confirmed it (stock: Q2 `580e4d38`; patched: P2 `f34f8651` and Q1 `580e4d38`). No GPU faults in either arm. In our own build (MTP, 256K context, our production settings), the same two pairs went from 27.5 to 22.7 s and from 44.7 to 37.3 s, and our reference outputs (greedy, a 512-token prompt set) are unchanged.

We're happy to re-run this on any build or settings you'd like, or to test any change you make to the patch.
