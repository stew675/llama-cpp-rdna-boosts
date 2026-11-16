# The in-place prefill read was never the carrier: a stale expert-cache ALIAS resolved a sibling ROLE's arena

**Status: CLOSED (r298, 2026-11-16).**  `TODO` **#335** and **#336** are both retired.  This directory holds the
campaign's reproducer harness; the finding, the numbers and the gates are in
[`benchmarks/2026-11-16-in-place-read-alias-role.md`](../../../benchmarks/2026-11-16-in-place-read-alias-role.md),
`WORKLOG.md` 261116.005, `INVARIANTS.md` r298 and `GREEDY-PURITY.md` §89 (corrected) + §94.  What it was folded into:
blocks **09** (the role-aware alias guard), **10** (`moe_cache_repair_copy` + its census + the consumer's
served/declined split) and **15** (the promise in the GTT prefill in-place path) — no new block.

## What was asked, and what it turned out to be

r297 left two items, both written as defects of `MOE_EXPERT_PREFILL_IN_PLACE`'s in-place **read**: a degenerate greedy
output under VRAM pressure (`#335`, blamed on a captured graph replaying an evicted arena pointer) and a coherent
near-tie between the in-place and staged arms (`#336`, "the two arms read the same bytes, so a logit difference is a
bug", magnitude owed).

Both are one bug, and it is not the read.  The expert cache resolves a table from the scheduler's **copy tensor
address**, validated by device + layer only; a layer's `gate`/`up`/`down` tables share both, and a layer's routed ops
are issued back to back, so a recycled buffer resolves a **sibling role**.  The take-over hooks resolve the host
master (right role, so they decline for a non-identity table and the scheduler fills the copy) while the consumer's
redirect resolves the copy pointer — so the redirect served the *neighbour's* arena.  Measured: `blk.43.ffn_up_exps`
computed out of `blk.43.ffn_down_exps` (`nan`, then `-2.1e7`).

* the in-place read itself: **KLD 0.000000 / same-top-p 100.000 %** against the staged arm (`#336`'s owed magnitude);
* both of r297's pairs collapse to `835 chars 7a3474cbe5f1` with the role check; `GGML_MOE_DEVREMAP=0`,
  `GGML_CUDA_DISABLE_GRAPHS=1` and `GGML_SCHED_DEVGATHER=0` do not move the pre-fix pair;
* `#335`'s workaround matrix is a stale-alias matrix (`DISABLE_GRAPHS` bakes the wrong pointer into a capture; no
  eviction means no alias churn; the staged arms never run the redirect; no drafter means no T=42 ingest batch), and
  the arena validator read 0 `MISMATCH` because *the arena was right*.

`#335`'s exact degenerate text did not reproduce on this box (the pressure arm is coherent pre-fix); the hazard is
present in it (a wrong-role refusal at layer 30), and the degenerate form needs the wrong pointer baked into a
capture.  Recorded honestly as such rather than claimed.

## The harness in this directory

`arm-335.sh` is the reproducer used for every number in the record:

```bash
# 2 devices -sm layer, Flash-Next GSQ-RCO IQ3_XXS + the mtp sidecar, greedy
PROMPT=$PWD/prompts/kld-corpus-35b-in-domain.txt CTX=32768 HEATMAP=0 \
  bash arm-335.sh <label> MOE_EXPERT_PREFILL_IN_PLACE=1
# -> "== <label> rc=0" then "<n> chars sha=<12 hex>" and the eviction/re-arm lines from stderr
```

Environment it honours: `CTX` (default 16384), `PROMPT` (default `prompts/prose-rdna-boosts.txt`), `NGEN` (64),
`HEATMAP` (`load`/`0`/unset), `LV`, `EXTRA`, `DEVS`, `NODRAFT`, plus any `K=V` assignment it is given (they are
`env`-prefixed onto `llama-cli`).  `corpus-1k.txt` is a 4 KB truncation of the KLD corpus, used only to show that a
short prompt does NOT reproduce the differential (it needs the 24 K-token prompt's allocator churn).

The instruments the investigation used are **not** in the delivery (they were temporary and are removed): a per-op
`dst` hash dump (`MOE_EXPERT_OP_DUMP`), a per-op kernel-family trace (`MOE_EXPERT_KERNEL_TRACE`), and a
`ggml_cuda_slab_classify()` + `moe_cache_debug_table()` slot/placement dump.  Re-add them the same way if another
"correct bytes, wrong answer" appears: the per-op hash is what localised this to a single dispatch (`blk.43.ffn_up_exps`
at T=42), and the name-keyed table dump is what showed the served table was a *different* one — an alias-keyed lookup
would have reported the wrong table's (valid) state.  What IS in the delivery: `MOE_EXPERT_ALIAS_ROLE_CHECK` (A/B),
`MOE_EXPERT_TAKEOVER_FAULT` (diagnostic) and the `take-over promises broken=N repaired=N` census line.
