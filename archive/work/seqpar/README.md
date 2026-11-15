# seqpar — PROMOTED into block 12 in r291 (qwen4exp sequence parallelism)

**Status: promoted.**  The contributor's series (PR #315, @briansp2020) is folded into **block 12** as a
**default-on** feature: `LLAMA_SEQPAR=1` (kill switch `=0`), `LLAMA_SEQPAR_MIN` 512 clamped to a floor of 128,
`LLAMA_SEQPAR_GATHER_BF16=1` (F32 wire via `=0`).  It was only ever reviewed and measured here, never applied
from `wip/`: the authoritative artifact is `patches/0012-…-campaign-remainder-…` and the record is
`WORKLOG.md` **261115.004**, `benchmarks/2026-11-15-seqpar.md`, `patches/README.md` block 12's r291 amendment,
`ENVIRONMENT.md` §21, `INVARIANTS.md` r291, `GREEDY-PURITY.md` §92 and `TODO` #330.

* `CONTRIBUTOR-README.md` — the submitter's own v2 design notes, verbatim from the PR's
  `wip/seqpar-r289/README.md` (the env table, their family-switch table, the `MIN` sweep, the per-arm raw-log
  index, the 2-rank-by-construction scope statement).  Their five `0001`-`0005` patch files are **not** kept:
  they are superseded by the block-12 commit, so the delivery's `patches/` is the artifact.  Their `logs/`
  directory (the gate, sweep and server logs their README indexes) lives in PR #315's head commit
  (`320954d3` on their branch), which GitHub keeps reachable for the closed PR; our own numbers are in the
  record named above.
* **What the promotion changed** relative to the submission: folded into an existing block instead of a new one
  (block 12 owns the `ALL_GATHER` collective and the op table the new op extends); the enable predicate
  generalised from exactly 2 devices to `>= 2` (the reduce-scatter still declines above 2 ranks -- the 3-card
  win is the gathers only); `LLAMA_SEQPAR_MIN` kept at **512** on a measurement of our own (`-1.0 %` at 128,
  `-0.8 %` at 256, `+6.4 %` at 512, ce) rather than their sweep's sub-512 wins; the **bf16 gather wire kept as
  the default** after measuring the F32 wire's logit win against its **-2.1 %** on the default chain; and the
  sub-128-token ubatch band **clamped off**, because the two failure modes they reported there (the Meta
  split-state recursion at 48..112 tokens and the token-scatter assertion on a 2-token graph) are block 09's to
  fix, not this feature's to document.
