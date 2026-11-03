# `wip/host-pool-speed/proposals/` — delivery changes awaiting the maintainer's call

**Each file is a diff against the tree, staged for a decision** (AGENTS.md: report findings before a
delivery-affecting commit).  An ACCEPTED entry stays here as the record of what was asked for and why.

| file | what / why it is needed |
|---|---|
| `gate-prefill-logits-extra.patch` | Adds `--extra "ARGS"` (`PPL_EXTRA`) to `scripts/gate-prefill-logits.sh`: extra `llama-perplexity` arguments, word-split, folded into the base-file name so a base recorded on one configuration is never reused for another.  **Why:** the gate had no way to judge a *non-default* configuration at all, so the host-expert arm this campaign must gate (`-ncmoe 40 --host-experts pool`) could not be measured — the alternative was a hand-rolled `llama-perplexity` invocation with no base-meta check.  It is additive (no existing arm changes) and it is the shape `--candidate-env`/`--ab` already use.  Measured use: `README.md` §14.4 (mean KLD `-0.000000`, same-top-p `100.000 %`).  **Committed 2026-11-02** (maintainer's call). |
