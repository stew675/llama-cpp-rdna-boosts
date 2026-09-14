# Recurrent-state snapshot budget — deeper MTP-at-depth follow-up

**Status:** investigative follow-up (2026-09-14).  Not a blocker for issue #30 — the reported load
failure is **fixed by the V4 policy** (`MEASUREMENTS.md` §C; the ~744 MiB F16 staging scratch V4 removes
was the missing margin).  This document is for the *next* configuration step: 24 GiB cards, deeper
contexts, higher adaptive ceilings, and any future model whose GDN state is larger.

Companion: `wip/issue-30-mtp-decode-regression/` (`README.md` Action C, `MEASUREMENTS.md` §C).

---

## 1. Where the memory goes

The recurrent (GDN) state is what makes qwen35-27B's MTP-at-depth expensive.  `llama_memory_recurrent`
allocates, per GDN layer, `n_rows = mem_size * (1 + n_rs_seq)` state rows, where

* `mem_size = n_seq_max` (the server's `n_parallel`, **4 by default**), and
* `n_rs_seq = draft.n_max` via `common_params_speculative::need_n_rs_seq()` (`has_mtp()` is true for the
  built-in MTP head) — the rollback snapshot count.

Measured on the 27B (`-c 32768`, verbose log): **`RS buffer = 7780.50 MiB`** at `n_rs_seq = 12`,
`mem_size = 4`, split `R (f32) 292.5 + S (f32) 7488.0`.  Per row/plane:

| component | per plane (4 seqs) | per plane (1 seq) |
|---|---|---|
| `R` (conv state) | 22.5 MiB | 5.6 MiB |
| `S` (SSM state) | 576.0 MiB | 144.0 MiB |
| **total** | **598.5 MiB** | **149.6 MiB** |

| configured ceiling `n_max` | `n_rs_seq` | planes | RS @ `n_parallel 4` | RS @ `n_parallel 1` |
|---|---|---|---|---|
| 3 | 3 | 4 | 2394 MiB | 599 MiB |
| 7 | 7 | 8 | 4788 MiB | 1197 MiB |
| **12** | 12 | 13 | **7781 MiB** | **1945 MiB** |

So the adaptive ceiling and the server's slot count **multiply**: ceiling 12 costs +2992 MiB over
ceiling 7, and `n_parallel 4` costs 4x the whole thing.  At `-c 196608` q8_0 the target KV is ~6528 MiB
and the weights ~16053 MiB, which leaves the RS set competing with the draft context's ~900 MiB
(768 KV + 130-260 compute) for the last few hundred MiB — hence the OOM at ceiling 12.

## 2. Why it needs a snapshot set at all

A verify batch decodes `K = n_max + 1` rows through the GDN.  On a partial accept of `a` draft tokens the
recurrent state must be rewound to the state after `a` tokens.  The cheapest way to make that exact is
to write the state after every position into its own plane during the forward pass (the current design),
so the rewind is a plane *selection*, not a recomputation.  `(1 + n_max)` planes = the live state plus
one per possible rewind point.  Any reduction is therefore a space/time (or precision) trade.

## 3. Levers, cheapest first

### L1 — `--parallel 1` (user knob; no code)
RS 7781 -> 1945 MiB.  Already verified to load the reporter's exact failing config.  Cost: one concurrent
request slot.  Good for a personal/single-user long-context server.

### L2 — memory-aware effective ceiling + a clear diagnostic (recommended stopgap)
The fit already **detects** the shortfall (`common_params_fit_impl: cannot meet free memory target ...
need to reduce device memory by 1608 MiB`) but aborts because `-ngl 99` is pinned.  Instead of an opaque
`cudaMalloc failed`, the loader should compute the RS requirement and, if it does not fit, **reduce the
effective adaptive ceiling** (i.e. `n_rs_seq`, and the controller's cap so it cannot draft past it) to
the largest value that does, print a one-line notice, and continue:

```
E --spec-draft-n-max 12 does not fit with -c <ctx> <type_k>/<type_v>; reducing the effective
  adaptive draft ceiling to <N> (snapshots are 598.5 MiB/plane at the current --parallel).
  Use --parallel 1 for the full ceiling, or lower -c.
```

*Where:* `common/common.cpp` (`common_context_params_to_llama`, `common_speculative_n_max`, the fit
call) and `common/speculative.cpp` (`common_speculative_init` already has a `n_max_effective` path, so
the controller can be told the cap).  Needs one VRAM estimate for the RS buffer, which is computable
from the model hparams + `n_seq_max` + `n_rs_seq` without a trial allocation.
*Risk:* low — the only behavior change is a lower draft depth on a config that would otherwise crash.
It does **not** touch the rollback path, so it needs no purity gate beyond "the effective ceiling still
loads and the MTP gate holds at the effective depth".
*This is the item to implement if we want the delivery to be self-explanatory on smaller cards.*

### L3 — lazy / shared snapshot planes (structural; `llama_memory_recurrent` redesign)
Allocate a slot's `(1 + n_rs_seq)` planes only when it actually spec-decodes, or share one snapshot set
across slots that never verify concurrently.  A single-user 4-slot server uses one active cell, so this
reclaims up to `(n_seq_max - 1) * (1 + n_rs_seq)` planes — **≈ 5835 MiB at ceiling 12 / 4 slots**.
Blocker: the graph allocator sizes `r_l[i]`/`s_l[i]` once (`mem_size * (1 + n_rs_seq)` rows, one tensor
per layer), and the graph builds views over that fixed extent, so "lazy" means either per-cell tensors
allocated on first use (a real redesign of `llama_memory_recurrent` + `llama-graph`) or a documented
constraint that only one sequence may speculate at a time.  Multi-sequence speculative decoding is the
reason the per-cell snapshots exist at all, so this must not silently break it.

### L4 — recompute-on-rollback (algorithmic; removes the `(1 + n_max)` factor)
Keep only the pre-batch state (the live plane), and on a partial accept **re-run the attention-free GDN
scan** over the accepted prefix to rebuild the state.  Removes the `(1 + n_max)` factor entirely (4x at
ceiling 12).  Cost: one extra GDN pass over `<= n_max` tokens per partial accept (the GDN is a fraction
of the verify cost, but partial accepts are common).  Purity: the recomputed state must be
**bit-identical** to the snapshotted one, which the deterministic `tests/test-recurrent-state-depth`
sweep (`n_rs_seq` 1..15, every rollback, deep drafts) can gate.  Biggest win, biggest change.

### L5 — f32 -> bf16 snapshot planes (opt-in; the maintainer's requested measurement)
The `S` planes are 7488 MiB of the 7781; storing the **snapshots** in bf16 would halve the dominant term
(`S` 7488 -> 3744 MiB; total 7781 -> 4037 MiB at ceiling 12 / 4 slots, or -> ~1010 MiB at 1 slot).

This is a **purity trade, not a free win**: the rewind would restore a bf16-rounded state instead of the
exact one, so `--spec-type none` and `draft-mtp` would no longer be bit-identical after a partial accept
(the live f32 state remains the reference; only the rewind path rounds).  It is explicitly a candidate
for **memory-constrained users who accept the loss of purity for the headroom**, behind an opt-in env
(e.g. `LLAMA_RS_SNAPSHOT_BF16=1`), and it should be **measured before it is offered**, not assumed:

* primary metric: **MTP acceptance at pos 1 / mean accepted length** at the default depth (`n_max 3`) and
  at a deep fixed depth, against the f32-snapshot reference over the four-axis gate
  (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`), because a slightly-off rewind changes the next
  token distribution and can change acceptance in either direction;
* the `test-recurrent-state-depth` sweep would be *expected to fail* (it compares against a reference
  context that never rewound) — that is the point of the trade, so the gate has to be replaced by a
  bounded-error assertion (e.g. max relative state diff) rather than bitwise;
* throughput: the rewind copy halves, so decode should be a touch faster at the widths where a rewind
  happens.
If the acceptance impact is small, this is the cheapest large headroom win for exactly the constrained
scenario this follow-up exists for.

## 4. Recommendation

Keep L2 as the near-term, safe, user-visible fix (so a too-large ceiling degrades gracefully instead of
crashing), and keep L3/L4 as the structural R&D.  Measure **L5** (f32 -> bf16 snapshots, opt-in) as the
maintainer asked — it is the only lever that buys memory with a *bounded, user-accepted* fidelity cost
rather than a redesign, and its acceptance impact is a measurable number we do not yet have.
