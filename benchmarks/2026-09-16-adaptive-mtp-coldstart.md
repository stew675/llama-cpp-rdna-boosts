# 2026-09-16 — adaptive-MTP cold start: `cap - 3` restored, `--spec-draft-n-start` added

**Outcome:** the brief move of the cold start to the floor/ceiling midpoint was measured and reverted.
The default is `max(floor, cap - 3)` again, and the start is now a runtime option.

`common_speculative_adaptive::reset()` (block 01) takes an optional `n_start`: `--spec-draft-n-start N`
(env `LLAMA_ARG_SPEC_DRAFT_N_START`) makes the first verify round of each generation start at `N`,
clamped to `[--spec-draft-n-min-adaptive, --spec-draft-n-max]`; unset (`0`) keeps the default `cap - 3`
bounded by the floor.  `tests/test-speculative-adaptive.cpp` covers the override and the clamp.

## Why the midpoint was wrong

The theory was that the `cap - 3` entry point made short reasoning/prose generations at a deep context
slow until the depth adjusted.  Same-build A/B (GPUs **1,2**, `-sm tensor -ts 1/1`, f16 KV, cap 12;
`reference-cell note:` the recorded 2026-09-15 tuning numbers were taken on this GPU pair — GPUs 0,1
measure ~2 % slower, which is the "fixed-n3 gap"):

Four-axis gate (shallow, `-n 3000`), adaptive cap 12:

| axis | cap - 3 (default) | midpoint | Δ |
|---|---:|---:|---:|
| reasoning | 60.7 | 57.7 | **−5.0 %** |
| prose | 80.9 | 80.3 | −0.7 % |
| code | 95.1 | 93.3 | −1.9 % |
| recall | 137.5 | 127.2 | **−7.5 %** |

~32k context (`-n 3000`): start 9 (default) **75.5**, start 7 71.6, start 3 70.5.
~32k context (`-n 300`, the short-form case): start 9 **71.0**, start 7 70.9, start 3 67.2.

End-to-end throughput tracks the round count `n/(1+mean_len)`.  The delivery's optimized multi-token
verify makes a wider batch cheap, so a lower start only lowers the settled mean depth and costs
rounds.  The "slow at first, then speeds up" effect is the **drafter warming up** (acceptance rises
with context), not the cold start.

## The pinned-depth oracle (shallow, `-n 3000`, cap 12)

Fixed `D` via `--spec-draft-n-min-adaptive D --spec-draft-n-max D`:

| axis | fixed 3 | fixed 5 | fixed 7 | fixed 8 | fixed 9 | adaptive (start 9) |
|---|---:|---:|---:|---:|---:|---:|
| reasoning | 57.7 | 57.5 | 51.6 | 48.3 | 48.9 | **60.6** |
| prose | 72.8 | **80.4** | 80.2 | 76.0 | 76.7 | 80.6 |

The controller beats the best fixed depth on reasoning and matches it on prose, so a fixed depth of 7
is **not** better.  The 2026-09-15 tuning record's `cap - 3` cold-start sentences remain accurate for
the current default.
