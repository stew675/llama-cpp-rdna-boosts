# Bucketed adaptive-MTP — the alternative controller (issue #35)

This is the maintainer's **bucketed** draft-depth controller, ported onto the delivery's
block-01 call site, as the preferred candidate for the #35 tuning work.

* Provenance: `~/stew675/llama-master`, branch **`bucketed-adaptive-mtp`**
  (`dc6f2aaf7` "spec : adaptive MTP draft depth" + `1207c1c24` "carry bucket surplus/deficit
  across depth changes, gate the ngram feed").
* It **does away with the per-depth `climb_threshold`/`drop_pressure` tables and the hard
  `n_drop = 0` reset**: one credit bucket `B` accumulates `delta` per verification round —
  a full accept adds `max(1, n_accepted - 1)` (a deeper accept is worth more), a shortfall
  subtracts `depth - n_accepted`. `B` starts at `drop_pressure() = max(20, 4*depth)`; the
  depth climbs when `B >= drop_pressure() + 20` and drops when `B <= 0`, with the surplus /
  deficit carried into the next depth's bucket. It is a smoothed running estimate of the
  right depth rather than a table lookup.

## Apply

In the delivery worktree (`~/llama-cpp-rebase`):

```sh
cd ~/llama-cpp-rebase
git apply ~/llama-cpp-rdna-boosts/wip/adaptive-mtp-ceiling-scaling/bucketed-port/port.patch
cmake --build build-rocm --target llama-cli -j16
```

`port.patch` replaces `common/speculative-adaptive.h` with the bucketed version and adapts
`adaptive_feedback()` in `common/speculative.cpp` to the bucketed `update(n_accepted, ...)`
signature (the delivery's `n_last`/replay guard and the own-draft-only feed are kept).

## Measured (Qwen3.8-27B **Q8_0**, 2-card `-sm tensor -ts 1/1`, f16 KV, `-n 3000`, 1 run/cell)

Four-axis vs fixed MTP-3, ceiling 12:

| axis | fixed n3 | bucketed adapt12 | vs fixed3 | (table adapt12, for reference) |
|---|---:|---:|---:|---:|
| reasoning | 58.3 | 57.8 | −0.9 % ✓ | 57.7 (−0.5 %) |
| prose | 73.2 | 78.8 | +7.7 % ✓ | 76.5 (+5.1 %) |
| code | 80.3 | 92.7 | +15.4 % ✓ | 91.3 (+13.4 %) |
| recall | 87.0 | 119.6 | +37.5 % ✓ | 124.9 (+43.7 %) |

**Verdict: the bucketed base is better than the table on prose and code, worse on recall —
but it still violates the newest constraint.** Code adaptive `n12` = 92.7 vs code adaptive
`n7` = 96.6 (bucketed cap-7), i.e. **C(n12) < C(n7)**.

## Why it under-performs (measured dynamics)

* **The ramp dominates.** Recall produces only ~61 *fed* verify rounds for 3000 tokens (high
  acceptance ⇒ many tokens/round, plus replayed rounds the controller is not fed on), and the
  climb from floor 3 to 12 costs ~39 of them. Its time-weighted mean depth is only ~7.5 even
  though it reaches 12. Code's distribution is spread across 3–12 (mean ~7.8).
* **Depth 10 is the code optimum, and the bucket's equilibrium is ~10, but it never *holds*
  it.** Pinned-depth throughput (same cell): 7 → 97.0, **10 → 99.5**, 11 → 98.0, 12 → 94.5 t/s.
  Adaptive at a mean depth of ~10 still only reaches ~90–92 because the controller wanders
  (and the profile of an adaptive run differs from a pinned one).
* **Reasoning is safely pinned by the negative drift at the floor**: at depth 3 its
  `E[delta] = 0.35*2 − 0.65*(3−0.87) ≈ −0.69`, so the bucket clamps at 0 and the depth stays.

## Tuning tried this session (all on the reporter's cell)

| variant | R | P | C(n12) | C(n7) | K | notes |
|---|---:|---:|---:|---:|---:|---|
| table +1 climb | — | — | 91.6 | 95.3 | — | no fix |
| table, partial relief (`n_drop -= 1`) | — | — | 91.5 | 94.2 | — | no fix |
| bucketed base | 57.8 | 78.8 | 92.7 | 96.6 | 119.6 | candidate base |
| bucketed, credit `(d−1)+(d−1)²/8` | 61.1 | 79.2 | 90.3 | 96.8 | 123.6 | helps R/P/K, **overshoots C to 12** |

The depth-weighted credit confirms the tension: anything that makes the climb easier also
lifts the code equilibrium **above** its ~10 optimum toward 12 (where the wide verify is
expensive). The lever is not the climb *rate* but the controller's **spread/stability**.

## Next steps for the tuning session

1. **Stabilise, then speed.** The win is keeping code pinned near 10 (pinned 99.5 vs the
   adaptive ~92) while letting recall ride at 12. Consider a depth-dependent climb budget
   (large at the floor to protect reasoning, small above ~8 to finish recall quickly) and/or
   a hysteresis that widens with depth so the controller *holds* a level instead of oscillating.
2. Re-validate every variant against the **five** constraints simultaneously
   (R ≤ fixed3·1.03, P ≥ fixed3, C ≥ fixed3·1.10 **and C(n12) ≥ C(n7)**, K reaches 12 fast) at
   `-n 3000`, on **both** the reporter's cell (Q8_0 × 2-card tensor) and the delivery's
   reference (UD-Q4_K_XL × 1 card).
3. The tools in the parent directory: `four-axis.sh` (the 4-axis gate), `depth-trace.sh`
   (throughput + depth histogram), and the pinned-depth trick
   `--spec-draft-n-min-adaptive D --spec-draft-n-max D` to isolate the depth cost.
4. Re-derive `tests/test-speculative-adaptive.cpp` for whichever controller wins (the bucketed
   branch has its own 435-line test suite at `~/stew675/llama-master`).
