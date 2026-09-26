# WIP — reasoning-aware adaptive-MTP cold start (NOT part of the delivery)

**Status:** experimental, measured, **not promoted**.  No delivery patch contains this.  Saved here
so the work is preserved for a later session.  Nothing under `wip/` is shipped (see `AGENTS.md`).

## What it is

An adaptive-MTP cold-start policy driven by the model's reasoning/thinking state:

- while the model is inside a reasoning block (`common_reasoning_budget_get_state()` is
  `COUNTING` / `WAITING_UTF8` / `FORCING`), the controller starts at `--spec-draft-n-reasoning N`
  (default **4**);
- when the model leaves the block (the reasoning end tag), the depth resets to
  `--spec-draft-n-start` / the default `cap - 3`, so the controller adapts to whatever content
  (prose / code / recall) follows.

The premise was that reasoning is floor-bound (pinned optimum 3-5) while code/recall are
ceiling-bound, so keeping the draft depth low while thinking and re-arming on `</think>` should help.
The measurements below do **not** support it in this build.

## Files touched (8)

- `common/common.h` — `common_params_speculative_draft::n_reasoning`
- `common/arg.cpp` — `--spec-draft-n-reasoning` (env `LLAMA_ARG_SPEC_DRAFT_N_REASONING`)
- `common/speculative.h` — `common_speculative_draft_params::in_reasoning`
- `common/speculative.cpp` — per-seq reasoning edge detection in the MTP `draft()`; resets the
  adaptive controller on the enter/leave edges
- `common/sampling.h` / `common/sampling.cpp` — `common_sampler_in_reasoning()` (queries the
  reasoning-budget sampler state)
- `tools/server/server-context.cpp` — fills `in_reasoning` from `slot.smpl`
- `examples/speculative-simple/speculative-simple.cpp` — one extra positional initializer field

## Applying

Applies on top of the delivered set at `d1d3c3396` (after `scripts/apply-all.sh`; the delivery
already carries the `cap - 3` default and `--spec-draft-n-start`):

```sh
cd <llama.cpp-checkout>
git apply /path/to/llama-cpp-rdna-boosts/archive/work/reasoning-aware-mtp/reasoning-aware-mtp.patch
cmake --build build-rocm --target llama-cli -j16
```

The reasoning state only exists when the reasoning-budget sampler is armed.  In `llama-cli` that
means passing a budget (the gate used `--reasoning-budget 65536`) or the server supplying
`reasoning_control`; with no sampler, `in_reasoning` is always false and the feature is inert.

## Measurement (2026-09-16, 2x R9700 gfx1201, GPUs 1,2, `-sm tensor -ts 1/1`, f16 KV, cap 12)

Shallow context, reasoning axis (`-n 3000`, `--reasoning on --reasoning-budget 65536`):

| `--spec-draft-n-reasoning` | t/s | acceptance | mean len |
|---|---:|---:|---:|
| 9 (the old / default behaviour) | **60.4** | 0.558 | 2.95 |
| 4 (the feature default) | 57.9 | 0.544 | 2.74 |
| 3 | 57.6 | 0.547 | 2.71 |

Full four-axis with the feature (`--reasoning off` for P/C/K, so it is inert there):
R 58.1, P 80.3, C 94.0, K 138.0.  P/C/K acceptance and mean length are **byte-identical** to the
no-feature baseline; only R moves, and it moves down.

Deep context (26.5k-token prefill, reasoning on, `-n 3000`): `n_reasoning=9` 52.6 t/s vs `4`
53.7 t/s (+2.1%) — **but the two runs generated different content** (3581 vs 3319 drafted tokens),
so this is not a clean comparison.

The reasoning prompt used never left the thinking block within 3000 tokens in these runs, so only
the enter edge was exercised; the reset-on-exit path was verified to engage but was not measured.

## Why it was not shipped

1. At cap 12 the verify batch crosses the 8-row kernel-family switch, so greedy output is
   depth-dependent and every comparison is content-confounded (the acceptance differs between arms).
2. In the purity range (`n_max <= 7`) the feature is a **no-op**: `cap - 3 = 4` already equals the
   proposed reasoning start, so the knob only acts where a clean measurement is impossible.
3. The shallow, same-config data (the least confounded) shows the low reasoning start is
   neutral-to-negative, consistent with the round-count finding: the descent from `cap - 3` raises
   the mean depth and reduces the number of target forward passes, which dominates in this build.

## If revisited

- Measure in the purity range (`--spec-draft-n-max 7`) with a reasoning start **below** `cap - 3`
  (e.g. 3) so both arms generate identical content; or gate the purity-impairing kernel family so a
  cap-12 comparison can be made cleanly.
- Use a prompt that actually leaves the thinking block (short think + long answer) to exercise the
  reset-on-exit path.
- Re-check the premise: "reasoning is floor-bound" (pinned optimum 3-5) does not imply "start low" —
  the transient matters, and in this build a higher transient is cheaper because the round count
  dominates.
