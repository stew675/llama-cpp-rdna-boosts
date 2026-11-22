# PROMOTION — what shipped, and how it differs from the PR

`v17-919794500-r338` (block 09).  The brief is `README.md` (verbatim from PR #342); the measurement record is
[`benchmarks/2026-11-21-ffn-even-split.md`](../../benchmarks/2026-11-21-ffn-even-split.md); the rules are
`GREEDY-PURITY.md` §104 and `INVARIANTS.md` r338; the release narrative is `WORKLOG.md` 261121.017.

## 1. Shipped OPT-IN, not default-on

The mechanism is taken as proposed: `get_split_granularity`'s FFN branch lowers the granularity from
`lcm(blck_size, 128)` toward `lcm(blck_size, 64)` when the split-axis size divides by the device count and the
`-ts` is absent or equal, so a 640-wide expert set at two devices splits **320 / 320** instead of **256 / 384**.
The floor of 64 is MMB's (`mmb_supported_mm`/`_mmid` decline `K % 64 != 0`), and gate/up/down keep identical
boundaries because they all take `blck_size` from the down (`tensor_axis_0`).

It is **OFF by default** because the granularity IS the reduction's grouping, so enabling it moves the logits of
every shape it reaches.  On the delivery's own 2-device Flash-Next point:

| | mean KLD | same-top-p |
|---|---|---|
| the feature | **0.013963** (gate limit 0.005) | **96.467 %** (floor 98 %) |
| control: same kernel, `-ub 4096` vs `-ub 3840` | 0.015248 | 96.516 % |

Below the point's own reorder floor, and the change's PPL effect (+0.59 %) is half the control's (+1.21 %) — a
reordering, not a defect.  But it is 2.8x the gate's rounding-level limit, and a default run must not silently
carry that, so the maintainer's decision was opt-in with the default path untouched.  After the promotion the
default reads **-0.000000 / 100.000 %** and its greedy text is unchanged (`c420dfe04c42`).

## 2. Reached by name: `-of tp-ffn-even`

This PR is what made the delivery decide how an opt-in feature should be reachable at all: an environment
variable is nearly invisible to a user who reads `--help`.  So the same release adds a generic
`--optional-features` / `-of TAG[,TAG...]` option, whose first (and currently only) tag is `tp-ffn-even`:

```bash
llama-cli -of tp-ffn-even ...          # or --optional-features tp-ffn-even, or LLAMA_TP_FFN_EVEN=1
llama-cli -of list                     # print the tags and what each one does
```

A tag sets the switch the feature already reads, so the environment variable keeps working on its own — the four
spellings are asserted to produce one text by `scripts/gate-ffn-even-split.sh`.  Every tag enables (there is no
"off" form; you drop the tag) and wins over a switch given at the same time.  The table in
`common/common.cpp` is the single place a tag exists, so a feature that later earns a first-class option deletes
its row; migrating the delivery's other environment-only opt-ins is `TODO` #363.

## 3. What it measured here (gfx1201, ROCm 10.2, 2 x R9700, `-ncmoe 47 -lme 47 --load-mode dio -ub 4096`)

* **+7.1 % `draft-mtp` decode** (70.4 -> 75.4 t/s, 2 reps/arm) and **+4.2 % plain prefill** (1498 -> 1560 t/s) on
  the 2-device Flash-Next host-expert shape — the contributor's claim reproduces in the shape it was made in.
* **+9.1 % prefill** on the 2-device Flash-Next IQ4_NL shape; **~+1.5 %** at 3 devices (`gpt-oss-20B`).
* Plain greedy decode is flat-to-slightly-down (-0.7 %), and prefill **with** a drafter is -2.3 %: a 320-row
  gate/up slice is not a multiple of 128, so it takes MMQ's bounds-checked variant and loses the fused gate/up
  kernel.  The balance still wins on the shapes above.

## 4. Three claims that were corrected

1. **The reach is not "Flash-Next on 2 GPUs".**  Enumerated from the GGUFs, the rule also fires for
   `gpt-oss-20B-MXFP4_MOE` at **3** devices (2880: 896/1024/960 -> 960 x3), `gemma-4-26B-A4B` at 3-4
   (2112/2816), `Inkling-2-Mini` at 2/4 (1152/2304), and the Flash-Next MTP draft models.  It is **inert** for
   every 3-GPU shape the delivery validates (35B-A3B 512-wide, 27B 17408-wide, 4B/9B/122B, DeepSeek-V4,
   Gemma4 12B/31B/E4B) and for Flash-Next at 3 and 4 devices.
2. **The 4-GPU example in the brief does not balance.**  "960 wide on 4 GPUs" is 240 per device, and
   240 % 64 = 48 — the rule correctly refuses it.  (The 768-wide/4-GPU example in the same paragraph, 192 each,
   does work.)
3. **The 64 floor protects a path MMB does not take on the affected MoE shapes.**  The routed MMB path is off by
   default (`routed=0`), and the census on this shape shows only the F32 routers reaching MMB, so the floor is
   currently defensive for dense shapes — correct, but not the thing that keeps the MoE down projection on MMB.

Also: the "+24 % on 8 short prompts" did not reproduce here (+4.2 % on a 5.2 k prompt), and the brief does not
say that the **default** would change numerics — the one thing that decided the opt-in.

## 5. Gate

`scripts/gate-ffn-even-split.sh` (`MANIFESTS.md` 37): the default is inert (no `FFN granularity` line, the
recorded default text), `-of tp-ffn-even` prints exactly one `128 -> 64 ... 320 / device` line and the recorded
opt-in text (asserted to DIFFER from the default), the four spellings agree, `-of nope` is refused with the tag
list, `-of list` exits 0 with it, and the 3-device 512-wide arm prints no line and reads identical text.
