# `prefill-duty-cycle` — batch the prompt into one step per cycle so a decoder gets a burst

**WIP campaign, ACTIVE.  NOT part of the delivery.**  The code exists only on branch `sched-prefill-duty` in
`~/llama.cpp` (commits `77163bf79`, `014a7957c`), which sits on top of the r300 release tree.  It is not in
`patches/`, not in `release.json`, and not in any release.  Nothing applies it.

Handover prompt for a fresh session: [`HANDOVER-PROMPT.md`](HANDOVER-PROMPT.md).
`TODO` handle: **#340**.

## The problem it targets

r300 fixed prompt-versus-prompt fairness.  It did nothing for **prefill versus decode**: a decoding slot gets
exactly one token per step, so its rate is `1 / step_time`, and `step_time` is dominated by the prompt tokens
sharing that step.  Measured on the Flash-Next field shape (3 GPU tensor, `-b 16384 -ub 16384`, no chunk cap),
16k-token prefill against an active decode: **one decode token per ~4.5 s step = 0.25 t/s**, while the prefill
runs at its normal rate.

`LLAMA_SERVER_PREFILL_CHUNK` (r300) addresses that by making every step smaller, which spreads the interference
thin.  The duty cycle addresses it from the other end: leave most steps **decode-only**, and put the prompt into
one wide step per cycle.

## What it does

`LLAMA_SERVER_PREFILL_DUTY=K` (0 or 1 = off, K > 1 = one prefill step every K steps).  The cycle is `K-1`
decode steps followed by one prefill step, so the decode burst comes first and the stall comes last.  The
deferral is taken **only when the step already carries decode work**: a step with no prompt tokens and no
decode tokens is an empty batch, which `decode()` calls "no tokens to decode" and aborts on after four in a
row.  So an idle server, or a prompt arriving with nothing decoding, is prefilled immediately and the tick does
not advance.

Census line (`-lv 4`): `prefill admission: fair share = on, chunk = 16384, skew = 1, duty = 16`.

## Measured (2026-11-16, 3 GPU tensor, Flash-Next IQ3_XXS, `-b/-ub 16384`, chunk unset, decoder already generating, 16k prefill fired at +1 s)

| | prefill wall | decode tokens delivered while it ran | arrival times relative to the prefill start |
|---|---|---|---|
| `DUTY=0` | 4.7 s | 4 | `0.00, 0.02, 4.53, 4.65` (a 4.5 s stall) |
| `DUTY=16` | 4.8 s | **15** | `~20 ms apart`, then the burst step |

So roughly **4x the decode tokens for the same prefill wall time**, which is the `b` amortisation the design
predicts: a step costs `a*tokens + b`, `b ≈ 196 ms` on this shape, and the cycle pays it once instead of K
times.

Guard check: a bare 16k prompt with `DUTY=16` and **no** decoder completes in 4.2 s at 3862 t/s, with 0
"no tokens to decode" warnings and 0 aborts.

## Why the phase is not visible on this shape

`K=16` at a ~15 ms decode step is a 225 ms burst against a 4.5 s prefill, so "burst first" and "burst last"
differ by 225 ms inside a 4.8 s cycle.  The order only becomes observable when `K x decode_step` is comparable
to the burst, i.e. a much larger `K` or a slower decode step.

## Open questions (these are what a fresh session must settle)

1. **The arena.**  The duty cycle is a sharper wide/narrow alternation than r300's chunking, and r292/`TODO`
   #317 exist because that alternation costs prefill on host-expert shapes.  The `-lv 4` MoE census
   (`evictions=`, the residency line) and `scripts/gate-prefill-server-slots.sh` are the instruments, not the
   latency numbers alone.
2. **The `K` sweet spot per shape** and whether it is one number at all, given `b` varies by 100x between a
   dense 4B (`b ≈ 2 ms`) and a host-expert + PLE + pool shape (`b ≈ 196 ms`).  On the dense shape there is
   probably nothing to win.
3. **Speculative decoding composes.**  An MTP decode step already yields ~2.7 tokens, so the same `K` gives
   roughly `2.7 x (K-1)` tokens per cycle.  This was started and not finished, so there is no number.
4. **Default on or off.**  This is a *trade* (bursty p99, coarser admission for new arrivals), not a free win,
   so the repo's default-on policy needs an explicit decision rather than an assumption.
5. **Where it ships.**  Block 12 again, a temporary `03xx`, or a new `0016`.  `TODO` #339 says block 12 is
   already carrying two unrelated features and should be split, so folding a third mechanism in makes that
   worse.

## State of the two repos

- delivery `~/llama-cpp-rdna-boosts`, `main` at the r300 release commit; clean.
- fork `~/llama.cpp`, worktree on `sched-prefill-duty` with this experiment built into `build-rocm`; the
  `rdna-boosts` branch there is clean at `24211c158` (the released tip).
- `release.json`: `v16-4f9dfa0e3-r300`, tip `24211c158`, tree `4baf1b67`.
