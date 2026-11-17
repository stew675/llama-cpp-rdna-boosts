# `prefill-duty-cycle` — batch the prompt into one step per cycle so a decoder gets a burst

**PROMOTED into block `0016` at r301 (2026-11-16) and archived here.**  The delivered code is
`tools/server/server-context.cpp` inside `patches/0016-…`; the knob is `LLAMA_SERVER_PREFILL_DUTY`
(**default 16**), catalogued in `ENVIRONMENT.md` §22; the gate is `scripts/gate-prefill-duty.sh`
(`MANIFESTS.md` gate 31).  This directory is the measurement record — nothing here is applied by anything.

`TODO` **#340** is retired by this record.  The campaign started from `wip/prefill-duty-cycle/` (branch
`sched-prefill-duty` in `~/llama.cpp`, commits `77163bf79`, `014a7957c`); the `wip/` entry was deleted with no
stub, and [`SESSION-BRIEF-2026-11-16.md`](SESSION-BRIEF-2026-11-16.md) is the brief that session was handed.

## 0. What the feature is, in one paragraph

A decoding slot gets exactly **one token per step**, so its rate is `1/step_time` — and a step that carries
prompt tokens is the slow one.  On the field shape (Flash-Next, 3 GPU tensor, one 16k-token step) a decoder
sharing the server with a 20k-token prefill delivered **5 tokens over 6.0 s**, at a 4.3 s worst gap: 0.25 t/s.
r300's per-step chunk bounds that stall; the duty cycle attacks it from the other end — leave most steps
**decode-only** and put the prompt into one step per cycle, so the cycle pays a step's fixed cost `b` once for
`K-1` decode tokens instead of once per token.

## 1. The design as shipped, and the one thing the measurements changed

`LLAMA_SERVER_PREFILL_DUTY=K`: up to `K-1` consecutive **decode-only** steps, then one step that carries prompt
tokens.  Two stop conditions end the burst, and either one is enough:

* `K-1` decode-only steps (`K` = the default **16**), and
* the idle time accumulated since the last prefill step reaching **that step's own measured duration**.

The second condition is the r301 design change, and it is the answer to the campaign's own open question 2.
The stretch of a concurrent prefill is

```
stretch = 1 + (K-1) * decode_step_time / prefill_step_time
```

which is **1.03x** on the field shape (one wide step, 15 ms decode steps) but **3.0x** on a dense 4B at a
512-token chunk and up to **16x** on a shape whose decode step is as slow as its prefill step.  `K` alone
therefore cannot be a default: the prefill stretch is unbounded in `K` and shape-dependent by 20x.  Capping the
*idle time* at the last prefill step's own duration bounds the stretch at ~2x **without any model knowledge** —
the budget is measured from the run, not configured.

The deferral is taken **only when the step already carries decode work**.  A step with no prompt tokens and no
decode tokens is an empty batch, which `decode()` calls "no tokens to decode" and aborts on after four in a row;
so an idle server, or a prompt arriving with nothing decoding, is prefilled immediately and neither counter
advances — which is also what calibrates the budget.

## 2. Measurements (2026-11-16, gfx1201 ×1 or ×3, ROCm 10.1, greedy, pinned slots, `-np 2`)

Both slots are **pinned** (`id_slot`) in every arm: `update_slots()` gives the lowest slot index priority and the
server assigns slots by LRU, so an unpinned two-slot arm measures whichever side of a race it gets (the
campaign that preceded this one had a 0.828 s vs 1.907 s pair from exactly that).

A "decode token delivered during the prefill" is an SSE arrival between the long prompt's request start and its
first generated token.  **`return_progress` changes this measurement**: a prompt-progress chunk carries
`tokens: [0]` with empty `content`, so a harness that counts "any chunk with content/reasoning/tokens" counts the
*arrival* progress chunk and collapses TTFT to the launch time (measured: 0.002 s "arrival" for a 52 ms
prefill).  Both gates now skip a chunk that carries `prompt_progress`.

| shape | duty 0 | strict cycle, `K=16` | shipped (time-bounded, `K=16`) |
|---|---|---|---|
| dense 4B Q8_0, chunk 512 (5601-token prompt) | 0.97 s prefill / **15** tokens | 2.86 s (2.96x) / 207 | **2.06 s (2.13x) / 126** |
| Flash-Next IQ3_XXS, streamed arena, 1 GPU, chunk 512 | 5.00 s / **9** | 8.42 s (1.68x) / 129 | **7.90 s (1.58x) / 109** |
| Flash-Next IQ3_XXS, 3 GPU tensor, `-b/-ub 16384`, chunk off, 20.3k prompt | 6.02 s / **5** | 6.56 s (1.09x) / 36 | **6.64 s (1.10x) / 41** |
| Qwen3.8-27B IQ3_S + `draft-mtp n3`, 1 GPU, chunk 512 | 5.41 s / **46** | 13.26 s (2.45x) / 879 | **11.50 s (2.13x) / 684** |

The gate (`scripts/gate-prefill-duty.sh`, its own arms and a 6000-token prompt) reads **16 -> 137** on the dense
shape, **46 -> 200** on the 27B + `draft-mtp n3` shape (acceptance 234/269 -> 233/271: two verify events differ
because the number of decode steps per cycle differs — judged for collapse, not equality, `GREEDY-PURITY.md`
§63) and **9 -> 109** on the streamed host-expert shape.  Greedy text is **byte-identical** in every arm; a lone
prompt (nothing decoding) is prefilled within 1 %; **0** "no tokens to decode" lines in every candidate arm.

## 3. The five open questions the campaign left, settled

**1. The arena — does a sharper wide/narrow alternation cost the expert cache?  No, measured.**  On the streamed
host-expert shape (`MOE_EXPERT_CACHE_MIB=8192`, `--n-cpu-moe 47`, 8 127 MiB arena of a 76 GiB model, so it really
streams) the `-lv 4` MoE census is *unchanged* by the duty cycle:

| arm | `h` | fills | evictions | hit |
|---|---|---|---|---|
| duty 0 | 0.7358 | 42 955 | 28 291 | 73.58 % |
| strict cycle | 0.7353 | 41 936 | 27 272 | 73.53 % |
| shipped | 0.7345 | 41 821 | 27 157 | 73.45 % |

The prefill rate falls by exactly the stretch (563 -> 360 t/s at 1.68x), which is the *idle* budget, not the
cache: fewer evictions, the same residency, the same table count.  `TODO` #317's warning is about a prefill that
*evicts and re-arms* the arena (a slab band move), which a step-shape change does not do.  This is the census
answer the campaign asked for, and it is why no arena-side change was made.

**2. The `K` sweet spot — there is no single number, so the bound is time, not `K`.**  `b` spans ~2 ms (dense 4B)
to ~196 ms (the Flash-Next field shape), which makes the stretch above range from 1.03x to 16x for one `K`.  The
default is **16** with the time cap; `K` is now only the burst *length*.  The maintainer asked for a default of 16
and got it, because the cap — not the number — is what makes it safe.

**3. MTP composition — it composes, and the drafter is why the win is larger.**  684 tokens delivered during an
11.5 s prefill against 46 with the duty off (14.9x) on the 27B + `draft-mtp n3` arm; acceptance moves by two
events out of ~680 (0.98532 -> 0.97380: the verify batch's composition changes with the number of decode steps
per cycle, `§63`), and the greedy text is byte-identical.  A drafter already yields ~2.7 tokens per step, so the
same `K` buys ~2.7x more.

**4. Default on or off — ON, at 16, with the time cap, and that is a policy decision, not a measurement.**  It is
a trade (a concurrent prefill's wall time against a decoder's smoothness), so it needed the maintainer's call; the
repo's default-on rule applies once the trade is bounded, and the cap bounds it at ~2x.  `DUTY=0` or `1` restores
the r300 behaviour exactly.

**5. Where it ships — block 16, with the fair share** (`TODO` #339).  See §5.

## 4. One thing this campaign established that is NOT about the duty cycle

`TODO` **#338**'s two parts were closed in the same session, and both are recorded in
`archive/work/session-scheduling/README.md` §6.2 and in the gate itself:

* **the fairness ratio bound is derived, not 3.0.**  The short prompt's wait is its own admission steps × the
  shape's per-step cost, measured from its `return_progress` chunks and its own `prompt_ms`; the ratio bound is
  `min(3.0, 0.8 × the baseline's wait / that floor)`.  On the streamed host-expert arm the arm reads **2.6x**
  where the old fixed 3.0 rejected it — and `L/(k·chunk)` promises 2.9x there because it assumes the baseline's
  step is *chunk*-wide and its fixed cost is zero; it is `n_batch`-wide, and its `b` does not scale with it.
* **the attendance-pool arm's "2.6x against a 5.7x ceiling" does not reproduce, and the carrier is the pool's
  per-step cost, not the scheduler.**  The written-down discriminating experiment (time slot 1's admission from
  the `-lv 5` `fair share: budget=…, added=…` line) shows slot 1 admitted on the **first step after its
  arrival** in both the pooled and the unpooled arm (`budget = 7, added = 3`, then `4` for the checkpoint tail).
  Same prompt (11 201 tokens), same rig, pinned slots:

  | arm | short TTFT | ratio | its own `prompt_ms` (2 steps) | per step |
  |---|---|---|---|---|
  | pooled, r299 fill (`FAIR=0 CHUNK=0`) | 17.913 s | 1.0 | 1 963 ms | 982 ms |
  | pooled, `FAIR=1 CHUNK=512` | 3.353 s | **5.34x** | 2 289 ms | 1 144 ms |
  | unpooled, `FAIR=1 CHUNK=512` | 2.741 s | **6.54x** | 1 755 ms | 878 ms |

  So the arm is at its floor (3 steps), the pool costs **1.30x per step** (a spilled-cell page read that does not
  shrink with the chunk), and `L/(k·chunk) = 7.3x` was never this shape's ceiling.  The gate now reports such a
  shape as "no room to improve" and judges it on ADMISSION instead of failing it for a ratio it cannot reach.
  The single-slot text oracle also had to move to a fresh server per arm: on a pooled cache the KV cells follow
  the earlier requests' page residency, so a near tie flipped `[P1]` between two arms that did nothing wrong
  (and read identically on fresh servers).

## 5. Where it shipped (`TODO` #339)

Block 12 carried the r300 fair share as an amendment.  r301 splits it out into a new **official block `0016`**
("the server's prompt admission policy"), with the duty cycle, rather than a `03xx` temporary: the policy is
settled and gated (two gates, one with a `--self-test`), it is a default-on product behaviour, and `03xx` means
"expect to drop it".  Block 12 goes back to being the campaign remainder; blocks 00-11 and 13-15 are
byte-identical, and the applied tree after `0016` is identical to the pre-split tree plus this campaign's change.
`n_blocks` 16 -> 17 (`v17-4f9dfa0e3-r301`).

## 6. What this campaign did not do

* No multi-device duty measurement beyond the field shape's 3-GPU tensor arm (a duty burst and a tensor split's
  all-reduce interact only through the step's shape, but it is one arm, not a matrix).
* No `--kv-window` arm: the windowed KV is a per-step spill like the pool, so the same "no room to improve"
  expectation applies, unmeasured here.
* `scripts/gate-prefill-server-slots.sh` was **not** re-run: its default model (122B-A10B **Q8_0**) is not on
  this box (only Q4_K_XL), and the duty gate's streamed host-expert arm measures the same two-slot question
  (request A's decode while B prefills, with text identity) far more cheaply.  That is a gap in the release
  evidence, recorded here rather than papered over.
