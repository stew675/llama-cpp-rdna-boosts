# `prefill-chunk-adaptive` — the block-16 admission chunk applies only while a decoder is waiting

**PROMOTED into block `0016` at r306 (2026-11-17) and archived here.**  The delivered code is
`tools/server/server-context.cpp` + `common/common.h` + `common/arg.cpp` inside `patches/0016-…`; the knob is
`-ib N` / `--interactive-batch-size N` (`LLAMA_SERVER_PREFILL_CHUNK` = deprecated alias),
catalogued in `ENVIRONMENT.md` §22; the gates are `MANIFESTS.md` 30/31 (`scripts/gate-prefill-fairness.sh`,
`scripts/gate-prefill-duty.sh` — 31 grew the two `-lv 5` width passes that enforce the rule); the record is
`WORKLOG.md` 261117.005 and `INVARIANTS.md` r306; the open gap is `TODO` #344.  This directory is the
measurement record — nothing here is applied by anything.  The prototype (branch `sched-chunk-contended`,
commits `0fc5bb60c` + `754f2a299`) is kept as `0001-…patch`; the delivered block supersedes it (it carries the
same hunk plus the option).

## 0. The defect, in one paragraph

`LLAMA_SERVER_PREFILL_CHUNK` (`ENVIRONMENT.md` §22) caps the prompt tokens admitted per step, and r300 applied
it **unconditionally**: `unset` = one `n_ubatch`, `0` = the whole `n_batch`, `N` = explicit. The cap exists to
bound a *decoding* slot's inter-token latency and the worst-case wait of a request arriving behind a running
prompt — "a batch is assembled once per step and cannot be joined mid-flight". With **nothing decoding** it
bounds nothing: the only work in flight is the prompts. It is not free, either — a step re-pays its fixed cost
`b` once whatever its width, so a chunk of `n_ubatch` costs `n_batch/n_ubatch` times that `b` per prompt. On
the field shape (`archive/work/session-scheduling/README.md` §4.2.1) `b ≈ 196 ms` against `b ≈ 2 ms` on the
dense 4B, and the measured cost of the r300 default was **1454 t/s against 3036 t/s** for the same 65536
tokens — **half of uncontended prefill**, for a bound nobody was waiting on.

That is also the shape a user hits when they lower the chunk to make decode interactive: the lever that fixes
decode responsiveness is the lever that destroys bulk prefill, so the two settings cannot both be right.

## 1. The rule

> The per-step chunk is a **contended** cap. A step that carries **no decode work** takes the whole logical
> window (`n_batch`, ubatch-split internally). The number of concurrent **prompts** is irrelevant — several
> prompts prefilling with nothing decoding is still the uncontended case, and the window is shared max-min
> fairly either way.

This gives the bimodality the setting was missing: an interactive decoder keeps its `chunk`-wide stall bound
while it is decoding, and bulk prefill (batch ingestion, a document/RAG upload, a fresh conversation on an idle
server, four requests arriving together) runs at the full window.

**The test is "some slot owes a decode token", not `batch.size() == 0`.** `generating.push_back()` is skipped
when `can_batch_with()` fails (`server-context.cpp:4386`) — two LoRA classes still serialise — so a generating
slot can be left *out* of this step's batch and `batch.size()` would read zero while a decoder is in fact
waiting. Audio/TTS slots are not in the shared batch either. Both cases stay chunked here, which is the safe
direction.

**A wide step never carries a decode token, by construction**: the decode token is added to the batch *before*
the budget is computed (`handle_last_sampled_token` → `n_batch_decode`), so the instant a decoder has work the
cap is the chunk again. A decoder's first contended step is therefore already chunk-bounded; the only latency
the rule can add is waiting out an in-flight *uncontended* step. That is the traded cost and the one thing a
gate must assert (§5).

## 2. The change (prototype)

`tools/server/server-context.cpp`, in `update_slots()` — the `want`/`cap` selection:

```cpp
bool decode_active = false;
for (const auto & slot : slots) {
    if (slot.state == SLOT_STATE_GENERATING) { decode_active = true; break; }
}

// -1 = follow the ubatch (the default), 0 = no cap, >0 = explicit
const int32_t want   = prefill_chunk > 0 ? prefill_chunk
                     : prefill_chunk < 0 ? (int32_t) n_ubatch
                     : n_batch;
const int32_t cap    = decode_active ? std::min(n_batch, want) : n_batch;
const int32_t budget = std::max(0, cap - (int32_t) batch.size());
```

plus the member comment and one census field (`..., duty = 16, no-decode = n_batch`). Every other line of the
path — the fair share, the skew, the one-token floor, the duty cycle, the slot-order kill-switch, the
`use_mixed_batch()` opt-out — is untouched.

**No new variable.** `LLAMA_SERVER_PREFILL_CHUNK` is its own switch: `unset` = one `n_ubatch` while decoding,
`0` = no cap at all, `N` = explicit. The only behaviour that stops being reachable is "cap a step that has no
decoder waiting", which *is* the defect; the r301 arm is simulated with `0` plus a smaller `-b`/`-ub`.

## 3. What moves and what does not

* **Moves:** a lone prompt / several prompts with nothing generating — from `chunk`-wide steps to `n_batch`-wide.
* **Does not move:** anything with a decoder in flight. With `DUTY=16` the prefill step of a cycle always
  carries decode tokens, so a contested prefill is chunked exactly as at r301 — this change does **not** make a
  decoder's smoothness better, and nobody should expect it to.
* **Interaction with the duty cycle:** the time bound is `prefill_step_us` = the previous prompt-carrying
  step's wall, so the *first* cycle after a decoder arrives is budgeted on the uncontended (wide) step and every
  later one re-calibrates on chunked steps. Harmless, and in the decoder's favour.
* **Numerics:** a lone prompt's ubatch boundaries are unchanged (consecutive `n_ubatch` blocks), so its greedy
  text must be byte-identical. A *multi-prompt* wide step interleaves the sequences across ubatches
  (`split_equal`), i.e. a different FA shape — judged on throughput and `gate-prefill-logits.sh`, **never** by
  text (`GREEDY-PURITY.md` §63).

## 4. Measurements

Box: one R9700 (gfx1201, 34 GiB), ROCm 10.1, `-np 1`, greedy, one request at a time unless stated.
`ab.py` reads the `-lv 5` `fair share: added=` line, so the step WIDTH is measured, not inferred.

### 4.1 The mechanism (this is binary, not a judgement)

**Qwen3.5-4B Q8_0, `-b 2048 -ub 512`, lone 8401-token prompt, nothing decoding:**

| arm | binary | `CHUNK` | prompt steps | max `added` | ms | t/s | text |
|---|---|---|---|---|---|---|---|
| `old-512` | r305 | 512 | **18** | 512 | 1177 | 7139.5 | `332/07a82d55e623` |
| `old-0` | r305 | 0 | 6 | 2048 | 1176 | 7144.8 | `332/07a82d55e623` |
| `new-512` | prototype | 512 | **6** | 2048 | 1178 | 7129.3 | `332/07a82d55e623` |
| `new-0` | prototype | 0 | 6 | 2048 | 1175 | 7151.9 | `332/07a82d55e623` |

The prototype's `CHUNK=512` step pattern is **identical to the old `CHUNK=0` arm** — the chunk stops applying when
nothing decodes — and `new-0` == `old-0`, i.e. the kill switch (`0`) is exact. Text is **byte-identical in all four
arms**, including across the two binaries: a lone prompt's ubatch boundaries really are unchanged by the wider
window (§3).

**Qwen3.6-35B-A3B Q8_0, 1 GPU, `-ncmoe 16 -fa on`, 5601-token prompt** (a host-expert shape, to try to reproduce
the field `b`): `old-512` 12 steps / 2488 ms / **2251.3 t/s**, `old-0` 5 steps / 2479 ms / 2259.6 t/s,
`new-512` 5 steps / 2477 ms / **2261.2 t/s**, `new-0` 5 steps / 2480 ms / 2258.5 t/s.

### 4.2 So what is the throughput win here? **~0 on this box**, and that is the honest answer

Both reachable shapes fit `step = a·tokens + b` with **b ≈ 0** (4B: a ≈ 0.147 ms/tok, b ≈ −10 ms; 35B-host-expert:
a ≈ 0.475, b ≈ −36), because the experts are in a resident pinned pool and nothing is read or synced per step. The
measured chunk cost is therefore **0.08 % (4B) and 0.4 % (35B)** — i.e. the r300 record's own "dominates nothing
where `b` is small" prediction, not a refutation of it.

The win lives where the r300 record's fit says it does: **`b ≈ 196 ms`** on the field `runme` (Flash-Next,
PLE-on-CPU + streamed host experts + attendance pool), where the chunk 512 arm cost **1454 t/s against 3036 t/s**
for the same 65536 tokens (`archive/work/session-scheduling/README.md` §4.2.1). That shape is a 76 GB load on this
box and was **not re-measured here** — so the headline number is the r300 record's, applied to a mechanism this
session proved changes exactly the step widths above. It is still the best available estimate and it is the shape
the change is for, but it is a *carry-over*, not a new measurement.

### 4.2.1 Several concurrent PROMPTS, nothing decoding — this IS a win even here

The user's constraint: multiple prefills at once must not be subject to the interactive chunk. Not only is the
cap off in that case, the fair share makes a small chunk *worse* than "not applied at all" — a 512-token window
split four ways is 128 tokens per prompt per step.

**4B Q8_0, `-np 4`, four concurrent ~7000-token prompts, nothing decoding** (`multi_prompt.py`):

| arm | admittance per step (4 slots) | slot-steps | wall |
|---|---|---|---|
| r305, `CHUNK=512` | 79 + 105 + 139 + 186 = 512 | **177** | **4.89 s** |
| prototype, `CHUNK=512` | 2048 for the first step, then 316 + 421 + 560 + 748 = 2045 | **46** | **3.40 s** (1.44x) |

This config fits `b ≈ 49 ms` (four slots, one GPU: the per-step cost is not just `a·tokens`), which is why the
4x step count costs 1.4 s here even though the lone-prompt arm (§4.1) shows nothing. So the multi-prompt case is
the one that reproduces the field shape's mechanism at a small `b`, and it is the case the r300 c4 bench measured
at 1454 vs 3036 t/s with `b ≈ 196 ms`.

### 4.3 The contended path is provably untouched

`contended_width.py`, 4B, `CHUNK=512`, `DUTY=0`, a decoder on slot 1 already generating when the long prompt
arrives, `-lv 5` step widths for the prefilling slot:

| binary | uncontended step list | contended step list | contended wall |
|---|---|---|---|
| r305 | `512×9, 477, 512, 4` (12) | `511×9, 486, 511, 1, 4` (13) | 950 ms |
| prototype | `2048, 2048, 989, 512, 4` (**5**) | `511×9, 486, 511, 1, 4` (13) | 950 ms |

**The contended step list is identical, element for element**, and so is its wall time: with a decoder in
`SLOT_STATE_GENERATING` the prototype admits exactly what r301 admitted. Note the tail `512, 4` in the
uncontended arm — that is the existing prompt-checkpoint placement (`4 + n_ubatch`, `4`), not the chunk.

### 4.4 The two server gates, on the prototype

`gate-prefill-duty.sh --quick`: **PASS** — decode burst **16 → 137** tokens (8.6x), prefill stretch 1.01 → 2.19 s
(**2.17x**, bound 2.5x), lone prefill −0.1 %, single-request text identical, census parsed, 0 "no tokens to
decode", GPU faults clean. These are the r301 numbers (137 tokens, 2.13x) reproduced on the prototype, which is
the same result as §4.3 from the other direction.

`gate-prefill-fairness.sh --quick`: **PASS** — short-prompt TTFT 1.901 → 0.830 s (2.3x), ITL max 329 → **77 ms**,
lone prefill **−0.4 %** (was −2 % at r300 with the chunk applied unconditionally; the gate's baseline is already
`CHUNK=0`, so the change tightens this arm), text identical, aggregate decode +2.3 %.

### 4.5 The arrival case (the cost, stated)

A request that arrives while a wide step is in flight cannot be joined mid-flight, so its wait is one
**uncontended** step — `n_batch/chunk ×` the r301 bound (158 ms vs 73 ms per step on the 4B at 512). Once it is
generating, the chunk applies on the next step. On a shape with `b ≈ 196 ms` that arrival wait is
~`n_batch/chunk` × the chunk step's cost, i.e. the same class of stall r300 documented as the chunk's other
purpose. This is the trade, and it is why the rule must be "wide only when nothing is generating".

### 4.6 What the chunk actually buys the decoder (the reason it exists)

`scripts/gate-prefill-duty.sh --quick --env LLAMA_SERVER_PREFILL_CHUNK=<N>` on the prototype, `-np 2`, one
decoder on slot 1 and a 6000-token prompt on slot 0 (the gate's own instrument, so these are comparable to the
r301 record). "Delivered tokens/s" is the decoder's rate *while the other slot prefills*:

| chunk | duty | prefill wall | decode tokens delivered | delivered t/s | decoder ITL med/max |
|---|---|---|---|---|---|
| **512** | 0 | 1.00 s | **15** | 15.0 | 10 / **90 ms** |
| 2048 | 0 | 0.88 s | 7 | 8.0 | 10 / **276 ms** |
| **512** | 16 (shipped) | 2.18 s (2.19x) | **137** | **62.8** | 10 / **89 ms** |
| 2048 | 16 (shipped) | 1.48 s (1.69x) | 69 | **46.6** | 10 / **276 ms** |

So yes, unequivocally: the small chunk **more than doubles** the decoder's delivered rate while a prompt is
running (15.0 vs 8.0 t/s bare; 62.8 vs 46.6 t/s with the duty cycle on, 1.35x) and cuts its worst stall 3x
(90 vs 276 ms). The price is the prefill itself (1.00 vs 0.88 s bare; 2.18 vs 1.48 s with duty). The *median*
ITL is 10 ms in every arm — the win is in the tail and in the count, not the median, which is why a median-only
harness cannot see this feature at all.

**This is the asymmetry the adaptive rule removes**: the chunk's price was being paid on steps that had no
decoder to buy smoothness for. After the change, `CHUNK=512` costs nothing when nothing decodes (§4.1/§4.2.1),
so the interactive setting is strictly better than it was at r301 — same contended numbers as the table above,
no bulk-prefill tax.

## 5. The gates this needs

* `scripts/gate-prefill-duty.sh` — contended arms unchanged (decode-token burst, prefill stretch), the
  lone-prompt arm now also asserts the **wide** step, text identical. Its `CENSUS_RE` must learn the new field.
* `scripts/gate-prefill-fairness.sh` — its baseline is already `FAIR=0 CHUNK=0`, so it is unchanged; its
  candidate's "lone long prompt within 3 %" arm becomes ~exact, which tightens it. Same `CENSUS_RE` update.
* **New HARD assertion** (the invariant of §1): from the `-lv 5` `fair share:` lines, *every step whose batch
  carried decode tokens admitted ≤ `chunk` prompt tokens*, and the uncontended arm has steps > `chunk`.
* `scripts/gate-prefill-logits.sh` on the multi-prompt shape (not a text oracle), `scripts/check-gpu-faults.sh`.

## 6. Open questions

1. Is one step of extra decoder wait (`n_batch/chunk ×` the r301 bound, only when the step was assembled with
   nothing decoding) worth the prefill recovery? Measured in §4; the answer is the gate's `--max-itl-ms`.
2. The natural second step (NOT this one): size the contended chunk from the **step time** the duty cycle
   already records (`a·chunk + b ≈ c × decode_step`, Sarathi-Serve style) rather than a fixed `N`. The
   instrument exists; the fixed chunk is the two-point version of the same idea and should land first.

## 7. Promotion (done at r306)

Cut as a block-`0016` **amendment** (the block already owns the admission policy; no new block): `patches/`
regenerated, `release.json` at `v17-4f9dfa0e3-r306` (tip `a93c90ffc`, tree `f12b1a0c8`), `ENVIRONMENT.md` §22
rewritten (`chunk`'s row became `-ib`'s), `patches/README.md` blocks 16/12 + the r306 headline, `MANIFESTS.md`
gates 30/31, `INVARIANTS.md` r306, `TODO` #344 for the unverified host-expert numerics, and `WORKLOG.md`
261117.005.
