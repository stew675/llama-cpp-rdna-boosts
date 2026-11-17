# `session-scheduling` — fair, preemptive prefill admission in `llama-server`

**PROMOTED into block `0012` at r300 (2026-11-16) and archived here.**  The delivered code is
`tools/server/server-context.cpp` inside `patches/0012-…`; the knobs are catalogued in `ENVIRONMENT.md` §22;
the gate is `scripts/gate-prefill-fairness.sh`; the apply order and the block's verification are in
`MANIFESTS.md`.  This directory keeps the measurements, the design record and the two open items
(`TODO` #338 and #339) — it is not itself applied by anything.

Status at promotion: **design + first-cut prototype + 9 validated layouts, promoted with `SKEW` defaulting to
the measured equal share.**  The prototype was backed up as a four-commit series plus a net diff (§9: trees,
sha256, apply instructions, size of the change) and that series is what block 12 now carries, squashed.  The
§2 baseline numbers were taken on the delivery at `rdna-boosts` tip `c215c77f3` (r299 tree), gfx1201 ×1,
ROCm 10.1.0, one `Qwen3.5-4B-Q8_0` (`/llm/models/Qwen3.5/4B/Q8_0/`), greedy, `--cache-prompt off`.


## 0. What this campaign is, and what it is deliberately not

**P0 (the whole point).**  `--parallel N` should behave concurrently:

1. **No prefill starvation.**  Every active prompt is admitted to every step; a request's prompt must not
   wait behind another request's entire prompt.
2. **Bounded decode interference.**  A slot that is decoding must not inherit another slot's prefill chunk
   as its inter-token latency.
3. **One slot prefilling while another decodes** must be the normal steady state, not an accident of the
   fill order.

All three are reachable **without touching the KV cache, the kernels or the backend scheduler**: they are
decisions in `update_slots()`' batch assembly.  That is the campaign.

**Not P0, kept separate on purpose (§5).**  Paged KV / fragmentation, cross-slot determinism, and true
parallel *execution* (multi-context or multi-stream).  Each is a real question and each is recorded below
with its mechanism and its falsification test — but folding any of them into P0 would replace a
server-side scheduling change (days) with a memory-format or runtime change (weeks) that does **not** fix
the observed symptom.

## 1. Today's scheduler: one context, one batch, one thread

| what | where |
|---|---|
| one `llama_context` for the model; every slot points at it | `slot.ctx_tgt = ctx_tgt`, `tools/server/server-context.cpp:1717` |
| one sequence id per slot (`slot.id = i`), `n_seq_max = n_parallel` | `server-context.cpp:1716`; `common/common.cpp:2148` |
| `n_ctx_seq = n_ctx / n_seq_max` unless `--kv-unified` | `src/llama-context.cpp:689-694` |
| one scheduler thread runs the whole loop | `server_queue::start_loop`, `tools/server/server-queue.cpp:283` |
| per iteration: build **one** batch = union of all slots | `update_slots()`, `server-context.cpp:3883` |
| decode tokens are added **first** (one per generating slot) | `:4318` `handle_last_sampled_token(batch)` |
| then prompts are admitted, slots **in index order**, each filling to `n_batch` | `:4328-4335`, inner loop `:4831` |
| the batch is executed as one `llama_process()` per (ub)batch | `decode()`, `:5025`; `llama_process` `:5060` |
| no preemption anywhere | `grep -rni preempt src/ tools/server/` → nothing |

Two consequences that matter:

* **Decode is genuinely batched.**  All generating slots' tokens are in the same graph, so the step is
  shared.  This is real continuous batching (measured in §2.1).
* **Prompt admission is a non-preemptive, slot-ordered batch filler.**  The inner loop is
  `while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch)`: slot 0 takes the
  whole window if it can, then slot 1 gets what is left.  Nothing rotates the cursor, nothing caps a
  slot's share, and nothing withdraws an admitted token.  This is the defect (measured in §2.2-§2.4).

Once tokens *are* in the batch, the ubatch splitter is already fair: `llama_batch_allocr::split_equal()`
takes one token per sequence set per round, so a batch holding two prompts is split evenly between them
(`src/llama-batch.cpp:668-913`).  **The sharing primitive exists; the admission policy is what is missing.**
That is why P0 is small.

### 1.1 Are decode and prefill concurrent?  Three layers, and the middle one is a surprise

Measured with `LLAMA_BATCH_DEBUG=1` on a mixed step (one slot decoding, one slot prefilling):

1. **Same batch / same `llama_process()` call — YES.**  `update_slots()` adds each generating slot's sampled
   token first (`:4318`), then fills prompt tokens.  Both are in the one `common_batch`.
2. **Same ubatch / same GEMM — NO.**  The ubatches of that step were:

   ```
   tok=2   n_seq_tokens=1  n_seqs_unq=2     <- 1 decode token + 1 prompt token
   tok=512 n_seq_tokens=512 n_seqs_unq=1    <- prompt only
   tok=512 n_seq_tokens=512 n_seqs_unq=1    <- prompt only
   tok=443 n_seq_tokens=443 n_seqs_unq=1    <- prompt only
   ```

   The decode token does **not** ride the wide prompt ubatch.  `split_equal()` advances one token per sequence
   set per round and stops when the shortest set is exhausted, and a decoding sequence has exactly **one**
   token — so the mixed ubatch is 2 tokens, and the prompt then continues alone.  This is why `n_seq_tokens`
   is 1 there and 512 elsewhere.
3. **Simultaneous execution (overlapping kernels/streams) — NO.**  One `ggml_backend_sched`, one stream; the
   ubatches are dispatched back to back.

**The consequence is the campaign's central number.**  `decode()` calls `llama_synchronize()` after the *whole*
`llama_process()` returns when the step has outputs (`:5061`), so a decoder's token is only visible once every
ubatch of the step is done.  Its latency is **the step**, not its own ubatch — which is why the fair share
alone moved nothing (336 ms before and after: it does not change the step's prompt tokens) and the per-step
chunk moved it to 87-99 ms.

**Why your "pack them together" layout is not what happens — and where it does happen.**  Qwen3.5 is a
**hybrid** model (GDN + attention: `llama_memory_recurrent` per GDN layer), so it goes through
`llama_memory_hybrid::init_batch()`, which always uses `split_equal` (`src/llama-memory-hybrid.cpp:99`) — that
is the 2-token ubatch above, and `--kv-unified` changes nothing (verified: identical shapes).  For a
**pure-attention** model, `llama_kv_cache::init_batch()` picks `split_simple` when `n_stream == 1`, i.e. under
`--kv-unified` (`src/llama-kv-cache.cpp:2055`, `n_stream = unified ? 1 : n_seq_max` at `:541`), and
`split_simple` packs tokens **in batch order** — which is exactly "decode at 0, prompt at 1..511".  So the
layout is reachable; it is just gated behind a cache mode and unavailable on the hybrid models this repo runs.

**A follow-up this suggests (not pursued):** if the decode token sat in the *first* ubatch and its logits were
extracted right after that ubatch, ITL would be bounded by one ubatch without any cap.  That is what the
optional `0100` two-lane-prefill artifact does per lane (`extract_outputs` per scheduler).  The chunk cap
reaches the same bound far more cheaply, so this is recorded, not scheduled.

## 2. Measurements

Server used throughout (the `-b`/`-np` are the only variables):

```bash
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-server \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -np 2 -c 65536 -b 2048 -ub 512 -t 8 --no-warmup --port 8099
```

Prompts are synthetic (a repeated sentence, ~11 tokens per repetition) so the token counts are exact; a
real gate should use `prompts/`.  Greedy (`temperature 0`, `seed 42`), `cache_prompt: false`, both requests
fired from threads so they are admitted concurrently.  The harness is [`probe.py`](probe.py); the exact
arm for each number below:

| § | command | prompt size |
|---|---|---|
| 2.1 | `probe.py decode` | short prompt, 40 generated tokens |
| 2.2 | `probe.py prefill --reps 180` | 2004 / 2205 tokens |
| 2.3 | `probe.py prefill --reps 1200` | 13203 tokens each |
| 2.4 | `probe.py starve --reps 1200` | 13204 + 7 tokens |
| 2.5 | `probe.py itl --reps 1200 --n 200` | 13203 tokens, fired mid-decode |
| 2.6 | `probe.py tps --n 200` | ~3.3 k tokens |
| 2.7 | `probe.py agg` (server at `-b 16384`) | 8803 tokens solo vs 2 × 4403 together |

### 2.1 Decode is batched (the user-visible "concurrency" already exists here)

Two 40-token streams, `-np 2 -b 2048`: both finish in **0.56 s wall**, 40 tokens each, and the token
arrival timestamps interleave at the *same* instants:

```
A@0.095 B@0.095 A@0.110 B@0.110 A@0.124 B@0.124 A@0.137 B@0.137 B@0.148 A@0.149 ...
```

That is one forward pass per step serving both sequences.  A statement like "llama.cpp only ever runs one
session at a time" is therefore **not true for decode**.

### 2.2 Prefill is co-batched only while both prompts fit the fill window

| arm | tokens | TTFT | prompt eval |
|---|---|---|---|
| A | 2004 | 0.68 s | 677 ms (2959 t/s) |
| B | 2205 | 0.65 s | 644 ms (3423 t/s) |

Both prompts were processed in the **same ~0.65 s window** (`-lv 5` shows one `decode:` per step, and both
`print_timing` lines land 12 ms apart).  They co-batch because A's prompt ran out before the 2048-token
window filled, leaving room for B.

### 2.3 Above the fill window, prompt 2 waits out prompt 1

| arm | TTFT | total |
|---|---|---|
| one 13203-token prompt alone | 1.90 s | 1.93 s |
| two 13203-token prompts | 2.60 s / **3.86 s** | 3.55 s / 3.90 s |

The second request's prefill starts after the first has consumed the 2048-token window ~6 times.  Total
work is unchanged; the *second* request simply has no share until the first is finished.

### 2.4 Admission is slot-index priority, and `-b` does not bound it

13204-token prompt + a 7-token prompt, second request fired 50 ms later:

| admission order | long TTFT | short TTFT | short `prompt_ms` |
|---|---|---|---|
| **long admitted first** (slot 0) | 1.99 s | **1.91 s** | 270 ms (reported 24 "t/s") |
| **short admitted first** (slot 0) | 1.96 s | **0.026 s** | 24 ms |
| long admitted first, `-b 512` | 2.00 s | **1.92 s** | 293 ms |

Two things to read off this:

* Who goes first is **admission order, i.e. slot index** — not length, not arrival-time service.  A 7-token
  prompt can wait 1.9 s.
* **`-b` does not change the block.**  At `-b 512` the long prompt grabs 512 tokens per step instead of
  2048, and the short prompt still waits ~1.9 s.  Lowering `-b` makes the monopoly finer-grained, not
  fairer.  There is **no knob** today that shares the batch between prompts.

The `prompt_per_second: 24` on the short request is the log-reading trap: `prompt_ms` is per-request wall
time, so a 7-token prompt interleaved into someone else's batch reports a nonsense rate.

### 2.5 A decoding slot's ITL is hostage to another slot's prefill chunk

One slot decodes 200 tokens; 350 ms in, the other slot is given a 13203-token prompt:

| | median token gap | during the other slot's prefill |
|---|---|---|
| decoder, no prefill | 10 ms | — |
| decoder while the other slot prefills | — | **266 / 281 / 294 / 305 / 317 / 331 ms** for 6 consecutive tokens, then 86 / 104 ms |

**26-33× the normal inter-token latency**, for as many tokens as the other prompt has 2048-token chunks.
The decode token *is* included in every step (that part already works); what it inherits is the other
slot's chunk duration.  This is the "not truly concurrent" a user actually feels, and it is the number P0
must move.

This is also why `scripts/gate-prefill-server-slots.sh` exists: it measures exactly this interference for
the expert-cache placement policy (`TODO` #317, block 16).  A fairness change is the same arm with a
different assertion.

### 2.6 Throughput scaling is *not* the bug

| concurrency | aggregate |
|---|---|
| 1 × 200 tokens | 82.6 t/s |
| 2 × 200 tokens | 120.5 t/s (**1.46×**) |

1.46× at concurrency 2 on one GPU is the memory-bandwidth wall, and vLLM shows the same shape on the same
card.  Do not read a sub-2× aggregate as evidence of missing concurrency — read §2.5 instead.

### 2.7 Co-batched prefill is throughput-neutral — the serialisation buys nothing

This is the arm that decides what a fair policy may cost.  `-b 16384` so the batch is not the constraint,
both requests admitted within **1 ms** of each other (`-lv 5`: the two `new prompt` lines, and both slots
report the same `prompt_ms` — a symmetric split, not a client race):

| arm | rows | wall | aggregate |
|---|---|---|---|
| 1 × 8803-token prompt | 8803 | 1.29 s | 6822 t/s |
| **2 × 4403-token prompts, together** | 8806 | 1.24 s | **7124 t/s** |
| 1 × 4403-token prompt | 4403 | 0.66 s | 6677 t/s |
| **2 × 2203-token prompts, together** | 4406 | 0.65 s | **6805 t/s** |

Two readings, and both matter:

* **Each request takes ~1.87× its solo time** (1.235 s vs 0.66 s) because the card is saturated — that is
  not a defect, it is the work.  The aggregate is conserved, and a bigger batch cannot beat it.
* **Co-batching two prompt sequences is throughput-neutral.**  The deltas above (+4.4 % / +1.9 %) are inside
  this box's run-to-run spread: re-running the same arm gave 2 × 2203 at **6893** t/s against a solo 4403 at
  **7067** (−2.5 %), and 2 × 4403 at 7098 against 6963.  So the honest statement is **within ±3 %**, and the
  only safe conclusion is the one that matters: **a second sequence in the batch costs nothing.**

So the serialisation the current fill policy imposes buys **nothing at all** — not throughput, not kernel
efficiency, not memory.  Two slots' rows ride the same layer GEMM at no extra cost, which is exactly the
shape the hardware prefers.  Today's slot-order fill is therefore a pure latency-distribution decision,
and §4.6 shows it is made in the direction that hurts the tail.

## 3. Reading the server log (why it *looks* like one session)

With `-lv 4`/`-lv 5` a run prints one `decode: n_batch (effective) = 2048, off = 0` line per **step**, and
that step is the union of all slots.  There is no per-slot "step" line, `prompt_per_second` is per-request
wall time (§2.4), and `slot print_timing: prompt eval time` is printed when the slot's prompt *finishes* —
so two prompts that ran in parallel print 12 ms apart (§2.2) while two that ran serially print ~2 s apart
(§2.3).  The instrumentation can distinguish them; the log layout just does not make it obvious.

For a bench harness like `llama-benchy`, the sequence at concurrency 2 with prompts above the fill window
is: request 2 admitted, no prompt progress, then request 1's prefill done, then request 2's prefill.  That
is §2.3/§2.4, and it is real — it is just *admission*, not execution.

### 3.1 `n_batch` is the admission granularity, not just the batch size

Three layers, and a request can only pass between them at one point:

```
update_slots() iteration
├── process_new_tasks()          <- a new request is admitted HERE (also on the worker thread)
├── assemble ONE common_batch    <- up to n_batch tokens, a flat list of (seq_id, pos) -- then FROZEN
└── decode() -> llama_process()
                 └── split into n_ubatch chunks, executed back to back
```

The `n_ubatch` chunks are a *subdivision* of a batch that is already fixed, so a request arriving mid-step
cannot join **any** chunk of that step — not the current ubatch, not a later one.  It joins the next
assembly.  What it *does* get immediately is a slot: `decode()` wraps `llama_process()` in
`yield_to_queue()` (`:5059`) and that worker runs `process_new_tasks(true)` (`server-queue.cpp:195`), so the
slot is launched and the HTTP headers go out while the GPU is busy.  **Noticed ≠ admitted to the batch.**

So the worst-case wait for a seat is the duration of one in-flight step, i.e.

```
worst-case admission delay  ≈  n_batch / prefill_rate
```

which is the same number as the decode ITL spike in §2.5: **~290 ms at `-b 2048`** (measured 266-331 ms),
**~2.3 s at `-b 16384`** (8803 tokens in 1.26 s).  That identity is the reason the per-step cap (§4.4) is
the primary knob and `n_batch` should be a ceiling rather than a target: at the stock batch sizes the
admission granularity is already the same 300 ms as the ITL jitter, so a cap that bounds one bounds both.

Worked example of the starvation this produces (measured, §2.4): long prompt in slot 0, 7-token prompt
arriving 50 ms later.  Slot 0 fills positions 0..`n_batch-1` on every step, the fill loop returns at the
`batch.size() >= n_batch` check before reaching slot 1, and this repeats until slot 0's *remaining* prompt
drops below `n_batch` (~1.64 s in, at step 6 of 7 for a 13204-token prompt at `-b 2048`).  The short
request then reports `prompt_ms = 270 ms` — it was never slow, it was *unseated*.  Total wait 1.91 s.

## 4. P0 — the change

### 4.1 Insertion point

Only `update_slots()` (`server-context.cpp:4328-5000`).  Keep everything else: one context, one batch, the
existing ubatch splitter, the existing `can_batch_with()` homogeneity rule.

### 4.2 Max-min fair share of the window — everyone is in every step

Two independent decisions, and they must not be confused:

* **who gets the window** — the policy, below;
* **how big the window is per step** — the chunk cap (§4.4), which is the ITL/admission bound.

**The policy is max-min fair share (water-filling): every active prompt slot is admitted to every step.**
Replace "fill to `n_batch` slot by slot" with:

```
budget   = max(0, n_batch - batch.size())     // after the decode tokens are in
active   = prompt slots that can join this batch (they have tokens left)
while active is not empty:
    share = budget / |active|                 // integer; 0 means the window is committed
    if share == 0: break
    for s in active, from a rotating cursor:
        take = min(remaining[s], share)
        admit[s] += take;  remaining[s] -= take
        budget -= take
    drop from `active` every slot with remaining[s] == 0   // its unspent share returns to the pool
```

Properties, and they are the point of the campaign:

* **Nobody hogs.**  A slot that needs less than its share returns the difference to the pool, and the pool
  is **redistributed equally** among the slots that can still use it — not handed to the head of the queue.
  The window is still fully used, so sharing costs no throughput (§2.7: a second sequence is free).
* **Everyone is in every step.**  With `7 + 20K` tokens at `-b 1024`: share 512 → the short prompt takes its
  7, the surplus 505 goes back and the long prompt takes 1017.  Both are admitted, window full.  The short
  request's wait is bounded by one step, which is exactly the property §3.1 says is missing today.
* **Equal-length prompts finish together** (§2.2 behaviour extended to any length), and a long prompt
  finishes no later than it does today in total.
* **Rotation makes the tie-break fair across steps**, so no slot is permanently first when several are
  clamped to the same share.

This is the same policy `split_equal()` already applies *inside* a batch (one token per sequence per
round), lifted one level to the decision of what enters the batch — so the two levels now agree instead of
fighting.

| policy | who is in the batch | mean TTFT | max TTFT | decode ITL |
|---|---|---|---|---|
| today: greedy block fill | slot 0 to exhaustion | best (equal-length) | unbounded by the peer's prompt (§2.4) | unbounded (§2.5) |
| **max-min fair share (this) + cap (§4.4)** | **all active slots, every step** | slightly worse (equal-length, §4.6) | **≤ one step** | **≤ the cap** |
| shortest-remaining-first, with aging | all active slots, biased to the small ones | best overall | ≤ one step | ≤ the cap |

### 4.2.2 `LLAMA_SERVER_PREFILL_SKEW` — the droop toward shortest-remaining-first (default 1 = off)

The fair share above is 1:1.  `LLAMA_SERVER_PREFILL_SKEW=N` weights **the slot with the shortest remaining
prompt** `N` times against the others (1..64, clamped; default 1).  It is *not* SRPT: every other slot keeps
advancing on every step, so it still gives feedback **and starvation stays impossible by construction** —
the one-token-each pre-pass plus strictly positive weights means no slot can be denied.  1 allocates exactly
what the plain share did, so it is bit-identical to the measured equal-share behaviour.

Measured (4B, `-np 4`, `-b 2048`, chunk = `n_ubatch` = 512, four prompts arriving together):

| prompt | skew 1 ttft / done | skew 5 | skew 20 |
|---|---|---|---|
| ~6000 | 2.201 / 2.236 s | 1.999 / 2.032 | 1.975 / 2.009 |
| ~3000 | 1.912 / 2.161 s | 1.356 / 1.595 | 1.267 / 1.503 |
| ~1500 | 1.266 / 1.542 s | 0.747 / 1.026 | 0.692 / 0.966 |
| ~400 | 0.223 / 0.543 s | 0.227 / 0.534 | 0.225 / 0.479 |
| **mean ttft** | **1.401 s** | **1.082 s** (−23 %) | **1.040 s** (−26 %) |
| **mean done** | **1.621 s** | **1.297 s** (−20 %) | **1.239 s** (−24 %) |

The result that matters: **the long prompts do not pay.**  Their TTFT improves too (2.201 → 1.975 s at skew 20)
because the short slots *retire* early and stop competing — the work is conserved, but the sharing set shrinks
sooner.  The shortest finishes marginally sooner (0.543 → 0.479 s) while every other prompt gains 10-26 %.
This is the "they accelerate in turn" effect, and it is why the skew dominates the equal share on *both* mean
and max, rather than trading one for the other.

So the ordering of the three policies on this shape is: **skewed share > equal share > FCFS** on mean, with the
skew also improving the long tail.  Pure SRPT would still be better on mean for the short jobs (it would finish
the 400-token prompt in one step) but it can starve; the skew is the safe part of that gain.

Harness: `./probe.py skew` runs the four-way heterogeneous mix (needs `-np 4`) and prints per-request ttft /
done plus the starvation check — run it once per skew value and compare.

**Max-min fair share is the default and the first thing to build.**  It is the policy that makes `--parallel
N` mean what a user expects, it needs no aging rule to be safe (a long prompt always receives its share, so
it cannot be starved), and it costs nothing measurable.  Shortest-remaining-first is a *follow-up*: it is
strictly better on mean TTFT, but it needs aging and a starvation bound to be defensible, and it is much
harder to write a gate for.  Do not conflate the two.

### 4.2.1 The chunk's cost model, and why the default is now one ubatch (llama-benchy, 2026-11-16)

`llama-benchy` on the field `runme` (Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS, 3 GPU tensor, `--parallel 4`,
`--batch-size 16384 --ubatch-size 16384`, attendance pool on, PLE on CPU), `pp16384 (c4)` = 65536 tokens:

| `LLAMA_SERVER_PREFILL_CHUNK` | total t/s | per-req t/s | ttfr | steps | s/step |
|---|---|---|---|---|---|
| 512 | 1453.6 | 364.6 | 40.6 s | 128 | 0.352 |
| 4096 | 2832.3 | 708.1 | 21.0 s | 16 | 1.447 |
| 16384 (= no cap here) | 3035.9 | 759.0 | 19.5 s | 4 | 5.398 |

Fitting `step = a·tokens + b` over the first two rows gives `a = 0.306 ms/token` and **`b ≈ 196 ms`**, which
predicts the third row within 4 %.  `b` is the **fixed per-step cost** — PLE on CPU, the host-expert tier and
the attendance-pool mirror, all read over PCIe once per step whatever its width.  On the dense 4B the same fit
gives **`b ≈ 2 ms`** (chunk 512 vs no cap on a lone 13.2 k prompt: 2.030 s vs 1.992 s, 19 extra steps,
38 ms of difference).  **`b` differs by 100×, and that is the whole story** — which is why "chunk = ubatch is
the sweet spot" was a single-GPU-dense conclusion that does not transfer.

**The ubatch width is the unit of work** (same shape, `-np 1`, one 16504-token prompt, experts resident,
`c` = 1).  This prices the "-b 16384 -ub 4096" recommendation directly:

| `-ub` | solo prefill | vs 16384 |
|---|---|---|
| 16384 | 3197 t/s | — |
| 8192 | 2921 t/s | −8.6 % |
| 4096 | 2828 t/s | −11.5 % |
| 2048 | 2736 t/s | −14.4 % |

2828 t/s at `-ub 4096` matches the c4 `chunk=4096` aggregate of 2832 almost exactly — the ubatch, not the
batch, is what the kernels see.  So buying a ~1.45 s decoder bound instead of ~5.4 s costs 7 % (c4) to 11.5 %
(c1) of prefill on this shape; the bound at `-ub 512` would cost ~half (§ above), which is why `b ≈ 196 ms`
sets a floor on how cheap a short chunk can be.

Two consequences, both now in the code:

1. **`chunk = n_ubatch` is a FLOOR, not a sweet spot.**  Below `n_ubatch` a step still issues one
   `llama_process()` and the same kernel launches, so it re-pays `b` for no reduction in launch count — it is
   dominated by construction.  The prototype's default therefore changed from `0` (off) to **follow
   `n_ubatch`**, with `0` now meaning explicitly off.  On the field `runme` (`ub == b == 16384`) that default
   is a **no-op that reproduces its best row**; on the 4B (`ub 512`, `b 2048`, `b≈2 ms`) it is the measured
   near-free point.
2. **A real ITL bound is only purchasable by lowering `-ub` as well.**  With `ub == b` there is no chunk that
   both binds and stays above the ubatch, so the coupling is honest rather than hidden: to bound a decoder's
   wait you must give up ubatch width, and the price is `(n_batch/chunk − 1) × b` per request's prefill.

**The mean-TTFT cost is real and is accepted deliberately.**  For two equal-length prompts, fair sharing
moves 1.90/3.86 s to ~3.7/3.7 s: mean rises (2.88 → 3.7 s), max falls.  A server should not hand one
caller a lottery win at the expense of an unbounded wait for the next — that is the whole argument of §3.1 —
so the campaign takes the mean cost knowingly and measures it (§4.6).  If a deployment is throughput-batch
shaped (offline, equal prompts, no arrivals) it can set the cap to 0 and get today's behaviour back.

### 4.3 Preemption is free here — and it is not KV preemption

"Preemptive prefill batching" does **not** require evicting anything:

* The KV cache is only written when tokens are **decoded**, so limiting this step's admitted prompt tokens
  costs nothing to undo.  The inner loop already breaks early for checkpoints and alora boundaries; a
  budget is one more `break` condition.
* A decode needs no space reclaimed to get its token: it is already added first (§1), and the batch is
  sized `max(n_batch, n_parallel)` (`:1778`), so every generating slot always fits.
* Real KV preemption is only needed for **admission control** under `--kv-unified`, where
  `llama_memory_context::init_batch()` can fail and the server today reacts by halving `n_batch` and
  clearing idle slots (`decode()`, `:5107-5109`).  That is a separate, later item (§7).  **P0 needs no KV
  work at all.**

### 4.4 Bound the per-step prefill to protect decode ITL (§2.5)

The fair share above still admits up to `n_batch` prompt tokens per step, and §3.1 shows that `n_batch` *is*
the admission granularity and the ITL bound.  To bound a decoder's ITL, cap the **per-step total** prompt
tokens: `cap = min(n_batch, chunk)`, then the same max-min share divides `cap`.  Knob: `--prefill-chunk N` /
`LLAMA_PREFILL_CHUNK` (proposed).

This is the only tuning decision in P0, and it is a direct trade: smaller chunk → lower ITL jitter and a
finer admission granularity, but more `llama_process()` calls per prompt and a little prefill throughput.
Measure both ends; do not guess the default.  The first cut ships the fairness **on with the cap off (=0)**,
because fairness is measured free (§2.7) and the cap is not yet measured — that keeps the default-on rule
(`AGENTS.md`) honest: the beneficial half is on, the unproven half is an explicit knob.

### 4.5 Guards a fair budget must not break

These are the reason P0 is a few hundred lines rather than twenty.  Each is an existing `break`/special
case in the same loop and each has a correctness meaning:

| guard | where | why a budget must respect it |
|---|---|---|
| checkpoint tail (`4 + n_ubatch`, `4`) | `:4885-4895` | a checkpoint must land on a state boundary; taking fewer tokens is safe, but the tail rule must still get its whole tail in one step |
| alora invocation boundary | `:4851-4855` | the adapter is switched at a token boundary; budget must not split a pre-invocation run |
| shared prompt prefix (`n_tokens_shared`) | `:4726-4747`, `:4839-4843` | children start from an exact parent state |
| decision tasks (`n_decision_first`, `decision.order`) | `:4844-4846` | logits are read from one batch; a decision prompt/option block must not be split |
| `can_batch_with()` | `:547-566` | batches are homogeneous in task type/LoRA/embd size; a rotation must re-check it rather than assume slot order |
| TTS / realtime slots | `:3938-3952`, `:4029` | they bypass the shared batch entirely |
| `n_keep_tail` in `split_equal` | `src/llama-batch.cpp:763-800` | the rollback tail (`n_rs_seq + 1`) must stay in one ubatch; the budget must not starve it |
| MoE arena band | `INVARIANTS.md` | admission changes the per-step token count, which is the arena's own band input — a wide prefill pays for a narrow decode (see `wip/bimodal-slab/`) |

### 4.6 What P0 is expected to do, and how to falsify it

| metric | today (§) | expected after P0 |
|---|---|---|
| short prompt behind a 13 k prompt, TTFT | 1.91 s (2.4) | ~1 step (10-50 ms) — **the win** |
| decoder ITL during another slot's 13 k prefill | 266-331 ms (2.5) | ≤ `chunk` / prefill rate — **the win** |
| two equal 13 k prompts, TTFT | 1.90 / **3.86** s (2.3) | both ≈ 3.6-3.8 s.  This **raises mean TTFT** (2.88 → 3.7 s) and lowers max (3.86 → 3.7 s): a deliberate redistribution, accepted because max TTFT is what a server must bound (§3.1, §4.2) |
| prefill throughput of a lone long prompt | 6788 t/s (2.4) | unchanged within a few percent (it still gets the whole remaining budget) |
| aggregate decode at concurrency 2 | 120.5 t/s (2.6) | unchanged (±2 %) |
| aggregate prefill, 2 concurrent prompts | within ±3 % of the co-batched case (§2.7) | unchanged — the serialisation is not buying throughput |

**Falsification:** if a long prompt's own total time degrades by more than a few percent, the budget is
stealing from the only consumer that can use a wide batch; if the short-prompt TTFT does not collapse, the
rotation is not running.  Both are cheap to measure (§2 harness).

### 4.7 First cut — implemented and measured (branch `sched-fair-prefill`, 2026-11-16)

Implemented in the fork at `~/llama.cpp`, branch **`sched-fair-prefill`** (commit `a1f8a1613`), off
`rdna-boosts`.  Not a delivery block, not on `rdna-boosts`, nothing in `patches/`.  One file:
`tools/server/server-context.cpp` (+130/-2), with `share_prompt_budget()` as the water-filling helper.

| config | short TTFT (behind 13.2 k) | decoder max ITL | long-prompt TTFT |
|---|---|---|---|
| `FAIR=0 CHUNK=0` (today) | 1.907 s | 336 ms | 1.992 s |
| `FAIR=1 CHUNK=0` | 0.834 s | **336 ms** | 2.027 s |
| `FAIR=1 CHUNK=1024` | 0.434 s | 177 ms | 2.047 s |
| **`FAIR=1 CHUNK=512`** | **0.293 s** | **87 ms** | 2.030 s |
| `FAIR=1 CHUNK=256` | 0.196 s | — | 2.465 s (**+24 %**) |

Four things the measurement *taught* that the design did not predict:

1. **The fair share alone does not fix decode ITL at all** (336 → 336 ms).  A decoding slot is not a *prompt*
   slot, so the long prompt still receives the whole window; only the per-step chunk bounds what a decoder
   waits for.  The two knobs are genuinely independent, which validates splitting them (§4.4).
2. **`chunk = n_ubatch` is the sweet spot**, and it is principled rather than tuned: one `llama_process()`
   per step carries exactly one ubatch, so the cap adds no ubatch split, and a lone long prefill pays only
   **+1.9 %** for a 4x better ITL bound.  `chunk = n_ubatch/2` (256) costs **+24 %** — the per-call overhead
   is the whole difference, so "smaller is better for latency" is false below the ubatch width.
3. **A 7-token prompt still needs two steps**, and not because of the budget: the checkpoint tail rule
   (`4 + n_ubatch`, then `4`) splits it at 3 tokens so a checkpoint can land on a state boundary.  Any
   gate's short-prompt bound must therefore be **two chunks**, not one.
4. **Purity holds on the single-slot oracle**: a single request's 64-token greedy continuation is
   byte-identical across `FAIR=0/0`, `FAIR=1/0` and `FAIR=1/512` (two prompts, `sha256` compared).  The
   two-slot pair is deliberately *not* a text oracle — co-batching changes the ubatch shape and hence the
   kernel family (`GREEDY-PURITY.md` §63, issue #234).

**A second shape, the one that could have regressed — 35B-A3B Q8_0 with the host-expert arena**
(`-ncmoe 48`, 1 GPU, arena 27.0 GiB at 83.6 % residency, `-c 32768 -b 2048 -ub 512`, gate = `--quick`):

| arm | short TTFT (behind 13.2 k) | decoder ITL med/p95/max | lone prefill | single-req text |
|---|---|---|---|---|
| `FAIR=0 CHUNK=0` | 7.799 s | 16 / 1021 / **1063** ms | 2070 t/s | ref |
| `FAIR=1 CHUNK=512` | **2.386 s** (3.3 ×) | 19 / 313 / **342** ms | 2073 t/s (**+0.1 %**) | identical |

So on the shape where the per-step token count *is* the arena's band input, the policy still costs nothing
in solo prefill and cuts the decoder's worst wait 3.1 ×.  The one-shot census line also confirms the arena
is unchanged (same 83.6 % residency in both arms).

**A third shape — MTP speculative decoding, the `LLAMA_SPEC_ALIGN` interaction** (4B, `--spec-type draft-mtp
--spec-draft-n-max 3`, `-c 32768`, gate = `--quick`):

| arm | short TTFT | decoder ITL max | lone prefill | single-req text | draft acceptance |
|---|---|---|---|---|---|
| `FAIR=0 CHUNK=0` | 2.091 s | 310 ms | 6162 t/s | ref | 0.853 (261/306) |
| `FAIR=1 CHUNK=512` | **0.220 s** (9.5 ×) | **89 ms** | 6173 t/s (+0.2 %) | identical | **0.853 (261/306)** |

**Acceptance is identical, and it is not a coincidence.**  `split_equal()` gives each drafting sequence its
`n_verify` rows *and* one prompt row per round, so the verify ubatch has the same shape in both arms — the
fair share only trims the prompt side.  The drafting slot's numerics are therefore untouched, which is the
result to want: a scheduling policy must not perturb the speculative path.  The **two-slot drafting** case
(the one `LLAMA_SPEC_ALIGN` exists for) is the aggregate-decode arm — both slots drafting at once — and it
also read identical (132 → 133 t/s, +0.6 %), with the same acceptance totals.  So the align pass and the
prompt budget do not fight over the same rows.

### The validation matrix so far (all `--quick`, gate = `gate.py`, 2026-11-16)

| shape | short TTFT (behind 13.2 k) | decoder ITL max | lone prefill | single-req text |
|---|---|---|---|---|
| 4B, 1 GPU, dense | 1.906 → **0.275 s** (6.9 ×) | 334 → **106 ms** | −0.8 % | identical |
| 4B, 1 GPU, **MTP n3** | 2.091 → **0.220 s** (9.5 ×) | 310 → **89 ms** | +0.2 % | identical (acceptance identical) |
| 35B-A3B, 1 GPU, **host experts** (`-ncmoe 48`) | 7.799 → **2.386 s** (3.3 ×) | 1063 → **342 ms** | +0.1 % | identical |
| 4B, **2 GPU `-sm tensor`** (`--device 0,2`) | 1.472 → **0.256 s** (5.7 ×) | 255 → **96 ms** | −0.6 % | identical |
| 4B, **3 GPU `-sm tensor`** | 1.904 → **0.272 s** (7.0 ×) | 321 → **122 ms** | −1.7 % | identical |
| 4B, **gfx1151** (Strix Halo, UMA APU) | 4.974 → **0.870 s** (5.7 ×) | 1089 → **410 ms** (0.38 ×) | −0.2 % | identical |
| Flash-Next IQ3_XXS, 2 GPU, `-ncmoe 47`, **arena fully resident** | 4.854 → **1.397 s** (3.5 ×) | 1528 → **509 ms** | +1.2 % | identical |
| Flash-Next IQ3_XXS, 2 GPU, **arena forced to 8 GiB — really streaming** | 7.397 → **2.757 s** (2.7 ×) ⚠ | 3473 → **1085 ms** (3.2 ×) | −0.3 % | identical |

Plus `scripts/gate-ckpt-reuse.sh` PASS on the prototype (5 vs 8 checkpoints, text identical, no fault) — the
context-checkpoint path survives prompts being split across more steps.

Reading the matrix: the **lone-prefill cost stays inside ±2 %** everywhere, which is the point — the policy
redistributes latency and does not tax the one consumer that can use a wide batch.  The 2-GPU split is the
*faster* prefill of the three (8618 t/s against 3-GPU 6632 — a 4B model pays the 3-way all-reduce), which is
why its derived ITL bound is the tightest (104 ms) and the one arm that trips the "near the bound" warning.

The gfx1151 arm ran on the `halo` box: the patch applied to an **identical blob** (the branch tips differ, the
file does not), the build was a single-file recompile, and the single-request greedy text came out with the
**same sha256 as gfx1201** — a bonus cross-arch purity signal, not a claim (text is per-config).

Two more layouts, both green, plus **one documented limitation**:

* **`--kv-unified`** passes unchanged (1.897 → 0.268 s, ITL 334 → 106 ms, prefill −0.6 %).  Note that for a
  *hybrid* model this changes the KV pool semantics but not the ubatch splitter (§1.1), so the mechanism is
  untouched.
* **`--kv-window 2048`** (streaming KV, 896 MiB spilled) passes (2.75 → 0.295 s, ITL 492 → 251 ms, prefill
  −0.9 %).  `--kv-window auto` **declined to engage** on this shape ("no window is needed — the 16384-cell KV
  fits fully resident"), so `auto` must be checked for engagement before any streaming claim: an arm that
  spills nothing tests nothing.
* **`--kv-attendance-pool on` — engaged on Qwen3.8-Flash-Next and measured** (this corrects an earlier claim
  in this doc that it could not be exercised; it declines on Qwen3.5-4B because "the compress ratio (page
  size) is not uniform across its layers", but Flash-Next has the uniform QSA indexer geometry it needs).
  With `--kv-unified --kv-attendance-pool on` on one discrete GPU the pool engages in **both** arms:
  `ATTENDANCE POOL ENABLED ... 4096 pages x 4 cells per layer, 384 MiB device pool + 384 MiB pinned host
  mirror, 12 layers x 128 dims`.  Measured (8800-token prompt, chunk 512): short TTFT 9.914 → **3.886 s**
  (2.6 ×), ITL 3535 → **1090 ms** (3.2 ×), lone prefill 636 → 634 t/s (**−0.3 %**), single-request text
  identical, no fault.  **Note `--kv-attendance-pool` is NOT supported on gfx1151 by design** (`TODO` #300,
  retired r264): on a UMA device the KV is already in the memory the GPU reads, so there is nothing to buy
  back — which is also why halo cannot be the box for this arm.

  The 2.6 × is **below** this shape's ceiling (`L/(3·chunk)` = 8800/1536 ≈ 5.7), so unlike the earlier
  streaming arm it is *not* explained by the step-count ceiling: the short prompt took ~5 chunks where 2-3
  were expected.  The plausible cause is the pool's prefill path (it reads the mirror over PCIe), with the
  short slot's page fills queueing behind the long slot's — **unexplained, recorded as an open question** in
  `TODO` #338 rather than asserted.

**The streamed-prefill case (`-ncmoe` with the experts really streaming) — the answer is good news.**  On
Flash-Next IQ3_XXS, 2 GPUs, `-ncmoe 47 -ot per_layer_token_embd.*=CPU`: with the default arena everything is
resident (arena 40012 MiB = **100.0 %**, `evictions=0`), so that arm does *not* test streaming.  Forcing
`MOE_EXPERT_CACHE_MIB=8192` gives real streaming (`evictions=48166`) and the numbers above: **lone prefill
563 → 561 t/s (−0.3 %)**, i.e. chunked admission costs the streaming prefill nothing, and the decoder's
worst wait falls 3.2 ×.  This is the case the campaign was most likely to get wrong, and it holds.

**The fairness ratio on the streaming arms is short of the 3.0 default — for two different reasons.**  On
the `-ncmoe` streaming arm it is 2.7 × *and that is the ceiling* (below).  On the attendance-pool arm it is
2.6 × against a ceiling of ~5.7, i.e. genuinely short and unexplained.  Both are recorded in `TODO` #338.  The short prompt needs ~3 steps (the checkpoint split makes a 7-token prompt
3 + 4, plus arrival alignment), so the ceiling is `L / (3 × chunk) = 4400 / 1536 = 2.86`.  Everything else on
the arm is clean.  Recorded as `TODO` **#338**: the fairness bound must be derived from `L/(k·chunk)` rather
than a fixed 3.0, or `--min-ttft-ratio` must be set per shape with the reason stated.

Knobs, as shipped in the prototype: `LLAMA_SERVER_PREFILL_FAIR` (on unless `0`, a kill-switch),
`LLAMA_SERVER_PREFILL_SKEW` (**default 1 = equal share**, `N` = weight the shortest-remaining slot N:1), and
`LLAMA_SERVER_PREFILL_CHUNK` (**unset = one `n_ubatch`**, `0` = off, `N` = explicit).  §4.2.1 has the cost
model behind the default.  The chunk is applied in **both** modes, because it is a separate
mechanism — so `FAIR=0 CHUNK=512` is a valid arm (today's fill, bounded per step), which is exactly the
bisection pair the gate needs.  One-shot census line at startup: `prefill admission: fair share = on/off,
chunk = N`.

Two hardening items landed with the gate (commit `1b4f9a705`), both inert on the causal text path:

* **No permanent starvation.**  `budget = n_batch - (decode tokens)` is 0 when the decode tokens alone commit
  the window — possible when `n_parallel >= n_batch` — and a plain max-min share then denies prompts forever
  (today's fill has the same hole via the `batch.size() >= n_batch` return).  The share now floors at one
  token per active prompt; the batch may exceed `n_batch` by at most `n_active` and the outer
  `llama_process()` loop splits it, so the extra pass is paid only in that degenerate case.
* **Mixed / non-causal batches opt out.**  `use_mixed_batch()` goes through `add_prompt_mixed()`, which
  places a whole prompt at once, so a per-sequence share is not expressible there.  The policy is left OFF
  for those contexts rather than half-applied.

Still open for the first cut to become a block: the `-lv 5` per-step `fair share: budget/added/batch` line
should become a `-lv 4` census (it is the gate's assertion source); and the `can_batch_with()` homogeneity
case — two LoRA classes still serialise, which is not fixable from the fill policy.

## 5. Not P0 — three questions kept separate

### 5.1 "Paged KV controls fragmentation"

Partly right, but the word means two different things here.

* **Wasted bytes:** llama.cpp is already *finer* than paged — one cell per token from a shared free list
  with a per-stream `head` cursor (`find_slot`, `src/llama-kv-cache.cpp:1350-1380`), plus `--kv-unified`
  for one shared pool.  Paged attention (16-token blocks) would *waste* up to 15 cells per sequence where
  llama.cpp wastes none.  The only block rounding today is per-slot: `GGML_PAD(n_ctx_seq, 256)`.
* **Scatter/alignment:** this is the real one.  Freed cells are reused, so a sequence's cells become
  non-contiguous, `slot_info::is_contiguous()` fails, and the vectorised contiguous path is replaced by a
  per-cell path (`llama-kv-cache.cpp:4806-4860`).  That costs throughput; it does **not** change any value,
  and the fork already carries compaction/parking machinery to move cells (`seq_park`, `LLAMA_KV_PARK_DEBUG`)
  precisely because scatter exists.

So a paged cache would buy *alignment for speed* and *block-granular preemption*.  Neither is the concurrency
symptom, and neither is needed by P0 (§4.3).  Recorded, not scheduled.

### 5.2 Would a paged KV fix cross-slot determinism? (hypothesis H3 — **untested, and probably no**)

The finding you are remembering is **`GREEDY-PURITY.md` §63** (r153, issue **#234**), with
`benchmarks/2026-10-31-multislot-verify-width.md`, `archive/work/spec-align-234/` and the gate
`scripts/gate-spec-align.sh`.  Its claim is narrower than "KV misalignment":

> "aligning the sequences of one step into one ubatch changes a slot's verify width … a two-slot *aligned*
> pair is not stable run to run either, so per-config determinism is a single-slot statement."

And the gate states the consequence explicitly: a two-slot pair is **not** gated on text, because *every*
fix for that defect has to change some slot's verify width, and `GREEDY-PURITY.md` promises greedy text
**per config**, never equality between configs.

Mechanism taxonomy — this is the part worth keeping straight, because it decides what paging can and cannot
buy:

| what changes | changes numerics? | why |
|---|---|---|
| **ubatch shape** (tokens per sequence, `n_seqs`, `equal_seqs`) | **yes** | selects the kernel family (`MMVQ_MAX_BATCH_SIZE == 8`, `ggml/src/ggml-cuda/mmvq.cuh:3`; the FA tile's `ncols2`; the WMMA `iu8` verify floor of 7), so the accumulation order changes — §63/#234, §91, §49 |
| **attention regime** (dense vs pooled/sparse/windowed) | **yes** | §77: the pooled arm is a different regime; a KV residency policy is a new numerics reference |
| **MoE arena band** (rows served from VRAM) | **yes** | the band selects the expert kernel set (`INVARIANTS.md`, §63 addendum) |
| **K/V cell index, offset, contiguity, block size, defrag, parking** | **no** | same values, same tile shape; only the copy/store path and the number of launches differ |

So the cross-slot nondeterminism is a **batch-composition** effect, not a KV-cell-layout effect.  A paged
cache does not constrain a slot's tokens-per-step, so it would not pin the kernel family, so it should not
fix the determinism claim.  Two things would: make the kernel family width-invariant (invariant 1 in
`GREEDY-PURITY.md`), or pin the step shape — which is what `LLAMA_SPEC_ALIGN_WIDTH_BAND` does, and it
*changes the width by design* on a 3+ slot config rather than making it stable.

**Falsification test if it is ever worth the time (cheap, do it before any paging work):** take one
two-slot config, force a fixed KV placement for slot 1 (even a deliberately scattered one via a diff/extra
alloc), keep the ubatch shape byte-identical (same token counts, same `n_seqs`), and diff raw logits.
Prediction: **0 difference**.  Then change only the token counts and predict a *nonzero* difference.  If the
first arm is nonzero, H3 is alive and this section is wrong — that is the whole reason to state it.

### 5.3 True parallel *execution* (multi-context / multi-stream) — costed, not scheduled

P0 gives concurrency *in scheduling*.  Two forward passes in flight is a different project:

| option | what it costs | what it buys |
|---|---|---|
| N `llama_context`s over one `llama_model`, own worker thread each | `server_context` assumes a single `ctx_tgt` everywhere (slots, batch, `spec`, MTP `ctx_dft`, MoE arenas, KV pools); one KV + one compute reserve **per context**; audit `llama_model` for mutable state | real overlap of prefill and decode kernels; on one GPU mostly latency isolation, not throughput (a 1-token batch × 2 is slower in aggregate than one 2-token batch) |
| N processes (already possible: the router spawns subprocesses) | one weight copy in VRAM per process | same, operationally, today |
| multi-stream inside one context (extend the optional `0100` two-lane prefill from across-devices to across-graphs) | per-stream compute buffers + graph copies + a hazard analysis; `ggml-backend` is where this gets hard | only meaningful with ≥2 devices |

The honest read for this box: one GPU is bandwidth-bound, so session parallelism is a **latency-isolation**
feature there; on 3 GPUs `-sm tensor` already spreads every step across all three, and per-session
parallelism would mean replicating weights.  P0 delivers most of the felt benefit for a fraction of the
cost, which is why it is first.

## 6. The gate — implemented (`gate.py`, this directory)

`./gate.py` — two arms on the same binary, `-np 2`, one GPU:

| arm | config |
|---|---|
| `baseline` | `FAIR=0 CHUNK=0` — today's slot-order fill |
| `candidate` | `FAIR=1 CHUNK=<ubatch>` — fair share + the per-step cap |

**HARD**
* short-prompt TTFT behind a 13.2 k prompt improves by ≥ `--min-ttft-ratio` (3 ×) — measured 6.9 ×;
* the candidate's worst decoder ITL during a foreign prefill is **proportional to the prompt tokens in the
  step**: `max ≤ baseline_max × (chunk / n_batch) × 1.75` — measured ratios 0.32 (1 GPU), 0.38 (gfx1151),
  0.38 (3 GPU), 0.32 (host experts).  The bound is deliberately **relative**, not an absolute millisecond
  count and not derived from the lone-prefill rate: on gfx1151 (UMA) a co-batched prefill runs at ~1550 t/s
  against 2604 t/s alone, so a lone-rate bound made a correct 405 ms read as a 344 ms failure.  `--max-itl-ms`
  overrides it with an absolute bound if a deployment wants one;
* a lone long prompt's prefill throughput within `--max-prefill-loss` (3 %) — measured −0.8 %;
* **one** request's greedy text byte-identical between the arms (the single-slot oracle; a two-slot pair is
  never a text oracle, §63);
* each arm's startup census states the configuration it claims (`prefill admission: fair share = ...`);
* no abort/assert in either log, and `scripts/check-gpu-faults.sh` clean — **with `--since` bounded to the
  run**.  An unbounded journal check fails on yesterday's faults (it did: the first run reported faults from
  2026-11-15, `journalctl -k --since "40 min ago"` showed 0).  A gate must scope its own window.

**SOFT** aggregate decode at concurrency 2 within `--max-decode-loss` (5 %) — measured +0.6 %.

`--self-test` runs the candidate **as** the baseline configuration and requires the HARD fairness checks to
fail.  Verified: it reports 1.0 × and 335 ms and exits 0 ("the gate rejects the defect").  A gate that
cannot reject the unfixed build is not a gate — run this before trusting a green run.

Cost: ~90 s (two cold loads, 13.2 k-token prompts, `-lv 4` for the census and the fault scan; the HTTP
responses are hashed, so `-lv 4` does not pollute the text — `AGENTS.md` → Verify).

**Speculative arms are self-checking.**  When `--server-args` carries a `--spec-type`, the gate asserts that
spec really initialized in **both** arms (a spec arm that silently failed to initialize would otherwise pass
every other check) and that acceptance did not *collapse* (≥ 50 % of the baseline).  Acceptance is
deliberately not asserted equal: the fair share changes the ubatch composition, and a verify width is a
numerics reference (`GREEDY-PURITY.md` §63), so it is reported and only a collapse fails.

Promotion path: on the block decision this becomes `scripts/gate-prefill-fairness.sh` (a bash wrapper,
repo conventions) and joins `MANIFESTS.md` → Release gates as the gate for "the server's prompt admission
policy".

**Census decision (settled — this revises the earlier plan).**  The startup line
`prefill admission: fair share = on/off, chunk = N` stays the `-lv 4` census, and it is what the gate asserts
(configuration identity: an arm cannot pass by measuring a policy it did not run).  The per-step
`fair share: budget/added/batch` line **stays at `-lv 5`** and is not promoted: it fires per slot per step, so
at `-lv 4` it would bury the log it is supposed to help.  The gate asserts *behaviour* (TTFT, ITL, text,
census), not per-step internals, so the promotion does not need it.

### 6.1 Three instrumentation traps the gate had to fix (each produced a wrong verdict first)

1. **Slot assignment is not under the harness's control.**  `update_slots()` gives the *lowest slot index*
   priority, and the server assigns slots by LRU/similarity — so an unpinned two-request arm silently
   measures whichever side of the race it gets.  Measured: the same starvation arm read **0.828 s unpinned**
   (the long prompt landed in slot 1) vs **1.907 s pinned** (slot 0).  Both `gate.py` and `probe.py` now pass
   `id_slot`, which the API supports.  **Any two-slot measurement in this repo without `id_slot` is
   unreliable** — that includes the §2 numbers, which happened to be the pinned case only because a fresh
   server puts the first request in slot 0.
2. **A "lone prefill" measured in the 4th position is not lone.**  The arena/graph state left by the earlier
   arms differs between a 512-token-step arm and a 2048-token-step one, and that reported **+25.1 %** on the
   host-expert shape.  Isolated on four fresh loads the two configs read **2065-2073 t/s — identical**.  The
   gate now measures solo prefill on its own server per arm, and that phantom win would have been shipped.
3. **A throughput metric that includes prefill is not a decode metric.**  The aggregate-decode arm first used
   wall time, which on a streamed-prefill shape is seconds of prefill against 80 decode tokens — it reported
   "−8 %" on one run and "+22.7 %" on another, both meaningless.  It now uses each request's first-to-last
   token span, and is **skipped as unmeasurable** (INFO, no verdict) on a shape whose responses stream a
   single chunk.  A metric that cannot be measured must not produce a verdict.
4. **`max` and `p95` disagree, and both are honest.**  The cap converts a *few* ~1 s stalls into *many*
   ~300 ms ones, so `max` falls 3 × while `p95` rises (334 → 95 ms on the 4 B it falls too, but on the MoE
   shape it goes 1021 → 313 ms only because the baseline's stalls are 3.5 % of steps).  The HARD check is
   `max` (the claim is a bound), and the distribution is always printed.

**RETRACTED observation — a first-step stall that does not reproduce (read this before quoting it).**  Two
early runs of this harness showed a one-off **~1.8 s (fair on) / ~2.5 s (fair off)** in the *first* gap after
a foreign prefill started, on the host-expert shape, then steady state.  It was reported as a pre-existing
idle-slot/parking defect.  **It is not one, and the claim is withdrawn.**  Six controlled runs since — three
reproduction designs (a 4 k warm context; a 13.2 k warm context on one slot; the gate's old
`text -> starve -> itl` order with both slots left with 13.2 k) × the three idle-slot knobs
(`LLAMA_SERVER_PARK_IDLE_SLOTS=0`, `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, `--no-cache-idle-slots`) — all read the
**same** maximum, 1023-1070 ms at `-b 2048`, i.e. exactly one 2048-token step:

| arm | max gap after the first | |
|---|---|---|
| default (parking on) | 1042 ms | |
| `LLAMA_SERVER_PARK_IDLE_SLOTS=0` | 1038 ms | parking is **not** the carrier |
| `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` | 1036 ms | nor lazy idle slots |
| `--no-cache-idle-slots` | 1040 ms | nor the save/clear path |

So there is **no knob to recommend and no trade-off to document**: disabling a default-on *purity* feature
(`GREEDY-PURITY.md` §57 — parking is what makes a lone request see the same attention range whoever else is
resident) would buy nothing.  The most likely explanation for the two original readings is the harness: slot
assignment was unpinned then, so the requests landed on different slots, and **pinning (`id_slot`) was added
in the same commit that retired the observation**.  If it ever resurfaces, the first thing to check is the
slot permutation, not the memory policy.

**Lesson, recorded on purpose:** an anomaly seen twice in a harness that was later found to be uncontrolled is
not a defect report.  It should have been held until it reproduced.

### 6.2 `TODO` #338 — the fairness ratio is shape-dependent, and one streaming arm is short of its ceiling

`gate.py` requires the short-prompt TTFT to improve by >= 3x.  Two arms miss it, for **different** reasons, and
neither is a defect in the policy:

**(a) `-ncmoe` streamed-prefill arm — 2.7x, and that IS the ceiling.**  Qwen3.8-Flash-Next IQ3_XXS, 2 GPU,
`MOE_EXPERT_CACHE_MIB=8192` (real streaming, `evictions=48166`): 7.397 → 2.757 s.  A 7-token prompt is split
3 + 4 by the checkpoint tail rule and needs one step of arrival alignment, i.e. ~3 steps, so the ceiling is
`L/(3*chunk) = 4400/1536 = 2.86`.  Everything else on the arm is clean (ITL 3473 → 1085 ms = 3.2x, lone prefill
−0.3 %, text identical, no fault).  **A fixed ratio cannot express this**: the same threshold is comfortable on
a 13.2 k prompt and unreachable on a 4.4 k one at the same chunk.

**(b) attendance-pool arm — 2.6x against a 5.7x ceiling, genuinely short and NOT explained.**  Qwen3.8-Flash-Next,
1 discrete GPU, `-ncmoe 47`, `--kv-unified --kv-attendance-pool on` (the pool engages: 4096 pages x 4 cells,
384 MiB device + 384 MiB pinned mirror): 9.914 → 3.886 s.  Here 2.6x is well below `8800/1536 = 5.7`, so the
step-count argument above does **not** cover it: the short prompt took ~5 chunks where 2-3 were expected.
Everything else is clean (ITL 3535 → 1090 ms = 3.2x, lone prefill −0.3 %, text identical, no fault).

**The likely carrier, not established:** the attendance pool's prefill reads the authoritative KV from the
pinned host mirror over PCIe, so the short slot's page fills may queue behind the long slot's.  But "the pool
adds a fixed per-step fill cost" and "the scheduler admits the short prompt late under the pool" are different
defects with different fixes, and this session did not disambiguate them.  **The discriminating experiment** is
to time the short slot's *admission* (the `-lv 5` `fair share: budget=…/added=…` line for slot 1) against
5 chunks of prefill: if it is admitted in step 1 and still takes 5 chunks, the carrier is the pool; if the
budget line shows it admitted late, the carrier is the scheduler.

**Fix for both:** derive the bound from `L/(k*chunk)` with `k` the short prompt's measured step count, rather
than a fixed ratio — the same correction the ITL bound already needed on the UMA box.  Until then a
streamed-prefill or pooled run must pass `--min-ttft-ratio 2.5` and say why; the alternative (a default low
enough for every shape) would stop the gate rejecting a partial fix.

## 7. Risks / open questions

* **Prefill throughput vs ITL.**  §4.4's cap is a real trade; the default must come from measurement on
  gfx1201 *and* gfx1151 (the slow box is where a wide prefill matters most).
* **MoE arena interaction.**  A per-step token count feeds the arena band; a fair scheduler changes the
  cadence of wide/narrow steps.  `wip/bimodal-slab/` and `INVARIANTS.md` r292 say a two-slot
  prefill/decode transition has already been made stable — re-measure, do not assume.
* **Speculative decoding.**  `LLAMA_SPEC_ALIGN` already caps verify width per step (§63); the two policies
  must not fight over the same budget.  Drafting happens before the batch is assembled (`:4190-4230`).
* **Admission queue.**  `n_parallel` slots are still a hard ceiling and a queued task is `defer()`red until
  a slot frees.  Real elastic admission is a shared-pool (`--kv-unified`) + preemption question (§4.3).
* **Interaction with block 14's window/pool.**  A streaming KV config changes the attention regime (§77);
  the fairness gate must run on a resident config, and a separate arm on a windowed one.

## 8. Pointers

* Code: `tools/server/server-context.cpp` `update_slots()` `:3883`, admission `:4328-4335`, inner fill
  `:4831`, `decode()` `:5025`; `src/llama-batch.cpp` `split_equal()` `:668`;
  `src/llama-kv-cache.cpp` `find_slot` `:1350`, contiguity `:4806`; `tools/server/server-queue.cpp:283`.
* Determinism: `GREEDY-PURITY.md` §63 (r153, issue #234), §77, §91; `INVARIANTS.md`;
  `benchmarks/2026-10-31-multislot-verify-width.md`; `scripts/gate-spec-align.sh`.
* Existing two-slot harness: `scripts/gate-prefill-server-slots.sh` (`TODO` #317, block 16);
  `scripts/gate-prefill-logits.sh` for any prefill change.
* Adjacent campaigns: `wip/bimodal-slab/` (wide/narrow ubatch modes), `wip/kv-residency-cleanup/`
  (window/pool), `optional/two-lane-prefill/` (device-level prefill overlap).
* Reproduce §2: `./probe.py --port 8099 {decode,prefill,starve,itl,tps}` (arms, defaults and the server
  invocation are in this README and the script's header).  The numbers here are one session on one card —
  a gate must re-take them on the release build and on gfx1151.  This is a scoping note, not a validated
  result.

## 9. The patch series (backup, and how to reproduce the branch)

The prototype lives on branch `sched-fair-prefill` in `~/llama.cpp` (and was applied over ssh to `halo` and
`fingon`).  It is backed up here as a **four-commit series with full messages** plus a **net diff**:

| file | what |
|---|---|
| `series/0001-*.patch` .. `series/0004-*.patch` | the four commits, `git format-patch` output (messages included) |
| `sched-fair-prefill.patch` | the net diff, applyable to a clean checkout of the delivery tree |

```
base tree  3da54a4c10d7350eddc5c0d8eae384086a306b0f   (= release.json.tree, i.e. the r299 delivery tree)
tip tree   4baf1b67ae591e6661a1e3c5365457a747b688ab
```

```
sha256  3d4a9d699ea64212749c259ff836a390e9433c34afb80525df57d61cb06f3ec8  sched-fair-prefill.patch
        6dc5c2b4...  d5e78cac...  7e9511bf...  fad9f9db...                 series/0001..0004
```

Reproduce the branch from the delivery tree:

```bash
git clone <llama.cpp> && cd llama.cpp && git checkout 4f9dfa0e3     # release.json.base
bash <delivery>/scripts/apply-all.sh .                              # the default set (tree 3da54a4c…)
git checkout -b sched-fair-prefill
git apply <delivery>/wip/session-scheduling/sched-fair-prefill.patch
# or, to keep the four commit messages:  git am <delivery>/wip/session-scheduling/series/*.patch
```

Verified: `git apply --check` of the net diff on a worktree at tree `3da54a4c…` passes (done 2026-11-16).

**Size of the change.**  One file, `tools/server/server-context.cpp`:

| | lines |
|---|---|
| added (net) | **207** — of which **128 code**, 54 comment, 25 blank |
| removed (net) | **2** (1 code, 1 comment) |
| **net** | **+205 lines, +127 code lines, 1 file** |

For scale: `wip/session-scheduling/gate.py` (the gate) is ~390 lines of Python and `probe.py` ~230, neither of
which is delivery code.
