# T2 — the server cut, in verified code points

**Status: BUILT AND PASSING TIER 1 (2026-11-22, pass 5) — `~/llama.cpp` `683deed67`.**  Every code point below was
implemented as written; the as-built notes, the two defects the cut produced, the VRAM finding (`741.48 MiB` 2-row
decode graph, fixed with `GGML_CUDA_SLAB_HEADROOM_MIB=1536`) and the measurement are in
[`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) **pass 5**.  Tier 1 passes cold and over three warm request pairs
(both slots byte-identical to the single-context server, every allocation handshake clean); the smoothness target is
met (~47-49 ms ITL p50 through a 2.0-5.3 s prefill) and the **cost target is not** (-24 to -27 % against the
instrument's -14.1 % — pass 5 §6 reads that honestly).

Three things differ from the plan below and are recorded in pass 5 §1: a `wait_handover_taken()` at the top of
`pre_decode()` as the second line of defence for the ordering rule, `arm_release()` for a cancel/abort that lands on
a slot the decode arm owns, and `finish_returned_slots()` owning the response's lifecycle on the main loop.

**AMENDED 2026-11-22 (pass 7): the return now does the REVERSE HANDOVER.**  Pass 5's cut cleared the slot's prompt
and let the decode arm free the shared cells, which cost a full re-prefill on the next request (measured 685 ms
against 33 ms).  The maintainer's direction is that prefix reuse is essential, so the return takes the recurrent
state **back** to the prefill arm and keeps the prompt (see `RESULTS-2026-11-22.md` pass 7 for the code points, the
two oracles and the numbers).  §2's "the decode arm calls `llama_memory_seq_rm(...)` — the shared cells are freed
once, here" is superseded: nothing is freed on a normal stop or a cancel, and only a failed take or a failed decode
step drops the sequence.

Original scope note (2026-11-22, against `tools/server/server-context.cpp` as it stood on `wip/prefill-overlap`):

The design is unchanged from `RESULTS-2026-11-21.md` §5: **one server with two arms**, where the decode arm owns
its slots' generation and nothing else.  The target is the instrument's own numbers: a decoder getting a token every
**~50 ms** through a **~2 s** prefill for **≤ ~14 %** of prefill (`RESULTS-2026-11-21.md` §2.1).

## 1. Contexts and batches (the plumbing)

| anchor | what to do |
|---|---|
| `llama_context * ctx_tgt` member (~1118), created ~1415-1700 | create **`ctx_dec`** right after it, from the same `llama_model_params`/`cparams`, with `cparams.ctx_other = ctx_tgt`, `ctx_other_share_memory = true`, `n_batch = n_ubatch = <decode width>` (the instrument's `ARMS_B_BATCH`), `n_seq_max = n_parallel`, `n_outputs_max = n_parallel`.  Refuse to create it when the opt-in is off, so a default server allocates nothing new |
| `struct server_batch` (151), the `batch` member | add **`batch_dec`** — the decode arm builds its own batch and never touches `batch` |
| `slot.ctx_tgt` wiring (~1829) | add `slot.ctx_dec` and `slot.arm` (`PREFILL`/`DECODE`); default `PREFILL` |
| `pre_decode()` (4329) — the slot loop at ~4409 | **skip `DECODE` slots entirely**: they are not the prefill arm's work any more |
| `post_decode()` (5502) — `slot.state = SLOT_STATE_GENERATING` at ~5562 | this is the **handover point**: see §2 |

## 2. The handover and the return

* **Out (prefill → decode)**: at the point `post_decode()` would set `SLOT_STATE_GENERATING`, instead post the
  slot to the decode arm (a queue + condvar), which executes
  `llama_memory_seq_state_take(llama_get_memory(ctx_dec), llama_get_memory(ctx_tgt), slot.id)`
  **under its own step mutex** (never under a lock the prefill thread waits on — measured 2.85 s starvation), then
  sets `slot.arm = DECODE`.  `llama_memory_alloc_sync(llama_get_memory(ctx_*), "boundary")` at the boundary is the
  T0 handshake and must return clean.
* **Back (decode → prefill)**: when the decode arm stops a slot (`n_predict`, EOG, abort, or a client disconnect),
  it calls `llama_memory_seq_rm(llama_get_memory(ctx_dec), slot.id, -1, -1)` — **the shared cells are freed once,
  here** — sets `slot.arm = PREFILL`, and wakes the main loop for `release()` / `send_final_response()`.  The decode
  arm does **not** own a response's lifecycle; it produces tokens.
* **Ordering rule (hard-won, `RESULTS-2026-11-22.md` pass 3 §3)**: the prefill arm must wait for a posted handover
  to complete before it starts the next prompt on that memory, or the take reads metadata a live prefill is
  writing — that silently changes the next prompt's recurrent cell and the reply's text.

## 3. The decode step (what to factor)

`post_decode()`'s per-slot work — `common_sampler_sample(slot.smpl.get(), ctx, i)`, the stop conditions, the
`generated`/stream emit, and the metrics — must become **one function taking an explicit context + batch + slot**, so
the two arms share it:

```
slot_step(llama_context * ctx, server_batch & batch, server_slot & slot, int32_t i)   // samples, emits, stops
```

The decode arm's loop is then: for each `DECODE` slot, add one token (`slot.pending`, `slot.pos`) to `batch_dec`,
`llama_process(ctx_dec, LLAMA_PROCESS_TYPE_DECODE, ...)`, then `slot_step(ctx_dec, batch_dec, slot, i)` per row.
**One token per active slot in one batch** — that is what keeps ITL flat (`RESULTS-2026-11-21.md` §2).

## 4. Ownership (the invariants the thread split must not break)

* **Main thread only**: `slots[]`, `batch`, `ctx_tgt`, `spec`, `checkpoints`, the prompt cache, mtmd, TTS, realtime,
  embeddings/rerank/decision, LoRA application, `send_final_response()`.
* **Decode thread only**: `ctx_dec`, `batch_dec`, the `DECODE` slots' `generated`/stream path.
* A slot is owned by exactly one arm at a time → **a per-slot mutex** (or the queue's ordering) covers the transition;
  the decode arm never calls `prompt_save`/`prompt_load`, `common_speculative_*`, or anything in the main-thread list.
* `llama_memory_seq_rm` on the decode arm frees the *shared* attention cells, so the main thread must not batch that
  slot again until the return is observed — the arm flag is the handshake, and the main loop's slot loop must read it
  under the slot mutex.

## 5. Guards (the opt-in gate, T4)

Refuse `--prefill-decode-arms` (working name; the flag and its default are a maintainer decision) when any of: a
speculator (`spec`), mtmd, TTS, realtime, embeddings, rerank, decision, LoRA, sleep/wake (`--sleep-idle`), or
`n_parallel < 2` is in play.  `-np 1` stays single-context.  The VRAM rule is measured
(`RESULTS-2026-11-21.md` §3.2): the pair needs the arena **+ one prefill graph + one decode graph per admitted slot
width** — a 2-row decode graph on 35B-A3B at `-ub 2048` asks 741.47 MiB, so `-ub 512` is the shape that fits on a
32 GiB card.  `auto` (enable on a measured residency deficit) vs a flag is **the maintainer's call** (README §8 Q1).

## 6. The gate (the two-tier oracle the maintainer approved) — **tier 1 is already built**

Run it first, before writing any server code, so the baseline is in hand:

```bash
cd archive/work/prefill-decode-arms/arms
./gate-server-arms.sh --mode ref                               # today's single-context server
./gate-server-arms.sh --mode arms                              # after the cut (needs --prefill-decode-arms)
./gate-server-arms.sh --compare ../runs/server-oracle/ref.txt ../runs/server-oracle/arms.txt
```

**The acceptance values are measured** (`RESULTS-2026-11-22.md` pass 4): product shape, two concurrent ~500-token
requests, `n_predict 32`, `temp 0`, `seed 42`, both purity pins → SLOT 0 `459865c93d55cbaf`, SLOT 1
`7dc20b0ce0bac255`, tier-2 census `h=0.6853`.  Those are deterministic across separate server processes, and the
gate is proven to fire (`-ub 256` moves slot 0 to `182a9fcfe1de6a70`).  The compare asserts the **text** only, with a
vacuity guard; the census is recorded, never asserted.

1. **Tier 1 — the cross-configuration hash**, which T0 made available on the product shape too: same prompt, same
   seed, `-np 2`, `ROCBLAS_USE_HIPBLASLT=0`, `MOE_EXPERT_HEATMAP=load`, `LLAMA_KV_ALLOC_SHARED` at its default;
   `--prefill-decode-arms` must read **byte-identical** text to the single-context server, for both slots, over
   ≥ 3 warm request pairs.  `llama_memory_alloc_sync()` must be clean at every request boundary.
2. **Tier 2 — KLD + the residency census**: `scripts/gate-prefill-logits.sh` for the prefill path, plus the
   `MoE expert cache: h=…` census reported (not asserted) — the §89 hazard is a risk for shapes this campaign has
   not run.
3. Server-side gates that follow from the touched code: `scripts/gate-prefill-fairness.sh` +
   `scripts/gate-prefill-duty.sh` (admission is now the prefill arm's alone), `scripts/gate-park-idle-slots.sh` if
   slot parking is reached, `scripts/gate-ckpt-reuse.sh` if a checkpoint crosses the arms,
   `scripts/check-gpu-faults.sh` on **every** arm (a two-arm race shows as a fault, not as a wrong token).
4. The measurement of record: the retired campaign's `arms/smooth.py` (`itl_d_p50`/`itl_d_p95` inside the prefill
   window, `pf_tps`), against its baseline table.

## 7. What NOT to do (each of these cost a session already)

* Do not put the take (or the handover) under a lock the prefill thread waits on while the decoder steps: post it.
* Do not let the prefill arm start the next prompt before a posted take completed.
* Do not run **two decode** contexts (the decode-band `GGML_MOE_DEVREMAP` map is per table, per process — pass 10 of
  the retired campaign).
* Do not re-enable the per-gather yield hook, a rate gate or stream priority on the hot path: all three are measured
  losers (`archive/work/prefill-transfer-overlap/RESULTS-2026-11-20.md` pass 12).
* Do not size the decode context for prefill (`ARMS_B_BATCH`-style widths only), and do not assume `-ub 2048` fits.
