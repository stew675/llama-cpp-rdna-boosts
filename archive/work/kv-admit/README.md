# `kv-admit` — the server's KV admission (contributor PR #370, `briansp2020`)

**Promoted into block 16 as an amendment at r365.**  This directory is the campaign record: the submitted patch as
the reproduction, the submitter's own repro log, and the integration work named below.  It is not open work and it is
not a `wip/` campaign (nothing here is applied by `scripts/apply-all.sh`).

## What was reported

With a unified KV cache the server starts every request that finds a free slot, whether or not its prompt fits beside
what the other slots already hold.  Two prompts whose sum exceeds the context both prefill until the cache is full and
then **decode** fails: the retry loop can only clear *idle* slots (`try_clear_idle_slots` skips a processing one), so it
halves `n_batch` down to 1 and answers HTTP 500 "Context size has been exceeded." — or, with a speculator,
"speculative batch index N is not inside the current sub-batch".  The delivery's **default** server is the failing
shape: `-np -1` resolves to 4 slots and `kv_unified = true`.

Verified on the delivery at 1/8 the scale and with no contributor code (r364, one R9700, Qwen3.5-4B, `-c 8192 -np 4
--kv-unified`): two ~4.85 k-token prompts answer **500 / 500** in 0.8 s, while the same two prompts run one at a time
answer 200 with `592ade35` / `e72118f2`.

## What shipped (block 16, r365)

The submitted patch (`submitted-0001-server-KV-admission.patch`, kept here unchanged as the reproduction) added the
fit check in `process_single_task`, after `slot_sched.select()` and using the chosen slot only to exclude itself from
the sum — a test that can never fire, since the candidate is always an idle slot — and logged a WARN per deferral.

The promotion moved the decision to where a deferral already belongs:

* `server_slot_sched` carries an injected **oracle** (`std::function<bool(const server_task &)> admit`);
  `server_slot_plan` carries a `why` (`none` / `no_slot` / `no_capacity`).
* `select()` asks the oracle **once, before a candidate is chosen**, so the predicate stays slot-INDEPENDENT (a
  whole-request go/no-go — never "does it fit on this candidate", which would have to be asked per candidate) and the
  scheduler learns nothing about cells, reserves or margins.
* A refusal travels the stock `plan.id_slot == -1 → queue_tasks.defer()` path: **one** defer site, so
  `pop_deferred_task()` and `/metrics`' `n_tasks_deferred` cover it for free, and a prompt pinned to an idle `id_slot`
  is refused by the same path.
* The accounting stays in the context that owns `n_ctx` (`kv_admit_ok` / `kv_admit_cells`): each **running** slot is
  charged `max(its prompt, its task's tokens)` plus a reply reserve.  Idle slots are not charged (the launch path parks
  them and then clears them for want of room) and neither is a realtime slot waiting for audio (`is_processing()` but
  no batch work — charging it would refuse a task no release could make room for).
* The reserve is `min(n_ctx / n_parallel, n_ctx_slot())` cells — the per-sequence share a NON-unified cache would give
  each slot — capped by the request's own `n_predict`, overridable with `LLAMA_SERVER_KV_ADMIT_RESERVE`.  (The
  submission used a flat 16384; the maintainer chose the per-sequence share at review.)
* `LLAMA_SERVER_KV_ADMIT=0` restores the pre-r365 server exactly.  Inert without `--kv-unified`, and for a parent task
  and a prompt that alone exceeds the context (documented limitations).

## The three defects stress-testing found (all fixed in the same block)

1. **The idle wake could spin at 100 % CPU.**  `update_slots()`' `all_idle` counts a realtime slot waiting for audio as
   idle while the admission counted it as running, so a re-offered task was refused again, the queue stayed non-empty
   and the loop never waited.  The wake now re-offers only when **no** slot is processing — which is also the proof it
   cannot spin, because the popped task is then always admitted.
2. **SIGTERM hung with a deferred queue** (pre-existing, made ordinary by the feature).  A task that was never launched
   is answered by nothing, and `server_http_context::join()` waits for its HTTP worker, so the exit died inside HIP's
   teardown (`__hip_module_dtor` → `amd_comgr_destroy_metadata`; three runs out of three, confirmed with eu-stack).
   `server_context::terminate()` now drains `queue_tasks_deferred` (`server_queue::take_deferred_tasks()`) and aborts
   the running slots: exit in 0.1 s.
3. **`LLAMA_SERVER_KV_ADMIT_RESERVE` was documented but not parsed** in the first integration cut; the census arm of
   the new gate caught it and it is now asserted.

## Measurements of record (r365)

`scripts/gate-kv-admit.sh` (MANIFESTS gate 46) — arm by arm: a fitting pair is not deferred and its text equals the
kill-switch arm; two copies of `prompts/prose-rdna-boosts.txt` (5246 tokens, `cache_prompt: false`) answer 500/500 on
`LLAMA_SERVER_KV_ADMIT=0` and 200/200 with it, serialised (2.1 s then 4.0 s), and **both** answers read
`1dfd5ced6f3e` — the same text the prompt gives when it runs alone, the deferred request re-prefilling all 5246 tokens;
four overlapping prompts drain in order (2.1/3.9/5.7/7.6 s); the reserve census reads 2048 at `-c 8192`, 4096 at
`-c 16384`, 256 with the override and prompt+`n_predict` when capped; a non-unified server defers nothing; SIGTERM with
three pending deferrals exits in 0.1 s; a waiting task is offered once (no log storm) and the server idles at 0 CPU
ticks after the drain.  `--self-test` (every arm on the kill switch) FAILS the gate on 12 checks, so the gate can
reject the pre-amendment server.

A 14-arm stress campaign (4B Qwen3.5 = a GDN hybrid, `-c 8192`/`-c 16384`, 1 GPU) also covered a 12-request
mixed-size soak, q8_0 K/V caches, a client aborting while deferred, and repeat-run text stability.

## Found for the reporter, not caused by this work

A **Gemma4** server aborts (`src/models/gemma4.cpp:646`, the prune mask must be on the host) when two sequences are
co-batched under `--kv-unified` — no MTP, no speculation, and it reproduces with `LLAMA_SERVER_KV_ADMIT=0`; `-np 4`
without `--kv-unified` is clean.  Upstream file, so it is `TODO` #384 and an `upstream/` candidate.

## Files

| file | what |
|---|---|
| `submitted-0001-server-KV-admission.patch` | the submitted patch, unchanged (`sha256 d71b45b2cc4a83fb…`), kept as the reproduction |
| `SUBMITTED-README.md` | the submitter's own campaign README |
| `repro-r362.log` | the submitter's repro log (stock r362 vs r362 + the patch, three arms) |
