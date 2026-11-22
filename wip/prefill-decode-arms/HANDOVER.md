# HANDOVER — `prefill-decode-arms`, state at 2026-11-22 end-of-session (passes 6-9)

**For a fresh session with no context. Read this file top to bottom before touching anything.**  It supersedes the
pass-5 handover; that one is in git history (`git -C ~/llama-cpp-rdna-boosts show 7b784bc0:wip/prefill-decode-arms/HANDOVER.md`),
and the dated record of every pass is [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) (passes 2-8) and
[`RESULTS-2026-11-21.md`](RESULTS-2026-11-21.md) (pass 1).

---

## 0. TL;DR — where this stands, and the ONE thing that is on fire

**What works.**  `llama-server --prefill-decode-arms` is built and measured: a prefill arm (main loop, `ctx_tgt`) and a
decode arm (`ctx_dec`, its own thread, co-owning the prefill arm's KV), a slot crossing only at a request boundary.
The tier-1 oracle passes on **every** shape tested — the 35B-A3B host tier, a fully resident 27B, a dense phi-4 and a
4B — cold and over warm pairs.  **Prefix reuse works** (the reverse handover, pass 7): a finished conversation stays
resident and the next request on that slot reuses it, at single-context parity.  Measured win: a decoder keeps
getting tokens every ~47-49 ms (p95 64-78 ms) through a 2-5 s prefill at **-21…-24 % prefill** (accepted cost,
`README.md` §3b), where the shipped single context stalls 300-880 ms.

**The one open bug (P0, this is why the session ended).**  With the arms enabled, on the maintainer's WebUI shape
(**Qwen3.8-Flash-Next IQ4_NL, `-sm tensor` on 3 GPUs, `--kv-unified -np 3 -c 204800`, WebUI**), a request whose prompt
is **~70K tokens (5 attached files + a text prompt, sent as ONE request)** behaves as if the model **cannot see the
files** — it answers as though only the trailing text prompt were present.  The server reports having read all ~70K
tokens.  With `--no-prefill-decode-arms` on the same shape the maintainer's UX was fine, and `-sm layer` + arms
"all worked" (slower decode).  **Not yet reproduced in a probe** (see §6 for exactly what was and was not ruled out,
and the four tests that will localise it in one session).

**Scope ruling (maintainer).**  This campaign ships **opt-in only** (`--prefill-decode-arms` / `mode on`).  Two items
are explicitly a **follow-up campaign**, not this one: (a) a **meta-aware device copy** for the handover (today a
`-sm tensor` boundary costs a serialised host round trip, §7), and (b) **speculative-decoding support** (the arms
currently *refuse* a speculator, §8).  Nothing is in `patches/`, no delivery block was created or amended, no tag.

---

## 1. State of every task

| task | state |
|---|---|
| **T0 — one allocator for the co-owner pair** | **DONE** (pass 3).  `enabling-0003-shared-kv-allocator.patch`: `struct llama_kv_alloc`, per-ubatch reserve/commit/release, `llama_memory_alloc_sync()` as the request-boundary handshake, `LLAMA_KV_ALLOC_SHARED=0` as the A/B control.  Called on both arms at every boundary and asserted clean |
| **T1 — the recurrent-state handover** | **DONE** (pass 1).  `enabling-0002-recurrent-handover.patch`, **0.45-0.66 ms** device-to-device against 34.5-36.1 ms serialised (52.7 MiB) |
| **T2 — the server cut** | **DONE** (pass 5).  `llama-server --prefill-decode-arms`; see §3 |
| **T3 — the two-tier oracle** | **DONE**: tier 1 = the cross-configuration greedy hash (`arms/gate-server-arms.sh`), tier 2 = KLD (`scripts/gate-prefill-logits.sh` → **0.000000 / 100 % top-p** on the WIP build, pass 6 §2) |
| **T4 — the opt-in gate** | **README §8 Q1 answered**: a tri-state `off/on/auto` is built (§3).  **Default is still `off`** — flipping it to `auto` is a delivery-behaviour change that belongs to the promotion decision |
| **T5 — measurement** | **DONE** (pass 5): ~47-49 ms ITL p50, 64-78 ms p95 through a 2.0-5.3 s prefill at **-24…-27 %** (`-ub 512` shape) / **-21…-24 %** flat from `-ub 2048` up.  Full residency: **-24.2 %** with a **34.9 ms** worst gap against the single context's 327.3 ms.  **`-ub`/`-b` 4096 is the sweet spot** |
| **T6 — the REVERSE handover (prefix reuse)** | **DONE** (pass 7).  The decode arm frees nothing at the return; the main thread takes the recurrent state **back** to the prefill arm and keeps the prompt.  `gate-ckpt-reuse.sh` on the arms path **SKIP → PASS** (3 restores, **33 ms** turns instead of 685 ms).  Oracles: `arms/gate-server-reuse.sh` (§4) |
| **T7 — dense + fully-resident coverage, and the option semantics** | **DONE** (pass 8).  Dense (phi-4): fork-below-prefix **also reuses** (`cache_n=220`, 20.2 vs 97.0 ms).  Fully resident 27B: parity with the single-context server's own reuse path, byte for byte.  Option tri-state + deprecation messages (§3) |
| **T8 — `-sm tensor` correctness** | **THREE CRASHES FOUND AND FIXED** (pass 9, `~/llama.cpp` `b822f89fc`), §7.  What remains on that path is the **serialised handover cost** (~74.6 ms per boundary) — follow-up campaign |
| **T9 — the long-prompt bug** | **OPEN — P0.  §6.  Start here.** |

---

## 2. What exists as code (do not re-derive any of it)

**Delivery repo** (`~/llama-cpp-rdna-boosts`, branch `prefill-decode-arms`):

| artifact | what |
|---|---|
| [`enabling-0001-second-context-shared-kv.patch`](enabling-0001-second-context-shared-kv.patch) | `llama_context_params::ctx_other_share_memory`, the generic same-model `ctx_other` gate, `llama_kv_cache::other_coowner` (a second context **owns** cells in the shared map; a mirror skips the mutable paths and reads garbage), the plain-hybrid wiring |
| [`enabling-0002-recurrent-handover.patch`](enabling-0002-recurrent-handover.patch) | `llama_memory_recurrent::seq_state_take()` (one batched device-to-device copy of a sequence's recurrent rows + cell metadata), `llama_memory_seq_state_take()` (llama level), `seq_rm_recr()`, `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY` (the serialised control) |
| [`enabling-0003-shared-kv-allocator.patch`](enabling-0003-shared-kv-allocator.patch) | T0, the one allocation owner |
| [`arms/two_arms.cpp`](arms/two_arms.cpp) + `build-two-arms.sh` | the instrument: two arms outside the server, modes `single`/`batch-pair`/`arms`/`handover`/`pf-alone`/`dec-alone`, `ARMS_*` knobs, `ARMS_LOGITS_DIR` row dumps |
| [`arms/gate-server-arms.sh`](arms/gate-server-arms.sh) | **tier-1 oracle**: the two-arm server's reply must be byte-identical to the single-context server's, both slots; `REPS=N` = N request pairs in one process (the warm reps — the interesting ones); `--compare` |
| [`arms/gate-server-reuse.sh`](arms/gate-server-reuse.sh) | **the reuse oracle** (pass 7/8).  `GATE=strict` = a reused prefix moves no token (`cache` vs `nocache`); `GATE=parity` = the arms' reuse path equals the single-context server's own (`cache` vs `ref`); `CONCURRENT=1` = two slots resuming in parallel (**stress only, no text claim** — §63); `TWO_SLOT=1` = two resident conversations, never co-batched (a valid text oracle); `NO_HOST_TIER=1` = fully resident shape; `FORK_REUSE=1` = require the fork to reuse (the dense claim) |
| [`arms/smooth.py`](arms/smooth.py) | the campaign's metric: the decoder's ITL **inside** the prefill window, tokens through it, `pf_tps`; `--warm N` |
| [`arms/interactive.py`](arms/interactive.py) | the user's version: a reply streaming while another chat pastes a long document (TTFT, in-window gaps, per-second token table) |
| [`runs/`](runs/) | every arm of passes 2-9 (see the results files for which directory is which) |

**The WIP fork** (`~/llama.cpp`, branch `wip/prefill-overlap`): 9 commits, tip **`b822f89fc`**, tree **clean**.

| commit | what |
|---|---|
| `6f2a74717` | the opt-in second context that shares the first one's KV (0001) |
| `3f2768131` | the per-gather split yield point — **a measured loser, kept as the record.  Do not re-enable** |
| `540d442a8` | stream priority — **also a measured loser, kept as the record** |
| `d8a696252` | the recurrent-state handover (0002) |
| `53ad4a6d4` | the compact env-gated cell censuses |
| `0aff8602f` | T0, one allocation owner (0003) |
| `9bb97b741` | **the server cut** (T2) |
| `7c114eec0` | `-ib` no longer counts a decode-arm slot (so `-ib` is inert here) |
| `16e942d2f` | **the REVERSE handover** (prefix reuse) |
| `51c1fb8ec` | the option tri-state + `auto` fallback + the `-ib`/`DUTY` deprecation messages |
| `b822f89fc` | **the `-sm tensor` safety work**: the take's views carry their parent buffer; the serialised fallback; the meta view-slot fix |

**Build** (the only tree that has any of this):

```bash
cd ~/llama.cpp && git checkout wip/prefill-overlap
BUILD_DIR=build-2ctx JOBS=24 ~/bin/build-llama-rocm-101      # ~10 s warm, 6-10 min cold
# binaries: ~/llama.cpp/build-2ctx/bin/{llama-server,llama-cli,llama-perplexity,llama-bench,...}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
export ROCBLAS_USE_HIPBLASLT=0                 # purity pin: a second context can move the hipBLASLt choice MID-PROCESS
```

---

## 3. The server cut and its option (as built)

**The flow.**  `post_decode()` samples the prompt's last row **on the prefill arm** (its logits live there) →
`hand_over_to_decode()`: allocation handshake → post the slot → **wait for the take**.  The decode thread runs
`llama_memory_seq_state_take(mem_dec, mem_tgt, id)` at the top of its next pass, then marks the slot ACTIVE and steps
**one token per active slot in ONE batch**.  On stop the decode arm frees **nothing**: the main thread runs
`take_state_back()` (the mirror take, `mem_tgt ← mem_dec`) under `arms_quiet` and **keeps the prompt**, so the sequence
stays resident and the next request reuses its prefix.  `slot.dec_state` (under `m_arms`) says where the recurrent
state currently lives.

**Hard-won rules inside that block (do not "simplify" them away):**
* the take is **POSTED** to the decode arm, never taken under a lock the prefill arm waits on (a direct mutex take
  starves the prefill arm for ~2.85 s);
* the prefill arm **waits** for a posted take before it starts the next prompt (`wait_handover_taken()`, also at the
  top of `pre_decode()` as a second line of defence) — breaking this silently changes the next prompt's recurrent cell
  index and the reply's text (`ARMS_NO_WAIT_TAKE=1` reproduces it deliberately);
* the allocation handshake (`arms_boundary_sync`) and the reverse take both run inside `arms_quiesce()` /
  `arms_resume()`, because a reservation is legal while a ubatch is prepared-but-not-yet-applied;
* **a failed reverse take DROPS the sequence** (`prompt_clear()`): an attention half with no recurrent half is an
  inconsistent state, and a forced re-prefill is the safe failure;
* never two **decode-band** contexts (the per-table `GGML_MOE_DEVREMAP` map is process-wide) — one arm prefills, the
  other decodes, always;
* a slot belongs to exactly one arm at a time, transitions only under `m_arms`.

**The option (as asked for by the maintainer):**

| form | behaviour |
|---|---|
| `--prefill-decode-arms` / `--no-prefill-decode-arms` (env `LLAMA_ARG_PREFILL_DECODE_ARMS`) | **explicit**: the pair, or **refuse to start** if it cannot be honoured (never a silent downgrade) |
| `--prefill-decode-arms-mode off\|on\|auto` (env `LLAMA_ARG_PREFILL_DECODE_ARMS_MODE`) | `auto` enables when the shape can carry the pair and **falls back to single-context with a warning** when it cannot (the refusal reason is logged above it).  Whoever comes last wins |
| **default today = `off`** | unchanged delivery behaviour.  **Promotion flips it to `auto`** — that is a decision, not a gate outcome |

**Verified option behaviour** (4B, pass 8 §3): `auto -np 2` creates the pair; `auto -np 1` warns and stays
single-context; `auto` + a refused feature (`--sleep-idle-seconds`) **falls back and keeps serving**; `on` + the same
**exits 1**; with no arms flag there are **no** new messages at all.

**Deprecated on this path (maintainer's ruling: ignore them, and say so).**  `-ib` / `--interactive-batch-size` and
`LLAMA_SERVER_PREFILL_DUTY` print a one-shot notice **only when set and only when the pair is running**: both exist to
bound a decoder's stall inside a **shared step**, and this path's decoder shares no step.  Measured inert: the duty
cycle moves 75 → 73 decoder tokens and the prefill stretch 1.01x.  **Block 16's fair-share half is NOT deprecated** —
it is live (1.85 s → 0.79 s on the fairness gate's shape).

---

## 4. Prefix reuse (the reverse handover) — and its two contracts

`gate-ckpt-reuse.sh` (4B, `cache_prompt:true`, four growing turns): **SKIP → PASS**, 3 checkpoint restores, resumed
turns **111.0 / 108.7 / 106.9 ms** on the 27B against the single-context control's 109.7 / 106.5 / 106.8 (33 ms on the
4B).  Before the fix the arms path re-prefilled everything: **684-686 ms** per appended turn (**~20x**).

`gate-server-reuse.sh` has **two contracts — do not conflate them**:

* **`GATE=strict`** (`cache` vs `nocache`): a reused prefix moves **no token**.  True on the 4B, phi-4 (dense) and the
  35B host tier.  **NOT true on the fully resident 27B — and the single-context server fails it identically there**, so
  it is a *delivery* property (a checkpoint-restored placement is not a re-prefilled one), not an arms defect.
* **`GATE=parity`** (`cache` on the arms vs `cache` on the single-context server): the reverse handover must be
  **equivalent to the reference server's own reuse path**.  Holds everywhere measured (`deda991b9954a821` on both, 27B
  fully resident) and **this is the contract a promotion should be judged on**.

Measured reuse wins: 35B host tier B (extends) 420.5 vs 851.4 ms (`cache_n=395`); dense phi-4 B 139.9 vs 256.6 and the
**fork below the prefix** 20.2 vs 97.0 ms (`cache_n=220`, `FORK_REUSE=1` enforces it); 27B fully resident, two slots
resuming **concurrently** 365.1 / 262.1 ms against 1295.1 / 1296.0 (**3.5x / 4.9x**).

---

## 5. Numbers of record (all: 1 x R9700 unless stated, `ROCBLAS_USE_HIPBLASLT=0`, host-tier arms also
`MOE_EXPERT_HEATMAP=load`, sequential, warm)

| | host tier (arena 5000, `-ncmoe 40 -lme auto`) | full residency |
|---|---|---|
| prefill alone | 2955 → 3800 t/s depending on `-ub` | 6320.8 t/s |
| prefill cost of the overlap | **-21…-24 %** (flat from `-ub 2048` up) | **-24.2 %** |
| single-context server, same run | -34.7 % and a 300-880 ms stall | -34.7 % and a 327 ms stall |
| two arms, decoder's worst in-window gap | 60-131 ms | **34.9 ms** |
| the pasted document (interactive) | 1.13 s / 3111 t/s at `-ub 4096` | 737 ms / 4786 t/s |

Tier-1 oracle acceptance values (35B-A3B Q4_K_M, arena 5000, `-c 8192 -b 2048 -ub 512 -np 2`, two concurrent ~500-token
requests, 32 tokens, temp 0, seed 42): **SLOT 0 `459865c93d55cbaf`, SLOT 1 `7dc20b0ce0bac255`**, census ~`h=0.690`,
`HANDSHAKE the pair shares one allocation owner`.  Gate-fires control: `-ub 256` moves slot 0 to `182a9fcfe1de6a70`.

---

## 6. **THE OPEN BUG (P0) — a ~70K-token prompt is "invisible" except its tail, arms only**

### 6.1 The maintainer's report, verbatim in substance

* Shape: the WebUI on **Qwen3.8-Flash-Next IQ4_NL** (`STEW675-IQ4_NL/runme`): `-sm tensor` on **3 GPUs**,
  `-c 204800 --parallel 3 --kv-unified --load-mode dio --lazy-mode mlock -b/-ub 4096`,
  `GGML_CUDA_SLAB_HEADROOM_MIB=1536`, `--prefill-decode-arms`.
* Prompt: **5 files attached in the WebUI + "Please perform a code-review of this project", ~70K tokens, ONE request.**
* Symptom: the model behaves as if it **cannot see the files** and answers as though only the trailing text prompt
  were present.  The server reports having read all ~70K tokens.
* Scope: **only with the arms enabled.**  With `--no-prefill-decode-arms` on the same shape the UX was fine; with
  `-sm layer` + arms "it all worked" (decode much slower).  Also seen: speculative decoding is refused (§8).

### 6.2 What is already ruled OUT (do not redo)

* **1 GPU, 4B Q8_0, 15.7K-token prompt, `-ub 4096` (4 chunks), arms ON vs OFF**: identical and **correct** —
  `prompt_n=15773`, the answer is the marker placed at the very start; and with `cache_prompt:true` a repeat reads
  `cache_n=15769` / `prompt_n=4` and is still correct.  So a multi-chunk prompt is not broken by the arms per se, and
  the reuse path is not broken on that shape.
* **Flash-Next, `-sm tensor`, 3 GPUs, 15.7K prompt, arms ON and OFF**: both returned **empty content** (that prompt
  framing makes this thinking model emit EOG immediately) → **inconclusive, NOT a reproduction**.  If you retry it,
  read `reasoning_content` as well as `content` and give it more tokens.

### 6.3 Hypotheses, most likely first

1. **H1 — phantom prefix reuse (the most likely, and it is code from pass 7).**  The reverse handover keeps
   `slot.prompt` across requests.  If the KV that prompt refers to is ever gone while the bookkeeping survives, the
   next request computes `n_past` = the full retained prefix, reports `cache_n` ≈ 70K, evaluates only the trailing
   tokens — and the model answers as if the files were never ingested.  This is exactly "forgetting everything prior
   to the very last prompt" and it can only happen with the arms, because only then does the prompt survive a request
   while the state moves between contexts.  Candidate sources of "cells gone, prompt kept": an eviction of the unified
   KV under pressure where the save/clear pairing fails, a sequence dropped by a failed take while the prompt was not
   cleared on some path, or a cancel (`arm_release`) that takes the state back onto a sequence whose cells the decode
   arm had already released.
2. **H2 — the long prompt itself on the meta path.**  `-sm tensor` means the *handover* is the **serialised state
   move** (§7) and the recurrent cache lives in the Meta split-state buffer.  If that move restores only part of the
   state (or the decode arm's `n_kv`/band range is derived from a co-owner's view of the cells), the decode arm would
   attend over the tail only.  It would be `-sm tensor`-specific — which matches "it all worked with `-sm layer`".
   Note this is a *decode-side attention* failure, so the prefill's own `prompt_n` would still read 70K.
3. **H3 — the context limit / shift.**  With `--kv-unified`, each slot's `n_ctx` is the whole `-c` (204800), so a 70K
   prompt fits — but confirm it in the log (`new slot, n_ctx = ...`) and look for `slot context shift` (a shift drops
   the oldest half by design and would look exactly like this).  Falsifiable in one grep.
4. **H4 — the WebUI sends the files as several requests.**  Then the failure would be in the *append* path (each
   request reusing the previous prefix).  Distinguishable from H1 only by the request count — check the log's
   `new prompt` / `prompt done` lines, or `prompt_n` per request.

### 6.4 The four tests, in order (each cheap, each bounded)

1. **Fresh server, ONE request, arms on, the 70K prompt** (no prior traffic on that slot).
   - **fails** → it is H2/H3 (not reuse).  Go to test 3.
   - **passes** → it is **H1**: reuse.  Go to test 2.
2. **Two requests with `cache_prompt:true`** (the files, then the question).  Read the second response's `timings`:
   a **large `cache_n`** with an answer that ignores the prefix = phantom cache confirmed.  Cross-check the log:
   `cached n_tokens = N, memory_seq_rm [N, end)` (what it thought it had) and whether
   `forcing full prompt re-processing due to lack of cache data` appears (a hybrid model without a usable checkpoint
   must print that and re-prefill).
3. **Same 70K prompt, `-sm layer`, arms on** — the maintainer says this worked.  If it also fails, the meta path is
   exonerated and H1/H4 is the whole story.
4. **Same 70K prompt on a single GPU** (27B fully resident, `-c 131072`, one device) with arms on, one request:
   removes `-sm tensor` from the picture entirely.

Always capture: the response `timings` (`prompt_n`, `cache_n`), and the server log with **`-lv 4`** (`new prompt`,
`prompt done: posted to the decode arm`, `allocation handshake clean`, `cached n_tokens`, `slot context shift`,
`forcing full prompt re-processing`, `restored context checkpoint`).

### 6.5 Containment while it is open (for real use of the server)

The reuse half is what H1 implicates, and it is one code path.  If a build that cannot exhibit a phantom prefix is
wanted **now**, the minimal, reversible change is a kill switch that restores pass-6 behaviour — `prompt_clear()` on
the return instead of keeping the prompt (i.e. no cross-request reuse; a re-prefill per request).  **Not implemented
yet** — it is a behaviour change and wants the maintainer's go-ahead, but it is ~6 lines in
`finish_returned_slots()` / `arm_release()` behind `LLAMA_ARMS_KEEP_PROMPT=0` (default ON = today's behaviour).

---

## 7. `-sm tensor` and the Meta split-state cache (three crashes, all fixed; one cost remains)

With `--split-mode tensor` the delivery puts state caches in the **Meta split-state buffer**, which has **neither
`cpy_tensor` nor `cpy_tensor_n`**.  Three distinct failures were found on the maintainer's shape and fixed in
`b822f89fc`:

1. **The take's views had `buffer == NULL`** (`ggml_new_tensor_impl` leaves an out-of-graph view unbacked; only the
   graph allocator fills it in) → `GGML_ASSERT(buffer)` in `ggml_backend_buffer_get_type`, on the first handover.  A
   backend with `cpy_tensor_n` (CUDA) never looks, which is why every 1-GPU shape had passed.  Fix: the views carry
   the parent's `buffer`/`data` (what `ggml_backend_view_init` does, minus its allocation hook).
2. **No device-side copy exists for that buffer**, and its host path cannot resolve a strided view of a split tensor
   (`SIGSEGV` in `ggml_backend_meta_get_tensor_impl`).  Fix: `llama_memory_seq_state_take()` **falls back to the
   serialised state move** (`state_write`/`state_read` with `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY`, the shipped
   prompt-cache path, i.e. the campaign's measured serialised control) when a new public query
   `ggml_backend_buffer_can_copy_tensor()` says the buffer has no device copy.  **This is NOT a fallback to the
   single-context mode** — the two arms keep running; only the per-boundary state move changes shape.  It logs its
   size and its cost:
   ```
   took seq N as a serialised state move (112.57 MiB: 38.5-39.2 ms out, 35.4-36.0 ms in, ~74.6 ms total)
   ```
   One per **request boundary** (so it lands in TTFT, not in decode).  Single-GPU shapes never take this path.
3. **A process-global view-slot selector raced between the two arms.**  `g_meta_view_slot` chose one of **two**
   rotating view containers per buffer, and it is read more than once per tensor (register a view's per-device
   children, then look them up).  With two contexts on two threads, one arm flipped it in between → a child lookup
   returned **null** → `GGML_ASSERT(tensor)` in the prefill arm's own `sched_copy_input`, or a 0-row tensor →
   *"tensor write out of bounds"*.  It landed on the **second slot's first prefill**.  Fix: the selector is now
   `thread_local`, `n_view_slots` is **4**, and a **co-owner** context (`ctx_other != nullptr`) takes the second pair
   (slots 2/3) while the owner keeps 0/1.
   **Verified**: 2 slots (slot 0 generating 128 tok while slot 1 prefilled 3000 chars) and **3 slots** (slot 0
   generating 192 tok, slots 1+2 prefilling) — **0 asserts**, 192/32/32 tokens at 44/35/26 tok/s.

**Still owed on this path (follow-up campaign, maintainer's scope ruling):** a **meta-aware device copy** — resolve
each device's *simple* tensor through the meta backend and issue the sub-range copy per device — which would take
those ~74.6 ms boundaries back to sub-millisecond.  Until then `-sm tensor` *works* but pays that per request.

---

## 8. What is refused by design (not bugs)

Refused at startup with a message: a **speculator** (`--spec-type`; it owns the draft context and a per-slot
checkpoint path that would have to cross the arms), **mtmd/mmproj**, TTS/realtime audio, embeddings/rerank/decision
(any pooling), **LoRA adapters**, `--sleep-idle`, backend sampling.  **`-np 1` warns and stays single-context** (no
second context, no second graph reserve).  Speculative decoding is therefore a **not-yet-supported**, and the
maintainer has scoped it to the follow-up campaign.

---

## 9. Gates: what passes, what needs an arm instead of a pass

| gate | arms path | note |
|---|---|---|
| `scripts/gate-prefill-logits.sh` (KLD) | **cannot reach the arms path** (`llama-perplexity` has one context) | on the WIP build it reads **0.000000 / 100 %** vs the canonical base — the library artifacts are bit-inert on the default path |
| `scripts/gate-prefill-fairness.sh` | reachable; **red as shipped** only on `ITL_c <= 1.75 x ITL_b x chunk/b` = **13 ms** (below the decode arm's own 28-30 ms step); **green** with `--max-itl-ms 120` | its baseline numbers are the campaign in one line: **single-context worst in-window ITL 323 ms, arms 29 ms**.  `--self-test` still rejects the unfixed build.  Needs an arms-aware ITL criterion |
| `scripts/gate-prefill-duty.sh` | **inert** (75 → 73 tokens, stretch 1.01x); reads 0.99x at a 1.0x floor — i.e. noise | retire the duty on this path (maintainer ruling) and give the gate a **two-sided band** |
| `scripts/gate-park-idle-slots.sh` | **SKIP by design** — a co-owner pair is not "one stream", the cache declines to park | its shipped shape also carries an MTP draft (refused) and its model is not on this box |
| `scripts/gate-ckpt-reuse.sh` | **PASS** (pass 7) | 3 restores, 33 ms turns |
| `arms/gate-server-arms.sh` | **PASS** on 35B host tier / 27B resident / phi-4 / 4B, cold + warm | the tier-1 oracle |
| `arms/gate-server-reuse.sh` | **strict** PASS on 4B/phi-4/35B, **parity** PASS everywhere | §4 |
| `scripts/gate-moe-promise.sh` | **cannot reach the path as shipped** (`-np 1` hard-coded, model absent) — but its **criteria are met** by the oracle's exit census: 0 repair lines, `broken=0 repaired=0 stale=3360` cold / `10338` warm | promotion owes an **arm**, not a pass |
| `scripts/gate-host-pool.sh` | **cannot reach the path** (llama-cli only) | a server-side pool arm is owed |
| `scripts/check-gpu-faults.sh` | **clean** over 50+ logs across passes 6-9 | run it on every arm |

---

## 10. Rules and traps (each of these cost a session; several cost this one)

**Measurement**
* **One GPU arm at a time**; pin `HIP_VISIBLE_DEVICES`; `ROCBLAS_USE_HIPBLASLT=0` on **every** arm (a second context
  can move the hipBLASLt choice **mid-process**); `MOE_EXPERT_HEATMAP=load` on every host-tier arm.
* **A leaked server answers `/health`.**  Twice this session a stale server made the next arm silently drive the
  *wrong* process (and the run read as a state failure).  **Always check the port first** (`ss -ltnp`), and remember
  the box runs a **`llama-slot-proxy` on 8037-8039** (`pid 1859`) that answers those ports and forwards elsewhere —
  never use them.
* **Never `wait` bare** in a script that backgrounded the server — it waits for the server too, and the run looks
  hung.  Bound everything: `timeout N`, a health loop with a max, `kill` at the end.
* **`jq --arg` cannot carry a 190 KB prompt** ("Argument list too long").  Use `jq -n --rawfile p FILE`.
* **The cold host-tier rep is not a stable oracle** (§89 residency class): the single-context arm's first pair can
  read `182a9fcfe1de6a70` with `h=0.6293` while the pair arm reads `459865c93d55cbaf` with `h=0.6905`, and reps 1-2
  then agree in both arms.  **Use the warm pairs** (`REPS=N`) or warm before recording.
* **A two-slot CONCURRENT pair is not a text oracle** (`GREEDY-PURITY.md` §63: co-batching changes the ubatch shape
  and hence the kernel family).  `gate-server-reuse.sh CONCURRENT=1` therefore asserts no text — only that the resume
  reused and nothing aborted.  `TWO_SLOT=1` (sequential, two resident) *is* a text oracle.
* **A float/pair diff needs `ROCBLAS_USE_HIPBLASLT=0` on that run only**; a per-*start* difference is the library.

**Model / shape**
* Flash-Next is a **thinking** model: check `reasoning_content` as well as `content`, and give it tokens; a 16-token
  budget can read as an empty reply.
* `--kv-unified` makes **each slot's `n_ctx` the whole `-c`**; an explicit `-np N` **without** `--kv-unified` leaves
  `kv_unified=false` (only `-np auto`, which expands to 4 slots, sets it) — and with it parking and block 05's r133
  band skip.
* `-ub`/`-b` 4096 is the sweet spot; 8192 buys +8 % prefill for a worse decoder tail.  `-ib` is inert here; do not
  reach for it.

**Repo**
* `llama-cli` text/hash arms take **no `-lv`**; census arms take **`-lv 4`**; the server hashes over HTTP so `-lv 4`
  is safe there.
* `llama_process()` is async — never time a prefill without `llama_synchronize()`.
* Never push from `~/llama.cpp` except the release-time `rdna-boosts` refresh to the personal fork.  Nothing today
  was pushed anywhere.
* `wip/` is **not** the delivery; a block is always a question, never a gate outcome.

---

## 11. Exact commands that work

```bash
# build
cd ~/llama.cpp && git checkout wip/prefill-overlap && BUILD_DIR=build-2ctx JOBS=24 ~/bin/build-llama-rocm-101

# tier 1 (product shape, 35B host tier): cold, then the 3 warm pairs, then the compare
cd ~/llama-cpp-rdna-boosts/wip/prefill-decode-arms/arms
OUT=$PWD/../runs/oracle-$(date +%H%M) ./gate-server-arms.sh --mode ref
OUT=$PWD/../runs/oracle-$(date +%H%M) ./gate-server-arms.sh --mode arms
./gate-server-arms.sh --compare ../runs/oracle-HHMM/ref.txt ../runs/oracle-HHMM/arms.txt
OUT=$PWD/../runs/oracle-warm REPS=3 ./gate-server-arms.sh --mode arms     # the WARM pairs are the oracle

# reuse: strict (reuse moves no token), parity (arms == single-context reuse), dense fork, two slots, concurrent
./gate-server-reuse.sh
GATE=parity ./gate-server-reuse.sh
NO_HOST_TIER=1 FORK_REUSE=1 MODEL=/llm/models/phi-4/IQ4_XS/phi-4-IQ4_XS.gguf ./gate-server-reuse.sh
NO_HOST_TIER=1 TWO_SLOT=1 MODEL=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf ./gate-server-reuse.sh
NO_HOST_TIER=1 CONCURRENT=1 MODEL=... ./gate-server-reuse.sh              # stress: no text claim

# the server (a 4-slot shape that worked, 27B fully resident, 1 GPU)
~/llama.cpp/build-2ctx/bin/llama-server -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -c 32768 --kv-unified -np 4 -b 2048 -ub 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8 \
  --host 0.0.0.0 --port 8080 -lv 4 --prefill-decode-arms

# the maintainer's Flash-Next shape (3 GPUs, tensor split, ~20 s to load) — the -sm tensor path
GGML_CUDA_SLAB_HEADROOM_MIB=1536 HIP_VISIBLE_DEVICES=0,1,2 ~/llama.cpp/build-2ctx/bin/llama-server \
  -m /llm/models/Qwen3.8/Flash-Next/STEW675-IQ4_NL/Qwen3.8-Flash-Next-MTP4-IQ4_NL-00001-of-00002.gguf \
  --fit off -ngl all -c 65536 --parallel 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0 \
  --load-mode dio --lazy-mode mlock -sm tensor -t 7 -b 4096 -ub 4096 -lv 4 --prefill-decode-arms

# diagnostics (both free when off)
LLAMA_RS_DEBUG_TAKE=1    # the tensors a take copies, with each side's buffer/ne/data
GGML_META_DEBUG_SET=1    # every meta init_tensor (verbose) + any set_tensor whose child is missing
LLAMA_KV_DEBUG_CELLS=1, LLAMA_RS_DEBUG_CELLS=1   # the cell censuses (engine side, -lv 4)
```

---

## 12. Repo state at the end of this session

* `~/llama-cpp-rdna-boosts` (branch `prefill-decode-arms`): this handover + `RESULTS-2026-11-22.md` passes 6-8 +
  `WORKLOG.md` 261122.011/012/013 committed.  **Nothing in `patches/`, no block, no tag, nothing pushed.**
* `~/llama.cpp` (branch `wip/prefill-overlap`): tip **`b822f89fc`**, **clean**.  Not pushed.
* No servers running; VRAM released.  The 4-slot test server the maintainer asked for was started on :8080 and then
  stopped at their request.

---

## 13. Do this, in this order

1. **Reproduce the P0 long-prompt bug** (§6.4, tests 1→4).  Do not start from a fresh theory; four tests settle it.
   If it is H1 (reuse), the phantom cache is in `finish_returned_slots()`'s retained prompt and fix there.  If it is
   H2, it is the meta/serialised-take path and the follow-up campaign's device copy is the real answer — in which
   case the honest first-shippable move is to **refuse `-sm tensor` with the arms** (a guard) rather than ship it.
2. **Report the finding before any delivery-affecting commit.**  The maintainer decides fix-now / defer / drop.
3. If the bug is fixed: the containment question (§6.5) and the promotion question come back to the table.  The
   promotion package would be: the two gate *arms* (§9), a long multi-turn soak (cells filling →
   `try_clear_idle_slots` on the arms path is **untested**), a dense fork arm, a server-side `gate-host-pool` arm,
   then `validate-set.sh` + `ENVIRONMENT.md` for the option — and the **block structure, which is the maintainer's
   call alone**.
4. **Still not run, and owed before promotion:** a long multi-turn soak; the arms-vs-single text comparison on a
   *streamed/windowed KV* shape (block 14's window and pool are off by default, so nothing measured here covers them).
