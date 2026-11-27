# wip/prefill-fair-restart — the fair-share rotation restarts at slot 0 when no prompt is left to place

Written by Claude (AI).  One `git am` file made on `v18-919794500-r368` (the tree `scripts/apply-all.sh` makes: tip
`af7a57ba8`, tree `bc7898a2c`); applied, the tree is `c17350d8f`.
`0001-server-the-fair-share-rotation-restarts-at-slot-0-wh.patch`, +22 lines in `tools/server/server-context.cpp`
(block 16's `share_prompt_budget()`).

## What it fixes

Block 16's max-min fair share (`share_prompt_budget()`) walks the slots from `slot_next` and advances `slot_next` at the end
of every call.  The call is made on every step that may take prompt tokens, decode-only steps included, and nothing ever
resets the rotation.  So when two prompts arrive together, which slot is walked first -- and gets the integer remainder of
the window and the shortest-remaining skew's tie -- depends on how many steps all the earlier requests of the process took.
That decides how both prompts are cut into ubatches, and the values of a prefill depend on how it is cut.

The visible effect: one `/completion` request with two prompts, sent three times to one server with `cache_prompt: false`
and `temperature: 0`, does not give the same replies three times.  The first request after start is always the same; the
later ones differ at near-tie tokens, and which reply a later request gets is decided by the step count of the requests
before it (the same sequence of replies in every fresh process, so it is not a race).  Erasing both slots between the
requests (`POST /slots/<id>?action=erase`) does not change it; `LLAMA_SERVER_PREFILL_FAIR=0` does.  One sequence alone is not
affected (the walk order of one active prompt does not matter).

## Upstream context

That a token's values depend on how its batch is composed is llama.cpp's own, long-standing behaviour, and upstream has
treated it as expected; this change does not touch it.  What it removes is one input on top of that, specific to block 16,
that makes the composition of an identical request depend on the requests before it.  As of 2026-11-28:

- On CUDA / HIP a quantized `MUL_MAT` takes MMVQ up to `MMVQ_MAX_BATCH_SIZE` columns and MMQ (or another path) above it,
  with a different summation order (since #1412 in 2023; the limit of 8 since #5434 in 2024; per-architecture thresholds
  in 2026, e.g. #26079).  On this machine every quantized type gives the same column at widths 1..8 as alone (r368 already
  keeps that band pure) and a different one at 9..16.
- #3014 (2023): perplexity changes with `-b`; the view was that identical results across batch sizes are not achievable and
  not important.  #7745 (2024) added the only documentation of it, a note under `cache_prompt` in the server README that
  logits are not guaranteed bit-for-bit identical for different batch sizes.
- #16016 (2025, a CUDA deterministic mode, still open): the CUDA code owner did not want to maintain bit-for-bit guarantees
  across batch sizes and pointed to a single slot without prompt caching for that.  #23335 (2026): "This is expected - we
  use different kernels for different batch sizes."
- #25618 (open since 2026-07): speculative greedy output differs from non-speculative output on quantized targets,
  reproduced on CUDA, ROCm, Vulkan, Metal and CPU; a member traced it to the MMVQ warp count changing with the column count
  and to FA taking different kernels for one token and for a batch, and asked whether an opt-in batch-invariant mode is
  wanted.  #32081 (draft, 2026-11): `--batch-invariant`, CUDA and Metal so far, not reviewed yet.  NVIDIA users report the
  same (#28111, open: IQ types differ at 5..8 columns on SM120); #27407 (an RTX 3090 Ti) and #28768 (an R9700 on Windows,
  HIP) were closed as stale without an answer.

Whether any of this should go further than this repository is your call; this PR only asks about block 16's rotation.  If
you would rather have the restart opt-in, the default is one line.

## The change

On a call that finds no prompt to place, `slot_next` goes back to 0 and the call returns (it would have shared nothing).
Every set of prompts then starts the rotation from the same state; the steps of one prompt set rotate exactly as before, so
the fairness the rotation is there for is kept.  The first request after start is unchanged bit for bit (the rotation is at
0 there already).

- Kill-switch: `LLAMA_SERVER_PREFILL_FAIR_RESTART=0` keeps the old behaviour (the rotation carries over).
- Census: a restart that moves the rotation logs `fair-share rotation restarts at slot 0 (no prompt to place; N restarts)`
  at debug level (`SRV_DBG`, so `-lv 5`), with the running count.
- Server only; no backend code is touched, so every backend behaves the same.

## Measurements

One gfx1201 (R9700, 32 GB), ROCm 10.0.0, Linux; Qwen3.8-27B UD-Q4_K_XL, `--parallel 2 --kv-unified -c 65536 -ub 2048
-ctk q8_0 -ctv q8_0 -fa on`, DFlash2 draft (`--spec-type draft-dflash --spec-draft-n-max 5`) or no speculation,
`ROCBLAS_USE_HIPBLASLT=0` (so the F32 GEMM solution is the same in every process), nothing else set.  The request: one
`/completion` with two prompts (a 11.5K-token chat prompt with tools and a 2.4K-token English text), 640 tokens, greedy,
`cache_prompt: false`, sent three times to one server.  The table gives a hash of each reply per request.

| build | setting | prompt 1, requests 1 / 2 / 3 | prompt 2, requests 1 / 2 / 3 |
|---|---|---|---|
| r368 | draft, `LLAMA_SERVER_PREFILL_DUTY=1` | `8aeb` `8aeb` `8aeb` | `3ddc` **`593a`** `3ddc` |
| this change | the same | `8aeb` `8aeb` `8aeb` | `3ddc` `3ddc` `3ddc` |
| r368 | no speculation, duty 1 | `4224` **`d988`** `4224` | `be3e` **`2ba4`** `be3e` |
| this change | the same | `4224` `4224` `4224` | `be3e` `be3e` `be3e` |
| this change | draft, duty 1, `LLAMA_SERVER_PREFILL_FAIR_RESTART=0` | `8aeb` `8aeb` `8aeb` | `3ddc` `593a` `3ddc` (= r368) |

The first request is the same on both builds; with the change the later ones repeat it.  The kill-switch gives r368's
sequence back exactly.  The replies that differ, differ at near-tie wording (for example the same figures under another
heading), not in content.

Speed (the same setting with the draft, r368 / this change / this change / r368, a fresh server each, a few seconds of load
first; the duty cycle at its default):

| | r368 | this change | |
|---|---|---|---|
| one 31.9K-token prompt, prefill | 1241.8 t/s | 1240.6 t/s | -0.1 % |
| one short prompt, 512 tokens, decode | 44.40 ms/step | 44.53 ms/step | +0.3 % |
| the two-prompt request, wall time (two requests per server) | 19.74 s | 19.99 s | +1.3 % |

The two-prompt row compares different splits, not an overhead.  r368's first two-prompt request in each server runs on the
rotation left over by the warm-up requests (prompt 1 / prompt 2 at 813 / 394 and 802 / 388 t/s, 344 tokens for prompt 1);
the change starts it at slot 0 (767 / 427 and 765 / 426 t/s, 330 tokens).  Where both builds made the same split (r368's
second request), the times are the same: 19.77 / 19.92 s on r368 against 19.88 / 19.92 s with the change.  Which order is
faster depends on the two prompts; the change only makes it the same every time.

## Not tested

- More than two slots, more than one GPU, other models, Windows, and the CPU / other backends (the change is in the server
  only and does not depend on the backend).
- Tasks with children (`n > 1`), and TTS / realtime slots: they are not in the walk, so the change does not reach them,
  but no arm ran them.
- The duty cycle at its default (`LLAMA_SERVER_PREFILL_DUTY` unset, 16) adds another input that this change does not touch:
  the deferral is decided by measured step times, and the first request after start has no measurement yet.  With the duty
  cycle at its default the first request still differs from the later ones on this request (prompt 1: `8aeb`, then `4224`
  `4224` with the change; r368 gave `8aeb` `4224` `1def`).  With `LLAMA_SERVER_PREFILL_DUTY=1` (the rows above) the change
  makes every request the same.
- Only greedy replies were compared; sampled replies follow the same split.
