# state-pipe

Pipelines the batched checkpoint copy (the #189 save path and the #209 restore path) so the host memcpy overlaps the device copies.

Patch: `0001-llama-context-pipeline-the-batched-state-save-restore.patch`, a format-patch for `stew675/llama-cpp-rdna-boosts` release `v15-6050b467c-r148` (tree `80bc2009`, `rdna-boosts-all.patch` sha256 `b955a3c86cb47cdefc3a64d8c3ef488426b6703c178dea2b94c94f13c23f89cd`). Applied tree with the patch: `d8b0374d`. `llama-context.{cpp,h}` are identical in r144, r147 and r148.

## What it does

Files: `src/llama-context.cpp` (+116 / -16), `src/llama-context.h` (+4); `libllama` only.

The batched copy used to run the host memcpy into/out of the pinned `state_stage` and the device copies back to back. Pieces are now cut into up to `LLAMA_STATE_PIPE_GROUPS` size-balanced groups.

- Restore: pack group g into the stage and queue its async sets, then pack group g+1 while those copies run. One sync per backend at the end, as before.
- Save: queue all async gets, record one event per (backend, group), and memcpy each group out of the stage as soon as its events complete.
- A backend without events (e.g. the CPU backend) falls back to the old single sync. The Meta backend proxies events to the simple devices, so under `-sm tensor` both sides pipeline.
- Same bytes, same offsets, same order; only timing changes.

Kill switch: `LLAMA_STATE_PIPE_GROUPS=1` gives the old unpipelined copy. Default is 8.

An independent logic review found no issues: the stage is never overwritten while a queued copy reads it; events per (backend, group) are synchronized before return; the group split handles 0, 1, many and zero-size pieces; the fallbacks equal the old code.

Illustration only, on a different baseline (our r131 + our patches, Flash-Next IQ3_S, a 112.6 MiB checkpoint): a profile showed about 5.3 + 6.4 ms (memcpy + device) per restore and 5.7 + 6.3 ms per save, back to back.

## Setup

2x R9700 (gfx1201), native Linux, second slot Gen4-limited. Only `libllama` differs between arms. Env is the `dgx.sh` defaults unless noted. Configs are those of #209:

- G1: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + MTP head (shared Q8_0), 2 GPU `-sm tensor --n-cpu-moe 10` (command as in #186 / #209 README).
- G1L: G1 with `-sm layer`.
- G2: Qwen3.6-35B-A3B Q4_K_M on 1 GPU (`--n-cpu-moe 10`).
- G3: the same model on 2 GPU `-sm tensor`.

## Gates

r144 (`LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, since #227 was open then):
- Greedy output identical to stock on all four configs.
- ckrest (a restored checkpoint's 48 greedy tokens vs a fresh run, twice): 44/44 SAME.
- 0 faults in 22 servers.

r148 (default env, lazy idle slots on):
- Greedy output identical to stock per config.
- ckrest: 24/24 SAME.
- 0 faults in 12 servers.
- Decode t/s unchanged (noise).

## Timing

The gains are small: a few ms per turn. The restore saves about 5 ms of a 33-80 ms restored request, and follow-up turns improve by 1 to 4 %.

### r144, restored 4-token request, median ms

Control = patch with `LLAMA_STATE_PIPE_GROUPS=1` (G1 only).

| config | stock | + patch | control |
|---|---|---|---|
| G1, `-sm tensor` | 72.2 | 67.1 | 72.2 |
| G1L, `-sm layer` | 80.6 | 75.8 | |
| G2, 1 GPU | 37.1 | 32.2 | |
| G3, `-sm tensor` | 40.8 | 37.5 | |

### r144, follow-up turns, server prompt ms

G1 is the mean of 6 turns, G3 the mean of 4.

| config | prompt | stock | + patch | control |
|---|---|---|---|---|
| G1 | 110-113 tok | 678 | 665 | 677 |
| G1 | 32/58 tok | 421 | 409 | 419 |
| G1 | 1.6k tok | 1540 | 1527 | 1542 |
| G3 | 29 tok | 165 | 161 | |
| G3 | 32 tok | 165 | 161 | |
| G3 | 1.5k tok | 503 | 480 | |

### r148, restored request, median ms

| config | stock | + patch |
|---|---|---|
| G1 | 72.6 | 66.8 |
| G1L | 79.6 | 77.4 |
| G2 | 37.7 | 32.4 |
| G3 | 41.2 | 38.3 |

### r148, follow-up turns, server prompt ms

| config | prompt | stock | + patch |
|---|---|---|---|
| G1 | 110 tok | 720 | 714 |
| G1 | 32/58 tok | 443 | 434 |
| G1 | 1.6k tok | 1581 | 1568 |
| G3 | 29 tok | 166 | 161 |
| G3 | 32 tok | 168 | 161 |
| G3 | 1.5k tok | 503 | 489 |

## How the arms were run

Per server: warm-up, then a 400-token greedy prose + code generation (hashed), then ckrest, then on G1/G3 a 4-turn chat at two context offsets, recording the server-side prompt time per turn. The r144 arms were stock, + patch, and + patch with `LLAMA_STATE_PIPE_GROUPS=1` (control, G1 only); G1 ran 3 servers per arm (order rotated), G3 2, G1L and G2 1. The r148 confirmation ran stock and + patch with 2 servers per arm on G1/G3 and 1 on G1L/G2.
