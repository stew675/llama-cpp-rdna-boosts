# wip/split-state-hit-copy

Removes a 2 KiB copy from the cache-hit path of the Meta split-state lookup (`ggml_backend_meta_get_split_state`), which runs for every tensor of every graph allocation under `-sm tensor`.

One format-patch on `v14-097cf33d9-r120` (tree `0fd9827d`); applied tree `dad68727`. Patch sha256 `7dc86e41…`.

## What changes

`ggml/src/ggml-backend-meta.cpp` only, +10 / −7:
- On a hit, the function used to do `ret = it->second.get(); ... return ret;`. `split_state_entry::get()` zero-fills and fills a full `ggml_backend_meta_split_state` (2128 bytes with `GGML_BACKEND_META_MAX_DEVICES` 16), and the named local was then copied into the return slot.
- Now the assertion (and the `NDEBUG`-off sum check) read the entry, and the function returns `entry.get()`, a prvalue of the return type, so C++17 guaranteed elision builds the state directly in the caller's object.
- Same values, same assertions, nothing else touched. `get()`'s zero-fill is left as it is: a build with the struct shrunk to 4 devices measured the same as this change, so the zero-fill does not seem to matter.

## Why

perf (dwarf call graphs, main thread) of short follow-up turns on our build: `ggml_backend_meta_get_split_state` 32-41 % of the main thread per turn (~85-123 ms of a 330-370 ms turn), via `ggml_backend_meta_buffer_init_tensor_impl` <- `ggml_gallocr_alloc_graph`. Of its self time, 53 % was one `rep movsq` of 0x10a qwords (the copy into the return slot) and 17 % one `rep stos` of the same size (the zero-fill in `get()`). After the change the function has one `rep movsq` left (2 before).

## Gates

Setup:
- 2x R9700 (gfx1201), native Linux.
- Stock r120 against r120 + the patch, same build dir; only `libggml-base` differs. Your default env (no local variables).
- The #209 configs:

| | model | split | other |
|---|---|---|---|
| G1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + shared Q8_0 MTP head | 2 GPU `-sm tensor` | `-ncmoe 10` |
| G1L | same | 2 GPU `-sm layer` | `-ncmoe 10` |
| G2 | Qwen3.6-35B-A3B Q4_K_M, no MTP | 1 GPU | `-ncmoe 10` |
| G3 | Qwen3.6-35B-A3B Q4_K_M, no MTP | 2 GPU `-sm tensor` | `-ncmoe 10` |

  G1's command is the one from #186 (reproduced in the #209 README); G1L uses `-sm layer`; G2 and G3 change `-m` and drop the MTP flags and the `-ot` / `-lm` / `-lzm` options; G2 sets `ROCR_VISIBLE_DEVICES=0`. The patch only runs under `-sm tensor` (G1, G3); G1L and G2 are there to show nothing else moved.

**Greedy hashes** (400 tokens, prose / code), identical to stock r120 on every config and to the #209 run on r115:

| config | prose / code |
|---|---|
| G1 | `a7a81f01b3bc` / `4bb43e3339c8` |
| G1L | `946ffc37b143` / `e2f7603679a0` |
| G2 | `047fd057911b` / `231119399578` |
| G3 | `c0cf0739bdab` / `930bcf1f4405` |

**`GGML_META_SS_VERIFY=1`** (your stale-entry check) with the patch, one server each on G1 and G3, greedy hashes plus the 4-turn chats: no abort, same hashes.

**Faults:** none in 16 servers.

## Timing

**4-turn chat** (a ~9k-char document + question, two short follow-ups, a ~6k-char tool result; prompt cache on; two chats per server at different document offsets). Server `prompt_ms`, mean over servers (individual values in brackets); G1 3 servers per arm, G3 2, alternating order:

| config | offset / turn | prompt tokens | stock r120 | + patch |
|---|---|---|---|---|
| G1 | 0 / first | 2378 | 2176 (2131 2219 2179) | 2159 (2159 2150 2167) |
| G1 | 0 / 1 | 92 | 678 (681 646 708) | 662 (683 643 659) |
| G1 | 0 / 2 | 32 | 443 (343 489 498) | 384 (469 340 342) |
| G1 | 0 / 3 | 1557 | 1589 (1541 1605 1621) | 1551 (1617 1524 1512) |
| G1 | 60000 / first | 2423 | 2088 | 2097 |
| G1 | 60000 / 1 | 113 | 693 | 689 |
| G1 | 60000 / 2 | 32 | 351 | 342 |
| G1 | 60000 / 3 | 1620 | 1473 | 1468 |
| G3 | 0 / first | 2348 | 649 | 634 |
| G3 | 0 / 1, 2 | 29, 32 | 168, 168 | 167, 166 |
| G3 | 0 / 3 | 1533 | 593 | 589 |
| G3 | 60000 / first | 2393 | 689 | 671 |
| G3 | 60000 / 1, 2 | 29, 32 | 168, 168 | 166, 167 |
| G3 | 60000 / 3 | 1595 | 402 | 393 |

The G1 offset-0 32-token turn is bimodal (~340 or ~490 ms) in both arms, so its mean moves with how many servers land in each mode.

**Decode** (the greedy hash runs, 400 tokens): G1 prose 124.8 -> 126.9 t/s, code 127.8 -> 130.1 (every pair positive); G3 109.8 -> 112.3, 111.0 -> 113.0. G1L (73.5 / 70.1 -> 73.5 / 72.7) and G2 (102.5 / 102.5 -> 108.3 / 108.7), where the patch does not run, moved by as much in their single unpaired runs, so I would not attribute the decode difference to the patch.

**Our build** (r118 + our patches; Flash-Next IQ3_S, `-ncmoe 47`, `-sm tensor`, MTP draft), 3 paired servers of the same 4-turn chat with 15 s idle gaps: 32-token follow-up 308 -> 291 ms (−5.4 %), 35-token 356 -> 334 ms (−6.0 %), 139-token −0.9 %, first turns −1.8 / −0.8 %, 1.6k-token turns −0.8 / −0.4 %; 16 of 16 multi-turn replies identical.
