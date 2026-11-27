# wip/idle-clear-first-r366: server -- with lazy idle slots off, save and clear the idle slots before the prompt-cache restore

Thank you for the prompt-cache room retry (`prompt_load_with_room`) and the idle-slot parking. With the lazy idle slots on, two long agents that cannot both fit in the KV cache take turns cleanly. We run with `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, because on our production setup the lazy idle slots still cost 4-6 % decode (the output is now stable, thanks to the parking). With the lazy idle slots off, we found that the same workload re-processes each agent's whole prompt on every turn after the first. We're sharing the data and a small patch (0001) in case it's useful. With the default settings nothing changes, nothing is urgent on our side, and we run it in our own build in the meantime.

**What we saw.** With `cache_idle_slots` on a unified KV cache and the lazy idle slots off, `process_single_task` restores the launching slot's prompt (`update_prompt_cache`) while the other idle slots still hold their cells, and saves and clears them only after the launch. Two things follow:
- A restore that needs those cells fails (`state_read_meta: failed to find 131522 available cells in kv cache`, then `failed to load prompt from cache`), and the whole prompt is processed again. The retry that makes room runs only with the lazy idle slots on.
- A restore that fits is placed above the idle slots' cells, which are cleared a moment later, so the sequence keeps that offset for its whole life. The QSA indexer top-k regular-layout fast path (cell j at position j) then declines on every ubatch of it. In the repro below, the first turn appends its 3.6K tokens at 1623 t/s on stock r366, against 2421 t/s with 0001.

**0001** (`tools/server/server-context.cpp`, +41): on the path above (`plan.update_cache`, a prompt cache, `cache_idle_slots`, lazy idle slots off, unified KV), the idle slots other than the launching slot and its children are saved to the prompt cache and cleared before the restore. The entry to be restored is held out of the cache while they are saved, as `update_prompt_cache` does on the lazy path, so their saving cannot evict it. The after-launch pass still runs and finds those slots empty. With the lazy idle slots on, the code is not reached. Kill switch: `LLAMA_SERVER_IDLE_CLEAR_FIRST=0`.

Limits we know of:
- The restore re-runs `find()` after the idle slots are saved, so an idle slot holding a near-prefix of the new prompt (one `select()` did not choose) can be picked instead of the held entry. That costs extra prefill, not wrong output.
- While the idle slots are saved, the cache briefly holds one more entry than before (the held one), so under a tight `-cram` the launching slot's own save can evict an older entry that stock would have kept.
- If the launch then fails, the idle slots are already saved and cleared (stock would have left them resident), so they come back with a restore.

**Repro** (`logs/repro-r366.log`, client `logs/agpar.py`): Flash-Next IQ3_XXS, 2x R9700, `-sm tensor -c 262144 -cram 10240 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0` (4 slots, unified KV, default admission), no MTP. `agpar.py` builds two agents of ~121K tokens and runs 3 rounds in which both send their next turn at once (each turn adds ~3.6K tokens; 256 generated; the "par" test, both agents fit), then does the same with ~130K-token agents, which cannot both fit (the "turn" test). One server per row of the table below: stock r366 (`e10879b0b3`) and r366 + 0001.

| server | turn test, 3 rounds | per round | "par" test (both fit) |
|---|---|---|---|
| stock r366, `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` | 184.6 s | 13.3 / 85.0 / 86.3 s | 30.9 s |
| + 0001, `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` | 35.4 s | 11.8 / 11.8 / 11.8 s | 31.3 s |
| stock r366, default (lazy idle slots on) | 37.0 s | 13.3 / 11.9 / 11.8 s | 32.0 s |

In rounds 2 and 3 of the stock `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` run, every restore fails as above and each agent processes its ~132K-token prompt again (about 38 s each). With 0001 there are no failed restores. The replies in those rounds are not identical between the first two rows, as expected: stock answers after a fresh prefill and 0001 after a restore plus the new turn. With `-cram 2048`, the ~2.1-3 GiB agent states do not fit in the prompt cache, nothing is restored, and 0001 makes no difference (252 s with and without it, same replies). No GPU faults in any run. In our own build (MTP, production settings, two ~130K agents taking turns), the turn test went from 190.4 to 29.9 s, and our reference outputs (greedy, a 512-token prompt set) are unchanged with 0001 on.

We're happy to re-run this on r368 or any build or settings you'd like, or to test any change you make to the patch.
