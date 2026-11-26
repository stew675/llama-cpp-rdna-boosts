# wip/kv-admit-r362: server -- defer a request whose prompt cannot fit next to the running slots (unified KV)

Thank you for the server's deferred-task path (`callback_on_release` -> `pop_deferred_task`), which re-offers a task when a slot is released. It let this patch add only a fit check and leave the scheduling untouched. We're sharing it in case it's useful. Nothing is urgent on our side, and we run it in our own build in the meantime.

**What we saw.** With a unified KV cache, the server starts every request that finds a free slot, whether or not its prompt fits next to what the other slots already hold. When two prompts together exceed the context, both prefill side by side until the cache is full, and then decode fails. On stock r362 (no patches of ours, no MTP, `-c 65536`), two ~40K-token prompts sent at once both return HTTP 500 "Context size has been exceeded" after about 33 s. Decode first retries with `n_batch` halved down to 256, then returns the 500. The server stays healthy. In our own build (`-sm tensor`, MTP, 256K context) the same overlap with two ~150K prompts makes one request fail with HTTP 500 "speculative batch index 2 is not inside the current sub-batch [0, 2)" from the halving retry, and the other completes.

**The patch** (one commit, `tools/server/server-context.cpp`, +48):
- In `process_single_task`, with `kv_unified`, a task is deferred when at least one other slot is processing and `sum over running slots (max(task tokens, tokens held) + reserve) + (prompt tokens + reserve) > n_ctx`. It is re-offered by the existing release path.
- The reserve stands for each request's reply: its `n_predict` (or the server's), capped at 16384 cells (`LLAMA_SERVER_KV_ADMIT_RESERVE`).
- A release wakes only one deferred task. If that task never takes a slot (it is dropped at launch, or cancelled after the wake), the remaining deferred tasks would wait until an unrelated request finishes. So when every slot is idle, `update_slots` wakes one deferred task.
- Parent tasks (`n > 1`) and tasks with no prompt tokens are not checked.
- Kill switch: `LLAMA_SERVER_KV_ADMIT=0` gives the previous behaviour. Without `kv_unified`, nothing changes.

Limits we know of:
- The reserve does not bound generation. Two admitted requests whose replies grow past their reserves can still fill the cache and fail as before.
- At small contexts the reserve also defers requests that would have fitted with short replies. At `-c 65536`, two prompts run side by side only while each is under about 16K tokens; two 20K prompts now take turns.

**Repro** (`logs/repro-r362.log`): Flash-Next IQ3_XXS, 2x R9700, `-sm layer -c 65536` (4 slots, unified KV), no MTP, default fit. `kadrep.py` sends two ~12K-token prompts at once, then two ~40K-token prompts at once, then one short prompt; greedy, 128 tokens each. We're happy to re-run it on any build or settings you'd like, or to test any change you make to the patch.

| arm (r362) | two ~12K at once | two ~40K at once | short request after |
|---|---|---|---|
| stock | both complete, side by side (`f7a7555f`, `7faff34a`) | both HTTP 500 after 33.0 s | completes |
| stock + patch | the same two hashes, side by side | both complete; the second waits until the first finishes (17.6 s, then 35.3 s; `d24cd1dc`, `8a0652fc`) | completes |
| patch, `LLAMA_SERVER_KV_ADMIT=0` | the same two hashes | both HTTP 500 after 33.4 s (= stock) | completes |

Faults 0 in every arm. In our own build at 256K, two ~150K prompts went from one HTTP 500 (134 s) to both completing (43 s and 87 s), with the same outputs on three repeat runs.

One observation from our setup, offered as data and not part of this patch: when we added MTP to this layer-split setup (Unsloth's shared Q8_0 head, `-ngld 99 -otd exps=CPU`), the draft's 2805 MiB compute buffer did not fit on device 1. Every draft step then failed (`llama_process[0] returned -2`), and the load reported that the movable-boundary slab could not be created. We therefore left MTP out of the repro; the 256K numbers above are from our `-sm tensor` build, where MTP runs normally.
