# wip/kv-full-stop-r366: server -- end a reply that no longer fits with a length stop instead of failing the decode (unified KV)

Thank you for taking #370 into r365, and for writing down the limit it leaves in the block-16 row of MANIFESTS.md: a reply that grows past its reserve can still fill the cache. This patch is a follow-up for that case. We're sharing it in case it's useful. Nothing is urgent on our side, and we run it in our own build in the meantime.

**What we saw.** With KV admission on, two requests can both be admitted and still fill a unified cache when their replies grow past their reserves. Decode then retries with `n_batch` halved down to 256 and returns HTTP 500 "Context size has been exceeded" to every request that is processing. On stock r366 (no patches of ours, no MTP, `-c 65536`, `LLAMA_SERVER_KV_ADMIT_RESERVE=512` so that both are admitted), two ~28K-token prompts with 8000-token replies both fail after 139 s. The server stays healthy. In our own build (`-sm tensor`, MTP, 256K context) the same overflow fails both requests with "speculative batch index 2 is not inside the current sub-batch [0, 2)" from the halving retry.

**The patch** (one commit, `tools/server/server-context.cpp`, +95/-4):
- Before each batch is built, `kv_full_check()` compares the generating slots' next tokens (1, plus `common_speculative_n_max` for a slot that speculates) with the free cells: `n_ctx` minus every slot's `prompt.n_tokens()`.
- If they do not fit, idle slots are cleared first through `try_clear_idle_slots()`, which saves them to the prompt cache as the decode retry does. If that is not enough, the generating slot that has generated the most tokens ends with a length stop (it sets `truncated` and `STOP_TYPE_LIMIT`, calls `send_final_response`, then releases the slot). Every token it produced has already been sent. The response reports `stop_type` "limit" and `truncated` true (`finish_reason` "length" on the OpenAI endpoints).
- The step's prompt tokens are bounded by a separate fill limit: the decode tokens plus the free cells, with idle slots' cells counted as room (the decode retry clears them). The batch therefore fits and is never split. `n_batch` itself is unchanged, so the checks that reject a request see the same value as before.
- The check is skipped while a parent or child task of an `n > 1` request is processing, because they share prefix cells through `seq_cp` and the per-slot count would double them.
- Kill switch: `LLAMA_SERVER_KV_FULL_STOP=0` gives the previous behaviour. Without `kv_unified`, nothing changes. While the free cells (idle slots' cells included) cover a full batch, the batch is the same as before.

We end the reply that has generated the most because it is the most likely to have outgrown its reserve. A prompt that is still prefilling keeps the cells it already holds. If you would prefer a different choice (for example the newest request), we're glad to change it.

Limits we know of:
- Cells are counted per slot from `prompt.n_tokens()`. A few paths can count low: a non-splittable embedding or rerank prompt between `keep_first` and `seq_rm`, media chunks (decoded inside the fill loop, outside the fill limit), and audio streams that add more than one row per step. Those fall back to the old retry path.
- A slot that has just started and still holds stale cells is skipped for one step when the cache is exactly full, so a reply can end one step earlier than strictly needed.

**Repro** (`logs/repro-r366.log`, client `logs/kfsrep.py`): Flash-Next IQ3_XXS, 2x R9700, `-sm layer -c 65536` (4 slots, unified KV), no MTP, default fit, `LLAMA_SERVER_KV_ADMIT_RESERVE=512`. With the default reserve (16K at this context) the second request would wait; the small reserve reaches the overflow in a few minutes. We expect the same overflow at the default reserve once replies grow past it. `kfsrep.py` sends two ~12K-token prompts at once (128 tokens each), then two ~28K-token prompts at once with 8000-token replies (`ignore_eos`), then one short prompt; greedy. We're happy to re-run it on any build or settings you'd like, or to test any change you make to the patch.

| arm (r366) | two ~12K at once | two ~28K at once, replies overflow | short request after |
|---|---|---|---|
| stock | both complete (`f7a7555f`, `7faff34a`) | both HTTP 500 after 139 s | completes (1.4 s) |
| stock + patch | the same two hashes | one ends with a length stop after 5,328 tokens (140 s); the other completes all 8,000 tokens (189 s) | completes (4.6 s; the log shows an idle long slot saved to the prompt cache and two older prompt-cache entries dropped) |
| patch, `LLAMA_SERVER_KV_FULL_STOP=0` | the same two hashes | both HTTP 500 after 140 s (the same as stock) | completes (1.4 s) |

No GPU faults in any arm. In our own build (MTP, `-sm tensor`, 256K), two ~125K prompts with 12000-token replies went from both HTTP 500 (196 s) to one ending with a length stop after 6,670 tokens and the other completing all 12,000; our reference outputs (greedy and prompt sets, where the cache never fills) are unchanged.
