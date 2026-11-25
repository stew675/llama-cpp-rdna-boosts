# wip/kv-pool-drop-r361: attendance pool -- drop the pages a prefill writes, so resident pool copies don't go stale

Thank you for the attendance pool, and especially for `LLAMA_KV_POOL_VERIFY`: comparing every selected pool page with its mirror page is what made this one quick to pin down. We're sharing the finding below in case it's useful. Nothing is urgent on our side: the pool is off by default and we don't run it in production.

**What we saw.** With `--kv-attendance-pool on`, the second of two requests that share a prompt prefix read stale K/V from the pool in our runs. On stock r361 (no patches of ours) VERIFY reports about 226 million mismatching words over two requests, and the answer differs from the resident-KV result. The second prompt holds 90 entries (Entry 0-89); stock's answer says "91 entries (Entry 0 through Entry 90)".

**What we think is happening.** The writer rule in `kv-pool.cu` re-fills a resident page when the batch that wrote it also resolves (`wids`, decode band). A prefill chunk writes the mirror without a resolve. Pool invalidation happens only on `clear`, `park_move`, the K-shift input and `state_read_data`, so after `seq_rm` + re-prefill, any page the previous request left resident keeps the old bytes. With a shared prefix the new K is nearly identical (q8_0 bytes off by one from a different ubatch shape), so the stale copies look plausible. The first VERIFY line is a prompt cell, e.g. `K cell=2060 blk=515 slot=203 word=256 mirror=541344e2 pool=531343e1`.

**The patch** (one commit, 7 files, +128/-2): `ggml_kv_pool_drop(wids, tbl, page)`, the same `GGML_OP_KV_POOL` with op param 1 = 1. Its CUDA kernel sets `tbl[cell / page] = -1` for every cell the batch wrote, so the next resolve claims and fills those blocks from the mirror. qwen4exp builds it per layer whenever the cache has a pool and the batch did not resolve through it: `build_attn_qsa` when `top_k_kv == nullptr`, and the dense `build_attn` arm. It runs in-graph on the compute stream and allocates nothing.
- `inv[slot]` is left naming the dropped block. A later claim of that slot only clears `tbl[prev]` through `atomicCAS(&tbl[prev], cand, -1)`, which fails once the block has moved, so a stale inverse is harmless.
- The writer-rule comment in `kv-pool.cu` is amended to say this.
- Kill switch: `LLAMA_KV_POOL_DROP=0` gives the previous behaviour.
- Without `--kv-attendance-pool` nothing is built, so the default path is unchanged.

**Repro** (`logs/repro-r361.log`). Flash-Next IQ3_XXS fully in VRAM, 2x R9700, `-sm layer -c 65536 --kv-attendance-pool on`, no MTP, `LLAMA_KV_POOL_VERIFY=1`. Requests: one fresh ~18.1k-token greedy request, then one ~4.1k-token request with the same prefix, 32 tokens each. We're happy to re-run the repro on any build you'd like to try.

| arm (r361) | request 1 | request 2 | VERIFY mismatches |
|---|---|---|---|
| stock | `8f4998e050cd` | `4af498db50ab` ("91 entries") | 227,074,287 |
| stock + patch | `8f4998e050cd` | `5c4eb5da3364` ("90 entries") | 0 |
| patch, `LLAMA_KV_POOL_DROP=0` | `8f4998e050cd` | `4af498db50ab` (= stock) | 227,979,668 |
| stock, no pool, `LLAMA_QSA_DENSE_DECODE_UNTIL=0` (resident KV, same sparse arm) | `8f4998e050cd` | `5c4eb5da3364` (= patch) | - |

With the patch the pool's text equals the resident-KV sparse arm byte for byte, which matches the design intent of the pool (it changes where the bytes live, not which bytes are read). Pool stats for the patched arm: hit rate 84.33 %, 0 bad. On a longer suite (our production model and settings, but `-sm layer`) we saw 94.7 % hits, 0 bad, and every hash equal to the resident sparse arm. Faults 0 in every arm. The table's runs also set our test harness's usual `LLAMA_KQ_MASK_DERIVED=0`, `GGML_SCHED_SYNC_GRAPH_INPUTS=1`, `LLAMA_MTP_SPARSE=0` and `LLAMA_MTP_DRAFT_OP_OFFLOAD=0`; a second stock/patch pair without them gives the same hashes (stock 225,752,352 mismatches, patch 0). Both pairs are in the log.

Two side notes from the same work, offered in case they help (not part of this patch):
- Under `-sm tensor` the pool declines, as its comment says (no per-device mirror on the Meta device).
- On our 2-GPU tensor-split box the sparse decode arm is slower than the dense one at every depth we tried. Most of the gap at depth comes from MTP verify steps (n_tokens 4) taking the per-op indexer chain: the fused `INDEXER_SCORE` and the derived indexer cache are n_tokens == 1 only, and each rollback `seq_rm` drops the derived watermark to 0. We're happy to measure anything along those lines if it's useful.
