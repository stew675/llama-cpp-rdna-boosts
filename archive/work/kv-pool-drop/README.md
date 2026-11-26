# kv-pool-drop — the attendance pool served stale K/V after a batch that wrote the mirror without resolving

Promoted into **block 14** at `v18-919794500-r363` (contributor PR #369, `briansp2020`; submitted as
`wip/kv-pool-drop-r361/` in that PR, never merged into `wip/` here — this directory is the record).
The code is `patches/0014-…-block-14-the-streamed-KV-a-fixed-VRAM-re.patch` (`ggml_kv_pool_drop`); the facts are
`INVARIANTS.md` r363, the narrative `WORKLOG.md` 261125.010 and 261125.011, the block reference
`patches/README.md` block 14 (the attendance-pool invariants' item 4), the switch `DIAGNOSTIC.md` §17.1, and the
gate `scripts/gate-kv-residency.sh`'s two-request section (`pstale_ref` / `pstale_res` / `pstale_ctl`).

## What the contributor found

With `--kv-attendance-pool on`, the second of two requests that share a prompt prefix read STALE K/V out of the
pool.  The pool's writer rule re-filled a resident page only when the batch that wrote it also RESOLVED (`wids`,
the decode band), so a **prefill chunk** — and the QSA dense/shortcut arm, which covers the first ~2051 cells of
every request — wrote the mirror and left the resident copy of every block it touched holding the PREVIOUS
request's bytes.  `kv_pool_invalidate()` only covers in-place rewriters (the K-shift, a state restore); the
non-resolving WRITER was the hole.  Their submitted patch is kept here verbatim as the reproduction
(`0001-…patch`, sha256 `82de7d0d1033daff9087571b62bf6ddbb08d0cbf929c1cbd4f7eca7ffe67935d`) together with their run
log (`logs/repro-r361.log`), and it is what shipped, with the integration work below.

## Verified here (r362 → r363)

| arm (r9700 2×`-sm layer`, Flash-Next GSQ-RCO-IQ3_XXS, `LLAMA_KV_POOL_VERIFY=1`) | request 1 | request 2 | mismatching words |
|---|---|---|---|
| r362 | `8f4998e050cd` | `4af498db50ab` ("91 entries" for a 90-entry prompt) | 228 717 907 |
| r363 | `8f4998e050cd` | `5c4eb5da3364` | **0** |
| r363, `LLAMA_KV_POOL_DROP=0` | `8f4998e050cd` | `4af498db50ab` | 230 744 396 |
| r363, no pool + `LLAMA_QSA_DENSE_DECODE_UNTIL=0` (resident SPARSE) | `8f4998e050cd` | `5c4eb5da3364` | – |

gfx1100 (`fingon`, 1 GPU, `-ncmoe 48 -c 32768`): 217 019 052 → **0**, and the same equality with the resident
sparse arm (`5c4eb5da3364`; request 1 differs by arch, `7ef47914cb40`).  gfx1151 (`halo`): INERT — the pool
declines on a shared host/device pool (IGPU), so the drop is never built and the text is identical with the flag on
and off.  A shape below the pool's capacity is vacuous (4.6k/1.4k reads `bad 0` on r362 too).

## Integration work this campaign needed (none of it in the submitted patch)

* **A gate arm, because the gate was blind.**  Every arm of `gate-kv-residency.sh` is ONE `llama-cli` request in
  one process, so none of them ever re-prefilled over a resident page — a 225 M-word stale read shipped green.
  The new section runs three `llama-server` arms over a prefix-sharing request pair and requires the control
  (`LLAMA_KV_POOL_DROP=0`) to read `bad > 0`, or the section is vacuous.
* **The reference needs `LLAMA_QSA_DENSE_DECODE_UNTIL=0`.**  gfx1201's DEFAULT decode arm is dense, so the pool's
  text only equals the resident **sparse** arm; against the default-dense arm it legitimately differs
  (`3a6ad0505fc6`) and comparing there blames the pool for the arch policy.
* **Docs/records.**  The block-14 invariants' item 4 said the only in-place rewriters were the K-shift and a state
  restore — corrected; `DIAGNOSTIC.md` §17.1 and `ENVIRONMENT.md` §17.1 catalogue `LLAMA_KV_POOL_DROP`;
  `INVARIANTS.md` r363 carries the fact; `MANIFESTS.md` gate 19 and `scripts/README.md` describe the new arms.
* **Two corrections to the submitted record** (from the contributor's own follow-up comment and this review): the
  repro model was **not** "fully in VRAM" (the IQ3_XXS is 75.8 GB on 2× 32 GB cards and the fit put experts in the
  host pool; the finding itself is unaffected — the stale copies are in the attention KV pool), and the
  resident-KV reference is the **sparse** arm, not the default one.

## Why it mattered beyond the pool: `TODO` #378

The two-arm server's unexplained corruption under heavy concurrent load (`beta/prefill-decode-arms/`, `TODO` #378)
was observed on a run **with the pool ON**: `LLAMA_KV_REGION_POOL=N` (`--kv-region-pool N`) is the same switch as
`--kv-attendance-pool on` — the numeric size is ignored since r247, the placement is not — and this box's
Flash-Next IQ4_NL `runme` sets `LLAMA_KV_REGION_POOL=32768`.  A pair under concurrent prefill + decode re-prefills
cells another slot left resident constantly, which is exactly this shape, so the pool alone is sufficient to
corrupt an answer.  #378 now carries that as suspect 1 and the closing run (r362+beta vs r363+beta with
`LLAMA_KV_POOL_VERIFY=1`); the artifact applies strictly on the r363 tree (applied tree `c1d50b664`).

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
