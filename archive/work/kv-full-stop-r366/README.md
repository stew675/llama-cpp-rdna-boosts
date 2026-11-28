# `kv-full-stop` — the server's KV full stop (contributor PR #372, `briansp2020`)

**Promoted into block 16 as an amendment at r377, DEFAULT ON** (`LLAMA_SERVER_KV_FULL_STOP=0` is the kill switch).
This directory is the campaign record: the submitter's own README, patch and repro are here unchanged
(`SUBMITTED-README.md`, `submitted-0001-server-KV-full-stop.patch`, `logs/`), and the integration work is named below.
It is not open work and it is not a `wip/` campaign (nothing here is applied by `scripts/apply-all.sh`).

## What was reported

r365's KV admission is a **pre-launch veto**, so it cannot bound a reply: a reply that grows past its reserve still
filled the unified cache, the decode retry could only clear *idle* slots, it halved `n_batch`, the batch split below
the MTP verify width, and the running requests answered **HTTP 500 "Context size has been exceeded."** — or, with a
speculator, "speculative batch index N is not inside the current sub-batch".  `MANIFESTS.md`'s block-16 row carried
that as an accepted limitation; the reporter filed the follow-up that removes it.

## Claim check (r377, r376 tree + the submitted patch, 1/4 scale)

Every claim held, and the shape was reproduced with no contributor code on the delivery build (`-c 8192 -np 4 -kvu`,
one R9700, Qwen3.5-4B-Q8_0, `LLAMA_SERVER_KV_ADMIT_RESERVE=512`, two ~3.4 k-token prompts with 3000-token replies):

| arm | stock r376 | submitted patch |
|---|---|---|
| two admitted requests, replies overflow | **both 500** (n_batch halved 1024→16) | one `200 truncated=true` after 708 tokens, one `200` with all 3000 |
| `LLAMA_SERVER_KV_FULL_STOP=0` | — | **both 500** (the same as stock) |
| `--spec-type draft-mtp --spec-draft-n-max 3` | 500 `speculative batch index 4 is not inside the current sub-batch [0, 4)` | the same two 200s, no spec-index failure |
| a pair whose replies FIT | — | text byte-identical to stock |
| the truncated reply's text | — | a **byte-exact prefix** of the same pair decoded at the same `-c` with room (9395/9395 chars), and the reply that survived was byte-identical to it (9047/9047) |
| streaming / OpenAI | — | SSE deltas end cleanly with `stop_type=limit`, `/v1/completions` reports `finish_reason: "length"` |
| `n = 2` (parent + child) | — | unchanged (the check is skipped: shared prefix cells would be double-counted) |
| `LLAMA_SERVER_KV_ADMIT=0` | both 500 | one truncated (17 tokens) + one complete reply |

The patch applied **strictly** to the r376 tree (the submitter targeted r366; the only change to the touched file
between the two releases is the r373 fair-share line, which the patch does not touch), and no regression appeared on
the block-16 gate family (`gate-kv-admit.sh`, `gate-prefill-fairness.sh`, `gate-prefill-duty.sh`,
`gate-park-idle-slots.sh` all PASS on the amended build).

## What shipped (block 16, r377)

The submitted design is what shipped — `kv_full_check()` at the top of `pre_decode()`, the separate `n_batch_fill`
cap in the prompt pass, the "most generated tokens" victim rule (the maintainer's decision: keep it, it is simple and
it ends the reply most likely to have outgrown its reserve).  The integration added:

* a **startup census line** (`KV full stop: on, a reply that no longer fits the cache ends with a length stop`), which
  every switch in this block prints, so a log states the arm and a gate has a non-vacuity reading;
* comments rewritten for this repo's records (the submitted ones cited an external `FINDINGS 1135`), including the
  `LLAMA_SERVER_KV_ADMIT=0` behaviour;
* the gate: `scripts/gate-kv-admit.sh`'s `full` / `full0` / `fullfit` / `fullstream` arms, and `--self-test` now stands
  **both** kill switches down (admission off alone leaves `full` passing on the pre-r377 build);
* `DIAGNOSTIC.md` §29 + `ENVIRONMENT.md` §29 (the switch, the startup line, the limitations), `MANIFESTS.md` gate 46 and
  the block-16 row, the block-16 note in `patches/README.md`, `CONTRIBUTORS.md`, `INVARIANTS.md` (r377) and the
  WORKLOG entry.

**What the gate deliberately does not check:** a cross-run TEXT comparison of the two co-batched requests.  Measured
(r377): the pair's text is reproducible for a FIXED arrival timing but moves when the timing moves (0.05/0.2/0.6/1.5 s
sweep, and a first-token barrier does not help), so a text oracle here would be flaky — `INVARIANTS.md` r377 records it
and the gate asserts the cut's token count in run instead.

Nothing here is owed: `TODO` carries no open item for this class, and the r365 limitation it closes was recorded in
`MANIFESTS.md`/`DIAGNOSTIC.md` rather than as a `TODO` row.
