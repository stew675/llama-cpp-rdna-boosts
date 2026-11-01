# Cold-session prompt — `wip/kv-residency` (the remaining items)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from a previous
session's context).  **Written 2026-11-01, end of session A18 (the r196 release).**

---

 Read [`wip/kv-residency/HANDOVER-A19.md`](HANDOVER-A19.md) first: it is the self-contained cold start and it
 supersedes `HANDOVER-A18.md` (which stays as the A17-era plan and the Goal A/B/C record).

 **Done, do not redo.** Goals A (the resident window, `LLAMA_KV_RESIDENT_CELLS`), B (the attendance pool,
 `LLAMA_KV_REGION_POOL`, plus the streamed cache taking the sparse decode arm) and C (the MTP draft window,
 `LLAMA_SPEC_DRAFT_WINDOW`, default on) are delivered inside **block 15**. `TODO` #123 (the pool's mirror
 grows on demand) and #125 (plain decode takes the top-k's block radix path) closed in **r196**.  The current
 release is `v16-6050b467c-r196` (tip `bb6599e80`, tree `164070eec…`); r195 was PR #261 (block 03). `TODO`
 #121 is closed, #122 is decided (P1).  Do not re-measure or re-decide any of it; the numbers and the records
 are in the handover.

 **What is left**, in the order the handover suggests:
 * **#129** `-nkvo` with host experts aborts (pre-existing; reproduces on the r191 binary, not our block).
 * **#128** the gfx1201 decode-module MTP cost (blocks #120's `q4_0` re-land and any decode-module change here).
 * **#120** / **#127** adjacent block-10 items (remaining KV write types; the F32-only 2-D copy), if there is time.
 * **#124** the `-sm tensor` window, hardware-blocked here (one card) and designed only.
 * **#116** the parked FP8 campaign.
 Plus two candidates the handover lists for a decision: the qwen4exp *separate* MTP draft is not windowed, and a
 pool-size policy for a 200k+ working set.

 **Rules that cost sessions** (the handover has the full list): `HIP_VISIBLE_DEVICES=0` always; one GPU job at a
 time; `-n 2000` and `-ncmoe` at the full layer count for arena numbers; interleaved arms; `-n 512`+ for a
 speculative arm and check `draft_n`/accepted before reading t/s; static MTP depth ≤ 4; every Flash-Next arm
 carries `-ot per_layer_token_embd.*=CPU -lm none -lzm off`; `ROCBLAS_USE_HIPBLASLT=0` for fresh-process numeric
 comparisons; text from runs **without** `-v` (`scripts/extract-generated.py`), timings from a separate `-v` run;
 always `--single-turn --no-display-prompt` and always `timeout`.  **`GGML_QSA_DERIVED_VIS=0` is not a no-op
 A/B** (use `LLAMA_INDEXER_NOBLOCK=1` for the top-k isolation arm), and **a pooled cache's park declines** by
 design.

 **Gates owed** are in the handover §5: pool text identity + `LLAMA_KV_POOL_VERIFY=1` + `LLAMA_KV_POOL_STATS=<path>`
 + park/ckpt/state on a pooled cache + `test-backend-ops -o FLASH_ATTN_QSA` + `check-gpu-faults.sh`;
 `INDEXER_TOPK` + the top-k census for qwen4exp changes; the standard prefill-logit point before any tag;
 `scripts/validate-set.sh` for any delivery change.

 **The delivery may move under you**: fetch the delivery `main` (https; the ssh key does not work) and check
 `release.json` / the tag list before cutting a release.

 **Report findings to the maintainer before any delivery-affecting commit and before any release or tag**
 (`AGENTS.md`), and never push anything except the delivery repo and the personal fork's `rdna-boosts` branch.
