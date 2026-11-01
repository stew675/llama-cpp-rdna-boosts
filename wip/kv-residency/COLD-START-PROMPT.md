# Cold-session prompt — `wip/kv-residency` (the remaining items)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from a previous
session's context).  **Written 2026-10-31, end of session A17 (the r194 release).**

---

 Read [`wip/kv-residency/HANDOVER-A18.md`](HANDOVER-A18.md) first: it is the self-contained cold start and it
 supersedes `HANDOVER.md` (which stays as the A-era record of Goals A and B).

 **Done, do not redo.** Goals A (the resident window, `LLAMA_KV_RESIDENT_CELLS`), B (the attendance pool,
 `LLAMA_KV_REGION_POOL`, plus the streamed cache taking the sparse decode arm) and C (the MTP draft window,
 `LLAMA_SPEC_DRAFT_WINDOW`, default on) are all delivered inside **block 15**. Current release
 `v16-6050b467c-r194` (tip `d7e708f08`, tree `977faeaa…`). `TODO` #121 is closed, #122 is decided (P1). Do not
 re-measure or re-decide any of it; the numbers and the records are in the handover.

 **What is left**, in the order the handover suggests:
 * **#123** grow-on-demand pool mirror (the mirror is pinned in full, 3264 MiB at `-c 262144` q8_0).
 * **#125** the radix top-k (0.9 ms per verify step under MTP, 2.2 ms/token plain at 103k, qwen4exp sparse).
 * **#129** `-nkvo` with host experts aborts (pre-existing, reproduces on the r191 binary, not our block).
 * **#120** / **#127** adjacent block-10 items, only if there is time.
 * **#124** the `-sm tensor` window, hardware-blocked here (one card) and designed only.
 Plus two candidates the handover lists for a decision: the qwen4exp *separate* MTP draft is not windowed, and a
 pool-size policy for a 200k+ working set.

 **Rules that cost sessions** (the handover has the full list): `HIP_VISIBLE_DEVICES=0` always; one GPU job at a
 time; `-n 2000` and `-ncmoe` at the full layer count for arena numbers; interleaved arms; `-n 512`+ for a
 speculative arm and check `draft_n`/accepted before reading t/s; static MTP depth ≤ 4; every Flash-Next arm
 carries `-ot per_layer_token_embd.*=CPU -lm none -lzm off`; `ROCBLAS_USE_HIPBLASLT=0` for fresh-process numeric
 comparisons; text from runs **without** `-v` (`scripts/extract-generated.py`), timings from a separate `-v` run;
 always `--single-turn --no-display-prompt` and always `timeout`.

 **Gates owed** are in the handover §6: pool text identity + `LLAMA_KV_POOL_VERIFY=1` + `LLAMA_KV_POOL_STATS=<path>`
 + park/ckpt/state on a pooled cache + `test-backend-ops -o FLASH_ATTN_QSA` + `check-gpu-faults.sh`; the standard
 prefill-logit point before any tag; `scripts/validate-set.sh` for any delivery change.

 **Report findings to the maintainer before any delivery-affecting commit and before any release or tag**
 (`AGENTS.md`), and never push anything except the delivery repo and the personal fork's `rdna-boosts` branch.
