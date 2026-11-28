# CONTRIBUTORS — who owns what in this delivery

**This file is the register of external contributions.**  `AGENTS.md` makes it mandatory: a patch derived from a PR
or from a fix suggested in an issue report is **co-owned** by the person who supplied it, and the credit is recorded
here *and* carried by the patch itself.  It exists because this repo is a patch set: a contributor's diff is almost
always adapted to fit the block structure, so the commit that ships is usually written (and committed) by the
maintainer — which used to leave their name alone on work that was not theirs alone.

## How the credit actually travels (the mechanism)

Three vehicles, strongest first.  All three survive the re-cut, so a consumer sees them:

| form | where it lives | what a consumer / GitHub sees |
|---|---|---|
| **author** | the block commit's `From:` line, which `git format-patch` writes from the commit's AUTHOR | `git log --format=%an` names them; GitHub shows their avatar as the commit's author on the fork branch and in every `git am`-ed clone.  `scripts/apply-all.sh` pins the COMMITTER to the series identity and stamps the committer date from the author date, so the author is the one that matters — and the tip stays reproducible. |
| **`Co-authored-by:`** | the block commit's message body, i.e. the `.patch` body | GitHub renders the commit with **both** authors; `git log` and `git shortlog -sne` count them.  `git am` preserves it verbatim, so it is in every consumer's history, not just in this repo. |
| **`Reported-by:` / `Suggested-by:`** | the same place | text only (GitHub does not link it), but greppable, and it is the honest form when the contribution was a diagnosis or a proposed fix rather than code. |

`scripts/make-patches.sh` exports the block commits' messages verbatim, so **a trailer added to a block commit is in
`patches/NNNN-*.patch` for ever after** (see "Debt" below for the rows that predate this policy).

## Code contributions that landed

`credit` = the form the shipped patch carries **today** (`author`, `co-author`, `reported`, `suggested`,
`none`), and `as "<identity>"` where the trailer spells the name differently from the handle: `author`, `co-author`, or **`none`** (a debt row — the
policy is to add the trailer at the next message re-cut that touches the block; the mechanism is a `git commit
--amend` of that block's message, which changes the patch's sha256 and the recorded `.tip` but **not its tree**).

| contributor | contribution | landed in | evidence | credit |
|---|---|---|---|---|
| [@overdoingism](https://github.com/overdoingism) | TQ1_0 / TQ2_0 (ternary) weights on the GPU — `MUL_MAT`, `GET_ROWS`, the dequantise rows | block `0017` | PR #358 | **author** ✅ |
| [@overdoingism](https://github.com/overdoingism) | the server's tool-text protection (a tool result's special-token strings stay TEXT) | block `0012` | PR #360 | **co-author** ✅ |
| [@overdoingism](https://github.com/overdoingism) | an FFN's GLU may take the memory of its gate (in-place fold) | block `0000` | PR #334, item 1 of 3 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | the hybrid HIP all-reduce's original contribution | block `0001` | existing trailer | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | BF16 dense weights take MMB while hipBLASLt is off | block `0006` | PR #340 | **co-author** ✅ *as "brian"* |
| [@briansp2020](https://github.com/briansp2020) | the qwen4exp hyper-connection chain's original contribution | block `0011` | existing trailer | **co-author** ✅ *as "brian"* |
| [@briansp2020](https://github.com/briansp2020) | the opt-in `-of tp-ffn-even` even FFN split, and the `--optional-features` tag layer | block `0009` | PR #342 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | sequence parallelism — the qwen4exp hyper-connection chain, token-split | block `0012` | PR #315 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | the server's KV admission: the max-min fair share, the reply reserve, the deferral | block `0016` | PR #370 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | the attendance pool's stale-KV drop (`ggml_kv_pool_drop`) | block `0014` | PR #369 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | the max-min prompt share divides the PASS's budget (`budget_before`), so two prompting slots get 1024/1024 instead of 1365/682; plus the analysis, the repro and the `-lv 5` evidence | block `0016` (r373) | PR #376 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | Windows host memory — the WDDM free figure and the working-set release of an uploaded mapping | blocks `0009`/`0011`/`0012`/`0014` | PR #269 | **co-author** ✅ |
| [@briansp2020](https://github.com/briansp2020) | the scheduler/slab/offload series (the async input ring, the wide-band redirect, the sizing rules, … — each landing is listed per block in `patches/README.md`) | blocks `0002`/`0005`/`0010` | PRs #169 #177 #182 #194 #253 #256 #266 #267 #335 #337, and others | **co-author** ✅ |
| [@burnoutIA](https://github.com/burnoutIA) | the VERIFY band's WMMA `iu8` matvec family (opt-in `--wmma-verify`) | block `0007` | PR #317 | **co-author** ✅ |
| [@burnoutIA](https://github.com/burnoutIA) | the fused MoE / k-quant decode contributions that carried this trailer | blocks `0008`/`0010` | existing trailers | **co-author** ✅ |
| [@slavap](https://github.com/slavap) | the cross-backend input copy ordered on a PER-COPY event (one event per copy, so a queued wait is never re-bound) | block `0012` | PR #268 | **co-author** ✅ |
| [@slavap](https://github.com/slavap) | two-lane prefill — `LLAMA_PREFILL_LANES` | `optional/two-lane-prefill` (`0100`) | PRs #268 #279 #284 #296 #301 | **co-author** ✅ |
| [@mjohanning99](https://github.com/mjohanning99) | the hybrid all-reduce's original contribution | block `0001` | existing trailer | **co-author** ✅ *as "Marvin Johanning"* |
| [@mjohanning99](https://github.com/mjohanning99) | the dense Q5_K VDR-4 dot product, and the opt-in RDNA3 Q6_K MMQ band | block `0007` | issue #374 | reported |
| [@overdoingism](https://github.com/overdoingism) | the shared-expert down epilogue's rounding: `__fmul_rn(...) + c` is contracted into an fma on HIP, so the epilogue rounded once where the chain it replaces rounds three times (diagnosis from the source plus the emitted ISA, with the two ways out spelled out) | block `0008` | issue #377 | reported |
| [@mjohanning99](https://github.com/mjohanning99) | the S2 exact top-k fast path now reproduces a penalty instead of standing down | block `0012` | issue #366 | reported |
| [@mjohanning99](https://github.com/mjohanning99) | mixed K/V cache types (K `q8_0` with V `q4_0`/`q5_0`/`q5_1` on 256-wide attention) | `optional/mixed-kv` (`0101`) | PR #312 | **co-author** ✅ |
| [@mjohanning99](https://github.com/mjohanning99) | head-256 prefill attention under a tensor split must not pad a GQA lane: the split-aware `ncols2` rule picks 8 for a group of 6, so a quarter of the prefill KQ/VKQ work is padding. The report supplied the switch, the RDNA3 half's diagnosis (spills, the `#31746` pointer, the two-arm mixed-pair reasoning) and every number in the table, including the perplexity/KLD controls | block `0005` | issue #381 | **co-author** ✅ *as "Marvin Johanning"* |
| [@ravel7524](https://github.com/ravel7524) | RDNA3 (gfx1100) head-256 prefill: Q streamed from shared memory, one K and one V batch per iteration and a sched barrier per WMMA — register spills 720 VGPRs / 1312 B scratch to 30 / 128 B, the attention call 10.16 -> 6.98 ms | block `0005` | upstream PR [ggml-org/llama.cpp#31746](https://github.com/ggml-org/llama.cpp/pull/31746) | **co-author** ✅ |
| [@jstamagal](https://github.com/jstamagal) | three RDNA2 (`gfx1030`) items: the FA tile-cap removal, the one-warp narrow `RMS_NORM`, the `H * n_seqs` GDN chunked gate and the rows-abreast MoE K walk (`RDNA2`-gated) | blocks `0005`/`0008`/`0010` | PR #338 | **co-author** ✅ |
| [@RikshaDriver](https://github.com/RikshaDriver) | the three kq-mask fills that lacked their sibling's buffer guard | block `0005` | PR #222 / issue #221 | **co-author** ✅ |
| [@d32768](https://github.com/d32768) | the masked-V HIP fixes and the FA prefill staging arena bound | block `0005` | PR #181, PR #198 | **co-author** ✅ |
| [@Nyshone](https://github.com/Nyshone) | the Q8_1 MUL fusion's reader count ignored VIEWS, so a LoRA's delta was computed from a buffer nobody wrote — reported with the guard analysis and a one-hunk patch (the shipped form is the shared helper plus the census line) | block `0010` | issue #331 | suggested |
| Simon Teixidor | the MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`) | block `0013` | existing trailer | **co-author** ✅ |
| Laurent Zuijdwijk | the MTP draft vocabulary subset | block `0013` | existing trailer | **co-author** ✅ |
| Martino B | the MTP draft vocabulary subset | block `0013` | existing trailer | **co-author** ✅ |

## Reports and diagnoses (their code is not in the tree, but the fix is theirs)

These are credited with `Reported-by:`/`Suggested-by:` at the next re-cut and are named in the block's
`patches/README.md` note **today**.  A report that produced a delivery change is an ownership claim: it is the
diagnosis, and often the patch was written against it.

| contributor | what they found / asked for | changed | evidence | credit |
|---|---|---|---|---|
| [@DanoPTT](https://github.com/DanoPTT) | the Flash-Next prefill deficit and the streamed-slot comparison that started `wip/flash-next-prefill`; issue #368's reserve/stand-down shape (an r369 block-10 amendment); #237, #278, #291, #363, #365, #138, #45 | block `0010` (r362/r365/r369 amendments), the whole r369 transport | issues #45 #138 #237 #278 #291 #363 #365 #368 | reported |
| [@hermeticvm](https://github.com/hermeticvm) | the cache-off host-tier GPU page fault (issue #272) — reproduced, then fixed | block `0002` (`spec_draft_align_cache()` realigns the drafter's KV with the batch it ingests; `TODO` #375 keeps the open half) | issue #272 | reported |
| [@jstamagal](https://github.com/jstamagal) | the BF16 `RMS_NORM` + `MUL` (+ `ADD`) abort (PR #339) — it reproduced a real r324-r327 defect, already gone at the r328 re-base | block `0010` (verify arm only) | PR #339 | **co-author** ✅ |
| [@plchldr](https://github.com/plchldr) | the `-sm layer` host-expert garbage-prefill report that produced the event-ordering fix | block `0009` | issue #33 | reported |
| [@tazmaniax](https://github.com/tazmaniax) | the gfx1100 `-lme`/GTT report behind `TODO` #347 | block `0015` (diagnosis) | issue #347 | reported |
| [@xxyygt](https://github.com/xxyygt) | the MoE device-remap findings folded into the cache's decode path | block `0009` | issue #219 | reported |
| [@a-n-t-0](https://github.com/a-n-t-0) | the original issue-#30 prefill reproduction | block `0005` | issue #30 | reported |
| [@autopeasant](https://github.com/autopeasant) | two offline ctests still encoded upstream's mixed-K/V rule while the delivery refuses a mixed pair on every model (`test-kv-cache`'s diagnostic, `test-fp8-kv`'s `K=f8_e4m3`/`V=f16` arm) -- reported with the exact command and the upstream control, and made the same reporter's tree green.  The sweep over the rest of the suite then found block 14's windowless device lookup and block 12's zero-length state copy, and block 14's long-first abbreviation pair | block `0010` (r374 amendment) | issue #382 | reported |

## Debt, and how it is paid

* **Trailer debt: CLEARED at r370 (2026-11-27).**  The 25 contributions that were shipped without a trailer were
  paid in one **message-only re-cut**: every block's message gained the trailer the row records (`Co-authored-by:`
  for code, `Reported-by:`/`Suggested-by:` for a report), so **all 19 block trees are byte-identical to r369**
  (`diff` of the two chains' tree lists is empty) while the patch sha256s and `release.json.tip` moved.  The
  register now reads **1 author, 24 co-author, 7 reported, 1 suggested, 0 none**, and `scripts/check-credit.sh
  --strict` is wired into `validate-set.sh`, so a future contribution cannot land uncredited.
* **Unattributed**: the docs name these contributions without the person's handle — fill them in when it is known
  rather than leaving the credit with the PR number alone:
  * PR #298 — the copy-engine `q8ce` all-reduce chain concept (blocks `0001`).
  * PR #344 — the DeepSeek-V4 hyper-connection band kernels (block `0011`).

## Adding a row (do this in the same change that lands the work)

1. Name the person in the block's `patches/README.md` note with their PR/issue — that is the note a consumer reads.
2. Add a row here with `landed in`, `evidence` and the `credit` form you actually put in the patch (not the one you
   intend to).
3. Put the credit in the patch: `git commit --amend` the block commit with
   `Co-authored-by: Name <email>` (or `Reported-by:`/`Suggested-by:`), using the identity from their commits when
   known.  For a contribution that IS whole-cloth a block, make them the block's **author** instead
   (`git commit --amend --author="…"`), as block `0017` does.  Never strip an existing trailer, and never rewrite
   someone's authorship at a re-cut.
4. `scripts/check-credit.sh` enforces the mechanical half: every handle named in `patches/README.md` has a row here,
   every row's claimed `credit` form is really in the named patch, and no row names a block that does not exist.
