# WORKLOG — the dated delivery record

**Today's entries are here, newest first.  Every earlier day is in [`archive/worklog/`](archive/worklog/), one
file per day (`WORKLOG-YYYYMMDD.md`, newest first inside the file), and the table at the end of this file indexes
them.**  That keeps the current day in front of you without a second place to look for it, and the record still
has exactly one entry point.

**Recording work.**  Add the entry at the top of this file with the id `YYMMDD.VVV`: `YYMMDD` is the real local
calendar date, and `VVV` is a per-date sequence that increases in file order, so the topmost entry of a day
carries the highest number.  Never invent or extrapolate a future date, and never reuse an id.

**At the first entry of a new day**, move the previous day's entries into
`archive/worklog/WORKLOG-<that day>.md` (file header in the same form as its neighbours) and add its row to the
index below.  That is the whole roll-over: one move, one row.

A reference written `WORKLOG.md` 261124.012 means entry `261124.012` — it is in this file if it is today's,
otherwise look its day up in the index.

## 261128.009 (r378, release cut) -- tagged, main and the fork pushed, the assets published, the images queued on the worker

**The cut.**  `v18-919794500-r378` (annotated tag, pushed) = fork re-cut tip
`d4dd34a93ba61748e642a3d76ddaf6c62e4f0482`, tree `976fd14dcd43b6fc8d90176cfb1c419ec72968fc`, 19 default patches
(18 official + temporary `0300`), 2 optional artifacts re-cut against the new tree (bodies unchanged),
optional tree `b23dc540337f3f8539c45140aefdeefaf3bd0b03` -- all recorded in `release.json`.  The taxonomy is
one message-only re-cut into the release: the first export dropped the `Co-authored-by: DanoPTT` trailer
(the assembler split the inserted paragraph at its own trailer line), which `check-credit.sh` caught
immediately -- so the block 07 and 10 commits were amended again, the whole set re-exported and re-applied,
the tree **unchanged** (`976fd14dcd`, which is why no rebuild was owed) and only the patch sha256s and `.tip`
moved.  Determinism holds: two independent `apply-all.sh` runs from the shipped patches produce tip
`d4dd34a93` and tree `976fd14dcd`.

**The ladder** (in `WORKLOG.md` 261128.008): `validate-set.sh` green (strict 19/19 on a fresh tarball, both
optional artifacts alone and cumulative, credit 45 handles / 40 rows / 0 debt, file-set reconciliation 582
files 582 / 0 loss / 0 gone / 0 shrunk / **+2 defines** -- exactly `VDR_Q6_K_Q8_1_MMVQ_DENSE4` and
`GGML_CUDA_Q6K_DENSE_VDR4`); `fork-refresh.sh --build --gates` PASS (re-cut asserted tree **and** tip, `all`
build, `check-shlib-symbols.sh` 16/16, coherence, `gate-prefill-logits.sh` **0.000000 / 100.000 %**);
`test-backend-ops test -o MUL_MAT` **2753/2753** on the default build; `check-gen-docs.sh` ok;
`gate-optional-build.sh` (gate 47) PASS in 248 s on tree `b23dc5403`; coherence vs the r377 build identical on
four arms (4B Q8_0 1 GPU and 3-GPU tensor `7e6229b260f6`, 27B UD-Q6_K_XL `da2e2d192e21`, 35B-A3B Q4_K_M
`461ca8cd0e88`, 27B `draft-mtp` `f2f9f9a596b7`).  The flag-on arm was measured in the scratch tree, whose tree
is **the same `976fd14dcd`**: `test-backend-ops -o MUL_MAT -p type_a=q6_K` 71/71 with the same max error as
VDR 2, wikitext PPL 6.9346 both arms chunk for chunk, MTP acceptance 0.49755 -> 0.50825, and the kernel /
batched tables in 261128.008.  Gate **49 (`check-offline-tests.sh`) was NOT triggered**: it applies to a runtime
policy that replaces an upstream one, or an option added/reordered in `common/arg.cpp`, and this change is a
build-time preprocessor guard with no argument surface.

**CI and assets.**  Tier-1 `Validate delivery set` green on `main` and on the tag, `Release patch set` green;
the GitHub Release carries its four assets (`SHA256SUMS`, `patches.tar.gz`, `rdna-boosts-all.patch`,
`release.json`).  The personal fork's `rdna-boosts` branch carries `d4dd34a93`:
`gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` reads it.

**The images.**  OWED: the `radiance` cron watcher picks the new tag up on its next poll and runs that tag's own
`scripts/build-containers.sh --push --prune` for both ROCm lines.  A push is not proof (issue #364), so this
entry is closed by `scripts/build-containers.sh --verify-only` reading every tag resolving to this release.  The
wait here was stopped on the maintainer's instruction (the worker needs roughly another hour for the two lines);
`--verify-only` read **8 missing / 15 of 23 present**, and all eight missing are this release's tags -- that is
the open obligation, not a passing check.

## 261128.008 (r378) -- contributor PR #386 promoted into blocks 07 + 10: the dense Q6_K VDR-4 dot product, the third RDNA4 build-time opt-in

**Contribution.**  `@DanoPTT`'s **PR #386** is the Q6_K counterpart of r368's dense Q5_K VDR-4 kernel (issue #374):
one 32-element call, each 16-value sub-block summed in ONE integer dp4a chain before its single scale multiply, the
-32 offset folded in through `vec_dot_q6_K_q8_1_impl_mmvq_vdr2`'s one-byte `__builtin_amdgcn_perm` table.  It was
submitted as a `wip/` campaign with its own README and one `git am` file cut on r368; the patch applied **strictly
(`git am -3`) to the r377 tree**, so no rebase was owed.  The maintainer's decision this session: **promote it as an
opt-in** (the third member of the family next to `-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4` / `-DGGML_HIP_RDNA4_Q5K_DENSE_VDR4`),
**not** default-on -- the kernel win is large but the end-to-end win is 1-2 %, and a default-on switch would move
every Q6_K consumer's greedy text for that.  Promoted in the same session, so it never had a `wip/` row and leaves no
stub: the record is `archive/work/rdna4-q6k-dense-vdr4/` with the submitted patch and README kept verbatim.

**Every claim checked, on this box (gfx1201, Linux, ROCm 10.2), not the reporter's Windows one.**  The independent
kernel rerun reproduces their table and is better at the wide end -- `m 4096 k 14336` µs/call, A/B/A/B, VDR 2 vs
VDR 4: **41.99 -> 33.86 at n=1 (-19.4 %, they said -20), 38.9 -> 33.9 at n=2 (-13.1 %, -13), 47.0 -> 33.8 at n=3
(-28.0 %), 54.9 -> 39.7 at n=4 (-27.7 %), 61.4 -> 42.2 at n=5 (-31.3 %, -24), 84.4 -> 53.9 at n=8 (-36.2 %, -32)**;
the k=5120 decode shapes -10.8..-15.2 %; the lm_head shape and every n >= 512 inert.  Controls match too: the 27B's
`tg128` is **flat** (23.10/23.15 -> 23.09/23.14 t/s) because this model decodes at ~584 GB/s, i.e. at the card's
DRAM roof, which is exactly what their "<= 1.1 % end to end" says.  What the reply did not measure and this session
did: **`llama-batched-bench -npp 512 -ntg 128` reads +0.2 / +0.3 / +6.9 % at B = 1/4/8** (S_TG 23.01/80.89/126.39 ->
23.06/81.13/135.08), and the MTP `draft-mtp` arm **+1.6 %** (43.0/43.0 -> 43.7/43.7 t/s, A B B A).  Correctness:
`test-backend-ops -o MUL_MAT -p type_a=q6_K` **71/71 with the same max error as the VDR-2 arm** (0.0005), wikitext
PPL **6.9346 in both arms, chunk for chunk identical**, MTP draft acceptance **0.49755 -> 0.50825**.

**The 2026-09-12 revert does not apply to this kernel.**  The dense VDR=4 set was reverted in September because the
32-element variants lost on the MTP verify widths (issue #30, `WORKLOG-20260912.md`).  That is why the band was the
question here, and this kernel wins it outright (-13..-36 % at widths 2..8) -- it is a different formulation, and the
merged integer chain is exact by construction (`|dot| <= 16*32*127` times `|scale| <= 128 < 2^24`).

**The promotion split is r368's own**: block **07** owns the kernel, `VDR_Q6_K_Q8_1_MMVQ_DENSE4` and the
`GGML_CUDA_Q6K_DENSE_VDR4` guard (`vecdotq.cuh`, +58); block **10** owns the host VDR (`get_vdr_mmvq_dense`, keyed on
the runtime `table_id` -- the host side must not read a device arch macro) and the dispatch case (`mmvq.cu` +6,
`mmvq-vecdot.cuh` +4/-1).  `GGML_HIP_Q6K_DENSE_VDR1` still wins over the new flag and the MoE path keeps
`vec_dot_q6_K_q8_1_vdr2` / `VDR_Q6_K_Q8_1_MMVQ_MOE` 2, so **the default path is byte-inert** -- verified rather than
assumed: the same generated text as the r377 build on 4B Q8_0 (1 GPU and 3-GPU `-sm tensor`, `7e6229b260f6`), 27B
UD-Q6_K_XL (`da2e2d192e21`), 35B-A3B Q4_K_M (`461ca8cd0e88`) and the 27B `draft-mtp` arm (`f2f9f9a596b7`), plus
`gate-prefill-logits.sh` **0.000000 / 100.000 %** and `llama-batched-bench` at parity on 4B.

**Gates.**  `validate-set.sh` green (strict 19/19 apply on a fresh tarball, tree `976fd14dcd`, both optional artifacts
and the cumulative tree re-cut and re-recorded, Windows path/portability lints, the system-header check, and
`check-rung-file-sets.py` **0 loss / 0 gone / 0 shrunk, +2 defines** -- exactly
`VDR_Q6_K_Q8_1_MMVQ_DENSE4` and `GGML_CUDA_Q6K_DENSE_VDR4`); `check-credit.sh` green with the new row (45 handles,
40 rows, 0 debt); `test-backend-ops test -o MUL_MAT` **2753/2753** on the default build; `check-shlib-symbols.sh`
**16/16**; `check-gen-docs.sh` ok; `gate-optional-build.sh` (gate 47) PASS on the re-cut optional tree;
`check-rung-file-sets.py`'s per-rung pass and `validate-set.sh` both run on the final export.  The determinism contract
holds: two independent `apply-all.sh` runs from the shipped patches produce tip `d4dd34a93` and tree `976fd14dcd`.

**Record.**  `archive/work/rdna4-q6k-dense-vdr4/` (the submitted patch and README verbatim, the maintainer's
kernel/PPL/MTP/batched tables, the reproduction command, and what is left open); `patches/README.md` blocks 07 and 10;
`ENVIRONMENT.md` §14 (the flag, its default and its measurements); `CONTRIBUTORS.md` (one code row, `co-author`);
`INVARIANTS.md` r378 (the verify-band fact, the DRAM-roof caveat, the inertness check).

## 261128.007 (r377, release cut) -- tagged, main and the fork pushed, PR #372 merged, the images queued on the worker

**The cut.**  `v18-919794500-r377` (annotated tag, pushed) = fork re-cut tip
`6651cef37fe2c9470bb47ddc6a7c551c7d71667a`, tree `3706b801e235bbb371bfd0a02bb4239bb4a89778` (recorded in
`release.json`), 19 default patches (18 official + temporary `0300`), 2 optional artifacts, optional tree
`45128b1b1345209e8272272f36caa2ad5ac0a947`.  The tree was derived twice independently (the amended chain and
`scripts/validate-set.sh`'s throwaway tarball) and `scripts/fork-refresh.sh --build --gates` re-cut the fork from
`patches/` and asserted tree **and** tip; `gh api repos/stew675/llama.cpp/branches/rdna-boosts` reads `6651cef37`.

**The re-cut exports from the CANONICAL chain.**  The first export came from the amended working chain, which moved
the `From <sha>` line of every block (a rebuilt fork produces its own commit ids).  Re-exporting from a fresh
`apply-all.sh` chain at `release.json.base` reproduced blocks `00`-`15` with the **exact ids r376 shipped**
(`a46ca023c` ... `0198bcd15`) and moved only block 16 (`5ee370656`), block 17 and `0300` -- so the delivered diff is
that one block plus the optional artifacts' parent lines, and the determinism contract in `AGENTS.md` still holds.

**The PR.**  Merged (`--rebase`, attribution kept, `40ec4dbc`) and the merged `wip/kv-full-stop-r366/` copy removed in
the next commit: the record is `archive/work/kv-full-stop-r366/` (their README, patch and repro kept unchanged) and a
promoted campaign leaves no `wip/` stub.

**The ladder.**  In `WORKLOG.md` 261128.006 (the promotion entry): `validate-set.sh` green, `fork-refresh.sh --build
--gates` PASS, `gate-kv-admit.sh` PASS on every arm plus `--self-test` 23 failing checks, `gate-prefill-duty.sh` and
`gate-park-idle-slots.sh` PASS, `gate-optional-build.sh` PASS (248 s), `check-shlib-symbols.sh` 16/16,
`check-gen-docs.sh` ok, coherence identical to the r376 build on 4B Q8_0 (1 GPU) and 27B UD-IQ3_S (3-GPU tensor).
`gate-prefill-fairness.sh` PASS in 2 of 3 runs, with the third failing only `share split` -- reproduced with the same
numbers (10/13 steps, worst 129.00) on the **r376** build, and unreachable from this change (that arm runs
`kv_unified=false`), so it is a pre-existing flake in that check and it is flagged for the maintainer in 261128.006.

**Images.**  The `radiance` watcher (cron, every 10 min) picks the new tag up on its next poll and runs that tag's
`scripts/build-containers.sh --push --prune`.  A release is only finished when `scripts/build-containers.sh
--verify-only` says every tag RESOLVES (issue #364) -- see the entry above the index for that verdict.

## 261128.006 (r377) -- contributor PR #372 promoted into block 16: the KV full stop (a reply that outgrows its reserve ends with a length stop, DEFAULT ON)

**Contribution.**  `@briansp2020`'s **PR #372** closes the limitation r365 wrote down for the KV admission: the
reserve is an admission heuristic, not a bound, so a reply that grew past it still filled the unified cache, the
decode retry could only clear *idle* slots, it halved `n_batch`, the batch split below the MTP verify width, and the
running requests answered HTTP 500 "Context size has been exceeded." (with a speculator, "speculative batch index N
is not inside the current sub-batch").  Their submission was a `wip/` campaign with a one-commit patch; the
maintainer's decisions this session: **keep their victim rule** (the generating slot that has generated the most),
**no `upstream/` candidate**, **fold into block 16 as a default-ON amendment**.

**Every claim checked out**, at 1/4 scale on the r376 tree + their patch (`-c 8192 -np 4 -kvu`, one R9700,
Qwen3.5-4B-Q8_0, `LLAMA_SERVER_KV_ADMIT_RESERVE=512`, two ~3.4 k-token prompts with 3000-token replies): stock
**500 / 500** in 9.8 s (n_batch halved 1024->16) -> one `200 truncated=true` after 708 tokens and one `200` with all
3000; `LLAMA_SERVER_KV_FULL_STOP=0` reads the two 500s again; `--spec-type draft-mtp --spec-draft-n-max 3` reads
`speculative batch index 4 is not inside the current sub-batch [0, 4)` on stock and the same two 200s with the
amendment; a pair whose replies fit is **byte-identical to stock**; the truncated reply is a **byte-exact prefix**
(9395/9395 chars) of the same pair decoded at the same `-c` with room, and the reply that survived is
byte-identical to that run (9047/9047); streaming ends cleanly (`stop_type=limit`, every delta delivered before the
terminal chunk) and `/v1/completions` reports `finish_reason: "length"`; `n=2` is unchanged.  The patch applied
**strictly** to r376.

**What changed in the integration (no code redesign, four additions).**  (a) a **startup census line**
(`KV full stop: on, a reply that no longer fits the cache ends with a length stop`), which every switch in this block
prints and which the gate's non-vacuity reads; (b) their comments cited an external `FINDINGS 1135` and omitted the
`LLAMA_SERVER_KV_ADMIT=0` behaviour, both rewritten for this repo's records; (c) **the gate**: four arms in
`scripts/gate-kv-admit.sh` (`full`: one reply ends truncated while the other completes all 5000 tokens, the cut logs
once with the count the client received, the decode retry never runs; `full0`: the kill switch answers 500 twice;
`fullfit`: the same pair whose replies fit is untouched; `fullstream`: the SSE path ends cleanly) and `--self-test`
now stands **both** kill switches down (`KV_ADMIT=0` alone leaves `full` passing on the pre-r377 build -- measured:
23 failing checks, so the gate can reject it); (d) the docs (`DIAGNOSTIC.md` §29, `ENVIRONMENT.md` §29,
`MANIFESTS.md` gate 46 + the block-16 row, the block-16 note in `patches/README.md`, `CONTRIBUTORS.md`,
`INVARIANTS.md` r377) and the campaign record in `archive/work/kv-full-stop-r366/` (their README, patch and repro kept;
no `wip/` stub).

**The gate deliberately has no text oracle for these arms, and that is a measured finding.**  Two co-batched requests
on this model are not reproducible across server runs: over a 0.05/0.2/0.6/1.5 s stagger sweep with `id_slot` pinned,
each FIXED timing repeats its hashes exactly while the pair moves when the timing moves, and a first-token barrier does
not help.  So a cross-run comparison would be flaky and a lone-run one is forbidden (`GREEDY-PURITY.md` §38/§63); the
arms assert the behaviour, the census and the cut's token count **in run**, and the prefix evidence above is a hand
measurement at one `-c` (`INVARIANTS.md` r377).

**Release ladder (r377, the amended set at the canonical tip `6651cef37`, tree `3706b801e`; scratch path, gfx1201 box).**

| gate | result |
|---|---|
| `validate-set.sh` | **GREEN** -- strict apply from the base tarball, both optional artifacts alone and cumulative (trees `7c27bcf25` / `a7a56868c` / `45128b1b1`), Windows lints, credit (43 handles / 39 rows / 0 debt), file-set reconciliation (582 files, 0 loss, 1 new env), system headers |
| whole-set `all` build at the tip | OK (ccache-warm, `all` target) |
| `check-shlib-symbols.sh` | 16/16 libraries resolve every symbol |
| `check-gen-docs.sh` | in sync (no `common/arg.cpp` change) |
| coherence vs the **r376** build | 4B Q8_0 1 GPU `1c5d32ac537d` and 27B UD-IQ3_S 3-GPU `-sm tensor` `fe3a5ad0f3ef` -- **identical** on both builds |
| `gate-kv-admit.sh` (gate 46) | **PASS** -- all arms, including the four new ones (cut at 2962 tokens, equal to the client's `predicted_n`; the 5000-token reply completes; no decode retry; `fullstream` 2963 == 2963) and `--self-test` PASS (23 failing checks on the two kill switches) |
| `gate-prefill-fairness.sh --quick` | **PASS** 2 of 3 runs; the third failed ONE check, `share split` (10/13 co-batched steps, worst 129.00), and the **r376 control build fails it with the same numbers**, so it is a pre-existing flake in that check, not this change -- the arm runs `kv_unified=false`, where the full stop cannot engage (0 census lines) |
| `gate-prefill-duty.sh --quick` | **PASS** (burst 16 -> 137 tokens, prefill stretch 2.12x, single-request text identical) |
| `gate-park-idle-slots.sh` | **PASS** (park == lazy-off on all four arms, the indexer cache moves with attention) |
| `gate-optional-build.sh` (gate 47) | `OPTIONAL_BUILD=PASS` in 248 s, cumulative tree `45128b1b1` (both artifacts re-cut on the new default tree; their patch BODIES are unchanged from r376 -- only the `From` line and the parent moved, so `gate-mixed-kv.sh` was not re-run) |
| `check-offline-tests.sh` (gate 49) | not owed: no `common/arg.cpp` change and no upstream runtime policy replaced |
| `gate-prefill-logits.sh` | not owed: `tools/server` only, no prefill kernel or approximate path |
| GPU faults | clean in every gate (`check-gpu-faults.sh` in each) |

**Fixed during the session, not deferred:** the gate's first `fullfit`/`fullpure` designs compared text across server
runs and flaked (measured, above) -- replaced by in-run checks; the reference arm for the prefix measurement had to
stop 64 tokens (not 8) below the cut, because within 8 the reference itself takes the stock retry and its text moves;
`check-credit.sh` gained a register row for the new contribution, and the release tag example in `README.md` moved to
r377.

**Open finding for the maintainer (not fixed here, not filed as a TODO):** `gate-prefill-fairness.sh`'s `share split`
check is intermittent at the same rate on r376 and r377 (10/13 steps, `worst 129.00`, identical on both).  It is a
release gate, so its threshold/semantics are the maintainer's call, and changing it in a release that is not about it
would be the wrong place; the numbers above are the record.

## 261128.005 (r376, release cut) -- tagged, main and the fork pushed, the assets published, the images queued on the worker

**The cut.**  `v18-919794500-r376` (annotated tag, pushed) = fork re-cut tip
`1cb8a186628ccb75e053d6aa09e8edc87307e8ca`, tree `223fbb80b1f2340b9bac698f5da8820d29d8f8f8` (recorded in
`release.json`), 19 default patches (18 official + temporary `0300`), 2 optional artifacts, optional tree
`9ee987063680ec8c8c15f806d6519e64e6f26725`.  The tree was derived three times independently -- the working chain,
`scripts/fork-refresh.sh` and the `validate-set.sh` throwaway tarball -- and all three agree; the fork branch
carries the recorded tip (`gh api repos/stew675/llama.cpp/branches/rdna-boosts` -> `1cb8a1866`).

**Gate ladder run for this cut** (gfx1100, r376 tree; gfx1201 on `soar` for the cross-arch arms):

| gate | result |
|---|---|
| `validate-set.sh` | GREEN -- strict apply, both optional orders (alone + cumulative), Windows lints, credit (43 handles / 38 rows / 0 debt), file-set reconciliation (0 loss, 2 new files, 1 new env), system headers |
| `fork-refresh.sh --build --gates` | tree/tip match, `all` build, 16/16 shared libraries resolve, coherence + prefill-logit gates PASS |
| `gate-gqa6-fold.sh` (new, gate 50) | PASS on gfx1100 (census 1/0, text `ebd02a60725c`/`ca199dcf8218`, off arm `-0.000000 / 100.000 %`, fold arm the same) and on gfx1201 (refusal: census 0 both arms, text `2245b10c2cde` single-device and `0f612e3cad2e` 3-GPU tensor, both KLD arms `-0.000000 / 100.000 %`), fault scan clean |
| `gate-optional-build.sh` (gate 47) | PASS (tree `9ee987063`, `all` target) |
| `gate-mixed-kv.sh` (on that build) | PASS, 14 checks, 2 refusal arms SKIPPED (no Ling model on this box) |
| `check-offline-tests.sh` (gate 49) | PASS on `soar` (4 environment-exempt tests, as at r374) |
| `check-shlib-symbols.sh` / `check-gen-docs.sh` | 16/16 resolve / docs in sync |
| coherence vs the r375 build | 9B Q8_0 `37b826ac5e2a`, 27B Q4_K_M `ddd04dc91b87`, 27B IQ3_S q8_0 with the fold ON `ca199dcf8218` -- all identical; the Ornith-35B-A3B IQ4_XS arm moves only through the block-06 row (`7a5911eefc77` -> `8d25aa669cfd`), and `MMB_ROUTED=1` reads `0.000001 / 100.000 %` against the r375 base |
| RDNA4 non-regression | r376 build vs the r375 base on gfx1201: default, `TILE_GQA6_FOLD=1` and `MMB_ROUTED=1` all `-0.000000 / 100.000 %`; `FLASH_ATTN_EXT` 7581/7581 with the fold on |

**Fixed during the cut, not deferred:** the new gate's arm C compared the caller's `~/llama.cpp` build with
itself instead of the build under test (it now exports `LLAMA_PPL=$BUILD_DIR/bin/llama-perplexity` and prints
it) -- on `soar`, where that checkout is the previous release, the arm had read a meaningless 0.000000; and
`gate-gqa6-fold.sh` picked the first 27B GGUF, which can be a Q8_0 no single card holds, so it now prefers the
delivery's IQ3_S test quant and names the text log when an arm produces nothing.  `gate-optional-build.sh` warns
when the GPU target is not on the box, because a build for an absent target compiles fine and then aborts every
gate run with "device kernel image is invalid" -- which is exactly what cost an hour of this session.

**Images.**  The `radiance` watcher (cron, every 10 min) had `r375` as `last-built` at the time of the cut; it
picks the new tag up on its next poll and runs that tag's `build-containers.sh --push --prune`.  A release is
only finished when `scripts/build-containers.sh --verify-only` says every tag RESOLVES -- see the entry above
the index for that verdict.

## 261128.004 (r376) -- PR #362 promoted: the padded GQA-6 tile fold is block 05, default ON; issue #385's MMB routed row is block 06; release v18-919794500-r376

**Contribution.**  `@kapojko`'s **PR #362** (padded GQA-6 tile flash-attention fold) was rejected as submitted on
2026-11-25 (`archive/work/pr362/`, rework R1-R6) and resubmitted on 2026-11-27 as a fully isolated kernel TU.  This session
assessed the resubmission, found R1-R6 met, and promoted it: it is now **part of block 05, DEFAULT ON**, with
`GGML_HIP_FA_TILE_GQA6_FOLD=0` as the kill switch (`DIAGNOSTIC.md` 8), the census proving the arm, and a new
`scripts/gate-gqa6-fold.sh` (gate 50) as its oracle.  `@noctrex`'s **issue #385** (the RDNA3_0 MMB routed prefill
row) is folded into block 06 as a 1-line `c.routed = 0` with the reporter's numbers reproduced here.

**What the fold does.**  Head 256/256, GQA ratio 6, a symmetric NATIVE q8_0 K/V pair, `n_q <= 8`, `RDNA3_0`: the
tile launcher's `ncols2 = 2` makes three logical K/V traversals per query row; the fold pins `ncols1 1, ncols2 8`,
placing all six real query heads in one padded eight-column block (columns 6/7 get zero Q, are never stored, and a
padding-only warp still joins the cooperative loads and barriers).  It is a **kernel copy** in its own TU, not a
template parameter on the shared kernel -- the runtime-branch form moved the ordinary kernels' register allocation,
and on gfx1201 that moved the decode/verify split the launcher derives from occupancy (VGPR 140 -> 94, mean KLD
0.000391 / same-top-p 98.97 %, tile 1.4-5.7 % slower with the fold UNREACHABLE).  The only shared change is
`launch_fattn`'s `force_split` template parameter, compiled out unless the fold's launcher sets it.

**Why `optional/` was not the answer.**  The PR shipped it as an unnumbered proposed artifact.  Three reasons it
could not stay there: (a) `optional/` is for a finished feature with **no** validated win on the delivery's own
configurations and this one has one on gfx1100; (b) `optional/0102` after `0100`+`0101` fails the strict cumulative
apply `validate-set.sh` requires (optional 0101 adds `type_V` to the very lines the fold hook anchors on); (c) the
default-on policy.  Integration needed no surgery on 0101 because the fold's dispatch tests the OPERANDS
(`K->type == V->type == Q8_0`), so a mixed pair can never take a kernel that has one K/V type -- 0101 was re-cut
mechanically against the new default tree instead.

**Verified here (r375 tree + the two amendments, 27B IQ3_S, `-ctk q8_0 -ctv q8_0`, `ROCBLAS_USE_HIPBLASLT=0`).**
gfx1100 (7900 XTX): the kill-switch arm is **inert against the unpatched release build** -- decode-width `--kv q8_0
--ub 1` mean KLD **0.000000 / same-top-p 100.000 %** with the raw u16 log-prob files **byte-identical**, the same
482-char greedy text, and tg within +-0.2 % -- and the **fold arm reads the same numbers** (identical file again),
so the fold is a rounding-level re-grouping at the only point that can see it.  Decode tg64 on/off: **52.71 ->
53.00 d2048 (+0.5 %), 51.58 -> 52.05 d8192 (+0.9 %), 47.11 -> 48.33 d32768 (+2.6 %), 42.29 -> 44.11 d65536
(+4.3 %), 38.40 -> 40.59 d98304 (+5.7 %)**.  `test-backend-ops -o FLASH_ATTN_EXT` **7581/7581** with the fold on
(the added coverage reaches it: 513 cases at `hsk=256.*nr23=[6,1]`).  gfx1201 (1x R9700, `soar`): both arms
`-0.000000 / 100.000 %` against the unpatched build, identical text single-device (`2245b10c2cde`) and 3-GPU
`-sm tensor` (`0f612e3cad2e`), `tg64` 27.95/26.24 unmoved, `FLASH_ATTN_EXT` 7581/7581 with the fold ON, and **no
census even with the env set** -- the arch gate holds, so RDNA4's obligation is non-regression only.

**#385, reproduced with the reporter's own model.**  Ornith-1.5-35B-A3B **IQ4_XS**, gfx1100, pp2048 r3: routed ON
-> OFF **4854/4070/3553 -> 5060/4256/3671 t/s at d0/d8192/d16384 (+4.2 / +4.6 / +3.3 %)**, matching his
+3.2/+3.9/+3.3 %.  The same A/B on **IQ4_NL is neutral** (5117/4267/3692 -> 5104/4246/3683), so the loss is the
IQ4_XS routed shape's, and the arm is not vacuous (`MMB_GLU fused` 2 lines with routed on, 0 off).

**Purity -- the routed stand-down is a kernel-family switch and the text DOES move on the IQ4_XS MoE.**  The
canonical point on that model reads mean KLD **0.007396 / same-top-p 96.667 %** against the routed-on arm
(`GGML_CUDA_MMB_ROUTED=1` reads **0.000001 / 100.000 %** against the r375 base, so the rest of r376 is inert --
the whole delta is this row), i.e. over the absolute 0.005 / 98 % limits but **no worse than the same-kernel
`-ub 256` control measured on the spot, 0.010867 / 93.333 %**: `GREEDY-PURITY.md` 46's calibration, the same
criterion r372's RDNA3_0 head-256 prefill row was taken on.  Same-seed greedy text moves at that near-tie
(**443 chars `7a5911eefc77` -> 428 chars `8d25aa669cfd`**); the 9B and 27B dense arms are unchanged
(`37b826ac5e2a`, `ddd04dc91b87`), and so is the 27B q8_0-KV arm with the fold ON (`ca199dcf8218`).  On the
IQ4_NL arm the point is 0.001836 / 99.216 % (inside the limits) with the text unchanged.  Decode never reaches
MMB (`mmb_min_t`), so this is prefill-only.

**Release.**  r376 (block 05 + block 06 amended, blocks 07-0300 replayed; 0100/0101 re-cut against the new default
tree), `release.json` regenerated, `validate-set.sh` green, `gate-gqa6-fold.sh` PASS, both Windows lints clean, the
credit register gains `@kapojko` (co-author) and `@noctrex` (reported) with the trailers in the patches.  The
`wip/pr362/` campaign is closed (moved to `archive/work/pr362/`, with its RESOLUTION.md).

**Findings reported, not fixed (all recorded in the records above):** (1) the fold's kernel copy is a maintenance
obligation, not a one-off -- a future change to `flash_attn_tile` must be mirrored (`INVARIANTS.md` r376); (2) two
coverage gaps remain unmeasurable on this fleet: multi-RDNA3 splits (no such box) and a speculator verify batch on
a q8_0/GQA-6 target (our 27B carries no MTP head, so the submitter's unchanged-acceptance result stands unverified
here); (3) `GGML_CUDA_MMB_CFG=1`'s dump prints the arch table and not the env overrides while claiming both, which
is why #385's arm was proved with `MMB_GLU` instead -- worth a fix at the next block-06 touch.

## 261128.003 (r375, release cut) -- tagged, main and the fork pushed, the assets published, the images verified 23/23, and the prefill-logit point reads 0.000000

**The cut.**  `v18-919794500-r375` (annotated tag, pushed) = fork re-cut tip
`5c35fca859bdae4fc9614bd5d48fac38248f973a`, tree `5fccfbcdc6c5dba3db5972362656b952d7a033ad` (recorded in
`release.json`), 19 default patches (18 official + temporary `0300`) and 2 optional artifacts.  The default tree was
re-derived twice independently (the throwaway `~/llama-scratch` apply and the fork refresh) and both produced the
same tip.  `main` is `91b6838a`; all three CI runs read **success** (`Validate delivery set` on `main` and on the tag,
`Release patch set` on the tag) and the Release carries its four assets.

**The re-cut itself.**  Two blocks were amended: **12** takes the whole transport-selector change (its final form and
the iface fills live there, and the gather counter is in the cache engine it owns) and **15** takes the
`moe_cache_residency`/`gather_read` signature on the iface row it writes last.  Blocks 09/10 are untouched: their
`From` ids move with the chain and nothing else.  The rebuild was `git am -3` per patch plus an apply-and-amend at 12
and 15, and it reproduces `r374 tree + the measured change` byte for byte (`git diff f5f7a8e25 recut-r375` equals the
working diff).  Both optional artifacts were re-cut against the new default tree; `0100` needed a real reconcile,
because the block's scheduler refactor (`pipe_begin`/`compute_split`/`pipe_end`) is exactly where its own interleaved
submission lives, so the gather-read sample was re-seated inside `pipe_end` (once per graph, after the last split).

**Fork refresh (`scripts/fork-refresh.sh --build --gates --push`).**  Branch deleted and re-applied at `919794500`
(strict `git am`), tree matched `release.json`, built the `all` target, `check-shlib-symbols.sh` **16/16**, the
coherence gate passed, and the prefill-logit gate reads **mean KLD 0.000000** (limit 0.005), top-p 100.000 % -- the
change is inert on the default path, as a transport decision should be.  `gh api
repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` reads `5c35fca85...`.

**Also verified before the tag**: `gate-host-pool.sh` **59/59** (its `tsplit` arms cover the Meta `stage_input`
forwarder the signature change touches), `gate-optional-build.sh` (gate 47) **PASS** on the cumulative optional tree
(all target, 100 s, tree `3a30b4377`), `validate-set.sh` green including the rung file-set reconciliation
(`blocks 21 -> 21, files 580 -> 580, defines 190 -> 190, envs 604 -> 607, options 14 -> 14`, 0 unexplained, 3 new),
and same-seed greedy text byte-identical on all four field requests.

**Containers, verified and not assumed.**  The worker watcher on `radiance` picked the tag up on its 05:10 poll,
built and pushed by 05:34, and `scripts/build-containers.sh --verify-only` then reads **23/23 tags PRESENT, every
moving tag carries this release** (`latest`, `rocm-*`, `server-rocm-*` at `sha256:7e734abe6140` on 10.2; `full` at
`9584242d0b09`, `light` at `d60ce6429859`).  No image build ran on the primary box.

**Also in this cut, and found while doing it**: `.github/workflows/release.yml` hardcoded the official block range
as `0000-0015`, so every release note since r356 said the wrong thing; it is derived from `n_blocks` now and this
tag's published body was corrected to `0000-0017`.

**Record.** `release.json`; `WORKLOG.md` 261128.002 (the change and its measurements); `INVARIANTS.md` r375;
`patches/README.md` blocks 12/15; `utilities/h2d-latency/`.


## 261128.002 (r375) -- issue #368: the stage-vs-gather transport is decided from a MEASURED gather read, three switches that lied are fixed, and the reporter's prefill regression has a mechanism (blocks 12/15)

**Reported** (`@DanoPTT`, issue #368, r373 on a single R9700 at PCIe 3.0 x16, Windows): r373 prefill **1818 t/s**
against r368's 2944 (-38%), S2 decode **50.5** against 65.8 after a long prefill "even with staging off", the
bimodal per-start state unchanged, `GGML_SCHED_PIPELINE_BUFS=0` ignored, and `GGML_SCHED_STAGE=0` thrashing a 64 GB
box.  `GGML_SCHED_STAGE_SELECT=0` restored his 2952 exactly, so the transport selector was the prefill cause.

**Reproduced, in part, on our own links.**  The r374 default could not be made to lose on any of our three: GPU0
(PCIe5 x8, 28.7 GB/s) staged **2423.5** against the gather's 1876.4; GPU1 (the M.2 riser, 14.4 GB/s) staged
**1977-2119** against 1579-1651; `radix` (PCIe5 x16, 55.1 GB/s) staged **2575.1** (r374) / **2590.0** (r375).  So the
default was not changed blind, and the fast-link arm is measurably untouched.

**What was wrong with the selector.**  It compared `(host - covered)` per token over the calibrated link against a
fixed **38 µs**.  Three findings:

1. `covered` is CAPACITY, not the hit rate of the routed working set.  Same model, flags and `-ub 8192`, and the
   field shape reads an arena hit rate of **0.95-0.97 at 51 % coverage** on his box against **0.59-0.66 at 51 %** on
   ours, so the gather reads an order of magnitude less there and the model over-stated the transport it was
   trading against.  His own census showed it: the slow state and the good state have the same stand-down set and
   the same residency map, and only the gather count moves (2070 against 2208 calls).
2. Dividing by the link made a SLOWER link more eager to stage: the same bytes over less bandwidth cross the same
   fixed µs threshold sooner, while the staged fill's own cost is those same bytes over that same link.  The
   trade had not improved; only the model's units had changed.
3. The physical break-even is a RATIO of volumes.  Staging's only advantage is the copy engine's overlap, and the
   overlap can save at most the read the gather would have done, so the fill may move at most `1/(1-alpha)` times
   it (alpha = the overlap efficiency, measured 50 % here).

**The fix.**  The gather kernel already chooses per expert between the arena (D2D), the pinned pool and the master,
so it now adds one `expert_bytes` per host-sourced expert into a device counter; the host mirrors it into pinned
memory behind an event and reads it **a graph later**, so there is no synchronize and, being a byte count, no
warm-up bias (a wall-clock sample of a cold first ubatch is 2x off and would have picked the wrong transport).
The selector stages only if the fill moves at most `GGML_SCHED_STAGE_FILL_RATIO` (default **2.0**) times the
measured read.  Measured ratios: **0.80x / 1.13x / 1.14x** on the three links above, all STAGE, so every delivery
decision is unchanged; his box is far above 2, which is why its gather wins.

**Three switches that lied, found while chasing it.**

* `GGML_SCHED_PIPELINE_BUFS=0` (documented GATHER) was parsed `if (v < 1) v = -1;` and silently became the
  residency rule's 3.  Now `0` is a real answer: `SLAB_INIT ... transport_req=0.00 GiB`, "transport region off".
* r362's zero-carve rule was asked ONCE at the model-load preflight; `slab_set_stage_cap`, `slab_set_stage_unit`
  and the residency decision (which runs under the cache mutex) each re-derived `unit x depth` and re-carved.  A
  gather arm therefore carried r374's **1408 MiB** carve, the same three evictions (116 + 158 + 3368 MiB) and the
  same 20-table stand-down.  The answer is latched where it can be asked and every late request reads the latch.
  At a generation length worth quoting (300 tokens on L1/L2, the natural 62-token stop on S2, one fresh server per
  arm) the fixed grabber reads L1 **1907.6 -> 1950.8 t/s (+2.3 %)** and L2 **36.69 -> 38.14 t/s (+4 %)**, S2
  **48.16 against 48.18**.  An earlier 30-token arm read an 8 % S2 gap between the same two arms: that was
  short-decode noise, it is recorded here as refuted, and it is why no decode claim in this entry is made from
  fewer than 300 generated tokens.
* `GGML_SCHED_STAGE=0` then took the scheduler's SERIAL whole-table upload instead of the gather (region carved,
  selector chose staging, pipeline off) -- the shape of his 64 GB thrash.  It now gathers: **1899.1 t/s**, no
  page-cache growth.

**And a per-copy cost, measured** (`utilities/h2d-latency/`, new): below ~1 MiB the link is latency-bound, not
bandwidth-bound.  On GPU1 a 64 KiB copy achieves **5.6 GB/s** and 1 MiB **10.0 GB/s** against a calibrated 14.5,
and the round-trip floor is **20-35 µs**.  The staged fill issues one DMA per host run (~1.9 MB on this shape:
`123258 host DMAs` for `235687 MiB` over an 11-ubatch prefill, i.e. ~11 200 per 8192-token ubatch), so its cost is
`DMA count x per-copy cost`, not `bytes / bandwidth`.  That is the quantity his box can be compared on, and it is
what the tool exists to report.

**Not fixed, and named for the follow-up issues**: the bimodal per-start state (un-reproduced on Linux; sub-issue),
the 23 % S2 loss *with staging off* (we see the eviction, 3642 MiB, but not that magnitude), and his H2D
calibration reading 9.0-9.2 GB/s where r368 read 11.7-12.6 (a per-start probe difference, `TODO` for the tool).

**Verified.**  Same-seed greedy text **byte-identical** on all four field requests (r374 vs r375, sha256 of every
response body); `gate-host-pool.sh` **59/59** (tsplit arms and the GPU fault scan included); `gate-prefill-logits.sh`
**mean KLD 0.000000**, top-p 100.000 %; `validate-set.sh` green (checksums, fresh-tarball strict apply, optional
0100/0101 re-cut onto the new default tree, Windows portability, credit register, and the rung file-set
reconciliation reading `0 unexplained`, 3 new env vars); the four-block re-cut (12 and 15 amended, 09/10 untouched)
reproduces the tree that was measured, byte for byte.

**Record.** `INVARIANTS.md` r375; `ENVIRONMENT.md` §31 (`GGML_SCHED_STAGE_FILL_RATIO`, the rule, the fallback) and
`DIAGNOSTIC.md` §31 (the three fixed switches) and §28 (`MOE_EXPERT_CACHE_TABLES` is report-only);
`patches/README.md` blocks 12/15; `utilities/h2d-latency/`.

## 261128.001 (r374, release cut) -- tagged, main and the fork pushed, the release assets published, the images verified 23/23, and the prefill-logit point reads 0.000000

**The cut.**  `v18-919794500-r374` (annotated tag, pushed) = fork re-cut tip
`f5f7a8e25dd4bc22099684c3aa0f10b5500c8862`, tree `760fa359026d1ce33cf22be3377311b95877bfa9` (recorded in
`release.json`), 19 default patches (18 official + temporary `0300`) and 2 optional artifacts.  `main` is
`eba74f69`; `gh run list` reads **Validate delivery set: success (47 s)** and **Release patch set: success** (the
tag's assets).  The default tree was re-derived three independent times with the deterministic re-cut
(`/tmp` throwaway, `~/llama-scratch`, and the fork refresh) and all three produced the same tip, so `.tip` is
reproducible.  `gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` reads `f5f7a8e25...`.

**Fork refresh (`scripts/fork-refresh.sh --build --gates --push`).**  Branch deleted and re-applied at `919794500`
(strict `git am`), tree matched `release.json`, built the `all` target, `check-shlib-symbols.sh` 16/16, the
coherence gate passed, and the prefill-logit gate reads **mean KLD 0.000000** (limit 0.005) -- the r374 changes are
inert on the default path, as the zero-length-piece and device-lookup fixes should be.  Force-with-lease pushed to
the personal fork only.

**Containers, verified and not assumed.**  The worker watcher on `radiance` picked the tag up on its next poll
(00:00:02), built and pushed by 00:08:07, and `scripts/build-containers.sh --verify-only` then reads
**23/23 tags PRESENT, every moving tag carries this release** (`latest`, `rocm-*`, `server-rocm-*` at
`sha256:004b4edfa987` on 10.2; `full` at `0e2a467184aa`, `light` at `13b26cdac7fd`; the 7.14.1 line the same shape).
No image build ran on the primary box.

**Also in this cut, and found while doing it**: the WORKLOG roll-over had been missed for 2026-11-26 and
2026-11-27 (37 entries still live in `WORKLOG.md`), so both days are archived now
(`archive/worklog/WORKLOG-20261126.md`, `WORKLOG-20261127.md`) and indexed below -- the r374 entry itself is
`261127.027` in the 27th's file.

**Record.** `release.json`; `WORKLOG.md` 261127.027 (the fix and its gates); `INVARIANTS.md` r374.

## Archive index — days before today

| day | entries | ids | file |
|---:|---:|---|---|
| 2026-11-27 | 27 | `261127.001-261127.027` | [`WORKLOG-20261127.md`](archive/worklog/WORKLOG-20261127.md) |
| 2026-11-26 | 10 | `261126.001-261126.010` | [`WORKLOG-20261126.md`](archive/worklog/WORKLOG-20261126.md) |
| 2026-11-25 | 16 | `261125.001-261125.016` | [`WORKLOG-20261125.md`](archive/worklog/WORKLOG-20261125.md) |
| 2026-11-24 | 17 | `261124.001-261124.017` | [`WORKLOG-20261124.md`](archive/worklog/WORKLOG-20261124.md) |
| 2026-11-23 | 27 | `261123.001-261123.027` | [`WORKLOG-20261123.md`](archive/worklog/WORKLOG-20261123.md) |
| 2026-11-22 | 40 | `261122.001-261122.040` | [`WORKLOG-20261122.md`](archive/worklog/WORKLOG-20261122.md) |
| 2026-11-21 | 22 | `261121.001-261121.022` | [`WORKLOG-20261121.md`](archive/worklog/WORKLOG-20261121.md) |
| 2026-11-20 | 7 | `261120.001-261120.007` | [`WORKLOG-20261120.md`](archive/worklog/WORKLOG-20261120.md) |
| 2026-11-19 | 8 | `261119.001-261119.008` | [`WORKLOG-20261119.md`](archive/worklog/WORKLOG-20261119.md) |
| 2026-11-18 | 12 | `261118.001-261118.012` | [`WORKLOG-20261118.md`](archive/worklog/WORKLOG-20261118.md) |
| 2026-11-17 | 9 | `261117.001-261117.009` | [`WORKLOG-20261117.md`](archive/worklog/WORKLOG-20261117.md) |
| 2026-11-16 | 8 | `261116.001-261116.008` | [`WORKLOG-20261116.md`](archive/worklog/WORKLOG-20261116.md) |
| 2026-11-15 | 9 | `261115.001-261115.010 (9; .003 absent)` | [`WORKLOG-20261115.md`](archive/worklog/WORKLOG-20261115.md) |
| 2026-11-14 | 10 | `261114.001-261114.010` | [`WORKLOG-20261114.md`](archive/worklog/WORKLOG-20261114.md) |
| 2026-11-13 | 16 | `261113.001-261113.016` | [`WORKLOG-20261113.md`](archive/worklog/WORKLOG-20261113.md) |
| 2026-11-12 | 10 | `261112.001-261112.010` | [`WORKLOG-20261112.md`](archive/worklog/WORKLOG-20261112.md) |
| 2026-11-11 | 37 | `261111.001-261111.037` | [`WORKLOG-20261111.md`](archive/worklog/WORKLOG-20261111.md) |
| 2026-11-10 | 14 | `261110.001-261110.014` | [`WORKLOG-20261110.md`](archive/worklog/WORKLOG-20261110.md) |
| 2026-11-09 | 12 | `261109.001-261109.012` | [`WORKLOG-20261109.md`](archive/worklog/WORKLOG-20261109.md) |
| 2026-11-08 | 17 | `261108.001-261108.017` | [`WORKLOG-20261108.md`](archive/worklog/WORKLOG-20261108.md) |
| 2026-11-07 | 14 | `261107.001-261107.014` | [`WORKLOG-20261107.md`](archive/worklog/WORKLOG-20261107.md) |
| 2026-11-06 | 22 | `261106.001-261106.022` | [`WORKLOG-20261106.md`](archive/worklog/WORKLOG-20261106.md) |
| 2026-11-05 | 20 | `261105.001-261105.020` | [`WORKLOG-20261105.md`](archive/worklog/WORKLOG-20261105.md) |
| 2026-11-04 | 8 | `261104.001-261104.008` | [`WORKLOG-20261104.md`](archive/worklog/WORKLOG-20261104.md) |
| 2026-11-03 | 15 | `261103.001-261103.015` | [`WORKLOG-20261103.md`](archive/worklog/WORKLOG-20261103.md) |
| 2026-11-02 | 22 | `261102.001-261102.020` | [`WORKLOG-20261102.md`](archive/worklog/WORKLOG-20261102.md) |
| 2026-11-01 | 20 | `261101.001-261101.021 (20; .006 absent)` | [`WORKLOG-20261101.md`](archive/worklog/WORKLOG-20261101.md) |
| 2026-10-31 | 16 | `261031.001-261031.016` | [`WORKLOG-20261031.md`](archive/worklog/WORKLOG-20261031.md) |
| 2026-10-30 | 9 | `261030.001-261030.009` | [`WORKLOG-20261030.md`](archive/worklog/WORKLOG-20261030.md) |
| 2026-10-29 | 8 | `261029.001-261029.008` | [`WORKLOG-20261029.md`](archive/worklog/WORKLOG-20261029.md) |
| 2026-10-28 | 14 | `261028.001-261028.014` | [`WORKLOG-20261028.md`](archive/worklog/WORKLOG-20261028.md) |
| 2026-10-27 | 11 | `261027.001-261027.011` | [`WORKLOG-20261027.md`](archive/worklog/WORKLOG-20261027.md) |
| 2026-10-26 | 8 | `261026.001-261026.008` | [`WORKLOG-20261026.md`](archive/worklog/WORKLOG-20261026.md) |
| 2026-10-25 | 17 | `261025.001-261025.017` | [`WORKLOG-20261025.md`](archive/worklog/WORKLOG-20261025.md) |
| 2026-10-24 | 12 | `261024.001-261024.012` | [`WORKLOG-20261024.md`](archive/worklog/WORKLOG-20261024.md) |
| 2026-10-23 | 6 | `261023.001-261023.006` | [`WORKLOG-20261023.md`](archive/worklog/WORKLOG-20261023.md) |
| 2026-10-22 | 12 | `261022.001-261022.012` | [`WORKLOG-20261022.md`](archive/worklog/WORKLOG-20261022.md) |
| 2026-10-21 | 16 | `261021.001-261021.016` | [`WORKLOG-20261021.md`](archive/worklog/WORKLOG-20261021.md) |
| 2026-10-20 | 15 | `261020.001-261020.015` | [`WORKLOG-20261020.md`](archive/worklog/WORKLOG-20261020.md) |
| 2026-10-19 | 15 | `261019.001-261019.015` | [`WORKLOG-20261019.md`](archive/worklog/WORKLOG-20261019.md) |
| 2026-10-18 | 10 | `261018.001-261018.010` | [`WORKLOG-20261018.md`](archive/worklog/WORKLOG-20261018.md) |
| 2026-10-17 | 8 | `261017.001-261017.008` | [`WORKLOG-20261017.md`](archive/worklog/WORKLOG-20261017.md) |
| 2026-10-16 | 12 | `261016.001-261016.012` | [`WORKLOG-20261016.md`](archive/worklog/WORKLOG-20261016.md) |
| 2026-10-15 | 6 | `261015.001-261015.006` | [`WORKLOG-20261015.md`](archive/worklog/WORKLOG-20261015.md) |
| 2026-10-14 | 8 | `261014.001-261014.008` | [`WORKLOG-20261014.md`](archive/worklog/WORKLOG-20261014.md) |
| 2026-10-13 | 7 | `261013.001-261013.007` | [`WORKLOG-20261013.md`](archive/worklog/WORKLOG-20261013.md) |
| 2026-10-12 | 7 | `261012.001-261012.007` | [`WORKLOG-20261012.md`](archive/worklog/WORKLOG-20261012.md) |
| 2026-10-11 | 18 | `261011.001-261011.018` | [`WORKLOG-20261011.md`](archive/worklog/WORKLOG-20261011.md) |
| 2026-10-10 | 4 | `261010.001-261010.004` | [`WORKLOG-20261010.md`](archive/worklog/WORKLOG-20261010.md) |
| 2026-10-09 | 3 | `261009.001-261009.003` | [`WORKLOG-20261009.md`](archive/worklog/WORKLOG-20261009.md) |
| 2026-10-08 | 10 | `261008.001-261008.010` | [`WORKLOG-20261008.md`](archive/worklog/WORKLOG-20261008.md) |
| 2026-10-07 | 9 | `261007.001-261007.009` | [`WORKLOG-20261007.md`](archive/worklog/WORKLOG-20261007.md) |
| 2026-10-06 | 7 | `261006.001-261006.007` | [`WORKLOG-20261006.md`](archive/worklog/WORKLOG-20261006.md) |
| 2026-10-05 | 16 | `261005.001-261005.016` | [`WORKLOG-20261005.md`](archive/worklog/WORKLOG-20261005.md) |
| 2026-10-04 | 8 | `261004.001-261004.008` | [`WORKLOG-20261004.md`](archive/worklog/WORKLOG-20261004.md) |
| 2026-10-03 | 2 | `261003.001-261003.002` | [`WORKLOG-20261003.md`](archive/worklog/WORKLOG-20261003.md) |
| 2026-10-02 | 8 | `261002.001-261002.008` | [`WORKLOG-20261002.md`](archive/worklog/WORKLOG-20261002.md) |
| 2026-10-01 | 1 | `261001.001-261001.001` | [`WORKLOG-20261001.md`](archive/worklog/WORKLOG-20261001.md) |
| 2026-09-30 | 3 | `260930.001-260930.003` | [`WORKLOG-20260930.md`](archive/worklog/WORKLOG-20260930.md) |
| 2026-09-29 | 5 | `260929.001-260929.005` | [`WORKLOG-20260929.md`](archive/worklog/WORKLOG-20260929.md) |
| 2026-09-28 | 6 | `260928.001-260928.006` | [`WORKLOG-20260928.md`](archive/worklog/WORKLOG-20260928.md) |
| 2026-09-27 | 6 | `260927.001-260927.006` | [`WORKLOG-20260927.md`](archive/worklog/WORKLOG-20260927.md) |
| 2026-09-26 | 4 | `260926.001-260926.004` | [`WORKLOG-20260926.md`](archive/worklog/WORKLOG-20260926.md) |
| 2026-09-25 | 9 | `260925.001-260925.009` | [`WORKLOG-20260925.md`](archive/worklog/WORKLOG-20260925.md) |
| 2026-09-24 | 1 | `260924.001-260924.001` | [`WORKLOG-20260924.md`](archive/worklog/WORKLOG-20260924.md) |
| 2026-09-22 | 3 | `260922.001-260922.003` | [`WORKLOG-20260922.md`](archive/worklog/WORKLOG-20260922.md) |
| 2026-09-21 | 12 | `260921.001-260921.012` | [`WORKLOG-20260921.md`](archive/worklog/WORKLOG-20260921.md) |
| 2026-09-20 | 2 | `260920.001-260920.002` | [`WORKLOG-20260920.md`](archive/worklog/WORKLOG-20260920.md) |
| 2026-09-19 | 3 | `260919.001-260919.003` | [`WORKLOG-20260919.md`](archive/worklog/WORKLOG-20260919.md) |
| 2026-09-18 | 7 | `260918.001-260918.007` | [`WORKLOG-20260918.md`](archive/worklog/WORKLOG-20260918.md) |
| 2026-09-17 | 2 | `260917.001-260917.002` | [`WORKLOG-20260917.md`](archive/worklog/WORKLOG-20260917.md) |
| 2026-09-16 | 2 | `260916.001-260916.002` | [`WORKLOG-20260916.md`](archive/worklog/WORKLOG-20260916.md) |
| 2026-09-15 | 7 | `260915.001-260915.007` | [`WORKLOG-20260915.md`](archive/worklog/WORKLOG-20260915.md) |
| 2026-09-14 | 2 | `260914.001-260914.002` | [`WORKLOG-20260914.md`](archive/worklog/WORKLOG-20260914.md) |
| 2026-09-13 | 7 | `260913.001-260913.007` | [`WORKLOG-20260913.md`](archive/worklog/WORKLOG-20260913.md) |
| 2026-09-12 | 17 | `260912.001-260912.017` | [`WORKLOG-20260912.md`](archive/worklog/WORKLOG-20260912.md) |
| 2026-09-11 | 11 | `260911.001-260911.011` | [`WORKLOG-20260911.md`](archive/worklog/WORKLOG-20260911.md) |

**Follow-up the same day (261128.002, docs/CI).**  The release note's block range was hardcoded `0000-0015` in
`.github/workflows/release.yml` and had been wrong since r356 (blocks `0016`/`0017` shipped while every release note
said otherwise); it is DERIVED from `n_blocks` now, and this tag's published body was corrected by hand to
`0000-0017`.  Found while reading the r375 release page.

**And one wording nit in r375's own code comment, for the next cut** (not worth a revision on its own):
`ggml-backend.cpp`'s measured-rule note says "a ratio near 1 on RDMA-class links and on the x8 box, against ~9 on
the reporter's PCIe 3.0 x16 box".  "RDMA-class" is the wrong term for a PCIe link and the ~9 is INFERRED from his
gather being 38 % faster, not measured; the honest form is "a ratio near 1 on our PCIe5 links ... against a value
well above 2 (his gather wins by 38 %, so the fill must move several times what it reads)".  `INVARIANTS.md` and
`ENVIRONMENT.md` already say it that way.
