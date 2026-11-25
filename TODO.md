# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) — today's entries at its top, earlier days under `archive/worklog/` — plus the
campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**How this file is organised.**  The **index table** is the only list: every **OPEN**, **PARKED** or
**WAITING** handle appears there exactly once, in handle order, and nothing else does.  The `## NNN.` detail
sections follow in the same order, one per row; a row never states a status its section disagrees with, and a
closed item leaves both (the record is a `WORKLOG.md` entry plus the campaign it points at).

**Accepted limitations** — deliberately not fixed, do not re-report — come last, as `### #NNN — subject`
entries, and they are **not** rows: an accepted limitation is not open work, so listing it in the table would
be the one place a "status" could read as a task.  The handle stays in the heading where one was ever
allocated (a dated record that cites it still resolves, and a re-entry condition needs a name); an entry that
never had one says `### No handle — …`.  Each entry names its re-entry condition, where it has one, and the
records it points at (`WORKLOG.md`, the campaign, the `benchmarks/` or `archive/work/` file).

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **19-patch default set** (18 official blocks `0000`-`0017` + the temporary `0300`)
at fork point **`919794500`**; read `.release` in `release.json` for the release id -- it is the source of
truth, not this line.  **Block 17 (the two-arm server) remains WITHDRAWN from the delivery** (r352) and its
repair is the standalone project `wip/prefill-decode-arms-v2/` -- **this file carries no row for that feature**
(the 2026-11-23 housekeeping note below says why).  `#375` is the cache-off host-tier GPU page fault (open,
**not reproducible at r352**) and is a *delivery* item (blocks 09/12/15), not a feature item.  Block 14 is the **streamed KV** (the resident window, the
attendance pool and the MTP draft window -- all three opt-in, default off); block 15 is the UMA GTT expert
store.  The **mixed K/V cache types** are **optional block `0101`** since the r288 block-number re-org (they
were numbered block 17 in r287, PR #312 -- a different block from today's `0017`; the one native pair, head 256
only), and `0100` is two-lane prefill.  `release.json` is the source of truth for the tip/tree, counts and hashes; the top of `WORKLOG.md` is the
newest record.  **Block `0300` is temporary** (item #104).
The r278 re-base (`archive/work/upstream-rebase-261113/`, base `4f9dfa0e3`) is **cut and released into
`patches/`**: its 17 blocks are replayed, both rung-repairs are squashed into their owning blocks, and the gates
are recorded in `WORKLOG.md` 261113.015.  What it left open -- #322 (a stale `gate-server-arms` baseline), #323 (the
prefill-logit base file name), #324 (both `optional/` artifacts stale), #325 (the APU-only gate set) and #326 (the
loader's async-upload lifetime) -- closed at **r285** (`WORKLOG.md` 261114.006), which is why nothing about them is
left in this file.  The r236 re-base
(`archive/work/upstream-rebase-261108/`) is superseded and closed.

## The open list

**`2026-11-23` housekeeping -- the block-17 rows were RETRACTED (the maintainer's decision).**  Every row whose
subject was the **two-arm server** left this file: `#364`, `#365`, `#366`, `#367`, `#368`, `#370`, `#371`, `#372`
and `#374`.  The feature is **withdrawn from the delivery** (r352) and is now a **standalone, self-contained
project**, `wip/prefill-decode-arms-v2/`, which carries its own goals, design, defect register and step-by-step
plan (`README.md`, `DESIGN.md`, `PLAN.md`).  It is a clean-slate redesign rather than a repair, so its work does
**not** live in this file: a forward-looking row here would duplicate a plan that is already five times more
specific, and the retracted rows described a design (a second context that *views* and *co-owns* the first's
cells, with the recurrent state copied between two memory modules) that the redesign deletes.  The numbers are
**void and never reused**; `#369` stays because its subject is **delivery gate debt** (three gate defaults
affected by a retired model tree), not the feature, and it no longer cites a campaign.  The old follow-up
campaign is archived at `archive/work/prefill-decode-arms-followups/` (`WORKLOG.md` 261123.005).

**`r333` housekeeping (2026-11-21):** three rows left the list at r333 and none is tracked any more -- `#349`
(gate-server-arms' two-slot text was a `--kv-unified` artifact, `GREEDY-PURITY.md` §38; fixed and its anchor
restored), `#351` (the `flint` / `-sm tensor` question: **retired by the maintainer's decision**, the temporary
block `0300` and its drop condition are theirs to re-test at a re-base, and no TODO row is owed for it) and
`#353` (Inkling-2: the support is upstream now and is picked up at the next re-base, so no reminder is kept).
**`r340` housekeeping (2026-11-21):** two more rows closed and left the list -- `#359` (the
`gate-fusion-alias-copy.sh` arm-A oracle re-derived as a logit point; the unfused chain's move at the r328 base
is block 09's host-offload relaxation, rounding level, `WORKLOG.md` 261121.020) and `#362` (the hipBLASLt lottery
measured on both ROCm lines -- 10.2 offers ONE solution per call where 10.1 offers 18, and 10.2 is recorded as
the canonical validated line in `BASELINE.md`; `WORKLOG.md` 261121.021).
**`r341` housekeeping (2026-11-21):** `#358` closed and left the list -- implemented, not deferred:
`scripts/check-rung-file-sets.py` reconciles every block against the patch it replaces and fails on a file,
`#define`, env var, option or block the previous release carried and this one does not, with decision rows in
`patches/FILE-SET-RETIREMENTS.tsv`; it reproduces all three r331 findings on r325->r328, fires on nothing else
across 70 consecutive release transitions (68 clean; the only other firing is the documented r288 re-org), and is wired into
`validate-set.sh`, `validate-gpu.sh` and `MANIFESTS.md` gate 38 (`WORKLOG.md` 261121.022).  The rows that
remain are (`#116` parked, `#273`, `#361`, `#363`): each carries its own re-entry condition.  A green gate
ladder is still not permission to skip a row.

**`r352` housekeeping (2026-11-23):** **block 17 (the two-arm server) was WITHDRAWN from the delivery** and moved
to `beta/prefill-decode-arms/` -- the artifact reproduces the r351 tree exactly, so the code is preserved, and
re-admission is the maintainer's decision.  One new row came out of that session: **`#375`**, the cache-disabled
host-tier **GPU page fault** recorded at r350 (`WORKLOG.md` 261122.040 §3) -- it reproduced at r349 with the pair
inactive and did **not** reproduce on any of the seven arms re-run at r352 (both builds), so it is open with a
recorded control set and **no fix**: a gate on a fault that does not fire would be vacuous.  Its subject is the
delivery's host-expert tier, so it stays.

| # | subject | kind | status |
|---|---|---|---|
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM | performance / new type (`wip/fp8-support/`) | **PARKED** (2026-09-26) — builds on gfx1201, gate 2 fails (0.69x Q8_0) — §116 |
| **273** | three prefill transports where one should do | architecture (delivery) | **OPEN** — the host-set SIZE is the primary lever, residency secondary; ONE transport for a partly resident table — §273 |
| **361** | the r334 RDNA2 items (PR #338) are validated on the contributor's V620, not on the released tree: blocks 05/08/10 now carry an RDNA2-only FA tile-cap removal, a one-warp narrow `RMS_NORM`, a `H * n_seqs` GDN chunked gate and a rows-abreast MoE K walk, and no RDNA2 device is in the maintainer's fleet | validation / external (delivery) | **WAITING on `@jstamagal`** -- re-run on the r334 tree: full `test-backend-ops` on `gfx1030` at default flags (**especially `FLASH_ATTN_EXT` with the caps gone: the failure mode is `GGML_ASSERT(max_blocks_per_sm > 0)` in `launch_fattn`, not a rate**), `MUL_MAT_ID` with `GGML_CUDA_MOE_ROWS_ABREAST` on and off, `RMS_NORM*` in the default/`NARROW=0`/`VEC=1` arms, a model-level decode+prefill (a GDN model for the chunked gate, a MoE for the K walk) and an MTP-acceptance run; **it should also confirm that PR #339's BF16 `RMS_NORM + MUL (+ ADD)` abort is gone there** (fixed at r328 by the re-base, `archive/work/can-fuse-bf16-rms-norm-r324/` -- `RMS_NORM_MUL_ADD` now includes `type=bf16,type_w=f32` and `type=bf16,type_w=f32,type_add=bf16` arms); §361 |
| **363** | `-of`/`--optional-features` exists with ONE tag (`tp-ffn-even`); the delivery's other env-only opt-in features are still environment-only, so a user who reads `--help` still cannot find them | tooling / docs (delivery) | **OPEN** -- migrate them one at a time: add the row to `common_optional_features()` with its switch, confirm the tag produces exactly what the env var produces (the arm `gate-ffn-even-split.sh` C-section runs), and delete the row if the feature gets a first-class option; candidates are the opt-in switches `ENVIRONMENT.md` already marks as such (§4 `LLAMA_KV_N_PAD_MIN`, §7.6 `--wmma-verify`-adjacent switches are options already, §1.2's arena knobs, block 14's `LLAMA_KV_SPILL_RAM_PCT`); §363 |
| **369** | three gates still name the RETIRED Flash-Next `GSQ-IQ3_XXS` tree (`gate-hc-base-copy`, `gate-hc-fusions`, `gate-host-expert-text`) and were deliberately NOT repointed at r346, because each carries recorded text/KLD anchors for that class | tooling (delivery) | **OPEN** -- repoint each AND re-record its anchors on the new class (the STEW675 IQ4_NL tree) in the same change, then run each once. §369 |
| **376** | the block-17 beta artifact (`beta/prefill-decode-arms/`) carries a fix for a DELIVERY-level defect -- the CUDA backend issued a 2-D device->host copy of a split tensor into a pinned host buffer another device owns, which the driver refuses (`2-D read of 'result_output' failed (invalid argument): 331092 bytes x 5 copies, tensor pitch 331092, host pitch 993280`; the mechanism is the peer grant, `hsa_amd_agents_allow_access ... err 4096`) -- so on promotion that fix must be folded into the OWNING delivery block (**block 09**, the CUDA iface/scheduler block; it is `ggml-cuda.cu`) **independently of the two-arm block**, and the beta artifact re-cut against the new tree | delivery / block structure (maintainer) | **OPEN (fold only)** -- the fix LANDED as STEP 15 (`WORKLOG.md` 261124.007, `INVARIANTS.md` r384-wip, `steps/STEP-15-0022-no-2d-d2h.patch`); the FOLD is owed at promotion and is a block amendment, so it is asked first.  **A SECOND fold rides with it (261124.009)**: STEP 16 -- the pair keeps the compute-buffer drop, fenced by the other arm's step window (`llama_context_set_drop_boundary()` in `include/llama.h` + `src/llama-context.*`, and the server fence) -- is delivery code too, so the delivery takes two amendments (or one, if block 09 is cut with both) |
| **377** | `scripts/gate-seqpar.sh` needs a **Flash-Next prefill-logit base** (`~/.cache/rdna-boosts/gates/prefill-logit-*Flash-Next*`) that this box did not have -- only the 27B one existed, so its arm 2 ("the `LLAMA_SEQPAR=0` arm is the recorded base") could not pass on ANY build, and arms 3a/3b scored nothing.  The base is machine-local, so a fresh box or CI hits the same red gate | delivery / gate plumbing | **OPEN (housekeeping)** -- recorded 2026-11-24 on the STEP-14 tip with the gate's own `--record` (the documented first-run path), which makes `gate-seqpar` meaningful again here; the open part is that a *fresh* box has no base and the gate's failure message does not say "record one first".  `gate-seqpar --record` must be run on a PRE-change tree, never on the build under test |
| **378** | the `beta/prefill-decode-arms/` two-arm server **corrupts output under heavy concurrent load**: the maintainer reproduced a corruption on 2026-11-24 by pushing it hard (pair ON, concurrent prefill + decode; the candidate otherwise passes its own crash/state/text/gate ladder).  No reproducer, no captured failing run and no localisation yet -- this is the campaign's last known open defect and it is why the artifact ships as BETA | beta candidate / correctness | **OPEN** -- needs a REPRODUCER first (the shape, the request pattern and the symptom are not recorded beyond "pushed hard"); then the usual split: text/KLD oracles on the same shape, the gate that owns the mechanism, and `INVARIANTS.md` for the fact.  Re-entry: any pair run that returns text differing from the single-context server at the same seed, or a `state-split`/`arms` gate failure that is not reproducible on the pair-off arm |
| **381** | block 17's **gfx1100** arm was not run: `fingon` rebooted twice mid-build and then dropped off the network during the r356 session, so the cross-arch matrix is gfx1201 + gfx1151 only | validation / external (delivery) | **WAITING on `fingon`** -- when the box is back: pull `main`, apply the set in a worktree at `release.json.base`, build, then `./scripts/gate-tq.sh --build <dir> --tq2 ~/models/Llama3-8B-1.58/TQ2_0.gguf --tq1 ~/models/Llama3-8B-1.58/TQ1_0.gguf --dense <a 4B Q8_0> --dense-ref '' --device 0 --no-split --bench`.  gfx1100 has no RDNA4 int8-WMMA kernel, so `mma=0` there is by design (gfx1151 already covers that path: **305/305 and 84/84 exactly 0**); what gfx1100 adds is a second non-MMA arch on the neutrality and rate arms.  Note the two instrument traps the gfx1151 run found: the CPU-placement text arm is not an oracle, and `TQ_CENSUS` can print more than one line per device (both now handled by the gate); §381 |
| **380** | block 17's `ggml/src/ggml-cuda/tq.cu` (689 lines of new CUDA) has **only ever been compiled by HIP/clang**: the contributor states it, the delivery's CI compiles nothing, and there is no NVIDIA device or CUDA toolkit on any box in this fleet — so the CUDA (nvcc) build of the ternary path, and its bit-identity there, are unverified | validation / external (delivery) | **WAITING on an NVIDIA box (or a CUDA toolkit)** -- compile `tq.cu` with nvcc (`-DGGML_CUDA=ON`, any recent CUDA), then `test-backend-ops -o MUL_MAT -p type_a=tq` and `-o GET_ROWS -p type=tq` with `LLAMA_TEST_OP_NMSE=1`: the `#pragma clang fp contract(off)` is clang-guarded and `TQ_FMUL`/`TQ_FADD` fall back to `__fmul_rn`/`__fadd_rn`, but nvcc's `-use_fast_math` flushes subnormals, so **all 305/84 values reading exactly 0 is the claim to test**, not the text.  A `MUL_MAT_ID` (TQ experts stay on the CPU backend by design) or a draft/MTP TQ arm would be the follow-up; §380 |
| **379** | `ggml_cuda_try_fuse` still costs ~7 ms per `graph_compute` on the 27B (~890 calls) even with the GDN norm-gate fold switched OFF: *other* matchers count readers over the whole graph too, and every count is a full node walk | performance / host time (delivery) | **OPEN** -- take the reader counts ONCE per graph (a per-graph `count_uses` cache, or a single pass that fills every tensor's count) and let the matchers read it instead of walking. It is a matcher-design change, not a fold: it must keep every matcher's decision identical (the r355 instrument -- re-run the pre-change matcher at every accept site and compare -- is the cheap oracle), and it pays only where the node loop runs (`GGML_CUDA_DISABLE_GRAPHS=1`, prefill, verify, recapture). Left out of PR #361 for that reason; `archive/work/gdn-norm-gate-once/`, `INVARIANTS.md` r355 |
| **382** | block 12's tool-text protection is a WORKAROUND for a missing piece: the renderer already carries `common_chat_prompt_part::is_input` ("from user input, special tokens inside must not be parsed") and nothing consumes it, so a tool result's special-token strings have to be marked and re-tokenized by hand | correctness / design (delivery) | **PARKED** -- the real fix is per-PART tokenization: one flag in `llama_vocab::tokenize`/`tokenizer_st_partition` meaning "no special-token matching at all" (today `parse_special=false` still matches USER_DEFINED ones), then tokenize `out_session.prompt().parts` with it on every `is_input` part.  That removes the marks, the second template render, the intactness heuristic and the space fallback, and covers `user`/`system` text too; take it at a re-base or when upstream offers a tokenizer-side switch -- `§382` |

## Handle numbering

**New handles append after the highest issued so far, so the next free handle is `#384` (`#383` is the tool-text protection's role-scoped limitation, `#382` its parked redesign, `#381` is block 17's owed gfx1100 arm, `#380` is block 17's unverified CUDA build, `#379` is the whole-graph reader-count cache the matchers still owe, `#378` the two-arm server's corruption under heavy load, `#377` the missing `gate-seqpar` prefill-logit base, `#376` the block-17 promotion-time fold, `#375` the cache-off page fault).  `#364`-`#372` and `#374` were issued for the block-17 follow-ups and **retracted (void) on 2026-11-23**; a number is never reused.**  `#NN` is a permanent handle: a `WORKLOG.md` entry that cites `TODO #74` still means the same
subject after that item has closed and left this file.  The recent ones, so the sequence is auditable:

* **issued and retired the same day** (2026-11-16): `#337` — the attendance pool *does* engage on
  Qwen3.8-Flash-Next under `--kv-unified`, so the "cannot be exercised" premise was wrong; the arm and its
  numbers are in `archive/work/session-scheduling/`.
* **`#329`** retired at r312 (`WORKLOG.md` 261118.005) — *refuted*, not implemented: at 2..4 verified columns the
  K-split `mmvq` kernel is already at 92-98 % of the DRAM roof, so no narrower tile or dot-product low-band arm
  has anything to win; the family stays opt-in and width-split, and what shipped instead is the measured
  **per-type** `auto` floor (5 / 5 / 8 / 8 / 8).  Record: `archive/work/rdna4-wmma-verify/LOW-BAND-2026-11-18.md`.
* **`#345`** (issued r308 -- the 256-dim band) retired at **r315** (`WORKLOG.md` 261118.010): implemented as the
  class's per-layer work gate, with its preferred route (an adaptive `P`) refuted by measurement.  It was
  mistakenly listed here as retired at r311 while the open table still carried it; the r315 record is the
  authoritative closure.
* **`#346`** (issued r315 -- the band's fixup period `P`) retired at **r316** (`WORKLOG.md` 261118.011): the
  quantized types' `P = nsm` shipped as `nsm/2` (one constant for every K/V type), chosen on the deep
  attention-bound arm; **`P` 8 was measured and rejected**, and the 2-byte family's own verify-width optimum is
  recorded in the benchmark rather than acted on.
* **`#338`-`#340`** retired at r301, **`#341`** (issued r303) retired at r309 (`WORKLOG.md` 261117.008).
* **`#335`/`#336`** retired at r298 — an alias ROLE gap in the expert cache, `WORKLOG.md` 261116.005;
  **`#282`** at r299 — the mid-run latch's mixture is gated on the logits, `WORKLOG.md` 261116.006.
* **`#349`** retired at **r333** (`WORKLOG.md` 261121.012): the arm was asserting a concurrent two-slot text
  hash under `--kv-unified`, which `GREEDY-PURITY.md` **§38** forbids; the fix is `--no-kv-unified` plus a
  deterministic admission (pinned `id_slot` + a warm-up request that holds the scheduler while both real
  requests queue), and the item's own prescribed fix (release slot 1 on slot 0's first streamed token) was
  implemented and **refuted by measurement** first.  `#351` and `#353` were retired at the same time (no
  reminder owed for either: the `flint`/`0300` question is a maintainer call at each re-base, and Inkling-2
  support is upstream and arrives with the next re-base).
* **`#342`** (r303) and **`#343`** (r304) are the newest still tracked; `#342` is WAITING on upstream, `#343`
  is an accepted limitation.

**Two allocation mistakes are on the record, and both are left in place because renumbering would break the dated
records** (the commit messages that cite them are immutable).  The r187-r189 sessions allocated `#190`-`#196` (and
`#218`) as if the sequence had run past `#119`; they are `#120`-`#127` wherever the file was later corrected.  The
2026-11-02 session allocated `#268`/`#269` out of sequence in the same way.  Neither set is reusable.

---

## 116. Native FP8 E4M3 for RDNA4 — PARKED behind a GEMM regression (campaign `wip/fp8-support/`)

**Status: parked (2026-09-26).  Not scheduled, not in `patches/`, and nothing may be applied from it without an
explicit go-ahead (`AGENTS.md`, WIP rules).  The campaign is `wip/fp8-support/` (`README.md`, `PLAN.md`,
`HANDOVER.md`, `RESULTS-2026-09-26-rebase-gate.md`).**

**Why it exists.**  Single-request prefill is **70-81 % `mul_mat_q`** (sibling campaign
`archive/work/prefill-gap-attribution/`), so the vLLM gap is a GEMM problem, and the int8 MMQ is
**epilogue-bound**: every per-16 quantization pays a scaled dequant epilogue in the MMA path (~57 TFLOPS
against Q8_0's 83 and F16's 107 — `archive/work/per16-f16-mma/`).  FP8 tensor cores need no int8 dequant epilogue, gfx1201's fp8 WMMA peak is
~171-370 TFLOP/s-class against the int8 MMQ's ~41.5 T-MAC/s, llama.cpp has **no FP8 weight type at all** (so
today's "75 % of vLLM" comparison is engine *and* precision), and the port already measured **+13-17 % prefill
over Q8_0** on a 4B / gfx1201 on the maintainer's `cllm` branch (last touched 2026-08-06, base `6ea215d17`).

**What exists.**  `patches/` = the original 23-commit `cllm` series (provenance; `~/cllm` tip `7c17faffc` is
one unpushed commit ahead of `origin/cllm` and is the source of truth), `patches-rebased/` = the delivered
re-base: **13 commits, 37 files, +5374/-12**, on the r9 base, with the 10 superseded commits (chunked GDN,
ssm-conv, perf docs) dropped because block 02 ships a newer and different GDN.  The port is
`GGML_TYPE_F8_E4M3` (43, self-contained `[f32 scale][128 fp8]` blocks), `ggml-cuda/fp8.cu` (`mul_mat_fp8_wmma`:
CTA 128x64, 8 warps, `GROUP_M=4` pid swizzle, 16x16x128 tiles, register-staged, repacked weights; plus
`quantize_fp8` activation staging and a `dot4` GEMV for decode), direct HF safetensors loading, a
`convert_hf_to_gguf.py --outtype fp8_e4m3` path, a CPU quant + `GET_ROWS` (bit-exact vs CPU), and RDNA4 gating
(compile-time `RDNA4` + runtime `CC_IS_RDNA4`, scalar fallback otherwise).

**Where it stopped (gate 2 fails).**  4B, `pp512`, gfx1201, same box: Q8_0 **8146 ± 221** (the `cllm`-era
figure was 6177, i.e. the delivery's MMB/GEMM work is +32 %) vs `F8_E4M3` **5607 ± 15** (the `cllm` figure was
7184-7260) — **0.69x** where it used to be **1.16x**.  The machine is not the problem, the fp8 path is: the
kernel runs ~65 TFLOP/s here against the ~77 the old notes recorded **on the same source**, and
`fp8_repack_weights` takes **7.3 %** because it appears to run **per call** instead of once (it is meant to be
a lazy one-time layout change cached on the context — check `fp8_backend_cuda_context::fp8_repack_buf`
invalidation on the newer base).  GDN/ssm is 3 % and attention 1.3 %, so this is GEMM + repack + staging, not
the dropped commits.

**Next actions** (`PLAN.md`, in order): (1) interleaved A/B of `mul_mat_fp8_wmma` on the re-based tree against
`~/cllm` at the same `pp512` to isolate codegen vs launch configuration; (2) fix/confirm the repack caching
(target its 7.3 % to ~0); (3) re-run the 4B gate (fp8 must beat Q8_0 again); (4) only then convert
`/llm/models/Qwen3.8/27B/FP8/` and measure `pp8192` against the int8 baselines (Q8_0 1371 / Q6_K 975 /
Q4_K_XL 1254 t/s), which is the number that decides the campaign; (5) lift AITER's gfx1201
`A8W8_BLOCKSCALE` tiles / `GROUP_SIZE_M` / `kpack` (target `mul_mat_fp8_wmma` ≥ 110 TFLOP/s effective; AITER's
Triton reaches 121-137 against our 77-98).

**Risks / constraints.**  The **conversion is the correctness cliff**: HF fp8 (per-tensor/per-channel weight
scales, dynamic per-token activation) → the 128-block `block_f8_e4m3` grid, and a wrong scale mapping produces
plausible-but-wrong text — so the oracle is PPL against the vLLM fp8 record (fp8 6.2250 vs Q8_0 6.2464) plus the
same-seed greedy gate, never "it runs" (there is **no CPU oracle** for an fp8 `MUL_MAT`).  FP8 is a different
numerics path, so it can never be bit-identical to int8 and the purity contract is a PPL/oracle comparison.  A
**non-RDNA4 policy** (reject vs dequantize rather than a silent scalar fallback) has to be settled before any
promotion.  "Dequantize to bf16 + hipBLAS" is a **trap** — upstream already measured that on RDNA4 MMQ beats
dequant+hipBLAS (`ggml-cuda/mmq.cu:608`, PR #18537) — so the win must come from the native WMMA GEMM.  And the
4B result may not scale (the 27B has different shapes, a bigger lm_head and more attention).  Promotion shape,
if it ever gets there: a new weight type + kernels + loader + converter is a **large block** and a maintainer
call.

**The index row's detail (kept here so the table can stay a dashboard):**

native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM (campaign `wip/fp8-support/`)

**Kind:** performance / new type.

**PARKED** (2026-09-26).  Re-based onto the delivery as 13 commits (`~/llama-fp8`, branch `fp8-rebase`), builds clean on gfx1201, fully RDNA4-gated — but **gate 2 fails**: on a 4B the fp8 arm is **0.69x Q8_0** where the pre-rebase `cllm` branch had **1.16x**, because the delivery's MMB/GEMM work lifted Q8_0 **+32 %** (8146 vs 6177 t/s pp512) and the fp8 WMMA path did not follow (5607 vs 7184-7260).  Kernel trace: `mul_mat_fp8_wmma` 71.9 %, **`fp8_repack_weights` 7.3 % (running per call instead of once — the likely bug)**, staging 3.3 %, FA 1.3 %.  Not delivery work: nothing here is in `patches/`.  Resume: A/B the GEMM against `~/cllm`, fix the repack cache, re-run the 4B gate, only then convert and measure the 27B.  Detail in §116

The index row also recorded the r209-era kernel trace split: `mul_mat_fp8_wmma` **71.9 %**, `fp8_repack_weights` **7.3 %**, staging **3.3 %**, FA **1.3 %**.

## 273. Three prefill transports where one should do

**Three prefill transports where one should do**: a host-resident expert table's prefill is served by the device gather, the staged fill or the scheduler's bulk upload, selected by `GGML_SCHED_DEVGATHER` x `GGML_MOE_CACHE_STAGE` and a tier test — each wins somewhere, so no single default is right.  Measured on r234: bulk wins at ≤ 67 % arena residency (+16 % at 44 %, +6 % at 67 %, 1 and 2 devices, both box phases), the staged fill wins at ≥ 83 % (up to +10 %; at 99 % it sends 0.0 % of the bytes over PCIe), on the pool tier the staged fill is mandatory (the alternative is the per-expert pinned bounce, 5-10x slower), and the gather is the pool default that wins at `-ub` ≤ 4096.  The trade is **bytes saved vs transactions issued**, and the resident set is scattered so the complement cannot be coalesced

**Kind:** architecture / delivery — a transport consolidation.

**OPEN — the current direction (2026-11-08, `TODO` #296 -> this row).**  The **host-set SIZE is the primary
lever**: every transport scales with the bytes of host-resident experts a ubatch must read (measured 1872 -> 1989
t/s, **+6.3 %**, from `-ncmoe 48` to `36`).  Arena residency is secondary — only a transport that *reads* the
duplicate can use it, which is why bulk measured flat across residency — and transaction count is third (the SDMA
phase).  **Leading candidate for one transport**: a *contiguous* static VRAM tier (the range chosen by a
heat-maximising sliding window) + a **two-transaction fill** — one bulk DMA of the contiguous complement and one D2D
overlay of the resident range — which has bulk's transaction count and staging's byte saving, and degenerates to
plain bulk at 0 % residency.  It needs the freeze-vs-slab problem from `wip/moe-static-hot-tier/` solved and its
decode hit-rate cost measured (a heat-*seeded* static set already costs ~1.8 points, r74).  The framing still on
the table as an alternative is that what remains is an **arena-sizing policy** (decode hits vs prefill bytes)
rather than a transport rule: the ≥ 83 %-residency cells are not a target, because high residency means the arena
was sized from VRAM the user did not need to offload, and the right answer there is fewer `-ncmoe` layers or a
capped arena.  Parked experiment and all numbers: [`wip/expert-transfer-engine/`](wip/expert-transfer-engine/README.md)
(`0001-no-pinned-staging.patch`: gated 41/41, text byte-identical, +6..16 % at low residency, **-4.6 %** on a
2-device 99 %-residency tensor split = the hole to close first).  **Next work: does a capped arena cost decode?**

**Method (`TODO` #296, r236) — `llama-bench` cannot measure this path.**  Its prompt is `std::rand() % n_vocab`, so
every ubatch routes over every expert and the heat-ranked arena has no signal: pp1450 measures flat at 414.1 /
414.5 / 414.5 / 414.5 t/s across `MOE_EXPERT_CACHE_MIB` 0 / 4096 / 8192 / auto, while the *same* arm on the shipped
prose prompt reads 401.6 / 583.6 / 676.9 / 1063.6.  Any A/B here is driven with real text (`llama-cli -f
prompts/prose-rdna-boosts.txt -n 0`, `scripts/gate-rebase-release.sh --only textprefill`), and the no-locality
floor to beat is unpatched `a691b2345`'s **508 t/s** against the delivery's 414: upstream computes the experts on
the CPU out of local RAM and never crosses the bus, and a uniform per-ubatch split is its best case.  The gap is
transport-shaped, not policy-shaped (it scales with the host-resident layer count and with the ubatch COUNT, not
its size).  Records: `WORKLOG.md` 261108.011, `benchmarks/2026-11-08-host-expert-prefill-transport.md`.

**One sub-case has closed and is NOT what this row is about** (`TODO` #133, r258): an identity (fully resident)
arena is now read IN PLACE at any width (`MOE_EXPERT_PREFILL_IN_PLACE`: 1489.6 -> 4033.9 t/s on 3 GPUs
`-sm tensor`, 3588.8 -> 4072.4 on the pool mirror).  Consolidating the three transports for a **partly resident**
table is the whole of this item.  Record: `WORKLOG.md` 261110.014, `benchmarks/2026-11-10-prefill-in-place.md`.

## 363. `-of` / `--optional-features` has one tag; migrate the rest of the env-only opt-in switches

The tag layer exists (`common/common.cpp`, `common_optional_features()`; documented in `ENVIRONMENT.md` §4.1) and
carries exactly one tag, `tp-ffn-even` (r338, `WORKLOG.md` 261121.017).  Every other opt-in delivery feature is
still reached only by an environment variable, which `--help` cannot show.

**The migration is a table row, not a new mechanism**: `{ "tag", "ENV", "1", "one paragraph" }`, then confirm the
tag produces exactly what the variable produces (the C section of `scripts/gate-ffn-even-split.sh` is the shape:
all spellings must read one text).  Do **not** add a second switch or a second code path; when a feature earns a
first-class option, delete its row instead.

Candidates (all already documented as opt-in in `ENVIRONMENT.md`): `LLAMA_KV_N_PAD_MIN` (§4), the block-14
spill/window budgets (`LLAMA_KV_SPILL_RAM_PCT`, §1.2's `MOE_EXPERT_CACHE_MIB`-adjacent knobs are tuning, not
opt-in -- check the row's class before migrating), and the `GGML_CUDA_*` opt-ins §7 lists as "opt-in, default
OFF".  A tag is only for a feature that is **off by default**; a kill-switch is not a tag.

## 369. Three more gate defaults still name the RETIRED Flash-Next IQ3_XXS tree

**Record:** the repointing work is recorded in `WORKLOG.md` 261122.027 (the r346 change that repointed the first
three gates); this row is the residue of it.

`gate-park-idle-slots.sh`, `gate-moe-promise.sh` and `gate-host-pool.sh` were repointed to the maintainer's
quality-first replacement, `Qwen3.8-Flash-Next/STEW675-IQ4_NL/Qwen3.8-Flash-Next-MTP4-IQ4_NL-00001-of-00002.gguf`
(r346; the IQ3_XXS build is gone from this box).  Three more still name it and were **deliberately not**
repointed in that change, because each carries *recorded text/KLD anchors for the BF16-hc / all-VRAM class* —
a new model is a new class, so a path edit alone would leave a gate that passes nothing:

* `gate-hc-base-copy.sh` (MODEL default),
* `gate-hc-fusions.sh` (`MODEL_ROOT/GSQ-IQ3_XXS/...`, and its header records the measured hashes for that class),
* `gate-host-expert-text.sh` (its header sizes the run at 75.8 GB / ~71 GiB peak RSS).

The work is one change: repoint each to the STEW675 IQ4_NL tree, re-record its anchors **on that class** (the
`--kld`/record paths already exist), and correct the headers' numbers.  Until then those three refuse at the
model check rather than run on a wrong anchor — which is the correct failure, not a defect.

**Not a row, and closed on the spot (2026-11-24):** the v2 campaign's sweep hit one more gate whose default model
was missing from the box, `gate-kv-residency.sh` (`MODEL_POOL` + the `-Q8_0` MTP sidecar its `DRAFT_POOL` derives).
That was an **environment gap, not gate debt** — those names are this gate's own documented shape, its arms are
self-referential (no recorded anchors to re-record), and once the model tree was back the gate passed on its
defaults unchanged.  Recorded in `WORKLOG.md` 261124.003 / `INVARIANTS.md` r380-wip so the next session does not
re-report it.



## 376. On promotion, the block-17 beta artifact's Meta-backend fix must be folded into its OWNING block

**Record:** the defect is `WORKLOG.md` 261124.004/005 (the original reading) and **261124.007** (the fix and the
corrected mechanism), `INVARIANTS.md` **r384-wip** (r383-wip item 4 carried the earlier, wrong reading and is
corrected in place); the work order is `wip/prefill-decode-arms-v2/HANDOVER.md`.  The maintainer's instruction
(2026-11-24) is explicit: the fix goes into the beta candidate **for now**, and at promotion it is folded back
**independently of the two-arm block**.

A 3-device `-sm tensor` split, with the two-arm server on and an MTP drafter, aborted ~6 s into a ~77K-token
prefill:

```
E ggml_backend_cuda_get_tensor_2d_async: device 1: 2-D read of 'result_output' failed (invalid argument):
    331092 bytes x 5 copies, tensor pitch 331092, host pitch 993280, offset 0 (ROCm1 buffer)
```

`host pitch = 3 x tensor pitch` -- a SPLIT 2-D tensor read against one device's slice.  **The split arithmetic is
exact** (per-device chunks 331092 / 331092 / 331096 sum to the meta row 993280 and device 0's copy of the same
arguments succeeded); the refusal is the **peer grant a 2-D D2H copy needs on the destination's pinned
allocation** (`hsa_amd_agents_allow_access ... err 4096`), which this box refuses.  The buggy code is therefore
**delivery** code in `ggml/src/ggml-cuda/ggml-cuda.cu` (the 2-D read entry points), not the Meta backend's chunked
path; the caller was the ordinary per-ubatch logits readback of a 5-row verify batch, which the two-arm decode arm
is what composes.

**Why it must not simply stay in the two-arm block:** block 17 is a beta candidate that may never be promoted, and
this is a defect in blocks 09/13's code.  Leaving the fix only in block 17 means a delivery that ships without the
pair still carries the bug -- and a promoted block 17 would silently modify a lower block's behaviour.

**The work at promotion:** cut the fix against the release tree as a delta on the owning block -- **block 09**
(`patches/0009-rdna-boosts-block-09-*.patch` owns `ggml-cuda.cu`'s 2-D iface and the scheduler; the `ggml-backend-meta.cpp`
reading is refuted and no block-13 delta is needed) -- re-cut that block's patch in place, run
`scripts/check-rung-file-sets.py` for the rung plus `scripts/validate-set.sh` (a block must not shrink), then re-cut
the beta artifact against the new tree so the candidate keeps only what is its own (the STEP-15 hunk becomes a
no-op there).  Record the transition in `WORKLOG.md` and in the beta artifact's README.

## 380. block 17's CUDA (nvcc) build is unverified — the ternary path has only ever been compiled by HIP

Block 17 adds `ggml/src/ggml-cuda/tq.cu` + `tq.cuh` (966 lines net) and one fusor, and it touches seven shared
`ggml-cuda` files.  Everything it does is inside `ggml/src/ggml-cuda/`, so the other backends keep refusing the two
types — but "CRANKS" is not "compiles": the CUDA build of those files is compiled by nvcc, the delivery's CI compiles
nothing at all, and no box in this fleet has an NVIDIA device or a CUDA toolkit.  The code reads nvcc-safe (the
`#pragma clang fp contract(off)` blocks are `defined(__clang__)`-guarded, `TQ_FMUL`/`TQ_FADD` fall back to
`__fmul_rn`/`__fadd_rn` which nvcc never contracts, and `TQ_MMA_*` is behind `AMD_WMMA_AVAILABLE && RDNA4`), and the
same tree builds under HIP/clang — but "reads safe" is what the scope policy explicitly refuses to accept as
evidence for a backend's build (`AGENTS.md`, "Scope policy").

What closes it: a CUDA build of the applied tree (`cmake -DGGML_CUDA=ON`, any recent toolkit), then
`test-backend-ops -o MUL_MAT -p type_a=tq` and `-o GET_ROWS -p type=tq` with `LLAMA_TEST_OP_NMSE=1`.  The claim to
test is **all 305 MUL_MAT and 84 GET_ROWS values reading exactly 0**, not the text: `-use_fast_math` flushes
subnormals, so a CUDA arm can be coherent and still not bit-identical (`GREEDY-PURITY.md` §109), and a nonzero
error with the `k=256`-vs-`k>256` split is the same fused-accumulation diagnosis `gate-tq.sh` prints on HIP.

Left out of r356 deliberately: the delivery is RDNA-first, the maintainer's fleet has no NVIDIA device, and an
`optional/`-style hold would have kept a validated-on-this-hardware win out of the default build for a
platform whose *behaviour* is out of scope anyway.  The build is the one part that is in scope.

## 381. block 17's gfx1100 arm is owed — `fingon` went down during the r356 session

The r356 cross-arch matrix is **gfx1201 (the release box) + gfx1151 (`halo`, and that box builds
`GGML_HIP_NO_VMM`, i.e. gate 40's other configuration)**.  The gfx1100 arm (`fingon`) started twice — the first run
was killed by a reboot that wiped `/tmp` mid-build, the second by a second reboot plus a corrupt tag ref, and then
the host stopped answering on port 22 for the rest of the session.  The delivery's rule is that an unvalidated arch
is a *documented* state, not a blocker (`AGENTS.md`, scope policy), and the specific risk here is low and named:
gfx1100 has **no RDNA4 int8-WMMA kernel**, so it takes exactly the code path gfx1151 already validated
(`mma=0`, the small-batch kernel for every batch, **305/305 and 84/84 NMSE values exactly 0**).

What closes it: the command in the index row, on the r356 tree, and its four results recorded in
`patches/README.md` block 17 (or in a follow-up release entry, whichever comes first).  Do not "close" it by quoting
the gfx1201 numbers.

## Re-entry / does it block promotion?

The two-arm candidate is **not** blocked on this fix for testing (the fd work landed; this defect needs a drafter
on a 3-device tensor split and a big prefill).  It **is** a blocker for folding the pair into the delivery: a
promotion that swallowed a lower block's bug would be the wrong shape.

## 382. block 12's tool-text protection is a WORKAROUND — the renderer's `is_input` parts are the real lever

**Status: PARKED** (r357, `ARCHIVE/work/tool-text/`, `INVARIANTS.md` r357, `MANIFESTS.md` gate 43).  The shipped block is
correct and gated; this row is the *design* debt behind it, recorded so the next session does not re-derive it.

The chat renderer already knows which span is data: `common/jinja` marks every interpolated value with `is_input`
(`common/jinja/string.cpp`, propagated through `lower`/`replace`/… by `mark_input_based_on`, and documented in
`common/chat.h` as *"from user input, special tokens inside must not be parsed"*).  `common_chat_prompt` carries it as
`parts`, and **nothing consumes it for tokenization**: `oaicompat_chat_params_parse()` flattens the prompt with
`prompt().str()` and `handle_completions_impl()` tokenizes that one string with `parse_special = true`.  That
flattening is the defect; the block the delivery ships is a workaround for it (mark the strings with U+FDD0/U+FDD1
before the template, then tokenize each marked string as ordinary text — see `patches/README.md` block 12).

**The real fix is two small changes:**

* a flag in `llama_vocab::tokenize` / `tokenizer_st_partition` meaning *no special-token matching at all*.  Today
  `parse_special = false` skips only CONTROL / UNKNOWN and **still matches USER_DEFINED tokens** (Qwen's `<think>`,
  `<tool_call>`, `<tool_response>`), which is why "just turn parse_special off" does not close the forgery
  (`INVARIANTS.md` r357 item 1);
* tokenize `out_session.prompt().parts` with that flag on every `is_input` part, ordinary parsing otherwise.

**What that buys:** the marks, the second template render (the +1 ms per request `/apply-template` shows), the
`tool_text_marks_intact()` heuristic and the whole space fallback disappear; a request carrying media and a non-BPE
vocab stop needing a fallback at all; and `user` / `system` text is covered too, which the shipped block leaves
forgeable **by design** (`#383`) — a forged turn in a `user` message still returns `PWNED` (measured).

**Why it did not ship now:** it is a tokenizer-API change (a fourth `parse_special` state, or a new argument) touching
`src/llama-vocab.cpp` and the one consumer, with a much wider blast radius than the block — every tokenization call
site is in scope for review — and the maintainer's r357 call was to take the submission's shape into the delivery and
park the redesign.  **Re-entry:** upstream offering a tokenizer-side "no special tokens" switch (then the block
becomes a 20-line call-site change), or a session that is already touching `llama_vocab::tokenize`.

## Accepted limitations (deliberately not fixed — do not re-report)

These are closed or accepted: no work is owed, and no row above tracks them (see *How this file is
organised*).  A handle is kept in the heading where one was ever allocated, so the dated records that cite
it still resolve and a re-entry condition has a name; the ones that never had a handle say `No handle`.

### #282 — the mid-run latch's MIXTURE is a ~1e-5 path difference, and it is gated on the LOGITS (retired r299)

`MOE_HOST_POOL_PIN_FAULT=late` leaves a run with the pools it built before the refusal serving and the rest of the
tier stood down, so some ops take the cache path and the rest the scheduler's.  That mixture is a different fp32
association order, and it flips one near-tie token in a 32-token greedy run on some layouts -- **not wrong bytes**:
measured under the field-shape latch **mean KLD 0.000001 / max 0.000051 / same-top-p 100.000 %** (limits
0.005 / 98 %), while the wholesale arms stay byte-identical to `ref` on every build.  Nothing is owed: the
byte-identical *text* assertion was simply the wrong instrument for a mixed state, so `gate-host-pool.sh` now gates
`pinlate` on the KLD (base recorded in-run from the `ref` config) and keeps the text assertion for the wholesale
arms.  The arm itself was also repaired -- sizing moved to registration, so its `g_sized`-only latch had degenerated
to 1 pool of 141 and 2 served ops, and `late` now waits for the first routed op.  **Re-entry:** a mixture whose KLD
leaves 0.005 / 98 %, or one that produces a wrong answer rather than a different near-tie.  Record:
[`benchmarks/2026-11-16-pinlate-mixture-logits.md`](benchmarks/2026-11-16-pinlate-mixture-logits.md), `WORKLOG.md`
261116.006, `INVARIANTS.md` r299, `GREEDY-PURITY.md` §95.

### #141 — the eager take-over's bound is a CAPPED ARENA x a sub-complement pool (retired r274)

`moe_cache_update_host` declines the whole routed op when a pool-backed table is neither a pinned master
(`table_cold_ok`) nor covered by its pool (`pool_covers_table`) and the arena holds fewer than `n_used * n_tok`
slots.  Measured on r273 (2 devices, 35B-A3B Q4_K_M `-ncmoe 40 -lme 8` = 43.8 % coverage):
`MOE_EXPERT_CACHE_MIB=2048` (53 slots/table) reads **46.1 tg** at `-npl 8` against **102.8** for the cache being
off; `2560` (67 slots) fixes `-npl 8` (**134.2**) but not `-npl 16` (**66.2** vs 149.0 off), and the band top needs
**5120** (134 slots, **289.9**).  The row's own premise was wrong: the sub-complement host budget alone does not
reach the band — at the **default** arena the same `-lme 8` arm is the fastest measured (**298.7 / 384.0**, `0`
declines), because auto sizing clamps each table at `n_experts` (256 here).  Both knobs are needed at once, i.e.
asking too much of too little.

**The answers, in order** (cheaper than a cold-source redesign, and why nothing is owed): `MOE_EXPERT_CACHE_MIB=0`
is 2.2x faster than the declining arm; or `-lme auto` (the covering mirror) / `--host-experts pinned` (exempt by
construction); or recover the capped VRAM with block 14's streaming KV (`--kv-window`, `--kv-attendance-pool`).
Not shipped on purpose: a WARN for the cell (a code change and a gate for a configuration the delivery does not
validate, when `decline_all`/`takeover=` already name it at `-lv 4`).

**Re-entry condition:** a real operator report on a supported configuration, or a campaign that lowers the band's
need on its own merits (a legal cold source for a neither-tier expert — the `#273`/gather side of the same
population).  Record: [`benchmarks/2026-11-12-eager-takeover-arena-bound.md`](benchmarks/2026-11-12-eager-takeover-arena-bound.md);
`WORKLOG.md` 261112.010; `ENVIRONMENT.md` (`MOE_EXPERT_CACHE_MIB`); `INVARIANTS.md` r274.

### #319 / #320 / #321 — the APU store's refusal band, the drafter's head budget, and "the MTP draft is not reproducible" (retired r284)

**#319 (the lock policy's refusal).**  The band it named -- a clamp that covers the RAM shortfall but not one slot
per expert table -- turns out to be **unreachable for any model more than one slot-set over RAM**, and the real
tier minimum is `max(one slot per expert table, the cache's own floor)`; the loader tested the first alone, so a
clamp in between let the arena decline and then reported *"the arena is unavailable on this device"*.  The loader
now asks the backend for the floor (`ggml_backend_dev_gtt_store_arena_min`) and refuses with the number to raise;
the four measured arms on `halo` (refusal at `-lme 1`, refusal under a 1 GiB clamp, and `-lme 3` **running**) are in
`INVARIANTS.md` r284.  Refusing inside the band is **necessary**, and an operator whose clamp is below it has two
documented escapes (raise `ulimit -l`, or `MOE_EXPERT_GTT_MLOCK=0`, which bypasses the lock policy so the arena is
reclaimable rather than reserved).  Nothing is owed.

**#320 (sizing the store for a separate drafter's head).**  Parked by the maintainer (r263) and **retired
unimplemented**: the head is pinned *outside* the `-lme` budget on purpose, its bytes are metadata overhead read on
every draft step, so a store sized to include it would trade that away; making it budget-visible is a `common`/
`--fit` term with no measurement behind it and nothing depending on it.  No course of action, so no item.

**#321 ("an MTP draft over 2 devices is not reproducible").**  The attribution was wrong: it is **not** the draft
path and not a race.  A run's greedy text depends on the **expert cache's residency**, i.e. on the persisted
`.moe-heatmap` prior: with the drafter model *loaded* (not used) on a 2-device `-sm layer` split, one prior state
gives `7ca535e1da91` and another `7a3474cbe5f1`, each reproducible within its own state, and
`MOE_EXPERT_PREFILL_IN_PLACE=0` collapses the two (the r258/r266 in-place arm serves a fully resident table from the
arena and stands its fusions down, so residency selects the kernel set).  The consequence is a *comparison* rule, not
a driver fix: a text comparison across processes pins the cache state, exactly as it pins
`ROCBLAS_USE_HIPBLASLT=0`.  `scripts/gate-kv-residency.sh` now forces `MOE_EXPERT_HEATMAP=load` and its multi-device
section is green on the same assertions (`b08d3dbcb1e9` x3, per-device counters intact).  **Still open on the record,
and deliberately not an item:** the two arms are documented as reading the same bytes, so their *magnitude* is
unmeasured (`llama-perplexity` cannot load a drafter, which is the state that produces the partial residency);
re-entry is a prefill-logit arm that can, and a measurement above the 0.005 tolerance.  The chosen way to remove the
coupling -- teach the prefill fusions the in-place contract -- was implemented in r297 and **refuted**: those
fusions cannot fire for a cache-managed table at all, so the coupling was never theirs (`INVARIANTS.md` r297).
**CLOSED r298:** what r297 then blamed -- the in-place prefill *read* (`TODO` #335, #336) -- was a **stale alias**
resolving a sibling ROLE's arena, and the read itself measures KLD 0.000000 / top-p 100.000 % against the staged arm.
The comparison rule stands (pin the prior); the residency-to-text coupling it was written for does not.
Record: [`benchmarks/2026-11-16-in-place-read-alias-role.md`](benchmarks/2026-11-16-in-place-read-alias-role.md),
[`benchmarks/2026-11-14-residency-reproducibility.md`](benchmarks/2026-11-14-residency-reproducibility.md),
`INVARIANTS.md` r284 + r298, `GREEDY-PURITY.md` §89 (corrected) + §94.

### #300 — the attendance pool is NOT supported on a shared host/device pool (an APU) (retired r264)

The pool buys device memory back by keeping the authoritative KV in a **pinned host mirror**, so it only has a
meaning where the host and the device are **separate tiers**.  On an APU they are one: the KV is already in the
memory the GPU reads, there is nothing to buy back, and the two-tier resolve has no model to be correct against
(measured 2026-11-08: a *fully resident* pool still reported ~93 M byte mismatches against its own mirror, and each
run gave a different text).  The cache therefore refuses the request with **one** WARN and leaves the KV as it is;
the resident **window** (`--kv-window`) and the sparse decode arm *are* supported there and validated on gfx1151.
This is not a defect to re-report.  **Re-entry:** a *discrete* device, or a future tier whose mirror really is a
second memory pool.  Record: `WORKLOG.md` 261111.033, `INVARIANTS.md` r264, `scripts/gate-kv-residency.sh`.

### #275 — a prefill-dominated multi-slot mix (`-npl 4`, `-npl 8`) is slower with `MOE_HOST_POOL_RANK=decode` (r224)

The decode ranking is a **single-user** win and its sign inverts under `llama-batched-bench`'s prefill-dominated
mix: on the frozen real-world prior at `-lme 36`, `-npl 4` prefill is 1144.69 t/s against 1292.88 for `combined`
(**-11.5 %**), and one `-npl 8` rep measured -21 % in total (memory was not the cause: `MemAvailable` stayed at
40 GiB).  Accepted because a single-user stream is the common case, and the workaround is documented rather than
hidden: **`MOE_HOST_POOL_RANK=combined`** restores the pre-r224 ranking while keeping the demote, and that is the
setting for a deployment whose traffic is parallel slots.  The demote itself is safe in every cell measured
(+5.3 % single-stream, +1.8 / -0.9 / +0.3 % at npl 1/4/8).  Record: `WORKLOG.md` 261106.012 + the r224 entry,
`archive/work/host-expert-split/HANDOVER.md` §4.3, `ENVIRONMENT.md` (`MOE_HOST_POOL_RANK`).

### No handle — the Windows pinned-host ceiling has no cap, only a measurement (r223)

`hipHostAlloc` there is budgeted by WDDM against the adapter's shared (aperture) segment, and the published model
for it — roughly `(total RAM / 2) / visible HIP devices` (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339) — is
**refuted as a predictor by the reporter's own numbers**: GPU shared usage stops at 9.32-9.35 GiB on a 47 GB box
and again on the same box at 64 GB, with one visible adapter and ~60 GB of commit headroom (`WORKLOG.md`
261106.016).  A ceiling derived from the model was implemented and **deliberately withdrawn**; r227 ships the
measurement instead — the exit census prints the pinned tier's realized total, and on Windows the aperture
segments' `CommitLimit`/`BytesResident` for every process (`ggml_hip_win_shared_mem_info`) — plus a loud, latching
refusal for a pool allocation the driver will not give.  The shipped guard is mechanism-independent: the first
refused `cudaHostAlloc` logs once and latches the pinned tier off, and no new pool is attempted after it.  The old
`TODO` #95 pointer (the reporter's r227 log) has left this file; **re-entry is that log.**

### No handle — block 15's GTT tier does not reach a multi-device split, or Windows, or a discrete box

Three limits of the same tier, recorded together because they are one hardware story:

* **Multi-device is untested, not merely unclaimed.**  The per-device budget, the per-device head subtraction and
  the free-bytes query have never seen a `-sm tensor` / `-sm layer` split across APUs, and the prefill-stream hook
  declines by geometry on a slice (by construction, not by measurement).  Every box in this repo's validation set
  is one APU or discrete cards, so the gap cannot be closed here.  **Re-entry: a multi-APU host.**
* **Windows: the tier declines by design.**  The single-address premise does not hold on WDDM, so the run keeps
  the previous behaviour (one WARN).  Nothing forecloses an arm later — the platform part is one `mmap`+`register`
  function — and that is the re-entry.
* **The `llama-batched-bench -npl 1,4,8` width arm is NOT owed for this tier.**  It was owed for a discrete box,
  but the store/arena is **inert on a discrete GPU by measurement** (forced-on envs produce zero `GTT` lines:
  `plan_gtt_expert_store()` is only reached from the UMA branch, `gtt_store_open()` is never called, and
  `moe_cache_prefill_stream()` declines on `!g_gtt_arena`), so that arm cannot witness this tier there.  The width
  arm's own subject is decode width (blocks 12/14) and it lives in the MTP protocol, not here.  Record:
  `archive/work/apu-expert-gtt/MEASUREMENTS.md` §27.1 and §28.

### #343 — Q5_K / Q6_K / IQ4_XS `ssm_alpha` projections are on the r304 path with no local GGUF to test it

### No handle — a merged-layout (`hc_*_down_inject`) qwen4exp GGUF with Q8_0 hc weights is untested (r313)

**Accepted limitation** (r313, contributor PR #326).  Block 11's Q8_0 hyper-connection mixer covers the
**separate-tensor** UD layout (`hc_attn_down`/`hc_attn_up` Q8_0, inject F32 or Q8_0), which is what every UD file
on this box is and what `scripts/gate-hc-fusions.sh` and `test-backend-ops` were run against (`UD-Q4_K_XL`,
`test-backend-ops -o HC_COMBINE_MIX` 13/13 with the Q8_0 arms).  A **post-#31028 merged** file (`hc_*_down_inject`
as ONE tensor) takes the same kernels through two `ggml_view_2d` row views of the merged tensor, and the kernels
address every weight row packed at `ggml_row_size(ne[0])`, which a row view preserves -- so the path is
structurally identical to the separate-tensor one -- but **no such GGUF exists here or at the contributor's**, and
the synthetic merged-HC fixture (`tests/test-llama-archs.cpp`, the r303 arm) is F32 with `hc_lr 8`, so it never
reaches the fused op at all.  What would close it: any merged-layout UD flash-next GGUF whose hc weights are Q8_0
(`scripts/gate-hc-fusions.sh MODEL.gguf`, whose class line then says `FUSED-capable ... merged`, with
`SOFT_EXPECT=""`), or a fixture that can emit Q8_0 rows for the merged tensor and pass the fused gating
(`hc == 4`, `hc_lr == 320`).  `LLAMA_FUSED_HC_MIX=0` is the reference for anyone holding such a file; the same
gap is recorded in `archive/work/hc-mix-q8/README.md` and in the PR #326 reply, so it is a stated limitation, not
a claim.

**Accepted limitation** (r304).  Block 10's GDN gate/beta epilogue takes `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS` and
`MXFP4` (`mmvq_gdn_epi_type`), but neither this box nor the contributor's has a GDN GGUF whose
`ssm_alpha`/`ssm_beta` are `Q5_K`, `Q6_K` or `IQ4_XS`, so those three quant types run the same code path
*untested* by anything: the class survey in `DIAGNOSTIC.md` §7.1 covers Q8_0 (most files), F32 (the UD and
`Q6_K` files), `Q4_K` (lmstudio-community's 35B-A3B Q4_K_M) and `IQ4_NL` (which the epilogue declines by type,
so it is the reorder-only control).  What would close it: any GDN-architecture GGUF quantized by llama.cpp's own
`llama-quantize` from a UD-Q4_K_* source with `--token-embedding-type`/default rules (a `Q6_K` requant is the
cheapest), then `scripts/gate-gdn-epi.sh MODEL_Q=<that file>` — the gate's mechanism check is what would catch a
wrong `qk`/`vdr` in the ksplit body for that type (the arithmetic is the body's, shared with `Q4_K`, and the
per-row dot product is unchanged by construction).  Do not add a KLD arm for it before the mechanism line is
observed: an arm that never fires passes vacuously.  The same gap is recorded in the r304 release entry and in
the PR #322 reply, so it is a stated limitation, not a claim.

### No handle — TQ1_0 / TQ2_0 (`MUL_MAT_ID`, draft decoding, a non-AVX2 CPU reference)

Block 17's `MUL_MAT_ID` is **not implemented**: a TQ (ternary) MoE keeps its expert tables on the CPU backend by
design, so such a model runs half on the GPU.  Draft/MTP decoding with a TQ model is untested (a speculator's TQ
weight would go through the same `MUL_MAT`, but no TQ model with an MTP head exists here), and the block's
bit-identity is stated against the **x86 AVX2** CPU path only — an ARM build accumulates in another order, so the
oracle itself does not exist there.  Re-entry: a TQ MoE GGUF (then `MUL_MAT_ID` is a feature, with the same
`gate-tq.sh` oracle extended to the routed form), a TQ model with a draft head, or an ARM box whose
`test-backend-ops` TQ run can be compared.  `scripts/gate-tq.sh` reports the first two as not covered rather than
skipping them silently, and `patches/README.md` block 17 names all three.

### #383 — the tool-text protection does not cover `user` / `system` text, meaning-level injection, or the realtime endpoint (r357)

**Accepted limitation** (r357, contributor PR #360; block 12's r357 amendment).  The block protects **by message role**:
`tool` / `function` content, an assistant's `tool_calls[].arguments`, and an assistant's own content/reasoning (the
CONTROL / UNKNOWN strings only there).  User and system text is deliberately untouched, so a *client* that returns tool
output in a `user` message — the text tool protocols, a late result, a sub-agent's summary, an environment dump — is
unprotected: measured on the delivery build, a forged turn placed in a `user` message still returns `PWNED`
(`archive/work/tool-text/README.md`).  Also not stopped: **meaning-level injection** (with the protection on, a forged
"second tool result" becomes plain words inside the one real tool result and the model still follows them), and
`server-realtime.cpp`, which builds its own chat session and never reaches `oaicompat_chat_params_parse()`.

A **request that carries media** and a **non-BPE vocab** take the space fallback, which keeps the strings from becoming
tokens by breaking them (`< |im_end|>` — the model reads a space the client did not send); that is a documented
degradation, now scoped to requests that really carry media (`INVARIANTS.md` r357 item 4).  A single-character
protected string is also unsplittable by either mechanism — no vocab in fleet has one (checked with
`llama_vocab_get_attr`: Qwen3.8 0/33, Gemma4 0/24, a Llama-3 vocab 0/256).

**Nothing is owed here**: the role-scoped rule is what the block claims, the gate asserts it, and the complete answer is
the parked redesign (`#382`) — tokenizing the prompt's `is_input` parts — which covers `user`/`system` for free.  The
re-entry for the *fallback* arms is a client-side escape on a media request, or a tokenizer-side switch; the re-entry for
the realtime endpoint is that endpoint gaining a chat-template path of its own.  Record: `patches/README.md` block 12,
`MANIFESTS.md` gate 43, `DIAGNOSTIC.md` §26, `archive/work/tool-text/`.
