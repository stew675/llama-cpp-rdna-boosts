# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
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
truth, not this line.  **Block 17 is the two-arm server** (a prefill arm and a decode arm co-owning one KV,
`--prefill-decode-arms`, **opt-in, default OFF**, r346), and its follow-ups are the `#365`-`#372` rows below,
written up in `wip/prefill-decode-arms-followups/`.  Block 14 is the **streamed KV** (the resident window, the
attendance pool and the MTP draft window -- all three opt-in, default off); block 15 is the UMA GTT expert
store.  The **mixed K/V cache types** are **optional block `0101`** since the r288 block-number re-org (they
were numbered block 17 in r287, PR #312 -- a different block from today's `0017`; the one native pair, head 256
only), and `0100` is two-lane prefill.  `release.json` is the source of truth for the tip/tree, counts and hashes; the top of `WORKLOG.md`
is the newest record.  **Block `0300` is temporary** (item #104).
The r278 re-base (`archive/work/upstream-rebase-261113/`, base `4f9dfa0e3`) is **cut and released into
`patches/`**: its 17 blocks are replayed, both rung-repairs are squashed into their owning blocks, and the gates
are recorded in `WORKLOG.md` 261113.015.  What it left open -- #322 (a stale `gate-server-arms` baseline), #323 (the
prefill-logit base file name), #324 (both `optional/` artifacts stale), #325 (the APU-only gate set) and #326 (the
loader's async-upload lifetime) -- closed at **r285** (`WORKLOG.md` 261114.006), which is why nothing about them is
left in this file.  The r236 re-base
(`archive/work/upstream-rebase-261108/`) is superseded and closed.

## The open list

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

**`r346` housekeeping (2026-11-22):** eight rows were opened for the follow-ups of **block 17** (the two-arm
server, `v18-919794500-r346`): `#364` (speculative decoding with the arms), `#365` (a meta-aware device copy),
`#366` (a derived decode reserve), `#367` (GLM5-Next), `#368` (a recurrent-only memory), `#369` (three gate
defaults still anchored to the retired Flash-Next model), `#370` (the pair's slot erase and cell accounting) and
`#371` (the prefill/decode speed balance, the maintainer's turn-skip idea).

**`r350` housekeeping (2026-11-22):** `#364` closed and left the list -- **implemented, not deferred**: a
speculator now runs ON the two-arm server (`draft-mtp`, `draft-mtp-adaptive`, `draft-simple` and the n-gram
family; the layer-input drafters and a gemma4 assistant are still refused, each because its target side is bound
to one context), the drafter reads the target's hidden rows from the context that ran the batch, the decode arm is
sized for the verify width, and the slab reserve is the sum of the decode arm's and the draft context's.  The
oracle is `scripts/gate-arms-spec.sh` (MANIFESTS gate 39): a **three-way cold** text claim
(`pair+draft-mtp` == `pair+none` == `single+draft-mtp`) at 3 devices f16 / 3 devices q8_0 / 2 devices f16, plus
the adaptive controller, the four-axis protocol on the pair and a concurrency arm -- **32/32**, and every arm pins
`LLAMA_SERVER_PARK_IDLE_SLOTS=0` because the co-owner guard declines parking on the pair while the single context
parks by default (`GREEDY-PURITY.md` §107).  `WORKLOG.md` 261122.040; campaign `wip/prefill-decode-arms-followups/`
§W1 is now DONE.

**`r347` housekeeping (2026-11-22):** `#365` and `#366` are **closed** (both shipped in the block-17 amendment:
the Meta buffer's per-child `cpy_tensor_n` and the width-keyed decode reserve — `WORKLOG.md` 261122.036), and the
crash row filed as `#373` is **closed** too (the five shared-state defects shipped with the same amendment; the
owed gate arm is `gate-arms-soak.sh boundary`).  `#374` (the vacuous `pool` arm) stays **open**.  Each has its detail section **before**
the accepted limitations (they were briefly appended after it, which read as if they were accepted limitations; they
are open work) and they are written up in one campaign, `wip/prefill-decode-arms-followups/`, as `W1`-`W7` plus
`G1`.  `#363` was in the same wrong place and moved up with them.

| # | subject | kind | status |
|---|---|---|---|
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM | performance / new type (`wip/fp8-support/`) | **PARKED** (2026-09-26) — builds on gfx1201, gate 2 fails (0.69x Q8_0) — §116 |
| **273** | three prefill transports where one should do | architecture (delivery) | **OPEN** — the host-set SIZE is the primary lever, residency secondary; ONE transport for a partly resident table — §273 |
| **361** | the r334 RDNA2 items (PR #338) are validated on the contributor's V620, not on the released tree: blocks 05/08/10 now carry an RDNA2-only FA tile-cap removal, a one-warp narrow `RMS_NORM`, a `H * n_seqs` GDN chunked gate and a rows-abreast MoE K walk, and no RDNA2 device is in the maintainer's fleet | validation / external (delivery) | **WAITING on `@jstamagal`** -- re-run on the r334 tree: full `test-backend-ops` on `gfx1030` at default flags (**especially `FLASH_ATTN_EXT` with the caps gone: the failure mode is `GGML_ASSERT(max_blocks_per_sm > 0)` in `launch_fattn`, not a rate**), `MUL_MAT_ID` with `GGML_CUDA_MOE_ROWS_ABREAST` on and off, `RMS_NORM*` in the default/`NARROW=0`/`VEC=1` arms, a model-level decode+prefill (a GDN model for the chunked gate, a MoE for the K walk) and an MTP-acceptance run; **it should also confirm that PR #339's BF16 `RMS_NORM + MUL (+ ADD)` abort is gone there** (fixed at r328 by the re-base, `archive/work/can-fuse-bf16-rms-norm-r324/` -- `RMS_NORM_MUL_ADD` now includes `type=bf16,type_w=f32` and `type=bf16,type_w=f32,type_add=bf16` arms); §361 |
| **363** | `-of`/`--optional-features` exists with ONE tag (`tp-ffn-even`); the delivery's other env-only opt-in features are still environment-only, so a user who reads `--help` still cannot find them | tooling / docs (delivery) | **OPEN** -- migrate them one at a time: add the row to `common_optional_features()` with its switch, confirm the tag produces exactly what the env var produces (the arm `gate-ffn-even-split.sh` C-section runs), and delete the row if the feature gets a first-class option; candidates are the opt-in switches `ENVIRONMENT.md` already marks as such (§4 `LLAMA_KV_N_PAD_MIN`, §7.6 `--wmma-verify`-adjacent switches are options already, §1.2's arena knobs, block 14's `LLAMA_KV_SPILL_RAM_PCT`); §363 |
| **367** | `GLM5-Next` is unvalidated on the pair: it uses the same `llama_memory_hybrid_idx` as Qwen3.8-Flash-Next (the module whose missing shared view was block 17's P0), but no checkpoint exists on this box | validation / external (delivery, block 17) | **OPEN (waiting on a model)** -- one coverage-matrix row: both arms, layer and tensor, 2 and 3 GPU, identical text, and no `implements no shared view` warning. Campaign §W4 |
| **368** | a **recurrent-only** memory (Mamba class, no attention KV at all) gets **no state** on the second arm: `llama_memory_seq_state_take()` refuses anything that is not a hybrid | capability (delivery, block 17) | **PARKED (no model on this box)** -- the same take with the recurrent half as the whole state; implementing it without a model would be a claim with no oracle. Campaign §W5 |
| **369** | three gates still name the RETIRED Flash-Next `GSQ-IQ3_XXS` tree (`gate-hc-base-copy`, `gate-hc-fusions`, `gate-host-expert-text`) and were deliberately NOT repointed at r346, because each carries recorded text/KLD anchors for that class | tooling (delivery) | **OPEN** -- repoint each AND re-record its anchors on the new class (the STEW675 IQ4_NL tree) in the same change, then run each once. Campaign §G1 |
| **370** | the pair's cell accounting after `POST /slots/N?action=erase`: with three resident prompts the erases release about **one** prompt's cells (used 8174 -> 5446 of 8192), and parking is declined by the co-owner guard so idle cells legitimately stay resident | correctness / accounting (delivery, block 17) | **OPEN** -- decide whether a slot whose saved state is erased should release all its cells, then turn the recorded number into an assertion in `scripts/gate-arms-soak.sh`. Campaign §W7 |
| **374** | the gate's **`pool` arm is VACUOUS**: `gate-arms-soak.sh pool` fails its own non-vacuity check (*no expert-access/teardown census*) with the 4B model on a 1-GPU host tier, both with the derived pair reserve and with `MOE_EXPERT_CACHE_AUX_RESERVE_MIB=3584` forced | **gate debt (block 17)** | **OPEN (found 2026-11-22)** -- the other six checks pass (`decline_all=0`, `broken=0 repaired=0`, text equals the single context's, no GPU faults), so the arm currently proves nothing about the pair's host-expert path.  Repair the census (or the arm's config, e.g. a `-ncmoe` value that really routes through the cache at this model size) before relying on it.  `WORKLOG.md` 261122.035 |
| **371** | the prefill/decode **speed balance** is untuned: the pair trades -21..-24 % prefill for a decoder that keeps moving, and the worst in-window gap is 95.5 / 86.5 / 60.5 / 32.8 ms on the recorded shapes; the idea is for the prefill arm to **skip one or more submission opportunities** while a decoder is active, so decode runs a burst of uninterrupted steps | tuning (delivery, block 17) | **OPEN** -- a measurement campaign, not a patch: the gap is four to twenty times SMALLER than the prefill arm's own step, so the mechanism must be identified first, and the deliverable is a Pareto curve for K in {0,1,2,4,8} with a **refutation acceptable** (bound stated up front: >= 50 % fewer worst-gap ms for <= 3 pp prefill). Campaign §W6 |
| **372** | multi-slot prefill batching: with several slots prefilling, does each slot really re-scan the full weight set per chunk, and can a "layer-major" schedule across slots batch them for better prefill throughput? | tuning / architecture (delivery, block 17) | **OPEN** -- the premise needs checking first: `share_prompt_budget()` already puts **every** active prompt into one step and one ubatch is one graph, so co-batchable slots already share the weight pass and the real cost is the **ubatch SPLIT** (which `-ub` moves: 1440 -> 3041 -> 3884 t/s at 512/2048/4096).  First task is a composition census (`LLAMA_KV_DEBUG_CELLS`), then a two-slot aggregate-scaling table.  **Do NOT merge the two arms' layer passes**: that is the shared step the feature exists to remove.  Campaign §W8 |


## Handle numbering

**New handles append after the highest issued so far, so the next free handle is `#374` (`#364`-`#372` were issued
for the block-17 follow-ups at r346); a number is never reused.**  `#NN` is a permanent handle: a `WORKLOG.md` entry that cites `TODO #74` still means the same
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

## 367. `GLM5-Next` on the two-arm server (block 17)

**Campaign:** `wip/prefill-decode-arms-followups/` §W4. — same memory module, no checkpoint on this box

`GLM5-Next` uses the same `llama_memory_hybrid_idx` memory as qwen4exp (attention KV + recurrent state + lightning
indexer KV), i.e. exactly the module whose missing shared view was block 17's P0.  The fix is module-level, so it
should cover GLM5-Next by construction — but "should by construction" is not a measurement, and no GLM5-Next
checkpoint exists on this box.  Whoever has one owes the coverage matrix's row: both arms, layer and tensor split,
2 and 3 GPUs, text identical (and the `implements no shared view` warning absent from the log).

## 368. The recurrent-ONLY memory has no handover (block 17)

**Campaign:** `wip/prefill-decode-arms-followups/` §W5. — and no such model here to measure it

`llama_memory_seq_state_take()` refuses anything that is not a **hybrid**: it moves the recurrent half of a hybrid
memory and leaves the attention cells alone (they are shared).  A pure recurrent model (a Mamba-class
architecture with no attention KV at all) would therefore get **no state at all** on the second arm, so the pair
would either refuse it or silently run it on an empty state.  This is a gap in the take, not in the sharing — there
is no attention KV to share.  No model of that class exists on this box, so the row is parked until one does;
the correct shape is the same take with the recurrent half as the whole state.

## 369. Three more gate defaults still name the RETIRED Flash-Next IQ3_XXS tree

**Campaign:** `wip/prefill-decode-arms-followups/` §G1. — repoint them **and** re-anchor, together

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

## 370. The pair's KV cells after `POST /slots/N?action=erase` (block 17)

**Campaign:** `wip/prefill-decode-arms-followups/` §W7.

Measured with `scripts/gate-arms-soak.sh soak` (4B, `-np 3 -c 8192`, six rounds of distinct ~2600-token prompts): the cache
fills to **8174 of 8192** cells and stays bounded (no leak -- the server keeps answering), and after erasing all three slots
the next tiny request's ubatch reads **used=5446**, i.e. the erases released about **one** resident prompt's cells, not three.
Two things are true and only one is a defect: `?action=erase` erases a slot's **saved state** (block 12's slot cache), not its
KV cells, and on the pair the idle slots' cells legitimately stay resident because **parking is declined** by the co-owner
guard.  What is open is whether the pair *should* release all three (a slot whose state is erased has no prompt to keep), and
what the accounting should read afterwards.  The gate records the number rather than asserting it; turning it into an
assertion is this item.

## 371. The prefill/decode SPEED BALANCE (block 17): the prefill arm SKIPS turns while a decoder is active

**Campaign:** `wip/prefill-decode-arms-followups/` §W6, which carries the mechanism table, the design constraints, the
knob shape and the measurement plan.  This entry is the handle; that document is the work.

**The idea (the maintainer's, 2026-11-22):** while the decode arm has pending work, the prefill arm **skips one or more
submission opportunities**, so the decoder runs several uninterrupted steps (a burst of `K` tokens at its own
uncontended step time) before the prefill arm grabs the device again.  It is a cross-arm **submission** policy at the
batch level, which is why it is not one of the three measured losers (a mid-gather yield, stream priorities, or block
16's duty cycle, all of which are inert or negative on this path: duty measured 50 -> 50 tokens, 1.00x).

**Why it is a measurement campaign and not a patch:** on the shapes where both are recorded the decoder's worst gap is
**four to twenty times smaller than the prefill arm's own step**: 86.5 ms against 771 ms p50 at `-ub 2048`, 60.5 against
1227 ms at `-ub 4096`, 32.8 against 339 ms fully resident (and 95.5 ms at `-ub 512`, whose step series was not
recorded).  So the naive model "the decoder waits for a whole ubatch" is already contradicted by the record.  The gap's actual composition
has to be identified (device time-slicing, internal chunking, or harness step granularity) before a policy is written,
or the patch will optimise a component that is not there.

**Hard constraints:** counters not clocks (a time-keyed policy is not reproducible); it changes WHEN the decode arm runs,
never WHAT it batches (batch composition is a numerics carrier, `GREEDY-PURITY.md` §36/§63); never skip with nothing
decoding and never on a request's first prefill step (pure TTFT loss); bound the streak so prefill cannot starve;
`--prefill-decode-arms-skip N` defaults to **0 = off**; `-ib` and `LLAMA_SERVER_PREFILL_DUTY` stay inert on this path.

**Done =** a Pareto curve for `K` in {0, 1, 2, 4, 8} (prefill cost against worst in-window gap) with purity green at
every `K`, and a stated bound accepted up front (**>= 50 % fewer worst-gap milliseconds for <= 3 percentage points of
extra prefill**).  **A flat or dominated curve is a legitimate result**: record it here and in `GREEDY-PURITY.md`, and
retire this handle rather than shipping a knob with no win.


## 372. Multi-slot prefill batching: does each prefilling slot really re-scan the weight set, and would a layer-major schedule help?

**Campaign:** `wip/prefill-decode-arms-followups/` §W8 (the premise check with code references, where the idea has teeth,
the measurement plan and the acceptance).

**The question (the maintainer's, 2026-11-22):** with several slots prefilling concurrently, "each slot scans through
the full set of layer weights per ubatch chunk"; what if the prefilling were processed layer by layer with each active
slot handling its ubatch for the active layer, so the slots share the weight read?

**The premise is mostly already the mechanism -- since r300, and because of block 16.**  Before r300 the batch was filled
**in slot index order**, so the lowest slot took the whole window and every other prompt waited for its entire prompt (the
fairness gate records 1.91 s for a 7-token prompt behind a 13204-token one): a prefill step carried ONE sequence, so every
step re-read the whole weight set for that one slot's chunk.  The **fair share** is what made the multi-slot prefill batch
exist.  Today `share_prompt_budget()` gives **every** active prompt a share of the step's window ("nobody fills n_batch
first and every active prompt is admitted to every step"), the resulting batch is split into `n_ubatch` chunks, and **each
chunk is one graph**, so the linear ops read the weights once for all slots' rows in that chunk.  The shortest-prompt
**priority droop** (`LLAMA_SERVER_PREFILL_SKEW`) is a live knob but **default 1 = the plain share**, i.e. off: its job is
now "who gets the larger share of one window" (a TTFT-vs-throughput tuning), not "who gets to run".  What is left, and is the real cost, is the **split**: a step whose admitted tokens exceed
`n_ubatch` becomes several chunks, and each re-reads the full weight set.  That is what `-ub` moves, and it is measured:
the pair's prefill reads **1440 -> 3041 -> 3884 t/s** at `-ub` 512 / 2048 / 4096, with 8192 buying only +8 % and a
worse decoder tail.

**Where the idea still has teeth** (and this is the work): (a) the fair share *admits* every prompt but does not make
them *arrive together*, so the composition of a real ubatch has to be censused before any scheduler change is designed;
(b) a host-expert MoE amortises the **union** of routed experts over the rows in the chunk, which is the one place
multi-slot batching is superlinear; (c) a cross-chunk layer-major schedule degenerates into a bigger ubatch, because it
needs several chunks' activations live at once.

**What must NOT be done:** merging the two **arms'** layer passes (decode rows plus prefill rows in one weight read)
would genuinely halve the weight traffic, and would also put the decode row's latency back inside the prefill layer's
work, which is exactly the 300-880 ms stall block 17 was built to delete.

**Done =** a composition census, an aggregate-scaling table for a two-slot prefill at four ubatches on a dense and a
host-expert MoE, and either a scheduler change that keeps more slots in the same ubatch (purity green) or a recorded
conclusion that the `-ub` knob already captures the available gain.  A refutation is a legitimate result.

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
*untested* by anything: the class survey in `ENVIRONMENT.md` §7.1 covers Q8_0 (most files), F32 (the UD and
`Q6_K` files), `Q4_K` (lmstudio-community's 35B-A3B Q4_K_M) and `IQ4_NL` (which the epilogue declines by type,
so it is the reorder-only control).  What would close it: any GDN-architecture GGUF quantized by llama.cpp's own
`llama-quantize` from a UD-Q4_K_* source with `--token-embedding-type`/default rules (a `Q6_K` requant is the
cheapest), then `scripts/gate-gdn-epi.sh MODEL_Q=<that file>` — the gate's mechanism check is what would catch a
wrong `qk`/`vdr` in the ksplit body for that type (the arithmetic is the body's, shared with `Q4_K`, and the
per-row dot product is unchanged by construction).  Do not add a KLD arm for it before the mechanism line is
observed: an arm that never fires passes vacuously.  The same gap is recorded in the r304 release entry and in
the PR #322 reply, so it is a stated limitation, not a claim.
