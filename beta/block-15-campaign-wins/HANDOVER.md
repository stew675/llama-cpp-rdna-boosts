# Block 0015 handover — campaign wins → beta delivery (+ upstream candidates)

**Written 2026-09-10, for the next session.**  Read in this order:

1. this file (the plan + the decisions + the state),
2. `README.md` in this directory (staging plan: inventory, gate audit, validation protocol, reference
   numbers),
3. `../../AGENTS.md` (repo rules — they override everything here),
4. `../../upstream/README.md` (the maintainer's PR-candidate list, with the two entries added here),
5. `../../wip/arch-independent-memory/DERIVED-MASK-DESIGN.md` (the design for the *future* wins
   V2/V3/V4 — not part of Block 15 unless they land first).

## 0. Decisions the maintainer took on 2026-09-10

| # | decision |
|---|---|
| D1 | `upstream/README.md` is the maintainer's own PR backlog.  Entries only need to be **self-sufficient enough for a fresh session to regenerate/pick them up later** — ggml-org permits one PR at a time and a PR has already been pending for four weeks, so a backlog is expected and acceptable.  (No STATUS.md machinery.) |
| D2 | **W4's gate is still undecided** — the maintainer asked for the dilemma to be explained (see §2). |
| D3 | **Add kill-switches for W1 (`GGML_QSA_SCORE_MEM`) and W3 (`LLAMA_QSA_KEYS_ONLY`)** — approved. |
| D4 | Merging the wins into one patch and validating them **as a combined set is a full session of work** — approved; it is the centrepiece of the campaign. |
| D5 | **Exactly one block: Block 0015.  No Block 16.**  Later campaign wins (V2/V3/V4) get folded into Block 15 as dated amendments, the same practice as the block-13/14 amendments.  Rationale: the patch count is already a maintenance burden at 14; it must not grow past 15. |
| D6 | **Prepare the two extra upstream candidates** (the keys-only dead-V removal and the `attn_k` null-mask guard) in the `upstream/` style (D1). |

Implication of D5 to settle early next session: Block 15 should be **cut with W1–W4 as soon as they are
merged and gated** (that starts the beta clock on a fixed artifact), and V2/V3/V4 are then folded in as
amendments to that block if/when they land.  The alternative (wait for V3 before cutting) delays beta
by multiple sessions.  Flag this for one-line confirmation before cutting.

## 1. Where the campaign stands

Four validated wins, all currently under `wip/` (nothing in `patches/` yet):

| win | what | source | measured effect (qwen4exp, ctx 204800, ub 2048, q8_0 KV, 3× R9700) |
|---|---|---|---|
| **W1** | L2 score-chain memory: relu before the 4-D reshape + `n_blocks`-chunked assembly with `ggml_concat` | `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch` (1 file) | compute reserve **6690.40 → 4450.40 MiB/GPU** (ub1024 3346.50 → 2274.35, ub512 1724.56 → 1188.56); bit-identical |
| **W2** | L1 derived QSA block bias + derived visibility, the **mask prune**, and the input-fill null guards (incl. the `attn_k` one) | `wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch` (10 files, +557/−80, base = W1) | compute **4450.40 → 3251.39**, host **1262.70 → 63.69** MiB (ub1024 1675.33/33.64); coherence byte-identical; MTP 0.61616 |
| **W3** | keys-only QSA indexer cache (the indexer V buffer is never read) | `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch` (3 files, 62 lines) | indexer KV 956.25 → **318.75 MiB**; box 88.58 → **86.70 GiB**; perf parity; broad validation matrix |
| **W4** | ggml-alloc: release view sources whose views are never consumed (the 3b leak) | `wip/arch-independent-memory/patches/0001-ggml-alloc-release-unused-view-sources.patch` (+35, `ggml-alloc.c`) — PR copy in `upstream/` | repro 56.00 → **16.00 MiB**; no change on current models (latent trap); **upstream-applicable, clean on master `9cf3bf256`** |

Not in Block 15: **3a** (through-view reuse — measured zero reserve win on the 27B/4B), and the
**future wins** V2 (1-bit packed mask), V3 (derived mask for dense models), V4 (native quantized K/V in
the MMA FA path) — all specced in `DERIVED-MASK-DESIGN.md`.

## 2. The W4 gate dilemma (D2 — decision needed)

**Why gates exist at all.**  Every other win is a *policy* change: running without it is a legitimate
configuration (it is what the previous build did), so a beta tester must be able to switch it off
individually — coherence divergence, a reserve regression or a perf drop can then be attributed to one
win in one run, without rebuilding.

**Why W4 is different.**  W4 is not a policy, it is a *bug fix* in shared ggml code:

* running "without" it means running the leak (the allocator keeps buffers alive that should be
  released) — nobody wants that as a configuration;
* it is bound for an upstream PR: an env knob that re-enables a leak is exactly the kind of thing
  ggml-org reviewers push back on, and it would make the beta artifact differ from the PR artifact;
* on every model we have it is a **measured no-op** (coherence byte-identical, reserves identical —
  see `README.md` §6), i.e. during the beta window it will only matter if some *other* graph hits the
  idiom (which is the whole point: the win is for graphs that assemble a large tensor by copying into
  views of it).

**Options:**

| option | what it means | pros | cons |
|---|---|---|---|
| **A. No gate** | W4 ships in Block 15 exactly as it would upstream | beta artifact == PR artifact; no dead code path; no reviewer friction | a tester cannot disable it from the CLI; "every win is switchable" is not literally true |
| **B. Temporary beta-only env gate** (`GGML_ALLOC_KEEP_UNUSED_VIEWS=1` restores the old behaviour) | gate lives in Block 15 during beta, is stripped before/at promotion; the upstream/ copy stays ungated | satisfies the "switchable by env var" policy during beta | the beta-tested code differs from the PR code; risk of forgetting to strip it; one more code path in a hot allocator loop |
| **C. Hybrid (recommended)** | **no env knob in the code**; instead ship a ready-made one-command revert for A/B testers: `beta/block-15-campaign-wins/ab/w4-revert.patch` (the reverse of the W4 hunk, applied with `git apply -R`) plus the documented binary pair | W4 stays byte-identical to the PR; A/B is still one command and needs no rebuild of the delivery source (or one `git apply -R` + rebuild); nothing to remember to strip | A/B costs a rebuild (~1–2 min) instead of an env var |

**Recommendation: C.**  It honours the intent of D3-style gating (a tester can isolate W4) without
putting a knob in shared ggml code that upstream would reject.  If the maintainer prefers the literal
policy, choose B and add "strip `GGML_ALLOC_KEEP_UNUSED_VIEWS` before cutting the promotion patch" to
§4's checklist.

## 3. Stage A — the two extra upstream candidates (D6)

Both follow the existing `upstream/` convention: `UPSTREAM-PR-<slug>.md` (self-sufficient notes: what,
why, root cause, evidence, base commit, apply-check result, validation) + `UPSTREAM-PR-<slug>.patch`.

### A1. `UPSTREAM-PR-kv-cache-keys-only` — dead indexer V buffer

* **What**: adds `bool v_enabled = true` to `llama_kv_cache`'s constructor (`src/llama-kv-cache.h/.cpp`:
  no V tensor allocated, and no V-side op may be issued) and passes `/* v_enabled */ false` for the
  qwen4exp QSA indexer store in `src/llama-memory-hybrid-idx.cpp` (~L60).  The indexer never reads V
  (keys-only scoring), so its V buffer is pure waste — measured 956.25 → 318.75 MiB of indexer KV, i.e.
  −1.9 GiB per box because the V copy was triplicated.
* **Upstream evidence (2026-09-10)**: the existing fork patch applies to **current master with
  0 failed hunks and 1 fuzz** (`patch -p1 --dry-run -F3`) — so upstream has the same waste and the same
  construction sites.
* **Work**: regenerate it as a clean-room patch against `origin/master` (rebuild the branch from
  master, apply the fork patch, resolve the fuzz, `git apply --check` must be clean), re-run the
  evidence (CPU-only build at minimum; the runtime numbers come from the fork campaign), write the
  `.md` in the same shape as `UPSTREAM-PR-ggml-alloc-unused-view.md`, add the `upstream/README.md` row.
* **Caveat to state in the `.md`**: the fork's numbers were measured with the fork's QSA ops; upstream
  validation would be a same-seed coherence run on a QSA model (which upstream supports —
  `src/models/qwen4exp.cpp` exists upstream; the CUDA ops it needs are fork-only, so upstream's QSA
  path may differ — verify before claiming coherence parity upstream).

### A2. `UPSTREAM-PR-attn-k-null-mask-guard` — latent crash in upstream code

* **What**: `llm_graph_input_attn_k::set_input` (`src/llama-graph.cpp`) calls
  `llama_kv_cache::set_input_kq_mask(self_kq_mask, ...)` unguarded, while its own `can_reuse_impl()`
  already accepts a null mask (`self_kq_mask == nullptr || can_reuse_kq_mask(...)`).  Any graph that
  legitimately leaves that input unallocated crashes at `GGML_ASSERT(ggml_backend_buffer_is_host(...))`
  in `llama_kv_cache::set_input_kq_mask`; the siblings (`llm_graph_input_attn_kv::set_input`,
  `llm_graph_input_dsv4_raw::set_input`, …) all guard it with `if (self_kq_mask && self_kq_mask->buffer)`.
* **Source**: the hunk currently inside W2 (`patches/0002-derived-qsa-block-bias.patch`) — extract only
  that hunk (it is generic: upstream code, no fork dependency).
* **Work**: extract, verify `git apply --check` on `origin/master` (expected clean), write the `.md`
  (include the inconsistency argument + the crash path + a note that the fork found it while pruning
  masks), add the `upstream/README.md` row.

## 4. Stage B — Block 15 (the beta delivery)

### B1. Merge the wins into one tree
The current fork tree already has **W1 + W2** applied and uncommitted (10 files).  **W3 is not
applied** and overlaps W2 in `src/llama-memory-hybrid-idx.cpp` (and both touch
`src/models/qwen4exp.cpp`), so:

```bash
cd ~/llama.cpp && git status --short            # expect exactly the 10 L1 files
git apply -3 <W3 patch>                         # staged 3-way; resolve conflicts by hand
# verify: 13ish modified files, no conflict markers
```
If the 3-way merge is awkward, the alternative order is a fresh tree: apply W1 → W3 → then W2 with
`git apply -3` (W2's patch is diffed against W1, not W3).  The merged tree is what must be validated —
do not trust the individual validations.

### B2. Gate the wins (D3) and strip diagnostics
| win | gate | where | default |
|---|---|---|---|
| W1 | `GGML_QSA_SCORE_MEM` | `src/models/qwen4exp.cpp`, `build_qsa_top_k` (~L1311–1360): one env read; `0` = the original path (reshape **then** relu, no chunking — the pristine 2×1600 MB behaviour) | 1 (both L2a + L2m on) |
| W2 | `GGML_QSA_DERIVED_BIAS`, `GGML_QSA_DERIVED_VIS`, `LLAMA_QSA_SPARSE_FA` | already present; **strip `GGML_QSA_DERIVED_BIAS=2|3`** (the `zero_bias`/`tiny_bias` placeholder modes and their extra tensor allocations) — they were diagnostics for the MTP-layout investigation | unchanged (1/1/sparse) |
| W3 | `LLAMA_QSA_KEYS_ONLY` | `src/llama-memory-hybrid-idx.cpp` ~L60: make the `/* v_enabled */ false` argument conditional; `0` = keep the V buffer | 1 |
| W4 | per §2 (recommended: no knob + a revert patch under `beta/block-15-campaign-wins/ab/`) | — | — |

Gate names follow the existing convention.  Document the interactions in the beta README: W2's prune is
only valid while `LLAMA_QSA_SPARSE_FA=1` (already enforced by the shared `qwen4exp_qsa_sparse()`
predicate), and W1's `0` path is bit-identical by construction (relu is elementwise).

### B3. Combined validation
Follow `README.md` §4 (protocol) and §6 (the reference numbers).  The specific interaction cases to
add beyond the per-win checks:

* **W3 + W2**: the indexer's `cell_vis`/`blk_*` arrays and the cache's V tensor are now both cache-side
  changes — test prefill + decode + MTP, and the K-store-only policies (`cell_blk == nullptr` path).
* **W1 + W2**: the top-k consumes the chunked score and adds the derived bias in-kernel — test with
  both gates on/off in all four combinations, byte-identical expectations.
* **`LLAMA_QSA_SPARSE_FA=0` + W3**: the dense FA fallback must still work (it reads the mask, not the
  indexer V).
* Everything under `README.md` §4's standard matrix (coherence on 4B/27B/qwen4exp, reserve matrix, MTP
  gate, bench parity, allocator tests, no reserve growth over repeated graph builds).

### B4. Cut the block (nothing is pushed, ever — see AGENTS.md)
1. Commit the merged+validated state on the fork's `rdna-boosts` branch as the **15th block commit**
   (match the style of the existing block messages; note the wins and the gates).
2. `scripts/make-patches.sh` with the **tip default updated to the new block-15 commit**; verify
   blocks 01–14 come out byte-identical to the current files and `0015-rdna-boosts-block-15-<slug>.patch`
   is the new one.
3. `scripts/apply-all.sh` 14 → 15 (the block count) + `patches/README.md` if it lists blocks.
4. `MANIFESTS.md` (apply order + per-block verification + a dated validation record),
   `README.md` (the patch table + the "current state" summary pointing at WORKLOG),
   `WORKLOG.md` (a dated entry at the top — this is a delivery-affecting change),
   `BASELINE.md` (provenance/tip), `rdna-boosts-all.patch` (regenerated by `make-patches.sh`).
5. **Clean-apply simulation** in a fresh worktree at the fork point (`scripts/apply-all.sh`) + build +
   coherence — the repo's standard pre-ship gate.
6. `beta/block-15-campaign-wins/block-15-campaign-wins.patch` = the beta copy, and this README +
   `README.md` updated as the **promotion record** (beta started <date>, gate table, validation
   results).  The ~4–5 day beta window starts here.
7. **Upstream-drop check**: if upstream has merged W4 (or any other `upstream/` entry) by then, drop
   that hunk from Block 15 before shipping and note it in WORKLOG.

### B5. Promotion (after the beta window)
`patches/0015-…` + `apply-all.sh` + `make-patches.sh` tip + `MANIFESTS.md`/`README.md` headers +
a WORKLOG entry + the beta README marked PROMOTED.  Then Block 15 is part of the delivery set, and the
campaign is closed.

## 5. State inventory

**Fork** `~/llama.cpp`, branch `rdna-boosts` at `e2380eb67` (= fork point `9113cc188` + blocks 01–14),
working tree = **10 modified files = W1 + W2 (incl. the `attn_k` guard)**, deliberately uncommitted.
W3 is **not** applied.  Never commit those 10 files except as the Block-15 commit (B4); never push
from that checkout.

**Campaign patches**
* W1 `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch`
* W2 `wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch` (regenerated 2026-09-10,
  10 files, +557/−80, `git apply --check` clean on a fresh W1 base)
* W3 `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch`
* W4 `wip/arch-independent-memory/patches/0001-ggml-alloc-release-unused-view-sources.patch` and its PR
  copy `upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch`

**Volatile helpers** (rebuild as needed; `/tmp` may be wiped between sessions):
`/tmp/bin-pristine` (14 blocks), `/tmp/bin-l2` (W1), `/tmp/bin-l1` (W1+W2 pre-guard),
`/tmp/bin-l1guarded` (= the current tree), `/tmp/bin-l3b` (tree + W4), `/tmp/bin-keysonly` (W3),
`/tmp/bin-l0base|l0c|l0d` (instrumented allocators), `/tmp/ggml-alloc.instrumented.c` (the L0b–L0e
instrumented allocator source), `/tmp/master-pr` (a master worktree with W4 applied — disposable),
`/tmp/alloc-leak-repro{,2}` (the built repros).

**Tools** (all in the repo, model-agnostic):
`wip/qwen4exp/qsa-memory/tools/{model-sweep.sh,mask-scaling.sh,bufsize.sh,ub-sweep.sh,ab-coherence.sh,mtp-ab.sh,l0a-scheddump.sh,peak-ledger.py,view-reuse-ledger.py}`;
repro: `wip/arch-independent-memory/repro/ggml-alloc-unused-view.c` (build/run recipe in its header).

**Rebuild** (AGENTS.md): `export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH && cmake --build build-rocm
--target llama-cli llama-bench -j 16`; run with
`LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0`.

**Patch-regeneration recipe** (worktree trick, used for W2): worktree at `HEAD` → apply the base patch
(W1) → commit → copy the modified files from `~/llama.cpp` into the worktree → `git diff` → remove the
worktree → verify `git apply --check` on a fresh base.  For conflicts across my own whitespace
cleanups, use `git add -A` + `git apply -3` (not `patch -F3`).

## 6. Rules that must not be broken

* Nothing under `wip/` enters `patches/` except through the promotion path (AGENTS.md WIP rule);
  Block 15 is that path, and it needs the maintainer's go-ahead (D4/D5 give it, per-stage).
* Never push from `~/llama.cpp`; the fork branch is disposable.  Delivery-repo pushes are the
  maintainer's.
* One GPU job at a time; `llama-cli` always gets `--single-turn`; compare **generated text only**
  (llama-cli's timing footer always differs).
* Memory numbers come from `bufsize.sh` / `model-sweep.sh` / `mask-scaling.sh`, never llama-bench.
* Any change that can move buffer layout must pass the adaptive-MTP gate
  (`tools/mtp-ab.sh`, acceptance ≥ ~0.45, MTP ≥ plain at depth 3).
* Check for stray `llama-cli`/`llama-bench` processes before measuring; `llama-slot-prox` owns ports
  8037–8039.
* Keep the delivery-repo working tree clean between sessions; commit with clear messages.

## 7. Next-session prompt (copy-paste)

```
Continue in /home/stew675/llama-cpp-rdna-boosts (read AGENTS.md first; its rules override this).
Start with beta/block-15-campaign-wins/HANDOVER.md - it holds the decisions taken on 2026-09-10, the
plan for Stage A (two upstream candidates) and Stage B (Block 15: merge W1-W4, gate them, validate the
combination, cut the block, stage it in beta/), the state inventory, and one open decision (the W4
gate, HANDOVER section 2 - the maintainer wanted the dilemma explained; re-explain it briefly in your
first message and propose option C, or implement whichever they pick).

Order of work, unless the maintainer says otherwise:
1. Stage A (small, independent): prepare upstream/UPSTREAM-PR-kv-cache-keys-only.{md,patch} and
   upstream/UPSTREAM-PR-attn-k-null-mask-guard.{md,patch}, each verified with git apply --check against
   the current origin/master (the keys-only patch currently applies with 1 fuzz, so rebase it), with
   self-sufficient notes and rows added to upstream/README.md.  No runtime validation is required
   beyond what the fork campaign already recorded - say so explicitly in each .md.
2. Stage B: merge W1-W4 on the fork tree (order W1 -> W2 -> W3, or W1 -> W3 -> W2; use git add -A +
   git apply -3 and resolve by hand), add GGML_QSA_SCORE_MEM (W1) and LLAMA_QSA_KEYS_ONLY (W3), strip
   the GGML_QSA_DERIVED_BIAS=2|3 diagnostics, then run the combined validation in
   beta/block-15-campaign-wins/README.md sections 4 and 6 (coherence byte-identical with every gate
   on and off, reserve matrix, MTP gate, bench parity, allocator tests, no reserve growth).
3. Cut Block 15 per HANDOVER section B4 (fork commit, make-patches.sh with the new tip, apply-all.sh
   14 -> 15, MANIFESTS/README/WORKLOG/BASELINE updates, clean-apply simulation, beta copy) and stage it
   in beta/block-15-campaign-wins/.  Do not create a Block 16 - later wins (V2/V3/V4, see
   wip/arch-independent-memory/DERIVED-MASK-DESIGN.md) are amended into Block 15.

Report at the end: the merged gate table with defaults, the combined validation results, the beta
artifact path, and anything that regressed or could not be validated.
```

## 8. Open questions

1. **W4 gate**: option A (no gate), B (temporary env gate) or C (no knob + `ab/w4-revert.patch` for
   A/B — recommended).  See §2.
2. **Block-15 cut timing** (D5): cut with W1–W4 as soon as they are combined and validated (recommended,
   starts the beta clock) vs. hold until V3/V4 land.  One line confirms it.
3. **Beta tester material**: do you want a one-page A/B checklist (env-var matrix + what to report) in
   `beta/block-15-campaign-wins/`?
