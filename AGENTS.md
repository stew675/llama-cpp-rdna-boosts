# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing anything in
`~/llama-cpp-rdna-boosts/` (or acting on its behalf). It is the **policy layer**: what is mandatory, what is
in and out of scope, how a change is verified and how a release is cut. It is deliberately short and it
carries **no findings and no numbers of record** — the engineering facts, invariants and "do not do that
again" rules live in [`INVARIANTS.md`](INVARIANTS.md), the purity rules in `GREEDY-PURITY.md`, the dated
record in `WORKLOG.md` (newest first) and `archive/`. The pre-2026-10-04 AGENTS.md is preserved verbatim in
`archive/docs/AGENTS-HISTORY.md`.

**Current state:** `release.json` is the single source of truth (fork point, canonical tip/tree, block
count, per-artifact sha256); the top of `WORKLOG.md` is the newest dated record. Do not restate release
history here.

## What this repo is

A **delivery repo**. It packages the RDNA/ROCm work of the [`stew675/llama.cpp`](https://github.com/stew675/llama.cpp)
fork (`rdna-boosts` branch) as a **18-patch default set** (17 official blocks `0000`-`0016` + the temporary
`0300`) that applies to a clean
llama.cpp checkout at the fork point named in `release.json`. This repo is NOT the fork: the fork
lives at `~/llama.cpp`; its `rdna-boosts` branch is re-cut from `patches/` on every release and
force-pushed to the personal fork so it never goes stale (see the Pushing policy). `patches/` is the
deliverable; `scripts/` is the apply/regenerate/validate flow; `README.md`
is the consumer overview; `MANIFESTS.md` is the apply order + verification contract, `BASELINE.md` the
provenance + drift policy (both are contracts now — their dated history is in `archive/docs/`);
`GREEDY-PURITY.md` is
the purity rulebook.

### Block numbering (r288, the ranges are policy)

* `0000`-`0099` — **official supported** blocks, contiguous from `0000`.  `N` in `v<N>-...` is the count of
  THESE blocks; the prefix is derived by `scripts/make-release.sh`, never hand-written.
* `0300`-`0399` — **temporary** blocks.  They ARE in the default apply (after every `00xx`), and they are
  **not** counted in `v<N>`.  Currently `0300` (`-sm relaxed-tensor`, `TODO` #104).
* `0100`-`0199` — **optional** blocks under `optional/<name>/<NNNN>-<name>.patch`.  They are never in the
  default apply, the default tree, `rdna-boosts-all.patch`, the fork branch or the default release gates;
  `scripts/apply-all.sh --include-optional[-all]` applies them on the verified default tree in **numeric**
  order.  Currently `0100` (two-lane prefill) and `0101` (mixed K/V cache types).

A default apply is therefore `00xx` then `03xx` (`release.json.n_default` patches), and
`validate-set.sh` / `make-release.sh` enforce the split.  Moving a block between ranges, adding one, or
renumbering is a maintainer decision (`ALWAYS ask before creating a new delivery patch block`, below).

## Before you change anything (the rules that damage if missed)

These are the ones that bite immediately; the reasoning, the numbers and the rest of the engineering
invariants are in [`INVARIANTS.md`](INVARIANTS.md). Read the entries that name the file you are about to
touch.

1. **Apply with `scripts/apply-all.sh`** (strict `git am`). Never `git apply` the concatenated series — it
   silently drops hunks.
2. **`llama-cli` needs `--single-turn`** (plus `--no-display-prompt` for scripted output), anything that can
   block goes inside `timeout`, and **`-lv` is picked by the arm, never by habit**: a **text/hash** arm takes
   **no `-lv` at all** (the same stream carries the generated text), a **census/mechanism** arm takes
   **`-lv 4`**.  `-lv 3` is the trap — it pollutes the text *and* still hides the ggml censuses.  Table under
   *Verify* below.
3. **Pin `HIP_VISIBLE_DEVICES` on every multi-device arm** — the iGPU is otherwise enumerated and the load
   aborts with `invalid kernel file`.
4. **Throughput is not a correctness signal.** Verify with same-seed greedy text, then the gate the change
   actually needs (`MANIFESTS.md` → Release gates). A uniform prefill shift is invisible to text.
5. **A fresh-process comparison needs `ROCBLAS_USE_HIPBLASLT=0`** on that run only; a per-start difference is
   the library, not the change (`INVARIANTS.md`, §44).  The same rule applies to the **expert-cache state** on a
   MoE arm: pin the residency prior (`MOE_EXPERT_HEATMAP=load`), because residency selects the MoE kernel set and
   the sets are not bit-identical (`GREEDY-PURITY.md` §89).
6. **Greedy text is a per-device-count reference** on a tensor split: judge a build change *within* one device
   count, or use `-sm layer`.
7. **Never run benchmarks in parallel**, and size `-t` (leave the GPU IRQ cores free).
8. **A re-base is a decode *and* prefill change**: measure before tagging, not only verify.
9. **Every new environment variable goes in `ENVIRONMENT.md`**, with its default and class (kill switch /
   opt-in / tuning / diagnostic); a beneficial gated feature is **on** by default.
10. **Do not regress a documented invariant**: if the code and an `INVARIANTS.md` entry disagree, fix the
    entry in the same change — and if the invariant itself must change, say so before shipping.
11. **Record the work**: a dated `WORKLOG.md` entry (newest first) for anything delivery-affecting, plus the
    per-block note in `patches/README.md` when a block's contents change.
12. **Report findings before you commit, not after** (see the next section).
13. **ALWAYS ask before creating a new delivery patch block.**  A new block, a block renumbering, or a
    change that moves a feature across block boundaries is the maintainer's decision — never a gate
    outcome.  A green gate ladder, a validated win, a passing PR or an optional artifact ready to
    promote are **not** approval.  Stop and ask before cutting, adding or renumbering a block, and before
    any release commit/tag/push that depends on that structure.  See the promotion rules below.

## Report findings before shipping (MANDATORY)

Surface review findings, open questions and small nits to the maintainer **before** any
delivery-affecting commit and **before** any release or tag, and wait for a decision. Do not bury them
in a commit message, a post-push summary or a GitHub reply. If something turns up mid-flight, stop and
ask instead of finishing the release and mentioning it afterwards. The maintainer decides whether a
finding is fixed now, deferred or dropped, and the decision is recorded (a `TODO.md` item, a block
amendment, or a note in the relevant record).

**A new delivery patch block is never created without the maintainer's explicit approval.**  ALWAYS ask
before adding a block to `patches/`, renumbering blocks, folding a feature into a different block in a
way that changes the block structure, or promoting an optional/beta artifact into the delivery.  The
maintainer owns the block structure; "the gates are green" is not permission.  If the block decision is
still open, stop before the release cut, the commit and the tag.
## Pushing policy (MANDATORY — read before any `git push`)

**Never push to upstream llama.cpp.** The only permitted push targets are this repo (the delivery,
`github.com:stew675/llama-cpp-rdna-boosts`) and the maintainer's personal fork
(`git@github.com:stew675/llama.cpp.git`). A bare `git push` in `~/llama.cpp` targets upstream
`ggml-org/llama.cpp` — never acceptable; repeated attempts can get the account banned.

- All deliverable changes live in THIS repo (`llama-cpp-rdna-boosts`) as the `patches/` set. That is
  the primary thing that gets pushed.
- **Every release MUST refresh the fork's `rdna-boosts` branch.**  Consumers clone the personal fork
  and check out `rdna-boosts` instead of applying the patch set, so a stale branch is what produced
  the 2026-10 Reddit "rdna-boosts is slow" incident (the branch was a month behind).  On every
  release (and whenever the block set changes) run **`scripts/fork-refresh.sh`** (add `--build
  --gates` for the full policy, and `--push` to publish once it has confirmed the tree):
  1. In a clean `~/llama.cpp`, check out upstream `master` at the **baseline point** — the `.base`
     commit in `release.json`, never the tip of `master`.
  2. Delete the existing `rdna-boosts` branch and re-create it with `scripts/apply-all.sh` (strict
     `git am`; the applied tree must equal `release.json`'s `.tree`).
  3. Build it and run the coherence + MTP gates and `scripts/gate-prefill-logits.sh`; `scripts/validate-set.sh` must be green.
  4. `git push --force-with-lease origin rdna-boosts` — the **personal fork only**, never upstream;
     never a bare `--force`.
- **`.tip` means the fork branch tip, and it is the only SHA to record.**  `release.json.tip` names
  the commit the personal fork's `rdna-boosts` branch carries.  `scripts/apply-all.sh` prints it
  (`Applied tip: <sha>`), `scripts/fork-refresh.sh` re-checks it against `.tip`, and since the
  deterministic re-cut (tooling change after r150: `apply-all.sh` pins the committer identity to the
  patch series' author and stamps each commit's committer date from its own author date) any checkout
  of `.base` on any machine produces **the same commit ids**, so the value is reproducible.  Three
  traps, all of which have bitten before:
  (a) the *working chain* tip in `~/llama.cpp` (the commits a release is developed on) is a
  **different commit with the same tree** — never record it; (b) a re-cut made before that tooling
  change (r150 and earlier) was stamped with the re-cut's wall-clock time, so its recorded tip is the
  commit the branch carries but the current `apply-all.sh` cannot reproduce that id (its tree still
  matches, which is the contract); (c) `scripts/validate-set.sh` cannot check the tip — it
  materialises `.base` from the commit tarball into a fresh repository, so its applied commit ids
  differ by construction (it checks the **tree**; `fork-refresh.sh` checks the tip against a real
  checkout).  If in doubt: **compare trees, never SHAs**; a matching tree plus a `rdna-boosts` branch
  whose tip is `.tip` is the whole contract.
- **Every release MUST also publish the ROCm container images.**  They are built and pushed
  **off-host** by the worker watcher (`scripts/container-watch.sh` on `radiance`), which polls for the
  new release tag and runs `scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune`; see
  `CONTAINERS.md` → "Offload the image build to a worker".  **Do not run the full container image build
  (all targets, both ROCm lines) on the primary box during a release**: it pins all cores for tens of
  minutes and contaminates any benchmark running beside it.  Nothing here changes the normal local
  **source** build for testing and development (`~/bin/build-llama-rocm`, Common tasks → "Build the
  fork"), which stays on the primary box.  The local `scripts/build-containers.sh` is for a smoke test or
  a manual worker-less container build only.  The Actions *gating* workflow no longer builds them (the tag-push matrix took ~5 h and its failure mode was a cold
  retry).  The fork-branch refresh and the worker image push are both part of tagging a release.
- The `~/llama.cpp` checkout exists to host the block commits and to apply/test the diff set locally.
  Apart from the release-time branch refresh above it is disposable: delete the pre-patched branch and
  re-apply rather than editing history in place.
- Confirm the exact branch name and remote before any push: the branch is always `rdna-boosts`, the
  permitted target is always `git@github.com:stew675/llama.cpp.git` (the personal fork).
- Repeated attempts to push directly to llama.cpp can result in an account ban. When in doubt: don't
  push, ask.
## Scope policy — RDNA first, other backends uninjured

This repo is **RDNA/ROCm-specific**: validation, tuning and claims cover the AMD devices the
maintainer runs (gfx1201 = RDNA4, plus validated gfx1151/RDNA3_5 and gfx1100/RDNA3_0). The patch set
is generic llama.cpp, so it must not *break* other backends (NVIDIA/CUDA, MUSA, SYCL, Vulkan, CPU):
keep the shared CMake lists, dispatch tables and predicates mutually consistent even where a change
is unreachable on AMD. **Behaviour and performance on non-AMD backends are explicitly out of scope** —
no tuning, no validation, no waiting on hardware there.

- An AMD-only fix may land without its non-AMD counterpart as long as the non-AMD paths stay
  consistent (no aborts, no uninstantiated pairs) and the difference is documented (see the F1 chooser
  example in `upstream/fa-decode-verify-kernel-family/`).
- New KV-cache types / instances / predicates **are** kept cross-backend consistent — an inconsistent
  set is a crash on whichever backend reaches it (`GREEDY-PURITY.md` §20). That is correctness, not
  scope creep.
- "Not validated on NVIDIA" is an acceptable, documented state — never a blocker for an RDNA win.
## Default-on policy — beneficial features are ON; env vars only disable

A feature that improves performance (or correctness) and has passed the relevant QA gates is
**enabled by default**. An environment variable for such a feature exists **only to disable it** — for
A/B testing, bisection or debugging — never to enable it.

**Why:** the maintainer's validation boxes (Strix Halo / gfx1151 in particular) are much slower than
gfx1201, so a gate that silently runs without a beneficial opt-in wastes wall-clock time and reports
slow numbers as if they were the product (the `mmb`/HC16 campaign measured ~835 t/s for a session
when the full set is ~1136 t/s, +36 %).

- When a gate passes, flip the default **in the same change** and turn the old opt-in into an opt-out
  (`FEATURE=0`). Keep reading the env var; just invert the default.
- A feature that is beneficial but not fully gated is a reason to **gate it**, not to leave it off.
  Default-on with the risk documented, then run the gate.
- A kill-switch for a *correctness* risk (a bisect knob, a known-bad-state workaround) may stay
  default-off — the distinction is "on because it helps" vs "off because it is known-risky".
- Before any benchmark, use the full feature set. If you write `FOO=1 <bench>`, ask why the default is
  not already `1`.

## WIP and promotion rules (MANDATORY)

Everything under `wip/` and `archive/work/` (including loose patches/diffs and experiment trees) is
**experimental work, NOT part of the delivery**. Never apply a `wip/`/`archive/work/` item to the fork
or any llama.cpp checkout, and never fold its content into `patches/`, **unless the user explicitly
asks for that specific item**. WIP stays env-gated OFF and excluded from `patches/`; the sanctioned way
out is promotion: collect *validated* wins, give each an env kill-switch, re-validate the
**combination** (individual validations do not carry over), then cut a new block with the maintainer's
go-ahead — **a new block is ALWAYS asked first and is never a gate outcome** (see "Report findings before
shipping"). Anything also applicable to unadulterated upstream gets a copy under `upstream/`.

**`beta/` is the external-validation holding area, not a third kind of source.**  `scripts/apply-all.sh
--include-beta` is the sanctioned way to *test* one (it applies `beta/*/*.patch` after the release blocks and
reports the resulting tree); nothing in `beta/` is ever picked up without that flag.  A candidate that
cannot sit in `wip/` because it edits a file a block owns, but is not yet validated on the reporter's
platform, goes to `beta/<name>/` (a `README.md`, a `BETA-TESTING.md` and a patch cut against the
current `release.json` tree; nothing in it is applied by `apply-all.sh`).  On confirmation or
refutation it moves to `archive/work/<name>/` and is recorded in `WORKLOG.md` — same no-stub rule as
`wip/`.

**`optional/` is for a finished feature that has no validated win here — it is not a fourth source
either.**  A contribution that is genuinely useful on hardware this repo does not validate (a
heterogeneous multi-GPU pair, say), or that is unreachable in the splits the delivery validates (a
backend-count gate is inert under `-sm tensor`, which is ONE backend — see `INVARIANTS.md`), earns an
`optional/<name>/` artifact instead of a block: `<name>.patch` cut against the **current `release.json`
tree**, a `README.md` that records its sha256, the tree it produces, the layouts it reaches and the
layouts it cannot, and what it measured.  `scripts/apply-all.sh --include-optional` is the sanctioned way
to apply one — it asks `[y/N]` per artifact (`--include-optional-all` for a headless apply) (strict `git am` on the verified release tree, one commit, tree reported; nothing here is
picked up without that flag).  An optional artifact **is rebased onto every new release** and is never
cut against a working branch or a part-applied set; it is promoted into `patches/` (with the full gate
ladder) only once it has a validated win on one of the delivery's own configurations, and archived to
`archive/work/` if it turns out wrong.  It is reviewed for build, default-path neutrality and
cross-backend consistency, but it is **not** covered by the delivery's gates and a consumer who applies
one is off the supported configuration — its `README.md` must say so.

**A finished campaign leaves `wip/` — it does not become a stub.**  The moment a campaign is promoted
(folded into `patches/`), closed, or refuted, its record moves to `archive/work/<campaign>/` and the
result is recorded in `WORKLOG.md`, then the `wip/` entry is **deleted** and `wip/CAMPAIGNS.md` is
updated.  Do **not** leave a "MOVED — see archive" placeholder behind: a redirect stub makes `wip/` claim
work is in flight when it is not, which is exactly the state this rule exists to prevent.  Dated records
(`WORKLOG.md`, `MANIFESTS.md`, `BASELINE.md`, `archive/`) keep the path that was correct when they were
written; when a campaign moves, fix those paths to the new `archive/work/` location rather than keeping a
stub alive to serve them, and treat a `wip/` path in an old record as a historical location.

## Layout

Docs each have one job: **this file** is policy; [`INVARIANTS.md`](INVARIANTS.md) is the engineering facts;
[`GREEDY-PURITY.md`](GREEDY-PURITY.md) is the purity rulebook; [`MANIFESTS.md`](MANIFESTS.md) is the apply
order + verification contract; [`BASELINE.md`](BASELINE.md) is provenance + the drift policy;
[`ENVIRONMENT.md`](ENVIRONMENT.md) catalogues every environment variable; [`WORKLOG.md`](WORKLOG.md) is the
dated record (newest first); [`release.json`](release.json) is the single source of truth for the release.

| path | what |
|------|------|
| `README.md` | consumer overview + workflow (start here) |
| `ENVIRONMENT.md` | **every environment variable** the delivery adds or repoints — default, and whether it is a kill-switch (default-on, set to disable), an opt-in, a tuning value or a diagnostic. Read it before adding a variable or changing a default |
| `INVARIANTS.md` | **the engineering facts** (the `do not re-derive` list: measured facts, invariants and "do not do that again" rules, each tagged with the `rNN` that established it). Correct an entry in place when it changes; the narrative is in `WORKLOG.md`. Extracted from this file in 2026-11 |
| `release.json` | **delivery single source of truth** (fork point, canonical tip/tree, block counts, per-artifact sha256) — read by `apply-all.sh`, `validate-set.sh` and CI; regenerate with `scripts/make-release.sh`, never hand-edit hashes.  `n_blocks` is the **official** count (it names `v<N>`), `n_temporary` the default-applied `03xx` blocks, `n_default` the **apply** count (`n_blocks + n_temporary`), `n_optional` (+ the `optional` map and `optional_tree`) the opt-in `01xx` artifacts |
| `MANIFESTS.md` | apply order + the verification contract (release gates, per-block verification, failure handling). Its dated validation history is in `archive/docs/validation-history.md` |
| `BASELINE.md` | fork point, what the patches are, how they are cut, drift policy. Its dated baseline/re-base records are in `archive/docs/baseline-history.md` |
| `TODO.md` | **forward-looking only**: open items, items waiting on someone else, accepted limitations, parked ideas. A closed item is one `WORKLOG.md` entry — never a TODO entry. `#NN` numbers are permanent handles and are never reused |
| `GREEDY-PURITY.md` | the purity rulebook: 11 invariants + one entry per finding (claim / rule / numbers / record), `§`-numbered and stable. Dated narratives are in `WORKLOG.md` and the `benchmarks/` record each entry names; the pre-r100 ones are also in `archive/docs/GREEDY-PURITY-FINDINGS.md` (`§§1-42`) |
| `patches/` | **the default delivery set** (official `0000`-`0016`, then the temporary `0300`; the optional `0100`/`0101` are in `optional/` and never here) + `README.md`, the per-block reference (what each block contains, its knobs, its purity notes). **Block 14 is the streamed KV**: both Goals A and B of `TODO` #118 — an opt-in fixed-VRAM resident **window** with a pinned-host tail (`--kv-window [auto|N]`, promoted from `LLAMA_KV_RESIDENT_CELLS`, which stays as its deprecated alias) *and* the opt-in attendance **pool** (`--kv-attendance-pool`, alias `--kv-region-pool`: the authoritative KV in a pinned host mirror plus a compact device pool the sparse decode resolves through, system-sized — **not** a size knob since r247) — plus the MTP draft ring (`--spec-draft-kv-window [auto|N]`).  **All three default to OFF since r247**, first-class options since r246: they are fittability features that cost decode on a config whose KV fits (the draft window measured −9.7 % of MTP decode past its depth, `benchmarks/2026-11-10-mtp-draft-window-depth.md`), so `auto`/`N` are a conscious opt-in and a default run carries no placement decision. **Block `0101` is the mixed K/V cache types** (PR #312; block 17 in r287, moved OUT of the default set by the r288 re-org, so it is an **optional** artifact under `optional/mixed-kv/`): K `q8_0` with V `q4_0` / `q5_0` / `q5_1` on a model whose attention layers are all 256-wide, read natively by the tile kernel's second V type instead of through the whole-cache F16 scratch; qwen4exp (QSA) and every other head size/pair stay refused, same-type caches are unchanged, and `GGML_CUDA_FA_KV_MIXED=0` restores the staged path.  Gate: `scripts/gate-mixed-kv.sh`, run on the optional-applied build; knobs: `ENVIRONMENT.md` §8.  **Block 15 is the UMA GTT expert store** (r263): an APU's MoE expert set in ONE registered, `mlock`ed region, with the shape following that budget (a budget covering the set PLACES it; one below it takes the ARENA: slot storage + LFRU residency + one `pread` into the vacated slot on a miss), the MTP head pinned outside that budget, the fills on block 12's adaptive transport (the buffered arm reads straight into the slot, and on RDNA3_5 UMA the ladder starts there), and a prefill stream that reads only what is not resident, coalesced and in parallel.  A model whose weights do not fit RAM is refused at load with the fix rather than left to swap, and the **store/arena** is **inert on a discrete GPU by measurement**.  **`-lme N|auto` is the host-expert budget in BOTH tiers, not an APU flag**: on an APU it sizes this GTT store, and on a discrete GPU it sizes the **pinned per-expert host pool** that `-ncmoe X` offloads into (`-lme` given is also the streaming opt-in; `--host-experts pinned|pool` are deprecated shorthands for the same budget).  Gate: `scripts/gate-uma-experts.sh`; knobs: `ENVIRONMENT.md` §20; record: `archive/work/apu-expert-gtt/`.  **Block `0300` is TEMPORARY** (`-sm relaxed-tensor`, a local relaxation of upstream #31081, no upstream candidate, tracked as `TODO` #104) — it is in the default apply but is not an official block (not counted in `v<N>`); keep it separate and expect to drop it. The dated amendment notes moved to `archive/docs/patch-notes-history.md` |
| `scripts/` | `apply-all.sh` (the tested apply flow, strict tree check + `git am -3` fallback, **deterministic commit ids**), `fork-refresh.sh` (the release-time fork refresh: re-cut, tree/tip check, optional build/gates/push — the `.tip` rule below), `gate-prefill-logits.sh` (the prefill-logit KLD release gate, issue #113), `gate-per-block-build.sh` (a clean build per block commit — `all` by default, `--fast` for the release profile), `gate-park-idle-slots.sh` (the server idle-slot parking gate: park == lazy-off text + the indexer cache moves with the attention cache, issue #227), `gate-hc-fusions.sh` (the qwen4exp hyper-connection fusion-equivalence gate: all four mix/combine arms must reproduce the unfused chain, once per hc-down weight class, issue #228), `gate-ckpt-reuse.sh` (the context-checkpoint reuse gate: a kept restored checkpoint changes no token, and only a hybrid/recurrent target reaches the keep path), `gate-state-pipe.sh` (the batched state save/restore gate: the pipelined / plain-memcpy-copy / pipelining-off / save-only / synchronous arms must restore byte-identical text, and it is the only gate that reaches that code), `gate-spec-width-band.sh` (the speculative verify-WIDTH gate: a band-0 3-GPU server at 1/2/4 slots, the bound must fire and win at 4 slots and be inert at 1-2 — the only gate that reaches it), `gate-spec-gdn-split.sh` (the speculator GDN-split gate, issue #309: four speculator sets on a hybrid target must report ONE chunked-GDN prefill tail in the `-lv 4` census while `n_rs_batch` differs, every set must reproduce the plain greedy text, and the default n-gram width must still out-draft the bounded one — the gate that keeps a speculator from choosing the prefill split), `gate-mtp-dev-rows.sh` (the MTP target h-rows gate, issue #318: the DEFAULT run must be the host readback and must not allocate the `h_nextn` device buffer at all, `LLAMA_MTP_DEV_ROWS=1` must reach the device path, and both arms must give the same greedy text and the same draft-acceptance counts), `gate-qwen4exp-quant-coherence.sh`, `gate-grid-stall-pad.sh` (the gfx1201 grid-stall pad gate: every covered launch site must really pad at its shape, a padded launch must match the CPU reference, an unpaddable grid must be refused, and the text must be identical with the pad off, issue #277), `gate-kv-residency.sh` (the KV residency gate: block 15's window and attendance pool, each with a resident arm and a forced-miss arm, plus a multi-device `-sm layer` pool section that also proves the per-device counter blocks -- text byte-identical with the expert-cache state pinned, and the arms' residency numbers must differ, `wip/kv-residency-cleanup/` P0), `gate-fusion-alias-copy.sh` (the fusion-window leaf-copy gate, issue #287: a shared leaf is only skipped when it cannot be a graph result, so a scheduler copy of a host expert table is checked -- the repro arm must reproduce the unfused chain's text byte for byte and log the decline, the failed-fill arm too, and the production arm must keep its anchor, issue #287), `gate-host-pool.sh` (the host-expert pool gate, issue #274: the pool's fill must never require the disk tier a platform may not have, a FAILED fill must be rolled back rather than served as resident, the call that fails a fill must DECLINE and not take its op over, and the exit census must print once from the model teardown — thirteen arms on one pool config (r277 adds `qd1` and `bulkbuf` for the placement read queue, plus the census checks that say the queue really ran), byte-identical text, `takeover=0` under the fill faults), `gate-moe-promise.sh` (the take-over promise's LIFETIME gate, issue #327: a promise is valid for one staging pass, so a pass that recycles the scheduler's `input_cpy` address must repair NOTHING (`broken=0 repaired=0` with `stale>0` as the non-vacuity check) while `MOE_EXPERT_TAKEOVER_FAULT=1` inside its own pass must still repair every table byte-for-byte), `check-gpu-faults.sh` (reads the kernel journal and scans a llama log for **real** GPU faults, filtering the echoed prompt lines — never grep a log for `page fault`/`slab` by hand), `make-patches.sh`, `make-release.sh`, `validate-set.sh` (checksums + fresh-tarball strict apply, runs in CI), `build-containers.sh` (the image build itself), `container-watch.sh` (the off-host release watcher, run on `radiance`), `cleanup-ghcr.py` (GHCR version retention; called by `build-containers.sh --push`), `extract-generated.py` (hashes llama-cli generated text; a naive sed/grep slice does not reproduce the hashes), `check-windows-paths.sh` (fails on a path that Windows cannot check out), `check-windows-portability.sh` (fails on a POSIX-only call outside a Windows-excluding region, or a `near`/`far`/`pascal`/`cdecl` identifier that collides with a `<windows.h>` macro, in a file the delivery adds **or on a line it inserts into an upstream file** — `tests/` included; issues #231/#306/#310; run by `validate-set.sh`). **Index: `scripts/README.md`** |
| `rdna-boosts-all.patch` | the entire 18-patch **default** net as ONE patch (fork point only; optional blocks excluded) |
|  `benchmarks/` | dated bench records + methodology; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) and **`prefill-logit-methodology.md` = the prefill-logit KLD gate** (run before shipping any prefill-kernel change or release) |
| `prompts/` | versioned, hash-stable test prompts; sizes/token counts/**sha256** in `prompts/README.md`. A shipped prompt is **never edited in place** (add a new file); a result is only valid against the prompt hash it names |
| `wip/` | **ACTIVE** exploration docs/tools — **NOT part of the delivery**; see the WIP rule. Each campaign is a self-contained handover under its own `README.md` (no `wip/README.md`); the open set is indexed in `wip/CAMPAIGNS.md` |
| `beta/` | **promotion-candidate patches** staged for a beta test — **NOT part of the delivery**. Each candidate is `<name>/` with `README.md` (what/why/status), `BETA-TESTING.md` (the tester's apply + gate checklist) and `<name>.patch` cut against the current `release.json` tree. Nothing here is applied by default: `scripts/apply-all.sh --include-beta` applies `beta/*/*.patch` on top of the verified release tree for a tester. A confirmed or refuted candidate moves to `archive/work/` and is recorded in `WORKLOG.md` |
| `optional/` | **opt-in blocks `01xx` that are NOT part of the default delivery and not a beta.** Each is `<name>/` with `README.md` (what, sha256, applied tree, the layouts it reaches and cannot, what it measured) and `<NNNN>-<name>.patch` cut against the current `release.json` default tree, rebased onto every release. For a *finished* feature that has no validated win on the delivery's own configurations (it needs hardware this repo does not validate, or it is unreachable in the splits it validates). `scripts/apply-all.sh --include-optional` offers them one at a time (`[y/N]`, default N) and applies the accepted ones strictly on the verified release tree (`--include-optional-all` = no prompt; they must still apply **strictly**, and `scripts/validate-set.sh` check 8 fails on a stale one); not covered by the delivery's gates. Promoted only on a validated win here |
| `upstream/` | **upstream-PR candidates** (self-contained changes for unadulterated `ggml-org/llama.cpp`), one sub-directory per candidate (`README.md` + `UPSTREAM-PR-<name>.patch`); the index is its `README.md`. None is part of the delivery |
| `utilities/` | **measurement/reporting tools, NOT part of the delivery** — nothing here is in `patches/`, nothing is applied by `scripts/apply-all.sh`, nothing is covered by a release gate; a tool measures, reports and changes nothing, needs no root, and says `UNKNOWN` rather than guess.  `storage-oracle/` takes a model file and reports which `--load-mode`/`--lazy-mode` to use for **that file on that device** (band map, cold per-read latency for `O_DIRECT`/buffered/`mmap`, sequential throughput, the idle-gap wake ladder, cold reads under a stream, page-cache residency, the `MemAvailable`/`RLIMIT_MEMLOCK` budget), with the measurement behind every verdict and an explicit avoid list; `--verify` checks the prediction against `llama-cli` head-to-head including a warm restart.  **Index: `utilities/README.md`** |
| `archive/docs/` | moved-out historical records (validation/baseline history, the pre-cleanup AGENTS.md) — reference only |
| `archive/work/` | closed experiments and campaigns, preserved for future re-evaluation (mmb/qsa3, moe-expert-cache, issues/PRs, …) — reference only |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |
| `.github/workflows/` | `validate.yml` (per-push delivery validation), `release.yml` (tag-triggered GitHub Release assets) and `docker-ghcr.yml` (manual-only ROCm image fallback); fork point read from `release.json`. **The release container images are built and pushed off-host by the worker watcher (`scripts/container-watch.sh` on `radiance`), not in CI and not on the primary box** — see `CONTAINERS.md` |

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout <release.json.base>
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 17 default commits

# with the beta candidates on top (NOT the release tree; for a beta test only)
bash <this-repo>/scripts/apply-all.sh --include-beta .   # 17 default blocks + beta/*/*.patch

# with the optional artifacts on top (NOT the release tree; opt-in features, see optional/README.md)
bash <this-repo>/scripts/apply-all.sh --include-optional .   # 17 default blocks + a [y/N] per optional block
```

### Verify (the coherence gate — mandatory after any change)

Local models live under `/llm/models/` (e.g. `Qwen3.5/4B/Q8_0/...`, `Qwen3.8/27B/Q4_K_XL/...`,
`Qwen3.6/35B-A3B/Q8_0/...`, the Gemma4 tree). Paths in older docs are host-specific; resolve them
against `/llm/models/`.

```bash
HIP_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn
```

**Pin `HIP_VISIBLE_DEVICES` for every multi-device arm on this box.**  The 9950X3D's iGPU (gfx1036) is
enumerated as a HIP device, and the delivery is built for `gfx1201` only, so an arm that lets `-sm tensor`
pick its devices without the pin aborts at model load with `ROCm error: invalid kernel file`
(`ggml-cuda.cu:130`) — it looks like a defect in the change under test and is not one.  Use
`HIP_VISIBLE_DEVICES=0,2` for the x8 pair and `0,1,2` for all three R9700s.  A **1 GPU** arm without the pin
happens to work (the iGPU is only reached when the split enumerates every device), which is what makes this
trap so easy to hit only on the 3-GPU arm.

Diff against a known-good build: same-seed output must be IDENTICAL. `GGML_CUDA_ALLREDUCE=nccl` is not
a bit-identical reference under `-sm tensor` (see `INVARIANTS.md`); treat it as a smoke comparison.

**`-lv` is a threshold you pick per arm, and `llama-cli` starts at errors-only.**  One flag, but there are
**two** ways a line reaches it, and that is the usual source of confusion:

1. **`common/`'s own macros self-filter.**  `LOG_TMPL` compares the macro's verbosity against the threshold, so
   `LOG_INF` / `LOG_CNT` (verbosity `LOG_LEVEL_INFO` = **3**) appear at **`-lv 3`**.  In practice the call site
   that matters is `common/sampling.cpp`'s **per-slot statistics block**, printed *between* streamed tokens.
2. **Anything logged through the ggml/llama logger is mapped up one notch.**  `LLAMA_LOG_INFO`
   (`src/llama-impl.h`) emits `GGML_LOG_LEVEL_INFO`, and `common_log_get_verbosity()` maps that level to
   **`LOG_LEVEL_TRACE` = 4** — so llama.cpp's *library* INFO (`sched_reserve: reserving …`, `two-lane prefill
   enabled`, the spec/`draft acceptance` lines) needs **`-lv 4`**, exactly like ggml's own `GGML_LOG_INFO`
   censuses.  Measured 2026-11-09: `llama-server`'s default and `llama-server -lv 3` produce **identical**
   startup logs (31 lines, 7 WARNs); only `-lv 4` adds its 191 INFO lines.

| `-lv` | what it adds | use it for |
|---|---|---|
| **none** (the `llama-cli` default, `LOG_LEVEL_ERROR`) | errors only — the banner, generic output, the generated text | **every text/hash arm**: `scripts/extract-generated.py` needs the clean stream, and the rates come from the `[ Prompt: … \| Generation: … ]` footer |
| `2` | WARN: the slab line, `narrow-2 … pinned at`, the pool/sink census WARNs | a quick mechanism spot-check |
| `3` | **`common/`'s** verbosity-3 INFO — in practice the per-slot statistics block, printed *between* streamed tokens | **nothing** — it corrupts a hash *and* adds no census, because llama's library INFO and every ggml census are one notch above it (see 2 above) |
| `4` | `LLAMA_LOG_INFO` (`sched_reserve`, `MoE expert cache:`, `expert access:`, `takeover=`, `cold admission=`, `two-lane prefill enabled`, the spec lines) **and** ggml INFO | **every census/mechanism arm** |

A hash and a census are therefore **two runs**: hash with no `-lv`, then repeat with `-lv 4` for the numbers
(that is what `wip`-style arm harnesses do; `archive/work/host-expert-parity/touch-ab.sh` is an example, and its
`VERBOSE=` is an explicit space, because `${VERBOSE:-default}` silently takes the default for an empty value).
`llama-server` is **not** more verbose where it matters: its default threshold is the common one
(`LOG_LEVEL_INFO` = 3) instead of `llama-cli`'s ERROR, so it already shows `common/`'s verbosity-3 lines — the
slot block, which is why a server arm *looks* chattier — and still hides every census and every
`LLAMA_LOG_INFO` until `-lv 4`.  What it does change is the hash: its generated text comes back over **HTTP**, so
`-lv 4` pollutes only the log — which is why `gate-ckpt-reuse.sh` hashes the API responses at `-lv 4` safely.
Every gate that hashes `llama-cli` text passes no `-lv` at all (`gate-host-pool.sh`, `gate-kv-residency.sh`, `gate-kv-window-split.sh`,
`gate-grid-stall-pad.sh`, `gate-norm-rows-alias.sh`, `gate-hc-fusions.sh`).  Code paths and the measured
consequences: `INVARIANTS.md` (2026-11-07).

The coherence gate is a decode/greedy check and is **blind to a uniform prefill-logit shift**. For any
change to a prefill kernel (GDN/SSM, FA, MMQ/MMVF/MMB, MoE prefill) or a default-on approximate path,
run the **prefill-logit gate** as well:

An **all-reduce arm** change (block 01) is only reachable with a multi-device `tensor` split, so pass
`--sm tensor --device 0,1,2` (r165): a single device has no comm layer and the default `layer` split
needs no all-reduce, so the bare gate would pass a broken arm.

```bash
./scripts/gate-prefill-logits.sh          # compare the current build against the known-good base
./scripts/gate-prefill-logits.sh --record # (re)record the base from a known-good build
./scripts/gate-prefill-logits.sh --ab "GGML_CUDA_GDN_CHUNKED_BF16=0"  # exact-fallback A/B
```

It requires mean KLD <= 0.005 and same-top-p >= 98 % (per ROCm line: a base recorded on another line is
refused). Protocol: `benchmarks/prefill-logit-methodology.md`.

### Pick the gate for what you changed

`MANIFESTS.md` → *Release gates* is the contract (all of it is mandatory for a release, and it says which
tier runs what); this table is the quick answer for a change in flight.

| you changed | gate |
|---|---|
| decode kernels, fusions, the MoE host-expert path, a speculator | same-seed greedy text, then the MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) and `llama-batched-bench -npl 1,4,8` |
| anything whose text you want to quote from the Flash-Next host-expert shape (or the admission rule's arm there) | `scripts/gate-host-expert-text.sh` (N repeats must read one text, the two pins are enforced; `--kld` adds the prefill-logit point) |
| the server's prompt admission: the fair share, the per-step chunk, the skew, or the prefill/decode duty cycle (block 16) | `scripts/gate-prefill-fairness.sh` (including `--self-test`) and `scripts/gate-prefill-duty.sh` on the dense, the `draft-mtp` and a streamed host-expert shape |
| a prefill kernel, an approximate default-on path, an all-reduce arm | `scripts/gate-prefill-logits.sh` |
| a mixed K/V cache pair (**optional block `0101`**): the tile `type_V`, `ggml_cuda_fattn_tile_type_V()`, the mixed instance list, the band predicate or the context gate | `scripts/gate-mixed-kv.sh` on the optional-applied build (+ `scripts/gate-prefill-logits.sh` for the default path) |
| the host-expert pool: a fill/staging path, `expert_host_src()`, the cache's admission, the `-lme`/`--host-experts pool` placement, the loader's source registration, or the `MOE_HOST_POOL_PIN_FAULT` latch | `scripts/gate-host-pool.sh` (its `pinlate` arm is gated on the KLD, never on the text -- a mixed fault state is a ~1e-5 fp32 path, `INVARIANTS.md` r299) |
| the offload decision for a host-expert routed MoE (`ggml_backend_cuda_device_offload_op`, `GGML_MOE_HOST_OFFLOAD`), i.e. which backend a `MUL_MAT_ID` whose expert bank is in the host tier runs on | `scripts/gate-host-pool.sh` **plus a short-prompt arm at the default `GGML_OP_OFFLOAD_MIN_BATCH`** (a <= 31-token prompt, `GGML_SCHED_DEBUG=2`: no `MUL_MAT_ID` node may be placed on the CPU).  The gate's own `-ub 1024`/6-token prompt never reaches the 17..31-token window, so it cannot see this class of regression -- `INVARIANTS.md` r305 items 1-2 |
| the MoE **weighted reduction**: `moe_weighted_reduction_f32*`, the fold's `fusion.red_dst` sum, `ggml_cuda_mwr_fold_serve_ok()`, the down-fold precedence, or `GGML_CUDA_MWR_EXACT` / `GGML_CUDA_MWR_FOLD_FIRST` | `scripts/gate-mwr-exact.sh` (the fold must equal the **fully unfused chain's** text and KLD, the census must show `FOLD == ASIDE` with `DECLINE 0`, and both kill switches must reproduce the pre-r324 build) **plus** `scripts/gate-host-pool.sh` -- the down fold is one of the gate-less `moe_cache_fusion_serve()` sites (MANIFESTS gates 37 + 21) |
| the expert-cache ALIAS resolution (`alias_find_checked`, `table_t.sem`, the redirect's served/declined split), the take-over PROMISE (`taken_cpy`, `moe_cache_repair_copy`), or a fused launch's placement (`moe_cache_fusion_serve`) | `MOE_EXPERT_ALIAS_ROLE_CHECK=0` vs the default + `gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` + a **staged** census arm (`mixed_lanes=0` with the gate-less kinds in the staged column -- a flat staged config reads 0 on the pre-fix build too, so it is vacuous) (MANIFESTS gate 28) |
| a promise's LIFETIME (`moe_cache_pass_end`, when the CUDA backend ends a staging pass) or any of the four promise sites | `scripts/gate-moe-promise.sh` (MANIFESTS gate 34: a stale promise must repair nothing and a same-pass break must still repair, byte for byte) |
| a `-sm tensor` **segmented (fused) tensor upload**, the meta backend's synchronous splice, or the loader's `--load-mode none` chunked upload | `scripts/gate-meta-seg-drain.sh` |
| the expert-cache prefill transport: the staging/gather decision, the pre-staging `moe_cache_in_place` query, `moe_cache_get_table`'s identity branch, or a fused kernel that reads a routed expert table directly | `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` on an **identity** arena (`-device 0,1,2 --sm tensor --ub 2048 --extra "-ncmoe 48"`, 35B-A3B Q8_0) must read KLD 0.000000, plus same-seed greedy text both arms and `gate-host-pool.sh` |
| the server's idle-slot parking / a memory type carrying a second cache | `scripts/gate-park-idle-slots.sh` |
| the node budget (`graph_max_nodes()`), the scheduler's sizing, or a memory module that adds graph nodes/leafs (the KV rotation is the current case) | `scripts/gate-node-budget.sh` |
| the server's context-checkpoint or speculative-checkpoint path | `scripts/gate-ckpt-reuse.sh` / `scripts/gate-spec-ckpt-device.sh` |
| a state save/restore (`llama_io_*`, `state_stage`) | `scripts/gate-state-pipe.sh` |
| the verify-batch alignment / a multi-slot draft width | `scripts/gate-spec-align.sh` / `scripts/gate-spec-width-band.sh` |
| the chunked-GDN prefill threshold (`GDN_CHUNKED_MIN_TOKENS`), `n_rs_batch`'s use as a kernel input, or a speculator's draft width | `scripts/gate-spec-gdn-split.sh` |
| the MTP target h-rows path (`LLAMA_MTP_DEV_ROWS`), the `h_nextn` device buffer, or `llama_set_ctx_state_src` | `scripts/gate-mtp-dev-rows.sh` |
| a qwen4exp hyper-connection op, chain or matcher | `scripts/gate-hc-fusions.sh MODEL.gguf` |
| a GDN gate/beta chain fusion (the mmvf launch, the ksplit epilogue, its memory rule) or the concat reorder's `GGML_CUDA_FUSE_GDN_EPI` branch | `scripts/gate-gdn-epi.sh` (+ `tests/test-fusion-gate-beta-multiseq`) |
| the narrow-row norm fusion (`norm-gated.cu`), its alloc-dep registration, or a new fusion that pre-dispatches a node | `scripts/gate-norm-rows-alias.sh` |
| the hc base copy, `bf16_src1_cache_put()`, or the device `pool_stack` | `scripts/gate-hc-base-copy.sh` |
| the fusion-window range check (`ggml_cuda_check_fusion_memory_ranges()`), a fusion's out-node set, or a host-expert scheduler copy | `scripts/gate-fusion-alias-copy.sh` |
| a fusion's READER COUNT (`ggml_cuda_find_mul_q8_1_matmul`, `ggml_cuda_should_fuse_mul_q8_1`, `ggml_cuda_count_uses`) or anything that makes another node read a fused intermediate (a LoRA, a second consumer) | `scripts/gate-lora-fusion.sh` (MANIFESTS gate 1b: a synthetic adapter's delta must survive the fused chain, and a LoRA-free graph must keep every fusion decision) |
| a fused-op PROBE (`resolve_fused_ops()`), a `cparams.auto_*` / `fused_*` enable decision, or a probe descriptor | `scripts/gate-fused-probe-splits.sh` (a probe that cannot verify an op must disable it and must SAY how it verified, so a bare `enabled` fails; `splits` bounded by the LAYOUT, never text -- `--fixture-only` is the cheap class-wide arm) |
| a grid-stall pad call site, or `ggml_cuda_grid_stall_pad()` itself | `scripts/gate-grid-stall-pad.sh` |
| the MTP draft KV window (block 14 Goal C) | `scripts/gate-spec-draft-window.sh` |
| a drafter's own KV cache: the DFlash/MTP prompt-ingest tail, a restored draft checkpoint, or anything that decides which rows a draft ingests | `scripts/gate-spec-reuse-window.sh` |
| the KV window/region provider, the attendance pool, `is_streaming()` or the sparse decode arm (block 14) | `scripts/gate-kv-residency.sh` |
| the tile FA launcher's decode/verify KV split (`parallel_blocks`), the band's own fixup period `P` (`ggml_cuda_fattn_band_wmma_split`), the band predicates, the KV extent/padding (`get_n_kv`, `LLAMA_KV_N_PAD_*`) or a speculator's verify-batch composition | `scripts/gate-spec-purity.sh` (plain == draft-mtp on q8_0 and bf16 at 3-device `-sm tensor`, plus the width probe's 256-cell-boundary arm; `--control` proves the pre-fix split diverges) |
| a GDN fold's own matcher (`gdn_conv_norm_match`, which layouts a fold accepts) or the conv-state gather deferral (`gdn_deferred_gathers`, `ggml_cuda_gdn_conv_gather_deferrable`) | `test-logits-width-probe` with `KV=q8_0` at `P=3584` and `P>=4096` on a dense and a MoE GDN model, plus a matcher **census**: a decision change moves no hash, so the arm is vacuous unless it shows the refusal (`deferred=1 base=overlap_state`) before and the fold after |
| the Meta split-state cache: the warm walk, a cache version, the resolve step or `calculate_split_state` (block 09) | `scripts/gate-seqpar.sh` plus the `-sm tensor` arms of `scripts/gate-host-pool.sh` and `scripts/gate-prefill-logits.sh`, and identical `-sm tensor` text against the previous build |
| the `--load-mode mlock`/`mmap+mlock` pin, `llama_mlock` itself, or where an `-lme` hole is placed | `scripts/gate-host-read-transport.sh` (its `mlock-hole` / `mlock-nohole` arms) |
| the GTT store, its residency/eviction, the fill transport, the prefill stream/IN-PLACE band, the admission ladder, the lock/budget policy or the shape decision (block 15) | `scripts/gate-uma-experts.sh` (its `inplace` and `pfcp` arms), `scripts/gate-prefill-server-slots.sh` (two slots: one decoding while the other prefills), plus `scripts/gate-prefill-logits.sh` and (discrete) `scripts/gate-host-pool.sh` |
| the allocator's in-place rule (`ggml_op_can_inplace`, which parent's memory a node may take) | `scripts/gate-glu-inplace.sh` (the unit leg plus the compute-buffer saving and ON/OFF text), and the **whole** per-block ladder when the block is an early one |
| the delivery set itself (any regenerated patch) | `scripts/validate-set.sh` |
| anything (before a tag) | `scripts/gate-per-block-build.sh --changed-from <prev tip>` + `scripts/check-gpu-faults.sh` |

Each gate's *why*, its arms and its pass criteria are in `MANIFESTS.md` and the gate's own header; a gate
that another change makes vacuous (a switch that another fusion owns at that width, an arm the config
cannot reach) is called out there too — do not "simplify" a gate to the arm that passes.

### Regenerate the patches (after a fork change)

`scripts/make-patches.sh` (defaults read from `release.json`) `git format-patch --start-number 0`s the block
commits (block 00 keeps prefix `0000`; `git diff <base>..<tip>` yields `rdna-boosts-all.patch`). Always
regenerate from a canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh`: the working
`~/llama.cpp` branch is not the canonical chain, and a raw range there can export upstream commits. Then
`scripts/make-release.sh` (refreshes `release.json`; the `.tip` is the value `scripts/fork-refresh.sh`
prints after a re-cut) and re-verify with `scripts/validate-set.sh`.

### Build the fork

```bash
cd ~/llama.cpp && ~/bin/build-llama-rocm          # default BUILD_DIR=build-rocm, ROCm 10.1.0
# fast loop: cmake --build build-rocm --target llama-cli llama-bench -j 16
# runtime libs: LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
```

`~/bin/build-llama-rocm` is the canonical tree (`build-rocm`); `~/bin/build-llama-rocm-714` is the older
ROCm 7.14 one (`build-rocm-714`) and is only for a 7.14-specific check. **Never pass
`EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=…"`** — both scripts inject a joined `-isystem<rocm>/include` there
and abort the configure if it does not reach the HIP compile line (the include-order guard; without it HIP
headers silently resolve from Fedora's ROCm 7.1). ccache is on by default (`clean` wipes the build dir,
which is free); give it room (`ccache --max-size=30G`, 60G if the disk allows) or a per-block ladder
thrashes. Do **not** outline the FA loader (measured 1.5-2.5 % prefill loss) — that is the sanctioned answer
to "the build is slow" (`archive/work/build-time-regression/`).

Keep every arm in **one** long-lived source path (`scripts/scratch-tree.sh` switches states inside it) —
ccache keys on the absolute path, so a new worktree is a full cold build (5-6 min) while a fresh build dir
at the same path is ~9 s. The per-block ladder's cost and the `--changed-from` shortcut are in
`MANIFESTS.md` → Release gates (gate 4) and the gate script's own header.

### Build the release container images (worker by default)

**The ROCm images are built and pushed off-host** by the worker watcher (`scripts/container-watch.sh` on
`radiance`), which polls for the new release tag and runs `scripts/build-containers.sh --rocm "7.14.1 10.1"
--push --prune` there. So **after pushing a tag nothing else is needed** (the worker picks it up within ~10
minutes), and **do not run the full two-line image build on the primary box during a release** — it pins
every core for tens of minutes. This is the *image* build only; the normal source build and the release
gates stay on the primary box. A fast smoke: `scripts/build-containers.sh --rocm 10.1 --arch gfx1201
--targets server` (no `--push`). Worker one-time setup, the build args, `--prepare-only`, the prune rules
(prune only what was just built, never the base images or the ccache) and the GHCR retention
(`scripts/cleanup-ghcr.py`, 7-day window, current tags always kept) are in [`CONTAINERS.md`](CONTAINERS.md).

## What NOT to do

- Do not `git apply` the concatenated block series (drops hunks); use `scripts/apply-all.sh`.
- Do not hand-edit the committed patches as a permanent drift fix — regenerate and re-verify.
- Do not mix the historical `baseline/*` branches or `block/*` tags with the current `patches/`.
- Do not push anything from the `~/llama.cpp` checkout except the release-time `rdna-boosts` refresh
  to the personal fork (see the Pushing policy). Never push to upstream `ggml-org/llama.cpp`.
- Do not present old docs as current: `MANIFESTS.md`/`BASELINE.md` are contracts, their dated history
  is in `archive/docs/`; current claims are their current-state headers + `patches/README.md` +
  `release.json` + the newest `WORKLOG.md` entry.
- Do not add new WIP experiments to the delivery set, and **never apply anything from `wip/`** or
  `archive/work/` unless the user explicitly asks for that specific item.

## Editing the docs

The docs have a freshness problem by design: historical records are kept, and the CURRENT state lives in the
current-state headers (`patches/README.md`, `README.md`, `MANIFESTS.md`, `BASELINE.md`, `INVARIANTS.md`,
`release.json`) plus the newest `WORKLOG.md` entry. When you change the delivery, update those headers and
add a new dated `WORKLOG.md` entry (newest first) — never edit dated validation records in place.

**Each top-level doc has one job — keep it that way.**  `README.md` describes the *set* (what it is, how to
apply it, what a consumer needs); `MANIFESTS.md` is the apply order + verification contract; `BASELINE.md` is
provenance + the drift policy; `GREEDY-PURITY.md` is the rulebook (claim, rule, numbers per `§N`, with the
narratives in `WORKLOG.md` + `benchmarks/`, and for `§§1-42` in `archive/docs/GREEDY-PURITY-FINDINGS.md`); `TODO.md` is **forward-looking only** (open,
waiting, accepted-limitation, parked — a closed item is one `WORKLOG.md` entry, never a TODO entry);
`ENVIRONMENT.md` is the variable catalogue; `INVARIANTS.md` is the engineering facts (current state — correct
an entry in place, with the `rNN` of the change); `AGENTS.md` is **policy only** (no findings, no numbers of
record, no release narrative).  **Release narrative belongs in `WORKLOG.md` and nowhere else** — do not paste
a release summary into `README.md`, `BASELINE.md` or `MANIFESTS.md`; point at `WORKLOG.md` instead.

**A new fact goes to `INVARIANTS.md`, appended** (its entries are ordered by when they were learned, so
append; and keep the `(rNN)`/PR tag so a reader can find the narrative). A *policy* goes here; a *number of
record* goes to `WORKLOG.md` or `benchmarks/`.  If `AGENTS.md` grows a section that is really a reference
table or a findings list, move it out — that is how this file ended up needing a cleanup.

**WORKLOG dated headers use `YYMMDD.VVV`** (e.g. `## 261011.005 (r47) -- ...`).  `YYMMDD` is the
real local calendar date and `VVV` is a per-date, monotonically increasing 3-digit sequence number in
file order (the topmost/newest entry of a date carries the highest number).  The date is only a
date — never treat it as a version; the revision is the explicit `(rNN)` tag.  A new entry takes
today's real date and the next free number for that date.  Do not invent or extrapolate a future
date: if the work is done today, the header says today.  Session/dev handovers belong under `wip/` or
`archive/docs/`, not at the repo top level.

**Keep this file terse: policies and pointers only.**  No findings, no benchmark tables, no per-release
numbers, no per-block content — those have their own docs above.  If a section is growing into a reference,
move it (that is why `INVARIANTS.md` exists).
