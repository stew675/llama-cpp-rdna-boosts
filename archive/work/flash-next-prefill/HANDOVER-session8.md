# HANDOVER — session 8: promotion, part 2 — the **C fold** (split by territory), then validate, remote sanity, tag

**Read this first, then `HANDOVER-session7.md` §-OUTCOME, `REVIEW-squashed.md` §10-§12 and `RESULTS-session7.md`
§6-§7.**  Session 7 did the final review, the gates below, and measured *why* the fold cannot be a one-block
amendment.  Session 8 does the fold the maintainer chose — **C: distribute the campaign's hunks to the blocks whose
territory they belong to** — and then the cut.  Nothing is tagged or pushed until the end of this list.

## 0. The ask, in order

1. **The C fold** (§3).  Acceptance is one command: `git diff <new-tip> a666f8637` must be **empty** — the promoted
   set must produce the tree we validated, byte for byte.
2. `scripts/make-patches.sh` from the new canonical chain, `scripts/make-release.sh`,
   `scripts/validate-set.sh`, `scripts/check-rung-file-sets.py` (§4).
3. The docs owed at promotion (§5) — the list is mechanical, and one item changed this session
   (`MOE_HOST_POOL_STAGE_STANDDOWN` is deleted, `moe_cache_stage_ready` retired).
4. **The remote sanity on `fingon` (gfx1100/RDNA3) and `halo` (gfx1151/RDNA3.5) on the CUT tree** (§6) — the
   maintainer's explicit requirement, before any tag.
5. The tag / release commit, `scripts/fork-refresh.sh --build --gates --push` (personal fork only), the container
   images by the worker + `--verify-only`, the `WORKLOG.md` cut entry.

## 1. State at handover

| | |
|---|---|
| `~/llama.cpp` | branch `wip/prefill-slot-pingpong-squashed`, tip **`a666f8637`** — ONE commit on `af7a57ba8`, clean, `build-rocm` built from it |
| patch | [`SQUASHED-campaign-r368.patch`](SQUASHED-campaign-r368.patch) **5327 lines / 18 files**, verified identical to the commit |
| delivery repo | `main` pushed; `release.json` **untouched**: `base` `919794500`, `tip` `af7a57ba8`, 18 official blocks + `0300` |
| the campaign in one line | the staging transport: a 1-3 buffer PIPELINE in the slab's transport region (derived size, cushion above the decode band, depth from the cache's settled arena), the stage-vs-gather SELECTOR, the Meta/tensor-split transport wired in, and the VMM pool's fail-soft growth |
| `/tmp/cut` | a **pristine canonical chain**: worktree at `919794500` + `apply-all.sh` → 20 commits ending at `af7a57ba8` (deterministic).  This is the starting point for the fold.  **Do not `apply-all.sh` on a checkout of the tip** — it starts at `release.json.base` (`919794500`), never at `af7a57ba8` |
| `/tmp/base-wt` | worktree at `af7a57ba8`, `build-rocm-base` (the MTP base arm's build) |
| local MTP runs | **in flight / chained** (§2).  `runs/mtp4-cand/summary.tsv` (candidate) and `runs/mtp4-base/summary.tsv` (base, chained behind the candidate).  `mtp4axis.sh` is the runner |
| `fingon` / `halo` | builds **in flight** (§6).  Both fetched over **HTTPS** (they have **no GitHub SSH key**): `git fetch https://github.com/stew675/llama.cpp.git rdna-boosts` and the same for the delivery repo |

## 1b. DO NOT (while the MTP chains are in flight)

* **Do not rebuild `~/llama.cpp/build-rocm` or `/tmp/base-wt/build-rocm-base`.  Do not run any other bench/arm.**
  The four-axis runner launches `llama-cli` once per arm, so a rebuild mid-chain makes the remaining arms measure a
  different binary and the gate is void.  The C fold is git-only and is safe to do meanwhile; the *cut* and any
  rebuild it wants must wait for `pgrep -f mtp4axis` to be empty.
* Read the verdict from `runs/mtp4-cand/summary.tsv` and `runs/mtp4-base/summary.tsv` once both are complete.
  The runner **truncates its own summary at start**, so a short file means "still running", not "few rows" — check
  `pgrep -f mtp4axis` first, and if the harness killed the chains, re-run `./mtp4axis.sh cand
  ~/llama.cpp/build-rocm` then base (never both at once).
* The candidate's **f16 cache is already complete and clean** (2026-11-27): `none` == `draft-mtp`, byte-identical on
  all four axes — R `218515b89b31` 41.6 -> 80.1 t/s, K `a87c4318b649` 41.8 -> 176.4, P `20ff19bc33b1`
  41.3 -> 120.0, C `48067cb92646` 42.4 -> 150.5.  The q8_0 cache was in flight; the base arm is chained behind the
  base build.

## 2. What is already green (do not re-run)

The ladder was run in session 7 on the candidate vs the delivery tip; every result is in `WORKLOG.md`
**261127.014-016** and `RESULTS-session7.md`.

| gate | result |
|---|---|
| `gate-prefill-logits` — dense canonical (27B Q8_0, `device 0`) | **PASS** KLD `0.000000`, top-p 100 % |
| `gate-prefill-logits` — host-expert 1 GPU (`-ncmoe 48 --host-experts pool`, `-ub 8192`, chunks 12) | **PASS** KLD `-0.000000`, top-p 100 %; base **re-recorded from `af7a57ba8`** (the old one was campaign-era) |
| `gate-prefill-logits` — host-expert 2-dev `-sm tensor` (`--pin MOE_EXPERT_HEATMAP=load,MOE_EXPERT_CACHE_MIB=8192`) | **PASS** KLD `-0.000000`, top-p 100 %; **caveat**: the Meta staged branch did not engage under `llama-perplexity`, so this point judged the gather regime |
| `gate-host-pool` | **PASS** 59/59 |
| `gate-kv-residency` | **PASS** |
| `gate-fused-probe-splits` | **PASS** with `EXTRA_ARGS="-ncmoe 48 -lme mirror -ot per_layer_token_embd.*=CPU" SPLITS_MAX=200` (worst 147).  Its default bound is 1 and **inapplicable to a `-ncmoe` arm**: the *tip* reads 144 splits on the same graph, so the count is pre-existing — record that as a gate TODO |
| `gate-moe-promise` | **PASS** 3/4 runs.  **One intermittent abort** (exit 134) on the `MOE_EXPERT_TAKEOVER_FAULT=1` arm, not reproduced in 3 re-runs, text identical each time; log lost (the gate defaults to `mktemp`).  The maintainer accepted it: **record it, proceed** |
| `check-shlib-symbols` on `build-rocm` **and** a `GGML_HIP_NO_VMM=ON` build | **PASS** both, 16/16 libs each |

**MTP four-axis (mandatory for this campaign — it alters the buffers MTP runs on).**  Protocol:
`benchmarks/mtp-adaptive-methodology.md` + the r328 record: **4 axes** (`reasoning` with `--reasoning on`, then
`recall`, `prose-rdna-boosts`, `code-python` with it off), **`none` vs `draft-mtp --spec-draft-n-max 7`**, on
**3 devices `-sm tensor`**, **`-n 3000`**, greedy, `ctk/ctv` = f16 **and** q8_0, model
`Qwen3.8-27B-Q8_0` (MTP head embedded, no `-md`), text on stdout (no `-lv` there) and acceptance from `-lv 4` on
stderr.  `./mtp4axis.sh TAG BUILD_DIR [CACHE…]` runs it; the two arms are chained already.  Judge: acceptance
> ~0.45 at pos 1, `none == draft-mtp` **byte-identical per axis and cache**, and MTP ≥ plain.

## 3. The C fold (the maintainer's decision, 2026-11-27)

**Why not one block**: the campaign is written against the tip; un-applying the blocks that come *after* the ones
it touches conflicts in — measured with `git revert --no-commit` on top of the campaign tree — **block 11: 16
files, block 12: 20, block 13: 3, block 14: 4, block 15: 7** (blocks 16, 17, 0300 un-apply cleanly).  So the campaign
is a cross-cutting layer, and the fold distributes it by **territory**:

| block | gets |
|---|---|
| **09** — host-resident MoE: iface, scheduler, expert cache, staging rings | the scheduler **pipeline + selector** (`ggml-backend.cpp`), the `stage_release`/`moe_cache_residency` iface members, the **Meta transport** (`ggml-backend-meta.cpp`), the `moe_cache_in_place` restoration and the `moe_cache_stage_ready` **retirement**, the cache's residency mirrors |
| **10** — the slab, the fusion dispatcher, the async input ring | `ggml-cuda-slab.cu`, `ggml-cuda-vmm.h`, the transport region / cushion / depth authority / `stage_candidate` / `transport_reset`, the **VMM pool's fail-soft growth**, the `h2d_stage_*` members in `common.cuh` and their hooks in `ggml-cuda.cu` |
| **12** — campaign remainder (CPU wins, backend iface fills, hooks) | the `ggml-cpu.cpp` / `ggml-rpc.cpp` iface rows, `ggml-cuda-moe-cache-hooks.{cu,cuh}` if they belong to its territory rather than 09's |
| 05/06/14/15 | nothing expected; check `moe-expert-cache.cu` hunk by hunk — the transport *decision*/fill belong with the cache engine, the GTT parts stay in 15 |

**Method that works** (the two that do not are below it):
1. In `/tmp/cut`, checkout block **09**; apply only the campaign's hunks for 09's file set
   (`git diff af7a57ba8..a666f8637 -- <files> | git apply -3`), resolve conflicts *in favour of the campaign* (a
   conflict in 09's territory means the hunk belongs to 09), `git commit --amend` keeping block 09's subject.
2. `git rebase --onto <new-09> <old-09> rdna-boosts`; the same for 10, then 12.
3. **Acceptance**: `git diff <new-tip-tree> a666f8637` empty, `git status` clean, and the chain's commit count
   unchanged (20).
4. `git diff af7a57ba8..<new-tip>` must be **empty** too (the chain's tip tree is unchanged — only the attribution
   moved).

**Do NOT**: `git apply -R -3` in a loop (it cascades: once the index is conflicted, later applies report
`does not exist in index`); whole-file `git restore --source=a666f8637` at a block (that pulls the *later* blocks'
content into the earlier block).  `git revert --no-commit` is the correct git machinery when you need to un-apply.

## 4. The cut

`scripts/make-patches.sh` (from the amended chain, not from `~/llama.cpp`'s working branch) →
`scripts/make-release.sh` → `scripts/validate-set.sh` (checksums, fresh-tarball strict apply, file-set
reconciliation) → `scripts/check-rung-file-sets.py` (per rung **while** porting, then the whole set; a deliberate
retirement is a row in `patches/FILE-SET-RETIREMENTS.tsv` — the `moe_cache_stage_ready` removal may need one).
The release prefix stays derived: 18 official blocks + `0300`.

## 5. Docs owed (updated for this campaign)

`ENVIRONMENT.md`: `GGML_SCHED_PIPELINE[_BUFS]`, `GGML_SCHED_STAGE_SELECT`, `GGML_SCHED_STAGE_EXPOSE_US`,
`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS`, `MOE_HOST_POOL_STAGE_BULK[_GBPS]`,
`MOE_HOST_POOL_STAGE_MERGE_FLOOR[_GBPS]`; **remove** `GGML_CUDA_SLAB_RING_MIB` (gone from the code — the region is
derived), `GGML_SCHED_STAGE_TABLE_REF_MB` (gone).
`DIAGNOSTIC.md`: `GGML_SCHED_PIPELINE=0`, `GGML_CUDA_SLAB_TRANSPORT_FALLBACK`,
`GGML_SCHED_STAGE_SELECT=0`, `GGML_SCHED_DEVGATHER=0|2` (now the two controls on BOTH tiers), `GGML_MOE_STAGE_DEBUG`,
`GGML_STAGE_META_REDIRECT` (**A/B only — it corrupts a host-offloaded expert table by measurement**),
`MOE_EXPERT_PREFILL_IN_PLACE` unchanged.  **`MOE_HOST_POOL_STAGE_STANDDOWN` is deleted** — do not document it.
`patches/README.md`: the block notes for 09/10 (what each now carries), the staging-ring → pipeline vocabulary, the
measured curve (§2's numbers), and the tensor-split section (**never say the gather "owns" that tier**: the transport
stays selectable whatever the policy is — the maintainer's instruction).
`INVARIANTS.md`: the derived region; the cushion; the depth authority; the `resident=`/`arena_region=` naming; the
over-carve/eviction-refill trade; the capture/consume pairing; **the two staging forms** (slots = in-place; Meta
`stage_input` = fill + drain) and the measured tensor-split curve; the reproducible bench flags
(`-lme mirror --lazy-mode off`).
`TODO.md`: (a) the `gate-fused-probe-splits` `SPLITS_MAX` bound does not model a `-ncmoe` arm; (b) the
`MOE_EXPERT_TAKEOVER_FAULT` arm aborted once and did not reproduce; (c) the exposure rule stages at 80 % coverage
where the gather measured +2.1 % — the threshold needs a per-tier calibration; (d) the follow-up campaign
`wip/tensor-split-stage-overlap/` (drain overlap + in-place, both gated on clean-text/tail checks).

## 6. The remote sanity (maintainer's requirement — after the cut, before the tag)

Both boxes already have: the delivery repo at `origin/main`, the fork fetched at `af7a57ba8`, and the campaign
patch applied as a branch (`cand`).  Builds were started in the background in session 7:

| box | GPU | reference build | candidate build |
|---|---|---|---|
| `fingon` | gfx1100 (RDNA3) + gfx1036 iGPU | `build-rocm` at `af7a57ba8` (already built) | `build-rocm-cand` (commit `2bfd09a95`) |
| `halo` | gfx1151 (RDNA3.5) | `build-rocm-tip` at `af7a57ba8` | `build-rocm-cand` |

Check `/tmp/cand-build.done` on each (`/tmp/tip-build.log`, `/tmp/cand-build.log`).  Then, **for the cut tree**
(`apply-all.sh` from `release.json.base` in a scratch checkout, then rebuild) or for the identical campaign branch
already there, run per box:

```sh
# both arms, same seeds, and the text must MATCH the reference build (each box is its own reference:
# the arch differs, so a cross-box hash is meaningless)
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
HIP_VISIBLE_DEVICES=0 "$BIN" -m $M -ngl 99 -ncmoe 30 -lme mirror --lazy-mode off --load-mode mmap \
  -c 4096 -fa on -ub 2048 -p "The capital of France is" -n 32 --seed 42 --temp 0 \
  --no-display-prompt --single-turn
```
plus one small dense model for the inert path (halo has `Qwen3.5/4B/Q8_0`; fingon does not — use the 35B only there).
Compare with `extract-generated.py` **and judge the tail** (`flash-next-prefill/tailcheck.sh`), take no `-lv` for the
text arm, and check `scripts/check-gpu-faults.sh` over the window.  A different *rate* is fine on other arches; a
text difference or an abort is not.

## 7. Traps this session paid for

* `apply-all.sh` starts at `release.json.base` (`919794500`), never at the tip; it also refuses an existing
  `rdna-boosts` branch (`git branch -D rdna-boosts` first — it is disposable by policy).
* The remote boxes have **no GitHub SSH key**: use `https://github.com/...` for `git fetch` there (push targets are
  still the SSH URLs on this box only).
* `~/bin/build-llama-rocm` rejects an absolute `BUILD_DIR` — use a relative name from the tree you are in
  (`BUILD_DIR=build-rocm-cand`), which is how the reference builds were kept alive.
* A text/coherence arm takes **no `-lv`** (the extractor does not strip log lines, so a `-lv 4` "hash" includes
  timestamps) and the **tail** is judged, not the head — `tailcheck.sh` does both checks.
* Never run two MTP/bench arms at once; the base and candidate arms are chained for exactly that reason.
* The Meta staged branch will not engage under `llama-perplexity`; its purity evidence is the **byte-identical clean
  text** (321 chars `sha=184dc307181b` at 40 / 80 / 100 % coverage), not a KLD point, until the follow-up campaign
  changes that.
