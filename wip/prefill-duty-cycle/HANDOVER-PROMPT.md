# Handover prompt: finish the prefill/decode duty cycle and cut the next release

Paste the block below into a fresh session.  Everything it needs is in the repo, and the records it points at
are the durable version of this handover.

---

You are continuing work in the llama.cpp delivery repo `~/llama-cpp-rdna-boosts` (branch `main`) and its fork
`~/llama.cpp`.  The current release is `v16-4f9dfa0e3-r300`.  Your job is to finish, validate, promote and
retire the remaining server prompt-admission work, close three TODO items, and cut the next release.

Read before touching anything, in this order:

1. `AGENTS.md` (mandatory policy: block numbering, WIP/promotion rules, the pushing policy, the default-on
   policy, "report findings before shipping")
2. `INVARIANTS.md` (engineering facts; do not re-derive them)
3. `archive/work/session-scheduling/README.md` (what shipped in r300, all nine validated layouts, the three
   measurement traps, and the retracted findings)
4. `wip/prefill-duty-cycle/README.md` (the work in flight, its measurements and its open questions)
5. `TODO.md` `#338`, `#339`, `#340`
6. `MANIFESTS.md` gate 30 and the Release gates section

## State you inherit

* r300 shipped **fair-share prompt admission** in **block 12**.  `release.json`: tip `24211c158`, tree
  `4baf1b67`, `n_blocks` 16.  Delivery `main` is at the tagged release commit.  The personal fork's
  `rdna-boosts` branch is at `24211c158`.  The gate is `scripts/gate-prefill-fairness.sh` (+ `.py`), and it
  must stay green.
* The **duty cycle** is an experiment.  It lives only on branch `sched-prefill-duty` in `~/llama.cpp`
  (commits `77163bf79`, `014a7957c`), based on the r300 tree.  It is not in `patches/`, `release.json` or any
  release.  `~/llama.cpp`'s worktree is currently on that branch and `build-rocm` contains it.  The
  `rdna-boosts` branch there is clean at the released tip.
* The code is one file, `tools/server/server-context.cpp`.  r300's part is `share_prompt_budget()` plus three
  variables; the experiment adds `LLAMA_SERVER_PREFILL_DUTY`.

## Deliverables, in order

1. **Finish the duty cycle.**  Settle the five open questions in `wip/prefill-duty-cycle/README.md`.  The one
   that is most likely to change the design is the arena: a wider prefill burst against decode-only steps is a
   sharper wide/narrow alternation than r300, and `TODO` #317 plus r292 exist because that costs prefill on
   host-expert shapes.  Use the `-lv 4` MoE census and `scripts/gate-prefill-server-slots.sh`, not latency
   alone.  Also finish the MTP composition measurement that was started and abandoned.
2. **Close `TODO` #338.**  Two parts.  (a) The gate's short-TTFT ratio must be derived from `L/(k*chunk)` with
   `k` the short prompt's measured step count, instead of a fixed 3x, because a 4.4k prompt at chunk 512 has a
   ceiling of 2.86 and a 13.2k prompt does not.  (b) The attendance-pool arm reads 2.6x against a 5.7x ceiling
   and the carrier is **not established**.  The discriminating experiment is written down: time slot 1's
   admission from the `-lv 5` `fair share: budget=.../added=...` line.  If it is admitted in step 1 and still
   takes 5 chunks the carrier is the pool; if it is admitted late it is the scheduler.  Do not guess.
3. **Close `TODO` #339.**  Block 12 now carries the campaign remainder plus fair-share admission plus (if it
   ships) the duty cycle.  Split it.  The maintainer has deferred this decision to this session, so pick the
   structure (a new official `0016`, a temporary `03xx`, or leaving it) and record the reasoning.
4. **Promote whatever survives.**  A new environment variable needs `ENVIRONMENT.md` with its default and class
   (kill-switch / opt-in / tuning / diagnostic).  A new policy needs a `MANIFESTS.md` gate entry and a gate
   arm.  `patches/README.md` needs the per-block note.  `WORKLOG.md` needs one dated entry, newest first, with
   a `YYMMDD.VVV` header and today's real date.
5. **Retire the campaign.**  Once promoted, closed or refuted, move `wip/prefill-duty-cycle/` to
   `archive/work/` and delete the `wip/` entry.  Do not leave a redirect stub.  Update `wip/CAMPAIGNS.md` and
   remove the row.
6. **Cut the release.**  Order matters:
   * build the canonical chain at `release.json.base` with `scripts/apply-all.sh`;
   * `scripts/fork-refresh.sh --build --gates` re-cuts `rdna-boosts` and **prints the authoritative tip**;
   * regenerate `patches/` from that re-cut with `scripts/make-patches.sh ~/llama.cpp <base> <printed tip>`,
     then `scripts/make-release.sh --tip <printed tip> --tree <tree> --release v16-<base>-r<N+1>`, plus the
     optional artifact trees, which must be re-measured against the new default tree;
   * `scripts/validate-set.sh` GREEN, `scripts/gate-per-block-build.sh --changed-from <prev tip> --fast` GREEN;
   * push the fork (`git -C ~/llama.cpp push --force-with-lease origin rdna-boosts`), then the delivery
     `main` and the tag;
   * the ROCm container images are built by the cron watcher on `radiance` from the tag.  Do not build them on
     the primary box.

## Traps that already cost time.  Do not re-derive them

* **Never compare commit ids to decide whether two trees match.  Compare trees.**  A locally rebuilt chain
  (amend plus rebase) has the right tree and the *wrong* ids, because `git rebase` re-stamps committer dates
  while `apply-all.sh` derives each committer date from the patch's author date.  Always take the tip from
  `apply-all.sh` / `fork-refresh.sh` and regenerate the patches from that re-cut, or the `From <sha>` headers
  name commits that do not exist.
* **Pin the slot on any two-slot measurement.**  `update_slots()` prioritises the lowest slot index and the
  server assigns slots by LRU, so an unpinned arm measures whichever side of a race it gets.  Measured: the
  same starvation arm read 0.828 s unpinned against 1.907 s pinned.  Both harnesses pass `id_slot`; keep that.
  Every two-slot number in the archives taken before that change is suspect.
* **Measure a "lone prefill" on its own server.**  In a shared arm it reported +25 % that was pure arm
  ordering; isolated on four fresh loads the two configs read 2065 to 2073 t/s, identical.
* **Bound the `check-gpu-faults.sh` window with `--since`.**  Unbounded, it fails on the previous day's faults.
* **`setsid nohup` servers outlive an aborted tool call.**  One of mine held 74 GiB of VRAM across three
  cards and made the maintainer's own `runme` fail with an OOM that looked like a config error.  Before
  blaming parameters, check `pgrep -af llama-server` and the per-card `mem_info_vram_used`.
* **The flag is `--n-cpu-moe`, not `-ncmoe`**, and the field `runme` does not use it at all: experts stay
  resident there.  A streaming arm has to add it deliberately, and `MOE_EXPERT_CACHE_MIB` if you want real
  streaming, since the default arena holds the whole 40 GiB set.
* **An empty environment variable is not "unset".**  `LLAMA_SERVER_PREFILL_CHUNK=` is `atoi("") == 0`, i.e.
  chunk OFF.  Use `env -u` or `unset`.
* **A harness that cannot tell "not ready" from "dead" will produce numbers.**  Poll the PID, not only
  `/health`.
* **Do not quote a gate threshold that was tuned to make an arm pass.**  Three of this campaign's numbers were
  corrected after a false verdict.  If a threshold moves, the reason goes in the record.

## Acceptance

* `TODO` `#338`, `#339` and `#340` are closed, each retired into one `WORKLOG.md` entry, and `TODO.md` carries
  no stale rows.
* `wip/prefill-duty-cycle/` is gone, its record is in `archive/work/`, `wip/CAMPAIGNS.md` is updated.
* `scripts/validate-set.sh` GREEN, `scripts/gate-per-block-build.sh` GREEN, `scripts/gate-prefill-fairness.sh`
  GREEN including `--self-test`, and whatever new gate the promotion adds.
* A new release tag is pushed, the delivery `main` is pushed, the fork's `rdna-boosts` is force-pushed to the
  recorded tip, and `release.json` names that tip.
* A short GitHub discussion announcing the change, in the style of discussion 324: what it does, how to use it,
  the measured results, and what it does not do.  No em dashes.

Report findings, open questions and anything you had to correct **before** the release commit and the tag.
