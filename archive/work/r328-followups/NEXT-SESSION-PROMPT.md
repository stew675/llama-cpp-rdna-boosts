> **SPENT — do not run this prompt.**  The session it describes ran on 2026-11-21 and closed every row it
> names; the outcomes are in the closure banner at the top of `HANDOVER.md` and in `WORKLOG.md` 261121.007.
> The text below is kept as the record of what a cold start would have been told.

# Paste this into a fresh session

> Work through `~/llama-cpp-rdna-boosts/archive/work/r328-followups/HANDOVER.md` — read it top to bottom first, then
> verify the state before touching anything:
>
> ```bash
> cd ~/llama-cpp-rdna-boosts && git pull --ff-only && git log --oneline -1 && git status --short
> jq -r '"release: "+.release+"\ntip: "+.tip+"\ntree: "+.tree' release.json
> cd ~/llama.cpp && git branch --show-current && git log --oneline -1 && git status --short
> ```
>
> Expect: delivery `main` at `ecfce7bd` or later (clean); `release.json` = `v17-919794500-r328`, tip
> `6d9d54238b402acc7d4b3a680ca585e771ceb00b`, tree `0c46248f7371a0eee84a980a1e16ad25a6fbd38f`; `~/llama.cpp` on
> branch `rdna-boosts` at `6d9d54238` (the re-cut r328 chain), clean.  The pre-cut ladder is branch
> `rebase-r328` (`73a74454b`) and the pre-squash tip is `backup-pre-squash-r328` — both are historical, do not
> re-derive from them.
>
> **`v17-919794500-r328` is already released and published** (tag, GitHub Release assets, fork `rdna-boosts`,
> container images by the `radiance` worker).  The MTP four-axis protocol, `gate-spec-purity.sh` and both other
> arches have since been validated and **all pass** (§2 of the handover) — do not re-run them "to be sure"
> unless your change touches decode/verify or the FA launcher.
>
> **Your job is `HANDOVER.md` §3, rows F1-F6, and then the `r329` gate in §4.**  In this order:
>
> 1. **F1 — `TODO` #357, gfx1151's GTT sizing-fill census.**  The one real open error.  Reproduce on `halo` with
>    one minimal arm and `cat -A` on the captured stderr, then decide **instrument vs transport** (the handover
>    spells out the two branches and the evidence already gathered).  It is probably a log-emitter or a
>    gate-expectation problem, not a queue regression — prove which before changing anything.  A delivery change
>    here is a **block amendment**: report it and get the maintainer's agreement before folding it in.
> 2. **F2 — the rest of `TODO` #354**: re-run `gate-prefill-server-slots.sh` on `halo` with a longer request
>    timeout (the harness aborted on a timeout, the arm never produced a verdict), record a **per-box**
>    prefill-logit base on `halo` and `fingon` (`MODEL=` explicit on `halo`), and either decide or record the
>    gfx1100 `gate-spec-align` +7.2 % vs the 10 % `RECOVER`.
> 3. **F3 — `TODO` #356**, the block-02 fix: align the **arena-band-derived** draft caps in the same call, so the
>    host-expert pair stops splitting 2 draft-length steps; then `gate-spec-align.sh` on gfx1201 and `fingon`.
> 4. **F4-F6** — the container image smoke (`#352`, check the worker's log first), R06.2's routed-MMB A/B, and a
>    decision on `gate-node-budget` (fetch phi-4 or record the gap).
> 5. Only then the `r329` sequence in §4 (fixups folded into owning blocks → gates → prefill-logit `--record` →
>    MTP + spec-purity → `make-patches`/`make-release`/`validate-set` incl. re-cutting `optional/` →
>    `fork-refresh --build --gates --push` → tag → worker images → WORKLOG/`patches/README`/`INVARIANTS`/`TODO`).
>
> **Non-negotiables** (all of these bit in the r328 session):
> * **Measure before you tag.**  A re-base or a decode change runs the four-axis MTP protocol
>   (`benchmarks/mtp-adaptive-methodology.md`), `gate-spec-purity.sh` and the prefill-logit gate **before** the
>   tag, not after.  That ordering mistake is why this handover exists.
> * **A red gate is not a delivery defect until r327's own binaries say so**: `LLAMA_CLI=~/llama.cpp/build-rocm/bin-r327/llama-cli
>   ./scripts/gate-<name>.sh` first.  Two "release blockers" this session were stale anchors, one of them because
>   the gate never anchored its own reference arm.
> * **No `-lv` on a text arm** (`-lv 4` only for a census; it lands on stderr, so one run can serve both).
>   Pin `HIP_VISIBLE_DEVICES` on every multi-device arm.  One GPU workload at a time, and check it with
>   `pgrep -cx llama-cli` (not `pgrep -af`, which matches your own command line and prints the whole prompt).
> * **`/tmp` is tmpfs** (46 GiB of RAM here): large artifacts to `~/.cache/rdna-boosts/gates`.  Never point a
>   KLD base, a state dump or a model at `/tmp`.
> * **Poll long jobs without blocking**: `nohup setsid … &`, then
>   `for i in $(seq 1 60); do grep -q <sentinel> && break; sleep 15; done`.
> * **Fold fixes into their owning block** with one `git rebase -i <base>` pass — never a trailing commit, never
>   inside `0300`.  `scripts/apply-all.sh` is the only apply flow; never `git apply` the concatenated series.
> * **Never push to upstream.**  Permitted: this delivery repo, and the personal fork's `rdna-boosts` refresh
>   (`scripts/fork-refresh.sh --push`, force-with-lease) at release time only.
> * **Report findings before any delivery-affecting commit**, and **ask before creating, renumbering or promoting
>   a block** — a green gate, a validated win or a passing PR is not approval.
> * **Never run the full container image build on the primary box** — the `radiance` worker watcher does it after
>   the tag.
>
> Evidence for every row is in `archive/work/r328-followups/logs/` (29 logs) — quote from those instead of re-running a
> 20-minute gate to read a number.  If a row turns out to be answered by an existing record, say so and retire it
> rather than re-measuring.
