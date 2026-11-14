# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the whole
  climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, expected clean at HEAD `767907a3f`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm;
  ~20-30 s incremental). The build dir currently holds the R17 tree.
- Reference builds: `build-rocm/bin-r277/llama-cli` = the FULL 17-block delivery (use it to tell "this block
  broke it" from "a later block fixes it"); `build-rocm-base2` = unpatched 4f9dfa0e3 with VMM=ON.
- Models: /llm/models/Qwen3.5/4B/Q8_0 (coherence oracle), Qwen3.6/35B-A3B/Q8_0 (host-expert/MTP field config),
  Qwen3.8/Flash-Next/{GSQ-IQ3_XXS,IQ4_NL,Q4_K_M,...} (qwen4exp — there is NO IQ4_XS, that arm skips by design),
  prompts/prose-rdna-boosts.txt.

READ FIRST, IN THIS ORDER
1. archive/work/upstream-rebase-261113/HANDOVER.md   (§0 state, §3 mechanics, §5 decisions, §7 env, §8 traps)
2. archive/work/upstream-rebase-261113/RUNGS.md      (§0 ledger — every rung is DONE — then §R18; §R13-R17 are what the
                                             rungs decided, which gate arms each cleared and which are owed to
                                             an APU box — HANDOVER §5.7/§5.9/§5.10/§5.11)
3. archive/work/upstream-rebase-261113/DECISIONS.md  (D7 is now re-verified; D6/D8/D9/D10/D11/D12/D5-a are closed)
4. archive/work/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
**R00..R17 are done — ALL 17 BLOCKS are on `rebase-r278` at `767907a3f` (tree `69161d3b0`), plus two
rung-repairs.**  Coherence is `100 chars sha=1c5d32ac537d` at every rung since R01, re-verified on the R17 tree.
Per-rung shas, resolutions and gate results: `RUNGS.md` §0 ledger + each §Rnn.  Rung chain in order:
block 00 `43a418c10`, 01 `ab433a7dd`, 02 `50a0b30f8`, 03 `43891459e`, 04 `4c6a5019a`, 05 `22fcde21e`,
**rung-repair `9c94ce139`** (block 05's FA instance matrix: heads 40/72, the FP8 KV column, a typed generator),
06 `c1d8b7c04`, 07 `626e398d2`, 08 `a95c0497b`, 09 `a6809f75b`, 10 `13176810a`, 11 `1d83b50e0`,
12 `75d83f231`, 13 `aca78625a`, 14 `f040d394f`, 15 `f745bbf1b`, **rung-repair `9c940db6f`** (upstream
`#31927`'s h_nextn assert vs the delivery's folded hyper-connection export), 16 `767907a3f`.

Every gate arm that was deferred mid-climb is now CLEARED by the block it was owed to — do not re-litigate any of
them: `gate-host-pool.sh` `tsplit`/`tsplitnosink` (R13, block 12's `moe_cache_pool_backed`), `gate-node-budget.sh`
(R13, 4/4, margin 440 entries per arm), quant-coherence **section E** (R16, block 15's `#30206` fix),
`gate-host-pool.sh`'s `fail`/`late`/`pinlate` (R16, block 15's pageable `ROCm_HostMap` master) and finally its
placement transport (**R17 — the gate is 54/54**, block 16's read queue).

Three things R18 must carry, all recorded:
  * **Squash BOTH rung-repairs** into their owning blocks: `9c94ce139` -> block 05, `9c940db6f` -> block 11.
    The series must be exactly 17 block commits (block 00 keeps prefix `0000`).
  * **APU-only work, owed to `halo`/`soar` (gfx1151)** — it cannot run on this gfx1201 box and must NOT be
    marked passed: `gate-uma-experts.sh` (incl. its `inplace`/`pfcp` arms), `gate-prefill-server-slots.sh`, and
    `D17`'s GTT store-region check (the store is inert on a discrete GPU — 0 `GTT` lines; HANDOVER §5.11).
  * **Known-red:** `gate-kv-residency.sh`'s **multi-device** pool section — a PRE-EXISTING 2-device MTP
    nondeterminism (`D20`/`TODO` #321; `bin-r277` reproduces it, `--spec-type none` and one device are stable).
    The gate was not weakened and must not be; the single-device, window and selftest sections are green.
  * `D7` was re-verified at R14: the device-buffer hidden-state row gather is live on BOTH splits (`target h rows
    on the device: yes`) because block 09's `ggml_backend_meta_device_event_new()` closed the R03 gap.

YOUR TASK
**R18 is the release ladder, and it is NOT a rung.**  It rewrites the delivery (`patches/`, `release.json`, the
fork branch) and it needs the maintainer's explicit go-ahead — if this session has none, do not start it: verify
the state, report, and stop.
1. Read `README.md` §4 and HANDOVER §10 (the ladder, in the order the traps fire) plus §9 (what must be
   re-verified).
2. `scripts/make-patches.sh` from the finished branch, then `scripts/make-release.sh --base 4f9dfa0e3 --tip <tip>
   --tree <tree>`.  **SQUASH BOTH RUNG-REPAIRS into the blocks that own them**: `9c94ce139` into block 05 (its FA
   instance matrix) and `9c940db6f` into block 11 (qwen4exp's folded h_nextn export).  The series must be exactly
   17 block commits (block 00 keeps prefix `0000`).
3. `scripts/validate-set.sh` (fresh tarball at `4f9dfa0e3`, tree equality, sha256s, the Windows lints), then
   `scripts/gate-per-block-build.sh` (a clean build per block commit).
4. The full gate set: coherence on the dense 4B / the 35B-A3B field config / a hybrid recurrent model / the
   qwen4exp class; `gate-prefill-logits.sh --record` from `build-rocm-base2` (VMM=ON, the asymmetry is stated in
   §9) plus the `--sm tensor --device 0,1,2`, `--kv q8_0` and FP8-KV arms; the MTP four-axis protocol +
   `llama-batched-bench -npl 1,4,8`; and every block gate whose code moved — **all of them pass except
   `gate-kv-residency.sh`'s multi-device section** (`D20`/`TODO` #321: the pre-existing 2-device MTP
   nondeterminism; the single-device section, the window section and the selftest are green).
5. **Owed to an APU box (`halo`/`soar`, gfx1151), because they cannot run on gfx1201** (§5.11):
   `gate-uma-experts.sh` (with its `inplace`/`pfcp` arms), `gate-prefill-server-slots.sh`, and `D17`'s store
   region check.  Say so in the release record; do not mark them passed.
6. Records: a `WORKLOG.md` entry (new base/tip/tree + the gate numbers), `patches/README.md` per-block notes
   (incl. the typed FA generator's joint ownership and the two squashed rung-repairs), `BASELINE.md`,
   `INVARIANTS.md` appended facts, `ENVIRONMENT.md` for every variable that appeared/vanished, `TODO.md`, and
   move this campaign to `archive/work/upstream-rebase-261113/` (**no `wip/` stub**).
7. Fork refresh (`scripts/fork-refresh.sh --build --gates --push` — **personal fork only, never upstream**) and
   the off-host container images, **only with the maintainer's go-ahead**.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Overlaps resolve as "both sides"
  unless a measurement says otherwise (D1, D2, D5, D12 set that precedent).
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A conflict that resolves "naturally" because upstream removed the same thing is a decision, not a silent
  delete: write it in the rung log.
- Never weaken a gate and never simplify it to the arm that passes; if an arm needs a later block, name the
  block and the symbol.
- If you run out of room: `git cherry-pick --abort`, record what you learned in RUNGS.md, update HANDOVER §0,
  and stop. Do not leave a half-resolved rung.

IF R18 COMPLETES
The climb is over: the delivery is re-based and released.  Report the new base/tip/tree, the gate numbers, the
APU-deferred set (which must be labelled "not run on this box") and the known-red `D20`/`TODO` #321 item to the
maintainer, then move this campaign to `archive/work/upstream-rebase-261113/` with no `wip/` stub.
```
