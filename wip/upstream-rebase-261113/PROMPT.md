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
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §3 mechanics, §5 decisions, §7 env, §8 traps)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger — every rung is DONE — then §R18; §R13-R17 are what the
                                             rungs decided, which gate arms each cleared and which are owed to
                                             an APU box — HANDOVER §5.7/§5.9/§5.10/§5.11)
3. wip/upstream-rebase-261113/DECISIONS.md  (D7 is now re-verified; D6/D8/D9/D10/D11/D12/D5-a are closed)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R14 are done on `rebase-r278`: blocks 00-08 (R01-R09), the rung-repair `9c94ce139` (block 05's FA instance
matrix: heads 40/72, the FP8 KV column, a typed generator), block 09 `a6809f75b`, block 10 `13176810a` (which
moved the whole fusion dispatcher into `ggml-cuda-try-fuse.cu`), block 11 `1d83b50e0` (qwen4exp;
GGML_OP_COUNT is 106 after R13), block 12 `75d83f231` (campaign remainder; tree `d862165c5`) and block 13
`aca78625a` (MTP draft vocabulary subset; tree `061a71a98`) and block 14 `f040d394f` (`-sm relaxed-tensor`,
TEMPORARY; tree `0c8a1fd58`), block 15 `f745bbf1b` (the streamed KV), the **R16 rung-repair `9c940db6f`**
(tree `14f99e579`) and block 16 `767907a3f` (the UMA GTT store; tree `69161d3b0`).  Coherence is `100 chars sha=1c5d32ac537d` at every rung since R01, re-verified on the R16
tree.  Block 14 is still load-bearing (upstream's `#31081` deny is verbatim at `4f9dfa0e3` and nothing in
`a691b2345..4f9dfa0e3` touches `llm_arch_supports_sm_tensor`) — do not drop it in passing; HANDOVER §5.8.
R16 cleared three gate arms and added a rung-repair (§5.9); R17 cleared the **last** one — `gate-host-pool.sh`
is **54/54** now — and block 16's own gates (`gate-uma-experts.sh`, `gate-prefill-server-slots.sh`) are **APU
gates that cannot run on this discrete box** (§5.11), as is `D17`'s store-region half.  `D7` was re-verified at R14: the device-buffer hidden-state
row gather is live on BOTH splits now (`target h rows on the device: yes`), because block 09's
`ggml_backend_meta_device_event_new()` closed the R03 gap — the `upstream/` candidate is purely upstream's.

R13 cleared both block-12 gates it owed: `gate-host-pool.sh` `tsplit`+`tsplitnosink` (pool built on a 2-device
`-sm tensor` split, exchange INERT, both arms `171 chars sha=8c39335bbceb`) and `gate-node-budget.sh` (4/4, every
arm `a1b7eb2ee7a6`, margin 440 entries: 2064/2504 q8_0, 1504/1944 f16 and rotation-off).  The pool gate is
**42/54** and its 12 remaining checks belong to LATER blocks — do not re-litigate them, do not weaken the gate:
  * `fail`, `late`, `pinlate` (8 checks) -> R16: `host_expert_buft()` still returns the PINNED `ROCm_Host`; block
    15 inserts the `want_mapping` -> `ggml_backend_dev_moe_host_master_buft(dev)` (`ROCm_HostMap`) branch, which
    is what makes the master pageable and the pool's fault/latch hooks reachable.
  * `ref`'s placement transport + placement queue, `qd1`, `bulkbuf` (4 checks) -> R17: `MOE_EXPERT_FILL_QD`,
    `MOE_HOST_POOL_BULK_DIO` and the placement read queue are block 16's.
  * Still open from R12: `gate-qwen4exp-quant-coherence.sh` section E (sparse QSA) -> R16 (block 15 replaces
    `ggml_rope_get_truncate(dst)` on the fused INDEXER_SCORE node, which upstream #30206's assert rejects).

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
APU-deferred set and `D20`/#321 to the maintainer.  It is not a rung: `make-patches.sh` (squashing BOTH rung-repairs — `R09r` into block 05 and the
R16r h_nextn repair into block 11), `make-release.sh`, `validate-set.sh`, `gate-per-block-build.sh`, the
prefill-logit record + arms, the coherence set (dense 4B, the 35B-A3B field config, a hybrid recurrent model,
the qwen4exp class), the MTP four-axis protocol, `gate-rebase-release.sh`, `check-gpu-faults.sh`, the fork
refresh (personal fork only) and the records.  Known-red items to carry: `gate-kv-residency.sh`'s **multi-device**
pool section (`D20`/`TODO` #321 — pre-existing 2-device MTP nondeterminism; nothing to do with the delivery).
```
