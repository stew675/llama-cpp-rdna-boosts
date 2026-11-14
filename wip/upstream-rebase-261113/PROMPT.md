# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the whole
  climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, expected clean at HEAD `9c940db6f`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm;
  ~20-30 s incremental). The build dir currently holds the R16 tree.
- Reference builds: `build-rocm/bin-r277/llama-cli` = the FULL 17-block delivery (use it to tell "this block
  broke it" from "a later block fixes it"); `build-rocm-base2` = unpatched 4f9dfa0e3 with VMM=ON.
- Models: /llm/models/Qwen3.5/4B/Q8_0 (coherence oracle), Qwen3.6/35B-A3B/Q8_0 (host-expert/MTP field config),
  Qwen3.8/Flash-Next/{GSQ-IQ3_XXS,IQ4_NL,Q4_K_M,...} (qwen4exp — there is NO IQ4_XS, that arm skips by design),
  prompts/prose-rdna-boosts.txt.

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §3 mechanics, §5 decisions, §7 env, §8 traps)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R17; §R13-R16 are what the last rungs decided —
                                             which arms they cleared and which they handed to R17 — and
                                             HANDOVER §5.7/§5.9/§5.10)
3. wip/upstream-rebase-261113/DECISIONS.md  (D7 is now re-verified; D6/D8/D9/D10/D11/D12/D5-a are closed)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R14 are done on `rebase-r278`: blocks 00-08 (R01-R09), the rung-repair `9c94ce139` (block 05's FA instance
matrix: heads 40/72, the FP8 KV column, a typed generator), block 09 `a6809f75b`, block 10 `13176810a` (which
moved the whole fusion dispatcher into `ggml-cuda-try-fuse.cu`), block 11 `1d83b50e0` (qwen4exp;
GGML_OP_COUNT is 106 after R13), block 12 `75d83f231` (campaign remainder; tree `d862165c5`) and block 13
`aca78625a` (MTP draft vocabulary subset; tree `061a71a98`) and block 14 `f040d394f` (`-sm relaxed-tensor`,
TEMPORARY; tree `0c8a1fd58`), block 15 `f745bbf1b` (the streamed KV) and the **R16 rung-repair `9c940db6f`**
(tree `14f99e579`).  Coherence is `100 chars sha=1c5d32ac537d` at every rung since R01, re-verified on the R16
tree.  Block 14 is still load-bearing (upstream's `#31081` deny is verbatim at `4f9dfa0e3` and nothing in
`a691b2345..4f9dfa0e3` touches `llm_arch_supports_sm_tensor`) — do not drop it in passing; HANDOVER §5.8.
R16 cleared the last three gate arms of the climb, added a rung-repair (§5.9) and found `D20`/`TODO` #321.  `D7` was re-verified at R14: the device-buffer hidden-state
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
Execute R17 = block 16 ("the UMA GTT expert store -- an APU's MoE expert set in ONE registered, mlock'd region,
with a slot-storage ARENA when the budget is below the set"), commit `9d7c0502c`, 16 files, **0 expected
conflicts**.
1. `cd ~/llama.cpp && git cherry-pick 9d7c0502c`.  Expect clean; if a path conflicts, re-measure and resolve as
   "both sides" unless a measurement says otherwise.
2. Build (`~/bin/build-llama-rocm`); `touch ggml/src/ggml-cuda/CMakeLists.txt` first if the block adds/removes
   files matched by a bare `file(GLOB ...)`; fix first-error passes, then
   `git add -A && GIT_EDITOR=true git commit --amend --no-edit`.
3. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, NO -lv; expect `sha=1c5d32ac537d` via
     scripts/extract-generated.py.
   - **`D17` (the VMM region check):** the halved VMM reservation + retry must not change the region the GTT
     store asks for.  `scripts/gate-uma-experts.sh --quick` (its `inplace` and `pfcp` arms), plus
     `scripts/gate-prefill-server-slots.sh` (two slots: one decoding while the other prefills).
   - **`scripts/gate-host-pool.sh` must reach 54/54**: the four checks R13/R16 handed here are the placement
     read transport (`ref`'s `host pool PLACEMENT reads: O_DIRECT` line + the placement queue census with its
     8 reader(s)), `qd1` (`MOE_EXPERT_FILL_QD=1` -> 1 reader) and `bulkbuf` (`MOE_HOST_POOL_BULK_DIO=0` -> a
     buffered placement read).  If they still do not appear, name the symbol that is missing — do not weaken the
     gate.
   - the 8 GB-class discrete path must stay inert (the store is UMA-only by measurement) and the discrete
     `gate-host-pool`/`gate-kv-residency` arms must not change.
   - `git grep -In '^<<<<<<< \|^=======$\|^>>>>>>> '` — must be empty.
4. RUNGS.md: ledger row + result/resolution log; DECISIONS.md `D17`; HANDOVER §0/§4/§5/§6; a dated WORKLOG entry;
   commit the delivery repo on `rebase-261113`; point PROMPT.md at R18.

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

IF R17 COMPLETES
R18 is the release ladder (`README.md` §4 + HANDOVER §10) — **do not start it without the maintainer's
go-ahead**.  It is not a rung: `make-patches.sh` (squashing BOTH rung-repairs — `R09r` into block 05 and the
R16r h_nextn repair into block 11), `make-release.sh`, `validate-set.sh`, `gate-per-block-build.sh`, the
prefill-logit record + arms, the coherence set (dense 4B, the 35B-A3B field config, a hybrid recurrent model,
the qwen4exp class), the MTP four-axis protocol, `gate-rebase-release.sh`, `check-gpu-faults.sh`, the fork
refresh (personal fork only) and the records.  Known-red items to carry: `gate-kv-residency.sh`'s **multi-device**
pool section (`D20`/`TODO` #321 — pre-existing 2-device MTP nondeterminism; nothing to do with the delivery).
```
