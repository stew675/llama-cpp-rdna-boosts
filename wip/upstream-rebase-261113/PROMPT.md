# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the
  whole climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, clean at HEAD `a95c0497b` (R09, green).
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm).
  The build dir currently holds the R09 tree (R10 was rolled back) — rebuild before measuring.
- Models: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf (coherence oracle), Qwen3.6/35B-A3B/Q8_0
  (the host-expert arm), Qwen3.8/Flash-Next (the pool gate's model).

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §5.5 the R10 blocker, §5.4 R09, traps, R18 ladder)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then the R10 result/blocker section)
3. wip/upstream-rebase-261113/DECISIONS.md  (D10 is the live suspect; D1 is ANSWERED; D5-a is open)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R09 are done = delivery blocks 00-08 on `rebase-r278` (HEAD `a95c0497b`, coherence
`sha=1c5d32ac537d`). R10 = block 09 (host-resident MoE) was MERGED, BUILT, COHERENT and then PARKED:
the resolved commit is the tag `r10-blocked-d4106d6f6` in ~/llama.cpp and
`wip/upstream-rebase-261113/R10-blocked-d4106d6f6.patch` here. It is parked for ONE reason:
`scripts/gate-host-pool.sh` fails its 2-device `-sm tensor` arm with a GPU page fault on a HOST address
inside `mul_mat_vec_q_moe`, while the same arm passes on r277 and passes on the merged tree with
`-sm layer`, with 1 device, and with `MOE_EXPERT_CACHE_MIB=0`.

YOUR TASK
Unblock R10, then continue the ladder.
1. `cd ~/llama.cpp && git am ~/llama-cpp-rdna-boosts/wip/upstream-rebase-261113/R10-blocked-d4106d6f6.patch`
   (it applies on `a95c0497b` exactly; the tag `r10-blocked-d4106d6f6` is the same commit).
2. Diagnose the `-sm tensor` arena fault. Check DECISIONS.md D10's hypothesis FIRST: whether the
   host-expert fill (`llama_context::sched_copy_experts`) runs at all under `-sm tensor`, i.e. whether
   upstream's `hv_tensor_requires_copy[hash_id(input)]` gate (merged at R10 into ggml-backend.cpp) marks a
   Meta-wrapped host expert weight as "not read". Compare the `-lv 4` fill census under `-sm tensor` against
   the 1-device 1991 fills. Do not delete upstream's gate to make the fault go away: if the gate is right,
   the fix is to register the delivery's staging as a read.
3. Rebuild, re-run `scripts/gate-host-pool.sh` to green (all 13 arms), coherence, and the 35B-A3B arena
   census. Then record: RUNGS.md R10 row + result, DECISIONS.md D10, HANDOVER §0/§5.5, a WORKLOG entry,
   and commit the delivery repo on `rebase-261113`.
4. Then R11 (block 10, `05f298fea`, 10 expected conflicts) — which owes `gate-node-budget.sh` and
   `gate-norm-rows-alias.sh`.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Upstream-vs-delivery overlaps
  resolve as "both sides" unless a measurement says otherwise; D2, D5 and D12 set that precedent.
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A rung that is blocked by a regression is rolled back and its resolution parked as a patch here
  (README.md §5) — never left half-resolved and never left red at HEAD.
- If you run out of room: `git cherry-pick --abort` / `git reset --hard` back to the last green tip, record
  what you learned in RUNGS.md, update HANDOVER §0, and stop.
```
