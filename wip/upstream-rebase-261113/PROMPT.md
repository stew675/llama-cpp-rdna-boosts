# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the
  whole climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, clean at HEAD `13176810a`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm).
- Models: /llm/models/Qwen3.5/4B/Q8_0 (coherence oracle), Qwen3.6/35B-A3B/Q8_0, Qwen3.8/Flash-Next
  (qwen4exp + the host-pool gate).

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §5.5 the recent rungs, traps, R18 ladder)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R12)
3. wip/upstream-rebase-261113/DECISIONS.md  (D11 is the R12 item; D6/D8/D10 and D5-a are closed)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R11 are done: blocks 00-08, the rung-repair `9c94ce139` (block 05's FA instance matrix: heads 40/72, the
FP8 KV column, a typed generator), block 09 `a6809f75b` and block 10 `13176810a` — the one that moved the whole
fusion dispatcher into `ggml-cuda-try-fuse.cu`.  Coherence is `100 chars sha=1c5d32ac537d` throughout.
`gate-norm-rows-alias.sh` PASSES.  Two gate arms are R13's by construction (block 12 ships the code they need):
`gate-host-pool.sh` `tsplit`/`tsplitnosink` and `gate-node-budget.sh`.

YOUR TASK
Execute R12 = block 11 ("qwen4exp: model, hyper-connections, lazy reader, QSA/indexer"), commit
`ea4963486`, ~5 expected conflicts, decision D11.
1. `cd ~/llama.cpp && git cherry-pick ea4963486`. Re-measure the conflict set.
2. Resolve as "both sides" unless a measurement says otherwise. Expect: the `ggml.c`/RPC op-table
   re-registration (D11 — one registration + one dispatch + one test oracle per new op, and a matching RPC
   handler on both ends), `ggml/src/ggml-cuda/ggml-cuda.cu` (op dispatch + the graph/stream state now
   living in graph.cuh), `src/models/qwen4exp.cpp` (upstream #32078 removed the ubatch-dependent nodes — see
   D2's precedent: the delivery's path is the surviving one, do not re-add removed enum values), and
   `tests/test-backend-ops.cpp` (the public-API test move). Watch D8's include trap after every prune.
3. `git add -A` (always), then `GIT_EDITOR=true git cherry-pick --continue`.
4. Build. Expect first-error passes (block 10 needed six); fix, rebuild, `git commit --amend` into the rung.
5. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, no -lv; expect `sha=1c5d32ac537d`.
   - scripts/gate-hc-fusions.sh            (the HC mix/combine arms on qwen4exp)
   - scripts/gate-qwen4exp-quant-coherence.sh
   - test-backend-ops -o HC_MIX / HC_COMBINE (or the ops the block adds)
   - RUNGS.md: ledger row + result/resolution log; DECISIONS.md D11; HANDOVER §0; a WORKLOG entry;
     commit the delivery repo on `rebase-261113`.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Upstream-vs-delivery overlaps
  resolve as "both sides" unless a measurement says otherwise (D1, D2, D5, D12 set that precedent).
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A conflict that resolves "naturally" because upstream removed the same thing is a decision, not a silent
  delete: write it in the rung log.
- If a gate arm belongs to a later block, name the block and the symbol that makes it impossible now — never
  weaken the gate and never simplify it to the arm that passes.
- If you run out of room: `git cherry-pick --abort`, record what you learned in RUNGS.md, update HANDOVER
  §0, and stop. Do not leave a half-resolved rung.

IF R12 COMPLETES
Continue with R13 (block 12, "campaign remainder": CPU wins, backend iface fills, DFlash, FA build grouping,
server prompt cache; commit `c7628e974`, 4 expected conflicts, decisions D10/D11/D15). R13 is the rung that
owes `gate-host-pool.sh` `tsplit`/`tsplitnosink` AND `gate-node-budget.sh` — and for the node budget it must
show the margin, not just a green line (at R11 arm B aborts where the unpatched base passed).
```
