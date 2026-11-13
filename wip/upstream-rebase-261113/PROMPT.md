# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the
  whole climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, clean at HEAD `a6809f75b`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm).
- Models: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf (coherence oracle), Qwen3.6/35B-A3B/Q8_0,
  Qwen3.8/Flash-Next (qwen4exp + the pool gate).

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §5.5 the last two rungs, traps, R18 ladder)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R11)
3. wip/upstream-rebase-261113/DECISIONS.md  (D6, D8 and D10 are the R11 items; D5-a and D10's staging
                                             question are already closed, do not re-litigate them)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R10 are done: blocks 00-08 on `rebase-r278`, then the rung-repair `9c94ce139` (block 05's FA instance
matrix — heads 40/72, the FP8 KV column, a typed generator; undefined FA symbols 32 -> 0), then block 09 as
`a6809f75b`. Coherence is `100 chars sha=1c5d32ac537d` throughout. One rung's gate arms are deferred by
construction: `gate-host-pool.sh` `tsplit`/`tsplitnosink` need block 12's `moe_cache_pool_backed` Meta
forwarding, so they are R13's (README §5's blocked-rung protocol was used once and is resolved).

YOUR TASK
Execute R11 = block 10 ("slab, fusion dispatcher and async input ring", commit `05f298fea`, 10 expected
conflicts — the largest cluster; decisions D6/D8/D10).
1. `cd ~/llama.cpp && git cherry-pick 05f298fea`. Re-measure the conflict set, do not trust the count.
2. Resolve as "both sides" unless a measurement says otherwise. Watch for: D6 (graph.cuh owns
   `ggml_cuda_graph`/`ggml_cuda_stream_context`, so any delivery hook that touches that state follows it),
   D8 (the prune removes INCLUDES as well as declarations — a build break, not a conflict: check the include
   surface of every delivery symbol, `llama-ext.h` especially), D10 (the scheduler upload/prefetch path).
3. `git add -A` (always), then `GIT_EDITOR=true git cherry-pick --continue`.
4. Build. Expect a first-error pass; fix, rebuild, `git commit --amend` into the rung commit.
5. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, no -lv; expect `sha=1c5d32ac537d`.
   - scripts/gate-node-budget.sh  (the async ring adds graph nodes)
   - scripts/gate-norm-rows-alias.sh  (the r225 narrow-row fusion arrives with this block; its driver now
     links, so a real PASS is expected — if an arm fails, that is the block's mechanism, not the harness)
   - scripts/gate-grid-stall-pad.sh if the pad site moved
   - RUNGS.md: ledger row + result/resolution log; DECISIONS.md D6/D8/D10; HANDOVER §0; a WORKLOG entry;
     commit the delivery repo on `rebase-261113`.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Upstream-vs-delivery overlaps
  resolve as "both sides" unless a measurement says otherwise; D2, D5, D12 and D1 set that precedent.
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A conflict that resolves "naturally" because upstream removed the same thing is a decision, not a silent
  delete: write it in the rung log.
- If a gate arm belongs to a later block, name the block and the symbol that makes it impossible now (that is
  how the R10 `tsplit` arms and R09's rows gate were settled) — never weaken the gate and never "simplify" it
  to the arm that passes.
- If you run out of room: `git cherry-pick --abort`, record what you learned in RUNGS.md, update HANDOVER
  §0, and stop. Do not leave a half-resolved rung.

IF R11 COMPLETES
Continue with R12 (block 11, qwen4exp: model, hyper-connections, lazy reader, QSA/indexer, commit `ea4963486`,
5 expected conflicts, decision D11). R12 owes `gate-hc-fusions.sh` and `gate-qwen4exp-quant-coherence.sh`.
```
