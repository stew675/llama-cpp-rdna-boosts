# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the
  whole climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, clean at HEAD `1d83b50e0`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm).
- Models: /llm/models/Qwen3.5/4B/Q8_0 (coherence oracle), Qwen3.6/35B-A3B/Q8_0, Qwen3.8/Flash-Next (qwen4exp
  + the pool/node-budget gates; `QUANTS` there is IQ3_XXS/IQ4_NL/Q4_K_M — there is no IQ4_XS on this box).

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §5.5 the recent rungs, traps, R18 ladder)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R13)
3. wip/upstream-rebase-261113/DECISIONS.md  (D10/D11/D15 are the R13 items; D6/D8/D9/D12/D5-a are closed)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R12 are done: blocks 00-08, the rung-repair `9c94ce139` (block 05's FA instance matrix), block 09
`a6809f75b`, block 10 `13176810a` (the fusion dispatcher moved into `ggml-cuda-try-fuse.cu`) and block 11
`1d83b50e0` (qwen4exp; `GGML_OP_COUNT` is **105**).  Coherence is `100 chars sha=1c5d32ac537d` throughout.
Three gate arms are deferred by construction, each to the block that ships the code they need:
`gate-host-pool.sh` `tsplit`/`tsplitnosink` -> R13, `gate-node-budget.sh` -> R13, sparse QSA
(`gate-qwen4exp-quant-coherence.sh` section E) -> R16.

YOUR TASK
Execute R13 = block 12 ("campaign remainder: CPU wins, backend iface fills, DFlash, FA build grouping, server
prompt cache"), commit `c7628e974`, ~4 expected conflicts, decisions D10/D11/D15.
1. `cd ~/llama.cpp && git cherry-pick c7628e974`. Re-measure the conflict set.
2. Resolve as "both sides" unless a measurement says otherwise. Expect `ggml/include/ggml-rpc.h` + `ggml/src/ggml.c`
   (the RPC **message** surface — re-check "one message, one handler on both ends", D11),
   `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` (the FA build grouping — remember it must still
   reproduce the tree byte for byte: run it twice and check `git status` is clean, that invariant is what the
   `R09r` repair established), and `src/llama-context.cpp` (`a8fad21e9`, `cb44fdb18`, D10).
3. `git add -A` (always), then `GIT_EDITOR=true git cherry-pick --continue`.
4. Build. Expect first-error passes; fix, rebuild, `git commit --amend` into the rung.
5. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, no -lv; expect `sha=1c5d32ac537d`.
   - scripts/gate-host-pool.sh   -> ALL 13 arms must now pass (`tsplit`/`tsplitnosink` included: this block
     adds `moe_cache_pool_backed` + the Meta forwarding).
   - scripts/gate-node-budget.sh -> all four arms, AND show the margin: at R11 arm B aborted where the
     unpatched base passed, so name the node count vs the budget for each arm rather than only the verdict.
   - scripts/gate-prefill-logits.sh if the FA build grouping changed the compiled instance set.
   - RUNGS.md: ledger row + result/resolution log; DECISIONS.md D10/D11/D15; HANDOVER §0; a WORKLOG entry;
     commit the delivery repo on `rebase-261113`.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Overlaps resolve as "both sides"
  unless a measurement says otherwise (D1, D2, D5, D12 set that precedent).
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A conflict that resolves "naturally" because upstream removed the same thing is a decision, not a silent
  delete: write it in the rung log.
- If a gate arm needs a later block, name the block and the symbol that makes it impossible now — never weaken
  the gate and never simplify it to the arm that passes. (Three such items are already recorded; clear the two
  R13 ones now.)
- If you run out of room: `git cherry-pick --abort`, record what you learned in RUNGS.md, update HANDOVER
  §0, and stop. Do not leave a half-resolved rung.

IF R13 COMPLETES
Continue with R14 (block 13, MTP draft vocabulary subset, commit `f0c7f3359`, 0 expected conflicts — verify the
vocab-subset filter still applies to the drafted tokens) and R15 (block 14, `-sm relaxed-tensor`, TEMPORARY —
re-check upstream has not started allowing those arches).
```
