# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the
  whole climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, expected clean at HEAD `626e398d2`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm).
- Models: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf (coherence oracle) and the 35B-A3B tree.

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (state, rung mechanics, traps, decision status, R18 ladder)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R09 + its pre-analysis at the end)
3. wip/upstream-rebase-261113/DECISIONS.md  (D6 and D12 are the R09 items)
4. wip/upstream-rebase-261113/README.md §1 and §3 (rung rules and the "report before dropping" policy)

STATE
R00..R08 are done = delivery blocks 00-07, each a resolved, build-green, coherent commit on
`rebase-r278` (chain and resolutions in HANDOVER §0/§5). The next rung is R09 = block 08
("fused kernels"), commit `7f1ed0723`.

YOUR TASK
Execute R09 exactly as a ladder rung:
1. In ~/llama.cpp: `git cherry-pick 7f1ed0723`. It produces 7 conflicted files.
2. Resolve per the table in RUNGS.md §R09 pre-analysis. The important one is ggml/src/ggml-cuda/norm.cu
   (D12): upstream #32135 rewrote BF16 RMS_NORM around rms_norm_store4() with a typed destination, while
   the delivery has a dual store (store_f32 -> dstc, plus a dst16c/norm_f2bf BF16 side channel feeding the
   HC16 producer slots). Merge as "if (store_f32) { rms_norm_store4(...) } + if (dst16c) { ... }" in both
   the 4-wide and the scalar path, and keep BOTH includes (convert.cuh and mmb.cuh). Never collapse the
   two stores into one. The other six conflict sites are listed there too (unions vs "take the delivery").
3. `git add -A` (always — the index is what git reports as unmerged), then
   `GIT_EDITOR=true git cherry-pick --continue`.
4. Build. Expect a first-error pass (the merge is not done when the text looks right — fix, rebuild).
5. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, no -lv; expect `sha=1c5d32ac537d` via
     scripts/extract-generated.py.
   - scripts/gate-norm-rows-alias.sh  (owes it: the fused norm path moved)
   - scripts/gate-hc-fusions.sh MODEL   (owes it: the bf16 side channel is the HC16 producer)
   - ./build-rocm/bin/test-backend-ops -o MUL_MAT_ID   (the routed fusion assert)
   - RUNGS.md: ledger row + result/resolution log; DECISIONS.md D6/D12; HANDOVER.md §0; commit the delivery
     repo on `rebase-261113`.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Upstream-vs-delivery overlaps
  resolve as "both sides" unless a measurement says otherwise; D2 and D5 already set that precedent.
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A conflict that resolves "naturally" because upstream removed the same thing is a decision, not a silent
  delete: write it in the rung log.
- If you run out of room: `git cherry-pick --abort`, record what you learned in RUNGS.md, update HANDOVER
  §0, and stop. Do not leave a half-resolved rung.

IF R09 COMPLETES
Continue with R10 (block 09, host-resident MoE, commit `495d00d45`, 8 expected conflicts, decisions
D1/D9/D15/D17/D18). D1 is a genuine stop-and-ask point: port the delivery's arena hook to upstream's new
llama-moe-cache seam and keep the r236 either/or ruling unless a measurement overturns it.
```
