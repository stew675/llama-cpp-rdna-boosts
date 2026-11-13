# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the
  whole climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, expected clean at HEAD `a95c0497b`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm).
- Models: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf (coherence oracle), the 35B-A3B tree
  (the field host-expert config) and /llm/models/Qwen3.8/Flash-Next (qwen4exp).

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (state, rung mechanics, traps, decision status, R18 ladder)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R10; §R09's result shows what the last
                                             rung decided and what it handed over)
3. wip/upstream-rebase-261113/DECISIONS.md  (D1/D9/D15/D17/D18 are the R10 items; D5-a is the open
                                             finding R09 handed over)
4. wip/upstream-rebase-261113/README.md §1 and §3 (rung rules and the "report before dropping" policy)

STATE
R00..R09 are done = delivery blocks 00-08, each a resolved, build-green, coherent commit on
`rebase-r278` (chain and resolutions in HANDOVER §0/§5). The next rung is R10 = block 09
("host-resident MoE: iface, scheduler, Meta, slab, cache engine"), commit `495d00d45`, 8 expected
conflicts. The coherence oracle is `100 chars sha=1c5d32ac537d`, unchanged since R01.

YOUR TASK
Execute R10 as a ladder rung, and note that it contains a genuine STOP-AND-ASK:
1. In ~/llama.cpp: `git cherry-pick 495d00d45`. It produces 8 conflicted files
   (common/fit.cpp, common/speculative.cpp, ggml/src/ggml-backend.cpp, ggml/src/ggml-cuda/common.cuh,
   ggml/src/ggml-cuda/ggml-cuda.cu, src/llama-context.cpp, src/llama-context.h, src/llama-mmap.cpp per
   RUNGS.md — re-measure, do not trust the count).
2. D1 is the stop: upstream `951d465f5` (#32157) moved the MoE cache out of `llama-graph.cpp` into
   `src/llama-moe-cache.{cpp,h}` (an `apply()`/`ensure()` split) and `cf34d3113` (#31888) stopped its CPU
   part waiting for cached experts. Port the delivery's arena hook to that seam WITHOUT changing the r236
   either/or ruling (arena default-on, upstream's cache shipped-but-dormant — it measured 84 % behind on
   prefill / 35 % on decode). If the move makes the hook impossible to keep, STOP and present options.
   `D18` is consistency only: the SYCL device-side MoE scheduler must not change the CUDA predicate.
3. `git add -A` (always — the index is what git reports as unmerged), then
   `GIT_EDITOR=true git cherry-pick --continue`.
4. Build (`~/bin/build-llama-rocm`). Expect a first-error pass — the merge is not done when the text
   looks right; fix, rebuild, `git commit --amend` into the rung commit.
5. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, no -lv; expect `sha=1c5d32ac537d` via
     scripts/extract-generated.py.
   - the host-expert arm: 35B-A3B Q8_0, `-ncmoe`/`-lme` as `runme-rocm` beside the model has it, `-lv 4`,
     and grep the arena/pool census (`MoE expert cache:`, `expert access:`, `takeover=`).
   - `scripts/gate-host-pool.sh` if the pool/fill path moved (it is the block's own gate).
   - RUNGS.md: ledger row + result/resolution log; DECISIONS.md for D1/D9/D15/D18; HANDOVER.md §0;
     a dated WORKLOG entry; commit the delivery repo on `rebase-261113`.

RULES
- Report to the maintainer and STOP before dropping any delivery mechanism. Upstream-vs-delivery overlaps
  resolve as "both sides" unless a measurement says otherwise; D2, D5 and D12 already set that precedent.
- Nothing from wip/, beta/, optional/ or archive/work/ is applied, rebased or consulted.
- A conflict that resolves "naturally" because upstream removed the same thing is a decision, not a silent
  delete: write it in the rung log.
- If you run out of room: `git cherry-pick --abort`, record what you learned in RUNGS.md, update HANDOVER
  §0, and stop. Do not leave a half-resolved rung.

IF R10 COMPLETES
Continue with R11 (block 10, slab + fusion dispatcher + async input ring, commit `05f298fea`, 10 expected
conflicts — the largest cluster). R11 owns two gates R09/R10 do not: `scripts/gate-node-budget.sh` (the
async ring adds graph nodes) and `scripts/gate-norm-rows-alias.sh` (the r225 narrow-row fusion's code
arrives with block 10) — the latter stays blocked until DECISIONS.md `D5-a` is repaired, because its
section A links a driver against libggml-hip.so and the merged FA instance matrix leaves 32
`flashattn_mma_f16_case<40|72,...>` symbols undefined.
```
