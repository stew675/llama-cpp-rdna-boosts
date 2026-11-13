# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the whole
  climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, expected clean at HEAD `1d83b50e0`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm;
  ~20-30 s incremental). The build dir currently holds the R12 tree.
- Reference builds: `build-rocm/bin-r277/llama-cli` = the FULL 17-block delivery (use it to tell "this block
  broke it" from "a later block fixes it"); `build-rocm-base2` = unpatched 4f9dfa0e3 with VMM=ON.
- Models: /llm/models/Qwen3.5/4B/Q8_0 (coherence oracle), Qwen3.6/35B-A3B/Q8_0 (host-expert/MTP field config),
  Qwen3.8/Flash-Next/{GSQ-IQ3_XXS,IQ4_NL,Q4_K_M,...} (qwen4exp — there is NO IQ4_XS, that arm skips by design),
  prompts/prose-rdna-boosts.txt.

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §3 mechanics, §5 decisions, §7 env, §8 traps)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R13; §R12/§R11 are what the last rungs decided)
3. wip/upstream-rebase-261113/DECISIONS.md  (D10, D11, D15 are the R13 items; D6/D8/D9/D11/D12/D5-a are closed)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R12 are done on `rebase-r278`: blocks 00-08 (R01-R09), the rung-repair `9c94ce139` (block 05's FA instance
matrix: heads 40/72, the FP8 KV column, a typed generator), block 09 `a6809f75b`, block 10 `13176810a` (which
moved the whole fusion dispatcher into `ggml-cuda-try-fuse.cu`) and block 11 `1d83b50e0` (qwen4exp;
GGML_OP_COUNT is 105). Coherence is `100 chars sha=1c5d32ac537d` at every rung since R01, re-verified on the
R12 tree.

Three gate arms are deferred by construction, each to the block that ships the code they need — R13 clears the
first two:
  * `gate-host-pool.sh` `tsplit` + `tsplitnosink`  -> R13 (block 12 adds `moe_cache_pool_backed` + the Meta
    forwarding; without it the fused kernel reads the pageable host master and both GPUs fault)
  * `gate-node-budget.sh` (all four arms)          -> R13 (block 12 has `graph_max_nodes()` and the
    "over its node budget" refusal)
  * `gate-qwen4exp-quant-coherence.sh` section E (sparse QSA) -> R16 (block 15 replaces
    `ggml_rope_get_truncate(dst)` on the fused INDEXER_SCORE node, which upstream #30206's assert rejects)

YOUR TASK
Execute R13 = block 12 ("campaign remainder: CPU wins, backend iface fills, DFlash, FA build grouping, server
prompt cache"), commit `c7628e974`, ~4 expected conflicts, decisions D10/D11/D15.
1. `cd ~/llama.cpp && git cherry-pick c7628e974`. Re-measure the conflict set (do not trust the count).
2. Resolve as "both sides" unless a measurement says otherwise. Expect:
   - `ggml/include/ggml-rpc.h` + `ggml/src/ggml.c` — the RPC *message* surface (D11): re-check "one message, one
     handler on both ends", and re-derive `GGML_OP_COUNT` from the merged enum if the block adds ops (never take
     either side's literal).
   - `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` — the FA build grouping. It MUST still
     reproduce the tree byte for byte: run it twice and require `git status` clean (that invariant is what the
     R09r repair established). If it changes the compiled instance set, run `gate-prefill-logits.sh`.
   - `src/llama-context.cpp` (`a8fad21e9`, `cb44fdb18`, D10).
   Watch D8's include trap: block 12 may re-add an include upstream pruned (check with
   `git log --oneline a691b2345..4f9dfa0e3 -- <file>`).
3. `git add -A` (always), then `GIT_EDITOR=true git cherry-pick --continue`.
4. Build (`~/bin/build-llama-rocm`). Expect first-error passes — R11 needed six; fix, rebuild, then
   `git add -A && GIT_EDITOR=true git commit --amend --no-edit`. If the block adds/removes files matched by a
   bare `file(GLOB ...)`, `touch ggml/src/ggml-cuda/CMakeLists.txt` first.
5. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, NO -lv; expect `sha=1c5d32ac537d` via
     scripts/extract-generated.py.
   - `scripts/gate-host-pool.sh` — ALL 13 arms must pass now, `tsplit`/`tsplitnosink` included. If the verdict
     block still cannot run, say exactly which symbol is still missing.
   - `scripts/gate-node-budget.sh` — all four arms AND the margin: print the node count vs the budget per arm.
     At R11 arm B (`q8_0 -ub 512`) aborted where the UNPATCHED base passed, so this is the one rung where a
     green line alone is not enough — show the margin or report it as a finding.
   - `git grep -In '^<<<<<<< \|^=======$\|^>>>>>>> '` — must be empty (stray end-markers bite).
   - RUNGS.md: ledger row + result/resolution log (incl. which arms are still deferred and to whom);
     DECISIONS.md D10/D11/D15; HANDOVER §0/§5/§6; a dated WORKLOG entry; commit the delivery repo on
     `rebase-261113`; point PROMPT.md at R14.

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

IF R13 COMPLETES
Continue with R14 (block 13, MTP draft vocabulary subset `--spec-draft-mtp-vocab N`, commit `f0c7f3359`, 0
expected conflicts — verify the subset filter still applies to the drafted tokens, and re-verify D7) and R15
(block 14, `-sm relaxed-tensor`, TEMPORARY — re-check upstream has not started allowing those arches; expect to
drop it eventually by decision, not by merge).
```
