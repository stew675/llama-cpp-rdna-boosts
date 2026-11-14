# Cold-start prompt

Paste the block below into a fresh session (it assumes `~/llama-cpp-rdna-boosts` is the cwd).  It is
self-contained; the detail lives in [`HANDOVER.md`](HANDOVER.md).

```text
You are continuing a multi-session upstream re-base of the "rdna-boosts" delivery patch set
(17 blocks of RDNA/ROCm work on llama.cpp). Work rung by rung; never batch rungs.

LAYOUT
- Delivery repo (cwd): ~/llama-cpp-rdna-boosts, branch `rebase-261113`. Never merge to `main` until the whole
  climb is green; never push anything anywhere.
- Fork checkout: ~/llama.cpp on branch `rebase-r278`, expected clean at HEAD `75d83f231`.
  `master` there is the pristine upstream snapshot `4f9dfa0e3` — never commit to it.
- Build: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1, gfx1201, GGML_HIP_NO_VMM=OFF, ccache warm;
  ~20-30 s incremental). The build dir currently holds the R13 tree.
- Reference builds: `build-rocm/bin-r277/llama-cli` = the FULL 17-block delivery (use it to tell "this block
  broke it" from "a later block fixes it"); `build-rocm-base2` = unpatched 4f9dfa0e3 with VMM=ON.
- Models: /llm/models/Qwen3.5/4B/Q8_0 (coherence oracle), Qwen3.6/35B-A3B/Q8_0 (host-expert/MTP field config),
  Qwen3.8/Flash-Next/{GSQ-IQ3_XXS,IQ4_NL,Q4_K_M,...} (qwen4exp — there is NO IQ4_XS, that arm skips by design),
  prompts/prose-rdna-boosts.txt.

READ FIRST, IN THIS ORDER
1. wip/upstream-rebase-261113/HANDOVER.md   (§0 state, §3 mechanics, §5 decisions, §7 env, §8 traps)
2. wip/upstream-rebase-261113/RUNGS.md      (§0 ledger, then §R14; §R13 is what the last rung decided and
                                             which gate arms it handed to R16/R17 — HANDOVER §5.7)
3. wip/upstream-rebase-261113/DECISIONS.md  (D7 is the R14 item; D6/D8/D9/D10/D11/D12/D5-a are closed)
4. wip/upstream-rebase-261113/README.md §1 and §5 (rung mechanics and the blocked-rung protocol)

STATE
R00..R13 are done on `rebase-r278`: blocks 00-08 (R01-R09), the rung-repair `9c94ce139` (block 05's FA instance
matrix: heads 40/72, the FP8 KV column, a typed generator), block 09 `a6809f75b`, block 10 `13176810a` (which
moved the whole fusion dispatcher into `ggml-cuda-try-fuse.cu`), block 11 `1d83b50e0` (qwen4exp;
GGML_OP_COUNT is now 106 after R13) and block 12 `75d83f231` (campaign remainder; tree `d862165c5`).
Coherence is `100 chars sha=1c5d32ac537d` at every rung since R01, re-verified on the R13 tree.

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
Execute R14 = block 13 ("MTP draft vocabulary subset, `--spec-draft-mtp-vocab N`"), commit `f0c7f3359`,
19 files, 0 expected conflicts.
1. `cd ~/llama.cpp && git cherry-pick f0c7f3359`. Expect it to apply cleanly; if a path conflicts, re-measure and
   resolve as "both sides" unless a measurement says otherwise.
2. Build (`~/bin/build-llama-rocm`). First-error passes are normal; fix, rebuild, then
   `git add -A && GIT_EDITOR=true git commit --amend --no-edit`.
3. Verify, then record:
   - coherence: HIP_VISIBLE_DEVICES=0 llama-cli 4B -ngl 99 -sm layer -p "The capital of France is" -n 20
     --seed 42 --temp 0 --no-display-prompt --single-turn, NO -lv; expect `sha=1c5d32ac537d` via
     scripts/extract-generated.py.
   - `D7` re-verify (HANDOVER §6): the vocab-subset filter must still apply to the DRAFTED tokens.  Run the field
     MTP arm on the 35B-A3B (`HIP_VISIBLE_DEVICES=0,2`, `-sm layer`, bf16 KV,
     `--spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 1`, `-lv 4`) and grep
     `adaptive draft depth enabled`, `target h rows on the device`, `draft acceptance`; confirm the
     `--spec-draft-mtp-vocab N` arm drafts only ids < N (the subset mask) and that `-sm tensor` still declines
     the device-buffer path and falls back to the host row gather.
   - `git grep -In '^<<<<<<< \|^=======$\|^>>>>>>> '` — must be empty.
4. RUNGS.md: ledger row + result/resolution log (incl. which gate arms remain deferred and to whom);
   DECISIONS.md `D7`; HANDOVER §0/§5/§6; a dated WORKLOG entry; commit the delivery repo on `rebase-261113`;
   point PROMPT.md at R15.

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

IF R14 COMPLETES
Continue with R15 (block 14, `-sm relaxed-tensor`, commit `6b4d9865e`, TEMPORARY — re-check upstream has not
started allowing those arches; expect to drop it eventually by decision, not by merge) and then R16 (block 15,
the streamed KV, `99d33117a`, 5 conflicts, which owes the KV residency gates, quant-coherence section E **and
`gate-host-pool.sh`'s `fail`/`late`/`pinlate`**).
```
