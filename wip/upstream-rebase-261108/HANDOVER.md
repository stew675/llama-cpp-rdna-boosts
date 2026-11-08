# HANDOVER — the upstream re-base, cold-start brief

**The live brief is now [`HANDOVER-REBASE-TODO.md`](HANDOVER-REBASE-TODO.md)** — the ladder is climbed (R01–R16), what remains is the re-base TODO list (four defects/gaps, three decisions, one measured regression) and then R17 (the release ladder) as the last task.

**Read this first, then [`HANDOVER-R13.md`](HANDOVER-R13.md) — the per-rung brief for the rung you are about to
do (R13 and everything after it) — then `RUNGS.md` (the ledger) and `DECISIONS.md` (the open questions).**  Everything below is
the state as of **2026-11-08, end of the session that reached rung R10**.

You are continuing a **ladder climb**: re-basing this repo's 16-block delivery patch set (`patches/`) from the
old upstream fork point onto the new one, one block per rung, pausing at every point where upstream's approach
and the delivery's may be the same thing.  Nothing here is delivery until R17 — `patches/`, `release.json` and
the fork stay untouched until then.

**R11 and R12 are DONE.**  R11 (block 10, `ed1d36436`) was blocked on a GPU fault that turned out to be a
re-base defect in the MUL_MAT pair fusion; unblocking it found two more latent delivery-code losses and a
two-copy dispatcher that had to be reconciled to one (`GREEDY-PURITY.md` §84, and the `INVARIANTS.md` entries).
R12 (block 11 qwen4exp, `b1b0fa807`) then turned out to be **`D3` in its full form**: upstream carries its own
qwen4exp, so the rung was a measured choice between two implementations — the delivery's wins by **+31 % decode
with byte-identical text** on the field config, so it stays.  Read `RUNGS.md` → R12 before touching that model.
A pre-rung fix also landed: `5f412b011` releases the per-graph MMB/BF16 pool blocks in LIFO order (the VMM pool
asserts it; every qwen3.8 run crawled to a halt on the first compute step without it).
**R13 is DONE** (`2b09a50b0`, tree `87e2841f7`, 2026-11-09): block 12 resolved as three measured decisions —
`D15` (F1 deleted, upstream's `embd_row` kept, the Meta refusal carried, the host-experts guard re-measured and
dropped), the state pipe (kept: 43.7 ms vs 74.7 ms restore, byte-identical), and the prompt cache (upstream's
API + the delivery's residency/parking policy, with the checkpoint recycle restricted to uniquely-owned
objects).  Two pre-existing R12 defects were fixed to get there (`-sm layer` ran for the first time after the
`hc_combine_norm` pool-stack fix; the DFlash drafter was silently dead from an unregistered graph input).
**R14 is DONE too** (`ed13a322e`, tree `c08c912a2`, 2026-11-09): block 13 was a clean rung (no decision) —
four conflict sites, all "both sides add code here", plus two drift fixes (`arch` gone from
`llm_graph_params::can_reuse` after `#31733`; upstream's `common_fit_apply()` already carried the delivery's
draft-reserve fold).  `--spec-draft-mtp-vocab N` verified on both head paths and inert when unset.
**R15 is DONE as well** (`617ee7152`, tree `f6ce37555`, 2026-11-09): block 14 applied with **zero conflicts and
no drift edits**, and the flint model on this box confirms the block's claim (`-sm relaxed-tensor` == `-sm layer`
text, while `-sm tensor` is still refused with the block's own message).
**R16 is DONE as well** (`2dc4511b8`, tree `7d6e24c12`, 2026-11-08): block 15 applied over **8** conflicted
files / 12 hunks (`qwen4exp.cpp` auto-merged), `D12`/`D13` re-derived, build green, the four coherence arms
identical to R13–R15, `gate-kv-residency.sh` / `gate-kv-window-split.sh` / `gate-spec-draft-window.sh` all PASS.
Two real defects were fixed to get there — the hybrid ctor's positional argument shift (upstream inserted
`is_indexer` before `v_enabled`, so no V tensor was allocated: every GDN model segfaulted at load) and a latent
**R12** defect in `indexer-score.cu` (`ggml_rope_get_truncate` on a non-ROPE node: the sparse QSA decode arm has
aborted since R12, no gate reached it) — both recorded as `GREEDY-PURITY.md` §85 and `TODO` #297.  The r193 draft
reserve was preserved by folding its `mtp_shared` branch into `common_draft_reserve_derive()` (arena 20 527.5 vs
17 977.5 MiB, +7.8 points of residency).  **Your next job is R17**, the release ladder.

**R16's brief** (block 15, the streamed KV) follows; it is kept as the record of how the rung was resolved.  **Read [`HANDOVER-R16.md`](HANDOVER-R16.md) first**: it is the cold-start brief for both — the conflict
map, the two decisions, the gate list with runtimes and prerequisites (all present on this box), the exact rung
procedure (deterministic commit ids, squash, identity), the R17 ladder in order, the documentation owed, and the
traps R13–R15 paid for (unregistered graph inputs, the pool-stack local rule, shared checkpoints, per-build
coherence oracles).  The R13 brief
below is kept as the record of how it was resolved.

**The pre-R13 brief (R13, block 12, the campaign remainder) — **read [`HANDOVER-R13.md`](HANDOVER-R13.md)**,
which replaces the R11/R12 history below as the working brief.  R13 is **surveyed and parked**, not started: the
cherry-pick was examined and rolled back, and `RUNGS.md` → R13 plus **`r13-survey.md`** in this directory are
the cold-start brief.  It is not one merge but three decisions, each with an oracle: `D15` (the delivery's F1
`llama_lf_*` layer-input device store vs upstream's `embd_row`/`#31506` — expected subsumption, keep the
delivery's host-experts guard), the state save/restore batching (delivery `state-pipe` vs upstream's new
batched `llama_io_write_host` — `gate-state-pipe.sh` decides), and the server prompt-cache residency/parking
vs upstream's server rework (`gate-park-idle-slots.sh` / `gate-ckpt-reuse.sh` decide).  Two of the eight
conflicts are mechanical (both event families; the F1 hooks in `llama-graph.cpp`).

---

## §0  Where we are

| item | value |
|---|---|
| old fork point | `6050b467c` (2026-10-25) — `release.json .base`, delivery = r235 = `69d36099b` / tree `ef5dcbcb0` |
| new fork point | **`a691b2345`** (2026-11-08 11:50 +0200) — 318 upstream commits, 947 files, +116932/−45160 |
| working branch | **`rebase-r236`** in `~/llama.cpp`, currently **`5654a5f95`** (block 09), tree `63ca1523`, clean |
| `master` in `~/llama.cpp` | the **pristine snapshot** at `a691b2345`.  Never commit to it, never rebase onto it directly. |
| blocks re-based | **12 of 16** — blocks 00-11 = rungs R01-R12 (see the table below) |
| blocked | none |
| next | **R13 = block 12 — PARKED with a survey** (`r13-survey.md`); then R14 … R16, then R17 (the release ladder) |
| delivery repo | branch **`rebase-r236`** (main stays at the pre-re-base delivery; the switch-over is R17) |
| historical | **`r11-resolution.patch`** in this directory — the R11 resolution as it was parked (83 files, +16987/−5484, sha256 `705b7a75f765e58a…`), applies to `5654a5f95`.  Kept as the record of the blocked state; the committed rung carries the fixes it was missing. |

```
4aaffd1c0 block 00 foundation            R01  +986/−340    MUL_MAT_ID 931/931
cadab8c45 block 01 hybrid HIP all-reduce R02                  AR pipeline on 3 GPUs
2f0385c71 block 02 adaptive MTP          R03                  draft-mtp-adaptive live (mean len 3.50)
5a214716b block 03 chunked GDN           R04                  chunked dispatch + kill-switch byte-identical
92cf39297 block 04 BF16 KV               R05  (clean)         K/V (bf16) + flash attention
b5d32e0c7 block 05 flash attention       R06  9 files / 48 h    derived kq mask enabled, derived==packed
75ba21116 block 06 MMB                   R07  (clean)         MMB_DENSE fires, MMB=0 byte-identical
f0a594fe1 block 07 k-quant boosts        R08  1 file / 2 h     MMVQ table split + RDNA4_M
af0cae2f1 block 08 fused kernels         R09  5 files / 8 h    D2 complementary; MUL_MAT_ID 931/931
5654a5f95 block 09 host-resident MoE     R10  7 files / 12 h   D1 first numbers; NO_VMM finding
ed1d36436 block 10 slab/dispatcher/ring R11  8 files / 22 h   pair ntj fault fixed; compact MoE dispatch revived (+1.9% pp);
                                                              one dispatcher (TODO #290 closed); GREEDY-PURITY §84
5f412b011 (fix) pool LIFO release      --                    pre-R12: VMM pool assert killed every qwen3.8 run
b1b0fa807 block 11 qwen4exp            R12  9 files / 34 h    D3 measured: ours is +31% tg on Flash-Next 3-GPU with
                                                              identical text; upstream's qwen4exp discarded
```

## §0.1  The hard rules of this repo (they are not optional)

1. **Apply with `scripts/apply-all.sh`** (strict `git am`).  Never `git apply` the concatenated series.
2. **Report findings before you commit, and before any release.**  Surface review findings, surprises and nits
   to the maintainer *before* a delivery-affecting commit; the maintainer decides fix-now / defer / drop, and
   the decision is recorded (a `TODO.md` item, a block amendment, or a note in the relevant record).
3. **Never push.**  The only permitted push targets are this repo and the personal fork
   (`git@github.com:stew675/llama.cpp.git`, branch `rdna-boosts`).  A bare `git push` in `~/llama.cpp` targets
   upstream `ggml-org/llama.cpp` — never acceptable.
4. **WIP is not delivery.**  Every open `wip/` campaign's patches are **ignored** by this re-base (maintainer's
   ruling, 2026-11-08): not applied, not rebased, not consulted while resolving a rung.  They get their own
   sessions afterwards.  Same for `beta/`, `optional/` and `archive/work/`.
5. **A beneficial feature ships ON by default**; an env var for it exists only to disable it.
6. **`llama-cli` needs `--single-turn`**; anything that can block goes inside `timeout`.
   `-lv` is chosen by the arm: **no `-lv` for a text/hash arm**, **`-lv 4` for a census/mechanism arm**
   (`-lv 3` is a trap: it pollutes the text *and* still hides the ggml censuses).
7. **Pin `HIP_VISIBLE_DEVICES`** on every multi-device arm (the 9950X3D iGPU is enumerated as a HIP device and
   a device-picking `-sm tensor` arm aborts at load with `ROCm error: invalid kernel file`).
8. **Never run benchmarks in parallel**, and never run a container build beside one.
9. **One long-lived source path** (`~/llama.cpp`): ccache keys on absolute paths, so a new worktree is a full
   cold build (≈6 min) while a fresh build dir at the same path is ≈10 s.

## §0.2  The environment

| | |
|---|---|
| GPUs | 3 × R9700 (gfx1201, 32 GiB each).  `HIP_VISIBLE_DEVICES=0` for one, `0,2` for the x8 pair, `0,1,2` for all three |
| ROCm | `/opt/rocm-10.1.0-gfx120X` (canonical), `LD_LIBRARY_PATH=$ROCM/lib` |
| build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` — `build-rocm/`, all targets, `-j 0.75·nproc`, ccache 30 GiB |
| **build flag (new)** | `~/bin/build-llama-rocm` now passes **`-DGGML_HIP_NO_VMM=OFF`** (maintainer's decision 2026-11-08 — the delivery's host-expert arena needs VMM; upstream's HIP default is `ON`).  `build-rocm-vmm/` is a leftover duplicate, harmless |
| clean configure | a rung build needs **`~/bin/build-llama-rocm clean`** when the tree's CMake options changed: a reused cache keeps the *tip's* `GGML_CUDA_FA_QUANTS` (`iq4_nl` instances) and blocks 00-04 fail to configure.  `clean` wipes only the build dir — ccache makes it cheap |
| models | `/llm/models/`: `Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf` (dense, head 128), `Qwen3.6/35B-A3B/Q8_0/…` (MTP MoE, field shape `runme-rocm`), `Qwen3/30B-A3B/Q4_K_M/…` (MoE, fits one card, `MMB_DENSE`/`moe-heatmap`), `Ling/3.0-tiny/IQ4_NL/…` (the norm-rows gate's model), `Gemma4/26B-A4B/Q8_0/…` (head 256, FA family-switch gate) |
| prompts | `prompts/prose-rdna-boosts.txt` (the gates' default), `prompts/fa-family-switch-gemma4-26B-A4B.txt` |
| gates | `scripts/gate-*.sh` — all are the delivery's own gates; `scripts/check-gpu-faults.sh --log F --since "10 min ago"` scans the kernel journal and filters echoed prompt lines (**never grep a log for `page fault` by hand**) |
| reference records | `release.json` (source of truth), `WORKLOG.md` (dated record, newest first), `INVARIANTS.md` (engineering facts), `patches/README.md` (per-block reference), `MANIFESTS.md` (gates), `GREEDY-PURITY.md` (purity rules) |

---

## §1  HISTORY: how R11 was unblocked (block 10 — slab, fusion dispatcher, async input ring)

*Resolved 2026-11-08; read `RUNGS.md` → R11 for the resolution.  The evidence below is what the blocked
state looked like, kept because the same shapes will recur on the later rungs.*

### §1.1  Restore the parked resolution

```bash
cd ~/llama.cpp
git log --oneline -1                     # must be 5654a5f95 (block 09)
git status --porcelain                   # must be empty
git apply --3way ~/llama-cpp-rdna-boosts/wip/upstream-rebase-261108/r11-resolution.patch
~/bin/build-llama-rocm                   # builds green (~2 min warm)
```

The patch is the *complete* R11 resolution: block 10 cherry-picked (`7342cd95a`), all 8 conflicted files /
22 hunks resolved, plus four clean-applied call sites repaired.  `RUNGS.md` → R11 has the per-file table.
**It carries one temporary debug gate you must remove before committing:** `ggml_cuda_try_fuse_legacy()` (see
§1.4) is wrapped in `if (getenv("GGML_CUDA_FUSE_LEGACY"))` and is therefore **off by default**; the committed
rung must not lose upstream's 7 arms, so either the gate goes or a reconciled single dispatcher replaces it.

### §1.2  The blocking fault — exact evidence

```
llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -c 8192 -ngl 99 \
          -f prompts/prose-rdna-boosts.txt -n 16 --seed 42 --temp 0 --single-turn
→ Aborted (exit 134)
  Memory access fault by GPU node-1 … on address 0x7faad5809000. Reason: Page not present
  kernel: void mul_mat_q<(ggml_type)…, …>   grid=[16384, 288, 1]
  (+11 "page fault / GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031" lines in the kernel journal)
```

Same result with `Ling-3.0-tiny-IQ4_NL` at `-c 8192`, i.e. **not model-specific**.  The fault lands during the
load/first-reserve banner (`Loading model... |-\|/ Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT`), and
the **shapes are context-size dependent**:

| tree | `-c 2048` | `-c 8192` |
|---|---|---|
| **R10** (`5654a5f95`) | OK | **OK — exit 0, zero faults** |
| **R11** (parked patch) | OK (Ling) | **abort + 11 faults** |

R10 also runs `-c 8192` fine on a **VMM** build (the `D1` arms in §0.2), so `GGML_HIP_NO_VMM=OFF` is **not** the
trigger on its own — block 10's content is.

**Already eliminated** (each tested on the faulting build, same command):

| switch | result |
|---|---|
| `GGML_CUDA_SLAB=0` | still faults → not the movable-boundary slab |
| `MOE_EXPERT_CACHE_MIB=0` | still faults → not the expert arena |
| `GGML_CUDA_FA_STAGE_MAX_MB=1` ("never stage") | still faults → not the FA prefill staging arena |
| legacy dispatcher disabled (rebuilt without it) | still faults → **not the interim two-dispatcher merge** |
| `GGML_CUDA_FA_STAGE_CHUNK` | opt-in and off already |

### §1.3  The plan, in order

The faulting kernel is `mul_mat_q` with a 16384-tile x-grid and 288 x-tiles, at a *large* context.  Two families
of suspects, cheapest first:

**A. Environment switches (one run each, ~1 min apiece).**  Block 10 adds all of these:

| # | switch | what it excludes |
|---|---|---|
| 1 | `GGML_CUDA_DISABLE_FUSION=1` | every fusion arm, including the new ones from this block (mwr fold, norm→q8_1, GDN/SSM cache, ssm prescan, dsv4 hc) |
| 2 | `GGML_CUDA_DISABLE_GRAPHS=1` | the CUDA-graph key/eviction change (`common.cuh`, upstream's idle window + our per-subgraph key) |
| 3 | `GGML_SCHED_STAGE=0` | the whole scheduler staging path (block 10's async input ring) |
| 4 | `GGML_CUDA_ASYNC_INPUT=0` | the async graph-input ring only |
| 5 | `GGML_CUDA_MMQ_J_MAX=<big>` / `GGML_Q6_COMPACT_J` / `GGML_CUDA_DISABLE_MMQ_ROUTED=1` | block 10's J-tile selection and the routed-compact dispatch — the fault is in `mul_mat_q`, so start here if (1)-(4) are all negative |
| 6 | `GGML_COMPUTE_BUFFER_CHUNK_MIB=<large>` | the compute-buffer chunking ("the compute-buffer chunk follows a live slab") |
| 7 | `GGML_CUDA_OP_TIMING=1` / `GGML_STAGE_TRACE=1` / `GGML_CUDA_FA_STAGE_CHUNK_DEBUG=1` | not kill switches — use them to learn *which* launch precedes the fault |

**B. If no switch isolates it, bisect block 10's own patch by subsystem.**  `patches/0010-*.patch` is 84 files;
split it into (i) the TUs split out of `ggml-cuda.cu` (slab/arena/vmm), (ii) the fusion dispatcher TU, (iii) the
async input ring + scheduler (`ggml-backend*`), (iv) the mmq/mmvq/mmid launch changes, (v) everything else — and
apply them cumulatively on top of R10 until the fault appears.  `git apply --3way` per subset, rebuilding at the
same path, is ~2 min per step.

**C. Then the R11 exit criteria** (as written in `RUNGS.md`): build green; coherence on the dense 4B **and** the
MoE 30B-A3B; `scripts/gate-norm-rows-alias.sh` **green** (this is the first rung where it can run at all — the
dispatcher arrives with this block; on a blocks-00-08 tree it reports `fused=0 declined=0`); and
`tests/test-backend-ops` for the ops block 10 pre-dispatches.  Record: the rung log, the ledger row, the commit
message, and `patches/README.md`'s block-10 note (at R17).

### §1.4  The dispatcher merge debt (do not lose this)

Block 10 **moves** the fusion dispatcher out of `ggml-cuda.cu` into `ggml-cuda-try-fuse.cu`, but upstream added
**7 arms of its own** to the inline copy in the same range:

1. the MoE weighted-reduction fold (`ggml_cuda_match_moe_weighted_reduction` / `_op_moe_weighted_reduction`),
2. the norm+MUL→mmvq Q8_1 fold (`ggml_cuda_op_rms_norm_q8_1`),
3. `ggml_cuda_op_gated_delta_net_fused_cache`, 4. `ggml_cuda_op_ssm_scan_fused_cache`,
5. `ggml_cuda_op_ssm_conv_l2_gatebeta`, 6. `ggml_cuda_op_dsv4_hc_post`,
7. the direct `ggml_cuda_mul_mat_q_swiglu` form.

The parked resolution keeps upstream's copy as `ggml_cuda_try_fuse_legacy()` **behind** the delivery's dispatcher
(delivery first, upstream's only for a node the delivery declined), so no arm is lost and none is displaced.
**`TODO #290`** is the debt: fold the 7 arms into the single dispatcher in a reconciled priority order, then
delete the duplicate.  The priority is a real decision — the norm+MUL→mmvq pattern exists on **both** sides, so
which one runs first changes the kernel and the numerics.  Ask, don't guess.

---

## §2  Then the rest of the ladder

Work one rung at a time; each rung ends committed, building and coherent.  Per-rung detail
(conflict lists, expected counts, exit criteria) is in `RUNGS.md`.

| rung | block | what to expect |
|---|---|---|
| **R12** | 11 qwen4exp (`0e1032213`) | 7 conflicts: `llama-kv-cache.*`, `llama-memory-hybrid-idx.cpp` (delivery-only memory type), `models/models.h` (#31142 output structs), `models/qwen4exp.cpp`.  **`D3`**: upstream reworked the same QSA/indexer/sparse selection (`#31243`, `#31009`, `#31443`) — the derived kq mask + pooled sparse decode may be redundant.  Gates: `gate-hc-fusions.sh MODEL.gguf`, `gate-qwen4exp-quant-coherence.sh` |
| **R13** | 12 campaign remainder (`be4749a56`) | the **refactor-heaviest** rung: `#31733` (model → graph-context member) touches 33 of its 81 files.  8 conflicts.  **`D8`** (which CPU wins survive `#31854` F32 src1), **`D15`** (upstream's own device-layer-input API `llama_set_embeddings_layer_inp_dev`/`llama_supports_embd_row`/`llama_batch_ext_add_embd_row` + `#31506` device-copy DFlash **vs** the delivery's F1 `llama_lf_*` — very likely a subsumption; both are in the tree now, unreconciled).  Gates: `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh` |
| **R14** | 13 MTP vocab subset (`6ee8eb290`) | 3 conflicts, all graph/model-member refactors.  Gate: the flag runs and is inert when unset |
| **R15** | 14 `-sm relaxed-tensor` (`59a3e5206`) | **0 conflicts**; upstream still denies those arches (`#31147` only added laguna to the deny list), so the block is not subsumed.  Test: a flint/mellum/mistral4 MoE loads and matches `-sm layer` text |
| **R16** | 15 streamed KV (`69d36099b`) | 9 conflicts; the memory instances after `#31141`/`#31200`, `#30541` state copies, `#31073` rotation, `#31570` kv_rows.  Gates: `gate-kv-residency.sh`, `gate-kv-window-split.sh`, `gate-spec-draft-window.sh` |
| **R17** | — | the release ladder: `make-patches.sh` → `make-release.sh --base a691b2345 --tip … --tree …` → `validate-set.sh` → `gate-per-block-build.sh` (all 16) → coherence (dense + host-expert + a hybrid) → `gate-prefill-logits.sh --record` **from a clean unpatched `a691b2345` build** then compare (+ `--sm tensor --device 0,1,2` for block 01, `--kv q8_0`) → MTP four-axis + `llama-batched-bench -npl 1,4,8` → every block gate → `check-gpu-faults.sh` → `fork-refresh.sh --build --gates --push` (personal fork only) → WORKLOG + headers + `release.json` |

**Per-rung recipe** (the same every time):

```bash
cd ~/llama.cpp && git status                 # clean
git cherry-pick <block sha>                  # RUNGS.md has every sha
# resolve; read both sides; never -Xours/-Xtheirs; never delete a hunk silently
git add -A && GIT_EDITOR=true git cherry-pick --continue
~/bin/build-llama-rocm                       # (clean) when CMake options changed
# coherence: dense 4B + the block's own mechanism (census at -lv 4) + its kill-switch A/B
```
Then add the rung log to `RUNGS.md` (resolutions table + findings), update the ledger row and the TODO/decision
register, and commit **the campaign** (this repo) with a message that states what was resolved and why.

**A rung's commit message** belongs to the delivery patch: keep the original block subject, append a short
"re-based onto `a691b2345`" paragraph naming the conflict shape and the resolutions (see any of R01-R10 for the
pattern).  Do **not** put campaign detail there — that is `RUNGS.md`/`WORKLOG.md`.

---

## §3  Open decisions and TODOs (the register)

`DECISIONS.md` holds the full text.  Status now:

| id | subject | status |
|---|---|---|
| `D1` | upstream's MoE cache vs the delivery's host-expert arena | **MEASURING** — first numbers only; see `TODO #291`; the maintainer deferred the analysis to its own session |
| `D2` | fused MoE gate/up+GLU (upstream `has_fusion` vs our `has_gate`) | **DECIDED** — complementary, two template arms, nothing dropped (R09) |
| `D3` | qwen4exp / indexer / sparse selection | **OPEN** — R12 |
| `D4` | recurrent rollback tracker | **DECIDED** — ours dropped, upstream's `rs_max` is the bound (R04) |
| `D5` | FA entry points after `#29510` | **DECIDED** — absorbed, no band kernel dropped (R06) |
| `D6` | `ggml : fix bf16` | closed — independent (R05) |
| `D7` | Meta compute views (upstream's 256 floor vs our margin) | `TODO #286` — measure at R17 |
| `D8` | which CPU wins survive upstream's CPU rewrite | **OPEN** — R13 |
| `D9` | graph cache / allocator / views | **OPEN** — R11/R13 |
| `D10` | all-reduce | resolved — both kept, crossover keyed on `n_ranks()` (R02) |
| `D11` | spec: hidden state, conv snapshots, checkpoints | **OPEN** — R13/R14 |
| `D12` | quantized-KV state / rotation / kv_rows | **OPEN** — R16 |
| `D13` | memory instances after `no_alloc` / `n_pad` | **OPEN** — mechanical, R11-R16 |
| `D15` | upstream's `embd_row` device-layer-input API vs the delivery's F1 `llama_lf_*` | **NEW, OPEN — R13.**  Likely a subsumption; decide with a measurement, not by preference |

Open TODO handles: **#286** (Meta headroom re-measure), **#287** (re-cut
`upstream/UPSTREAM-PR-mul-mat-id-dup.patch` against `#31047`'s `tile_spec` grid), **#289** (upstream PR: `find_slot()`
must swap `mem_cell::rs_max` — found and fixed in the delivery at R04), **#290** (the dispatcher merge debt —
**resolved at R11**, it lands with r236), **#291** (the arena-vs-upstream investigation).  New handles continue
from **#292**.

## §4  Documentation owed at R17 (collected per rung — do not lose these)

* `ENVIRONMENT.md`: `LLAMA_RS_ROLLBACK_DEBUG` loses its RSROLL half (RSPREP only, R04); the `GGML_HIP_NO_VMM`
  row (line ~962) says "open investigation" and must state the delivery's build requirement (R10); any variable
  the re-base made dead or new.
* `patches/README.md`: block 00's MUL_MAT_ID shape and its `-Wshadow` note (R01), block 03's tracker/warn text
  (R04), block 09's shape (R10), block 10's dispatcher split (R11).
* `INVARIANTS.md`: the recurrent rollback guard entry becomes "upstream's `rs_max` + our diagnostic" (R04);
  append the new facts the climb produced (`FLASH_ATTN_EXT`'s `src[5]` belongs to `kv_rows`, R06).
* `BASELINE.md` current baseline + `release.json` (R17), `WORKLOG.md` entry with the new base/tip/tree and the
  gate numbers, and this campaign moved to `archive/work/`.

## §5  Facts and traps this climb has already paid for (do not re-derive)

1. **`-Wshadow` rename sweeps are not carried.**  R01 dropped 1126 lines of pure renames across 21 files: they
   re-skin every later block and every future re-base.  Cost: ~94 `-Wshadow` warnings (none are errors;
   `-Werror` is off).  Maintainer approved.  Do not "restore" them; do not add new ones.
2. **`FLASH_ATTN_EXT`'s `src[5]` is upstream's `kv_rows`.**  Our derived kq mask (`cell_pos`/`tok_lo`/`tok_hi`)
   lives in `src[6..8]`.  Merged as-is, the derived probe silently failed and every batch fell back to the packed
   mask — no error, no warning, just the memory it exists to save (R06, TODO-grade trap for any new op storage).
3. **`ggml_cuda_launch_mm_ids_helper` emits upstream's `tile_spec` and, optionally, per-expert `expert_bounds`.**
   One compaction serves MMQ/MMF (`tile_spec`), MMB and the delivery's routed-compact dispatch and Q8_1 pair
   fusion (`expert_bounds`).  `tile_spec` may be `nullptr` (the seam writer returns early).  `mmq_args` carries
   both.  Upstream itself has **no** duplicate-id handling (one slot per token) — the delivery's per-occurrence
   compaction is the fix, re-expressed on their grid (R01/R07/R09).
4. **The host-expert arena needs VMM.**  `ggml_cuda_slab_init_locked()` declines without it and the run streams
   the experts with a W line.  `~/bin/build-llama-rocm` now passes `-DGGML_HIP_NO_VMM=OFF`.  On a `NO_VMM=ON`
   build *every* host-expert measurement is really stock `-ncmoe` (R10).
5. **A rung build needs a clean configure** when CMake options moved (`GGML_CUDA_FA_QUANTS` carries block 05's
   `iq4_nl` instances and makes blocks 00-04 fail to configure).  `~/bin/build-llama-rocm clean` — cheap.
6. **The fusion gates are inert until the dispatcher exists** (block 10).  On a blocks 00-09 tree
   `gate-norm-rows-alias.sh` reports `fused=0 declined=0` while `nofuse` passes: that is "cannot run yet", not a
   regression (R09 correction).
7. **Upstream absorbed three delivery mechanisms** — keep the *invariant*, drop the duplicate code: the conv-state
   snapshot store (`build_conv_state_store`, one `set_rows`, `Kw = min(n, K)`; R03/R04), the per-ubatch rotation
   fill (`#31073` consumes `get_k_rot()` at graph-build time; R06), and the pre-batch conv snapshot slot (the
   `rs_max` bound makes it unreachable; R04).
8. **`find_slot()` must swap `mem_cell::rs_max`** with `pos`/`src`/`seq_id` — upstream does not, and the delivery's
   moved-cell rollback then refuses legal rollbacks.  Fixed in the delivery, upstream PR owed (R04, TODO #289).
9. **A re-base moves numerics on purpose** (`#31854` F32 src1 on the CPU backend, the conv-state refactor, the MoE
   layout linearisation).  **r235-identical text is not a rung oracle**; a rung is judged by build + coherence +
   its own kill-switch A/B, and the KLD gate at R17.
10. **The measure-before-deciding rule is load-bearing.**  Two upstream changes looked like subsumptions and were
    not (`D4` was, `D2`/`-sm relaxed-tensor` were not); the arena one (`D1`) is measured but unexplained
    (TODO #291).  Bring numbers, not preferences.

## §6  Reading order for a cold start

1. this file, 2. `RUNGS.md` (the rung you are on, plus the ledger), 3. `DECISIONS.md` (the open question that
rung touches), 4. `CONFLICT-MAP.md` (what to expect and why), 5. `~/llama-cpp-rdna-boosts/AGENTS.md` (policy:
what not to do, the gate table, the pushing policy), 6. `INVARIANTS.md` §-entries the rung cites.

**Then run the §1.2 repro once on the parked patch** — it takes two minutes and puts you in the same state of
knowledge this session ended with.  Do not start R12 before R11 is committed and green: the ladder is ordered
because a later block's conflict is judged against the already-rebased earlier blocks.
