# HANDOVER — `TODO` #303: the dense MMQ kernel is 1.4 % slower than r235 at identical geometry

> **CLOSED 2026-11-08 — no delivery change (`RUNGS.md` `P7`).**  The premise below is refuted: r236's dense
> `mul_mat_q` is **instruction-for-instruction identical to unpatched `a691b2345`'s** (1436 instructions, 0 diff
> lines), so the delivery has no code presence in that kernel.  Four-build attribution (2 interleaved rounds
> each): base commit **+1.31 %**, r235's old body **−0.36 %**, delivery on the new base **+0.09 %**.  Two candidate
> fixes were compiled and refuted at ISA level (a `min()` removal: 1436 → 1435 instructions; a type-gated NVFP4
> prologue: byte-identical) and reverted; the 0.36 % allocation win was dropped by decision.  Record and the
> corrected numbers (794 dense launches, not 986; single-run census floor ±1.1 %):
> [`../../benchmarks/2026-11-08-dense-mmq-303.md`](../../benchmarks/2026-11-08-dense-mmq-303.md).  Everything below
> is kept as the item's history.

**Read order:** this file → [`RUNGS.md`](RUNGS.md) `P4`/`P5` (how #301/#302 were found and fixed) →
[`../../benchmarks/2026-11-08-rebase-r235-vs-r236-localisation.md`](../../benchmarks/2026-11-08-rebase-r235-vs-r236-localisation.md)
(the census and the gate numbers) → `DECISIONS.md` (`D2` is the MMQ-adjacent decision) →
[`HANDOVER.md`](HANDOVER.md) §0.1/§0.2 (hard rules, the boxes) → `../../AGENTS.md` → `../../TODO.md` #303.

## §0 State (verified at handover, 2026-11-08, end of the #294/#297/#298/#301/#302 session)

| item | value |
|---|---|
| fork checkout | `~/llama.cpp`, branch **`rebase-r236`**, tip **`2ff384571`**, tree **`eda002cc2`**, clean |
| `master` there | pristine upstream **`a691b2345`** — never commit to it |
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-r236`**, head **`3ab80808`**, pushed |
| builds on the primary box | `build-rocm` = r236 (newest), `build-rocm-r235` = the shipped set on the old base, **`build-rocm-base` = unpatched `a691b2345`** |
| prefill-logit base | recorded for the **f16** arm from `build-rocm-base` (`~/.cache/rdna-boosts/gates/prefill-logit-gfx1201-Qwen3.8-27B-Q8_0.kld`); the **`--kv q8_0`** arm is still owed at R17 |
| MoE prefill state | **at parity** after #301 (ids compaction) + #302 (FA loader): 30B-A3B +0.1/+1.8/−0.7 %, 35B-A3B 3 GPU −1.5/−0.3/+1.9 % |
| dense prefill state | **−1.3 … −1.6 %** on the 27B — this item |

## §1 What is known, and what is not

**Measured** (`rocprofv3 --kernel-trace`, dense 27B Q8_0, `llama-bench -p 512 -n 0 -ngl 99 -r 1`, one GPU,
r235 vs r236 after #302):

| | r235 | r236 | note |
|---|---|---|---|
| total kernel time (all kernels) | 649.9 ms | 658.0 ms | **+1.26 %**, matches the −1.3…−1.5 % wall clock |
| `mul_mat_q` | 535.4 ms | 543.0 ms | **+1.4 %** — the dominant kernel, 986 launches in both |
| the instantiation | `void mul_mat_q<(ggml_type)8, 128, false, (ggml_prec)30, false, false>` | **identical name** | type Q8_0, J-ish template arg 128, same prec/flag bools |
| grid / workgroup / count | (128, …) / 32 / 986 | **identical** | from the trace's `Grid_Size_*`, `Workgroup_Size_X` |
| **VGPR count** | **232** | **216** | the one thing that differs — same source name, different register footprint |
| `flash_attn_ext_f16` (27B) | 8.5 ms / 476 B scratch | 8.8 ms / 640 B | #302 improved this from 10.3 ms / 768 B; still not r235's 476 B |
| everything else | — | — | within noise; `rms_norm_f32` −2 %, `gdn_*` ±3 %, `k_bin_bcast` flat |

**So:** the same template instantiation, the same launch count, the same grids and blocks, and a *different*
register allocation. That is either (a) a different runtime `J`/`ntj`/`nthreads` choice feeding the same
instantiation, (b) a body difference (the delivery's `mmq.cuh`/`mmq_args` hunks on upstream's `tile_spec` grid),
or (c) both.  **Not yet known:** which.  Do not guess — the census distinguishes (a) from (b) cheaply (below).

Related facts worth carrying:
* the 27B's FA is *also* still 164 B above r235's scratch (640 vs 476) after #302 — a **different**
  `flash_attn_ext_f16` instantiation from the 30B's (head/shape differs).  If #303 turns out to be small,
  closing that 164 B is the next-best dense win, and `P5`'s recipe (compile-time specialisation, `__noinline__`
  for the rare variant) is the tool that worked.
* `mul_mat_q`'s J selection lives in `ggml_cuda_mmq_select_J()` (`mmq.cu`), and the row-tile count is
  `ntj = ceil(n_tokens*n_expert_used/J) + n_experts` for the ids path and the plain `ceil(ne01/J)`-style count
  for a dense matmul.  **R01 and R08/R09 rewrote exactly this** (upstream's `tile_spec` grid, `mmq_args`
  carrying `expert_bounds` next to `tile_spec`), so the re-base is the prime suspect.

## §2 The instrument (this is the whole session's first hour)

```bash
cd /tmp && mkdir -p prof303 && cd prof303
M=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
# one trace per binary — the profiler perturbs wall clock, so judge KERNEL time, not t/s
HIP_VISIBLE_DEVICES=0 rocprofv3 --kernel-trace -f csv -o r235 -- ~/llama.cpp/build-rocm-r235/bin/llama-bench \
    -m $M -p 512 -n 0 -ngl 99 -r 1 >/dev/null 2>&1
HIP_VISIBLE_DEVICES=0 rocprofv3 --kernel-trace -f csv -o r236 -- ~/llama.cpp/build-rocm/bin/llama-bench \
    -m $M -p 512 -n 0 -ngl 99 -r 1 >/dev/null 2>&1
```

Then aggregate per kernel name (`Kernel_Name`, `Grid_Size_X/Y/Z`, `Workgroup_Size_X/Y/Z`, `VGPR_Count`,
`SGPR_Count`, `LDS_Block_Size`, `Scratch_Size`, `Start/End_Timestamp`) — a ready-made script is in this file's
history in `P4`/`P5` (`python3` + `csv.DictReader`, group by `Kernel_Name`); **normalise the name's `<...>`** when
you want base names, but **keep the full name** when you compare instantiations (the template args are the J and
the flags).

**To separate (a) from (b)** — a runtime `J`/`ntj` difference shows up in the *launch geometry*, which the trace
records: `Grid_Size_Y` on the ids path is `ntj`-ish and `Workgroup_Size_X` is `nthreads`.  If those are identical
too, the difference is in the body (b).  If they differ, read `ggml_cuda_mmq_select_J()`'s inputs for this shape
(`type=Q8_0`, `fallback = ne[1] % 128 != 0`, `ncols_opt`, `has_ids=false`, `nchannels_y`, `ncols_dst`) in both
trees — `git diff rdna-boosts rebase-r236 -- ggml/src/ggml-cuda/mmq.cu | grep -n "select_J\|ntj\|nthreads"`.

Runtimes: one trace ≈ 20 s (27B load) — cheap enough to iterate.

## §3 Ranked hypotheses (cheapest test first)

1. **A different runtime J / nthreads for the dense shape.**  Check the trace's `Workgroup_Size_X` and
   `Grid_Size_Y` first; then diff `ggml_cuda_mmq_select_J` and the `ntj` computation between the trees.  If the
   *inputs* changed (e.g. `ncols_opt`, or `has_ids`/`nchannels_y` being passed differently after the `tile_spec`
   rewrite), the fix is to pass the pre-re-base values for the dense path — a 1-3 line change with no kernel
   change, and the census confirms it immediately.
2. **The `mmq_args`/`tile_spec` re-expression changed the dense body.**  Same geometry, different VGPR ⇒ plausible.
   Diff `ggml/src/ggml-cuda/mmq.cuh` between the trees around `mul_mat_q`'s template and `process_tile`'s ids
   handling; remember R01's log: the delivery's `mmq.*` hunks were **dropped in favour of upstream's**
   (`tile_spec` grid, `ntj`, `fusion_device`) and only `mmul_mat_f_switch_ids` and the per-shape `nwarps` were
   kept, so a *lost* delivery tuning is a live possibility (compare `P4`'s lesson: a lost win, not an added cost).
3. **Upstream's own body is slower for this shape.**  Test with `build-rocm-base` on the same arm: if
   unpatched `a691b2345` is also ~543 ms, it is upstream's change and the honest options are "accept + document"
   or an upstream-style local fix — measure before choosing.

## §4 Gates and acceptance

* **Instrument**: the `rocprofv3` census above (kernel time + VGPR + geometry).  Wall clock is the final word but
  needs 3+ interleaved rounds at `-r 5` and the box is noisy (±4 % swings have been seen on the 3-GPU arm).
* **`scripts/gate-prefill-logits.sh`** — mandatory: `mul_mat_q` is a prefill kernel.  A base recorded from
  **unpatched `a691b2345`** already exists for the f16 arm (recorded 2026-11-08); the run must read mean KLD
  ≤ 0.005 and same-top-p ≥ 98 %.  If a J/tile change moves the numerics, `--ab` the exact-fallback arm.
* **`test-backend-ops -o MUL_MAT_ID`** 931/931 and the dense `MUL_MAT` suite (a J change touches both paths).
* **Same-seed greedy text** on the four coherence arms (`AGENTS.md` → Verify).
* `scripts/check-gpu-faults.sh --since <session start>`.
* **Acceptance**: the dense 27B pp512 arm within ~1 % of r235 (it is −1.3…−1.6 % now), or a measured, understood
  cost with the mechanism named in `benchmarks/`.

## §5 Traps this campaign already paid for (relevant to this item)

1. **"It passed once" is not evidence** for anything pool/layout shaped, and **register allocation is layout
   shaped**: #302's first two fixes *looked* right and made things worse (19.3 ms, then 35.5 ms with 1504 B of
   scratch).  Always re-read the trace's `Scratch_Size`/`VGPR_Count` and the kernel duration together.
2. **A no-op switch reads like a measurement.**  `GGML_CUDA_DISABLE_MMQ` **does not exist** in this tree (only
   `GGML_CUDA_DISABLE_MMQ_ROUTED`); an A/B on it measured the default twice.  Before quoting an env arm, grep
   that the variable is read.
3. **`-lv 4` is a census, not a text arm**; and the prefill runs need `HIP_VISIBLE_DEVICES` pinned.
4. **The box is noisy**: firefox/kwin at 2-3 % CPU moved a 3-GPU arm by ±4 %.  Interleave arms, use `-r 5`,
   quote per-round ratios, never a min/max of raw t/s.
5. **`build-rocm-base` must keep existing** — R17 step 5 re-records the prefill-logit base from it.
6. Never `git apply` the block series; never push to upstream; the delivery's `patches/` are cut only at R17.
7. **The attribution base is a SHA, not a branch.**  `build-rocm-base` must be built from a detached
   `git checkout a691b2345`, never from `master` — `master` is only *supposed* to be that commit.  A local
   commit on `master` (a #289-style candidate staged for `git apply --check`) silently changes what every
   "unpatched upstream" arm measures and what R17 step 5 records as the KLD reference.

## §6 What else is left in this campaign (context, not this item)

| handle | state |
|---|---|
| `#279` | narrow-2 `cudaMalloc` vs arena eviction: one A/B on r236 (closed pre-re-base) |
| `#286` | Meta compute headroom (`D7`): measure which constant sizes the buffer, then narrow or keep |
| `#287` | re-cut `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` onto `#31047`'s `tile_spec` |
| `#289` | upstream PR owed: `find_slot()` must swap `mem_cell::rs_max` |
| `#290` | the dispatcher merge debt: verify it in the **re-cut** patch at R17 step 1 |
| `#291` | the arena-vs-upstream-MoE-cache measurement (`D1`) — its own session, maintainer-deferred |
| `#295`/`#296` | R17 ladder arms (host-expert two-slot DFlash; `D8`'s CPU wins) |
| `#300` | gfx1151 attendance pool — **parked**, mitigated (the knob declines there) |
| `#303` | **this item** |

Then **R17** (the release ladder) — `HANDOVER-REBASE-TODO.md` §3, with the deltas this session added: block 11
owns `indexer-score.cu`, block 10 owns the `mul_mat_cublas_impl` pool fix, block 00/08 owns the ids compaction,
block 05 owns the FA loader, and the prefill-logit base for the **f16** arm is already recorded from unpatched
`a691b2345` (the `--kv q8_0` arm is not).

## §7 Command reference

```bash
# state
cd ~/llama.cpp && git log --oneline -1 && git status --short --branch && git rev-parse HEAD^{tree}
cd ~/llama-cpp-rdna-boosts && git log --oneline -1

# build (incremental is fine unless a widely-included header moved)
~/bin/build-llama-rocm                 # r236
# unpatched a691b2345, for attribution and R17 step 5's KLD base.  PIN THE SHA, never `master`: a local
# commit on `master` would silently move the reference base (it did once, #289, 2026-11-08).
git checkout --detach a691b2345 && BUILD_DIR=build-rocm-base ~/bin/build-llama-rocm && git checkout rebase-r236

# the census (kernel time + registers + geometry, one arm)
HIP_VISIBLE_DEVICES=0 rocprofv3 --kernel-trace -f csv -o out -- ./build-rocm/bin/llama-bench \
    -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -p 512 -n 0 -ngl 99 -r 1

# wall clock, the only defensible shape: interleaved, per-round ratios
for rd in 1 2 3; do
  a=$(HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench -m $M -p 512 -n 0 -ngl 99 -r 5 | grep -oE "\|\s+[0-9.]+ ±" | tr -d '|± ')
  b=$(HIP_VISIBLE_DEVICES=0 ./build-rocm-r235/bin/llama-bench -m $M -p 512 -n 0 -ngl 99 -r 5 | grep -oE "\|\s+[0-9.]+ ±" | tr -d '|± ')
  python3 -c "print('round $rd: r236', $a, 'r235', $b, '%+.1f%%' % (100*(float('$a')/float($b')-1)))"
done

# gates
./scripts/gate-prefill-logits.sh
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/test-backend-ops -o MUL_MAT_ID
./scripts/check-gpu-faults.sh --since "90 min ago"
```
