# HANDOVER — TODO items 6 (the gfx1201 half) and 10 (column-block the fused shared-expert epilogue)

> **OUTCOME (2026-09-12) — both items are closed; this brief is a historical record now.**
> **Item 10 landed** as a block-13 amendment: `shexp_down_gated_q8_0` is `ncols_dst`-templated with the
> token loop inside the k-block loop and `grid = (nrows)` — a **bit-identical** restructure (old-vs-new
> `libggml-hip.so` A/B: every gate hash equal, incl. all `W = 1..8` probe hashes, the §5 matrix, the §19
> `plain == n_max 3 == n_max 7` gate `68c0a24ed8d4` and MTP `0.87179`) that repays the band amendment's
> cost: `pl=8` 461.0 -> 475.4 t/s (+3.1 %), `pl=4` 299.1 -> 306.5 (+2.4 %), `pl=1` flat, with the fused
> default now ahead of the unfused reference at every width.  Canonical tip `124abba9e` (tree
> `d7c8e8984b8bd65838d8ae58c0f5de449d9c5d4d`), beta re-cut an 11th time (`a90f75896` / tree
> `ed6ee74df8b690c5a1584adb3f85c45eda70a09b`, patch 3 811 lines, only the `From` line changed).  See
> `WORKLOG.md` 2026-09-12, `patches/README.md` (the new section + the block-13 notes) and
> `GREEDY-PURITY.md` §24.
> **Item 6 is closed too, with two corrections to this brief's own premise:** the 35B-A3B Q4_K_M **does**
> take the routed-compact path (480 `mul_mat_q_routed_compact<(ggml_type)12, 32>` launches per pp512/ub512;
> the control is that the dispatch is prefill-only — 0 compact launches at decode), and
> `GGML_CUDA_DISABLE_MMQ_ROUTED=1` isolates only the compact enumeration, not the per-expert J selection.
> The port's bit-identity claim holds on both MoE models; perf reproduces (+4-11 % qwen4exp, +5-7.8 %
> 35B-A3B, tg flat).  See `wip/qwen4exp/gfx1201-porting.md` (2026-09-12 entry) and TODO item 6.

**Read this file first; it is self-contained.**  Shared environment/instrument rules live in
`wip/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` §2–§5/§12/§13 and in
`AGENTS.md` ("Critical facts"); the purity doctrine is `GREEDY-PURITY.md` (§11 the purity range,
§15–§17 the MoE band work, §19 the purity-first policy, §23 the shadowing-bug lessons).

## 1. Mission — two items, item 10 first

### Item 10 (the real work): make the fused shared-expert epilogue column-blocked (~2.4 % at wide verify batches)

`shexp_down_gated_q8_0` (`ggml/src/ggml-cuda/mmvq.cu`) is launched as **one block per `(output row,
token)`** — `const dim3 block_nums(nrows, ncols);` with `row = blockIdx.x`, `t = blockIdx.y` — so the
down-weight row is re-read once per token.  That is the *deliberate* cost of the 2026-09-11
band-uniformity amendment (item 5, block 13): the fused gate+down chain serves the whole band
(`n_tokens <= MMVQ_MAX_BATCH_SIZE`) and is pinned to the single-token reduction order so decode ==
verify.  Measured cost (35B-A3B Q4_K_M, `llama-batched-bench`, fused vs the
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` A/B): **pl=8 332.1 vs 341.5 t/s, pl=4 252.0 vs 254.2, pl=1
unchanged** ⇒ ~2.4–2.8 % at the widest verify batches.

**Task:** restructure the down-projection kernel into the `mul_mat_vec_q` pattern — template it on
`ncols_dst`, keep the **token loop inside the k-block loop**, per-token accumulators, one weight read
per (row, k-block) for the whole band — so the wall clock recovers ≈ the fused-vs-unfused gap while
**every (row, token) result stays bit-identical** to today's per-(row, token) version (same
per-token accumulation order ⇒ decode == verify preserved).

**Definition of done:** the new kernel is in block 13; on the 35B-A3B the pl=4/pl=8 numbers move to
≈ the unfused values (or better) with pl=1 unchanged; **every** purity/MTP gate below reproduces the
*current* hashes exactly (the change must be numerically invisible); the 15-patch set is regenerated,
the clean-apply sim passes, the beta is re-cut an 11th time, and the docs/TODO are updated.

**The two constraints the code imposes — read them before writing any code:**
1. **`nwarps` must stay pinned to the single-token value.**  The launcher calls
   `calc_nwarps(GGML_TYPE_Q8_0, 1, get_device_table_id(cc))` and says why: "*nwarps is pinned to the
   single-token value: it sets `blocks_per_iter` and hence the reduction order of the down projection,
   so a width-dependent `nwarps` would make the verify batch arithmetically different from the decode
   (the decode == verify invariant).*"  Keep that call, and keep the cross-warp reduction order
   (`sh_down[nwarps-1][warp_size]` then `warp_reduce_sum`) identical per token.
2. **Disabling the fusion at wide widths is NOT an acceptable fix.**  The fusion guard
   (`ggml/src/ggml-cuda/ggml-cuda.cu:4704-4770`) documents the opt-out as making "*decode and verify
   differ, as before the 2026-09-11 band amendment*" — i.e. the impure route.  Only column-blocking
   (bit-identical per token) satisfies the purity policy.
3. Also keep the epilogue's explicit `__fmul_rn` (no FMA contraction) and the
   `dst[t*nrows_dst + row]` layout, and leave `shexp_gate_sigmoid` alone (it is already one block per
   token and is not the bottleneck).

**Files:** `ggml/src/ggml-cuda/mmvq.cu` (`shexp_gate_sigmoid` ~2440, `shexp_down_gated_q8_0` ~2461,
`ggml_cuda_op_shexp_down_gate` ~2510 with the launch at ~2565), the declaration in
`ggml/src/ggml-cuda/mmvq.cuh:36`, and the fusion/launch contract in `ggml/src/ggml-cuda/ggml-cuda.cu`
(~4704–4770).  **The reference pattern to copy is `mul_mat_vec_q`** in the same file (its
`ncols_dst` template, `calc_nwarps`, `calc_rows_per_block` and per-token accumulation).

### Item 6 (small, mostly a validation record): the gfx1201 fallback/routed-compact probe

TODO item 6 is now scoped to this.  The plan's Phase 2.5 text is **stale** ("*with the RDNA3_5 kernels
inert*") — the routed-compact MoE MMQ is **enabled by default on RDNA4**
(`mmq_routed_compact_arch_ok(cc) = GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc)`,
`ggml/src/ggml-cuda/mmq.cuh` section "*RDNA3.5/RDNA4 routed-compact MoE MMQ*"), validated 2026-09-06
(`wip/archive/qwen4exp/discovery/2026-09-06-gfx1201-rdna4-routed-moe-mmq.md`: ub2048 tensor-split
prefill +4–8 %, tg flat, **same-seed text byte-identical compact vs plain**, 846 compact
launches/pp2048; opt-out `GGML_CUDA_DISABLE_MMQ_ROUTED=1`).  Its section comment *claims*
"Numerics are bit-identical to the plain `mul_mat_q` path" — that claim has not been re-checked since
the 2026-09-11 block-13 amendments (the mmvq ksplit, the rms_norm fold gate, the per-type mmvq cap,
the shexp epilogue).

**Task:** re-verify that claim on the current tip, and confirm the *plain/fallback* path is
unperturbed, i.e.
1. **Routed ON vs OFF must be numerically identical** on the model that actually exercises it —
   **qwen4exp** (`IQ4_XS` experts; the 35B-A3B Q4_K_M does *not* take the routed path, which makes it
   the natural *fallback* control): same-seed greedy text with `GGML_CUDA_DISABLE_MMQ_ROUTED` unset vs
   `=1`, plus the perf delta re-measured (expect the recorded +4–8 % prefill on the routed model, and
   no change on the Q4_K model).  Also record the compact launch count on qwen4exp for the same
   config to confirm the kernel still fires (the 2026-09-06 record saw 846/pp2048-ubatch).
2. **The whole purity/§19 gate list passes under both settings** (see §5) — the routed change must not
   have re-introduced a width- or batch-dependence.
3. If the claim does **not** hold (a divergence appears), that is a *finding*: stop, root-cause it (the
   kernel's numerics claim is that only the *tile enumeration* differs), and treat it as a block-13
   bug.  If it does hold, this is a **record-only** deliverable: a dated entry appended to
   `wip/qwen4exp/gfx1201-porting.md` (+ the TODO item-6 wording, + a `benchmarks/` record if the numbers
   are worth keeping).  No patch, no re-cut, in that case.

## 2. Where things stand (state as of 2026-09-11 (12))

* **Delivery repo** `/home/stew675/llama-cpp-rdna-boosts`, `main` = `5c58453` (== `origin/main`, clean).
* **Canonical fork** `/tmp/canon-llama`, branch `rdna-boosts`, **tip `484231cb9`**, net tree
  **`fc3c73da4ac68e92348043b992fb963b006e14df`**, 15 blocks, clean.  Build dir `build-base`
  (`build-base/bin/llama-cli`, `llama-bench`, `llama-batched-bench`, `llama-perplexity`,
  `test-backend-ops`).
* **The block chain (current SHAs — always re-read them, they shift with every amendment):**

  | block | sha | block | sha |
  |---|---|---|---|
  | 00 `1c7ab0e89` | structural/arch fixes (FA KV-split, Vulkan masked-V) | 08 `3500a6e26` | fused-core prefill + GPU bit-identical |
  | 01 `7611ef816` | adaptive MTP draft depth (**+ the n_max ≤ 7 cap**) | 09 `1aa602fc2` | meta-buffer headroom |
  | 02 `595bbdbc6` | fused chunked GDN prefill (**+ K-independent whole-batch**) | 10 `17adb14a7` | k-quant boosts (mmvq VDR) |
  | 03 `90f43d78e` | BF16 KV + native-BF16 FA | 11 `3e3d369ad` | skip CUDA graphs for multi-token prefill |
  | 04 `86c4ee33e` | RDNA4 WMMA FA + Q6_K mmq | 12 `702e2ccb8` | hybrid HIP all-reduce (RDNA4) |
  | 05 `7b56c022e` | CPU bit-identical decode/verify | **13 `1672225bc`** | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split ← item 10's owner** |
  | 06 `d57443439` | host-buffer revert marker | 14 `484231cb9` | qwen4exp support (**+ mixed-K/V hard reject**) |
  | 07 `ea63fe857` | meta device-wrapper skip | | |

  `git log --format='%h %s' | grep 'block 13:'` gives the current block-13 SHA.
* **Beta:** the Block 15 patch is the **10th re-cut** — base `484231cb9` → beta tip `a796a1d49`, tree
  `b48565e69f77f0c20a20cd75d87c2559d11e6de2`, patch 3 811 lines, in
  `beta/block-15-campaign-wins/block-15-campaign-wins.patch`.  It builds (`/tmp/blk15y/build-rec10`)
  and its smoke gates pass.  **Amending block 13 for item 10 invalidates it ⇒ an 11th re-cut is part
  of the item-10 landing** (the flow is in §6).
* **Models:** 35B-A3B MoE `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (item 10's
  A/B + the MoE purity gates); qwen4exp `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
  (**`/models/…`, not `/llm/models/…`**) + its MTP draft `/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`;
  27B `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (own MTP head, no `-md`); 4B
  `/home/stew675/Qwen3.5-4B-Q8_0.gguf`; gemma-4-E4B `/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf`.
* **Build:** `BUILD_DIR=build-base EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
  (the override is mandatory with CMake ≥ 4.3), or the fast loop
  `cmake --build build-base --target llama-cli llama-batched-bench llama-perplexity test-backend-ops -j 16`.
  `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`.  A from-scratch build is ~2–4 min when the box is
  healthy; a HIP-backend rebuild after touching a `.cu` is ~1–3 min.
* **Instruments:** `wip/kv-quant-purity-followups/tools/` — `qperf.sh` (parameterised perf table),
  `leakgate.sh` (random-text causal-leak gate), `qsa-ppl-oracle.sh` (dense-vs-sparse perplexity gate),
  `rv.sh`/`f3run.sh` (text/width/MTP drivers), `node-dump-instrumentation.patch` (per-node hashes),
  `logits-dump-kv.cpp` (the width probe).  `benchmarks/mtp-adaptive-methodology.md` is the MTP gate.

## 3. The item-10 measurement (do this first, on the *current* build, to confirm the premise)

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
B=/tmp/canon-llama/build-base/bin
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
bench() { # $1 = label, $2.. = env
  shift 0; local lab=$1; shift
  printf '%-22s ' "$lab"
  env "$@" HIP_VISIBLE_DEVICES=0,1,2 "$B/llama-batched-bench" -m $M -c 32768 -b 2048 -ub 512 \
    -npp 2048 -ntg 128 -npl 1,2,4,8 -ctk f16 -ctv f16 -fa on -ngl 99 -sm tensor -mg 0 --output-format jsonl \
    | python3 -c 'import sys,json
row=[]
for l in sys.stdin:
    l=l.strip()
    if l.startswith("{"):
        d=json.loads(l); row.append("%d:%7.2f" % (d["pl"], d["speed_tg"]))
print(" ".join(row))'
}
bench "fused (default)"      X=1
bench "unfused (reference)"  GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1
```

Run each twice, interleaved, and keep the CSV rows plausible (see the traps in §7).  The recorded
baseline for this exact A/B is **pl=8 332.1 (fused) vs 341.5 (unfused), pl=4 252.0 vs 254.2, pl=1
equal** — that is the gap item 10 exists to close (the *fused* path is the default and must win or at
least not lose).

After the kernel change, re-run the same A/B: expect the fused numbers at pl=4/pl=8 to reach ≈ the
unfused ones (and the unfused ones to be unchanged, since that path is untouched).

## 4. Gate list (all mandatory; the change must be numerically invisible)

**Band purity (§19 — the invariant this kernel exists to protect):**
* MoE 35B-A3B: the probe default `W = 1,2,3,4,8` → all **`ac8825358d9adfda`** (the post-item-5 uniform
  fused value); with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` all **`bd138ad2326fbbf2`** (the uniform
  unfused reference).  Both must be unchanged.
* `--spec-type none == draft-mtp n_max 3 == n_max 7` byte-identical on 35B-A3B and on qwen4exp
  (`804de0576868` f16 tensor, `886292b17a93` q4_1 tensor, `acd18ad2d55c`/`fcb2d47f94cf` iq4_nl — see
  `BETA-TESTING.md` §4d for the iq4_nl caveat) with `-ctk`/`-ctv` matched.
* **MTP acceptance must not regress:** 35B-A3B **`0.81707`** (the post-item-5 value; plain 96.9 t/s vs
  MTP 167.3 t/s), qwen4exp f16 `0.47009` (pos-1 0.615), 27B f16 `0.82716`.  Note the acceptance line
  needs `--log-verbosity 4` and it interleaves into the text ⇒ run the text gates separately.
  (`benchmarks/mtp-adaptive-methodology.md` §"purity range": the guarantee is `n_max <= 7`, and the CLI
  now clamps to 7 — `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` is the escape hatch.)
* 27B f16 layer `4089b4d40b91090c`, 27B f16 tensor `91434ea90f2cbfa0`, qwen4exp tensor f16
  `dcf1ae667f730879`, MoE f16 `ac8825358d9adfda` (probe, `W=1..8` single hash) — all from
  `WORKLOG.md`'s item-5/§19 entries; re-measure before/after rather than trusting them blindly.
* Backend suites: `test-backend-ops test -o FLASH_ATTN_EXT`, `-o FLASH_ATTN_QSA`, `-o GATED_DELTA_NET`
  (all must report the pass summary), plus `test-arg-parser` if `common/` is touched.
* Determinism: two identical runs → identical hash (the shexp kernel is where a stride/accumulator bug
  would show up).

## 5. Item 6's gate list

* qwen4exp: routed ON vs `GGML_CUDA_DISABLE_MMQ_ROUTED=1` → **same-seed greedy text identical** (the
  2026-09-06 claim) + the prefill delta re-measured (`tools/qperf.sh` or `llama-bench`), and a
  `rocprofv3 --kernel-trace` launch count to confirm the compact kernel still fires (or `--log-verbosity 4`
  if a log line names it).
* 35B-A3B Q4_K_M (the fallback/plain path): routed ON vs OFF → identical text and identical perf (the
  routed kernel must not be involved) — this is the "fallback path unperturbed" half of Phase 2.5.
* The full §19 purity + MTP list under both settings.

## 6. Landing (item 10; item 6 is records-only unless it finds a bug)

```sh
# 1) amend the owner block (block 13) — ascending order matters if you amend more than one
cd /tmp/canon-llama
B13=$(git log --format='%h %s' | grep 'block 13:' | cut -d' ' -f1)
B12=$(git log --format='%h %s' | grep 'block 12:' | cut -d' ' -f1)
# (edit files, then) — note: `git rebase -i` refuses a dirty tree ⇒ stash first if you already edited
GIT_SEQUENCE_EDITOR="sed -i -e '/^pick $B13 /s/^pick /edit /'" git rebase -i $B12
git add -A && git commit --amend --no-edit -q && git rebase --continue

# 2) regenerate the delivery set + refresh the combined patch
cd /home/stew675/llama-cpp-rdna-boosts
NEW=$(git -C /tmp/canon-llama rev-parse HEAD)
./scripts/make-patches.sh /tmp/canon-llama 9113cc188 ${NEW:0:9}
sed -i "s/^TIP=.*/TIP=\"\${3:-${NEW:0:9}}\"/" scripts/make-patches.sh      # update the default tip (line ~44)
git -C /tmp/canon-llama diff 9113cc188 $NEW > rdna-boosts-all.patch        # NOT written by the script!

# 3) clean-apply sim (strict 15/15, 0 whitespace, tree equality)
rm -rf /tmp/simx && git clone --no-local -q /tmp/canon-llama /tmp/simx && cd /tmp/simx && \
  git checkout -q 9113cc188 && git branch -D rdna-boosts -q && \
  bash /home/stew675/llama-cpp-rdna-boosts/scripts/apply-all.sh . ; \
  git rev-list --count 9113cc188..HEAD ; git rev-parse HEAD^{tree}         # vs the canonical tree
cd /tmp/simx && BUILD_DIR=build-simx EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714

# 4) beta 11th re-cut (block 13 changed ⇒ the 10th re-cut no longer applies)
cd /tmp/canon-llama && git worktree add -f -q /tmp/blk15z $NEW && cd /tmp/blk15z && \
  git checkout -q -b blk15-rec11 && \
  git am -3 /home/stew675/llama-cpp-rdna-boosts/beta/block-15-campaign-wins/block-15-campaign-wins.patch
# resolve (expect a qwen4exp.cpp conflict; a llama-context.cpp one is possible), build, run the smoke
# gates (qwen4exp f16 sparse 804de0576868, oracle sparse 6.5394 / dense 6.5377), then export:
git format-patch --stdout --start-number 15 -1 HEAD > /tmp/b15-11.patch
sed -i '4s/^Subject: \[PATCH\] /Subject: [PATCH 15\/15] /' /tmp/b15-11.patch   # format-patch drops the /15
cp /tmp/b15-11.patch /home/stew675/llama-cpp-rdna-boosts/beta/block-15-campaign-wins/block-15-campaign-wins.patch
# round-trip: fresh worktree at $NEW + `git am -3` the exported file → tree must match
```

**Docs to update** (newest-first; never rewrite a dated record — annotate): `WORKLOG.md` (a new entry);
`patches/README.md` (the `0013` row + a dated section with the kernel diff, the A/B table and the gates);
`AGENTS.md` (the block-13 clause; if the change alters any policy/expectation, the Critical-facts
bullet); `TODO.md` (item 10 → Closed with the numbers, item 6 → whatever the probe concludes);
`GREEDY-PURITY.md` (a §24 only if the change teaches something about band purity); the beta
`README.md`/`BETA-TESTING.md`/`HANDOVER.md` (the 11th re-cut + the re-run gate table); and for item 6 a
dated entry appended to `wip/qwen4exp/gfx1201-porting.md`.  Commit + push **only** to the delivery
repo's `origin`.

## 7. Traps (each of these cost time in the session that wrote this)

* **`-ctk` and `-ctv` must now MATCH** — mixed K/V cache types were hard-rejected on 2026-09-11
  (block 14): `-ctk q8_0` alone (V defaults to f16) fails context creation.  Every command in this
  document therefore passes both.
* **`--spec-draft-n-max` is clamped to 7** with a visible `E`-level notice; add
  `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` to test a larger depth (and note `n_max > 15` also re-introduces the
  K-dependent chunked-GDN boundary).
* **`rdna-boosts-all.patch` is not generated by `make-patches.sh`** — regenerate it by hand
  (`git diff 9113cc188 <tip>`), and remember `make-patches.sh`'s default tip string lives near line 44.
* `git rebase -i` **refuses to start with a dirty tree**: edit first, `git stash push -q -- <paths>`,
  rebase with the `edit` marker, `git stash pop` at the stop, then `git add -A && git commit
  --amend --no-edit`.  A bare `--amend` after `git apply` commits nothing (unstaged changes).
* `scripts/apply-all.sh` fails with `ERROR: branch rdna-boosts already exists` in a clone that carries
  the branch — delete it first (the sim recipe above does).
* **Warnings emitted at *argument-parse* time are invisible** at the default verbosity (llama.cpp-wide:
  its own DEPRECATED notices too).  Only `E`-level messages reach `llama-cli` by default — and if you
  add a user-facing notice, put it where the logger is live (see `common/common.cpp`'s clamp for the
  pattern that works: `LOG_ERR` + name the env var).
* **Never run benches in parallel** and never run a build during a bench.  This box can intermittently
  run ~2× slow (93 GiB tmpfs `/tmp`, a 93 GB mmap'd model, zram swap): if a CSV row looks implausible
  or is missing (`LOADFAIL`), re-run and interleave a control you know.  Batching two builds at once
  (as happened while writing this) is also a self-inflicted slow phase.
* `rocprofv3 --kernel-trace` distorts wall-clock (it serialises) — use it for structure/counts, not for
  deltas; `llama-bench`/`llama-batched-bench` are the timing instruments.
* `--log-verbosity 4`(or `--verbose`) **interleaves log lines into the generated text** ⇒ take text
  hashes at the default verbosity, and the `draft acceptance` line in a separate run.
* `llama-cli` needs `--single-turn` (or it waits for input); `llama-batched-bench` needs
  `--output-format jsonl` for the per-`pl` rows parsed above.
* A change that is a **no-op at the gate config** is the strongest control: for item 10 that means
  "same hash everywhere, faster wall clock" — if any hash moves, the kernel is not bit-identical and
  the change must be fixed, not explained.

## 8. Reference numbers worth protecting (from the item-5 record, `WORKLOG.md` + `GREEDY-PURITY.md` §17)

| what | value |
|---|---|
| MoE probe f16, fused (default) | `W = 1,2,3,4,8` all `ac8825358d9adfda` |
| MoE probe f16, unfused (`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`) | all `bd138ad2326fbbf2` |
| 35B-A3B MTP acceptance (post-item-5) | `0.81707`, plain 96.9 t/s vs MTP 167.3 t/s |
| item-10 A/B, `llama-batched-bench` tg/batch | pl=8 332.1 fused vs 341.5 unfused; pl=4 252.0 vs 254.2; pl=1 equal |
| qwen4exp f16 sparse text | `804de0576868` |
| qwen4exp oracle (perplexity) | sparse `6.5394` / dense `6.5377` |
| routed-compact MoE MMQ (RDNA4) | enabled by default; +4–8 % prefill (ub2048 tensor-split), tg flat, compact-vs-plain byte-identical, 846 launches/pp2048; opt-out `GGML_CUDA_DISABLE_MMQ_ROUTED=1` |
