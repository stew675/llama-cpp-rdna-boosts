# BETA TESTING — `--host-experts window` (3-GPU / `-sm tensor` in particular)

The candidate is [`host-experts-window.patch`](host-experts-window.patch) (sha256 in
[`README.md`](README.md)).  What is already proven on 1 GPU, and what is being asked of you here:

* **proven** — same-seed text identical, MTP four-axis acceptance identical to 5 decimals, all kill
  switches identical, correct under a 10 GiB cgroup, +17-35 % decode and −24.8 GB peak RSS on the
  cells that spill;
* **not proven** — anything with more than one device, and specifically the **per-host-tensor
  registration refcount** (`g_win_tensors`): under `-sm tensor` three per-device children share one
  host master, and a second `cudaHostRegister` of the same range is refused as an overlap.  The code
  shares one registration across them; that path has never run.

## 1. Get the tree

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 6050b467c                       # release.json `.base`
bash <repo>/scripts/apply-all.sh --include-beta .
```

Expected: 16 block commits, then one beta commit, then

```
NOTE: this checkout includes 1 BETA candidate(s) and is NOT the release tree.
      release.json tree: 1bac1001f1780a5907fbce256ceeeaac8b8561ec
      applied tree:      01e1abc96d9ffb83eea6fef66a71649940ba3996
```

If the applied tree is not `01e1abc96d9ffb83eea6fef66a71649940ba3996`, stop: your base or your repo
checkout has drifted (report it, do not chase it).  The patch sha256 is
`3548105406642dff8afbfa0fb82422f8d18dd603549451d22691855077927d63`.

Build with the usual flags — **keep the ROCm include-order guard** (`-isystem<rocm>/include`); without
it Fedora's ROCm 7.1 headers win and every flash-attention TU fails on `__ocml_exp10_f32`:

```bash
~/bin/build-llama-rocm                      # the fork's canonical script (BUILD_DIR=build-rocm)
# or, by hand:
cmake -S . -B build-rocm -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGGML_RPC=1 -DGGML_NATIVE=1 \
      -DGPU_TARGETS="gfx1201;gfx1100;gfx1151" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_FLAGS="-isystem/opt/rocm-10.1.0-gfx120X/include" \
      -DCMAKE_CXX_FLAGS="-isystem/opt/rocm-10.1.0-gfx120X/include" \
      -DCMAKE_C_FLAGS="-isystem/opt/rocm-10.1.0-gfx120X/include"
cmake --build build-rocm --target llama-cli llama-bench llama-server llama-batched-bench -j 16
```

A cold build of the beta tree was verified clean (13 files differ from r206).

## 2. Smoke (must pass before anything else)

```bash
export HIP_VISIBLE_DEVICES=0,1,2            # NEVER leave this unset on the multi-GPU box
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
MOE_EXPERT_CACHE_MIB=4096 MOE_HOST_POOL_MIB=2048 ./build-rocm/bin/llama-cli -m $M \
  -ngl 99 -ncmoe 40 --host-experts window -c 2048 \
  -p "The capital of France is" -n 32 --seed 42 --temp 0 --no-display-prompt --single-turn
```

Pass: **150 chars, `sha=4ac3a523160f`**, and the mode is announced once:
`host_experts_mode=WINDOW: ONE residency tier ...` (add `-v` to see it; llama-cli hides INFO).

Then the same with `--host-experts pool`.  Both must give the reference hash.  (On this box's Q4_K_M
and Q8_0 they do; on Q8_K_XL they do not — that file's MTP head is damaged, see the README.)

## 3. The arm that matters: `-sm tensor`

```bash
export HIP_VISIBLE_DEVICES=0,1,2
for he in pool window; do
  MOE_EXPERT_WINDOW_TRACE=1 ./build-rocm/bin/llama-cli -m $M -ngl 99 -ncmoe 40 -sm tensor -mg 0 \
    --host-experts $he -c 2048 -p "The capital of France is" -n 32 --seed 42 --temp 0 \
    --no-display-prompt --single-turn
done
```

**The one assertion this beta exists for** — `MOE_EXPERT_WINDOW_TRACE=1` prints at exit:

```
WINDOW EXIT tensors=N reg_fail=0 ml_fail=0 alias_d2d=0 reg_bytes=... cold_eager=... cold_dev=... remap_maps=...
```

Read it in this order:

* **`reg_fail` must be `0` — this is the headline.**  The registration is shared per HOST TENSOR: the
  first per-device child of a master registers the range and the other children adopt it.  If the
  refcount is wrong, the 2nd/3rd child issues its own `cudaHostRegister` for the same range, the
  runtime refuses it as an overlap, and that device silently loses its alias — which shows up as
  **`reg_fail` = (devices − 1) × H**, *not* as a short `tensors`.  So `tensors = H` alone does not mean
  it worked; `reg_fail = 0` is what does.
* `tensors` should equal the number of **host expert tensors**, one per master (the `-v` line
  `host pool (...): H host tensors` gives `H`).  Cross-check: `MOE_EXPERT_TENSOR_REG=0` must give
  `tensors=0`.
* `cold_dev > 0` on the decode band (the DEVREMAP is encoding cold experts at all) and it should move
  when the registration moves; `cold_dev = 0` while the arena is sub-residency means the cold seam is
  not live somewhere.
* The functional check that covers everything the counters cannot: on the `-sm tensor` arm the
  `window` and `pool` runs must agree on **acceptance** (exactly) and on **text** (expected; if they
  differ, first re-check with both arms on `MOE_EXPERT_CACHE_MIB` fixed — see §5 of the README).

**NEW, UNVERIFIED HERE: the gather's capability predicate under `-sm tensor`.**  The arena-aware gather is
default-on, and the scheduler now asks `moe_cache_gather_ok` before it skips whole-table staging.  The meta
backend forwards that query (true only if *every* device says yes) — that forwarding has **never run on
hardware**, because a 1-GPU box never constructs the meta backend.  Its failure mode is loud and cheap to
see: if a device declines *after* staging was already skipped, the split drops onto the host path and
prefill collapses (measured on a simple backend in the pool case: **49.6 t/s** on the first request, versus
~232-285 for the same arm with the predicate in place).  So measure prefill with the switch **on and off**:

**AND: run a multi-ubatch prefill, not only the 32-token smoke.**  `window` is FATAL when the host expert
set can be evicted (2026-11-02, `WORKLOG.md` 261102.012): on a box where the set does not fit comfortably in
RAM, a `-ncmoe 44` run faults the GPU (`mul_mat_vec_q_moe` → `Page not present`) on the first completion,
and `llama-bench -p 2048` dies even with `GGML_MOE_DEVREMAP=0`; `-ncmoe 32` was clean there.  A short prompt
does not exercise it.  So for every arm you report, include
`llama-bench -p 2048 -n 128 -r 2 --host-experts window -lm mmap` at the running `-ncmoe`, and
`scripts/check-gpu-faults.sh` on the log.  If your box has RAM headroom this will be clean — say so
explicitly, because it is the difference between "works" and "works on a big-RAM box".

```bash
# arm A: gather on (the default)   arm B: gather off (staging, the pre-gather baseline)
for g in 1 0; do GGML_SCHED_DEVGATHER=$g ./build-rocm/bin/llama-bench -m $M -ngl 99 -ncmoe 40 -sm tensor \
    --host-experts window -p 2048 -n 128 -r 2 2>&1 | tail -3; done
```

Expected: A >= B for `window` (1 GPU measured 1038 vs 698).  A **far below** B is the forwarding failing —
report it with the log; it is a scheduler-side bug, not a data one.

Also worth one look: 2 devices at `-sm tensor` (the x8 pair, `HIP_VISIBLE_DEVICES=0,2`) at
`-ncmoe 48`, 2000+ tokens, since the pair is a different regime for the pool.

Record, for each arm: t/s (generation), acceptance if you enable `--spec-type draft-mtp`, the trace
line, and the peak RSS (`/usr/bin/time -v`).  The interesting comparison is `window` vs `pool`; both
should be stable, and `window` should not be *slower*.

## 4. The gate set (same commands as 1 GPU)

* **Text oracle**: the smoke above, 3 runs per arm; within an arm, identical.  Across `pool`/`window`,
  acceptance must match and text *may* differ on a MoE (do not gate on cross-mode text; `INVARIANTS.md`).
* **MTP four-axis** (`benchmarks/mtp-adaptive-methodology.md` Protocol A): `-n 3000`, `--reasoning on`
  for R and `off` for P/C/K, `-c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -lv 4`,
  `--spec-type draft-mtp --spec-draft-n-max 3`, prompts `prompts/{reasoning,prose-rdna-boosts,code-python,recall}.txt`.
  1-GPU reference (window): acceptance 0.81370 / 0.75408 / 0.89175 / 0.98148, mean len
  3.44 / 3.26 / 3.68 / 3.94, and `none` == `draft-mtp` text.  On 3 devices the acceptance numbers
  should match the multi-GPU baselines in that file, not these; what must hold is
  **`pool` vs `window` acceptance identical**.
* **Batched** (rule 5): `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8 -c 16384`.
* **Prefill logits** (the fill serves prefill staging):
  `scripts/gate-prefill-logits.sh` against a base recorded on your box with `--host-experts pool`;
  the candidate arm is `--host-experts window`.  If the script's flags do not reach `--host-experts`,
  reproduce its protocol by hand: `llama-perplexity -m $M -f /llm/models/wikitext-2-raw/wiki.test.raw
  -ngl 99 -ncmoe 40 --host-experts {pool,window} -c 512 --chunks 40 -fa on -ctk f16 -ctv f16 -t 8`
  with arm 1 `--kl-divergence-base <file>` and arm 2 `--kl-divergence-base <file> --kl-divergence`;
  pass = mean KLD <= 0.005 and same-top-p >= 98 %.
* **GPU faults**: `scripts/check-gpu-faults.sh <log>` (never grep a log for `page fault` by hand).
* **`-sm layer`** as well as `-sm tensor` for the smoke, one arm each.

## 5. What to report

1. `tensors=` / `reg_fail=` / `cold_dev=` from the trace on the `-sm tensor` arm (the headline).
2. t/s and peak RSS, `pool` vs `window`, per split mode and device count.
3. Acceptance per axis, `pool` vs `window` (must be identical).
4. Anything from §4 that fails, with the log.

Known and expected, so not a finding: `MOE_EXPERT_WINDOW_MIB`/`MOE_EXPERT_WINDOW` no longer exist;
`MOE_HOST_POOL_MIB` now only sizes a fallback ring in window mode (there is no pool tier);
`MOE_EXPERT_WINDOW_ALIAS_FILL` is now **default on** (it is the kill switch `=0` that is the exception)
and its visible effect is a **decode-and-prefill RSS** saving, not a text change.

Two more numbers from the 2026-11-02 revision, so you can tell "expected" from "broken" on your box:

* **Prefill** on the persistent-server cell is **at parity with `pinned`** through the arena-aware device
  gather (`window` 1037.9 against `pinned` 1036.3 t/s, `-ncmoe 48`, arena auto, request 3 of a 5246-token
  prompt), decode 60.1 vs 59.1, RSS 68.5 vs 68.6 GB.  With the gather killed (`GGML_SCHED_DEVGATHER=0`)
  `window` runs the staging path at ~698 against pinned's ~735.  Do **not** read a prefill number from a
  process-restarting `llama-cli`/`llama-bench` run: those mix page-cache fill into the cost and understate
  the gap.
* **The gather must be validated with its positive control, not just for speed.**  `GGML_MOE_GATHER_ONCE=1`
  restores the shipped-broken once-only guard (Hole B) and *must* reproduce the corruption — 1706 t/s and
  ~2000 chars of repeated-`/` garbage on this cell.  If it does **not**, your box is not exercising the
  multi-ubatch prefill and a clean run proves nothing.  The prefill-logit gate is blind to Hole B, so a KLD
  pass is not evidence of soundness.
* The `srv-test` **request-3 text hash is not reproducible** (two `pinned` runs gave `4c88b625a400` and
  `4e2fdac2bfab`), so do not use it as a purity gate: use the short-horizon oracle and acceptance.

Gates on this revision that are **already green** at 1 GPU (re-run them, they are cheap): greedy text
150 chars `sha=4ac3a523160f` in `pinned` / `pool` / `window` / `window + ALIAS_FILL=0`, 3 prompts × n96
byte-identical across the four arms, and the prefill-logit gate against a `pool` base (mean KLD
-0.000000, same-top-p 100.000 %).
