# RUNS — dated measurement record for `prefill-decode-arms` v2

One section per landing, newest first.  A line here is the **record of a claim**: the command, the
shape, the hash/rate, and (for a comparison) the arm it is compared against.  A gate's own output is
not the record — it is the instrument.

Anchors (the inertness reference set) live in [`baseline/r353-anchors.md`](baseline/r353-anchors.md).

---

## 2026-11-23 — STEP 4: the Meta view slot is a per-context identity (`WORKLOG.md` 261123.010)

3 files, +74/-7 (`ggml/src/ggml-backend-meta.cpp`, `src/llama-context.{h,cpp}`).

| arm | shape | result |
|---|---|---|
| `gates/probe-meta-slots.cpp` (new) | 27B Q8_0, 3 devices `-sm tensor` (the Meta backend is really built), 3 contexts (`owner`, `borrower`, a third live context) | **PASS 11/11**: bases **0 / 2 / 4**, distinct, even, pairs inside the backend's 8; identical token (11751) with 2 and with 3 contexts live |
| `GGML_META_DEBUG_SET=1` | same model/shape, `-n 16` | **0 `MISSING child`** lines; text `26b1dc6dd4ad` |
| A1 / A2 | 4B 1 GPU / 27B 3 GPU tensor | `be904e86f248` / `10a82b5559a9` |
| A6 + startup shape | 2 slots `--kv-unified`, 2 concurrent | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}`, shape `diff` empty |
| A4 | prefill-logit | 0.000000 / 100.000 % |

The per-context base line is DEBUG by design (an INFO line per context would add a line to every
single-context run's startup log, breaking the STEP 0 anchor for a diagnostic only a multi-context run
reads); the census rides on the STEP 2 borrow line: `Meta view slots 2/3 and 0/1`.

---

## 2026-11-23 — STEP 3: one allocator, one lock (`WORKLOG.md` 261123.009)

13 files, +314/-19 (`src/llama-memory.h`, `src/llama-kv-cache.{h,cpp}`, the 8 composite memory headers,
`src/llama-memory.cpp`, `src/llama-context.cpp`).

| arm | shape | result |
|---|---|---|
| `gates/probe-kv-reservation.cpp` | phi-4 IQ4_XS, CPU only (`n_gpu_layers=0`), **unified** cache so two sequences share one cell array, `--ntok 512` | **PASS**: unborrowed A/B `[0..511]` each (bitmap off by design); borrowed A `[0..511]`, B `[512..1023]`, C `[1024..1535]`; a dropped batch context releases (later prepare starts at 0); teardown census `0 of 8192` |
| … the STEP 2 build (control) | same probe, same shape | **FAIL at the I4 line**: `prepare A: [0..511]`, `prepare B: [0..511]`.  The other 10 checks pass, so the probe isolates the mechanism |
| `gates/probe-n1-threads.cpp` | 4B Q8_0; owner `n_batch 2048 / n_ubatch 512` (`n_ctx 8192`, `n_seq_max 4`, unified ⇒ 2048 cells), borrower `n_batch 8`; **two threads**: prefill 1900 tokens on seq 0 × 3 rounds while seq 1 decodes 32 single-token steps on the borrower | **PASS** and `DIGEST prefill=64a94dac76706dfd decode=897f0e3809ab747f` **identical to `--solo`**; wall 982 ms vs 1140 ms solo; teardown census `0 of 2048 reserved`.  Declared as the I3/no-corruption oracle: the pre-STEP-3 build reads the same digests |
| A1 / A2 / A3 | 4B 1 GPU / 27B 3 GPU tensor / 35B-A3B host tier | `be904e86f248` / `10a82b5559a9` / `e9db5cb8db0b` — unchanged |
| A6 | 2 slots `--kv-unified`, 2 concurrent greedy | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}`; **startup shape byte-identical** (`diff` empty) |
| A4 | `gate-prefill-logits.sh` | **0.000000 / 100.000 %** |
| `gate-state-pipe.sh` | ROCm 10.2, pinned | **PASS** — every arm byte-identical to the synchronous reference; pipe 18.7 ms vs sync 41.4 ms (0.45×) |
| `gate-park-idle-slots.sh` | Flash-Next host-tier IQ4_NL, `-ncmoe 48` | **PASS with `ROCBLAS_USE_HIPBLASLT=0`** (2 runs).  Failure without it, and the attribution grid, is in the WORKLOG entry §4 |

### The one thing this step's own record must not overstate

The reservation is proven by a **deterministic set-algebra probe**, not by the soak: the two-thread arm reads
the same digests on the STEP 2 build too.  That is not a weakness of the step, it is the reason the probe was
written -- a collision needs two batches that are prepared and not yet applied, which the public API does not
expose, so a timing-based arm would be luck.  The soak still earns its place as the I3 oracle (no deadlock, no
corruption, census clean) and it is the shape STEP 6's server will use.

---

## 2026-11-23 — STEP 2: the borrow (`llama_context_params::ctx_memory`) (`WORKLOG.md` 261123.008)

Files: `include/llama.h`, `src/llama-context.h`, `src/llama-context.cpp` (79 insertions, 6 deletions;
`src/llama-cparams.h` needs no field — the borrow is resolved in the constructor).

| arm | shape | result |
|---|---|---|
| `gates/probe-ctx-memory.cpp` | 4B Q8_0 (owner/borrower) + phi-4 IQ4_XS (the different-model refusal), 1 GPU | **PASS 11/11**: identity `0x55fe1f3ea260` both; negative control (an unborrowed context's memory differs); `t1=11751 t2=13` from the owner-only arm **and** from the arm whose last two rows were decoded on the borrower; both refusals; owner still decodes after the borrower is freed |
| A1 / A2 | 4B 1 GPU `-n 64` / 27B 3 GPU `-sm tensor -mg 0` `-n 96` | `be904e86f248` / `10a82b5559a9` — unchanged |
| A6 | 2 slots `--kv-unified`, 2 concurrent greedy | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}` — unchanged |
| A6 startup shape | `baseline/norm-log.sh … --drop-prefix 'slot (launch_slot_\|print_timing\|release\|select)'` vs `baseline/A6-startup-shape.txt` | **identical** (`diff` empty, 11 lines) |
| A4 | `gate-prefill-logits.sh` (no `--record`) | mean KLD **0.000000**, same-top-p **100.000 %**, journal clean |
| `gate-draft-ctx.sh --quick` | external MTP head (Flash-Next GSQ-RCO-IQ3_S + shared Q8_0 head), 1 device | text arms **5/5 PASS** — plain and 2 `draft-mtp` runs all `e7aae6034b4f`, acceptance 0.81429, no same-model refusal.  **census arm FAIL by construction**: it greps for `is a context of ANOTHER model (a draft head)`, which exists only in `beta/prefill-decode-arms/prefill-decode-arms.patch` — the r352 withdrawal deleted the generic `ctx_other` arm it belonged to (`AGENTS.md`: this gate is **beta-only since r352**).  Verified not mine: `grep -rn "ANOTHER model" src/*.cpp` finds nothing |
| build | `~/bin/build-llama-rocm` | clean |

### The two probe failures found on the way (both were MINE, and both are now facts)

1. **The first version of check 3 compared the wrong two tokens** (the argmax over the prompt's last row
   against the argmax after decoding one more token) — a probe bug, not a code bug; it now runs two
   independent (owner, borrower) pairs through the *same* three steps and compares like with like.
2. **The `ctx_other` guard did not fire.**  Written against `cparams.ctx_other`, which the constructor filters
   to `nullptr` for every architecture that does not need one — so it was dead on qwen35 (the probe's model)
   and alive only where `ctx_other` is meaningful.  Now on `params.ctx_other`.  `INVARIANTS.md` r355-wip item 3.

---

## 2026-11-23 — STEP 1: the three backend prerequisites (`WORKLOG.md` 261123.007)

**Tree:** same r353 tree, `~/llama.cpp` branch `arms-v2`.  Files: `ggml/src/ggml-cuda/ggml-cuda.cu`,
`batch-copy.cu`, `common.cuh`, `ggml-cuda-slab.cu`, `ggml-cuda-vmm.h`.  No delivery file touched.

| arm | command / shape | result |
|---|---|---|
| shlib symbols, default (`GGML_HIP_NO_VMM=OFF`) | `scripts/check-shlib-symbols.sh` | **OK**, 16/16 libraries resolve every symbol |
| shlib symbols, `GGML_HIP_NO_VMM=ON` | `BUILD_DIR=build-rocm-novmm EXTRA_CMAKE_FLAGS="-DGGML_HIP_NO_VMM=ON" ~/bin/build-llama-rocm` then `check-shlib-symbols.sh ~/llama.cpp/build-rocm-novmm` | **OK**, 16/16.  The `#else` definition is the one compiled: `strings … \| grep -c "peer access to a"` is **1** in the VMM build and **0** in the NO_VMM build |
| host-stream probe (non-vacuity for fix 1) | `gates/probe-host-stream.cpp` linked against `build-rocm/bin/libggml-hip.so` | **PASS**: devices 0/1/2 → `0x558a914d5c50` / `0x558a91dee140` / `0x558a91eb9c70`, repeat calls identical, `hipStreamPerThread` = `0x2`, legacy default = `(nil)`; a second **thread** gets `0x7f4b68000d90`, distinct |
| A1 | 4B Q8_0, 1 GPU, prose, `-n 64`, 2 runs × 2 builds | `be904e86f248` ×4 (anchor) |
| A2 | 27B Q8_0, 3 GPU `-sm tensor -mg 0`, `-n 96` | `10a82b5559a9` (anchor) |
| A2 + `GGML_CUDA_P2P=1` | exercises the new split grant (P2P on ⇒ `use_peer_access`) | `10a82b5559a9`, **no** `peer access to a …` warning (all peers granted), kernel journal + log clean |
| A3 | 35B-A3B Q8_0, `-ncmoe 40` host tier, `MOE_EXPERT_HEATMAP=load` | `e9db5cb8db0b` (anchor) |
| A4 | prefill-logit, base recorded from the **pre-change** build (`gates-baseline/`), candidate = post-change build | mean KLD **0.000000**, same-top-p **100.000 %**; the two base files are **byte-identical** (`sha256 ce8bb26f…` both) |
| GPU faults | `check-gpu-faults.sh --since` over the 3-GPU and host-tier logs | clean |
| copy/tensor plumbing | `test-backend-ops -o CPY` on 3 ROCm backends; `test-backend-tensor` | 286/286 + 286/286 passed; all backends passed |

### Non-vacuity of the pool fix, and its one uncovered branch

`GGML_CUDA_P2P=1` takes the `use_peer_access` path, so the **split grant ran** (owner located, per-peer
`cuMemSetAccess`, mask registered) and every peer was granted — the copy behaviour is therefore unchanged,
which is what the identical text shows.  The **refusal** branch needs a peer grant the driver refuses, i.e. an
arena that has taken the VRAM *and* a second context whose pool grows at request time.  This box cannot
produce that without the pair: the branch stays *exercised but not fired* until STEP 6/8's pair gate on a full
host-expert arena.  Recorded so it is not mistaken for a validated path.

### Observation: the pre-change A4 base disappeared from `~/.cache/rdna-boosts/gates/`

The STEP 0 base (5.1 GiB, `sha256 ce8bb26f…`) was gone when A4 was re-run after the STEP 1 build: the
directory existed and was empty, and `~/.cache/rdna-boosts` + `gates/` both carry `ctime == mtime ==
10:20:39`, while `arms-v2/` and `gpu.lock` in the same parent were untouched.  Nothing in the delivery writes
that path (`grep -rn rdna-boosts` over `src/ ggml/src/ common/ tools/` finds no writer), no `tmpfiles.d` rule
names it, and the system journal has no entries for the window.  Cause unknown; the consequence was ~2
minutes of re-recording, and the workaround is the one used here: record the cross-build reference into a
**separate** directory (`gates-baseline/`), which is also the right protocol for a change whose oracle is a
KLD point.  A recurrence would be worth chasing; a one-off is not.

---

## 2026-11-23 — STEP 0: workspace, branch, baseline anchors (r353 tree)

**Tree:** delivery `v17-919794500-r353` (base `919794500`, tree `27db9bfd4`, tip `089f90067`, build
banner `b12615-089f90067`), ROCm **10.2.0** (`/opt/rocm-10.2.0-gfx120X`), 3 × gfx1201 R9700.
**Branches:** delivery `prefill-decode-arms-v2` (off `main`); `~/llama.cpp` `arms-v2` (local, never
pushed).

### The seven anchors

| # | shape | result |
|---|---|---|
| A1 | 4B Q8_0, 1 GPU, prose prompt, `-n 64`, greedy seed 42 | **251 chars `be904e86f248`** (prompt 7107.9 t/s, gen 101.3 t/s) |
| A2 | 27B Q8_0, 3 GPU `-sm tensor -mg 0`, `-n 96` | **399 chars `10a82b5559a9`** |
| A3 | 35B-A3B Q8_0, `-ncmoe 40` host tier, 1 GPU, `-n 64` | **262 chars `e9db5cb8db0b`** (prompt 1104.0 t/s, gen 52.3 t/s) |
| A4 | prefill-logit base, default point (27B Q8_0, `--device 0`, f16 KV, `-ub` default, 40×512) | base PPL **5.9325**, base sha256 `ce8bb26f…`, self-check **KLD 0.000000 / 100.000 %** |
| A5 | MTP four-axis, 27B Q8_0 3 GPU `-sm tensor -mg 0`, `-n 3000`, f16 KV, adaptive ceiling 7 | see the A5 table below |
| A6 | single-context server, 2 slots `--kv-unified`, 2 concurrent greedy completions | `{142 chars 7bd542418112, 207 chars 6ce1f88dd5f7}` + an 11-line startup shape |
| A7 | `scripts/validate-set.sh` | **OK** (18 default blocks, 2 optional, tree `27db9bfd4`, reconciliation 0 unexplained) |

### A5 — the four-axis table (this build's own baseline)

`baseline/mtp-axis.sh --axis <R|K|P|C> --spec none|adaptive`, one arm at a time, f16 KV,
`ROCBLAS_USE_HIPBLASLT=0`, `-n 3000`.

| axis | none t/s | adaptive t/s | speedup | acc drafts/gen | acc tok/gen | mean len | pos 1 | text (none == adaptive) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| R | 41.7 | 79.0 | 1.89× | 0.72456 | 0.51987 | 2.63 | 0.725 | `f27add1937e2` |
| K | 41.8 | 151.4 | 3.62× | 0.98649 | 0.97241 | 6.72 | 0.986 | `a87c4318b649` |
| P | 41.4 | 111.2 | 2.69× | 0.91558 | 0.66833 | 4.48 | 0.916 | `c494eb660f49` |
| C | 41.6 | 128.3 | 3.08× | 0.93692 | 0.69121 | 5.56 | 0.937 | `5f6c662c4e0f` |

Purity (`none` == drafter, byte for byte) holds on **4/4 axes**; pos-1 acceptance is 0.725-0.986 against
the ~0.45 floor; the drafter wins on every axis.  The **K hash matches the delivery's r328 record**
(`a87c4318b649`) and every `none` rate agrees with it to ~2 %, so this command reproduces the delivery
baseline's plain arms.  Its adaptive trajectory does **not** reproduce that record's (R mean-acc-len
2.63 here vs 3.40 there: different `n_min_adaptive`/`n_max` trajectory), so r328's `draft-mtp`
rates must not be quoted against this table, and vice versa.

### Harness defects found and fixed at STEP 0 (recorded because they will bite again)

1. **A bare `wait` in the server driver also waited on the server** (started as a background job in the
   same shell), so the arm "completed" only when the server's own `--timeout` expired — a 900 s
   unguarded wait behind a process that was already healthy and idle.  Fixed: background request jobs
   are waited on **by PID**, and every wait is bounded (readiness 60 s, per request `--max-time`, a hard
   server `--timeout`); a server that dies fails the arm immediately with its log tail.
2. **Two arms launched with `&` contaminated a text anchor**: the R axis's first `none`/`draft-mtp`
   pair ran concurrently and read *different* texts at the same seed.  Re-run sequentially they are
   identical.  Fixed (and enforced): `baseline/mtp-axis.sh` takes an exclusive `flock` on
   `~/.cache/rdna-boosts/gpu.lock` and refuses after a bounded wait, so "one GPU job at a time" is a
   mechanism, not a habit.

### Non-anchor observations

* `~/bin/build-llama-rocm` configures **ROCm 10.2.0** (`/opt/rocm-10.2.0-gfx120X`), not 10.1 as
  `AGENTS.md`'s "Build the fork" section states; the prefill-logit gate agrees (its base file carries
  `rocm-10.2.0-gfx120X`).  Reported to the maintainer rather than edited, since `AGENTS.md` is policy.
* The A6 server log's shutdown tail is `Received second interrupt, terminating immediately` plus the
  block-09/15 sink census lines: the harness kills the process group, so `baseline/norm-log.sh` cuts the
  shape at the first termination line.
