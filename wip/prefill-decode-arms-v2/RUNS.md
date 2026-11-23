# RUNS — dated measurement record for `prefill-decode-arms` v2

One section per landing, newest first.  A line here is the **record of a claim**: the command, the
shape, the hash/rate, and (for a comparison) the arm it is compared against.  A gate's own output is
not the record — it is the instrument.

Anchors (the inertness reference set) live in [`baseline/r353-anchors.md`](baseline/r353-anchors.md).

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
