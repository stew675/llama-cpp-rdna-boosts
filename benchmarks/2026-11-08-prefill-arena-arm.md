# A prefill-only workload never arms the expert arena — the `-ncmoe` + `llama-bench` regression (`TODO` #305)

**Date:** 2026-11-08 · **Box:** `soar`, 1 × / 2 × R9700 (gfx1201, 32 GiB each, three PCIe 5.0 **x16** links),
Ryzen 9950X3D, ROCm 10.1.0, VMM build (`~/bin/build-llama-rocm`)
**Models:** Qwen3.6-35B-A3B **Q8_0** (48 × 256 experts, 32 640 MiB host set at `-ncmoe 48`);
Qwen3.8-Flash-Next **GSQ-RCO-IQ3_S** (qwen4exp, the field config)
**Builds:** `build-rocm` = the candidate (`0001-prefill-arena-arm.patch` on r236 tip `0e6daae4e`); `build-rocm` before
the patch and `build-rocm-r235` (the shipped pre-rebase release, same host path) = the baseline; `build-rocm-base` =
unpatched `a691b2345`
**Caveat observed on this box:** the host-expert tier's locked memory is released a few seconds **after** the process
exits, so every arm below is separated by `sleep 10`; back-to-back arms contaminate each other.

## 1. The symptom, and the census that names it

`llama-bench -m <35B Q8_0> -ngl 99 -ncmoe 48 -fa 1 -t 8 -ub 512 -p 512 -n 0 -r 2`, 1 GPU, pinned masters:

| build | pp512 |
|---|---:|
| shipped r236 (pre-patch, measured directly) | 438.5 ± 0.2 |
| shipped r235 binary | 438.3 |
| unpatched `a691b2345` | 516.4 / 528.9 / 531.9 / 532.9 / 529.0 (5 runs) |
| **candidate** | **1183-1204** (1184.4, 1190.6, 1195.2, 1197.6, 1198.8, 1201.8, 1204.0) |

`-v` on the failing arm:

```
report_locked: prefill cache staging: ok=0 geom_decline=0 unusable=0 nomatch=234 unsized=0 (the priming pass); arena=0.0 MiB host=0.0 MiB
report_locked: MoE expert cache: h=0.0000 (0/0 reaches), fills=0 evictions=0, tables=0, slots=0, arena=0.0 MiB total (budget auto MiB/device)
moe_cache_forget_all: model teardown: dropped 3 expert-cache table(s) (0.0 MiB of arena)
```

`tables=0`: nothing is registered.  Arithmetic check of what that costs: `32640 MiB / 28.7 GB/s` (the box's measured
`ggml_backend_cuda_stage_h2d_gbps` calibration) = **1.11 s per ubatch**; the arm reads 512/438.5 = **1.17 s per ubatch**.

Proof that the arena is the difference (same flags, one 8-token row first so the deferred sizing latches):
`-p 8,512` → pp512 = **1116-1162 t/s**.

## 2. Arming the arena — the candidate

`wip/prefill-arena-arm/0001-prefill-arena-arm.patch` (7 files, +148/−1, sha256 `75b0fcfb…`): two new
`ggml_backend_i` queries — `moe_cache_pending` (enabled + host experts expected on this device + not armed) and
`moe_cache_stage_ready` (sized + THIS table has slots) — used by `sched_stage_issue`, plus `moe_cache_update`'s
priming-time sizing no longer requiring `n_tok <= 8`.

### 2a. The reported arm and the ub sweep (`-p 4096`)

| arm | shipped | candidate | Δ |
|---|---:|---:|---:|
| `-p 512 -ub 512` | 438.5 | 1198.8 | **+173 %** |
| `-p 4096 -ub 512` | 439.4 | 1191.0 | **+171 %** |
| `-p 4096 -ub 1024` | 876.5 | 1857.3 | **+112 %** |
| `-p 4096 -ub 2048` | 1741.8 | 2655.5 | **+52 %** |
| `-p 4096 -ub 4096` | 1741.9 | 2648.4 | **+52 %** |

(`-b` clamps at 2048 on this shape, so the last two rows are one ubatch size.  The "3446 t/s at `-ub 4096`" that
`TODO` #273 quotes is **not** this invocation — the shipped binary reads 1741 there, verified today.)

### 2b. `-ncmoe` sweep (pp512) and device count

| arm | shipped | upstream | candidate |
|---|---:|---:|---:|
| `-ncmoe 12` | — | 1185.4 | 1186.5 |
| `-ncmoe 24` | — | 755.6 | 1143.2 |
| `-ncmoe 48` | — | 536.0 | 1192.2 |
| 2 GPU `-sm layer` (`-ncmoe 48`) | 453.8 | — | 1344-1350 |
| 2 GPU `-sm tensor` (`-ncmoe 48`) | 798.6 | — | 798.5 (deliberately unchanged) |

### 2c. The field model (`--lazy-mode mlock`, as required for this model)

Qwen3.8-Flash-Next GSQ-RCO-IQ3_S, `-ncmoe 48`, 1 GPU, `-ub 1024`, `-p 512 -n 64 -r 2 -lzm mlock`:

| build | pp512 | tg64 |
|---|---:|---:|
| shipped r235 | 295.8 ± 0.5 | 45.3 ± 8.9 |
| candidate | **387.8 ± 1.4** (+31 %) | 44.6 ± 7.0 (−1.6 %, inside the arm's own spread) |

## 3. Neutrality (the general case is a server or a CLI, not a benchmark)

| arm | shipped | candidate |
|---|---:|---:|
| `tg128` decode (`-ncmoe 48`, 1 GPU, `-r 3`) | 82.98 ± 3.56 | 82.94 ± 3.82 |
| real-text prefill, `--host-experts auto` (5256-token prose, `-ub 512`) | 1423.3 | 1428.4 |
| real-text prefill, `--host-experts pinned` | 999.0 | 998.7 |
| real-text prefill, `--host-experts pool` | 1426.9 | 1426.5 |
| 35B-A3B greedy text (prose, `-n 32 --seed 42 --temp 0`), `auto` and `pinned` | `139 chars e2dde701fc04` | **identical** |
| 4B Q8_0 coherence anchor (AGENTS' arm) | `1c5d32ac537d` | `1c5d32ac537d` |
| cache-off contract (`MOE_EXPERT_CACHE_MIB=0`, `-p 4096 -ub 4096`) | 1741 (r235) | 1739 |
| cache-off contract (`MOE_EXPERT_CACHE_MIB=0`, pp512, `-r 3`) | 438.5 (r235) | 437.5 |
| dense 27B Q8_0 pp512 (interleaved `-r 5`, 3 rounds) | 1543 (r235) | 1524 (**−1.2 %** = r236's documented inherited dense cost, `TODO` #303; the candidate's two queries are inert on a dense model — the preflight only marks a device when host expert bytes > 0) |

## 4. Notes that cost measurements (do not repeat them)

1. **The `-sm tensor` Meta forward is unsound.**  `ggml_backend_meta_moe_cache_update` declines every prefill width
   (`n_tok > band`), so a forwarded `pending` query never clears and the ring stays disabled for the whole run:
   2 GPU `-sm tensor` pp512 read **356** against 799.  The patch therefore does not touch the Meta backend; a
   tensor-split pp-only run keeps its r236 behaviour (799).
2. **A residency threshold is the wrong gate.**  "Stage only at ≥ 75 % residency" was tried: a 73 %-resident arena
   still wins **2×** at `-ub 512` (1195 staged vs 585 bulk), so the threshold cost the whole win.  Reverted.
3. `-p 8` alone is a *cold* arm (215 t/s at pp8) — it is the sizing trigger, not a measurement.
4. The candidate's own kill switches restore r236 exactly: `MOE_EXPERT_CACHE_MIB=0` (both queries become no-ops) and
   `GGML_SCHED_STAGE=0` (no ring).  No new environment variable is introduced.

## 5. r237: the registration pre-pass — the tensor-split half, and the same fix on one GPU becomes unconditional

The r237 candidate moves table discovery off the routing path entirely (`moe_cache_register`, called for every
split's routed host-expert inputs at the top of `ggml_backend_sched_compute_splits` while `moe_cache_pending`) and
arms the arena at registration.  Consequences measured on the box (all 30 s apart unless stated):

| arm (35B-A3B Q8_0, `-ncmoe 48`, `-ub 512`, `-p 512 -n 0`) | shipped r236 | r237 |
|---|---:|---:|
| 1 GPU, `-r 3` / `-r 5` | 438.5 | **1209 / 1198** |
| 1 GPU, `-r 1` (first rep, so no warm pass at all) | 438 | **1192** |
| 2 devices `-sm layer` | 454 | **1438** |
| 2 devices `-sm tensor` (4 runs, 30 s apart) | 798.6 | **2458 / 2455 / 2408 / 2337** |
| `MOE_EXPERT_CACHE_MIB=0` (cache-off contract), 2 interleaved rounds | 437.1 / 437.2 | 437.6 / 437.3 |
| `tg128` decode | 82.98 | 83.58 |
| real-text prefill, `auto` / `pinned` / `pool` | 1423 / 999 / 1427 | 1414 / 1046 / 1420 |
| 35B greedy text (`e2dde701fc04`), 4B anchor (`1c5d32ac537d`) | — | identical |

Flash-Next GSQ-RCO-IQ3_S, 2 devices `-sm tensor -ncmoe 48`, `-p 512 -r 3`, per-rep values:

| `--lazy-mode` | reps (t/s) | note |
|---|---|---|
| `mlock` (r235/r236, arena never armed) | 494.8 / 494.5 / 495.7 | r235, i.e. the whole half-host-set per ubatch |
| `mlock` (r237, arena armed) | 1267 / 1275 / 1278 | **flaps** under memory pressure: one run read 171.8 / **13.1** / 371.3, another 153 / 1264 / 1278 |
| `mlockr` (new mode: populate, then munlock) | 1245 / 1258 / 1264 and 1251 / 1275 / 1280 | stable; the same 1-device real-text prefill is **747-781 t/s** against `mlock`'s 492-505 |
| `off` | 1263 / 1286 / 1292 | stable, 3.5x slower on the 1-device real-text prefill (216 t/s) |
| `auto` (the field value) | 1155-1172 | stable |

**The `mlock` flap is memory overcommit, not the mlock release latency.**  Sampled during a 2-device run:
`MemAvailable` bottomed at **1.0 GiB**, `VmRSS` peaked at **76.1 GiB**, `VmLck` 26.8 GiB, on a 91 GiB box — an
unevictable PLE table + a ~43.7 GiB auto arena leaves the 47 GB model mapping as the only reclaimable memory, so
the 8.8 % non-resident staging bytes intermittently come from disk.  It reproduces with 30 s and with 60 s + `sync`
between runs, with `Dirty` = `Writeback` = 0 and `MemAvailable` ~78 GiB before each run (the mlock accounting
clears within a second of process exit), which rules out the "10-15 s to release mlocked pages" theory for *this*
symptom.  `mlockr` avoids it by construction: the pages stay hot but are reclaimable.
