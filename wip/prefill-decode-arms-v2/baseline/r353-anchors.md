# STEP 0 baseline anchors — r353 delivery tree (`v17-919794500-r353`)

Recorded 2026-11-23 on the primary box, one GPU job at a time (the `baseline/` harnesses take an
exclusive `flock` so two arms cannot overlap).  These are the **anchors every later step must leave
unchanged when the pair is off** (`README.md` §"Definition of done", `PLAN.md` "Failure handling").

Everything below is *reproducible from the commands as written*; a hash without its command and prompt
is not an anchor.  The revision is the explicit `(rNN)` tag; `r353` is the delivery rung this tree is at.

## Environment of record

| what | value |
|---|---|
| delivery tree | `release.json`: base `919794500`, tree `27db9bfd4ef7f41c7df60f704410de28863beac3`, tip `089f90067` (build banner `b12615-089f90067`) |
| build | `~/bin/build-llama-rocm` → `~/llama.cpp/build-rocm` (canonical; warm incremental) |
| ROCm | **10.2.0** (`/opt/rocm-10.2.0-gfx120X`) — the line `~/bin/build-llama-rocm` configures.  (The prefill-logit base records it as `rocm-10.2.0-gfx120X`; a base recorded on another line is refused by the gate.) |
| devices | 3 × `gfx1201` (R9700) + the 9950X3D iGPU `gfx1036` that must stay unpinned-from → `HIP_VISIBLE_DEVICES` always set |
| llama.cpp branch | `arms-v2`, checked out at the applied `rdna-boosts` set (local only, never pushed) |

## A1 — 4B Q8_0, 1 GPU, greedy, `-n 64`

```sh
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 \
  -p "$(cat prompts/prose-rdna-boosts.txt)" -n 64 --seed 42 --temp 0 \
  --no-display-prompt --single-turn > A1.log 2>&1
scripts/extract-generated.py A1.log
```

| anchor | value |
|---|---|
| A1 text | **251 chars, sha=`be904e86f248`** |
| A1 rates (informational) | prompt 7107.9 t/s, generation 101.3 t/s |

## A2 — 27B Q8_0, 3 GPU `-sm tensor -mg 0`, greedy, `-n 96`

```sh
HIP_VISIBLE_DEVICES=0,1,2 ROCBLAS_USE_HIPBLASLT=0 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ngl 99 -sm tensor -mg 0 \
  -p "$(cat prompts/prose-rdna-boosts.txt)" -n 96 --seed 42 --temp 0 \
  --no-display-prompt --single-turn > A2.log 2>&1
```

| anchor | value |
|---|---|
| A2 text | **399 chars, sha=`10a82b5559a9`** |

## A3 — 35B-A3B Q8_0, host expert tier, 1 GPU, greedy, `-n 64`

```sh
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load \
  ~/llama.cpp/build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf -ngl 99 -ncmoe 40 \
  -p "$(cat prompts/prose-rdna-boosts.txt)" -n 64 --seed 42 --temp 0 \
  --no-display-prompt --single-turn > A3.log 2>&1
```

| anchor | value |
|---|---|
| A3 text | **262 chars, sha=`e9db5cb8db0b`** |
| A3 rates (informational) | prompt 1104.0 t/s, generation 52.3 t/s |

## A4 — prefill-logit KLD base (issue #113), default point

```sh
HIP_VISIBLE_DEVICES=0 scripts/gate-prefill-logits.sh --record     # then: scripts/gate-prefill-logits.sh
```

| anchor | value |
|---|---|
| base file | `~/.cache/rdna-boosts/gates/prefill-logit-gfx1201-Qwen3.8-27B-Q8_0-rocm-10.2.0-gfx120X.kld` |
| base build / PPL | `089f90067`, base PPL **5.9325** (matches the r353 release record) |
| base sha256 | `ce8bb26fceb4eacf24fb34091cf9a408f4fe578d224be7041b3a1294802b0c91` |
| self-check | mean KLD **0.000000**, same-top-p **100.000 %**, GPU journal clean |

## A5 — MTP four-axis protocol (text + acceptance)

Shape: **3 devices `-sm tensor -mg 0`**, 27B Q8_0 (MTP head embedded in the target GGUF, no `-md`),
`-n 3000`, greedy (`--seed 42 --temp 0`), `--reasoning on` for **R** and off for K/P/C, f16 KV,
`ROCBLAS_USE_HIPBLASLT=0` pinned (this is a *fresh-process text* anchor, `INVARIANTS.md` §44), drafter
`--spec-type draft-mtp-adaptive --spec-draft-n-max 7` (the delivery's adaptive drafter at the ceiling
its own four-axis records use).  Run with `baseline/mtp-axis.sh --axis <A> --spec none|adaptive`.

| axis | prompt | `none` t/s | adaptive t/s | speedup | acc drafts/gen drafts | acc tok/gen tok | mean acc len | acc pos 1 | text |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| R reasoning | `reasoning.txt` | 41.7 | 79.0 | 1.89× | 0.72456 | 0.51987 | 2.63 | 0.725 | `f27add1937e2` == |
| K recall | `recall.txt` | 41.8 | 151.4 | 3.62× | 0.98649 | 0.97241 | 6.72 | 0.986 | `a87c4318b649` == |
| P prose | `prose-rdna-boosts.txt` | 41.4 | 111.2 | 2.69× | 0.91558 | 0.66833 | 4.48 | 0.916 | `c494eb660f49` == |
| C code | `code-python.txt` | 41.6 | 128.3 | 3.08× | 0.93692 | 0.69121 | 5.56 | 0.937 | `5f6c662c4e0f` == |

Verdicts at STEP 0: gate rule 1 (acceptance ≥ ~0.45 pos-1) **PASS** on all four axes; rule 2 (MTP must
not lose to plain) **PASS** (1.89×-3.62×); rule 4 (purity `n_max ≤ 7`) **PASS** — `none` and the
adaptive drafter read **byte-identical text on all four axes**.

Cross-check against the delivery's own record (`benchmarks/2026-11-21-r328-mtp-4axis.md`): the **K
axis hash is identical** (`a87c4318b649`) and the `none` rates agree to ~2 % on all four axes, so this
command reproduces the delivery baseline's *plain* arms.  The **adaptive trajectory does not** (their R
mean-acc-len 3.40 vs 2.63 here): that record's controller settings differ from `n_min_adaptive`'s
default, so its `draft-mtp` rates/acceptance are **not** comparable with this table and must not be
quoted against it.  Nothing in STEP 2-5 depends on that — this table is self-consistent, and the
property those steps are judged on (`none` == drafter, byte for byte) holds on all four axes.

## A6 — single-context server, 2 slots `--kv-unified`, two concurrent greedy completions

```sh
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 \
  wip/prefill-decode-arms-v2/baseline/server-arm.sh --out DIR --tag A6-server-2slot \
  --model /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf --np 2 --ctx 8192 --concurrent
```

| anchor | value |
|---|---|
| responses | `{142 chars sha=7bd542418112, 207 chars sha=6ce1f88dd5f7}` — compared as a **set** (the two slots and their prompts are assigned by LRU, so the pairing is not part of the anchor) |
| startup shape | 11 lines, see below and `baseline/A6-startup-shape.txt` |

The server's startup shape (normalised by `baseline/norm-log.sh --drop-prefix 'slot (launch_slot_|print_timing|release|select)'`):

```
I srv llama_server: initializing ...
I cmn common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
I srv init_listene: The UI is disabled
I srv init_listene: Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable
W srv server_setup: security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)
I srv load_model: loading model '/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf'
I cmn init: llama threadpool init, n_threads = 16
I srv load_model: initializing, n_slots = 2, n_ctx_slot = 8192, kv_unified = 'true'
I srv load_model: prefill admission: fair share = on, chunk = n_ubatch, skew = 1, duty = 16, no-decode = n_batch
I srv server_run_i: model loaded
I srv server_on_li: listening on http://N:PORT
```

**This 13-line startup shape is the STEP 2/3/5/6 inertness assertion** for the default path: each of those
steps must print exactly these lines (a new INFO line about the arms/memory is expected only when the pair is
ON; the borrow/allocator/recurrent/Meta steps are inert with it OFF).

*Widened at STEP 6 (2026-11-23):* `norm-log.sh` used to keep only `<level> <component> <function>:` lines and
silently DROPPED the library's own single-tag lines (`llama_context:`, `llama_kv_cache:`, `resolve_fused_ops:`,
`report_locked:`) -- so a changed cache geometry would not have shown in this anchor at all.  It now keeps both
forms (addresses masked to `PTR`), and the shape is cut at the `listening on` line (the teardown census lines
are per-run).  The re-baseline was taken from the **pair-off** arm of a build whose text (`be904e86f248` /
`10a82b5559a9` / `7bd542418112`+`6ce1f88dd5f7`) and KLD (0.000000 / 100.000 %) still match STEP 0, so nothing
was waved through: the two new lines are `resolve_fused_ops` WARNs every run had always printed.
With the pair ON, the delta is exactly: those two WARNs again (the second context resolves fused ops itself)
plus the pair's own `two-arm server: ... ONE memory ...` INFO line.

## A7 — the delivery's cheap ladder arm

```sh
scripts/validate-set.sh
```

| anchor | value |
|---|---|
| verdict | **OK** — base `919794500`, 17 official + 1 temporary = 18 default blocks, 2 optional; applied tree `27db9bfd4ef7f41c7df60f704410de28863beac3`; checksums, reconciliation (`20 blocks`, 0 unexplained), system-header and Windows-portability scans all clean |

## Not recorded at STEP 0 (deliberately)

* **The q8_0 KV half of A5.**  The delivery's r328 record runs both cache types; the anchors above run
  f16 only.  STEP 5's oracle needs the four-axis protocol on the **recurrent** models (35B-A3B, qwen4exp)
  anyway, and those arms are recorded there — running a second 27B cache type here would spend ~7
  minutes of the box on an anchor no step consumes.
* **The shape matrix of STEP 9** (phi-4, Flash-Next, Gemma4-iswa, ITL, memory).  It is the *feature's*
  record, not an inertness anchor.
