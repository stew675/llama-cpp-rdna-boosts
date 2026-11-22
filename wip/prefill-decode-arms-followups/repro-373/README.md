# repro-373 -- the crash harness for `TODO` #373 / campaign §W9 (the two-arm server's P0)

**NOT part of the delivery.** A load driver and a server runner for the shape that reproduces the pair's crash on demand.
Recorded 2026-11-22 (`WORKLOG.md` 261122.032); the mechanism, the numbers and the two fixes are there.

## Why it exists

Nothing in `scripts/` drove **three slots with two prefills in flight and one decoding slot**, which is why this reached a
user (`MANIFESTS.md` owes the gate arm; this is its skeleton). The crash reproduces in ~30 s on this shape, first try.

```
./run.sh <logfile>                          # the maintainer's runme, driven with 3 concurrent requests
EXTRA_ARGS="--no-prefill-decode-arms" ./run.sh <logfile>   # the pair OFF (the control: clean)
```

`run.sh` starts `/llm/models/Qwen3.8/Flash-Next/STEW675-IQ4_NL/runme` under `setsid` (process-group kill, refusal to start
while any `llama-server` exists -- the usual landmines), waits for `/health`, runs `armload.py`, and stops the group.
Everything the server prints (including any temporary `[DBG_*]` instrumentation) lands in the log file.

## The workload (`armload.py`, env-tunable)

| slot | request | env |
|---|---|---|
| 0 | a short prompt + a **long generation** -- keeps the decode arm applying for the whole prefill window | `HEAD_TOK` (2000), `GEN_TOK` (800) |
| 1 | the long prompt | `LONG_TOK` (71000) |
| 2 | the medium prompt, launched `MED_DELAY` s later (as in the crash log) | `MED_TOK` (23000), `MED_DELAY` (13) |

`ROUNDS` repeats the whole set (a fresh set of requests, so the request boundary and the prefills recur). Prompts are
sliced from `wikitext-2-raw/wiki.train.raw` and sized with the server's own `/tokenize`, so the token counts are the real
ones. Reproduced with `LONG_TOK=20000 MED_TOK=6000 GEN_TOK=200` in ~30 s and with the defaults above in 40-55 s.

## What it does NOT do (yet)

No assertions and no pass/fail: it drives and it logs. The owed gate (`MANIFESTS.md`, campaign §W9) adds: repeat N >= 3,
require every request to complete, shrink the pad buckets (`LLAMA_KV_N_PAD_MIN=256 LLAMA_KV_N_PAD_PREFILL=0`) so a
mismatch cannot hide in rounding, compare a fourth sequential request's text against the single-context server's (a
class-matched oracle, `GREEDY-PURITY.md` §105), and run `scripts/check-gpu-faults.sh` over the window.

## Status at the time of writing

The KV-cell-map defects (the qwen4exp assert and a libstdc++ RB-tree SIGSEGV in `llama_kv_cache::prepare`) are fixed by
the memory-group apply lock on the scratch branch and no longer appear here. The shape still dies on a shared-**meta-buffer**
SIGSEGV (`ggml-backend-meta.cpp`) and, on another run, a HIP `operation not permitted when stream is capturing`
(`ggml_cuda_async_input_set`); the pair OFF is clean on every shape tried. See `WORKLOG.md` 261122.032.
