# 0021 - env-gated decode-phase instrumentation in llama_decode

llama.cpp commit `7c7d9eeff` (parent `d9b1ef288`), NOT included in the beta
patch (debug scaffolding, not promotable) - preserved here so the diagnostic
capability survives.

## What it does

Adds per-phase timers inside `llama_context::decode()` and the sched sync,
printed to stderr only when `LLAMA_DECODE_PHASE_DEBUG=1`:

    [dphase] balloc_init 0.001 ms
    [dphase] mem_update 0.002 ms
    [dphase] mctx_ready 0.006 ms
    [dphase] ubatch_compute_done 20.043 ms     <- the full decode wall
    [dphase] sched_sync 0.121 ms               <- llama_synchronize cost

Touch points: `src/llama-context.cpp` only (~36 lines, env-gated, zero cost
when the env is unset - a `static const bool` read once + `ggml_time_us()`
pairs). The compute-phase timers were added in `process_ubatch` first, then
moved to the decode top level for the final measurements.

## Why it exists (the host-overhead hunt, 2026-09-03)

The llama-server was ~1 ms/token slower than the bseq_pp bench harness at
the same real KV, and the GPUs were 96% busy - pointing at either a hidden
GPU-work difference or host stalls. The instrumentation proved the decode
is NOT the gap:

- `llama_context::decode` in the server = 20.05 ms/step steady (last-150
  avg over a 3000-token run) + sched sync 0.121 = 20.16 ms/token - at
  parity with the bseq harness (19.6-20.1) at the same real KV.
- Graph reuse works (1991/2000; the 12 rebuilds/token-run amortize to
  nothing).
- The remaining ~1.0 ms/token (tg 21.15 vs decode 20.16) = the server's
  update_slots LOOP overhead: t_pre 0.036 + t_post 0.148 + t_sampl 0.107
  (the server's own DEBUG_TIMINGS, compile-time enabled in
  server-context.cpp) + ~0.67 ms/token of untimed queue/event-loop churn
  (the NEXT_RESPONSE task round-trip + update_slots wakeups + metrics).

## Apply

```
git checkout d9b1ef288      # or the beta tree (identical content)
git apply 0021-decode-phase-instrumentation.patch
```

## Usage

```
LLAMA_DECODE_PHASE_DEBUG=1 <server or harness> ...
# grep '[dphase]' the log
```

## Companion tools / files

- `/tmp/bseq_pf` = bseq_pp variant taking `BSEQ_PROMPT_FILE=` for a real
  prompt prefill (used to rule out prompt-content effects: cyclic vs real
  prefill both 19.6 ms/step).
- llama-cli == llama-server for decode purposes (`cli_server` wraps the
  server context) - use the server's DEBUG_TIMINGS or this env instead.
- Full write-up: `../HANDOVER.md` (host-overhead hunt section).
