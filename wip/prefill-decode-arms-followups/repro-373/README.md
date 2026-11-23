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

## `FIX-candidate-373.patch`

The fix set, cut from `~/llama.cpp` on the scratch branch `crash373-fix` (`ccfd7042f..fdc58fc78`, five commits, 832
insertions in 16 files) against the release tree.  **SHIPPED in the r347 block-17 amendment** (`WORKLOG.md`
261122.036) -- the patch is kept as the review/diff artifact, not as a delivery patch: it is not in `patches/` and
`patches/0017-...` carries the same change.  It contains:

1. **the cell-map apply lock** — `llama_memory_i::apply_mutex()` + one lock per memory **GROUP** shared through
   `create_memory()`, taken at `init_batch`/`apply`/`clear`/`state_*`/every `llama_memory_*` op/`seq_park`, plus one
   piece of **hardening that is still unproven** (`sched_alt_select()` sets the Meta view slot on every call);
2. **the Meta split-state cache made per VIEW SLOT** (`[4][8]`), the slot bound for every graph build/alloc
   (`meta_bind_view_slot()`), and the async-input ring waiting on each **origin's own** stream (its wrap path called
   `cudaDeviceSynchronize`, illegal while a peer arm captures a graph);
3. **the VMM pool's peer grant made best effort** (owner alone, then each peer alone, a per-pool mask, ranges in a
   registry, and `ggml_cuda_peer_copy_ok()` answering for pools as well as slabs, so a refused pair stages);
4. **the synchronous host paths' private stream** (`ggml_cuda_host_stream()`, one per (thread, device)) instead of the
   `cudaStreamPerThread` sentinel, plus `TODO` #366's derived decode reserve (`common_arms_dec_reserve()`);
5. **`TODO` #365 — the Meta buffer's per-child `cpy_tensor_n`**, so a `-sm tensor` handover is a device copy
   (1.4-1.9 ms against 72-77 ms), with `LLAMA_RS_VERIFY_TAKE` extended to whichever arm runs and
   `LLAMA_RS_VERIFY_TAKE_DUMP=<file>` making two arms comparable with `cmp(1)`.

## Harness knobs

| env | what |
|---|---|
| `LONG_TOK` / `MED_TOK` / `HEAD_TOK` / `GEN_TOK` / `MED_DELAY` / `ROUNDS` | the workload (defaults: 71000 / 23000 / 2000 / 800 / 13 s / 1) |
| `EXTRA_ARGS` | appended to `runme`'s own args (e.g. `--no-prefill-decode-arms` for the control, `-lv 4` for a census) |
| `TEXTDUMP=<dir>` | write every response body to `<dir>/<tag>.txt`, so two arms driving the same workload can be `diff`ed |
| `ARM_TEMP=0` | greedy: at the runme's 0.7 a tiny numeric difference flips a sampled token, and a CO-BATCHED pair is stress only, never a text claim (`GREEDY-PURITY.md` §63).  **Not `TEMP`**: exporting `TEMP` to a GPU process breaks the ROCm 10.1 runtime (its JIT dies with a stack-write fault at load — `WORKLOG.md` 261122.014), which cost this session one detour |
| `SERIAL=1` | fire the requests one at a time instead of overlapping them, so the pair's step composition (and every handover's cell placement) is reproducible and two arms' state dumps can be compared byte for byte |
| `LLAMA_RS_VERIFY_TAKE=1`, `LLAMA_RS_VERIFY_TAKE_DUMP=<file>`, `LLAMA_RS_FORCE_SERIALISED=1` | the faithful-copy instrument and the A/B control (the server's own, read from the environment) |

## Status

**Closed.**  The harness reproduced a P0 (a qwen4exp abort at a request boundary plus, behind it, a Meta split-state
SIGSEGV and a libstdc++ RB-tree SIGSEGV) in ~30 s on the maintainer's shape, and the fixed set has run it clean in
both configurations.  The **gate** that replaces this harness for regression purposes is
`scripts/gate-arms-soak.sh boundary` (`MANIFESTS.md` gate 17b): it asserts what this driver only reported -- every
request completes, no abort/assert/`ROCm error`, the take ran, a sequential fresh request's text matches a
single-context server, `check-gpu-faults.sh` clean -- on a validated hybrid split (a small model over a 3-way split
is *not* a shape anyone runs; it faults in `find_slot()` on the released build too).  Keep this harness for driving an
arbitrary shape by hand on the maintainer's own model.
