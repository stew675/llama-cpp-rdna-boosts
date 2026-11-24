# `repro/` — the live reproducers behind the three beta-tester crashes

These are the scripts that **actually reproduced** the failures on this box (2026-11-24).  They are kept
verbatim-ish so the next session does not have to rediscover the recipes; each one was run repeatedly and its
numbers are in `../RUNS.md`.  They are **not** gates — the gates are in `../gates/`.

Every script: `setsid` + ONE `TERM` + wait, `HIP_VISIBLE_DEVICES` pinned, `MOE_EXPERT_HEATMAP=load`, and (where
text matters) `ROCBLAS_USE_HIPBLASLT=0`.  Wrap a run in `flock -w 1800 ~/.cache/rdna-boosts/gpu.lock` unless the
script says otherwise, and never run two GPU jobs at once.

| script | what it reproduces | runtime |
|---|---|---|
| `make-bigprompt.sh` | builds the ~77K-token prompt the recipe needs (`~/.cache/rdna-boosts/prompts/bigprefill-3x.txt` = the 25 763-token corpus × 3) | instant |
| `repro-mtp-prefill.sh` | **the crash recipe**: a 10 000-word generation in flight, then 5 s later a ~77K-token prefill on the other slot. Pre-fix it aborted with `bit out of range 0 - FD_SETSIZE` (crashes #2/#3); post-STEP-14 it reaches the **2-D read of `result_output`** that `../HANDOVER.md` is about | ~3-4 min for the first round (model load dominates) |
| `fd-census.sh` | prints an arm's open-fd count and its `/dev/shm/nccl-*` count right after `/health` — the measurement that proved the FD ceiling (`MODE=off\|on AR=auto\|ce DEV=0,1,2`) | ~2 min per arm |
| `fd-1gpu.sh` | the 1-GPU pair-ON fd sanity (10 fds — no comm layer at all) | ~30 s |
| `repro-cancel.sh` | crash #1: two long streaming generations, `kill -9` one client, check `/health` (`MODE=on` dies pre-STEP-13, `MODE=off` survives) — now covered by `../gates/gate-arms-cancel.sh` | ~1 min |
| `probe-hip-2d-copy.cpp` | **isolation attempt for the 2-D read refusal** (STEP 15): every D2H `hipMemcpy2DAsync` argument combination the failing read used -- pitches, 1..6 copies, 1/4/16/64-aligned destinations, pinned (`Default`/`Portable`/`Mapped`) and pageable hosts, a destination pinned by ANOTHER device, two copies in flight from different devices, the bounce+scatter shape. **All succeed on an idle box**, which is why that crash could only be diagnosed end-to-end (the driver log, `AMD_LOG_LEVEL=3`, is the evidence -- see `../gates/gate-arms-state-split.sh`'s header). Build: `hipcc -O2 --offload-arch=gfx1201 -o /tmp/probe-hip-2d-copy probe-hip-2d-copy.cpp`, run `HIP_VISIBLE_DEVICES=0,1,2 /tmp/probe-hip-2d-copy`. ~10 s | instant |
| `repro-statesave.sh` | a prompt-cache save on a 3-device `-sm tensor` split **with the pair off** (`MODE=off`), + optionally the drafter: it SURVIVES, which is how the 2-D read was attributed to the pair's per-round draft checkpoint rather than to prompt-cache saves | ~3 min |

## Things that cost time, already learned

* **`-c` must leave the 77K prompt room in ONE slot.** With `-np 2` and no `--kv-unified` each slot gets `c/2`;
  at `-c 16384` the big request never even reaches a slot (`n_ctx_slot = 8192`, and the log shows only the small
  generation task) — a vacuous run.  Use `-c 204800` (slot 102 400).
* **Check that the request actually ran.** `repro-mtp-prefill.sh` prints the big request's HTTP status and body
  and the log's `task.n_tokens`; a silent reject looks exactly like a pass.
* The exponential-notation mistake: `python3 -c 'json.dumps({..."stream":false})'` fails (`false` is not Python) —
  use `False`, or the body is never built and the "prefill" is an empty request.
* `-ncmoe N` is the delivery's spelling (`--n-cpu-moe` also parses, `-n-cpu-moe` does not).
* The transcript in `RUNS.md` says which `-lv` each arm needs: **no `-lv`** for a text/hash arm, **`-lv 4`** for a
  census/mechanism arm, never `-lv 3`.
