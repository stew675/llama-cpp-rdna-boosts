# BASE BLOCKER B1 — still live at `919794500` (measured 2026-11-20)

**This is the r278 campaign's blocker, unchanged, re-measured on the new base.**  Full mechanism, stack
traces and the candidate fix: [`../../archive/work/upstream-rebase-261113/BASE-BLOCKER.md`](../../archive/work/upstream-rebase-261113/BASE-BLOCKER.md).
The standing decision from r278 was "carry the revert in block 00; drop it at the next re-base".  It is
carried again — see `DECISIONS.md` **D2**.

## What was measured today, on the pristine `919794500` snapshot

Three builds, all `--offload-arch=gfx1201` against `/opt/rocm-10.1.0-gfx120X`:

| build | `GGML_HIP_NO_VMM` | command | result |
|---|---|---|---|
| `build-base` | `OFF` (the delivery's config) | `HIP_VISIBLE_DEVICES=0 llama-cli -m Qwen3.5-4B-Q8_0 -ngl 99 -sm layer -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn` | **HANG** — `SIGKILL`ed at 150 s (rc 137), no `load time` line |
| `build-base-novmm` | `ON` (control) | same | **works** — 2 s, `Prompt: 247.7 t/s \| Generation: 92.2 t/s`, `100 chars sha=1c5d32ac537d` |

So the failure is **VMM-pool-specific**, exactly as at r278, and the base is otherwise healthy (it loads,
prefills and decodes, and its greedy text is the `1c5d32ac537d` reference the r278 ladder also used).

**`GGML_HIP_NO_VMM` is a CMake option, not an environment variable** (`ggml/CMakeLists.txt:219`, default
`ON`; `ggml/src/ggml-hip/CMakeLists.txt:104` turns it into a compile define).  A runtime
`GGML_HIP_NO_VMM=ON` changes nothing — `build-base` and the "VMM=ON" arm of the first attempt were the same
binary, which is why both hung.  Vary it at configure time and use a separate `BUILD_DIR`.

## The trigger is still the `#30542` code

`git grep -n AddressReserve master -- ggml/src/ggml-cuda` finds `ggml-cuda.cu:709-712`: the VMM pool still
reserves the address range **before** `cuMemCreate`, and still halves `max_size` while the error is
`CUDA_ERROR_OUT_OF_MEMORY`.  Nothing in `4f9dfa0e3..919794500` touched that sequence (the 13 upstream commits
to `ggml-cuda.cu` are MMID/bf16/norm-fusion/set_n/warp-sync work).

## Consequence for the ladder

- R00 carries the full revert of `ac931b5bc` again, on top of upstream's 13 `ggml-cuda.cu` commits.
- **No rung can be verified without it**, because every coherence arm is a GPU arm.  A rung that cannot be
  verified is not a rung.
- The APU-only gates (`gate-uma-experts.sh`, `gate-prefill-server-slots.sh`) still cannot run on this
  gfx1201 box — that limitation is unchanged and is recorded in `MANIFESTS.md`.
