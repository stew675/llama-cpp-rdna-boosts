# win-setenv-shim: `setenv`/`unsetenv` in `test-llama-archs.cpp` on Windows

*Written by Claude (AI) on behalf of DanoPTT.*

**Applies to:** r279 (`v17-4f9dfa0e3-r279`, tree `d6372d47a`), `git apply --check` clean. The hunk belongs in **block 11**: that is the block that adds the `LLAMA_LAZY_BUF_MB` export to `tests/test-llama-archs.cpp`.

## The problem

Block 11 hands the managed PLE reader budget to the model load as `LLAMA_LAZY_BUF_MB`. In `tests/test-llama-archs.cpp` it exports the variable around `llama_model_load` with the POSIX `setenv` / `unsetenv`. The MSVC CRT has neither of them, and clang++ under the Windows HIP SDK uses the MSVC CRT. So a Windows build that includes the tests stops in `test-llama-archs.cpp`. Everything else builds: the libraries, `ggml-hip` and `llama-server`.

We first hit this at r236 and have carried the patch below in our Windows builds ever since (r236 through r277, now r279). Our builds always include the tests, because we want `test-backend-ops` in the artifact.

## The patch

`0001-tests-setenv-unsetenv-shim-for-Windows-in-test-llama.patch`, 15 added lines under `#ifdef _WIN32`:

- `setenv(name, value, overwrite)` → `_putenv_s(name, value)`. It keeps the POSIX `overwrite == 0` semantics: a variable that is already set is left alone.
- `unsetenv(name)` → `_putenv_s(name, "")`. On Windows, an empty value removes the variable.

Non-Windows builds do not change. The only call site passes `overwrite = 1`.

## Why `validate-set.sh` does not catch it

`scripts/check-windows-portability.sh` misses this on two independent counts:

1. **Scope.** `PATHS=(ggml/ src/ common/ tools/)`, so `tests/` is never scanned, not even with `--modified-too`.
2. **Call list.** `setenv` / `unsetenv` are not in the list of POSIX-only calls.

On top of that, the default `--diff-filter=A` scope only covers files the delivery adds. `test-llama-archs.cpp` is an upstream file that block 11 edits. The script's header explains the edited-file exclusion as "upstream's Windows CI's business". That holds for code upstream wrote. It does not hold for a call the delivery itself inserts into an upstream file, because upstream CI never builds the delivery.

For reference, running the r279 script on the r279 tree:

- default: `OK: 337 file(s) scanned`
- `--modified-too`: 6 hits (`sysconf`, `getpid`, hexagon `mmap()`, …), all in upstream code, and still nothing in `tests/`

A possible lint change, which this PR does not make (it's your call how wide the sweep should go):

- add `tests/` to `PATHS`
- add `setenv` / `unsetenv` to the call list
- for edited files, scan only the lines the delivery adds (`git diff -U0` `+` lines) rather than the whole file. That catches calls the delivery inserts into upstream files without pulling in upstream's own hexagon/zDNN noise.
