# archive/work/win-setenv-shim (CLOSED, promoted in r280) — contributor PR #310

**Status: CLOSED 2026-11-14 — promoted into the delivery as a block-11 amendment in `v17-4f9dfa0e3-r280`.**
Nothing here is delivery work and nothing is open.  The code is in `patches/0011-*`; the record of *shipping* it is
the `WORKLOG.md` entry `261114.001` and `INVARIANTS.md` r280; the block note is `patches/README.md` block 11; the
per-block verification contract row is `MANIFESTS.md` gate 12.  This file is the campaign record: what the
contributor offered, what we measured, what we changed, and what we deliberately did not take.

Contributor: **DanoPTT (Daniel Petrik)** — <https://github.com/stew675/llama-cpp-rdna-boosts/pull/310>, one patch
`0001-tests-setenv-unsetenv-shim-for-Windows-in-test-llama.patch` (`tests/test-llama-archs.cpp`, +15) cut against
**r279**, plus a README, filed under `wip/win-setenv-shim/`.

Artifacts kept here:

| file | what | sha256 |
|---|---|---|
| `0001-win-setenv-shim-as-submitted.patch` | the patch exactly as the PR carries it (`git format-patch`, applies clean on the r279 tree) | `0520b055cd3cdd01ee0336eacf2bfc910f8418f4129262040b0807d67558e20b` |
| `0001-win-setenv-shim-final-cut.diff` | the **delivery's** form (`git diff f61ac4bd8 f25fd3ff2`, i.e. r279 -> r280): the call-site platform split plus `<cstdlib>` | `37da017d2d3d604be7974428b2fb4c49f9f5dac3a11c9ad3d1c251cd3b6a8e14` |

## What the defect is

Block 11 hands the managed PLE reader budget to the model load as `LLAMA_LAZY_BUF_MB`; in
`tests/test-llama-archs.cpp`'s `get_model()` it exported it around `llama_model_load` with the POSIX
`setenv`/`unsetenv`.  The MSVC CRT that clang-cl uses under the Windows HIP SDK has neither (`getenv`,
`_putenv`, `_putenv_s` only), so **a Windows build that enables the tests does not compile** — the
libraries, `ggml-hip` and `llama-server` are unaffected, which is exactly why no gate on this box ever
saw it.  The reporter carried the equivalent patch from **r236** (when block 11 gained the export) to r279.

## What we measured before accepting

Every claim in the PR was re-run on r279 and held:

* `scripts/check-windows-portability.sh` default: **OK: 337 files scanned** — `tests/` is not in the
  default path set, so the file was never looked at.
* `--modified-too`: **6 hits** (`common/common.cpp` `sysconf` on the ppc64 branch, hexagon's
  `mmap(bool)` + `mmap()`, zDNN's `sysconf`, `ggml.c`'s `getpid`, `llama-mmap.cpp`'s `sysconf`), all
  upstream code, and nothing in `tests/`.
* `setenv`/`unsetenv`/`putenv` were absent from the lint's POSIX call list.
* The patch `git apply --check`es clean on the r279 tree.

## What the delivery took, and what it changed

1. **Placement — folded into block 11**, not committed as `wip/win-setenv-shim/`.  `wip/` is experimental
   work by policy (`AGENTS.md`), and this is a Windows build fix for shipped code: a `wip/` patch nobody
   applies is the same defect with a README.  Folded with the r278/r279 procedure (`edit` + `fixup` +
   `rebase --autosquash`), so blocks 12-16 were re-cut with an unchanged delta.
2. **Implementation — the call-site split, not a shim.**  The PR defines file-static `setenv`/`unsetenv`
   in the global namespace under `#ifdef _WIN32`.  Two problems with that: a platform header may already
   declare them (redefinition error) and the portability lint reads a *definition* as a call site.  The
   delivery already had the idiom in block 01 (`NCCL_P2P_DISABLE` in `ggml-cuda.cu`), and the test now uses
   it: `#if defined(_WIN32)` -> `_putenv_s(name, value)` / `_putenv_s(name, "")` (an empty value **removes**
   the variable, the exact mirror of `unsetenv`), `#else` -> `setenv`/`unsetenv`.  `#include <cstdlib>` was
   added because block 11 had been relying on a transitive `<stdlib.h>` for the POSIX pair.
3. **The detector gap — fixed, which is the part worth keeping.**  `check-windows-portability.sh` gained
   `tests/` in its default path set, `setenv`/`unsetenv`/`putenv` in the call list, and an **added-line
   report mask** for files the delivery edits: the whole file is still parsed (an upstream `#ifdef _WIN32`
   above an inserted line still excludes it) but only the lines `git diff -U0 base..HEAD` adds are reported.
   That is precisely the PR's own suggestion, and it is the only thing that could have caught this in r236.
   **Negative control:** the r279 tree reports `tests/test-llama-archs.cpp:844 setenv` and `:846 unsetenv`,
   the r280 tree reports none, both at **509 files** scanned.

## Gates

`validate-set.sh` green (strict apply on a fresh tarball at `4f9dfa0e3`, tree `b83424ef5`, 509 files linted);
`gate-per-block-build.sh --changed-from f61ac4bd8` **2/2** on the `all` profile (the commit touches `tests/`);
coherence `1c5d32ac537d` (dense 4B, unchanged since r152); no GPU fault.  No kernel, model-graph, loader or
API code moved, so the prefill-logit gate and the block gates whose code did not move were not re-run, and
r279's numbers stand.

**The Windows claim is review-level, not build-level** — there is no Windows runner here (same standing as
issue #307 in r279).  What backs it: the shape now matches the idiom block 01 already ships in a Windows-HIP
build, `_putenv_s`'s empty-value-removes semantics is the CRT contract, and the reporter builds Windows with
the tests enabled.  A hypothetical MinGW build is the one arm neither of us has compiled.
