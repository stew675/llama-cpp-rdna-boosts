# win-ssize-t: `(ssize_t)`/`%zd` in block 17's state-move VERIFY stops a Windows build

*Written by Claude (AI) on behalf of noctrex.*

**Applies to:** r347 (`v18-919794500-r347`, tree `d4ab6461`; the block-17 amendment release, cut 261123) -- `git apply --check` clean on both the r346 tree (`a6f877eab`, hunk at `llama-context.cpp:7213-7214`) and the r347 tree (same hunk, offset 54 lines -> `7263-7268` region; the amendment added code earlier in the file, the VERIFY hunk context is verbatim). The hunk belongs in **block 17**: that is the block that adds the serialised state-move VERIFY logging to `src/llama-context.cpp`. The two lines are present verbatim in `patches/0017-...patch` and `rdna-boosts-all.patch` at `main` (r347 block-17 patch lines 2024-2025).

## The problem

Block 17's VERIFY for the serialised state move (`llama_memory_seq_state_take`, the two-arm server's state-handover check) logs the first differing byte as `"first at %zd"` with an `(ssize_t)` cast of the `size_t` offset. `ssize_t` is a POSIX type: the MSVC/UCRT headers that `clang++` uses under the Windows HIP SDK never declare it, and `llama-context.cpp` includes no header that would (its POSIX-only siblings guard this - `llama-lazy-reader.cpp` gates its `pread`/`ssize_t` behind `LLAMA_LAZY_READER_POSIX`, `llama-mmap.cpp`'s `::read` arm sits in the non-Windows `impl` branch - but the new VERIFY logging is unguarded). A Windows build of the library stops at the first `src/` target:

```
FAILED: [code=1] src/CMakeFiles/llama.dir/llama-context.cpp.obj
...clang++.exe -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP ...
    -std=gnu++17 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt ...
src/llama-context.cpp:7498:80: error: use of undeclared identifier 'ssize_t'
 7498 |                 __func__, (int) seq_id, n_diff, size, n_diff_meta, meta_size, (ssize_t) first,
      |                                                                                ^~~~~~~
1 error generated.
...
ninja: build stopped: subcommand failed.
```

HIP SDK 10.x clang++, Windows 11, `-std=gnu++17 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt` (MSVC STL + UCRT headers). Everything before it builds: the whole ggml/ggml-hip kernel set (the ~215k-line log is warnings only) and 468 of the `llama` target's objects; this is target `[469/722]` of the default ninja set. With the optional set applied (`apply-all.sh --include-optional-all`) the same two lines sit at `llama-context.cpp:7497-7498`; on the r346 default tree they are 7213-7214.

## The patch

`0001-src-print-the-block-17-state-move-VERIFY-offset-with.patch`, 2 changed lines in `src/llama-context.cpp`:

- `(ssize_t) first` -> `(long long) first`
- `"first at %zd"` -> `"first at %lld"`

`long long` and `%lld` are standard everywhere the delivery builds and UCRT's printf supports them. Non-Windows output is unchanged: the value is a byte offset that is never negative, and the `(size_t)-1` "no difference" sentinel still prints as `-1` through the signed cast. No other POSIX-only type or call was found on the delivery's inserted lines in the affected file: every other `ssize_t`/`pread`/`::read` use in the applied tree is inside a Windows-excluding region.

## Why `validate-set.sh` does not catch it

`scripts/check-windows-portability.sh` cannot see this class, by its own header's "WHAT IT IS NOT": it lints POSIX-only **calls** (`close`, `pread`, `setenv`, ...) on lines compiled on Windows, and explicitly not a compiler - it "cannot see ... a type that differs under MSVC". `(ssize_t)` is a type name in a cast, so no entry in its call list can carry it, and both of its run gates (check 7's portability lint, check 10's `check-system-headers.py` include-closure scan, which gates macros the delivery calls, not types it names) pass on the r346 tree. With no Windows runner, a reporter's build log is the detector - as the header says for #231, and as happened here.

A possible lint extension, which this PR does not make (your call how wide the sweep should go): flag the identifier `ssize_t` on a Windows-compiled line the way the script already flags `near`/`far`/`pascal`/`cdecl` (issue #306) - a name rule rather than a call rule, so it carries this class without re-opening the upstream noise the call-list design avoids.

## Verification

- `git apply --check` clean on the r346 default applied tree, reconstructed from the `919794500` tarball + `scripts/apply-all.sh` (base tree `9169416` and applied tree `a6f877eab` both match `release.json` exactly), and re-verified against the r347 tree the same way (applied tree `d4ab6461` matches `release.json`).
- **The fix builds green on Windows.** With only these two lines changed, the same Windows build (HIP SDK 10.x clang++, repo-compile.py, default backend set) that stopped at `[469/722]` with the `ssize_t` error now completes: the resume rebuilds 238 targets from `llama-context.cpp.obj` down -- `llama.dll`, `llama-server`, `llama-cli`, `llama-mtmd-cli` and the rest of the tools -- 0 errors, `ninja` exits at `[238/238]` with every executable linked. Block 17's VERIFY path is opt-in-only code (`--prefill-decode-arms`, default OFF); the failure was compile-time on the unguarded line, and the change cannot affect non-Windows builds or output behaviour (standard cast + format swap).
