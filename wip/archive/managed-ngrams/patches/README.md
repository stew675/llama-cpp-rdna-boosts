# Patches — managed ngram LRU loading

7 commits generated against **llama.cpp master `a7cc83bba`** (same fork point
as the main 12-patch set), on the `ngram-disk` branch of `~/llama.cpp`.

## Apply

```sh
git checkout a7cc83bba
git checkout -b managed-ngrams
git am patches/0001-*.patch      # in this directory
```

or review the combined diff:

```sh
git apply --check managed-ngrams-combined.diff
```

## Series

| Patch | Content |
|---|---|
| 0001 | `llama_lazy_reader` cache class + `tests/test-lazy-reader.cpp` |
| 0002 | plumbing: `n_lazy_buf_size`, `--lazy-buffer-size` |
| 0003 | qwen4exp integration (reader, TENSOR_SKIP_MANAGED branch, user-path guards) |
| 0004 | PLE fixture + managed roundtrip in `test-llama-archs`; F32 reader path |
| 0005 | `lazy_read::add_range` — keep the skipped table out of the load-time WILLNEED prefetch |
| 0006 | silence the misleading "unused tensor" warning in managed mode |
| 0007 | ggml meta-backend: host tensors in tensor-split |

## Notes

- This is a WIP patch set, not part of the push distro (blocks 01-12).
- `tests/` in the parent directory are the *working copies* of the test
  sources; the patch files carry the repo-tree versions.
- The ggml change (0007) touches core `ggml/src/ggml-backend-meta.cpp` and
  was validated on 3x R9700 (608 arch-test configs) plus CPU (128 configs)
  with zero regressions; it may be worth upstreaming separately.
