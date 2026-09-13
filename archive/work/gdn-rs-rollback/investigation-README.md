# Fix: recurrent-state rollback for long/non-verify batches

Two related defects in the recurrent (GDN) rollback snapshots:

1. **ngram-mod long verify batches** (the reported bug): the whole-batch chunked
   GDN kernel writes no rollback snapshots, but ngram-mod verify batches are
   longer than the snapshot depth, so a partial acceptance restored a stale slot.
2. **rollback of the whole last batch**: the kernel only snapshots per-token
   states (slots `0..n_tokens-1`), so a rollback of exactly `n_tokens` tokens
   needs the pre-batch state at slot `n_tokens`, which was never written. This is
   what `test_multi_seq_split_replay` exercises.

Both are fixed here.

## Part 1 - reported bug

`llama-server` with `--spec-type draft-mtp-adaptive,ngram-mod` produced:

```
W seq_rm: rollback crossed a batch boundary: seq 0 rollback=1 but the last batch
decoded 49 tokens ... The recurrent snapshot path assumes only verify batches ...
if that batch exceeded the chunked-GDN threshold it wrote no usable snapshot and
the restored state is wrong ...
```

Root cause: the rollback keeps `K = n_rs_seq + 1` snapshots and only the
**sequential** GDN kernel writes them. The **whole-batch chunked prefill** kernel
writes slot 0 only. It was gated by `n_tokens > max(K, 16)`, assuming a verify
batch satisfies `n_tokens <= K = n_rs_seq + 1`.

`n_rs_seq` is sized only from `speculative.draft.n_max` (`need_n_rs_seq()`), i.e.
7 here, while ngram-mod can draft up to `--spec-ngram-mod-n-max` (64). A verify
batch is therefore up to `1 + 64 = 65` tokens: it takes the chunked path, writes
no snapshots, and a small tail rollback restores a stale plane.

MTP alone is fine (drafts <= 7, batch <= 8 <= 16, sequential). ngram-mod alone is
fine (`n_rs_seq == 0`, demoted to checkpoints). Only the combination breaks.

### Fix

Decouple the chunked threshold from `K`. The sequential kernel already writes the
last `K` per-token snapshots of any batch, and only `n_rollback <= n_rs_seq` is
served from snapshots, so the requirement is just: **any batch that can be rolled
back into must run the sequential kernel.** Pass the longest possible draft to the
kernel as `n_rs_batch` and use:

```c
GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16, n_rs_batch);
```

`common_speculative_n_max(&params.speculative) + 1` flows through
`llama_context_params::n_rs_batch` -> `llama_cparams` -> `ggml_gated_delta_net()`
op param 1 -> the CUDA dispatch, and also into `llama_memory_recurrent` so the
`seq_rm` guard stays a correct invariant check.

| spec config            | n_rs_seq | n_rs_batch | chunked threshold |
|------------------------|---------:|-----------:|------------------:|
| none                   | 0        | 1          | 16 (K == 1, N/A)  |
| draft-mtp n_max=7      | 7        | 8          | 16 (unchanged)    |
| mtp 7 + ngram-mod 64   | 7        | 65         | 65                |
| ngram-mod only 64      | 0        | 65         | N/A (K == 1)      |

Snapshot memory is unchanged (`n_rs_seq + 1` planes, ~1.2 GiB on this model).
Setting `n_rs_seq = 64` instead would have cost ~+8 GiB.

## Part 2 - rollback of the whole last batch

For a batch of `m` tokens the kernel writes slots `0..m-1` (`slot s` = state `s`
tokens back). A rollback of `m` tokens needs slot `m` = the state *before* the
batch. When `m < K` there is room for it; it just was not written, so the rollback
restored whatever older state happened to be in slot `m`. The dsv4 cache keeps
such a pre-batch (restore) state; the recurrent cache did not.

### Fix

When `0 < n_tokens < K`, also write the pre-batch state to slot `n_tokens`, for
both the ssm state and the conv state, at the graph level in
`llm_build_delta_net_base`:

* `build_recurrent_attn`: `ggml_cpy(reshape_2d(s), ssm_states_all @ slot n_tokens)`
* `build_conv_state`:    `ggml_cpy(reshape_2d(conv_states), conv_states_all @ slot n_tokens)`

`s` and `conv_states` are the `build_rs` outputs (fresh tensors), so this is a
plain graph copy with no kernel change and no effect on the model output for long
batches (`n_tokens >= K` guards it off).

## Files

Patch: `gdn-rs-rollback-bound.patch`, applies cleanly to `rdna-boosts` @ `55819a887`.

Kernel / op:
* `ggml/include/ggml.h`, `ggml/src/ggml.c` - `ggml_gated_delta_net()` gains
  `n_rs_batch` (op param 1)
* `ggml/src/ggml-cuda/gated_delta_net.cu` - threshold uses it
* `ggml/src/ggml-vulkan/ggml-vulkan.cpp` - clone preserves op param 1
* `src/models/delta-net-base.cpp` - passes `cparams.n_rs_batch`; writes the
  pre-batch ssm/conv state slots

llama / common:
* `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp` -
  `n_rs_batch` plumbing + log
* `common/common.cpp` - derive from `common_speculative_n_max()`
* `common/speculative.cpp` - draft context keeps `n_rs_batch = 1` (K == 1)
* `src/llama-memory-recurrent.{h,cpp}`, the three hybrid memories,
  `src/llama-model.cpp` - guard bound
* `tests/test-backend-ops.cpp` - new op argument

## Verification

Build: `~/bin/build-llama-rocm-714` (or build the needed targets directly; the
`llama-ui-assets` step can hang on a HuggingFace download).

### A/B regression (`test-long-batch-rollback.cpp`)

Decodes a 49-token batch, rolls back 3 tokens through the snapshot path, replays,
and compares logits against a reference that never decoded past the rollback
point. `TEST_RS_BATCH` sets the bound:

```
TEST_RS_BATCH=64  ->  PASS: max logit diff = 0        (sequential kernel)
TEST_RS_BATCH=8   ->  FAIL: max logit diff = 8.47255  (chunked kernel)
                      + the seq_rm warning fires
```

`TEST_RS_BATCH=8` reconstructs the old `K`-only bound, so one binary shows both
the bug and the fix.

### Whole-batch rollback (`test-recurrent-state-rollback`, qwen35)

```
test_rollback                        : recurrent rollback checkpoint restored successfully
test_multi_seq_split_replay          : multi-seq split replay matched (max diff 0)
test_multi_seq_split_replay          : seq-1-only decode independent of seq 0 (max diff 0)
```

for both cache fill `0x00` and `0x3e`, exit 0. On the unmodified base the same
test fails with `max diff 6.51987, first at seq 0 pos 16`.

### Op correctness

`test-backend-ops test -o GATED_DELTA_NET`: **46/46 passed** on ROCm0, ROCm1 and
CPU, including `K=2/3/4` and multi-seq cases.

### End-to-end with the exact reported config

llama-server with
`--spec-type draft-mtp-adaptive,ngram-mod --spec-draft-n-max 7
--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
(`LLAMA_TRACE=1`) produced verify batches of 49 and 65 tokens:

```
accepted 48/48 draft tokens
accepted 64/64 draft tokens   (x many)
accepted 56/64 draft tokens (restore checkpoint)
```

A repeated-prompt run does not reliably reject only the last 1-7 of a long draft,
so the exact snapshot path was forced deterministically with the synth rates
`--spec-synth-rates 1.0,...,1.0,0.0` (63 x 1.0, 1 x 0.0), which always rejects
draft position 64:

```
accepted 63/64 draft tokens   (x 23, NO restore checkpoint -> snapshot path)
```

23 occurrences of a 65-token verify batch followed by a 1-token snapshot rollback
(the reported `rollback=1` shape), with **zero** `rollback crossed a batch
boundary` warnings and 1500 generated tokens.

### End-to-end plumbing

`check_rs_batch` (no model needed):

```
no spec:            n_rs_seq=0 n_rs_batch=1
mtp n_max=7:        n_rs_seq=7 n_rs_batch=8
mtp 7 + ngram 64:   n_rs_seq=7 n_rs_batch=65
ngram only 64:      n_rs_seq=0 n_rs_batch=65
```

## Performance

Prefill is unaffected: real prefills stay chunked. Only short batches
`(max(K,16), n_rs_batch]` - speculative verify batches - move to the sequential
kernel, which is what writes the snapshots they are rolled back into.

`test-backend-ops perf -o GATED_DELTA_NET` on one R9700 (gfx1201), head_count 32,
head_size 128, n_seq_tokens 64:

| path                                     | us/op |
|------------------------------------------|------:|
| chunked (default)                        | 22.58 |
| sequential (`GGML_CUDA_GDN_CHUNKED=0`)   | 86.46 |
| autoregressive n_seq_tokens=1            | 26.49 |

A ~49-token verify batch pays roughly 44 us more per GDN layer, ~2.1 ms across 48
recurrent layers (less after tensor split) - low single-digit percent of a
long-draft verify step, and cheaper than restoring a checkpoint and replaying the
accepted prefix on every small tail rejection. `GGML_CUDA_GDN_CHUNKED=0` is no
longer needed for correctness.

## Known limitation

The analogous Mamba `ggml_ssm_scan` path (`src/models/mamba-base.cpp`) has the
same two characteristics (K-snapshot kernel, no pre-batch slot). It is not
touched here because this work targets the qwen35/delta-net model. A Mamba model
would need the same `n_rs_batch` bound and pre-batch ssm/conv slots.

## Reproducing

```sh
# build (avoid the UI download: build the needed targets)
cmake --build ~/llama.cpp/build-rocm --target test-recurrent-state-rollback -j 16

# regression test binary
./build-regression-test.sh

export LD_LIBRARY_PATH=~/llama.cpp/build-rocm/bin
MODEL=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf

HIP_VISIBLE_DEVICES=1,2 TEST_RS_BATCH=64 ./test-long-batch-rollback -m $MODEL -ngl 99
HIP_VISIBLE_DEVICES=1,2 TEST_RS_BATCH=8  ./test-long-batch-rollback -m $MODEL -ngl 99

HIP_VISIBLE_DEVICES=1,2 ~/llama.cpp/build-rocm/bin/test-recurrent-state-rollback \
    -m $MODEL -ngl 99 -c 512 -b 512 -ub 512
```
