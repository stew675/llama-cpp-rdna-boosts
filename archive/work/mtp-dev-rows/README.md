# mtp-dev-rows — the MTP device h-rows path, its Windows cost, and the ordering rewrite that measured worse

Issue [#318](https://github.com/stew675/llama-cpp-rdna-boosts/issues/318) (reporter `DanoPTT`, Windows 11,
one R9700, dense Qwen3.8-27B UD-Q5_K_XL, `--spec-type draft-mtp` n-max 3, `-np 2`, 262144 ctx, q8_0 KV).
Closed by **r293**: the delivery defaults to the host readback (`LLAMA_MTP_DEV_ROWS` unset/`0`) and keeps
upstream `#31927`'s device path behind `LLAMA_MTP_DEV_ROWS=1`.

## What the reporter measured (Windows/HIP, their box)

Bisected to one line: the MTP target-h-rows-on-device arm that r281 turned on when its base gained
`#31927` (`llama_set_embeddings_nextn_dev(ctx_tgt, true)` + `llama_set_ctx_state_src(ctx_dft, ctx_tgt)`).

| depth | r277 (host rows) | r281 (device rows) | r281 + device path off | device vs r277 |
|---|---|---|---|---|
| 62 857 | 60.27 | 59.38 | 60.15 | **-1.48 %** |
| 68 767 | 48.59 | 47.96 | 48.71 | **-1.30 %** |
| 100 607 | 53.95 | 53.09 | 53.95 | **-1.59 %** |

Identical text and identical draft acceptance (93/101, 85/124, 91/106) in all three arms; a `GGML_CUDA_OP_TIMING`
census shows the kernels unchanged, so the cost is host-side: +0.25-0.30 ms per generated token, about 1 ms per
verify+draft round.  Their hypothesis: every round goes through the `ev_h_nextn_written` /
`ev_h_nextn_read` pair plus the per-round host row copies, and on Windows `hipEventSynchronize` and a stream
wait carry a fixed latency larger than the host readback they replace.

## What this box measured (Linux, gfx1201, one R9700, 10.1.0, `-sm layer`, q8_0 KV, `--spec-type draft-mtp`
`--spec-draft-n-max 3`, `--seed 42 --temp 0`, two prompt depths)

| arm | deep (61 525-token prompt, `-n 128`) | shallow (5 246-token prompt, `-n 512`) |
|---|---|---|
| host rows (pre-`#31927`, now the default) | 47.9 / 47.9 / 48.0 t/s | 49.2 / 49.2 / 49.2 t/s |
| device rows (`LLAMA_MTP_DEV_ROWS=1`) | 47.9 / 48.0 / 47.9 t/s | 49.2 / 49.2 / 49.2 t/s |

So the path is **neutral here** (the r281 record already noted the decode half did not reproduce on Linux) and
**negative on Windows**.  Draft acceptance and greedy text are identical in every arm on every model measured
(27B `542 chars sha=d56a982e90c2`, 79/141 accepted; 4B Q8_0 `sha=bbb051e2aeab`, 87/119).

## The ordering rewrite, and why it is not shipped

The reporter's alternative ask ("is there a cheaper ordering for HIP, e.g. skipping the event pair when target
and draft run on the same device and stream") was implemented and measured before being dropped.  Two changes:

1. **The carry row stays on the device.**  Upstream fetches the accepted row of the last verify batch into
   `pending_h` on the host (`llama_get_embeddings_nextn_rows`, a blocking copy behind a blocking event wait) and
   uploads it back to `row n_batch + seq` for the next round (`llama_set_embeddings_nextn_rows`, a second blocking
   copy behind a second blocking wait).  The rewrite copies the row **within the h_nextn device buffer**
   (`ggml_backend_tensor_copy_async` on the context's own backend, ~10 KB), re-records the write event behind the
   copy, and feeds the drafter's first draft step from that row too, so a decode round does no host row traffic
   at all.  Tested: same greedy text, same acceptance (`79/141`, `87/119`).
2. **A stream wait instead of a host wait** for the h_nextn source ordering in `decode()` (`ggml_backend_event_wait`
   on the reader's backend on the source's device, host sync kept when there is no such backend and on Metal,
   where a command buffer waiting for a whole target batch trips the Metal timeout).  This half measured
   **neutral**, on its own and combined.

The complete change as measured is in `device-rows-carried-on-device.patch.diff` (against the r292 tree).

| arm (deep prompt, `-n 128`) | tg t/s |
|---|---|
| host rows (default) | 47.9 / 48.0 / 48.0 |
| device rows + carry row + stream wait | 47.7 / 47.7 / 47.7 |
| device rows + carry row + host wait | 47.7 / 47.6 |
| device rows, upstream host round trip | 47.9 / 48.0 / 47.9 |

**-0.5 % on Linux**, i.e. the round trip it removes was cheaper here than the two tiny device copies that
replace it (each `ggml_backend_tensor_copy_async` also drains the async-input ring and re-records an event, and
the drafter's first step gains a cross-context source copy in the scheduler).  It was not shipped: a change that
measures worse on the only platform we can measure, for a path that is neutral there and behind an opt-in,
buys nothing.  The Windows half of the question stays open — the reporter offered to test a candidate build, and
`LLAMA_MTP_DEV_ROWS=1` is how they A/B it.

## Reproducing

```bash
# deep arm (~85 s per run on a dense 27B; the prompt is not a shipped artefact, it was 170 650 bytes of this
# repo's WORKLOG.md, 61 525 tokens on the Qwen3.8-27B tokenizer)
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/27B/Q5_K_XL/Qwen3.8-27B-UD-Q5_K_XL.gguf -f longprompt.txt \
  -c 65536 -ngl 99 -sm layer -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 \
  -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn        # LLAMA_MTP_DEV_ROWS=1 for the other arm
```

The gate for what was shipped is `scripts/gate-mtp-dev-rows.sh` (default arm = host rows and no h_nextn buffer
at all; opt-in arm = device rows; identical text, identical acceptance, census-checked on a 4B single-head MTP
model in ~20 s).
