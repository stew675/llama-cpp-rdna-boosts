# state-stage-per-device — give the host-read destination to the device that fills it (r369's guarantee, other direction)

**Status: OPEN — a plan, NOT started; nothing in `patches/`, no block, no knob, no gate change.**  Opened
2026-11-28 on the maintainer's instruction, when contributor **PR #375** was assessed and **held** (not rejected):
its fix is measured and correct, but it patches the *copy* where the *destination* is the defect.  The submission
and its assessment are kept here (`held-pr375/`, [`PR375-ASSESSMENT.md`](PR375-ASSESSMENT.md)); if this campaign
fails, that patch is the fallback (§7).

**One sentence.**  A device→host row-block read is strided only because the destination is ONE device's pinned
buffer that every device of a `-sm tensor` split reads into; give each device its own stage — the guarantee r369
gave the H2D staging slots — and every copy becomes a tight 1-D D2H whose destination the copying device owns, which
retires the 2-D D2H, the peer grant, the owner query, the per-row fallback and the host callback at once.

## 0. Read first, and what this is NOT

* [`../tensor-split-stage-overlap/README.md`](../tensor-split-stage-overlap/README.md) — r369's **fill-ahead /
  in-place consume** follow-up (`TODO` #388).  Same subsystem, **opposite direction**: that campaign makes the
  H2D *input* staging overlap the consumer; this one makes the D2H *state* read destination device-local.
  They do not share a fix and neither blocks the other.
* `INVARIANTS.md` **r364** (the row-block read must not be a 2-D copy on ROCm) and **r369** (the transport
  pipeline: derived region, unit = one split input, depth = a residency decision, exposure = the selector).
  r369 is the *pattern* this campaign follows; r364 is the *symptom* it explains.
* `INVARIANTS.md` **r379-wip** — the facts measured on r378 that this campaign must not re-derive.
* `../../patches/README.md` block 12 (the batched state path) and block 14 (the streamed KV, which owns
  `ggml_cuda_rows_to_host` and the region's copy helpers).

**This is not an "optimisation" campaign.**  Nothing here is about throughput: r378's state-save *bytes* already
move at 3.1-7.5 GiB/s.  It is about a copy **shape** that only exists because of where its destination lives, and
about a class of hard abort (a refused peer grant, r364) that shape keeps alive.  A rate arm is not the oracle;
text identity and the gate that owns the state path are (§6).

## 1. The problem, as measured on r378 (2026-11-28)

Flash-Next GSQ-RCO IQ3_XXS, the x8 pair (`HIP_VISIBLE_DEVICES=0,2`), `-sm tensor -c 65536 --cache-ram 8192`, four
slots, unified KV, one request per arm, `LLAMA_STATE_TIMING=1`.  "short" = a 4 290-token prompt, "long" = 17 189.

| arm | short saves (226.5 MiB) | short "A again" wall | long saves (566 MiB) |
|---|---|---|---|
| r378 base | **428.7 / 505.1 / 375.8 ms** | 1 092.9 ms | 84.6 / 88.3 / 77.4 ms |
| PR #375 `0001` | 82.0 / 96.8 / 39.3 ms | 688.2 ms | 84.0 / 93.6 / 77.2 ms |
| PR #375 `0001`+`0002` | 86.6 / 98.2 / 32.8 ms | 688.2 ms | **147.6 / 90.6 / 102.4 ms** |
| `0001`, gather only (`ROWS_OWNER_2D=0`) | 89.8 / 101.0 / 34.6 ms | — | — |
| `0001`, kill switch (`ROWS_ASYNC_GATHER=0`) | 431.8 / 493.9 / 374.4 ms | 1 079.5 ms | — |

The interesting column is the *inside* of one save, because it names the mechanism:

| read in the same save | pieces | enqueue | copy+wait |
|---|---|---|---|
| the recurrent/GDN state (mirrored, **tight** destination) | 73 / 112.6 MiB | **0.4 ms** | 23.2 ms (4.7 GiB/s) |
| the column-split KV (**strided** into device 0's stage) | 109 / 226.5 MiB | **313.2 ms** | 71.4 ms (3.1 GiB/s) |

Same function, same stream, same run.  The tight read is one 1-D `cudaMemcpyAsync` per piece
(`ggml-cuda.cu:3926`); the strided one falls to the per-row loop (`ggml-cuda.cu:3956`) — one queued 1-D copy per
token row per (tensor, device) piece, of the order of 10^5 submissions for this size (4 290 rows against ~109
pieces), i.e. ~0.7 µs of pure submission cost each.  **The enqueue cost, not the link, is the wall time**, and the
stride exists only because each device's slice of a row must land at `nb[1]` inside device 0's buffer.

## 2. Why the destination is the defect, not the copy

Three layers were each "fixed" in turn, and each fix is a different way of paying for the same interleave:

1. **r362** — issue one 2-D D2H.  Legal only from the buffer's *owner*: on another device the driver refuses the
   peer grant (r364's measured `invalid argument`, the abort behind issue #124/#313).
2. **r364** — never 2-D: one 1-D copy per row (the delivery's current async form), correct everywhere and 25x
   slower on a column split because of the submission cost above.
3. **PR #375** — 2-D on the owner (gated by `hipPointerGetAttributes`, refused → fall back), and *gather on the
   device* elsewhere: a 2-D D2D into a per-(device, stream) staging pair, a chunked 1-D D2H into the device's own
   pinned bounce, then a host scatter **from a `cudaLaunchHostFunc` callback** so nothing synchronizes.  Measured:
   a real 4-6x win on r378 (§1), and 0002's rounds a *regression* here.

All three keep the destination as it is: `llama_context::state_stage()` and `output_host_buft()` both allocate from
**simple device 0's** pinned buffer type (`llama-context.cpp:5801`, `:4282`; the per-device types exist —
`ggml-cuda.cu:6128`, `ggml_backend_cuda_device_get_host_buffer_type` — but the context never asks for them), so
every device of the split reads into one device's allocation.  That is the whole reason the copy is strided, that
is the whole reason it can be a 2-D D2H, and that is what makes a grant necessary.  The delivery has been paying
the interleave inside the copy; the campaign moves it to a place where it is free.

**And it is not Python-level cleverness**: r369 solved exactly this shape for fills (per-device slots in a derived
region, the unit and depth owned by the layer that knows them, 0 = gather).  A refused destination does not occur
there any more because no copy is ever given a destination another device owns.

## 3. The design

**3.1 The destination.**  `state_stage()` becomes a *set* of pinned regions, one per simple device of the split
(`ggml_backend_dev_host_buffer_type(ggml_backend_meta_dev_simple_dev(output_dev, j))`), each grown once and kept,
**packed**: device *j*'s slice of each window sits contiguously in region *j*.  `LLAMA_STATE_ASYNC_MAX_MIB` keeps its
meaning as the cap on the **total** (`sum_j` of a window's shares), so each region is ≤ that cap and the pinned
footprint is unchanged in the limit (sum of shares = the state), not `N × state`.

**3.2 The read.**  Each device reads its own slice, tight: `stride_tensor == stride_data == size` → the branch that
already exists at `ggml-cuda.cu:3926`, one `cudaMemcpyAsync` per window per device, queued, never synchronizing.
No 2-D D2H anywhere, so:

* no peer grant, so **the r364 refusal class cannot occur by construction** (not "is caught and retried");
* no `hipPointerGetAttributes`, no owner arm, no per-device "turn this off" latch;
* no `cudaLaunchHostFunc`, no per-stream staging pair (the source of PR #375's only novel API assumption), no
  16 MiB VRAM + 16 MiB pinned per (device, stream);
* the ASYNC form finally needs no host work at all in the queue path — it queues every piece and lets the caller's
  one backend synchronize do its job (which is what `llama_io_write_host::flush_async`, `llama-context.cpp:5083`,
  was built for).

**3.3 The splice.**  After the caller's existing synchronize, the per-device packed slices are interleaved into the
caller's buffer by **host** work: one `llama_state_copy_piece` per (window, device) — the same list, the same
`llama_state_copy_pieces` machinery and the same `LLAMA_STATE_COPY_THREADS` streaming copy that already performs
this exact copy today (stage → caller), only sourced from N packed regions instead of one interleaved one.  Host
copy count is unchanged; what changes is that the *stride* is now on the source side of a `memcpy` instead of inside
a DMA descriptor.

**3.4 The plan (the only genuinely new surface).**  The frontend cannot compute the splice: the split geometry is
the Meta backend's (`ggml-backend-meta.cpp:2858`'s axis-0 branch already computes exactly `nbytes` per device and
`simple_offsets[j] + row_start*simple_tensor->nb[1]`).  So the Meta backend exposes its plan:

```
// one call per window; out_pieces[j] = { dst_off, dst_size, src_off } for device j (0 size = not on that device)
bool get_tensor_split_plan(tensor, offset, size, std::vector<split_piece> & out_pieces);
```

Appended to the iface per the append rule, with a default ("the whole window on device 0") so every non-meta backend
keeps today's single-piece behaviour.  `llama_io_write_host` then builds the pack offsets from it, exactly as it
already builds `pcs` from `winfos`.

**3.5 The mirror.**  The restore direction gets the same shape: each device's packed slice is one tight 1-D H2D
(`swap_blocks`-free, no `set_tensor_split` interleave).  It removes that side's 2-D form too; it is a smaller win
(the current 2-D H2D is not pathological) and it is included because leaving one direction asymmetric is how the
next reader gets the rule wrong.

**3.6 What it retires.**  For these destinations, `ggml_cuda_rows_to_host`'s staged (sync) and gather (async) arms
both become unreachable, and PR #375's 116 lines are not needed at all.  The staged arm is still reached by a
pageable caller (`ggml_backend_tensor_get_2d` into a pageable buffer), so it is kept; the async row loop stays as the
fallback for a non-tight destination.  **Do not delete a form until the census says nothing reaches it** (a
`state stage: pieces=<n> tight=<n>` line, §6).

**3.7 The same shape has two more consumers, and they are in scope.**  `state_stage()` is not the only buffer
allocated from simple device 0's pinned type: `output_host_buft()` is, and so is `buf_layer_inp` — which is why the
split `result_output` read is the copy that actually aborted in r364 (`2-D read of 'result_output' failed (invalid
argument)`, 5 rows, device 1, issue #124/#313).  Both get the same treatment from the same plan query, and each is
the *same* change with a different consumer:

* the **output** read ends in llama.cpp's contiguous logits/embeddings array, so its splice is per (row, device) —
  tiny pieces, but it is the arm that aborts today, and it must not be left as the only 2-D D2H in the tree;
* `buf_layer_inp` (the layer-input buffer, `llama-context.cpp:4258`) is one allocation per context and is read the
  same way.

So "the `result_output` abort goes away for free" means *free in design* — the guarantee is the same and the plan is
already computed — but it is a **named step in section 5**, not a side effect: a per-device destination for the
output buffer is owed its own arm (`-sm tensor`, several rows, `LLAMA_OUTPUT_PINNED` as the A/B) before it is claimed.

## 4. What must not break (the invariants this campaign is judged against)

| fact | why it binds here |
|---|---|
| r364: a row-block D2H is never 2-D on ROCm | this campaign makes it *structurally* impossible, and must not re-introduce a 2-D arm for the "few rows" case |
| r96: the stage is grow-only, one allocation per context, ≤ `LLAMA_STATE_ASYNC_MAX_MIB` | becomes N grow-only regions; the cap must stay a **total** cap or pinned memory doubles per device |
| r151/r152: one event per (backend, group), recorded *after* the group's copies, synchronized before its copy-out; a worker must never cross a group boundary | per-device pieces multiply the pieces in a group: the event count per (backend, group) is unchanged, but the *piece* list grows and the group split must still be size-balanced (`llama_state_pipe_groups`) |
| the state buffer layout is the CALLER's (a state blob is portable, and `--slot-save-path` writes it to a file) | the splice must produce byte-identical bytes to today, or a restored state moves the text |
| `-sm layer`, `-nkvo` and a single device must be untouched | outside a meta split there is one device, one region, one piece: the code must take today's path exactly |
| the region-backed KV's 2-D directions are 1-D only (issue #313, r204) | its read helper (`ggml-cuda.cu:3964`) already syncs and stages; a per-device destination must not reach it as a 2-D |

## 5. The plan, in order (each step gated before the next)

0. **Census first, no behaviour change.**  Add a `state stage:` line to `flush_async` (pieces, windows, MiB, how
   many were tight, per-device shares) and to `ggml_cuda_rows_to_host` (which arm each read took).  Run the §1 arms
   and the gate on r378.  This is the non-vacuity instrument for every later step; without it a "no change" arm is
   indistinguishable from a spliced one.
1. **The plan query** (§3.4) on the Meta backend, with the default for other backends, plus a unit arm in
   `tests/test-llama-context` or a small harness: for each state tensor shape of a 2/3-device split, the plan's
   `sum_j dst_size == size` and its offsets are inside their region.  Nothing else proceeds until that holds.
2. **The per-device stage** (§3.1) as an *opt-in* (`LLAMA_STATE_STAGE_PER_DEVICE=1`) with the single-region path
   untouched beside it, so the A/B exists from the first run.
3. **The tight read** (§3.2) + **the host splice** (§3.3), still opt-in; text must be byte-identical to the
   single-stage path on every arm (a splice bug is a torn state, i.e. a silent wrong answer, so the oracle is the
   existing synchronous arm, never "the text looks fine").
4. **The restore mirror** (§3.5), same arm structure.
5. **The census must prove the reach**: on the two-device shape the `tight=` count must be > 0 and the async
   per-row loop must read 0 for these destinations.  A step whose census reads 0 is vacuous — the r221 rule.
6. **The output and layer-input buffers** (§3.7) on the same plan: the split `result_output` read (the r364 abort)
   and `buf_layer_inp` become per-device tight reads, with `LLAMA_OUTPUT_PINNED=0` as the A/B and a multi-row
   `-sm tensor` arm.  This is the step that makes the tree's *only* remaining 2-D D2H a pageable-caller fallback.
7. **Flip the default, delete the switch**, and only then decide whether `ggml_cuda_rows_to_host`'s staged/gather
   arms still have a caller.

## 6. The gate ladder (what promotion owes)

Owning gate: **`scripts/gate-state-pipe.sh` (MANIFESTS gate 10)**, which is also the campaign's *performance* oracle
for this class — it compares the batched save+load against the synchronous arm inside one run, which is how the
r286 cliff was caught.  On r378 base it reads **pipe 64.7 ms vs sync 42.2 ms (1.53x)**; with PR #375's 0001 it reads
**18.8 ms (0.45x)**, and with `KV_WINDOW=4096` **23.4 ms (0.54x)**.  A per-device stage must land at least as good
as the 0001 reference, on both arms:

* `scripts/gate-state-pipe.sh` — all four arms, **with and without `KV_WINDOW`** (the windowed region's state copies
  are issue #313's, and the gate doc requires both);
* `scripts/gate-ckpt-reuse.sh` — the context checkpoint path reads the same stage;
* `scripts/gate-spec-ckpt-device.sh` if a draft context is involved;
* `tests/test-llama-context` — the 15 state cases, 1 GPU and 2 GPU `-sm tensor`, PASS/FAIL set identical to r378
  (on this box that set is `rf` + `rollback` OOM — compare the sets, not the count alone);
* `scripts/gate-meta-seg-drain.sh` — the Meta backend's other copy direction, which shares the simple-backend
  plumbing this changes;
* `scripts/gate-prefill-logits.sh` (release gate, unchanged text expected) and same-seed greedy coherence on the
  usual four arms;
* `scripts/check-gpu-faults.sh` on every arm's log, and `test-backend-ops` is *not* an oracle here (no kernel);
* the standing set: `validate-set.sh`, `check-rung-file-sets.py` (blocks 12 + the Meta backend are both touched),
  `check-shlib-symbols.sh` (an appended iface member is exactly gate 40's class), `check-windows-*`,
  `check-gen-docs.sh` (only if an option moves — an env switch does not).

**A 3-device arm is owed** (0,1,2): the x8 pair hides a per-device asymmetry, and the delivery validates gfx1201 at
both ranks.  `-sm layer` and 1 GPU are the inertness arms.

## 7. The fallback (maintainer's decision, 2026-11-28)

If this campaign fails or is abandoned, the fallback is **`held-pr375/0001`** — PR #375's first patch — as a
**block-14 amendment**, with two integration changes:

1. **the owner-2D arm is deleted** (`GGML_CUDA_ROWS_OWNER_2D` does not ship): gather-only keeps the r364 invariant
   literally true (no 2-D D2H anywhere), and the measured cost on this box is **nil** — 34.6-101.0 ms against
   39.3-96.8 ms, inside noise (§1);
2. **the scatter's error path declines instead of aborting**: a failed `cudaLaunchHostFunc`/`cudaMemcpyAsync` marks
   the staging pair abandoned and falls through to the per-row copies for the rest of the block (nothing has reached
   the caller's buffer at that point, so it is exact).  `check()` here would turn a state save into an abort, which
   is the class r364 exists to remove.

**`held-pr375/0002` (the state-copy rounds) does not ship in any form**: it re-opens the banded-copy half that was
*declined* at r173 (`archive/work/prompt-cache-save-bands/ASSESSMENT.md`) and it measures **slower** here
(84.6/88.3/77.4 → 147.6/90.6/102.4 ms on a 566 MiB state; the plan only cuts a window above the cap on a flat
tensor, so what it really adds is "pack the windows into rounds", i.e. the batched path's extra pinned-stage copy
on top of an already-fast synchronous path).  Its motivating pathology ("a 2 GB agent save takes 17-20 s") is the
per-row async read, which `0001` removes on its own — in that build the rounds are what *put* the large states on
the batched path in the first place.

## 8. How to run it

The probe harness used for §1 is in [`probe/`](probe/README.md) (`run-arm.sh` = the Flash-Next A/B/A/B, `save-probe.sh`
= one save at a chosen size, `run-arm2.sh` = the gate-shaped 35B-A3B server).  One arm:

```sh
# r378 base bin first (scratch-tree.sh keep), then the arm
./scripts/scratch-tree.sh keep r378base
./scripts/scratch-tree.sh arm /path/to/patch            # applies on release.json's tree, rebuilds
bash probe/run-arm.sh short "$HOME/llama-scratch/build-rocm/bin" 15500 LLAMA_STATE_TIMING=1
bash probe/run-arm.sh base-short "$HOME/llama-scratch/build-rocm/bin-r378base" 15500 LLAMA_STATE_TIMING=1
```

`nch` is characters of `$SRC` (see [`probe/README.md`](probe/README.md) for generating it: 6 concatenated `src/*.cpp`):
15 500 ≈ 4 290 tokens, 60 000 ≈ 17 189, 140 000 ≈ 41 678.  Read `prompt cache save took` (the pure copy) and `state get:` (the enqueue/copy split); do not
compare `wall` across processes.  The extracted arms of 2026-11-28 are in
[`logs/arms-2026-11-28.txt`](logs/arms-2026-11-28.txt).

## 9. Numbers of record (do not re-derive)

2026-11-28, gfx1201, ROCm 10.2, x8 pair, Flash-Next GSQ-RCO IQ3_XXS, `-sm tensor -c 65536`:

* r378's column-split state read enqueues at **0.6-0.7 µs per row copy** (313 ms / 226.5 MiB in 109 pieces);
  the *same* save's mirrored state read enqueues **0.4 ms for 112.6 MiB** because it is tight.
* `gate-state-pipe.sh` on r378: pipe **64.7 ms (1.53x sync)**; with `0001`: **18.8 ms (0.45x)**;
  with `0001` + `KV_WINDOW=4096`: **23.4 ms (0.54x)**.  Both PASS on the arm; r378 passes at 1.53x, i.e. the
  gate's 3x bound does not see this class — the *ratio* is the number to keep, not the pass.
* 0002 is a **regression** on this box at 566 MiB (above) and does not fix anything below the cap that 0001 does
  not fix better.
* The r369 pattern does **not** apply to this D2H path as-is: r369's slots are VRAM staging for a compute consumer,
  and a D2H's consumer is the host, so a ring would still need a drain point (see `PR375-ASSESSMENT.md` §5).  The
  part of r369 that transfers is the **destination ownership**, which is this campaign.
