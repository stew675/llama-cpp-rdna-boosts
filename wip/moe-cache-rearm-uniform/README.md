# moe-cache-rearm-uniform: re-arm a stood-down layer's other tables with it, so the layer stays slot-uniform

Three format-patches on top of `v16-a55e952b8-r37`, applied with `git am`, all in
`ggml/src/ggml-cuda/moe-expert-cache.cu` (34 lines).  0001 is the original re-arm fix; 0002 and 0003 (added
later) are the cause of the remaining tight-headroom output difference described below.  Each stands alone.

## What

When the work pool takes slab chunks, `moe_cache_evict_slab_range` drops only the tables whose storage lay in the
taken range, and `moe_cache_rearm` re-allocates just those.  So a layer can come back with one role re-sized and the
others not.  Sizing already refuses such a layer ("the shared remap would name different experts per role"), and
`moe_cache_redirect_fused` reads the up table's remap for both the gate and up lanes.

The patch makes `moe_cache_rearm` add every allocated table of a stood-down layer to the set it re-arms, so the
existing attempt loop sizes them to one target.

## Seen

A local build (r34 plus a few patches of ours), `-sm layer -ts 59,41`, host experts (`--n-cpu-moe 48`): the
production warm-up's 17.8k-token prefill made the work pool take 302 MiB of slab chunks; `blk.47.ffn_up_exps` was the
one table in that range and came back at 193 slots while `ffn_gate_exps` / `ffn_down_exps` kept 502.  Greedy output
then changed on every request (five different shas, against a stable `9f826c79de59` before the warm-up); with
admissions frozen it settled on a stable but different sha.  `MOE_EXPERT_CACHE_VALIDATE=2` reported every table
consistent.  With the patch: `9f826c79de59` on every request before and after the warm-up (log: `re-armed 3
stood-down expert-cache tables`).

## Stock r37 repro

`-sm tensor --n-cpu-moe 48` (pinned host experts), `-ub 6144 -b 6144 -c 204800`, greedy 400-token request before and
after the 17.8k-token warm-up prompt; reference sha `45e68f1bdfeb` (also the all-VRAM sha).

| build | `GGML_CUDA_SLAB_HEADROOM_MIB` | tables re-armed | greedy after the warm-up |
|---|---|---|---|
| stock r37 | 4096 | 9 | `45e68f1bdfeb`, `45e68f1bdfeb` |
| stock r37 | 3072 | 11 | `3b99183e4a92`, then `02c7fce840cd` |
| r37 + patch | 4096 | 18 | `45e68f1bdfeb`, `45e68f1bdfeb` |
| r37 + patch | 3072 | 18 | stable but not the reference (`992a1e108d05` x2; another run `8237a68abb22` x2) |

At 2048 both builds abort with OOM in the warm-up prefill.

## 0002 and 0003: the rest of the tight-headroom difference

With 0001 alone, 3072 MiB headroom still gave a wrong greedy output after the warm-up (stable within a run,
different per run), and - found later - the second request after start differed too (`44c0b3703427`, stock as
well).  A debug check (not included) that compares every resident slot with the host master, the device
`expert -> slot` map against the host slot map in both directions, and gate vs up slot order found two causes:

- **0002, pipelined used-list readback across a re-allocation.**  `alloc_table_locked` re-allocates
  `used_dev` / `used_host` / `used_host2`, but `used_pending` / `used_toggle` carried over, so the first
  `moe_cache_promote_host` after a re-arm read a buffer no readback had written (zeros) and admitted expert 0 into
  slot 0 - in some of a layer's tables but not the others.  Every later admission landed one slot further there:
  e.g. layer 45 on device 1, gate and up holding the same 502 experts with ~250 in different slots (gate slot 0 =
  expert 0, up slot 0 = expert 303).  The fused gate+up redirect reads up's remap for both lanes, so the gate lane
  read the wrong expert for those.  Fix: start the readback pipeline over on a (re)allocation, `used_dev` = -1.
- **0003, the prompt-routing seed did not refresh `slot_dev`.**  `apply_prefill_seed_rank_locked` evicts and places
  residents in a live table but neither uploads the device map nor marks it dirty, so an evicted expert kept pointing
  at its old slot, now holding another expert (layer 42: expert 501 -> slot 177, which held 499, in gate, up and down
  on both devices) until some later admission refreshed the map.  Fix: upload it at the end of the seed, as the
  promotion does.

The 600 MiB `cudaMalloc` failure at 3072 (the meta backend's tmp buffer for the butterfly all-reduce, allocation
not checked) is not part of this output difference, but it is not harmless either: if the internal all-reduce
times out ("peer arrival not observed") after it, the butterfly fallback hits `GGML_ASSERT(buffer)`.  We only saw
that when the debug check stalled one GPU for seconds.  Not addressed here.

## Validation (r37 + 0001-0003, no debug code; 2 greedy requests before and 2 after the warm-up, one run each)

| config (pinned host experts) | before | after the warm-up | other |
|---|---|---|---|
| `-sm tensor -ub 6144`, headroom 3072 | `45e68f1bdfeb` x2 | `45e68f1bdfeb` x2 | stock: `44c0b3703427` 2nd request, then drifting |
| `-sm tensor -ub 6144`, headroom 4096 | `45e68f1bdfeb` x2 | `45e68f1bdfeb` x2 | |
| `-sm tensor -ub 6144`, default headroom | `45e68f1bdfeb` x2 | `45e68f1bdfeb` x2 | |
| `-sm tensor -ub 2048`, default | `45e68f1bdfeb` x2 | `45e68f1bdfeb` x2 | 900-building 36k recall OK; prefill 1.8k / 37k / 155k 1988 / 2448-2499 / 2008 t/s |
| `-sm layer -ts 59,41 -ub 2048`, default | `9f826c79de59` x2 | `9f826c79de59` x2 | 900-building recall OK; prefill 1737 / 1962-1989 / 1648 t/s |

0 GPU faults in every run.

In the earlier `-ts 59,41` / `-ts 50,50` / default `-ub 2048` runs on stock r37 the eviction took gate and up of a
layer together (re-armed to one size) and the output stayed correct; `ffn_down_exps` of that layer stayed at 509
slots while gate/up went to 10, so down apparently does not have to match.
