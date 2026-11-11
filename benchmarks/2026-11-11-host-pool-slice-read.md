# The host pool's demand read on a `-sm tensor` split: the overlap premise, and the slice that was never needed (2026-11-11, r261)

`TODO` #132 said the pool's pair decode deficit is *the synchronous demand read* (~29 % of a decode step) and
that the fix is the **demand-fill overlap worker** the 1-GPU ~6 % ceiling declined.  This is the measurement that
decided it: the overlap worker cannot pay on this design, and the read itself was pulling **twice the bytes it
needs** on a tensor split.  The shipped change is the byte half (`MOE_HOST_POOL_SLICE_READ`, block 12); the
overlap half is retired with the numbers below and the residue is pointed at `TODO` #141 / the devremap
(`archive/work/host-pool-demand-overlap/`).

Box: `soar` — 3 x R9700 gfx1201 (32.6 GiB each, in-process 0/1/2; the x8 pair is `HIP_VISIBLE_DEVICES=0,2`),
91 GiB RAM, ROCm 10.1.0, `/llm` on `nvme2n1`.  Model: `Qwen3.5-122B-A10B-UD-Q4_K_XL` shard 1, `-ncmoe 48`
(67 110 MiB of host experts, 144 host tensors = 48 layers x gate/up/down, 256 experts x 1 769 472 B each),
`-sm tensor`, `-c 8192 -ctk/-ctv q8_0 -b 2048 -ub 2048 --spec-type none`, `prompts/prose-rdna-boosts.txt`
(5 258 tokens), `--seed 42 --temp 0`, `-n 1000`, `MOE_HOST_POOL_DIO=1` (the disk tier: this box cannot hold the
74 GiB mapping *and* a 54 GiB pool, so a buffered fill re-faults), `-lme 54` = the pool budget (80.5 % of the
host set — the sub-complement band the item was opened on).  Runner: `arm-run.sh` in the campaign directory
(hygiene check, `/proc/diskstats` and `pswpout` accounting, text hashed with `scripts/extract-generated.py`).

## 1. Where the step actually goes (the instrument that decided the item)

A temporary probe (kept as `probe-132.patch` in the campaign directory, **not shipped**) timed the four
host-side pieces of a decode step.  `-lme 54`, pair, `DIO=1`, 300 tokens, 58.6 ms/step (17.06 t/s):

| segment | per call | per token | share of the 58.6 ms step |
|---|---|---|---|
| ids readback + `ggml_backend_synchronize(ids_backend)` | 0.499 ms | 48 calls = **24.0 ms** | this *is* the device's compute: the host waits for the router |
| `moe_cache_update_host` (policy + fills + copies) | 0.229 ms | 145 calls = **33.2 ms** | the device is IDLE for all of it |
| — of which the demand source (`expert_host_src`, decode band) | | 96 calls = **31.7 ms** | 54 % of the step |
| — of which the per-table **bounce** read | 0.531 ms | 44.6 calls = **23.7 ms** | 40 % of the step |
| — of which a pool-slot fill attempt | 0.135 ms | 59.6 calls = 8.0 ms | most fail instantly (the layout is full); the successes are ~0.3 ms |

**The overlap premise is refuted by that split, and the mechanism is in the scheduler, not the pool.**  A
decode graph is 50 splits (one per layer, `sched_reserve: graph (pp bs=2048, tg bs=1): ... splits = 144 / 50`).
For every layer the host copies the split's host-weight inputs *before* dispatching the split's compute, and that
copy begins with `ggml_backend_synchronize(input_backend)` + an ids readback + `ggml_backend_synchronize(ids_backend)`
— a full meta-backend synchronize.  So at the instant the routing (and therefore the miss set) is known, the
device has **nothing in flight**: the "overlap the current ubatch's misses with the previous ubatch's compute"
shape the proposal assumed needs a host that runs ahead of the device, and this path structurally does not have
one.  The 24 ms of device time per token is spent *inside* the readback, and the 33 ms of demand service that
follows is strictly additive.  Nothing was built for it.  (A hook-side worker cannot fix it either: the whole
hook runs under one global `g_mutex`, so any worker would have to do its read outside the lock to run
concurrently, and it would still be reading bytes the device needs *now*.)

## 2. What the demand read is made of

Two candidate byte-level wins, both measured:

**(a) A shared demand ring across the devices — REFUTED.**  The pool is one object per *host tensor* (144, not
288) and is shared by both devices, so a policy-slot fill already serves both; the fallback bounce ring is per
`table_t` = per (layer, role, device).  If the two devices both missed the *same* cold expert, a shared reusable
ring would halve the reads.  Implemented (4 whole experts per host tensor, LRU, `slots + i` in the pool's own
allocation, sharing its per-(entry, device) drain events, 1 188 MiB pinned) and measured: **20 860 ring reads
against 96 hits = 0.5 %**, decode 16.95 against 17.06 t/s (a tie), disk unchanged.  The cold tail is genuinely
cold — an expert that neither tier holds is not needed again within a handful of fills — and the devices do not
duplicate one another.  Dropped, nothing shipped.

**(b) Read only THIS device's slice — SHIPPED, and it is the whole win.**  On a `-sm tensor` split each device
owns a slice of every host expert, and for `gate`/`up` that slice is **contiguous** inside the expert
(`CACHE_GEOM`: `expert_bytes 884736 host_bytes 1769472 src_off 0 host_pitch 0 split_axis 1`, i.e. device 1's
bytes are `884736 .. 1769471`); the consumer copies exactly that run (`copy_slot_from_src` -> `base + src_off`,
1-D).  The fill read the **whole** expert anyway, so every cold expert was read at double size.  The fix reads
`expert_bytes` at `off + src_off` into the entry at `src_off`, which keeps `copy_slot_from_src` untouched and
works for all four consumers (`expert_host_src`'s bounce, `fill_slot_from_host`, the prefill gather and the
staging fill).  A strided slice (`host_pitch > 0`, an axis-0 `down_exps` split) or a fused table (`n_slices > 0`)
keeps the whole-expert read: its slice is not one run, and one sequential read beats `rows` strided ones.

### The A/B (same binary, `MOE_HOST_POOL_SLICE_READ=0` is the old whole-expert read)

`-n 1000`, everything else as above; arms alternate, never parallel:

| shape | arm | decode t/s | prefill t/s | disk MiB/run |
|---|---|---|---|---|
| 2 GPUs x8 pair | slice read (default) | **20.12** | **917.8** | 173 354 |
| | whole expert (`=0`) | 18.98 | 844.9 | 193 221 |
| | `pinned` (control) | 43.55 | 1617.9 | 66 544 |
| 3 GPUs | slice read | **49.33** | **1406.8** | 86 487 |
| | whole expert | 48.96 | 1126.9 | 102 998 |
| 1 GPU | slice read | 20.45 | 580.7 | 87 326 |
| | whole expert | 20.44 | 569.9 | 87 155 |

**+6.0 % decode / +8.6 % prefill / −10 % disk on the pair; +24.8 % prefill at 3 GPUs; inert at 1 GPU** — which
is the prediction the geometry makes (a single device owns the whole expert, so `expert_bytes == host_bytes`
and the two arms are the same read).  3-GPU decode is flat because that shape's arena holds 99.6 % of the set
(`arenaH 0.9961`), so there is almost no demand read to halve.

### Purity

Same-seed greedy text is byte-identical across the arms, the pre-change binary and the prefill-logit A/B:

| arm | text |
|---|---|
| pair, 122B host-experts, `-n 128`: slice read / whole expert / **the r259 binary** | `fa20f8883726` (490 chars) — all three identical |
| dense 4B Q8_0, 1 GPU, `-n 24`: r259 binary / r261 binary | `51daef9ef546` (111 chars) |
| `gate-host-pool.sh` (Flash-Next GSQ, 8 arms) | **42/42 checks passed**, every arm `a99ccbd97ecb` except the known `pinlate` mixture (`TODO` #282) |
| `gate-prefill-logits.sh --record --ab "MOE_HOST_POOL_SLICE_READ=0"` (35B-A3B Q8_0, 3 devices `-sm tensor`, `-ncmoe 48`, `-ub 2048`) | mean KLD **−0.000000**, same-top-p **100.000 %** |

The gate-host-pool and text rows matter more than the throughput ones: this is a **read-geometry** change, so
its failure mode is wrong bytes in a slot, and both gates compare generated text.

## 2b. The cross-role prefetch: built, exact, and still not worth it (measured)

The one *exact* lookahead the design has is **within a layer**: a layer's six host-weight visits are
consecutive and carry the SAME routing —

```
IDS132 call=0 layer=0 role=blk.0.ffn_gate_exps.weight dev=0 ids=[138 85 242 117 196 10 101 206 246 206 61 103 41 196 29 186]
IDS132 call=1 layer=0 role=blk.0.ffn_gate_exps.weight dev=1 ids=[... identical ...]
IDS132 call=2 layer=0 role=blk.0.ffn_up_exps.weight   dev=0 ids=[... identical ...]
...                                                    down dev=1 ids=[... identical ...]
```

so a prefetch issued on the first visit needs no prediction at all.  It was built the simple way (one worker
thread over a small job table, the destination is the sibling's **existing** pinned bounce entry, no new
buffer or tier, no eviction, release-on-consume with a lazy reclaim; first cut kept as
`prefetch-experiment-first-cut.patch` in the campaign, plus the two fixes it needed: free the job slot on
consume/reclaim — without it four slots latch at "done" and the whole run stages 4 reads — and expire a
sibling's unused entry on the next pass).  It ran, and it is **not worth shipping**:

| | count | per token |
|---|---|---|
| prefetch passes (one per layer per token) | 14 496 | 48 |
| sibling tables tested | 72 480 | 240 |
| ids tested for "is this expert in neither tier" | 397 240 | 1 315 |
| … arena-resident | 368 676 (92.8 %) | |
| … pool-resident | 25 009 (6.3 %) | |
| … **neither tier** (the only thing a prefetch can help) | **3 555 (0.9 %)** | **17.6** |
| staged / consumed by their visit / reclaimed unused (best run) | 1 719 / 968 / 748 | 5.7 / 3.2 / 2.5 |

and end to end on the arm: **18.35 / 18.26 t/s (`MOE_HOST_POOL_LAYER_PREFETCH=0`) against 18.63 / 18.56 /
18.32 t/s (on)** — i.e. **±1.5 %, noise**, for ~150 lines of worker, a new lifecycle and a teardown join.

**Why it cannot do better, which is the part worth keeping:** the bound is not the worker, it is the pool.
The demand path tries an *empty policy slot* before it falls back to the bounce (`pool_fill_empty_locked`),
so most of the misses a prefetch would stage are served by a placing fill anyway — that is a **better**
outcome (the expert stays resident) and it costs 0.131 ms against the fallback's 0.449 ms.  Of the misses the
fallback actually sees, each is a first-request latency the consumer must wait for regardless: moving the
read one visit earlier only helps if the *consumer's own* wait disappears, and it does not, because the
consumer is the one that discovers the placement decision.  A prefetch that targeted **pool slots** instead
would attack the right thing but needs a "filling" slot state, i.e. exactly the background-fill/eviction
machine `TODO` #87 removed and r113 measured as a 2x loss.

The composition this leaves for anyone tempted to try again (shipping tree, 300 tokens, 54.0 ms/step):

| term | calls/token | wall/token | share of the step |
|---|---|---|---|
| ids readback + meta synchronize (the device's compute) | 48 | 24.0 ms | the device is *working* here |
| demand service: fallback reads | 44.5 @ 0.449 ms | **20.0 ms** | 37 % |
| demand service: pool-slot fill attempts | 58 @ 0.131 ms | 7.6 ms | 14 % |
| whole demand pass (`expert_host_src`, decode band) | 95 | **27.6 ms** | 51 % |

## 3. Why an overlap worker cannot be the answer even in principle (the queue-depth probe)

If the read is bandwidth-bound, the only remaining lever is more in-flight reads.  64 random 1 769 472-byte
`O_DIRECT` reads of the real shard from `N` threads (`readprobe.py`, campaign directory):

| depth | wall | effective |
|---|---|---|
| 1 | 0.032 s | 3.3 GiB/s (0.50 ms/read — the measured 0.531 ms bounce) |
| 2 | 0.021 s | 4.9 GiB/s |
| 4 | 0.018 s | 5.8 GiB/s |
| 8 | 0.018 s | 6.0 GiB/s |

So ~1.8x is on the table at depth 4 — but a depth needs *several misses known at once*, and section 1 is why
there are none: the host learns one layer's routing at a time, and it is the only reader.  The one exact
lookahead that exists is **within a layer** (gate/up/down share the routing ids, and `prev_ids_tensor` already
relies on it), which bounds the depth at ~2 (the measured demand set is 0.5-1 miss per table, ~2 per layer).
That is a +14-20 % estimate on this arm for a cross-table prefetch plus a lock-free fill worker — **and it
was built and measured at ±1.5 %, see §2b** — so it is recorded, not shipped: the hook's global mutex and the per-table staging make it a
scheduler change, not a pool one, and `TODO` #141 (a sub-complement budget's static bound) and the devremap
(a cold source the device can resolve) both attack the same bytes with less machinery.

## 4. Reproduce

```bash
campaign=archive/work/host-pool-demand-overlap
Out=/tmp/132 OUT=$Out NGEN=1000 bash $campaign/ab-261111.sh     # the whole matrix (sequential)
# one arm by hand:
OUT=$Out NGEN=1000 bash $campaign/arm-run.sh pair-slice MOE_HOST_POOL_DIO=1 ::: -lme 54
OUT=$Out NGEN=1000 bash $campaign/arm-run.sh pair-whole MOE_HOST_POOL_DIO=1 MOE_HOST_POOL_SLICE_READ=0 ::: -lme 54
```

Raw logs and `summary.tsv` per verdict: `archive/work/host-pool-demand-overlap/runs/`.
