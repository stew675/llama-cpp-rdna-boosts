# 2026-11-09 -- GitHub issue #290: an hc base copy per layer stayed live to the end of a `-sm layer` prefill graph

Reproduction, fix and gates for release `v16-a691b2345-r240` (fork point `a691b2345`, 16 blocks).  Reported
by **@briansp2020** in [issue #290](https://github.com/stew675/llama-cpp-rdna-boosts/issues/290) against stock
`r236` (with stock `r235` as the passing control), with a one-line diff, a mechanism analysis that held up on
every point, and a follow-up snapshot that identified the *remaining* half exactly.

Box: 3x R9700 (gfx1201, 32 GiB each; the driver reports `34208743424` B), Ryzen 9 9950X3D, 91 GiB, ROCm
10.1.0-gfx120X, `~/bin/build-llama-rocm` (ccache), `HIP_VISIBLE_DEVICES=0,1,2` on every arm.
Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (`qwen4exp`, BF16 hc-down -> the FUSED mixer class).
**Before** = the shipped `build-rocm-r239ship` (tree `7339c1bc7`) and, where the pool census is needed, a build
of the same tree carrying the census only ("pre-fix" below: the canonical caches, the pre-r240 ordering in
`hyperconn.cu`); **after** = the r240 tree.

## The defect

The `hc_combine_norm` [repeat] matcher (`hyperconn.cu`) fuses `repeat -> mul -> add -> rms_norm -> mul(gamma)`
into one kernel.  The narrow base (`block_out`, `[n_embd,1,T]` F32) is not pinned -- it dies at the repeat --
so when `add` or `mulg` reuses its buffer the matcher reads the base from a **pool copy** rather than falling
back to the unfused chain.  r236 registered that copy on the device `pool_stack` **before** the window's
`set_bf16*()` calls, which push an mmb activation block and the BF16 src1 block.  The pool is LIFO -- a killed
block is freed only when it reaches the top -- so the copy sat under them and stayed live until the **end of
the graph**.  Under `-sm layer` one graph covers every layer of a device, i.e. one copy per layer.

The reporter's follow-up shows the other half of the same class: the two per-graph activation caches
**leapfrog**.  The BF16 src1 cache holds one block and the MMB activation cache `cache_max` (4 on RDNA4), and
each replaced its previous block only *after* the other had pushed a newer one above it, so no killed block
ever reached the top either -- their snapshot is `abababab…A` (41 dead, 1 live on device 0).  Both chains were
freed only at `pool_stack_release_all`.

`GGML_CUDA_POOL_STACK_CENSUS=1` (new, r240) is the instrument.  Once per graph per device, at the end of the
graph, it prints `live=<entries> dead=<entries>/<MiB> bytes=<MiB> | hc=<live>/<MiB> made=<copies made>
mmb=<live>/<MiB> bf16=<live>/<MiB>`: the per-kind counts are LIVE entries, `dead` is the killed-but-blocked
pile, and `bytes` is what the graph really still holds.  The worst prefill graph of a run:

```
# Flash-Next GSQ-RCO-IQ3_XXS, 3 devices, -sm layer -ts 60,20,20 -c 8192 -ub 2048 -fa on,
# -n 0 -f prompts/prose-rdna-boosts.txt -lv 2       (worst graph by retained bytes)
pre-fix  dev0: live=36 dead=59/521.8  bytes=1205.8 | hc=31/620.0  made=31  mmb=4/54.0   bf16=1/10.0
         dev1: live=15 dead=15/115.0  bytes= 365.0 | hc=10/200.0  made=10  mmb=4/40.0   bf16=1/10.0
         dev2: live=12 dead=12/ 93.8  bytes= 283.8 | hc= 7/140.0  made= 7  mmb=4/40.0   bf16=1/10.0
         TOTAL retained at graph end: 1854.6 MiB
r240     dev0: live= 5 dead= 1/ 24.0  bytes= 132.0 | hc= 0/  0.0  made=31  mmb=4/68.0   bf16=1/40.0
         dev1: live= 5 dead= 0/  0.0  bytes=  80.0 | hc= 0/  0.0  made=10  mmb=4/40.0   bf16=1/40.0
         dev2: live= 5 dead= 0/  0.0  bytes=  80.0 | hc= 0/  0.0  made= 7  mmb=4/40.0   bf16=1/40.0
         TOTAL retained at graph end:  292.0 MiB
```

`made=31` is the vacuity control: the copies are still **made** (the fusion still runs, the copy is still the
kernel's base), they are just released.  The copy count equals the layers that device carries (48 layers in a
3+1-of-10 split at `-ts 60,20,20`) -- the reporter's "the stack holds 26 live entries on device 0 and 19 on
device 1 ... which equals the layers per device".  The single dead 24 MiB block left on device 0 is one MMB
cache block that was evicted *and* too small for the next entry (a growth reallocation, not churn).

## What shipped (three changes, two blocks)

1. **The copy is the last pool allocation of the window, and it is killed right after the launch**
   (`hyperconn.cu`, block 12).  `set_bf16*()` only *reads* `block_out`, so the copy is taken from the same,
   still-intact base *after* those pushes; it is then the newest block on the stack and the kill frees it
   instead of deferring it.  `blk_in_bf16` (`LLAMA_HC_BLK16`, default off) is re-pointed at the copy so an
   armed BF16 block read follows the same bytes.  This is the reporter's own fix, made to actually free.
2. **The BF16 src1 cache reuses its block** (`common.cuh`, block 10): `cap >= n` on the same stream re-keys
   the existing block instead of killing it and allocating a new one.  `cap` is a capacity field independent
   of the key's `n`, because the converted tensors of a graph are a handful of sizes repeated per layer
   (`hc_mixed-N`, the fused `xn`).
3. **The MMB activation cache re-keys its evicted block** (`mmb.cu`, block 10) instead of killing it: the
   oldest entry is dead the moment its key goes, so when it is big enough for the new entry it becomes the
   newest entry (rotated to the back, so the LRU order is unchanged); it is only killed when the new request
   does not fit.  Together with (2) this removes the leapfrog the reporter's snapshot shows.

`GGML_CUDA_POOL_STACK_CENSUS` (a diagnostic, `common.cuh` + the graph-end hook in `ggml-cuda.cu`) is the
fourth piece: the pool-stack entries now carry a byte size and a kind tag so the census can report the split.

## Effect on VRAM (the point of the report)

Peak VRAM per card, `mem_info_vram_used` polled at 200 ms, interleaved fresh processes, the 5246-token prose
prefill (`-n 0`):

| arm (Flash-Next GSQ-RCO-IQ3_XXS, `-sm layer -ts 60,20,20`) | shipped r239 | r240 | delta |
|---|---|---|---|
| `-c 8192`, peaks card1/2/3 | 28522 / 13562 / 11188 MiB | 27448 / 13278 / 10994 MiB | **-1074 / -284 / -194 MiB** |
| `-c 262144 -ctk q8_0 -ctv q8_0`, peaks | 31600 / 18500 / 14717 MiB | 30526 / 18216 / 14523 MiB | **-1074 / -284 / -194 MiB** |
| `-c 262144`, headroom on the tightest card (32624 MiB total) | **1024 MiB** | **2098 MiB** | +1074 MiB |

The `-c 262144` row is the reporter's situation: on their `-ts 59,41` shape device 0 sat at 29728 MiB at load
and their 259.6k-token prompt tipped it over.  Here the shipped build comes within **1 GiB** of the card and
the fix doubles that headroom.  Prefill is flat (1782.0 / 1783.2 t/s shipped vs 1788.9 / 1779.7 t/s fixed, two
interleaved rounds), and the per-graph retention drops 1854.6 -> 292.0 MiB (with the BF16 cache off as well:
172.0 MiB) -- what remains is the MMB cache's own live working set plus one BF16 block, which is the cache
working as designed.

## Purity

Same-seed greedy text, `-n 32`, `--seed 42 --temp 0`, `-sm layer -ts 60,20,20`:

| arm | chars | sha |
|---|---|---|
| pre-fix build | 153 | `8447c7032117` |
| r240 | 153 | `8447c7032117` |
| r240, `GGML_CUDA_BF16_SRC1_CACHE=0` | 153 | `8447c7032117` |
| r240, `GGML_CUDA_MMB_CACHE=0` | 153 | `8447c7032117` |

`scripts/gate-hc-fusions.sh` (all four mix/combine arms + the two DSV4 numeric arms, 3 devices `-sm layer`):
**PASS** -- 5 hard arms byte-identical to the unfused chain (449 chars `d62340fbf566`), the two numeric arms at
their recorded anchor `6a102dbc62d6`.  No GPU fault in any arm's window.

## The gate

`scripts/gate-hc-base-copy.sh` (new): two text arms (delivery defaults and `GGML_CUDA_BF16_SRC1_CACHE=0`, both
anchored to `8447c7032117`), then two census arms -- every device `hc=0` with `made>0`, `bf16 <= 1`,
`mmb <= 8`, retained under `MAX_RETAINED_MIB` (default 512; measured 292), and with the cache off no BF16
block at all -- plus a GPU-fault scan.  **Negative-tested**: the pre-fix build (canonical caches, pre-r240
ordering, census present) fails it with `hc=31/620.0` live on device 0, `dead=59/521.8` and 1854.6 MiB
retained; a build without the census fails on "no census lines" rather than passing silently.

## What this does NOT claim

* The reporter's 259.6k-token OOM itself is not reproduced here (this box carries the model on 3 devices, so
  there is room); what is reproduced is the retention that caused it and the headroom it costs.
* The MMB cache still holds its live working set (`mmb=4`, up to 68 MiB on device 0 at this shape) and one
  BF16 src1 block: that is the caches working as designed, it was present before r236, and it is not part of
  this defect.
* The fix is inert under `-sm tensor` -- that path emits `DSV4_HC_POST` instead of the repeat chain
  (`cparams.fused_dsv4_hc_post = meta_split_active`), so the matcher never runs there, which is why the
  reporter's production mode never saw the bug.
