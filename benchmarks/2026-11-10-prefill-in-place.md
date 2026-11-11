# An identity expert table is served IN PLACE in the prefill band (TODO #133, r258)

**Ship / do not ship:** SHIPPED (r258, `v16-a691b2345-r258`, block 12).
**What it is:** above the decode/verify band, an **identity** expert table (arena holds every expert,
`slot == expert`) is read by the op **directly from the arena**. The scheduler takes the input over and
skips the whole-table staging/gather, instead of copying the table out of VRAM into a ring slot and then
D2D-ing it into the split input — once per table per prefill ubatch.
**Switch:** `MOE_EXPERT_PREFILL_IN_PLACE=0` restores the staged/gathered path (A/B and bisect only).

## Why this is not the r40 mechanism coming back

`ENVIRONMENT.md` §1.2's r40 note and `TODO` #273 both warn against an "in-place/remap prefill read". That
removed machinery was a **device-side policy**: it decided per expert (resident vs complement) and read
the resident half in place while fetching the complement, with a device map, deferred promotion and
device mirrors. It was slower than staging, and the archive refutes it.

This change contains none of that:

* it is **identity-only** — there is no complement to fetch, so no policy, no per-expert decision, no
  device map, no deferral. The arena is byte-for-byte the table (`expert_bytes` per slot, `slot e` = expert
  `e`), so the redirect is a base-pointer swap and the strides are unchanged.
* it reuses the **decode band's existing redirect** (`moe_cache_get_table` / `moe_cache_take_over`), which
  has served identity tables in place since session 7 — the band gate was the only thing keeping the
  prefill band out of it.
* it is measured at **2.4x**, not as a loss.

`TODO` #133 asked for exactly this ("let an identity (fully resident) table serve the prefill op in
place"). Non-identity tables are untouched: they still stage or gather, byte for byte as before.

## The defect being fixed (r254 measurement, reproduced here)

Three RDNA4 GPUs (`HIP_VISIBLE_DEVICES=0,1,2`), `-sm tensor`, Qwen3.6-35B-A3B **Q8_0**, `-ncmoe 48`,
`-ub 2048`, `prompts/prose-rdna-boosts.txt` (5246 tokens), `-n 0`, arena **100 % resident** (identity):

| arm | prefill t/s | staging/gather calls |
|---|---:|---:|
| pinned master (`MOE_HOST_TIER=tensor`), staged | **1489.6** | 702 stage ops, **63 648 MiB copied out of the arena**, 0.0 % from host |
| pinned master, 1 GPU (69.1 % residency — not identity) | 1966.8 | 273 stage ops, 51 341 MiB arena + 22 915 MiB host |
| pool mirror (the default), gathered above the band | 3588.8 | 1053 gather calls |

3 GPUs with 3x the VRAM prefill **slower than 1 GPU**: the re-copy is serialised against the compute
(copy table → compute → copy next) and is issued once per table per device per ubatch.

## The measurement after the change

Same box, same command, same prompt. Every row is the shipped tree `c1cd962e0`
(`apply-all.sh` re-cut, tree `108d298f63`).

| arm | before | after | |
|---|---:|---:|---|
| 3 GPUs, `-sm tensor`, pinned master (identity arena) | **1125-1508** (4 reps) | **3921-4100** (4 reps) | **2.6-3.6x** |
| 3 GPUs, `-sm tensor`, pinned master, `-ub/-b 1024`, `-c 16384` (MTP arm) | 398.6 | **3996.6** | **+10.0x** |
| 3 GPUs, `-sm tensor`, pool mirror (default) | **3505-3589** (3 reps) | **3968-4072** (3 reps) | **+13-14 %** |
| 1 GPU, pool, 69.1 % residency (not identity) | 2241.3 | 2275.6 | inert (+1.5 %, noise) |
| 1 GPU, pinned master, 69.1 % residency (not identity) | 1966.8 | 1982.8 | inert (+0.8 %, noise) |
| 2 GPUs, `-sm layer`, `MOE_EXPERT_CACHE_MIB=2048` (`-npl 1,4,8`) | pp 1030 / 1258 / 1259 | pp 1055 / 1253 / 1275 | inert (11 % residency) |

The **staged** pinned arm is the one that spreads (1125-1508 t/s over four reps): it is the arm that moves
63 GiB per prefill, so it tracks the card's clocks/power state after a long session.  The in-place arm is tight
(3921-4100), which is what a compute-bound prefill looks like.  Quote the ratio as 2.6-3.6x, never a single
pair.

The kill-switch arm reproduces the pre-change numbers (1507.8 / 398.6 against 1489.6 / 398.6), so the A/B is
within one binary.

**The mechanism is visible in the census:** the same 4-chunk `llama-perplexity` arm logs **360 device
gather calls** with `MOE_EXPERT_PREFILL_IN_PLACE=0` and **3** without it, at the same PPL (5.1540).

## Gates (all on the shipped tree `c1cd962e0`, tree `108d298f63`)

| gate | result |
|---|---|
| `scripts/gate-prefill-logits.sh --model 35B-Q8_0 --device 0,1,2 --sm tensor --ub 2048 --extra "-ncmoe 48" --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` | **mean KLD −0.000000, same-top-p 100.000 %** (base PPL 5.8316) — the two arms are the same bytes.  The kill-switch arm is the **full pre-r258 path**, prefill MoE fusions INCLUDED (`GGML_CUDA_FUSE_TRACE` on the two arms: `ne2=390 allow (prefill)` vs `ne2=390 BLOCK (prefill-in-place)`), so this also proves the fused MMB/MMQ gate+up and the per-op chain are **bit-identical** at this shape |
| the same, pinned tier (`MOE_HOST_TIER=tensor`, own base file) | **−0.000000 / 100.000 %** |
| same-seed greedy text, in place vs staged | `reasoning.txt` 385 tok `f36300612861`; `prose` 5246 tok + 64 decode `e9db5cb8db0b`; 1 GPU pool `4a9dd8395acc`; 2 GPU `-sm layer` `4a9dd8395acc`; 35B 3-GPU pool `065817c76b95`; 4B anchor `1c5d32ac537d` — **all identical** |
| MTP (`draft-mtp`, 3 GPUs, identity) | acceptance **0.67258 identical** both arms; MTP 165.4 vs plain 92.8 t/s (rule 2 holds) |
| `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`, 2 devices `-sm layer`, `MOE_EXPERT_CACHE_MIB=2048` | equal within the first-batch constant (ABBA: 1278/1282/1249/1276 pp, 56.7/57.0/57.9/57.9 tg) — 11 % residency, so inert |
| the same on the **identity** config (3 devices `-sm tensor`, Q8_0, `-ncmoe 48`) | pp **+22 % / +28 %** at `-npl 4 / 8` (3459/3912 vs 2832/3052); at `-npl 1` the b=1 row is the warm-up artefact, and `-npl 1,1` ABBA (two rounds, 8 runs) gives steady state **3699/3672 vs 2874/2857 = +29 %**; tg identical (94.2 vs 94.4) |
| a **sequentially driven 2-slot server** (`llama-server -np 2`, 2 devices `-sm layer`, `-ncmoe 40`, two greedy requests) | both responses byte-identical: 214 chars `4b8106497433` and 179 chars `7a8bd060ffff` in both arms |
| the coherence kill-switch arms `GGML_MOE_DEVREMAP=0`, `MOE_HOST_POOL_COLD=0`, `--host-experts pool` (3 devices `-sm tensor`, 48-token greedy on `reasoning.txt`) | `702 chars sha=df25cc87d385` on all six (three arms x in-place/staged) |
| `scripts/gate-host-pool.sh` (9 arms) | **42/42 checks passed, GATE PASSED** — `takeover=0` under the fill faults, every text unchanged |
| `scripts/gate-per-block-build.sh --fast` | **16/16 built, 0 failed** (987 s) |
| `scripts/validate-set.sh` | checksums OK, strict `git am`, applied tree `108d298f63` = `release.json.tree`, Windows portability 289 files clean |
| `scripts/check-gpu-faults.sh --since "150 min ago"` | clean (the whole session window) |

## How it works (for the next reader)

1. `moe_cache_serves_in_place(weight, n_tok, device)` — the side-effect-free identity query behind the new
   `moe_cache_in_place` iface hook (block 09's `moe_cache_identity_locked` is the one decision it and the
   take-over share, so the pre-staging query and the take-over cannot disagree).
2. `sched_stage_issue` skips the ring / `stage_input` hand-off for a split whose routed expert input will
   be served in place (the same early-out the device gather already uses), so no slot is filled and no D2D
   is issued.
3. The input loop's take-over no longer has a band restriction (it was `ne[2] <= band`); the take-over
   itself was already identity-only.
4. `moe_cache_get_table`'s identity fast path serves any width; the **slot-remapped** (non-identity) path
   stays decode/verify-band only, explicitly.
5. A fused MUL_MAT_ID kernel that reads the expert table directly (MMB GLU, the fused MMQ gate+up, the
   MMQ swiglu→down fold) must not run when the table is served in place — it bypasses the dispatch's
   redirect. `ggml_cuda_cache_blocks_fusion` stands those down per table above the band; the per-op chain
   then redirects to the arena. Measured cost of standing them down: ~0-2 % (`GGML_CUDA_MMB=0`,
   `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`).
6. `ggml_backend_buffer_get_usage(NULL)` now answers `GGML_BACKEND_BUFFER_USAGE_ANY` instead of asserting:
   the redirect hands the kernel a tensor with `buffer == nullptr` (a raw `cudaMalloc` arena), and the
   compute-buffer padding clears all read `== GGML_BACKEND_BUFFER_USAGE_COMPUTE`.  Harmless before because
   only the MMVQ decode path took it; MMQ/MMB take it now, and `ggml_cuda_mul_mat_q` read `t->buffer`
   unguarded.

## Reproduce

```sh
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
P=prompts/prose-rdna-boosts.txt

for e in "" "MOE_EXPERT_PREFILL_IN_PLACE=0"; do
  env HIP_VISIBLE_DEVICES=0,1,2 MOE_HOST_TIER=tensor $e \
    ~/llama.cpp/build-rocm/bin/llama-cli -m $M -ngl 99 -sm tensor -ncmoe 48 -ub 2048 -fa on \
      -f $P -n 0 --no-display-prompt --single-turn -lv 4 2>&1 \
    | grep -E "Prompt:|prefill cache staging"
done
# → 4033.9 with no staging census / 1507.8 with `ok=702 arena=63648.0 MiB`

# the default (pool mirror) arm: drop MOE_HOST_TIER, expect ~4072 vs ~3589, and the
# "device gather: N calls" line at 3 vs 351-360.
```
