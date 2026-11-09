# The host-expert prefill "regression": `llama-bench`'s random-token prompt, plus one real transport floor

**Date:** 2026-11-08 · **Box:** primary, 1 × / 3 × R9700 (gfx1201, 32 GiB), Ryzen 9950X3D, ROCm 10.1.0, VMM build
**Closes** nothing on its own; it is the measurement behind `TODO` #296's closure, the `TODO` #273 note and the new
`scripts/gate-rebase-release.sh` step 5b.  Related: `INVARIANTS.md`, `WORKLOG.md` 261108.011.

## The claim being tested

On a host-resident-expert config (`-ncmoe`), the delivery's prefill was reported at **−15…−18 %** against unpatched
`a691b2345`, consistently, from `llama-bench -p 512`.  That is the number the campaign had been chasing.  It is not
a delivery defect: it is a property of **`llama-bench`'s synthetic prompt**, on which the expert arena cannot work
by construction, and it hides a narrower real cost (the no-locality transport floor).

## 1. The tool builds its prompt from random token ids

`tools/llama-bench/llama-bench.cpp` (prompt construction):

```cpp
tokens[0] = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
for (int i = 1; i < n_tokens; i++) {
    tokens[i] = std::rand() % n_vocab;
}
```

Uniformly random *token ids*, not text.  Every ubatch therefore routes over **every** expert of every layer (a
random 512-token ubatch selects ≈ 256 of 256 experts per layer at top-8), so the heat-ranked arena has no
locality to exploit — and a uniform 16-rows-per-expert split is the *best* case for a CPU GEMM, which is exactly
what unpatched upstream runs.

## 2. Same arm, same flags, two prompt shapes

35B-A3B Q8_0, 1 × R9700, `-ncmoe 48`, `-ub 512 -b 2048`, ~1450-token prompt, `-t 8`.

| build | `llama-bench` (random ids) | `llama-cli -f` (the shipped prose prompt) |
|---|---:|---:|
| r236 (delivery) | **414.5** | **1064** |
| unpatched `a691b2345` | **508.3** | **379** |

So the sign of the comparison flips with the prompt, and the delivery is +181 % where production traffic lives.

## 3. The arena is inert on the random-token prompt (the proof)

Same arm, `MOE_EXPERT_CACHE_MIB` sweep (35B-A3B Q8_0, 1 × R9700, `-ncmoe 48`, `-ub 512`, ~1450-token prompt):

| `MOE_EXPERT_CACHE_MIB` | `llama-bench` pp1450 | `llama-cli` real-text pp1450 |
|---|---:|---:|
| `0` (streaming, arena off) | 414.10 | 401.6 |
| `4096` | 414.50 | 583.6 |
| `8192` | 414.48 | 676.9 |
| `auto` (22 567.5 MiB = 69.1 % residency) | 414.48 | **1063.6** |

A 22 GiB arena changes the `llama-bench` number by **0.09 %**.  On real text the same sweep is the whole win
(401.6 → 1063.6, **+165 %** over its own streaming arm).  No `llama-bench`-based A/B can therefore see the arena,
the host-expert transport or the pool — which is where this repo's prefill risk lives.

## 4. What the delivery actually moves on a real-text prefill

`-lv 4` census, 35B-A3B Q8_0, `-ncmoe 48`, ~4000-token real-text prompt, 69.1 % arena residency:

```
expert access: hits=1629 fills=289 colds=42 (hit=83.11% fill=14.74% cold=2.14%)
expert H2D traffic: fills=307.1 MiB cold-reads=44.6 MiB total=351.7 MiB
device gather: 234 calls (234 pool-backed), 0 routed experts copied ... (neither tier)
```

**351.7 MiB of expert traffic for the whole prefill**, at 1077 t/s.  Under uniform routing the same model must
move the un-resident complement (31 % of a 31 962 MiB host set ≈ 9.9 GiB) per ubatch, at the measured pageable
H2D calibration of **28.5-28.7 GB/s** (`ggml_backend_cuda_stage_h2d_gbps`).

## 5. The transport floor (the part that is real)

**Scales with the host fraction** (`llama-bench`, random ids, 1 × R9700, `-p 512 -n 0`):

| `-ncmoe` | delivery | unpatched base | delta |
|---|---:|---:|---:|
| 12 | 1290.5 | 1180.6 | **+9.3 %** |
| 24 | 713.1 | 755.1 | −5.6 % |
| 48 | 438.3 | 533.4 | **−17.8 %** |

**Scales with the ubatch COUNT, not its size** (`llama-bench`, random ids, `-p 4096`, delivery):
`-ub` 512 / 1024 / 2048 / 4096 → **439.3 / 876.6 / 1744.4 / 3446.2** t/s = exactly 1/2/4/8 × a constant
per-ubatch cost (unpatched base over the same sweep: 528.9 / 907.6 / 1483.1 / 2228.2, i.e. CPU-compute-bound and
overtaken above `-ub` 1024).  That constant is the resident-complement re-read.

**Not any of the obvious knobs** (all measured at `-ncmoe 48`, `-p 4096 -ub 512`, delivery, `llama-bench`):
`MOE_EXPERT_CACHE_MIB=0` 436 (pp512 arm), `GGML_CUDA_SLAB=0` 435, `GGML_SCHED_DEVGATHER=0` 439.35,
`MOE_HOST_TIER=tensor` 439.33, `MOE_HOST_POOL_COLD=0` 439.29, `LLAMA_TENSOR_HOST_BUFT=0` 438, and
`GGML_CPU_MOE_OFFLOAD_THREADS=0|32` 435.9 / 437.4 — every one inside noise of the 438-439 default.  The dense
path is not involved either: all-experts-on-GPU 3 × R9700 `-sm tensor` pp512 is **5357-5393 (delivery) vs
4738-4749 (base), +13 %**.

So the floor is the **host→device expert transport under no routing locality** (the `TODO` #273 family, and the
`-ncmoe`-proportional cost `TODO` #133 already describes): upstream computes those experts on the CPU out of
local RAM and never crosses the bus.  On real text the delivery's streaming arm is +6 % over that CPU path
(401.6 vs 379) and the arena then takes it to +181 % — so the floor only binds when locality is absent.

## 6. Consequences recorded

* `TODO` #273 carries the numbers and the method change (real text, never `llama-bench`, for any transport A/B);
  the random-token floor (414 vs 508) is the target it should be measured against.
* `scripts/gate-rebase-release.sh` gains step **5b `textprefill`**: real-text prompt-eval on
  `-ncmoe 48` against the shipped build, `--text-tolerance` default 3.0 %.  First run (3 rounds, r236 vs r235):
  **−0.45 % (−0.46 / −0.45 / −0.45, spread 0.01)**, PASS.
* `TODO` #296 (block 12's CPU hunks) closed in the same pass: the two dead hunks are gone from the tree, and the
  measurement above is `D8`'s "host-expert CPU path" arm — read on **real text**, where the delivery is +181 %,
  not on `llama-bench`, where it is −18 % by construction.
