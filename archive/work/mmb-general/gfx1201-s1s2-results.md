# gfx1201 port — S1/S2 measurement record (2026-09-21)

Session: first gfx1201 (3× R9700, RDNA4) session on the `mmb-general` WIP.  This is the raw
evidence behind the port decisions in `gfx1201-porting.md`; the plan is the source of truth, this
file is the dated record.

## Setup

* **Base** = delivery r12, `~/llama-base/build-rocm`, commit `c3ee45747`.
* **WIP** = `~/llama.cpp` branch `rdna-boosts-mmb-port`, the 5 WIP patches + the 2026-09-21
  `gfx1201` G3a gate commit `bdf97a390`.
* **no-G5** = `v-no-g5` branch (`bdf97a390` + a revert of the indexer patch) = old indexer + G4 +
  G3a gate.
* Hardware: 3× Radeon AI PRO R9700 (gfx1201), ROCm `/opt/rocm-7.14.1-gfx102X`, `-sm tensor`,
  `GGML_CUDA_ALLREDUCE=hybrid`.
* Models: 27B Q8_0 (dense `qwen35`), Flash-Next UD-IQ4_XS (`qwen4exp`, 94 GiB).
* `llama-bench -b 2048 -ub 2048`, q8_0 KV for Flash-Next unless stated.
* `r=3` unless stated; `r=5` where the short-prefill cold-start noise mattered.

**Caveat learned the hard way:** `llama-bench`'s *first* prefill test in an invocation is
cold-start-limited (pp2048 moves by up to ~10 % between `r=3` and `r=5`).  Trust the deep
(32768+) numbers and the `r=5` numbers; treat a single `r=3` pp2048 as indicative only.

## Reproduce

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx102X/lib:$LD_LIBRARY_PATH
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
ARGS="-m $M -ngl 99 -sm tensor -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto -p 32768,65536,98304 -n 0 -r 3"

# base (delivery r12)
cd ~/llama-base && HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_ALLREDUCE=hybrid ./build-rocm/bin/llama-bench $ARGS
# WIP / a variant (rebuild first with BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714)
cd ~/llama.cpp && HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_ALLREDUCE=hybrid ./build-rocm/bin/llama-bench $ARGS
```

The G4 interleaved A/B was run by copying the all-G4 build's `bin/` (rpath `$ORIGIN`, so the copied
dir is self-contained) to `/tmp/allg-bin` and alternating the two executables in one warm session.
That `/tmp` copy is **ephemeral** — recreate it by copying `build-rocm/bin/.` before rebuilding the
other variant if the A/B is repeated.

## 1. Coherence / correctness (identical on Base and WIP)

| gate | result |
|---|---|
| B1 dense 27B same-seed greedy | `110 chars sha=da2e2d192e21` on **both** |
| B2 qwen4exp Flash-Next same-seed greedy | `35 chars sha=359ff4337837` on **both** (and on the WIP with G3a off) |
| B6 `test-backend-ops -o FLASH_ATTN_QSA` | 2/2 backends passed |
| B6 `-o GATED_DELTA_NET` | 2/2 |
| B6 `-o INDEXER_TOPK` | 2/2 (the G5 oracle) |
| B7 `test-logits-width-probe` (27B, prose, P=1024) | `width_purity=PASS (worst maxdiff 0)`, f16 KV |

## 2. Dense 27B Q8_0, 3-GPU tensor (no qwen4exp, so no QSA/indexer)

| test | Base (`c3ee45747`) | WIP (`bdf97a390`) | delta |
|---|---:|---:|---:|
| pp2048 | 2368.35 | 2400.97 | +1.4 % |
| pp8192 | 2302.38 | 2326.48 | +1.0 % |
| pp32768 | 2028.23 | 2037.32 | +0.4 % |
| tg128 | 36.71 | 36.75 | flat |
| tg128 @ d16384 | 36.63 | 36.67 | flat |

No regression.  (The small prefill delta is within +1 %; the dense model exercises neither the
indexer nor the non-temporal HC kernels, so this is mostly noise/allocator.)

## 3. qwen4exp Flash-Next IQ4_XS, q8_0 KV, 3-GPU tensor — the main result

### 3a. The always-QSA (G3a) regression, isolated with the env override

`LLAMA_QSA_DENSE_SHORTCUT` default is ON in the delivery and was flipped OFF by the WIP (the
gfx1151 always-QSA policy).  On gfx1201 that is a large regression.  `r=3`:

| config | pp2048 | pp8192 | pp32768 |
|---|---:|---:|---:|
| Base (shortcut ON) | 1898.31 | 2326.31 | 2335.44 |
| WIP, shortcut OFF (always-QSA) | 1539.63 | 2169.76 | 2359.36 |
| WIP + `LLAMA_QSA_DENSE_SHORTCUT=1` | 2097.43 | 2429.52 | 2420.54 |

The shortcut gates the **first ~2051 tokens of every prefill** (`n_kv <= width`, width =
`indexer_top_k + r - 1`), so it is not a pp2048-only effect.  **Decision: default the shortcut ON
on every arch except gfx1151** (commit `bdf97a390`), i.e. keep the delivery's dense-shortcut
behaviour on gfx1201 until the packed-WMMA `qsa3` path (G2) lands and can be re-measured.

### 3b. Group attribution at `r=5` (`r=3` for the deep points)

| config | pp2048 | pp8192 | pp32768 | pp65536 | pp98304 |
|---|---:|---:|---:|---:|---:|
| Base (delivery) | 1898¹ | 2326¹ | 2379 | 2216 | 2074 |
| no-G5 (G4 + G3a gate) | 2492 | 2520 | 2382 | 2203 | 2047 |
| no-G4 (G5 + G3a gate) | 2672 | 2596 | 2445 | 2317 | 2203 |
| all-groups (G4 + G5 + G3a gate) | 2531 | 2572 | 2455 | 2327 | 2211 |

¹ base pp2048/pp8192 were `r=3` and cold-start-limited; re-measure pending.

**G5 (indexer top-k) = all-groups − no-G5:** **+1.6 % at pp2048, +2.1 % at pp8192, +3.1 % at
pp32768, +5.6 % at pp65536, +8.0 % at pp98304** — a clear win that grows with depth, exactly the
indexer cost scaling with `n_kv`.

**G4 (non-temporal) = all-groups − no-G4, interleaved rounds (the shipping config, G5 on):**

| depth | G4 on | G4 off | Δ |
|---|---:|---:|---:|
| 32768 | 2451.74 / 2455.60 | 2444.20 / 2445.94 | +0.31 % / +0.40 % |
| 65536 | 2325.58 / 2325.78 | 2316.94 / 2316.59 | +0.37 % / +0.40 % |
| 98304 | 2210.12 / 2210.90 | 2202.12 / 2203.76 | +0.36 % / +0.32 % |

**G4 is a small but consistent win at every deep point in the config we ship → keep it** (maintainer
rule 2026-09-21: "if G4 produces any win at all, without losses, keep it").  An earlier
base-vs-no-G5 comparison had suggested G4 was slightly negative, but that used the **old** indexer
and is not the shipping config; the interleaved A/B above is the decisive one.

**WIP (G4+G5, G3a-gated) vs base at depth:** with no-G4/no-G5 accounting, the all-groups config is
**+3.2 % / +5.0 % / +6.6 %** at 32k/64k/98k over the delivery — the headline gfx1201 win from the
arch-neutral groups, growing monotonically with depth (the QSA/indexer scaling the WIP predicted).

## 4. Conclusions carried into the port

1. **G3a must be gated off on RDNA4** — done (`bdf97a390`, default shortcut ON except gfx1151).
2. **G5 is a real gfx1201 win and scales with depth** (+2 % at 8k → **+8.0 % at 98k**) — keep it,
   no gate needed.
3. **G4 is a small consistent win at depth (+0.3–0.4 %)** — keep it, no gate needed.
4. **The gfx1201 port is G5 + G4 + the G3a gate — nothing removed.**  The WMMA groups (G1 `mmb`,
   G2 `qsa3`) stay gfx11-gated for now; their RDNA4 fragment port is the multi-session follow-up in
   `gfx1201-porting.md`.
5. **No correctness regression** — all oracles and both same-seed hashes match on every variant.
6. **Optional re-measure to close:** base `r=5` at 2048/8192 (cold-start correction only; does not
   change any decision).
