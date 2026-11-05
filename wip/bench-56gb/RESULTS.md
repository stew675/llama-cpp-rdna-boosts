# measured matrix, 2026-11-05

Machine: one R9700 (gfx1201, 32 GB), ROCm 10.1, 93 GiB host RAM.  Every row: server in a
`MemoryMax=56G`/`MemorySwapMax=0` scope, 31 GiB mlocked ballast outside it so `MemAvailable` reads 57.3 GiB
(a 59 GiB box, i.e. `briansp2020`'s).  Model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, draft
`mtp-...-shared-Q4_K_M`, `-c 204800`, `-ctk/-ctv q8_0`, `--parallel 1`, one load per run, greedy.  Build: the
block 15 tip (`790686b53`); the pool fill and THP pinning are the r215 shape.

Host expert set as the loader counts it: **47962.5 MiB = 46.84 GiB** (144 tensors x 512 experts).  PLE
(`per_layer_token_embd`, 26.8 GiB) is lazy-read and never resident; the dense weights (4.2 GiB) are in VRAM.

| run | host expert tier | ub | KV pool | decode t/s | prefill 50k | prefill 179.5k | sec | host anon |
|---|---|---|---|---|---|---|---|---|
| A | `-lme 47` + `RAM_PCT=0`, mirror 512/512 | 2048 | 16384 | 80.8 | 1749 | 1456 | 123.3 | 50.7 GiB |
| B | mirror | 6144 | 16384 | 81.1 | 2061 | 1814 | 99.0 | 51.2 GiB |
| C | mirror | 2048 | off | 80.0 | 1798 | 1565 | 114.7 | 48.4 GiB |
| D | mirror | 6144 | off | 78.5 | 2080 | 1820 | 98.6 | 48.9 GiB |
| E | mirror | 6144 | 65536 | 80.7 | 1957 | 1630 | 110.1 | 51.3 GiB |
| F | `-lme 47`, `RAM_PCT=50` -> 473/512 (92.4 %) | 6144 | off | 76.6 | 1061 | 1733 (*) | 103.6 | 45.6 GiB |
| G | `auto` -> 46858.8 MiB (97.7 %) | 6144 | off | 75.6 | 1443 | 1795 (*) | 100.0 | 47.8 GiB |
| H | `-lme 47` + `RAM_PCT=0`, mirror (the field config) | 6144 | off | 78.5 | 1941 | 1802 | 99.6 | 48.9 GiB |
| J2 | `-lme 51`, default ceiling -> 93.6 % | 6144 | off | 74.9 | 1770 | 1762 | 101.9 | 46.1 GiB |
| K | the model `runme` as shipped (`auto` 98.1 %, kvpool, `-cram 6144`) | 6144 | 16384 | 78.0 | 1869 | 1489 | 120.6 | 50.2 GiB |
| Z | `-lme 52` | 6144 | off | refused at argument parse | | | | |

(*) second pass: the non-pooled remainder was page-cached by then, so the 50k row is the honest first-pass number.

Reference rows from `briansp2020`'s box, same model, 56 GB container, `-c 204800`, 201.7k prompt
([PR #267 comment](https://github.com/stew675/llama-cpp-rdna-boosts/pull/267#issuecomment-6644149816)):

| his run | decode t/s | prefill |
|---|---|---|
| the model `runme`, `--load-mode mmap` | 53.4 / 66.4 | 252 s, 799 t/s |
| the model `runme`, `--load-mode dio` | 81.4 / 86.7 | 170 s, 1188 t/s |
| his branch (`--host-experts mmap`) | 82.7 / 81.7 | 104 s, 1949 t/s |

What the matrix supports, per campaign:

* **coverage**: `auto` on a 59 GiB box is 97.7 % (80 % of 57 GiB against a 46.84 GiB set) and can never mirror;
  a ceiling shave to 92.4 % halves the first prefill; `-lme 52` is refused (>90 % of `MemAvailable`) and `-lme 51`
  is accepted and then shaved to 93.6 %, which costs 5 % decode and 9 % prefill.  `-lme 47` + `RAM_PCT=0` is a
  mirror, 0 MiB of drive reads during prefill, and it is the only row at 100 %.
* **mirror transport**: the `-ub` dependence (1456 at `-ub 2048` against 1814 at `-ub 6144` with the KV pool) is
  the staging copy's amortisation, not a compute change; the 179.5k pass streams about 1.36 TB.
* **split**: coverage 100 / 97.7 / 93.6 / 92.4 % gives prefill 1941 / 1443 / 1770 / 1061, i.e. the cliff is the
  miss path (about 125 MB/s) and not the split.
* **KV pool** (feeds `kv-residency-cleanup/` §2.4): at `-c 204800` the pool reports `hits 8408 (4.27 %)` of
  196 896 ids; +2.3 GiB host anon, +2.3 GiB of arena VRAM, prefill 0 to -17 % depending on `-ub` and `-cram`,
  decode neutral; 65536 cells is worse than 16384.
* everything fits a 56 GiB container: peak charge reaches the limit only through reclaimable page cache, anon
  stays at 48.4 to 51.3 GiB, so the field config leaves about 7 GiB.
