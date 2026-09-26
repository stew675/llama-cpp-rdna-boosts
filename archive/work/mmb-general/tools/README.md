# `archive/work/mmb-general/tools/` — the measurement instruments

Two kinds of thing live here: **hardware probes** (written during the campaign) and the **A/B
harness** (added 2026-09-21, because it had been re-derived from scratch once too often).

## The A/B harness

* **`ab-interleaved.sh`** — the interleaved delivery-vs-WIP benchmark.  Alternates the two binaries
  in one warm session and reports the mean per test plus the delivery's own spread.  Reads the
  binaries from `BASE_BIN`/`WIP_BIN` (defaults `~/llama-base` and `~/llama.cpp`), takes the WIP-side
  environment with `--wip-env`.

  ```sh
  # the campaign's headline model, deep prefill, 3-GPU tensor
  ./ab-interleaved.sh "qwen4exp deep" 0,1,2 \
      /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf 2 \
      -- -sm tensor -ctk q8_0 -ctv q8_0 -fa auto -p 32768,65536,98304 -n 0 -r 2

  # dense, with the WIP master switch on
  ./ab-interleaved.sh "27B IQ3_S" 0 /llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf 3 \
      --wip-env GGML_CUDA_MMB=1 -- -p 8192,32768 -n 0 -r 3
  ```

* **`lbparse.py`** — the output parser, in two modes: llama-bench tables (`lbparse.py`) and
  `test-backend-ops` results (`lbparse.py --ops FLASH_ATTN_EXT`, exit 1 on any FAIL).

**Why the harness exists and why it is interleaved:** the first prefill test of an invocation is
cold-start-limited (~-9 %) and the clock ramps on the first compute-dense run — not heat.  On 3-GPU
qwen4exp the *same* configuration varied **+/-3 % at pp8192** but only **+/-0.1-0.4 % at
pp65536/98304**.  So: **prefer depth for a verdict; treat a shallow-only delta as +/-2 %**, and never
run two benches at once.

**Why the parser exists rather than a `grep`** — two traps each cost real time in this campaign:

1. `llama-bench` names a depth-variant test **`tg128 @ d16384`**, so a regex on the test name silently
   drops it.
2. **Never count `test-backend-ops` from a `2>&1`-merged log.**  The status is ANSI-wrapped
   (`printf("\033[1;32mOK\033[0m\n")`) and `print_test_console` writes the name to **stdout** while the
   test emits CUDA-graph-warmup/allocation notices to **stderr**; with `2>&1` into a file, stdout is
   block-buffered and stderr is not, so the status is orphaned onto its own line.  Counting
   `NAME(...): OK` then reports a *different* number each run (1947/1949/1951 were observed) and it
   looks like the matrix is randomised — it is not, it is deterministic (8090 name lines every run,
   **5954 OK / 0 FAIL**).  `lbparse.py --ops` pairs an orphaned status with its preceding name, so the
   totals are stable from a merged *or* a separated log.

Verified 2026-09-21 on gfx1201: `--ops` returns `OK=5953 not_supported=11 FAIL=0` from both the raw
merged and the separated captures of the same binary, and `ab-interleaved.sh` reproduces the 27B
UD-IQ3_S `+0.45 %` pp2048 figure.

## The hardware probes

* **`dram-bw-probe.cpp`** — DRAM bandwidth probe.
* **`wmma-peak-gfx1151.cpp`** — the gfx1151 WMMA peak-throughput probe (the measurement that
  established what the `mmb` bf16-WMMA path could reach on Strix Halo, and hence the campaign's
  headroom case).
