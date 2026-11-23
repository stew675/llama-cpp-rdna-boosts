# BETA-TESTING — `prefill-decode-arms` (the withdrawn two-arm server)

You are testing a **withdrawn** feature, not the delivery.  Nothing here is covered by the delivery's
release gates; a result here is evidence for or against re-admitting the block, not a release claim.

## 0. What you are applying

One extra `git am` commit on top of the r352 release tree.  Applied alone it reproduces the **r351 release
tree** (`8851b0d02`) exactly — i.e. the delivery as it shipped from r346 to r351 — so a text or throughput
comparison against r351 is a comparison against the same code.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 919794500
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh --include-beta .
```

`apply-all.sh` prints the applied tree and warns that the checkout is not the release tree.  Record that tree
in your report.

## 1. Build

```bash
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Keep `GGML_HIP_RCCL=1` (the host-expert all-reduce arm uses it).  Report your ROCm line, GPU model and
whether the build was clean.

## 2. Hygiene (this feature has taken a box down before)

* **Pin `HIP_VISIBLE_DEVICES`** on every arm (an iGPU enumerated as a device aborts the load).
* Start every server under `setsid` + `timeout`, and kill the **process group** afterwards: after a GPU fault
  the server survives and keeps its port, so the next run fails to bind and looks like a different bug.
* `rm -f gpucore.*.gpu` after any fault, and check every log with
  `~/llama-cpp-rdna-boosts/scripts/check-gpu-faults.sh --since <date-before-the-run> --log <log>`.
* Pin the expert-cache residency prior on any MoE arm (`MOE_EXPERT_HEATMAP=load`) and use
  `ROCBLAS_USE_HIPBLASLT=0` for a fresh-process text comparison.
* Do not run two GPU jobs at once.

## 3. The smoke ladder (report each result)

| # | arm | what it shows |
|---|---|---|
| 1 | `llama-server -m <dense 27B> -ngl 99 -c 16384 --parallel 2 --kv-unified -np 2`, **no** `--prefill-decode-arms`, one request | the block is inert: text and throughput equal the r352 build without it |
| 2 | the same shape with `--prefill-decode-arms`, two concurrent requests (one decoding, one prefilling) | the feature: a decoder's token cadence through a prefill |
| 3 | a **hybrid** model (e.g. 35B-A3B) on `-sm layer`, `-ncmoe 48`, `--prefill-decode-arms` | the recurrent-state handover at a request boundary |
| 4 | a qwen4exp model (Flash-Next) on 3 devices `-sm tensor`, 3 slots, `--kv-unified`, two long prefills + one decoding slot | the shape that aborted at r350 (`qwen4exp.cpp:1138`) — **the one to watch** |
| 5 | the cache-OFF host tier: `-ncmoe 48` + `MOE_EXPERT_CACHE_MIB=0` + `-sm tensor`, cold prefill >= 2k tokens | the r350 page fault — see §4 |

For arm 1, the same-seed greedy text must be byte-identical between the two builds; for arm 2, quote the
decoder's ITL and the prefill cost.

## 4. The page fault (open, not reproduced at r352)

Recorded at r350 and **not** reproduced at r352 on any of seven arms (host tier `-sm layer` 1 and 3 devices,
`-sm tensor` with `-ncmoe 8` and 48, the recorded server command twice, and with the pinned tier forced off
two ways).  If you HAVE a reproduction, it is the most valuable thing you can send:

* the exact command line, the model, its quant and the number of shards;
* the server/`llama-cli` log at `-lv 4`;
* `dmesg` around the fault (the line that names an address and `client 10` / `GCVM_L2_PROTECTION_FAULT_STATUS`);
* `free -g` and `ulimit -l` **before** the load, and the `host experts:` census lines from the load.

That set is what turns "a device read a pageable page" into a code path.  Until then the finding stays open in
`TODO.md` and no fix is claimed.

## 5. Reporting

Send: the applied tree, the model/quant, the ROCm line, each arm's result (text hash or throughput + the gate
output), any fault with its log + dmesg, and whether you consider the feature usable.  A refutation is as
useful as a confirmation: the block is out of the delivery until someone can show it is ready.
