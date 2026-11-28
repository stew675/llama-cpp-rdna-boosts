# h2d-latency: what the host->device path costs at the sizes a model uses

A single-file HIP micro-benchmark. It allocates, measures, frees and prints. It changes nothing, needs no root,
and says `UNKNOWN` rather than guess.

**Nothing here is part of the delivery set** in `patches/`, nothing here is applied by `scripts/apply-all.sh`, and
nothing here is covered by a release gate.

## Why it exists

The delivery derives its transport decisions from the H2D link's **bandwidth**
(`ggml_cuda_device_h2d_gbps`, the `H2D bandwidth calibration: <x> GB/s (512 MiB x3 ...)` line). Bandwidth is the
right number for one huge copy and the wrong number for everything else: the staged fill moves **one DMA per
host run of experts**, about 1.9 MB on the Flash-Next shape (measured: `123258 host DMAs` for `235687 MiB` over an
11-ubatch prefill, i.e. ~11 200 DMAs per 8192-token ubatch), and the device gather reads ~0.5 MB per expert.

At 1-2 MB a copy the achieved rate is well below the calibrated one, and the gap is **per-copy latency** (issue,
driver round trip, DMA setup). Two boxes that calibrate within 20% of each other can behave very differently
there, which is the shape of GitHub issue #368: the reporter's PCIe 3.0 x16 box calibrates at 9.2 GB/s against
our slowest card's 14.4 GB/s, yet his device gather beats the staged transport by 38% where our staging beats the
gather by 30%.

So this reports, per size and per memory kind, the **median** single-copy time (not the best one), the first
(cold) copy, and the steady rate, plus a 4-byte round-trip ping for the pure latency floor.

## Build

```bash
# Linux
hipcc -O2 -std=c++17 h2d-latency.cpp -o h2d-latency
# Windows (in the same ROCm shell you build llama.cpp in)
hipcc -O2 -std=c++17 h2d-latency.cpp -o h2d-latency.exe
```

`hipcc` cannot always infer the target on a mixed box; `--offload-arch=gfx1201` (or your card's arch) pins it.
Nothing else is needed: no cmake, no llama.cpp checkout, no model.

## Run

```bash
HIP_VISIBLE_DEVICES=0 ./h2d-latency                          # the default sweep
./h2d-latency --device 1 --reps 100                          # a specific card, more reps
./h2d-latency --sizes 4096,1048576,16777216                  # the sizes your shape uses
./h2d-latency --markdown                                     # a table to paste into an issue
```

The default sweep is 4 KiB, 64 KiB, 1 MiB, 8 MiB, 64 MiB, 512 MiB, pageable and pinned. **The 512 MiB row is the
delivery's own bandwidth probe**, so the two numbers are comparable on the same box.

## Reading it

* **A fit or tuning decision wants the median, not the min.** A rate quoted from the luckiest copy is exactly the
  kind of number this tool exists to stop people tuning on.
* The 64 KiB and 1 MiB rows are the ones that predict prefill, because they bracket what a staged fill and a
  gathered expert actually move.
* A large **cold vs median** gap is first-touch cost. On Windows a large probe allocation often does not come back
  after `hipFree` (issue #97), which is why the calibration line differs between visits on some boxes.
* A large **pageable vs pinned** gap is why the host expert tier is pinned (`--host-experts pinned|pool`).
* `UNKNOWN` means the allocation or the copy was refused, not that it was slow.

## Reference numbers

Our three links, one R9700 each, ROCm 10.2, median of 30 copies (`--sizes 4096,65536,1048576,8388608`). The
`calibrated` column is the delivery's own 512 MiB probe on the same card.

| card | calibrated | 4 KiB pinned | 64 KiB pinned | 1 MiB pinned | 8 MiB pinned | 4 B round trip |
|---|---:|---:|---:|---:|---:|---:|
| GPU0, PCIe5 x8 | 28.7 GB/s | 21.9 us (0.19 GB/s) | 9.1 us (7.2 GB/s) | 46.4 us (22.6 GB/s) | 310 us (27.1 GB/s) | 35 us |
| GPU1, M.2 riser (x4 lanes) | 14.5 GB/s | 20.4 us (0.20 GB/s) | 11.6 us (5.6 GB/s) | 104 us (10.0 GB/s) | 610 us (13.8 GB/s) | 32 us |
| GPU2, PCIe5 x8 | 28.7 GB/s | 22.2 us (0.18 GB/s) | 10.7 us (6.1 GB/s) | 47.7 us (22.0 GB/s) | 309 us (27.2 GB/s) | 35 us |

Two things to take from it:

1. **Below ~1 MiB the link is latency-bound, not bandwidth-bound.** A 64 KiB copy achieves 5-7 GB/s against a
   calibrated 14-29 GB/s, and a 4 KiB copy about 0.2 GB/s. This is why move *count* matters as much as move
   *volume*: 11 200 DMAs of 1.9 MB per 8192-token ubatch cost ~150 us each on GPU1, which is the whole staged fill.
2. **The per-copy floor is 20-35 us here.** It is a property of the platform (driver, DMA setup, WDDM on Windows),
   not of the link generation, so comparing it between two boxes is how you tell "his link is slower" apart from
   "his platform pays more per copy".

## Where the finding goes

If what it measures changes a delivery decision, the *fact* goes to `INVARIANTS.md` and the dated record to
`WORKLOG.md`; the tool stays here.
