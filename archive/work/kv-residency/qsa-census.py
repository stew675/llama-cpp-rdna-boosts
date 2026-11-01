#!/usr/bin/env python3
"""QSA attended-set census (archive/work/kv-residency B1(c), TODO #118).

Reads the binary the diagnostic in `ggml/src/ggml-cuda/indexer-topk.cu` writes when
`LLAMA_QSA_STREAM_CENSUS=<path>` is set (campaign branch only), and answers the one question a residency
POOL's design hangs on: **how stable is the attended set from step to step**, i.e. what hit rate would a pool
of N cells get if it admitted what attention selected?

Record layout (little-endian int32, repeated): [nrows, k, n_kv, n_blocks, name_len, name bytes, ids (k*nrows)].
Only decode-band calls (`nrows <= 8`) are recorded, so every record is one layer of one decode step.

Usage:
  ./qsa-census.py /tmp/census.bin [--cell-bytes 1088] [--layers 12]

The arm that produces it must be run with `GGML_CUDA_DISABLE_GRAPHS=1` and the sparse decode arm
(`LLAMA_QSA_DENSE_DECODE_UNTIL=0`), otherwise no top-k tensor is ever computed.  Timings from that arm are
meaningless (an uncaptured graph); only the selection statistics are.
"""
import argparse
import struct
import sys
from collections import OrderedDict, defaultdict


def read_records(path):
    recs = []
    with open(path, "rb") as f:
        blob = f.read()
    off = 0
    while off + 5 * 4 <= len(blob):
        nrows, k, n_kv, n_blocks, name_len = struct.unpack_from("<5i", blob, off)
        off += 5 * 4
        if name_len < 0 or name_len > 4096 or off + name_len > len(blob):
            break
        name = blob[off:off + name_len].decode("utf-8", "replace")
        off += name_len
        n = k * nrows
        if off + 4 * n > len(blob):
            break
        ids = struct.unpack_from("<%di" % n, blob, off)
        off += 4 * n
        recs.append((nrows, k, n_kv, n_blocks, name, ids))
    return recs


def lru_hits(seq_per_step, capacity):
    """Simulate an LRU pool of `capacity` cells over a sequence of per-step cell sets.

    Returns (hits, misses).  A step is per-layer: one query column (nrows == 1) or a verify batch, whose
    rows are unioned (the attention reads all of them in that step).
    """
    pool = OrderedDict()
    hits = misses = 0
    for step in seq_per_step:
        for cell in step:
            if cell in pool:
                pool.move_to_end(cell)
                hits += 1
            else:
                misses += 1
                pool[cell] = None
                if len(pool) > capacity:
                    pool.popitem(last=False)
    return hits, misses


def flru_hits(seq_per_step, capacity, period=32):
    """F-LRU (the delivery's expert-cache policy shape: a decayed frequency plus an LRU tiebreak).

    `period` steps halve every counter, so the policy forgets a topic shift instead of carrying a stale
    hot set forever (the reason the arena chose F-LRU over pure LRU; AGENTS.md block 09/12).
    """
    freq = defaultdict(int)
    last = {}
    hits = misses = 0
    tick = 0
    for step in seq_per_step:
        tick += 1
        if tick % period == 0:
            for k in list(freq):
                freq[k] //= 2
                if freq[k] == 0:
                    del freq[k]
        for cell in step:
            if cell in freq:
                freq[cell] += 1
                last[cell] = tick
                hits += 1
            else:
                misses += 1
                freq[cell] = 1
                last[cell] = tick
                if len(freq) > capacity:
                    victim = min(freq, key=lambda k: (freq[k], last[k]))
                    del freq[victim]
                    del last[victim]
    return hits, misses


def static_hits(seq_per_step, capacity, warmup):
    """A PRE-CALCULATED ordering (Strata's static arena): after `warmup` steps freeze the `capacity` most
    attended cells and admit nothing afterwards.  This is the policy the campaign rejected for the expert
    cache (content beats a frozen ordering) -- measured here on the KV's attended set for the same reason."""
    counts = defaultdict(int)
    hits = misses = 0
    frozen = None
    for i, step in enumerate(seq_per_step):
        if frozen is None and i >= warmup:
            frozen = set(sorted(counts, key=lambda k: -counts[k])[:capacity])
        for cell in step:
            if frozen is None:
                counts[cell] += 1
                continue          # warmup: nothing resident yet, but also nothing to charge? charge it
            if cell in frozen:
                hits += 1
            else:
                misses += 1
    return hits, misses


def unit_policy_hits(seq_per_step, capacity_cells, unit_cells, lag=1):
    """The CHEAP form of an attendance pool: block 15's region, with WHICH units are device decided by the
    attended set instead of by recency.

    Constraints it models, because they are what the region can actually do:
      * residency moves in whole map units (`unit_cells` cells/layer at the region's 4 MiB map unit);
      * the decision lags one step (`lag`), because the top-k is device data and a per-step readback would
        serialise the decode pipeline (block 15's invariant 6: a capture holds the VA, so a *lagged* host-side
        remap is legal, a synchronous one is not);
      * at most `capacity_cells` cells are resident, as a whole number of units.

    Returns (hits, misses, swaps) over every step, where a swap is one unit evicted + one admitted (the
    region flip: stage out + flip + stage in, ~8 MiB of copies at 55.7 GB/s).
    """
    assert unit_cells > 0
    cap_units = max(capacity_cells // unit_cells, 1)
    hits = misses = swaps = 0
    prev = None                   # lagged admitted set (cells of the previous step)
    resident = OrderedDict()      # unit -> None, LRU order
    for step in seq_per_step:
        cells = set(step)
        if prev is not None:
            want = []
            for c in prev:        # admit the units the previous step attended, in its order
                u = c // unit_cells
                if u not in want:
                    want.append(u)
            for u in want:
                if u in resident:
                    resident.move_to_end(u)
                    continue
                while len(resident) >= cap_units:
                    resident.popitem(last=False)
                    swaps += 1
                resident[u] = None
                swaps += 1
        for c in cells:
            if (c // unit_cells) in resident:
                hits += 1
            else:
                misses += 1
        prev = cells
    return hits, misses, swaps


def block_policy_hits(seq_per_step, capacity_cells, page_cells, lag=1):
    """A compact POOL with a page table: residency is per PAGE of `page_cells` cells (the QSA selection is
    block-granular, `compress_ratio` = 4 cells, so that is the natural page), and the pool holds
    `capacity_cells // page_cells` pages.  Unlike the region-unit case this is a *compact* pool: a logical
    page can sit anywhere in it, which is what a scattered attended set needs.
    """
    cap_pages = max(capacity_cells // page_cells, 1)
    hits = misses = fills = 0
    resident = OrderedDict()
    prev = None
    for step in seq_per_step:
        pages = set(c // page_cells for c in step)
        if prev is not None:
            for p in sorted(prev):
                if p in resident:
                    resident.move_to_end(p)
                    continue
                while len(resident) >= cap_pages:
                    resident.popitem(last=False)
                resident[p] = None
                fills += 1
        for p in pages:
            if p in resident:
                hits += 1
            else:
                misses += 1
        prev = pages
    return hits, misses, fills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cell-bytes", type=float, default=1088.0,
                    help="bytes per cell per layer (Flash-Next q8_0: 2 KV heads x 256 dims x K+V x ~1.06 B)")
    ap.add_argument("--layers", type=int, default=0,
                    help="expected layers per step (0 = infer from the record order)")
    args = ap.parse_args()

    recs = read_records(args.path)
    if not recs:
        sys.exit("no records -- was the arm run with GGML_CUDA_DISABLE_GRAPHS=1 and the sparse arm?")

    # order of layers = the distinct names, in first-seen order
    names = []
    for r in recs:
        if r[4] not in names:
            names.append(r[4])
    n_layers = args.layers or len(names)
    print("records: %d, layers per step: %d (%s), k per record: %s, n_kv: %s"
          % (len(recs), n_layers, ", ".join(names[:4]) + ("..." if len(names) > 4 else ""),
             sorted({r[1] for r in recs}), sorted({r[2] for r in recs})))

    # group into steps: a step is one record per layer, in the layer order
    steps = defaultdict(list)   # layer index -> list of sets
    cur = {}
    step_no = 0
    for nrows, k, n_kv, n_blocks, name, ids in recs:
        li = names.index(name)
        if li in cur:                       # the layer repeated -> a new step started without all layers
            for _, cells in sorted(cur.items()):
                steps[_].append(cells)
            cur = {}
            step_no += 1
        cur[li] = set(ids[i] for i in range(0, len(ids), max(nrows, 1)))
    if cur:
        for li, cells in sorted(cur.items()):
            steps[li].append(cells)

    n_steps = max(len(v) for v in steps.values())
    print("decode steps analysed: %d" % n_steps)

    # attended set size and step-to-step stability, per layer
    sizes = [len(s) for v in steps.values() for s in v]
    sizes.sort()
    print("attended cells per layer per step: min %d  median %d  max %d"
          % (sizes[0], sizes[len(sizes) // 2], sizes[-1]))

    jac = []
    for li, seq in steps.items():
        for a, b in zip(seq, seq[1:]):
            u = len(a | b)
            if u:
                jac.append(len(a & b) / u)
    if jac:
        jac.sort()
        print("consecutive-step Jaccard (same layer): median %.3f  p10 %.3f  p90 %.3f"
              % (jac[len(jac) // 2], jac[int(0.10 * len(jac))], jac[int(0.90 * (len(jac) - 1))]))

    # the deciding number: an LRU pool's hit rate, and the host traffic it implies
    n_att = sum(sizes) / max(len(sizes), 1)     # mean attended cells per layer per step
    print("\npool (cells/layer)   hit rate   host read MiB/token   vs an all-miss pool")
    base = None
    for cap in (2048, 4096, 8192, 16384, 32768, 65536):
        hits = misses = 0
        for li, seq in steps.items():
            h, m = lru_hits(seq, cap)
            hits += h
            misses += m
        tot = hits + misses
        rate = hits / tot if tot else 0.0
        mib = misses * args.cell_bytes / (1024.0 * 1024.0) / max(n_steps, 1)
        if base is None:
            base = mib
        print("%20d   %7.3f%%   %18.2f   %+8.1f%%" % (cap, 100.0 * rate, mib,
                                                      100.0 * (mib / base - 1.0) if base else 0.0))

    # policy comparison at equal capacity: LRU (Strata's CLOCK approximates it), the delivery's F-LRU, and a
    # pre-calculated static ordering.  The expert-cache campaign chose adaptive over static for the arena; this
    # is the same question on the KV's attended set.
    print("\npolicy comparison (hit rate, %d steps):" % n_steps)
    print("%8s %10s %10s %13s" % ("cells", "LRU", "F-LRU", "static(warm 1/2)"))
    for cap in (2048, 4096, 8192, 16384):
        tot = hit = hit_f = miss_f = 0
        for li, seq in steps.items():
            h, m = lru_hits(seq, cap)
            hit += h
            tot += h + m
            hf, mf = flru_hits(seq, cap)
            hit_f += hf
            miss_f += mf
        # static: count over the whole sequence with the second half frozen on the first half's heat
        hs = ms = 0
        for li, seq in steps.items():
            combined = list(seq)
            half = max(len(combined) // 2, 1)
            counts = defaultdict(int)
            for s2 in combined[:half]:
                for c in s2:
                    counts[c] += 1
            frozen = set(sorted(counts, key=lambda k: -counts[k])[:cap])
            for s2 in combined[half:]:
                for c in s2:
                    if c in frozen:
                        hs += 1
                    else:
                        ms += 1
        print("%8d %9.2f%% %9.2f%% %12.2f%%" % (cap,
              100.0 * hit / tot if tot else 0.0,
              100.0 * hit_f / (hit_f + miss_f) if (hit_f + miss_f) else 0.0,
              100.0 * hs / (hs + ms) if (hs + ms) else 0.0))

    # the cheap form: unit-granular, lagged (the region's map unit is 4 MiB -> unit_cells cells/layer)
    unit_cells = int(4 * 1024 * 1024 // args.cell_bytes)
    print("\nattendance pool in REGION UNITS (4 MiB = %d cells/layer), lag-1 admission:" % unit_cells)
    print("%18s %10s %12s %10s %14s" % ("pool cells/layer", "hit rate", "read MiB/tok", "swaps/step", "swap ms/step"))
    for cap in (8192, 16384, 32768, 65536):
        hit = mis = sw = 0
        for li, seq in steps.items():
            h, m, s2 = unit_policy_hits(seq, cap, unit_cells, lag=1)
            hit += h
            mis += m
            sw += s2
        tot = hit + mis
        per_step_sw = sw / max(n_steps, 1)
        # a swap = 4 MiB staged out + 4 MiB staged in, at the measured 55.7 GB/s link
        print("%18d %9.2f%% %11.2f %10.1f %13.2f" % (cap, 100.0 * hit / tot if tot else 0.0,
              mis * args.cell_bytes / (1024.0 * 1024.0) / max(n_steps, 1), per_step_sw,
              per_step_sw * 2 * 4 * 1024.0 * 1024.0 / (55.7e9) * 1e3))

    # the POOL with a page table, page = 4 cells (one indexer block: the selection is block-granular)
    print("\nattendance POOL with a page table, page = 4 cells (indexer block), lag-1 admission:")
    print("%18s %10s %12s %12s %14s" % ("pool cells/layer", "hit rate", "read MiB/tok", "fills/step", "fill ms/step"))
    for cap in (4096, 8192, 16384, 32768):
        hit = mis = fills = 0
        for li, seq in steps.items():
            h, m, fl = block_policy_hits(seq, cap, 4, lag=1)
            hit += h
            mis += m
            fills += fl
        tot = hit + mis
        per_step = fills / max(n_steps, 1)
        # a fill = one 4-cell page copied host->device = 4 x cell_bytes
        print("%18d %9.2f%% %11.2f %12.1f %13.2f" % (cap, 100.0 * hit / tot if tot else 0.0,
              mis * args.cell_bytes / (1024.0 * 1024.0) / max(n_steps, 1), per_step,
              per_step * 4 * args.cell_bytes / (55.7e9) * 1e3))

    print("\nall-miss (zero-hit) read for the same selection: %.2f MiB/token"
          % (sum(len(s) for v in steps.values() for s in v) * args.cell_bytes
             / (1024.0 * 1024.0) / max(n_steps, 1)))
    print("note: read volume is per layer only; multiply by the QSA layer count for the whole step, and")
    print("      compare with the measured cost (bytes / 55.7 GB/s) before believing a design.")


if __name__ == "__main__":
    main()
