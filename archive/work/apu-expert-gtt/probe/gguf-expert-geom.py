#!/usr/bin/env python3
"""gguf-expert-geom.py -- read a GGUF's tensor table and print the MoE expert geometry an expert store needs:
per layer, the absolute file offset of each expert tensor, the number of experts, and the contiguous per-expert
slab size (an expert's rows are contiguous: for a 3-D `[n_embd, n_ff, n_expert]` tensor, expert e's bytes are
one contiguous run, which is what makes a single `pread` a whole expert).

This is a MEASUREMENT HELPER, not a probe: it changes nothing, needs no GPU and no root.  Its output feeds
`14-expert-cost.cpp <file> <offset> <slab_bytes> <count>`.

Usage:
  ./gguf-expert-geom.py <model.gguf> [--layer N] [--shell]
    --layer N   print only layer N (default: every layer that has expert tensors, first 3 + last)
    --shell     print `OFFSET SLAB COUNT` for layer N on one line (for `$(...)` in a bash arm)
"""
import struct, sys

GGML_TYPE_SIZE = {   # (block_size, type_size) for the types that appear in expert tensors
    0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24), 8:(32,34), 9:(32,36),
    10:(256,84), 11:(256,110), 12:(256,144), 13:(256,176), 14:(256,210), 15:(256,292),
    16:(256,66), 17:(256,74), 18:(256,98), 19:(256,50), 20:(32,18), 21:(256,110),
    22:(256,82), 23:(256,136), 24:(1,1), 25:(1,2), 26:(1,4), 27:(1,4), 28:(1,8),
    29:(256,56), 30:(1,2), 34:(256,54), 35:(256,66), 39:(32,17),
}

class R:
    def __init__(s, f): s.f = f
    def raw(s, n):
        b = s.f.read(n)
        if len(b) != n: raise EOFError("short read")
        return b
    def u32(s): return struct.unpack('<I', s.raw(4))[0]
    def u64(s): return struct.unpack('<Q', s.raw(8))[0]
    def i8(s):  return struct.unpack('<b', s.raw(1))[0]
    def string(s): return s.raw(s.u64()).decode('utf-8', 'replace')
    def value(s, t):
        if t == 8:  return s.string()
        if t == 9:
            et = s.u32(); n = s.u64()
            return [s.value(et) for _ in range(n)]
        fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
        if t not in fmt: raise ValueError(f"bad gguf kv type {t}")
        v = struct.unpack(fmt[t], s.raw(struct.calcsize(fmt[t])))[0]
        return None if t == 7 else v

def main():
    args = [a for a in sys.argv[1:]]
    if not args: print(__doc__); return 2
    path = args.pop(0)
    only = None; shell = False
    while args:
        a = args.pop(0)
        if a == '--layer': only = int(args.pop(0))
        elif a == '--shell': shell = True
        else: print(f"unknown arg {a}"); return 2

    f = open(path, 'rb'); r = R(f)
    magic = r.raw(4)
    if magic != b'GGUF': print(f"{path}: not a GGUF (magic {magic!r})"); return 2
    ver = r.u32(); n_tensors = r.u64(); n_kv = r.u64()
    align = 32
    for _ in range(n_kv):
        key = r.string(); t = r.u32(); v = r.value(t)
        if key == 'general.alignment' and isinstance(v, int): align = v
    tensors = []
    for _ in range(n_tensors):
        name = r.string(); nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32(); off = r.u64()
        tensors.append((name, dims, ttype, off))
    data_start = (f.tell() + align - 1)//align*align

    def slab_bytes(dims, ttype):
        if len(dims) < 3: return None
        bs, ts = GGML_TYPE_SIZE.get(ttype, (None, None))
        if bs is None: return None
        per_expert_elems = 1
        for d in dims[:-1]: per_expert_elems *= d      # everything but the expert dim
        return per_expert_elems // bs * ts

    layers = {}
    for name, dims, ttype, off in tensors:
        if '_exps' not in name or not name.startswith('blk.'): continue
        il = int(name.split('.')[1])
        layers.setdefault(il, []).append((name, dims, ttype, off))
    if not layers: print(f"{path}: no *_exps tensors"); return 1

    if shell:
        if only is None: print("--shell needs --layer N"); return 2
        ents = layers.get(only)
        if not ents: print(f"layer {only} has no expert tensor"); return 1
        name, dims, ttype, off = min(ents, key=lambda e: e[3])   # the first slab run in the file
        print(f"{data_start + off} {slab_bytes(dims, ttype)} {dims[2] if len(dims) > 2 else 0}")
        return 0

    keys = sorted(layers)
    show = [only] if only is not None else (keys[:3] + (['…'] if len(keys) > 4 else []) + keys[-1:])
    print(f"{path}: {n_tensors} tensors, alignment {align}, data at {data_start}, "
          f"{len(keys)} layers with *_exps")
    for il in show:
        if il == '…': print("  …"); continue
        for name, dims, ttype, off in sorted(layers[il], key=lambda e: e[3]):
            sb = slab_bytes(dims, ttype)
            if sb is None:
                print(f"  blk.{il:<3} {name:<34} dims={dims} type={ttype:<3} off={data_start+off} "
                      f"(unknown quant: slab not computed)")
            else:
                print(f"  blk.{il:<3} {name:<34} dims={dims} type={ttype:<3} "
                      f"off={data_start+off:<12} slab/expert={sb} bytes ({sb/1048576.0:.3f} MiB)")
    n_exp = max((d[2] for v in layers.values() for (_, d, t, _) in v if len(d) > 2), default=0)
    per_expert = sum(slab_bytes(d, t) or 0 for (_, d, t, _) in layers[keys[0]])
    known = sum((slab_bytes(d, t) or 0) * (d[2] if len(d) > 2 else 1)
                for v in layers.values() for (_, d, t, _) in v)
    print(f"  per-expert slab (one layer, all *_exps tensors): {per_expert/1048576.0:.3f} MiB, "
          f"n_expert={n_exp}")
    print(f"  whole expert set: {known/2**30:.2f} GiB over {len(keys)} layers "
          f"(≈{per_expert*n_exp*len(keys)/2**30:.2f} GiB if every layer matches layer {keys[0]})"
          + ("  [lower bound: an unknown quant was skipped]" if known < per_expert*n_exp*len(keys) else ""))
    return 0

if __name__ == '__main__':
    sys.exit(main())
