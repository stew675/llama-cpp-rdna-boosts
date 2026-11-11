#!/usr/bin/env python3
"""moe-heatmap-read.py -- read the delivery's persisted MoE heat map (`<model>.moe-heatmap`, "MHT2") and report
how concentrated the expert traffic is.  This is the number that decides how big a UMA expert store has to be,
and how much of the expert set is effectively never touched (which is what an Nth-touch admission policy would
leave out).

Format (ggml/src/ggml-cuda/moe-expert-cache.cu:3979-3993, `mht1_hdr_t` / `mht1_rec_t`):
  32-byte header: magic[4] "MHT2" | u32 version | u32 n_tables | u32 reserved0 | u64 model_bytes | u32
  n_tensors | u32 reserved1
  per table (80-byte record): i32 layer | char role[64] | u32 n_experts | u64 host_bytes
  then `all`  : u32 x n_experts   (routing count over the run's whole history)
  then `dec`  : u32 x n_experts   (the decode band only)

MEASUREMENT HELPER: reads one file, changes nothing, needs no GPU and no root.

Usage:
  ./moe-heatmap-read.py <model>.moe-heatmap [--role exps] [--layer N] [--quiet]
"""
import struct, sys, collections

def main():
    args = sys.argv[1:]
    if not args: print(__doc__); return 2
    path = args.pop(0)
    role_filter = None; only = None; quiet = False
    while args:
        a = args.pop(0)
        if a == '--role': role_filter = args.pop(0)
        elif a == '--layer': only = int(args.pop(0))
        elif a == '--quiet': quiet = True
        else: print(f"unknown arg {a}"); return 2
    b = open(path, 'rb').read()
    magic, ver, n_tables, _r0, model_bytes, n_tensors, _r1 = struct.unpack_from('<4sIIIQII', b, 0)
    if magic not in (b'MHT2', b'MHT1'):
        print(f"{path}: not a heat map (magic {magic!r})"); return 1
    off = 32
    tables = []
    for _ in range(n_tables):
        layer, = struct.unpack_from('<i', b, off)
        role = b[off+4:off+68].split(b'\0')[0].decode('utf-8', 'replace')
        n_exp, = struct.unpack_from('<I', b, off+68)
        host_b, = struct.unpack_from('<Q', b, off+72)
        off += 80
        allv = list(struct.unpack_from(f'<{n_exp}I', b, off)); off += 4*n_exp
        decv = allv
        if magic == b'MHT2':
            decv = list(struct.unpack_from(f'<{n_exp}I', b, off)); off += 4*n_exp
        tables.append((layer, role, n_exp, host_b, allv, decv))
    print(f"{path}: {'MHT2' if magic==b'MHT2' else 'MHT1'} v{ver}, {len(tables)} tables, "
          f"model {model_bytes/2**30:.2f} GiB, {n_tensors} tensors, {len(b)} bytes")

    sel = [t for t in tables if (role_filter is None or t[1] == role_filter) and (only is None or t[0] == only)]
    if not sel: print("  no table matched"); return 1
    if not quiet:
        by_role = collections.Counter(t[1] for t in tables)
        print(f"  roles: {dict(by_role)}; per-table experts {sorted({t[2] for t in tables})}, "
              f"slot bytes {sorted({t[3] for t in tables})}")

    for band, idx in (('all', 4), ('decode', 5)):
        tot = sum(sum(t[idx]) for t in sel)
        if tot == 0:
            print(f"  [{band}] no counts"); continue
        # aggregate every selected table's experts into one distribution (share of total traffic per expert)
        vals = []
        for t in sel: vals += t[idx]
        vals.sort(reverse=True)
        n = len(vals)
        zero = sum(1 for v in vals if v == 0)
        cum = 0; marks = {}
        targets = [0.50, 0.80, 0.90, 0.95, 0.99]
        for i, v in enumerate(vals, 1):
            cum += v
            for tg in targets:
                if tg not in marks and cum >= tg*tot: marks[tg] = i
        print(f"  [{band}] {n} expert-slots, {tot:,} accesses; {zero} never touched ({100.0*zero/n:.1f} %), "
              f"top expert {100.0*vals[0]/tot:.2f} % of traffic")
        print("        coverage: " + "  ".join(f"{int(tg*100)}% -> {marks.get(tg, n)} experts "
                                              f"({100.0*marks.get(tg, n)/n:.1f} %)" for tg in targets))
        # the same as a footprint, using the mean slot size of the selection
        mean_slot = sum(t[3]*t[2] for t in sel)/n if n else 0
        for tg in (0.90, 0.95, 0.99):
            print(f"        at {int(tg*100)} % coverage the resident set is "
                  f"{marks.get(tg, n)*mean_slot/2**30:.2f} GiB of {n*mean_slot/2**30:.2f} GiB")
    # how much the decode band concentrates vs all traffic (the LFRU input)
    da = [sum(t[5]) for t in sel]; aa = [sum(t[4]) for t in sel]
    if sum(aa):
        print(f"  decode band is {100.0*sum(da)/sum(aa):.1f} % of all counted accesses in this selection")
    return 0

if __name__ == '__main__':
    sys.exit(main())
