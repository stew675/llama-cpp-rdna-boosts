#!/usr/bin/env python3
"""Header-only GGUF tensor-type scanner.

Reads only the GGUF header + tensor-info table (first few hundred KB of each
file), never the tensor data, so it is fast over a large model tree.
"""
import os, struct, sys, json, collections

TYPES = {
    0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
    10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
    16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
    22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",
    29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4",40:"NVFP4",
    41:"Q1_0",42:"Q2_0",
}
TARGET = {"Q2_K","Q2_0","IQ2_XXS","IQ2_XS","IQ2_S"}

class R:
    def __init__(self, f): self.f = f
    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n: raise EOFError("short read")
        return b
    def u32(self): return struct.unpack("<I", self.raw(4))[0]
    def u64(self): return struct.unpack("<Q", self.raw(8))[0]
    def s(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

def skip_value(r, vt):
    """Skip one metadata value of type vt. Returns True if it was a scalar we can ignore."""
    if vt in (0,1,7): r.raw(1)
    elif vt in (2,3): r.raw(2)
    elif vt in (4,5): r.raw(4)
    elif vt == 6:     r.raw(4)
    elif vt in (10,11,12): r.raw(8)
    elif vt == 8:     r.s()
    elif vt == 9:     # array
        et = r.u32(); n = r.u64()
        # arrays of strings/arrays are the only non-fixed sizes
        for _ in range(n):
            skip_value(r, et)
    else:
        raise ValueError(f"unknown gguf value type {vt}")

def scan(path):
    with open(path, "rb") as f:
        r = R(f)
        magic = r.u32()
        if magic != 0x46554747:
            return {"error": f"bad magic {magic:#x}"}
        ver = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            r.s()               # key
            vt = r.u32()
            skip_value(r, vt)
        hist = collections.Counter()
        tnames = []
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            r.raw(8 * nd)       # dims
            t = r.u32()
            r.raw(8)            # offset
            tn = TYPES.get(t, f"?{t}")
            hist[tn] += 1
            tnames.append(tn)
        return {"version": ver, "n_tensors": n_tensors, "types": dict(hist),
                "targets": sorted(set(hist) & TARGET)}

def main():
    roots = sys.argv[1:] or ["/llm/models"]
    files = []
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(".gguf"):
                    files.append(os.path.join(dp, fn))
    files.sort()
    rows = []
    for p in files:
        try:
            res = scan(p)
        except Exception as e:
            res = {"error": repr(e)}
        rows.append((p, res))
    # print a compact per-file line
    for p, res in rows:
        if "error" in res:
            print(f"ERR  {p}: {res['error']}")
        else:
            tgt = (" TARGET=" + ",".join(res["targets"])) if res["targets"] else ""
            print(f"{res['n_tensors']:6d}  {p}{tgt}")
            print(f"          types: {res['types']}")
    # summary
    print("\n=== files containing Q2_*/IQ2_* ===")
    hits = [(p, res) for p, res in rows if "error" not in res and res["targets"]]
    if not hits:
        print("(none)")
    for p, res in hits:
        print(f"{p}\n    targets={res['targets']}  all={res['types']}")
    print(f"\nscanned {len(rows)} files, {len(hits)} with Q2_*/IQ2_* weights")

if __name__ == "__main__":
    main()
