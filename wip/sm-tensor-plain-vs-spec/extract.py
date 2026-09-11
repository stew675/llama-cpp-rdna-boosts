import sys,hashlib
def extract(fn):
    lines=open(fn,errors='replace').read().splitlines()
    start=None
    for i,l in enumerate(lines):
        if l.startswith('> '): start=i+1
    if start is None: return None
    end=len(lines)
    for i in range(start,len(lines)):
        if lines[i].startswith('[ Prompt:'): end=i; break
    return '\n'.join(lines[start:end]).strip()
def div(a,b):
    n=min(len(a),len(b))
    for i in range(n):
        if a[i]!=b[i]: return i
    return None if len(a)==len(b) else n
if __name__=='__main__':
    import glob
    for fn in sys.argv[1:]:
        t=extract(fn)
        print(f"{fn}: len={len(t) if t else 0} sha1={hashlib.sha1((t or '').encode()).hexdigest()[:8]}")
    # pair compare if two files
    if len(sys.argv)==3:
        a=extract(sys.argv[1]); b=extract(sys.argv[2])
        x=div(a,b)
        print("  ->", "IDENTICAL" if x is None else f"diverge@{x}")
