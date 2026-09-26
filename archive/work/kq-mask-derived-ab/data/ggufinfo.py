#!/usr/bin/env python3
"""Print the attention geometry of a GGUF file (parses the header only)."""
import struct, sys

def read_meta(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        struct.unpack('<I', f.read(4))          # version
        struct.unpack('<Q', f.read(8))          # n_tensors
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def rstr():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        def rval(t):
            fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
            if t in fmt:
                return struct.unpack(fmt[t], f.read(struct.calcsize(fmt[t])))[0]
            if t == 8:
                return rstr()
            if t == 9:
                et = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                return [rval(et) for _ in range(n)]
            raise ValueError(f'type {t}')

        meta = {}
        for _ in range(n_kv):
            k = rstr()
            t = struct.unpack('<I', f.read(4))[0]
            meta[k] = rval(t)
        return meta

m = read_meta(sys.argv[1])
arch = m.get('general.architecture', '?')
keys = ['block_count', 'embedding_length', 'context_length',
        'attention.head_count', 'attention.head_count_kv',
        'attention.key_length', 'attention.value_length',
        'attention.sliding_window', 'rope.freq_base']
print(f"== {sys.argv[1]}")
print(f"   arch = {arch}   name = {m.get('general.name','?')}   size(meta) = {m.get('general.size_label','?')}")
nl = m.get(f'{arch}.block_count', 0)
nh = m.get(f'{arch}.attention.head_count', 0)
nkv = m.get(f'{arch}.attention.head_count_kv', nh)
hk = m.get(f'{arch}.attention.key_length', 0)
hv = m.get(f'{arch}.attention.value_length', hk)
nemb = m.get(f'{arch}.embedding_length', 0)
ctx = m.get(f'{arch}.context_length', 0)
sw = m.get(f'{arch}.attention.sliding_window', 0)
if isinstance(nkv, list):
    nkv = nkv[0]
print(f"   n_layer={nl} n_embd={nemb} n_head={nh} n_head_kv={nkv} head_dim_k={hk} head_dim_v={hv} n_ctx_train={ctx} swa={sw}")
if nl and nkv and hk:
    per_tok = nl * nkv * (hk + hv) * 2          # f16 K+V bytes per token
    print(f"   KV/token (f16, K+V) = {per_tok/1024:.1f} KiB  ->  32k={per_tok*32768/2**30:.2f} GiB  64k={per_tok*65536/2**30:.2f} GiB  100k={per_tok*100000/2**30:.2f} GiB")
