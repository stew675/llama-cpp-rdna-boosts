#!/usr/bin/env python3
"""Write a deterministic rank-N LoRA adapter GGUF for a model, without training anything.

Usage:
    scripts/make-test-lora.py MODEL.gguf OUT.gguf [options]

Options:
    --modules a,b,c   projection names to target (default: ffn_down,ssm_out,attn_output)
                      matched as `blk.<N>.<module>.weight`; a module the model does not
                      have is skipped, so one invocation covers several architectures
    --rank N          LoRA rank (default: 2)
    --std X           stddev of the A/B entries, N(0, X) (default: 0.05)
    --scale-alnum     accepted and ignored (documentation aid: `alpha` is written as 0, which
                      llama.cpp reads as "scale = the adapter scale", exactly like the
                      publisher LoRAs this gate was written against)
    --seed N          numpy RNG seed (default: 331, the issue this exists for)
    --include-nextn   also target the appended nextn/MTP block (default: skipped, see below)
    --gguf-py DIR     gguf-py directory (default: $GGUF_PY, else <llama.cpp>/gguf-py)
    --llama-dir DIR   the llama.cpp checkout (default: $LLAMA_DIR, else ../llama.cpp)

Why this exists: `scripts/gate-lora-fusion.sh` has to prove that a LoRA's delta survives the
fused CUDA chain (issue #331).  A published adapter only exists for one specific model, so a gate
using one could not run on the models this box actually validates.  This generator reads the
model's own tensor shapes and writes `lora_a`/`lora_b` for the requested projections, so the
adapter is always shape-correct for the model in hand and its bytes are reproducible (same seed,
same model => same file).

The values are noise on purpose: the gate only needs the adapter to CHANGE the greedy text
(that is its non-vacuity arm) and to be bit-identical between the fused and the unfused chain.
"""

import argparse
import os
import re
import sys

import numpy as np


def find_gguf_py(args_gguf_py: str | None, llama_dir: str | None) -> str:
    cands = []
    if args_gguf_py:
        cands.append(args_gguf_py)
    if os.environ.get("GGUF_PY"):
        cands.append(os.environ["GGUF_PY"])
    roots = []
    if llama_dir:
        roots.append(llama_dir)
    if os.environ.get("LLAMA_DIR"):
        roots.append(os.environ["LLAMA_DIR"])
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llama.cpp"))
    roots.append(os.path.expanduser("~/llama.cpp"))
    for r in roots:
        cands.append(os.path.join(r, "gguf-py"))
    for c in cands:
        if os.path.isdir(os.path.join(c, "gguf")):
            return os.path.abspath(c)
    raise SystemExit("make-test-lora: no gguf-py found (pass --gguf-py DIR or set GGUF_PY/LLAMA_DIR)")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--modules", default="ffn_down,ssm_out,attn_output")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--std", type=float, default=0.05)
    ap.add_argument("--scale-alnum", action="store_true")
    ap.add_argument("--seed", type=int, default=331)
    ap.add_argument("--include-nextn", action="store_true")
    ap.add_argument("--gguf-py", default=None)
    ap.add_argument("--llama-dir", default=None)
    args = ap.parse_args()

    sys.path.insert(0, find_gguf_py(args.gguf_py, args.llama_dir))
    import gguf  # noqa: E402
    from gguf import GGUFReader, GGUFWriter  # noqa: E402

    want = [m.strip() for m in args.modules.split(",") if m.strip()]
    if not want:
        raise SystemExit("make-test-lora: --modules is empty")

    rd = GGUFReader(args.model)
    arch = str(rd.fields["general.architecture"].contents())
    pat = re.compile(r"^blk\.(\d+)\.([A-Za-z0-9_]+)\.weight$")

    # The appended nextn/MTP block is a duplicate of the last trunk layer, and llama.cpp does not expose
    # its tensors by name: an adapter carrying `blk.<block_count-1>.<module>.weight` is REFUSED at load
    # with "LoRA tensor ... does not exist in base model" (measured on qwen35 4B/27B, both block_count
    # 33/65 with nextn_predict_layers 1).  Published adapters are converted from the HF checkpoint, which
    # has no such block, so they never carry one.  Skip it unless asked.
    def field_int(key: str, default: int = 0) -> int:
        f = rd.fields.get(key)
        return int(f.contents()) if f is not None else default

    n_block = field_int(f"{arch}.block_count")
    n_nextn = field_int(f"{arch}.nextn_predict_layers")
    first_nextn = n_block - n_nextn if n_block > 0 else None

    targets = []
    skipped_nextn = 0
    for t in rd.tensors:
        m = pat.match(t.name)
        if m is None or m.group(2) not in want:
            continue
        if (not args.include_nextn) and first_nextn is not None and int(m.group(1)) >= first_nextn:
            skipped_nextn += 1
            continue
        shape = [int(x) for x in t.shape]
        if len(shape) != 2 or min(shape) <= 0:
            continue
        targets.append((t.name, shape[0], shape[1]))  # ne0 = in, ne1 = out
    targets.sort()

    if not targets:
        raise SystemExit(f"make-test-lora: no 2-D tensor of {want} in {args.model} "
                         "(the model has none of those projections)")

    rng = np.random.default_rng(args.seed)
    r = args.rank
    w = GGUFWriter(args.out, arch)    # `general.type = adapter` and alpha = 0 (`llama_adapter.h: get_scale`: alpha 0 => the
    # adapter scale itself, i.e. 1.0 by default) is what convert_lora_to_gguf.py writes for a
    # rank-only adapter, and what the published abliteration LoRA of issue #331 carries.
    w.add_type("adapter")
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 0.0)
    # GGUFWriter(arch) writes `general.architecture` itself; adding it again duplicates the key.
    for name, nin, nout in targets:
        # gguf-py writes a numpy shape reversed into the ggml dims, so `ne0 = in, ne1 = r` is a numpy
        # array of shape (r, in) -- the layout convert_lora_to_gguf.py produces.
        a = (rng.standard_normal((r, nin)) * args.std).astype(np.float32)
        b = (rng.standard_normal((nout, r)) * args.std).astype(np.float32)
        w.add_tensor(name + ".lora_a", a)
        w.add_tensor(name + ".lora_b", b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    mods: dict[str, int] = {}
    for name, _, _ in targets:
        mods[name.split(".")[-2]] = mods.get(name.split(".")[-2], 0) + 1
    print(f"make-test-lora: {args.out} arch={arch} rank={r} std={args.std} seed={args.seed} "
          f"tensors={len(targets) * 2} modules=" + ",".join(f"{k}x{v}" for k, v in sorted(mods.items()))
          + (f" (skipped {skipped_nextn} nextn-block tensors)" if skipped_nextn else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
