# issue-44 HC_COMBINE CPU-reference oracle

The standalone op-level oracle used to validate the block-14 `hc_combine` CPU-reference fix
(issue #44).  It links against a built llama.cpp (`libggml` / `libggml-cpu` / `libggml-hip`) and
builds `GGML_OP_HC_COMBINE` with the exact layouts the qwen4exp model produces:

* `block_out` is a contiguous `[n_embd, nt]` tensor (row stride `n_embd`, *not* `nt`);
* `inject` is a strided view into the mix-output tail `[hc, nt]` whose row stride is `n_embd + hc`.

It runs the op on the CPU and on the HIP backend and compares both against a host reference.  With
the pre-fix CPU reference the multi-token cases fail on CPU (HIP stays exact); with the fix all cases
pass.

## Build / run (from a llama.cpp checkout, after `~/bin/build-llama-rocm-714`)

```sh
cd ~/llama.cpp
g++ -std=c++17 -O2 -I ggml/include \
    <repo>/wip/issue-44-hc-combine-oracle/hctest.cpp \
    -o build-rocm/bin/hctest \
    -Lbuild-rocm/bin -lggml -lggml-base -lggml-cpu -lggml-hip -lggml-rpc \
    -Wl,-rpath,'$ORIGIN'
./build-rocm/bin/hctest
```

The binary must sit next to the backend `.so` files so `ggml_backend_load_all()` finds them.

## Result (gfx1100, release `v16-84e76d8a2-r3`)

| case | pre-fix cpu-vs-ref | post-fix |
|---|---|---|
| `nt = 1` | PASS | PASS |
| `nt = 2,4,7`, `hc = 4/8`, contiguous/broadcast variants | **FAIL (up to 1.06)** | **PASS** |
| HIP (all cases) | exact | exact |

`nt == 1` (decode) was accidentally correct pre-fix because `t == 0`; every `t >= 1` read the wrong
rows.  See `WORKLOG.md` 2026-09-25 (r3).
