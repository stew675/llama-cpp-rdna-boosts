// qsa-support-probe — print what the ACTING DEVICE answers for the fused sparse QSA op, per K/V
// cache type, and compare it with the hand-maintained list block 14 used before 2026-09-12.
//
// Build (from a llama.cpp tree with `build-rocm` built, run from the repo root):
//   clang++ -O2 -std=c++17 -I include -I ggml/include qsa-support-probe.cpp \
//     -Lbuild-rocm/bin -lggml -lggml-base -lllama -Wl,-rpath,$PWD/build-rocm/bin -o /tmp/qsa-support-probe
//   LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib /tmp/qsa-support-probe [head_dim]
//
// Why: `src/models/qwen4exp.cpp` must not mirror ggml_cuda_flash_attn_qsa_supported()'s type list
// (a stale mirror is an abort in the meta splitter under -sm tensor, not a fallback).  The gate now
// asks the device with a shaped probe tensor; this prints that answer next to the old list so the
// two can be compared on a real backend.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static ggml_tensor * make_probe(ggml_context * ctx, ggml_type kv, int64_t D) {
    // minimal shapes: the back-end predicate reads the source types and the head size (D) only
    ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, 1, 1);
    ggml_tensor * k   = ggml_new_tensor_4d(ctx, kv,            D, 1, 1, 1);
    ggml_tensor * v   = ggml_new_tensor_4d(ctx, kv,            D, 1, 1, 1);
    ggml_tensor * idx = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1);
    ggml_tensor * msk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 1, 1);
    return ggml_flash_attn_qsa(ctx, q, k, v, idx, msk, 1.0f, 0.0f);
}

// the list the graph gate used before 2026-09-12
static bool legacy_list(ggml_type t) {
    return t == GGML_TYPE_F16  || t == GGML_TYPE_BF16 ||
           t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0 ||
           t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_0 ||
           t == GGML_TYPE_Q5_1 || t == GGML_TYPE_IQ4_NL;
}

int main(int argc, char ** argv) {
    const int64_t D = argc > 1 ? atoll(argv[1]) : 128;

    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (dev == nullptr) {
        printf("no GPU/IGPU device found (dev_count=%zu)\n", ggml_backend_dev_count());
        return 1;
    }
    printf("device: %s (%s), head_dim D=%lld\n\n", ggml_backend_dev_name(dev),
           ggml_backend_dev_description(dev), (long long) D);

    const ggml_type types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL,
        GGML_TYPE_Q6_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS,
    };

    int mismatches = 0;
    printf("%-12s %-8s %-8s %s\n", "kv type", "device", "legacy", "verdict");
    for (ggml_type t : types) {
        const ggml_init_params params = {
            /*.mem_size   =*/ 8*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        const bool ok = ggml_backend_dev_supports_op(dev, make_probe(ctx, t, D));
        ggml_free(ctx);

        const bool was = legacy_list(t);
        const char * verdict = ok == was ? "agree" : "*** MISMATCH ***";
        if (ok != was) {
            mismatches++;
        }
        printf("%-12s %-8s %-8s %s\n", ggml_type_name(t), ok ? "yes" : "no", was ? "yes" : "no", verdict);
    }

    // head size: the legacy list ignored it, the back-end predicate does not
    for (int64_t d : { int64_t(64), int64_t(128), int64_t(256), int64_t(80) }) {
        const ggml_init_params params = {
            /*.mem_size   =*/ 8*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        const bool ok = ggml_backend_dev_supports_op(dev, make_probe(ctx, GGML_TYPE_F16, d));
        ggml_free(ctx);
        printf("head_dim %-4lld f16: device=%s  (legacy list always said yes)\n",
               (long long) d, ok ? "yes" : "no");
    }

    printf("\nmismatches vs the legacy list: %d\n", mismatches);
    return mismatches == 0 ? 0 : 2;
}
