// Standalone oracle for GGML_OP_HC_COMBINE (block 14, qwen4exp).
//
// Exercises the CPU reference `ggml_compute_forward_hc_combine_f32` against a
// host reference and against the stride-aware CUDA/HIP kernel, using the exact
// tensor layouts build_hc_mix/build_hc_combine produce:
//   * block_out is a contiguous [n_embd, nt] tensor (row stride n_embd, NOT nt)
//   * inject is a strided view of the hc_mix output tail [hc, nt] whose row
//     stride is n_embd + hc
// Pre-fix the CPU reference indexes block_out with t*ne[1] and inject with t*hc,
// so every token t >= 1 reads the wrong rows -> mismatch.  nt == 1 is fine.
//
// Build (from ~/llama.cpp):
//   g++ -std=c++17 -O2 -I ggml/include hctest.cpp -o build-rocm/bin/hctest \
//       -Lbuild-rocm/bin -lggml -lggml-base -lggml-cpu -lggml-hip -lggml-rpc \
//       -Wl,-rpath,'$ORIGIN'
// Run:
//   ./build-rocm/bin/hctest

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static float gen(int i) { return sinf(0.7f * (float) i + 1.3f) * 0.5f; }

struct Case {
    int64_t n_embd;
    int64_t hc;
    int64_t nt;
    bool broadcast_bo;  // block_out [n_embd, 1]
    bool broadcast_inj; // inject     [hc, 1]
    std::string name() const {
        char buf[128];
        snprintf(buf, sizeof(buf), "n_embd=%lld hc=%lld nt=%lld %s%s",
                 (long long) n_embd, (long long) hc, (long long) nt,
                 broadcast_bo ? "bo=bcast " : "bo=cont  ",
                 broadcast_inj ? "inj=bcast" : "inj=view");
        return buf;
    }
};

// Build the graph, run it on `backend`, return the flat [n_embd*hc*nt] output.
static std::vector<float> run_backend(ggml_backend_t backend, const Case & c,
                                      std::vector<float> & res_v,
                                      std::vector<float> & mix_v,
                                      std::vector<float> & bo_v) {
    const int64_t n_embd = c.n_embd, hc = c.hc, nt = c.nt;
    const int64_t inj_stride = c.broadcast_inj ? 0 : (n_embd + hc);
    const int64_t bo_rows    = c.broadcast_bo ? 1 : nt;

    ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /* .mem_base   = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);

    // hc_mix output tail host tensor; inject is a view into it when not broadcast
    ggml_tensor * mix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd + hc, nt);
    ggml_tensor * inject;
    if (c.broadcast_inj) {
        inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, 1);
    } else {
        // [hc, nt] view with row stride (n_embd + hc)
        inject = ggml_view_2d(ctx, mix, hc, nt, (n_embd + hc) * sizeof(float),
                              n_embd * sizeof(float));
    }

    ggml_tensor * block_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, bo_rows);

    ggml_tensor * out = ggml_hc_combine(ctx, residual, block_out, inject, hc);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc failed on %s\n", ggml_backend_name(backend));
        exit(2);
    }

    ggml_backend_tensor_set(residual, res_v.data(), 0, res_v.size() * sizeof(float));
    ggml_backend_tensor_set(mix, mix_v.data(), 0, mix_v.size() * sizeof(float));
    if (c.broadcast_inj) {
        ggml_backend_tensor_set(inject, mix_v.data() + n_embd, 0, hc * sizeof(float));
    }
    ggml_backend_tensor_set(block_out, bo_v.data(), 0, bo_v.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out_v(n_embd * hc * nt);
    ggml_backend_tensor_get(out, out_v.data(), 0, out_v.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    GGML_UNUSED(inj_stride);
    return out_v;
}

// Host reference matching ggml_hc_combine's true semantics.
static std::vector<float> reference(const Case & c,
                                    const std::vector<float> & res_v,
                                    const std::vector<float> & mix_v,
                                    const std::vector<float> & bo_v) {
    const int64_t n_embd = c.n_embd, hc = c.hc, nt = c.nt;
    std::vector<float> out(n_embd * hc * nt);
    const float inv_hc = 1.0f / (float) hc;
    for (int64_t t = 0; t < nt; ++t) {
        const float * inj = mix_v.data() + (c.broadcast_inj ? n_embd : n_embd + t * (n_embd + hc));
        for (int64_t c0 = 0; c0 < hc; ++c0) {
            const float w = 2.0f / (1.0f + expf(-inj[c0] * inv_hc));
            for (int64_t r = 0; r < n_embd; ++r) {
                const float b = bo_v[r + (c.broadcast_bo ? 0 : t * n_embd)] * w;
                out[r + c0 * n_embd + t * n_embd * hc] = res_v[r + c0 * n_embd + t * n_embd * hc] + b;
            }
        }
    }
    return out;
}

static float maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, fabsf(a[i] - b[i]));
    }
    return m;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!cpu) { fprintf(stderr, "no CPU backend\n"); return 2; }
    if (!gpu) { fprintf(stderr, "no GPU backend\n"); return 2; }

    printf("CPU backend: %s\n", ggml_backend_name(cpu));
    printf("GPU backend: %s\n", ggml_backend_name(gpu));
    printf("%-44s %10s %10s   %s\n", "case", "cpu-vs-ref", "gpu-vs-ref", "verdict");

    struct Cfg { int64_t n_embd, hc, nt; bool bbo, binj; };
    const std::vector<Cfg> cfgs = {
        {256, 4, 1, false, false}, // decode: must pass pre- and post-fix
        {256, 4, 2, false, false}, // verify width 2
        {256, 4, 4, false, false}, // verify width 4
        {256, 4, 7, false, false}, // the nt=2,2,7 prefill ubatches from issue #44
        {256, 8, 5, false, false}, // hc=8 (qwen4exp max), strided inject
        {256, 4, 4, true,  false}, // block_out broadcast
        {256, 4, 4, false, true }, // inject broadcast
        {256, 4, 4, true,  true }, // both broadcast
    };

    int failures = 0;
    for (const Cfg & f : cfgs) {
        Case c{f.n_embd, f.hc, f.nt, f.bbo, f.binj};

        std::vector<float> res_v(c.n_embd * c.hc * c.nt);
        for (size_t i = 0; i < res_v.size(); ++i) res_v[i] = gen((int) i);
        std::vector<float> mix_v((c.n_embd + c.hc) * c.nt);
        for (size_t i = 0; i < mix_v.size(); ++i) mix_v[i] = gen((int) (i + 1000));
        std::vector<float> bo_v(c.n_embd * (c.broadcast_bo ? 1 : c.nt));
        for (size_t i = 0; i < bo_v.size(); ++i) bo_v[i] = gen((int) (i + 2000));

        std::vector<float> ref = reference(c, res_v, mix_v, bo_v);
        std::vector<float> cpu_out = run_backend(cpu, c, res_v, mix_v, bo_v);
        std::vector<float> gpu_out = run_backend(gpu, c, res_v, mix_v, bo_v);

        const float d_cpu = maxdiff(cpu_out, ref);
        const float d_gpu = maxdiff(gpu_out, ref);
        const bool ok = d_cpu < 1e-5f && d_gpu < 1e-5f;
        if (!ok) failures++;

        printf("%-44s %10.3e %10.3e   %s\n", c.name().c_str(),
               (double) d_cpu, (double) d_gpu, ok ? "PASS" : "FAIL");
    }

    printf("\n%s (%d case(s) failed)\n", failures ? "OVERALL: FAIL" : "OVERALL: PASS", failures);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return failures ? 1 : 0;
}
