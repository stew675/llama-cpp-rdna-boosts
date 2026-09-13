// Elementwise op micro-bench: F32 [10240, 2048] ops from the hc path.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Case { const char * name; int op; };

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "no backend\n"); return 1; }
    const int64_t N = 10240, M = 2048;

    struct { const char * name; ggml_tensor * (*build)(ggml_context *, ggml_tensor *, ggml_tensor *); } cases[] = {
        { "mul_bcast [10240,2048]x[10240]", [](ggml_context * c, ggml_tensor * a, ggml_tensor * b){ return ggml_mul(c, a, b); } },
        { "sigmoid   [10240,2048]",         [](ggml_context * c, ggml_tensor * a, ggml_tensor * b){ return ggml_sigmoid(c, a); } },
        { "rms_norm  [10240,2048]",         [](ggml_context * c, ggml_tensor * a, ggml_tensor * b){ return ggml_rms_norm(c, a, 1e-6f); } },
        { "add       [10240,2048]+[10240,2048]", [](ggml_context * c, ggml_tensor * a, ggml_tensor * b){ return ggml_add(c, a, b); } },
        { "scale     [10240,2048]",         [](ggml_context * c, ggml_tensor * a, ggml_tensor * b){ return ggml_scale(c, a, 0.25f); } },
        { "repeat+add [10240,2048]+[2560,2048]x4", [](ggml_context * c, ggml_tensor * a, ggml_tensor * b){ return ggml_add(c, a, ggml_repeat_4d(c, ggml_reshape_3d(c, b, 2560, 1, 2048), 2560, 4, 2048, 1)); } },
    };
    const int n_cases = sizeof(cases)/sizeof(cases[0]);

    ggml_context * ctx = ggml_init({ 64*1024*1024, nullptr, true });
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    std::vector<float> av(N*M), bv(N*M);
    for (size_t i = 0; i < av.size(); i++) { av[i] = (float)(i % 997) / 1000.0f; bv[i] = 0.5f; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(a, av.data(), 0, av.size()*4);
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size()*4);

    for (int ci = 0; ci < n_cases; ci++) {
        ggml_context * gctx = ggml_init({ 64*1024*1024, nullptr, true });
        ggml_tensor * out = cases[ci].build(gctx, a, b);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, backend);
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const int n_iter = 30;
        const double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const double dt = (now_ms() - t0) / n_iter;
        const double bytes = (double) N*M*4*2/1e9; // read+write GB
        fprintf(stderr, "%-42s %8.3f ms  %6.1f GB/s\n", cases[ci].name, dt, bytes/dt*1000);
        ggml_backend_buffer_free(gbuf);
        ggml_free(gctx);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
