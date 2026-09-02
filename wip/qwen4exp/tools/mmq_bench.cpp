// Micro-benchmark for the qwen4exp prefill GEMM shapes.
// Times each op in a loop with ggml_backend_graph_compute + synchronize.
// Shapes chosen from the pp8192 op timing:
//   A. MoE fused gate+up:  src0 Q8_0 [1280, 2560, 512], src1 F32 [2560, 2048], ids [10, 2048] -> [1280, 10, 2048]
//   B. MoE down:           src0 Q8_0 [640, 2560, 512],  src1 F32 [640, 2048],  ids [10, 2048] -> [2560, 10, 2048]
//   C. hc LoRA up:         src0 Q8_1 [320, 10240],      src1 F32 [320, 2048]                -> [10240, 2048]
//   D. hc LoRA down:       src0 Q8_1 [10240, 320],      src1 F32 [10240, 2048]               -> [320, 2048]
//   E. router:             src0 F32  [2560, 512],       src1 F32 [2560, 2048]                -> [512, 2048]
//   F. dense Qcur ref:     src0 Q8_1 [2560, 6144],      src1 F32 [2560, 2048]                -> [6144, 2048]
//   G. dense gate_inp big: src0 F32  [2560, 10240],     src1 F32 [2560, 2048]                -> [10240, 2048]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Case {
    const char * name;
    ggml_type    wtype;
    int64_t      w_ne[3];   // weight dims [K, N, (n_expert)]
    int64_t      x_ne[2];   // activation [K, M]
    bool         is_moe;    // use mul_mat_id
    int64_t      n_exp;     // number of experts (for is_moe)
};

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "no cuda backend\n"); return 1; }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));

    const int64_t n_exp = 512;
    const int64_t n_used = 10;
    const int64_t M = 2048;

    std::vector<Case> cases = {
        { "moe_gate_up  Q4_K [1280,2560,512]", GGML_TYPE_Q4_K, {2560, 1280, n_exp}, {2560, M}, true, n_exp },
        { "moe_gate_up  Q4_K [1280,2560,10] M=2048", GGML_TYPE_Q4_K, {2560, 1280, 10}, {2560, M}, true, 10 },
        { "hc_up M=1    Q8_0 [320,10240]",       GGML_TYPE_Q8_0, {320,  10240, 1},     {320,  1},  false },
        { "hc_down M=1  Q8_0 [10240,320]",       GGML_TYPE_Q8_0, {10240,320, 1},       {10240,1},  false },
        { "moe_gate M=1 Q4_K [1280,2560,512]",   GGML_TYPE_Q4_K, {2560, 1280, n_exp}, {2560, 1}, true, n_exp },
        { "moe_down     Q5_1 [640,2560,512]",  GGML_TYPE_Q5_1, {640,  2560, n_exp}, {640,  M},  true, n_exp },
        { "moe_down     Q5_1 [640,2560,10] M=2048", GGML_TYPE_Q5_1, {640, 2560, 10}, {640, M}, true, 10 },
        { "dense Q4_K   [2560,6144]",          GGML_TYPE_Q4_K, {2560, 6144, 1},      {2560, M},  false },
        { "dense Q5_1   [2560,6144]",          GGML_TYPE_Q5_1, {2560, 6144, 1},      {2560, M},  false },
        { "dense Q4_K   [2560,1280]",          GGML_TYPE_Q4_K, {2560, 1280, 1},      {2560, M},  false },
        { "dense Q5_1   [640,2560]",           GGML_TYPE_Q5_1, {640,  2560, 1},      {640,  M},   false },
        { "hc_lora_up   Q8_0 [320,10240]",     GGML_TYPE_Q8_0, {320,  10240, 1},     {320,  M},  false },
        { "hc_lora_down Q8_0 [10240,320]",     GGML_TYPE_Q8_0, {10240,320, 1},       {10240,M},  false },
        { "router       F32  [2560,512]",      GGML_TYPE_F32,  {2560, 512,  1},      {2560, M},  false },
        { "qcur_ref     Q8_0 [2560,6144]",     GGML_TYPE_Q8_0, {2560, 6144, 1},      {2560, M},  false },
        { "iso K=2560 N=10240 Q8_0",            GGML_TYPE_Q8_0, {2560, 10240, 1},     {2560, M},  false },
        { "iso K=320  N=6144  Q8_0",            GGML_TYPE_Q8_0, {320,  6144, 1},      {320,  M},  false },
        { "iso K=10240 N=6144 Q8_0",            GGML_TYPE_Q8_0, {10240,6144, 1},      {10240,M},  false },
        { "dense Q4_K [2560,1280] M=20480",       GGML_TYPE_Q4_K, {2560, 1280, 1},      {2560, 20480}, false },
        { "dense Q5_1 [640,2560]  M=20480",       GGML_TYPE_Q5_1, {640,  2560, 1},      {640,  20480}, false },
        { "moe_fat Q4_K [1280,2560,10] M=2048",    GGML_TYPE_Q4_K, {2560, 1280, 10},     {2560, M}, true  },
        { "moe_fat Q5_1 [640,2560,10]  M=2048",    GGML_TYPE_Q5_1, {640,  2560, 10},     {640,  M},  true  },
        { "moe_smallw Q4_K [256,256,512] M=2048",   GGML_TYPE_Q4_K, {256,  256, 512},     {256,  M},  true, 512 },
        { "moe_smallw Q4_K [1280,2560,512] M=2048 1t", GGML_TYPE_Q4_K, {2560, 1280, 512},  {2560, 2048}, true, 512 },
        { "moe_gate_up  Q4_K M=8192",          GGML_TYPE_Q4_K, {2560, 1280, 128},    {2560, 8192}, true },
        { "moe_down     Q5_1 M=8192",          GGML_TYPE_Q5_1, {640,  2560, 128},    {640,  8192}, true },
    };

    for (const auto & c : cases) {
        ggml_context * ctx = ggml_init({ 16*1024*1024, nullptr, true });
        ggml_tensor * w = ggml_new_tensor_3d(ctx, c.wtype, c.w_ne[0], c.w_ne[1], c.w_ne[2]);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.x_ne[0], c.is_moe ? 1 : c.x_ne[1], c.is_moe ? c.x_ne[1] : 1);
        ggml_tensor * out;
        ggml_tensor * ids = nullptr;
        if (c.is_moe) {
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, c.x_ne[1]);
            out = ggml_mul_mat_id(ctx, w, x, ids);
        } else {
            out = ggml_mul_mat(ctx, w, x);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // fill with pseudo-random bytes (values irrelevant for timing)
        std::mt19937 rng(42);
        std::vector<uint8_t> wb(ggml_nbytes(w)), xb(ggml_nbytes(x));
        for (auto & b : wb) b = (uint8_t) rng();
        for (auto & b : xb) b = (uint8_t) rng();
        ggml_backend_tensor_set(w, wb.data(), 0, wb.size());
        ggml_backend_tensor_set(x, xb.data(), 0, xb.size());
        if (ids) {
            std::vector<int32_t> idv(n_used * c.x_ne[1]);
            for (size_t i = 0; i < idv.size(); i++) idv[i] = rng() % n_exp;
            ggml_backend_tensor_set(ids, idv.data(), 0, idv.size()*4);
        }

        // warmup
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);

        const int n_iter = 30;
        const double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const double dt = (now_ms() - t0) / n_iter;

        const double macs = (double) c.w_ne[0] * c.w_ne[1] * c.x_ne[1] * (c.is_moe ? 1.0 : 1.0);
        const double slots = c.is_moe ? (double) c.x_ne[1] * n_used : (double) c.x_ne[1];
        const double eff_macs = c.is_moe
            ? (double) c.w_ne[0] * c.w_ne[1] * slots
            : macs;
        fprintf(stderr, "%-32s %8.3f ms  %7.1f TFLOPS(eff)\n", c.name, dt, eff_macs / dt / 1e9);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
