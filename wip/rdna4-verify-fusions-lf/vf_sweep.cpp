// vf_sweep: geometry sweep for the three verify-step fusions (GLU -> Q8_1, GDN conv at 2..255 tokens, batched copies).
// Builds the model's subgraphs on the CUDA/HIP backend with fixed pseudo-random inputs and prints one FNV-1a hash of all
// outputs per case.  Run it once with the fusions on (default) and once with them off and diff the two outputs:
//
//   ./vf_sweep all > on.txt
//   GGML_CUDA_FUSE_GLU_Q8_1=0 GGML_CUDA_FUSE_GDN_CONV_VERIFY=0 GGML_CUDA_FUSE_CPY_BATCH=0 ./vf_sweep all > off.txt
//   diff on.txt off.txt        # must be empty
//
//   conv C T K   : the shared build_conv_state graph (delta-net-base.cpp) of one GDN layer, n_seqs 1: concat(state,
//                  transpose(qkv)), K snapshot copies of the last 3 columns (K = n_rs_seq + 1; K 0 = the single
//                  last-state copy), ssm_conv, silu.  Hash: silu output + every snapshot slot.
//   glu op F N type E reshape : GLU (op 0 swiglu split, 1 swiglu fused-halves, 2 geglu split, 3 reglu split) over
//                  [F, N] f32, then mul_mat(W [F x E] of `type`, glu) (through an identity reshape when reshape = 1).
//                  Hash: glu output + matmul output.
//
// Build against a llama.cpp build tree:
//   g++ -O2 -std=c++17 -I<tree>/ggml/include vf_sweep.cpp -L<tree>/build/bin -lggml -lggml-base -lggml-hip -o vf_sweep
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint32_t rng_state = 1;
static float frand() { // xorshift32 in [-1, 1)
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (float) (rng_state >> 8) / (float) (1u << 23) - 1.0f;
}
static std::vector<float> rand_vec(size_t n, float scale, uint32_t seed) {
    rng_state = seed * 2654435761u + 12345u;
    std::vector<float> v(n);
    for (auto & x : v) x = frand() * scale;
    return v;
}

static ggml_backend_t be;

struct graph_ctx {
    ggml_context * ctx;
    graph_ctx() {
        ggml_init_params ip = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), nullptr, true };
        ctx = ggml_init(ip);
    }
    ~graph_ctx() { ggml_free(ctx); }
};

static uint64_t hash_tensor(uint64_t h, ggml_tensor * t) {
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    return fnv(h, buf.data(), buf.size());
}

static void run_conv(int64_t C, int64_t T, int64_t K) {
    graph_ctx g;
    ggml_context * ctx = g.ctx;
    const int64_t row_count = 3 * C;
    const int64_t n_slots   = K > 0 ? K + 1 : 1;
    ggml_tensor * states_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_count, n_slots);
    ggml_tensor * state_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, row_count);
    ggml_tensor * qkv        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_tensor * w          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, C);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors(ctx, be);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * conv_states = ggml_reshape_3d(ctx, state_in, 3, C, 1);
    ggml_tensor * qkv_t       = ggml_transpose(ctx, qkv);
    ggml_tensor * conv_input  = ggml_concat(ctx, conv_states, qkv_t, 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_F32, row_count);
    if (K == 0) {
        const int64_t s_idx = conv_input->ne[0] - 3;
        ggml_tensor * last = ggml_view_3d(ctx, conv_input, 3, C, 1, conv_input->nb[1], conv_input->nb[2], ggml_row_size(GGML_TYPE_F32, s_idx));
        ggml_tensor * upd  = ggml_view_2d(ctx, states_all, row_count, 1, states_all->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
    } else {
        const int64_t t_min = std::max<int64_t>(1, K - T + 1);
        for (int64_t t = t_min; t <= K; ++t) {
            const int64_t s_idx  = std::max<int64_t>(0, conv_input->ne[0] - 3 - K + t);
            const int64_t s_slot = K - t;
            ggml_tensor * last = ggml_view_3d(ctx, conv_input, 3, C, 1, conv_input->nb[1], conv_input->nb[2], ggml_row_size(GGML_TYPE_F32, s_idx));
            ggml_tensor * upd  = ggml_view_2d(ctx, states_all, row_count, 1, states_all->nb[1], (size_t) s_slot * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
        }
        if (T < K) {
            ggml_tensor * pre = ggml_reshape_2d(ctx, conv_states, row_count, 1);
            ggml_tensor * dst = ggml_view_2d(ctx, states_all, row_count, 1, states_all->nb[1], (size_t) T * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, pre, dst));
        }
    }
    ggml_tensor * conv = ggml_ssm_conv(ctx, conv_input, w);
    ggml_tensor * out  = ggml_silu(ctx, conv);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);

    auto vs = rand_vec(row_count, 1.0f, (uint32_t) (C * 7 + T));
    auto vq = rand_vec(C * T, 1.0f, (uint32_t) (C * 13 + T));
    auto vw = rand_vec(4 * C, 0.5f, (uint32_t) (C * 17));
    std::vector<float> zero(row_count * n_slots, 0.0f);
    ggml_backend_tensor_set(state_in, vs.data(), 0, ggml_nbytes(state_in));
    ggml_backend_tensor_set(qkv, vq.data(), 0, ggml_nbytes(qkv));
    ggml_backend_tensor_set(w, vw.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(states_all, zero.data(), 0, ggml_nbytes(states_all));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("conv %lld %lld %lld FAIL\n", (long long) C, (long long) T, (long long) K); exit(1); }
    uint64_t h = 1469598103934665603ull;
    h = hash_tensor(h, out);
    h = hash_tensor(h, states_all);
    printf("conv %lld %lld %lld %016llx\n", (long long) C, (long long) T, (long long) K, (unsigned long long) h);
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(inbuf);
}

static void run_glu(int op, int64_t F, int64_t N, ggml_type type, int64_t E, bool reshape, const std::vector<uint8_t> & q) {
    graph_ctx g;
    ggml_context * ctx = g.ctx;
    const bool split = op != 1;
    ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, split ? F : 2 * F, N);
    ggml_tensor * b  = split ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, N) : nullptr;
    ggml_tensor * wd = ggml_new_tensor_2d(ctx, type, F, E);
    ggml_backend_buffer_t inbuf = ggml_backend_alloc_ctx_tensors(ctx, be);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * glu = op == 0 ? ggml_swiglu_split(ctx, a, b) : op == 1 ? ggml_swiglu(ctx, a) :
                        op == 2 ? ggml_geglu_split(ctx, a, b)  : ggml_reglu_split(ctx, a, b);
    ggml_tensor * in  = reshape ? ggml_reshape_2d(ctx, glu, F, N) : glu;
    ggml_tensor * mm  = ggml_mul_mat(ctx, wd, in);
    ggml_build_forward_expand(gf, glu);
    ggml_build_forward_expand(gf, mm);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);

    auto va = rand_vec(ggml_nelements(a), 2.0f, (uint32_t) (F * 3 + N + op));
    ggml_backend_tensor_set(a, va.data(), 0, ggml_nbytes(a));
    if (b) { auto vb = rand_vec(ggml_nelements(b), 2.0f, (uint32_t) (F * 5 + N)); ggml_backend_tensor_set(b, vb.data(), 0, ggml_nbytes(b)); }
    ggml_backend_tensor_set(wd, q.data(), 0, ggml_nbytes(wd));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("glu FAIL\n"); exit(1); }
    uint64_t h = 1469598103934665603ull;
    h = hash_tensor(h, glu);
    h = hash_tensor(h, mm);
    printf("glu %d %lld %lld %s %lld %d %016llx\n", op, (long long) F, (long long) N, ggml_type_name(type), (long long) E, (int) reshape, (unsigned long long) h);
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(inbuf);
}

int main(int argc, char ** argv) {
    const char * which = argc > 1 ? argv[1] : "all";
    be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA/HIP device\n"); return 1; }
    if (!strcmp(which, "all") || !strcmp(which, "conv")) {
        const int64_t Cs[] = { 256, 768, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 384, 1152 }; // the last two: not a multiple of 256, so the fusion must not take them
        for (int64_t C : Cs) {
            for (int64_t T = 1; T <= 300; ++T) {
                const int64_t Ks[] = { 0, 2, 4, 8, 16 };
                for (int64_t K : Ks) {
                    if (C > 4096 && T > 20 && T % 17 != 0 && T != 255 && T != 256 && T != 257) continue; // keep the run short
                    run_conv(C, T, K);
                }
            }
        }
    }
    if (!strcmp(which, "all") || !strcmp(which, "glu")) {
        const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                                    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4 };
        const int64_t Fs[] = { 256, 1536, 4096, 17408 };
        const int64_t Es[] = { 256, 5120 };
        for (ggml_type t : types) for (int64_t F : Fs) for (int64_t E : Es) {
            auto vw = rand_vec(F * E, 0.1f, (uint32_t) (F * 11 + E + t));
            std::vector<uint8_t> q(ggml_row_size(t, F) * E);
            ggml_quantize_chunk(t, vw.data(), q.data(), 0, E, F, nullptr);
            for (int64_t N = 1; N <= 9; ++N) for (int op = 0; op < 4; ++op) {
                for (int r = 0; r < 2; ++r) {
                    if (r == 1 && op != 0) continue;
                    run_glu(op, F, N, t, E, r == 1, q);
                }
            }
        }
    }
    ggml_backend_free(be);
    return 0;
}
