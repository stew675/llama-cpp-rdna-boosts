// Per-type column-invariance probe for a quantized MUL_MAT (issue #384's first claim).
//
// For each quantized type: compute dst = mul_mat(src0 [K, M], src1 [K, N]) for N = 1..16 with the
// SAME weights and the SAME first-column data, then compare each column of the N-column result with
// the 1-column result BIT FOR BIT.  The question the delivery's GREEDY-PURITY invariant 1 / §11 asks:
// is the decode/verify band (N <= 8) one numerics class, and does it break at N = 9?
//
// Build: g++ -O2 -std=c++17 probe.cpp -I <repo>/ggml/include -I <repo>/ggml/src \
//          -L <build>/bin -lggml -lggml-base -Wl,-rpath,<build>/bin -o probe
// Run:   GGML_BACKEND_PATH=<build>/bin ./probe [M] [K]

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float * gen_rand_f32(int64_t n, unsigned seed) {
    float * d = (float *) malloc(n * sizeof(float));
    srand(seed);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = (float) rand() / (float) RAND_MAX - 0.5f;
        d[i] *= 2.0f;
    }
    return d;
}

// t is [K, rows]; src is rows*K floats, row-major (row r contiguous)
static void fill_quant(struct ggml_tensor * t, const float * src, int64_t rows, int64_t K, ggml_type type) {
    void * q = malloc(ggml_nbytes(t));
    const float * imatrix = nullptr;
    float * im = nullptr;
    if (ggml_quantize_requires_imatrix(type)) {
        im = (float *) malloc(rows * K * sizeof(float));
        for (int64_t i = 0; i < rows * K; i++) im[i] = 1.0f;
        imatrix = im;
    }
    ggml_quantize_chunk(type, src, q, 0, rows, K, imatrix);
    ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
    free(q);
    free(im);
}

// run one MUL_MAT with a src1 of N columns; returns dst [M*N] (element (m,j) at m + j*M)
static std::vector<float> run_mul_mat(ggml_backend_t backend, ggml_type type,
                                      const float * w, int64_t M, int64_t K,
                                      const float * x, int64_t N) {
    struct ggml_init_params ip = { 1024ull*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, type, K, M);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_cgraph * gf   = ggml_new_graph(ctx);
    struct ggml_tensor * dst  = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    fill_quant(src0, w, M, K, type);
    ggml_backend_tensor_set(src1, x, 0, (size_t) K * N * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out((size_t) M * N);
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    const int64_t M = argc > 1 ? atoll(argv[1]) : 512;   // output rows of the weight
    const int64_t K = argc > 2 ? atoll(argv[2]) : 4096;  // reduction width

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "no GPU backend\n");
        return 2;
    }
    printf("device: %s  (M=%lld K=%lld)\n", ggml_backend_dev_name(dev), (long long) M, (long long) K);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    static const ggml_type types[] = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    };
    const int N_MAX = 16;

    printf("%-8s |", "type");
    for (int n = 1; n <= N_MAX; ++n) printf("%2d ", n);
    printf("| verdict\n");

    int n_band_impure = 0;
    for (ggml_type type : types) {
        float * w = gen_rand_f32(M * K, 0x1234);
        float * x = gen_rand_f32(K * N_MAX, 0x5678);

        std::vector<std::vector<float>> ref(N_MAX + 1); // ref[n] = the N=1 result for column n-1
        for (int n = 1; n <= N_MAX; ++n) {
            ref[n] = run_mul_mat(backend, type, w, M, K, x + (size_t) (n - 1) * K, 1);
        }

        printf("%-8s |", ggml_type_name(type));
        bool impure_in_band = false;
        int first_diff = -1;
        for (int n = 1; n <= N_MAX; ++n) {
            std::vector<float> got = run_mul_mat(backend, type, w, M, K, x, n);
            bool same = true;
            for (int j = 0; j < n && same; ++j) {
                // column j of the n-column result vs the 1-column result of the same data
                if (memcmp(&got[(size_t) j * M], ref[j + 1].data(), (size_t) M * sizeof(float)) != 0) {
                    same = false;
                }
            }
            printf("%2s ", same ? "=" : "X");
            if (!same) {
                if (n <= 8) impure_in_band = true;
                if (first_diff < 0) first_diff = n;
            }
        }
        printf("| first diff at N=%d%s\n", first_diff, impure_in_band ? "  <-- IN-BAND IMPURE" : "");
        if (impure_in_band) n_band_impure++;

        free(w);
        free(x);
    }

    printf("\nlegend: '=' every column equals the same column computed alone; 'X' differs.\n");
    printf("in-band impure types: %d\n", n_band_impure);

    ggml_backend_free(backend);
    return 0;
}
