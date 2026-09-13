// fattn-probe.cpp — masked-column stale-K/V probe for ggml FLASH_ATTN_EXT.
//
// Asks: does the FA implementation on a given backend leak the *contents* of
// cells that are masked out (P == +0.0) into the output? A correct kernel must
// produce bit-identical outputs whether the fully-masked tail of the KV cache
// holds +0.0 or garbage of arbitrary magnitude/sign.
//
// usage:
//   fattn-probe <backend-lib> <ktype f16|bf16> <nq> <kv> <n_valid> [trials]
//   backend-lib: absolute path to libggml-hip.so or libggml-vulkan.so
//
// Outputs, per trial: LEAK (outputs differ), OK (identical), or DET-<n> if two
// identical-input computes differ (kernel nondeterminism).
//
// Build (against one build dir's ggml libs; run with LD_LIBRARY_PATH=that dir):
//   g++ -O2 -std=c++17 fattn-probe.cpp -I<llama>/ggml/include \
//       -L<build>/bin -lggml-base -lggml-cpu -ldl -o fattn-probe
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <dlfcn.h>

// ---------------------------------------------------------------------------
typedef ggml_backend_t (*init_fn)(int);
typedef int (*count_fn)(void);

static void * g_lib = nullptr;
static init_fn g_init = nullptr;
static count_fn g_count = nullptr;

// load the backend .so; returns device count via its own exported counter
static int load_backend(const char * lib_path, const char * init_sym, const char * count_sym) {
    g_lib = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!g_lib) { fprintf(stderr, "dlopen %s: %s\n", lib_path, dlerror()); exit(2); }
    g_init  = (init_fn)  dlsym(g_lib, init_sym);
    g_count = (count_fn) dlsym(g_lib, count_sym);
    if (!g_init || !g_count) { fprintf(stderr, "dlsym %s/%s failed\n", init_sym, count_sym); exit(2); }
    return g_count();
}

static ggml_fp16_t f16(float x) { return ggml_fp32_to_fp16(x); }
static float f16f(ggml_fp16_t h) { return ggml_fp16_to_fp32(h); }

// build a graph: out = FLASH_ATTN_EXT(q, k, v, mask)
// q: f32 [hsk, nq], k: f16/bf16 [hsk, kv], v: f16/bf16 [hsv, kv], mask f16 [kv, nq]
struct fa_case {
    int64_t hsk, hsv, nq, kv;
    ggml_type kv_type;
    std::vector<float> q;                 // f32
    std::vector<ggml_fp16_t> k, v_zero;   // f16 storage (converted)
    std::vector<ggml_fp16_t> v_garb;      // masked tail garbage
    std::vector<ggml_fp16_t> mask;        // f16, 0 / -inf
    std::vector<ggml_bf16_t> k_b, v_zero_b, v_garb_b;
    std::vector<ggml_bf16_t> mask16;      // placeholder unused for bf16 path below
};

static float rnd(unsigned * s) { // xorshift, uniform (-1,1)
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return ((float)(*s & 0xFFFFFF) / 8388608.0f) - 1.0f;
}

static void init_case(fa_case & c, unsigned seed) {
    c.hsk = c.hsv = 128;
    const int64_t nq = c.nq, kv = c.kv;
    const int64_t n_valid = kv > c.hsk ? kv / 2 : kv;   // caller sets c.kv already scaled; see main
    (void)n_valid;
    c.q.assign((size_t)c.hsk * nq, 0.f);
    c.k.assign((size_t)c.hsk * kv, 0);
    c.v_zero.assign((size_t)c.hsv * kv, 0);
    c.v_garb.assign((size_t)c.hsv * kv, 0);
    c.mask.assign((size_t)kv * nq, 0);
    unsigned s = seed;
    for (auto & x : c.q) x = rnd(&s);
    for (auto & x : c.k) x = f16(2.0f * rnd(&s));
    for (int64_t i = 0; i < c.hsv * kv; ++i) {
        // realistic-ish magnitudes in the valid part, wild garbage in the masked tail
        c.v_zero[i] = f16(0.0f);                    // never used for valid cols except A/B baseline
        c.v_garb[i] = f16(2.0f * rnd(&s));
    }
    // build bf16 mirrors if needed
    if (c.kv_type == GGML_TYPE_BF16) {
        c.k_b.resize(c.k.size()); c.v_zero_b.resize(c.v_zero.size()); c.v_garb_b.resize(c.v_garb.size());
        for (size_t i = 0; i < c.k.size(); ++i) c.k_b[i]  = ggml_fp32_to_bf16(f16f(c.k[i]));
        for (size_t i = 0; i < c.v_zero.size(); ++i) c.v_zero_b[i] = ggml_fp32_to_bf16(f16f(c.v_zero[i]));
        for (size_t i = 0; i < c.v_garb.size(); ++i) c.v_garb_b[i] = ggml_fp32_to_bf16(f16f(c.v_garb[i]));
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <lib> <f16|bf16> <nq> <kv> <n_valid> [hsk=128] [nh=1] [nh_kv=1]\n", argv[0]);
        return 2;
    }
    const char * lib = argv[1];
    const char * tstr = argv[2];
    const bool bf16  = strcmp(tstr, "bf16") == 0;
    const bool quant = !strcmp(tstr, "q8_0") || !strcmp(tstr, "q4_0") || !strcmp(tstr, "q4_1") ||
                       !strcmp(tstr, "q5_0") || !strcmp(tstr, "q5_1") || !strcmp(tstr, "iq4_nl");
    ggml_type kv_type;
    if (bf16)      kv_type = GGML_TYPE_BF16;
    else if (quant) {
        if (!strcmp(tstr, "q8_0")) kv_type = GGML_TYPE_Q8_0;
        else if (!strcmp(tstr, "q4_0")) kv_type = GGML_TYPE_Q4_0;
        else if (!strcmp(tstr, "q4_1")) kv_type = GGML_TYPE_Q4_1;
        else if (!strcmp(tstr, "q5_0")) kv_type = GGML_TYPE_Q5_0;
        else if (!strcmp(tstr, "q5_1")) kv_type = GGML_TYPE_Q5_1;
        else kv_type = GGML_TYPE_IQ4_NL;
    } else            kv_type = GGML_TYPE_F16;
    const int64_t nq = atoll(argv[3]);
    const int64_t kv = atoll(argv[4]);
    const int64_t n_valid = atoll(argv[5]);
    const bool is_hip = strstr(lib, "hip") != nullptr;

    int ndev = load_backend(lib,
        is_hip ? "ggml_backend_cuda_init" : "ggml_backend_vk_init",
        is_hip ? "ggml_backend_cuda_get_device_count" : "ggml_backend_vk_get_device_count");
    if (ndev < 1) { fprintf(stderr, "no device\n"); return 2; }
    ggml_backend_t backend = g_init(0);
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 2; }
    fprintf(stderr, "backend: %s device0\n", ggml_backend_name(backend));

    const int64_t hsk = argc > 6 ? atoll(argv[6]) : 128;
    const int64_t hs  = hsk; // hsv == hsk for this probe
    const int64_t nh   = argc > 7 ? atoll(argv[7]) : 1;    // q heads
    const int64_t nh_kv= argc > 8 ? atoll(argv[8]) : 1;    // kv heads

    // tensors
    struct ggml_init_params ip = {
        /*.mem_size   =*/ 64ull*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, nq, nh);
    ggml_tensor * k     = ggml_new_tensor_3d(ctx, kv_type,       hs, kv, nh_kv);
    ggml_tensor * v     = ggml_new_tensor_3d(ctx, kv_type,       hs, kv, nh_kv);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nq);
    ggml_tensor * out   = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf((float)hs), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);  // match llama.cpp decode/prefill (f32acc)
    ggml_set_name(out, "out");
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(mask);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    fprintf(stderr, "supports_op(FLASH_ATTN_EXT)=%d\n", (int)ggml_backend_supports_op(backend, out));
    if (!ggml_backend_supports_op(backend, out)) {
        printf("%s %s nq=%lld kv=%lld nvalid=%lld hsk=%lld nh=%lld nh_kv=%lld  -> SKIP (unsupported kv type for FA)\n",
            is_hip ? "hip" : "vk", tstr, (long long)nq, (long long)kv, (long long)n_valid,
            (long long)hsk, (long long)nh, (long long)nh_kv);
        fflush(stdout);
        ggml_free(ctx);
        ggml_backend_free(backend);
        dlclose(g_lib);
        return 0;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); return 2; }

    std::vector<ggml_fp16_t> q16(hs*nq*nh);
    for (size_t i = 0; i < q16.size(); ++i) q16[i] = f16(0.5f*((float)(i%17)/8.0f - 1.0f));

    // Q data
    std::vector<float> qdata(hs*nq*nh);
    unsigned s = 12345;
    for (auto & x : qdata) x = rnd(&s);
    ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size()*4);

    // mask: 0 for c < n_valid, -inf for c >= n_valid (fully masked tail)
    std::vector<ggml_fp16_t> maskdata((size_t)kv*nq, f16(-INFINITY));
    for (int64_t r = 0; r < nq; ++r)
        for (int64_t c = 0; c < n_valid && c < kv; ++c)
            maskdata[(size_t)r*kv + c] = f16(0.0f);

    // K: valid content for c < n_valid, +0.0 for c >= n_valid (K must not matter)
    std::vector<ggml_fp16_t> kdata((size_t)hs*kv, f16(0.0f));
    for (int64_t c = 0; c < n_valid && c < kv; ++c)
        for (int64_t d = 0; d < hs; ++d)
            kdata[(size_t)c*hs + d] = f16(rnd(&s)*2.0f);

    // V runs: mode A = +0 tail, mode B = garbage tail (valid cells identical)
    auto make_v = [&]() {
        std::vector<ggml_fp16_t> vdata((size_t)hs*kv, f16(0.0f));
        for (int64_t c = 0; c < n_valid && c < kv; ++c)
            for (int64_t d = 0; d < hs; ++d)
                vdata[(size_t)c*hs + d] = f16(rnd(&s)*2.0f);
        return vdata;
    };

    auto garb_v = [&](const std::vector<ggml_fp16_t> & valid) {
        std::vector<ggml_fp16_t> vdata = valid;
        for (int64_t c = n_valid; c < kv; ++c)
            for (int64_t d = 0; d < hs; ++d) {
                float x = (float)((c*131 + d*17 + 7) % 101 - 50) / 100.0f; // -0.5..0.5 (realistic)
                if (((c+d) & 1) == 0) x = -x;
                vdata[(size_t)c*hs + d] = f16(x);
            }
        return vdata;
    };

    auto upload = [&](const std::vector<ggml_fp16_t> & kd, const std::vector<ggml_fp16_t> & vd, const std::vector<ggml_fp16_t> & md) {
        // replicate the single-head plane across nh_kv heads
        const size_t plane = (size_t)hs*kv;
        const size_t rowbytes = quant ? ggml_row_size(kv_type, (int32_t)hs) : (size_t)hs*2;
        std::vector<uint8_t> kq, vq;
        if (quant) {
            std::vector<float> f32(plane);
            auto do_quant = [&](const std::vector<ggml_fp16_t> & src, std::vector<uint8_t> & dst) {
                dst.resize(rowbytes*kv);
                for (int64_t c = 0; c < kv; ++c) {
                    for (int64_t d = 0; d < hs; ++d) f32[c*hs + d] = f16f(src[c*hs + d]);
                }
                ggml_quantize_chunk(kv_type, f32.data(), dst.data(), 0, kv, hs, nullptr);
            };
            do_quant(kd, kq); do_quant(vd, vq);
        }
        for (int64_t h = 0; h < nh_kv; ++h) {
            if (quant) {
                fprintf(stderr, "quant set k h=%lld off=%zu size=%zu nb=%zu | v off=%zu size=%zu nb=%zu\n",
                    (long long)h, (size_t)h*kv*rowbytes, (size_t)kv*rowbytes, (size_t)ggml_nbytes(k),
                    (size_t)h*kv*rowbytes, (size_t)kv*rowbytes, (size_t)ggml_nbytes(v));
                ggml_backend_tensor_set(k, kq.data() + h*kv*rowbytes, h*kv*rowbytes, kv*rowbytes);
                ggml_backend_tensor_set(v, vq.data() + h*kv*rowbytes, h*kv*rowbytes, kv*rowbytes);
            } else if (kv_type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> kbb(kd.size()), vbb(vd.size());
                for (size_t i = 0; i < kd.size(); ++i) kbb[i] = ggml_fp32_to_bf16(f16f(kd[i]));
                for (size_t i = 0; i < vd.size(); ++i) vbb[i] = ggml_fp32_to_bf16(f16f(vd[i]));
                ggml_backend_tensor_set(k, kbb.data(), h*plane*2, plane*2);
                ggml_backend_tensor_set(v, vbb.data(), h*plane*2, plane*2);
            } else {
                ggml_backend_tensor_set(k, kd.data(), h*plane*2, plane*2);
                ggml_backend_tensor_set(v, vd.data(), h*plane*2, plane*2);
            }
        }
        ggml_backend_tensor_set(mask, md.data(), 0, md.size()*2);
    };

    auto compute = [&](std::vector<float> & o) {
        o.assign((size_t)hs*nq*nh, 0.f);
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(st)); exit(2); }
        ggml_backend_tensor_get(out, o.data(), 0, o.size()*4);
    };

    std::vector<ggml_fp16_t> vA = make_v();        // zeroed tail
    std::vector<ggml_fp16_t> vB = garb_v(vA);      // garbage tail (valid == vA)
    // garbage K tail variant (stale K, like the zeroing-off server cache)
    std::vector<ggml_fp16_t> kB = kdata;
    for (int64_t c = n_valid; c < kv; ++c)
        for (int64_t d = 0; d < hs; ++d) {
            float x = (float)((c*97 + d*11 + 3) % 1001 - 500) / 100.0f; // -5..5
            if (((c+d) & 1) == 0) x = -x;
            kB[(size_t)c*hs + d] = f16(x);
        }
    std::vector<ggml_fp16_t> kz(kdata.size(), f16(0.0f));

    // control: run A twice (determinism), then A vs B (leak)
    // A = zero K tail + zero V tail ; B = (env FPROBE_KGARB!=0 ? garbage K : zero) tail + (env FPROBE_VGARB!=0 ? garbage V : zero) tail
    const bool kgarb = getenv("FPROBE_KGARB") == nullptr || atoi(getenv("FPROBE_KGARB"));
    const bool vgarb = getenv("FPROBE_VGARB") == nullptr || atoi(getenv("FPROBE_VGARB"));
    std::vector<float> o1, o2, o3;
    upload(kdata, vA, maskdata); compute(o1);
    upload(kdata, vA, maskdata); compute(o2);
    upload(kgarb ? kB : kdata, vgarb ? vB : vA, maskdata); compute(o3);

    int det_diff = 0, leak_diff = 0;
    float leak_max = 0.f, leak_pos = -1;
    for (size_t i = 0; i < o1.size(); ++i) {
        if (o1[i] != o2[i]) det_diff++;
        if (o1[i] != o3[i]) {
            leak_diff++;
            float d = fabsf(o1[i] - o3[i]);
            if (d > leak_max) { leak_max = d; leak_pos = (float)i; }
        }
    }

    const char * verdict;
    if (det_diff) {
        verdict = "DET";
    } else if (leak_diff) {
        verdict = "LEAK";
    } else {
        verdict = "OK";
    }
    printf("%s %s nq=%lld kv=%lld nvalid=%lld hsk=%lld nh=%lld nh_kv=%lld  -> %s (nz_diff=%d max=%.3e @%d)\n",
        is_hip ? "hip" : "vk", tstr, (long long)nq, (long long)kv, (long long)n_valid, (long long)hsk,
        (long long)nh, (long long)nh_kv, verdict, leak_diff, leak_max, (int)leak_pos);
    fflush(stdout);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    dlclose(g_lib);
    return 0;
}
