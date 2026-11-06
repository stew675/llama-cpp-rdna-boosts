// norm-rows-alias.cpp -- the driver behind scripts/gate-norm-rows-alias.sh (issue #276).
//
// The narrow-row (norm-rows) fusion reads x, w and the gate output z and writes its own dst, all as
// the dense [ncols, rows] layout its kernel indexes.  The sigmoid-gated form dispatches the gate
// MUL_MAT (`nodes[nm.pre]`) BEFORE the fused launch, so the gate output is written while x is still
// live -- but in graph order x is dead as soon as the RMS_NORM has consumed it, and the gate output
// has exactly x's shape, so the allocator's best fit for it is x's own block.
//
// On the delivery's own path ggml_backend_sched's optimize pass keeps x/w/z alive until the fused
// output (`ggml_backend_cuda_graph_optimize`), and that is what makes the shipped configuration
// correct.  This driver deliberately bypasses the scheduler: it drives ggml_backend_graph_compute()
// directly, which is a supported use of the backend and registers no alloc deps at all, so the
// fusion's own aliasing guard (`norm_rows_alias_conflict()` in norm-gated.cu) is the only thing
// standing between the graph and a wrong answer.  Each aliasing arm is produced by pointing a tensor
// at another one's buffer AFTER the allocation, which is exactly what the allocator's graph-order
// reuse does (measured on Ling-3.0-tiny: an exact, full-block alias).
//
// Arms (all on the gated form's exact shape, RMS_NORM -> MUL(w) -> MUL_MAT -> SIGMOID -> MUL):
//   ref              no forced alias: the fusion must run and its output is the reference
//   alias_gate       gate output == x's block          -> the fusion must DECLINE, output unchanged
//   alias_dst_row    fused output == x's block + 1 row -> the fusion must DECLINE (partial overlap:
//                    the launch would write rows other waves have not read yet)
//   alias_dst_exact  fused output == x's block exactly -> the fusion must RUN (a read-then-write of the
//                    same address by the same thread) and still reproduce the reference
//
// Prints one `arm=` line per arm plus `RESULT PASS|FAIL`; exit status is 0 only on PASS.  The digests
// are stable across processes so the gate can also compare this driver's fused arms against a
// --no-fuse run of it (the unfused chain) byte for byte.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fused = 0;
static int g_declined = 0;

static void log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (std::strncmp(text, "norm-rows:", 10) == 0) {
        if (std::strstr(text, "fused=1")) {
            g_fused++;
        } else if (std::strstr(text, "declined=")) {
            g_declined++;
            std::fputs(text, stderr);
        }
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    bool no_fuse = false;
    bool dump    = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-fuse") == 0) {
            no_fuse = true;
        } else if (std::strcmp(argv[i], "--dump") == 0) {
            dump = true;
        } else {
            std::fprintf(stderr, "usage: %s [--no-fuse] [--dump]\n", argv[0]);
            return 2;
        }
    }
    if (no_fuse) {
        setenv("GGML_CUDA_DISABLE_NORM_ROWS", "1", 1);
    }
    // Diagnostic only: the fusion's decision line is what makes "declined" an observation instead of
    // an inference, and it never changes what is computed.
    setenv("GGML_CUDA_NORM_ROWS_DEBUG", "1", 1);
    ggml_log_set(log_cb, nullptr);

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d;
            break;
        }
    }
    if (dev == nullptr) {
        std::fprintf(stderr, "FAIL: no GPU backend registered (link the backend lib, e.g. -lggml-hip)\n");
        return 2;
    }
    std::fprintf(stderr, "device: %s fuse=%s\n", ggml_backend_dev_name(dev), no_fuse ? "off" : "on");
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 2;
    }

    // The fusion's shape gate: ne0 <= 256 (multiple of 32), rows >= 4096, F32 and contiguous.
    const int    ncols = 128;
    const int    nrows = 8192;
    const size_t n     = (size_t) ncols * nrows;

    ggml_init_params ip = { 32u * 1024u * 1024u, nullptr, /*.no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);       // the norm's input
    struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ncols);             // the norm's weight
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, ncols);      // the gate projection
    struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);      // the gate's input
    ggml_set_name(x, "x");
    ggml_set_name(w, "w");
    ggml_set_name(W, "gate_w");
    ggml_set_name(y, "gate_in");

    struct ggml_tensor * rms = ggml_rms_norm(ctx, x, 1e-5f);   ggml_set_name(rms, "rms");
    struct ggml_tensor * mul = ggml_mul(ctx, rms, w);          ggml_set_name(mul, "normed");
    struct ggml_tensor * pre = ggml_mul_mat(ctx, W, y);        ggml_set_name(pre, "gate_out");
    struct ggml_tensor * sig = ggml_sigmoid(ctx, pre);         ggml_set_name(sig, "gate_sig");
    struct ggml_tensor * out = ggml_mul(ctx, mul, sig);        ggml_set_name(out, "out");
    ggml_set_output(out);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    std::fprintf(stderr, "graph: %d nodes\n", ggml_graph_n_nodes(gf));
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const struct ggml_tensor * t = ggml_graph_node(gf, i);
        std::fprintf(stderr, "  %d %s %s\n", i, ggml_op_name(t->op), t->name);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "FAIL: tensor allocation failed\n");
        return 2;
    }

    std::vector<float> hx(n), hw(ncols), hW((size_t) ncols * ncols), hy(n);
    for (size_t i = 0; i < n; ++i) {
        hx[i] = 1.0f + 0.5f * std::sin((float) i * 0.0007f);
        hy[i] = 0.3f * std::cos((float) i * 0.0011f);
    }
    for (int i = 0; i < ncols; ++i) {
        hw[i] = 1.0f + 0.01f * std::sin((float) i);
    }
    for (size_t i = 0; i < (size_t) ncols * ncols; ++i) {
        hW[i] = 0.05f * std::sin((float) i * 0.37f);
    }

    std::vector<float> ref(n), got(n);

    // `force_gate` / `force_out` are the allocator's reuse, made explicit: point that tensor at another
    // tensor's buffer for the duration of one compute (the arm re-sets its inputs first, so a previous
    // arm's writes cannot leak in).
    int fails = 0;
    auto run = [&](const char * arm, void * force_gate, void * force_out, int want_fused, int want_declined) {
        ggml_backend_tensor_set(x, hx.data(), 0, n * sizeof(float));
        ggml_backend_tensor_set(w, hw.data(), 0, ncols * sizeof(float));
        ggml_backend_tensor_set(W, hW.data(), 0, (size_t) ncols * ncols * sizeof(float));
        ggml_backend_tensor_set(y, hy.data(), 0, n * sizeof(float));

        void * gate_save = pre->data;
        void * out_save  = out->data;
        if (force_gate != nullptr) { pre->data = force_gate; }
        if (force_out  != nullptr) { out->data = force_out; }

        g_fused = 0;
        g_declined = 0;
        const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(out, got.data(), 0, n * sizeof(float));

        pre->data = gate_save;
        out->data = out_save;

        if (st != GGML_STATUS_SUCCESS) {
            std::printf("arm=%-16s FAIL compute status %d\n", arm, (int) st);
            fails++;
            return;
        }
        if (std::strcmp(arm, "ref") == 0) {
            ref = got;
        }
        const bool match = std::memcmp(ref.data(), got.data(), n * sizeof(float)) == 0;
        const bool ok    = match && g_fused == want_fused && g_declined == want_declined;
        if (!ok) {
            fails++;
        }
        std::printf("arm=%-16s fused=%d declined=%d ref_match=%s expect=fused:%d,declined:%d %s digest=%016llx\n",
                arm, g_fused, g_declined, match ? "yes" : "NO", want_fused, want_declined,
                ok ? "ok" : "FAIL", (unsigned long long) fnv1a(got.data(), n * sizeof(float)));
        if (dump) {
            for (size_t i = 0; i < 4; ++i) {
                std::printf("    [%zu] %a\n", i, got[i]);
            }
        }
    };

    std::printf("ptrs x=%p w=%p gate_out=%p out=%p\n", x->data, w->data, pre->data, out->data);

    // With the fusion disabled every arm is the plain chain, which is correct under all of these
    // aliases (in graph order x is consumed before the gate output is written): the arms are then a
    // cross-process reference for the fused run, not a guard test.
    const int f = no_fuse ? 0 : 1;
    const int d = no_fuse ? 0 : 1;
    run("ref",             nullptr,                        nullptr,                              f, 0);
    run("alias_gate",      x->data,                        nullptr,                              0, d);
    run("alias_dst_row",   nullptr, (char *) x->data + ncols * sizeof(float),                     0, d);
    run("alias_dst_exact", nullptr, x->data,                                                      f, 0);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    std::printf("RESULT %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
