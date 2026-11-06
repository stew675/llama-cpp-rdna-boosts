// norm-rows-alias.cpp -- the driver behind scripts/gate-norm-rows-alias.sh (issue #276).
//
// The narrow-row (norm-rows) fusion reads x, w and (for the sigmoid-gated form) the gate output z, and
// writes its own dst, all as the dense [ncols, rows] layout its kernel indexes.  The gated form dispatches
// the gate MUL_MAT (`nodes[nm.pre]`) BEFORE the fused launch, so the gate output is written while x is still
// live -- but in graph order x is dead as soon as the RMS_NORM has consumed it, and the gate output has
// exactly x's shape, so the allocator's best fit for it is x's own block.
//
// On the delivery's own path ggml_backend_sched's optimize pass keeps x/w/z alive until the fused output
// (`ggml_backend_cuda_graph_optimize`), and that is what makes the shipped configuration correct.  This
// driver deliberately bypasses the scheduler: it drives ggml_backend_graph_compute() directly, which is a
// supported use of the backend and registers no alloc deps at all, so the fusion's own aliasing guard
// (`norm_rows_alias_conflict()` in norm-gated.cu) is the only thing standing between the graph and a wrong
// answer.  Each aliasing arm is produced by pointing a tensor at another one's buffer AFTER the allocation,
// which is exactly what the allocator's graph-order reuse does (measured on Ling-3.0-tiny: an exact,
// full-block alias on all 180 gated nodes).
//
// Two graph shapes, each with its own arms:
//
//   gated  (RMS_NORM -> MUL(w) -> MUL_MAT -> SIGMOID -> MUL, the qwen4exp / bailingmoe3 per-head gate)
//     ref              no forced alias: the fusion must run, and its output is the reference
//     alias_gate       gate output == x's block          -> must DECLINE
//     alias_dst_row    fused output == x's block + 1 row -> must DECLINE (partial overlap: the launch would
//                      write rows other waves have not read yet)
//     alias_dst_exact  fused output == x's block exactly -> must RUN (a read-then-write of the same address
//                      by the same thread) and still reproduce ref
// Only the GATED shape is driven here.  The plain form's arms would need upstream's own
// `RMS_NORM + MUL` fusion (`ggml_cuda_can_fuse`) disabled as well, and only the master
// `GGML_CUDA_DISABLE_FUSION` reaches it -- which would also disable the fusion under test, so there is no
// reference to compare against inside one process.  The guard is the same function either way, and the
// exact-alias exception it makes for the plain form is exercised by `alias_dst_exact` below.
//
// Prints one `case=... arm=...` line per arm plus `RESULT PASS|FAIL`; exit status is 0 only on PASS.  The
// digests are stable across processes, so the gate also compares each case's fused output against a
// --no-fuse run of this driver (the unfused chain) byte for byte.

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

// The fusion's shape gate: ne0 <= 256 (multiple of 32), rows >= 4096, F32 and contiguous.
static const int    NCOLS = 128;
static const int    NROWS = 8192;
static const size_t N     = (size_t) NCOLS * NROWS;

struct input_t {
    struct ggml_tensor * t;
    const void * data;
    size_t bytes;
};

struct case_t {
    const char * name;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buf;
    struct ggml_tensor * arena;  // slack around x, so a forced partial alias stays inside one allocation
    struct ggml_tensor * x;
    struct ggml_tensor * w;
    struct ggml_tensor * pre;   // the gate output
    struct ggml_tensor * dst;
    struct ggml_cgraph * gf;
    std::vector<input_t> inputs;
    std::vector<float> ref, got;
};

static void build_case(ggml_backend_t backend, case_t & cs) {
    ggml_init_params ip = { 32u * 1024u * 1024u, nullptr, /*.no_alloc =*/ true };
    cs.ctx = ggml_init(ip);
    cs.name = "gated";

    // N + one row of slack: the allocator's own reuse always lives inside an arena, so an arm that forces a
    // partial overlap must too.  Writing one row past the end of x's own block would clobber whatever tensor
    // the linear context allocation put behind it (w), which would make the arm an out-of-bounds write
    // instead of an aliasing test.
    cs.arena = ggml_new_tensor_1d(cs.ctx, GGML_TYPE_F32, (int64_t) (N + NCOLS));
    ggml_set_name(cs.arena, "arena");
    cs.x = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, NCOLS, NROWS);   // the norm's input
    cs.w = ggml_new_tensor_1d(cs.ctx, GGML_TYPE_F32, NCOLS);          // the norm's weight
    ggml_set_name(cs.x, "x");
    ggml_set_name(cs.w, "w");

    struct ggml_tensor * rms = ggml_rms_norm(cs.ctx, cs.x, 1e-5f);
    ggml_set_name(rms, "rms");
    struct ggml_tensor * mul = ggml_mul(cs.ctx, rms, cs.w);
    ggml_set_name(mul, "normed");

    std::vector<float> hx(N), hw(NCOLS);
    for (size_t i = 0; i < N; ++i) {
        hx[i] = 1.0f + 0.5f * std::sin((float) i * 0.0007f);
    }
    for (int i = 0; i < NCOLS; ++i) {
        hw[i] = 1.0f + 0.01f * std::sin((float) i);
    }
    static std::vector<float> sx, sw;   // the host arrays must outlive the case
    if (sx.empty()) {
        sx = hx;
        sw = hw;
    }

    {
        struct ggml_tensor * W = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, NCOLS, NCOLS);
        struct ggml_tensor * y = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, NCOLS, NROWS);
        ggml_set_name(W, "gate_w");
        ggml_set_name(y, "gate_in");
        cs.pre = ggml_mul_mat(cs.ctx, W, y);
        ggml_set_name(cs.pre, "gate_out");
        struct ggml_tensor * sig = ggml_sigmoid(cs.ctx, cs.pre);
        ggml_set_name(sig, "gate_sig");
        cs.dst = ggml_mul(cs.ctx, mul, sig);
        ggml_set_name(cs.dst, "out");

        static std::vector<float> sW, sy;
        if (sW.empty()) {
            sW.resize((size_t) NCOLS * NCOLS);
            sy.resize(N);
            for (size_t i = 0; i < sW.size(); ++i) {
                sW[i] = 0.05f * std::sin((float) i * 0.37f);
            }
            for (size_t i = 0; i < N; ++i) {
                sy[i] = 0.3f * std::cos((float) i * 0.0011f);
            }
        }
        cs.inputs.push_back({ cs.x,  sx.data(),  N * sizeof(float) });
        cs.inputs.push_back({ cs.w,  sw.data(),  NCOLS * sizeof(float) });
        cs.inputs.push_back({ W,     sW.data(),  sW.size() * sizeof(float) });
        cs.inputs.push_back({ y,     sy.data(),  N * sizeof(float) });
    }
    ggml_set_output(cs.dst);

    cs.gf = ggml_new_graph(cs.ctx);
    ggml_build_forward_expand(cs.gf, cs.dst);
    std::fprintf(stderr, "case %s: %d nodes\n", cs.name, ggml_graph_n_nodes(cs.gf));
    for (int i = 0; i < ggml_graph_n_nodes(cs.gf); ++i) {
        const struct ggml_tensor * t = ggml_graph_node(cs.gf, i);
        std::fprintf(stderr, "  %d %s %s\n", i, ggml_op_name(t->op), t->name);
    }

    cs.buf = ggml_backend_alloc_ctx_tensors(cs.ctx, backend);
    if (cs.buf == nullptr) {
        std::fprintf(stderr, "FAIL: tensor allocation failed (case %s)\n", cs.name);
        std::exit(2);
    }
    // x lives in the arena (which has one row of slack after it), so every forced alias below stays inside
    // one allocated block, exactly as a gallocr block reuse does.
    if (cs.arena->data == nullptr || cs.arena->nb[0] != sizeof(float)) {
        std::fprintf(stderr, "FAIL: arena is not a contiguous F32 block (case %s)\n", cs.name);
        std::exit(2);
    }
    cs.x->data = cs.arena->data;
    cs.ref.assign(N, 0.0f);
    cs.got.assign(N, 0.0f);
}

// `force_pre` / `force_dst` are the allocator's reuse, made explicit: point that tensor at another tensor's
// buffer for the duration of one compute.  The inputs are re-set first, so a previous arm's writes cannot
// leak into this one.
static void run_arm(ggml_backend_t backend, case_t & cs, const char * arm, void * force_pre,
                    void * force_dst, int want_fused, int want_declined, int & fails) {
    for (const input_t & in : cs.inputs) {
        ggml_backend_tensor_set(in.t, in.data, 0, in.bytes);
    }
    void * pre_save = cs.pre ? cs.pre->data : nullptr;
    void * dst_save = cs.dst->data;
    if (cs.pre && force_pre) { cs.pre->data = force_pre; }
    if (force_dst)           { cs.dst->data = force_dst; }

    g_fused = 0;
    g_declined = 0;
    const enum ggml_status st = ggml_backend_graph_compute(backend, cs.gf);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(cs.dst, cs.got.data(), 0, N * sizeof(float));

    if (cs.pre) { cs.pre->data = pre_save; }
    cs.dst->data = dst_save;

    if (st != GGML_STATUS_SUCCESS) {
        std::printf("case=%-5s arm=%-16s FAIL compute status %d\n", cs.name, arm, (int) st);
        fails++;
        return;
    }
    if (std::strcmp(arm, "ref") == 0) {
        cs.ref = cs.got;
    }
    const bool match = std::memcmp(cs.ref.data(), cs.got.data(), N * sizeof(float)) == 0;
    const bool ok    = match && g_fused == want_fused && g_declined == want_declined;
    if (!ok) {
        fails++;
    }
    std::printf("case=%-5s arm=%-16s fused=%d declined=%d ref_match=%s expect=fused:%d,declined:%d %s"
                " digest=%016llx\n",
                cs.name, arm, g_fused, g_declined, match ? "yes" : "NO", want_fused, want_declined,
                ok ? "ok" : "FAIL", (unsigned long long) fnv1a(cs.got.data(), N * sizeof(float)));
}

int main(int argc, char ** argv) {
    bool no_fuse = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-fuse") == 0) {
            no_fuse = true;
        } else {
            std::fprintf(stderr, "usage: %s [--no-fuse]\n", argv[0]);
            return 2;
        }
    }
    if (no_fuse) {
        setenv("GGML_CUDA_DISABLE_NORM_ROWS", "1", 1);
    }
    // Diagnostic only: the fusion's decision line is what makes "declined" an observation instead of an
    // inference, and it never changes what is computed.
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

    int fails = 0;
    const int f = no_fuse ? 0 : 1;   // want_fused on a plain "no forced alias" arm
    const int d = no_fuse ? 0 : 1;   // want_declined on a forced-conflict arm

    case_t gated;
    build_case(backend, gated);
    auto row_after_x = [](const case_t & cs) { return (void *) ((char *) cs.x->data + NCOLS * sizeof(float)); };

    std::printf("ptrs arena=%p x=%p gate_out=%p out=%p\n",
                gated.arena->data, gated.x->data, gated.pre->data, gated.dst->data);

    run_arm(backend, gated, "ref",             nullptr,             nullptr, f, 0, fails);
    run_arm(backend, gated, "alias_gate",      gated.x->data,       nullptr, 0, d, fails);
    run_arm(backend, gated, "alias_dst_row",   nullptr, row_after_x(gated), 0, d, fails);
    run_arm(backend, gated, "alias_dst_exact", nullptr, gated.x->data,       f, 0, fails);

    ggml_backend_buffer_free(gated.buf);
    ggml_free(gated.ctx);
    ggml_backend_free(backend);

    std::printf("RESULT %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
