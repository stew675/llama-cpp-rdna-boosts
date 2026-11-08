// Standalone HIP probe: is the copy-engine `hipMemcpyAsync` H2D the fastest way to fill one expert
// table, and what does it cost in DMA issues / interference?  No llama.cpp, no model, `hipcc` only.
//
// Shapes are the field ones (Qwen3.8-Flash-Next GSQ-RCO-IQ3_S on one R9700):
//   table       512 experts x 681574 B  = 333 MiB   (one MUL_MAT_ID src0)
//   complement  290 of 512 experts (43.4 % arena residency) in 126 runs, avg 2.3 experts/run
// The delivery's staged fill issues ONE copy per expert for those 290 (today) or, if the runs were
// coalesced, one per run (126).  This measures what each shape actually costs.
//
// Tests
//   A  one 333 MiB copy, per host-memory kind (pinned / pageable / registered-mmap / write-combined)
//   B  the complement as 290 per-expert copies        (today's per-expert fill)
//   C  the complement as 126 run copies               (run-coalesced fill)
//   D  C split over 2 copy streams                    (is the per-copy cost a serialized dispatch?)
//   E  one copy of the complement's total bytes       (the bytes-only lower bound)
//   F  a hand-written vectorised PULL kernel, CTA sweep (the 33 GB/s-vs-28.6 GB/s hypothesis)
//   G  interference: a compute kernel alone, and beside the DMA / beside the pull kernel
//
// Build (see the campaign README):
//   hipcc -O2 --offload-arch=gfx1201 -o h2d-probe h2d-probe.cpp
// Run: HIP_VISIBLE_DEVICES=0 ./h2d-probe

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <sys/mman.h>
#include <chrono>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("ERR %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1);} } while (0)

static const size_t EXPERT = 681574;              // one expert's bytes (field config)
static const int    N_EXP  = 512;
static const size_t TABLE  = EXPERT * N_EXP;      // 333 MiB

// The complement: 126 runs over 290 experts (alternate 2/3 to average 2.3).
struct run_t { size_t off, len; };                // off/len in experts
static std::vector<run_t> make_runs() {
    std::vector<run_t> r;
    int e = 0, i = 0;
    while (e < 290) {
        int len = (i % 3 == 2) ? 3 : 2;           // 2,2,3 -> avg 2.33
        if (e + len > 290) len = 290 - e;
        r.push_back({(size_t) e, (size_t) len});
        e += len;
        i++;
    }
    return r;
}

static double ev_ms(hipEvent_t a, hipEvent_t b) { float ms = 0; CHECK(hipEventElapsedTime(&ms, a, b)); return ms; }

static double median3(double a, double b, double c) {
    double v[3] = {a, b, c};
    for (int i = 0; i < 2; i++) for (int j = i + 1; j < 3; j++) if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[1];
}

// ONE async copy of `bytes` from `src`.  (The A arm used to divide the whole-table time by the
// complement's bytes -- that inflated every "one copy" figure 1.77x.  Bytes are passed explicitly.)
static double time_one(hipStream_t s, void * d, const char * src, size_t bytes) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipMemcpyAsync(d, src, bytes, hipMemcpyHostToDevice, s));
    CHECK(hipStreamSynchronize(s));
    double best = 1e9;
    for (int r = 0; r < 3; r++) {
        CHECK(hipEventRecord(e0, s));
        CHECK(hipMemcpyAsync(d, src, bytes, hipMemcpyHostToDevice, s));
        CHECK(hipEventRecord(e1, s));
        CHECK(hipStreamSynchronize(s));
        double ms = ev_ms(e0, e1);
        if (ms < best) best = ms;
    }
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return best;
}

// The SYNCHRONOUS API -- what the delivery's own `stage_h2d_gbps()` calibration times ("timed with a
// synchronous copy so it needs no event API").  If this is much slower than `time_one`, the delivery's
// 28.6 GB/s link figure (and the width gate calibrated from it) is measuring the wrong path.
static double time_one_sync(hipStream_t s, void * d, const char * src, size_t bytes) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipEventRecord(e0, s));
    CHECK(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
    CHECK(hipStreamSynchronize(s));
    CHECK(hipEventRecord(e1, s));
    CHECK(hipStreamSynchronize(s));
    double ms = ev_ms(e0, e1);
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return ms;
}

// One table-sized fill = all runs, issued back to back on one stream.
static double time_fill(hipStream_t s, void * d, const char * src, const std::vector<run_t> & runs,
                        int exp_per_copy /*1 = per-expert, >1 = coalesced runs*/) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipMemcpyAsync(d, src, TABLE, hipMemcpyHostToDevice, s));   // warm
    CHECK(hipStreamSynchronize(s));
    CHECK(hipEventRecord(e0, s));
    if (exp_per_copy == 1) {
        for (const run_t & r : runs) {
            for (size_t k = 0; k < r.len; k++) {
                const size_t off = (r.off + k) * EXPERT;
                CHECK(hipMemcpyAsync((char *) d + off, (const char *) src + off, EXPERT,
                                     hipMemcpyHostToDevice, s));
            }
        }
    } else {
        for (const run_t & r : runs) {
            const size_t off = r.off * EXPERT, n = r.len * EXPERT;
            CHECK(hipMemcpyAsync((char *) d + off, (const char *) src + off, n, hipMemcpyHostToDevice, s));
        }
    }
    CHECK(hipEventRecord(e1, s));
    CHECK(hipStreamSynchronize(s));
    double ms = ev_ms(e0, e1);
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return ms;
}

static size_t fill_bytes(const std::vector<run_t> & runs) {
    size_t b = 0;
    for (const run_t & r : runs) b += r.len * EXPERT;
    return b;
}

// Vectorised host->device pull: one CTA per run segment, grid-strided int4 loads.
__global__ void pull_kernel(char * __restrict__ d, const char * __restrict__ s,
                            const size_t * offs, const size_t * lens, int n_runs) {
    const int r = blockIdx.x;
    if (r >= n_runs) return;
    const size_t off = offs[r] / 16, n = lens[r] / 16;    // 16-byte words
    const int4 * sp = (const int4 *) (s + offs[r]);
    int4 * dp = (int4 *) (d + offs[r]);
    for (size_t i = blockIdx.y * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.y) {
        dp[i] = sp[i];
    }
}

static double time_pull(hipStream_t s, void * d, const char * src, const std::vector<run_t> & runs,
                        int ctas_per_run, int threads) {
    size_t n = runs.size();
    std::vector<size_t> hoff(n), hlen(n);
    for (size_t i = 0; i < n; i++) { hoff[i] = runs[i].off * EXPERT; hlen[i] = runs[i].len * EXPERT; }
    size_t *doff = nullptr, *dlen = nullptr;
    CHECK(hipMalloc(&doff, n * sizeof(size_t))); CHECK(hipMalloc(&dlen, n * sizeof(size_t)));
    CHECK(hipMemcpy(doff, hoff.data(), n * sizeof(size_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dlen, hlen.data(), n * sizeof(size_t), hipMemcpyHostToDevice));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    dim3 grid((unsigned) n, (unsigned) ctas_per_run);
    pull_kernel<<<grid, threads, 0, s>>>((char *) d, src, doff, dlen, (int) n);   // warm
    CHECK(hipStreamSynchronize(s));
    CHECK(hipEventRecord(e0, s));
    pull_kernel<<<grid, threads, 0, s>>>((char *) d, src, doff, dlen, (int) n);
    CHECK(hipEventRecord(e1, s));
    CHECK(hipStreamSynchronize(s));
    double ms = ev_ms(e0, e1);
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    CHECK(hipFree(doff)); CHECK(hipFree(dlen));
    return ms;
}

// A compute-heavy kernel: fixed iteration count, so its TIME is the throughput measure.
__global__ void spin(float * p, long long iters) {
    long long t = clock64();
    float x = p[threadIdx.x];
    while (clock64() - t < iters) x += 1.0f;
    if (x == 12345.0f) p[threadIdx.x] = x;
}
static double time_spin(hipStream_t s, float * dp, long long iters, int blocks) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    spin<<<blocks, 256, 0, s>>>(dp, iters);      // warm
    CHECK(hipStreamSynchronize(s));
    CHECK(hipEventRecord(e0, s));
    spin<<<blocks, 256, 0, s>>>(dp, iters);
    CHECK(hipEventRecord(e1, s));
    CHECK(hipStreamSynchronize(s));
    double ms = ev_ms(e0, e1);
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return ms;
}

int main() {
    CHECK(hipSetDevice(0));
    hipDeviceProp_t pr{};
    CHECK(hipGetDeviceProperties(&pr, 0));
    printf("device: %s (%s), %zu CUs\n", pr.name, pr.gcnArchName, (size_t) pr.multiProcessorCount);

    // Host kinds
    void * h_pin = nullptr;
    CHECK(hipHostMalloc(&h_pin, TABLE, hipHostMallocDefault));
    void * h_pin_wc = nullptr;
    CHECK(hipHostMalloc(&h_pin_wc, TABLE, hipHostMallocWriteCombined));
    void * h_page = malloc(TABLE);
    void * h_reg = mmap(nullptr, TABLE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h_reg == MAP_FAILED) { printf("mmap failed\n"); return 1; }
    CHECK(hipHostRegister(h_reg, TABLE, hipHostRegisterDefault));
    memset(h_pin, 1, TABLE); memset(h_pin_wc, 1, TABLE); memset(h_page, 1, TABLE); memset(h_reg, 1, TABLE);

    void * d = nullptr;
    CHECK(hipMalloc(&d, TABLE));
    hipStream_t s1, s2;
    CHECK(hipStreamCreate(&s1)); CHECK(hipStreamCreate(&s2));

    const std::vector<run_t> runs = make_runs();
    const size_t comp_bytes = fill_bytes(runs);
    printf("table %.1f MiB, complement %.1f MiB in %zu runs (%.2f experts/run), %zu experts\n",
           TABLE / 1048576.0, comp_bytes / 1048576.0, runs.size(),
           comp_bytes / (double) EXPERT / runs.size(), comp_bytes / EXPERT);

    // A: one whole-table copy per host-memory kind
    printf("\nA  one %.0f MiB H2D copy, by host kind\n", TABLE / 1048576.0);
    struct { void * p; const char * n; } kinds[] = {
        { h_pin, "pinned (hipHostMalloc)" },
        { h_reg, "registered mmap" },
        { h_pin_wc, "pinned write-combined" },
        { h_page, "pageable malloc" },
    };
    for (auto & k : kinds) {
        double ms = time_one(s1, d, (const char *) k.p, TABLE);
        double msy = time_one_sync(s1, d, (const char *) k.p, TABLE);
        printf("   %-26s async %6.2f ms %6.1f GB/s   |  sync %6.2f ms %6.1f GB/s\n",
               k.n, ms, TABLE / (ms * 1e6), msy, TABLE / (msy * 1e6));
    }

    // B/C/D/E: the complement, per-expert vs run-coalesced vs 2 streams
    printf("\nB-E  the 43%%-residency complement (%.0f MiB)\n", comp_bytes / 1048576.0);
    double ms_b = median3(time_fill(s1, d, (const char *) h_pin, runs, 1),
                          time_fill(s1, d, (const char *) h_pin, runs, 1),
                          time_fill(s1, d, (const char *) h_pin, runs, 1));
    double ms_c = median3(time_fill(s1, d, (const char *) h_pin, runs, 2),
                          time_fill(s1, d, (const char *) h_pin, runs, 2),
                          time_fill(s1, d, (const char *) h_pin, runs, 2));
    // 2-stream: odd runs on s2, then wait for both
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipEventRecord(e0, s1));
    for (size_t i = 0; i < runs.size(); i++) {
        hipStream_t s = (i & 1) ? s2 : s1;
        CHECK(hipMemcpyAsync((char *) d + runs[i].off * EXPERT, (const char *) h_pin + runs[i].off * EXPERT,
                             runs[i].len * EXPERT, hipMemcpyHostToDevice, s));
    }
    CHECK(hipEventRecord(e1, s2));
    CHECK(hipStreamSynchronize(s1)); CHECK(hipStreamSynchronize(s2));
    double ms_d = ev_ms(e0, e1);
    double ms_e = time_one(s1, d, (const char *) h_pin, comp_bytes);
    const int n_exp_copies = (int) (comp_bytes / EXPERT);
    printf("   B  %3d per-expert copies     %8.2f ms  %6.1f GB/s  (%5.1f us/copy)\n",
           n_exp_copies, ms_b, comp_bytes / (ms_b * 1e6), ms_b * 1000.0 / n_exp_copies);
    printf("   C  %3zu run copies            %8.2f ms  %6.1f GB/s  (%5.1f us/copy)\n",
           runs.size(), ms_c, comp_bytes / (ms_c * 1e6), ms_c * 1000.0 / runs.size());
    printf("   D  %3zu runs, 2 copy streams  %8.2f ms  %6.1f GB/s\n",
           runs.size(), ms_d, comp_bytes / (ms_d * 1e6));
    printf("   E  1 copy, same bytes        %8.2f ms  %6.1f GB/s\n", ms_e, comp_bytes / (ms_e * 1e6));

    // F: the pull kernel, CTA sweep
    printf("\nF  PULL kernel (vectorised int4, one block-group per run)\n");
    for (int ctas : {1, 2, 4, 8}) {
        double ms = time_pull(s1, d, (const char *) h_pin, runs, ctas, 256);
        printf("   %d CTA/run  %8.2f ms  %6.1f GB/s\n", ctas, ms, comp_bytes / (ms * 1e6));
    }

    // G: interference with a concurrent compute kernel
    printf("\nG  interference: a 64-block spin kernel alone, then beside each fill\n");
    float * dp = nullptr;
    CHECK(hipMalloc(&dp, 4096 * sizeof(float)));
    const long long iters = 60000000;
    double base = time_spin(s2, dp, iters, 64);
    printf("   spin alone                %8.2f ms\n", base);
    for (int mode = 0; mode < 3; mode++) {
        hipEvent_t g0, g1; CHECK(hipEventCreate(&g0)); CHECK(hipEventCreate(&g1));
        double spin_ms = 0;
        for (int rep = 0; rep < 3; rep++) {
            CHECK(hipEventRecord(g0, s2));
            spin<<<64, 256, 0, s2>>>(dp, iters);
            CHECK(hipEventRecord(g1, s2));
            if (mode == 0) {
                time_fill(s1, d, (const char *) h_pin, runs, 2);                 // DMA alongside
            } else if (mode == 1) {
                time_pull(s1, d, (const char *) h_pin, runs, 1, 256);            // pull kernel alongside
            }
            CHECK(hipStreamSynchronize(s2));
            spin_ms += ev_ms(g0, g1);
        }
        spin_ms /= 3;
        const char * nm = mode == 0 ? "spin + DMA fill" : mode == 1 ? "spin + pull kernel" : "spin alone (again)";
        printf("   %-25s %8.2f ms  (%+.1f %%)\n", nm, spin_ms, 100.0 * (spin_ms - base) / base);
    }

    // H: streams vs copy granularity, INTERLEAVED (this box alternates run-to-run, so an A/B/A/B is
    // the only way to compare).  Tests 1/2/4 copy streams x {per-expert, run-coalesced}.
    printf("\nH  interleaved: copy streams x granularity (%d reps per arm, alternating)\n", 4);
    hipStream_t st[4] = { s1, s2, nullptr, nullptr };
    CHECK(hipStreamCreate(&st[2])); CHECK(hipStreamCreate(&st[3]));
    for (int k = 0; k < 4; k++) {   // warm every stream over the whole run list
        for (size_t i = 0; i < runs.size(); i++) {
            CHECK(hipMemcpyAsync((char *) d + runs[i].off * EXPERT, (const char *) h_pin + runs[i].off * EXPERT,
                                 runs[i].len * EXPERT, hipMemcpyHostToDevice, st[k]));
        }
    }
    for (int k = 0; k < 4; k++) CHECK(hipStreamSynchronize(st[k]));
    double best_h[2][3];
    for (int g = 0; g < 2; g++) for (int n = 0; n < 3; n++) best_h[g][n] = 1e9;
    const int nstr[3] = { 1, 2, 4 };
    for (int rep = 0; rep < 4; rep++) {
        for (int g = 0; g < 2; g++) {
            for (int ni = 0; ni < 3; ni++) {
                const int ns = nstr[ni];
                CHECK(hipDeviceSynchronize());            // HOST clock: spans every stream
                auto t0 = std::chrono::steady_clock::now();
                size_t k = 0;
                if (g == 0) {                        // per-expert
                    for (const run_t & r : runs) for (size_t j = 0; j < r.len; j++, k++) {
                        const size_t off = (r.off + j) * EXPERT;
                        CHECK(hipMemcpyAsync((char *) d + off, (const char *) h_pin + off, EXPERT,
                                             hipMemcpyHostToDevice, st[k % ns]));
                    }
                } else {                             // run-coalesced
                    for (const run_t & r : runs) {
                        const size_t off = r.off * EXPERT;
                        CHECK(hipMemcpyAsync((char *) d + off, (const char *) h_pin + off, r.len * EXPERT,
                                             hipMemcpyHostToDevice, st[k % ns]));
                        k++;
                    }
                }
                CHECK(hipDeviceSynchronize());
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
                if (ms < best_h[g][ni]) best_h[g][ni] = ms;
            }
        }
    }
    for (int g = 0; g < 2; g++) {
        for (int ni = 0; ni < 3; ni++) {
            printf("   %-14s %d stream(s)  best %6.2f ms  %6.1f GB/s\n",
                   g == 0 ? "per-expert" : "run-coalesced", nstr[ni], best_h[g][ni],
                   comp_bytes / (best_h[g][ni] * 1e6));
        }
    }

    // I: TIME TO FIRST COMPLETION.  Aggregate bandwidth is invariant for a link-bound fill, but the copy the
    // CONSUMER waits on arrives later when the link is shared with a copy it does not need yet.  This is the
    // measurement behind the "one stream per consumer chain" rule (INVARIANTS.md) and the -4 % the global
    // stream knob measured on Flash-Next.
    printf("\nI  time to FIRST copy vs all copies, %d x one-table copies\n", 8);
    {
        const int K = 8;
        void * dk = nullptr;
        CHECK(hipMalloc(&dk, (size_t) K * TABLE));
        hipEvent_t ev0;
        CHECK(hipEventCreate(&ev0));
        for (int ns : {1, 2, 4}) {
            double bf = 1e9, ba = 1e9;
            for (int r = 0; r < 3; r++) {
                CHECK(hipDeviceSynchronize());
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < K; i++) {
                    CHECK(hipMemcpyAsync((char *) dk + (size_t) i * TABLE, h_pin, TABLE,
                                         hipMemcpyHostToDevice, st[i % ns]));
                    if (i == 0) { CHECK(hipEventRecord(ev0, st[0])); }   // ONE record: copy 0 only
                }
                CHECK(hipEventSynchronize(ev0));
                double f = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                CHECK(hipDeviceSynchronize());
                double a = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (f < bf) { bf = f; }
                if (a < ba) { ba = a; }
            }
            printf("   %d stream(s)  first %6.2f ms   all %6.2f ms\n", ns, bf, ba);
        }
        CHECK(hipEventDestroy(ev0));
        CHECK(hipFree(dk));
    }

    CHECK(hipHostUnregister(h_reg));
    munmap(h_reg, TABLE);
    free(h_page);
    CHECK(hipHostFree(h_pin));
    CHECK(hipHostFree(h_pin_wc));
    CHECK(hipFree(d)); CHECK(hipFree(dp));
    return 0;
}
