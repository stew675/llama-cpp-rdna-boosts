// Does `hipHostRegister`/`hipHostUnregister` churn stay cheap, or does its cost grow with the number of
// register/unregister cycles?  This decides whether the window may be promoted per FILL (churny) or must be
// a managed resident SET (like the pool's contents) that a fill only reads.
// build: ./build.sh    run: ./build/probe-churn
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static double now_us() {
    timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec * 1e6 + (double) t.tv_nsec / 1e3;
}

__global__ void busy_kernel(float * out, int iters) {
    float acc = 0.f;
    for (int i = 0; i < iters; i++) {
        acc += (float) threadIdx.x * 0.000001f;
    }
    if (acc == 1234.5f) { out[threadIdx.x] = acc; }
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
        : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const int cycles = argc > 2 ? atoi(argv[2]) : 20000;
    const size_t len = 1740800;
    
    hipSetDevice(0);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st{};
    fstat(fd, &st);
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    char * map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    const size_t base = ((size_t) st.st_size / 4) & ~(page - 1);
    char * dev = nullptr;
    hipMalloc((void **) &dev, len);
    // warm the page cache for the region we cycle over
    volatile char sink = 0;
    for (size_t p = base; p < base + len * 4096; p += page) sink ^= map[p];
    (void) sink;

    double bucket_reg = 0, bucket_copy = 0, bucket_unreg = 0;
    int    bucket_n   = 0, bucket     = 0;
    for (int i = 0; i < cycles; i++) {
        char * src = map + base + ((size_t) i % (len * 4096 / len)) * len;
        double t0 = now_us();
        hipError_t e = hipHostRegister(src, len, hipHostRegisterReadOnly);
        double t1 = now_us();
        if (e != hipSuccess) { printf("register failed at %d: %s\n", i, hipGetErrorString(e)); return 1; }
        hipMemcpyAsync(dev, src, len, hipMemcpyHostToDevice, 0);
        hipStreamSynchronize(0);
        double t2 = now_us();
        hipHostUnregister(src);
        double t3 = now_us();
        bucket_reg += t1 - t0; bucket_copy += t2 - t1; bucket_unreg += t3 - t2; bucket_n++;
        if (bucket_n == 256) {
            printf("cycle %6d: register %7.1f us | copy %7.1f us | unregister %7.1f us\n",
                   bucket * 256 + 256, bucket_reg / bucket_n, bucket_copy / bucket_n, bucket_unreg / bucket_n);
            bucket_reg = bucket_copy = bucket_unreg = 0; bucket_n = 0; bucket++;
            if (bucket > 80) break;
        }
    }
    // ---- does hipHostRegister SYNCHRONIZE the stream? -------------------------------------
    // Enqueue ~10 ms of work, then time a register and an unregister.  If either waits for the kernel,
    // the cost of a per-fill registration is the whole exposed pipeline, not the ~18 us shown above.
    {
        float * out = nullptr;
        hipMalloc((void **) &out, 4096);
        double t0 = now_us();
        busy_kernel<<<64, 256, 0, 0>>>(out, 200000);
        double t1 = now_us();
        // find the iteration count that gives ~10 ms by doubling (bounded)
        double kernel_us = 0;
        for (int it = 200000; it < 20000000; it *= 2) {
            t0 = now_us();
            busy_kernel<<<64, 256, 0, 0>>>(out, it);
            hipStreamSynchronize(0);
            kernel_us = now_us() - t0;
            if (kernel_us > 8000.0) break;
        }
        hipStreamSynchronize(0);
        // now: launch the kernel, then immediately register (no sync in between)
        (void) t1;
        const double tlaunch = now_us();
        busy_kernel<<<64, 256, 0, 0>>>(out, 20000000);
        char * src = map + base;
        const double treg0 = now_us();
        hipHostRegister(src, len, hipHostRegisterReadOnly);
        const double treg1 = now_us();
        hipStreamSynchronize(0);
        const double tend = now_us();
        hipHostUnregister(src);
        printf("\nbusy kernel alone      : %8.1f us\n", kernel_us);
        printf("register AFTER launch  : %8.1f us   <- ~0 means it waited for the kernel (sync!)\n",
               treg1 - treg0);
        printf("kernel+register total  : %8.1f us   (launch-to-drain)\n", tend - tlaunch);
    }
    return 0;
}
