// Is a `hipMemcpyAsync(dev, alias, D2D)` out of a REGISTERED mapping range byte-stable, and does it depend on
// the registration SIZE?  (v2 registered one small range per expert and was byte-identical; the per-TENSOR
// registration version produced nondeterministic text, which is the tearing signature.)
//   build: ./build.sh    run: ./build/probe-alias-d2d
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
        : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    hipSetDevice(0);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st{};
    fstat(fd, &st);
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    char * map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    const size_t base = ((size_t) st.st_size / 4) & ~(page - 1);

    const size_t reg_sizes[] = { 1740800, 8u << 20, 64u << 20, 256u << 20, 1024u << 20 };
    const size_t chunk = 1740800;   // copy an expert-sized chunk out of each registration
    char * dev = nullptr;
    hipMalloc((void **) &dev, chunk);
    char * rb = (char *) malloc(chunk);

    for (size_t reg : reg_sizes) {
        char * win = map + base;
        hipError_t e = hipHostRegister(win, reg, hipHostRegisterReadOnly);
        if (e != hipSuccess) { printf("register %zu MiB: %s\n", reg >> 20, hipGetErrorString(e)); continue; }
        void * alias = nullptr;
        e = hipHostGetDevicePointer(&alias, win, 0);
        if (e != hipSuccess) { printf("alias %zu MiB: %s\n", reg >> 20, hipGetErrorString(e)); hipHostUnregister(win); continue; }
        if (reg < chunk * 8) { printf("registered %6zu MiB: skipped (smaller than 8 chunks)\n", reg >> 20); hipHostUnregister(win); continue; }
        int bad = 0;
        for (int i = 0; i < 24; i++) {
            const size_t off = (size_t) (i % 8) * chunk;   // stay inside the registration
            hipMemcpyAsync(dev, (char *) alias + off, chunk, hipMemcpyDeviceToDevice, 0);
            hipStreamSynchronize(0);
            hipMemcpy(rb, dev, chunk, hipMemcpyDeviceToHost);
            if (memcmp(rb, map + base + off, chunk) != 0) bad++;
        }
        printf("registered %6zu MiB, alias D2D copy of %zu KiB: %s (%d/24 bad)\n", reg >> 20, chunk >> 10,
               bad == 0 ? "BYTE-STABLE" : "TEARING", bad);
        hipHostUnregister(win);
    }
    printf("\n(also: the same test with the H2D kind, which ROCm stages for pageable sources)\n");
    for (size_t reg : { (size_t) 1740800, (size_t) (256u << 20) }) {
        char * win = map + base;
        if (hipHostRegister(win, reg, hipHostRegisterReadOnly) != hipSuccess) continue;
        if (reg < chunk * 8) { hipHostUnregister(win); continue; }
        int bad = 0;
        for (int i = 0; i < 24; i++) {
            const size_t off = (size_t) (i % 8) * chunk;
            hipMemcpyAsync(dev, win + off, chunk, hipMemcpyHostToDevice, 0);
            hipStreamSynchronize(0);
            hipMemcpy(rb, dev, chunk, hipMemcpyDeviceToHost);
            if (memcmp(rb, map + base + off, chunk) != 0) bad++;
        }
        printf("registered %6zu MiB, H2D copy from the range: %s (%d/24 bad)\n", reg >> 20,
               bad == 0 ? "BYTE-STABLE" : "TEARING", bad);
        hipHostUnregister(win);
    }
    return 0;
}
