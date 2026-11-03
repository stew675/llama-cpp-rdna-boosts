// wip/host-expert-pinned-window: is a `hipMemcpy` H2D out of a *registered* file mapping a direct
// DMA by the copy engine, or a host-side staged (bounce) copy?  Three sources, one copy each after a
// warmup, so a `rocprofv3 --memory-copy-trace --kernel-trace` shows what the runtime actually does.
//
//   build: ./build.sh
//   trace: ./trace-copy-path.sh
//
// The question matters for design B's budget: a bounce means the "window" is not zero-copy (it is
// "one host copy instead of two"), and that the pages must be resident or the copy pays the faults.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIP_CHECK(call)                                                                       \
    do {                                                                                      \
        hipError_t e_ = (call);                                                               \
        if (e_ != hipSuccess) {                                                               \
            fprintf(stderr, "HIP error %s: %s\n", #call, hipGetErrorString(e_));              \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
                                 : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const size_t n = 8 << 20;
    HIP_CHECK(hipSetDevice(0));

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    char * map = (char *) mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    char * reg_src = map + (((size_t) st.st_size / 2) & ~(page - 1));

    void * pin = nullptr;
    HIP_CHECK(hipHostMalloc(&pin, n, hipHostMallocDefault));
    void * pgbl = malloc(n);
    memcpy(pin, reg_src, n);
    memcpy(pgbl, reg_src, n);

    char * dev = nullptr;
    HIP_CHECK(hipMalloc((void **) &dev, n));
    HIP_CHECK(hipHostRegister(reg_src, n, hipHostRegisterReadOnly));

    struct arm { const char * name; void * src; };
    const arm arms[] = {
        { "REGISTERED file mapping", reg_src },
        { "PINNED hipHostMalloc",    pin },
        { "PAGEABLE malloc",         pgbl },
    };
    // one warmup per arm (lazy pin / staging allocation), then one traced copy per arm
    for (const auto & a : arms) { HIP_CHECK(hipMemcpy(dev, a.src, n, hipMemcpyHostToDevice)); }
    for (const auto & a : arms) {
        printf("COPYARM %s\n", a.name);
        fflush(stdout);
        HIP_CHECK(hipMemcpy(dev, a.src, n, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipHostUnregister(reg_src));
    printf("COPYARM done\n");
    return 0;
}
