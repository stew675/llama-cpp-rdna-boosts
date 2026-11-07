#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    /* GiB, fractional allowed (`./ballast 32.4`): the emulation needs MemAvailable to land on a target
     * (57.3 GiB for a 59 GiB box) and a whole-GiB step is ~1 GiB of MemAvailable -- too coarse. */
    double gib = argc > 1 ? atof(argv[1]) : 31.0;
    size_t n = (size_t) (gib * 1073741824.0);
    char * p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    memset(p, 1, n);
    if (mlock(p, n) != 0) { perror("mlock"); return 1; }
    printf("ballast %.1f GiB locked (%zu B)\n", gib, n);
    fflush(stdout);
    pause();
    return 0;
}
