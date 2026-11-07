// dqbench -- isolate the PLE row read: to_float() from (a) a local buffer, (b) the mmap'd
// file with the loader's MADV_RANDOM flags, (c) the same mapping after the pages are warm.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include "ggml.h"
#include "ggml-quants.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
int main(int argc, char **argv) {
    const char *path = argv[1];
    int fd = open(path, O_RDONLY);
    const size_t total = 28800138240LL;
    uint8_t *map = mmap(NULL, total, PROT_READ, MAP_PRIVATE, fd, 0);
    madvise(map, total, MADV_RANDOM);
    int32_t *rows = malloc(8u<<20);
    FILE *rf = fopen("/tmp/ple/rows_ids.bin","rb"); fseek(rf,0,SEEK_END); size_t np=ftell(rf)/4; fseek(rf,0,SEEK_SET);
    if (fread(rows,4,np,rf)!=np) return 1; fclose(rf);
    void (*tf)(const void*, float*, int64_t) = (void(*)(const void*,float*,int64_t)) dequantize_row_iq4_nl;
    printf("type %s block %lld size %zu to_float %p\n", ggml_type_name(GGML_TYPE_IQ4_NL),
           (long long) ggml_blck_size(GGML_TYPE_IQ4_NL), ggml_type_size(GGML_TYPE_IQ4_NL), (void*)tf);

    float buf[160];
    // (a) local buffer
    double t0=now(); for (int i=0;i<20000;i++) tf((const void*)((const uint8_t*)rows+0), buf, 160);
    printf("to_float from a local buffer: %.1f ns/row\n", (now()-t0)/20000);

    // (b) cold mapping, 16 rows per "step", 64 steps
    const int G=64, R=16; size_t cur=0; float dst[16][160];
    posix_fadvise(fd, 192, total, POSIX_FADV_DONTNEED);
    t0=now();
    for (int g=0; g<G; g++) {
        for (int i=0;i<R;i++) posix_fadvise(fd, 192 + (size_t)rows[cur+i]*90, 90, POSIX_FADV_WILLNEED);
        for (int i=0;i<R;i++) tf(map + 192 + (size_t)rows[cur+i]*90, dst[i], 160);
        cur += R;
    }
    printf("cold mapping, fadvise hint : %.1f us/step, %.1f us/row\n", (now()-t0)/1e3/G, (now()-t0)/1e3/(G*R));

    // (c) same rows, now resident in the page cache
    cur -= G*R; t0=now();
    for (int g=0; g<G; g++) {
        for (int i=0;i<R;i++) posix_fadvise(fd, 192 + (size_t)rows[cur+i]*90, 90, POSIX_FADV_WILLNEED);
        for (int i=0;i<R;i++) tf(map + 192 + (size_t)rows[cur+i]*90, dst[i], 160);
        cur += R;
    }
    printf("warm mapping, fadvise hint : %.1f us/step, %.1f us/row\n", (now()-t0)/1e3/G, (now()-t0)/1e3/(G*R));

    // (d) warm mapping, no hint
    cur -= G*R; t0=now();
    for (int g=0; g<G; g++) for (int i=0;i<R;i++) tf(map + 192 + (size_t)rows[cur+i]*90, dst[i], 160);
    printf("warm mapping, no hint       : %.1f us/step, %.1f us/row\n", (now()-t0)/1e3/G, (now()-t0)/1e3/(G*R));
    return 0;
}
