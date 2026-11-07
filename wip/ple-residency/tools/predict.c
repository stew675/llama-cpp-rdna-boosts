// predict -- cost of computing every PLE row a prompt will need, ahead of time.
// uint64 mul/xor/mod exactly as llm_graph_input_ple::set_input() does, then a bitmap dedup.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static const uint64_t mult[3] = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
static const uint32_t off[16] = {0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,
                                 160000373,180000451,200000534,220000623,240000718,260000819,280000926,300001039};
static const uint32_t vsz[16] = {20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
                                 20000085,20000093,20000103,20000115,20000131,20000143,20000153,20000167};
#define NROWS 320001536ULL
#define RPP 45ULL
#define NPAGES (NROWS/RPP + 1)
int main(int argc, char **argv) {
    long ntok = argc > 1 ? atol(argv[1]) : 189809;
    int32_t *tok = malloc(ntok*4);
    // a synthetic but realistic token stream (the trace's ids, wrapped)
    FILE *f = fopen("/tmp/ple/rows_ids.bin","rb"); (void) f;
    for (long i = 0; i < ntok; i++) tok[i] = (int32_t)((i * 2654435761u) % 248000);
    uint8_t *rowbit = calloc(NROWS/8, 1);
    uint8_t *pgebit = calloc(NPAGES/8, 1);
    double t0 = now();
    for (long i = 0; i < ntok; i++) {
        uint64_t ctx[3] = { (uint64_t) tok[i], i>=1 ? (uint64_t) tok[i-1] : 248044ULL,
                            i>=2 ? (uint64_t) tok[i-2] : 248044ULL };
        for (int n = 2; n <= 3; n++) {
            uint64_t mixed = ctx[0]*mult[0];
            for (int j = 1; j < n; j++) mixed ^= ctx[j]*mult[j];
            const int base = (n-2)*8;
            for (int g = 0; g < 8; g++) {
                uint32_t row = (uint32_t)(mixed % vsz[base+g]) + off[base+g];
                rowbit[row>>3] |= 1u << (row&7);
                uint32_t pg = row / RPP;
                pgebit[pg>>3] |= 1u << (pg&7);
            }
        }
    }
    double t1 = now();
    uint64_t nrow = 0, npage = 0;
    for (uint64_t i = 0; i < NROWS/8; i++) nrow += __builtin_popcount(rowbit[i]);
    for (uint64_t i = 0; i < NPAGES/8; i++) npage += __builtin_popcount(pgebit[i]);
    double t2 = now();
    printf("%ld tokens: predict+bitmap %.1f ms, census %.1f ms | %llu rows, %llu pages, 16 hashes/token\n",
           ntok, (t1-t0)*1e3, (t2-t1)*1e3, (unsigned long long)nrow, (unsigned long long)npage);
    return 0;
}
