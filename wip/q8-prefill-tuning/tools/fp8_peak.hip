#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
using v2i = __attribute__((__vector_size__(2*sizeof(int)))) int;
using v8f = __attribute__((__vector_size__(8*sizeof(float)))) float;
#define NACC 8
__global__ void __launch_bounds__(256) peak(v8f * o, int reps) {
    v2i a = {(int)threadIdx.x, (int)threadIdx.x+1};
    v2i b = {(int)threadIdx.x+2, (int)threadIdx.x+3};
    v8f acc[NACC];
#pragma unroll
    for (int i=0;i<NACC;++i) acc[i]=(v8f){0,0,0,0,0,0,0,0};
    for (int r=0;r<reps;++r) {
#pragma unroll
        for (int i=0;i<NACC;++i)
            acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc[i]);
    }
    v8f s=(v8f){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i=0;i<NACC;++i) for (int l=0;l<8;++l) s[l]+=acc[i][l];
    if (threadIdx.x==0 && blockIdx.x==0) o[0]=s;
}
int main(int argc,char**argv){
    int blocks=argc>1?atoi(argv[1]):4096, threads=256, reps=argc>2?atoi(argv[2]):20000;
    v8f* d; hipMalloc(&d,sizeof(v8f));
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    printf("device=%s CUs=%d blocks=%d reps=%d\n",p.name,p.multiProcessorCount,blocks,reps);
    peak<<<blocks,threads>>>(d,reps); hipDeviceSynchronize();
    hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0); peak<<<blocks,threads>>>(d,reps); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms=0; hipEventElapsedTime(&ms,e0,e1);
    double s=ms/1e3, warps=(double)blocks*(threads/32), macs=warps*NACC*(double)reps*4096.0;
    printf("fp8 wmma: %.3f ms -> %.1f T-MAC/s (%.1f TOPS)\n", ms, macs/s/1e12, 2*macs/s/1e12);
    return 0;
}
