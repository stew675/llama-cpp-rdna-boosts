#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
    int mb = argc>1?atoi(argv[1]):20;
    size_t n=(size_t)mb<<20;
    hipSetDevice(0); void *a; hipMalloc(&a,n); hipMemset(a,1,n);
    hipSetDevice(1); void *b; hipMalloc(&b,n); hipMemset(b,2,n);
    int can=0; hipDeviceCanAccessPeer(&can,0,1);
    printf("canAccessPeer(0,1)=%d size=%dMB\n",can,mb);
    hipSetDevice(0); hipDeviceEnablePeerAccess(1,0);
    hipDeviceSynchronize();
    hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
    int iters=20;
    // D2D peer copy 0->1
    hipEventRecord(e0);
    for(int i=0;i<iters;i++) hipMemcpyPeerAsync(b,1,a,0,n,nullptr);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms,e0,e1);
    double s=ms/1e3/iters;
    printf("peer copy 0->1: %.3f ms  -> %.2f GB/s\n", s*1e3, n/s/1e9);
    // D2D peer copy 1->0
    hipEventRecord(e0);
    for(int i=0;i<iters;i++) hipMemcpyPeerAsync(a,0,b,1,n,nullptr);
    hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms,e0,e1); s=ms/1e3/iters;
    printf("peer copy 1->0: %.3f ms  -> %.2f GB/s\n", s*1e3, n/s/1e9);
    // same-device copy baseline
    hipSetDevice(0); void*c; hipMalloc(&c,n);
    hipEventRecord(e0); for(int i=0;i<iters;i++) hipMemcpyAsync(c,a,n,hipMemcpyDeviceToDevice,nullptr);
    hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms,e0,e1); s=ms/1e3/iters;
    printf("local  copy 0->0: %.3f ms  -> %.2f GB/s (control)\n", s*1e3, n/s/1e9);
    return 0;
}
