#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
__global__ void sumw(const unsigned int* p, unsigned long long* out, size_t n){ size_t s=(size_t)gridDim.x*blockDim.x,i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; unsigned long long a=0; for(;i<n;i+=s)a+=((volatile const unsigned int*)p)[i]; atomicAdd(out,a);}
static unsigned long long gs(const void*p,size_t b){ unsigned long long*d; CK(hipMalloc(&d,8)); CK(hipMemset(d,0,8)); sumw<<<512,256>>>((const unsigned int*)p,d,b/4); CK(hipDeviceSynchronize()); unsigned long long h; CK(hipMemcpy(&h,d,8,hipMemcpyDeviceToHost)); hipFree(d); return h;}
int main(int argc,char**argv){
  size_t SZ=1ull<<30;
  unsigned char* d; CK(hipMalloc(&d,SZ));
  printf("  hipMalloc pointer = %p (host dereference test)\n", d);
  fflush(stdout);
  volatile unsigned char v = d[0]; printf("  host READ  of device memory: OK (0x%02x)\n", v); fflush(stdout);
  d[0] = 0xAB; printf("  host WRITE of device memory: OK\n"); fflush(stdout);
  double t=now(); memset(d,0x44,SZ); double m=now()-t;
  printf("  host memset 1 GiB into hipMalloc: %.3f s = %.1f GB/s\n", m, SZ/m/1e9);
  printf("  GPU sum after host memset: %016llx  %s\n", gs(d,SZ), gs(d,SZ)==(0x44444444ULL*(SZ/4))?"FRESH":"STALE");
  int fd=open(argv[1],O_RDONLY|O_DIRECT); t=now(); ssize_t got=pread(fd,(void*)d,SZ,0); double od=now()-t;
  printf("  O_DIRECT pread 1 GiB into hipMalloc: %.3f s = %.1f GB/s (%zd)\n", od, SZ/od/1e9, got);
  printf("  GPU sum after disk fill:   %016llx\n", gs(d,SZ));
  return 0;
}
