// 12-pread-registered.cpp -- the keystone of the GTT expert store: pread(O_DIRECT) INTO an
// already-registered (ReadOnly) region, then have the GPU read exactly those bytes.  (P1.0 assumption #2.)
//
// WHY: the three halves are measured and the combination is not --
//   * pread(O_DIRECT) into an ANON buffer works, 3.6 GB/s                        (probe 04)
//   * a filled PROT_WRITE buffer can be registered ReadOnly, GPU reads it at 233  (probe 04)
//   * a host write into a REGISTERED range is GPU-visible with no flush          (probe 10)
// If the combination is refused (EINVAL) or the GPU reads stale bytes, the "zero-copy load into the vacated
// expert slot" is not zero-copy and the design must be re-costed before any code is written.
//
// The arm that matters is the REUSE arm: a slot that already holds a different expert's bytes (the eviction
// case), overwritten by a pread, then read by the GPU.  A fresh (zero) region would hide a coherence defect.
//
// Build: hipcc -O2 -o 12-pread-registered 12-pread-registered.cpp
// Run:   ./12-pread-registered [src_file] [MiB=256]
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

// GPU sums the buffer as bytes; the CPU sum of the same bytes is printed alongside, so a stale read is visible.
__global__ void sum(const unsigned char* p, unsigned long long* out, size_t n){
  size_t stride=(size_t)gridDim.x*blockDim.x, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long acc=0;
  for(; i<n; i+=stride){ const volatile unsigned char* q=p; acc+=q[i]; }
  atomicAdd(out,acc);   // the total is order-independent, so it must equal the CPU sum
}
static unsigned long long gpu_sum(const void* p,size_t n){
  unsigned long long* d; CK(hipMalloc(&d,8)); CK(hipMemset(d,0,8));
  sum<<<512,256>>>((const unsigned char*)p,d,n); CK(hipDeviceSynchronize());
  unsigned long long v; CK(hipMemcpy(&v,d,8,hipMemcpyDeviceToHost)); hipFree(d); return v;
}
static unsigned long long cpu_sum(const unsigned char* p,size_t n){ unsigned long long v=0; for(size_t i=0;i<n;i++) v+=p[i]; return v; }

int main(int argc,char**argv){
  const char* src=argc>1?argv[1]:"/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf";
  size_t SZ=((argc>2?(size_t)atoi(argv[2]):256)*1024*1024);
  const size_t STEP=64ull<<20;                       // each round preads a different 64 MiB slice
  printf("12-pread-registered: %zu MiB from %s\n", SZ>>20, src);

  int fd=open(src,O_RDONLY|O_DIRECT);
  if(fd<0){ perror("open O_DIRECT"); return 2; }
  int fdb=open(src,O_RDONLY);                                       // buffered arm
  unsigned char* buf=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(buf==MAP_FAILED){ perror("mmap"); return 2; }
  madvise(buf,SZ,MADV_HUGEPAGE);
  memset(buf,0xAA,SZ);                                              // fault it in, and give it a known pattern
  CK(hipHostRegister(buf,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly));
  void* d=buf; CK(hipHostGetDevicePointer(&d,buf,0));
  printf("  registered ReadOnly, prefilled 0xAA: CPU sum %llu / GPU sum %llu %s\n",
         cpu_sum(buf,SZ), gpu_sum(d,SZ), cpu_sum(buf,SZ)==gpu_sum(d,SZ)?"(match)":"(MISMATCH)");

  int fails=0;
  for(int round=0; round<3; ++round){
    const off_t off=(off_t)round*STEP;
    double t=now(); ssize_t got=pread(fd,buf,SZ,off); double dt=now()-t;
    if(got<0){ printf("  round %d: pread(O_DIRECT) into the REGISTERED region: %s\n", round, strerror(errno)); fails++; break; }
    const unsigned long long c=cpu_sum(buf,got), g=gpu_sum(d,got);
    printf("  round %d: O_DIRECT pread %zu MiB in %.1f ms = %6.2f GB/s | CPU %llu GPU %llu %s\n",
           round, (size_t)got>>20, dt*1e3, got/dt/1e9, c, g, c==g?"(match)":"(MISMATCH)");
    if(c!=g) fails++;
    if(round==0) printf("    ^ this round overwrote 0xAA with real bytes: a still-0xAA GPU read would be stale\n");
  }
  // the buffered arm, for comparison (what a page-cache hit costs)
  {
    const off_t off=0; double t=now(); ssize_t got=pread(fdb,buf,SZ,off); double dt=now()-t;
    if(got<0) printf("  buffered pread into the registered region: %s\n", strerror(errno));
    else {
      const unsigned long long c=cpu_sum(buf,got), g=gpu_sum(d,got);
      printf("  buffered pread %zu MiB in %.1f ms = %6.2f GB/s | CPU %llu GPU %llu %s\n",
             (size_t)got>>20, dt*1e3, got/dt/1e9, c, g, c==g?"(match)":"(MISMATCH)");
      if(c!=g) fails++;
    }
  }
  CK(hipHostUnregister(buf));
  close(fd); close(fdb);
  printf("  verdict: %s\n", fails? "FAILED -- the cold fill is NOT zero-copy into a registered region"
                                   : "OK -- pread lands in the registered region and the GPU reads it fresh");
  return fails?1:0;
}
