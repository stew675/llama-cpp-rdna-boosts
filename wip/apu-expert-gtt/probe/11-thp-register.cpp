// 11-thp-register.cpp -- does an MADV_HUGEPAGE anon region keep its transparent huge pages across
// hipHostRegister(Mapped|ReadOnly)?  (P1.0 assumption #1 of the GTT expert store.)
//
// WHY: the design's tier is "a mass of HIP registered THPs".  Registration walks pages (get_user_pages); if it
// splits every 2 MiB page into 4 KiB ones, the tier is a 4 KiB tier and both the TLB cost and the registration
// cost change.  The 2026-11-09 probes counted BASE pages only (46 GiB shard = 11.9M pages), so this was never
// answered.
//
// Measures, for one region: smaps AnonHugePages / Rss / Private_Dirty before and after registering, the
// registration time, and the GPU read rate in both states.  Also the other fault order (register first, fault
// later) because a growable store does exactly that.
//
// Build: hipcc -O2 -o 11-thp-register 11-thp-register.cpp     (ROCm 10.1, see probe/README.md)
// Run:   ./11-thp-register [MiB=1024]
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

__global__ void rd(const unsigned int* p, unsigned long long* out, size_t n4){
  size_t stride=(size_t)gridDim.x*blockDim.x, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long acc=0;
  for(; i<n4; i+=stride){ const volatile unsigned int* q=p; unsigned int a0=q[i*4],a1=q[i*4+1],a2=q[i*4+2],a3=q[i*4+3]; acc+=(unsigned long long)a0+a1+a2+a3; }
  if(acc==0xdeadbeefULL) atomicAdd(out,acc);
}
static void gpu_read(const void* p,size_t bytes,const char* tag){
  size_t n4=bytes/16; unsigned long long* d; CK(hipMalloc(&d,8)); double best=1e9;
  for(int r=0;r<6;r++){ hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
    rd<<<512,256>>>((const unsigned int*)p,d,n4); hipEventRecord(b); CK(hipEventSynchronize(b));
    float ms=0; hipEventElapsedTime(&ms,a,b); if(ms<best)best=ms; hipEventDestroy(a); hipEventDestroy(b);}
  hipFree(d); printf("    %-44s %7.1f GB/s\n",tag,bytes/(best*1e6));
}

// The VMA that contains `p`, from /proc/self/smaps: AnonHugePages, Rss, Private_Dirty.
static void smaps(const void* p, const char* tag){
  FILE* f=fopen("/proc/self/smaps","r"); if(!f){ printf("    (no smaps)\n"); return; }
  char line[512]; bool in=false; unsigned long ahp=0,rss=0,pd=0,rss_kb=0;
  unsigned long long base=0, end=0;
  while(fgets(line,sizeof line,f)){
    if(strchr(line,'-') && strstr(line," r")){ // a VMA header line
      if(in) break;
      if(sscanf(line,"%llx-%llx",&base,&end)==2 && (unsigned long long)(uintptr_t)p>=base && (unsigned long long)(uintptr_t)p<end) in=true;
      continue;
    }
    if(!in) continue;
    if(sscanf(line,"AnonHugePages: %lu kB",&rss_kb)==1) ahp+=rss_kb;
    else if(sscanf(line,"Rss: %lu kB",&rss_kb)==1) rss+=rss_kb;
    else if(sscanf(line,"Private_Dirty: %lu kB",&rss_kb)==1) pd+=rss_kb;
  }
  fclose(f);
  printf("    %-30s AnonHugePages %8.1f MiB | Rss %8.1f MiB | Private_Dirty %8.1f MiB\n",
         tag, ahp/1024.0, rss/1024.0, pd/1024.0);
}

int main(int argc,char**argv){
  size_t SZ=(argc>1? (size_t)atoi(argv[1]) : 1024)*1024*1024;
  printf("11-thp-register: %zu MiB, THP=%s\n", SZ>>20, "madvise");

  // ---- arm A: fault everything in FIRST (the load-then-register order the store uses at creation)
  unsigned char* buf=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(buf==MAP_FAILED){ printf("mmap failed\n"); return 2; }
  madvise(buf,SZ,MADV_HUGEPAGE);
  memset(buf,0x5a,SZ);
  smaps(buf,"A: faulted in, not registered");
  double t=now();
  hipError_t e=hipHostRegister(buf,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly);
  double reg=now()-t;
  printf("  hipHostRegister(Mapped|ReadOnly): %s in %.1f ms (%.2f GB/s)\n",
         e==hipSuccess?"OK":hipGetErrorString(e), reg*1e3, SZ/reg/1e9);
  if(e!=hipSuccess) return 2;
  smaps(buf,"A: REGISTERED ReadOnly");
  void* d=buf; CK(hipHostGetDevicePointer(&d,buf,0));
  gpu_read(d,SZ,"A: GPU read after registration");
  CK(hipHostUnregister(buf));
  smaps(buf,"A: after hipHostUnregister");

  // ---- arm B: register FIRST, fault later (the growable-store order)
  unsigned char* b2=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  madvise(b2,SZ,MADV_HUGEPAGE);
  smaps(b2,"B: reserved, never touched");
  t=now(); e=hipHostRegister(b2,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly); reg=now()-t;
  printf("  register an untouched region: %s in %.1f ms\n", e==hipSuccess?"OK":hipGetErrorString(e), reg*1e3);
  if(e!=hipSuccess) return 2;
  memset(b2,0x33,SZ);
  smaps(b2,"B: registered, then faulted");
  CK(hipHostGetDevicePointer(&d,b2,0));
  gpu_read(d,SZ,"B: GPU read (registered then faulted)");
  CK(hipHostUnregister(b2));
  printf("  (VmPin/VmLck read 0 for a registration hold -- use mincore(), probe 09)\n");
  return 0;
}
