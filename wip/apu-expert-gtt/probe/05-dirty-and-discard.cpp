#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static unsigned long vmstat_kb(const char* key){
  FILE* f=fopen("/proc/vmstat","r"); char k[64]; unsigned long v, out=0;
  while(fscanf(f,"%63s %lu",k,&v)==2) if(!strcmp(k,key)) out=v;
  fclose(f); return out;
}
static void smaps(void* p,size_t SZ,const char* tag){
  /* find the region containing p, print the Private_Dirty / Shared_Clean bytes */
  FILE* f=fopen("/proc/self/smaps","r"); if(!f) return;
  char line[512]; unsigned long long start=0,end=0; bool in=false;
  unsigned long long pd=0,sc=0,sd=0,ra=0; char hdr[256]=""; bool want=false;
  while(fgets(line,sizeof line,f)){
    unsigned long long a,b;
    if(sscanf(line,"%llx-%llx",&a,&b)==2){
      if(in) break;
      if((void*)a<=p && p<(void*)b){ in=true; want=true; snprintf(hdr,sizeof hdr,"%s",line); }
    } else if(want){
      unsigned long long v;
      if(sscanf(line,"Private_Dirty: %llu kB",&v)==1) pd=v;
      if(sscanf(line,"Shared_Clean: %llu kB",&v)==1)  sc=v;
      if(sscanf(line,"Shared_Dirty: %llu kB",&v)==1)  sd=v;
      if(sscanf(line,"Rss: %llu kB",&v)==1)           ra=v;
    }
  }
  fclose(f);
  printf("    %-46s Rss=%6llu MiB  Private_Dirty=%6llu MiB  Shared_Clean=%6llu MiB  Shared_Dirty=%llu MiB\n",
         tag, ra>>10, pd>>10, sc>>10, sd>>10);
}
int main(int argc,char**argv){
  size_t SZ=4ull<<30; const char* src=argv[1];
  unsigned long sw0=vmstat_kb("pswpout"), pg0=vmstat_kb("pgpgout");

  printf("ARM 1: anonymous pool (what a writable slot pool is)\n");
  unsigned char* a=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  madvise(a,SZ,MADV_HUGEPAGE); memset(a,0x33,SZ);
  smaps(a,SZ,"anon, filled by the host");
  void* da=a; CK(hipHostRegister(a,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly)); CK(hipHostGetDevicePointer(&da,a,0));
  smaps(a,SZ,"anon, registered ReadOnly for the GPU");
  double t=now(); madvise(a,SZ,MADV_DONTNEED); 
  printf("    MADV_DONTNEED while REGISTERED: %.2f ms\n",(now()-t)*1e3);
  smaps(a,SZ,"  ...after DONTNEED while registered");
  t=now(); CK(hipHostUnregister(a));
  madvise(a,SZ,MADV_DONTNEED);
  printf("    unregister + MADV_DONTNEED: %.2f ms\n",(now()-t)*1e3);
  smaps(a,SZ,"  ...after unregister + DONTNEED");
  printf("    vmstat delta: pswpout=%lu pages (swap writeback), pgpgout=%lu pages\n",
         vmstat_kb("pswpout")-sw0, vmstat_kb("pgpgout")-pg0);

  printf("\nARM 2: read-only mapping of the GGUF (the 'arena is the file' idea)\n");
  int fd=open(src,O_RDONLY);
  unsigned char* f=(unsigned char*)mmap(nullptr,SZ,PROT_READ,MAP_SHARED,fd,0);
  for(size_t i=0;i<SZ;i+=4096)(void)f[i];
  smaps(f,SZ,"file PROT_READ MAP_SHARED, hot");
  void* df=f; CK(hipHostRegister(f,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly)); CK(hipHostGetDevicePointer(&df,f,0));
  smaps(f,SZ,"file, registered ReadOnly for the GPU");
  return 0;
}
