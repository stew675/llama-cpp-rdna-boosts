#include <cstdio>
#include <hip/hip_runtime.h>
#include <rccl.h>

static void try_cfg(int ngpus, const char* label, int maxP2p) {
    printf("maxP2pPeers=%d (%s): ", maxP2p, label); fflush(stdout);
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) { printf("uniqueid FAILED\n"); return; }
    ncclComm_t comms[4];
    ncclConfig_t cfgs[4];
    ncclGroupStart();
    for (int i = 0; i < ngpus; i++) {
        cfgs[i] = NCCL_CONFIG_INITIALIZER;
        if (maxP2p >= 0) cfgs[i].maxP2pPeers = maxP2p;
        hipSetDevice(i);
        ncclCommInitRankConfig(&comms[i], ngpus, id, i, &cfgs[i]);
    }
    ncclResult_t r = ncclGroupEnd();
    if (r != ncclSuccess) { printf("FAILED: %s\n", ncclGetErrorString(r)); return; }
    printf("OK\n"); fflush(stdout);
    for (int i = 0; i < ngpus; i++) ncclCommDestroy(comms[i]);
}

int main() {
    int n = 2;
    try_cfg(n, "undefined", -1);
    try_cfg(n, "zero", 0);
    try_cfg(n, "one", 1);
    try_cfg(n, "2", 2);
    return 0;
}
