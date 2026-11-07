import os, sys
path = '/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'
fd = os.open(path, os.O_RDONLY)
off, ln = 192, 28800138240
os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED)
os.close(fd)
print('evicted %d MiB of PLE pages' % (ln//2**20))
