import os, re, subprocess
CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/dense-combo-logs"
MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
ARMS = {
 "c6s6_nonm": ["--spec-type","draft-mtp-adaptive","--spec-draft-n-max","6","--spec-draft-n-start","6"],
 "c12nm42":   ["--spec-type","draft-mtp-adaptive,ngram-mod","--spec-ngram-mod-n-match","42","--spec-draft-n-max","12","--spec-draft-n-start","12"],
 "c9nm45":    ["--spec-type","draft-mtp-adaptive,ngram-mod","--spec-ngram-mod-n-match","45","--spec-draft-n-max","9","--spec-draft-n-start","9"],
}
def accml(p):
    a=m=None
    for L in open(p, errors='ignore'):
        x=re.search(r"draft acceptance = ([\d.]+).*?mean len =\s*([\d.]+)", L)
        if x: a,m=x.groups()
    return a,m
for arm, spec in ARMS.items():
    for f in ("c1-code.txt","c4-code.txt","p3-prose.txt"):
        env=dict(os.environ); env["LD_LIBRARY_PATH"]="/opt/rocm-7.14-gfx1201/lib"; env["HIP_VISIBLE_DEVICES"]="0"
        log=f"{LOGD}/{arm}_{f}.log"
        cmd=[BIN,"-m",MODEL,"--reasoning","off","-f",f"{CORPUS}/{f}","-n","3000","--seed","42","--temp","0",
             "--single-turn","--no-display-prompt","-c","32768","-b","2048","-ub","2048",
             "-ctk","f16","-ctv","f16","-fa","auto","-ngl","99","-lv","4"]+spec
        with open(log,"w") as fh:
            rc=subprocess.run(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT,timeout=1800).returncode
        a,m=accml(log) if rc==0 else ("FAIL","")
        tps=""
        for L in open(log,errors='ignore'):
            x=re.search(r"([\d.]+) tokens per second\)",L)
            if x and 'prompt eval' not in L: tps=x.group(1)
        print(f"{arm:10s} {f:14s} rc={rc} {tps:>7} t/s acc={a} ml={m}", flush=True)
print("iso done")
