# probe — the harness behind `../logs/arms-2026-11-28.txt` (PR #375's assessment)

Four scripts and one generated prompt file.  They exist because the state path has no `llama-cli` arm at all
(`llama-cli` never saves a state; the file-based slot save goes through `llama_io_write_file` and never touches the
stage — see `MANIFESTS.md` gate 10), so a state-copy measurement has to drive the **in-memory prompt cache** of a
server: request A, then B on the same slot (A is saved), then A again (restored).

* `client.py PORT SRC NCH` — A / B / A again / B again, 32 greedy tokens each, prints wall, `prompt_n`, `prompt_ms`
  and a sha256 prefix of each reply.  **The shas are the correctness arm**: every arm of one build must read the
  same two shas, and a restored prompt must reply as the original did.
* `client2.py PORT SRC NCH` — A (long) then B (7 tokens): one save at a chosen size, for the above-cap arms.
* `run-arm.sh TAG BINDIR NCH [ENV=V ...]` — starts `llama-server` on the Flash-Next IQ3_XXS tree with
  `-sm tensor -ngl 99 -c 65536 --cache-ram 8192 -fa auto -lv 4`, runs `client.py`, kills the server, and prints the
  arms' `prompt cache save took` / `state get:` / `state set:` lines.
* `save-probe.sh TAG BINDIR NCH [ENV=V ...]` — the same for a single save at `-c 131072` (the >1 GiB state arm).
* `run-arm2.sh TAG BINDIR PROMPT_A PROMPT_B` — the **gate-shaped** server (35B-A3B UD-Q4_K_M, `-ncmoe 10`,
  `-c 16384`, 2 GPUs `-sm tensor`), i.e. the shape `scripts/gate-state-pipe.sh` uses.  This is the arm that shows the
  pathology is not a Flash-Next artefact: its own 165.3 MiB state read enqueues at **305-318 ms** on r378.

Every arm needs `LLAMA_STATE_TIMING=1` in the environment variable list to print `state get:` / `state set:`
(that is what splits **enqueue** from **copy+wait** — the two numbers that name the mechanism).

## The prompt file

`client.py` wants a text file of N characters; the numbers in `../README.md` §1 use `nch` = 15 500 (≈4 290 tokens),
60 000 (≈17 189) and 140 000 (≈41 678).  Generate it from the tree under test so it is reproducible:

```sh
PY=python3
$PY - <<'PY' > /tmp/state-probe-src.txt
import glob
print("\n".join(open(f, encoding="utf-8", errors="replace").read()
                for f in sorted(glob.glob("src/*.cpp"))[:6]))
PY
export SRC=/tmp/state-probe-src.txt     # save-probe.sh and run-arm.sh read $SRC
```

## One arm, end to end

```sh
cd ~/llama-cpp-rdna-boosts
./scripts/scratch-tree.sh keep r378base                                   # the release build, as the reference
./scripts/scratch-tree.sh arm held-pr375/0001-*.patch                     # the arm under test
bash wip/state-stage-per-device/probe/run-arm.sh short \
     "$HOME/llama-scratch/build-rocm/bin" 15500 LLAMA_STATE_TIMING=1
bash wip/state-stage-per-device/probe/run-arm.sh base-short \
     "$HOME/llama-scratch/build-rocm/bin-r378base" 15500 LLAMA_STATE_TIMING=1
```

Read `prompt cache save took` (the pure copy), and inside it `enqueue` vs `copy+wait`.  **Do not compare `wall`
across processes** — it contains the prefill.  Pin `HIP_VISIBLE_DEVICES` (the scripts use `0,2`, the x8 pair): an
unpinned arm lets `-sm tensor` enumerate the iGPU and abort at load with `invalid kernel file`.

An env var passed after `NCH` is a server environment variable (`GGML_CUDA_ROWS_ASYNC_GATHER=0`,
`GGML_CUDA_ROWS_OWNER_2D=0`, `LLAMA_STATE_TIMING=1`, ...), which is how the kill-switch and A/B arms in §1 were
taken.
