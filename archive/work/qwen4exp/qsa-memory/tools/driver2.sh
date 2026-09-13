#!/bin/bash
P=/tmp/kt/progress2.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$P"; }
say "T6 prompt-cache round-trip";      bash /tmp/kt/t6.sh 2>&1 | tee -a "$P"
say "T7 ctx-checkpoints + 4600-token gen"; bash /tmp/kt/t7.sh 2>&1 | tee -a "$P"
say "T8 MTP draft";                    bash /tmp/kt/t8.sh 2>&1 | tee -a "$P"
say "f16 coherence A/B";               bash /tmp/kt/f16-ab.sh 2>&1 | tee -a "$P"
say "driver2 complete"
