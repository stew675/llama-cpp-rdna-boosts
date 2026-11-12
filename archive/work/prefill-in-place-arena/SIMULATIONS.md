# The admission replay (corrected ladder, r266)

`./admission-sim.sh` on `traces/q8-lme90-code533-n200.trace` (Q8_0 122B, `-lme 90` = 200 of 256 slots per table,
538-token code prompt + 200 decode tokens; the heat prior comes from that run's frozen sidecar).  "d miss" is the
decode-miss delta against the copy path over the 200-token follow-up (11 559 misses there).  The trace itself is not
committed (4 MB) -- `traces/README.md` has the exact capture command.

prompt arm                                         h   misses    d miss prefill D2D  admitted   reclaim declined
200    copy path (r265)                       0.9515    11559       +0    66392.4         0         0        0
200    in place: landing reserve only         0.9529    11226     -333        0.0       261         0       99
200    in place: ladder (delivered)           0.9561    10458    -1101        0.0       945       393       57
200    in place: ladder, no reserve           0.9562    10443    -1116        0.0       945       945       57
200      first 16 decode passes (follow-up)   0.0000      600      -87

300    copy path (r265)                       0.9515    11559       +0    71259.8         0         0        0
300    in place: landing reserve only         0.9522    11403     -156        0.0       156         0      120
300    in place: ladder (delivered)           0.9555    10605     -954        0.0       942       426       63
300    in place: ladder, no reserve           0.9555    10614     -945        0.0       942       942       63
300      first 16 decode passes (follow-up)   0.0000      609      -78

400    copy path (r265)                       0.9515    11559       +0    73774.7         0         0        0
400    in place: landing reserve only         0.9516    11532      -27        0.0        15         0      138
400    in place: ladder (delivered)           0.9555    10608     -951        0.0      1047       573       66
400    in place: ladder, no reserve           0.9555    10608     -951        0.0      1047      1047       66
400      first 16 decode passes (follow-up)   0.0000      603      -84

538    copy path (r265)                       0.9515    11559       +0    76500.0         0         0        0
538    in place: landing reserve only         0.9517    11523      -36        0.0        18         0      138
538    in place: ladder (delivered)           0.9557    10551    -1008        0.0      1257       831       66
538    in place: ladder, no reserve           0.9557    10551    -1008        0.0      1257      1257       66
538      first 16 decode passes (follow-up)   0.0000      615      -72

`prefill D2D` is the arena->input_cpy copy the copy path makes on the prefill band; the in-place arms
make none.  `read` (the misses both arms read from storage) is identical in every arm and not printed.
