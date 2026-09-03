# 0016 - hc_mix: fuse the head call (il=-1, no inject) - full mix coverage

On top of 0015 (af3640f67 -> 1f05646fd). The head mixer was the last
unfused hc chain in the decode (~9 graph evals, once per token). The op
now accepts w_inject == NULL: no inject tail (dst [n_embd]), no inject
blocks in the collapse_inject dispatch (grid = collapse blocks only), and
the meta splitter allows the null src's UNKNOWN split state. The model
uses the fused branch whenever nt == 1 (drop the w_inject != nullptr
requirement); the head mixed output = the dst directly (no views).
Decode logitdump byte-identical. tg128 ~45.6 (flat: the head runs once
per token - the change completes coverage, it does not move the needle).

ALSO (scoping correction for the handover): the ffn-moe routing is
ALREADY fused by the tree's topk_moe pass (SOFTMAX->RESHAPE->ARGSORT->
VIEW->GET_ROWS -> one dispatch, labeled "FUSED SOFT_MAX" in the profile;
the logits MUL_MAT stays separate because the kernel takes the logits
values, not x+W). The moe routing = logits mm (~12-15us M=1 floor) + the
fused topk_moe kernel. The router-fusion lever in the session-6 handover
note is DEAD - nothing left to fuse there.
