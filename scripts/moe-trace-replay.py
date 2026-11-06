#!/usr/bin/env python3
"""Offline MoE expert-cache policy replayer (TODO #61 P1).

Scores expert-cache admission/placement policies against a REAL routing trace, so a policy change can
be answered from one GPU run instead of one benchmark per hypothesis.  The trace is produced by the
delivery's own cache:

    MOE_EXPERT_TRACE=/tmp/trace.txt ./llama-cli ... -n 300 --temp 0 --single-turn

and this tool replays it (stdlib only, no model, no GPU).  See `ENVIRONMENT.md`
(`MOE_EXPERT_TRACE`) for the file format, and `benchmarks/moe-cache-coverage/` for the study that
this tool was built for.

Two-tier policies (`--pool-frac`, what happens to the arena victim) are `pool-none`, `pool-retire`,
`pool-shipped`, `pool-swap-slot`, `pool-swap-weakest` and **`pool-layout`** -- the LAST is the delivered
policy: the arena is heat-seeded and the pool is RE-LAID OUT at the first scored pass as the arena's
complement plus its mirrored residents (`pool_rerank_from_heat_locked`).  Read a comparison against
`pool-none`/`pool-retire` as a lower bound for the delivery: neither models that re-rank, so both carry the
L3 cost of a pool that no longer matches the arena's actual residency.

Policies scored (all on the same trace, same arena):

  shipped   the delivered policy: per-table uniform slots, F-LRU eviction with a decay period,
            `admit=touch` doorkeeper, misses served cold.  This is the REFERENCE: its hit rate must
            land on the run's logged `h=`, otherwise the model is wrong and no comparison means
            anything.
  static    an oracle static placement: each table pre-filled with its globally hottest experts and
            never evicted.  It is an upper bound for any placement that cannot re-rank at run time.
  quota     the shipped policy, but the per-table slot count comes from the trace's per-LAYER routed
            mass instead of being uniform (campaign P2: "per-layer capacity quota").  A floor keeps
            the whole decode/verify band resident (`--min-slots`, default 8 = `--spec-draft-n-max 7`
            + 1), because a table that cannot hold a full verify batch declines the cache entirely.
  window    the shipped policy plus pass-start pre-admission: at the start of each pass the whole
            distinct expert set that pass will route is admitted up front (bounded by slots), which
            is what a prefetch-on-miss scheme can hope to emulate.

Accounting caveats (read before quoting a number):

  * The trace dedupes the identical record a tensor split produces per device, so the table set is ONE
    device's; `--arena-mib` is therefore a PER-DEVICE budget.
  * An IDENTITY (fully resident) table is not traced on the decode band (its routing is never read
    back).  Such a table cannot miss, so a trace from a fully-resident run has no decode records at
    all -- trace a partially-resident config (e.g. `MOE_EXPERT_CACHE_MIB=4096`).
  * The trace records the routing, not the timing.  `h` is comparable to the logged `h=`, but ONLY on the
    decode band: the runtime's `h=` comes from `moe_cache_update_host`'s access path, and the PREFILL band
    returns before it (whole-table staging, or a decline), so a prefill position never reaches `hits`/
    `misses`.  Measured on the 122B-A10B campaign trace: 2,145,816 reaches = 1,863 decode steps x 144 tables
    x 8 experts, i.e. the trace's 740,814 prefill positions contribute exactly zero.  Scoring them anyway
    (`--score-band all`, the old default) biases `shipped` low by ~0.15 (0.1956 vs the run's 0.3488 at
    3,932 MiB, 0.6328 vs 0.7543 at 22,517 MiB); on the decode band the model lands within 0.3 %
    (0.3492 / 0.7513).  `wip/host-pool-speed/HANDOVER.md` §4/F2 has the table.
"""

import argparse
import collections
import json
import sys


# ---------------------------------------------------------------------------------------------
# trace parsing
# ---------------------------------------------------------------------------------------------

class Table:
    __slots__ = ("layer", "role", "n_experts", "expert_bytes", "freq", "slots")

    def __init__(self, layer, role, n_experts, expert_bytes):
        self.layer = layer
        self.role = role
        self.n_experts = n_experts
        self.expert_bytes = expert_bytes
        self.freq = collections.Counter()   # expert -> touches in the whole trace
        self.slots = 0


class Rec:
    __slots__ = ("band", "layer", "role", "n_tok", "n_used", "ids")

    def __init__(self, band, layer, role, n_tok, n_used, ids):
        self.band = band
        self.layer = layer
        self.role = role
        self.n_tok = n_tok
        self.n_used = n_used
        self.ids = ids


def parse(path, mass_band="all"):
    tables = {}
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first.startswith("# moe-trace v1"):
            sys.exit("not a moe-trace v1 file: %s" % path)
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if parts[0] == "T":
                if len(parts) != 5:
                    continue
                layer, role, n_exp, eb = int(parts[1]), parts[2], int(parts[3]), int(parts[4])
                tables[(layer, role)] = Table(layer, role, n_exp, eb)
            elif parts[0] in ("P", "D"):
                if len(parts) < 6:
                    continue
                layer, role = int(parts[2]), parts[3]
                n_tok, n_used = int(parts[4]), int(parts[5])
                ids = [int(x) for x in parts[6:]]
                if len(ids) != n_tok * n_used:
                    continue   # a truncated final line
                recs.append(Rec(parts[0], layer, role, n_tok, n_used, ids))
                if mass_band != "all" and parts[0] != ("D" if mass_band == "decode" else "P"):
                    continue
                t = tables.get((layer, role))
                if t is not None:
                    for e in ids:
                        if 0 <= e < t.n_experts:
                            t.freq[e] += 1
    return tables, recs


def segment_passes(recs):
    """Split the ordered records into passes.

    A pass is one sweep of the model: the same (layer, role) table is never visited twice in it.  The
    trace is written in dispatch order, so a repeat of a table means the next sweep started.  This is
    how a decode token (or a verify batch, or a prefill ubatch) is recovered without a pass id in the
    file.
    """
    passes = []
    cur = []
    seen = set()
    for r in recs:
        key = (r.layer, r.role)
        if key in seen:
            passes.append(cur)
            cur = []
            seen = set()
        seen.add(key)
        cur.append(r)
    if cur:
        passes.append(cur)
    return passes


# ---------------------------------------------------------------------------------------------
# the cache model
# ---------------------------------------------------------------------------------------------

class TableState:
    __slots__ = ("slots", "slot_expert", "expert_slot", "count", "ghost", "last", "clock",
                 "last_decay", "hits", "misses", "cold", "fills", "evictions")

    def __init__(self, t, slots):
        self.slots = min(slots, t.n_experts)
        self.slot_expert = [-1] * self.slots
        self.expert_slot = {}
        self.count = collections.Counter()
        self.ghost = collections.Counter()
        self.last = collections.Counter()
        self.clock = 0
        self.last_decay = 0
        self.hits = 0
        self.misses = 0
        self.cold = 0
        self.fills = 0
        self.evictions = 0


def decay(st, period):
    if st.clock - st.last_decay < period:
        return
    for d in (st.count, st.ghost):
        for k in list(d.keys()):
            d[k] >>= 1
    st.last_decay = st.clock - (st.clock % period)


def access(st, expert, period, admit, touch, allow_cold=True, protect=None, no_evict=False):
    """Mirror of `access_locked`: F-LRU eviction + the fill-vs-cold admission rule.

    `protect` is the record's own expert set: an expert admitted earlier in the same token/pass must
    not be chosen as this miss's victim (the real hook passes exactly that list), which is what keeps
    one token's experts from evicting each other.

    `no_evict` is the campaign's `churn=off` arm (TODO #86): a miss may take an EMPTY slot but never
    displaces a resident, so the arena is a pure function of its seed plus the first-touch order.  No
    real policy does this; it exists to separate the seed's value from what eviction adds on top.
    """
    if no_evict:
        st.clock += 1
        decay(st, period)
        if expert in st.expert_slot:
            st.hits += 1
            st.count[expert] += 1
            st.last[expert] = st.clock
            return True
        st.misses += 1
        st.ghost[expert] += 1
        slot = next((s for s in range(st.slots) if st.slot_expert[s] < 0), -1)
        if slot < 0:
            if allow_cold:
                st.cold += 1
            return False
        st.slot_expert[slot] = expert
        st.expert_slot[expert] = slot
        st.fills += 1
        st.count[expert] += 1
        st.last[expert] = st.clock
        return True
    st.clock += 1
    decay(st, period)
    if expert in st.expert_slot:
        st.hits += 1
        st.count[expert] += 1
        st.last[expert] = st.clock
        return True
    st.misses += 1
    st.ghost[expert] += 1
    if st.slots <= 0:
        if allow_cold:
            st.cold += 1
        return False
    slot = -1
    for s in range(st.slots):
        if st.slot_expert[s] < 0:
            slot = s
            break
    victim = None
    if slot < 0:
        for s in range(st.slots):
            e = st.slot_expert[s]
            if e < 0:
                continue
            if protect is not None and e in protect:
                continue
            if victim is None or st.count[e] < st.count[victim] or \
                    (st.count[e] == st.count[victim] and st.last[e] < st.last[victim]):
                victim = e
                slot = s
        if slot < 0:
            if allow_cold:
                st.cold += 1
            return False
        if allow_cold:
            if admit == "value" and st.ghost[expert] <= st.count[victim]:
                st.cold += 1
                return False
            if admit == "touch" and st.ghost[expert] < touch:
                st.cold += 1
                return False
    if victim is not None:
        del st.expert_slot[victim]
        st.slot_expert[slot] = -1
        st.evictions += 1
    st.slot_expert[slot] = expert
    st.expert_slot[expert] = slot
    st.fills += 1
    st.count[expert] += 1           # the real code counts the use that admitted it
    st.last[expert] = st.clock
    return True


def resident_bytes(st, t):
    return len(st.expert_slot) * t.expert_bytes


# ---------------------------------------------------------------------------------------------
# slot allocation
# ---------------------------------------------------------------------------------------------

def uniform_slots(tables, arena_bytes):
    one = sum(t.expert_bytes for t in tables.values())
    return max(0, int(arena_bytes / one)) if one else 0


def quota_slots(tables, arena_bytes, mass_by_layer, min_slots):
    """Per-layer slot counts sized by routed mass (campaign P2), with a verify-band floor.

    `mass_by_layer[l]` is the routed mass of layer `l` (summed over its tables).  Solve for the scale
    `alpha` such that `sum_l slots_l * bytes_l == arena`, with `slots_l = clamp(alpha*mass_l/bytes_l,
    floor_l, cap_l)`.  `floor_l` = min_slots and `cap_l` = the layer's smallest `n_experts`, so
    gate/up/down stay aligned (the C++ rearm path keys on a per-layer slot count) and a layer can
    always hold a full decode/verify batch.
    """
    layers = collections.defaultdict(list)
    for t in tables.values():
        layers[t.layer].append(t)
    bytes_l = {l: sum(t.expert_bytes for t in ts) for l, ts in layers.items()}
    cap_l = {l: min(t.n_experts for t in ts) for l, ts in layers.items()}
    floor_l = {l: min(min_slots, cap_l[l]) for l in layers}

    def alloc(alpha):
        return {l: int(min(cap_l[l], max(floor_l[l], alpha * mass_by_layer.get(l, 0) / bytes_l[l])))
                for l in layers}

    def total(a):
        return sum(a[l] * bytes_l[l] for l in a)

    lo, hi = 0.0, 1.0
    while total(alloc(hi)) <= arena_bytes and hi < 1e12:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(alloc(mid)) <= arena_bytes:
            lo = mid
        else:
            hi = mid
    a = alloc(lo)
    # Hand any rounding leftovers to the hottest layers, largest remaining mass first, while they fit.
    order = sorted(layers, key=lambda l: -mass_by_layer.get(l, 0))
    for l in order:
        while a[l] < cap_l[l] and total(a) + bytes_l[l] <= arena_bytes:
            a[l] += 1
    return a


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------

def seed_prior(tables, passes, seed, sidecar=None, probe_passes=0, mass_band="decode"):
    """The arena's load-time prior: {(layer, role): [count per expert]} (TODO #86 step 0).

    `none`  no seed at all -- the model the campaign calibrated against an `MOE_EXPERT_HEATMAP=0` run.
    `heat`  this trace's own routing on the scored mass band: an ORACLE prior, and exactly the ranking
            `static` uses (so `heat` x `churn=off` must reproduce `static` bit for bit).
    `probe` the trace's first `probe_passes` passes only -- a same-session proxy for a prior, which is
            the most a run can know before it starts generating.
    `sidecar:<path>` a REAL persisted `.moe-heatmap` from a previous run: the achievable prior and the one the
            delivery actually seeds from (`apply_heat_seed_locked`).  An **MHT2** map is ranked by its
            DECODE counter (the band the arena serves, TODO #88); an **MHT1** map has only the combined
            one, which is then used for both -- the pre-r114 behaviour.
    """
    if seed == "none":
        return None
    if seed == "heat":
        return {k: [t.freq.get(e, 0) for e in range(t.n_experts)] for k, t in tables.items()}
    if seed == "probe":
        want = None if mass_band == "all" else ("D" if mass_band == "decode" else "P")
        counts = collections.defaultdict(collections.Counter)
        for p in passes[:max(0, probe_passes)]:
            for r in p:
                if want is not None and r.band != want:
                    continue
                for e in r.ids:
                    counts[(r.layer, r.role)][e] += 1
        return {k: [counts[k].get(e, 0) for e in range(t.n_experts)] for k, t in tables.items()}
    if seed == "sidecar":
        return sidecar or {}
    sys.exit("unknown --seed %r" % seed)


def apply_seed(states, tables, prior):
    """Mirror of `apply_heat_seed_locked`: rank by the prior (count desc, expert id asc) and place the top
    `slots` in slot order with `count=0, ghost=0, last=0` -- a pre-filled expert has not been USED yet, so it
    must be the first thing eviction gives up (seeding `count=1` made arbitrary residents outlive genuinely
    hot ones and cost the real cache ~800 tokens of hit rate)."""
    for key, t in tables.items():
        st = states.get(key)
        prior_t = prior.get(key) if prior else None
        if st is None or st.slots <= 0 or not prior_t or len(prior_t) != t.n_experts:
            continue
        if not sum(prior_t):
            continue
        order = sorted(range(t.n_experts), key=lambda e: (-prior_t[e], e))[:st.slots]
        for s, e in enumerate(order):
            st.slot_expert[s] = e
            st.expert_slot[e] = s
            st.count[e] = 0
            st.ghost[e] = 0
            st.last[e] = 0


def score(tables, passes, slots_by_layer=None, arena_bytes=0, period=32, admit="touch", touch=2,
          policy="shipped", prior=None, churn=True, score_band="all"):
    states = {}
    for key, t in tables.items():
        slots = slots_by_layer.get(t.layer, 0) if slots_by_layer is not None else t.slots
        states[key] = TableState(t, slots)

    if policy == "static":
        # oracle: each table pre-filled with its globally hottest experts, never evicted
        for key, t in tables.items():
            st = states[key]
            top = [e for e, _ in t.freq.most_common(st.slots)]
            for s, e in enumerate(top):
                st.slot_expert[s] = e
                st.expert_slot[e] = s
                st.last[e] = 0
    elif prior is not None:
        apply_seed(states, tables, prior)

    tokens = 0
    for p in passes:
        band_tokens = p[0].n_tok if p else 0
        if score_band == "decode" and band_tokens > 64:
            continue    # the prefill band stages whole tables; scoring it only blurs a policy comparison
        tokens += band_tokens
        if policy == "window":
            # pass-start pre-admission of the whole distinct routed set, hottest-miss first
            want = collections.defaultdict(set)
            for r in p:
                want[(r.layer, r.role)].update(r.ids)
            for key, experts in want.items():
                st = states.get(key)
                t = tables.get(key)
                if st is None or t is None:
                    continue
                have = len(st.expert_slot)
                if have >= st.slots:
                    continue
                for e in experts:
                    if have >= st.slots:
                        break
                    if e in st.expert_slot or not (0 <= e < t.n_experts):
                        continue
                    slot = next((s for s in range(st.slots) if st.slot_expert[s] < 0), -1)
                    if slot < 0:
                        break
                    st.slot_expert[slot] = e
                    st.expert_slot[e] = slot
                    have += 1
        for r in p:
            key = (r.layer, r.role)
            st = states.get(key)
            t = tables.get(key)
            if st is None or t is None:
                continue
            protect = set(r.ids)
            for e in r.ids:
                if 0 <= e < t.n_experts:
                    if policy == "static":
                        st.clock += 1
                        if e in st.expert_slot:
                            st.hits += 1
                        else:
                            st.misses += 1
                    else:
                        access(st, e, period, admit, touch, protect=protect, no_evict=not churn)

    hits = sum(s.hits for s in states.values())
    misses = sum(s.misses for s in states.values())
    cold = sum(s.cold for s in states.values())
    fills = sum(s.fills for s in states.values())
    evict = sum(s.evictions for s in states.values())
    reaches = hits + misses
    # Traffic: a fill and a cold read are both one expert's bytes over the host link.
    traffic = sum((s.fills + s.cold) * tables[k].expert_bytes for k, s in states.items())
    arena = sum(resident_bytes(s, tables[k]) for k, s in states.items())
    # Routed-mass coverage of the FINAL resident set, on this trace's routing.
    total_mass = sum(sum(t.freq.values()) for t in tables.values())
    covered = sum(t.freq[e] for k, t in tables.items() for e in states[k].expert_slot)
    return {
        "policy": policy,
        "seeded": prior is not None, "churn": churn,
        "hits": hits, "misses": misses, "reaches": reaches,
        "h": (hits / reaches) if reaches else 0.0,
        "cold": cold, "fills": fills, "evictions": evict,
        "tokens": tokens,
        "mib_per_token": (traffic / tokens / 1048576.0) if tokens else 0.0,
        "resident_mib": arena / 1048576.0,
        "covered_mass_pct": (100.0 * covered / total_mass) if total_mass else 0.0,
        "slots": {str(k): s.slots for k, s in states.items()},
    }


# ---------------------------------------------------------------------------------------------
# two-tier model (TODO #87): arena (L1) + pinned host pool (L2) + mapping/disk (L3)
# ---------------------------------------------------------------------------------------------

class TwoTier:
    """Per-table arena + host pool.  Mirrors the C++ slot maps so the copy/read accounting is literal:
    every expert that reaches the arena crosses the host link once (pool->arena, or L3->pool->arena),
    and a hand-over is whatever extra copies the policy needs to move the arena victim into the pool."""

    __slots__ = ("warm_on_evict", "slots_a", "slots_p", "a_slot", "a_map", "a_count", "a_ghost", "a_last",
                 "p_slot", "p_map", "p_last", "clock", "pclock", "last_decay",
                 "hits", "misses", "p_hits", "p_cold", "l3", "promote", "handover", "evictions",
                 "p_evict", "rebalances", "warm", "warm_hits")

    def __init__(self, t, slots_a, slots_p, warm_on_evict=False):
        self.warm_on_evict = warm_on_evict
        self.slots_a = min(slots_a, t.n_experts)
        self.slots_p = min(slots_p, t.n_experts)
        self.a_slot = [-1] * self.slots_a
        self.a_map = {}
        self.a_count = collections.Counter()
        self.a_ghost = collections.Counter()
        self.a_last = collections.Counter()
        self.clock = 0
        self.p_slot = [-1] * self.slots_p
        self.p_map = {}
        self.p_last = collections.Counter()
        self.pclock = 0
        self.last_decay = 0
        self.hits = self.misses = self.p_hits = self.p_cold = self.l3 = self.promote = 0
        self.handover = self.evictions = self.p_evict = self.rebalances = 0
        self.warm = set()      # experts whose bytes were (or will be) pulled into the page cache off-path
        self.warm_hits = 0     # misses served by such a warmed page instead of a cold storage read

    # -- pool helpers ---------------------------------------------------------
    def p_free(self):
        for s in range(self.slots_p):
            if self.p_slot[s] < 0:
                return s
        return -1

    def p_weakest(self):
        s_best, e_best = -1, None
        for s in range(self.slots_p):
            e = self.p_slot[s]
            if e < 0:
                continue
            if e_best is None or self.p_last[e] < self.p_last[e_best]:
                s_best, e_best = s, e
        return s_best

    def p_put(self, slot, expert):
        old = self.p_slot[slot]
        if old >= 0:
            self.p_map.pop(old, None)
        prev = self.p_map.get(expert)   # mirror `pool_slot_assign_locked`: never two indices for one expert
        if prev is not None and prev != slot and 0 <= prev < self.slots_p and self.p_slot[prev] == expert:
            self.p_slot[prev] = -1
        self.p_slot[slot] = expert
        self.p_map[expert] = slot
        self.p_last[expert] = self.pclock

    def p_drop(self, expert):
        slot = self.p_map.pop(expert, None)
        if slot is not None and self.p_slot[slot] == expert:
            self.p_slot[slot] = -1

    # -- one routed expert ------------------------------------------------
    def cold_read(self, e):
        """A miss that is NOT admitted (or cannot be) still reads the expert's bytes once.  Where from?
        In pool mode with `MOE_HOST_POOL_COLD` (default 1) a pool-resident expert is read IN PLACE from
        the pool's device alias -- a PCIe read of pinned RAM, NOT a disk read.  Only an expert in neither
        tier goes to the mapping/disk (L3).  Mirroring that is what makes `l3` comparable to
        /proc/diskstats: measured on the 122B at `-ncmoe 48`, the disk reads per decode token are
        ~163 / ~53 / ~13 MiB at pool FRAC 50 / 80 / 100, and conflating pool cold reads with L3 reads
        predicted ~460 MiB/token at every FRAC."""
        if self.slots_p > 0 and e in self.p_map:
            self.p_cold += 1
            self.p_last[e] = self.pclock
            return
        if e in self.warm:
            self.warm.discard(e)            # the page cache already has it: a PCIe read, not a storage read
            self.warm_hits += 1
            return
        self.l3 += 1

    def rebalance(self, mode="carry", prior_row=None, k=1):
        """Up to `k` swaps in one round: the hottest POOL expert that is not resident takes the COLDEST arena
        slot, and only if it is hotter than that slot's resident.  The counters travel with the bytes, so the
        decision uses exactly what the cache already keeps: `a_count` (decaying uses while resident) and
        `a_ghost` (decaying demand while not resident).  Returns the extra copies paid (2 per swap: the victim
        D2H and the promotion H2D -- a later miss would have paid the H2D anyway, but here it is proactive).

        This is the maintainer's proposal (2026-11-05) and it is the cheapest possible demote: unlike
        `pool-swap-slot` it never moves a victim merely because something was demanded, so a routing detour
        cannot churn the tier, and unlike the layout it can repair a prior that guessed wrong.

        `k > 1` repairs more of the ranking mismatch out of the SAME pass-boundary decision.  Each iteration
        re-reads the coldest resident, which the previous swap has already replaced with a hot expert, so the
        round terminates early (`_rebalance_one` returns 0) once the guard refuses."""
        paid = 0
        for _ in range(max(1, k)):
            n = self._rebalance_one(mode, prior_row)
            if n == 0:
                break          # the guard refused: nothing left to repair in this round
            paid += n
        return paid

    def _rebalance_one(self, mode, prior_row):
        if self.slots_a <= 0 or self.slots_p <= 0:
            return 0
        heat = (lambda e: prior_row[e]) if mode == "prior" and prior_row else \
               (lambda e: self.a_ghost[e])
        v_best, v_slot = None, -1
        for s in range(self.slots_a):
            e = self.a_slot[s]
            if e < 0:
                continue
            if v_best is None or self.a_count[e] < self.a_count[v_best] or \
                    (self.a_count[e] == self.a_count[v_best] and self.a_last[e] < self.a_last[v_best]):
                v_best, v_slot = e, s
        if v_best is None:
            return 0
        c_best = None
        for e in self.p_map:
            if e in self.a_map:
                continue
            if c_best is None or heat(e) > heat(c_best) or (heat(e) == heat(c_best) and e < c_best):
                c_best = e
        v_heat = prior_row[v_best] if (mode == "prior" and prior_row) else self.a_count[v_best]
        if c_best is None:
            return 0
        if mode != "blind" and heat(c_best) <= v_heat:
            return 0   # `blind` = the guard-off A/B arm: always rotate, whatever the heat says
        p_slot = self.p_map.pop(c_best, -1)
        if 0 <= p_slot < self.slots_p and self.p_slot[p_slot] == c_best:
            self.p_slot[p_slot] = -1
        old = self.a_count[v_best]
        self.a_map.pop(v_best, None)
        self.a_slot[v_slot] = c_best
        self.a_map[c_best] = v_slot
        if mode == "carry":
            # the hotness travels with the expert, between the two counters
            self.a_count[c_best] = max(self.a_count[c_best], self.a_ghost[c_best])
        else:
            # the SHIPPED bookkeeping: an admission starts the resident count from 1 ...
            self.a_count[c_best] = 1
        self.a_last[c_best] = self.clock
        self.a_ghost[c_best] = 0
        self.a_count[v_best] = 0
        # ... and an eviction does NOT remember what the victim had demonstrated while resident
        self.a_ghost[v_best] = max(self.a_ghost[v_best], old) if mode == "carry" else 0
        if 0 <= p_slot:
            self.pclock += 1
            self.p_put(p_slot, v_best)
        self.rebalances += 1
        return 2

    def access(self, e, policy, protect, period, admit, touch):
        self.clock += 1
        if self.clock - self.last_decay >= period:
            for d in (self.a_count, self.a_ghost):
                for k in list(d.keys()):
                    d[k] >>= 1
            self.last_decay = self.clock - (self.clock % period)
        if e in self.a_map:
            self.hits += 1
            self.a_count[e] += 1
            self.a_last[e] = self.clock
            return
        self.misses += 1
        self.a_ghost[e] += 1
        if self.slots_a <= 0:
            self.cold_read(e)               # served cold: the pool alias when resident, else L3
            return
        slot = -1
        for s in range(self.slots_a):
            if self.a_slot[s] < 0:
                slot = s
                break
        victim = None
        if slot < 0:
            for s in range(self.slots_a):
                v = self.a_slot[s]
                if v < 0 or v in protect:
                    continue
                if victim is None or self.a_count[v] < self.a_count[victim] or \
                        (self.a_count[v] == self.a_count[victim] and self.a_last[v] < self.a_last[victim]):
                    victim, slot = v, s
            if slot < 0:
                self.cold_read(e)           # every arena slot is protected: served cold
                return
            if admit == "value" and self.a_ghost[e] <= self.a_count[victim]:
                self.cold_read(e)
                return
            if admit == "touch" and self.a_ghost[e] < touch:
                self.cold_read(e)
                return
        # ---- where do the incoming bytes come from?
        if self.slots_p > 0 and e in self.p_map:
            self.p_hits += 1
            self.promote += 1               # pool -> arena
            self.p_last[e] = self.pclock
        else:
            self.l3 += 1                    # mapping / disk
            self.promote += 1               # whatever filled it is copied into the arena
            if self.slots_p > 0:
                free = self.p_free()
                if free >= 0:
                    self.pclock += 1
                    self.p_put(free, e)     # an L3 read fills an EMPTY slot, never evicts
        # ---- what happens to the arena victim?
        if victim is not None:
            self.a_map.pop(victim, None)    # the victim leaves the arena (its bytes are overwritten below)
            self.evictions += 1
            if self.warm_on_evict:
                self.warm.add(victim)       # the warmer reads it into the page cache, off the critical path
            if self.slots_p <= 0:
                pass
            elif policy == "pool-none":
                pass                        # the pool keeps whatever it had
            elif policy == "pool-shipped":
                self.handover += 1          # the victim is re-read into the pool (1.0 copy equivalent)
                if self.p_free() < 0:
                    w = self.p_weakest()
                    if w >= 0:
                        self.p_evict += 1
                        # 2026-11-02: this was `self.p_slot[w] = -1`, which freed the slot but left the
                        # evicted expert in `p_map` forever -- p_map then grew past `slots_p` and
                        # `e in p_map` turned phantom experts into pool hits (measured: 26,769 "resident"
                        # experts in 9,216 slots at --pool-frac 25).  Drop it properly, as the C++ does.
                        self.p_drop(self.p_slot[w])
                free = self.p_free()
                if free >= 0:
                    self.pclock += 1
                    self.p_put(free, victim)
            elif policy == "pool-retire":
                # No copies at all: a promoted expert leaves the pool (its slot is freed), the arena victim
                # is simply dropped, and a later L3 read fills a free slot.  The pool drifts toward "the
                # experts the arena does not hold plus what was demanded from L3".
                self.p_drop(e)
            elif policy == "pool-swap-slot":
                if e in self.p_map:          # the slot the incoming just left takes the victim
                    s_in = self.p_map.pop(e)
                    self.p_slot[s_in] = -1
                    self.pclock += 1
                    self.p_put(s_in, victim)
                    self.handover += 2        # staged out through a pinned bounce, then in
                else:
                    free = self.p_free()
                    if free >= 0:
                        self.pclock += 1
                        self.p_put(free, victim)
                        self.handover += 1
            elif policy == "pool-swap-weakest":
                if e in self.p_map:
                    self.p_drop(e)            # retire the incoming's slot (it is in VRAM now)
                if victim not in self.p_map:
                    w = self.p_weakest()
                    if w >= 0:
                        self.p_evict += 1
                        self.pclock += 1
                        self.p_put(w, victim)
                        self.handover += 1
        self.a_slot[slot] = e
        self.a_map[e] = slot
        self.a_count[e] += 1
        self.a_last[e] = self.clock


def load_heat_sidecar(path, band="decode"):
    """Read a real `.moe-heatmap` sidecar: {(layer, role): [heat per expert]} for `band`.

    Same layout as `heat_map_save_locked` in `ggml/src/ggml-cuda/moe-expert-cache.cu`: a 32-byte header
    (magic, version, n_tables, reserved, model_bytes, n_tensors, reserved) then, per table, an 80-byte
    record (int32 layer, char role[64], uint32 n_experts, uint64 host_bytes) followed by the counters.
    Two formats exist:

      MHT1/v1  one counter per expert -- the prefill and decode bands MIXED, which ranks the arena by a
               prefill-dominated distribution (TODO #88).  `band="decode"` has to fall back to it.
      MHT2/v2  TWO arrays per table: `all` (prefill + decode, what the host pool's layout ranks by) then
               `dec` (the decode band alone, what the arena seed ranks by).

    `band="decode"` (the arena seed) prefers `dec` and falls back to the combined vector when a map has no
    decode mass; `band="all"` is the pool's ranking.  A truncated/corrupt file is ignored (a cold start),
    never fatal.
    """
    import struct
    out = {}
    with open(path, "rb") as f:
        hdr = f.read(32)
        if len(hdr) != 32:
            return out
        magic = hdr[:4]
        if magic not in (b"MHT1", b"MHT2"):
            return out
        _magic, version, n_tables, _r0, _mb, _nt, _r1 = struct.unpack("<4sIIIQII", hdr)
        v2 = magic == b"MHT2" and version == 2
        if not v2 and not (magic == b"MHT1" and version == 1):
            return out
        for _ in range(n_tables):
            rec = f.read(80)
            if len(rec) != 80:
                break
            layer, role, n_exp, _hb = struct.unpack("<i64sIQ", rec)
            role = role.split(b"\0", 1)[0].decode("utf-8", "replace")
            # The sidecar stores the GGUF tensor name (`blk.0.ffn_gate_exps.weight`); the trace stores the
            # cache's role (`ffn_gate_exps.weight`) and the layer number separately.  Normalise, or every
            # lookup misses -- which used to be invisible: the pool init fell back to the ORACLE (the
            # trace's own counts) and reported the oracle's L3 as if it were the sidecar's.
            if role.startswith("blk."):
                parts = role.split(".", 2)
                if len(parts) == 3 and parts[1].isdigit():
                    role = parts[2]
            if n_exp == 0 or n_exp > (1 << 22):
                break
            data = f.read(4 * n_exp)
            if len(data) != 4 * n_exp:
                break
            comb = list(struct.unpack("<%dI" % n_exp, data))
            dec = None
            if v2:
                data = f.read(4 * n_exp)
                if len(data) != 4 * n_exp:
                    break
                dec = list(struct.unpack("<%dI" % n_exp, data))
            if band == "all":
                out[(layer, role)] = comb
            else:
                out[(layer, role)] = dec if (dec is not None and sum(dec) > 0) else comb
    return out


def score_pool(tables, passes, slots_a, slots_p, policy, pool_init="heat", period=32, admit="touch",
               touch=2, decode_only=True, sidecar=None, rebalance=0, rebalance_heat="carry", evict_warm=False,
               rebalance_prior=None, rebalance_k=1):
    states = {}
    for key, t in tables.items():
        states[key] = TwoTier(t, slots_a, slots_p.get(key, 0) if isinstance(slots_p, dict) else slots_p,
                              warm_on_evict=evict_warm)

    # The pool's ranking source (`pool_rank_experts_locked`): the prior, desc, ties by expert id, and
    # expert id where this table has no prior.  NEVER fall back to `heat` for a missing sidecar entry --
    # that silently turns a sidecar measurement into the oracle's number (the r114 correction).
    base_init = pool_init[:-len("-complement")] if pool_init.endswith("-complement") else pool_init
    complement = pool_init.endswith("-complement")

    def pool_order(key, t):
        if base_init == "heat":
            # The ORACLE prior (this trace's own counts) -- an upper bound, never the achievable one.
            order = [e for e, _ in sorted(t.freq.items(), key=lambda kv: -kv[1])]
            return order + [e for e in range(t.n_experts) if t.freq.get(e, 0) == 0]
        prior = (sidecar or {}).get(key) if base_init == "sidecar" else None
        if prior is not None and len(prior) == t.n_experts:
            return sorted(range(t.n_experts), key=lambda e: (-prior[e], e))
        return list(range(t.n_experts))

    def lay_pool(key, st, t, mirror_residents):
        """Lay the pool out over its slots.  `mirror_residents` is the shipped re-rank
        (`pool_rerank_from_heat_locked`): the COMPLEMENT first, then the arena-resident experts in the same
        rank order while capacity remains.  Without it a pool that no longer matches the arena's actual
        residency is never repaired -- which is what made the old `pool-none` row read 2-4x too much L3."""
        order = pool_order(key, t)
        if mirror_residents:
            order = [e for e in order if e not in st.a_map] + [e for e in order if e in st.a_map]
        else:
            order = [e for e in order if e not in st.a_map]
        for s_ in range(st.slots_p):
            st.p_slot[s_] = -1
        st.p_map.clear()
        for s_, e_ in enumerate(order[:st.slots_p]):
            st.p_slot[s_] = e_
            st.p_map[e_] = s_
            st.p_last[e_] = 0

    # Initial pool content: the load-time prewarm.  `heat` is the ORACLE prior (ranked by the whole
    # trace's routing), `id` is what a prewarm with no tally does, `none` starts empty (the prefill
    # then fills it in staging order, which is expert id order).
    if pool_init != "none":
        for key, t in tables.items():
            st = states[key]
            if st.slots_p <= 0:
                continue
            if complement or policy == "pool-layout":
                # The real system heat-seeds the ARENA at sizing, so the pool's job is the complement of
                # the arena, ranked by the same prior: that is what makes arena U pool cover the set.
                for s_, e_ in enumerate(pool_order(key, t)[:st.slots_a]):
                    st.a_slot[s_] = e_
                    st.a_map[e_] = s_
            # `pool-layout` also mirrors the arena's residents into whatever capacity the complement does
            # NOT use -- without that a budget ABOVE the complement leaves slots EMPTY at init (a 100 %
            # budget left 220 of 512 empty and read 16.6 MiB/token), which the shipped re-rank never does.
            lay_pool(key, st, t, mirror_residents=(policy == "pool-layout"))

    # `pool-layout` = the SHIPPED policy: the arena is heat-seeded, and at the first scored pass the pool is
    # re-laid out as the arena's complement plus its mirrored residents (`pool_rerank_from_heat_locked`).
    # Without this relayout the pool keeps the set it was prewarmed with, the arena drifts away from its seed,
    # and the residual L3 reads are attributed to the policy instead of to the missing re-rank: that is what
    # made the old `pool-none` row read 26.1 MiB/token where the shipped cache reads the uncovered fraction.
    relayout_pending = policy == "pool-layout"
    scored_pass = 0
    tokens = 0
    for p in passes:
        band_tokens = p[0].n_tok if p else 0
        is_decode = band_tokens <= 64
        if is_decode:
            tokens += band_tokens
        # `--rebalance N`: every N scored passes, give each table one hottest-pool/coldest-arena swap if the
        # pool side is hotter.  Applied AFTER the pass whose index is a multiple of N has run (a pass boundary
        # is where the shipped cache does its own policy work, and where a graph is not in flight).
        if rebalance > 0 and (is_decode or not decode_only) and scored_pass % rebalance == 0 and scored_pass > 0:
            for k_, st_ in states.items():
                row = (rebalance_prior or {}).get(k_) if rebalance_heat == "prior" else None
                st_.handover += st_.rebalance(mode=rebalance_heat, prior_row=row, k=rebalance_k)
        if is_decode or not decode_only:
            scored_pass += 1
        if relayout_pending and (is_decode or not decode_only):
            for key_, t_ in tables.items():
                if states[key_].slots_p > 0:
                    lay_pool(key_, states[key_], t_, mirror_residents=True)
            relayout_pending = False
        for r in p:
            key = (r.layer, r.role)
            st = states.get(key)
            t = tables.get(key)
            if st is None or t is None:
                continue
            if decode_only and not is_decode:
                continue
            protect = set(r.ids)
            for e in r.ids:
                if 0 <= e < t.n_experts:
                    st.access(e, policy, protect, period, admit, touch)

    hits = sum(s.hits for s in states.values())
    misses = sum(s.misses for s in states.values())
    p_hits = sum(s.p_hits for s in states.values())
    warm_hits = sum(s.warm_hits for s in states.values())
    l3 = sum(s.l3 for s in states.values())
    p_cold = sum(s.p_cold for s in states.values())
    promote = sum(s.promote for s in states.values())
    handover = sum(s.handover for s in states.values())
    evict = sum(s.evictions for s in states.values())
    bytes_of = lambda k: tables[k].expert_bytes
    l3_mib = sum(s.l3 * bytes_of(k) for k, s in states.items()) / 1048576.0
    promote_mib = sum(s.promote * bytes_of(k) for k, s in states.items()) / 1048576.0
    handover_mib = sum(s.handover * bytes_of(k) for k, s in states.items()) / 1048576.0
    reaches = hits + misses
    resident = sum(len(s.a_map) * bytes_of(k) for k, s in states.items()) / 1048576.0
    pool_res = sum(len(s.p_map) * bytes_of(k) for k, s in states.items()) / 1048576.0
    total_mass = sum(sum(t.freq.values()) for t in tables.values())
    covered = sum(t.freq[e] for k, t in tables.items() for e in states[k].a_map)
    return {
        "policy": policy, "hits": hits, "misses": misses, "reaches": reaches,
        "h": (hits / reaches) if reaches else 0.0,
        "pool_h": (p_hits / misses) if misses else 0.0,
        "pool_cold": p_cold,
        "pool_cold_pct": (100.0 * p_cold / misses) if misses else 0.0,
        "l3": l3, "l3_mib": l3_mib, "l3_mib_per_tok": (l3_mib / tokens) if tokens else 0.0,
        "promote_mib": promote_mib, "handover_mib": handover_mib,
        "hostlink_mib": promote_mib + handover_mib,
        # PER-TOKEN columns, using the same `tokens` denominator as `l3_mib_per_tok` (the DECODE band only:
        # a prefill position never reaches the cache's access path).  Dividing by the trace's total token
        # count instead understates this by ~6x on a 5k-token prompt and silently inflates any
        # "L3 MiB saved per MiB copied" ratio -- quote THESE columns, never hand-divided ones.
        "handover_mib_per_tok": (handover_mib / tokens) if tokens else 0.0,
        "hostlink_mib_per_tok": ((promote_mib + handover_mib) / tokens) if tokens else 0.0,
        "warm_hits": warm_hits,
        "warm_mib_per_tok": (warm_hits * sum(tables[k].expert_bytes for k in tables) / len(tables) / 1048576.0 / tokens) if tokens else 0.0,
        "copies_per_saved": (handover_mib / max(1e-9, (l3_mib))) if l3_mib else 0.0,
        "handover_per_miss": (handover / misses) if misses else 0.0,
        "evictions": evict, "tokens": tokens,
        "resident_mib": resident, "pool_resident_mib": pool_res,
        "covered_mass_pct": (100.0 * covered / total_mass) if total_mass else 0.0,
    }


def static_bound(tables, arena_bytes):
    """Routed-mass coverage of a GLOBAL top-f resident set of the same total slot count."""
    flat = []
    for t in tables.values():
        flat.extend(t.freq.values())
    n_slots = sum(t.slots for t in tables.values())
    flat.sort(reverse=True)
    total = sum(flat)
    if total == 0:
        return 0.0
    return 100.0 * sum(flat[:max(0, min(n_slots, len(flat)))]) / total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--arena-mib", type=float, default=None,
                    help="per-device arena budget in MiB (default: derive the uniform slot count from "
                         "--slots, else 4096)")
    ap.add_argument("--slots", type=int, default=None, help="force the uniform per-table slot count")
    ap.add_argument("--devices", type=int, default=1,
                    help="tensor-split device count.  The trace records the FULL host expert size, while "
                         "the arena holds one device's slice, so the per-device expert bytes are "
                         "expert_bytes/devices (default 1 = unsplit)")
    ap.add_argument("--period", type=int, default=32, help="F-LRU decay period (MOE_EXPERT_CACHE_PERIOD)")
    ap.add_argument("--admit", choices=("always", "value", "touch"), default="touch",
                    help="admission rule (MOE_EXPERT_CACHE_ADMIT); 'always' = 0, 'value' = 1, 'touch' = 2")
    ap.add_argument("--touch", type=int, default=2, help="MOE_EXPERT_CACHE_TOUCH")
    ap.add_argument("--min-slots", type=int, default=8,
                    help="quota floor: the whole decode/verify band must fit (default 8 = n_max 7 + 1)")
    ap.add_argument("--mass-band", choices=("all", "decode", "prefill"), default="decode",
                    help="which band's routed mass drives the quota policy (default decode: that is the "
                         "band the arena serves; 'all' is dominated by the prefill, which routes ~n_tok "
                         "experts per layer in one pass, and a prefill pass can also lose one table to "
                         "the staged path, which would distort the ranking)")
    ap.add_argument("--policies", default="shipped,static,quota,window")
    ap.add_argument("--seed", default="none",
                    help="arena pre-fill at sizing, i.e. `apply_heat_seed_locked` (TODO #86 step 0): 'none' = the "
                         "unseeded model the campaign calibrated against an MOE_EXPERT_HEATMAP=0 run; 'heat' = "
                         "this trace's whole-trace routing (an ORACLE prior -- the ranking `static` uses); "
                         "'probe:N' = the trace's first N passes (a same-session prior, the most a run can know "
                         "before generating); 'sidecar:<path>' = a REAL persisted `.moe-heatmap` (MHT2 ranked by "
                         "its DECODE counter, or MHT1 read as dec=all) from a previous run -- the achievable "
                         "prior, and what the delivery seeds from")
    ap.add_argument("--churn", choices=("on", "off"), default="on",
                    help="'off' = a miss may take an EMPTY arena slot but never evicts a resident, so the arena "
                         "is a pure function of its seed.  No shipped policy does this: it isolates the seed's "
                         "value (foreknowledge) from what eviction adds (churn), which is the #86 decomposition")
    ap.add_argument("--pool-frac", type=int, default=None,
                    help="model `--host-experts pool`: a pinned L2 of this %% of each table's experts in "
                         "front of the arena.  Selects the two-tier policy family (pool-*) and is REQUIRED "
                         "for them")
    ap.add_argument("--rebalance", type=int, default=0, metavar="N",
                    help="every N scored passes, swap each table's COLDEST arena resident with its HOTTEST "
                         "non-resident pool expert, but only if the pool expert is hotter (decaying "
                         "`a_ghost` > `a_count`).  0 = off.  A rate-limited, conditional demote: it repairs a "
                         "prior that guessed wrong without churning the tier on every admission.")
    ap.add_argument("--rebalance-heat", choices=("carry", "reset", "prior", "blind"), default="carry",
                    help="which heat the `--rebalance` swap compares: 'carry' transfers the hotness between "
                         "the resident (`count`) and non-resident (`ghost`) counters as the bytes move; 'reset' "
                         "is the SHIPPED bookkeeping (an admission starts `count` at 1, an eviction forgets "
                         "what the victim demonstrated); 'prior' compares the static persisted prior instead "
                         "of either decayed counter (the `heat`/`heat_dec` vector).")
    ap.add_argument("--rebalance-k", type=int, default=1, metavar="K",
                    help="swaps per table per `--rebalance` round (default 1).  A round at K>1 repairs more of "
                         "the ranking mismatch at once out of the SAME pass-boundary decision, so it should be "
                         "more copy-efficient than K rounds of K=1: each swap re-reads the coldest resident, "
                         "which a previous swap in the same round has already made hot.")
    ap.add_argument("--evict-warm", action="store_true", default=False,
                    help="model an eviction WARMER: each arena victim is pulled into the page cache off the "
                         "critical path (a background/readahead read, nothing moved by us), so its next miss "
                         "is a page-cache hit instead of a storage read.  The bandwidth is still spent -- it "
                         "is reported as `warm/tok` -- but the latency is not on the token path.")
    ap.add_argument("--pool-band", choices=("all", "decode"), default="all",
                    help="which persisted counter ranks the POOL (default 'all', what the delivery uses: "
                         "`pool_rerank_from_heat_locked` ranks by the combined prefill+decode heat because the "
                         "pool stages whole tables for prefill).  'decode' ranks by the decode-only counter "
                         "(the one the ARENA seeds from, TODO #88).  Measured 2026-11-05 on the Flash-Next "
                         "prose trace: the combined counter is WORSE THAN CHANCE for decode routing (top-64 "
                         "overlap 10.5 %% against 12.5 %% chance) while the decode counter is 2x chance "
                         "(24.2 %%), which is why the pool answers only 56 %% of arena misses.")
    ap.add_argument("--pool-init", default="heat",
                    help="initial host-pool content: 'heat' = ranked by this trace's routing (an ORACLE "
                         "prior, the best any prior can do), 'id' = expert order (a tally-less prewarm, "
                         "what the delivery does today without a heat map), 'none' = empty (the "
                         "prefill/staging fills it), or 'sidecar:<path>' = rank by a REAL persisted "
                         "`.moe-heatmap` sidecar (the achievable prior; MHT1 format).  A '-complement' "
                         "suffix first heat-seeds the ARENA with the same prior and then gives the pool "
                         "only the non-resident experts (the real layout, and the one that makes "
                         "arena U pool cover the set)")
    ap.add_argument("--score-band", choices=("all", "decode"), default="decode",
                    help="which passes the arena policies are scored on (default 'decode', which is what the "
                         "runtime's logged h= counts: the prefill band returns before the cache's access path, "
                         "so its positions never reach hits/misses.  'all' includes them and biases `shipped` "
                         "low by ~0.15 on the 122B trace -- it is a diagnostic, never a calibration)")
    ap.add_argument("--decode-only", action="store_true", default=True,
                    help="score the decode band only (the prefill stages whole tables, so it does not "
                         "discriminate between pool policies); default on")
    ap.add_argument("--all-bands", dest="decode_only", action="store_false")
    ap.add_argument("--measured-h", type=float, default=None,
                    help="the run's logged `MoE expert cache: h=` -- printed beside the model's own "
                         "`shipped` h so the model's error is visible (it is a model, not the cache)")
    ap.add_argument("--json", metavar="PATH", help="write the scored table as JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    tables, recs = parse(args.trace, args.mass_band)
    if not tables or not recs:
        sys.exit("empty trace (is it a fully-resident run? see the caveat in --help)")
    if args.devices > 1:
        for t in tables.values():
            t.expert_bytes = t.expert_bytes // args.devices

    # Drop any table that never routed: it cannot affect a hit rate, and the sizing in the log only
    # counted tables that registered (all of them), so keep it for the byte accounting either way.
    passes = segment_passes(recs)
    bands = collections.Counter(r.band for r in recs)
    tokens = sum(r.n_tok for p in passes for r in p if r is p[0])

    one = sum(t.expert_bytes for t in tables.values())
    if args.slots is not None:
        uni = args.slots
    elif args.arena_mib is not None:
        uni = uniform_slots(tables, args.arena_mib * 1048576.0)
    else:
        args.arena_mib = 4096.0
        uni = uniform_slots(tables, args.arena_mib * 1048576.0)
    arena_bytes = args.arena_mib * 1048576.0 if args.arena_mib is not None else uni * one
    for t in tables.values():
        t.slots = min(uni, t.n_experts)

    if not args.quiet:
        print("trace      : %s" % args.trace)
        print("records    : %d (%s), %d passes, ~%d tokens"
              % (len(recs), ", ".join("%s=%d" % kv for kv in sorted(bands.items())), len(passes), tokens))
        print("tables     : %d (one device's set), %d experts, %.1f MiB per slot-set (one slot per table)"
              % (len(tables), sum(t.n_experts for t in tables.values()), one / 1048576.0))
        print("arena      : %.0f MiB budget -> uniform %d slots/table (%.1f%% of the expert set)"
              % (arena_bytes / 1048576.0, uni,
                 100.0 * sum(min(uni, t.n_experts) for t in tables.values()) /
                 max(1, sum(t.n_experts for t in tables.values()))))
        print()

    sidecar = None
    if args.pool_init.startswith("sidecar"):
        sc_path = args.pool_init.split(":", 1)[1]
        sidecar = load_heat_sidecar(sc_path, band=args.pool_band)
        args.pool_init = args.pool_init.split(":", 1)[0]
        matched = sum(1 for k, t in tables.items()
                      if len((sidecar or {}).get(k) or []) == t.n_experts)
        if not sidecar:
            sys.exit("could not read a heat sidecar at %s" % sc_path)
        if matched == 0:
            sys.exit("the heat sidecar at %s matches NONE of this trace's %d tables (wrong model?)"
                     % (sc_path, len(tables)))
        if not args.quiet:
            print("heat prior : %d tables, %d/%d matched to the trace" % (len(sidecar), matched, len(tables)))

    # The arena seed (TODO #86 step 0).  `sidecar:<path>` is the real persisted prior; it is parsed here so
    # the same path can also drive --pool-init sidecar:<path>.
    seed = args.seed
    probe_passes = 0
    seed_sidecar = None
    if seed.startswith("sidecar"):
        p = seed.split(":", 1)[1]
        seed_sidecar = load_heat_sidecar(p, band="decode")
        if not seed_sidecar:
            sys.exit("could not read a heat sidecar at %s" % p)
        seed = "sidecar"
        matched = sum(1 for k, t in tables.items() if len(seed_sidecar.get(k) or []) == t.n_experts)
        if matched == 0:
            sys.exit("the heat sidecar at %s matches NONE of this trace's %d tables (wrong model?)"
                     % (p, len(tables)))
        if not args.quiet:
            print("arena seed : sidecar %s (%d tables, %d/%d matched to the trace)"
                  % (p, len(seed_sidecar), matched, len(tables)))
    elif seed.startswith("probe"):
        probe_passes = int(seed.split(":", 1)[1]) if ":" in seed else 16
        seed = "probe"
        if not args.quiet:
            print("arena seed : probe = the trace's first %d passes" % probe_passes)
    elif seed not in ("none", "heat"):
        sys.exit("unknown --seed %r" % args.seed)
    prior = seed_prior(tables, passes, seed, sidecar=seed_sidecar, probe_passes=probe_passes,
                       mass_band=args.mass_band)
    if not args.quiet and prior is not None:
        mass = sum(sum(v) for v in prior.values())
        in_slots = sum(sum(sorted(prior.get(k) or [], reverse=True)[:min(t.slots, t.n_experts)])
                       for k, t in tables.items())
        print("arena seed : %s, %.1f%% of the prior's mass lands in the seeded slots"
              % (args.seed, 100.0 * in_slots / mass if mass else 0.0))

    results = {}
    pols = [p.strip() for p in args.policies.split(",") if p.strip()]
    pool_pols = [p for p in pols if p.startswith("pool-")]
    if pool_pols and args.pool_frac is None:
        sys.exit("the pool-* policies need --pool-frac (they model --host-experts pool)")
    if pool_pols:
        slots_p = {k: min(int(t.n_experts * args.pool_frac / 100), t.n_experts) for k, t in tables.items()}
        for pol in pool_pols:
            r = score_pool(tables, passes, uni, slots_p, pol, pool_init=args.pool_init,
                           period=args.period, admit=args.admit, touch=args.touch, rebalance=args.rebalance,
                           decode_only=args.decode_only, sidecar=sidecar,
                           rebalance_heat=args.rebalance_heat, rebalance_prior=prior,
                           rebalance_k=args.rebalance_k, evict_warm=args.evict_warm)
            results[pol] = r
        pols = [p for p in pols if not p.startswith("pool-")]
    for pol in pols:
        if pol == "quota":
            mass_by_layer = collections.Counter()
            for t in tables.values():
                mass_by_layer[t.layer] += sum(t.freq.values())
            sq = quota_slots(tables, arena_bytes, mass_by_layer, args.min_slots)
            r = score(tables, passes, slots_by_layer=sq, arena_bytes=arena_bytes,
                      period=args.period, admit=args.admit, touch=args.touch, policy="quota",
                      prior=prior, churn=args.churn == "on", score_band=args.score_band)
            r["layer_slots"] = {str(k): v for k, v in sorted(sq.items())}
        else:
            r = score(tables, passes, slots_by_layer=None, arena_bytes=arena_bytes,
                      period=args.period, admit=args.admit, touch=args.touch, policy=pol,
                      prior=prior, churn=args.churn == "on", score_band=args.score_band)
        results[pol] = r

    bound = static_bound(tables, arena_bytes)
    if pool_pols and not args.quiet:
        hdr = "%-17s %8s %8s %8s %11s %9s %9s" % ("pool policy", "h", "pool_h", "L3/MiB", "L3 MiB/tok",
                                                   "copy/tok", "link/tok")
        print(hdr)
        print("-" * len(hdr))
        for pol in pool_pols:
            r = results[pol]
            print("%-17s %8.4f %8.4f %8d %11.3f %9.2f %9.2f"
                  % (pol, r["h"], r["pool_h"], r["l3"], r["l3_mib_per_tok"],
                     r["handover_mib_per_tok"], r["hostlink_mib_per_tok"]))
        print("(L3 = experts in NEITHER tier: the disk reads a pool policy exists to remove; "
              "handover = the extra device copies a policy pays to keep the pool useful)")
        print()
    if not args.quiet:
        hdr = "%-8s %8s %9s %7s %9s %11s %8s" % ("policy", "h", "misses", "cold", "MiB/tok",
                                                 "resident MiB", "covered%")
        print(hdr)
        print("-" * len(hdr))
        for pol in pols:
            r = results[pol]
            print("%-8s %8.4f %9d %7d %9.2f %11.1f %8.2f"
                  % (pol, r["h"], r["misses"], r["cold"], r["mib_per_token"], r["resident_mib"],
                     r["covered_mass_pct"]))
        print()
        print("static bound (global top-f, same slot count): %.2f%% of the routed mass" % bound)
        print("arena seed: %s, churn: %s" % (args.seed, args.churn))
        sh = results.get("shipped")
        if sh and args.measured_h is not None:
            print("calibration: model 'shipped' h=%.4f vs the run's logged h=%.4f (error %+.4f)"
                  % (sh["h"], args.measured_h, sh["h"] - args.measured_h))
            print("             (pass --seed sidecar:<the run's .moe-heatmap> for the real prior, and --slots <the")
            print("              run's per-table count> when the log has it -- with both, the decode band lands within")
            print("              0.3 %.  See wip/host-pool-speed/HANDOVER.md §4/F2)")
        if sh and "quota" in results:
            q = results["quota"]
            print("quota vs shipped: h %+.4f, MiB/token %+.1f%% (%s)"
                  % (q["h"] - sh["h"],
                     100.0 * (q["mib_per_token"] / sh["mib_per_token"] - 1.0) if sh["mib_per_token"] else 0.0,
                     "quota WINNER" if q["h"] > sh["h"] else "shipped wins/ties"))
            print("layer slots (quota): %s" % ", ".join("%s:%d" % kv for kv in
                                                        sorted(results["quota"]["layer_slots"].items(),
                                                               key=lambda kv: int(kv[0]))))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"trace": args.trace, "arena_mib": arena_bytes / 1048576.0,
                       "uniform_slots": uni, "passes": len(passes), "tokens": tokens,
                       "seed": args.seed, "churn": args.churn,
                       "static_bound_pct": bound, "results": results}, f, indent=2)
        if not args.quiet:
            print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
