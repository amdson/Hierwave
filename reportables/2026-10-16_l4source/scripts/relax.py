"""l4source relax: bench probes with fine tiles fixed and coarse levels resampled exactly (p*(coarse | t)); scored like the block check.  l4src header follows.\nl4source: does the population's L4 drift to the probe-like states with more kernel sweeps (mixing),
or stay put (stationary)?  Params frozen (cfitpen_ml2, 300 s).

usage: l4src.py <variant> <params.pkl> <outdir> [iters_max]
Arms (all on GPU, one process):
  ref    : 64 own draws (bench seeds) -> the own-draw Theil-Sen line per level (as block_check);
           128 probe worlds (bench probe seeds) -> probe numbers.
  l4only : fine levels fixed; exact colour Gibbs sweeps of level 4 only (hard, lambda = inf) on
           32 fresh own draws and 32 probe worlds; 1, 10, 100 sweeps.
  chain  : the population's refresh cycle (refine phase 2 + cg_sweeps soft + hard_sweeps, exactly the
           CKernelPop loop body), no replacement, on the same 32 own + 32 probe worlds (one batch of 64);
           snapshots at 0, 32 (= 1x a member's lifetime at refresh 1/32), 100, 320 (10x), 1000, 3200 (100x).
Each snapshot: per-level rows (logq share, direct, subtree weight, set, world, slack, dirt) + L4 extras
(logq of rho4 / cert4 per block, noise4 and smooth4 energy per block, lift-exactness per block).
"""
import sys, time, json, pickle, os
import numpy as np, jax, jax.numpy as jnp
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region
from castlegen.bench import shares, energy as en
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.kernels import colour_gibbs as cg

name, path, outd = sys.argv[1:4]
ITMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 3200
SNAPS = [s for s in (0, 32, 100, 320, 1000, 3200) if s <= ITMAX]
os.makedirs(outd, exist_ok=True)
p = get("ground"); T = p.T; reg = Region(128)
V = load_variants(); v = V[name](p, 0, "/tmp/l4src_" + name)
ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
nbs = {l: shares._nblk(p, reg, l) for l in range(T + 1)}
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}]", *a, flush=True)

def blockmean(a, l):
    nb = nbs[l]; pad = np.full((2 * nb[0], 2 * nb[1]), np.nan); pad[:a.shape[0], :a.shape[1]] = a
    return np.nanmean(pad.reshape(nb[0], 2, nb[1], 2), axis=(1, 3)).ravel()

def blocksum(a, l):
    nb = nbs[l]; pad = np.zeros((2 * nb[0], 2 * nb[1])); pad[:a.shape[0], :a.shape[1]] = a
    return pad.reshape(nb[0], 2, nb[1], 2).sum(axis=(1, 3)).ravel()

def rows(x, setid, wid):
    sites = v.score_sites(x, reg)
    lq = shares.logq_shares(p, sites, reg); D, S = shares.weight_shares_signed(p, x, reg)
    R = {}
    for l in (2, 3, 4):
        h = 1 << l
        ch = p.chan[f"cert{l}"]; c = np.asarray(x[f"cert{l}"])
        slack = blockmean(np.asarray(ch.views["B"])[c] - np.asarray(ch.views["A"])[c], l)
        rho = blockmean(np.asarray(x[f"rho{l}"], float) / h ** 2, l)
        n = lq[l].size
        cols = [lq[l].ravel(), D[l].ravel(), S[l].ravel(), np.full(n, setid), np.full(n, wid), slack, rho]
        if l == 4:
            r4 = np.asarray(x["rho4"], float); f = np.asarray(x["fill"], float) / 16
            noise = G.BETA * (r4 - 256 * f) ** 2 / 256
            sm = G.LAM * np.abs(r4 - np.roll(r4, -1, axis=1)) / 16      # pair (p, right neighbour), whole to p
            L = G.lift(np.asarray(x["t"]))
            A4, B4 = np.asarray(ch.views["A"]), np.asarray(ch.views["B"])
            cx, cl = np.asarray(x["cert4"]), np.asarray(L["cert4"])
            cols += [blocksum(np.asarray(sites["rho4"], float), 4), blocksum(np.asarray(sites["cert4"], float), 4),
                     blocksum(noise, 4), blocksum(sm, 4),
                     blockmean(np.abs(r4 - L["rho4"]), 4),                       # |rho4 - exact count|
                     blockmean((cx == cl).astype(float), 4),                     # cert4 tight (lift) fraction
                     blockmean((A4[cl] - A4[cx]).astype(float), 4),              # A below tight
                     blockmean((B4[cx] - B4[cl]).astype(float), 4)]              # B above tight
        R[l] = np.stack(cols, 1)
    return R

def score_batch(st, setids, tag, extra=None):
    P = jax.device_get(st); n = P["t"].shape[0]
    store = {l: [] for l in (2, 3, 4)}; nv = []; lm = []
    for b in range(n):
        x = {k: np.asarray(a[b]) for k, a in P.items()}
        R = rows(x, setids[b], b)
        for l in R: store[l].append(R[l])
        E = en.energy(p, x, reg); nv.append(sum(E.viol.values()))
        lm.append(p.stats(x)["line_match"])
    out = {f"L{l}": np.concatenate(store[l]) for l in store}
    out["nviol"] = np.asarray(nv); out["line_match"] = np.asarray(lm)
    np.savez(os.path.join(outd, f"{tag}.npz"), **out)
    log("scored", tag, n)
    return out

# ---------------- probes with tiles fixed, coarse channels (levels 4..1) resampled from p*(coarse | t)
prb = []
for i in range(32):
    seed = EVAL_SEED0 + 600 + i
    for y in p.probes(seed, reg):
        prb.append(_with_designed(p, y, seed, reg))
keys = [c.name for c in p.chans]
stk = lambda L: {k: jnp.asarray(np.stack([np.asarray(x[k]) for x in L]), jnp.int32) for k in keys}
v._build(reg)
st = stk(prb); n = len(prb)
key = jax.random.PRNGKey(99); done = 0
for k in (1, 10, 100):
    while done < k:
        key, sub = jax.random.split(key); st = cg.sweep_all(v.lay_tr, st, 1.0, None, sub, levels=[4, 3, 2, 1]); done += 1
    score_batch(st, [1] * n, f"relax_{k}")
print("DONE", time.time() - t0, flush=True)
