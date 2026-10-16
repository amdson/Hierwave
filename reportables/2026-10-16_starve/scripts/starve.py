"""starve: probes / population / own draws on the same block axes.  usage: starve.py <variant> <params.pkl> <out.npz> [iters]"""
import sys, time, json, pickle
import numpy as np, jax, jax.numpy as jnp
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench import shares
from castlegen.bench.run import EVAL_SEED0, _with_designed, N_POOL_OWN, N_POOL_PROBE_SEEDS
name, path, out = sys.argv[1:4]; iters = int(sys.argv[4]) if len(sys.argv) > 4 else 320
p = get("ground"); T = p.T; reg = Region(128)
V = load_variants(); v = V[name](p, 0, "/tmp/starve_" + name)
ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
nbs = {l: shares._nblk(p, reg, l) for l in range(T + 1)}

def blockmean(a, l):
    nb = nbs[l]; pad = np.full((2 * nb[0], 2 * nb[1]), np.nan); pad[:a.shape[0], :a.shape[1]] = a
    return np.nanmean(pad.reshape(nb[0], 2, nb[1], 2), axis=(1, 3)).ravel()

def rows(x, sites, setid, wid, extra):
    lq = shares.logq_shares(p, sites, reg); D, S = shares.weight_shares_signed(p, x, reg)
    R = {}
    for l in range(T + 1):
        h = 1 << l
        if l == 0: slack = np.zeros(nbs[0]).ravel(); rho = blockmean(np.asarray(x["t"], float), 0)
        else:
            ch = p.chan[f"cert{l}"]; c = np.asarray(x[f"cert{l}"])
            slack = blockmean(np.asarray(ch.views["B"])[c] - np.asarray(ch.views["A"])[c], l)
            rho = blockmean(np.asarray(x[f"rho{l}"], float) / h ** 2, l)
        n = lq[l].size
        R[l] = np.stack([lq[l].ravel(), D[l].ravel(), S[l].ravel(), np.full(n, setid), np.full(n, wid), slack, rho, np.full(n, extra)], 1)
    return R
store = {l: [] for l in range(T + 1)}
def add(R):
    for l in R: store[l].append(R[l])
t0 = time.time()
for i in range(N_POOL_OWN):
    seed = EVAL_SEED0 + 200 + i
    x = _with_designed(p, v.generate(seed, reg)["raw"], seed, reg)
    add(rows(x, v.score_sites(x, reg), 0, i, 0))
npr = 0
for i in range(N_POOL_PROBE_SEEDS):
    seed = EVAL_SEED0 + 600 + i
    for y in p.probes(seed, reg):
        y = _with_designed(p, y, seed, reg)
        add(rows(y, v.score_sites(y, reg), 1, npr, 0)); npr += 1
print("own+probe done", time.time() - t0, npr, flush=True)
# population: replicate CKernelPop.train's loop with the ML step removed (params fixed)
c = v.cfg; n = c["nctx"]; nref = max(1, int(round(c["refresh"] * n)))
from castlegen.kernels import colour_gibbs as cg
v._build(reg); v.ctx_seed = 20_000_000
key = jax.random.PRNGKey(4242)
pop = v._own_ctx(n, reg); age = np.zeros(n, int)
w = 0
for it in range(1, iters + 1):
    fresh = v._own_ctx(nref, reg); old = np.argsort(-age)[:nref]; oj = jnp.asarray(old)
    pop = {k: x.at[oj].set(fresh[k]) for k, x in pop.items()}; age[old] = 0
    key, sub = jax.random.split(key)
    for _ in range(c["n_cycles"]):
        sub, s2 = jax.random.split(sub); pop, kd, vv = v.refine(pop, s2, 2)
    for _ in range(c["cg_sweeps"]):
        sub, s3 = jax.random.split(sub); pop = cg.sweep_all(v.lay_tr, pop, 1.0, c["kernel_lam"], s3)
    for _ in range(c["hard_sweeps"]):
        sub, s4 = jax.random.split(sub); pop = cg.sweep_all(v.lay_tr, pop, 1.0, None, s4)
    age += 1
    if it >= 48 and it % 16 == 0:
        P = jax.device_get(pop)
        for b in range(n):
            x = {k: np.asarray(a[b]) for k, a in P.items()}
            add(rows(x, v.score_sites(x, reg), 2, w, age[b])); w += 1
        print("snap", it, time.time() - t0, flush=True)
np.savez(out, **{f"L{l}": np.concatenate(store[l]) for l in range(T + 1)})
print("DONE", time.time() - t0)
