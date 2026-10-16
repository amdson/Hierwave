"""relax4 part 2: L3 block classes (starve's measure) under relaxed probes.  usage: classes.py <out.json> <label>=<variant>:<params.pkl> ...
Own draws (block_check's seeds) fit the Theil-Sen line (direct weight vs log q share, as starve); relaxed probes (tiles fixed, 10 hard Gibbs sweeps of levels 4..1, as relaxcheck) are scored against it.
Classes at L3: slack0 (block mean B-A == 0), hi_cross (slack > 1.5 and 0.05<dirt<0.95), hi_all (slack>1.5), plus all.  starved = deficit > 5 nats."""
import json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench import shares
from castlegen.bench.run import EVAL_SEED0, _with_designed, N_POOL_OWN, N_POOL_PROBE_SEEDS
from castlegen.kernels import colour_gibbs as cg
p = get("ground"); T = p.T; reg = Region(128); V = load_variants(); keys = [c.name for c in p.chans]
nbs = {l: shares._nblk(p, reg, l) for l in range(T + 1)}
def blockmean(a, l):
    nb = nbs[l]; pad = np.full((2 * nb[0], 2 * nb[1]), np.nan); pad[:a.shape[0], :a.shape[1]] = a
    return np.nanmean(pad.reshape(nb[0], 2, nb[1], 2), axis=(1, 3)).ravel()
def row3(v, x):
    lq = shares.logq_shares(p, v.score_sites(x, reg), reg); D, S = shares.weight_shares_signed(p, x, reg)
    ch = p.chan["cert3"]; c = np.asarray(x["cert3"])
    slack = blockmean(np.asarray(ch.views["B"])[c] - np.asarray(ch.views["A"])[c], 3)
    dirt = blockmean(np.asarray(x["rho3"], float) / 64, 3)
    return np.stack([lq[3].ravel(), D[3].ravel(), slack, dirt], 1)
v0 = V["cfitpen_ml2"](p, 0, "/tmp/r4c"); v0._build(reg); lay = v0.lay_tr
probes = []
for i in range(N_POOL_PROBE_SEEDS):
    seed = EVAL_SEED0 + 600 + i
    ys = [_with_designed(p, y, seed, reg) for y in p.probes(seed, reg)]
    st = {k: jnp.asarray(np.stack([np.asarray(y[k]) for y in ys]), jnp.int32) for k in keys}
    key = jax.random.PRNGKey(seed % (2**31))
    for _ in range(10):
        key, sub = jax.random.split(key); st = cg.sweep_all(lay, st, 1.0, None, sub, levels=[4, 3, 2, 1])
    S = jax.device_get(st)
    for b in range(len(ys)):
        x = {k: np.asarray(S[k][b]) for k in keys}; assert (x["t"] == ys[b]["t"]).all()
        probes.append(_with_designed(p, {k: x[k] for k in keys if p.chan[k].designed is None}, seed, reg))
print("probes", len(probes), flush=True)
res = {}
for a in sys.argv[2:]:
    lab, rest = a.split("="); vn, path = rest.split(":")
    ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
    v = V[vn](p, 0, "/tmp/r4c_" + lab)
    v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
    own = []
    for i in range(N_POOL_OWN):
        seed = EVAL_SEED0 + 200 + i
        own.append(row3(v, _with_designed(p, v.generate(seed, reg)["raw"], seed, reg)))
    own = np.concatenate(own); pr = np.concatenate([row3(v, y) for y in probes])
    s, ic = shares._robust_line(own[:, 1], own[:, 0], np.random.default_rng(0))
    out = {}
    for setn, a_ in (("own", own), ("probe", pr)):
        lq, dw, sl, di = a_.T; d = (s * dw + ic) - lq
        for cn, m in (("all", np.ones(len(lq), bool)), ("slack0", sl == 0), ("hi_cross", (sl > 1.5) & (di > .05) & (di < .95)), ("hi_all", sl > 1.5)):
            if m.sum():
                out[f"{setn}/{cn}"] = dict(n=int(m.sum()), frac_blocks=float(m.mean()), starved=float((d[m] > 5).mean()), med_deficit=float(np.median(d[m])),
                    mean_lq=float(lq[m].mean()), share_lq=float(lq[m].sum() / lq.sum()), share_starved=float(((d > 5) & m).sum() / max(1, (d > 5).sum())))
    res[lab] = out
    for k, r in out.items(): print(lab, k, json.dumps({q: round(z, 3) for q, z in r.items()}), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1); print("DONE", flush=True)
