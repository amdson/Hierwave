"""Guard (2): fine cost per cell (sum of the draw costs soft + lam nv + log q of levels < L, tau = 1, per fine cell)
with levels >= L clamped to a probe lift (fill-following, problem.probes jitter 0-3) vs the model's own coarse draw,
same noise seeds; L = 4, 3 (reportables/2026-10-14_cstart/gate.py numbers; cfast reference 1.97 vs 0.72 at L=4,
2.23 vs 0.34 at L=3).  Uses crecur's split_pass (one seed for all levels; clamp at levels >= L).
usage: clamp_gate.py <out.json> <label=params.pkl> ...   (NS = 32 seeds x 4 probes)"""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.variants._crecur_core import split_pass
p = get("ground"); T = p.T; reg = Region(p.W_train); V = load_variants()
NS = 32
seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]
probes = [p.probes(int(s), reg) for s in seeds]
NP = len(probes[0])
cells = p.H * reg.W
v = V["crecur"](p, 0, "/tmp/recur_gate", rec_n=0)
lay = v.layout(reg)
out = {}
for arg in sys.argv[2:]:
    label, path = arg.split("=")
    ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); lam = jnp.asarray(ck["lam"], jnp.float32)
    res = {}
    for L in (4, 3):
        names = [c.name for c in lay.p.sampled() if c.level >= L]
        f = jax.jit(jax.vmap(lambda s, d, c, fl: split_pass(lay, params, s, s, d, 1.0, lam, 0, record=False,
                                                              clamp=c, cflag=fl, Lc=L)))
        for cond in ("own", "probe"):
            sd = np.repeat(seeds, NP); dl = [d for d in des for _ in range(NP)]
            prs = [pr for ps in probes for pr in ps]
            tot = []
            for i in range(0, len(sd), 32):
                d = {k: jnp.asarray(np.stack([x[k] for x in dl[i:i + 32]]), jnp.int32) for k in dl[0]}
                c = {n: jnp.asarray(np.stack([x[n] for x in prs[i:i + 32]]), jnp.int32) for n in names}
                fl = jnp.full(len(sd[i:i + 32]), cond == "probe")
                st, costs = f(jnp.asarray(sd[i:i + 32], jnp.int32), d, c, fl)
                tot.append(np.stack([sum(np.asarray(costs[k]).sum((1, 2)) for k in costs if int(k[1:k.index("_")]) == l)
                                     for l in range(L)], 1))
            tot = np.concatenate(tot) / cells                             # (N, L) per level per fine cell
            res[f"L{L}_{cond}"] = {"fine_cost_per_cell": float(tot.sum(1).mean()),
                                   "se": float(tot.sum(1).std(ddof=1) / np.sqrt(len(tot))),
                                   "per_level": tot.mean(0).round(4).tolist()}
    out[label] = res
    print(label, {k: round(r["fine_cost_per_cell"], 3) for k, r in res.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE")
