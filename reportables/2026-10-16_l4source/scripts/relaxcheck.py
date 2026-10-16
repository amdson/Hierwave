"""v2 block check with exact-lift vs coarse-relaxed probes (tiles fixed, 10 hard Gibbs sweeps of levels 4..1,
key fixed per probe seed).  usage: relaxcheck.py <out.json> <label>=<variant>:<params.pkl> ...
Report-only: the bench is unchanged; relaxation is applied by passing a copy of the problem to run.block_check."""
import json, pickle, sys, dataclasses
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import block_check, _with_designed
from castlegen.bench.problems import ground as G
from castlegen.kernels import colour_gibbs as cg
p = get("ground"); V = load_variants(); reg = Region(p.W_eval)
keys = [c.name for c in p.chans]
v0 = V["cfitpen_ml2"](p, 0, "/tmp/l4rc"); v0._build(reg); lay = v0.lay_tr
cache = {}
def relaxed(seed, region):
    if seed not in cache:
        ys = [_with_designed(p, y, seed, region) for y in p.probes(seed, region)]
        st = {k: jnp.asarray(np.stack([np.asarray(y[k]) for y in ys]), jnp.int32) for k in keys}
        key = jax.random.PRNGKey(seed % (2**31))
        for _ in range(10):
            key, sub = jax.random.split(key); st = cg.sweep_all(lay, st, 1.0, None, sub, levels=[4, 3, 2, 1])
        S = jax.device_get(st)
        out = []
        for b in range(len(ys)):
            x = {k: np.asarray(S[k][b]) for k in keys}
            assert (x["t"] == ys[b]["t"]).all()
            out.append({k: x[k] for k in keys if p.chan[k].designed is None})
        cache[seed] = out
    return [dict(x) for x in cache[seed]]
pr = dataclasses.replace(p, probes=relaxed)
res = {}
for a in sys.argv[2:]:
    lab, rest = a.split("="); vn, path = rest.split(":")
    ck = pickle.load(open(path, "rb")); ck = ck.get("trained", ck)
    v = V[vn](p, 0, "/tmp/l4rc_" + lab)
    v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)
    for kind, prob in (("exact", p), ("relaxed", pr)):
        b = block_check(prob, v, 0, reg)["levels"]
        res[f"{lab}/{kind}"] = {k: [b[str(l)]["direct"][k] for l in range(5)] for k in ("rho_pool", "starved_probe", "starved_own")}
        print(lab, kind, json.dumps({k: [None if x is None else round(x, 3) for x in vv] for k, vv in res[f"{lab}/{kind}"].items()}), flush=True)
# relaxed-probe statistics vs the chain: tiles (unchanged by construction) and the coarse rho ladder in pure air
lad, lm, rough = {l: [] for l in range(1, 5)}, [], []
for seed, ys in cache.items():
    fl = np.asarray(p.designed_values(seed, reg)["fill"])
    for x in ys:
        st = p.stats({**x, "fill": fl}); lm.append(st["line_match"]); rough.append(st["roughness"])
        for l in range(1, 5):
            s = 1 << (4 - l); air = np.repeat(np.repeat(fl == 0, s, 0), s, 1)
            lad[l].append(np.asarray(x[f"rho{l}"], float)[air])
res["relaxed_probe_stats"] = {"line_match": float(np.mean(lm)), "roughness": float(np.mean(rough)),
                              "air_rho_L1_L4": [float(np.concatenate(lad[l]).mean()) for l in range(1, 5)]}
print(json.dumps(res["relaxed_probe_stats"]))
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("DONE", flush=True)
