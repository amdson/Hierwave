"""Fraction of draws that change when the coarse levels (>= L) are clamped to a
fill-following probe (jitter 0) instead of the model's own draw, same noise.
8 seeds; L = 4 and 3; trained and untrained params."""
import json, pickle, sys, jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.variants import _cpass_core as core
from clamp_tools import clamped_pass_stats
p = get_problem("ground"); reg = Region(p.W_train); lay = core.Layout(p, reg.W, True, 0, resid=False)
D = sys.argv[1]
ck = pickle.load(open(D + "/params_ground.pkl", "rb")); NS = 16; seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]; prs = [p.probes(int(s), reg)[0] for s in seeds]
f = jax.jit(jax.vmap(lambda pr, lm, s, d, c: clamped_pass_stats(lay, pr, s, d, 1.0, lm, c), in_axes=(None, None, 0, 0, 0)))
dj = {k: jnp.asarray(np.stack([d[k] for d in des]), jnp.int32) for k in des[0]}
out = {}
for m in ("untrained", "trained"):
    params = jax.tree_util.tree_map(jnp.asarray, ck[m]["params"]); lam = jnp.asarray(ck[m]["lam"], jnp.float32)
    own, _ = f(params, lam, jnp.asarray(seeds, jnp.int32), dj, {})
    for L in (4, 3):
        cl = {c.name: jnp.asarray(np.stack([pr[c.name] for pr in prs]), jnp.int32) for c in p.sampled() if c.level >= L}
        cla, _ = f(params, lam, jnp.asarray(seeds, jnp.int32), dj, cl)
        for c in p.sampled():
            if c.level < L:
                out[f"{m}/L{L}/{c.name}"] = float((np.asarray(own[c.name]) != np.asarray(cla[c.name])).mean())
        # fraction of fine draws (all levels < L) that change
        tot = sum((np.asarray(own[c.name]) != np.asarray(cla[c.name])).sum() for c in p.sampled() if c.level < L)
        n = sum(np.asarray(own[c.name]).size for c in p.sampled() if c.level < L)
        out[f"{m}/L{L}/all_fine"] = float(tot / n)
        # and the tiles' agreement with the probe's tiles
        out[f"{m}/L{L}/t_eq_probe"] = float((np.asarray(cla["t"]) == np.stack([pr["t"] for pr in prs])).mean())
json.dump(out, open(D + "/same_check.json", "w"), indent=1)
for k, v in out.items():
    print(k, round(v, 3))
