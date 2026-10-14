import pickle, jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.variants import _cpass_core as core
from clamp_tools import clamped_pass_stats
p = get_problem("ground"); reg = Region(p.W_train); lay = core.Layout(p, reg.W, True, 0, resid=False)
ck = pickle.load(open(__import__("sys").argv[1] + "/params_ground.pkl", "rb")); NS = 8; seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]; prs = [p.probes(int(s), reg)[0] for s in seeds]
f = jax.jit(jax.vmap(lambda pr, lm, s, d, c: clamped_pass_stats(lay, pr, s, d, 1.0, lm, c), in_axes=(None, None, 0, 0, 0)))
dj = {k: jnp.asarray(np.stack([d[k] for d in des]), jnp.int32) for k in des[0]}
params = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"]); lam = jnp.asarray(ck["trained"]["lam"], jnp.float32)
own, _ = f(params, lam, jnp.asarray(seeds, jnp.int32), dj, {})
cl = {c.name: jnp.asarray(np.stack([pr[c.name] for pr in prs]), jnp.int32) for c in p.sampled() if c.level >= 4}
cla, _ = f(params, lam, jnp.asarray(seeds, jnp.int32), dj, cl)
for c in p.sampled():
    a, b = np.asarray(own[c.name]), np.asarray(cla[c.name])
    print(c.name, "equal frac", (a == b).mean().round(3), "own mean", a.mean().round(2), "clamped mean", b.mean().round(2))
pr = np.stack([q["rho3"] for q in prs]); print("probe rho3 mean", pr.mean(), "probe rho4", np.stack([q["rho4"] for q in prs]).mean())
r3 = np.asarray(own["rho3"]); print("rho3 row means", r3.mean((0, 2)).round(1), "std across worlds+cols per row", r3.std((0, 2)).round(1))
r1 = np.asarray(own["rho1"]); print("rho1 row means", r1.mean((0, 2)).round(2))
un = jax.tree_util.tree_map(jnp.asarray, ck["untrained"]["params"])
o2, _ = f(un, lam, jnp.asarray(seeds, jnp.int32), dj, {}); c2, _ = f(un, lam, jnp.asarray(seeds, jnp.int32), dj, cl)
for c in ("rho3", "rho2", "rho1"): print("untrained", c, "equal frac own vs clamped", (np.asarray(o2[c]) == np.asarray(c2[c])).mean().round(3))
