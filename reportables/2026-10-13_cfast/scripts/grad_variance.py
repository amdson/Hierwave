"""Variance of the single-world actor gradient with cone returns vs all-later
returns (cpass), same worlds, same parameters (untrained or briefly trained),
critic off (Q = 0 at init) so the returns alone carry the score term.

Per level: mean gradient of each estimator (they must agree up to noise:
a statistical check of the cone's unbiasedness) and the total variance
E||g - mean||^2.  usage: grad_variance.py problem N [train_updates]"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cfast import CFast

prob = problems.get(sys.argv[1])
N = int(sys.argv[2])
pre = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out = {}
vc = CFast(prob, 0, "/tmp", B=1)
vf = CFast(prob, 0, "/tmp", B=1, cone=False)
region = Region(prob.W_train)
lay = vc.layout(region)
fc, ff = vc.train_fns(lay), vf.train_fns(lay)
params, cparams = vc.params, vc.cparams
gs = {k: jnp.float32(1.0) for k in params}
lam = jnp.asarray(vc.lam, jnp.float32)
tau = jnp.float32(1.0)
if pre:                                      # brief training (cone, fused) to move off the init
    vc.fn = fc
    vc.cfg["B"] = 8
    vc._measure_baseline(lay, lambda d: None)
    for u in range(pre):
        seeds = jnp.arange(8, dtype=jnp.int32) + 10_000 + 8 * u
        des = vc._batch(np.asarray(seeds), region)
        o = fc["fused"](vc.params, vc.cparams, vc.opt, vc.copt, vc.gs, seeds, des, tau, lam, 1e-3, 1e-3, 10.0,
                        jax.random.PRNGKey(u))
        vc.params, vc.cparams, vc.opt, vc.copt = o[1:5]
    params = vc.params
    cparams = jax.tree_util.tree_map(jnp.zeros_like, vc.cparams)  # critic off: returns only
levels = sorted({int(k[1:k.index("_")]) for k in params})


def flat_by_level(g):
    return {l: np.concatenate([np.ravel(np.asarray(x)) for k in sorted(g) if k.startswith(f"l{l}_")
                               for x in jax.tree_util.tree_leaves(g[k])]) for l in levels}


gC, gF = {l: [] for l in levels}, {l: [] for l in levels}
t = time.time()
for i in range(N):
    seeds = jnp.asarray([2_000 + i], jnp.int32)
    des = vc._batch(np.asarray(seeds), region)
    st, recs, GC, *_ = fc["rec"](params, seeds, des, tau, lam)
    GF = ff["rec"](params, seeds, des, tau, lam)[2]
    a = flat_by_level(fc["actor_grad"](params, cparams, recs, GC, tau, lam, gs))
    b = flat_by_level(fc["actor_grad"](params, cparams, recs, GF, tau, lam, gs))
    for l in levels:
        gC[l].append(a[l]); gF[l].append(b[l])
print(f"{N} worlds in {time.time() - t:.1f}s")
for l in levels:
    C, F = np.stack(gC[l]), np.stack(gF[l])
    mc, mf = C.mean(0), F.mean(0)
    vC, vF = ((C - mc) ** 2).sum(1).mean(), ((F - mf) ** 2).sum(1).mean()
    # mean difference relative to its standard error (paired: same worlds)
    d = C - F
    md = d.mean(0)
    se2 = ((d - md) ** 2).sum(1).mean() / N
    z = float((md ** 2).sum() / max(se2, 1e-30))
    out[l] = {"var_cone": float(vC), "var_full": float(vF), "ratio": float(vF / max(vC, 1e-30)),
              "mean_norm_cone": float(np.linalg.norm(mc)), "mean_norm_full": float(np.linalg.norm(mf)),
              "diff_sq_over_se2": z}
    print(f"level {l}: var full {vF:.4g}  cone {vC:.4g}  ratio {vF / max(vC, 1e-30):.1f}  "
          f"|mean| full {np.linalg.norm(mf):.3g} cone {np.linalg.norm(mc):.3g}  "
          f"|mean diff|^2 / se^2 {z:.2f} (about 1 if unbiased)")
json.dump(out, open(f"reportables/2026-10-13_cfast/gradvar_{sys.argv[1]}_pre{pre}.json", "w"), indent=1)
