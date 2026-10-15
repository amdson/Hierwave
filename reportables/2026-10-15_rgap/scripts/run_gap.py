import sys, time, json
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants._crevise_gradcheck import tiny_flat_problem
from castlegen.variants.crevise import CRevise
from castlegen.variants._rgap import enumerate_with_logr, RGap, LAMS

def build(which, seed):
    if which == "flat":
        var = CRevise(tiny_flat_problem(), 0, "/tmp/rgap_f", R=1, rho=0.5, guide=True); lam = np.array([20.0])
    else:
        var = CRevise(gc.tiny_problem(), 0, "/tmp/rgap_t", R=[0, 1], rho=0.5, guide=False); lam = np.array([2.0, 3.0])
    var.cparams = gc.zero_critic(var.cparams)
    lay = var.layout(Region(var.problem.W_train)); sch = var.sched(lay)
    t = time.time()
    forced, final, masks, nl = enumerate_with_logr(sch, 0.5, seed)
    ex = RGap(var, 1.0, lam, seed, forced, final, nlogr=nl); ex.masks = masks
    print(which, "seed", seed, "N", ex.N, "erased", int(sum(m.sum() for m in masks.values())), "enum s", round(time.time() - t), flush=True)
    return var, ex

def adam(var, ex, mode, secs, lr=0.05):
    p = var.params; m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p)
    t0 = time.time(); i = 0
    while time.time() - t0 < secs:
        i += 1
        J, g = ex.grad_with(p, mode)
        m = jax.tree_util.tree_map(lambda a, b: .9 * a + .1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: .999 * a + .001 * b * b, v, g)
        p = jax.tree_util.tree_map(lambda x, a, b: x - lr * (a / (1 - .9 ** i)) / (jnp.sqrt(b / (1 - .999 ** i)) + 1e-8), p, m, v)
    print("  trained", mode, i, "steps, J", round(J, 3), flush=True)
    return p

def row(tag, which, ex, params, ncell):
    r = ex.parts(params)
    nM = r["uniform"]["charge"]
    out = {"problem": which, "gen": tag, "rkl": r["rkl"], "H_x0": r["H_x0"]}
    for k in ("uniform", "des5", "des10", "des20"):
        out[k] = {**r[k], "gap_per_cell": r[k]["gap"] / ncell}
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    which, seed, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    var, ex = build(which, seed)
    ncell = int(sum(int(np.prod(var.layout(Region(var.problem.W_train)).hw[c.level])) for c in var.problem.sampled()))
    print("cells", ncell)
    res = []
    p0 = var.params
    res.append(row("untrained", which, ex, p0, ncell))
    pu = adam(var, ex, None, secs); res.append(row("trained_uniform", which, ex, pu, ncell))
    pd = adam(var, ex, 10.0, secs); res.append(row("trained_des10", which, ex, pd, ncell))
    json.dump(res, open(f"reportables/2026-10-15_rgap/runs/gap_{which}_{seed}.json", "w"), indent=1)
