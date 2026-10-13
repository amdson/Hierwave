"""Gradient check numbers (tiny enumerable problem): crb vs cfast (rb off) vs
the biased negative control, zero and wrong critic.  Run with JAX_PLATFORMS=cpu.
usage: gradcheck.py M"""
import json, sys
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.variants import _crb_core, _gradcheck as gc
from castlegen.variants.crb import CRB

M = int(sys.argv[1])
TAU, LAM, B = 1.5, np.array([2.0, 3.0]), 16
p = gc.tiny_problem()
var = CRB(p, 0, "/tmp/crb_gc")
var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
var.cparams = gc.zero_critic(var.cparams)
ex = gc.Exact(var, LAM, reference=False, ref_check=200)
J, g, info = ex.J_and_grad(var.params, TAU)
gf = gc.flat(g)
lay = var.layout(Region(8))
out = {"J": J, "info": info, "M": M, "B": B}
for name, rb, neg, crit in [("crb", True, None, "zero"), ("cfast", False, None, "zero"),
                            ("crb", True, None, "wrong"), ("cfast", False, None, "wrong"),
                            ("neg_uniform", True, "uniform", "zero")]:
    fn = _crb_core.build_train(var, lay, rb=rb, neg=neg)
    cp = var.cparams if crit == "zero" else gc.randomize(var.cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
    gs = {k: jnp.float32(1.0) for k in var.params}
    est = []
    for m in range(M):
        seeds = jnp.arange(40_000 + m * B, 40_000 + (m + 1) * B, dtype=jnp.int32)
        gg, tot = fn["agrad"](var.params, cp, gs, seeds, {}, jnp.float32(TAU), jnp.asarray(LAM, jnp.float32),
                              jax.random.PRNGKey(m))
        gg = {k: jax.tree_util.tree_map(lambda x: x * fn["nval"][int(k[1:].split("_")[0])], v) for k, v in gg.items()}
        est.append(gc.flat(gg))
    est = np.array(est)
    res = {}
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], B)
        res[k] = {f: z[f] for f in ("proj_ratio", "proj_z", "rand_z", "hotelling_p", "rel_noise", "snr_world", "cos")}
        print(name, crit, k, json.dumps(res[k]), flush=True)
    out[f"{name}_{crit}"] = res
json.dump(out, open("reportables/2026-10-13_crb/gradcheck.json", "w"), indent=1)
