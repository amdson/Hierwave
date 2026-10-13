"""Blocking timings of fused cfast updates for several configurations,
interleaved so that machine load hits them alike.
usage: profile_cfast.py problem 'cfg1' 'cfg2' ...   (cfg: python dict literal)"""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cfast import CFast

prob = problems.get(sys.argv[1])
vs = []
for a in sys.argv[2:]:
    cfg = eval(a)
    v = CFast(prob, 0, "/tmp", **cfg)
    region = Region(v.cfg["W_train"] or prob.W_train)
    lay = v.layout(region)
    t = time.time()
    v.fn = v.train_fns(lay)
    v._measure_baseline(lay, lambda d: None)
    print(f"{a}: setup {time.time() - t:.2f}", flush=True)
    B = v.cfg["B"]
    v.args = (jnp.arange(B, dtype=jnp.int32), v._batch(np.arange(B), region))
    vs.append((a, v))
res = {a: [] for a, _ in vs}
for it in range(6):
    for a, v in vs:
        tau, lam = jnp.float32(2.0), jnp.asarray(v.lam, jnp.float32)
        seeds, des = v.args
        t = time.time()
        out = jax.block_until_ready(v.fn["fused"](v.params, v.cparams, v.opt, v.copt, v.gs, seeds + it, des, tau,
                                                  lam, 1e-3, 1e-3, 10.0, jax.random.PRNGKey(it)))
        v.params, v.cparams, v.opt, v.copt = out[1:5]
        res[a].append(time.time() - t)
for a, t in res.items():
    print(f"{a}: first {t[0]:.2f}  median {np.median(t[1:]):.3f}  {[round(x, 3) for x in t[1:]]}")
