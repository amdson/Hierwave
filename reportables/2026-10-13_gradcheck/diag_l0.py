"""Why does Hotelling reject l0 at M = 300 (test seeds 14000..)? Heavy tails or bias?"""
import sys

import jax
import numpy as np

from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

TAU, LAM = 1.5, np.array([2.0, 3.0])
p = gc.tiny_problem()
var = CPass(p, 0, "/tmp/gcw", K=4)
var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
var.cparams = gc.zero_critic(var.cparams)
ex = gc.Exact(var, LAM, cache="/tmp/gradcheck_tiny_exact.npz")
J, g, _ = ex.J_and_grad(var.params, TAU)
s = gc.block_slices(g)["l0_t"]
x = gc.flat(g)[s]
for seed0, M in ((14_000, 300), (14_000, 1500), (8_000_000, 300), (9_000_000, 300)):
    est = np.array([gc.flat(gc.estimator_grad(var, np.arange(seed0 + 16 * m, seed0 + 16 * (m + 1)), TAU, LAM)[0])[s]
                    for m in range(M)])
    z = gc.ztest(est, x, 16)
    u = x / np.linalg.norm(x)
    d = est @ u
    big = np.argsort(-np.abs(d - d.mean()))[:5]
    zt = gc.ztest(np.delete(est, big, 0), x, 16)
    print(seed0, M, {k: round(z[k], 4) for k in ("hotelling_p", "hotelling_k", "unseen_frac", "proj_z", "proj_ratio", "rand_z")},
          "max |dev|/sd along u", round(float(np.abs(d - d.mean()).max() / d.std()), 1),
          "without top-5 batches p", round(zt["hotelling_p"], 4), flush=True)
