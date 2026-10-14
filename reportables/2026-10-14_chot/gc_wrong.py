"""Re-check of the wrong-critic gradcheck with fresh seeds and more batches,
eps = 0 (cfast's estimator through chot's path) as reference."""
import sys, jax, numpy as np
sys.path.insert(0, "tests/variants")
import test_chot as T
from castlegen.variants import _gradcheck as gc
p = gc.tiny_problem()
from castlegen.variants.chot import CHot
var = CHot(p, 0, "/tmp/chot_gc", tau_b=3.0)
var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
var.cparams = gc.zero_critic(var.cparams)
ex = gc.Exact(var, T.LAM, reference=False, ref_check=50)
J, g, _ = ex.J_and_grad(var.params, T.TAU)
gf = gc.flat(g)
cw = gc.randomize(var.cparams, jax.random.PRNGKey(5), scale=3.0, last_only=True)
for name, eps in (("eps0", [0.0, 0.0]), ("all", [0.3, 0.3]), ("top", [0.0, 0.3])):
    est, cost, W = T._estimate(var, cw, 800, 16, 900_000, eps)
    for k, s in gc.block_slices(g).items():
        z = gc.ztest(est[:, s], gf[s], 16)
        print(name, k, {x: round(z[x], 3) for x in ("proj_z", "rand_z", "hotelling_p", "proj_ratio", "rel_noise")}, flush=True)
