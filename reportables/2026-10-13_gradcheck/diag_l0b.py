"""Which component drives the M = 300 Hotelling rejection on l0?"""
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
M = 1500
est = np.array([gc.flat(gc.estimator_grad(var, np.arange(14_000 + 16 * m, 14_000 + 16 * (m + 1)), TAU, LAM)[0])[s]
                for m in range(M)])
np.save("/tmp/gc_est_l0.npy", est)
R = np.random.default_rng(1).normal(size=(x.size, 16))
R /= np.linalg.norm(R, axis=0)
R = np.concatenate([(x / np.linalg.norm(x))[:, None], R], 1)
for m in (300, 600, 1500):
    P = est[:m] @ R
    dev = P.mean(0) - x @ R
    C = np.cov(P, rowvar=False) / m
    w, V = np.linalg.eigh(C)
    comp = (V.T @ dev) ** 2 / w
    print(m, "eig min/max", w.min(), w.max(), "top contributions", np.sort(comp)[-4:].round(1), "T2", comp.sum().round(1))
    v = V[:, np.argmax(comp)]
    pv = P @ v
    print("   along worst: values distinct", len(np.unique(pv.round(6))), "sd", pv.std(), "mean-exact", (pv.mean() - (x @ R) @ v))
