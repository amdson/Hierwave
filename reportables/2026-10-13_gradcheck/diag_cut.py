"""Eigenvalue cutoff for the whitened T^2 test, on saved l0 batch gradients."""
import jax
import numpy as np
from scipy.stats import f as fdist

from castlegen.variants import _gradcheck as gc
from castlegen.variants.cpass import CPass

p = gc.tiny_problem()
var = CPass(p, 0, "/tmp/gcw", K=4)
var.params = gc.randomize(var.params, jax.random.PRNGKey(1))
ex = gc.Exact(var, np.array([2.0, 3.0]), cache="/tmp/gradcheck_tiny_exact.npz")
J, g, _ = ex.J_and_grad(var.params, 1.5)
x = gc.flat(g)[gc.block_slices(g)["l0_t"]]
est = np.load("/tmp/gc_est_l0.npy")
R = np.random.default_rng(1).normal(size=(x.size, 16))
R /= np.linalg.norm(R, axis=0)
R = np.concatenate([(x / np.linalg.norm(x))[:, None], R], 1)
for m0, m in ((0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (0, 1500)):
    P = est[m0:m] @ R
    M = m - m0
    dev = P.mean(0) - x @ R
    C = np.cov(P, rowvar=False) / M
    w, V = np.linalg.eigh(C)
    row = [f"{m0}-{m}"]
    for cut in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        keep = w >= cut * w.max()
        k = keep.sum()
        T2 = np.sum((V[:, keep].T @ dev) ** 2 / w[keep])
        F = T2 * (M - k) / (k * (M - 1))
        row.append(f"{cut:g}: k{k} p{fdist.sf(F, k, M - k):.3g}")
    print("  ".join(row))
print("eigs (rel):", np.sort(w)[::-1] / w.max())
