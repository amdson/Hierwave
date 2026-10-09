"""Identifiability of the dem paint potential (stage 1 helper): rank of the
window feature differences and the null-space component of the fit."""
import numpy as np
from castlegen.channels.circles import Circles
from castlegen.channels.induce_circles import paint_region_mid
from castlegen.channels import paintpot as pp

C = Circles(6, 6)
rng = np.random.default_rng(21)
Z = ((rng.random((3000, 5, 5)) < 0.5) * rng.integers(1, C.D, (3000, 5, 5))).astype(np.int32)
r = np.zeros((3, 3), np.int32)
Pz = np.stack([paint_region_mid(C, z[1:-1, 1:-1], z) for z in Z])
Pr = np.stack([paint_region_mid(C, r, z) for z in Z])
X = pp.features(Pz, 4) - pp.features(Pr, 4)
y = C.fz[Pz].sum((1, 2)) - C.fz[Pr].sum((1, 2))
s = np.linalg.svd(X, compute_uv=False)
print("features", X.shape[1], "rank", int((s > 1e-8 * s[0]).sum()))
th, d = pp.fit(X, y, 1e-6)
th0 = pp.pack(C.fz, np.zeros((4, 4, 4)))
print("rel", d["rel"], "|X (th - th0)| rms", np.sqrt(np.mean((X @ (th - th0)) ** 2)), "|th - th0|", np.linalg.norm(th - th0))
used = np.flatnonzero(np.abs(X).sum(0) > 0)
print("pair features never varying:", X.shape[1] - len(used))
