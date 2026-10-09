"""Which learned levels to switch on: evaluate saved certland parameters with
subsets of levels learned (the rest Phi = 0) on the held-out noise fields of
certland_run.py.  Usage: python certland_levels.py tag"""
import os
import sys

import numpy as np

from castlegen.channels import certland as cl, certland_fit as cf

IMG = os.environ.get("CERTLAND_IMG", "/Users/amdson/dev/Hierwave/images")
tag = sys.argv[1]
z = np.load(f"{IMG}/certland_{tag}_params.npz")
P = [(z[f"th{i}"], int(z["m"][i])) for i in range(4)]
EVAL = [101, 102, 103, 104]
w = cl.World(128, 256)
rows = []
for name, on in [("none", ()), ("h2", (3,)), ("h2-4", (3, 2)), ("h2-8", (3, 2, 1)), ("all", (3, 2, 1, 0))]:
    w.params = [P[i] if i in on else None for i in range(4)]
    mk, ims = [], []
    for s in EVAL:
        cf.set_noise(w, s)
        bad = cl.generate(w, seed=s)
        m = cl.metrics(w)
        m["stuck"] = sum(bad)
        m["solid"] = float(w.rho[cl.NL - 1].mean())
        mk.append(m)
        ims.append(cl.render(w))
    f = lambda k: np.mean([m[k] for m in mk])
    print(f"{name:5s}: overhangs {f('overhangs'):.0f} stuck {f('stuck'):.0f} fill_err {f('fill_err'):.3f} "
          f"solid {f('solid'):.2f} (target {w.fill.mean():.2f})  |tiles-rho|/h^2 at h16 {np.mean([m['h16']['rho_err'] for m in mk]):.3f}",
          flush=True)
    rows.append((name, ims))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(len(rows), 4, figsize=(16.8, 2.4 * len(rows)), squeeze=False)
for r, (name, ims) in enumerate(rows):
    for c, im in enumerate(ims):
        ax[r, c].imshow(im, interpolation="nearest")
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if c == 0:
            ax[r, c].set_ylabel(f"learned: {name}", fontsize=9)
fig.tight_layout()
fig.savefig(f"{IMG}/certland_{tag}_levels.png", dpi=100)
print("wrote", f"{IMG}/certland_{tag}_levels.png")
