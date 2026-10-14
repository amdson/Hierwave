"""Clamped completions (trained, L = 3 and L = 4, probe jitter 0 and 2, seed 0) for every run side by side,
plus the probe itself."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
p = get_problem("ground"); reg = Region(p.W_train)
runs = sys.argv[1:]
pr = p.probes(3_000_000, reg)
cols = [("probe j0", None, 0), ("probe j2", None, 2), ("L=3 j0", "probe_L3", 0), ("L=3 j2", "probe_L3", 2),
        ("L=4 j0", "probe_L4", 0)]
fig, ax = plt.subplots(len(runs) + 1, len(cols), figsize=(3.2 * len(cols), 1.9 * (len(runs) + 1)))
for j, (nm, c, k) in enumerate(cols):
    if c is None:
        ax[0, j].imshow(p.render({"t": pr[k]["t"]}, scale=1), interpolation="nearest")
    else:
        z = np.load(f"runs/{runs[0]}/ground_raw.npz")
        ax[0, j].imshow(p.render({"t": z[f"untrained__{c}__t"][k]}, scale=1), interpolation="nearest")
        nm = "untrained " + nm
    ax[0, j].set_title(nm, fontsize=8); ax[0, j].axis("off")
for i, r in enumerate(runs):
    z = np.load(f"runs/{r}/ground_raw.npz")
    for j, (nm, c, k) in enumerate(cols):
        a = ax[i + 1, j]; a.axis("off")
        if c is None:
            continue
        a.imshow(p.render({"t": z[f"trained__{c}__t"][k]}, scale=1), interpolation="nearest")
        a.set_title(f"{r}: {nm}", fontsize=8)
plt.tight_layout(); plt.savefig("images/clamped_completions.png", dpi=100)
