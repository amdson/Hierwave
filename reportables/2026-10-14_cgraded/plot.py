"""Violations per level vs sweep, binary vs graded penalty, one row per lambda."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "reportables/2026-10-14_cgraded"
lams = [float(a) for a in sys.argv[1:]] or [20.0, 200.0]
COL = {"binary": "#2a78d6", "graded": "#eb6834"}
fig, axs = plt.subplots(len(lams), 6, figsize=(16, 3.0 * len(lams)), sharex=True, squeeze=False)
for r, lam in enumerate(lams):
    res = json.load(open(f"{D}/result_lam{lam:g}.json"))
    start = np.array(res["start"])[:, :-1].sum(0)
    for name, arm in res["arms"].items():
        sw = np.array([0] + [t["sweep"] for t in arm["trace"]])
        v = np.array([start] + [np.array(t["viol"])[:, :-1].sum(0) for t in arm["trace"]])
        v = np.concatenate([v, v.sum(1, keepdims=True)], 1)
        for c in range(6):
            ax = axs[r, c]
            ax.plot(sw + 1, v[:, c], color=COL[name], lw=2, marker="o", ms=4, label=name)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_title((f"level {c}" if c < 5 else "total") + f"   λ = {lam:g}", fontsize=10)
    axs[r, 0].set_ylabel(f"violated instances ({res['B']} worlds)")
for ax in axs[-1]:
    ax.set_xlabel("sweep + 1")
axs[0, 0].legend(frameon=False)
fig.suptitle("ground, colour Gibbs at τ = 1 from greedy-pass starts: binary count vs graded distance penalty", fontsize=11)
fig.tight_layout()
out = "images/cgraded_violations.png"
fig.savefig(out, dpi=110)
print(out)
