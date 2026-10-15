"""Valid refinements vs sweep: rows coarse source, columns depth, lines penalty."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "reportables/2026-10-14_cmix"
res = json.load(open(f"{D}/refine.json"))
COL = {"bin": "#2a78d6", "grad": "#eb6834"}
srcs, depths = ["tight", "chain"], [1, 2, 4]
fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharey=True, sharex=True)
for r in res:
    ax = axs[srcs.index(r["source"]), depths.index(r["depth"])]
    c = np.array(r["curve"])
    s, nv = c[:, 0], c[:, 2]
    s = np.append(s, 600)
    nv = np.append(nv, nv[-1])                 # runs stop once every world is valid
    ax.step(s, nv, where="post", color=COL[r["pen"]], lw=2, label=r["pen"])
    ax.set_title(f"{r['source']} coarse, levels < {r['depth']} free", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
for ax in axs[:, 0]:
    ax.set_ylabel("valid worlds (of 16)")
for ax in axs[-1]:
    ax.set_xlabel("sweep")
axs[0, 0].legend(frameon=False)
fig.suptitle("ground: refining from uniform-random fine levels under clamped valid coarse levels (λ = 20)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{D}/refine.png", dpi=110)
fig.savefig("images/cmix_refine.png", dpi=110)
