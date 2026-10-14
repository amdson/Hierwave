"""Training curves of the equal-update (300) ground comparison: creplay R=(2,2),K=4 vs blev=0 (cfast estimator)."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = {"creplay_cap300": ("#2a78d6", "creplay R=(2,2), K=4"), "creplay_b0_cap300": ("#eb6834", "cfast estimator (blev=0)")}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
sm = lambda x: np.convolve(x, np.ones(25) / 25, mode="valid")
for v, (col, lab) in C.items():
    d = sorted(glob.glob(f"results/runs/*_{v}_ground_creplay2_eq300"))[-1]
    for s, ls in ((0, "-"), (1, "--")):
        U = [e for e in map(json.loads, open(f"{d}/train_s{s}.jsonl")) if "update" in e]
        u = np.array([e["update"] for e in U])[24:]
        ax[0].plot(u, sm([e["cost_per_world"] for e in U]), ls, color=col, lw=2, label=f"{lab}, seed {s}")
        for i, l in ((1, 3), (2, 2)):
            ax[i].plot(u, sm([e["raw_viol"][l] for e in U]), ls, color=col, lw=2)
for a, t in zip(ax, ["cost per world (25-update mean)", "raw violation frac, level 3", "raw violation frac, level 2"]):
    a.set_title(t, fontsize=10); a.set_xlabel("update"); a.grid(alpha=0.2)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
ax[0].legend(fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig("reportables/2026-10-13_creplay/images/r3_eq300_curves.png", dpi=110)
