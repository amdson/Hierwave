"""Saturation curves from csat train logs: per level H (non-forced entropy of q),
H0 (b = 0), forced fraction, |b| spread, dirt fraction, tau; one column per run."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "reportables/2026-10-14_csat"
runs = sys.argv[1].split(",")
out = sys.argv[2]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#a855c9", "#d4a514"]
rows = [("H", "entropy of q, non-forced draws (nats)"), ("H0", "same with b = 0"),
        ("spread", "b spread over A, non-forced"), ("forced", "forced fraction |A| = 1"), ("dirt", "dirt fraction / tau")]
fig, ax = plt.subplots(len(rows), len(runs), figsize=(4.2 * len(runs), 12), squeeze=False)
for j, r in enumerate(runs):
    L = [json.loads(x) for x in open(f"{D}/runs/{r}/train_ground.jsonl")]
    dg = [d for d in L if "diag" in d]
    u = np.array([d["update"] for d in dg])
    levels = sorted(int(k[1:]) for k in dg[0]["diag"] if k.startswith("L"))
    for i, (m, lab) in enumerate(rows):
        a = ax[i, j]
        if m == "dirt":
            a.plot(u, [d["diag"]["dirt"] for d in dg], c="k", label="dirt fraction")
            uu = [d["update"] for d in L if "update" in d]
            a.plot(uu, [d["tau"] - 1 for d in L if "update" in d], c="0.6", ls="--", label="tau - 1")
            a.set_ylim(-0.02, 1.05)
        else:
            for l in levels:
                a.plot(u, [d["diag"][f"L{l}"][m] for d in dg], c=cols[l], label=f"level {l}")
        if j == 0:
            a.set_ylabel(lab)
        if i == 0:
            nu = max(d.get("update", 0) for d in L)
            a.set_title(f"{r} ({nu} updates)")
        a.grid(alpha=.3)
        if j == len(runs) - 1:
            a.legend(fontsize=7, frameon=False)
    ax[-1, j].set_xlabel("update")
plt.tight_layout(); plt.savefig(out, dpi=110)
