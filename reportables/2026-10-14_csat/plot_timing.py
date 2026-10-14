"""Collapse timing: per-channel entropy (non-forced) of cert4 and rho3/rho2/rho1 vs dirt fraction."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_csat"
runs = sys.argv[1].split(",")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
ch = [("l4_cert4", "#d4a514", "-"), ("l4_rho4", "#d4a514", ":"), ("l3_rho3", "#a855c9", "-"), ("l2_rho2", "#1baf7a", "-"),
      ("l1_rho1", "#eb6834", "-"), ("l3_cert3", "#a855c9", ":")]
fig, ax = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.4), squeeze=False)
for j, r in enumerate(runs):
    dg = [json.loads(x) for x in open(f"{D}/runs/{r}/train_ground.jsonl")]
    dg = [d for d in dg if "diag" in d and d["update"] <= 1000]
    u = [d["update"] for d in dg]
    a = ax[0, j]
    for k, c, ls in ch:
        a.plot(u, [d["diag"][k]["H"] for d in dg], c=c, ls=ls, label=k[3:])
    a2 = a.twinx(); a2.plot(u, [d["diag"]["dirt"] for d in dg], c="k", lw=2, label="dirt"); a2.set_ylim(0.8, 1.005)
    a2.set_ylabel("dirt fraction"); a.set_ylabel("entropy, non-forced (nats)"); a.set_xlabel("update")
    a.set_title(r); a.grid(alpha=.3)
    if j == 0:
        a.legend(fontsize=7, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(f"{D}/images/collapse_timing.png", dpi=110)
