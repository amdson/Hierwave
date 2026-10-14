import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_dimers_kernel"; ARM = sys.argv[1]
runs = {"from cfast 60 s": f"{D}/runs/{ARM}_cfast60.jsonl", "from init": f"{D}/runs/{ARM}_init.jsonl"}
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
names = ["horizontal", "vertical", "diagonal"]; col = ["#c0504d", "#4f81bd", "#4a9d4a"]
summ = {}
for j, (title, f) in enumerate(runs.items()):
    R = [json.loads(l) for l in open(f)]; R = [r for r in R if "iter" in r]
    u = np.array([r["update"] for r in R]); raw = np.array([r["orient_raw"] for r in R]); ref = np.array([r["orient_refined"] for r in R])
    for i in range(3):
        ax[0, j].plot(u, raw[:, i], color=col[i], label=names[i]); ax[1, j].plot(u, ref[:, i], color=col[i], label=names[i])
    ax[0, j].set_title(f"{title}: own draws (raw)"); ax[1, j].set_title(f"{title}: refined states (after kernel cycle)")
    for a in ax[:, j]: a.axhline(0.5, ls=":", c="gray", lw=.7); a.axhline(.25, ls=":", c="gray", lw=.7); a.set_ylim(-.02, .8)
    ax[1, j].set_xlabel("ML update")
    summ[title] = dict(updates=int(u[-1]), raw_first=raw[:3].mean(0).round(3).tolist(), raw_last=raw[-10:].mean(0).round(3).tolist(),
                       ref_first=ref[:3].mean(0).round(3).tolist(), ref_last=ref[-10:].mean(0).round(3).tolist(),
                       viol_raw_last=float(np.mean([r["viol_raw"] for r in R[-10:]])), viol_ref_last=float(np.mean([r["viol_refined"] for r in R[-10:]])),
                       acc_mt=float(np.mean([r["acc_mt"] for r in R])), chg_k2=float(np.mean([r["chg_k2"] for r in R])), chg_k3=float(np.mean([r["chg_k3"] for r in R])),
                       infeasible=float(np.mean([r["infeasible_per_state"] for r in R[-10:]])))
ax[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{D}/images/orientation_vs_update_{ARM}.png", dpi=120)
json.dump(summ, open(f"{D}/runs/summary_{ARM}.json", "w"), indent=1); print(json.dumps(summ, indent=1))
