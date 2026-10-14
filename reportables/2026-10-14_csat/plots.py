import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("ground_summary.json"))["summary"]; ps = json.load(open("probe_soft.json"))["per_level_mean"]
conds = [("own", "own coarse draw", "#2a78d6"), ("probe_L4", "probe-clamped L>=4", "#eb6834"), ("probe_L3", "probe-clamped L>=3", "#1baf7a")]
mets = [("cost_per_cell", "cost per fine cell (nats)"), ("viol_frac", "violated fraction (raw, energy)"),
        ("forced_frac", "forced draws (|A| = 1)"), ("noopt_frac", "draws with no valid option")]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
for r, model in enumerate(("untrained", "trained")):
    for c, (m, lab) in enumerate(mets):
        a = ax[r, c]
        for i, (cond, cl, col) in enumerate(conds):
            Lmax = {"own": 4, "probe_L4": 3, "probe_L3": 2}[cond]
            ls = np.arange(Lmax + 1)
            v = [d[f"{model}/{cond}"][str(l)][m][0] for l in ls]
            e = [d[f"{model}/{cond}"][str(l)][m][1] for l in ls]
            a.bar(ls + (i - 1) * 0.27, v, 0.25, yerr=e, color=col, label=cl, edgecolor="white", linewidth=1)
        if m == "cost_per_cell":
            a.scatter(np.arange(4), ps[:4], marker="_", s=200, c="k", zorder=5, label="probe's own soft energy")
        a.set_xticks(range(5)); a.set_xlabel("level"); a.set_title(f"{model}: {lab}"); a.grid(axis="y", alpha=.3)
        a.axhline(0, color="#999", lw=.8)
ax[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("ground, cresid 150 s: levels below the clamp, fill-following (probe) vs own coarse context; 32 seeds x 4 probes, SE bars")
plt.tight_layout(); plt.savefig("images/ground_per_level_bars.png", dpi=110)

# total fine cost below L
fig, a = plt.subplots(figsize=(6.5, 3.6))
rows = []
for L in (4, 3):
    for model in ("untrained", "trained"):
        own = sum(d[f"{model}/own"][str(l)]["cost_per_cell"][0] for l in range(L))
        cl = sum(d[f"{model}/probe_L{L}"][str(l)]["cost_per_cell"][0] for l in range(L))
        rows.append((f"L={L}\n{model}", own, cl))
x = np.arange(len(rows))
a.bar(x - .2, [r[1] for r in rows], .38, color="#2a78d6", label="own coarse context", edgecolor="white")
a.bar(x + .2, [r[2] for r in rows], .38, color="#eb6834", label="fill-following (probe) context", edgecolor="white")
for i, r in enumerate(rows):
    a.text(i - .2, r[1] + .03, f"{r[1]:.2f}", ha="center", fontsize=8); a.text(i + .2, r[2] + .03, f"{r[2]:.2f}", ha="center", fontsize=8)
a.set_xticks(x); a.set_xticklabels([r[0] for r in rows]); a.set_ylabel("cost of levels < L per fine cell (nats)")
a.legend(frameon=False, fontsize=8); a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("images/ground_total_below_L.png", dpi=110)
for r in rows: print(r)
