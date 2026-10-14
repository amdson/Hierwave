"""Gate bars: fine cost per fine cell below L, own coarse vs probe-clamped."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
rows = []
for r in ("chot_off", "chot"):
    s = json.load(open(f"reportables/2026-10-14_chot/runs/{r}/ground_summary.json"))["summary"]
    c = lambda m, k, L: sum(s[f"{m}/{k}"][str(l)]["cost_per_cell"][0] for l in range(L))
    if r == "chot_off":
        rows.append(("untrained", [c("untrained", k, L) for k, L in (("own", 4), ("probe_L4", 4), ("own", 3), ("probe_L3", 3))]))
    rows.append((r if r == "chot" else "chot_off (= cfast)", [c("trained", k, L) for k, L in (("own", 4), ("probe_L4", 4), ("own", 3), ("probe_L3", 3))]))
fig, ax = plt.subplots(figsize=(7, 3.4))
lab = ["L=4 own", "L=4 fill", "L=3 own", "L=3 fill"]
x = np.arange(4); w = 0.27
for i, (n, v) in enumerate(rows):
    ax.bar(x + (i - 1) * w, v, w, label=n)
ax.axhline(0.014, color="k", lw=0.8, ls=":", label="probe's own soft energy")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("fine cost per fine cell (nats)")
ax.set_title("ground, 300 s, clamp gate"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("reportables/2026-10-14_chot/images/gate.png", dpi=90)
