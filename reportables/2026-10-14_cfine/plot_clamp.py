"""Clamp gate: fine cost per fine cell below L, own coarse vs probe-clamped, per run."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
D = "reportables/2026-10-14_cfine"
runs = [("cfine300", "cfine\nlr 3e-3"), ("cfast300", "cfast est.\nlr 3e-3"), ("cfine300_lr1", "cfine\nlr 1e-3"),
        ("cfast300_lr1", "cfast est.\nlr 1e-3")]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, L in zip(axs, (4, 3)):
    own, fill = [], []
    for r, _ in runs:
        s = json.load(open(f"{D}/runs/{r}/ground_summary.json"))["summary"]
        c = lambda k: sum(s[f"trained/{k}"][str(l)]["cost_per_cell"][0] for l in range(L))
        own.append(c("own")); fill.append(c(f"probe_L{L}"))
    x = np.arange(len(runs))
    ax.bar(x - 0.2, own, 0.38, color="#2a78d6", label="own coarse context")
    ax.bar(x + 0.2, fill, 0.38, color="#eb6834", label="probe-clamped (fill-following)")
    for i in range(len(runs)):
        ax.text(x[i] - 0.2, own[i] + 0.03, f"{own[i]:.2f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.2, fill[i] + 0.03, f"{fill[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [n for _, n in runs], fontsize=8); ax.set_title(f"fine cost per fine cell below L = {L} (300 updates)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
axs[0].set_ylabel("nats per fine cell"); axs[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{D}/images/clamp_gate_eq300.png", dpi=110)
