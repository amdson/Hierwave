"""Per-level entropy and line_match / dirt vs update from a train_diag.py log.
usage: plot_diag.py <run name> [<run name> ...]  -> images/diag_<run>.png"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
for run in sys.argv[1:]:
    L = [json.loads(l) for l in open(f"runs/{run}/train_ground.jsonl")]
    D = [d for d in L if "diag" in d]
    u = np.array([d["update"] for d in D])
    fr = np.array([d.get("frac", 0) for d in D]); tau = np.array([d["tau"] for d in D])
    fig, ax = plt.subplots(3, 1, figsize=(8, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
    for l in range(5):
        y = [d["diag"][f"L{l}"]["H"] for d in D]
        ax[0].plot(u, y, color=COL[l], lw=2, label=f"level {l}")
        ax[0].annotate(f"L{l}", (u[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color="#444", va="center")
    ax[0].set_ylabel("entropy of q, non-forced draws (nats)"); ax[0].legend(frameon=False, fontsize=8, ncol=5, loc="upper right")
    ax[1].plot(u, [d["diag"]["line_match"] for d in D], color=COL[0], lw=2, label="line_match (probe 0.88)")
    ax[1].plot(u, [d["diag"]["dirt"] for d in D], color=COL[1], lw=2, label="dirt fraction")
    ax[1].set_ylim(0.4, 1.02); ax[1].legend(frameon=False, fontsize=8, loc="center right"); ax[1].set_ylabel("fraction")
    ax[2].plot(u, fr, color="#555", lw=2, label="clamped fraction")
    ax[2].plot(u, (tau - 1) / 7, color="#555", lw=1.5, ls="--", label="(tau - 1) / 7")
    ax[2].set_ylabel("schedule"); ax[2].legend(frameon=False, fontsize=8, loc="upper right"); ax[2].set_xlabel("update")
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
        a.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax[0].set_title(f"{run}: ground, diagnostics on 8 held-out worlds every 10 updates (tau, lambda in use)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"images/diag_{run}.png", dpi=110); print("saved", run)
