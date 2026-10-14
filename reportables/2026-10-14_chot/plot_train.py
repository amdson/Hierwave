"""Per-level entropy of q (non-forced draws, training batch), ESS and W_max
over training, for chot runs vs chot_off (cfast through the same path)."""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
D = "reportables/2026-10-14_chot/runs"
runs = sys.argv[1:] or ["chot", "chot_off"]
fig, ax = plt.subplots(1, 6, figsize=(22, 3.6))
cols = {"chot": "#c0392b", "chot_off": "#2c3e50", "chot_t2": "#e67e22", "chot_all": "#8e44ad"}
for r in runs:
    L = [json.loads(l) for l in open(f"{D}/{r}/train_ground.jsonl")]
    u = [d for d in L if "H_level" in d]
    up = [d["update"] for d in u]
    H = np.array([d["H_level"] for d in u])
    for l in range(H.shape[1]):
        ax[4 - l].plot(up, H[:, l], label=r, color=cols.get(r))
        ax[4 - l].set_title(f"level {l}: mean H(q), nats")
    a = [d for d in L if "ess" in d]
    ax[5].plot([d["update"] for d in a], np.convolve([d["ess"] for d in a], np.ones(20) / 20, "same"),
               label=r, color=cols.get(r))
    ax[5].set_title("ESS of batch weights (B = 8), 20-update mean")
for x in ax:
    x.set_xlabel("update"); x.legend(fontsize=7)
plt.tight_layout(); plt.savefig("reportables/2026-10-14_chot/images/train_entropy.png", dpi=90)
