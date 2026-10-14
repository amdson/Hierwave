"""Training cost per fine cell (levels 0-2, every world clamped at L = 3), 100-update running mean."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = [("cstart_f1L3", "tanh 4x4"), ("cstart_f1L3_ln", "LN+SiLU 4x4"), ("cstart_f1L3_ln6", "LN+SiLU 6x6"),
        ("cstart_f1L3_res", "tanh + count resid")]
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for (r, nm), c in zip(runs, cols):
    L = [json.loads(l) for l in open(f"runs/{r}/train_ground.jsonl")]
    y = np.array([d["cost_per_world"] for d in L if "cost_per_world" in d]) / (64 * 128)
    k = 100; ys = np.convolve(y, np.ones(k) / k, "valid")
    ax.plot(np.arange(len(ys)) + k, ys, color=c, lw=2, label=nm)
    ax.annotate(nm, (len(ys) + k, ys[-1]), fontsize=8, color="#444", xytext=(4, 0), textcoords="offset points", va="center")
ax.axhline(0.014, color="#888", lw=1, ls="--"); ax.text(50, 0.03, "probe soft energy 0.014", fontsize=8, color="#555")
ax.set_ylim(-0.1, 1.0); ax.set_xlim(0, 5800); ax.set_xlabel("update"); ax.set_ylabel("cost per fine cell (training λ)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Fine levels trained only in fill contexts (f = 1, L = 3), 300 s each", fontsize=10)
plt.tight_layout(); plt.savefig("images/train_cost.png", dpi=110)
