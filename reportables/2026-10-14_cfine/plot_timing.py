"""ms per replay by branched level: windowed vs full-world (ground, B = 8, R = 16, K = 4; shared GPU)."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
D = "reportables/2026-10-14_cfine"
t = json.load(open(f"{D}/timing_levels.json"))
lv = [2, 1, 0]
w = [t[f"l{l}_window"]["ms_per_replay"] for l in lv]
f = [t[f"l{l}_full"]["ms_per_replay"] for l in lv]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
b1 = ax.bar(x - 0.2, f, 0.38, color="#eb6834", label="full-world replay")
b2 = ax.bar(x + 0.2, w, 0.38, color="#2a78d6", label="windowed (cone) replay")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03, f"{r.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x, [f"level {l}" for l in lv]); ax.set_ylabel("ms per replay")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=8)
ax.set_title("Replay cost by branched level (ground, shared GPU)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{D}/images/replay_timing.png", dpi=110)
