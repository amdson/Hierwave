"""usage (report folder): plot_w.py run1,run2 -> images/w_curves.png"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = sys.argv[1].split(",")
fig, ax = plt.subplots(1, 4, figsize=(19, 3.7))
for i, r in enumerate(runs):
    L = [d for d in (json.loads(l) for l in open(f"runs/{r}/loop.jsonl")) if "iter" in d]
    c = f"C{i}"
    u = [d["update"] for d in L]
    ax[0].plot(u, [d["pop"]["line_match"] for d in L], c + ":", label=f"{r} population")
    m = [d for d in L if "own8" in d]
    ax[0].plot([d["update"] for d in m], [d["own8"]["line_match"] for d in m], c, label=f"{r} own draws (8)")
    ax[1].plot(u, [d["pop"]["dirt"] for d in L], c + ":", label=f"{r} population")
    ax[1].plot([d["update"] for d in m], [d["own8"]["dirt"] for d in m], c, label=f"{r} own draws (8)")
    for l, ls in zip(range(4), ["-", "--", "-.", ":"]):
        ax[2].plot([d["update"] for d in m], [max(d["lv_own"][l], 1e-6) for d in m], c + ls, label=f"{r} L{l}")
    ax[3].plot(u, [d["pop"]["valid"] for d in L], c, label=f"{r} population valid")
for a, t in zip(ax, ["line_match", "dirt fraction", "own-draw violated hard fraction per level (8 draws)", "population valid fraction"]):
    a.set_title(t, fontsize=9); a.set_xlabel("ML updates"); a.legend(fontsize=6)
ax[2].set_yscale("log"); ax[2].axhline(2e-4, color="gray", lw=0.5)
plt.tight_layout(); plt.savefig("images/w_curves.png", dpi=110)
