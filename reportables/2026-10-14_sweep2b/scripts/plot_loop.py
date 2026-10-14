"""usage: plot_loop.py run1[,run2...] out.png"""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_sweep2b/runs"
runs, out = sys.argv[1].split(","), sys.argv[2]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for i, r in enumerate(runs):
    L = [json.loads(l) for l in open(f"{D}/{r}/loop.jsonl")]
    L = [d for d in L if "iter" in d]
    u = [d["update"] for d in L]
    c = f"C{i}"
    ax[0].plot(u, [d["own_raw"]["line_match"] for d in L], c, label=f"{r} own draws")
    ax[0].plot(u, [d["own_refined"]["line_match"] for d in L], c + ":", label=f"{r} refined own")
    ax[1].plot(u, [d["own_raw"]["dirt"] for d in L], c)
    ax[2].plot(u, [d["soft_own"] for d in L], c, label=f"{r} own")
    ax[2].plot(u, [d["soft_own_refined"] for d in L], c + ":", label=f"{r} own refined")
    ax[2].plot(u, [d.get("soft_probe_refined") for d in L], c + "--", label=f"{r} probe refined")
    for k, ls in (("acc_flip_L4", "-"), ("acc_flip_L3", "--"), ("acc_mtm_L2", ":")):
        ax[3].plot(u, [d.get(k) for d in L], c + ls, label=f"{r} {k}")
ax[0].axhline(0.5, color="gray", lw=0.5); ax[0].set_title("line_match (own draws)"); ax[0].legend(fontsize=6)
ax[1].axhline(0.5, color="gray", lw=0.5); ax[1].set_title("dirt fraction (own draws)")
ax[2].set_title("soft energy per cell"); ax[2].legend(fontsize=6)
ax[3].set_title("acceptance per kernel"); ax[3].legend(fontsize=6)
for a in ax: a.set_xlabel("ML updates")
plt.tight_layout(); plt.savefig(out, dpi=110)
