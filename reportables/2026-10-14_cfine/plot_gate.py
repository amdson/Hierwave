"""Per-level non-forced entropy, b-spread and dirt fraction against update for gate-2 runs."""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_cfine"
SETS = {"gate2_entropy_eq300": {"cfine (300 updates)": ("#2a78d6", f"{D}/runs/cfine300"),
                                "cfast estimator (300 updates)": ("#eb6834", f"{D}/runs/cfast300")},
        "gate2_entropy_eq300_lr1": {"cfine lr 1e-3 (300 updates)": ("#2a78d6", f"{D}/runs/cfine300_lr1"),
                                    "cfast estimator lr 1e-3 (300 updates)": ("#eb6834", f"{D}/runs/cfast300_lr1")},
        "gate2_entropy_120s": {"cfine (120 s, 124 updates)": ("#2a78d6", f"{D}/runs/cfine"),
                               "cfast (120 s, 1508 updates)": ("#eb6834", f"{D}/runs/csat120")}}
out = sys.argv[1] if len(sys.argv) > 1 else "gate2_entropy_eq300"
runs = SETS[out]
fig, ax = plt.subplots(2, 6, figsize=(17, 5.6), sharex=True)
for lab, (col, path) in runs.items():
    rows = [json.loads(l) for l in open(f"{path}/train_ground.jsonl")]
    rows = [r for r in rows if "diag" in r]
    u = [r["update"] for r in rows]
    for i, l in enumerate(range(4, -1, -1)):
        ax[0, i].plot(u, [r["diag"][f"L{l}"]["H"] for r in rows], color=col, lw=2, label=lab)
        ax[1, i].plot(u, [r["diag"][f"L{l}"]["spread"] for r in rows], color=col, lw=2)
        ax[0, i].set_title(f"level {l}: entropy of q (non-forced)", fontsize=9)
        ax[1, i].set_title(f"level {l}: spread of b over A", fontsize=9)
    ax[0, 5].plot(u, [r["diag"]["dirt"] for r in rows], color=col, lw=2)
    ax[1, 5].plot(u, [r["diag"]["L0"]["forced"] for r in rows], color=col, lw=2)
ax[0, 5].set_title("dirt fraction", fontsize=9); ax[1, 5].set_title("level 0 forced fraction", fontsize=9)
for a in ax.ravel():
    a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
for a in ax[1]: a.set_xlabel("update")
ax[0, 0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(f"{D}/images/{out}.png", dpi=110)
