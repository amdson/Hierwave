"""Per-level entropy (non-forced, nats), dirt fraction and line_match over updates,
ccur vs the uninstrumented-schedule baseline (ccur_off = cfast + diagnostics), one figure."""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_ccur"
runs = sys.argv[1].split(",")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
lv = [("L4", "#d4a514"), ("L3", "#a855c9"), ("L2", "#1baf7a"), ("L1", "#eb6834"), ("L0", "#3a7bd5")]
fig, ax = plt.subplots(2, len(runs), figsize=(5.2 * len(runs), 6.2), squeeze=False, sharex=True)
for j, r in enumerate(runs):
    L = [json.loads(x) for x in open(f"{D}/runs/{r}/train_ground.jsonl")]
    dg = [d for d in L if "diag" in d]
    u = [d["update"] for d in dg]
    rel = [(d["update"], d["released"]) for d in L if d.get("released")]
    a = ax[0, j]
    for k, c in lv:
        a.plot(u, [d["diag"][k]["H"] for d in dg], c=c, label=k)
    a.plot(u, [d["diag"]["l4_cert4"]["H"] for d in dg], c="#d4a514", ls=":", label="cert4")
    a.set_ylabel("entropy of q, non-forced (nats)"); a.set_title(r); a.grid(alpha=.3)
    b = ax[1, j]
    b.plot(u, [d["diag"]["dirt"] for d in dg], c="k", lw=2, label="dirt fraction")
    b.plot(u, [d["diag"].get("line_match", float("nan")) for d in dg], c="#c0392b", lw=1.5, label="line_match")
    b.axhline(0.5, c="0.6", lw=.8, ls="--")
    b.set_ylim(0.3, 1.02); b.set_xlabel("update"); b.grid(alpha=.3)
    for x in (a, b):
        for ur, ls in rel:
            x.axvline(ur, c="0.3", lw=.8, ls="--")
            x.text(ur, x.get_ylim()[1], " release L" + ",".join(map(str, ls)), fontsize=7, va="top")
    if j == 0:
        a.legend(fontsize=7, frameon=False, loc="upper right"); b.legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(f"{D}/images/curves.png", dpi=110)
