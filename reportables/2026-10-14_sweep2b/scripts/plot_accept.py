import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_sweep2b"
new = json.load(open(f"{D}/runs/accept_kA_mlinit.json"))
old = json.load(open("reportables/2026-10-14_kernel_b/runs/ground_cfast.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, st in zip(ax, ("probe", "own")):
    for data, nm, ls in ((new, "kernel-a params", "-"), (old, "cfast", ":")):
        for K, tau, c in ((1, 1.0, "C0"), (8, 1.0, "C1"), (1, 2.0, "C2"), (8, 2.0, "C3")):
            r = sorted([x for x in data if x["start"] == st and x["K"] == K and x["tau"] == tau], key=lambda x: x["L"])
            a.plot([x["L"] for x in r], [x["acc"] for x in r], ls, color=c, marker="o", label=f"{nm} K{K} tau{tau:g}" if st == "probe" else None)
    a.set_title(f"from {st} {'lifts' if st == 'probe' else 'draws'}"); a.set_xlabel("level L of the subtree block"); a.set_ylabel("acceptance")
    a.set_xticks([1, 2, 3, 4]); a.set_ylim(-0.02, 1.02)
ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{D}/images/accept_per_level.png", dpi=110)
