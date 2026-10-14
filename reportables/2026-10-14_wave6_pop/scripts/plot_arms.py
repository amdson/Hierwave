import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = {"C rkl123x2 (L1-3 step x2)": "runs/C_rkl123x2", "D rkl123x2 + 40 sweeps": "runs/D_rkl123x2_cg40",
        "ref ckpop_hard_rkl bench seed 0 (wave5)": "../2026-10-14_wave5_pop/runs/bench_run/train_s0.jsonl"}
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for lab, d in runs.items():
    it = [json.loads(l) for l in open(d if d.endswith(".jsonl") else d + "/loop.jsonl")]; it = [r for r in it if "lv_own" in r]
    t = [r["t"] for r in it]; lv = np.array([r["lv_own"] for r in it])
    ax[0].plot(t, lv[:, 2], label=lab); ax[1].plot(t, lv[:, 3], label=lab)
    ax[2].plot(t, [r["own8"]["line_match"] for r in it], label=lab)
    ax[3].plot(t, [r["lam"][1] for r in it], label=lab)
for a, ti in zip(ax, ["own-draw viol frac L2", "own-draw viol frac L3", "own-draw line_match (8 draws)", "lambda L1"]):
    a.set_title(ti); a.set_xlabel("train s (incl. setup)")
ax[0].set_yscale("symlog", linthresh=1e-4); ax[1].set_yscale("symlog", linthresh=1e-4)
ax[0].axhline(2e-4, c="k", ls=":"); ax[1].axhline(2e-4, c="k", ls=":"); ax[2].legend(fontsize=7)
fig.suptitle("wave6-pop L1-L3 arms, own draws only, inside the frozen constitution (ground, seed 0, 300 s)")
fig.tight_layout(); fig.savefig("images/arms_curves.png", dpi=90)
