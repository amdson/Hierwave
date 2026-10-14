"""usage (from report folder): plot_pop.py run1,run2,...  -> images/pop_curves.png, summary lines."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = sys.argv[1].split(",")
load = lambda r: [d for d in (json.loads(l) for l in open(f"runs/{r}/loop.jsonl")) if "iter" in d]
fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
for i, r in enumerate(runs):
    L = load(r); u = [d["update"] for d in L]; c = f"C{i}"
    ax[0].plot(u, [d["own_raw"]["dirt"] for d in L], c, label=f"{r} own draws")
    ax[0].plot(u, [d["pop"]["dirt"] for d in L], c + ":", label=f"{r} population")
    ax[1].plot(u, [d["own_raw"]["line_match"] for d in L], c, label=f"{r} own draws")
    ax[1].plot(u, [d["pop"]["line_match"] for d in L], c + ":", label=f"{r} population")
    ax[2].plot(u, [d["pop"]["B_top_row"] for d in L], c + ":", label=f"{r} population")
    ax[2].plot(u, [d["own_raw"].get("B_top_row", np.nan) for d in L], c, label=f"{r} own draws")
    ax[3].plot(u, [d["pop"]["valid"] for d in L], c + ":", label=f"{r} population valid frac")
    ax[3].plot(u, [d["own_raw"]["valid"] for d in L], c, label=f"{r} own valid frac")
for a, t in zip(ax, ["dirt fraction", "line_match", "cert B4 mean, top row (/16)", "valid states (0 violations)"]):
    a.set_title(t); a.set_xlabel("ML updates"); a.legend(fontsize=6)
ax[1].axhline(0.5, color="gray", lw=0.5)
plt.tight_layout(); plt.savefig("images/pop_curves.png", dpi=110)
for r in runs:
    L = load(r); last = L[-5:]
    m = lambda f: float(np.mean([f(d) for d in last]))
    tt = {k: (sum(d.get(f"pure2part_tt_{k}", 0) for d in L), sum(d.get(f"n_pure_tt_{k}", 0) for d in L)) for k in ("L3", "L4")}
    print(json.dumps({"run": r, "updates": L[-1]["update"], "iters": len(L),
                      "own_dirt": round(m(lambda d: d["own_raw"]["dirt"]), 3), "own_lm": round(m(lambda d: d["own_raw"]["line_match"]), 3),
                      "own_Btop": round(m(lambda d: d["own_raw"].get("B_top_row", np.nan)), 3), "own_valid": round(m(lambda d: d["own_raw"]["valid"]), 3),
                      "own_viol": round(m(lambda d: d["own_raw"]["viol_per_cell"]), 5),
                      "pop_dirt": round(m(lambda d: d["pop"]["dirt"]), 3), "pop_lm": round(m(lambda d: d["pop"]["line_match"]), 3),
                      "pop_Btop": round(m(lambda d: d["pop"]["B_top_row"]), 3), "pop_valid": round(m(lambda d: d["pop"]["valid"]), 3),
                      "pop_viol": round(m(lambda d: d["pop"]["viol_per_cell"]), 5),
                      "acc_tt_L3": round(float(np.mean([d.get("acc_tt_L3", np.nan) for d in L])), 3),
                      "acc_tt_L4": round(float(np.mean([d.get("acc_tt_L4", np.nan) for d in L])), 3),
                      "tt_pure2part": {k: f"{a}/{b}" for k, (a, b) in tt.items()},
                      "time_refine": round(float(np.mean([d["time_refine"] for d in L[3:]])), 2)}))
