"""usage: plot_runs.py run1[,run2...] (from the report folder).  Writes images/loop_dirt_line.png,
images/loop_accept.png; prints a summary table (last 5 iterations mean)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = sys.argv[1].split(",")
ref = [("sweep2b ck300_own", "../2026-10-14_sweep2b/runs/ck300_own/loop.jsonl")]
def load(path):
    return [d for d in (json.loads(l) for l in open(path)) if "iter" in d]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
allr = [(r, f"runs/{r}/loop.jsonl") for r in runs] + ref
for i, (r, path) in enumerate(allr):
    L = load(path); u = [d["update"] for d in L]; c = f"C{i}"
    ax[0].plot(u, [d["own_raw"]["dirt"] for d in L], c, label=r)
    ax[0].plot(u, [d["own_refined"]["dirt"] for d in L], c + ":")
    ax[1].plot(u, [d["own_raw"]["line_match"] for d in L], c, label=r)
    ax[1].plot(u, [d["own_refined"]["line_match"] for d in L], c + ":")
    ax[2].plot(u, [d["viol_own"] for d in L], c, label=r)
ax[0].set_title("own-draw dirt fraction (solid raw, dotted refined)"); ax[1].set_title("own-draw line_match")
ax[2].set_title("own-draw violations per cell"); ax[2].set_yscale("log")
ax[1].axhline(0.5, color="gray", lw=0.5)
for a in ax: a.set_xlabel("ML updates"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("images/loop_dirt_line.png", dpi=110)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for i, r in enumerate(runs):
    L = load(f"runs/{r}/loop.jsonl"); c = f"C{i}"
    u = [d["update"] for d in L]
    for k, ls in (("acc_flip_L4", "-"), ("acc_flip_L3", "--"), ("acc_tt_L4", "-"), ("acc_tt_L3", "--"), ("acc_mtm_L2", ":")):
        if not any(k in d for d in L): continue
        ax[0].plot(u, [d.get(k) if d.get(k) is not None else np.nan for d in L], c + ls, label=f"{r} {k}")
    for k, ls in (("L4", "-"), ("L3", "--"), ("tt_L4", "-"), ("tt_L3", "--")):
        if not any(f"n_pure_{k}" in d for d in L): continue
        ax[1].plot(u, [d.get(f"pure2part_{k}", np.nan) / max(d.get(f"n_pure_{k}", 0), 1) if f"n_pure_{k}" in d else np.nan for d in L],
                   c + ls, label=f"{r} {k}: accepted pure->partial / pure proposals")
    if any("an_valid" in d for d in L):
        ax[2].plot(u, [d.get("an_valid", np.nan) for d in L], c, label=f"{r} anneal valid")
        ax[2].plot(u, [d.get("an_ess_mean", np.nan) / 8 for d in L], c + "--", label=f"{r} ESS/K")
        ax[2].plot(u, [d.get("an_pure2part_w", np.nan) for d in L], c + ":", label=f"{r} weighted pure->partial")
ax[0].set_title("acceptance per kernel and level"); ax[1].set_title("accepted pure-dirt block -> partial / pure proposals")
ax[2].set_title("annealed L4 candidates")
for a in ax: a.set_xlabel("ML updates"); a.legend(fontsize=6)
plt.tight_layout(); plt.savefig("images/loop_accept.png", dpi=110)
keys = ["dirt", "line_match"]
for r, path in allr:
    L = load(path); last = L[-5:]
    def m(f):
        v = [f(d) for d in last]; v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else float("nan")
    s = {"run": r, "updates": L[-1]["update"], "dirt": m(lambda d: d["own_raw"]["dirt"]), "line_match": m(lambda d: d["own_raw"]["line_match"]),
         "viol_own": m(lambda d: d["viol_own"]), "acc_L3": m(lambda d: d.get("acc_flip_L3")), "acc_L4": m(lambda d: d.get("acc_flip_L4")),
         "acc_tt_L3": m(lambda d: d.get("acc_tt_L3")), "acc_tt_L4": m(lambda d: d.get("acc_tt_L4"))}
    for k in ("L3", "L4", "tt_L3", "tt_L4"):
        P = sum(d.get(f"pure2part_{k}", 0) for d in L); N = sum(d.get(f"n_pure_{k}", 0) for d in L)
        A = sum(d.get(f"chg_val_{k}", 0) for d in L)
        s[f"pure2part_{k}"] = f"{P}/{N} pure proposals ({A} accepted value changes)"
    if any("an_valid" in d for d in L):
        s["an_valid"] = float(np.mean([d["an_valid"] for d in L if "an_valid" in d]))
        s["an_ess"] = float(np.mean([d["an_ess_mean"] for d in L if "an_ess_mean" in d]))
        s["an_pure_old"] = float(np.mean([d["an_pure_old"] for d in L if "an_pure_old" in d]))
        s["an_pure2part_w"] = float(np.mean([d["an_pure2part_w"] for d in L if "an_pure2part_w" in d]))
        s["an_pure2part_valid_unw"] = float(np.mean([d["an_pure2part_valid_unw"] for d in L if "an_pure2part_valid_unw" in d]))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
