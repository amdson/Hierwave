"""cbound figures: exact-gradient schedules (dimers 4x4) and sampled dimers runs."""
import json, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "reportables/2026-10-14_cbound/"
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for f in sorted(glob.glob(D + "runs/dimers4_exact_mlp_*.json")):
    d = json.load(open(f)); c = d["curve"]; tag = f.split("mlp_")[1][:-5]
    s = [m["step"] for m in c]
    ax[0].plot(s, [m["kl"] for m in c], label=tag); ax[1].plot(s, [m["entropy"] / 16 for m in c])
    ax[2].plot(s, [m["diagonal"] for m in c])
ax[0].set_yscale("log"); ax[0].set_title("exact KL (tau 1, lam 20)"); ax[1].set_title("entropy / cell (target 0.377)")
ax[2].set_title("diagonal fraction (target 0.248)")
for a in ax: a.axvline(600, color="grey", ls=":"); a.set_xlabel("step (schedule ends at 600)")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(D + "images/exact_schedules.png", dpi=110)

def runs(path):
    rows = [json.loads(l) for l in open(path) if l.startswith("{")]
    out = []
    for r in rows:
        if r["update"] == 1: out.append([])
        out[-1].append(r)
    return out
R = {"cfast": runs(D + "runs/dimers_log_cfast.txt")[0]}
w, l = runs(D + "runs/dimers_log_whole8_lam.txt")
R["cbound_whole8"] = w; R["cbound_lam"] = l
R["cbound_wholelam"] = runs(D + "runs/dimers_log_wholelam.txt")[0]
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
for k, rs in R.items():
    u = [r["update"] for r in rs]
    ax[0].plot(u, [r["raw"][2] for r in rs], label=k); ax[1].plot(u, [r["ent"] for r in rs])
    ax[2].plot(u, [max(r["raw_viol"], 1e-5) for r in rs]); ax[3].plot(u, [r["lam"] for r in rs])
ax[0].set_title("diagonal fraction (raw, 4 worlds)"); ax[1].set_title("entropy -log q / cell")
ax[2].set_title("raw violation fraction"); ax[2].set_yscale("log"); ax[3].set_title("lambda used")
for a in ax: a.set_xlabel("update")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(D + "images/dimers_sampled.png", dpi=110)
