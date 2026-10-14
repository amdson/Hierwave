import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob("runs/*.json"))}
rows = []
for n, r in runs.items():
    c = r["curve"][-1]; ref = r["ref"]
    ex = {k: c[k] for k in ("flat_block_frac", "mode_other", "mode_L", "broken_bonds") if k in c}
    rows.append((n, c["step"], c["fwd_kl"], c["rev_kl_lam10"], c["p_viol"], c["F_per_cell"], ref["Fstar"], c["ent_per_cell"], ref["ent_pstar_per_cell"], ex, {k: ref[k] for k in ex}))
with open("summary.md", "w") as f:
    f.write("| run | steps | fwd KL(p*\\|q) | rev KL(q\\|p_lam10) | p_viol | F/cell | F* | H/cell | H(p*)/cell | stats (q) | stats (p*) |\n|" + "---|" * 11 + "\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.4f} | " +
                ", ".join(f"{k} {v:.4f}" for k, v in r[9].items()) + " | " + ", ".join(f"{k} {v:.4f}" for k, v in r[10].items()) + " |\n")
print(open("summary.md").read())
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for pi, prob in enumerate(["support", "modes"]):
    for n, r in runs.items():
        if r["prob"] != prob: continue
        t = [c["t"] for c in r["curve"]]
        for j, (k, lab) in enumerate([("fwd_kl", "forward KL"), ("rev_kl_lam10", "reverse KL (lam 10)"), ("ent_per_cell", "entropy / cell")]):
            ax[pi, j].plot(t, [c[k] for c in r["curve"]], label=n.replace(prob + "_", "")); ax[pi, j].set_title(f"{prob}: {lab}")
            if k != "ent_per_cell": ax[pi, j].set_yscale("log")
    ax[pi, 2].axhline(next(r["ref"]["ent_pstar_per_cell"] for r in runs.values() if r["prob"] == prob), color="k", ls=":")
ax[0, 0].legend(fontsize=6); [a.set_xlabel("s") for a in ax.ravel()]
plt.tight_layout(); plt.savefig("images/kl_entropy.png", dpi=110)
