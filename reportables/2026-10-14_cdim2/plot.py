import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = {}
for f in sorted(glob.glob("runs/dimers4_exact_*.json")):
    d = json.load(open(f)); n = os.path.basename(f)[8:-5]
    R[n] = d
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
col = {"mlp": "C0", "table_feat": "C1", "table_hist": "C2"}
for n, d in R.items():
    w6 = "w6" in n; kind = n.replace("exact_", "").replace("_w6x", "").replace("_w6", "").replace("_x", "")
    col.setdefault(kind, "C%d" % (3 + len(col) - 3))
    c = d["curve"]; x = [m["step"] for m in c]
    kw = dict(color=col[kind], ls="--" if w6 else "-", label=f"{kind} {'6x6' if w6 else '4x4'}")
    ax[0].plot(x, [m["kl"] for m in c], **kw)
    ax[1].plot(x, [m["entropy"] / 16 for m in c], **kw)
    ax[2].plot(x, [m["p_viol"] for m in c], **kw)
meta = next(iter(R.values()))["meta"]
ax[1].axhline(meta["log_nvalid_per_cell"], color="k", lw=0.8, ls=":", label="target (log 412 / 16)")
ax[0].set_yscale("log"); ax[0].set_title("KL(q || target)"); ax[1].set_title("entropy per cell (nats)"); ax[2].set_title("P(violation) under q")
ax[2].set_yscale("symlog", linthresh=1e-4)
for a in ax: a.set_xlabel("step")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("images/exact_dimers.png", dpi=110)
print("%-24s %8s %8s %8s %7s %7s %7s %7s" % ("run", "kl", "p_viol", "H/cell", "h", "v", "diag", "pvalid"))
for n, d in R.items():
    m = d["curve"][-1]
    print("%-24s %8.4f %8.4f %8.4f %7.3f %7.3f %7.3f" % (n, m["kl"], m["p_viol"], m["entropy"]/16, m["horizontal"], m["vertical"], m["diagonal"]))
