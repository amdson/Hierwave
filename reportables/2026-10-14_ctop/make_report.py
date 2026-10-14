import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("res_native_summary.json")); J = json.load(open("res_joint_summary.json"))
ck = [("init", "init@tau2.00", "init|2.0"), ("u50", "u50@tau1.90", "u50|1.897"), ("u200", "u200@tau1.54", "u200|1.538")]
TERM = ["soft energy", "lambda * violations", "tau * log q"]
COL = ["#4c78a8", "#e45756", "#72b7b2"]
def single(setname, ch=None):
    out = []
    for lab, k, _ in ck:
        d = S[k]["sets"][setname]["probe-dirt"]; out.append(d)
    return out
sel = {"rho4 draws (probe value vs 256)": "rho4 probe!=dirt & both in A",
       "cert4 draws (probe cert vs (16,16)), both in A": "cert4 probe!=dirt & both in A"}
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
x = np.arange(3); w = 0.26
def bars(ax, data_terms, tot, title, ylabel=True):
    for t in range(3):
        ax.bar(x + (t - 1) * w, [d[t] for d in data_terms], w, color=COL[t], label=TERM[t])
    ax.plot(x, tot, "ko", ms=7, label="total (downstream)")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["init\n(tau 2.0)", "50 updates\n(tau 1.9)", "200 updates\n(tau 1.54)"])
    ax.set_title(title, fontsize=10)
    if ylabel: ax.set_ylabel("partial fill minus all dirt, nats per fine cell\n(positive: all dirt is cheaper downstream)")
for ax, (ttl, sn) in zip(axs[:2], sel.items()):
    ds = single(sn); bars(ax, [d["by_term"] for d in ds], [d["per_cell"] for d in ds], ttl, ax is axs[0])
# joint is below-L4 only (list has 4 levels); add L4 own cost difference
jt2, jtot2 = [], []
for lab, _, jk in ck:
    a = J[jk + "|abs"]; dd = {k: np.array(a["probe1"][k]) - np.array(a["dirt"][k]) for k in ("soft", "viol", "lq", "cost")}
    jt2.append([dd["soft"][:4].sum(), dd["viol"][:4].sum(), dd["lq"][:4].sum()]); jtot2.append(dd["cost"].sum())
bars(axs[2], jt2, jtot2, "all top-level draws clamped together (probe vs all dirt)\nbars: levels 0-3 by term; dot: levels 0-4 total", False)
axs[0].legend(fontsize=8, loc="best")
plt.tight_layout(); plt.savefig("images/ctop_cost_diff.png", dpi=130)
# tables
print("JOINT incl L4 (probe1 - dirt) native tau:", [(l, np.round(t, 3).tolist(), round(c, 3)) for (l, _, _), t, c in zip(ck, jt2, jtot2)])
for tau in ("1.0", "2.0"):
    for n in ("init", "u50", "u200"):
        a = J[f"{n}|{tau}|abs"] if f"{n}|{tau}|abs" in J else None
        if a is None: continue
        res = []
        for pn in ("probe0", "probe1", "probe3"):
            dd = {k: np.array(a[pn][k]) - np.array(a["dirt"][k]) for k in ("soft", "viol", "lq", "cost")}
            res.append((pn, round(dd["cost"].sum(), 3), [round(dd[k].sum(), 3) for k in ("soft", "viol", "lq")], np.round(dd["cost"], 3).tolist()))
        print("joint", n, "tau", tau, res)
