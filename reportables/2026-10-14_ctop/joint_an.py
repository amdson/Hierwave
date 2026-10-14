import pickle, numpy as np, json
R = pickle.load(open("res_joint.pkl", "rb")); CELLS = 64 * 128
# acc fields: n, cost, soft, nv, logq, ...
rows = {}
for (name, tau, cn), r in R.items():
    a = r["acc"].mean(0)           # (5, nf)
    lam = np.array(r["lam"])
    soft = a[:, 2] / CELLS; viol = lam * a[:, 3] / CELLS; lq = tau * a[:, 4] / CELLS      # tau used for the pass
    cost = a[:, 1] / CELLS
    rows[(name, tau, cn)] = dict(soft=soft, viol=viol, lq=lq, cost=cost, nv=a[:, 3] / CELLS)
out = {}
for name in ("init", "u50", "u200"):
    taus = sorted({t for (n, t, c) in rows if n == name})
    for tau in taus:
        print("==", name, "tau", tau)
        for cn in ("own", "probe0", "probe1", "probe3", "dirt"):
            d = rows[(name, tau, cn)]
            below = slice(0, 4)
            print(f"  {cn:7s} below-L4 cost {d['cost'][below].sum():8.4f} = soft {d['soft'][below].sum():8.4f} + lam*viol {d['viol'][below].sum():8.4f} + tau*logq {d['lq'][below].sum():8.4f} | per level cost {np.round(d['cost'][:4],3)} | L4 clamp cost {d['cost'][4]:.3f}")
        for cn in ("probe0", "probe1", "probe3"):
            a, b = rows[(name, tau, cn)], rows[(name, tau, "dirt")]
            dd = {k: (a[k][:4] - b[k][:4]) for k in ("soft", "viol", "lq", "cost")}
            print(f"  {cn}-dirt: total {dd['cost'].sum():+.4f}  soft {dd['soft'].sum():+.4f} viol {dd['viol'].sum():+.4f} logq {dd['lq'].sum():+.4f} | by level {np.round(dd['cost'],3)}")
            out[f"{name}|{tau}|{cn}-dirt"] = {k: v.tolist() for k, v in dd.items()}
        out[f"{name}|{tau}|abs"] = {cn: {k: v.tolist() for k, v in rows[(name, tau, cn)].items()} for cn in ("own", "probe0", "probe1", "probe3", "dirt")}
json.dump(out, open("res_joint_summary.json", "w"), indent=1)
