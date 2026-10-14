import pickle, sys, json, numpy as np
CELLS = 64 * 128
f = sys.argv[1]
R = pickle.load(open(f, "rb"))
CHN = ["rho4", "cert4"]
TERM = ["soft", "lam*viol", "tau*logq"]
LEV = ["L0", "L1", "L2", "L3", "L4"]

def stats(x, w):
    """x (NW, ND) values, w (NW, ND) bool -> mean over selected draws pooled per world, SE across worlds."""
    per = np.array([x[b][w[b]].mean() if w[b].any() else np.nan for b in range(x.shape[0])])
    return np.nanmean(per), np.nanstd(per) / np.sqrt(np.sum(~np.isnan(per))), int(w.sum())

out = {}
for (name, tau), r in R.items():
    G, own, inA, tab, cand = r["G"], r["own"], r["inA"], r["tab"], r["cand"]
    NW, ND = r["nA"].shape
    chan = np.broadcast_to(tab[:, 0][None], (NW, ND))
    sets = {
        "all": np.ones((NW, ND), bool),
        "probe!=dirt": cand["probe"] != cand["dirt"],
        "probe!=dirt & both in A": (cand["probe"] != cand["dirt"]) & inA["probe"] & inA["dirt"],
    }
    for ch in (0, 1):
        sets[f"{CHN[ch]} probe!=dirt"] = sets["probe!=dirt"] & (chan == ch)
        sets[f"{CHN[ch]} probe!=dirt & both in A"] = sets["probe!=dirt & both in A"] & (chan == ch)
    o = {"tau": tau, "lam": r["lam"], "frac_probe_in_A": float(inA["probe"].mean()), "frac_dirt_in_A": float(inA["dirt"].mean()),
         "frac_actual_dirt": float((cand["actual"] == cand["dirt"]).mean()),
         "frac_actual_dirt_cert4": float((cand["actual"] == cand["dirt"])[chan == 1].mean()),
         "frac_actual_dirt_rho4": float((cand["actual"] == cand["dirt"])[chan == 0].mean()),
         "mean_nA": float(r["nA"].mean()), "sets": {}}
    for sn, w in sets.items():
        d = {}
        for a, b in (("probe", "dirt"), ("actual", "dirt")):
            D = (G[a] - G[b])                                  # (NW, ND, 3, 5)
            tot = D.sum((2, 3)) / CELLS
            m, se, n = stats(tot, w)
            fr = float(np.mean([(tot[i][w[i]] < 0).mean() for i in range(NW) if w[i].any()]))
            d[f"{a}-{b}"] = {"per_cell": m, "se": se, "n": n, "frac_cheaper": fr,
                             "by_term": [stats(D[:, :, t].sum(-1) / CELLS, w)[0] for t in range(3)],
                             "by_level": [stats(D[:, :, :, l].sum(-1) / CELLS, w)[0] for l in range(5)],
                             "term_x_level": [[stats(D[:, :, t, l] / CELLS, w)[0] for l in range(5)] for t in range(3)]}
        # own cost of the draw plus downstream, in-A both only
        wa = w & inA["probe"] & inA["dirt"]
        own_d = np.nansum(own["probe"] - own["dirt"], -1)
        tot = (own_d + (G["probe"] - G["dirt"]).sum((2, 3))) / CELLS
        d["own_probe-dirt"] = stats(own_d / CELLS, wa)[0]
        d["own_by_term_probe-dirt"] = [stats((own["probe"] - own["dirt"])[:, :, t] / CELLS, wa)[0] for t in range(3)]
        d["total_probe-dirt_inA"] = {"per_cell": stats(tot, wa)[0], "se": stats(tot, wa)[1], "n": stats(tot, wa)[2],
                                     "frac_cheaper": float(np.mean([(tot[i][wa[i]] < 0).mean() for i in range(NW) if wa[i].any()])) if wa.any() else None}
        o["sets"][sn] = d
    out[f"{name}@tau{tau:.2f}"] = o
json.dump(out, open(f.replace(".pkl", "_summary.json"), "w"), indent=1, default=float)
for k, o in out.items():
    print("==", k, "lam", np.round(o["lam"], 1), "A-frac probe/dirt", round(o["frac_probe_in_A"], 2), round(o["frac_dirt_in_A"], 2),
          "actual dirt frac cert4/rho4", round(o["frac_actual_dirt_cert4"], 2), round(o["frac_actual_dirt_rho4"], 2), "mean|A|", round(o["mean_nA"], 1))
    for sn, d in o["sets"].items():
        s = d["probe-dirt"]; a = d["actual-dirt"]
        print(f"  [{sn}] n={s['n']}: probe-dirt {s['per_cell']:+.4f}+-{s['se']:.4f} /cell cheaper {s['frac_cheaper']:.2f} | terms {np.round(s['by_term'],4)} | lev {np.round(s['by_level'],4)} | actual-dirt {a['per_cell']:+.4f} cheaper {a['frac_cheaper']:.2f} | own {d['own_probe-dirt']:+.4f} total {d['total_probe-dirt_inA']['per_cell']:+.4f} cheaper {d['total_probe-dirt_inA']['frac_cheaper']}")
