"""Summaries of l4src.py snapshots.  usage: analyse.py <outdir>"""
import sys, os, glob, json
import numpy as np
from castlegen.bench import shares
d = sys.argv[1]
ref = np.load(os.path.join(d, "ref_own.npz"))
lines = {}
for l in (3, 4):
    a = ref[f"L{l}"]; rng = np.random.default_rng(0)
    lines[(l, "direct")] = shares._robust_line(a[:, 1], a[:, 0], rng)
    lines[(l, "subtree")] = shares._robust_line(a[:, 2], a[:, 0], rng)
SET = {0: "own", 1: "probe", 2: "ownchain", 3: "prbchain"}

def ovl(a, b, bins):
    ha = np.histogram(a, bins)[0] / a.size; hb = np.histogram(b, bins)[0] / b.size
    return float(np.minimum(ha, hb).sum())

def summ(f):
    z = np.load(f); out = {}
    pr4 = np.load(os.path.join(d, "ref_probe.npz"))["L4"][:, 1]
    bins = np.linspace(-40, 5, 46)
    for setid in np.unique(z["L4"][:, 3]).astype(int):
        r = {}
        for l in (3, 4):
            a = z[f"L{l}"]; m = a[:, 3] == setid; a = a[m]
            for nm, col in (("direct", 1), ("subtree", 2)):
                s, ic = lines[(l, nm)]; dfc = s * a[:, col] + ic - a[:, 0]
                r[f"L{l}_starved_{nm}"] = float((dfc > 5).mean())
            w = a[:, 1]
            r[f"L{l}_w_mean"], r[f"L{l}_w_sd"], r[f"L{l}_w_med"] = float(w.mean()), float(w.std()), float(np.median(w))
            r[f"L{l}_logq_mean"] = float(a[:, 0].mean()); r[f"L{l}_slack"] = float(a[:, 5].mean())
            if l == 4:
                r["L4_overlap_probe_w"] = ovl(w, pr4, bins)
                for i, k in enumerate(("lq_rho4", "lq_cert4", "noise4", "smooth4", "abs_drho4", "cert_tight", "A_below", "B_above")):
                    r[f"L4_{k}"] = float(a[:, 7 + i].mean())
        wl = z["line_match"]; nv = z["nviol"]
        sel = (np.arange(nv.size) >= 0) if setid in (0, 1) else ((np.arange(nv.size) < 32) if setid == 2 else (np.arange(nv.size) >= 32))
        r["line_match"] = float(wl[sel].mean()); r["valid_frac"] = float((nv[sel] == 0).mean())
        out[SET[setid]] = r
    return out

res = {}
names = ["ref_own", "ref_probe"] + [f"l4only_{k}" for k in (1, 10, 100)] + [f"relax_{k}" for k in (1, 10, 100)] + [f"chain_{k}" for k in (0, 32, 100, 320, 1000, 3200)]
for n in names:
    f = os.path.join(d, n + ".npz")
    if os.path.exists(f):
        res[n] = summ(f)
json.dump({"lines": {f"{k[0]}_{k[1]}": v for k, v in lines.items()}, "res": res}, open(os.path.join(d, "summary.json"), "w"), indent=1)
cols = ["L4_starved_direct", "L4_starved_subtree", "L4_w_mean", "L4_w_sd", "L4_overlap_probe_w", "L4_logq_mean", "L4_lq_rho4", "L4_lq_cert4",
        "L4_noise4", "L4_smooth4", "L4_abs_drho4", "L4_cert_tight", "L4_A_below", "L4_B_above", "L4_slack",
        "L3_starved_direct", "L3_w_mean", "line_match", "valid_frac"]
print("snapshot set " + " ".join(cols))
for n, r in res.items():
    for s, x in r.items():
        print(n, s, " ".join(f"{x[c]:.3f}" for c in cols))
